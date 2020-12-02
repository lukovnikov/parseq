from abc import abstractmethod
from copy import deepcopy
from typing import Dict

import torch
from nltk import Tree
import numpy as np
from torch.utils.data import DataLoader

import qelos as q

from parseq.datasets import OvernightDatasetLoader, autocollate
from parseq.grammar import tree_to_lisp_tokens
from parseq.scripts_insert.overnight_treeinsert import extract_info
from parseq.scripts_insert.util import reorder_tree, flatten_tree
from parseq.transformer import TransformerConfig, TransformerStack
from parseq.vocab import Vocab, SequenceEncoder
from transformers import BertTokenizer, BertModel


def tree_to_seq(x:Tree):
    xstr = tree_to_lisp_tokens(x)
    xstr = ["@BOS@"] + xstr + ["@EOS@"]
    return xstr


def load_ds(domain="restaurants", nl_mode="bert-base-uncased",
            trainonvalid=False, noreorder=False):
    """
    Creates a dataset of examples which have
    * NL question and tensor
    * original FL tree
    * reduced FL tree with slots (this is randomly generated)
    * tensor corresponding to reduced FL tree with slots
    * mask specifying which elements in reduced FL tree are terminated
    * 2D gold that specifies whether a token/action is in gold for every position (compatibility with MML!)
    """
    orderless = {"op:and", "SW:concat"}     # only use in eval!!

    ds = OvernightDatasetLoader().load(domain=domain, trainonvalid=trainonvalid)
    # ds contains 3-tuples of (input, output tree, split name)

    if not noreorder:
        ds = ds.map(lambda x: (x[0], reorder_tree(x[1], orderless=orderless), x[2]))
    ds = ds.map(lambda x: (x[0], tree_to_seq(x[1]), x[2]))

    vocab = Vocab(padid=0, startid=2, endid=3, unkid=1)
    vocab.add_token("@BOS@", seen=np.infty)
    vocab.add_token("@EOS@", seen=np.infty)
    vocab.add_token("@STOP@", seen=np.infty)

    nl_tokenizer = BertTokenizer.from_pretrained(nl_mode)

    tds, vds, xds = ds[lambda x: x[2] == "train"], \
                    ds[lambda x: x[2] == "valid"], \
                    ds[lambda x: x[2] == "test"]

    seqenc = SequenceEncoder(vocab=vocab, tokenizer=lambda x: x,
                             add_start_token=False, add_end_token=False)
    for example in tds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=True)
    for example in vds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=False)
    for example in xds.examples:
        query = example[1]
        seqenc.inc_build_vocab(query, seen=False)
    seqenc.finalize_vocab(min_freq=0)

    def mapper(x):
        seq = seqenc.convert(x[1], return_what="tensor")
        ret = (nl_tokenizer.encode(x[0], return_tensors="pt")[0], seq)
        return ret

    tds_seq = tds.map(mapper)
    vds_seq = vds.map(mapper)
    xds_seq = xds.map(mapper)
    return tds_seq, vds_seq, xds_seq, nl_tokenizer, seqenc, orderless


class SeqInsertionTagger(torch.nn.Module):
    """ A tree insertion tagging model takes a sequence representing a tree
        and produces distributions over tree modification actions for every (non-terminated) token.
    """
    @abstractmethod
    def forward(self, tokens:torch.Tensor, **kw):
        """
        :param tokens:      (batsize, seqlen)       # all are open!
        :return:
        """
        pass


class TransformerTagger(SeqInsertionTagger):
    def __init__(self, dim, vocab:Vocab=None, numlayers:int=6, numheads:int=6,
                 dropout:float=0., maxpos=512, bertname="bert-base-uncased", **kw):
        super(TransformerTagger, self).__init__(**kw)
        self.vocab = vocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim
        config = TransformerConfig(vocab_size=self.vocabsize, d_model=self.dim, d_ff=self.dim * 4,
                                   num_layers=numlayers, num_heads=numheads, dropout_rate=dropout,
                                   use_relative_position=False)

        self.emb = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.posemb = torch.nn.Embedding(maxpos, config.d_model)
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_causal_mask = False
        self.decoder = TransformerStack(decoder_config)

        self.out = torch.nn.Linear(self.dim * 2, self.vocabsize)
        # self.out = MOS(self.dim, self.vocabsize, K=mosk)

        vocab_mask = torch.ones(self.vocabsize)
        # for excl_token in self.exclude:
        #     if excl_token in self.vocab:
        #         vocab_mask[self.vocab[excl_token]] = 0
        self.register_buffer("vocab_mask", vocab_mask)

        self.bertname = bertname
        self.bert_model = BertModel.from_pretrained(self.bertname)
        def set_dropout(m:torch.nn.Module):
            if isinstance(m, torch.nn.Dropout):
                m.p = dropout
        self.bert_model.apply(set_dropout)

        self.adapter = None
        if self.bert_model.config.hidden_size != decoder_config.d_model:
            self.adapter = torch.nn.Linear(self.bert_model.config.hidden_size, decoder_config.d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None):
        padmask = (tokens != 0)
        padmask = padmask[:, 1:]
        embs = self.emb(tokens)
        posembs = self.posemb(torch.arange(tokens.size(1), device=tokens.device))[None]
        embs = embs + posembs
        ret = self.decoder(inputs_embeds=embs, attention_mask=padmask,
                     encoder_hidden_states=enc,
                     encoder_attention_mask=encmask, use_cache=False)
        ret = ret[0]
        c = torch.cat([ret[:, 1:], ret[:, :-1]], -1)
        logits = self.out(c)
        # logits = logits + torch.log(self.vocab_mask[None, None, :])
        return logits, padmask
        # probs = self.out(ret[0], self.vocab_mask[None, None, :])
        # return probs


class SeqInsertionDecoder(torch.nn.Module):
    def __init__(self, tagger:SeqInsertionTagger,
                 max_steps:int=50,
                 max_size:int=100,
                 **kw):
        super(SeqInsertionDecoder, self).__init__(**kw)
        self.tagger = tagger
        self.max_steps = max_steps
        self.max_size = max_size

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return self.test_forward(x, y)

    @abstractmethod
    def train_forward(self, x, y):  # --> implement one step training of tagger
        # extract a training example from y: different for different versions
        # run through tagger: the same for all versions
        # compute loss: different versions do different masking and different targets
        pass

    @abstractmethod
    def test_forward(self, x, y):   # --> implement how decoder operates end-to-end
        pass


class SeqInsertionDecoderUniform(SeqInsertionDecoder):
    def __init__(self, tagger:SeqInsertionTagger,
                 **kw):
        super(SeqInsertionDecoderUniform, self).__init__(**kw)
        # TODO


class SeqInsertionDecoderBinary(SeqInsertionDecoderUniform):
    """ Differs from Uniform only in computing and using non-uniform weights for gold output distributions """
    def __init__(self, tagger:SeqInsertionTagger,
                 **kw):
        super(SeqInsertionDecoderBinary, self).__init__(**kw)
        # TODO


class SeqInsertionDecoderLTR(SeqInsertionDecoder):
    def __init__(self, tagger:SeqInsertionTagger,
                 **kw):
        super(SeqInsertionDecoderLTR, self).__init__(**kw)
        # TODO


def run(lr=0.001,
        batsize=10,
        hdim=768,
        numlayers=6,
        numheads=12,
        dropout=0.1,
        noreorder=False,
        trainonvalid=False):

    tt = q.ticktock("script")
    tt.tick("loading")
    tds_seq, vds_seq, xds_seq, nltok, flenc, orderless = load_ds("restaurants", trainonvalid=trainonvalid, noreorder=noreorder)
    tt.tock("loaded")

    tdl_seq = DataLoader(tds_seq, batch_size=batsize, shuffle=True, collate_fn=autocollate)
    vdl_seq = DataLoader(vds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)
    xdl_seq = DataLoader(xds_seq, batch_size=batsize, shuffle=False, collate_fn=autocollate)

    # model
    tagger = TransformerTagger(hdim, flenc.vocab, numlayers, numheads, dropout)

    # test run
    batch = next(iter(tdl_seq))
    out = tagger(batch[1])


# TODO: EOS balancing ?!


if __name__ == '__main__':
    q.argprun(run)