import os
import re
import sys
from functools import partial
from typing import *

import numpy as np
import torch

import qelos as q
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from nltk import PorterStemmer

from torch.utils.data import DataLoader

# from funcparse.decoding import TransitionModel, TFActionSeqDecoder, LSTMCellTransition, BeamActionSeqDecoder, \
#     GreedyActionSeqDecoder, TFTokenSeqDecoder
# from funcparse.grammar import FuncGrammar, passtr_to_pas
# from funcparse.states import FuncTreeState, FuncTreeStateBatch, BasicState, BasicStateBatch
# from funcparse.vocab import VocabBuilder, SentenceEncoder, FuncQueryEncoder
# from funcparse.nn import TokenEmb, PtrGenOutput, SumPtrGenOutput, BasicGenOutput
from parseq.decoding import SeqDecoder
from parseq.eval import CELoss, SeqAccuracies, make_loss_array, DerivedAccuracy, TreeAccuracy
from parseq.grammar import prolog_to_pas, lisp_to_pas, pas_to_prolog, prolog_to_tree
from parseq.nn import TokenEmb, BasicGenOutput, PtrGenOutput, PtrGenOutput2, GRUEncoder
from parseq.states import DecodableState, BasicDecoderState, State, batchstack, TreeDecoderState
from parseq.transitions import TransitionModel, LSTMCellTransition, GRUTransition
from parseq.vocab import SequenceEncoder, Vocab


def stem_id_words(pas, idparents, stem=False, strtok=None):
    if stem is True:
        assert(not isinstance(pas, tuple))
    if not isinstance(pas, tuple):
        if stem is True:
            assert(isinstance(pas, str))
            if re.match(r"'([^']+)'", pas):
                pas = re.match(r"'([^']+)'", pas).group(1)
                pas = strtok(pas)
                return [("str", pas)]
            else:
                return [pas]
        else:
            return [pas]
    else:
        tostem = pas[0] in idparents
        children = [stem_id_words(k, idparents, stem=tostem, strtok=strtok)
                    for k in pas[1]]
        children = [a for b in children for a in b]
        return [(pas[0], children)]


def pas2toks(pas):
    if not isinstance(pas, tuple):
        return [pas]
    else:
        children = [pas2toks(k) for k in pas[1]]
        ret = [pas[0]] if pas[0] != "@NAMELESS@" else []
        ret[0] = f"_{ret[0]}("
        for child in children:
            ret += child
            # ret.append(",")
        # ret.pop(-1)
        ret.append("_)")
        return ret


def basic_query_tokenizer(x:str, strtok=None):
    pas = prolog_to_pas(x)
    # idpreds = set("_cityid _countryid _stateid _riverid _placeid".split(" "))
    idpreds = set("cityid stateid countryid riverid placeid".split(" "))
    pas = stem_id_words(pas, idpreds, strtok=strtok)[0]
    ret = pas2toks(pas)
    return ret

def try_basic_query_tokenizer():
    stemmer = PorterStemmer()
    x = "answer(cityid('new york', _))"
    y = basic_query_tokenizer(x, strtok=lambda x: [stemmer.stem(xe) for xe in x.split()])
    # print(y)


class GeoQueryDatasetFunQL(object):
    def __init__(self,
                 p="../../datasets/geoquery/",
                 sentence_encoder:SequenceEncoder=None,
                 min_freq:int=2, **kw):
        super(GeoQueryDatasetFunQL, self).__init__(**kw)
        self._initialize(p, sentence_encoder, min_freq)

    def _initialize(self, p, sentence_encoder:SequenceEncoder, min_freq:int):
        self.data = {}
        self.sentence_encoder = sentence_encoder
        questions = [x.strip() for x in open(os.path.join(p, "questions.txt"), "r").readlines()]
        queries = [x.strip() for x in open(os.path.join(p, "queries.funql"), "r").readlines()]
        trainidxs = set([int(x.strip()) for x in open(os.path.join(p, "train_indexes.txt"), "r").readlines()])
        testidxs = set([int(x.strip()) for x in open(os.path.join(p, "test_indexes.txt"), "r").readlines()])
        splits = [None]*len(questions)
        for trainidx in trainidxs:
            splits[trainidx] = "train"
        for testidx in testidxs:
            splits[testidx] = "test"
        if any([split == None for split in splits]):
            print(f"{len([split for split in splits if split == None])} examples not assigned to any split")

        self.query_encoder = SequenceEncoder(tokenizer=partial(basic_query_tokenizer, strtok=sentence_encoder.tokenizer), add_end_token=True)

        # build vocabularies
        unktokens = set()
        for i, (question, query, split) in enumerate(zip(questions, queries, splits)):
            question_tokens = self.sentence_encoder.inc_build_vocab(question, seen=split=="train")
            query_tokens = self.query_encoder.inc_build_vocab(query, seen=split=="train")
            unktokens |= set(query_tokens) - set(question_tokens)
        for word in self.sentence_encoder.vocab.counts.keys():
            self.query_encoder.vocab.add_token(word, seen=False)
        self.sentence_encoder.finalize_vocab(min_freq=min_freq, keep_rare=True)
        self.query_encoder.finalize_vocab(min_freq=min_freq, keep_rare=True)
        unktokens = unktokens & self.query_encoder.vocab.rare_tokens

        self.build_data(questions, queries, splits, unktokens=unktokens)

    def build_data(self, inputs:Iterable[str], outputs:Iterable[str], splits:Iterable[str], unktokens:Set[str]=None):
        if unktokens is not None:
            gold_map = torch.arange(0, self.query_encoder.vocab.number_of_ids(last_nonrare=False))
            for rare_token in unktokens:
                gold_map[self.query_encoder.vocab[rare_token]] = \
                    self.query_encoder.vocab[self.query_encoder.vocab.unktoken]
        for inp, out, split in zip(inputs, outputs, splits):

            inp_tensor, inp_tokens = self.sentence_encoder.convert(inp, return_what="tensor,tokens")
            gold_tree = prolog_to_tree(out)
            assert(gold_tree is not None)
            out_tensor, out_tokens = self.query_encoder.convert(out, return_what="tensor,tokens")
            if gold_map is not None:
                out_tensor = gold_map[out_tensor]

            state = TreeDecoderState([inp], [gold_tree],
                                      inp_tensor[None, :], out_tensor[None, :],
                                      [inp_tokens], [out_tokens],
                                      self.sentence_encoder.vocab, self.query_encoder.vocab)
            if split not in self.data:
                self.data[split] = []
            self.data[split].append(state)

    def get_split(self, split:str):
        return DatasetSplitProxy(self.data[split])

    @staticmethod
    def collate_fn(data:Iterable):
        goldmaxlen = 0
        inpmaxlen = 0
        data = [state.make_copy(detach=True, deep=True) for state in data]
        for state in data:
            goldmaxlen = max(goldmaxlen, state.gold_tensor.size(1))
            inpmaxlen = max(inpmaxlen, state.inp_tensor.size(1))
        for state in data:
            state.gold_tensor = torch.cat([
                state.gold_tensor,
                state.gold_tensor.new_zeros(1, goldmaxlen - state.gold_tensor.size(1))], 1)
            state.inp_tensor = torch.cat([
                state.inp_tensor,
                state.inp_tensor.new_zeros(1, inpmaxlen - state.inp_tensor.size(1))], 1)
        ret = data[0].merge(data)
        return ret

    def dataloader(self, split:str=None, batsize:int=5):
        if split is None:   # return all splits
            ret = {}
            for split in self.data.keys():
                ret[split] = self.dataloader(batsize=batsize, split=split)
            return ret
        else:
            assert(split in self.data.keys())
            dl = DataLoader(self.get_split(split), batch_size=batsize, shuffle=split=="train",
             collate_fn=GeoQueryDatasetFunQL.collate_fn)
            return dl


class GeoQueryDatasetSub(GeoQueryDatasetFunQL):
    def __init__(self,
                 p="../../datasets/geo880dong/",
                 sentence_encoder:SequenceEncoder=None,
                 min_freq:int=2, **kw):
        super(GeoQueryDatasetSub, self).__init__(p, sentence_encoder, min_freq, **kw)

    def _initialize(self, p, sentence_encoder:SequenceEncoder, min_freq:int):
        self.data = {}
        self.sentence_encoder = sentence_encoder
        trainlines = [x.strip() for x in open(os.path.join(p, "train.txt"), "r").readlines()]
        testlines = [x.strip() for x in open(os.path.join(p, "test.txt"), "r").readlines()]
        splits = ["train"]*len(trainlines) + ["test"] * len(testlines)
        questions, queries = zip(*[x.split("\t") for x in trainlines])
        testqs, testxs = zip(*[x.split("\t") for x in testlines])
        questions += testqs
        queries += testxs

        queries = self.lisp2prolog(queries)

        self.query_encoder = SequenceEncoder(tokenizer=partial(basic_query_tokenizer, strtok=sentence_encoder.tokenizer), add_end_token=True)

        # build vocabularies
        for i, (question, query, split) in enumerate(zip(questions, queries, splits)):
            self.sentence_encoder.inc_build_vocab(question, seen=split=="train")
            self.query_encoder.inc_build_vocab(query, seen=split=="train")
        for word, wordid in self.sentence_encoder.vocab.D.items():
            self.query_encoder.vocab.add_token(word, seen=False)
        self.sentence_encoder.finalize_vocab(min_freq=min_freq)
        self.query_encoder.finalize_vocab(min_freq=min_freq)

        self.build_data(questions, queries, splits)

    def lisp2prolog(self, data:List[str]):
        ret = []
        for x in data:
            pas = lisp_to_pas(x)
            prolog = pas_to_prolog(pas)
            ret.append(prolog)
        return ret


def try_dataset():
    tt = q.ticktock("dataset")
    tt.tick("building dataset")
    ds = GeoQueryDatasetFunQL(sentence_encoder=SequenceEncoder(tokenizer=lambda x: x.split()))
    train_dl = ds.dataloader("train", batsize=19)
    test_dl = ds.dataloader("test", batsize=20)
    examples = set()
    examples_list = []
    duplicates = []
    for b in train_dl:
        print(len(b))
        for i in range(len(b)):
            example = b.inp_strings[i] + " --> " + b.gold_strings[i]
            if example in examples:
                duplicates.append(example)
            examples.add(example)
            examples_list.append(example)
            # print(example)
        pass
    print(f"duplicates within train: {len(duplicates)} from {len(examples_list)} total")
    tt.tock("dataset built")


class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item].make_copy()

    def __len__(self):
        return len(self.data)


class BasicGenModel(TransitionModel):
    def __init__(self, inp_emb, inp_enc, out_emb, out_rnn:LSTMCellTransition,
                 out_lin, att, enc_to_dec=None, feedatt=False, nocopy=False, dropout=0., **kw):
        super(BasicGenModel, self).__init__(**kw)
        self.dropout = torch.nn.Dropout(dropout)
        self.inp_emb, self.inp_enc = inp_emb, inp_enc
        self.out_emb, self.out_rnn, self.out_lin = out_emb, out_rnn, out_lin
        self.enc_to_dec = enc_to_dec
        self.att = att
        # self.ce = q.CELoss(reduction="none", ignore_index=0, mode="probs")
        self.feedatt = feedatt
        self.nocopy = nocopy

    def forward(self, x:State):
        if not "mstate" in x:
            x.mstate = State()
        mstate = x.mstate
        if not "ctx" in mstate:
            # encode input
            inptensor = x.inp_tensor
            mask = inptensor != 0
            inpembs = self.inp_emb(inptensor)
            inpembs = self.dropout(inpembs)
            # inpembs = self.dropout(inpembs)
            inpenc, final_encs = self.inp_enc(inpembs, mask)
            init_states = []
            for i in range(len(final_encs)):
                init_states.append(self.enc_to_dec[i](final_encs[i][0]))
            mstate.ctx = inpenc
            mstate.ctx_mask = mask

        ctx = mstate.ctx
        ctx_mask = mstate.ctx_mask

        emb = self.out_emb(x.prev_actions)
        emb = self.dropout(emb)

        if not "rnnstate" in mstate:
            init_rnn_state = self.out_rnn.get_init_state(emb.size(0), emb.device)
            init_rnn_state.h = torch.stack(init_states, 1).contiguous()
            mstate.rnnstate = init_rnn_state
        if "prev_summ" not in mstate:
            # mstate.prev_summ = torch.zeros_like(ctx[:, 0])
            mstate.prev_summ = final_encs[-1][0]
        _emb = emb
        if self.feedatt == True:
            _emb = torch.cat([_emb, mstate.prev_summ], 1)
        enc, new_rnnstate = self.out_rnn(_emb, mstate.rnnstate)
        mstate.rnnstate = new_rnnstate

        alphas, summ, scores = self.att(enc, ctx, ctx_mask)
        mstate.prev_summ = summ
        enc = torch.cat([enc, summ], -1)

        if self.nocopy is True:
            outs = self.out_lin(enc)
        else:
            outs = self.out_lin(enc, x.inp_tensor, scores)
        outs = (outs,) if not q.issequence(outs) else outs
        # _, preds = outs.max(-1)
        return outs[0], x


def create_model(embdim=100, hdim=100, dropout=0., numlayers:int=1,
                 sentence_encoder:SequenceEncoder=None,
                 query_encoder:SequenceEncoder=None,
                 feedatt=False, nocopy=False):
    inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
    inpemb = TokenEmb(inpemb, rare_token_ids=sentence_encoder.vocab.rare_ids, rare_id=1)
    encoder_dim = hdim * 2
    encoder = GRUEncoder(embdim, hdim, numlayers, bidirectional=True, dropout=dropout)
    # encoder = PytorchSeq2SeqWrapper(
    #     torch.nn.LSTM(embdim, hdim, num_layers=numlayers, bidirectional=True, batch_first=True,
    #                   dropout=dropout))
    decoder_emb = torch.nn.Embedding(query_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
    decoder_emb = TokenEmb(decoder_emb, rare_token_ids=query_encoder.vocab.rare_ids, rare_id=1)
    dec_rnn_in_dim = embdim + (encoder_dim if feedatt else 0)
    decoder_rnn = GRUTransition(dec_rnn_in_dim, hdim, dropout=dropout)
    # decoder_out = BasicGenOutput(hdim + encoder_dim, query_encoder.vocab)
    decoder_out = PtrGenOutput(hdim + encoder_dim, out_vocab=query_encoder.vocab)
    decoder_out.build_copy_maps(inp_vocab=sentence_encoder.vocab)
    attention = q.Attention(q.MatMulDotAttComp(hdim, encoder_dim), dropout=min(0.1, dropout))
    enctodec = torch.nn.ModuleList([torch.nn.Sequential(
        torch.nn.Linear(encoder_dim, hdim),
        torch.nn.Tanh()
    ) for _ in range(numlayers)])
    model = BasicGenModel(inpemb, encoder,
                          decoder_emb, decoder_rnn, decoder_out,
                          attention,
                          enc_to_dec=enctodec,
                          feedatt=feedatt, nocopy=nocopy)
    return model


def do_rare_stats(ds:GeoQueryDatasetSub):
    # how many examples contain rare words, in input and output, in both train and test
    def get_rare_portions(examples:List[State]):
        total = 0
        rare_in_question = 0
        rare_in_query = 0
        rare_in_both = 0
        rare_in_either = 0
        for example in examples:
            total += 1
            question_tokens = example.inp_tokens[0]
            query_tokens = example.gold_tokens[0]
            both = True
            either = False
            if len(set(question_tokens) & example.sentence_vocab.rare_tokens) > 0:
                rare_in_question += 1
                either = True
            else:
                both = False
            if len(set(query_tokens) & example.query_vocab.rare_tokens) > 0:
                either = True
                rare_in_query += 1
            else:
                both = False
            if both:
                rare_in_both += 1
            if either:
                rare_in_either += 1
        return rare_in_question / total, rare_in_query/total, rare_in_both/total, rare_in_either/total
    print("RARE STATS:::")
    print("training data:")
    ris, riq, rib, rie = get_rare_portions(ds.data["train"])
    print(f"\t In question: {ris} \n\t In query: {riq} \n\t In both: {rib} \n\t In either: {rie}")
    print("test data:")
    ris, riq, rib, rie = get_rare_portions(ds.data["test"])
    print(f"\t In question: {ris} \n\t In query: {riq} \n\t In both: {rib} \n\t In either: {rie}")
    return


def tensor2tree(x, D:Vocab=None):
    # x: 1D int tensor
    x = list(x.detach().cpu().numpy())
    x = [D(xe) for xe in x]
    x = [xe for xe in x if xe != D.padtoken]
    # find first @END@ and cut off
    parentheses_balance = 0
    for i in range(len(x)):
        if x[i] ==D.endtoken:
            x = x[:i]
            break
        elif x[i] == "(" or x[i][-1] == "(":
            parentheses_balance += 1
        elif x[i] == ")" or x[i][-1] == ")":
            parentheses_balance -= 1
        else:
            pass

    # balance parentheses
    while parentheses_balance > 0:
        x.append("_)")
        parentheses_balance -= 1
    i = len(x) - 1
    while parentheses_balance < 0 and i > 0:
        if x[i] == ")" or x[i][-1] == ")":
            x.pop(i)
            parentheses_balance += 1
        i -= 1

    # introduce comma's
    i = 1
    while i < len(x):
        if x[i-1][-1] == "(":
            pass
        elif x[i][-1] == ")":
            pass
        else:
            x.insert(i, ",")
            i += 1
        i += 1

    # convert to nltk.Tree
    try:
        prolog = " ".join(x)
        prolog = prolog.replace("_)", ")")
        tree, parsestate = prolog_to_tree(prolog, None)
    except Exception as e:
        tree = None
    return tree



def run(lr=0.001,
        batsize=20,
        epochs=100,
        embdim=100,
        encdim=200,
        numlayers=1,
        dropout=.5,
        wreg=1e-10,
        cuda=False,
        gpu=0,
        minfreq=2,
        gradnorm=3.,
        cosine_restarts=1.,
        ):
    # DONE: Porter stemmer
    # DONE: linear attention
    # DONE: grad norm
    # DONE: beam search
    # DONE: lr scheduler
    print(locals())
    tt = q.ticktock("script")
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt.tick("loading data")
    stemmer = PorterStemmer()
    tokenizer = lambda x: [stemmer.stem(xe) for xe in x.split()]
    ds = GeoQueryDatasetFunQL(sentence_encoder=SequenceEncoder(tokenizer=tokenizer), min_freq=minfreq)

    train_dl = ds.dataloader("train", batsize=batsize)
    test_dl = ds.dataloader("test", batsize=batsize)
    tt.tock("data loaded")

    do_rare_stats(ds)

    # batch = next(iter(train_dl))
    # print(batch)
    # print("input graph")
    # print(batch.batched_states)

    model = create_model(embdim=embdim, hdim=encdim, dropout=dropout, numlayers=numlayers,
                             sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder, feedatt=True)

    tfdecoder = SeqDecoder(model, tf_ratio=1.,
                           eval=[CELoss(ignore_index=0, mode="logprobs"),
                            SeqAccuracies(), TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab))])
    # beamdecoder = BeamActionSeqDecoder(tfdecoder.model, beamsize=beamsize, maxsteps=50)
    freedecoder = SeqDecoder(model, maxtime=100, tf_ratio=0.,
                             eval=[CELoss(ignore_index=0, mode="logprobs"),
                            SeqAccuracies(), TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab))])

    # # test
    # tt.tick("doing one epoch")
    # for batch in iter(train_dl):
    #     batch = batch.to(device)
    #     ttt.tick("start batch")
    #     # with torch.no_grad():
    #     out = tfdecoder(batch)
    #     ttt.tock("end batch")
    # tt.tock("done one epoch")
    # print(out)
    # sys.exit()

    # beamdecoder(next(iter(train_dl)))

    # print(dict(tfdecoder.named_parameters()).keys())

    losses = make_loss_array("loss", "elem_acc", "seq_acc", "tree_acc")
    vlosses = make_loss_array("loss", "seq_acc", "tree_acc")

    # 4. define optim
    optim = torch.optim.Adam(tfdecoder.parameters(), lr=lr, weight_decay=wreg)
    # optim = torch.optim.SGD(tfdecoder.parameters(), lr=lr, weight_decay=wreg)

    # lr schedule
    if cosine_restarts >= 0:
        # t_max = epochs * len(train_dl)
        t_max = epochs
        print(f"Total number of updates: {t_max} ({epochs} * {len(train_dl)})")
        lr_schedule = q.WarmupCosineWithHardRestartsSchedule(optim, 0, t_max, cycles=cosine_restarts)
        reduce_lr = [lambda: lr_schedule.step()]
    else:
        reduce_lr = []

    # 6. define training function (using partial)
    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(tfdecoder.parameters(), gradnorm)
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=tfdecoder, dataloader=train_dl, optim=optim, losses=losses,
                         _train_batch=trainbatch, device=device, on_end=reduce_lr)

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=freedecoder, dataloader=test_dl, losses=vlosses, device=device)
    # validepoch = partial(q.test_epoch, model=tfdecoder, dataloader=test_dl, losses=vlosses, device=device)

    # 7. run training
    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done training")



if __name__ == '__main__':
    # try_basic_query_tokenizer()
    # try_build_grammar()
    # try_dataset()
    q.argprun(run)