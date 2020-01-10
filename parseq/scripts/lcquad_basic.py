import math
import os
import random
import re
import sys
from abc import ABC
from functools import partial
from typing import *

import dill as dill
import torch
import numpy as np
import ujson

import qelos as q
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from nltk import PorterStemmer, Tree

from torch.utils.data import DataLoader

# from funcparse.decoding import TransitionModel, TFActionSeqDecoder, LSTMCellTransition, BeamActionSeqDecoder, \
#     GreedyActionSeqDecoder, TFTokenSeqDecoder
# from funcparse.grammar import FuncGrammar, passtr_to_pas
# from funcparse.states import FuncTreeState, FuncTreeStateBatch, BasicState, BasicStateBatch
# from funcparse.vocab import VocabBuilder, SentenceEncoder, FuncQueryEncoder
# from funcparse.nn import TokenEmb, PtrGenOutput, SumPtrGenOutput, BasicGenOutput
from parseq.decoding import SeqDecoder, TFTransition, FreerunningTransition, BeamDecoder, BeamTransition
from parseq.eval import CELoss, SeqAccuracies, make_loss_array, DerivedAccuracy, TreeAccuracy
from parseq.grammar import prolog_to_pas, lisp_to_pas, pas_to_prolog, pas_to_tree, tree_size, tree_to_prolog, \
    tree_to_lisp, lisp_to_tree
from parseq.nn import TokenEmb, BasicGenOutput, PtrGenOutput, PtrGenOutput2, load_pretrained_embeddings, GRUEncoder, \
    LSTMEncoder
from parseq.states import DecodableState, BasicDecoderState, State, TreeDecoderState, ListState
from parseq.transitions import TransitionModel, LSTMCellTransition, LSTMTransition
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


def stem_id_words_tree(tree:Tree, idparents, stem=False, strtok=None):
    if stem is True:
        assert(len(tree) == 0)  # should be leaf
    if len(tree) == 0:
        if stem is True:
            if re.match(r"'([^']+)'", tree.label()):
                pas = re.match(r"'([^']+)'", tree.label()).group(1)
                pas = strtok(pas)
                return [Tree("str", pas)]
            else:
                return [tree]
        else:
            return [tree]
    else:
        tostem = tree.label() in idparents
        children = [stem_id_words_tree(k, idparents, stem=tostem, strtok=strtok)
                    for k in tree]
        children = [a for b in children for a in b]
        return [Tree(tree.label(), children)]


def pas2toks(pas):
    if not isinstance(pas, tuple):
        return [pas]
    else:
        children = [pas2toks(k) for k in pas[1]]
        ret = [pas[0]] if pas[0] != "@NAMELESS@" else []
        ret.insert(0, "(")
        for child in children:
            ret += child
            # ret.append(",")
        # ret.pop(-1)
        ret.append(")")
        return ret

def tree2toks(tree:Tree):
    if len(tree) == 0:
        return [tree.label()]
    else:
        children = [tree2toks(x) for x in tree]
        ret = [tree.label()] if tree.label() != "@NAMELESS@" else []
        ret.insert(0, "(")
        for child in children:
            ret += child
        ret.append(")")
        return ret


def basic_query_tokenizer(x:str, strtok=None):
    pas = lisp_to_pas(x)
    # idpreds = set("_cityid _countryid _stateid _riverid _placeid".split(" "))
    # idpreds = set("cityid stateid countryid riverid placeid".split(" "))
    idpreds = set()
    pas = stem_id_words(pas, idpreds, strtok=strtok)[0]
    ret = pas2toks(pas)
    return ret


def tree_query_tokenizer(x:Tree, strtok=None):
    idpreds = set()
    _x = stem_id_words_tree(x, idpreds, strtok=strtok)[0]
    ret = tree2toks(_x)
    return ret


def try_basic_query_tokenizer():
    stemmer = PorterStemmer()
    x = "answer(cityid('new york', _))"
    y = basic_query_tokenizer(x, strtok=lambda x: [stemmer.stem(xe) for xe in x.split()])
    # print(y)


class LCQuaDnoENTDataset(object):
    def __init__(self,
                 p="../../datasets/lcquad/",
                 sentence_encoder:SequenceEncoder=None,
                 min_freq:int=2,
                 splits=None, **kw):
        super(LCQuaDnoENTDataset, self).__init__(**kw)
        self._simplify_filters = True        # if True, filter expressions are converted to orderless and-expressions
        self._initialize(p, sentence_encoder, min_freq)
        self.splits_proportions = splits

    def lines_to_examples(self, lines:List[str]):
        maxsize_before = 0
        avgsize_before = []
        maxsize_after = 0
        avgsize_after = []
        afterstring = set()

        def convert_to_lispstr(_x):
            splits = _x.split()
            assert(sum([1 if xe == "~" else 0 for xe in splits]) == 1)
            assert(splits[1] == "~")
            splits = ["," if xe == "&" else xe for xe in splits]
            pstr = f"{splits[0]} ({' '.join(splits[2:])})"
            return pstr

        ret = []
        ltp = None
        j = 0
        for i, line in enumerate(lines):
            question = line["question"]
            query = line["logical_form"]
            query = convert_to_lispstr(query)
            z, ltp = prolog_to_pas(query, ltp)
            if z is not None:
                ztree = pas_to_tree(z)
                maxsize_before = max(maxsize_before, tree_size(ztree))
                avgsize_before.append(tree_size(ztree))
                lf = ztree
                ret.append((question, lf))
                # print(f"Example {j}:")
                # print(ret[-1][0])
                # print(ret[-1][1])
                # print()
                ltp = None
                maxsize_after = max(maxsize_after, tree_size(lf))
                avgsize_after.append(tree_size(lf))
                j += 1

        avgsize_before = sum(avgsize_before) / len(avgsize_before)
        avgsize_after = sum(avgsize_after) / len(avgsize_after)

        print("Sizes ({j} examples):")
        # print(f"\t Max, Avg size before: {maxsize_before}, {avgsize_before}")
        print(f"\t Max, Avg size: {maxsize_after}, {avgsize_after}")

        return ret

    def _initialize(self, p, sentence_encoder:SequenceEncoder, min_freq:int):
        self.data = {}
        self.sentence_encoder = sentence_encoder

        jp = os.path.join(p, "lcquad_dataset.json")
        with open(jp, "r") as f:
            examples = ujson.load(f)

        examples = self.lines_to_examples(examples)

        questions, queries = tuple(zip(*examples))
        trainlen = int(round(0.8 * len(examples)))
        validlen = int(round(0.1 * len(examples)))
        testlen = int(round(0.1 * len(examples)))
        splits = ["train"] * trainlen + ["valid"] * validlen + ["test"] * testlen
        random.seed(42)
        random.shuffle(splits)
        assert(len(splits) == len(examples))

        self.query_encoder = SequenceEncoder(tokenizer=partial(tree_query_tokenizer, strtok=sentence_encoder.tokenizer), add_end_token=True)

        # build vocabularies
        for i, (question, query, split) in enumerate(zip(questions, queries, splits)):
            self.sentence_encoder.inc_build_vocab(question, seen=split=="train")
            self.query_encoder.inc_build_vocab(query, seen=split=="train")
        for word, wordid in self.sentence_encoder.vocab.D.items():
            self.query_encoder.vocab.add_token(word, seen=False)
        self.sentence_encoder.finalize_vocab(min_freq=min_freq)
        self.query_encoder.finalize_vocab(min_freq=min_freq)

        self.build_data(questions, queries, splits)

    def build_data(self, inputs:Iterable[str], outputs:Iterable[str], splits:Iterable[str]):
        maxlen_in, maxlen_out = 0, 0
        eid = 0
        for inp, out, split in zip(inputs, outputs, splits):
            state = TreeDecoderState([inp], [out], self.sentence_encoder, self.query_encoder)
            state.eids = np.asarray([eid], dtype="int64")
            maxlen_in, maxlen_out = max(maxlen_in, len(state.inp_tokens[0])), max(maxlen_out, len(state.gold_tokens[0]))
            if split not in self.data:
                self.data[split] = []
            self.data[split].append(state)
            eid += 1
        self.maxlen_input, self.maxlen_output = maxlen_in, maxlen_out

    def get_split(self, split:str):
        splits = split.split("+")
        data = []
        for split in splits:
            data += self.data[split]
        return DatasetSplitProxy(data)

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
            dl = DataLoader(self.get_split(split), batch_size=batsize, shuffle=split in ("train", "train+valid"),
             collate_fn=type(self).collate_fn)
            return dl


class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item].make_copy()

    def __len__(self):
        return len(self.data)


def try_dataset():
    tt = q.ticktock("dataset")
    tt.tick("building dataset")
    ds = LCQuaDnoENTDataset(sentence_encoder=SequenceEncoder(tokenizer=lambda x: x.split()), splits=(80, 10, 10))
    train_dl = ds.dataloader("train+valid", batsize=20)
    test_dl = ds.dataloader("test", batsize=20)
    examples = set()
    examples_list = []
    duplicates = []
    testexamples = set()
    testexamples_list = []
    testduplicates = []
    for b in train_dl:
        for i in range(len(b)):
            example = b.inp_strings[i] + " --> " + str(b.gold_trees[i])
            if example in examples:
                duplicates.append(example)
            examples.add(example)
            examples_list.append(example)
            # print(example)
    for b in test_dl:
        for i in range(len(b)):
            example = b.inp_strings[i] + " --> " + str(b.gold_trees[i])
            if example in examples:
                testduplicates.append(example)
            testexamples.add(example)
            testexamples_list.append(example)

    print(f"duplicates within train: {len(duplicates)} from {len(examples_list)} total")
    print(f"duplicates from test to train: {len(testduplicates)} from {len(testexamples_list)} total:")
    for x in testduplicates:
        print(x)
    tt.tock("dataset built")


class BasicGenModel(TransitionModel):
    def __init__(self, embdim, hdim, numlayers:int=1, dropout=0.,
                 sentence_encoder:SequenceEncoder=None,
                 query_encoder:SequenceEncoder=None,
                 feedatt=False, store_attn=True, **kw):
        super(BasicGenModel, self).__init__(**kw)

        inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids(), embdim, padding_idx=0)

        # _, covered_word_ids = load_pretrained_embeddings(inpemb.emb, sentence_encoder.vocab.D,
        #                                                  p="../../data/glove/glove300uncased")  # load glove embeddings where possible into the inner embedding class
        # inpemb._do_rare(inpemb.rare_token_ids - covered_word_ids)
        self.inp_emb = inpemb

        encoder_dim = hdim * 2
        encoder = LSTMEncoder(embdim, hdim, num_layers=numlayers, dropout=dropout, bidirectional=True)
        # encoder = q.LSTMEncoder(embdim, *([encoder_dim // 2] * numlayers), bidir=True, dropout_in=dropout)
        self.inp_enc = encoder

        decoder_emb = torch.nn.Embedding(query_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
        self.out_emb = decoder_emb

        dec_rnn_in_dim = embdim + (encoder_dim if feedatt else 0)
        decoder_rnn = LSTMTransition(dec_rnn_in_dim, hdim, numlayers, dropout=dropout)
        self.out_rnn = decoder_rnn

        decoder_out = BasicGenOutput(hdim + encoder_dim, vocab=query_encoder.vocab)
        # decoder_out.build_copy_maps(inp_vocab=sentence_encoder.vocab)
        self.out_lin = decoder_out

        self.att = q.Attention(q.SimpleFwdAttComp(hdim, encoder_dim, hdim), dropout=min(0.1, dropout))

        self.enc_to_dec = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(encoder_dim, hdim),
            torch.nn.Tanh()
        ) for _ in range(numlayers)])

        self.feedatt = feedatt
        self.nocopy = True

        self.store_attn = store_attn

        self.reset_parameters()

    def reset_parameters(self):
        def _param_reset(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    torch.nn.init.uniform_(m.bias, -0.1, 0.1)
            elif type(m) in (torch.nn.LSTM, torch.nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name or "bias" in name:
                        torch.nn.init.uniform_(param, -0.1, 0.1)
            elif type(m) == torch.nn.Embedding:
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                torch.nn.init.constant_(m.weight[0], 0)
        self.apply(_param_reset)

    def forward(self, x:State):
        if not "mstate" in x:
            x.mstate = State()
        mstate = x.mstate
        init_states = []
        if not "ctx" in mstate:
            # encode input
            inptensor = x.inp_tensor
            mask = inptensor != 0
            inpembs = self.inp_emb(inptensor)
            # inpembs = self.dropout(inpembs)
            inpenc, final_encs = self.inp_enc(inpembs, mask)
            for i, final_enc in enumerate(final_encs):    # iter over layers
                _fenc = self.enc_to_dec[i](final_enc[0])
                init_states.append(_fenc)
            mstate.ctx = inpenc
            mstate.ctx_mask = mask

        ctx = mstate.ctx
        ctx_mask = mstate.ctx_mask

        emb = self.out_emb(x.prev_actions)

        if not "rnnstate" in mstate:
            init_rnn_state = self.out_rnn.get_init_state(emb.size(0), emb.device)
            # uncomment next line to initialize decoder state with last state of encoder
            # init_rnn_state[f"{len(init_rnn_state)-1}"]["c"] = final_enc
            if len(init_states) == init_rnn_state.c.size(1):
                init_rnn_state.h = torch.stack(init_states, 1).contiguous()
            mstate.rnnstate = init_rnn_state

        if "prev_summ" not in mstate:
            mstate.prev_summ = torch.zeros_like(ctx[:, 0])
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

        if self.store_attn:
            if "stored_attentions" not in x:
                x.stored_attentions = torch.zeros(alphas.size(0), 0, alphas.size(1), device=alphas.device)
            x.stored_attentions = torch.cat([x.stored_attentions, alphas.detach()[:, None, :]], 1)

        return outs[0], x


def do_rare_stats(ds, sentence_rare_tokens=None, query_rare_tokens=None):
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
            _sentence_rare_tokens = example.sentence_encoder.vocab.rare_tokens if sentence_rare_tokens is None else sentence_rare_tokens
            if len(set(question_tokens) & _sentence_rare_tokens) > 0:
                rare_in_question += 1
                either = True
            else:
                both = False
            _query_rare_tokens = example.query_encoder.vocab.rare_tokens if query_rare_tokens is None else query_rare_tokens
            if len(set(query_tokens) & _query_rare_tokens) > 0:
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
        if x[i] == D.endtoken:
            x = x[:i]
            break
        elif x[i] == "(" or x[i][-1] == "(":
            parentheses_balance += 1
        elif x[i] == ")":
            parentheses_balance -= 1
        else:
            pass

    # balance parentheses
    while parentheses_balance > 0:
        x.append(")")
        parentheses_balance -= 1
    i = len(x) - 1
    while parentheses_balance < 0 and i > 0:
        if x[i] == ")":
            x.pop(i)
            parentheses_balance += 1
        i -= 1

    # convert to nltk.Tree
    try:
        tree, parsestate = lisp_to_tree(" ".join(x), None)
    except Exception as e:
        tree = None
    return tree


def split_tokenizer(x):
    x = x.replace("?", " ?"). \
        replace(".", " ."). \
        replace(",", " ,"). \
        replace("'", " '")
    x = re.sub("\s+", " ", x)
    return x.lower().split()


def run(lr=0.001,
        batsize=50,
        epochs=50,
        embdim=100,
        encdim=100,
        numlayers=1,
        beamsize=1,
        dropout=.2,
        wreg=1e-10,
        cuda=False,
        gpu=0,
        minfreq=3,
        gradnorm=3.,
        cosine_restarts=1.,
        ):
    localargs = locals().copy()
    print(locals())
    tt = q.ticktock("script")
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt.tick("loading data")
    ds = LCQuaDnoENTDataset(sentence_encoder=SequenceEncoder(tokenizer=split_tokenizer), min_freq=minfreq)
    print(f"max lens: {ds.maxlen_input} (input) and {ds.maxlen_output} (output)")
    tt.tock("data loaded")

    do_rare_stats(ds)
    # batch = next(iter(train_dl))
    # print(batch)
    # print("input graph")
    # print(batch.batched_states)

    model = BasicGenModel(embdim=embdim, hdim=encdim, dropout=dropout, numlayers=numlayers,
                             sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder, feedatt=True)

    # sentence_rare_tokens = set([ds.sentence_encoder.vocab(i) for i in model.inp_emb.rare_token_ids])
    # do_rare_stats(ds, sentence_rare_tokens=sentence_rare_tokens)

    tfdecoder = SeqDecoder(TFTransition(model),
                           [CELoss(ignore_index=0, mode="logprobs"),
                            SeqAccuracies(), TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab),
                                                          orderless={"select", "count", "ask"})])
    # beamdecoder = BeamActionSeqDecoder(tfdecoder.model, beamsize=beamsize, maxsteps=50)
    freedecoder = SeqDecoder(FreerunningTransition(model, maxtime=40),
                             eval=[SeqAccuracies(),
                                   TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab),
                                                orderless={"select", "count", "ask"})])
    # freedecoder = BeamDecoder(model, maxtime=50, beamsize=beamsize,
    #                           eval=[SeqAccuracies()],
    #                           eval_beam=[TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab),
    #                                              orderless={"select", "count", "ask"})])

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
    vlosses = make_loss_array("seq_acc", "tree_acc")
    # if beamsize >= 3:
    #     vlosses = make_loss_array("seq_acc", "tree_acc", "tree_acc_at3", "tree_acc_at_last")
    # else:
    #     vlosses = make_loss_array("seq_acc", "tree_acc", "tree_acc_at_last")

    # trainable_params = tfdecoder.named_parameters()
    # exclude_params = set()
    # exclude_params.add("model.model.inp_emb.emb.weight")   # don't train input embeddings if doing glove
    # trainable_params = [v for k, v in trainable_params if k not in exclude_params]

    # 4. define optim
    # optim = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wreg)
    optim = torch.optim.Adam(tfdecoder.parameters(), lr=lr, weight_decay=wreg)

    # lr schedule
    if cosine_restarts >= 0:
        # t_max = epochs * len(train_dl)
        t_max = epochs
        print(f"Total number of updates: {t_max}")
        lr_schedule = q.WarmupCosineWithHardRestartsSchedule(optim, 0, t_max, cycles=cosine_restarts)
        reduce_lr = [lambda: lr_schedule.step()]
    else:
        reduce_lr = []

    # 6. define training function
    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(tfdecoder.parameters(), gradnorm)
    # clipgradnorm = lambda: None
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=tfdecoder, dataloader=ds.dataloader("train", batsize), optim=optim, losses=losses,
                         _train_batch=trainbatch, device=device, on_end=reduce_lr)

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=freedecoder, dataloader=ds.dataloader("valid", batsize), losses=vlosses, device=device)
    # validepoch = partial(q.test_epoch, model=freedecoder, dataloader=valid_dl, losses=vlosses, device=device)

    # p = q.save_run(freedecoder, localargs, filepath=__file__)
    # q.save_dataset(ds, p)
    # _freedecoder, _localargs = q.load_run(p)
    # _ds = q.load_dataset(p)
    # sys.exit()

    # 7. run training
    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done training")

    # testing
    tt.tick("testing")
    testresults = q.test_epoch(model=freedecoder, dataloader=ds.dataloader("valid", batsize), losses=vlosses, device=device)
    print("validation test results: ", testresults)
    tt.tock("tested")
    tt.tick("testing")
    testresults = q.test_epoch(model=freedecoder, dataloader=ds.dataloader("test", batsize), losses=vlosses, device=device)
    print("test results: ", testresults)
    tt.tock("tested")

    # save model?
    tosave = input("Save this model? 'y(es)'=Yes, <int>=overwrite previous, otherwise=No) \n>")
    if tosave.lower() == "y" or tosave.lower() == "yes" or re.match("\d+", tosave.lower()):
        overwrite = int(tosave) if re.match("\d+", tosave) else None
        p = q.save_run(model, localargs, filepath=__file__, overwrite=overwrite)
        q.save_dataset(ds, p)
        _model, _localargs = q.load_run(p)
        _ds = q.load_dataset(p)

        _freedecoder = BeamDecoder(_model, maxtime=50, beamsize=beamsize,
                                  eval_beam=[TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab),
                                                          orderless={"op:and", "SW:concat"})])

        # testing
        tt.tick("testing reloaded")
        _testresults = q.test_epoch(model=_freedecoder, dataloader=_ds.dataloader("test", batsize),
                                    losses=vlosses, device=device)
        print(_testresults)
        assert(testresults == _testresults)
        tt.tock("tested")


class ScoreModel(ABC, torch.nn.Module):
    def forward(self, inp_tensor:torch.Tensor, cand_tensors:torch.Tensor, attn_tensors:torch.Tensor=None):
        """
        :param inp_tensor:      (batsize, inpseqlen) tensor of int ids
        :param cand_tensors:    (batsize, beamsize, outseqlen) tensor of int ids
        :param attn_tensors:    (batsize, beamsize, outseqlen, inpseqlen) tensor of attention probs
        :return:                (batsize, beamsize) scores or probabilities
        """
        pass


class SimpleScoreModel(ScoreModel):
    """ Simple scoring model that uses two encoders and produces .dot-based scores. """
    def __init__(self, inpemb, inpenc, outemb, outenc, comp, **kw):
        super(SimpleScoreModel, self).__init__(**kw)
        self.inpemb, self.outemb = inpemb, outemb
        self.inpenc, self.outenc = inpenc, outenc

        self.comp = comp

    def forward(self, inp_tensor:torch.Tensor, cand_tensors:torch.Tensor, attn_tensors:torch.Tensor=None):
        mask = inp_tensor != 0
        inp_embedded = self.inpemb(inp_tensor)
        inp_encoding = self.inpenc(inp_embedded, mask=mask)

        out_encs = []
        for i in range(cand_tensors.size(1)):
            mask = cand_tensors[:, i] != 0
            out_embedded = self.outemb(cand_tensors[:, i])
            out_encoding = self.outenc(out_embedded, mask=mask)
            out_encs.append(out_encoding)
        out_encs = torch.stack(out_encs, 1)     # (batsize, beamsize, encdim)

        scores = self.comp(inp_encoding, out_encs)
        return scores


class BeamReranker(TransitionModel):
    def __init__(self, genmodel, scoremodel, beamsize=5, maxtime=100,
                 copy_deep=False, endid=3, use_beamcache=True, **kw):
        super(BeamReranker, self).__init__(**kw)
        self.genmodel = genmodel
        self.beammodel = BeamTransition(self.genmodel, beamsize=beamsize, maxtime=maxtime, copy_deep=copy_deep)
        self.scoremodel = scoremodel
        self.endid = endid
        self._beamcache_actions = {}
        self._beamcache_attentions = {}         # TODO: cache attentions too
        self._beamcache_complete = False
        self._use_beamcache = use_beamcache

    def finalize_beamcache(self):
        numex = max(self._beamcache_actions.keys()) + 1
        ex = self._beamcache_actions[0]
        beamsize, seqlen = ex.size(1), ex.size(2)
        beamcache = torch.zeros(numex, beamsize, seqlen, dtype=torch.long, device=ex.device)
        for k in self._beamcache_actions:
            beamcache[k] = self._beamcache_actions[k]
        self._beamcache_actions = beamcache
        self._beamcache_complete = True

    def forward(self, x:State):
        _x = x
        # run genmodel in beam decoder in non-training mode
        if self._beamcache_complete and "eids" in x and self._use_beamcache:
            eids = torch.tensor(x.eids, device=self._beamcache_actions.device)
            predactions = self._beamcache_actions[eids]
        else:
            with torch.no_grad():
                self.beammodel.eval()
                x.start_decoding()
                i = 0
                all_terminated = x.all_terminated()
                while not all_terminated:
                    outprobs, predactions, x, all_terminated = self.beammodel(x, timestep=i)
                    i += 1
                if not self._beamcache_complete  and "eids" in _x and self._use_beamcache:
                    for j, eid in enumerate(list(_x.eids)):
                        self._beamcache_actions[eid] = predactions[j:j+1]

        # if training, add gold to beam
        if self.training:
            golds = _x.get_gold()
            # align gold dims
            pass    # TODO

        # replace everything after end id with padding (0)
        # TODO

        # run scoring model
        # TODO

        # re-arrange the beam
        # TODO

        # run eval


class LSTMEncoderWrapper(torch.nn.Module):
    def __init__(self, lstmencoder, **kw):
        super(LSTMEncoderWrapper, self).__init__(**kw)
        self.core = lstmencoder

    def forward(self, x, mask=None):
        enc, finalenc = self.core(x, mask)
        return finalenc


class DotSimilarity(torch.nn.Module):
    def forward(self, x, y):
        """
        :param x:   (batsize, encdim)
        :param y:   (batsize, beamsize, encdim)
        :return:    (batsize, beamsize)
        """
        scores = torch.einsum("xd,xbd->xb", x, y)
        return scores


def run_rerank(lr=0.001,
        batsize=20,
        epochs=1,
        embdim=301,     # not used
        encdim=200,
        numlayers=1,
        beamsize=5,
        dropout=.2,
        wreg=1e-10,
        cuda=False,
        gpu=0,
        minfreq=2,
        gradnorm=3.,
        cosine_restarts=1.,
        domain="restaurants",
        gensavedp="overnight_basic/run{}",
        genrunid=1,
        ):
    localargs = locals().copy()
    print(locals())
    gensavedrunp = gensavedp.format(genrunid)
    tt = q.ticktock("script")
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt.tick("loading data")
    ds = q.load_dataset(gensavedrunp)
    # ds = OvernightDataset(domain=domain, sentence_encoder=SequenceEncoder(tokenizer=split_tokenizer), min_freq=minfreq)
    print(f"max lens: {ds.maxlen_input} (input) and {ds.maxlen_output} (output)")
    tt.tock("data loaded")

    do_rare_stats(ds)
    # batch = next(iter(train_dl))
    # print(batch)
    # print("input graph")
    # print(batch.batched_states)

    genmodel, genargs = q.load_run(gensavedrunp)
    # BasicGenModel(embdim=embdim, hdim=encdim, dropout=dropout, numlayers=numlayers,
    #                          sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder, feedatt=True)

    # sentence_rare_tokens = set([ds.sentence_encoder.vocab(i) for i in model.inp_emb.rare_token_ids])
    # do_rare_stats(ds, sentence_rare_tokens=sentence_rare_tokens)

    inpenc = q.LSTMEncoder(embdim, *([encdim // 2] * numlayers), bidir=True, dropout_in=dropout)
    outenc = q.LSTMEncoder(embdim, *([encdim // 2] * numlayers), bidir=True, dropout_in=dropout)
    scoremodel = SimpleScoreModel(genmodel.inp_emb, genmodel.out_emb,
                                  LSTMEncoderWrapper(inpenc), LSTMEncoderWrapper(outenc),
                                  DotSimilarity())

    model = BeamReranker(genmodel, scoremodel, beamsize=beamsize, maxtime=50)

    # todo: run over whole dataset to populate beam cache
    testbatch = next(iter(ds.dataloader("train", batsize=2)))
    model(testbatch)

    sys.exit()

    tfdecoder = SeqDecoder(TFTransition(model),
                           [CELoss(ignore_index=0, mode="logprobs"),
                            SeqAccuracies(), TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab),
                                                          orderless={"op:and", "SW:concat"})])
    # beamdecoder = BeamActionSeqDecoder(tfdecoder.model, beamsize=beamsize, maxsteps=50)
    freedecoder = BeamDecoder(model, maxtime=50, beamsize=beamsize,
                              eval_beam=[TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab),
                                                 orderless={"op:and", "SW:concat"})])

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

    losses = make_loss_array("loss", "seq_acc", "tree_acc")
    vlosses = make_loss_array("tree_acc", "tree_acc_at3", "tree_acc_at_last")

    trainable_params = tfdecoder.named_parameters()
    exclude_params = {"model.model.inp_emb.emb.weight"}   # don't train input embeddings if doing glove
    trainable_params = [v for k, v in trainable_params if k not in exclude_params]

    # 4. define optim
    optim = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wreg)
    # optim = torch.optim.SGD(tfdecoder.parameters(), lr=lr, weight_decay=wreg)

    # lr schedule
    if cosine_restarts >= 0:
        # t_max = epochs * len(train_dl)
        t_max = epochs
        print(f"Total number of updates: {t_max}")
        lr_schedule = q.WarmupCosineWithHardRestartsSchedule(optim, 0, t_max, cycles=cosine_restarts)
        reduce_lr = [lambda: lr_schedule.step()]
    else:
        reduce_lr = []

    # 6. define training function
    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(tfdecoder.parameters(), gradnorm)
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=tfdecoder, dataloader=ds.dataloader("train", batsize), optim=optim, losses=losses,
                         _train_batch=trainbatch, device=device, on_end=reduce_lr)

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=freedecoder, dataloader=ds.dataloader("valid", batsize), losses=vlosses, device=device)
    # validepoch = partial(q.test_epoch, model=freedecoder, dataloader=valid_dl, losses=vlosses, device=device)

    # p = q.save_run(freedecoder, localargs, filepath=__file__)
    # q.save_dataset(ds, p)
    # _freedecoder, _localargs = q.load_run(p)
    # _ds = q.load_dataset(p)
    # sys.exit()

    # 7. run training
    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done training")

    # testing
    tt.tick("testing")
    testresults = q.test_epoch(model=freedecoder, dataloader=ds.dataloader("test", batsize), losses=vlosses, device=device)
    print(testresults)
    tt.tock("tested")

    # save model?
    tosave = input("Save this model? 'y(es)'=Yes, <int>=overwrite previous, otherwise=No) \n>")
    if tosave.lower() == "y" or tosave.lower() == "yes" or re.match("\d+", tosave.lower()):
        overwrite = int(tosave) if re.match("\d+", tosave) else None
        p = q.save_run(model, localargs, filepath=__file__, overwrite=overwrite)
        q.save_dataset(ds, p)
        _model, _localargs = q.load_run(p)
        _ds = q.load_dataset(p)

        _freedecoder = BeamDecoder(_model, maxtime=50, beamsize=beamsize,
                                  eval_beam=[TreeAccuracy(tensor2tree=partial(tensor2tree, D=ds.query_encoder.vocab),
                                                          orderless={"op:and", "SW:concat"})])

        # testing
        tt.tick("testing reloaded")
        _testresults = q.test_epoch(model=_freedecoder, dataloader=_ds.dataloader("test", batsize),
                                    losses=vlosses, device=device)
        print(_testresults)
        assert(testresults == _testresults)
        tt.tock("tested")



if __name__ == '__main__':
    # try_basic_query_tokenizer()
    # try_build_grammar()
    # try_dataset()
    q.argprun(run)
    # q.argprun(run_rerank)