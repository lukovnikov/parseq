import math
import os
import random
import re
import sys
from functools import partial
from typing import *

import torch
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
from parseq.decoding import SeqDecoder, TFTransition, FreerunningTransition, BeamDecoder
from parseq.eval import CELoss, SeqAccuracies, make_loss_array, DerivedAccuracy, TreeAccuracy
from parseq.grammar import prolog_to_pas, lisp_to_pas, pas_to_prolog, pas_to_tree, tree_size, tree_to_prolog, \
    tree_to_lisp, lisp_to_tree
from parseq.nn import TokenEmb, BasicGenOutput, PtrGenOutput, PtrGenOutput2, load_pretrained_embeddings
from parseq.states import DecodableState, BasicDecoderState, State, TreeDecoderState
from parseq.transitions import TransitionModel, LSTMCellTransition
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


class OvernightDataset(object):
    def __init__(self,
                 p="../../datasets/overnightData/",
                 pcache="../../datasets/overnightCache/",
                 domain:str="restaurants",
                 sentence_encoder:SequenceEncoder=None,
                 usecache=True,
                 min_freq:int=2, **kw):
        super(OvernightDataset, self).__init__(**kw)
        self._simplify_filters = True        # if True, filter expressions are converted to orderless and-expressions
        self._pcache = pcache if usecache else None
        self._domain = domain
        self._usecache = usecache
        self._initialize(p, domain, sentence_encoder, min_freq)

    def lines_to_examples(self, lines:List[str]):
        maxsize_before = 0
        avgsize_before = []
        maxsize_after = 0
        avgsize_after = []
        afterstring = set()

        def simplify_tree(t:Tree):
            if t.label() == "call":
                assert(len(t[0]) == 0)
                # if not t[0].label().startswith("SW."):
                #     print(t)
                # assert(t[0].label().startswith("SW."))
                t.set_label(t[0].label())
                del t[0]
            elif t.label() == "string":
                afterstring.update(set([tc.label() for tc in t]))
                assert(len(t) == 1)
                assert(len(t[0]) == 0)
                t.set_label(f"arg:{t[0].label()}")
                del t[0]
            if t.label().startswith("edu.stanford.nlp.sempre.overnight.SimpleWorld."):
                t.set_label("SW:" + t.label()[len("edu.stanford.nlp.sempre.overnight.SimpleWorld."):])
            if t.label() == "SW:getProperty":
                assert(len(t) == 2)
                ret = simplify_tree(t[1])
                ret.append(simplify_tree(t[0]))
                return ret
            elif t.label() == "SW:singleton":
                assert(len(t) == 1)
                assert(len(t[0]) == 0)
                return t[0]
            elif t.label() == "SW:ensureNumericProperty":
                assert(len(t) == 1)
                return simplify_tree(t[0])
            elif t.label() == "SW:ensureNumericEntity":
                assert(len(t) == 1)
                return simplify_tree(t[0])
            elif t.label() == "SW:aggregate":
                assert(len(t) == 2)
                ret = simplify_tree(t[0])
                assert(ret.label() in ["arg:avg", "arg:sum"])
                assert(len(ret) == 0)
                ret.set_label(f"agg:{ret.label()}")
                ret.append(simplify_tree(t[1]))
                return ret
            else:
                t[:] = [simplify_tree(tc) for tc in t]
                return t

        def simplify_further(t):
            """ simplifies filters and count expressions """
            # replace filters with ands
            if t.label() == "SW:filter" and self._simplify_filters is True:
                if len(t) not in (2, 4):
                    raise Exception(f"filter expression should have 2 or 4 children, got {len(children)}")
                children = [simplify_further(tc) for tc in t]
                startset = children[0]
                if len(children) == 2:
                    condition = Tree("cond:has", [children[1]])
                elif len(children) == 4:
                    condition = Tree(f"cond:{children[2].label()}", [children[1], children[3]])
                conditions = [condition]
                if startset.label() == "op:and":
                    conditions = startset[:] + conditions
                else:
                    conditions = [startset] + conditions
                # check for same conditions:
                i = 0
                while i < len(conditions) - 1:
                    j = i + 1
                    while j < len(conditions):
                        if conditions[i] == conditions[j]:
                            print(f"SAME!: {conditions[i]}, {conditions[j]}")
                            del conditions[j]
                            j -= 1
                        j += 1
                    i += 1

                ret = Tree(f"op:and", conditions)
                return ret
            # replace countSuperlatives with specific ones
            elif t.label() == "SW:countSuperlative":
                assert(t[1].label() in ["arg:max", "arg:min"])
                t.set_label(f"SW:CNT-{t[1].label()}")
                del t[1]
                t[:] = [simplify_further(tc) for tc in t]
            elif t.label() == "SW:countComparative":
                assert(t[2].label() in ["arg:<", "arg:<=", "arg:>", "arg:>=", "arg:=", "arg:!="])
                t.set_label(f"SW:CNT-{t[2].label()}")
                del t[2]
                t[:] = [simplify_further(tc) for tc in t]
            else:
                t[:] = [simplify_further(tc) for tc in t]
            return t

        def simplify_furthermore(t):
            """ replace reverse rels"""
            if t.label() == "arg:!type":
                t.set_label("arg:~type")
                return t
            elif t.label() == "SW:reverse":
                assert(len(t) == 1)
                assert(t[0].label().startswith("arg:"))
                assert(len(t[0]) == 0)
                t.set_label(f"arg:~{t[0].label()[4:]}")
                del t[0]
                return t
            elif t.label().startswith("cond:arg:"):
                assert(len(t) == 2)
                head = t[0]
                head = simplify_furthermore(head)
                assert(head.label().startswith("arg:"))
                assert(len(head) == 0)
                headlabel = f"arg:~{head.label()[4:]}"
                headlabel = headlabel.replace("~~", "")
                head.set_label(headlabel)
                body = simplify_furthermore(t[1])
                if t.label()[len("cond:arg:"):] != "=":
                    body = Tree(t.label()[5:], [body])
                head.append(body)
                return head
            else:
                t[:] = [simplify_furthermore(tc) for tc in t]
                return t

        def simplify_final(t):
            assert(t.label() == "SW:listValue")
            assert(len(t) == 1)
            return t[0]

        ret = []
        ltp = None
        j = 0
        for i, line in enumerate(lines):
            z, ltp = lisp_to_pas(line, ltp)
            if z is not None:
                print(f"Example {j}:")
                ztree = pas_to_tree(z[1][2][1][0])
                maxsize_before = max(maxsize_before, tree_size(ztree))
                avgsize_before.append(tree_size(ztree))
                lf = simplify_tree(ztree)
                lf = simplify_further(lf)
                lf = simplify_furthermore(lf)
                lf = simplify_final(lf)
                question = z[1][0][1][0]
                assert(question[0] == '"' and question[-1] == '"')
                ret.append((question[1:-1], lf))
                print(ret[-1][0])
                print(ret[-1][1])
                ltp = None
                maxsize_after = max(maxsize_after, tree_size(lf))
                avgsize_after.append(tree_size(lf))

                print(pas_to_tree(z[1][2][1][0]))
                print()
                j += 1

        avgsize_before = sum(avgsize_before) / len(avgsize_before)
        avgsize_after = sum(avgsize_after) / len(avgsize_after)

        print("Simplification results ({j} examples):")
        print(f"\t Max, Avg size before: {maxsize_before}, {avgsize_before}")
        print(f"\t Max, Avg size after: {maxsize_after}, {avgsize_after}")

        return ret

    def _load_cached(self):
        train_cached = ujson.load(open(os.path.join(self._pcache, f"{self._domain}.train.json"), "r"))
        trainexamples = [(x, Tree.fromstring(y)) for x, y in train_cached]
        test_cached = ujson.load(open(os.path.join(self._pcache, f"{self._domain}.test.json"), "r"))
        testexamples = [(x, Tree.fromstring(y)) for x, y in test_cached]
        print("loaded from cache")
        return trainexamples, testexamples

    def _cache(self, trainexamples:List[Tuple[str, Tree]], testexamples:List[Tuple[str, Tree]]):
        train_cached, test_cached = None, None
        if os.path.exists(os.path.join(self._pcache, f"{self._domain}.train.json")):
            try:
                train_cached = ujson.load(open(os.path.join(self._pcache, f"{self._domain}.train.json"), "r"))
                test_cached = ujson.load(open(os.path.join(self._pcache, f"{self._domain}.test.json"), "r"))
            except (IOError, ValueError) as e:
                pass
        trainexamples = [(x, str(y)) for x, y in trainexamples]
        testexamples = [(x, str(y)) for x, y in testexamples]

        if train_cached != trainexamples:
            with open(os.path.join(self._pcache, f"{self._domain}.train.json"), "w") as f:
                ujson.dump(trainexamples, f, indent=4, sort_keys=True)
        if test_cached != testexamples:
            with open(os.path.join(self._pcache, f"{self._domain}.test.json"), "w") as f:
                ujson.dump(testexamples, f, indent=4, sort_keys=True)
        print("saved in cache")

    def _initialize(self, p, domain, sentence_encoder:SequenceEncoder, min_freq:int):
        self.data = {}
        self.sentence_encoder = sentence_encoder

        trainexamples, testexamples = None, None
        if self._usecache:
            try:
                trainexamples, testexamples = self._load_cached()
            except (IOError, ValueError) as e:
                pass

        if trainexamples is None:

            trainlines = [x.strip() for x in
                         open(os.path.join(p, f"{domain}.paraphrases.train.examples"), "r").readlines()]
            testlines = [x.strip() for x in
                        open(os.path.join(p, f"{domain}.paraphrases.test.examples"), "r").readlines()]

            trainexamples = self.lines_to_examples(trainlines)
            testexamples = self.lines_to_examples(testlines)

            if self._usecache:
                self._cache(trainexamples, testexamples)

        questions, queries = tuple(zip(*(trainexamples + testexamples)))
        trainlen = int(round(0.8 * len(trainexamples)))
        validlen = int(round(0.2 * len(trainexamples)))
        splits = ["train"] * trainlen + ["valid"] * validlen
        random.seed(1223)
        random.shuffle(splits)
        assert(len(splits) == len(trainexamples))
        splits = splits + ["test"] * len(testexamples)

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
        for inp, out, split in zip(inputs, outputs, splits):
            state = TreeDecoderState([inp], [out], self.sentence_encoder, self.query_encoder)
            maxlen_in, maxlen_out = max(maxlen_in, len(state.inp_tokens[0])), max(maxlen_out, len(state.gold_tokens[0]))
            if split not in self.data:
                self.data[split] = []
            self.data[split].append(state)
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
             collate_fn=OvernightDataset.collate_fn)
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
    ds = OvernightDataset(sentence_encoder=SequenceEncoder(tokenizer=lambda x: x.split()))
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
            example = b.inp_strings[i] + " --> " + b.gold_strings[i]
            if example in examples:
                duplicates.append(example)
            examples.add(example)
            examples_list.append(example)
            # print(example)
    for b in test_dl:
        for i in range(len(b)):
            example = b.inp_strings[i] + " --> " + b.gold_strings[i]
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
    def __init__(self, inp_emb, inp_enc, out_emb, out_rnn:LSTMCellTransition,
                 out_lin, att, enc_to_dec=None, feedatt=False, nocopy=False, **kw):
        super(BasicGenModel, self).__init__(**kw)
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
            # inpembs = self.dropout(inpembs)
            inpenc, final_enc = self.inp_enc(inpembs, mask)
            final_enc = final_enc.view(final_enc.size(0), -1).contiguous()
            final_enc = self.enc_to_dec(final_enc)
            mstate.ctx = inpenc
            mstate.ctx_mask = mask

        ctx = mstate.ctx
        ctx_mask = mstate.ctx_mask

        emb = self.out_emb(x.prev_actions)

        if not "rnnstate" in mstate:
            init_rnn_state = self.out_rnn.get_init_state(emb.size(0), emb.device)
            # uncomment next line to initialize decoder state with last state of encoder
            # init_rnn_state[f"{len(init_rnn_state)-1}"]["c"] = final_enc
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
        return outs[0], x


def create_model(embdim=300, hdim=100, dropout=0., numlayers:int=1,
                 sentence_encoder:SequenceEncoder=None,
                 query_encoder:SequenceEncoder=None,
                 feedatt=False, nocopy=False):
    inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids(), 300, padding_idx=0)
    inpemb = TokenEmb(inpemb, adapt_dims=(300, embdim), rare_token_ids=sentence_encoder.vocab.rare_ids, rare_id=1)
    _, covered_word_ids = load_pretrained_embeddings(inpemb.emb, sentence_encoder.vocab.D, p="../../data/glove/glove300uncased")     # load glove embeddings where possible into the inner embedding class
    inpemb._do_rare(inpemb.rare_token_ids - covered_word_ids)
    # TODO: use fasttext
    encoder_dim = hdim
    encoder = q.LSTMEncoder(embdim, *([encoder_dim // 2]*numlayers), bidir=True, dropout_in=dropout)
    # encoder = PytorchSeq2SeqWrapper(
    #     torch.nn.LSTM(embdim, hdim, num_layers=numlayers, bidirectional=True, batch_first=True,
    #                   dropout=dropout))
    decoder_emb = torch.nn.Embedding(query_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
    decoder_emb = TokenEmb(decoder_emb, rare_token_ids=query_encoder.vocab.rare_ids, rare_id=1)
    dec_rnn_in_dim = embdim + (encoder_dim if feedatt else 0)
    decoder_rnn = [torch.nn.LSTMCell(dec_rnn_in_dim, hdim)]
    for i in range(numlayers - 1):
        decoder_rnn.append(torch.nn.LSTMCell(hdim, hdim))
    decoder_rnn = LSTMCellTransition(*decoder_rnn, dropout=dropout)
    # decoder_out = BasicGenOutput(hdim + encoder_dim, query_encoder.vocab)
    decoder_out = PtrGenOutput2(hdim + encoder_dim, out_vocab=query_encoder.vocab)
    decoder_out.build_copy_maps(inp_vocab=sentence_encoder.vocab)
    attention = q.Attention(q.MatMulDotAttComp(hdim, encoder_dim))
    enctodec = torch.nn.Sequential(
        torch.nn.Linear(encoder_dim, hdim),
        torch.nn.Tanh()
    )
    model = BasicGenModel(inpemb, encoder,
                          decoder_emb, decoder_rnn, decoder_out,
                          attention,
                          enc_to_dec=enctodec,
                          feedatt=feedatt, nocopy=nocopy)
    return model


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
        if x[i] ==D.endtoken:
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
        tree, parsestate = lisp_to_tree(" ".join(x), "empty")
    except Exception as e:
        tree = None
    return tree



def run(lr=0.001,
        batsize=20,
        epochs=100,
        embdim=301,
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
        ):
    print(locals())
    tt = q.ticktock("script")
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt.tick("loading data")
    tokenizer = lambda x: x.split()
    ds = OvernightDataset(domain=domain, sentence_encoder=SequenceEncoder(tokenizer=tokenizer), min_freq=minfreq)
    print(f"max lens: {ds.maxlen_input} (input) and {ds.maxlen_output} (output)")
    train_dl = ds.dataloader("train", batsize=batsize)
    fulltrain_dl = ds.dataloader("train+valid", batsize=batsize)
    valid_dl = ds.dataloader("valid", batsize=batsize)
    test_dl = ds.dataloader("test", batsize=batsize)
    tt.tock("data loaded")

    do_rare_stats(ds)

    # batch = next(iter(train_dl))
    # print(batch)
    # print("input graph")
    # print(batch.batched_states)

    model = create_model(embdim=embdim, hdim=encdim, dropout=dropout, numlayers=numlayers,
                             sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder, feedatt=True)

    sentence_rare_tokens = set([ds.sentence_encoder.vocab(i) for i in model.inp_emb.rare_token_ids])
    do_rare_stats(ds, sentence_rare_tokens=sentence_rare_tokens)

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
        print(f"Total number of updates: {t_max} ({epochs} * {len(train_dl)})")
        lr_schedule = q.WarmupCosineWithHardRestartsSchedule(optim, 0, t_max, cycles=cosine_restarts)
        reduce_lr = [lambda: lr_schedule.step()]
    else:
        reduce_lr = []

    # 6. define training function
    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(tfdecoder.parameters(), gradnorm)
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=tfdecoder, dataloader=fulltrain_dl, optim=optim, losses=losses,
                         _train_batch=trainbatch, device=device, on_end=reduce_lr)

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=freedecoder, dataloader=test_dl, losses=vlosses, device=device)
    # validepoch = partial(q.test_epoch, model=freedecoder, dataloader=valid_dl, losses=vlosses, device=device)

    # 7. run training
    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done training")


if __name__ == '__main__':
    # try_basic_query_tokenizer()
    # try_build_grammar()
    # try_dataset()
    q.argprun(run)