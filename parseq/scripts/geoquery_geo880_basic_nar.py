import os
import re
import sys
from functools import partial
from typing import *

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
from parseq.decoding import SeqDecoder, TFTransition, FreerunningTransition, merge_dicts
from parseq.eval import CELoss, SeqAccuracies, make_loss_array, DerivedAccuracy
from parseq.grammar import prolog_to_pas
from parseq.nn import TokenEmb, BasicGenOutput
from parseq.states import DecodableState, BasicDecoderState, State
from parseq.tm import TransformerConfig, Transformer
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
                return [("_str", pas)]
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
        ret[0] += "("
        for child in children:
            ret += child
            # ret.append(",")
        # ret.pop(-1)
        ret.append(")")
        return ret


def basic_query_tokenizer(x:str, strtok=None):
    pas = prolog_to_pas(x)
    idpreds = set("_cityid _countryid _stateid _riverid _placeid".split(" "))
    idpreds = set("cityid stateid countryid riverid placeid".split(" "))
    pas = stem_id_words(pas, idpreds, strtok=strtok)[0]
    ret = pas2toks(pas)
    return ret

def try_basic_query_tokenizer():
    stemmer = PorterStemmer()
    x = "answer(cityid('new york', _))"
    y = basic_query_tokenizer(x, strtok=lambda x: [stemmer.stem(xe) for xe in x.split()])
    # print(y)


class GeoQueryDataset(object):
    def __init__(self,
                 p="../../datasets/geoquery/",
                 sentence_encoder:SequenceEncoder=None,
                 min_freq:int=2, **kw):
        super(GeoQueryDataset, self).__init__(**kw)
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
        for i, (question, query, split) in enumerate(zip(questions, queries, splits)):
            self.sentence_encoder.inc_build_vocab(question, seen=split=="train")
            self.query_encoder.inc_build_vocab(query, seen=split=="train")
        self.sentence_encoder.finalize_vocab(min_freq=min_freq)
        self.query_encoder.finalize_vocab(min_freq=min_freq)

        self.build_data(questions, queries, splits)

    def build_data(self, inputs:Iterable[str], outputs:Iterable[str], splits:Iterable[str]):
        for inp, out, split in zip(inputs, outputs, splits):
            state = BasicDecoderState([inp], [out], self.sentence_encoder, self.query_encoder)
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
             collate_fn=GeoQueryDataset.collate_fn)
            return dl


def try_dataset():
    tt = q.ticktock("dataset")
    tt.tick("building dataset")
    ds = GeoQueryDataset(sentence_encoder=SequenceEncoder(tokenizer=lambda x: x.split()))
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


class NARTMModel(TransitionModel):
    def __init__(self, tm, out, maxinplen=50, maxoutlen=50, numinpids:int=None, eval=tuple(), **kw):
        super(NARTMModel, self).__init__(**kw)
        self.tm = tm
        self.out = out
        self.maxinplen = maxinplen
        self.maxoutlen = maxoutlen
        self._metrics = eval
        self._numinpids = numinpids

    def forward(self, x:State):
        inpseq = x.inp_tensor
        position_ids = torch.arange(inpseq.size(1), dtype=torch.long, device=inpseq.device)[None, :].repeat(inpseq.size(0), 1)
        inpseq = torch.cat([inpseq, torch.arange(self.maxoutlen, dtype=inpseq.dtype, device=inpseq.device)[None, :].repeat(inpseq.size(0), 1)+self._numinpids], 1)
        position_ids_out = torch.arange(self.maxoutlen, dtype=torch.long, device=inpseq.device)[None, :].repeat(inpseq.size(0), 1) + self.maxinplen
        position_ids = torch.cat([position_ids, position_ids_out], 1)
        attention_mask = (inpseq != 0)
        y = self.tm(inpseq, attention_mask=attention_mask, position_ids=position_ids)
        outprobs = self.out(y[0])
        outprobs = outprobs[:, self.maxinplen:]
        _, predactions = outprobs.max(-1)

        metrics = [metric(outprobs, predactions, x) for metric in self._metrics]
        metrics = merge_dicts(*metrics)
        return metrics, x


def create_model(hdim=128, dropout=0., numlayers:int=1, numheads:int=4,
                 sentence_encoder:SequenceEncoder=None,
                 query_encoder:SequenceEncoder=None,
                 feedatt=False, maxtime=100):
    inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids()+maxtime, hdim, padding_idx=0)
    inpemb = TokenEmb(inpemb, rare_token_ids=sentence_encoder.vocab.rare_ids, rare_id=1)
    tm_config = TransformerConfig(vocab_size=inpemb.emb.num_embeddings, num_attention_heads=numheads,
                                  num_hidden_layers=numlayers, hidden_size=hdim, intermediate_size=hdim*4,
                                  hidden_dropout_prob=dropout)
    tm = Transformer(tm_config)
    tm.embeddings.word_embeddings = inpemb
    decoder_out = BasicGenOutput(hdim, query_encoder.vocab)
    model = NARTMModel(tm, decoder_out, maxinplen=maxtime, maxoutlen=maxtime, numinpids=sentence_encoder.vocab.number_of_ids())
    return model


def do_rare_stats(ds:GeoQueryDataset):
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
            if len(set(question_tokens) & example.sentence_encoder.vocab.rare_tokens) > 0:
                rare_in_question += 1
                either = True
            else:
                both = False
            if len(set(query_tokens) & example.query_encoder.vocab.rare_tokens) > 0:
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

    # introduce comma's
    i = 1
    while i < len(x):
        if x[i-1][-1] == "(":
            pass
        elif x[i] == ")":
            pass
        else:
            x.insert(i, ",")
            i += 1
        i += 1
    return " ".join(x)



def run(lr=0.001,
        batsize=20,
        epochs=100,
        embdim=100,
        encdim=164,
        numlayers=4,
        numheads=4,
        dropout=.0,
        wreg=1e-10,
        cuda=False,
        gpu=0,
        minfreq=2,
        gradnorm=3000.,
        cosine_restarts=1.,
        ):
    print(locals())
    tt = q.ticktock("script")
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt.tick("loading data")
    stemmer = PorterStemmer()
    tokenizer = lambda x: [stemmer.stem(xe) for xe in x.split()]
    ds = GeoQueryDataset(sentence_encoder=SequenceEncoder(tokenizer=tokenizer), min_freq=minfreq)

    train_dl = ds.dataloader("train", batsize=batsize)
    test_dl = ds.dataloader("test", batsize=batsize)
    tt.tock("data loaded")

    do_rare_stats(ds)

    # batch = next(iter(train_dl))
    # print(batch)
    # print("input graph")
    # print(batch.batched_states)

    model = create_model(hdim=encdim, dropout=dropout, numlayers=numlayers, numheads=numheads,
                         sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder)

    model._metrics = [CELoss(ignore_index=0, mode="logprobs"),
                      SeqAccuracies()]

    losses = make_loss_array("loss", "elem_acc", "seq_acc")
    vlosses = make_loss_array("loss", "seq_acc")

    # 4. define optim
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wreg)
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
    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), gradnorm)
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    trainepoch = partial(q.train_epoch, model=model, dataloader=train_dl, optim=optim, losses=losses,
                         _train_batch=trainbatch, device=device, on_end=reduce_lr)

    # 7. define validation function (using partial)
    validepoch = partial(q.test_epoch, model=model, dataloader=test_dl, losses=vlosses, device=device)
    # validepoch = partial(q.test_epoch, model=tfdecoder, dataloader=test_dl, losses=vlosses, device=device)

    # 7. run training
    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done training")



if __name__ == '__main__':
    try_basic_query_tokenizer()
    # try_build_grammar()
    # try_dataset()
    q.argprun(run)