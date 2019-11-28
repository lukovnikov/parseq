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
from parseq.decoding import SeqDecoder, TFTransition, FreerunningTransition
from parseq.eval import StateLoss, StateMetric
from parseq.grammar import prolog_to_pas
from parseq.nn import TokenEmb, BasicGenOutput
from parseq.states import BasicState, batch, StateBatch, DecodableStateBatch, DecodableState
from parseq.transitions import TransitionModel, LSTMCellTransition
from parseq.vocab import SentenceEncoder


def stem_id_words(pas, idparents, stem=False, stemmer=None):
    if stem is True:
        assert(not isinstance(pas, tuple))
    if not isinstance(pas, tuple):
        if stem is True:
            return stemmer.stem(pas)
        else:
            return pas
    else:
        tostem = pas[0] in idparents
        children = [stem_id_words(k, idparents, stem=tostem, stemmer=stemmer)
                    for k in pas[1]]
        return (pas[0], children)


def pas2toks(pas):
    if not isinstance(pas, tuple):
        return [pas]
    else:
        children = [pas2toks(k) for k in pas[1]]
        ret = [pas[0]] if pas[0] != "@NAMELESS@" else []
        ret.append("(")
        for child in children:
            ret += child
            # ret.append(",")
        # ret.pop(-1)
        ret.append(")")
        return ret


def basic_query_tokenizer(x:str, stem=True):
    stemmer = PorterStemmer()
    pas = prolog_to_pas(x)
    idpreds = set("_cityid _countryid _stateid _riverid _placeid".split(" "))
    pas = stem_id_words(pas, idpreds, stemmer=stemmer)
    ret = pas2toks(pas)
    return ret


class GeoQueryDataset(object):
    def __init__(self,
                 geoquery_path:str="../../data/geo880/",
                 train_file:str="geo880_train600.tsv",
                 test_file:str="geo880_test280.tsv",
                 sentence_encoder:SentenceEncoder=None,
                 min_freq:int=2,
                 **kw):
        super(GeoQueryDataset, self).__init__(**kw)
        self.data = {}

        self.sentence_encoder = sentence_encoder

        train_lines = [x.strip() for x in open(os.path.join(geoquery_path, train_file), "r").readlines()]
        test_lines = [x.strip() for x in open(os.path.join(geoquery_path, test_file), "r").readlines()]
        train_pairs = [x.split("\t") for x in train_lines]
        test_pairs = [x.split("\t") for x in test_lines]
        inputs = [x[0].strip() for x in train_pairs] + [x[0] for x in test_pairs]
        outputs = [x[1] for x in train_pairs] + [x[1] for x in test_pairs]
        outputs = [x.replace("' ", "") for x in outputs]
        split_infos = ["train" for _ in train_pairs] + ["test" for _ in test_pairs]

        # build input vocabulary
        for i, (inp, split_id) in enumerate(zip(inputs, split_infos)):
            self.sentence_encoder.inc_build_vocab(inp, seen=split_id == "train")
        self.sentence_encoder.finalize_vocab(min_freq=min_freq)

        self.query_encoder = SentenceEncoder(tokenizer=basic_query_tokenizer, add_end_token=True)

        # build output vocabulary
        for i, (out, split_id) in enumerate(zip(outputs, split_infos)):
            self.query_encoder.inc_build_vocab(out, seen=split_id == "train")
        self.query_encoder.finalize_vocab(min_freq=min_freq)

        self.build_data(inputs, outputs, split_infos)

    def build_data(self, inputs:Iterable[str], outputs:Iterable[str], splits:Iterable[str]):
        for inp, out, split in zip(inputs, outputs, splits):
            if split not in self.data:
                self.data[split] = []
            self.data[split].append((inp, out))

    def get_split(self, split:str):
        return DatasetSplitProxy(self.data[split])

    @staticmethod
    def collate_fn(data:Iterable):
        goldmaxlen = 0
        inpmaxlen = 0
        for state in data:
            goldmaxlen = max(goldmaxlen, state.gold_tensor.size(0))
            inpmaxlen = max(inpmaxlen, state.nn_state.inp_tensor.size(0))
        for state in data:
            state.gold_tensor = torch.cat([
                state.gold_tensor,
                state.gold_tensor.new_zeros(goldmaxlen - state.gold_tensor.size(0))], 0)
            state.nn_state.inp_tensor = torch.cat([
                state.nn_state.inp_tensor,
                state.nn_state.inp_tensor.new_zeros(inpmaxlen - state.nn_state.inp_tensor.size(0))], 0)
        ret = batch(data)
        return ret


def try_dataset():
    tt = q.ticktock("dataset")
    tt.tick("building dataset")
    ds = GeoQueryDataset(sentence_encoder=SentenceEncoder(tokenizer=lambda x: x.split()))
    tt.tock("dataset built")


class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item].make_copy()

    def __len__(self):
        return len(self.data)


def get_dataloaders(ds:GeoQueryDataset, batsize:int=5):
    dls = {}
    for split in ds.data.keys():
        dls[split] = DataLoader(ds.get_split(split), batch_size=batsize, shuffle=split=="train",
             collate_fn=GeoQueryDataset.collate_fn)
    return dls


class BasicGenModel(TransitionModel):
    def __init__(self, inp_emb, inp_enc, out_emb, out_rnn:LSTMCellTransition,
                 out_lin, att, dropout=0., enc_to_dec=None, feedatt=False, **kw):
        super(BasicGenModel, self).__init__(**kw)
        self.inp_emb, self.inp_enc = inp_emb, inp_enc
        self.out_emb, self.out_rnn, self.out_lin = out_emb, out_rnn, out_lin
        self.enc_to_dec = enc_to_dec
        self.att = att
        # self.ce = q.CELoss(reduction="none", ignore_index=0, mode="probs")
        self.dropout = torch.nn.Dropout(dropout)
        self.feedatt = feedatt

    def forward(self, x:StateBatch):
        if "ctx" not in x.nn_state:
            # encode input
            inptensor = x.nn_state["inp_tensor"]
            mask = inptensor != 0
            inpembs = self.inp_emb(inptensor)
            # inpembs = self.dropout(inpembs)
            inpenc, final_enc = self.inp_enc(inpembs, mask)
            final_enc = final_enc.view(final_enc.size(0), -1).contiguous()
            final_enc = self.enc_to_dec(final_enc)
            x.nn_state["ctx"] = inpenc
            x.nn_state["ctx_mask"] = mask

        ctx = x.nn_state["ctx"]
        ctx_mask = x.nn_state["ctx_mask"]

        emb = self.out_emb(x.nn_state["prev_action"])

        if "rnn" not in x:
            init_rnn_state = self.out_rnn.get_init_state(emb.size(0), emb.device)
            # uncomment next line to initialize decoder state with last state of encoder
            # init_rnn_state[f"{len(init_rnn_state)-1}"]["c"] = final_enc
            x.nn_state["rnn"] = init_rnn_state

        # DONE: concat previous attention summary to emb
        if "prev_summ" not in x:
            x.nn_state["prev_summ"] = torch.zeros_like(ctx[:, 0])
        _emb = emb
        if self.feedatt == True:
            _emb = torch.cat([_emb, x.nn_state["prev_summ"]], 1)
        enc = self.out_rnn(_emb, x.nn_state["rnn"])

        alphas, summ, scores = self.att(enc, ctx, ctx_mask)
        x.nn_state["prev_summ"] = summ
        enc = torch.cat([enc, summ], -1)

        outs = self.out_lin(enc)
        outs = (outs,) if not q.issequence(outs) else outs
        # _, preds = outs.max(-1)
        return outs[0], x


def create_model(embdim=100, hdim=100, dropout=0., numlayers:int=1,
                 sentence_encoder:SentenceEncoder=None,
                 query_encoder:SentenceEncoder=None,
                 feedatt=False):
    inpemb = torch.nn.Embedding(sentence_encoder.vocab.number_of_ids(), embdim, padding_idx=0)
    inpemb = TokenEmb(inpemb, rare_token_ids=sentence_encoder.vocab.rare_ids, rare_id=1)
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
    decoder_out = BasicGenOutput(hdim + encoder_dim, query_encoder.vocab)
    attention = q.Attention(q.MatMulDotAttComp(hdim, encoder_dim))
    enctodec = torch.nn.Sequential(
        torch.nn.Linear(encoder_dim, hdim),
        torch.nn.Tanh()
    )
    model = BasicGenModel(inpemb, encoder,
                          decoder_emb, decoder_rnn, decoder_out,
                          attention,
                          dropout=dropout,
                          enc_to_dec=enctodec,
                          feedatt=feedatt)
    return model


def do_rare_stats(ds:GeoQueryDataset):
    # how many examples contain rare words, in input and output, in both train and test
    def get_rare_portions(examples:List[BasicState]):
        total = 0
        rare_in_question = 0
        rare_in_query = 0
        rare_in_both = 0
        rare_in_either = 0
        for example in examples:
            total += 1
            question_tokens = example.inp_actions
            query_tokens = example.gold_actions
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


class StateCELoss(StateLoss):
    def __init__(self, weight=None, reduction="mean", ignore_index=-100, mode="logits", **kw):
        super(StateCELoss, self).__init__(**kw)
        self.ce = q.CELoss(weight=weight, reduction=reduction, ignore_index=ignore_index, mode=mode)

    def forward(self, x:DecodableStateBatch):   # must be BasicStates
        # collect all probs:
        probs = torch.stack([x.out_probs[i] for i in range(x.out_probs.listlen())], 1)
        # collect all golds:
        golds = x.gold_tensor
        probs = probs[:, :x.gold_tensor.size(1)]
        loss = self.ce(probs, golds)
        return {"loss": loss}


class StateSeqAccuracies(StateMetric):
    def forward(self, x:DecodableStateBatch):   # must be BasicStates
        # x has a batched .gold_tensor tensor and a batched State-list of .out_probs tensors
        # collect all probs:
        probs = torch.stack([x.out_probs[i] for i in range(x.out_probs.listlen())], 1)
        golds = x.gold_tensor
        probs = probs[:, :x.gold_tensor.size(1)]
        _, preds = probs.max(-1)
        mask = golds != 0
        same = golds == preds
        seq_accs = (same | ~mask).all(1)
        elem_accs = (same & mask).sum(1) / mask.sum(1)
        ret = {"seq_acc": seq_accs.sum().detach().cpu().item() / seq_accs.size(0),
               "elem_acc": elem_accs.sum().detach().cpu().item() / elem_accs.size(0)}
        return ret


def run(lr=0.001,
        batsize=20,
        epochs=50,
        embdim=100,
        encdim=200,
        numlayers=1,
        dropout=.2,
        wreg=1e-6,
        cuda=False,
        gpu=0,
        minfreq=2,
        gradnorm=3.,
        beamsize=1,
        smoothing=0.,
        fulltest=False,
        cosine_restarts=-1.,
        ):
    # DONE: Porter stemmer
    # DONE: linear attention
    # DONE: grad norm
    # DONE: beam search
    # DONE: lr scheduler
    tt = q.ticktock("script")
    ttt = q.ticktock("script")
    device = torch.device("cpu") if not cuda else torch.device("cuda", gpu)
    tt.tick("loading data")
    stemmer = PorterStemmer()
    tokenizer = lambda x: [stemmer.stem(xe) for xe in x.split()]
    ds = GeoQueryDataset(sentence_encoder=SentenceEncoder(tokenizer=tokenizer), min_freq=minfreq)
    dls = get_dataloaders(ds, batsize=batsize)
    train_dl = dls["train"]
    test_dl = dls["test"]
    tt.tock("data loaded")

    do_rare_stats(ds)

    # batch = next(iter(train_dl))
    # print(batch)
    # print("input graph")
    # print(batch.batched_states)

    model = create_model(embdim=embdim, hdim=encdim, dropout=dropout, numlayers=numlayers,
                             sentence_encoder=ds.sentence_encoder, query_encoder=ds.query_encoder, feedatt=True)

    tfdecoder = SeqDecoder(TFTransition(model),
                           [StateCELoss(ignore_index=0, mode="logits"),
                            StateSeqAccuracies()])
    # beamdecoder = BeamActionSeqDecoder(tfdecoder.model, beamsize=beamsize, maxsteps=50)
    freedecoder = SeqDecoder(FreerunningTransition(model),
                             [StateCELoss(ignore_index=0, mode="logits"),
                              StateSeqAccuracies()])
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

    losses = [q.LossWrapper(q.SelectedLinearLoss(x, reduction=None), name=x) for x in ["loss", "elem_acc", "seq_acc"]]
    vlosses = [q.LossWrapper(q.SelectedLinearLoss(x, reduction=None), name=x) for x in ["loss", "elem_acc", "seq_acc"]]

    # 4. define optim
    optim = torch.optim.Adam(tfdecoder.parameters(), lr=lr, weight_decay=wreg)
    # optim = torch.optim.SGD(tfdecoder.parameters(), lr=lr, weight_decay=wreg)

    # lr schedule
    if cosine_restarts >= 0:
        t_max = epochs * len(train_dl)
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

    # 7. run training
    tt.tick("training")
    q.run_training(run_train_epoch=trainepoch, run_valid_epoch=validepoch, max_epochs=epochs)
    tt.tock("done training")



if __name__ == '__main__':
    # try_build_grammar()
    # try_dataset()
    q.argprun(run)