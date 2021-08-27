import re
from typing import Union

import torch
import qelos as q
from nltk.translate.bleu_score import sentence_bleu
from parseq.grammar import taglisp_to_tree, are_equal_trees, tree_to_lisp
from parseq.scripts_compgen_ae.models.rnn import GRUDecoderCell
from parseq.scripts_compgen_ae.models.tm import TransformerDecoderCell
from parseq.vocab import Vocab


ORDERLESS = {"@WHERE", "@OR", "@AND", "@QUERY", "(@WHERE", "(@OR", "(@AND", "(@QUERY"}


class SeqDecoderBaseline(torch.nn.Module):
    # default_termination_mode = "sequence"
    # default_decode_mode = "serial"

    def __init__(self, cell:Union[TransformerDecoderCell,GRUDecoderCell],
                 vocab=None,
                 max_size:int=100,
                 smoothing:float=0.,
                 **kw):
        super(SeqDecoderBaseline, self).__init__(**kw)
        self.cell = cell
        self.vocab = vocab
        self.max_size = max_size
        self.smoothing = smoothing
        if self.smoothing > 0:
            self.loss = q.SmoothedCELoss(reduction="none", ignore_index=0, smoothing=smoothing, mode="logprobs")
        else:
            self.loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)

        self.logsm = torch.nn.LogSoftmax(-1)

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return self.test_forward(x, y)

    def compute_loss(self, logits, tgt):
        """
        :param logits:      (batsize, seqlen, vocsize)
        :param tgt:         (batsize, seqlen)
        :return:
        """
        mask = (tgt != 0).float()

        logprobs = self.logsm(logits)
        if self.smoothing > 0:
            loss = self.loss(logprobs, tgt)
        else:
            loss = self.loss(logprobs.permute(0, 2, 1), tgt)      # (batsize, seqlen)
        loss = loss * mask
        loss = loss.sum(-1)

        best_pred = logits.max(-1)[1]   # (batsize, seqlen)
        best_gold = tgt
        same = best_pred == best_gold

        elemacc = same.float().sum(-1) / mask.float().sum(-1)

        same = same | ~(mask.bool())
        acc = same.all(-1)  # (batsize,)
        return loss, acc.float(), elemacc

    def test_forward(self, x:torch.Tensor, gold:torch.Tensor=None):   # --> implement how decoder operates end-to-end
        preds = self.get_prediction(x)

        def tensor_to_trees(x, vocab:Vocab):
            xstrs = [vocab.tostr(x[i]).replace("@START@", "") for i in range(len(x))]
            xstrs = [re.sub("::\d+", "", xstr) for xstr in xstrs]
            trees = []
            for xstr in xstrs:
                # drop everything after @END@, if present
                xstr = xstr.split("@END@")
                xstr = xstr[0]
                # add an opening parentheses if not there
                xstr = xstr.strip()
                if len(xstr) == 0 or xstr[0] != "(":
                    xstr = "(" + xstr
                # balance closing parentheses
                parenthese_imbalance = xstr.count("(") - xstr.count(")")
                xstr = xstr + ")" * max(0, parenthese_imbalance)        # append missing closing parentheses
                xstr = "(" * -min(0, parenthese_imbalance) + xstr       # prepend missing opening parentheses
                try:
                    tree = taglisp_to_tree(xstr)
                    if isinstance(tree, tuple) and len(tree) == 2 and tree[0] is None:
                        tree = None
                except Exception as e:
                    tree = None
                trees.append(tree)
            return trees

        # compute loss and metrics
        gold_trees = tensor_to_trees(gold, vocab=self.vocab)
        pred_trees = tensor_to_trees(preds, vocab=self.vocab)
        treeaccs = [float(are_equal_trees(gold_tree, pred_tree, orderless=ORDERLESS, unktoken="@UNK@"))
                    for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {"treeacc": torch.tensor(treeaccs).to(x.device)}

        # compute bleu scores
        bleus = []
        for gold_tree, pred_tree in zip(gold_trees, pred_trees):
            if pred_tree is None or gold_tree is None:
                bleuscore = 0
            else:
                gold_str = tree_to_lisp(gold_tree)
                pred_str = tree_to_lisp(pred_tree)
                bleuscore = sentence_bleu([gold_str.split(" ")], pred_str.split(" "))
            bleus.append(bleuscore)
        bleus = torch.tensor(bleus).to(x.device)
        ret["bleu"] = bleus

        d, logits = self.train_forward(x, gold)
        nll, acc, elemacc = d["loss"], d["acc"], d["elemacc"]
        ret["nll"] = nll
        ret["acc"] = acc
        ret["elemacc"] = elemacc
        return ret, pred_trees

    def train_forward(self, x:torch.Tensor, y:torch.Tensor):  # --> implement one step training of cell
        # extract a training example from y:
        x, newy, tgt = self.extract_training_example(x, y)
        enc, encmask = self.cell.encode_source(x)
        # run through cell: the same for all versions
        logits, cache = self.cell(tokens=newy, enc=enc, encmask=encmask, cache=None, _full_sequence=True)
        # compute loss: different versions do different masking and different targets
        loss, acc, elemacc = self.compute_loss(logits, tgt)
        return {"loss": loss, "acc": acc, "elemacc": elemacc}, logits

    def extract_training_example(self, x, y):
        ymask = (y != 0).float()
        ylens = ymask.sum(1).long()
        newy = y
        newy = torch.cat([torch.ones_like(newy[:, 0:1]) * self.vocab["@START@"], newy], 1)
        newy = torch.cat([newy, torch.zeros_like(newy[:, 0:1])], 1)       # append some zeros
        # append EOS
        for i, ylen in zip(range(len(ylens)), ylens):
            newy[i, ylen+1] = self.vocab["@END@"]

        goldy = newy[:, 1:]
        # tgt = torch.zeros(goldy.size(0), goldy.size(1), self.vocab.number_of_ids(), device=goldy.device)
        # tgt = tgt.scatter(2, goldy[:, :, None], 1.)
        # tgtmask = (goldy != 0).float()

        newy = newy[:, :-1]
        return x, newy, goldy

    def get_prediction(self, x:torch.Tensor):
        steps_used = torch.ones(x.size(0), device=x.device, dtype=torch.long) * self.max_size
        # initialize empty ys:
        y = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@START@"]
        # yend = torch.ones(x.size(0), 1, device=x.device, dtype=torch.long) * self.vocab["@EOS@"]

        # run encoder
        enc, encmask = self.cell.encode_source(x)

        step = 0
        newy = None
        ended = torch.zeros_like(y[:, 0]).bool()
        cache = None
        while step < self.max_size and not torch.all(ended):
            y = newy if newy is not None else y
            # run cell
            # y = torch.cat([y, yend], 1)
            logits, cache = self.cell(tokens=y, enc=enc, encmask=encmask, cache=cache)
            probs = torch.softmax(logits, -1)
            maxprobs, preds = probs.max(-1)
            preds, maxprobs = preds[:, -1], maxprobs[:, -1]
            newy = torch.cat([y, preds[:, None]], 1)
            y__ = torch.cat([y, torch.zeros_like(newy[:, :newy.size(1) - y.size(1)])], 1)
            newy = torch.where(ended[:, None], y__, newy)     # prevent terminated examples from changing
            _ended = (preds == self.vocab["@END@"])
            ended = ended | _ended
            step += 1
        return newy


class SeqDecoderAE(torch.nn.Module):
    def __init__(self, celli2o, cello2i, celli2i, cello2o,
                 vocab=None, inpvocab=None, max_size=100, smoothing=0.,
                 main_contrib=0.25, rev_contrib=0.25, i2i_contrib=0.25, o2o_contrib=0.25,
                 **kw):
        super(SeqDecoderAE, self).__init__(**kw)
        self.i2o = SeqDecoderBaseline(celli2o, vocab=vocab, max_size=max_size, smoothing=smoothing)
        self.o2i = SeqDecoderBaseline(cello2i, vocab=inpvocab, max_size=max_size, smoothing=smoothing)
        self.i2i = SeqDecoderBaseline(celli2i, vocab=inpvocab, max_size=max_size, smoothing=smoothing)
        self.o2o = SeqDecoderBaseline(cello2o, vocab=vocab, max_size=max_size, smoothing=smoothing)
        self.main_contrib, self.rev_contrib, self.i2i_contrib, self.o2o_contrib \
            = main_contrib, rev_contrib, i2i_contrib, o2o_contrib

    def forward(self, x, y, xi, yi, xo, yo):
        if self.training:
            return self.train_forward(x, y, xi, yi, xo, yo)
        else:
            return self.test_forward(x, y)

    def train_forward(self, x:torch.Tensor, y:torch.Tensor,
                      xi:torch.Tensor, yi:torch.Tensor,
                      xo:torch.Tensor, yo:torch.Tensor):  # --> implement one step training of cell
        # extract a training example from y:
        blank_ret = {"loss": 0, "acc": -1, "elemacc": -1}
        ret, logits = self.i2o.train_forward(x, y) if self.main_contrib > 0 else blank_ret, None
        reti, _ = self.i2i.train_forward(xi, yi) if self.i2i_contrib > 0 else blank_ret, None
        reto, _ = self.o2o.train_forward(xo, yo) if self.o2o_contrib > 0 else blank_ret, None
        retr, _ = self.o2i.train_forward(y, x) if self.rev_contrib > 0 else blank_ret, None
        loss = ret["loss"] * self.main_contrib + reti["loss"] * self.i2i_contrib \
               + reto["loss"] * self.o2o_contrib + retr["loss"] * self.rev_contrib
        return {"loss": loss, "acc": ret["acc"], "elemacc": ret["elemacc"],
                "loss_m": ret["loss"], "loss_i": reti["loss"], "loss_o": reto["loss"], "loss_r": retr["loss"],
                "acc_i": reti["acc"], "acc_o": reto["acc"], "acc_r": retr["acc"]}, logits

    def test_forward(self, x:torch.Tensor, gold:torch.Tensor=None):   # --> implement how decoder operates end-to-end
        return self.i2o.test_forward(x, gold)

