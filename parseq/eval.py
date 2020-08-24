from abc import ABC, abstractmethod
from functools import partial
from typing import Union, Dict, Callable, List

import nltk
import qelos as q

import torch
import numpy as np

from parseq.grammar import are_equal_trees
from parseq.states import State, DecodableState, TrainableDecodableState, BeamState


class SelectedLoss(q.SelectedLinearLoss):
    """ Same as LinearLoss, but with selection from tuple of outputs from model (that specifies losses)
        To be used to output multiple losses from the model/ select one model output as training loss
    """
    def forward(self, model_outs, gold, **kw):
        metrics = model_outs[0]
        x = metrics[self.which]
        if self.reduction in ["elementwise_mean", "mean"]:
            ret = x.mean()
        elif self.reduction == "sum":
            ret = x.sum()
        else:
            ret = x
        return ret


def make_array_of_metrics(*lossnames, reduction=None):
    ret = []
    for lossname in lossnames:
        ret.append(q.MetricWrapper(SelectedLoss(lossname, reduction=reduction), name=lossname))
    return ret


class Metric(ABC):
    @abstractmethod
    def forward(self, probs, predactions, gold, x:State=None) -> Dict:
        pass

    def __call__(self, probs, predactions, gold, x:State=None) -> Dict:
        return self.forward(probs, predactions, gold, x)


class Loss(torch.nn.Module, ABC):
    def __init__(self, contrib=1., **kw):
        super(Loss, self).__init__(**kw)
        self.contrib = contrib

    @abstractmethod
    def forward(self, probs, predactions, gold, x:State=None)->Dict:
        pass


class BCELoss(Loss):
    def __init__(self, weight=None, reduction="mean", mode="logits", smoothing:float=0., **kw):
        super(BCELoss, self).__init__(**kw)
        if mode == "logits":
            self.bce = torch.nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
        elif mode == "logprobs":
            self.bce = torch.nn.BCELoss(weight=weight, reduction=reduction)
        self.smoothing = smoothing

    def forward(self, probs, predactions, gold, x:State=None) ->Dict:
        if self.smoothing > 0:
            gold = gold.clamp(self.smoothing, 1-self.smoothing)
        ret = self.bce(probs, gold) * self.contrib
        return {"loss": ret, "ce": ret}


class KLLoss(Loss):
    def __init__(self, weight=None, reduction="mean", mode="logits", goldmode="logits", maximize=False, **kw):
        super(KLLoss, self).__init__(**kw)
        self.reduction = reduction
        self.mode = mode
        self.goldmode = goldmode
        self.sm = torch.nn.Softmax(-1)
        self.logsm = torch.nn.LogSoftmax(-1)
        self.mult = -1 if maximize else 1
        self.kldiv = torch.nn.KLDivLoss(reduction="none")

    def forward(self, probs, predactions, golds, mask=None, x:State=None) ->Dict:
        if probs.size(1) < golds.size(1):
            extension = torch.ones(probs.size(0), golds.size(1) - probs.size(1), probs.size(2), dtype=probs.dtype, device=probs.device)
            extension /= extension.size(2)  # makes uniform dist
            probs = torch.cat([probs, extension], 1)
        else:
            probs = probs[:, :golds.size(1)]

        if self.mode == "logits":
            logprobs = self.logsm(probs)
        elif self.mode == "probs":
            logprobs = torch.log(probs).clamp_min(-1e9)
        elif self.mode == "logprobs":
            logprobs = probs.clamp_min(-1e9)
        else:
            raise Exception(f"mode '{self.mode}' unknown. ")

        if self.goldmode == "logits":
            goldprobs = self.sm(golds)
        elif self.goldmode == "probs":
            goldprobs = golds
        elif self.goldmode == "logprobs":
            goldprobs = torch.exp(golds)
        else:
            raise Exception(f"goldmode '{self.goldmode}' unknown. ")

        kl = self.kldiv(logprobs, goldprobs)    # (batsize, seqlen, vocabsize)
        kl = kl.sum(-1)

        if mask is not None:
            assert mask.dim() == 2, f"mask dim must be 2"
            kl = kl * mask.float()

        if self.reduction == "mean":
            ret = kl.sum() / mask.float().sum()
        elif self.reduction == "sum":
            ret = kl.sum()
        elif self.reduction == "none" or self.reduction is None:
            ret = kl
        else:
            raise Exception(f"Unknown reduction '{self.reduction}'")

        ret = ret * self.contrib * self.mult

        return {"kl": ret, "loss": ret}


class EntropyLoss(Loss):
    def __init__(self, weight=None, reduction="mean", ignore_index=-100, mode="logits", maximize=True, **kw):
        super(EntropyLoss, self).__init__(**kw)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.mode = mode
        self.sm = torch.nn.Softmax(-1)
        self.logsm = torch.nn.LogSoftmax(-1)
        self.mult = -1 if maximize else 1

    def forward(self, probs, predactions, golds, mask=None, x:State=None) ->Dict:
        if probs.size(1) < golds.size(1):
            extension = torch.ones(probs.size(0), golds.size(1) - probs.size(1), probs.size(2), dtype=probs.dtype, device=probs.device)
            extension /= extension.size(2)  # makes uniform dist
            probs = torch.cat([probs, extension], 1)
        else:
            probs = probs[:, :golds.size(1)]

        if self.mode == "logits":
            probs = self.sm(probs)
            logprobs = self.logsm(probs).clamp_min(-1e9)
        elif self.mode == "probs":
            logprobs = torch.log(probs).clamp_min(-1e9)
        elif self.mode == "logprobs":
            logprobs = probs.clamp_min(-1e9)
            probs = torch.exp(probs)
        else:
            raise Exception(f"mode '{self.mode}' unknown. ")

        entropy = -(probs * logprobs)
        if mask is not None and mask.dim() == 3:
            entropy = entropy * mask

        entropy = entropy.sum(-1)        # (batsize, seqlen)

        gold_mask = mask if mask.dim() == 2 else None
        if self.ignore_index >= 0:
            if gold_mask is None:
                gold_mask = golds != self.ignore_index
            else:
                gold_mask = gold_mask & (golds != self.ignore_index)

        if gold_mask is not None:
            entropy = entropy * gold_mask.float()

        if self.reduction == "mean":
            ret = entropy.sum() / gold_mask.float().sum()
        elif self.reduction == "sum":
            ret = entropy.sum()
        elif self.reduction == "none" or self.reduction is None:
            ret = entropy
        else:
            raise Exception(f"Unknown reduction '{self.reduction}'")

        ret = ret * self.contrib * self.mult

        return {"entropy": ret, "loss": ret}


class CELoss(Loss):
    def __init__(self, weight=None, reduction="mean", ignore_index=0, mode="logits", smoothing:float=0., **kw):
        super(CELoss, self).__init__(**kw)
        self.mode = mode
        self.ce = q.CELoss(weight=weight, reduction=reduction, ignore_index=ignore_index, mode=mode)
        if smoothing != 0.:
            assert(smoothing < 1. and smoothing > 0.)
            assert(mode in ["logits", "logprobs"])
            self.ce = q.SmoothedCELoss(reduction=reduction, ignore_index=ignore_index, smoothing=smoothing, mode=mode, weight=weight)

    def forward(self, probs, predactions, golds, x:State=None):   # must be BasicStates
        # golds = x.get_gold()
        if probs.size(1) < golds.size(1):
            extension = torch.ones(probs.size(0), golds.size(1) - probs.size(1), probs.size(2), dtype=probs.dtype, device=probs.device)
            extension /= extension.size(2)  # makes uniform dist
            probs = torch.cat([probs, extension], 1)
        else:
            probs = probs[:, :golds.size(1)]
        if probs.size(1) != golds.size(1):
            print(probs, golds)

        selected = probs.gather(2, golds[:, :, None])
        if torch.any(selected == (-np.infty if self.mode in ("logits", "logprobs") else 0.)):
            print("gold id could not be generated")

        loss = self.ce(probs, golds)
        loss = loss * self.contrib
        return {"loss": loss, "ce": loss}


def state_path_penalty_getter(x, spec=None):
    path = spec.split(".")
    o = x
    for path_e in path:
        o = getattr(o, path_e)
    return o


class StatePenalty(Loss):
    def __init__(self, getter, weight=1., reduction="mean", name="penalty", **kw):
        super(StatePenalty, self).__init__(**kw)
        if isinstance(getter, str):
            getter = partial(state_path_penalty_getter, spec=getter)
        self.getter = getter
        self.reduction = reduction
        self.weight = weight
        self._name = name

    def forward(self, probs, predactions, gold, x:State=None) ->Dict:
        # get tensor from state
        penalty_vec = self.getter(x)
        assert(penalty_vec.dim() == 1 and penalty_vec.size(0) == probs.size(0))
        if self.reduction in ("mean", "default"):
            penalty = penalty_vec.mean()
        elif self.reduction == "sum":
            penalty = penalty_vec.sum()
        elif self.reduction in ("none", None):
            penalty = penalty_vec
        else:
            raise Exception(f"unknown reduction mode: {self.reduction}")
        ret = penalty * q.v(self.weight)
        ret = ret * self.contrib
        return {"loss": ret, self._name: ret}


class SeqAccuracies(Metric):
    padid = 0
    unkid = 1
    def forward(self, probs, predactions, golds, x:State=None):   # must be BasicStates
        # TODO: GOLD MUST CONTAIN END TOKEN !!!!!
        # golds = x.get_gold()
        mask = golds != self.padid
        if predactions.size(1) < golds.size(1):
            extension = torch.zeros(predactions.size(0), golds.size(1) - predactions.size(1), dtype=predactions.dtype, device=predactions.device)
            predactions = torch.cat([predactions, extension], 1)
        else:
            predactions = predactions[:, :golds.size(1)]
        same = golds == predactions
        same = same & (predactions != self.unkid)
        seq_accs = (same | ~mask).all(1).float()
        elem_accs = (same & mask).sum(1).float() / mask.sum(1).float()
        ret = {"seq_acc": seq_accs.sum().detach().cpu().item() / seq_accs.size(0),
               "elem_acc": elem_accs.sum().detach().cpu().item() / elem_accs.size(0)}
        return ret


class DerivedAccuracy(Metric):
    def __init__(self, name:str="derived_acc", tensor2tree:Callable[[torch.Tensor], nltk.Tree]=None, **kw):
        super(DerivedAccuracy, self).__init__(**kw)
        self.name = name
        self.tensor2tree = tensor2tree

    def forward(self, probs, predactions, golds, x:State=None):
        # golds = x.get_gold()
        gold_trees = [self.tensor2tree(gold) for gold in golds]
        pred_trees = [self.tensor2tree(predactionse) for predactionse in predactions]
        ret = [float(gold_tree == pred_tree) for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {self.name: sum(ret) / len(ret)}
        return ret


class TreeAccuracy(Metric):
    unktoken = "@UNK@"
    def __init__(self, name:str="tree_acc", tensor2tree:Callable[[torch.Tensor], nltk.Tree]=None, orderless=set(), **kw):
        super(TreeAccuracy, self).__init__(**kw)
        self.name = name
        self.tensor2tree = tensor2tree
        self.orderless = orderless

    def forward(self, probs, predactions, golds, x:State=None):
        def compare(_gold_trees, _predactions):
            pred_trees = [self.tensor2tree(predactionse) for predactionse in _predactions]
            ret = [float(are_equal_trees(gold_tree, pred_tree, orderless=self.orderless, unktoken=self.unktoken))
                   for gold_tree, pred_tree in zip(_gold_trees, pred_trees)]
            return ret
        if isinstance(predactions, torch.Tensor) and predactions.dim() == 3:      # beam states
            # assert(isinstance(x, BeamState))
            # golds = x.bstates.get(0).get_gold()
            gold_trees = [self.tensor2tree(goldse) for goldse in golds]
            rets = []
            for i in range(predactions.size(1)):
                ret_i = compare(gold_trees, predactions[:, i])
                rets.append(ret_i)
            rets = np.asarray(rets).T
            acc_cum = np.cumsum(rets, 1)
            acc_cum = np.clip(acc_cum, 0, 1)
            r = {}
            batsize = acc_cum.shape[0]
            r[self.name] = sum(acc_cum[:, 0]) / batsize
            for j in range(acc_cum.shape[1]):
                r[f"{self.name}_at{j+1}"] = sum(acc_cum[:, j]) / batsize
            r[f"{self.name}_at_last"] = sum(acc_cum[:, -1]) / batsize
            return r
        else:
            # assert(predactions.dim() == 2)
            # golds = x.get_gold()
            # _gold_trees = x.gold_trees
            gold_trees = [self.tensor2tree(goldse) for goldse in golds]
            ret = compare(gold_trees, predactions)
            ret = {self.name: sum(ret) / len(ret)}
            return ret


class BeamSeqAccuracies(Metric):
    def forward(self, probs, predactions, golds, x:State=None):
        # golds = x.bstates.get(0).get_gold()
        # for i in range(len(x.bstates._list)):
        #     assert(torch.allclose(x.bstates.get(i).get_gold(), golds))
        mask = golds != 0

        if predactions.size(2) < golds.size(1):
            extension = torch.zeros(predactions.size(0), predactions.size(1), golds.size(1) - predactions.size(2), dtype=predactions.dtype, device=predactions.device)
            predactions = torch.cat([predactions, extension], 2)
        else:
            predactions = predactions[:, :, :golds.size(1)]
        same = golds[:, None, :] == predactions
        seq_accs = (same | ~mask[:, None, :]).all(2)   # (batsize, beamsize)
        assert(torch.allclose((seq_accs.float().sum(-1) <= 1).float(), torch.ones_like(seq_accs[:, 0]).float()))
        batsize, beamsize = seq_accs.size(0), seq_accs.size(1)
        seq_accs_cum = (seq_accs.cumsum(-1) > 0).float()
        seq_accs_cum_sum = list((seq_accs_cum.sum(0) / batsize).detach().cpu().numpy())      # (beamsize,)

        ret = {}
        for j in range(0, beamsize):
            ret[f"beam_seq_recall_at{j+1}"] = seq_accs_cum_sum[j]
        ret["beam_recall"] = seq_accs_cum_sum[-1]
        ret["beam_seq_acc"] = seq_accs_cum_sum[0]
        ret["beam_seq_acc_bottom"] = seq_accs[:, -1].float().sum().detach().cpu().item() / batsize

        elem_accs = (same & mask[:, None, :]).sum(2).float() / mask[:, None, :].sum(2).float()
        elem_accs = elem_accs.max(1)[0]
        ret["beam_best_elem_acc"] = elem_accs.sum().detach().cpu().item() / batsize
        return ret



