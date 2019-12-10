from abc import ABC, abstractmethod
from typing import Union, Dict, Callable

import nltk
import qelos as q

import torch

from parseq.states import State, DecodableState, TrainableDecodableState


class SelectedLoss(q.SelectedLinearLoss):
    """ Same as LinearLoss, but with selection from tuple of outputs from model (that specifies losses)
        To be used to output multiple losses from the model/ select one model output as training loss
    """
    def forward(self, model_outs, gold, **kw):
        metrics, state = model_outs
        x = metrics[self.which]
        if self.reduction in ["elementwise_mean", "mean"]:
            ret = x.mean()
        elif self.reduction == "sum":
            ret = x.sum()
        else:
            ret = x
        return ret


def make_loss_array(*lossnames):
    ret = []
    for lossname in lossnames:
        ret.append(q.LossWrapper(SelectedLoss(lossname, reduction=None)))
    return ret


class StateMetric(ABC):
    @abstractmethod
    def forward(self, probs, predactions, x:State) -> Dict:
        pass

    def __call__(self, probs, predactions, x:State) -> Dict:
        return self.forward(probs, predactions, x)


class StateLoss(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, probs, predactions, x:State)->Dict:
        pass


class StateCELoss(StateLoss):
    def __init__(self, weight=None, reduction="mean", ignore_index=-100, mode="logits", **kw):
        super(StateCELoss, self).__init__(**kw)
        self.ce = q.CELoss(weight=weight, reduction=reduction, ignore_index=ignore_index, mode=mode)

    def forward(self, probs, predactions, x:TrainableDecodableState):   # must be BasicStates
        golds = x.get_gold()
        if probs.size(1) < golds.size(1):
            extension = torch.ones(probs.size(0), golds.size(1) - probs.size(1), probs.size(2), dtype=probs.dtype, device=probs.device)
            extension /= extension.size(2)  # makes uniform dist
            probs = torch.cat([probs, extension], 1)
        else:
            probs = probs[:, :golds.size(1)]
        if probs.size(1) != golds.size(1):
            print(probs, golds)
        loss = self.ce(probs, golds)
        return {"loss": loss}


class StateSeqAccuracies(StateMetric):
    def forward(self, probs, predactions, x:TrainableDecodableState):   # must be BasicStates
        golds = x.get_gold()
        mask = golds != 0
        if predactions.size(1) < golds.size(1):
            extension = torch.zeros(predactions.size(0), golds.size(1) - predactions.size(1), dtype=predactions.dtype, device=predactions.device)
            predactions = torch.cat([predactions, extension], 1)
        else:
            predactions = predactions[:, :golds.size(1)]
        same = golds == predactions
        seq_accs = (same | ~mask).all(1).float()
        elem_accs = (same & mask).sum(1).float() / mask.sum(1).float()
        ret = {"seq_acc": seq_accs.sum().detach().cpu().item() / seq_accs.size(0),
               "elem_acc": elem_accs.sum().detach().cpu().item() / elem_accs.size(0)}
        return ret


class StateDerivedAccuracy(StateMetric):
    def __init__(self, name:str="derived_acc", tensor2tree:Callable[[torch.Tensor], nltk.Tree]=None, **kw):
        super(StateDerivedAccuracy, self).__init__(**kw)
        self.name = name
        self.tensor2tree = tensor2tree

    def forward(self, probs, predactions, x:TrainableDecodableState):
        golds = x.get_gold()
        gold_trees = [self.tensor2tree(gold) for gold in golds]
        pred_trees = [self.tensor2tree(predactionse) for predactionse in predactions]
        ret = [float(gold_tree == pred_tree) for gold_tree, pred_tree in zip(gold_trees, pred_trees)]
        ret = {self.name: sum(ret) / len(ret)}
        return ret


class BeamSeqAccuracies(StateMetric):
    def forward(self, probs, predactions, x):
        golds = x.bstates.get(0).get_gold()
        for i in range(len(x.bstates._list)):
            assert(torch.allclose(x.bstates.get(i).get_gold(), golds))
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



