from abc import ABC, abstractmethod
from typing import Union, Dict
import qelos as q

import torch

from parseq.states import State, DecodableState, TrainableDecodableState


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
        probs = probs[:, :golds.size(1)]
        loss = self.ce(probs, golds)
        return {"loss": loss}


class StateSeqAccuracies(StateMetric):
    def forward(self, probs, predactions, x:TrainableDecodableState):   # must be BasicStates
        golds = x.get_gold()
        mask = golds != 0
        predactions = predactions[:, :golds.size(1)]
        same = golds == predactions
        seq_accs = (same | ~mask).all(1).float()
        elem_accs = (same & mask).sum(1).float() / mask.sum(1).float()
        ret = {"seq_acc": seq_accs.sum().detach().cpu().item() / seq_accs.size(0),
               "elem_acc": elem_accs.sum().detach().cpu().item() / elem_accs.size(0)}
        return ret


class BeamSeqAccuracy(StateMetric):
    def forward(self, probs, predactions, x):
        golds = x.bstates.get(0).get_gold()
        for i in range(len(x.bstates._list)):
            assert(torch.allclose(x.bstates.get(i).get_gold(), golds))
        mask = golds != 0
        predactions = predactions[:, :, :golds.size(1)]
        same = golds[:, None, :] == predactions
        seq_accs = (same | ~mask[:, None, :]).all(2).any(1).float()
        elem_accs = (same & mask[:, None, :]).sum(2).float() / mask[:, None, :].sum(2).float()
        elem_accs = elem_accs.max(1)[0]
        ret = {"beam_seq_acc": seq_accs.sum().detach().cpu().item() / seq_accs.size(0),
               "beam_best_elem_acc": elem_accs.sum().detach().cpu().item() / elem_accs.size(0)}
        return ret



