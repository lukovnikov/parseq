from abc import ABC, abstractmethod
from typing import Union, Dict

import torch

from parseq.states import State, DecodableState


class StateMetric(ABC):
    @abstractmethod
    def forward(self, x:State):
        pass

    def __call__(self, x:State):
        return self.forward(x)


class StateLoss(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x:State)->Dict:
        pass



