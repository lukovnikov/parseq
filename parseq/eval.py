from abc import ABC, abstractmethod
from typing import Union, Dict

import torch

from parseq.states import StateBatch, State, batch, DecodableState, DecodableStateBatch


class StateMetric(ABC):
    @abstractmethod
    def forward(self, x:Union[State, StateBatch]):
        pass

    def __call__(self, x:Union[State, StateBatch]):
        return self.forward(x)


class StateLoss(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Union[State, StateBatch])->Dict:
        pass



