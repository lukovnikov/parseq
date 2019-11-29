from typing import Dict, List, Union

import torch
import qelos as q
import numpy as np

from parseq.eval import StateLoss, StateMetric
from parseq.states import DecodableState, TrainableDecodableState, ListState, State, BasicDecoderState
from parseq.transitions import TransitionModel


class SeqDecoder(torch.nn.Module):
    def __init__(self, model:TransitionModel, eval:List[Union[StateMetric, StateLoss]]=tuple(), **kw):
        super(SeqDecoder, self).__init__(**kw)
        self.model = model
        self._metrics = eval

    def forward(self, x:DecodableState) -> Dict:
        # sb = sb.make_copy()
        x.start_decoding()

        outprobs = []

        i = 0

        while not x.all_terminated():
            probs, x = self.model(x, timestep=i)
            outprobs.append(probs)
            i += 1

        outprobs = torch.stack(outprobs, 1)

        metrics = [metric(outprobs, x) for metric in self._metrics]
        metrics = merge_dicts(*metrics)
        return metrics, x


def merge_dicts(*dicts):
    ret = {}
    for d in dicts:
        for k, v in d.items():
            if k in ret:
                raise Exception(f"Key '{k}' already in return dict.")
            ret[k] = v
    return ret


class SeqDecoderTransition(TransitionModel):
    endtoken = "@END@"
    padtoken = "@PAD@"
    """
    States must have a "out_probs" substate and "gold_actions"
    """
    def __init__(self, model:TransitionModel, **kw):
        super(SeqDecoderTransition, self).__init__(**kw)
        self.model = model


class TFTransition(SeqDecoderTransition):
    """
    Teacher forcing transition.
    """
    def forward(self, x:TrainableDecodableState, timestep:int):
        actionprobs, x = self.model(x)
        # feed next
        goldactions = x.get_gold(timestep)
        # goldactions = [x.gold_tensor[i][x.get_decoding_step()[i]]
        #                if not x.is_terminated()[i] else self.padtoken
        #                for i in range(len(x))]
        x.step(goldactions)
        return actionprobs, x


class FreerunningTransition(SeqDecoderTransition):
    """
    Freerunning transition.
    """
    def __init__(self, model:TransitionModel, maxtime=100, **kw):
        super(FreerunningTransition, self).__init__(model, **kw)
        self.maxtime = maxtime

    def forward(self, x:DecodableState, timestep:int):
        actionprobs, x = self.model(x)
        # feed next
        _, predactions = actionprobs.max(-1)
        predactions = [predactions[i] if timestep != self.maxtime else self.endtoken for i in range(len(x))]
        x.step(predactions)
        return actionprobs, x


class BeamState(DecodableState):
    def __init__(self, *states:DecodableState, scores=None):
        bstates = ListState(*states)
        batsize, beamsize = len(bstates), len(states)
        bscores = torch.zeros(batsize, beamsize) if scores is None else scores
        super(BeamState, self).__init__(bstates=bstates, bscores=bscores)

    def start_decoding(self):
        raise Exception("states must have already been started decoding.")

    def is_terminated(self)-> List[List[bool]]:
        """
        Returns a list (over beam elements) of lists (over batch size) of booleans whether each elem in this state is terminated.
        """
        return [x.is_terminated() for x in self.bstates._list]

    def all_terminated(self):
        return all([all(ist) for ist in self.is_terminated()])

    def step(self, action:Union[torch.Tensor, List[Union[str, torch.Tensor]]]=None):
        raise Exception("this should not be used")


class BeamTransition(SeqDecoderTransition):
    """
    Transition for beam search.
    """
    def __init__(self, model:TransitionModel, beamsize=1, maxtime=100, **kw):
        super(BeamTransition, self).__init__(model, **kw)
        self.maxtime = maxtime
        self.beamsize = beamsize

    def do_single_state_init(self, x):
        actionprobs, x = self.model(x)
        logprobs, actionids = torch.sort(actionprobs, 1, descending=True)
        beamstates = []
        for i in range(self.beamsize):
            x_copy = x.make_copy(detach=False, deep=True)
            x_copy.step(actionids[:, min(i, actionids.size(1) - 1)])
            beamstates.append(x_copy)
        y = BeamState(*beamstates, scores=logprobs[:, :min(self.beamsize, actionids.size(1))])
        return y

    @classmethod
    def gather_states(cls, x:List[State], indexes:torch.Tensor)->List[State]:
        ret = []
        for i in range(indexes.size(1)): # for every element in new beam
            proto = x[0].make_copy(detach=True, deep=True)
            uniq = indexes[:, i].unique()
            for x_id in uniq:            # for every id pointing to a batch in current beam
                x_id = x_id.cpu().item()
                idxs_where_x_id = torch.where(indexes[:, i] == x_id)[0]
                proto[idxs_where_x_id] = x[x_id][idxs_where_x_id].make_copy(detach=False, deep=True)
            ret.append(proto)
        return ret

    def forward(self, x:Union[DecodableState, BeamState], timestep:int):
        if timestep == 0:
            assert(isinstance(x, DecodableState))
            y = self.do_single_state_init(x)
            return y
        else:
            assert(isinstance(x, BeamState))
            # gather logprobs and actionids from individual beam elements in batches
            logprobses, actionidses = [], []
            modul = None
            for i in range(len(x.bstates._list)):
                actionprobs, xi = self.model(x.bstates.get(i))      # get i-th batched state in beam, run model over it
                logprobs, actionids = torch.sort(actionprobs, 1, descending=True)   # sort output actions
                logprobs = logprobs[:, :min(self.beamsize, actionids.size(1))]  # clip logprob to beamsize
                logprobs = logprobs + x.bscores[:, i:i+1]                       # compute logprob of whole seq so far
                actionids = actionids[:, :min(self.beamsize, actionids.size(1))]# clip actionids to beamsize
                logprobses.append(logprobs)         # save
                actionidses.append(actionids)       # save
                modul = actionids.size(1) if modul is None else modul   # make sure modul is correct and consistent (used later for state copying)
                assert(modul == actionids.size(1))
            logprobses = torch.cat(logprobses, 1)       # concatenate
            actionidses = torch.cat(actionidses, 1)     # concatenate
            # sort and select actionids for new beam, update logprobs etc
            logprobs, selection_ids = torch.sort(logprobses, 1, descending=True)     # sort: selection_ids are the ids in logprobses
            logprobs = logprobs[:, :min(self.beamsize, selection_ids.size(1))]       # clip to beamsize
            selection_ids = selection_ids[:, :min(self.beamsize, selection_ids.size(1))]       # clip to beamsize
            actionids = actionidses.gather(1, selection_ids)         # select action ids using beam_ids
            stateids = selection_ids % modul        # select state ids based on modul and selection_ids
            gatheredstates = self.gather_states(x.bstates._list, stateids)    # gatheredstates is a ListState
            # apply selected actions to selected states
            for i in range(actionids.size(1)):
                gatheredstates[i].step(actionids[:, i] if timestep != self.maxtime else [self.endtoken]*len(gatheredstates[i]))
            # create new beamstate from updated selected states
            y = BeamState(*gatheredstates, scores=logprobs)
            return y