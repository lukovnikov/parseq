from typing import Dict, List, Union, Tuple

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

    def forward(self, x:DecodableState) -> Tuple[Dict, State]:
        # sb = sb.make_copy()
        x.start_decoding()

        outprobs = []
        predactions = []

        i = 0

        all_terminated = x.all_terminated()
        while not all_terminated:
            probs, preds, x, all_terminated = self.model(x, timestep=i)
            outprobs.append(probs)
            predactions.append(preds)
            i += 1

        outprobs = torch.stack(outprobs, 1)
        predactions = torch.stack(predactions, 1)

        metrics = [metric(outprobs, predactions, x) for metric in self._metrics]
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
        _, predactions = actionprobs.max(-1)
        # feed next
        goldactions = x.get_gold(timestep)
        x.step(goldactions)
        return actionprobs, predactions, x, x.all_terminated()


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
        x.step(predactions)
        return actionprobs, predactions, x, x.all_terminated() or timestep >= self.maxtime - 1


class BeamDecoder(SeqDecoder):
    def __init__(self, model:TransitionModel, eval:List[Union[StateMetric, StateLoss]]=tuple(), eval_beam:List[Union[StateMetric, StateLoss]]=None,
                 beamsize=1, maxtime=100, copy_deep=False, **kw):
        model = BeamTransition(model, beamsize=beamsize, maxtime=maxtime, copy_deep=copy_deep)
        super(BeamDecoder, self).__init__(model, eval, **kw)
        self._beam_metrics = eval_beam

    def forward(self, x:DecodableState) -> Dict:
        # sb = sb.make_copy()
        x.start_decoding()

        i = 0

        all_terminated = x.all_terminated()
        outprobs = None
        while not all_terminated:
            outprobs, predactions, x, all_terminated = self.model(x, timestep=i)
            i += 1

        beam_metrics = [metric(outprobs, predactions, x) for metric in self._beam_metrics]
        beam_metrics = merge_dicts(*beam_metrics)
        # get top of the beam and run eval on top of the beam
        top_outprobs, top_predactions, top_x = outprobs[:, 0], predactions[:, 0], x.bstates.get(0)
        metrics = [metric(top_outprobs, top_predactions, top_x) for metric in self._metrics]
        metrics = merge_dicts(beam_metrics, *metrics)
        return metrics, x


class BeamState(DecodableState):
    def __init__(self, states:List[DecodableState], scores:torch.Tensor=None,
                 actionprobs:List[torch.Tensor]=None, predactions:torch.Tensor=None):
        bstates = ListState(*states)
        batsize, beamsize = len(bstates), len(states)
        bscores = torch.zeros(batsize, beamsize) if scores is None else scores
        kw = {"bstates": bstates, "bscores": bscores}
        if actionprobs is not None:
            actionprobs = ListState(*actionprobs)
            kw["actionprobs"] = actionprobs
        kw["predactions"] = predactions
        super(BeamState, self).__init__(**kw)

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
    # TODO: problem: beam search is continuing from terminated states
    # ==> TODO: don't expand terminated states (??? what about preference for shorter sequences under beam search?)
    """
    Transition for beam search.
    """
    def __init__(self, model:TransitionModel, beamsize=1, maxtime=100, copy_deep=False, **kw):
        super(BeamTransition, self).__init__(model, **kw)
        self.maxtime = maxtime
        self.beamsize = beamsize
        self.copy_deep = copy_deep

    def do_single_state_init(self, x):
        actionprobs, x = self.model(x)
        is_term = torch.tensor(x.is_terminated()).to(actionprobs.device)
        logprobs, actionids = torch.sort(actionprobs, 1, descending=True)
        beamstates = []
        beamsize = min(self.beamsize, actionids.size(1))
        for i in range(beamsize):
            x_copy = x.make_copy(detach=False, deep=True)
            x_copy.step(actionids[:, i])
            beamstates.append(x_copy)
        scores = logprobs[:, :beamsize]
        # take into account terminated states
        if is_term.any().cpu().item():
            scores[is_term, 0] = 0
            scores[is_term, 1:] = -np.infty
            raise Exception("terminated after first timestep! wtf!")
        # create beam state
        y = BeamState(beamstates,
                      scores=scores,
                      actionprobs=[actionprobs.clone()[:, None, :] for _ in range(beamsize)],
                      predactions=actionids[:, :beamsize, None])
        return y

    def gather_states(self, x:List[State], indexes:torch.Tensor)->List[State]:
        ret = []
        for i in range(indexes.size(1)): # for every element in new beam
            uniq = indexes[:, i].unique()
            if len(uniq) == 1:
                x_id = uniq[0].cpu().item()
                proto = x[x_id].make_copy(detach=False, deep=self.copy_deep)
            else:
                proto = x[0].make_copy(detach=True, deep=self.copy_deep)
                for x_id in uniq:            # for every id pointing to a batch in current beam
                    # x_x_id_copy = x[x_id].make_copy(detach=False, deep=True)
                    x_id = x_id.cpu().item()
                    idxs_where_x_id = torch.where(indexes[:, i] == x_id)[0]
                    proto[idxs_where_x_id] = x[x_id][idxs_where_x_id].make_copy(detach=False, deep=self.copy_deep)
                    # proto[idxs_where_x_id] = x_x_id_copy[idxs_where_x_id]
            ret.append(proto)
        return ret

    def forward(self, x:Union[DecodableState, BeamState], timestep:int):
        if timestep == 0:
            assert(isinstance(x, DecodableState))
            y = self.do_single_state_init(x)
        else:
            assert(isinstance(x, BeamState))

            # gather logprobs and actionids from individual beam elements in batches
            scoreses, actionidses, actionprobses = [], [], []
            modul = None
            for i in range(len(x.bstates._list)):
                actionprobs, xi = self.model(x.bstates.get(i))      # get i-th batched state in beam, run model over it
                is_term = torch.tensor(xi.is_terminated()).to(actionprobs.device)
                x.bstates.set(i, xi)
                logprobs, actionids = torch.sort(actionprobs, 1, descending=True)   # sort output actions
                beamsize = min(self.beamsize, actionids.size(1))
                _actionprobs = torch.cat([x.actionprobs.get(i), actionprobs[:, None, :]], 1)
                actionprobses.append(_actionprobs)
                scores = logprobs[:, :beamsize]  # clip logprob to beamsize
                if is_term.any().cpu().item():
                    scores[is_term, 0] = 0
                    scores[is_term, 1:] = -np.infty
                scores = scores + x.bscores[:, i:i+1]                       # compute logprob of whole seq so far
                actionids = actionids[:, :beamsize]# clip actionids to beamsize
                scoreses.append(scores)         # save
                actionidses.append(actionids)       # save
                modul = beamsize if modul is None else modul   # make sure modul is correct and consistent (used later for state copying)
                assert(modul == actionids.size(1))
            scoreses, actionidses, actionprobses = torch.cat(scoreses, 1), torch.cat(actionidses, 1), torch.stack(actionprobses, 1)       # concatenate

            # sort and select actionids for new beam, update logprobs etc
            scores, selection_ids = torch.sort(scoreses, 1, descending=True)     # sort: selection_ids are the ids in logprobses
            beamsize = min(self.beamsize, selection_ids.size(1))
            scores = scores[:, :beamsize]       # clip to beamsize
            selection_ids = selection_ids[:, :beamsize]       # clip to beamsize
            actionids = actionidses.gather(1, selection_ids)         # select action ids using beam_ids
            stateids = selection_ids // modul        # select state ids based on modul and selection_ids
            actionprobs = actionprobses.gather(1, stateids[:, :, None, None].repeat(1, 1, actionprobses.size(2), actionprobses.size(3)))
            predactions = x.predactions.gather(1, stateids[:, :, None].repeat(1, 1, x.predactions.size(2)))
            predactions = torch.cat([predactions, actionids[:, :, None]], 2)
            gatheredstates = self.gather_states(x.bstates._list, stateids)    # gatheredstates is a ListState

            # apply selected actions to selected states
            for i in range(actionids.size(1)):
                gatheredstates[i].step(actionids[:, i])         # if timestep != self.maxtime-1 else [self.endtoken]*len(gatheredstates[i]))

            # create new beamstate from updated selected states
            y = BeamState(gatheredstates, scores=scores, actionprobs=[e[:, 0] for e in actionprobs.split(1, 1)], predactions=predactions)
        return torch.stack(y.actionprobs._list, 1), y.predactions, y, y.all_terminated() or timestep >= self.maxtime - 1