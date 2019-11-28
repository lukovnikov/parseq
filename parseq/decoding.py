from typing import Dict, List, Union

import torch
import qelos as q
import numpy as np

from parseq.eval import StateLoss, StateMetric
from parseq.states import DecodableStateBatch
from parseq.transitions import TransitionModel


class SeqDecoder(torch.nn.Module):
    def __init__(self, model:TransitionModel, eval:List[Union[StateMetric, StateLoss]]=tuple(), **kw):
        super(SeqDecoder, self).__init__(**kw)
        self.model = model
        self._metrics = eval

    def forward(self, sb:DecodableStateBatch) -> Dict:
        # sb = sb.make_copy()
        sb.start_decoding()

        while not sb.all_terminated():
            self.model(sb)

        metrics = [metric(sb) for metric in self._metrics]
        metrics = merge_dicts(*metrics)
        return metrics


def merge_dicts(*dicts):
    ret = {}
    for d in dicts:
        for k, v in d.items():
            if k in ret:
                raise Exception(f"Key '{k}' already in return dict.")
            ret[k] = v
    return ret


class SeqDecoderTransition(TransitionModel):
    """
    States must have a "out_probs" substate and "gold_actions"
    """
    def __init__(self, model, **kw):
        super(SeqDecoderTransition, self).__init__(**kw)
        self.model = model


class TFTransition(SeqDecoderTransition):
    """
    Teacher forcing transition.
    """
    padtoken = "@PAD@"
    def forward(self, x:DecodableStateBatch):
        actionprobs, x = self.model(x)
        # store predictions
        x.out_probs.append(actionprobs)
        # feed next
        # goldactions = [state.gold_actions[state.get_decoding_step()]
        #                if len(state.gold_actions) > state.get_decoding_step()
        #                else self.padtoken
        #                for state in x.states]
        goldactions = [state.gold_tensor[state.get_decoding_step()]
                       if not state.is_terminated() else self.padtoken
                       for state in x.states]
        x.step(goldactions)
        return x


class FreerunningTransition(SeqDecoderTransition):
    """
    Freerunning transition.
    """
    endtoken = "@END@"
    def __init__(self, model, maxtime=100, **kw):
        super(FreerunningTransition, self).__init__(model, **kw)
        self.maxtime = maxtime

    def forward(self, x:DecodableStateBatch):
        actionprobs, x = self.model(x)
        # store predictions
        x.out_probs.append(actionprobs)
        # feed next
        _, predactions = actionprobs.max(-1)
        if self.maxtime == x.states[0].get_decoding_step():
            predactions = [self.endtoken for _ in x.states]
        x.step(predactions)
        return x


# TODO: convert to use new state-transition API
class BeamActionSeqDecoder(torch.nn.Module):
    def __init__(self, model:TransitionModel, beamsize=1, maxsteps=25, **kw):
        super(BeamActionSeqDecoder, self).__init__(**kw)
        self.model = model
        self.beamsize = beamsize
        self.maxsteps = maxsteps

    def forward(self, fsb:DecodableStateBatch):
        hasgold = []
        with torch.no_grad():
            fsb_original = fsb
            fsb = fsb_original.make_copy()
            states = fsb.unbatch()
            numex = 0
            for state in states:
                hasgold.append(state.has_gold)
                state.use_gold = False  # disable gold
                state.start_decoding()
                numex += 1

            assert(all([_hg is True for _hg in hasgold]) or all([_hg is False for _hg in hasgold]))
            hasgold = hasgold[0]

            all_terminated = False

            beam_batches = None

            step = 0
            while not all_terminated:
                all_terminated = True
                if beam_batches is None:    # first time
                    fsb.batch()
                    probs, fsb = self.model(fsb)
                    fsb.unbatch()
                    best_probs, best_actions = (-torch.log(probs)).topk(self.beamsize, -1, largest=False)    # (batsize, beamsize) scores and action ids, sorted
                    if (best_probs == np.infty).any():
                        print("whut")
                    beam_batches = [fsb.make_copy() for _ in range(self.beamsize)]
                    best_actions_ = best_actions.cpu().numpy()
                    for i, beam_batch in enumerate(beam_batches):
                        for j, state in enumerate(beam_batch.states):
                            if not state.is_terminated:
                                open_node = state.open_nodes[0]
                                action_str = state.query_encoder.vocab_actions(best_actions_[j, i])
                                state.apply_action(open_node, action_str)
                                all_terminated = False
                else:
                    out_beam_batches = []
                    out_beam_actions = []
                    out_beam_probs = []
                    for k, beam_batch in enumerate(beam_batches):
                        beam_batch.batch()
                        beam_probs, beam_batch = self.model(beam_batch)
                        beam_batch.unbatch()
                        beam_best_probs, beam_best_actions = (-torch.log(beam_probs)).topk(self.beamsize, -1, largest=False)
                        out_beam_probs.append(beam_best_probs + best_probs[:, k:k+1])
                        out_beam_batches.append([k]*self.beamsize)
                        out_beam_actions.append(beam_best_actions)
                    out_beam_probs = torch.cat(out_beam_probs, 1)
                    out_beam_batches = [xe for x in out_beam_batches for xe in x]
                    out_beam_actions = torch.cat(out_beam_actions, 1)

                    beam_best_probs, beam_best_k = torch.topk(out_beam_probs, self.beamsize, -1, largest=False)
                    out_beam_actions_ = out_beam_actions.cpu().numpy()
                    beam_best_k_ = beam_best_k.cpu().numpy()
                    new_beam_batches = []
                    for i in range(beam_best_k.shape[1]):
                        _state_batch = []
                        for j in range(beam_best_k.shape[0]):
                            _state = beam_batches[out_beam_batches[beam_best_k_[j, i]]].states[j]
                            _state = _state.make_copy()
                            if not _state.is_terminated:
                                all_terminated = False
                                open_node = _state.open_nodes[0]
                                action_id = out_beam_actions_[j, beam_best_k_[j, i]]
                                action_str = _state.query_encoder.vocab_actions(action_id)
                                _state.apply_action(open_node, action_str)
                            _state_batch.append(_state)
                        _state_batch = fsb.new(_state_batch)
                        new_beam_batches.append(_state_batch)
                    beam_batches = new_beam_batches
                    best_probs = beam_best_probs
                step += 1
                if step >= self.maxsteps:
                    break
            pass

        all_out_states = beam_batches       # states for whole beam
        all_out_probs = best_probs

        best_out_states = all_out_states[0]
        best_out_probs = all_out_probs[0]

        if hasgold:     # compute accuracies (seq and tree) with top scoring states
            seqaccs = 0
            treeaccs = 0
            # elemaccs = 0
            total = 0
            for best_out_state in best_out_states.states:
                seqaccs += float(best_out_state.out_rules == best_out_state.gold_rules)
                treeaccs += float(str(best_out_state.out_tree) == str(best_out_state.gold_tree))
                # elemaccs += float(best_out_state.out_rules[:len(best_out_state.gold_rules)] == best_out_state.gold_rules)
                total += 1
            return {"output": best_out_states, "output_probs": best_out_probs,
                    "seq_acc": seqaccs/total, "tree_acc": treeaccs/total}
        else:
            return {"output": all_out_states, "probs": all_out_probs}



