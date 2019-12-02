import time
from unittest import TestCase

import torch
import numpy as np

from parseq.decoding import SeqDecoder, TFTransition, FreerunningTransition, BeamTransition, BeamState
from parseq.states import BasicDecoderState, ListState, State
from parseq.transitions import TransitionModel
from parseq.vocab import SentenceEncoder


class TestSeqDecoder(TestCase):
    def test_tf_decoder(self):
        texts = ["i went to chocolate @END@", "awesome is @END@", "the meaning of life @END@"]
        se = SentenceEncoder(tokenizer=lambda x: x.split())
        for t in texts:
            se.inc_build_vocab(t)
        se.finalize_vocab()
        x = BasicDecoderState(texts, texts, se, se)

        class Model(TransitionModel):
            def forward(self, x:BasicDecoderState):
                outprobs = torch.rand(len(x), x.query_encoder.vocab.number_of_ids())
                return outprobs, x

        dec = SeqDecoder(TFTransition(Model()))

        y = dec(x)
        print(y[1].out_actions)
        outactions = y[1].out_actions.detach().cpu().numpy()
        print(outactions[0])
        print(se.vocab.print(outactions[0]))
        print(se.vocab.print(outactions[1]))
        print(se.vocab.print(outactions[2]))
        self.assertTrue(se.vocab.print(outactions[0]) == texts[0])
        self.assertTrue(se.vocab.print(outactions[1]) == texts[1])
        self.assertTrue(se.vocab.print(outactions[2]) == texts[2])

    def test_free_decoder(self):
        texts = ["i went to chocolate a b c d e f g h i j k l m n o p q r @END@", "awesome is @END@", "the meaning of life @END@"]
        se = SentenceEncoder(tokenizer=lambda x: x.split())
        for t in texts:
            se.inc_build_vocab(t)
        se.finalize_vocab()

        texts = ["@END@"] * 100

        x = BasicDecoderState(texts, texts, se, se)


        class Model(TransitionModel):
            def forward(self, x:BasicDecoderState):
                outprobs = torch.rand(len(x), x.query_encoder.vocab.number_of_ids())
                return outprobs, x

        MAXTIME = 10
        dec = SeqDecoder(FreerunningTransition(Model(), maxtime=MAXTIME))

        y = dec(x)
        print(y[1].out_actions)
        print(max([len(y[1].out_actions[i]) for i in range(len(y[1]))]))
        print(min([len(y[1].out_actions[i]) for i in range(len(y[1]))]))
        self.assertTrue(max([len(y[1].out_actions[i]) for i in range(len(y[1]))]) <= MAXTIME + 1)


class TestBeamTransition(TestCase):
    def test_gather_states(self):
        x = [State(data=torch.rand(3, 4), substate=State(data=np.asarray([0, 1, 2], dtype="int64"))) for _ in range(5)]
        x[1].substate.data[:] = [3, 4, 5]
        x[2].substate.data[:] = [6, 7, 8]
        x[3].substate.data[:] = [9, 10, 11]
        x[4].substate.data[:] = [12, 13, 14]
        print(len(x))
        for i in range(5):
            print(x[i].substate.data)

        indexes = torch.tensor([[0, 0, 1, 0], [1, 1, 1, 0], [2, 4, 2, 0]])
        bt = BeamTransition(None)
        y = bt.gather_states(x, indexes)
        print(indexes)
        print(y)

        for ye in y:
            print(ye.substate.data)

        yemat = torch.tensor([ye.substate.data for ye in y]).T
        print(yemat)

        a = torch.arange(0, 15).reshape(5, 3).T
        b = a.gather(1, indexes)
        print(a)
        print(b)

        self.assertTrue(torch.allclose(b, yemat))

        # for i in range(len(y._list)):
        #     print(y.get(i).substate.data)

    def test_beam_transition(self):
        texts = ["i went to chocolate @END@", "awesome is @END@", "the meaning of life @END@"]
        from parseq.vocab import SentenceEncoder
        se = SentenceEncoder(tokenizer=lambda x: x.split())
        for t in texts:
            se.inc_build_vocab(t)
        se.finalize_vocab()
        x = BasicDecoderState(texts, texts, se, se)
        x.start_decoding()

        class Model(TransitionModel):
            def forward(self, x: BasicDecoderState):
                outprobs = torch.randn(len(x), x.query_encoder.vocab.number_of_ids())
                outprobs = torch.nn.functional.log_softmax(outprobs, -1)
                return outprobs, x

        model = Model()

        beamsize = 50
        maxtime = 10
        beam_xs = [x.make_copy(detach=False, deep=True) for _ in range(beamsize)]
        beam_states = BeamState(beam_xs)

        print(len(beam_xs))
        print(len(beam_states))

        bt = BeamTransition(model, beamsize, maxtime=maxtime)
        i = 0
        _, y, _ = bt(x, i)
        i += 1
        _, y, _ = bt(y, i)

        all_terminated = y.all_terminated()
        while not all_terminated:
            _, y, all_terminated = bt(y, i)
            i += 1

        print("timesteps done:")
        print(i)
        print(y)

    def test_beam_search_vs_greedy(self):
        with torch.no_grad():
            texts = ["a b"] * 10
            from parseq.vocab import SentenceEncoder
            se = SentenceEncoder(tokenizer=lambda x: x.split())
            for t in texts:
                se.inc_build_vocab(t)
            se.finalize_vocab()
            x = BasicDecoderState(texts, texts, se, se)
            x.start_decoding()

            class Model(TransitionModel):
                transition_tensor = torch.tensor([[0, 0, 0, 0, .51, .49],
                                                  [0, 0, 0, 0, .51, .49],
                                                  [0, 0, 0, 0, .51, .49],
                                                  [0, 0, 0, 0, .51, .49],
                                                  [0, 0, 0, 0, .51, .49],
                                                  [0, 0, 0, 0, .01, .99]])
                def forward(self, x: BasicDecoderState):
                    prev = x.prev_actions
                    outprobs = self.transition_tensor[prev]
                    outprobs = torch.log(outprobs)
                    return outprobs, x

            model = Model()

            beamsize = 50
            maxtime = 10
            beam_xs = [x.make_copy(detach=False, deep=True) for _ in range(beamsize)]
            beam_states = BeamState(beam_xs)

            print(len(beam_xs))
            print(len(beam_states))

            bt = BeamTransition(model, beamsize, maxtime=maxtime)
            i = 0
            _, y, _ = bt(x, i)
            i += 1
            _, y, _ = bt(y, i)

            all_terminated = y.all_terminated()
            while not all_terminated:
                start_time = time.time()
                _, y, all_terminated = bt(y, i)
                i += 1
                # print(i)
                end_time = time.time()
                print(f"{i}: {end_time - start_time}")

            print(y)
            print(y.bstates.get(0).out_actions)

    def test_beam_search_stored_probs(self):
        with torch.no_grad():
            texts = ["a b"] * 2
            from parseq.vocab import SentenceEncoder
            se = SentenceEncoder(tokenizer=lambda x: x.split())
            for t in texts:
                se.inc_build_vocab(t)
            se.finalize_vocab()
            x = BasicDecoderState(texts, texts, se, se)
            x.start_decoding()

            class Model(TransitionModel):
                transition_tensor = torch.tensor([[0, 0, 0, 0.01, .51, .48],
                                                  [0, 0, 0, 0.01, .51, .48],
                                                  [0, 0, 0, 0.01, .51, .48],
                                                  [0, 0, 0, 0.01, .51, .48],
                                                  [0, 0, 0, 0.01, .51, .48],
                                                  [0, 0, 0, 0.01, .01, .98]])

                def forward(self, x: BasicDecoderState):
                    prev = x.prev_actions
                    outprobs = self.transition_tensor[prev]
                    outprobs = torch.log(outprobs)
                    outprobs -= 0.01 * torch.rand_like(outprobs)
                    return outprobs, x

            model = Model()

            beamsize = 3
            maxtime = 10
            beam_xs = [x.make_copy(detach=False, deep=True) for _ in range(beamsize)]
            beam_states = BeamState(beam_xs)

            print(len(beam_xs))
            print(len(beam_states))

            bt = BeamTransition(model, beamsize, maxtime=maxtime)
            i = 0
            _, y, _ = bt(x, i)
            i += 1
            _, y, _ = bt(y, i)

            all_terminated = False
            while not all_terminated:
                start_time = time.time()
                _, y, all_terminated = bt(y, i)
                i += 1
                # print(i)
                end_time = time.time()
                print(f"{i}: {end_time - start_time}")

            print(y)
            print(y.bstates.get(0).out_actions)

            best_actions = y.bstates.get(0).out_actions
            best_actionprobs = y.actionprobs.get(0)
            for i in range(len(best_actions)):
                print(i)
                i_prob = 0
                for j in range(len(best_actions[i])):
                    action_id = best_actions[i, j]
                    action_prob = best_actionprobs[i, j, action_id]
                    i_prob += action_prob
                print(i_prob)
                print(y.bscores[i, 0])
                self.assertTrue(torch.allclose(i_prob, y.bscores[i, 0]))



