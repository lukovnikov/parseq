from unittest import TestCase

import torch

from parseq.states import State, batch, unbatch, make_copy, StateBatch


class Test_tensor_state_batch(TestCase):

    def test_nnstate_batching(self):
        states = [State(*[
            torch.randn(3, 4),
            torch.randint(0, 10, (5,)),
            State(**{"a": torch.rand(1, 3)})
        ]) for _ in range(3)]
        print("Input states:")
        for state in states:
            print(state[0])
            pass
        batched_states = batch(states)
        print("Batched states:")
        print(batched_states[0])
        _batched_states = batched_states.make_copy()
        unbatched_states = unbatch(_batched_states)
        print("Unbatched states:")
        for state in unbatched_states:
            print(state[0])
            pass
        rebatched_state = batch(unbatched_states)
        print("Rebatched states:")
        print(rebatched_state[0])
        for i in range(len(states)):
            self.assertTrue(torch.allclose(states[i][0], unbatched_states[i][0]))
            self.assertTrue(torch.allclose(states[i][1], unbatched_states[i][1]))
            self.assertTrue(torch.allclose(states[i][2]["a"], unbatched_states[i][2].a))

        self.assertTrue(torch.allclose(batched_states[0], rebatched_state[0]))
        self.assertTrue(torch.allclose(batched_states[1], rebatched_state[1]))
        self.assertTrue(torch.allclose(batched_states[2].a, rebatched_state[2].a))

        self.assertTrue(torch.allclose(rebatched_state[0][0], states[0][0]))
        self.assertTrue(torch.allclose(rebatched_state[0][1], states[1][0]))
        self.assertTrue(torch.allclose(rebatched_state[0][2], states[2][0]))

        self.assertTrue(torch.allclose(rebatched_state[1][0], states[0][1]))
        self.assertTrue(torch.allclose(rebatched_state[1][1], states[1][1]))
        self.assertTrue(torch.allclose(rebatched_state[1][2], states[2][1]))

        self.assertTrue(torch.allclose(rebatched_state[2].a[0], states[0][2].a))
        self.assertTrue(torch.allclose(rebatched_state[2].a[1], states[1][2].a))
        self.assertTrue(torch.allclose(rebatched_state[2].a[2], states[2][2].a))

        print(torch.allclose(states[0][0], unbatched_states[0][0]))
        self.assertTrue(torch.allclose(states[0][0], unbatched_states[0][0]))
        unbatched_states[0][0][0] = 1
        print(torch.allclose(states[0][0], unbatched_states[0][0]))
        self.assertFalse(torch.allclose(states[0][0], unbatched_states[0][0]))

    def test_modifications_to_batched_state(self):
        states = [State(torch.randn(5)) for _ in range(3)]
        batched = batch(states)
        y = False
        y = True
        if y:
            batched = make_copy(batched)
        batched["a"] = torch.randn(3, 4)
        unstates = unbatch(batched)

        print(states[0])
        print(states[0][0])
        print(unstates[0].a)
        print(unstates[0][0])
        print(unstates[0][0] is states[0][0])

    def test_unbatch_created_statebatch(self):
        batched = StateBatch()
        batched[0] = torch.randn(3, 5)
        unbatched = batched.unbatch()
        for i, state in enumerate(unbatched):
            print(state[0])
            self.assertTrue(torch.allclose(batched[0][i], state[0]))

    def test_unbatch_statebatch_after_substate_addition(self):
        states = [State(torch.randn(5)) for _ in range(3)]
        batched = batch(states)

        batched[1] = StateBatch()
        batched[1]["a"] = torch.randn(3, 4)

        unbatched = unbatch(batched)

        print(states[0][0])
        print(unbatched[0][0])
        self.assertTrue(states[0][0] is unbatched[0][0])
        print(states[0][1]["a"])
        print(unbatched[0][1]["a"])
        self.assertTrue(torch.allclose(states[0][1]["a"], batched[1]["a"][0]))
