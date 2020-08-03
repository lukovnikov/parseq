from unittest import TestCase
import numpy as np
import torch

from parseq.scrips_mem_meta.overnight_mem_meta import MetaSeqMemNN, DecoderOutputLayer


class TestMetaSeqMemNN(TestCase):
    def test_align_inputs(self):
        m = MetaSeqMemNN(None, None, None, None, 8)
        x_enc_base = torch.randn(2, 3, 8)
        xsup_enc = torch.randn(2, 5, 4, 8)
        supmask = torch.tensor([
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
             ],
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
             ],
        ])

        att, summ, attweights = m.align_inputs(x_enc_base, xsup_enc, supmask=supmask)
        print(att.size())
        print(summ.size())
        self.assertTrue(att.size() == (2, 3, 5, 4))
        self.assertTrue(summ.size() == (2, 3, 8))

    def test_decoder_output_layer(self):
        m = DecoderOutputLayer(8, 10, {4, 6, 7, 8, 9})
        enc = torch.nn.Parameter(torch.randn(2, 8))
        summ = torch.nn.Parameter(torch.randn(2, 8))
        memencs = torch.nn.Parameter(torch.randn(2, 5, 4, 8))
        memids = torch.tensor([
            [
                [1, 2, 3, 4],
                [5, 6, 3, 4],
                [1, 3, 4, 2],
                [1, 7, 3, 4],
                [1, 2, 0, 4],
             ],
            [
                [1, 7, 5, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
             ],
        ])
        memmask = torch.tensor([
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
             ],
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
             ],
        ])

        probs, mem_summ = m(enc, summ, memids=memids, memencs=memencs, memmask=memmask)
        print(probs.size())
        # self.assertTrue(np.isclose(probs.sum(1)[0].item(), 1))

        # try grad
        loss = - torch.log(probs[0, 9]) - torch.log(probs[1, 5])
        loss.backward()
        print(summ.grad)
        print(memencs.grad)


