from unittest import TestCase
import numpy as np
import torch
from torch.utils.data import DataLoader

from parseq.datasets import autocollate
from parseq.scrips_mem_meta.overnight_mem_meta_abs import MetaSeqMemNN, DecoderOutputLayer, DecoderInputLayer, \
    LSTMDecoderCellWithMemory, StateDecoderWithMemory, StateInnerDecoder, InnerLSTMDecoderCell, load_ds, pack_loaded_ds, \
    load_data, create_lstm_model, SuperBasicDecoderState
import qelos as q


class TestLoadDS(TestCase):
    def test_load_ds_lex(self):
        """dummy commits"""
        tt = q.ticktock()
        batsize = 4
        numworkers = 0
        traindl, validdl, testdl, nltok, flenc, tokenmasks = \
            load_data(traindomains=("restaurants", "blocks"), testdomain="housing",
                supportsetting="lex", batsize=batsize, numworkers=numworkers)
        tt.tick("first round")
        for batch in traindl:
            l = len(batch)
            # print(len(batch))
        tt.tock("first round done")

        tt.tick("second round")
        for batch in traindl:
            l = len(batch)
            # print(len(batch))
        tt.tock("second round done")
        print(tokenmasks)

    def test_load_ds_min(self):
        batsize = 4
        numworkers = 0
        tt = q.ticktock()
        traindl, validdl, testdl, nltok, flenc, tokenmasks = \
            load_data(traindomains=("restaurants", "blocks"), testdomain="housing",
                supportsetting="min", batsize=batsize, numworkers=numworkers)
        tt.tick("first round")
        for batch in traindl:
            l = len(batch)
            # print(len(batch))
        tt.tock("first round done")

        tt.tick("second round")
        for batch in traindl:
            l = len(batch)
            # print(len(batch))
        tt.tock("second round done")
        print(tokenmasks)



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

    def test_decoder_input_layer(self):
        m = DecoderInputLayer(10, 8, unkid=1, unktoks={4, 6, 7, 8, 9})
        x = torch.tensor([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ])
        a = torch.randn(10, 8)
        b = torch.randn(10, 8)
        y = m(x, a, b)
        print(y)

    def create_state_with_memory(self, cell, vocsize, batsize, dim):
        x_enc = torch.randn(batsize, 3, dim)
        y = torch.randint(0, vocsize, (batsize, 4))
        memencs = torch.randn(batsize, 5, 4, dim)
        memids = torch.randint(0, vocsize, (batsize, 5, 4))
        xmask = torch.ones_like(x_enc[:, :, 0])
        memmask = torch.ones_like(memencs[:, :, :, 0])
        state = StateDecoderWithMemory.create_state(cell, x_enc, y,
                                                    memencs, memids, xmask, memmask)
        return state

    def test_lstm_decoder_cell_with_memory(self):
        unktoks = {4, 6, 7, 9}
        inplayer = DecoderInputLayer(10, 8, unktoks=unktoks)
        outlayer = DecoderOutputLayer(8, 10, unktoks=unktoks)
        m = LSTMDecoderCellWithMemory(inplayer, 8, outlayer=outlayer)
        state = self.create_state_with_memory(m, 10, 2, 8)
        print(state)
        out, newstate = m(state)
        print(out)
        print(out.sum(1))
        self.assertTrue(torch.allclose(out.sum(1), torch.ones_like(out.sum(1))))

    def create_inner_state(self, cell, vocsize, batsize, dim):
        ctx = torch.randn(batsize, 3, dim)
        y = torch.randint(0, vocsize, (batsize, 4))
        ctxmask = torch.ones_like(ctx[:, :, 0])
        state = StateInnerDecoder.create_state(cell, y, ctx, ctxmask=ctxmask)
        return state

    def test_inner_lstm_decoder_cell(self):
        unktoks = {4, 6, 7, 9}
        inplayer = DecoderInputLayer(10, 8, unktoks=unktoks)
        m = InnerLSTMDecoderCell(inplayer, dim=8)
        state = self.create_inner_state(m, 10, 2, 8)
        print(state)
        out, newstate = m(state)
        print(out)

    def test_inner_decoder(self):
        unktoks = {4, 6, 7, 9}
        inplayer = DecoderInputLayer(10, 8, unktoks=unktoks)
        cell = InnerLSTMDecoderCell(inplayer, dim=8)
        m = StateInnerDecoder(cell)
        batsize, dim, vocsize = 2, 8, 10
        y = torch.randint(0, vocsize, (batsize, 4))
        y[:, -1] = 3
        ctx = torch.randn(batsize, 3, dim)
        ctxmask = torch.ones_like(ctx[:, :, 0])
        out = m(y, ctx, ctxmask=ctxmask)
        print(out)
        print(out.size())

    def test_dummy_meta_seq_mem_nn(self):
        class Dummy(torch.nn.Module):
            def __init__(self, shape, **kw):
                super(Dummy, self).__init__(**kw)
                self.shape = shape
            def forward(self, *args, **kw):
                return torch.randn(*self.shape)

        memenc = Dummy((10, 4, 8))
        memdec = Dummy((10, 6, 8))
        enc = Dummy((2, 3, 8))
        unktoks = {4, 6, 7, 9}
        inplayer = DecoderInputLayer(10, 8, unktoks=unktoks)
        outlayer = DecoderOutputLayer(8, 10, unktoks=unktoks)
        cell = LSTMDecoderCellWithMemory(inplayer, 8, outlayer=outlayer)
        dec = StateDecoderWithMemory(cell)
        m = MetaSeqMemNN(enc, memenc, dec, memdec, 8, 0.2)

        x = torch.randint(0, 10, (2, 3))
        y = torch.randint(0, 10, (2, 5))
        y[:, -1] = 3
        xsup = torch.randint(1, 10, (2, 5, 4))
        ysup = torch.randint(1, 10, (2, 5, 6))
        xsup[0, 0, 1:] = 0
        xsup[0, 1, 2:] = 0
        xsup[1, :, 1:] = 0
        ysup[0, 0, 3:] = 0
        ysup[0, 1, 2:] = 0
        ysup[1, :, 1:] = 0
        supmask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0]
        ])

        o = m(x, y, xsup, ysup, supmask)
        print(o)
        print(o[1].followed_actions)
        print(y)


    def test_meta_seq_mem_nn(self):
        class Encoder(torch.nn.Module):
            def __init__(self, vocsize, dim, **kw):
                super(Encoder, self).__init__(**kw)
                self.emb = torch.nn.Embedding(vocsize, dim)
                self.enc = torch.nn.LSTM(dim, dim//2, 1, batch_first=True, bidirectional=True)

            def forward(self, x:torch.LongTensor, **kw):
                emb = self.emb(x)
                enc = self.enc(emb)
                return enc[0]

        enc = Encoder(100, 8)
        unktoks = {4, 6, 7, 9}
        inplayer = DecoderInputLayer(10, 8, unktoks=unktoks)
        outlayer = DecoderOutputLayer(8, 10, unktoks=unktoks)
        cell = LSTMDecoderCellWithMemory(inplayer, 8, outlayer=outlayer)
        dec = StateDecoderWithMemory(cell)
        memcell = InnerLSTMDecoderCell(inplayer, 8)
        memdec = StateInnerDecoder(memcell)
        m = MetaSeqMemNN(enc, enc, dec, memdec, 8, 0.2)

        x = torch.randint(0, 100, (2, 3))
        y = torch.randint(0, 10, (2, 5))
        y[:, -1] = 3
        xsup = torch.randint(1, 100, (2, 5, 4))
        ysup = torch.randint(1, 10, (2, 5, 6))
        xsup[0, 0, 1:] = 0
        xsup[0, 1, 2:] = 0
        xsup[1, :, 1:] = 0
        ysup[0, 0, 3:] = 0
        ysup[0, 1, 2:] = 0
        ysup[1, :, 1:] = 0
        ysup[:, :, -1] = 3
        supmask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0]
        ])

        o = m(x, y, xsup, ysup, supmask)
        print(o)
        print(o[1].followed_actions)
        print(y)

    def test_lstm_model_with_real_data(self):
        traindl, validdl, testdl, nltok, flenc, tokenmasks = \
            load_data(traindomains=("restaurants", "blocks"), testdomain="housing",
                supportsetting="lex")

        inp_vocabsize = max(nltok.vocab.values()) + 1
        out_vocabsize = flenc.vocab.number_of_ids()

        dim = 100

        unktokens = set(tokenmasks["_metarare"].nonzero()[:, 0].cpu().numpy())

        class Encoder(torch.nn.Module):
            def __init__(self, vocsize, dim, **kw):
                super(Encoder, self).__init__(**kw)
                self.emb = torch.nn.Embedding(vocsize, dim)
                self.enc = torch.nn.LSTM(dim, dim//2, 1, batch_first=True, bidirectional=True)

            def forward(self, x:torch.LongTensor, **kw):
                emb = self.emb(x)
                enc = self.enc(emb)
                return enc[0]

        enc = Encoder(inp_vocabsize, dim)
        m = create_lstm_model(enc, out_vocabsize, dim, unktokens=unktokens, eos_id=3)

        batch = next(iter(traindl))

        outs = m(*batch)
        print(outs)


if __name__ == '__main__':
    tlds = TestLoadDS()
    tlds.test_load_ds_lex()