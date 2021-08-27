import torch
from parseq.rnn1 import Encoder
from parseq.vocab import Vocab


class Embeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, vocab_size, hidden_size, dropout=0., pad_token_id=0,
                 layer_norm_eps=1e-12):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        ret = inputs_embeds

        ret = self.LayerNorm(ret)
        ret = self.dropout(ret)
        return ret


class WordDropout(torch.nn.Module):
    def __init__(self, p=0., maskid=1, keepids=None, **kw):
        super(WordDropout, self).__init__(**kw)
        self.dropout = torch.nn.Dropout(p)
        self.maskid = maskid
        self.keepids = [] if keepids is None else keepids

    def forward(self, x):
        if self.training and self.dropout.p > 0:
            worddropoutmask = self.dropout(torch.ones_like(x).float()) > 0
            for keepid in self.keepids:
                worddropoutmask = worddropoutmask | (x == keepid)
            x = torch.where(worddropoutmask, x, torch.ones_like(x) * self.maskid)
        return x


class GRUDecoderCellDecoder(torch.nn.Module):
    def __init__(self, dim, vocab:Vocab=None, inpvocab:Vocab=None, numlayers:int=2,
                 dropout:float=0., worddropout:float=0., **kw):
        super(GRUDecoderCellDecoder, self).__init__(**kw)
        self.vocab = vocab
        self.inpvocab = inpvocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim

        self.dec_emb = torch.nn.Embedding(self.vocabsize+3, self.dim)
        dims = [self.dim + self.dim] + [self.dim for _ in range(numlayers)]
        self.dec_stack = torch.nn.ModuleList([torch.nn.GRUCell(dims[i], dims[i+1]) for i in range(numlayers)])
        self.dropout = torch.nn.Dropout(dropout)
        self.attn_linQ = None
        self.attn_linK = None
        self.attn_linV = None
        # self.attn_linQ = torch.nn.Linear(self.dim, self.dim)
        # self.attn_linK = torch.nn.Linear(self.dim, self.dim)
        # self.attn_linV = torch.nn.Linear(self.dim, self.dim)


        self.preout = torch.nn.Linear(self.dim + self.dim, self.dim)
        self.out = torch.nn.Linear(self.dim, self.vocabsize+3)

        self.worddropout = WordDropout(worddropout, self.vocab[self.vocab.masktoken], [self.vocab[self.vocab.padtoken]])

        self.reset_parameters()

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None):
        # assert tokens.size(1) == 1
        if tokens.size(1) > 1:
            assert cache is not None
        tokens = tokens[:, -1]
        padmask = (tokens != 0)
        embs = self.dec_emb(tokens)
        if cache is None:
            cache = {"states": [{"h_tm1": torch.zeros(enc.size(0), self.dim, device=enc.device)} for _ in self.dec_stack],
                     "prevatt": torch.zeros_like(enc[:, 0])}

        prev_att = cache["prevatt"]
        inps = torch.cat([embs, prev_att], -1)
        for l, layer in enumerate(self.dec_stack):
            prev_state = cache["states"][l]["h_tm1"]
            inps = self.dropout(inps)
            h_t = layer(inps, prev_state)
            cache["states"][l]["h_tm1"] = h_t
            inps = h_t

        if self.attn_linQ is not None:
            h_t = self.attn_linQ(h_t)
        if self.attn_linK is not None:
            encK = self.attn_linK(enc)
        else:
            encK = enc
        if self.attn_linV is not None:
            encV = self.attn_linV(enc)
        else:
            encV = enc

        # attention
        weights = torch.einsum("bd,bsd->bs", h_t, encK)
        weights = weights.masked_fill(encmask == 0, float('-inf'))
        alphas = torch.softmax(weights, -1)
        summary = torch.einsum("bs,bsd->bd", alphas, encV)
        cache["prevatt"] = summary

        out = torch.cat([h_t, summary], -1)
        out = self.preout(out)
        logits = self.out(out)

        logits = logits[:, None]
        return logits, cache


class GRUDecoderCell(torch.nn.Module):
    def __init__(self, dim, vocab:Vocab=None, inpvocab:Vocab=None, numlayers:int=2,
                 dropout:float=0., worddropout:float=0., useskip=False, **kw):
        super(GRUDecoderCell, self).__init__(**kw)

        self.vocab = vocab
        self.inpvocab = inpvocab
        self.vocabsize = vocab.number_of_ids()
        self.dim = dim

        self.useskip = useskip
        inpvocabsize = inpvocab.number_of_ids()
        self.encoder = Encoder(inpvocabsize + 5, self.dim, int(self.dim / 2), num_layers=numlayers, dropout=dropout, useskip=self.useskip)

        self.inpworddropout = WordDropout(worddropout, self.inpvocab[self.inpvocab.masktoken],
                                          [self.inpvocab[self.inpvocab.padtoken]])

        self.decoder = GRUDecoderCellDecoder(dim, vocab=vocab, inpvocab=inpvocab, numlayers=numlayers, dropout=dropout, worddropout=worddropout, **kw)
        self.reset_parameters()

    def reset_parameters(self):
        pass
        # self.posemb.weight.fill_(0.)

    def encode_source(self, x):
        encmask = (x != 0)
        x = self.inpworddropout(x)
        encs = self.encoder(x, attention_mask=encmask)[0]
        return encs, encmask

    def forward(self, tokens:torch.Tensor=None, enc=None, encmask=None, cache=None, _full_sequence=False):
        if _full_sequence:
            # iterate over tokens
            logitses = []
            for i in range(tokens.size(1)):
                logits, cache = self.decoder(tokens=tokens[:, i:i+1], enc=enc, encmask=encmask, cache=cache)
                logitses.append(logits)
            logitses = torch.cat(logitses, 1)
            return logitses, cache
        else:
            return self.decoder(tokens=tokens, enc=enc, encmask=encmask, cache=cache)