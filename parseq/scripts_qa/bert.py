import copy
from functools import partial
from typing import List, Union, Dict

import torch
import numpy as np
import qelos as q
from torch import nn
import transformers
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, BertTokenizer, BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import BertModel, BertAttention, BertLayer, BertOutput, BertEmbeddings


# use 100k LM pre-trained T5 weights instead of normal weights! --> https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md

class CosineWithRestart(q.sched.Cosine):
    def _get_lr(self, i):
        perc = (i / (self.num_steps / self.cycles)) % 1.0
        ret = (np.cos((perc / 2 + self.phase) * 2 * np.pi) * self._scale) + self._add
        return ret


def try_cosine():
    import matplotlib.pyplot as plt
    sched = CosineWithRestart(cycles=3, steps=100) * q.sched.Linear(1, 0, steps=100)
    x = list(range(100))
    y = [sched.get_lr(i) for i in x]
    plt.plot(x, y)
    plt.show()


class KVMem(torch.nn.Module):
    def __init__(self, dim, memdim, memsize):
        super().__init__()
        self.dim, self.memdim, self.memsize = dim, memdim, memsize

        self.keys = torch.nn.Linear(self.memdim, self.memsize)
        self.sm = torch.nn.Softmax(-1)
        self.values = torch.nn.Linear(self.memsize, self.memdim)

        self.reset_parameters()

    def reset_parameters(self):
        self.keys.reset_parameters()
        self.values.reset_parameters()

    def forward(self, h):
        weights = self.keys(h)
        weights = self.sm(weights)
        ret = self.values(weights)
        return ret


class MemAdapter(torch.nn.Module):
    gated = False
    gatebias = -4

    def __init__(self, dim, memdim, memsize, dropout=0., kvmem=None, **kw):
        super().__init__(**kw)
        self.dim = dim
        self.memdim, self.memsize = memdim, memsize

        self.mem = KVMem(self.dim, self.memdim, self.memsize) if kvmem is None else kvmem

        self.wi = torch.nn.Linear(self.dim, self.memdim)
        self.wo = torch.nn.Linear(self.memdim, self.dim)
        if self.gated == "full":
            self.wg = torch.nn.Linear(self.memdim, self.dim)  # gate weights
        elif self.gated == "bias":
            self.gb = torch.nn.Parameter(torch.zeros(self.dim))
        self.sg = torch.nn.Linear(self.memdim, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.nonlin = torch.nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        small = 0.001
        if self.gated == "full":
            torch.nn.init.uniform_(self.wo.weight, -small, +small)
            torch.nn.init.constant_(self.wg.bias, 0)
        elif self.gated == "bias":
            torch.nn.init.constant_(self.gb, 0)
        torch.nn.init.uniform_(self.sg.weight, -small, +small)
        torch.nn.init.constant_(self.sg.bias, 0)
        self.wi.reset_parameters()
        self.wo.reset_parameters()
        # torch.nn.init.uniform_(self.wi.weight, -small, +small)
        # torch.nn.init.uniform_(self.wo.weight, -small, +small)
        # torch.nn.init.constant_(self.wi.bias, 0)
        # torch.nn.init.constant_(self.wo.bias, 0)

    def forward(self, h):
        _h = h
        _h = self.wi(_h)
        __h = self.nonlin(_h)
        __h = self.dropout(__h)

        _h = self.kvmem(_h)

        new_h = self.wo(_h)

        gate = None

        if self.gated == "full":
            gate = torch.sigmoid(self.wg(__h) + self.sg(__h) + self.gatebias)
        elif self.gated == "bias":
            gate = torch.sigmoid(self.gb + self.sg(__h) + self.gatebias)

        if gate is not None:
            new_h = gate * new_h + (1 - gate) * h
        else:
            new_h = new_h + h
        return new_h


class AdapterFF(torch.nn.Module):
    gated = False
    gatebias = -4
    def __init__(self, dim, adapterdim, dropout=0., **kw):
        super().__init__(**kw)
        self.dim = dim
        self.adapterdim = adapterdim
        self.wi = torch.nn.Linear(self.dim, self.adapterdim)
        self.wo = torch.nn.Linear(self.adapterdim, self.dim)
        if self.gated == "full":
            self.wg = torch.nn.Linear(self.adapterdim, self.dim)  # gate weights
        elif self.gated == "bias":
            self.gb = torch.nn.Parameter(torch.zeros(self.dim))
        self.dropout = torch.nn.Dropout(dropout)
        self.nonlin = torch.nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        small = 0.001
        if self.gated == "full":
            torch.nn.init.uniform_(self.wo.weight, -small, +small)
            torch.nn.init.constant_(self.wg.bias, 0)
        elif self.gated == "bias":
            torch.nn.init.constant_(self.gb, 0)
        torch.nn.init.uniform_(self.wi.weight, -small, +small)
        torch.nn.init.uniform_(self.wo.weight, -small, +small)
        torch.nn.init.constant_(self.wi.bias, 0)
        torch.nn.init.constant_(self.wo.bias, 0)

    def forward(self, h):
        _h = h
        _h = self.wi(_h)
        _h = self.nonlin(_h)
        _h = self.dropout(_h)
        new_h = self.wo(_h)
        if self.gated == "full":
            g = torch.sigmoid(self.wg(_h) + self.gatebias)
            new_h = g * new_h + (1 - g) * h
        elif self.gated == "bias":
            g = torch.sigmoid(self.gb + self.gatebias)
            new_h = g * new_h + (1 - g) * h
        else:
            new_h = new_h + h
        return new_h


class AdapterGatedFF(AdapterFF):
    gated = "full"


class AdapterBiasGatedFF(AdapterFF):
    gated = "bias"


class AdapteredBertOutput(torch.nn.Module):
    def __init__(self, bertoutput:BertOutput, adapter):
        super(AdapteredBertOutput, self).__init__()
        self.m = bertoutput
        self.adapterff = adapter

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.m.dense(hidden_states)
        hidden_states = self.m.dropout(hidden_states)

        hidden_states = self.adapterff(hidden_states)

        hidden_states = self.m.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def get_ft_params(self):
        return list(self.adapterff.parameters()) + list(self.m.LayerNorm.parameters())


def insert_adapters(m:torch.nn.Module, dim=None, adapterdim=None, adaptmode="ada"):
    selectors = set([x for x in adaptmode.split("+") if x in ("ff", "self", "cross")])
    if len(selectors) == 0:
        selectors = {"ff", "self", "cross"}
        # print(f"Adapting based on selectors: {selectors}")
    if isinstance(m, BertAttention) and "self" in selectors:
        m.output = AdapteredBertOutput(m.output, dim=dim, adapterdim=adapterdim)
    elif isinstance(m, BertLayer) and "ff" in selectors:
        m.output = AdapteredBertOutput(m.output, dim=dim, adapterdim=adapterdim)


class VanillaBERT(BertModel):
    def get_tunable_params_and_set_requires_grad(self):
        ret = list(self.parameters())
        return ret

    def set_dropout(self, p=0.):
        def _set_dropout(m, _p=0.):
            if isinstance(m, torch.nn.Dropout):
                m.p = _p
        self.apply(partial(_set_dropout, _p=p))


def mem_adapterize_layer(layer, dim, adapterdim, memsize=1000, kvmem=None, adaptmode="ada"):
    selectors = set([x for x in adaptmode.split("+") if x in ("ff", "self", "cross")])
    if len(selectors) == 0:
        selectors = {"ff", "self", "cross"}
        # print(f"Adapting based on selectors: {selectors}")
    if "ff" in selectors:
        memadapter = MemAdapter(dim=dim, memdim=adapterdim, memsize=memsize, kvmem=kvmem)
        layer.output = AdapteredBertOutput(layer.output, memadapter)
    if "self" in selectors:
        adapterff = AdapterFF(dim=dim, adapterdim=adapterdim)
        layer.attention.output = AdapteredBertOutput(layer.attention.output, adapterff)


class AdaptedBertWordEmbeddings(torch.nn.Module):
    def __init__(self, original_emb:torch.nn.Embedding, tok:Union[BertTokenizer, BertTokenizerFast]):
        super(AdaptedBertWordEmbeddings, self).__init__()
        self.original_emb = original_emb
        # self.tok = tok

        if isinstance(tok, BertTokenizer):
            self.added_tokens = tok.added_tokens_encoder
        elif isinstance(tok, BertTokenizerFast):
            self.added_tokens = dict(zip(tok.additional_special_tokens, tok.additional_special_tokens_ids))


        orig_mapper = torch.arange(0, max(list(tok.vocab.values()) + list(self.added_tokens.values()))+1, dtype=torch.long)
        xtra_mapper = torch.zeros_like(orig_mapper)
        masker = torch.zeros_like(orig_mapper).to(torch.bool)
        newidx = 1
        for added_token in self.added_tokens.values():
            orig_mapper[added_token] = tok.vocab["[UNK]"]
            xtra_mapper[added_token] = newidx
            masker[added_token] = 1
            newidx += 1

        self.register_buffer("orig_mapper", orig_mapper)
        self.register_buffer("xtra_mapper", xtra_mapper)
        self.register_buffer("masker", masker)

        self.xtra_emb = torch.nn.Embedding(max(xtra_mapper)+1, original_emb.embedding_dim)

    def forward(self, x:torch.Tensor):
        orig_x = self.orig_mapper[x]
        xtra_x = self.xtra_mapper[x]
        originalemb = self.original_emb(orig_x)
        xtraemb = self.xtra_emb(xtra_x)
        mask = self.masker[x]
        emb = torch.where(mask.unsqueeze(-1), xtraemb, originalemb)
        return emb

    def get_ft_params(self):
        return self.xtra_emb.parameters()


def adapt_embeddings(embeddings:BertEmbeddings, tok:Union[BertTokenizer, BertTokenizerFast]):
    adaptedembs = AdaptedBertWordEmbeddings(embeddings.word_embeddings, tok)
    embeddings.word_embeddings = adaptedembs
    return embeddings


class MemAdapteredBERT(BertModel):
    def adapt(self,
              adapterdim=64,  # dimension of adapters, only used if pt_type is "ada(pters)"
              memsize=1000,
              topk=4,        # insert adapters only in higher topk layers
              sharemem=True,
              tokenizer=None,
              ):
        self.adapterdim = adapterdim
        self.memsize = memsize
        self.sharemem = sharemem

        if sharemem:
            kvmem = KVMem(self.config.hidden_size, self.adapterdim, self.memsize)
        else:
            kvmem = None

        numlayer = len(self.encoder.layer)
        for i, layer in enumerate(self.encoder.layer):
            if numlayer - i <= topk:
                mem_adapterize_layer(layer, dim=self.config.hidden_size, adapterdim=self.adapterdim, memsize=memsize, kvmem = kvmem)

        self.embeddings = adapt_embeddings(self.embeddings, tokenizer)

        # self.encoder.apply(partial(insert_adapters, dim=self.config.hidden_size, adapterdim=self.adapterdim))

    def get_tunable_params_and_set_requires_grad(self):
        ret = []

        def _get_ft_params(m, out=None):
            if hasattr(m, "get_ft_params"):
                out.extend(m.get_ft_params())

        collectedparams = []
        self.apply(partial(_get_ft_params, out=collectedparams))

        ret += collectedparams

        for param in self.parameters():
            computegrad = False
            for tunableparam in ret:
                if param is tunableparam:
                    computegrad = True
            param.requires_grad = computegrad

        return ret

    def set_dropout(self, p=0.):
        def _set_dropout(m, _p=0.):
            if isinstance(m, torch.nn.Dropout):
                m.p = _p
        self.apply(partial(_set_dropout, _p=p))


def adapterize_layer(layer, dim, adapterdim, adaptmode="ada"):
    selectors = set([x for x in adaptmode.split("+") if x in ("ff", "self", "cross")])
    if len(selectors) == 0:
        selectors = {"ff", "self", "cross"}
        # print(f"Adapting based on selectors: {selectors}")
    if "ff" in selectors:
        adapterff = AdapterFF(dim=dim, adapterdim=adapterdim)
        layer.output = AdapteredBertOutput(layer.output, adapterff)
    if "self" in selectors:
        adapterff = AdapterFF(dim=dim, adapterdim=adapterdim)
        layer.attention.output = AdapteredBertOutput(layer.attention.output, adapterff)


class AdapteredBERT(BertModel):
    def adapt(self,
              adapterdim=64,  # dimension of adapters, only used if pt_type is "ada(pters)"
              topk=1000,        # insert adapters only in higher topk layers
              ):
        self.adapterdim = adapterdim
        numlayer = len(self.encoder.layer)
        for i, layer in enumerate(self.encoder.layer):
            if numlayer - i <= topk:
                adapterize_layer(layer, dim=self.config.hidden_size, adapterdim=self.adapterdim)

        # self.encoder.apply(partial(insert_adapters, dim=self.config.hidden_size, adapterdim=self.adapterdim))

    def get_tunable_params_and_set_requires_grad(self):
        ret = []

        def _get_ft_params(m, out=None):
            if hasattr(m, "get_ft_params"):
                out.extend(m.get_ft_params())

        collectedparams = []
        self.apply(partial(_get_ft_params, out=collectedparams))

        ret += collectedparams

        for param in self.parameters():
            computegrad = False
            for tunableparam in ret:
                if param is tunableparam:
                    computegrad = True
            param.requires_grad = computegrad

        return ret

    def set_dropout(self, p=0.):
        def _set_dropout(m, _p=0.):
            if isinstance(m, torch.nn.Dropout):
                m.p = _p
        self.apply(partial(_set_dropout, _p=p))


def try_generate(inp:str=None, model=None, tokenizer=None):
    if inp is None:
        inp = "When I get up in the morning, the first thing I do is eat breakfast. I should also take a "
    input_ids = tokenizer(inp, return_tensors='pt').input_ids
    print(f"input ids: {input_ids}")
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def try_train(inps:List[str], outs:List[str], model=None, tokenizer=None):
    params = model.get_tunable_params_and_set_requires_grad()

    inputs = tokenizer(inps, return_tensors='pt', padding=True)
    outputs = tokenizer(outs, return_tensors='pt', padding=True)

    modelout = model(input_ids=inputs.input_ids,
                 attention_mask=inputs.attention_mask,
                 labels=outputs.input_ids)

    print(f"Loss: {modelout.loss}")

    modelout.loss.backward()
    print("done backward")


def load_bert_tokenizer(modelsize="base"):
    print(f"Transformers version: {transformers.__version__}")
    tt = q.ticktock("script")
    modelname = "BERT"
    tt.tick(f"loading {modelsize} {modelname}")
    modelname = f"bert-{modelsize}-uncased"
    tt.msg(f"modelname: {modelname}")
    tt.tick("loading tokenizer")
    tokenizer = BertTokenizer.from_pretrained(modelname)
    tt.tock("loaded")
    return tokenizer


def load_vanilla_bert(modelsize="base"):
    print(f"Transformers version: {transformers.__version__}")
    tt = q.ticktock("script")
    modelname = "BERT"
    tt.tick(f"loading {modelsize} {modelname}")
    modelname = f"bert-{modelsize}-uncased"
    tt.msg(f"modelname: {modelname}")
    tt.tick("loading tokenizer")
    tokenizer = BertTokenizer.from_pretrained(modelname)
    tt.tock("loaded")
    tt.tick(f"loading normal {modelname} from huggingface")
    model = VanillaBERT.from_pretrained(modelname)

    tt.tock("loaded")
    tt.tock("loaded everything")
    return tokenizer, model


def load_adaptered_bert(modelsize="base", adapterdim=64, usemem=False):
    print(f"Transformers version: {transformers.__version__}")
    tt = q.ticktock("script")
    modelname = "BERT"
    tt.tick(f"loading {modelsize} {modelname}")
    modelname = f"bert-{modelsize}-uncased"
    tt.msg(f"modelname: {modelname}")
    tt.tick("loading tokenizer")
    tokenizer = BertTokenizer.from_pretrained(modelname)
    tt.tock("loaded")

    if not usemem:
        tt.tick(f"loading Adapter-inserted {modelname} from huggingface")
        model = AdapteredBERT.from_pretrained(modelname)
    else:
        tt.tick(f"loading KV Mem Adapter-inserted {modelname} from huggingface")
        model = MemAdapteredBERT.from_pretrained(modelname)

    tt.tock("loaded")
    tt.msg(f"Inserting {adapterdim}-dim KV mem adapter layers.")
    model.adapt(adapterdim=adapterdim)

    tt.tock("loaded everything")
    return tokenizer, model


def main(lr=0.001):
    out_vocab_size = 1000000
    tokenizer, model = load_vanilla_bert("small", out_vocab_size=out_vocab_size)

    # try_generate(model=model, tokenizer=tokenizer)

    inps = ["Hello, my name is", "I will find you"]
    outs = ["Hallo, mijn naam is", "Ik zal je vinden"]
    try_train(inps, outs, model=model, tokenizer=tokenizer)

    # inputs = tokenizer(['0 translate English to German: The house is wonderful.', 'translate English to German: the house is.'], return_tensors='pt', truncation=True, padding=True)
    # decinp = tokenizer(['0 Das Haus ist wunderbar', '0 Das haus'], return_tensors='pt', truncation=True, padding=True)
    # labels = tokenizer(['Das Haus ist wunderbar.', 'Das haus.'], return_tensors='pt', truncation=True, padding=True)
    # # the forward function automatically creates the correct decoder_input_ids
    # loss = model(input_ids=inputs.input_ids,
    #              attention_mask=inputs.attention_mask,
    #              decoder_input_ids=decinp.input_ids,
    #              decoder_attention_mask=decinp.attention_mask,
    #              labels=labels.input_ids)
    # loss = loss.loss
    # print(f"Loss: {loss}")


if __name__ == '__main__':
    # try_load_vanilla()
    q.argprun(main)
    # try_cosine()