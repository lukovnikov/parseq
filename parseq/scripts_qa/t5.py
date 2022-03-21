import copy
from functools import partial
from typing import List, Union

import torch
import numpy as np
import qelos as q
from torch import nn
import transformers
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm, load_tf_weights_in_t5, T5LayerFF, \
    T5LayerSelfAttention, T5LayerCrossAttention


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


class AdapteredT5LayerFF(T5LayerFF):
    @classmethod
    def cast(cls, obj:T5LayerFF, dim=None, adapterdim=None):
        obj.__class__ = cls
        obj.adapterff = AdapterBiasGatedFF(dim, adapterdim)

    def forward(self, h):
        _h = h
        _h = self.layer_norm(_h)
        _h = self.DenseReluDense(_h)
        _h = self.adapterff(_h)
        newh = h + self.dropout(_h)
        return newh

    def set_custom_dropout(self, p):
        self.adapterff.dropout.p = p

    def get_ft_params(self):
        return list(self.adapterff.parameters()) + list(self.layer_norm.parameters())


class AdapteredT5LayerSelfAttention(T5LayerSelfAttention):
    @classmethod
    def cast(cls, obj:T5LayerFF, dim=None, adapterdim=None):
        obj.__class__ = cls
        obj.adapterff = AdapterBiasGatedFF(dim, adapterdim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        x = self.adapterff(attention_output[0])
        hidden_states = hidden_states + self.dropout(x)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

    def get_ft_params(self):
        return list(self.adapterff.parameters()) + list(self.layer_norm.parameters())


class AdapteredT5LayerCrossAttention(T5LayerCrossAttention):
    @classmethod
    def cast(cls, obj:T5LayerFF, dim=None, adapterdim=None):
        obj.__class__ = cls
        obj.adapterff = AdapterBiasGatedFF(dim, adapterdim)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        x = self.adapterff(attention_output[0])
        layer_output = hidden_states + self.dropout(x)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

    def get_ft_params(self):
        return list(self.adapterff.parameters()) + list(self.layer_norm.parameters())


def insert_adapters(m:torch.nn.Module, dim=None, adapterdim=None, adaptmode="ada"):
    selectors = set([x for x in adaptmode.split("+") if x in ("ff", "self", "cross")])
    if len(selectors) == 0:
        selectors = {"ff", "self", "cross"}
        # print(f"Adapting based on selectors: {selectors}")
    if isinstance(m, T5LayerFF) and "ff" in selectors:
        AdapteredT5LayerFF.cast(m, dim=dim, adapterdim=adapterdim)
    elif isinstance(m, T5LayerSelfAttention) and "self" in selectors:
        AdapteredT5LayerSelfAttention.cast(m, dim=dim, adapterdim=adapterdim)
    elif isinstance(m, T5LayerCrossAttention) and "cross" in selectors:
        AdapteredT5LayerCrossAttention.cast(m, dim=dim, adapterdim=adapterdim)


class VanillaT5Gen(T5ForConditionalGeneration):
    def get_tunable_params_and_set_requires_grad(self):
        ret = list(self.parameters())
        return ret

    def set_dropout(self, p=0.):
        def _set_dropout(m, _p=0.):
            if isinstance(m, torch.nn.Dropout):
                m.p = _p

        self.apply(partial(_set_dropout, _p=p))


class AdapteredT5Gen(T5ForConditionalGeneration):
    def adapt(self,
              out_vocab_size=None,
              adapterdim=64,  # dimension of adapters, only used if pt_type is "ada(pters)"
              adaptencoder=True,
              ):
        self.adapterdim = adapterdim
        self.out_vocab_size = out_vocab_size

        assert self.out_vocab_size is not None
        # change input embeddings and output layer
        self.decoder.embed_tokens = torch.nn.Embedding(out_vocab_size, self.shared.embedding_dim)
        # CustomT5Stack.cast(self.decoder)
        self.lm_head = torch.nn.Linear(self.lm_head.in_features, out_vocab_size, bias=False)

        dim = self.shared.embedding_dim

        if adaptencoder:
            self.encoder.apply(partial(insert_adapters, dim=dim, adapterdim=self.adapterdim))
        self.decoder.apply(partial(insert_adapters, dim=dim, adapterdim=self.adapterdim))

        self.shared = None  # make sure self.shared is not used!

    def get_tunable_params_and_set_requires_grad(self):
        ret = []

        def _get_ft_params(m, out=None):
            if hasattr(m, "get_ft_params"):
                out.extend(m.get_ft_params())

        collectedparams = []
        self.apply(partial(_get_ft_params, out=collectedparams))

        ret += collectedparams

        ret += list(self.lm_head.parameters())
        ret += list(self.decoder.embed_tokens.parameters())

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


def load_t5_tokenizer(modelsize="small"):
    print(f"Transformers version: {transformers.__version__}")
    tt = q.ticktock("script")
    modelname = f"google/t5-v1_1-{modelsize}"
    tt.msg(f"modelname: {modelname}")
    tt.tick("loading tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(modelname)
    tt.tock("loaded")
    return tokenizer


def load_vanilla_t5(modelsize="small", use_lm100k=True, out_vocab_size=None):
    print(f"Transformers version: {transformers.__version__}")
    tt = q.ticktock("script")
    modelname = "T5-LM100k" if use_lm100k else "T5"
    tt.tick(f"loading {modelsize} {modelname}")
    modelname = f"google/t5-v1_1-{modelsize}"
    tt.msg(f"modelname: {modelname}")
    tt.tick("loading tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(modelname)
    tt.tock("loaded")
    config = T5Config.from_pretrained(modelname)
    if use_lm100k:
        tt.tick(f"loading normal 100k-LM T5-{modelsize} model from checkpoint")
        model = VanillaT5Gen.from_pretrained(f"../../t5lm100k/t5.1.1.lm100k.{modelsize}/model.ckpt-1100000.index",
                                    from_tf=True, config=config)
    else:
        tt.tick(f"loading normal T5-{modelsize} from huggingface")
        model = VanillaT5Gen.from_pretrained(modelname)

    model.decoder.embed_tokens = torch.nn.Embedding(out_vocab_size, config.d_model)
    model.lm_head = torch.nn.Linear(config.d_model, out_vocab_size)
    tt.tock("loaded")
    tt.tock("loaded everything")
    return tokenizer, model, config


def load_adaptered_t5(modelsize="small", use_lm100k=True, adapterdim=64, out_vocab_size=None):

    print(f"Transformers version: {transformers.__version__}")
    tt = q.ticktock("script")
    tt.tick(f"loading {modelsize} T5-LM100k")
    modelname = f"google/t5-v1_1-{modelsize}"
    tt.msg(f"modelname: {modelname}")
    tt.tick("loading tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(modelname)
    tt.tock("loaded")
    config = T5Config.from_pretrained(modelname)
    if use_lm100k:
        tt.tick(f"loading Adapter-inserted 100k-LM T5-{modelsize} model from checkpoint")
        model = AdapteredT5Gen.from_pretrained(f"../../t5lm100k/t5.1.1.lm100k.{modelsize}/model.ckpt-1100000.index",
                                                     from_tf=True, config=config)
    else:
        tt.tick(f"loading Adapter-inserted T5-{modelsize} from huggingface")
        model = AdapteredT5Gen.from_pretrained(modelname)
    tt.tock("loaded")
    tt.msg(f"Inserting {adapterdim}-dim adapter layers.")
    model.adapt(adapterdim=adapterdim, out_vocab_size=out_vocab_size)

    tt.tock("loaded everything")
    return tokenizer, model, config


def try_load_vanilla():
    tokenizer, model, _ = load_vanilla_t5("small", out_vocab_size=1000)


def main(lr=0.001):
    out_vocab_size = 1000000
    tokenizer, model, _ = load_vanilla_t5("small", out_vocab_size=out_vocab_size)

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