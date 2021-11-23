import copy
import torch
import qelos as q
from torch import nn
import transformers
from torch.nn import CrossEntropyLoss
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm, load_tf_weights_in_t5


# use 100k LM pre-trained T5 weights instead of normal weights! --> https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md


class T5BlockPT(torch.nn.Module):       # wrapper for T5 Blocks with PT
    def __init__(self, block:T5Block,
                 dim=None,
                 pt_type="default",
                 pt_size=5,
                 first=False,
                 ):
        super().__init__()
        self.block = block

        if pt_type == "default":
            pt_type = "shallow+static+add"

        self.pt_type = pt_type
        self.pt_size = pt_size
        self.dim = dim

        self.first = first

        for x in self.pt_type.split("+"):
            if x.startswith("a"):
                self.replace = False
            elif x.startswith("r"):
                self.replace = True
            elif x.startswith("sh"):
                self.deep = False
            elif x.startswith("de"):
                self.deep = True
            elif x.startswith("st"):
                self.dynamic = False
            elif x.startswith("dy"):
                self.dynamic = True

        if not self.dynamic:
            if self.first or self.deep:
                self.pt_emb = torch.nn.Embedding(self.pt_size, self.dim)
        else:
            raise NotImplementedError("use static")


class T5BlockPTEncoder(T5BlockPT):
    """ An encoder-side T5 transformer block for prompt-tuning"""
    def forward(
        self,
        hidden_states,                  # (batsize, seqlen, dim)
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if self.first or self.deep:
            # take hidden states, compute prefix and integrate prefix between first hidden states and the rest
            prefix = self.pt_emb.weight[None, :, :].repeat(hidden_states.size(0), 1, 1)

            if self.first or self.replace: # replace
                hidden_states[:, :self.pt_size, :] = prefix
            else:
                hidden_states[:, :self.pt_size, :] = hidden_states[:, :self.pt_size, :] + prefix

        ret = self.block(hidden_states,
                   attention_mask=attention_mask,
                   position_bias=position_bias,     #???
                   encoder_hidden_states=encoder_hidden_states,
                   encoder_attention_mask=encoder_attention_mask,
                   encoder_decoder_position_bias=encoder_decoder_position_bias, # ???
                   layer_head_mask=layer_head_mask,
                   cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                   past_key_value=past_key_value,
                   use_cache=use_cache,
                   output_attentions=output_attentions,
                   return_dict=return_dict
                   )
        return ret


class T5StackPTEncoder(T5Stack):
    """ Wraps a T5Stack from the encoder to support prompt tuning """
    DUMMYID = 0

    @classmethod
    def cast(cls, obj:T5Stack, pt_type="default", pt_size=5, **kw):
        obj.__class__ = cls
        obj.pt_type = pt_type
        obj.pt_size = pt_size
        return obj

    # def __init__(self, encoder:T5Stack, pt_type="default", pt_size=5, **kw):
    #     super(T5StackPTEncoder, self).__init__(**kw)
    #     self.encoder = encoder
    #     self.pt_type = pt_type
    #     self.pt_size = pt_size

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None:
            insertids = torch.ones(input_ids.size(0), self.pt_size, dtype=input_ids.dtype, device=input_ids.device) * self.DUMMYID
            input_ids = torch.cat([insertids, input_ids], 1)
        if inputs_embeds is not None:
            insert_embeds = torch.zeros(inputs_embeds.size(0), self.pt_size, inputs_embeds.size(2),
                                        dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds = torch.cat([insert_embeds, inputs_embeds], 1)
        if input_ids is not None or inputs_embeds is not None:
            insert_attention_mask = torch.ones(attention_mask.size(0), self.pt_size, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([insert_attention_mask, attention_mask], 1)
        ret = super(T5StackPTEncoder, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return ret


class T5PTGen(T5ForConditionalGeneration):
    DUMMYID = 5
    def adapt(self,
              out_vocab_size=None,
              pt_type="default",               # de(ep)/sh(allow)+a(dd)/r(epl)+st(atic)/dy(namic)      default: deep+static+add
              pt_size=5,                    # number of prefix/prompt pseudo-tokens
              ):
        self.pt_size = pt_size
        self.pt_type = pt_type
        self.out_vocab_size = out_vocab_size
        if out_vocab_size is not None:
            # change input embeddings and output layer
            self.decoder.embed_tokens = torch.nn.Embedding(out_vocab_size, self.shared.embedding_dim)
            self.lm_head = torch.nn.Linear(self.lm_head.in_features, out_vocab_size, bias=False)

        dim = self.shared.embedding_dim

        # adapt the transformer layers -- every adapted layer is responsible for generating its own prompt
        for i, block in enumerate(self.encoder.block):      # encoder layers
            block = T5BlockPTEncoder(block, pt_type=pt_type, pt_size=pt_size, first=i==0, dim=dim)
            self.encoder.block[i] = block

        self.encoder = T5StackPTEncoder.cast(self.encoder, pt_type=pt_type, pt_size=pt_size)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if encoder_outputs is not None:
            # assumption: encoder_outputs is None when doing training and is not None when doing generation.
            # => in case encoder_outputs is not None, we still need to fix the attention mask over the input!
            # insert attention mask here as well because during generation, decoder is called with old attention mask while hidden states are from extended
            insert_attention_mask = torch.ones(attention_mask.size(0), self.pt_size, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([insert_attention_mask, attention_mask], 1)
        ret = super(T5PTGen, self).forward(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     decoder_input_ids=decoder_input_ids,
                                     decoder_attention_mask=decoder_attention_mask,
                                     head_mask=head_mask,
                                     decoder_head_mask=decoder_head_mask,
                                     cross_attn_head_mask=cross_attn_head_mask,
                                     encoder_outputs=encoder_outputs,
                                     past_key_values=past_key_values,
                                     inputs_embeds=inputs_embeds,
                                     decoder_inputs_embeds=decoder_inputs_embeds,
                                     labels=labels,
                                     use_cache=use_cache,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states,
                                     return_dict=return_dict
                                     )
        return ret


def try_generate(inp:str=None, model=None, tokenizer=None):
    if inp is None:
        inp = "When I get up in the morning, the first thing I do is eat breakfast. I should also take a "
    input_ids = tokenizer(inp, return_tensors='pt').input_ids
    print(f"input ids: {input_ids}")
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def load_t5(modelsize="small",  use_lm100k=True):
    print(f"Transformers version: {transformers.__version__}")
    tt = q.ticktock("script")
    tt.tick(f"loading {modelsize} T5-LM100k")
    modelname = f"google/t5-v1_1-{modelsize}"
    tt.msg(f"modelname: {modelname}")
    tt.tick("loading tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(modelname)
    tt.tock("loaded")
    config = T5Config.from_pretrained(modelname)
    # model = T5PTGen.from_pretrained(modelname)
    if use_lm100k:
        tt.tick("loading 100k-LM T5 model from checkpoint")
        model = T5PTGen.from_pretrained(f"../../t5lm100k/t5.1.1.lm100k.{modelsize}/model.ckpt-1100000.index", from_tf=True, config=config)
    else:
        tt.tick("loading T5 from huggingface")
        model = T5PTGen.from_pretrained(modelname)
    tt.tock("loaded")
    tt.tock("loaded everything")
    return tokenizer, model, config


def main(lr=0.001):
    tokenizer, model, _ = load_t5("small")
    model.adapt(out_vocab_size=50)
    try_generate(model=model, tokenizer=tokenizer)

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
    q.argprun(main)