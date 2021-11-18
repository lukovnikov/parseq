import copy
import torch
import qelos as q
from torch import nn
import transformers
from torch.nn import CrossEntropyLoss
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm


class T5BlockTP(torch.nn.Module):       # wrapper for T5 Blocks with PT
    def __init__(self, block:T5Block,
                 dim=None,
                 pt_type="default",
                 pt_size=5,
                 ):
        super().__init__()
        self.block = block
        self.pt_type = pt_type
        self.pt_size = pt_size
        self.dim = dim

        self.pt_emb = torch.nn.Embedding(self.pt_size, self.dim)


class T5BlockTPEncoder(T5BlockTP):
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
        # take hidden states, compute prefix and integrate prefix between first hidden states and the rest
        prefix = self.pt_emb.weight[None, :, :].repeat(hidden_states.size(0), 1, 1)
        if self.additive:
            hidden_states[:, 1:self.pt_size, :] = hidden_states[:, 1:self.pt_size, :] + prefix
        else: # replace
            hidden_states[:, 1:self.pt_size, :] = prefix
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


# class T5BlockTPDecoder(T5BlockTPEncoder):
#     # TODO: avoid re-feeding the prompt
#     def forward(
#         self,
#         hidden_states,                  # (batsize, seqlen, dim)
#         attention_mask=None,
#         position_bias=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         encoder_decoder_position_bias=None,
#         layer_head_mask=None,
#         cross_attn_layer_head_mask=None,
#         past_key_value=None,
#         use_cache=False,
#         output_attentions=False,
#         return_dict=True,
#     ):
#         # take hidden states, compute prefix and integrate prefix between first hidden states and the rest
#         # prefix = self.pt_emb.weight[None, :, :].repeat(hidden_states.size(0), 1, 1)
#         # if self.additive:
#         #     hidden_states[:, 1:self.pt_size, :] = hidden_states[:, 1:self.pt_size, :] + prefix
#         # else: # replace
#         #     hidden_states[:, 1:self.pt_size, :] = prefix
#         ret = self.block(hidden_states,
#                    attention_mask=attention_mask,
#                    position_bias=position_bias,     #???
#                    encoder_hidden_states=encoder_hidden_states,
#                    encoder_attention_mask=encoder_attention_mask,
#                    encoder_decoder_position_bias=encoder_decoder_position_bias, # ???
#                    layer_head_mask=layer_head_mask,
#                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
#                    past_key_value=past_key_value,
#                    use_cache=use_cache,
#                    output_attentions=output_attentions,
#                    return_dict=return_dict
#                    )
#         return ret


class T5PTGen(T5ForConditionalGeneration):
    DUMMYID = 5
    def adapt(self,
              out_vocab_size=None,
              pt_type="default",               # de(ep)/e(mb)+a(dd)/r(epl)+s(tatic)/dy(namic)      default: emb+static
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
            block = T5BlockTPEncoder(block, pt_type=pt_type, pt_size=pt_size, dim=dim)
            self.encoder.block[i] = block

        # for i, block in enumerate(self.decoder.block):      # decoder layers
        #     block = T5BlockTPDecoder(block, pt_type=pt_type, pt_size=pt_size, dim=dim)
        #     self.decoder.block[i] = block

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
        #  if input_ids, add dummy tokens
        #       if input_embeds, add zero vectors
        #       in both cases, change attention_mask

        if input_ids is not None:
            insertids = torch.ones(input_ids.size(0), self.pt_size, dtype=input_ids.dtype, device=input_ids.device) * self.DUMMYID
            input_ids = torch.cat([input_ids[:, 0:1], insertids, input_ids[:, 1:]], 1)
        if inputs_embeds is not None:
            insert_embeds = torch.zeros(inputs_embeds.size(0), self.pt_size, inputs_embeds.size(2),
                                        dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds = torch.cat([inputs_embeds[:, 0:1, :], insert_embeds, inputs_embeds[:, 1:]], 1)
        if input_ids is not None or inputs_embeds is not None:
            insert_attention_mask = torch.ones(attention_mask.size(0), self.pt_size, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask[:, 0:1], insert_attention_mask, attention_mask[:, 1:]], 1)
        super(T5PTGen, self).forward(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     decoder_input_ids=decoder_input_ids,
                                     decoder_attention_mask=decoder_attention_mask,
                                     head_mask=head_mask,
                                     decoder_head_mask=decoder_head_mask,
                                     cross_attn_head_mask=cross_attn_head_mask,
                                     encoder_outputs=encoder_outputs,
                                     past_key_values=past_key_values,
                                     inputs_embeds=inputs_embeds,
                                     decoder_input_embeds=decoder_inputs_embeds,
                                     labels=labels,
                                     use_cache=use_cache,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states,
                                     return_dict=return_dict
                                     )


def main(lr=0.001):
    print(transformers.__version__)

    modelname = "google/t5-v1_1-small"
    tokenizer = T5Tokenizer.from_pretrained(modelname)
    model = T5PTGen.from_pretrained(modelname)
    model.adapt()

    inputs = tokenizer(['0 translate English to German: The house is wonderful.', 'translate English to German: the house is.'], return_tensors='pt', truncation=True, padding=True)
    decinp = tokenizer(['0 Das Haus ist wunderbar', '0 Das haus'], return_tensors='pt', truncation=True, padding=True)
    labels = tokenizer(['Das Haus ist wunderbar.', 'Das haus.'], return_tensors='pt', truncation=True, padding=True)
    # the forward function automatically creates the correct decoder_input_ids
    loss = model(input_ids=inputs.input_ids,
                 attention_mask=inputs.attention_mask,
                 decoder_input_ids=decinp.input_ids,
                 decoder_attention_mask=decinp.attention_mask,
                 labels=labels.input_ids)
    loss = loss.loss
    print(f"Loss: {loss}")


if __name__ == '__main__':
    q.argprun(main)