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
        # take hidden states, compute prefix and insert prefix between first hidden states and the rest
        prefix = self.pt_emb.weight[None, :, :].repeat(hidden_states.size(0), 1, 1)
        new_hidden_states = torch.cat([hidden_states[:, 0:1, :], prefix, hidden_states[:, 1:, :]])
        ret = self.block(new_hidden_states,
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



class T5PTGen(T5ForConditionalGeneration):
    def adapt(self,
              out_vocab_size=None,
              pt_type="default",               # de(ep)/e(mb)+a(dd)/r(epl)+s(tatic)/dy(namic)      default: emb+static
              pt_size=5,                    # number of prefix/prompt pseudo-tokens
              ):
        if out_vocab_size is not None:
            # change input embeddings and output layer
            self.decoder.embed_tokens = torch.nn.Embedding(out_vocab_size, self.shared.embedding_dim)
            self.lm_head = torch.nn.Linear(self.lm_head.in_features, out_vocab_size, bias=False)

        dim = self.shared.embedding_dim

        # adapt the transformer layers -- every adapted layer is responsible for generating its own prompt
        for i, block in enumerate(self.encoder.block):      # encoder layers
            if i == 0:
                block = T5BlockTPEncoderFirst(block, pt_type=pt_type, pt_size=pt_size, dim=dim)
            else:
                block = T5BlockTPEncoder(block, pt_type=pt_type, pt_size=pt_size, dim=dim)
            self.encoder.block[i] = block

        for i, block in enumerate(self.decoder.block):      # decoder layers
            if i == 0:
                block = T5BlockTPDecoderFirst(block, pt_type=pt_type, pt_size=pt_size, dim=dim)
            else:
                block = T5BlockTPDecoder(block, pt_type=pt_type, pt_size=pt_size, dim=dim)
            self.decoder.block[i] = block


def main(lr=0.001):
    print(transformers.__version__)

    modelname = "google/t5-v1_1-small"
    tokenizer = T5Tokenizer.from_pretrained(modelname)
    config = transformers.PretrainedConfig.from_pretrained(modelname)
    config.out_vocab_size = 10
    model = T5PTGen.from_pretrained(modelname, config=config)
    model.adapt()

    input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
    labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids
    # the forward function automatically creates the correct decoder_input_ids
    loss = model(input_ids=input_ids, labels=labels)
    loss = loss.loss
    print(f"Loss: {loss}")


if __name__ == '__main__':
    q.argprun(main)