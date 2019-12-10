import math
import sys

import torch



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.torch.nn.functional.relu, "swish": swish}


class TransformerConfig(object):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """

    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(TransformerConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = False
        self.output_hidden_states = False


class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.register_buffer("_np_mask", None)        # (dim,) float of zeros and ones

    def set_np_mask(self, mask):
        self._np_mask = mask

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, x):
        if self._np_mask is None:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            ret = self.weight * x + self.bias
            return ret
        else:
            mask = self._np_mask.unsqueeze(0).unsqueeze(0)
            x = x * mask.float()
            u = x.sum(-1, keepdim=True) / mask.sum()
            s = ((x - u) * mask).pow(2).sum(-1, keepdim=True) / mask.sum()
            x = ((x - u) * mask) / torch.sqrt(s + self.variance_epsilon)
            ret = self.weight * x + self.bias
            return ret * mask


class TransformerEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(TransformerEmbeddings, self).__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self._no_grad = False
        self.register_buffer("_np_mask", None)

    def set_np_mask(self, mask):
        self._np_mask = mask
        self.LayerNorm.set_np_mask(mask)

    def forward(self, input_ids, position_ids=None):
        if self._no_grad:
            with torch.no_grad():
                return self._forward(input_ids, position_ids=position_ids)
        else:
            return self._forward(input_ids, position_ids=position_ids)

    def _forward(self, input_ids, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings

        if self._np_mask is not None:
            mask = self._np_mask.unsqueeze(0).unsqueeze(0)
            embeddings = embeddings * mask

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super(TransformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)
        self.register_buffer("_np_att_mask", None)
        self.register_buffer("_np_val_mask", None)

    def set_np_mask(self, attmask, valmask):
        self._np_att_mask = attmask    # must be (numheads, dim_per_head)
        self._np_val_mask = valmask    # must be (numheads, dim_per_head)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_head_size = self.attention_head_size
        if self._np_att_mask is not None:
            mask = self._np_att_mask.unsqueeze(1).unsqueeze(0)
            query_layer = query_layer * mask
            key_layer = key_layer * mask
            attention_head_size = mask[0, 0, 0].sum()

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        if self._np_val_mask is not None:
            mask = self._np_val_mask.unsqueeze(1).unsqueeze(0)
            context_layer = context_layer * mask

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        ret_attention = attention_scores
        outputs = (context_layer, ret_attention) if self.output_attentions else (context_layer,)
        return outputs


class TransformerSelfOutput(torch.nn.Module):
    def __init__(self, config):
        super(TransformerSelfOutput, self).__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("_np_mask", None)

    def set_np_mask(self, mask):
        self._np_mask = mask
        self.LayerNorm.set_np_mask(mask)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if self._np_mask is not None:
            mask = self._np_mask.unsqueeze(0).unsqueeze(0)
            hidden_states = hidden_states * mask
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerAttention(torch.nn.Module):
    def __init__(self, config):
        super(TransformerAttention, self).__init__()
        self.self = TransformerSelfAttention(config)
        self.output = TransformerSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TransformerIntermediate(torch.nn.Module):
    def __init__(self, config):
        super(TransformerIntermediate, self).__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.register_buffer("_np_mask", None)

    def set_np_mask(self, mask):
        self._np_mask = mask

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        if self._np_mask is not None:
            mask = self._np_mask.unsqueeze(0).unsqueeze(0)
            hidden_states = hidden_states * mask
        return hidden_states


class TransformerOutput(torch.nn.Module):
    def __init__(self, config):
        super(TransformerOutput, self).__init__()
        self.dense = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("_np_mask", None)

    def set_np_mask(self, mask):
        self._np_mask = mask
        self.LayerNorm.set_np_mask(mask)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if self._np_mask is not None:
            mask = self._np_mask.unsqueeze(0).unsqueeze(0)
            hidden_states = hidden_states * mask
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(torch.nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.attention = TransformerAttention(config)
        self.intermediate = TransformerIntermediate(config)
        self.output = TransformerOutput(config)
        self._no_grad = False

    def forward(self, hidden_states, attention_mask):
        if self._no_grad:
            with torch.no_grad():
                return self._forward(hidden_states=hidden_states, attention_mask=attention_mask)
        else:
            return self._forward(hidden_states=hidden_states, attention_mask=attention_mask)

    def _forward(self, hidden_states, attention_mask):
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = torch.nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class Transformer(torch.nn.Module):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config, **kw):
        super(Transformer, self).__init__(**kw)
        self.config = config

        self.embeddings = TransformerEmbeddings(config)
        self.encoder = TransformerEncoder(config)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        embedding_output = self.embeddings(input_ids, position_ids=position_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output, ) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, (hidden_states), (attentions)


def try_transformer():
    config = TransformerConfig(vocab_size=1000, num_attention_heads=8, num_hidden_layers=4, hidden_size=256, intermediate_size=512)
    t = Transformer(config)
    x = torch.randint(0, 1000, (3, 5))
    y = t(x)
    print(len(y))
    print(y[0].size())
    print(y)


if __name__ == '__main__':
    try_transformer()