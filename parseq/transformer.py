
""" Copied from Hugging face T5 model code in Pytorch. """


import copy
import logging
import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PretrainedConfig
from transformers.configuration_t5 import T5Config
from transformers.file_utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer


logger = logging.getLogger(__name__)

####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of torch.nn.Module)
####################################################


class TransformerConfig(PretrainedConfig):
    r"""
        :class:`~transformers.T5Config` is the configuration class to store the configuration of a
        `T5Model`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `T5Model`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `T5Model`.
            initializer_factor: A factor for initializing all weight matrices (should be kept to 1.0, used for initialization testing).
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    # pretrained_config_archive_map = T5_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "transformer"

    def __init__(
        self,
        vocab_size=32128,
        n_positions=512,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        is_encoder_decoder=True,
        pad_token_id=0,
        eos_token_id=1,
        use_position_bias=False,
        use_causal_mask=True,       # use causal mask in decoder blocks
        use_relative_position=False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, is_encoder_decoder=is_encoder_decoder, **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_position_bias = use_position_bias
        self.use_causal_mask = use_causal_mask
        self.use_relative_position = use_relative_position

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers


FACTOR = 1.


class TransformerDenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        self.wi.weight.data.normal_(mean=0.0, std=FACTOR * ((self.config.d_model) ** -0.5))
        if hasattr(self.wi, "bias") and self.wi.bias is not None:
            self.wi.bias.data.zero_()
        self.wo.weight.data.normal_(mean=0.0, std=FACTOR * ((self.config.d_ff) ** -0.5))
        if hasattr(self.wo, "bias") and self.wo.bias is not None:
            self.wo.bias.data.zero_()

    def forward(self, hidden_states):
        h = self.wi(hidden_states)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class TransformerLayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = TransformerDenseReluDense(config)
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output


class TransformerAttention(nn.Module):
    def __init__(self, config: TransformerConfig, has_relative_attention_bias=False):
        super().__init__()
        self.config = config
        self.is_decoder = config.is_decoder
        self.use_position_bias = config.use_position_bias
        self.use_relative_position = config.use_relative_position
        self.has_relative_attention_bias = has_relative_attention_bias

        self.output_attentions = config.output_attentions
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

        if self.use_relative_position:
            self.relative_position_embedding = nn.Embedding(self.relative_attention_num_buckets, self.d_model)

        self.pruned_heads = set()

        self.reset_parameters()

    def reset_parameters(self):
        d_model = self.config.d_model
        d_kv = self.config.d_kv
        n_heads = self.config.num_heads
        self.q.weight.data.normal_(mean=0.0, std=FACTOR * ((d_model * d_kv) ** -0.5))
        self.k.weight.data.normal_(mean=0.0, std=FACTOR * (d_model ** -0.5))
        self.v.weight.data.normal_(mean=0.0, std=FACTOR * (d_model ** -0.5))
        self.o.weight.data.normal_(mean=0.0, std=FACTOR * ((n_heads * d_kv) ** -0.5))
        if self.has_relative_attention_bias:
            self.relative_attention_bias.weight.data.normal_(mean=0.0, std=FACTOR * ((d_model) ** -0.5))

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_heads, self.d_kv)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not (self.is_decoder and self.config.use_causal_mask),
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def compute_relative_position_scores(self, q, klen):
        """
        :param q:   (batsize, numheads, qlen, headdim)
        :param klen:
        :return:
        """
        qlen = q.size(2)
        bs = q.size(0)
        relposembs = self.relative_position_embedding.weight
        relposembs = self.k(relposembs)
        relposembs = relposembs.view(-1, self.n_heads, self.d_kv).transpose(0, 1)
        # (numheads, numrelpos, dim_per_head)

        # compute the relative position based score using the queries
        relposscores = torch.einsum("bhsd,hzd->bhsz", q, relposembs)
        # (batsize, numheads, qlen, numrelpos)

        # compute relative positions in buckets
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not (self.is_decoder and self.config.use_causal_mask),
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_position_embedding.weight.device)
        # (qlen, klen)
        rp_bucket = rp_bucket[None, None].repeat(bs, self.n_heads, 1, 1)

        scores = relposscores.gather(-1, rp_bucket)
        return scores


    def forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        use_position_bias=True,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value_state[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        if past_key_value_state is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert (
                len(past_key_value_state) == 2
            ), "past_key_value_state should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value_state)
            )
            real_qlen = qlen + past_key_value_state[0].shape[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_value_state is None:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_value_state is not None:
            if kv is None:
                k_, v_ = past_key_value_state
                k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value_state

        if self.is_decoder and use_cache is True:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

        if use_position_bias and self.use_position_bias:
            if position_bias is None:
                if not self.has_relative_attention_bias:
                    raise ValueError("No position_bias provided and no weights to compute position_bias")
                position_bias = self.compute_bias(real_qlen, klen)

                # if key and values are already calculated
                # we want only the last query position bias
                if past_key_value_state is not None:
                    position_bias = position_bias[:, :, -1:, :]

                if mask is not None:
                    position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

            scores += position_bias

        if use_position_bias and self.use_relative_position:    # only use rel pos if not cross-attention
            relative_position_scores = self.compute_relative_position_scores(q, klen)
            if past_key_value_state is not None:
                relative_position_scores = relative_position_scores[:, :, -1:, :]
            if mask is not None:
                relative_position_scores = relative_position_scores + mask
            scores += relative_position_scores

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,) + present_key_value_state

        if self.output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


class TransformerLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = TransformerAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class TransformerLayerCrossAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.EncDecAttention = TransformerAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        kv,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
        query_length=None,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x,
            mask=attention_mask,
            kv=kv,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            query_length=query_length,
            use_position_bias=False,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class TransformerBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(TransformerLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(TransformerLayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))

        self.layer.append(TransformerLayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        past_key_value_state=None,
        use_cache=False,
    ):

        if past_key_value_state is not None:
            assert self.is_decoder, "Only decoder can use `past_key_value_states`"
            expected_num_past_key_value_states = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_value_states,
                "2 (past / key) for cross attention" if expected_num_past_key_value_states == 4 else "",
                len(past_key_value_state),
            )
            assert len(past_key_value_state) == expected_num_past_key_value_states, error_message

            self_attn_past_key_value_state = past_key_value_state[:2]
            cross_attn_past_key_value_state = past_key_value_state[2:]
        else:
            self_attn_past_key_value_state, cross_attn_past_key_value_state = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value_state=self_attn_past_key_value_state,
            use_cache=use_cache,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value_state=cross_attn_past_key_value_state,
                query_length=query_length,
                use_cache=use_cache,
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,)

        # Add attentions if we output them
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class TransformerPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = T5Config
    # pretrained_model_archive_map = T5_PRETRAINED_MODEL_ARCHIVE_MAP
    # load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        # if isinstance(module, torch.nn.LayerNorm):
        #     module.weight.data.fill_(factor * 1.0)
        # elif isinstance(module, (TransformerModel, T5ForConditionalGeneration)):
        # #     Mesh TensorFlow embeddings initialization
        #     # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
        #     module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        # if isinstance(module, TransformerDenseReluDense):
        #     # Mesh TensorFlow FF initialization
        #     # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
        #     # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
        #     module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
        #     if hasattr(module.wi, "bias") and module.wi.bias is not None:
        #         module.wi.bias.data.zero_()
        #     module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
        #     if hasattr(module.wo, "bias") and module.wo.bias is not None:
        #         module.wo.bias.data.zero_()
        # elif isinstance(module, TransformerAttention):
        #     # Mesh TensorFlow attention initialization to avoid scaling before softmax
        #     # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
        #     d_model = self.config.d_model
        #     d_kv = self.config.d_kv
        #     n_heads = self.config.num_heads
        #     module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
        #     module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
        #     module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
        #     module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))
        #     if module.has_relative_attention_bias:
        #         module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in lm_labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `lm_labels` has only positive values and -100"

        return shifted_input_ids


class TransformerStack(TransformerPreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [TransformerBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_value_states=None,
        use_cache=False,
    ):
        # assert(use_cache == False)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            if self.is_decoder:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if past_key_value_states is not None:
            assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
                input_shape, (batch_size, 1)
            )
            # required mask seq length can be calculated via length of past
            # key value states and seq_length = 1 for the last token
            mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length
        else:
            mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length).to(inputs_embeds.device)

        # initialize past_key_value_states with `None` if past does not exist
        if past_key_value_states is None:
            past_key_value_states = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, self.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = ()
        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if self.output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[4 if self.output_attentions else 3]
            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)
            outputs = outputs + (present_key_value_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: tuple, device: torch.device):
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.

        Arguments:
            attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
            input_shape: tuple, shape of input_ids
            device: torch.Device, usually self.device

        Returns:
            torch.Tensor with dtype of attention_mask.dtype
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder and self.config.use_causal_mask:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


T5_START_DOCSTRING = r"""    The T5 model was proposed in
    `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`_
    by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.
    It's an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`:
        https://arxiv.org/abs/1910.10683

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            T5 is a model with relative position embeddings so you should be able to pad the inputs on both the right and the left.
            Indices can be obtained using :class:`transformers.T5Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
            To know more on how to prepare :obj:`input_ids` for pre-training take a look at
            `T5 Training <./t5.html#training>`_ .
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for sequence to sequence training. T5 uses the pad_token_id as the starting token for decoder_input_ids generation.
            If `decoder_past_key_value_states` is used, optionally only the last `decoder_input_ids` have to be input (see `decoder_past_key_value_states`).
            To know more on how to prepare :obj:`decoder_input_ids` for pre-training take a look at
            `T5 Training <./t5.html#training>`_ .
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up decoding.
            If `decoder_past_key_value_states` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all `decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If `use_cache` is True, `decoder_past_key_value_states` are returned and can be used to speed up decoding (see `decoder_past_key_value_states`).
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded representation.
            If `decoder_past_key_value_states` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `decoder_past_key_value_states`).
            This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        head_mask: (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
)
class TransformerModel(TransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = TransformerStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = TransformerStack(decoder_config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_value_states=None,
        use_cache=True,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `decoder_past_key_value_states` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
            Note that when using `decoder_past_key_value_states`, the model only outputs the last `hidden-state` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            from transformers import T5Tokenizer, T5Model

            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            model = T5Model.from_pretrained('t5-small')
            input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
            outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
            last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
            )

        hidden_states = encoder_outputs[0]

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )

        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        return decoder_outputs + encoder_outputs

#
# @add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
# class T5ForConditionalGeneration(T5PreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.model_dim = config.d_model
#
#         self.shared = nn.Embedding(config.vocab_size, config.d_model)
#
#         encoder_config = copy.deepcopy(config)
#         self.encoder = TransformerStack(encoder_config, self.shared)
#
#         decoder_config = copy.deepcopy(config)
#         decoder_config.is_decoder = True
#         self.decoder = TransformerStack(decoder_config, self.shared)
#
#         self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
#
#         self.init_weights()
#
#     def get_input_embeddings(self):
#         return self.shared
#
#     def set_input_embeddings(self, new_embeddings):
#         self.shared = new_embeddings
#         self.encoder.set_input_embeddings(new_embeddings)
#         self.decoder.set_input_embeddings(new_embeddings)
#
#     def get_output_embeddings(self):
#         return self.lm_head
#
#     def get_encoder(self):
#         return self.encoder
#
#     def get_decoder(self):
#         return self.decoder
#
#     @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         encoder_outputs=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         decoder_past_key_value_states=None,
#         use_cache=True,
#         lm_labels=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         head_mask=None,
#     ):
#         r"""
#         lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#                 Labels for computing the sequence classification/regression loss.
#                 Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
#                 All labels set to ``-100`` are ignored (masked), the loss is only
#                 computed for labels in ``[0, ..., config.vocab_size]``
#
#     Returns:
#         :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.T5Config`) and inputs.
#         loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_label` is provided):
#             Classification loss (cross entropy).
#         prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#             If `past_key_value_states` is used only the last prediction_scores of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
#         decoder_past_key_value_states (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`, `optional`, returned when ``use_cache=True``):
#             Contains pre-computed key and value hidden-states of the attention blocks.
#             Can be used to speed up sequential decoding (see `decoder_past_key_value_states` input).
#             Note that when using `decoder_past_key_value_states`, the model only outputs the last `prediction_score` of the sequence of shape :obj:`(batch_size, 1, config.vocab_size)`.
#         hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#             of shape :obj:`(batch_size, sequence_length, hidden_size)`.
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
#             Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#             :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention.
#
#     Examples::
#
#         from transformers import T5Tokenizer, T5ForConditionalGeneration
#
#         tokenizer = T5Tokenizer.from_pretrained('t5-small')
#         model = T5ForConditionalGeneration.from_pretrained('t5-small')
#         input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
#         outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
#         loss, prediction_scores = outputs[:2]
#
#         tokenizer = T5Tokenizer.from_pretrained('t5-small')
#         model = T5ForConditionalGeneration.from_pretrained('t5-small')
#         input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
#         outputs = model.generate(input_ids)
#         """
#
#         # Encode if needed (training, first prediction pass)
#         if encoder_outputs is None:
#             # Convert encoder inputs in embeddings if needed
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask
#             )
#
#         hidden_states = encoder_outputs[0]
#
#         if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
#             # get decoder inputs from shifting lm labels to the right
#             decoder_input_ids = self._shift_right(lm_labels)
#
#         # If decoding with past key value states, only the last tokens
#         # should be given as an input
#         if decoder_past_key_value_states is not None:
#             assert lm_labels is None, "Decoder should not use cached key value states when training."
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids[:, -1:]
#             if decoder_inputs_embeds is not None:
#                 decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
#
#         # Decode
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             inputs_embeds=decoder_inputs_embeds,
#             past_key_value_states=decoder_past_key_value_states,
#             encoder_hidden_states=hidden_states,
#             encoder_attention_mask=attention_mask,
#             head_mask=head_mask,
#             use_cache=use_cache,
#         )
#
#         # insert decoder past at right place
#         # to speed up decoding
#         if use_cache is True:
#             past = ((encoder_outputs, decoder_outputs[1]),)
#             decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]
#
#         sequence_output = decoder_outputs[0]
#         # Rescale output before projecting on vocab
#         # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
#         sequence_output = sequence_output * (self.model_dim ** -0.5)
#         lm_logits = self.lm_head(sequence_output)
#
#         decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
#         if lm_labels is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-100)
#             loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
#             # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
#             decoder_outputs = (loss,) + decoder_outputs
#
#         return decoder_outputs + encoder_outputs
#
#     def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, **kwargs):
#         assert past is not None, "past has to be defined for encoder_outputs"
#
#         # first step
#         if len(past) < 2:
#             encoder_outputs, decoder_past_key_value_states = past, None
#         else:
#             encoder_outputs, decoder_past_key_value_states = past[0], past[1]
#
#         return {
#             "decoder_input_ids": input_ids,
#             "decoder_past_key_value_states": decoder_past_key_value_states,
#             "encoder_outputs": encoder_outputs,
#             "attention_mask": attention_mask,
#             "use_cache": use_cache,
#         }
#
#     def _reorder_cache(self, past, beam_idx):
#         # if decoder past is not included in output
#         # speedy decoding is disabled and no need to reorder
#         if len(past) < 2:
#             logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
#             return past
#
#         decoder_past = past[1]
#         past = (past[0],)
#         reordered_decoder_past = ()
#         for layer_past_states in decoder_past:
#             # get the correct batch idx from layer past batch dim
#             # batch dim of `past` is at 2nd position
#             reordered_layer_past_states = ()
#             for layer_past_state in layer_past_states:
#                 # need to set correct `past` for each of the four key / value states
#                 reordered_layer_past_states = reordered_layer_past_states + (
#                     layer_past_state.index_select(0, beam_idx),
#                 )
#
#             assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
#             assert len(reordered_layer_past_states) == len(layer_past_states)
#
#             reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
#         return past + (reordered_decoder_past,)
