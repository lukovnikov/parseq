from itertools import product
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

from parseq.transformer import TransformerConfig, TransformerLayerFF


FACTOR = 1.


class TransformerAttentionConv(MessagePassing):
    def __init__(self, config: TransformerConfig):
        super().__init__(node_dim=0)
        self.config = config
        # self.is_decoder = config.is_decoder

        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = torch.nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = torch.nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = torch.nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = torch.nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.attn_dropout = torch.nn.Dropout(config.attention_dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        d_model = self.config.d_model
        d_kv = self.config.d_kv
        n_heads = self.config.num_heads
        self.q.weight.data.normal_(mean=0.0, std=FACTOR * ((d_model * d_kv) ** -0.5))
        self.k.weight.data.normal_(mean=0.0, std=FACTOR * (d_model ** -0.5))
        self.v.weight.data.normal_(mean=0.0, std=FACTOR * (d_model ** -0.5))
        self.o.weight.data.normal_(mean=0.0, std=FACTOR * ((n_heads * d_kv) ** -0.5))

    def forward(self, edge_index: Adj, x:torch.Tensor=None, kv:torch.Tensor=None):
        H, C = self.n_heads, self.d_kv

        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        assert kv.dim() == 2

        q = self.layer_norm(x)

        q = self.q(q).view(q.size(0), H, C)
        k = self.k(kv).view(kv.size(0), H, C)
        v = self.v(kv).view(kv.size(0), H, C)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, q=(None, q), k=(k, None), v=(v, None))

        out = out.view(-1, self.n_heads * self.d_kv)

        out = self.o(out)

        out = x + self.dropout(out)
        return out

    def message(self,
                q_i: Tensor,    # key
                k_j: Tensor,    # query
                v_j: Tensor,    # value
                edge_index_i: Tensor,
                edge_index_j: Tensor) -> Tensor:
        attention_scores = (q_i * k_j).sum(-1)
        alpha = softmax(attention_scores, edge_index_i)
        alpha = self.attn_dropout(alpha)
        ret = v_j * alpha.unsqueeze(-1)  # weigh the incoming states by alphas
        return ret


class RelationalTransformerAttentionConv(MessagePassing):
    GATE_BIAS = 3.
    def __init__(self, config: TransformerConfig):
        super().__init__(node_dim=0)
        self.config = config
        # self.is_decoder = config.is_decoder

        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.d_ff = config.d_ff
        self.inner_dim = self.n_heads * self.d_kv

        self.q = torch.nn.Linear(self.d_model, self.inner_dim, bias=False)

        self.relA = torch.nn.Linear(self.d_model * 2, self.d_ff)
        self.relB = torch.nn.Linear(self.d_ff, self.d_model)
        self.relG = torch.nn.Linear(self.d_ff, self.d_model)

        self.k = torch.nn.Linear(self.d_model * 2, self.inner_dim, bias=False)
        self.v = torch.nn.Linear(self.d_model, self.inner_dim, bias=False)

        self.o = torch.nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.attn_dropout = torch.nn.Dropout(config.attention_dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        d_model = self.config.d_model
        d_kv = self.config.d_kv
        n_heads = self.config.num_heads
        self.q.weight.data.normal_(mean=0.0, std=FACTOR * ((d_model * d_kv) ** -0.5))
        self.k.weight.data.normal_(mean=0.0, std=FACTOR * (d_model ** -0.5))
        self.v.weight.data.normal_(mean=0.0, std=FACTOR * (d_model ** -0.5))
        self.o.weight.data.normal_(mean=0.0, std=FACTOR * ((n_heads * d_kv) ** -0.5))

    def forward(self, edge_index: Adj, x:torch.Tensor=None, kv:torch.Tensor=None,
                edge_features:torch.Tensor=None):
        H, C = self.n_heads, self.d_kv

        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        assert kv.dim() == 2

        q = self.layer_norm(x)

        q = self.q(q).view(q.size(0), H, C)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, q=(None, q), kv=(kv, None), edge_features=edge_features)

        out = out.view(-1, self.n_heads * self.d_kv)

        out = self.o(out)

        out = x + self.dropout(out)
        return out

    def message(self,
                q_i: Tensor,    # key
                kv_j: Tensor,    # values
                edge_index_i: Tensor,
                edge_index_j: Tensor,
                edge_features: torch.Tensor=None) -> Tensor:
        assert edge_features is not None, "'edge_features' can not be None"

        H, C = self.n_heads, self.d_kv

        inter_k = torch.celu(self.relA(torch.cat([kv_j, edge_features], -1)))
        k_gate = torch.sigmoid(self.relG(inter_k) + self.GATE_BIAS)
        k_add = self.relB(inter_k)
        _kv_j = kv_j * k_gate + k_add * (1 - k_gate)
        k_j = self.k(torch.cat([_kv_j, edge_features], -1)).view(_kv_j.size(0), H, C)
        v_j = self.v(_kv_j).view(_kv_j.size(0), H, C)

        attention_scores = (q_i * k_j).sum(-1)
        alpha = softmax(attention_scores, edge_index_i)
        alpha = self.attn_dropout(alpha)

        ret = v_j * alpha.unsqueeze(-1)  # weigh the incoming states by alphas
        return ret


class TransformerConv(MessagePassing):
    def __init__(self,
                 config:TransformerConfig,
                 **kw):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = torch.nn.ModuleList()
        self.layer.append(TransformerAttentionConv(config))
        if self.is_decoder:
            self.layer.append(TransformerAttentionConv(config))

        self.layer.append(TransformerLayerFF(config))

    def forward(self,
                states: torch.Tensor,
                edge_index: Adj,
                ctx_states=None,
                ctx_edge_index: Adj=None,
                **kwargs):

        self_attn_out = self.layer[0](edge_index, x=states, kv=states)
        states = self_attn_out
        if self.is_decoder:
            assert(ctx_states is not None and ctx_edge_index is not None)
            ctx_attn_out = self.layer[1](ctx_edge_index, x=states, kv=ctx_states)
            states = ctx_attn_out

        states = self.layer[-1](states)
        return states


class RelationalTransformerConv(MessagePassing):
    def __init__(self,
                 config:TransformerConfig,
                 **kw):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = torch.nn.ModuleList()
        self.layer.append(RelationalTransformerAttentionConv(config))
        if self.is_decoder:
            self.layer.append(TransformerAttentionConv(config))

        self.layer.append(TransformerLayerFF(config))

    def forward(self,
                states: torch.Tensor,
                edge_index: Adj,
                edge_features: torch.Tensor=None,
                ctx_states=None,
                ctx_edge_index: Adj=None,
                **kwargs):

        self_attn_out = self.layer[0](edge_index, x=states, kv=states, edge_features=edge_features)
        states = self_attn_out
        if self.is_decoder:
            assert(ctx_states is not None and ctx_edge_index is not None)
            ctx_attn_out = self.layer[1](ctx_edge_index, x=states, kv=ctx_states)
            states = ctx_attn_out

        states = self.layer[-1](states)
        return states


class RelationalTransformer(torch.nn.Module):
    def __init__(self, config:TransformerConfig, **kw):
        super(RelationalTransformer, self).__init__(**kw)
        self.config = config
        self.hdim = self.config.d_model
        self.vocabsize = self.config.vocab_size
        self.relvocabsize = self.config.relvocab_size

        self.node_emb = torch.nn.Embedding(self.vocabsize, self.hdim)
        self.edge_emb = torch.nn.Embedding(self.relvocabsize, self.hdim)

        self.layers = torch.nn.ModuleList()
        for i in range(self.config.num_layers):
            layer = RelationalTransformerConv(config=config)
            self.layers.append(layer)

    def forward(self,
                node_ids: torch.Tensor,
                edge_index: Adj,
                edge_ids: torch.Tensor,
                ctx: torch.Tensor=None,
                ctx_edge_index: Adj=None,
                **kwargs):
        node_states = self.node_emb(node_ids)
        edge_feats = self.edge_emb(edge_ids)

        for layer in self.layers:
            node_states = layer(node_states, edge_index, edge_feats,
                                ctx_states=ctx, ctx_edge_index=ctx_edge_index)
        return node_states


class RelativePositionTransformer(RelationalTransformer):
    PADID = 0
    MAXLEN = 256

    def __init__(self, config:TransformerConfig, **kw):
        config.relvocab_size = self.MAXLEN
        super(RelativePositionTransformer, self).__init__(config, **kw)

    def forward(self,
                x: torch.Tensor,      # (batsize, seqlen) ids
                ctx: torch.Tensor=None, # (batsize, ctxlen, dim)
                ctx_mask: torch.Tensor=None,    # (batsize, ctxlen) bool
                **kwargs):
        edges = []
        edge_ids = []
        ctx_edges = [] if ctx is not None else None
        B, L = x.size()
        ctxL = ctx.size(1) if ctx is not None else None
        for i, x_i in enumerate(x.detach().cpu().numpy()):
            for j in range(len(x_i)):
                x_ij = x_i[j]
                if x_ij != self.PADID:
                    edges.append([j + i*L, j + i*L])
                    edge_ids.append(0)
                    for k in range(j+1, len(x_i)):
                        x_ik = x_i[k]
                        if x_ik != self.PADID:
                            edges.append([j + i*L, k + i*L])
                            edges.append([k + i*L, j + i*L])
                            assert(abs(j - k) < self.MAXLEN/2)
                            edge_ids.append((k - j + self.MAXLEN) % self.MAXLEN)
                            edge_ids.append((j - k + self.MAXLEN) % self.MAXLEN)
                    if ctx is not None:
                        for k in range(0, ctxL):
                            if ctx_mask[i, k].detach().cpu().item() != 0:
                                ctx_edges.append([k + i*ctxL, j + i*L])

        edges = torch.tensor(edges).to(x.device).T
        edge_ids = torch.tensor(edge_ids).to(x.device)
        ctx_edges = torch.tensor(ctx_edges).to(x.device).T if ctx_edges is not None else None

        # flatten everything
        x = x.view(-1)
        if ctx is not None:
            ctx = ctx.view(-1, ctx.size(-1))

        out = super(RelativePositionTransformer, self).forward(x, edges, edge_ids, ctx=ctx, ctx_edge_index=ctx_edges)
        out = out.view(B, L, out.size(-1))

        return out


class MGATConv(MessagePassing):
    def __init__(self,
                 indim: int,
                 outdim: int=None,
                 numheads: int = 1,
                 negative_slope: float = 0.2,
                 dropout: float = 0.,
                 add_self_loops: bool = True,
                 bias: bool = True,
                 **kwargs):
        super(MGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.indim = indim
        self.outdim = outdim if outdim is not None else outdim
        self.numheads = numheads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        assert outdim % numheads == 0, f" 'outdim' must be divisible by 'numheads' but {outdim} resp. {numheads} given"
        self.size_per_head = outdim // numheads

        self.lin = Linear(indim, outdim, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(outdim))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=False):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        assert return_attention_weights == False, "Returning attention weights not supported. "
        H, C = self.numheads, self.size_per_head

        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x_l = x_r = self.lin(x).view(x.size(0), H, C)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                num_nodes = size[1] if size is not None else num_nodes
                num_nodes = x_r.size(0) if x_r is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size)

        out = out.view(-1, self.heads * self.size_per_head)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self,
                x_j: Tensor,
                x_i: Tensor,
                edge_index_i: Tensor,
                edge_index_j: Tensor,
                size_i: Optional[int]) -> Tensor:
        attention_scores = (x_j * x_i).sum(-1)
        alpha = softmax(attention_scores, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        ret = x_j * alpha.unsqueeze(-1)     # weigh the incoming states by alphas
        return ret

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


def try_mgat_conv():
    m = MGATConv(5, 6, numheads=2)

    x = torch.randn(4, 5)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    out = m(x, edge_index)

    print(out)


def try_transformer_attn_conv():
    conf = TransformerConfig(d_model=6, num_heads=2, d_kv=3, d_ff=24, num_layers=2)
    m = TransformerAttentionConv(conf)

    q = torch.randn(4, 6)
    kv = torch.randn(7, 6)

    edge_index = torch.tensor([[0, 2, 4, 6],[0, 1, 2, 3]])
    # edge_features = torch.randn(4, 6)

    out = m(edge_index, x=q, kv=kv)

    print(out)


def try_relational_transformer_attn_conv():
    conf = TransformerConfig(d_model=6, num_heads=2, d_kv=3, d_ff=24, num_layers=2)
    m = RelationalTransformerAttentionConv(conf)

    q = torch.randn(4, 6)
    kv = torch.randn(7, 6)

    edge_index = torch.tensor([[0, 2, 4, 6],[0, 1, 2, 3]])
    edge_features = torch.randn(4, 6)

    out = m(edge_index, x=q, kv=kv, edge_features=edge_features)

    print(out)


def try_relational_transformer():
    conf = TransformerConfig(d_model=6, num_heads=2, d_kv=3, d_ff=24, num_layers=2)
    conf.vocab_size = 11
    conf.relvocab_size = 3

    node_ids = torch.randint(1, conf.vocab_size, (5,))
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 4, 0, 1, 2, 3]])
    edge_ids = torch.randint(1, conf.relvocab_size, (10,))

    m = RelationalTransformer(conf)

    out = m(node_ids, edge_index, edge_ids)

    print(out)


def try_relative_position_transformer():
    conf = TransformerConfig(d_model=6, num_heads=2, d_kv=3, d_ff=24, num_layers=2)
    conf.vocab_size = 11

    m = RelativePositionTransformer(conf)

    x = torch.tensor([
        [1, 2, 3, 0, 0, 0],
        [4, 5, 3, 6, 7, 9],
        [10, 0, 0, 0, 0, 0]
    ])

    out = m(x)
    print(out)

    print(out.size())

    out[1, -1].sum().backward()
    print(m.node_emb.weight.grad)


def try_relative_position_transformer_with_context():
    conf = TransformerConfig(d_model=6, num_heads=2, d_kv=3, d_ff=24, num_layers=2)
    conf.vocab_size = 11
    conf.is_decoder = True

    m = RelativePositionTransformer(conf)

    x = torch.tensor([
        [1, 2, 3, 0, 0, 0],
        [4, 5, 3, 6, 7, 9],
        [10, 0, 0, 0, 0, 0]
    ])

    ctx = torch.nn.Parameter(torch.randn(3, 4, 6))
    ctx_mask = torch.tensor([
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ])

    out = m(x, ctx=ctx, ctx_mask=ctx_mask)
    print(out)

    print(out.size())

    out[0, 0].sum().backward()
    print(m.node_emb.weight.grad)
    print(ctx.grad)


if __name__ == '__main__':
    # try_mgat_conv()
    # try_transformer_attn_conv()
    # try_relational_transformer_attn_conv()
    # try_relational_transformer()
    # try_relative_position_transformer()
    try_relative_position_transformer_with_context()