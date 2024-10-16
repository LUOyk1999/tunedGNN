from genericpath import exists
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

from vq import VectorQuantizerEMA
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

class TransformerConv(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_dim: int,
        num_nodes: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        skip: bool = True,
        dist_count_norm: bool = True,
        conv_type: str = 'local',
        num_centroids: Optional[int] = None,
        # centroid_dim: int = 64,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and skip
        self.skip = skip
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.dist_count_norm = dist_count_norm
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)
        # if edge_dim is not None:
        #     self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        # else:
        #     self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels, heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels, out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)


        if self.conv_type != 'local' :
            self.vq = VectorQuantizerEMA(
                num_centroids, 
                global_dim, 
                decay=0.99
            )
            c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.short)
            self.register_buffer('c_idx', c)
            self.attn_fn = F.softmax

            self.lin_proj_g = Linear(in_channels, global_dim)
            self.lin_key_g = Linear(global_dim, heads * out_channels)
            self.lin_query_g = Linear(global_dim, heads * out_channels)
            self.lin_value_g = Linear(global_dim, heads * out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        # if self.edge_dim:
        #     self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()



    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, 
                    pos_enc=None):

        if self.conv_type == 'local' :
            out = self.local_forward(x, edge_index, edge_attr)

        elif self.conv_type == 'global' :
            out = self.global_forward(x, pos_enc)

        elif self.conv_type == 'full' :
            out_local = self.local_forward(x, edge_index, edge_attr)
            out_global = self.global_forward(x, pos_enc)
            out = torch.cat([out_local, out_global], dim=1)
            
        else :
            raise NotImplementedError

        return out


    def global_forward(self, x, pos_enc=None):

        d, h = self.out_channels, self.heads
        scale = 1.0 / math.sqrt(d)

        q_x = torch.cat([self.lin_proj_g(x)], dim=1)

        k_x = self.vq.get_k()
        v_x = self.vq.get_v()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=h), (q, k, v))
        dots = torch.einsum('h i d, h j d -> h i j', q, k) * scale

        c, c_count = self.c_idx.unique(return_counts=True)
        # print(f'c count mean:{c_count.float().mean().item()}, min:{c_count.min().item()}, max:{c_count.max().item()}')

        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
        centroid_count[c.to(torch.long)] = c_count
        dots += torch.log(centroid_count.view(1,1,-1))

        attn = self.attn_fn(dots, dim = -1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum('h i j, h j d -> h i d', attn, v)
        out = rearrange(out, 'h n d -> n (h d)')

        # Update the centroids
        if self.training :
            x_idx = self.vq.update(q_x)
            self.c_idx = x_idx.squeeze().to(torch.short)

        return out

    def local_forward(self, x: Tensor, edge_index: Adj,
                    edge_attr: OptTensor = None):
            
        H, C = self.heads, self.out_channels

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                            edge_attr=edge_attr, size=None)
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.skip:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:

        # if self.lin_edge is not None:
        #     assert edge_attr is not None
        #     edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        #     key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)


        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class Transformer(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, global_dim, num_layers, heads, ff_dropout, attn_dropout, skip, dist_count_norm, conv_type,num_centroids, no_bn, norm_type):
        super(Transformer, self).__init__()

        # self.fc_in = nn.Linear(in_channels, hidden_channels) ###################
        if norm_type == 'batch_norm' :
            norm_func = nn.BatchNorm1d
        elif norm_type == 'layer_norm' :
            norm_func = nn.LayerNorm

        if no_bn :
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels)
            )            
        else :
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                norm_func(hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels)
            )

        self.convs = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()

        # assert num_layers == 1
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    heads=heads,
                    dropout=attn_dropout, 
                    skip=skip, 
                    dist_count_norm=dist_count_norm,
                    conv_type=conv_type,
                    num_centroids=num_centroids
                )
            )
            h_times = 2 if conv_type == 'full' else 1

            if no_bn :
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(h_times*hidden_channels*heads, hidden_channels*heads),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels*heads, hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )
            else :
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(h_times*hidden_channels*heads, hidden_channels*heads),
                        norm_func(hidden_channels*heads),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels*heads, hidden_channels),
                        norm_func(hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )

        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fc_out.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, pos_enc=None):
        x = self.fc_in(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.ffs[i](x)
        x = self.fc_out(x)
        return x
