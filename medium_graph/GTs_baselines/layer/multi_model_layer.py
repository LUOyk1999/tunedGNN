import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from layer.Exphormer import ExphormerAttention


class LocalModel(nn.Module):
    def __init__(self, dim_h, local_gnn_type, edge_type, edge_attr_type, num_heads,
                pna_degrees=None, equivstable_pe=False, dropout=0.0,
                layer_norm=False, batch_norm=True):
        super().__init__()

        self.dim_h = dim_h
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.edge_type = edge_type
        self.edge_attr_type = edge_attr_type

        if self.edge_type == 'edge_index' or self.edge_type is None:
            self.edge_type = 'edge_index'
            self.edge_attr_type = 'edge_attr'
        elif self.edge_type == 'exp':
            self.edge_type = 'expander_edge_index'
            self.edge_attr_type = 'expander_edge_attr'

        # Local message-passing model.
        if local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GCN':
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GraphSAGE':
            self.local_model = pygnn.SAGEConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)

    def reset_parameters(self):
        self.local_model.reset_parameters()
            
    def forward(self, x, edge_index, virt_h, virt_edge_index):
        h = x

        h_in1 = h  # for first residual connection

        if edge_index is None:
            raise ValueError(f'edge type {self.edge_type} is not stored in the data!')

        self.local_model: pygnn.conv.MessagePassing  # Typing hint.
        
    
        if self.local_gnn_type == 'GCN':
            h_local = self.local_model(h, edge_index)
        else:
            h_local = self.local_model(h, edge_index)
        h_local = self.dropout_local(h_local)
        h_local = h_in1 + h_local  # Residual connection.

        if self.layer_norm:
            h_local = self.norm1_local(h_local)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)

        return h_local


class GlobalModel(nn.Module):
    """
    Attention layer
    """

    def __init__(self, dim_h, global_model_type, edge_type, use_edge_attr, edge_attr_type, num_heads,
                dropout=0.0, attn_dropout=0.0, layer_norm=False,
                batch_norm=True, bigbird_cfg=None, exp_edges_cfg=None):

        super().__init__()

        self.dim_h = dim_h
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.num_heads = num_heads
        self.edge_type = edge_type
        self.edge_attr_type = edge_attr_type

        # Global attention transformer-style model.
        if global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Exphormer':
            self.self_attn = ExphormerAttention(dim_h, dim_h, num_heads,
                                          use_bias=False, 
                                          use_virt_nodes= False)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for Self-Attention representation.
        if self.layer_norm:
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_attn = nn.Dropout(dropout)
        
    def reset_parameters(self):
        self.self_attn.reset_parameters()
            
    def forward(self, x, edge_index, virt_h, virt_edge_index):
        h = x
        h_in1 = h  # for first residual connection

        # Multi-head attention.
        if self.global_model_type in ['Exphormer']:
            h_attn = self.self_attn(x, edge_index, virt_h, virt_edge_index)
        else:
            raise RuntimeError(f"Unexpected {self.global_model_type}")

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        return h_attn

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x


class MultiLayer(nn.Module):
    """Any combination of different models can be made here.
      Each layer can have several types of MPNN and Attention models combined.
      Examples:
      1. GCN
      2. GCN + Exphormer
      3. GINE + CustomGatedGCN
      4. GAT + CustomGatedGCN + Exphormer + Transformer
    """

    def __init__(self, dim_h,
                 local_gnn_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, exp_edges_cfg=None):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
       

        # Local message-passing models.
        self.models = []
        
        edge_attr_type = 'edge_attr'
        edge_type = 'edge_index'
        use_edge_attr = False
            
        self.models.append(GlobalModel(dim_h=dim_h,
                                    global_model_type='Exphormer',
                                    edge_type = edge_type,
                                    use_edge_attr = use_edge_attr,
                                    edge_attr_type = edge_attr_type,
                                    num_heads=self.num_heads,
                                    dropout=dropout,
                                    attn_dropout=self.attn_dropout,
                                    layer_norm=self.layer_norm,
                                    batch_norm=self.batch_norm,
                                    bigbird_cfg=bigbird_cfg,
                                    exp_edges_cfg=exp_edges_cfg))
                                            
        self.models.append(LocalModel(dim_h=dim_h,
                                    local_gnn_type=local_gnn_type,
                                    edge_type = edge_type,
                                    edge_attr_type = edge_attr_type,
                                    num_heads=num_heads,
                                    pna_degrees=pna_degrees,
                                    equivstable_pe=self.equivstable_pe,
                                    dropout=dropout,
                                    layer_norm=self.layer_norm,
                                    batch_norm=self.batch_norm))
            

        self.models = nn.ModuleList(self.models)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_index, virt_h, virt_edge_index):
        h_out_list = []
        # representations from the models
        for model in self.models:
            h_out_list.append(model(x, edge_index, virt_h, virt_edge_index))

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h)
        if self.batch_norm:
            h = self.norm2(h)

        return h
    
    def reset_parameters(self):
        for model in self.models:
            model.reset_parameters()
        
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'heads={self.num_heads}'
        return s
