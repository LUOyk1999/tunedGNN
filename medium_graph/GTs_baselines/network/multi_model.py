import torch
from layer.multi_model_layer import MultiLayer
import torch.nn as nn
from network.expander_edges import generate_random_expander


class ExpanderEdgeFixer(nn.Module):
    '''
        Gets the batch and sets new edge indices + global nodes
    '''
    def __init__(self, dim_hidden, add_edge_index=False, num_virt_node=0):
        
        super().__init__()


        self.add_edge_index = add_edge_index
        self.num_virt_node = num_virt_node
        self.use_exp_edges = True

        if self.num_virt_node > 0:
            self.virt_node_emb = nn.Embedding(self.num_virt_node, dim_hidden)


    def forward(self, x, edge_index, expander_edges):
        virt_h=None
        virt_edge_index=None
        edge_types = []
        device = edge_index.device
        edge_index_sets = []
        if self.add_edge_index:
            edge_index_sets.append(edge_index)

        num_node = x.shape[0]

        if self.use_exp_edges:
            edge_index_sets.append(expander_edges)

        if self.num_virt_node > 0:
            global_h = []
            virt_edges = []
            virt_edge_attrs = []
            for idx in range(self.num_virt_node):
                global_h.append(self.virt_node_emb(torch.zeros(1, dtype=torch.long).to(device)+idx))
                virt_edge_index = torch.cat([torch.arange(num_node).view(1, -1).to(device),
                                        ((num_node+idx*1)).view(1, -1)], dim=0)
                virt_edges.append(virt_edge_index)

                virt_edge_index = torch.cat([((num_node+idx*1)).view(1, -1), 
                                                    torch.arange(num_node).view(1, -1).to(device)], dim=0)
                virt_edges.append(virt_edge_index)
                
            virt_h = torch.cat(global_h, dim=0)
            virt_edge_index = torch.cat(virt_edges, dim=1)

        
        if len(edge_index_sets) > 1:
            edge_index = torch.cat(edge_index_sets, dim=1)
        else:
            edge_index = edge_index_sets[0]

        return edge_index, virt_h, virt_edge_index

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, dim_inner):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in

        self.exp_edge_fixer = ExpanderEdgeFixer(dim_inner, add_edge_index=True, 
                                                    num_virt_node=0)

    def forward(self, x, edge_index, expander_edges):
        edge_index, virt_h, virt_edge_index = self.exp_edge_fixer(x, edge_index, expander_edges)
        return edge_index, virt_h, virt_edge_index


class MultiModel(torch.nn.Module):
    """Multiple layer types can be combined here.
    """

    def __init__(self, args, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in, args.dim_inner)
        dim_in = self.encoder.dim_in
        self.pre_mp = torch.nn.Linear(dim_in, args.dim_inner)
        self.layers = torch.nn.ModuleList()
        for _ in range(args.layers):
            # print(args)
            self.layers.append(MultiLayer(
                dim_h=args.dim_inner,
                local_gnn_type=args.local_gnn_type,
                num_heads=args.n_heads,
                dropout = args.dropout
            ))
        self.expander_edges = None
        self.post_mp = torch.nn.Linear(args.dim_inner, dim_out)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.pre_mp.reset_parameters()
        self.post_mp.reset_parameters()

    def forward(self, x, edge_index):
        if self.expander_edges is None:
            self.expander_edges = generate_random_expander(edge_index).to(edge_index.device)
        # print(self.expander_edges.shape)
        edge_index, virt_h, virt_edge_index = self.encoder(x, edge_index, self.expander_edges)
        if self.pre_mp is not None:
            x = self.pre_mp(x)
        for layer in self.layers:
            x = layer(x, edge_index, virt_h, virt_edge_index)
        
        x = self.post_mp(x)
        
        return x
