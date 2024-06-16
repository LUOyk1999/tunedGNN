import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class MPNNs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3,
            in_dropout=0.15, dropout=0.5, heads=1,
            pre_ln=False, post_bn=True, local_attn=False, res=True, dp=True, ln=True, add_=True, sage=False):
        super(MPNNs, self).__init__()

        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.post_bn = post_bn

        self.res = res
        self.dp = dp
        self.add_ = add_
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.post_bn:
            self.post_bns = torch.nn.ModuleList()

        ## first layer
        if local_attn:
            self.local_convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        elif sage:
            self.local_convs.append(SAGEConv(in_channels, heads*hidden_channels))
        else:
            self.local_convs.append(GCNConv(in_channels, heads*hidden_channels,
                cached=False, normalize=True))

        self.lins.append(torch.nn.Linear(in_channels, heads*hidden_channels))
        self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
        if self.pre_ln:
            self.pre_lns.append(torch.nn.LayerNorm(in_channels))
        if self.post_bn:
            self.post_bns.append(torch.nn.BatchNorm1d(heads*hidden_channels))

        ## following layers
        for _ in range(local_layers-1):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if local_attn:
                self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads))
            elif sage:
                self.local_convs.append(SAGEConv(heads*hidden_channels, heads*hidden_channels))
            else:
                self.local_convs.append(GCNConv(heads*hidden_channels, heads*hidden_channels,
                    cached=False, normalize=True))

            self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.post_bn:
                self.post_bns.append(torch.nn.BatchNorm1d(heads*hidden_channels))

        self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
        self.ln = torch.nn.LayerNorm(heads*hidden_channels)
        self.pred_local = torch.nn.Linear(heads*hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        if self.post_bn:
            for p_bn in self.post_bns:
                p_bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.pred_local.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)

        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
                
            if self.post_bn:
                x = self.post_bns[i](x)
            x = F.relu(x)
            if self.dp:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.add_:
                x_local = x_local + x
            else:
                x_local = x
        
        x = self.pred_local(x_local)

        return x
