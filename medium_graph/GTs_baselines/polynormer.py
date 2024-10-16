import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GlobalAttn(torch.nn.Module):
    def __init__(self, hidden_channels, heads, num_layers, beta, dropout, qk_shared=True):
        super(GlobalAttn, self).__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared

        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(num_layers, heads*hidden_channels)*self.beta)

        self.h_lins = torch.nn.ModuleList()
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if not self.qk_shared:
                self.q_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.k_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.v_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
        self.lin_out = torch.nn.Linear(heads*hidden_channels, heads*hidden_channels)

    def reset_parameters(self):
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)
        self.lin_out.reset_parameters()

    def forward(self, x):
        seq_len, _ = x.size()
        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = F.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            if self.qk_shared:
                q = k
            else:
                q = F.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            v = self.v_lins[i](x).view(seq_len, self.hidden_channels, self.heads)

            # numerator
            kv = torch.einsum('ndh, nmh -> dmh', k, v)
            num = torch.einsum('ndh, dmh -> nmh', q, kv)

            # denominator
            k_sum = torch.einsum('ndh -> dh', k)
            den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

            # linear global attention based on kernel trick
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (num/den).reshape(seq_len, -1)
            x = self.lns[i](x) * (h+beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x



class Polynormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, global_layers=2,
            in_dropout=0.15, dropout=0.5, global_dropout=0.5, heads=1, beta=-1, pre_ln=False):
        super(Polynormer, self).__init__()

        self._global = False
        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln

        ## Two initialization strategies on beta
        self.beta = beta
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(local_layers,heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(local_layers,heads*hidden_channels)*self.beta)

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        for _ in range(local_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads,
                concat=True, add_self_loops=False, bias=False))
            self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))

        self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
        self.ln = torch.nn.LayerNorm(heads*hidden_channels)
        self.global_attn = GlobalAttn(hidden_channels, heads, global_layers, beta, global_dropout)
        self.pred_local = torch.nn.Linear(heads*hidden_channels, out_channels)
        self.pred_global = torch.nn.Linear(heads*hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.global_attn.reset_parameters()
        self.pred_local.reset_parameters()
        self.pred_global.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        ## equivariant local attention
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (1-beta)*self.lns[i](h*x) + beta*x
            x_local = x_local + x

        ## equivariant global attention
        if self._global:
            x_global = self.global_attn(self.ln(x_local))
            x = self.pred_global(x_global)
        else:
            x = self.pred_local(x_local)

        return x
