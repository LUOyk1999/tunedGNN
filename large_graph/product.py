import os.path as osp
import time
import argparse
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear
from tqdm import tqdm
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.utils import index_to_mask


class GNNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout, ln, gnn='gcn'):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        if gnn=='gcn':
            self.conv = GCNConv(in_channels, out_channels)
        elif gnn=='sage':
            self.conv = SAGEConv(in_channels, out_channels)
        elif gnn=='gat':
            self.conv = GATConv(in_channels, out_channels)
        self.dropout = dropout
        self.ln = ln
    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        if self.ln:
            x = self.norm(x).relu()
        else:
            x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv(x, edge_index)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, ln, gnn, jk, res):
        super().__init__()

        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.norm = LayerNorm(hidden_channels, elementwise_affine=True)
        self.ln = ln
        self.jk = jk
        self.res = res
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GNNConv(
                hidden_channels,
                hidden_channels,
                dropout,
                self.ln,
                gnn=gnn
            )
            self.convs.append(conv)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x_final = 0
        x = self.lin1(x)
        x_final += x
        for (conv) in self.convs:
            if self.res:
                x = conv(x, edge_index) + x
            else:
                x = conv(x, edge_index)
            x_final += x
        if self.ln:
            x = self.norm(x).relu()
        else:
            x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.jk:
            x = x_final
        else:
            pass

        return self.lin2(x)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--hidden_channels', type=int, default=160)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--epochs', type=int, default=1001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gnn', type=str, default='gcn')
parser.add_argument('--ln', action='store_true')
parser.add_argument('--jk', action='store_true')
parser.add_argument('--res', action='store_true')
args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
transform = T.Compose([T.ToDevice(device), T.ToSparseTensor()])
root = osp.join(osp.dirname(osp.realpath(__file__)), 'data/ogb')
dataset = PygNodePropPredDataset('ogbn-products', root,
                                 transform=T.AddSelfLoops())
evaluator = Evaluator(name='ogbn-products')

data = dataset[0]
split_idx = dataset.get_idx_split()
for split in ['train', 'valid', 'test']:
    data[f'{split}_mask'] = index_to_mask(split_idx[split], data.y.shape[0])

train_loader = RandomNodeLoader(data, num_parts=10, shuffle=True,
                                num_workers=5)
# Increase the num_parts of the test loader if you cannot fit
# the full batch graph into your GPU:
test_loader = RandomNodeLoader(data, num_parts=1, num_workers=5)

model = GNN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,  
    dropout=args.dropout,
    ln = args.ln,
    gnn = args.gnn,
    jk = args.jk,
    res = args.res,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


def train(epoch):
    model.train()


    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()

        data = transform(data)
        out = model(data.x, data.adj_t)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].view(-1))
        (loss).backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

    return total_loss / total_examples


@torch.no_grad()
def test(epoch):
    model.eval()

    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}

    for data in test_loader:
        data = transform(data)
        out = model(data.x, data.adj_t)
        out = out.argmax(dim=-1, keepdim=True)
        for split in ['train', 'valid', 'test']:
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

    train_acc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['acc']

    test_acc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['acc']

    return train_acc, valid_acc, test_acc


times = []
best_val = 0.0
final_train = 0.0
final_test = 0.0
for epoch in range(1, args.epochs):
    start = time.time()
    loss = train(epoch)
    train_acc, val_acc, test_acc = test(epoch)
    if val_acc > best_val:
        best_val = val_acc
        final_train = train_acc
        final_test = test_acc
    print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * val_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * final_test:.2f}%')
    times.append(time.time() - start)

print(f'Final Train: {100 * final_train:.2f}%, '
        f'Best Valid: {100 * best_val:.2f}%, '
        f'Best Test: {100 * final_test:.2f}%')
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")