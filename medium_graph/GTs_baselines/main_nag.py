import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from eval import *
from parse import parse_method, parser_add_main_args

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
args.dim_inner = args.hidden_channels
args.n_heads = args.num_heads
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

if args.model == "nagphormer":
    from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                    to_undirected, to_dense_adj, scatter)
    from torch_geometric.utils.num_nodes import maybe_num_nodes

    def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
        """
        Implement different eigenvector normalizations.
        """

        EigVals = EigVals.unsqueeze(0)

        if normalization == "L1":
            # L1 normalization: eigvec / sum(abs(eigvec))
            denom = EigVecs.norm(p=1, dim=0, keepdim=True)

        elif normalization == "L2":
            # L2 normalization: eigvec / sqrt(sum(eigvec^2))
            denom = EigVecs.norm(p=2, dim=0, keepdim=True)

        elif normalization == "abs-max":
            # AbsMax normalization: eigvec / max|eigvec|
            denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

        elif normalization == "wavelength":
            # AbsMax normalization, followed by wavelength multiplication:
            # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
            denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
            eigval_denom = torch.sqrt(EigVals)
            eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
            denom = denom * eigval_denom * 2 / np.pi

        elif normalization == "wavelength-asin":
            # AbsMax normalization, followed by arcsin and wavelength multiplication:
            # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
            denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
            EigVecs = torch.asin(EigVecs / denom_temp)
            eigval_denom = torch.sqrt(EigVals)
            eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
            denom = eigval_denom

        elif normalization == "wavelength-soft":
            # AbsSoftmax normalization, followed by wavelength multiplication:
            # eigvec / (softmax|eigvec| * sqrt(eigval))
            denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
            eigval_denom = torch.sqrt(EigVals)
            eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
            denom = denom * eigval_denom

        else:
            raise ValueError(f"Unsupported normalization `{normalization}`")

        denom = denom.clamp_min(eps).expand_as(EigVecs)
        EigVecs = EigVecs / denom

        return EigVecs

    def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
        """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

        Args:
            evals, evects: Precomputed eigen-decomposition
            max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
            eigvec_norm: Normalization for the eigen vectors of the Laplacian
        Returns:
            Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
            Tensor (num_nodes, max_freqs) of eigenvector values per node
        """
        N = len(evals)  # Number of nodes, including disconnected nodes.

        # Keep up to the maximum desired number of frequencies.
        idx = evals.argsort()[:max_freqs]
        evals, evects = evals[idx], np.real(evects[:, idx])
        evals = torch.from_numpy(np.real(evals)).clamp_min(0)

        # Normalize and pad eigen vectors.
        evects = torch.from_numpy(evects).float()
        evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
        if N < max_freqs:
            EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
        else:
            EigVecs = evects

        # Pad and save eigenvalues.
        if N < max_freqs:
            EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
        else:
            EigVals = evals.unsqueeze(0)
        EigVals = EigVals.repeat(N, 1).unsqueeze(2)

        return EigVals, EigVecs

    def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                            num_nodes=None, space_dim=0):
        """Compute Random Walk landing probabilities for given list of K steps.

        Args:
            ksteps: List of k-steps for which to compute the RW landings
            edge_index: PyG sparse representation of the graph
            edge_weight: (optional) Edge weights
            num_nodes: (optional) Number of nodes in the graph
            space_dim: (optional) Estimated dimensionality of the space. Used to
                correct the random-walk diagonal by a factor `k^(space_dim/2)`.
                In euclidean space, this correction means that the height of
                the gaussian distribution stays almost constant across the number of
                steps, if `space_dim` is the dimension of the euclidean space.

        Returns:
            2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
        """
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        source, dest = edge_index[0], edge_index[1]
        deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
        deg_inv = deg.pow(-1.)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        if edge_index.numel() == 0:
            P = edge_index.new_zeros((1, num_nodes, num_nodes))
        else:
            # P = D^-1 * A
            P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
        rws = []
        if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
            # Efficient way if ksteps are a consecutive sequence (most of the time the case)
            Pk = P.clone().detach().matrix_power(min(ksteps))
            for k in range(min(ksteps), max(ksteps) + 1):
                rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                        (k ** (space_dim / 2)))
                Pk = Pk @ P
        else:
            # Explicitly raising P to power k for each k \in ksteps.
            for k in ksteps:
                rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                        (k ** (space_dim / 2)))
        rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

        return rw_landing

    # # Eigen values and vectors.
    # evals, evects = None, None
    
    # # Eigen-decomposition with numpy, can be reused for Heat kernels.
    # L = to_scipy_sparse_matrix(
    #     *get_laplacian(dataset.graph['edge_index'],
    #                     num_nodes=n)
    # )
    # evals, evects = np.linalg.eigh(L.toarray())
    
    # EigVals, pe = get_lap_decomp_stats(
    #     evals=evals, evects=evects,
    #     max_freqs=5,
    #     eigvec_norm='L2')
        
rw_landing = get_rw_landing_probs(ksteps=range(1,5),
                                    edge_index=dataset.graph['edge_index'],
                                    num_nodes=n)
pe = rw_landing

dataset.graph['node_feat'] = torch.concat([dataset.graph['node_feat'],pe], dim=-1)
print(dataset.graph['node_feat'].shape)
d += pe.shape[1]

if args.model == 'nagphormer':
    import utils
    processed_features = utils.re_features(dataset.graph['edge_index'], dataset.graph['node_feat'], args.hops)  # return (N, hops+1, d)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), processed_features.to(device)

### Load method ###
model = parse_method(args, n, c, d, device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

args.method = args.gnn
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    best_test = float('-inf')
    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        out = model(dataset.graph['node_feat'])
        if args.model=="nodeformer":
            out, loss_add = out
            print(loss_add)
        if args.dataset in ('questions'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        if args.model=="nodeformer":
            loss += loss_add[0]
        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            if args.save_model:
                save_model(args, model, optimizer, run)

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()
### Save results ###
save_result(args, results)

