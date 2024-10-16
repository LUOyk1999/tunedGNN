import math
import numpy as np
import scipy as sp
from typing import Any, Optional
import torch


def laplacian_matrix(senders: np.ndarray, receivers: np.ndarray,
        weights: Optional[np.ndarray] = None, n: Optional[int] = None) -> Any:

  if weights is None:
    weights = 0*senders + 1

  if n is None:
    n = senders.max()
    if receivers.max() > n:
      n = receivers.max()
    n += 1

  s = senders.tolist() + list(range(n))
  t = receivers.tolist() + list(range(n))
  w = weights.tolist() + [0.0] * n
  adj = sp.sparse.csc_matrix((w, (s, t)), shape=(n, n))
  lap = adj * -1.0
  lap.setdiag(np.ravel(adj.sum(axis=0)))
  return lap

def laplacian_eigenv(senders: np.ndarray,
                     receivers: np.ndarray,
                     weights: Optional[np.ndarray] = None,
                     k=2,
                     n: Optional[int] = None):
    """Computes the k smallest non-trivial eigenvalue and eigenvectors of the Laplacian matrix corresponding to the given graph.
    Skips all constant vector.
    Args:
        senders: The sender nodes of the graph
        receivers: The receiver nodes of the graph
        weights: The weights of the edges
        k: number of eigenvalue/vector pairs (excluding trivial eigenvector)
        n: # of nodes (optional)
    Returns:
        eigen_values: array of eigenvalues
        eigen_vectors: array of eigenvectors
    """
    m = senders.shape[0]
    if weights is None:
        weights = np.ones(m)

    if n is None:
      n = senders.max()
      if receivers.max() > n:
        n = receivers.max()
      n += 1
    
    lap_mat = laplacian_matrix(senders, receivers, weights, n = n)
    # n = lap_mat.shape[0]
    k = min(n - 2, k + 1)
    # rows of eigenv correspond to graph nodes, cols correspond to eigenvalues
    eigenvals, eigenvecs = sp.sparse.linalg.eigs(lap_mat, k=k, which='SM')
    eigenvals = np.real(eigenvals)
    eigenvecs = np.real(eigenvecs)

    # sort eigenvectors in ascending order of eigenvalues
    sorted_idx = np.argsort(eigenvals)
    eigenvals = eigenvals[sorted_idx]
    eigenvecs = eigenvecs[:, sorted_idx]

    constant_eigenvec_idx = 0

    for i in range(0, k):
        # normalize the i^th eigenvector
        eigenvecs[:, i] = eigenvecs[:, i] / np.sqrt((eigenvecs[:, i]**2).sum())
        if eigenvecs[:, i].var() <= 1e-7:
            constant_eigenvec_idx = i

    non_constant_idx = [*range(0, k)]
    non_constant_idx.remove(constant_eigenvec_idx)

    eigenvals = eigenvals[non_constant_idx]
    eigenvecs = eigenvecs[:, non_constant_idx]

    return eigenvals, eigenvecs

def generate_random_regular_graph1(num_nodes, degree, rng=None):
  """Generates a random 2d-regular graph with n nodes using permutations algorithm.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  if rng is None:
    rng = np.random.default_rng()

  senders = [*range(0, num_nodes)] * degree
  receivers = []
  for _ in range(degree):
    receivers.extend(rng.permutation(list(range(num_nodes))).tolist())

  senders, receivers = [*senders, *receivers], [*receivers, *senders]

  senders = np.array(senders)
  receivers = np.array(receivers)

  return senders, receivers



def generate_random_regular_graph2(num_nodes, degree, rng=None):
  """Generates a random 2d-regular graph with n nodes using simple variant of permutations algorithm.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  if rng is None:
    rng = np.random.default_rng()

  senders = [*range(0, num_nodes)] * degree
  receivers = rng.permutation(senders).tolist()

  senders, receivers = [*senders, *receivers], [*receivers, *senders]

  return senders, receivers


def generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng=None):
  """Generates a 2d-regular graph with n nodes using d random hamiltonian cycles.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  if rng is None:
    rng = np.random.default_rng()

  senders = []
  receivers = []
  for _ in range(degree):
    permutation = rng.permutation(list(range(num_nodes))).tolist()
    for idx, v in enumerate(permutation):
      u = permutation[idx - 1]
      senders.extend([v, u])
      receivers.extend([u, v])

  senders = np.array(senders)
  receivers = np.array(receivers)

  return senders, receivers


def generate_random_expander(x, degree=3, algorithm='Random-d', rng=None, max_num_iters=100, exp_index=0):
  """Generates a random d-regular expander graph with n nodes.
  Returns the list of edges. This list is symmetric; i.e., if
  (x, y) is an edge so is (y,x).
  Args:
    num_nodes: Number of nodes in the desired graph.
    degree: Desired degree.
    rng: random number generator
    max_num_iters: maximum number of iterations
  Returns:
    senders: tail of each edge.
    receivers: head of each edge.
  """

  num_nodes = x.shape[0]

  if rng is None:
    rng = np.random.default_rng()
  
  eig_val = -1
  eig_val_lower_bound = max(0, 2 * degree - 2 * math.sqrt(2 * degree - 1) - 0.1)

  max_eig_val_so_far = -1
  max_senders = []
  max_receivers = []
  cur_iter = 1

  if num_nodes <= degree:
    degree = num_nodes - 1    
    
  # if there are too few nodes, random graph generation will fail. in this case, we will
  # add the whole graph.
  if num_nodes <= 10:
    for i in range(num_nodes):
      for j in range(num_nodes):      
        if i != j:
          max_senders.append(i)
          max_receivers.append(j)
  else:
    while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters:
      if algorithm == 'Random-d':
        senders, receivers = generate_random_regular_graph1(num_nodes, degree, rng)
      elif algorithm == 'Random-d-2':
        senders, receivers = generate_random_regular_graph2(num_nodes, degree, rng)
      elif algorithm == 'Hamiltonian':
        senders, receivers = generate_random_graph_with_hamiltonian_cycles(num_nodes, degree, rng)
      else:
        raise ValueError('prep.exp_algorithm should be one of the Random-d or Hamiltonian')
      [eig_val, _] = laplacian_eigenv(senders, receivers, k=1, n=num_nodes)
      if len(eig_val) == 0:
        print("num_nodes = %d, degree = %d, cur_iter = %d, mmax_iters = %d, senders = %d, receivers = %d" %(num_nodes, degree, cur_iter, max_num_iters, len(senders), len(receivers)))
        eig_val = 0
      else:
        eig_val = eig_val[0]

      if eig_val > max_eig_val_so_far:
        max_eig_val_so_far = eig_val
        max_senders = senders
        max_receivers = receivers

      cur_iter += 1

  # eliminate self loops.
  non_loops = [
      *filter(lambda i: max_senders[i] != max_receivers[i], range(0, len(max_senders)))
  ]

  senders = np.array(max_senders)[non_loops]
  receivers = np.array(max_receivers)[non_loops]

  max_senders = torch.tensor(max_senders, dtype=torch.long).view(-1, 1)
  max_receivers = torch.tensor(max_receivers, dtype=torch.long).view(-1, 1)

  if exp_index == 0:
    expander_edges = torch.cat([max_senders, max_receivers], dim=1)
    
  return expander_edges
