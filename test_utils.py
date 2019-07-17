# ====================================================================================== #
# Useful functions for simulating percolation.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
from .utils import *


def test_find_all_clusters():
    from .utils import _find_all_clusters

    np.random.seed(0)
    adj = (np.random.rand(50,50)<.05).astype(int)
    adj += adj.T
    adj[np.diag_indices_from(adj)] = 0
    adj[adj>0] = 1
    adj = csr_matrix(adj)

    clusters1 = _find_all_clusters(adj, True)
    clusters2 = find_all_clusters(adj, True)

    assert all([sorted(i)==sorted(j) for i,j in zip(clusters1,clusters2)])
