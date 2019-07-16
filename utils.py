# ====================================================================================== #
# Useful functions for simulating percolation.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from scipy.sparse import csr_matrix


def find_all_clusters(adj):
    """Find all connected clusters of points by searching through all points and their
    neighbors given the adjacency matrix.

    Parameters
    ---------
    adj : scipy.sparse.csr_matrix

    Returns
    -------
    list of lists of ints
    """
    
    assert type(adj)==csr_matrix
    assert (adj.data==1).all()

    clusters = []
    remainingpts = set(range(adj.shape[0]))
    
    while remainingpts:
        # seed pt
        thisCluster = [remainingpts.pop()]
        toSearch = set(adj.getrow(thisCluster[-1]).indices)

        # sequentially search through neighborhood starting with seed pt
        while toSearch:
            thisCluster.append(toSearch.pop())
            remainingpts.remove(thisCluster[-1])
            for neighbor in adj.getrow(thisCluster[-1]).indices:
                if neighbor in remainingpts and not neighbor in toSearch:
                    toSearch.add(neighbor)

        clusters.append(thisCluster)
    return clusters
