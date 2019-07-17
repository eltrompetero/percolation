# ====================================================================================== #
# Useful functions for simulating percolation.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from scipy.sparse import csr_matrix
from numba import jit, njit


def _find_all_clusters(adj, fast=False):
    """Slow version to compare with find_all_clusters().
    """
    
    if not fast:
        assert type(adj)==csr_matrix
        assert (adj.data==1).all()
        assert (adj-adj.transpose()).count_nonzero()==0

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

def find_all_clusters(adj, fast=False):
    """Find all connected clusters of points by searching through all points and their
    neighbors given the adjacency matrix.

    Parameters
    ---------
    adj : scipy.sparse.csr_matrix
        Make sure this is symmetric.
    fast : bool, False

    Returns
    -------
    list of lists of ints
        Each list contains the indices of the rows in adj that belong to the same cluster.
    """
    
    if not fast:
        assert type(adj)==csr_matrix
        assert (adj.data==1).all()
        assert (adj-adj.transpose()).count_nonzero()==0
        assert (adj.diagonal()==0).all()
    
    indices = adj.indices
    indptr = adj.indptr
    n = adj.shape[0]
    
    #@jit  # there seems to be a bug with using jit here (think it's a numba bug)
    def jit_wrapper(indices, indptr, n):
        clusters = []
        remainingpts = set(range(n))
        
        while len(remainingpts)>0:
            # seed pt
            thisCluster = [remainingpts.pop()]
            if thisCluster[-1]<indptr.size:
                toSearch = set(indices[indptr[thisCluster[-1]]:indptr[thisCluster[-1]+1]])
            else:
                toSearch = set(indices[indptr[thisCluster[-1]]:])

            # sequentially search through neighborhood starting with seed pt
            while len(toSearch)>0:
                thisCluster.append(toSearch.pop())
                remainingpts.remove(thisCluster[-1])
                if thisCluster[-1]<indptr.size:
                    for neighbor in indices[indptr[thisCluster[-1]]:indptr[thisCluster[-1]+1]]:
                        if neighbor in remainingpts and not neighbor in toSearch:
                            toSearch.add(neighbor)
                else:
                    for neighbor in indices[indptr[thisCluster[-1]]:]:
                        if neighbor in remainingpts and not neighbor in toSearch:
                            toSearch.add(neighbor)

            clusters.append(thisCluster)
        return clusters

    return jit_wrapper(indices, indptr, n)

def randomly_close_bonds(adj, p, rng=np.random):
    """Given adjacency matrix keep bonds open with probability p.
    
    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
    p : float

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    
    assert type(adj) is csr_matrix
    assert p>0
    adj = adj.copy()
    
    for i in range(adj.shape[0]):
        row = adj.getrow(i)
        for j in row.indices[row.indices>i]:
            if rng.rand()>p:
                adj[i,j] = adj[j,i] = 0
    adj.eliminate_zeros()
    return adj

def random_walk(xy, adj, tmax, rng=np.random):
    """Random walk starting from a random site.
    
    Parameters
    ----------
    xy : ndarray
        Coordinates for measuring distance.
    adj : scipy.sparse.csr_matrix
    tmax : int
        Max number of steps to take before stopping.
    rng : np.random.RandomState
        
    Returns
    -------
    list of ints
        Path given by indices of xy visited.
    """

    path = np.zeros(tmax, dtype=int)
    path[0] = rng.randint(len(xy))
    xy0 = xy[path[0]]  # for avoiding adding element access time in loop
    radius = np.zeros(tmax)
    
    indices = adj.indices
    indptr = adj.indptr

    for i in range(1, tmax):
        if path[i-1]<indptr.size:
            path[i] = rng.choice(indices[indptr[path[i-1]]:indptr[path[i-1]+1]])
        else:
            path[i] = rng.choice(indices[indptr[path[i-1]]:])
        newd = np.linalg.norm(xy[path[i]]-xy0)
        if newd>radius[i-1]:
            radius[i] = newd
        else:
            radius[i] = radius[i-1]
    
    return path, radius

@njit
def cum_unique(x):
    """Keep track of cumulative number of unique elements.

    Parameters
    ----------
    x : ndarray
        One-dimensional vector.

    Returns
    -------
    ndarray
    """
    
    s = set((x[0],))
    c = np.zeros(len(x))
    c[0] = 1
    counter = 1
    for i,x_ in enumerate(x[1:]):
        if not x_ in s:
            s.add(x_)
            counter += 1
        c[i+1] = counter
    return c

def digitize_by_x(x, y, bins):
    """For measuring scaling relations.
    
    Parameters
    ----------
    x : ndarray
    y : ndarray
    bins : ndarray
    
    Returns
    -------
    ndarray
        Binned then averaged x.
    ndarray
        Binned then averaged y.
    """
    
    ix = np.digitize(x, bins)
    binx = np.zeros(ix.max()+1)
    biny = np.zeros(ix.max()+1)
    for i in np.unique(ix):
        binx[i] = x[ix==i].mean()
        biny[i] = y[ix==i].mean()
    return binx, biny
