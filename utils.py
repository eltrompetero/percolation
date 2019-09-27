# ====================================================================================== #
# Useful functions for simulating percolation.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from numba import jit, njit


def invert_shell_by_site_dict(shellsBySite):
    """Invert dictionary with keys as (x,y) coordinates and values as shell indices to a
    dictionary with shell indices as the keys. This is a non-invertible operation.

    Parameters
    ----------
    shellsBySite : dict

    Returns
    -------
    dict
        Each key is the shell index and the value is a list of (x,y) tuples. Dict is
        formed sequentially by the index of the shell so it is chronological.
    """
    
    ushells = sorted(set(shellsBySite.values()))
    shellsOfxy = {}
    for k in ushells:
        shellsOfxy[k] = []
    for xy,shell in shellsBySite.items():
        shellsOfxy[shell].append(xy)

    return shellsOfxy

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
        _check_adj(adj) 
    
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
    """Average y data points by binned x for measuring scaling relations.
    
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

def construct_adj_from_xy(xy):
    """Construct adjacency matrix from the coordinates of occupied sites on a square
    lattice.

    Parameters
    ----------
    xy : list of twoples

    Returns
    -------
    scipy.sparse.coo_matrix
    """
    
    n = len(xy)
    setxy = set(xy)  # for quick look up
    xy = xy[:]
    ix = []  # row index
    iy = []  # col index
    
    for i,xy_ in list(enumerate(xy)):
        # check for all four possible neighbors
        if (xy_[0]-1,xy_[1]) in setxy:
            ix.append(i)
            iy.append(xy.index((xy_[0]-1,xy_[1]))+i)
        if (xy_[0]+1,xy_[1]) in setxy:
            ix.append(i)
            iy.append(xy.index((xy_[0]+1,xy_[1]))+i)
        if (xy_[0],xy_[1]-1) in setxy:
            ix.append(i)
            iy.append(xy.index((xy_[0],xy_[1]-1))+i)
        if (xy_[0],xy_[1]+1) in setxy:
            ix.append(i)
            iy.append(xy.index((xy_[0],xy_[1]+1))+i)
        
        # remove point whose neighbors have been found and just symmetrize the matrix at the end
        setxy.remove(xy.pop(0))

    return coo_matrix((np.ones(2*len(ix)),(ix+iy,iy+ix)),
                      shape=(n,n))
