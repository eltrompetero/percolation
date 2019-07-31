# ====================================================================================== #
# Useful functions for simulating percolation.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from scipy.sparse import csr_matrix
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

def random_walk(xy, adj, tmax,
                rng=np.random,
                return_radius=True,
                fast=False):
    """Random walk starting from a random site.
    
    Parameters
    ----------
    xy : ndarray
        Coordinates for measuring distance.
    adj : scipy.sparse.csr_matrix
    tmax : int
        Max number of steps to take before stopping.
    rng : np.random.RandomState
    return_radius : bool, True
    fast : bool, False
        
    Returns
    -------
    ndarray of ints
        Path given by indices of xy visited.
    ndarray
        Distance from origin.
    ndarray
        Max radius during trajectory.
    """
    
    if not fast:
        assert len(xy)==adj.shape[0]
        _check_adj(adj)

    path = []
    path.append(rng.randint(len(xy)))
    xy0 = xy[path[0]]  # for avoiding adding element access time in loop
    
    indices = adj.indices.tolist()
    indptr = adj.indptr.tolist()
    lenindptr = len(indptr)

    for i in range(1, tmax):
        if path[-1]<lenindptr:
            path.append(rng.choice(indices[indptr[path[i-1]]:indptr[path[i-1]+1]]))
        else:
            path.append(rng.choice(indices[indptr[path[i-1]]:]))
    
    if return_radius:
        d = np.linalg.norm(xy[path]-xy0, axis=1)
        return np.array(path), d, np.maximum.accumulate(d)
    return np.array(path)

def random_walk_blind(xy, adj, tmax, rng=np.random, fast=False):
    """Random walk starting from a random site with "blind ant" that can make a bad choice
    and be forced to stay in the same spot.
    
    Parameters
    ----------
    xy : ndarray
        Coordinates for measuring distance.
    adj : scipy.sparse.csr_matrix
    tmax : int
        Max number of steps to take before stopping.
    rng : np.random.RandomState
    fast : bool, False
        If False, check adjacency matrix.
        
    Returns
    -------
    ndarray of ints
        Path given by indices of xy visited.
    ndarray
        Distance from origin.
    ndarray
        Max radius during trajectory.
    """
    
    if not fast:
        assert len(xy)==adj.shape[0]
        _check_adj(adj)

    path = []
    path.append(rng.randint(len(xy)))
    xy0 = xy[path[0]]  # for avoiding adding element access time in loop
    
    indices = adj.indices.tolist()
    indptr = adj.indptr.tolist()
    data = adj.data.tolist()
    lenindptr = len(indptr)

    for i in range(1, tmax):
        if path[-1]<lenindptr:
            p = rng.choice(indices[indptr[path[i-1]]:indptr[path[i-1]+1]])
        else:
            p = rng.choice(indices[indptr[path[i-1]]:])

        if data[p]:
            path.append(p)
        else:
            path.append(path[-1])
    
    d = np.linalg.norm(xy[path]-xy0, axis=1)
    return np.array(path), d, np.maximum.accumulate(d)

def random_walk_with_cost(xy, adj, plus_factor, minus_factor, tmax,
                          rng=np.random,
                          return_radius=True,
                          start_site=None):
    """Random walk starting from a random site with a cost in resource when exploring a
    old site and a gain in resource when exploring a new site. Walk stops when resource is
    depleted.
    
    Parameters
    ----------
    xy : ndarray
        Coordinates for measuring distance.
    adj : scipy.sparse.csr_matrix
    plus_factor : float
        Amount of resource gained per new site.
    minus_factor : float
        Amount of resource lost per old site.
    tmax : int
        Max number of steps to take before stopping.
    rng : np.random.RandomState
    return_radius : bool, True
    start_site : int, None
        
    Returns
    -------
    ndarray of ints
        Path given by indices of xy visited.
    ndarray
        Radius during trajectory.
    """
    
    assert plus_factor>0 and minus_factor<0
    start_site = start_site or rng.randint(len(xy))
    path = []
    path.append( start_site )
    radius = [0]
    visited = set((path[0],))
    resource = plus_factor
    
    indices = adj.indices
    indptr = adj.indptr

    i=0
    while resource>0 and i<tmax:
        # move to a random neighbor
        if path[-1]<indptr.size:
            path.append(rng.choice(indices[indptr[path[i-1]]:indptr[path[i-1]+1]]))
        else:
            path.append(rng.choice(indices[indptr[path[i-1]]:]))
        
        # save path and new distance
        if path[-1] in visited:
            resource += minus_factor
        else:
            visited.add(path[-1])
            resource += plus_factor
        i += 1
    
    if return_radius:
        xy0 = xy[path[0]]
        return np.array(path), np.maximum.accumulate(np.linalg.norm(xy[path]-xy0, axis=1))
    return np.array(path)

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

def _check_adj():
    """Helper function for checking adjacency matrices.
    
    Checks that matrix is of type csr_matrix, matrix is square, nonzero elements are all
    1, that matrix symmetric, and that diagonal elements are 0.
    """

    assert type(adj)==csr_matrix
    assert adj.shape[0]==adj.shape[1]
    assert ((adj.data==0)|(adj.data==1)).all()
    assert (adj-adj.transpose()).count_nonzero()==0
    assert (adj.diagonal()==0).all()
