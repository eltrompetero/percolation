# ====================================================================================== #
# Useful functions for simulating random walks on percolation clusters.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
from .utils import *


def construct_planar_adj(xy):
    """Construct adjacency matrix just using set of points in plane and only connecting
    points that are neighbors of one another.

    Not assuming that boundaries are periodic.

    Parameters
    ----------
    xy : list of twoples
        Sites. Will be converted to a set.

    Returns
    -------
    scipy.sparse.csr_matrix
        Adjacency matrix ordered in the same way the sites are given in xy.
    """

    xydict = set(xy)

    def find_neighbors(xy_):
        neighbors = []
        if (xy_[0]-1,xy_[1]) in xydict:
            neighbors.append((xy_[0]-1,xy_[1]))
        if (xy_[0]+1,xy_[1]) in xydict:
            neighbors.append((xy_[0]+1,xy_[1]))
        if (xy_[0],xy_[1]-1) in xydict:
            neighbors.append((xy_[0],xy_[1]-1))
        if (xy_[0],xy_[1]+1) in xydict:
            neighbors.append((xy_[0],xy_[1]+1))
        return neighbors
    
    # loop through all sites and all 4 of its possible neighbors
    i = []
    j = []
    for counter,xy_ in enumerate(xy):
        neighbors = find_neighbors(xy_)
        if neighbors:
            for n in neighbors:
                i.append(counter)
                j.append(xy.index(n))
    
    return coo_matrix((np.ones(len(i)), (i,j))).tocsr()

def myopic_ant(xy, adj, tmax,
              rng=np.random,
              return_radius=True,
              fast=False,
              xy0=None):
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
    xy0 : int, None
        
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
    path.append(xy0 or rng.randint(len(xy)))
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

def multiple_walkers(n_walkers, xy, adj, tmax,
                     rng=np.random,
                     return_radius=True,
                     fast=False):
    """Random walk of n walkers starting from same random point.
    
    Parameters
    ----------
    n_walkers : int
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
    distFromOrigin = []
    radius = []
    origin = rng.randint(len(xy))
    
    for i in range(n_walkers):
        p, d, r = random_walk(xy, adj, tmax, fast=True)
        path.append(p)
        distFromOrigin.append(d)
        radius.append(r)

    # combine distance paths to get overall growth
    radius = np.vstack(radius).max(0)

    return path, distFromOrigin, radius

def blind_ant(xy, adj, tmax, rng=np.random, fast=False):
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

def myopic_ant_with_cost(xy, adj, plus_factor, minus_factor, tmax,
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

def absorbing(xy, adj, absorbing_sites,
              tmax=np.inf,
              rng=np.random,
              return_radius=True,
              fast=False,
              xy0=None):
    """Random walk that ends when an absorbing site is reached.
    
    Parameters
    ----------
    xy : ndarray
        Coordinates for measuring distance.
    adj : scipy.sparse.csr_matrix
    tmax : int, np.inf
        Max number of steps to take before stopping.
    rng : np.random.RandomState
    return_radius : bool, True
    fast : bool, False
    xy0 : int, None
        
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
    path.append(xy0 or rng.randint(len(xy)))
    xy0 = xy[path[0]]  # for avoiding adding element access time in loop
    absorbing_sites = set(absorbing_sites)
    
    indices = adj.indices.tolist()
    indptr = adj.indptr.tolist()
    lenindptr = len(indptr)
    
    i = 0
    while i<tmax and not (xy[path[-1]] in absorbing_sites):
        if path[-1]<lenindptr:
            path.append(rng.choice(indices[indptr[path[-1]]:indptr[path[-1]+1]]))
        else:
            path.append(rng.choice(indices[indptr[path[-1]]:]))
        i += 1
    
    if return_radius:
        d = np.linalg.norm(np.array(xy)[path]-xy0, axis=1)
        return np.array(path), d, np.maximum.accumulate(d)
    return np.array(path)

def saw(xy, adj, tmax, rng=np.random, fast=False):
    """Self-avoiding random walk.
    
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
            p = rng.choice(indices[indptr[path[-1]]:indptr[path[-1]+1]])
        else:
            p = rng.choice(indices[indptr[path[-1]]:])

        if data[p] and not p in path:
            path.append(p)
    
    d = np.linalg.norm(xy[path]-xy0, axis=1)
    return np.array(path), d, np.maximum.accumulate(d)

def _check_adj(adj):
    """Helper function for checking adjacency matrices.
    
    Checks that matrix is of type csr_matrix, matrix is square, nonzero elements are all
    1, that matrix symmetric, and that diagonal elements are 0.
    """

    assert type(adj)==csr_matrix
    assert adj.shape[0]==adj.shape[1]
    assert ((adj.data==0)|(adj.data==1)).all()
    assert (adj-adj.transpose()).count_nonzero()==0
    assert (adj.diagonal()==0).all()
