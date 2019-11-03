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


# ======= #
# Classes #
# ======= #
class Backbone():
    def __init__(self, sites, p1, p2):
        assert p1 in sites and p2 in sites
        self.p1 = p1
        self.p2 = p2
        self.sites = sites

    def find_backbone(self):
        self.burn1()
        self.burn2()
        self.burn3()

    def burn1(self):
        """Burning algorithm step 1 as described in Herrman et al (1984).

        Speedup possible by keeping separate sets of sites in current shell and next shell.

        Returns
        -------
        list
            Time step at which site first burned.
        list of tuples
        """
        
        sitesAsSet = set(self.sites)
        counts = np.zeros(len(self.sites), dtype=int)  # no. of times each site is approached by fire
        time = np.zeros(len(self.sites), dtype=int)-1  # timestep at which site burned
        p2ix = self.sites.index(self.p2)
        loopSites = []  # self.sites that are first burnt by multiple neighbors in the same time step
        
        def check_one_site(xy):
            if xy in sitesAsSet:
                ix = self.sites.index(xy)
                counts[ix] += 1
                # if this site has not burned before, add it to list
                if counts[ix]==1:
                    newBurningSites.add(xy)
                # if this loop has been burned already in the same time step, it's a loop site
                elif xy in newBurningSites:
                    loopSites.append(xy)

        # Starting at self.p1, burn neighboring self.sites and keep track of how many
        # times each site tries to be burned in counts.
        burningSites = set([self.p1])
        counts[self.sites.index(self.p1)] += 1
        counter = 0
        while counts[p2ix]<1:
            shellSize = len(burningSites)
            newBurningSites = set()

            # iterate through only the neighbors of currently burning self.sites
            for i in range(shellSize):
                thisSite = burningSites.pop()
                time[self.sites.index(thisSite)] = counter

                xy = (thisSite[0]-1,thisSite[1])
                check_one_site(xy)
                
                xy = (thisSite[0]+1,thisSite[1])
                check_one_site(xy)

                xy = (thisSite[0],thisSite[1]-1)
                check_one_site(xy)

                xy = (thisSite[0],thisSite[1]+1)
                check_one_site(xy)
            burningSites = newBurningSites
            counter += 1
        
        # last shell
        shellSize = len(burningSites)

        for i in range(shellSize):
            thisSite = burningSites.pop()
            time[self.sites.index(thisSite)] = counter

            xy = (thisSite[0]-1,thisSite[1])
            check_one_site(xy)
            
            xy = (thisSite[0]+1,thisSite[1])
            check_one_site(xy)

            xy = (thisSite[0],thisSite[1]-1)
            check_one_site(xy)

            xy = (thisSite[0],thisSite[1]+1)
            check_one_site(xy)

        self.loopSites = loopSites
        self.time = time
        return time, loopSites

    def burn2(self):
        """
        Returns
        -------
        list
            Tuples indicating self.sites in elastic backbone.
        """
        
        sitesAsSet = set(self.sites)
        ebackbone = []
        counts = np.zeros(len(self.sites), dtype=int)  # no. of times each site is approached by fire
        burningSites = [self.p2]
        counter = self.time.max()

        def check_one_site(xy):
            if xy in sitesAsSet:
                ix = self.sites.index(xy)
                counts[ix] += 1
                # if this site burned earlier, add to list
                if self.time[ix]<thistime and counts[ix]<=1:
                    burningSites.append(xy)
        
        while counter:
            # iterate through only the neighbors of currently burning self.sites
            for i in range(len(burningSites)):
                thisSite = burningSites.pop(0)
                thistime = self.time[self.sites.index(thisSite)]
                ebackbone.append(thisSite)
    
                # for each neighbor check if that point should be burned
                xy = (thisSite[0]-1,thisSite[1])
                check_one_site(xy) 

                xy = (thisSite[0]+1,thisSite[1])
                check_one_site(xy) 
                
                xy = (thisSite[0],thisSite[1]-1)
                check_one_site(xy) 

                xy = (thisSite[0],thisSite[1]+1)
                check_one_site(xy) 
            counter -= 1

        self.ebackbone = ebackbone
        return ebackbone

    def burn3(self):
        """Iterate through each loop site that is connected to (or part of the backbone). Continue looping
        through any new loop self.sites that are connected to the backbone.
        
        Could give each site a fixed index instead having to search for index every
        time.
        
        Returns
        -------
        list of tuples
            backbone excluding elastic backbone
        """
        
        assert len(self.sites)==self.time.size
        
        sitesAsSet = set(self.sites)
        backbone = []  # growing backbone
        loopSites = self.loopSites[:]

        def check_one_site(xy, thistime):
            if xy in sitesAsSet and not xy in burnedSites:
                # if site is in elastic backbone, don't do anything, but note that it's been burnt
                if (xy in self.ebackbone) or (xy in backbone):
                    return xy
                ix = self.sites.index(xy)
                # if this site burned earlier, add to list
                if self.time[ix]<thistime:
                    burnedSites.add(xy)
                    burningSites.add(xy)
        
        counter = 0  # number of self.sites we've passed thru without having to add new self.sites
        while loopSites:
            burnedebackbone = []
            site = loopSites.pop(0)
            burningSites = set([site])
            burnedSites = set([site])  # keeping track of all self.sites that will be burned and have burned
            while burningSites:
                # loop thru one shell
                for j in range(len(burningSites)):
                    thisSite = burningSites.pop() 
                    thistime = self.time[self.sites.index(thisSite)]
                    
                    # check all neighbors and then light them if appropriate
                    xy = (thisSite[0]-1,thisSite[1])
                    burnedebackbone.append( check_one_site(xy, thistime) )

                    xy = (thisSite[0]+1,thisSite[1])
                    burnedebackbone.append( check_one_site(xy, thistime) )

                    xy = (thisSite[0],thisSite[1]-1)
                    burnedebackbone.append( check_one_site(xy, thistime) )

                    xy = (thisSite[0],thisSite[1]+1)
                    burnedebackbone.append( check_one_site(xy, thistime) )
            
            # if multiple burned sites are in backbone, 
            if len(set([i for i in burnedebackbone if not i is None]))>1:
                backbone.extend(burnedSites)

            counter += 1
        self.backbone = list(set(backbone))
        return backbone
#end Backbone
