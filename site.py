# ====================================================================================== #
# Site percolation in various graphs.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from scipy.sparse import coo_matrix,lil_matrix
from numba import njit
import multiprocess as mp


def simulate_brownian_walker(component, L,
                             n_samples=1,
                             max_steps=10000,
                             fcn_on_visited=lambda x: None,
                             rng=np.random):
    """Let a single Brownian walker percolate through component til all sites have been
    visited. At each step, the walker *always* takes a step. An alternative set of
    dynamics belonging to the same universality class is if walkers are permitted to
    attempt to move in a blocked direction and fail.
    
    Parameters
    ----------
    component : list of tuples
    L : int
        Width or height of system.
    n_samples : int, 1
    max_steps : int, 10000
    fcn_on_visited : function
        Acts on set of visited sites.
    rng : np.random.RandomState, np.random

    Returns
    -------
    ndarray (int)
    list
    ndarray (bool)
    """
    
    unzippedComp = list(zip(*component))
    assert max(unzippedComp[0])<L and max(unzippedComp[1])<L
    componentSet = set(component)
    
    # build adjacency matrix where each site can have max four neighbors
    adj = lil_matrix((len(component),len(component)), dtype=np.uint8)
    for i,c in enumerate(component):
        # going clockwise from right
        if ((c[0]+1)%L,c[1]) in componentSet:
            adj[i,component.index(((c[0]+1)%L,c[1]))] = 1
        if (c[0],(c[1]+1)%L) in componentSet:
            adj[i,component.index((c[0],(c[1]+1)%L))] = 1
        if ((c[0]-1)%L,c[1]) in componentSet:
            adj[i,component.index(((c[0]-1)%L,c[1]))] = 1
        if (c[0],(c[1]-1)%L) in componentSet:
            adj[i,component.index((c[0],(c[1]-1)%L))] = 1

    def single_loop():
        counter = 0
        pos = rng.randint(len(component))
        visited = set((pos,))
        while (counter<max_steps) and len(visited)<len(component):
            pos = rng.choice(adj.rows[pos])
            visited.add(pos)
            counter += 1
        found = len(component)==len(visited)  # did we find all the sites?
        return counter, found, fcn_on_visited(visited)
        
    diffusionTime = np.zeros(n_samples, dtype=np.uint)

    success = np.zeros(n_samples, dtype=bool)
    visitedProcessing = []
    for i in range(n_samples):
        diffusionTime[i], success[i], v = single_loop()
        visitedProcessing.append(v)
        
    return diffusionTime, visitedProcessing, success

def _simulate_brownian_walker_with_length_cutoff(component, L,
                             n_samples=1,
                             max_steps=10000,
                             rng=np.random):
    """Let a single Brownian walker percolate through component til the percolation cluster has been spanned
    along longest axis.
    
    Parameters
    ----------
    component : list of tuples
    L : int
        Width or height of system.
    n_samples : int, 1
    max_steps : int, 10000
    rng : np.random.RandomState, np.random

    Returns
    -------
    ndarray (int)
    ndarray (bool)
    """
    
    from misc.utils import convex_hull, max_dist_pair2D
    unzippedComp = list(zip(*component))
    assert max(unzippedComp[0])<L and max(unzippedComp[1])<L
    componentSet = set(component)
    maxdistix = max_dist_pair2D(component)
    maxdist = np.sqrt((component[maxdistix[0]][0]-component[maxdistix[1]][0])**2 +
                      (component[maxdistix[0]][1]-component[maxdistix[1]][1])**2)
    
    # build adjacency matrix where each site can have max four neighbors
    adj = lil_matrix((len(component),len(component)), dtype=np.uint8)
    for i,c in enumerate(component):
        # going clockwise from right
        if ((c[0]+1)%L,c[1]) in componentSet:
            adj[i,component.index(((c[0]+1)%L,c[1]))] = 1
        if (c[0],(c[1]+1)%L) in componentSet:
            adj[i,component.index((c[0],(c[1]+1)%L))] = 1
        if ((c[0]-1)%L,c[1]) in componentSet:
            adj[i,component.index(((c[0]-1)%L,c[1]))] = 1
        if (c[0],(c[1]-1)%L) in componentSet:
            adj[i,component.index((c[0],(c[1]-1)%L))] = 1
    
    def single_loop():
        """Walker walks til conditions are violated."""
        counter = 0
        pos = rng.randint(len(component))
        visited = set((pos,))
        dist = 0
        chull = []  # convex hull
        while (counter<max_steps) and len(visited)<len(component) and dist<maxdist:
            pos = rng.choice(adj.rows[pos])
            
            if not pos in visited:
                # any new sites visited will belong to the current convex hull but some other points that are
                # currently in the hull might need to be removed
                # then, it is quick to calculate the largest separation because we know that the max pairwise
                # distance will be between points on the hull
                chull.append(pos)
                chull = [chull[i] for i in convex_hull(np.vstack([component[i] for i in chull]))]
                if len(chull)>2:
                    dist = max_dist_pair2D(np.vstack([component[i] for i in chull]),
                                           force_slow=True,
                                           return_dist=True)[-1]

                visited.add(pos)
            counter += 1
        found = len(component)==len(visited)
        return counter, found
        
    diffusionTime = np.zeros(n_samples, dtype=np.uint)
    success = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        diffusionTime[i], success[i] = single_loop()
        
    return diffusionTime, success



# ======= #
# Classes #
# ======= #
class Bethe():
    def __init__(self, z, p, rng=np.random):
        """
        Parameters
        ----------
        z : int
        p : float
        rng : np.random.RandomState, np.random
        """

        self.z = z
        self.p = p
        self.rng = rng

    def generate_clusters(self):
        """Generate a random cluster that lives on a Bethe lattice seeding on a single
        site.
        """
        
        z = self.z
        p = self.p
        rng = self.rng
        nodes = []
        edges = []

        nodesToParse = [0]
        nodeCounter = 0
        while nodesToParse:
            nNeighbors = (rng.rand(z)<p).sum()
            nodes.append(nodesToParse.pop(0))
            if nNeighbors:
                for i in range(nNeighbors):
                    nodeCounter += 1
                    edges.append((nodes[-1],nodeCounter))
                    nodesToParse.append(nodeCounter)
        return nodes, edges

    def generate_family_tree(self):
        """Generate a random cluster that lives on a Bethe lattice seeding on a single
        site and keep track of root history.

        Returns
        -------
        list of str
            Each string recounts the lineage of the node. The length of the string is the
            generation.
        """
        
        nodes = []
        parents = ['0']
        while parents:
            children = self._produce_children(parents)
            nodes += parents
            parents = children
        return nodes

    def _produce_children(self, parents):
        """
        Parameters
        ----------
        parents : list of str

        Returns
        -------
        list of str
            children
        """
        
        children = []
        for p in parents:
            n = self.rng.binomial(self.z, self.p)
            for i in range(n):
                children.append(p+str(i))
        return children

    def sample_family_tree(self, n_samples):
        """
        Parameters
        ----------
        n_samples : int

        Returns
        -------
        list of lists of str
            As return from self.generate_family_tree().
        """
        
        avalanches = []
        for i in range(n_samples):
            avalanches.append(self.generate_family_tree())
        return avalanches

    def sample(self, n_iters):
        """Generate many samples of avalanches.

        Parameters
        ----------
        n_iters : int

        Returns
        -------
        list
            Size of avalanches. 
        """

        s = []
        for i in range(n_iters):
            s.append(len(self.generate_clusters()[0]))
        return s
#end Bethe


class Square2D():
    """2D periodic square lattice."""
    def __init__(self, N, p, rng=np.random):
        """
        Parameters
        ----------
        N : int
            Width or height.
        p : float
            Percolation probability.
        rng : np.random.RandomState
        """

        self.N = N
        self.p = p
        self.rng = rng
        self._reset_lattice()

    def _reset_lattice(self):
        self.lattice = self.rng.rand(self.N,self.N)<=self.p

    def find_components(self):
        components = []
        toSearch = list(zip(*np.where(self.lattice)))
        searched = set()

        while toSearch:
            # seed cluster on lattice site that is active
            thisComponent = [toSearch.pop(0)]
            searched.add(thisComponent[0])

            # now iteratively search through neighbors of neighbors while there are more to look at
            # don't have to ignore elements in searched because if any neighbor was already clustered
            # then this point should have been already clustered
            thisSearch = [ne for ne in square_lattice_neighbors(thisComponent[0][0], thisComponent[0][1], self.N)
                          if self.lattice[ne[0],ne[1]]]
            while thisSearch:
                ij = thisSearch.pop(0)
                thisComponent.append(ij)
                searched.add(toSearch.pop(toSearch.index(ij)))
                # only find viable neighbors that haven't been searched yet or are listed to be searched
                thisSearch += [ne for ne in square_lattice_neighbors(ij[0],ij[1],self.N)
                               if self.lattice[ne[0],ne[1]] and (not ne in searched) and (not ne in thisSearch)]
            components.append(thisComponent)
        return components
#end Square2D

@njit
def square_lattice_neighbors(i,j,n):
    """Square 2D lattice."""
    return ((i+1)%n,j), ((i-1)%n,j), (i,(j+1)%n), (i,(j-1)%n)




# Code that remains to be adapted.
def square_lattice_forest(p, L, rng=np.random):
    """Generate a bond percolation cluster that lives on a 2D lattice seeding it from
    (0,0) and growing out.  This does not yield the standard percolation cluster size
    distribution.
    
    Explore each site and its neighbors (with probability p). Keep track of which bonds
    you have crossed.
    
    Parameters
    ----------
    p : float
        Probability of percolating through a bond.
    L : int
        Boundaries of system.
    rng : np.random.RandomState
    """
    
    nodes = set(((0,0),))
    openEdges = []
    edgesParsed = set()  # sites visited but not invaded
    nodesToParse = [(0,0)]
    
    def _test_node(xy0, xy):
        """origin, destination"""
        if not xy in nodes:
            # order the coordinates from bottom to top or left to right
            if xy0[1]==xy[1]:  # same y
                if xy0[0]<xy[0]:
                    edge = xy0, xy
                else:
                    edge = xy, xy0
            else:  # same x
                if xy0[1]<xy[1]:
                    edge = xy0, xy
                else:
                    edge = xy, xy0
            
            if not edge in edgesParsed:
                # each edge gets one chance of being opened with probability p
                edgesParsed.add(edge)
                if rng.rand()<p:
                    openEdges.append(edge)
                    nodes.add(xy0)
                    nodes.add(xy)
                    
                    # you could get rid of the one corresponding to xy0
                    nodesToParse.append(xy)
    
    def test_node(xy):
        # iterate through all four neighbors of xy
        _test_node(xy, ((xy[0]+1)%L, xy[1]))
        _test_node(xy, ((xy[0]-1)%L, xy[1]))
        _test_node(xy, (xy[0], (xy[1]+1)%L))
        _test_node(xy, (xy[0], (xy[1]-1)%L))
        
    while nodesToParse:
        test_node(nodesToParse.pop(0))
    return list(nodes), openEdges

def sample(p, L, n_iters=1000):
    return list(zip(*[square_lattice_forest(p, L) for i in range(n_iters)]))
