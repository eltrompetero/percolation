# ===================================================================================== #
# Site percolation in various graphs.
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
import numpy as np
from scipy.sparse import coo_matrix
from numba import njit


class Bethe():
    def __init__(self, z, p, rng=np.random):
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

    def sample(self, n_iters):
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
