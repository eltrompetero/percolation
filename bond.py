# ===================================================================================== #
# Bond percolation in various graphs.
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
import numpy as np
from scipy.sparse import coo_matrix


class Square2D():
    def __init__(self, L, p, rng=np.random):
        """Generate a bond percolation cluster that lives on a 2D lattice and collect all clusters.
        
        Parameters
        ----------
        p : float
            Probability of percolating through a bond.
        L : int
            Boundaries of system.
        rng : np.random.RandomState
        """

        self.L = L
        self.p = p
        self.rng = rng

        # sample from all possible edges
        self.edges = self.random_graph()

    def random_graph(self):
        # sample from all possible edges
        edges = set()
        for i in range(self.L):
            for j in range(self.L-1):
                if self.rng.rand()<self.p:
                    edges.add(((j,i),(j+1,i)))
                if self.rng.rand()<self.p:
                    edges.add(((i,j),(i,j+1)))
            if self.rng.rand()<self.p:
                edges.add(((0,i),(self.L-1,i)))
            if self.rng.rand()<self.p:
                edges.add(((i,0),(i,self.L-1)))
        return edges
        
    def find_components(self):
        """Identify all connected components."""

        L = self.L
        p = self.p
        components = []
        remainingNodes = set([(i,j) for i in range(L) for j in range(L)])  # all nodes

        def _test_node_01(xy0, xy, nodesToParse):
            """Are origin and destination connected by an edge?"""
            if xy in remainingNodes and (xy0,xy) in self.edges:
                nodesToParse.append(xy)
                remainingNodes.discard(xy)

        def _test_node_10(xy, xy0, nodesToParse):
            """Are origin and destination connected by an edge?"""
            if xy in remainingNodes and (xy,xy0) in self.edges:
                nodesToParse.append(xy)
                remainingNodes.discard(xy)

        while remainingNodes:
            thisComponent = []
            nodesToParse = [remainingNodes.pop()]
            
            while nodesToParse:
                xy = nodesToParse.pop(0)
                thisComponent.append( xy )
                _test_node_01(xy, ((xy[0]+1)%L, xy[1]), nodesToParse)
                _test_node_10(((xy[0]-1)%L, xy[1]), xy, nodesToParse)
                _test_node_01(xy, (xy[0], (xy[1]+1)%L), nodesToParse)
                _test_node_10((xy[0], (xy[1]-1)%L), xy, nodesToParse)
                
            components.append(thisComponent)
        return components

    def sample(self, n_iters):
        """
        Parameters
        ----------
        n_iters : int

        Returns
        -------
        list
            components
        list
           edges
        """
          
        components = []
        edges = []
        for i in range(n_iters):
            self.edges = self.random_graph()
            edges.append(self.edges)
            components.append(self.find_components())
        return components, edges
#end Square2D
