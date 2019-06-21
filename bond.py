# ===================================================================================== #
# Bond percolation in various graphs.
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
import numpy as np
from scipy.sparse import coo_matrix

def simulate_brownian_walker(component, edges, L,
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
    edges : set
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


# ======= #
# Classes #
# ======= #
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
        """Identify all connected components.
        
        Returns
        -------
        list
            Coordinates of points in each connected component.
        """

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
        """Generate a random graph by sampling bonds independently.
        
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
