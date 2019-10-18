# ====================================================================================== #
# Bond percolation in various graphs.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt


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
    def __init__(self, L, p, rng=np.random, sample=True):
        """Generate a bond percolation cluster that lives on a 2D lattice and collect all
        clusters.
        
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
        
        if sample: 
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
        
    def find_components(self, periodic=True):
        """Identify all connected components.
        
        Parameters
        ----------
        periodic : bool, True
            If True, identifies neighbors with periodic boundary conditions.
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
        
        if periodic:
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
        else:
            while remainingNodes:
                thisComponent = []
                nodesToParse = [remainingNodes.pop()]
                
                while nodesToParse:
                    xy = nodesToParse.pop(0)
                    thisComponent.append( xy )
                    _test_node_01(xy, (xy[0]+1, xy[1]), nodesToParse)
                    _test_node_10((xy[0]-1, xy[1]), xy, nodesToParse)
                    _test_node_01(xy, (xy[0], xy[1]+1), nodesToParse)
                    _test_node_10((xy[0], xy[1]-1), xy, nodesToParse)
                    
                components.append(thisComponent)

        return components

    def sample(self, n_iters):
        """Generate an entirely new random graph by sampling bonds independently.
        
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

    def grow_cluster(self, n_samples, tmax=np.inf, lmax=None, return_zeros=False):
        """Grow a percolation cluster starting from (0,0). This is pretty fast and can do
        1000x1000 without trouble.
        
        Parameters
        ----------
        n_samples : int
            Number of clusters to generate.
        tmax : int, inf
            Max number of steps.
        lmax : int, self.L
            Walls at x=+/-lmax and y=+/-lmax. Cluster growth ends when it hits a wall.
        return_zeros : bool, False
            If False, do not return a cluster that failed to grow.
        
        Returns
        -------
        list of lists
            All directed bonds (startxy, endxy) in the cluster.
        list of lists
            All (x,y) coordinates in the cluster.
        """
        
        lmax = lmax or self.L
        bonds = []
        sites = []
        for i in range(n_samples):
            s = []
            while not return_zeros and len(s)==0:
                b, s = self._grow_cluster(tmax, lmax)
            bonds.append(b)
            sites.append(s)
        return bonds, sites

    def grow_cluster_by_shell(self, n_samples,
                              tmax=np.inf,
                              lmax=None,
                              return_zeros=False,
                              min_shells=1):
        """See self.grow_cluster().
        
        Parameters
        ----------
        n_samples : int
        tmax : int, inf
        lmax : int, self.L
        return_zeros : bool, False
        
        Returns
        -------
        list of lists
            All directed bonds (startxy, endxy) in the cluster.
        list of lists
            All (x,y) coordinates in the cluster.
        list of dict
            Keys are sites and values are shell indices.
        """
        
        lmax = lmax or self.L
        bonds = []
        sites = []
        siteShells = []
        counter = 0
        while counter<n_samples:
            s = []
            while not return_zeros and len(s)==0:
                b, s, sh = self._grow_cluster_by_shell(tmax, lmax)
            if len(sh)>=min_shells:
                bonds.append(b)
                sites.append(s)
                siteShells.append(sh)
                counter += 1
        return bonds, sites, siteShells

    def _grow_cluster(self, tmax, lmax):
        """Grow a single cluster. All bonds are explored once with probability p. Growth
        stops when no more bonds can be explored."""

        visitedBonds = set()
        # tuples of start to end
        bondsToVisit = set(( ((0,0),(1,0)), ((0,0),(0,1)), ((0,0),(-1,0)), ((0,0),(0,-1)) ))
        clusterSites = set()
        clusterBonds = []
        thisSite = (0,0)
        
        counter = 0
        while (bondsToVisit and
               counter<tmax and
               abs(thisSite[0])<=lmax and
               abs(thisSite[1])<=lmax):
            thisBond = bondsToVisit.pop()

            if self.rng.rand()<self.p:
                clusterBonds.append(thisBond)
                # iterate through all potential neighbors of this bond unless this site has already been
                # visited
                thisSite = thisBond[1]
                if not thisSite in clusterSites:
                    clusterSites.add(thisSite)
                    potentialNewSites = ((thisBond[1][0]-1, thisBond[1][1]),
                                         (thisBond[1][0], thisBond[1][1]-1),
                                         (thisBond[1][0]+1, thisBond[1][1]),
                                         (thisBond[1][0], thisBond[1][1]+1))
                    # check if any of these bonds are allowed to be explored
                    for xy in potentialNewSites:
                        if (not xy==thisBond[0] and  # not the original site
                            not ((thisSite,xy) in visitedBonds or (xy,thisSite) in visitedBonds) and
                            not ((thisSite,xy) in bondsToVisit or (xy,thisSite) in bondsToVisit)):
                            bondsToVisit.add((thisSite,xy))
                counter += 1
        return clusterBonds, list(clusterSites)

    def _grow_cluster_by_shell(self, tmax, lmax):
        """Addition to self._grow_cluster() to include shells. This is included as an
        additional time index that is incremented with every bond and site. Thus this code
        include memory access and shell index incrementing time."""

        visitedBonds = set()
        # tuples of start to end
        bondsToVisit = set(( ((0,0),(1,0)), ((0,0),(0,1)), ((0,0),(-1,0)), ((0,0),(0,-1)) ))
        clusterSites = set(((0,0),))
        clusterBonds = []
        siteShell = {}  # stores the shell index for each site
        thisSite = (0,0)
        siteShell[thisSite] = 0
        
        counter = 0
        while (bondsToVisit and
               counter<tmax and
               abs(thisSite[0])<=lmax and
               abs(thisSite[1])<=lmax):
            thisBond = bondsToVisit.pop()

            if self.rng.rand()<self.p:
                clusterBonds.append(thisBond)
                # iterate through all potential neighbors of this bond unless this site has already been
                # visited
                thisSite = thisBond[1]
                if not thisSite in clusterSites:
                    clusterSites.add(thisSite)
                    siteShell[thisSite] = siteShell[thisBond[0]]+1
                    potentialNewSites = ((thisBond[1][0]-1, thisBond[1][1]),
                                         (thisBond[1][0], thisBond[1][1]-1),
                                         (thisBond[1][0]+1, thisBond[1][1]),
                                         (thisBond[1][0], thisBond[1][1]+1))
                    # check if any of these bonds are allowed to be explored
                    for xy in potentialNewSites:
                        if (not xy==thisBond[0] and  # not the original site
                            not ((thisSite,xy) in visitedBonds or (xy,thisSite) in visitedBonds) and
                            not ((thisSite,xy) in bondsToVisit or (xy,thisSite) in bondsToVisit)):
                            bondsToVisit.add((thisSite,xy))
                counter += 1
        return clusterBonds, list(clusterSites), siteShell

    def find_edge_sites(self, sites, radius=3, density=.5):
        """Find sites that do not have a certain number of neighbors within a fixed
        distances in every direction (a square centered about each point). This can be
        used as a heuristic to identify boundary points.

        Parameters
        ----------
        radius : int, 3
        density : float, .5

        Returns
        -------
        list of (x,y) twoples
        """
        
        sites = set(sites)
        thresholdNeighbors = ((radius*2+1)**2 - 1) * density
        edgeSites = []

        for xy in sites:
            # count number of neighbors of xy
            nNeighbors = 0
            for dx in range(-radius,radius+1):
                for dy in range(-radius,radius+1):
                    if (xy[0]+dx,xy[1]+dy) in sites:
                        nNeighbors += 1 
            if nNeighbors<thresholdNeighbors:
                edgeSites.append(xy)
        
        return edgeSites
    
    def _adj(self, sites):
        """More general algorithm for returning adjacency matrix for a given set of sites.
        
        Parameters
        ----------
        sites : dict

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        
        ix = []
        iy = []

        # look at every pair of sites
        for i in range(len(sites)-1):
            for j in range(i+1,len(sites)):
                if ((sites[i],sites[j]) in self.edges or
                    (sites[j],sites[i]) in self.edges):
                    ix.append(i)
                    iy.append(j)
        adj = coo_matrix((np.ones(2*len(ix)), (ix+iy,iy+ix)), shape=(len(sites),len(sites)))
        return adj.tocsr()

    def adj(self, xy):
        """Construct adjacency matrix from the coordinates of occupied sites on a square
        lattice.

        Parameters
        ----------
        xy : list of twoples

        Returns
        -------
        scipy.sparse.csr_matrix
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
                          dtype=np.uint8,
                          shape=(n,n)).tocsr()
#end Square2D


class RandomFixedRadius():
    """Random points in 2D connected when any two points are within radius.

    See "2019-07-11 random walk on random graph.ipynb" for examples.
    """
    def __init__(self, L,
                 radius=1.,
                 density=3/np.pi,
                 rng=None):
        """Generate a random connected cluster using a cutoff radius 1 starting from four 
        boxes around the origin.
        
        Parameters
        ----------
        L : int
            Max distance cluster is permitted to go in both x and y directions.
        radius : float, 1.
        density : float, 1.
            Number of points per unit area.
        rng : np.random.RandomState
        """

        assert L>1 and radius>0 and density>0
        self.L = L
        self.rng = rng or np.random
        self.r = radius
        self.density = density

    def initialize(self):
        """Generate points and cluster them for the four boxes centered about the origin.
        """
        
        # initial set of random points centered around origin
        # boxes are labeled by their bottom left point
        allPointsByBox = {(-1,0):[], (0,0):[], (-1,-1):[], (0,-1):[]}  # labeled by bottom left corner
        while all([len(i)==0 for i in allPointsByBox.values()]):
            for b in allPointsByBox.keys():
                xy = self.rng.rand(self.rng.poisson(self.density),2)
                xy[:,0] += b[0]
                xy[:,1] += b[1]
                allPointsByBox[b] = xy

        # points that form part of cluster only
        # first set of points are all ones that are within r of the origin and all the points in 
        # those boxes that form part of the connected cluster
        cPointsByBox = {}
        for k in allPointsByBox:
            withinRadiusIx = np.linalg.norm(allPointsByBox[k], axis=1)<=1
            cPointsByBox[k] = allPointsByBox[k][withinRadiusIx]
        for k,xy in cPointsByBox.items():
            cPointsByBox[k] = self.find_shared_neighbors(xy, allPointsByBox[k])
        
        self.allPointsByBox = allPointsByBox
        self.cPointsByBox = cPointsByBox
        
    def populate_box(self, x, y):
        """Populate box with a uniformly random sample."""

        xy = self.rng.rand(self.rng.poisson(self.density),2)
        xy[:,0] += x
        xy[:,1] += y
        self.allPointsByBox[(x,y)] = xy

    def find_shared_neighbors(self, xy1, xy2):
        """Given points in box1 and box2, return points in box2 that are neighbors of points 
        in box1 within given distance threshold.

        Parameters
        ----------
        xy1 : ndarray
        xy2 : ndarray
        r : float, 1

        Returns
        -------
        ndarray
            Set of points in xy2.
        """

        d = cdist( xy1, xy2 )
        return xy2[(d<=self.r).any(0)]

    def _compare_with_one_box(self, thisbx, neighborbx):
        """Wrapper for comparing this box with a particular neighboring box and updating
        all the variables in the loop.

        If neighboring box is not in the record, it is populated.

        Parameters
        ----------
        thisbx : tuple
        neighborbx : tuple

        Returns
        -------
        None
        """
        
        if abs(neighborbx[0])>self.L:
            if not self.hitx:
                print("Hit x boundary.")
                self.hitx = True
            return
        if abs(neighborbx[1])>self.L:
            if not self.hity:
                print("Hit y boundary.")
                self.hity = True
            return
        allPointsByBox = self.allPointsByBox
        cPointsByBox = self.cPointsByBox
        
        # add neighboring box to record if it hasn't been added
        if not neighborbx in allPointsByBox.keys():
            self.populate_box(*neighborbx)

        # get all first order neighbors from neighboring box
        xy = self.find_shared_neighbors(cPointsByBox[thisbx], allPointsByBox[neighborbx])

        # if we have any neighbors...
        if len(xy):
            # add any points in the neighboring box building from the connected cluster with found first order
            # neighbors 
            el = len(xy)
            xy = self.find_shared_neighbors(xy, allPointsByBox[neighborbx])
            while len(xy)>el:
                el = len(xy)
                xy = self.find_shared_neighbors(xy, allPointsByBox[neighborbx])
            if not neighborbx in self.boxesConsidered:
                self.boxesToConsider.append(neighborbx)
                cPointsByBox[neighborbx] = xy
            else:
                cPointsByBox[neighborbx] = np.unique(np.append(cPointsByBox[neighborbx], xy, axis=0), axis=0)
        
    def grow(self, max_steps=10_000):
        """Grow cluster from origin.
        
        For each box we consider, we add all pieces that form a connected component (given
        the cluster points we have already identified. Then, we iterate through all
        neighbors, again building connected components. Since each box is checked as many
        times as it has neighbors, we should capture all points that are part of the
        connected cluster even when they are disconnected within a box."""
        
        self.hitx = False
        self.hity = False
        allPointsByBox = self.allPointsByBox
        cPointsByBox = self.cPointsByBox
        
        counter = 0
        self.boxesToConsider = list(cPointsByBox.keys())
        self.boxesConsidered = set()
        while self.boxesToConsider and counter<max_steps:
            bx = self.boxesToConsider.pop(0)
            if not bx in self.boxesConsidered:
                self.boxesConsidered.add(bx)

                # search all neighbors if there are points in this box
                if allPointsByBox[bx].size:
                    # first, get connected component in this box since we may have added more components from
                    # searching neighbors
                    xy = self.cPointsByBox[bx]
                    el = len(xy)
                    xy = self.find_shared_neighbors(xy, self.allPointsByBox[bx])
                    while len(xy)>el:
                        el = len(xy)
                        xy = self.find_shared_neighbors(xy, self.allPointsByBox[bx])
                    cPointsByBox[bx] = np.unique(np.append(cPointsByBox[bx], xy, axis=0), axis=0)
                
                    # look for neighboring points in all surrounding neighboring boxes
                    self._compare_with_one_box(bx, (bx[0]-1, bx[1]))
                    self._compare_with_one_box(bx, (bx[0]-1, bx[1]-1))
                    self._compare_with_one_box(bx, (bx[0], bx[1]-1))
                    self._compare_with_one_box(bx, (bx[0]+1, bx[1]-1))
                    self._compare_with_one_box(bx, (bx[0]+1, bx[1]))
                    self._compare_with_one_box(bx, (bx[0]+1, bx[1]+1))
                    self._compare_with_one_box(bx, (bx[0], bx[1]+1))
                    self._compare_with_one_box(bx, (bx[0]-1, bx[1]+1))
                counter += 1
        if counter==max_steps:
            print("Max steps reached.")
            
    def plot(self,
             fig=None,
             ax=None,
             figure_kw={'figsize':(5,5)},
             show_bounding_circles=False):
        """
        Parameters
        ----------
        fig : mpl.Figure, None
        ax : mpl.Axes, None
        figure_kw : dict, {'figsize':(5,5)}
        show_bounding_circles : bool, False
            If True, circles draw around every single point not included in 
            cluster. This can be useful for debugging.
            
        Returns
        -------
        mpl.Figure
        mpl.Axes
        """

        allPointsByBox = self.allPointsByBox
        cPointsByBox = self.cPointsByBox
        
        # setup
        if fig is None:
            fig = plt.figure(**figure_kw)
            ax = fig.add_subplot(1,1,1,aspect='equal')
        elif ax is None:
            ax = fig.add_subplot(1,1,1,aspect='equal')

        # get max bounds
        xmx = max([abs(k[0]) for k in cPointsByBox.keys()])
        ymx = max([abs(k[1]) for k in cPointsByBox.keys()])

        # plot all points
        for xy in allPointsByBox.values():
            ax.plot(xy[:,0], xy[:,1], 'k.')

        # plot cluster points
        for xy in cPointsByBox.values():
            ax.plot(xy[:,0], xy[:,1], 'rx')
        ax.set(xlim=(-xmx-1,xmx+2), ylim=(-ymx-1,ymx+2),
               xlabel='x', ylabel='y')

        # plot origin circle
        circle = plt.Circle((0, 0), 1, color=None, fill=None, linestyle='--')
        ax.add_artist(circle)

        # plot circles around points that are not included in cluster
        if show_bounding_circles:
            for k in allPointsByBox:
                # ignore points that overlap
                if k in cPointsByBox:
                    plotix = np.where( (cdist(allPointsByBox[k], cPointsByBox[k])>1e-14).all(1) )[0]
                else:
                    plotix = range(len(allPointsByBox[k]))
                for ix in plotix:
                    xy = allPointsByBox[k][ix]
                    circle = plt.Circle(xy, 1, color=None, fill=None)
                    ax.add_artist(circle)
                    
        return fig, ax
    
    def sample_sizes(self, n_samples, iprint_every=False, grow_kw={}):
        sizes = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            self.initialize()
            self.grow(**grow_kw)
            sizes[i] = sum([len(i) for i in self.cPointsByBox])
            if iprint_every and (i%iprint_every)==0:
                print("Done with sample %d."%i)
        return sizes
    
    def n_points(self):
        """Size of cluster.
        """
        return sum([len(i) for i in self.cPointsByBox])
#end RandomFixedRadius
