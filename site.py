# ====================================================================================== #
# Site percolation in various graphs.
# Author: Eddie Lee, edl56@cornell.edu
# ====================================================================================== #
import numpy as np
from scipy.sparse import coo_matrix,lil_matrix
from numba import njit
import multiprocess as mp
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt


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

    def generate_family_tree(self, max_size=1_000_000):
        """Generate a random cluster that lives on a Bethe lattice seeding on a single
        site and keep track of root history.
        
        Parameters
        ----------
        max_size : int, 1_000_000

        Returns
        -------
        list of str
            Each string recounts the lineage of the node. The length of the string is the
            generation.
        """
        
        nodes = []
        parents = ['0']
        while parents and len(nodes)<=max_size:
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

    def sample_family_tree(self, n_samples, **kwargs):
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
            avalanches.append(self.generate_family_tree(**kwargs))
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


class Random2DBox():
    def __init__(self, L,
                 radius=1.,
                 density=1.,
                 rng=None):
        """Percolation cluster. This has distribution exponent tau of 3/2 when density is
        low and 1 when density is high.
        
	By uniformly generating points within a box, count all connected clusters
        consisting of points that are within distance r of each other.
        
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
        """Generate random and uniform set points for every single box in the grid. Select
        the number of points using a Poisson distribution then randomly place that number
        into the box.

        These are saved into self.allPointsByBox which will be clustered and clusters
        saved into self.cPointsByBox.
        """
        
        # initial set of random points centered around origin
        # boxes are labeled by their bottom left point
        allPointsByBox = {}  # labeled by bottom left corner
        for x in range(-self.L,self.L):
            for y in range(-self.L,self.L):
                xy = self.rng.rand(self.rng.poisson(self.density),2)
                xy[:,0] += x
                xy[:,1] += y
                allPointsByBox[(x,y)] = xy

        self.allPointsByBox = allPointsByBox
        self.cPointsByBox = {}

    def find_shared_neighbors(self, xy1, xy2, return_ix=False):
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
        ix = (d<=self.r).any(0)
        if return_ix:
            return xy2[ix], np.where(ix)[0]
        return xy2[ix]

    def _compare_with_one_box(self, thisbx, neighborbx):
        """Wrapper for comparing this box with a particular neighboring box and updating
        all the variables in the loop.

        Parameters
        ----------
        thisbx : tuple
        neighborbx : tuple

        Returns
        -------
        None
        """
        
        if not -self.L<=neighborbx[0]<self.L:
            return
        if not -self.L<=neighborbx[1]<self.L:
            return
        allPointsByBox = self.allPointsByBox
        cPointsByBox = self.cPointsByBox
        
        if neighborbx in allPointsByBox.keys():
            xy = self.find_shared_neighbors(cPointsByBox[thisbx], allPointsByBox[neighborbx])
            if len(xy):
                if not neighborbx in self.boxesConsidered:
                    self.boxesToConsider.append(neighborbx)

                    # check if there is a connected component is this box alone in which case 
                    # points from this box that are neighbors of each other must be added
                    cPointsByBox[neighborbx] = self.find_shared_neighbors(xy, allPointsByBox[neighborbx])
                else:
                    # in case there are clusters that are not connected within a single box but by virtue of a
                    # link extending thru neighboring boxes
                    cPointsByBox[neighborbx] = np.unique(np.append(cPointsByBox[neighborbx], xy, axis=0), axis=0)
        
    def grow(self, bx, ix, max_steps=10_000):
        """Grow cluster from given starting point.
        
        Parameters
        ----------
        bx : tuple
            (x,y) position of box in which starting point is to be found.
        ix : int
            ixth point the list of points in bx.
        max_steps : int, 10_000
        """
        
        allPointsByBox = self.allPointsByBox
        self.cPointsByBox = {bx:allPointsByBox[bx][ix][None,:]}
        cPointsByBox = self.cPointsByBox
        
        # first set of points are all ones that are within r of the origin and all the points in 
        # those boxes that form part of the connected cluster
        for k,xy in cPointsByBox.items():
            cPointsByBox[k] = self.find_shared_neighbors(xy, allPointsByBox[k])
        
        counter = 0
        self.boxesToConsider = list(cPointsByBox.keys())
        self.boxesConsidered = set()
        while self.boxesToConsider and counter<max_steps:
            bx = self.boxesToConsider.pop(0)
            if not bx in self.boxesConsidered:
                self.boxesConsidered.add(bx)

                # search all neighbors if there are points in this box
                if allPointsByBox[bx].size:
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
        
    def find_all_clusters(self):
        """Find all connected clusters of points by searching through all points and their neighbors.
        """
        
        clusters = []
        
        # create directory of all points (using a set for quick search)
        remainingpts = set()
        for x in range(-self.L, self.L):
            for y in range(-self.L, self.L):
                if (x,y) in self.allPointsByBox.keys():
                    for i in range(len(self.allPointsByBox[(x,y)])):
                        remainingpts.add(((x,y),i))
        
        while remainingpts:
            self.grow(*remainingpts.pop())
            
            # remove all explored points in this cluster from remainingpts
            for k,xy in self.cPointsByBox.items():
                matchix = np.where((cdist(xy, self.allPointsByBox[k])<=1e-14).any(0))[0]
                for ix in matchix:
                    # one of these will fail because it's the original point that has already been removed
                    try:
                        remainingpts.remove((k,ix))
                    except KeyError:
                        pass
            clusters.append(self.cPointsByBox)
        return clusters
    
    def adj_matrix(self, cPointsByBox=None):
        """Return adjacency matrix for given cluster.
        
        Parameters
        ----------
        cPointsByBox : dict, None
        """
        
        if cPointsByBox is None:
            cPointsByBox = self.cPointsByBox
            
        # setup for quickly calculating distances only between points in neighboring boxes
        xy = np.vstack([i for i in cPointsByBox.values()])
        bx = np.vstack([[k]*len(val) for k,val in cPointsByBox.items()])
        bx = [tuple(i) for i in bx]
        ixdict = {}
        counter = 0
        for k in cPointsByBox.keys():
            ixdict[k] = np.zeros(len(cPointsByBox[k]))
            for i in range(len(cPointsByBox[k])):
                ixdict[k][i] = counter
                counter += 1
        
        n = len(xy)
        adj = lil_matrix((n,n), dtype=np.uint8)
        for i in range(len(xy)):
            # look within own box
            ix = self.find_shared_neighbors(xy[i][None,:], cPointsByBox[bx[i]], return_ix=True)[1]
            adj[i,ixdict[bx[i]][ix]] = 1
            # ignore self
            adj[i,i] = 0
            
            # iterate thru all existing neighboring boxes
            neighborbx = bx[i][0]-1, bx[i][1]
            if neighborbx in cPointsByBox.keys():
                ix = self.find_shared_neighbors(xy[i][None,:], cPointsByBox[neighborbx], return_ix=True)[1]
                adj[i,ixdict[neighborbx][ix]] = 1
                
            neighborbx = bx[i][0]-1, bx[i][1]-1
            if neighborbx in cPointsByBox.keys():
                ix = self.find_shared_neighbors(xy[i][None,:], cPointsByBox[neighborbx], return_ix=True)[1]
                adj[i,ixdict[neighborbx][ix]] = 1
                
            neighborbx = bx[i][0], bx[i][1]-1
            if neighborbx in cPointsByBox.keys():
                ix = self.find_shared_neighbors(xy[i][None,:], cPointsByBox[neighborbx], return_ix=True)[1]
                adj[i,ixdict[neighborbx][ix]] = 1
                
            neighborbx = bx[i][0]+1, bx[i][1]-1
            if neighborbx in cPointsByBox.keys():
                ix = self.find_shared_neighbors(xy[i][None,:], cPointsByBox[neighborbx], return_ix=True)[1]
                adj[i,ixdict[neighborbx][ix]] = 1
                
            neighborbx = bx[i][0]+1, bx[i][1]
            if neighborbx in cPointsByBox.keys():
                ix = self.find_shared_neighbors(xy[i][None,:], cPointsByBox[neighborbx], return_ix=True)[1]
                adj[i,ixdict[neighborbx][ix]] = 1
                
            neighborbx = bx[i][0]+1, bx[i][1]+1
            if neighborbx in cPointsByBox.keys():
                ix = self.find_shared_neighbors(xy[i][None,:], cPointsByBox[neighborbx], return_ix=True)[1]
                adj[i,ixdict[neighborbx][ix]] = 1
                
            neighborbx = bx[i][0], bx[i][1]+1
            if neighborbx in cPointsByBox.keys():
                ix = self.find_shared_neighbors(xy[i][None,:], cPointsByBox[neighborbx], return_ix=True)[1]
                adj[i,ixdict[neighborbx][ix]] = 1
                
            neighborbx = bx[i][0]-1, bx[i][1]+1
            if neighborbx in cPointsByBox.keys():
                ix = self.find_shared_neighbors(xy[i][None,:], cPointsByBox[neighborbx], return_ix=True)[1]
                adj[i,ixdict[neighborbx][ix]] = 1
        
        return adj
    
    def plot(self,
             fig=None,
             ax=None,
             figure_kw={'figsize':(5,5)},
             show_bounding_circles=False,
             cPointsByBox=None,
             adj=None):
        """
        Parameters
        ----------
        fig : mpl.Figure, None
        ax : mpl.Axes, None
        figure_kw : dict, {'figsize':(5,5)}
        show_bounding_circles : bool, False
            If True, circles draw around every single point not included in 
            cluster. This can be useful for debugging.
        cPointsByBox : dict, None
            Provide a cluster organized by dictionary of box coordinates in case you want 
            to plot a specific cluster.
        adj : ndarray, None
            If given, edges will be drawn.
        
        Returns
        -------
        mpl.Figure
        mpl.Axes
        """

        allPointsByBox = self.allPointsByBox
        if cPointsByBox is None:
            cPointsByBox = self.cPointsByBox
        
        # setup
        if fig is None:
            fig = plt.figure(**figure_kw)
            ax = fig.add_subplot(1,1,1,aspect='equal')
        elif ax is None:
            ax = fig.add_subplot(1,1,1,aspect='equal')

        # get max bounds
        xmx = max([abs(k[0]) for k in allPointsByBox.keys()])
        ymx = max([abs(k[1]) for k in allPointsByBox.keys()])

        # plot all points
        for xy in allPointsByBox.values():
            ax.plot(xy[:,0], xy[:,1], 'k.', zorder=1)

        # plot cluster points
        for xy in cPointsByBox.values():
            ax.plot(xy[:,0], xy[:,1], 'rx', zorder=2)
        ax.set(xlim=(-xmx-1,xmx+1), ylim=(-ymx-1,ymx+1),
               xlabel='x', ylabel='y')
        
        # plot edges
        if not adj is None:
            xy = np.vstack(cPointsByBox.values())
            assert adj.shape==(len(xy),len(xy))
            for i,row in enumerate(adj):
                for endptix in np.where(row==1)[0]:
                    ax.plot([xy[i][0], xy[endptix][0]], [xy[i][1], xy[endptix][1]], 
                            'k-', zorder=0, alpha=.2)

        # plot circles around points that are not included in cluster
        if show_bounding_circles:
            for k in allPointsByBox:
                if k in cPointsByBox:
                    plotix = np.where( (cdist(allPointsByBox[k], cPointsByBox[k])>1e-14).all(1) )[0]
                else:
                    plotix = range(len(allPointsByBox[k]))
                for ix in plotix:
                    xy = allPointsByBox[k][ix]
                    circle = plt.Circle(xy, 1, color=None, fill=None)
                    ax.add_artist(circle)
                    
        return fig, ax
    
    def sample_sizes(self, n_samples):
        sizes = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            self.initialize()
            self.grow()
            sizes[i] = sum([len(i) for i in self.cPointsByBox])            
        return sizes
#end Random2D



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
