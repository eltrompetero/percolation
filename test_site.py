# ===================================================================================== #
# Site percolation in various graphs.
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
from .site import *


def test_Bethe():
    z = 4  # coordination number
    p = .999/z  # site occupation probability

    model = Bethe(z, p, rng=np.random.RandomState(1))
    nodes, edges = model.generate_clusters()

    assert len(np.unique(nodes))==len(nodes)==len(np.unique(concatenate(edges)))
    print("Test passed: nodes that appear in nodes and edges list are consistent.")

    assert adj.sum(1).max()<=z
    print("Test passed: each node has at max z activated neighbors.")

def test_Square2D():
    model = Square2D(20, .59)
    components = model.find_components()
    sizes = [len(c) for c in components]

    # each site appears once
    assert len(unique(concatenate(components),axis=0))==sum([len(i) for i in components])
