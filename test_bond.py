# ===================================================================================== #
# Bond percolation in various graphs.
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
from .bond import *


def test_Square2D():
    model = Square2D(10, .4, rng=np.random.RandomState(0))
    components = model.find_components()
    assert sum([len(i) for i in components])==100, sum([len(i) for i in components])
