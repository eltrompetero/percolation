# ===================================================================================== #
# For visualizing percolation.
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def edges(edges, L, fig=None, ax=None, fig_kw={'figsize':(8,8)}, plot_kw={}):
    """
    Parameters
    ----------
    edges : iterable
    """

    if fig is None and ax is None:
        fig = plt.figure(**fig_kw)
        ax = fig.add_subplot(1,1,1, aspect='equal')

    # draw edges
    for edge_ in edges:
        if abs(edge_[0][0]-edge_[1][0])<=1 and abs(edge_[0][1]-edge_[1][1])<=1:
            l = mlines.Line2D([edge_[0][0],edge_[1][0]], [edge_[0][1],edge_[1][1]])
            ax.add_line(l)

    ax.set(xlim=(-1,L), ylim=(-1,L))
    return fig, ax
