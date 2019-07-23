# ===================================================================================== #
# For visualizing percolation.
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
from .utils import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


def edges(edges, L,
          fig=None,
          ax=None,
          fig_kw={'figsize':(8,8)},
          plot_kw={}):
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

def nodes(components, L,
          fig=None,
          ax=None,
          fig_kw={'figsize':(8,8)},
          marker='.',
          plot_kw={}):
    """Show each occupied site.
    """
    if fig is None and ax is None:
        fig = plt.figure(**fig_kw)
        ax = fig.add_subplot(1,1,1, aspect='equal')
    
    for c in components:
        c = np.vstack(c)
        ax.plot(c[:,0], c[:,1], marker, **plot_kw)

    ax.set(xlim=(-1,L), ylim=(-1,L))
    return fig, ax

def nodes_by_shell(siteShell,
                   fig=None,
                   ax=None,
                   fig_kw={'figsize':(8,8)},
                   marker='.',
                   cmap=plt.cm.hot,
                   plot_kw={}):
    """Show each occupied site by shell.

    Parameters
    ----------
    siteShell : dict
        Dictionary whose keys are the site and the value is the shell index. This is
        exactly what is returned by bond.Square2D.grow_cluster_by_shell().
    """

    if fig is None and ax is None:
        fig = plt.figure(**fig_kw)
        ax = fig.add_subplot(1,1,1, aspect='equal')
   
    shellsOfxy = invert_shell_by_site_dict(siteShell)

    for i,xy in enumerate(shellsOfxy.values()):
        ax.plot([i[0] for i in xy], [i[1] for i in xy], marker,
                c=cmap(i/len(shellsOfxy)),
                **plot_kw)
    # emphasize initial point
    ax.plot(shellsOfxy[0][0][0], shellsOfxy[0][0][1], 'k*', ms=15)

    ax.set(xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)
    #ax.set(xlim=(-1,L), ylim=(-1,L))
    return fig, ax
