import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.patches import Ellipse, Circle
# https://matplotlib.org/api/axes_api.html?

def plotter(x,y,w,scattersize,title):
    fig,ax = plt.subplots()
    a = ax.scatter(x,y,c=w,s=scattersize)
    ax.axis('equal')
    plt.colorbar(a)
    ax.set_title(f'{title}')
    
def plotterrainbow(x,y,w,scattersize,title, ax):
    # fig,ax = plt.subplots()
    a = ax.scatter(x,y,c=w,s=scattersize,cmap='rainbow')
    ax.axis('equal')
    plt.colorbar(a)
    ax.set_title(f'{title}')

def get_quiver(x_,y_,u_,v_, ax):
    """Generate quiver plot

    Args:
        x_ (1d array): x-coordinate
        y_ (1d array): y-coordinate
        u_ (1d array): u-velocity
        v_ (1d array): v-velocity
    """
    # fig,ax = plt.subplots()
    ax.quiver(x_,y_,u_,v_)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_xlim(min(x_), max(x_))
    ax.set_ylim(min(y_), max(y_))
    ax.set_aspect('equal')
    # ax.add_artist(Circle((5, 5), 1/2, color='k'))

def get_tricontour(x_,y_,psi_, lvl, fig, ax):
    """plot tricontour

    Args:
        x_ (1d np array): x-coordinates
        y_ (1d np array): y-coordinates
        psi_ (1d np array): values
        lvl (int or array): determines the number and positions of the contour lines / regions.
    """
    # fig,ax = plt.subplots()
    ax.set_aspect('equal')
    tcf = ax.tricontourf(x_,y_,psi_, levels=lvl)
    fig.colorbar(tcf)
    tcf = ax.tricontour(x_,y_,psi_, levels=lvl, colors='k')
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_xlim(min(x_), max(x_))
    ax.set_ylim(min(y_), max(y_))

def get_contour(x_2d,y_2d,psi_2d, lvl=None):
    """plot contour

    Args:
        x_ (2d np array): x-coordinates (meshgrid type)
        y_ (2d np array): y-coordinates (meshgrid type)
        psi_2d (2d np array): values
        lvl (int or array): determines the number and positions of the contour lines / regions.
    """
    fig,ax = plt.subplots()
    ax.contour(x_2d, y_2d, psi_2d, levels=lvl)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_xlim(min(x_2d.ravel()), max(x_2d.ravel()))
    ax.set_ylim(min(y_2d.ravel()), max(y_2d.ravel()))
    ax.set_aspect('equal')

def get_plot(x_, y_, xcompare_=[], ycompare_=[], xlabel='variable', ylabel='variable'):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(x_, y_)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlim(min(x_), max(x_))
    ax.set_ylim(min(y_), max(y_)*1.05)
    ax.plot(xcompare_, ycompare_,'xk')

def get_neigh_plot(ax, check_indices, nodes_2d, neighbors_2d, n_neigh_, brinkman):
    """Plotting neighbors of the particles in check_indices.

    Args:
        ax (axes object): axes from the (import geometry) section
        check_indices (list): list
        nodes_2d (np 2d array): coordinates x|y
        neighbors_2d (2d np array): indices of neighbors of every particles 
        n_neigh_ (1d np array): number of neighbors of each particle
    """
    if brinkman:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for idx in check_indices:
        ax.scatter(nodes_2d[idx,0], nodes_2d[idx,1], nodes_2d[idx,2], "o")
        # ax.scatter(nodes_2d[neighbors_2d[idx,0:n_neigh_[idx]],0], nodes_2d[neighbors_2d[idx,0:n_neigh_[idx]],1], nodes_2d[neighbors_2d[idx,0:n_neigh_[idx]],2], s=40 , alpha=0.5, marker="X")
        ax.scatter(nodes_2d[neighbors_2d[idx],0], nodes_2d[neighbors_2d[idx],1], nodes_2d[neighbors_2d[idx],2], s=40 , alpha=0.5, marker="X")
    
    fig1, ax1 = plt.subplots()
    ax1.plot( n_neigh_,"x")    

