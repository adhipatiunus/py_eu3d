import numpy as np
from scipy import sparse
from numpy import linspace, meshgrid
from scipy.interpolate import griddata
from scipy.sparse import linalg
import plotter  
from matplotlib import pyplot as plt

def get_grid_interp(x, y, z, resX=100, resY=100):
    """Interpolate 3 column data to matplotlib grid

    Args:
        x (1d np.array): x-
        y (1d np.array): y-
        z (1d np array): values
        resX (int, optional): x-resolution. Defaults to 100.
        resY (int, optional): y-resolution. Defaults to 100.

    Returns:
        X,Y,Z: set of 2D arrays ready to be plotted into contour
    """
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z_2d = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
    X_2d, Y_2d = meshgrid(xi, yi)
    
    return X_2d, Y_2d, Z_2d

def get_streamfunction(u_, v_, rhs_psi_, psi_bound_2d, dx_2d, dy_2d, dxx_2d, dyy_2d):
    """Compute streamfunction

    Args:
        u_ (1d np array): u-velocity
        v_ (1d np array): v-velocity
        psi_bound_2d (2d sparse matrix): boundary operator for streamfunction
        rhs_psi_ (1d np array): rhs for boundary particles
        dx_2d (2d CSR sparse matrix): x-deriv operator (inner particles only)
        dy_2d (2d CSR sparse matrix): y-deriv operator (inner particles only)
        dxx_2d (2d sparse matrix): xx-derivative operator (inner particles only)
        dyy_2d (2d sparse matrix): yy-derivative operator (inner particles only)
    
    Returns:
        psi_ (1d np array): streamline vector
    """
    LHS_2d  = sparse.vstack([psi_bound_2d, dxx_2d + dyy_2d])
    rhs_    = -dx_2d.dot(v_) + dy_2d.dot(u_)
    rhs_    = np.concatenate((rhs_psi_, rhs_))
    
    psi_ = linalg.spsolve(LHS_2d, rhs_)

    return psi_

def get_profile(loc_str1, loc_val1, loc_str2, loc_val2, V_, nodes_2d, xcompare_=[], ycompare_=[]):
    """obtain profile at an intended location, sorting, and plotting

    Args:
        loc_str1 (str): 1st loc
        loc_val1 (float): 1st loc value
        loc_str2 (str): 2nd axis
        loc_val2 (float): 2nd axis value
        V_ (1d np.array): velocity
        nodes_2d (2d np.array): coordinate
    """
    axis = {"x":0, "y":1, "z":2}
    # find indices of the wanted location
    ind_ = np.logical_and(abs(nodes_2d[:,axis[loc_str1]] - loc_val1) < 1e-8, \
        abs(nodes_2d[:,axis[loc_str2]] - loc_val2) < 1e-8)
    
    # find the axis along which the profile is to be plotted                               
    if loc_str1 == "x" or loc_str2 == "x":
        if loc_str1 == "y" or loc_str2 == "y":
            third = "z"
        elif loc_str1 == "z" or loc_str2 == "z":
            third = 'y'
    else:
        third = 'x'
    
    all_ = np.concatenate((nodes_2d[ind_,axis[third]].reshape(-1,1), V_[ind_].reshape(-1,1)), axis=1)
    all_.view('float, float').sort(order=['f0'], axis=0)

    plotter.get_plot(all_[:,0],all_[:,1], xcompare_, ycompare_, xlabel=third)
    plotter.get_plot(all_[:,1],all_[:,0], xcompare_, ycompare_, ylabel=third)


def get_sphere_streamline(h_, xs_min, xs_max, zs_min, zs_max, nodes_2d, Y0, u_, w_):
    
    hmin = min(h_) # generate h min as the spacing of the region of interest (sphere surface)

    # interpolation source
    midy_ = abs(nodes_2d[:,1] - Y0) < 1e-8

    # interpolation location
    nx = round((xs_max-xs_min)/hmin)
    nz = round((zs_max-zs_min)/hmin)
    xi_ = np.linspace(xs_min, xs_max, int(nx))
    zi_ = np.linspace(zs_min, zs_max, int(nz))
    xi_2d, zi_2d = np.meshgrid(xi_, zi_)

    # interpolate
    u_2d = griddata((nodes_2d[midy_,0], nodes_2d[midy_,2]), u_[midy_], (xi_2d, zi_2d), method='linear')
    w_2d = griddata((nodes_2d[midy_,0], nodes_2d[midy_,2]), w_[midy_], (xi_2d, zi_2d), method='linear')

    # plot streamplot
    fig,ax = plt.subplots(figsize=(5*1.2,2*1.2))
    # fig.figsize = [5,2]
    ax.streamplot(xi_2d, zi_2d, u_2d, w_2d, density=3, color='k', linewidth=1., arrowstyle='-')
    # ax.streamplot(xi_2d, zi_2d, u_2d, w_2d, density=2, color=u_2d**2+w_2d**2, linewidth=2, cmap=plt.cm.autumn) # old
    ax.axis('equal')
    # ax.add_artist(Circle((geom['X0'], geom['Z0']), geom['RAD'], color='k'))
    # ax.aspect = 0.4
    # ax.set_xlim(min(xi_), max(xi_))
    # ax.set_ylim(min(zi_), max(zi_))

def get_Young():
    with open("Young/u100Y.dat",'r') as file:
        u100Y = file.readlines()    
    u100Y = [float(i.strip()) for i in u100Y]
    with open("Young/w100Y.dat",'r') as file:
        w100Y = file.readlines()    
    w100Y = [float(i.strip()) for i in w100Y]
    with open("Young/u200Y.dat",'r') as file:
        u200Y = file.readlines()    
    u200Y = [float(i.strip()) for i in u200Y]
    with open("Young/w200Y.dat",'r') as file:
        w200Y = file.readlines()    
    w200Y = [float(i.strip()) for i in w200Y]
    with open("Young/zY.dat",'r') as file:
        zY = file.readlines()    
    inc_Y = [float(i.strip()) for i in zY]

    return inc_Y, u100Y, w100Y, u200Y, w200Y
