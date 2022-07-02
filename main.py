# %% created by Tobias Samuel Sugandi / 23619009
import time
import numpy as np
from numpy.core.fromnumeric import argmax
from scipy import sparse, linalg, stats
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import json
from neighbor_search import direct, verlet, multiple_verlet
from dcpse import dcpse
from pressure_correction import rotinc_pc, get_dn_operator
import post
import plotter 
start = time.process_time()

# %% 
# =============================================================================
# Solver parameters
# =============================================================================

nu = 1/100                   # kinematic viscosity
DT = 1                      # time-step
TF = 10                     # final time

contd = False # NS time-stepping: start from t=0 (False) or continue(True)? 
sphere = False # Sphere or LCF

derivatives = "compute"         # compute or load derivative operator
if derivatives == "compute":    
    deriv_contd = False       # continue or start fresh
    RC_MULTIPLIER = 3.5         # cutoff radius multiplier 
    RC_MULTIPLIER_BOUND = 3.5   # cutoff radius multiplier for boundary particles
    P = 3                       # order of polynomial basis. ex: 1st derivative with r=2 requires P=2 (second order) polynomial basis

    upwind = False # whether or not to also find one-sided neighbors
    NNPS = "compute"            # compute or load
    if NNPS == "compute":
        METHOD_NNPS = 3 # always use 3
        # 1. direct_search
        # 2. verlet_list
        # 3. multiple_verlet
    check_indices = [0,1]# Indices of particles which neighbors are to be checked 

elif derivatives == "load": 
    # BYPASS neighbor search and derivatives computation, 
    # load the derivatives from a file
    pass


# Brinkman
brinkman = False
eta = 1e-4

# For Reynolds number computation RE = (V*D/nu)
RE_D = 1 
RE_V = 1

# Rotating sphere angular velocity
omega = 0.0 # 0.2, 0.5, 0.8, 1.2, 2.0

#! INITIAL condition, jump to NS Solver algorithm section
#! BOUNDARY condition, jump to NS Solver algorithm section

# %% 
# =============================================================================
# Import Geometry from file
# =============================================================================
print("reading geometry file...")
# import geometry data (a dictionary)
with open('geom.txt','r') as file:
    geom = json.load(file)
h, n_bound, n_total, idx_inner_1 = geom['h'],  geom['n_bound'], geom['n_total'], geom['idx_inner_1']
print("h, n_bound, n_total: ", h, n_bound, n_total)

# import nodes coordinates and boundary normal and h
nodes_2d = np.load('nodes_2d.npy')
normal_2d = np.load('normal_2d.npy')
h_ = np.load('h_.npy')
normal_x_bound_, normal_y_bound_, normal_z_bound_ = normal_2d[:,0].reshape(-1,1), normal_2d[:,1].reshape(-1,1), normal_2d[:,2].reshape(-1,1) 
# ! The coordinates must be in the form of [x|y] with boundary particles sorted in the top of the row

#* check the imported geometry
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
if 'n_sphere' in geom:
    idx_111 = geom['n_bound'] - geom['n_sphere']
else:
    idx_111 = 0
ax.scatter(nodes_2d[idx_111:n_bound,0], nodes_2d[idx_111:n_bound,1], nodes_2d[idx_111:n_bound,2], alpha = 0.1)
# ax.quiver(nodes_2d[:n_bound,0], nodes_2d[:n_bound,1], nodes_2d[:n_bound,2], normal_x_bound_, normal_y_bound_, normal_z_bound_)

# %% 
# =============================================================================
# Neighbor-search 
# =============================================================================
if derivatives == "compute":
    # cutoff - radius
    rc_ = np.concatenate((h_[:n_bound]*RC_MULTIPLIER_BOUND, h_[n_bound:]*RC_MULTIPLIER))
    if NNPS == "compute":   
        print("performing NNPS...")
        # approx max. number of neighbor
        max_n_neigh = round(5/3*np.pi*(RC_MULTIPLIER_BOUND)**3 ) #! unused
        print("approx max n_neigh = ", max_n_neigh)
        # neighbor search
        if METHOD_NNPS == 1:
            neighbors_2d, n_neigh_ = direct(nodes_2d, n_bound, rc_, max_n_neigh)
        elif METHOD_NNPS == 2:
            neighbors_2d, n_neigh_ = verlet(nodes_2d, n_bound, rc_, max_n_neigh)
        elif METHOD_NNPS == 3:
            neighbors_2d, n_neigh_ = multiple_verlet(nodes_2d, n_bound, rc_, upwind)
        mode = int(stats.mode(n_neigh_[idx_inner_1:])[0]) # compute mode
                
        # print NNPS data
        print(f"Max n_neighs = {max(n_neigh_)}")
        print(f"Min n_neighs = {min(n_neigh_)}")
        print(f"Mode n_neighs  = {mode}")

        # save neighbors
        # np.save("neighbors_2d.npy",neighbors_2d)
        with open('neighbors_2d.txt', 'w') as file:
            json.dump(neighbors_2d, file)
        np.save("n_neigh_.npy",n_neigh_)

        # plot Neighboring particles
        plotter.get_neigh_plot(ax, check_indices, nodes_2d, neighbors_2d, n_neigh_, brinkman)

    elif NNPS == "load":
        # load neighbors
        #! if neighbors_2d is in the old format, check also plotter.py, lsmps.py, dcpse.py
        # neighbors_2d = np.load("neighbors_2d.npy")
        with open('neighbors_2d.txt', 'r') as file:
            neighbors_2d = json.load(file)
        n_neigh_ =  np.load("n_neigh_.npy")
        mode = int(stats.mode(n_neigh_[idx_inner_1:])[0]) # compute mode
        with open("Output.txt", "w") as text_file:
            print(f"mode: {mode}", file=text_file)    
        # plot Neighboring particles
        plotter.get_neigh_plot(ax, check_indices, nodes_2d, neighbors_2d, n_neigh_, brinkman)
    fig, ax = plt.subplots()
    ax.plot(rc_, ".")
# plt.show(block=True)
elapsed_NNPS = (time.process_time() - start)/3600
print(f"elapsed time (NNPS) =  {elapsed_NNPS} hrs" ) 

# %%
# =============================================================================
# Derivatives computations
# =============================================================================
if derivatives == "compute":
    print("computing derivatives...")

# DC-PSE
if derivatives == "compute":
    if P == 1:
        dx_2d, dy_2d, dz_2d = dcpse(deriv_contd, P, h_, nodes_2d, neighbors_2d, n_neigh_, n_bound, upwind)
    else:
        dx_2d, dy_2d, dz_2d, \
            dxx_2d, dyy_2d, dzz_2d = dcpse(deriv_contd, P, h_, nodes_2d, neighbors_2d, n_neigh_, n_bound, upwind)
elapsed_deriv = (time.process_time() - start)/3600 - elapsed_NNPS
print(f"elapsed time (Derivatives) =  {elapsed_deriv} hrs" )
# plt.show(block=True)
 
# %%
# =============================================================================
# # Load derivatives from a file
# =============================================================================
deriv = "dcpse"

if derivatives == "load":
    print("loading derivatives...")
    dx_2d = sparse.load_npz(deriv +'/dx_'+deriv+'_2d.npz')
    dy_2d = sparse.load_npz(deriv +'/dy_'+deriv+'_2d.npz')
    dz_2d = sparse.load_npz(deriv +'/dz_'+deriv+'_2d.npz')
    dxx_2d = sparse.load_npz(deriv +'/dxx_'+deriv+'_2d.npz')
    dyy_2d = sparse.load_npz(deriv +'/dyy_'+deriv+'_2d.npz')  
    dzz_2d = sparse.load_npz(deriv +'/dzz_'+deriv+'_2d.npz')

    # check
    fcn = nodes_2d[:,0] + nodes_2d[:,1] + nodes_2d[:,2] # test-function 
    fig, ax = plt.subplots(2,3 )
    ax[0,0].plot(dx_2d.dot(fcn) )
    ax[0,1].plot(dy_2d.dot(fcn) )
    ax[0,2].plot(dz_2d.dot(fcn) )
    ax[1,0].plot(dxx_2d.dot(fcn) )
    ax[1,1].plot(dyy_2d.dot(fcn) )        
    ax[1,2].plot(dzz_2d.dot(fcn) )

# # upwind
# dx_xneg_dcpse_2d = sparse.load_npz(deriv +'/dx_xneg_'+deriv+'_2d.npz')
# dx_xpos_dcpse_2d = sparse.load_npz(deriv +'/dx_xpos_'+deriv+'_2d.npz')
# dy_yneg_dcpse_2d = sparse.load_npz(deriv +'/dy_yneg_'+deriv+'_2d.npz')
# dy_ypos_dcpse_2d = sparse.load_npz(deriv +'/dy_ypos_'+deriv+'_2d.npz')
# dz_zneg_dcpse_2d = sparse.load_npz(deriv +'/dz_zneg_'+deriv+'_2d.npz')
# dz_zpos_dcpse_2d = sparse.load_npz(deriv +'/dz_zpos_'+deriv+'_2d.npz') 

# # check
# fig2, ax2 = plt.subplots(3,2)
# ax2[0,0].plot(dx_xneg_dcpse_2d.dot(fcn) )
# ax2[0,1].plot(dx_xpos_dcpse_2d.dot(fcn) )
# ax2[1,0].plot(dy_yneg_dcpse_2d.dot(fcn) )
# ax2[1,1].plot(dy_ypos_dcpse_2d.dot(fcn) )
# ax2[2,0].plot(dz_zneg_dcpse_2d.dot(fcn) )
# ax2[2,1].plot(dz_zpos_dcpse_2d.dot(fcn) )

# plt.show(block=True)
# %% 
# =============================================================================
# NS-Solver Algorithm
# =============================================================================
ntotal = nodes_2d.shape[0]

#! INITIAL condition
# a uniform and divergence free field if possible, default is zero
V0_2d = np.zeros((ntotal,3))

#! BOUNDARY CONDITION
# Both for velocity and pressure
# 1. LHS MATRIX corresponding to the boundary particles
# 2. RHS VECTOR corresponding to the boundary particles

# Initialize LHS boundary operators
I_2d = sparse.identity(ntotal).tocsr() # Identity
#* LHS:: default boundary operator for velocity (& streamfcn) is Dirichlet
u_bound_2d = I_2d[:n_bound]
v_bound_2d = u_bound_2d.copy()
w_bound_2d = u_bound_2d.copy()
# psi_bound_2d = u_bound_2d.copy()
#* LHS:: default boundary operator for pressure is Neumann
p_bound_2d = get_dn_operator(0, n_bound, dx_2d, dy_2d, dz_2d, normal_x_bound_[:n_bound], normal_y_bound_[:n_bound], normal_z_bound_[:n_bound] )

#* RHS:: default value is 0
rhs_u_ = np.zeros(n_bound) # Initialize rhs vector
rhs_v_ = np.zeros(n_bound) # Initialize rhs vector
rhs_w_ = np.zeros(n_bound) # Initialize rhs vector
rhs_p_ = np.zeros(n_bound) # Initialize rhs vectory
# rhs_psi_ = np.zeros(n_bound) # Initialize rhs vector

#* Template for NON-default-boundary-operator 
#* Neumann velocity
# u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
# v_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
# w_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
#* Dirichlet pressure
# p_bound_2d[idx_begin:idx_end] = I_2d[idx_begin:idx_end]
#%% 
  
# The order(E-N-W-S-T-B-) of the boundary locations is based on the geometry creation
#? east
idx_begin   = 0
idx_end     = geom['n_east']    
if sphere:
    u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end] #* neumann velocity
    p_bound_2d[idx_begin:idx_end] = I_2d[idx_begin:idx_end] # * dirichlet pressure
# p_bound_2d[idx_begin] = I_2d[idx_begin]
#* rhs
# rhs_u_[idx_begin:idx_end] = 1.0

#? north
idx_begin   = idx_end
idx_end     = idx_begin + geom['n_north']
# u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
#* rhs
if sphere:
    rhs_u_[idx_begin:idx_end] = 1.0

#? west 
idx_begin   = idx_end
idx_end     = idx_begin + geom['n_west']
#* rhs
if sphere:
    rhs_u_[idx_begin:idx_end] = 1.0

#? south
idx_begin   = idx_end
idx_end     = idx_begin + geom['n_south']
# u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
#* rhs
if sphere:
    rhs_u_[idx_begin:idx_end] = 1.0

#? top
idx_begin   = idx_end
idx_end     = idx_begin + geom['n_top']
# u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
#* rhs
rhs_u_[idx_begin:idx_end] = 1.0 # both cavity and sphere flow have this BC

#? bottom
idx_begin   = idx_end
idx_end     = idx_begin + geom['n_bottom']
#* dirichlet pressure
if not sphere:
    p_bound_2d[idx_begin] = I_2d[idx_begin]
# u_bound_2d[idx_begin:idx_end] = p_bound_2d[idx_begin:idx_end]
#* rhs
if sphere:
    rhs_u_[idx_begin:idx_end] = 1.0

if not brinkman and sphere: # sphere flow without IBM
    #? sphere
    # idx_begin   = idx_end
    # idx_end     = idx_begin + geom['n_sphere']
    idx_begin   = n_bound - geom['n_sphere']
    idx_end     = n_bound
    
    #* rhs
    # rhs_psi_[idx_begin:idx_end] = 0.0
    rhs_u_[idx_begin:idx_end] = omega * (nodes_2d[idx_begin:idx_end,2] - geom['Z0'])
    rhs_w_[idx_begin:idx_end] = omega * (geom['X0'] - nodes_2d[idx_begin:idx_end,0])
    pass

# %%
# poisson matrix
poisson_2d = dxx_2d + dyy_2d + dzz_2d
# stack with boundary condition
poisson_2d = sparse.vstack([p_bound_2d, poisson_2d[n_bound:]])
poisson_2d = sparse.linalg.factorized(poisson_2d.tocsc()) # pre-factorized: makes LU decomposition

# %% 
#! Perform NS-solver algorithm
print("solving N-S...")
u_, v_, w_, p_ = rotinc_pc(contd, poisson_2d, omega, sphere, V0_2d, nu, DT, TF, dx_2d, dy_2d, dz_2d, dxx_2d[n_bound:], dyy_2d[n_bound:], dzz_2d[n_bound:], u_bound_2d, v_bound_2d, w_bound_2d, p_bound_2d, rhs_u_, rhs_v_, rhs_w_, rhs_p_, n_bound, brinkman, eta, nodes_2d)

# %% 
# =============================================================================
# Post-processing
# =============================================================================
print("RE = ", RE_V*RE_D/nu)

# saving velocities and pressure
V_2d = np.concatenate((u_,v_,w_)).reshape((-1,3), order='F')
np.save("V_2d.npy", V_2d)
np.save("p_.npy", p_)

#* Cavity post
if not sphere:
    # reference
    inc_Y, u100Y, w100Y, u200Y, w200Y = post.get_Young()

    # velocity profile
    post.get_profile('x', 0.5, 'y', 0.5, u_, nodes_2d, u100Y, inc_Y)
    post.get_profile('y', 0.5, 'z', 0.5, w_, nodes_2d, inc_Y, w100Y)

    # contour
    midy = abs(nodes_2d[:,1] - 0.5) < 1e-8
    fig,ax = plt.subplots()
    # plotter.plotterrainbow(nodes_2d[midy,0],nodes_2d[midy,2],u_[midy],10,'streamwise velocity', ax)
    # plotter.get_tricontour(nodes_2d[midy,0],nodes_2d[midy,2],u_[midy],15, fig, ax)
    plotter.get_tricontour(nodes_2d[midy,0],nodes_2d[midy,2],p_[midy],15, fig, ax)
    plotter.get_quiver(nodes_2d[midy,0],nodes_2d[midy,2], u_[midy], w_[midy], ax)

    midx = abs(nodes_2d[:,0] - 0.5) < 1e-8
    fig,ax = plt.subplots()
    plotter.get_tricontour(nodes_2d[midx,1],nodes_2d[midx,2],p_[midx],15, fig, ax)
    plotter.get_quiver(nodes_2d[midx,1],nodes_2d[midx,2], u_[midx], w_[midx], ax)

#* Sphere post
else:
    # Inlet pressure contour
    idx_begin = geom['n_east'] + geom['n_north']
    idx_end = idx_begin + geom['n_west']
    fig, ax = plt.subplots()
    plotter.get_tricontour(nodes_2d[idx_begin:idx_end,1],nodes_2d[idx_begin:idx_end,2],p_[idx_begin:idx_end],25, fig, ax)
    # idx_mid_west = idx_begin + np.argmin(p_[idx_begin:idx_end]) # unused

    # velocity profiles through the center of the sphere
    if (geom['Y0'] % h) < 1e-8:
        post.get_profile('x', geom['X0'], 'y', geom['Y0'], u_, nodes_2d)
        post.get_profile('y', geom['Y0'], 'z', geom['Z0'], w_, nodes_2d)
    else:
        post.get_profile('x', 4.4, 'y', 4.4, u_, nodes_2d)
        post.get_profile('y', 4.4, 'z', 4.4, w_, nodes_2d)

    # streamwise velocity scatter plot
    fig,ax = plt.subplots()
    if (geom['Y0'] % h) < 1e-8:
        midy = abs(nodes_2d[:,1] - geom['Y0']) < 1e-8
        # plotter.plotterrainbow(nodes_2d[midy,0],nodes_2d[midy,2],u_[midy],10,'streamwise velocity', ax)
        plotter.get_tricontour(nodes_2d[midy,0],nodes_2d[midy,2],u_[midy],15, fig, ax) # u-velocity contour
        plotter.get_quiver(nodes_2d[midy,0],nodes_2d[midy,2], u_[midy], w_[midy], ax) # u-w velocity quiver
    else:
        midy = abs(nodes_2d[:,1] - 4.4) < 1e-8
        midu_ = griddata(nodes_2d, u_, (nodes_2d[midy,0], 4.5*np.ones_like(nodes_2d[midy,0]), nodes_2d[midy,2]), method='linear')
        midw_ = griddata(nodes_2d, w_, (nodes_2d[midy,0], 4.5*np.ones_like(nodes_2d[midy,0]), nodes_2d[midy,2]), method='linear')
        plotter.plotterrainbow(nodes_2d[midy,0],nodes_2d[midy,2],midu_,10,'streamwise velocity', ax)
        plotter.get_tricontour(nodes_2d[midy,0],nodes_2d[midy,2],midu_,15, fig, ax)
        plotter.get_quiver(nodes_2d[midy,0],nodes_2d[midy,2], midu_, midw_, ax)
    if not brinkman:
        ax.add_artist(Circle((geom['X0'], geom['Z0']), geom['RAD'], color='k'))

    # pressure contour in the streamwise direction at ymid
    fig,ax = plt.subplots()
    plotter.get_tricontour(nodes_2d[midy,0],nodes_2d[midy,2],p_[midy],15, fig, ax)

    # streamline for sphere simulation
    xs_min, xs_max = geom['X0']-2*geom['RAD'], geom['X0']+8*geom['RAD']
    zs_min, zs_max = geom['Z0']-2*geom['RAD'], geom['Z0']+2*geom['RAD']
    post.get_sphere_streamline(h_, xs_min, xs_max, zs_min, zs_max, nodes_2d, geom['Y0'], u_, w_)

    # u-velocity in the x axis along the geometric center (ymid, zmid)
    ymidzmid_ = (abs(nodes_2d[:,1] - geom['Y0']) < 1e-8) * (abs(nodes_2d[:,2] - geom['Z0']) < 1e-8)
    plotter.get_plot(nodes_2d[ymidzmid_,0], u_[ymidzmid_], xlabel='x', ylabel='u')

elapsed = (time.process_time() - start)/3600
print(f"Total elapsed time =  {elapsed} hrs" ) 
# plt.show(block=True)
