# 3D-sphere geometry creation : sphere IBM 

import numpy as np
import matplotlib.pyplot as plt
import json
import time
start = time.process_time()

# %% Input Parameters

XMIN, XMAX  = 0.0, 20.0
YMIN, YMAX  = 0.0, 10.0
ZMIN, ZMAX  = 0.0, 10.0
lx  = XMAX - XMIN
ly  = YMAX - YMIN
lz  = ZMAX - ZMIN

# h1 = 1/128
h1 = 1/64
h2 = 1/32
h3 = 1/16
h4 = 1/8 
h5 = 1/4 
h6 = 1/2 
# h6 = 1/2
hmin = h1
hmax = h6

# Sphere properties
RAD = 0.5
X0, Y0, Z0 = 5, 5, 5 #4.5, 4.5, 4.5
print("Blockage ratio = ", np.pi*RAD**2/(ly*lz))

# %%  Construct Boundary Nodes

h   = hmax # uniform spacing in all directions
NX  = int(lx/h +1) 
NY  = int(ly/h +1) 
NZ  = int(lz/h +1) 

# 1. east (x-max)
y_ = np.linspace(YMIN, YMAX, NY)
z_ = np.linspace(ZMIN, ZMAX, NZ)

Y_3d, Z_3d = np.meshgrid(y_,z_)
y_east_ = Y_3d.flatten()
z_east_ = Z_3d.flatten()
x_east_ = XMAX*np.ones_like(y_east_)

n_east = x_east_.shape[0] # number of particles in the east boundary 
normal_x_east_ = np.ones(n_east)
normal_y_east_ = np.zeros(n_east)
normal_z_east_ = np.zeros(n_east)

# 2. north(y-max)
x_ = np.linspace(XMIN+h, XMAX-h, NX-1-1)
z_ = np.linspace(ZMIN, ZMAX, NZ)

X_3d, Z_3d = np.meshgrid(x_,z_)
x_north_ = X_3d.flatten()
z_north_ = Z_3d.flatten()
y_north_ = YMAX*np.ones_like(x_north_)

n_north = x_north_.shape[0] # number of particles in the east boundary 
normal_x_north_ = np.zeros(n_north)
normal_y_north_ = np.ones(n_north) 
normal_z_north_ = np.zeros(n_north)

# 3. west (x-min)
y_ = np.linspace(YMIN, YMAX, NY)
z_ = np.linspace(ZMIN, ZMAX, NZ)

Y_3d, Z_3d = np.meshgrid(y_,z_)
y_west_ = Y_3d.flatten()
z_west_ = Z_3d.flatten()
x_west_ = XMIN*np.ones_like(y_west_)

n_west = x_west_.shape[0] # number of particles in the west boundary 
normal_x_west_ = -1*np.ones(n_west)
normal_y_west_ = np.zeros(n_west)
normal_z_west_ = np.zeros(n_west)

# 4. south(y-min)
x_ = np.linspace(XMIN+h, XMAX-h, NX-1-1)
z_ = np.linspace(ZMIN, ZMAX, NZ)

X_3d, Z_3d = np.meshgrid(x_,z_)
x_south_ = X_3d.flatten()
z_south_ = Z_3d.flatten()
y_south_ = YMIN*np.ones_like(x_south_)

n_south = x_south_.shape[0] # number of particles in the east boundary 
normal_x_south_ = np.zeros(n_south)
normal_y_south_ = -1*np.ones(n_south) 
normal_z_south_ = np.zeros(n_south)

# 5. top(z-max)
x_ = np.linspace(XMIN+h, XMAX-h, NX-1-1)
y_ = np.linspace(YMIN+h, YMAX-h, NY-1-1)

X_3d, Y_3d = np.meshgrid(x_,y_)
x_top_ = X_3d.flatten()
y_top_ = Y_3d.flatten()
z_top_ = ZMAX*np.ones_like(x_top_)

n_top = x_top_.shape[0] # number of particles in the east boundary 
normal_x_top_ = np.zeros(n_top)
normal_y_top_ = np.zeros(n_top) 
normal_z_top_ = np.ones(n_top) 

# 6. bottom(z-min)
x_ = np.linspace(XMIN+h, XMAX-h, NX-1-1)
y_ = np.linspace(YMIN+h, YMAX-h, NY-1-1)

X_3d, Y_3d = np.meshgrid(x_,y_)
x_bottom_ = X_3d.flatten()
y_bottom_ = Y_3d.flatten()
z_bottom_ = ZMIN*np.ones_like(x_bottom_)

n_bottom = x_bottom_.shape[0] # number of particles in the east boundary 
normal_x_bottom_ = np.zeros(n_bottom)
normal_y_bottom_ = np.zeros(n_bottom) 
normal_z_bottom_ = -1*np.ones(n_bottom) 

# # 7. sphere
# #! different h from other boundaries
# h = hmin
# n_teta  = int(np.pi*RAD/h)
# n_teta  = n_teta - (n_teta % 2) + 1 # so that there exist nodes at 0,90,180 deg
# teta_ = np.linspace(0, np.pi, n_teta)
# n_phi_ = np.ceil(2*np.pi*(RAD*np.sin(teta_))/h).astype('int') 
# for i in range(n_phi_.shape[0]):
#     if n_phi_[i] == 0:
#         n_phi_[i] += 1
#     if n_phi_[i] > 4:
#         n_phi_[i] -= np.remainder(n_phi_[i], 4)

# x_sphere_ = np.array([], dtype=float)
# y_sphere_ = np.array([], dtype=float)
# z_sphere_ = np.array([], dtype=float)
# normal_x_sphere_, normal_y_sphere_, normal_z_sphere_ = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
# for i in range(len(teta_)):
#     teta = teta_[i]
#     phi_ = np.linspace(0, 2*np.pi, n_phi_[i], endpoint=False)
#     x_sphere_ = np.concatenate((x_sphere_, X0 + RAD* np.sin(teta) * np.cos(phi_) )) 
#     y_sphere_ = np.concatenate((y_sphere_, Y0 + RAD* np.sin(teta) * np.sin(phi_) )) 
#     z_sphere_ = np.concatenate((z_sphere_, Z0 + RAD* np.cos(teta) * np.ones_like(phi_) )) 
#     normal_x_sphere_ = np.concatenate((normal_x_sphere_, -np.sin(teta) * np.cos(phi_) )) 
#     normal_y_sphere_ = np.concatenate((normal_y_sphere_, -np.sin(teta) * np.sin(phi_) )) 
#     normal_z_sphere_ = np.concatenate((normal_z_sphere_, -np.cos(teta) * np.ones_like(phi_) )) 

# n_sphere = x_sphere_.shape[0] # number of particles on the sphere

#* Concatenating boundary nodes
x_bound_ = np.concatenate((x_east_,x_north_,x_west_,x_south_,x_top_,x_bottom_)) #!
y_bound_ = np.concatenate((y_east_,y_north_,y_west_,y_south_,y_top_,y_bottom_)) #!
z_bound_ = np.concatenate((z_east_,z_north_,z_west_,z_south_,z_top_,z_bottom_)) #!
normal_x_bound_ = np.concatenate((normal_x_east_,normal_x_north_,normal_x_west_,normal_x_south_,normal_x_top_,normal_x_bottom_)) #!
normal_y_bound_ = np.concatenate((normal_y_east_,normal_y_north_,normal_y_west_,normal_y_south_,normal_y_top_,normal_y_bottom_))#!
normal_z_bound_ = np.concatenate((normal_z_east_,normal_z_north_,normal_z_west_,normal_z_south_,normal_z_top_,normal_z_bottom_))#!
normal_2d = np.concatenate((normal_x_bound_.reshape(-1,1),normal_y_bound_.reshape(-1,1),normal_z_bound_.reshape(-1,1)),axis=1)
n_bound = len(x_bound_) #!
print("number of boundary nodes: ", n_bound)
np.save("normal_2d.npy",normal_2d)

# store the spacing of boundary nodes
# n_bounding_box = n_bound - n_sphere
# h_bound_ = np.concatenate( (hmax*np.ones(n_bound), hmin*np.ones(n_sphere)) ) #!
h_bound_ = hmax*np.ones(n_bound) #!

# plotting boundary nodes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_sphere_,y_sphere_,z_sphere_)
ax.scatter(x_bound_,y_bound_,z_bound_)

# %% Create adaptive-resolution nodes
def get_nodes_spherical(h, R_in, R_out, x_node_, y_node_, z_node_, h_ ):
    print(f'h = {h}', end=' ')
    safety = 1e-10

    # generate a box of nodes 
    x_min    = X0 - 2*RAD 
    x_max    = X0 + 2*RAD
    y_min    = Y0 - 2*RAD
    y_max    = Y0 + 2*RAD
    z_min    = Z0 - 2*RAD
    z_max    = Z0 + 2*RAD
    nx  = round((x_max-x_min) / h) + 1
    ny  = round((y_max-y_min) / h) + 1
    nz  = round((z_max-z_min) / h) + 1
    print((x_max-x_min)/(nx-1))

    x_ = np.linspace(x_min, x_max, nx)
    y_ = np.linspace(y_min, y_max, ny)
    z_ = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(x_, y_, z_)
    x_adapt_ = X.flatten()
    y_adapt_ = Y.flatten()
    z_adapt_ = Z.flatten()

    # find unwanted nodes
    complement_inner_ = (x_adapt_ - X0)**2 + (y_adapt_ - Y0)**2 + (z_adapt_ - Z0)**2 <= (R_in + safety)**2
    complement_outer_ = (x_adapt_ - X0)**2 + (y_adapt_ - Y0)**2 + (z_adapt_ - Z0)**2 > (R_out + safety)**2
    complement_ = complement_inner_ + complement_outer_

    x_adapt_ = x_adapt_[~complement_]
    y_adapt_ = y_adapt_[~complement_]
    z_adapt_ = z_adapt_[~complement_]
    h_adapt_ = h*np.ones(len(x_adapt_))

    x_node_ = np.concatenate((x_node_, x_adapt_))
    y_node_ = np.concatenate((y_node_, y_adapt_))
    z_node_ = np.concatenate((z_node_, z_adapt_))
    h_ = np.concatenate((h_, h_adapt_))
    # ax.plot(x_adapt1_,y_adapt1_,'.') # plotting

    return x_node_, y_node_, z_node_, h_

def get_nodes_box(h, n_layer, extend_mult, x_min_prev, x_max_prev, y_min_prev, y_max_prev, z_min_prev, z_max_prev, x_node_, y_node_, z_node_, h_, x_min_add=0, x_max_add=0, y_min_add=0, y_max_add=0, z_min_add=0, z_max_add=0 ):
    print(f'h = {h}')
    safety = 1e-10

    # generate a box of nodes 
    x_min    = XMIN+hmax
    x_max    = XMAX-hmax
    y_min    = YMIN+hmax
    y_max    = YMAX-hmax
    z_min    = ZMIN+hmax
    z_max    = ZMAX-hmax
    nx  = round((x_max-x_min) / h) + 1
    ny  = round((y_max-y_min) / h) + 1
    nz  = round((z_max-z_min) / h) + 1

    # generate a box of nodes 
    x_ = np.linspace(x_min, x_max, nx)
    y_ = np.linspace(y_min, y_max, ny)
    z_ = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(x_, y_, z_)
    x_adapt_ = X.flatten()
    y_adapt_ = Y.flatten()
    z_adapt_ = Z.flatten()

    # find unwanted nodes
    complement_inner_ = (x_adapt_ >= x_min_prev - safety) * (x_adapt_ <= x_max_prev + safety) \
                        *(y_adapt_ >= y_min_prev - safety) * (y_adapt_ <= y_max_prev + safety) \
                        *(z_adapt_ >= z_min_prev - safety) * (z_adapt_ <= z_max_prev + safety) 

    # outer boundary
    x_min = x_min_prev - n_layer*h - safety + x_min_add
    x_max = x_max_prev + extend_mult*n_layer*h + safety + x_max_add
    y_min = y_min_prev - n_layer*h - safety + y_min_add
    y_max = y_max_prev + n_layer*h + safety + y_max_add
    z_min = z_min_prev - n_layer*h - safety + z_min_add
    z_max = z_max_prev + n_layer*h + safety + z_max_add

    complement_outer_ = (x_adapt_ < x_min) + (x_adapt_ > x_max) +  (y_adapt_ < y_min) + (y_adapt_ > y_max) +  (z_adapt_ < z_min) + (z_adapt_ > z_max)
    complement_ = complement_inner_ + complement_outer_

    x_adapt_ = x_adapt_[~complement_]
    y_adapt_ = y_adapt_[~complement_]
    z_adapt_ = z_adapt_[~complement_]
    h_adapt_ = h*np.ones(len(x_adapt_))

    x_node_ = np.concatenate((x_node_, x_adapt_))
    y_node_ = np.concatenate((y_node_, y_adapt_))
    z_node_ = np.concatenate((z_node_, z_adapt_))
    h_ = np.concatenate((h_, h_adapt_))
    # ax.plot(x_adapt1_,y_adapt1_,'.') # plotting

    return x_min, x_max, y_min, y_max, z_min, z_max, x_node_, y_node_, z_node_, h_


#* 1. inside the sphere
h = h3

R_in = 0
R_out = RAD - 2*h

x_node_, y_node_, z_node_, h_ = get_nodes_spherical(h, R_in, R_out, x_bound_, y_bound_, z_bound_, h_bound_ )

#* 2. inside the sphere
h = h2

R_in = R_out + 1e-5
R_out = RAD - 1*h

x_node_, y_node_, z_node_, h_ = get_nodes_spherical(h, R_in, R_out, x_node_, y_node_, z_node_, h_ )

#* 3. inside the sphere
h = h1

R_in = R_out + 1e-5
R_out = RAD

x_node_, y_node_, z_node_, h_ = get_nodes_spherical(h, R_in, R_out, x_node_, y_node_, z_node_, h_ )

#
#
#
# outside the sphere
#* 1. h1 : smallest spacing
h = h1
n_layer = 3

R_in = RAD+ 1e-9
R_out = RAD + n_layer*h

x_node_, y_node_, z_node_, h_ = get_nodes_spherical(h, R_in, R_out, x_node_, y_node_, z_node_, h_ )

#* 2. h2
h = h2
n_layer = 3

R_in = R_out + h*0.1
R_out = R_out + n_layer*h

x_node_, y_node_, z_node_, h_ = get_nodes_spherical(h, R_in, R_out, x_node_, y_node_, z_node_, h_ )

# #* 2. h2
# h = h3
# n_layer = 3

# R_in = R_out + h*0.1
# R_out = R_out + n_layer*h

# x_node_, y_node_, z_node_, h_ = get_nodes_spherical(h, R_in, R_out, x_node_, y_node_, z_node_, h_ )

# #* 2. h2
# h = h4
# n_layer = 3

# R_in = R_out + h*0.1
# R_out = R_out + n_layer*h

# x_node_, y_node_, z_node_, h_ = get_nodes_spherical(h, R_in, R_out, x_node_, y_node_, z_node_, h_ )

# #* 2. h2
# h = h5
# n_layer = 3

# R_in = R_out + h*0.1
# R_out = R_out + n_layer*h

# x_node_, y_node_, z_node_, h_ = get_nodes_spherical(h, R_in, R_out, x_node_, y_node_, z_node_, h_ )

#* 3. intermediate
# inner boundary: sphere
# outer boundary: box
# function of (h, R_out_prev, n_layer, extend_mult, x_bound_, y_bound_, z_bound_, h_bound_ ) 
h = h3
n_layer = 3
extend_mult = 7
print(f'h = {h}')
safety = 1e-10

# generate a box of nodes 
x_min    = XMIN+hmax
x_max    = XMAX-hmax
y_min    = YMIN+hmax
y_max    = YMAX-hmax
z_min    = ZMIN+hmax
z_max    = ZMAX-hmax
nx  = round((x_max-x_min) / h) + 1
ny  = round((y_max-y_min) / h) + 1
nz  = round((z_max-z_min) / h) + 1

# generate a box of nodes 
x_ = np.linspace(x_min, x_max, nx)
y_ = np.linspace(y_min, y_max, ny)
z_ = np.linspace(z_min, z_max, nz)

X, Y, Z = np.meshgrid(x_, y_, z_)
x_adapt_ = X.flatten()
y_adapt_ = Y.flatten()
z_adapt_ = Z.flatten()

# find unwanted nodes
complement_inner_ = (x_adapt_ - X0)**2 + (y_adapt_ - Y0)**2 + (z_adapt_ - Z0)**2 <= (R_out + safety)**2

# outer boundary
x_min = (X0 - R_out) - n_layer*h - safety
x_max = (X0 + R_out) + extend_mult*n_layer*h + safety
y_min = (Y0 - R_out) - n_layer*h - safety
y_max = (Y0 + R_out) + n_layer*h + safety
z_min = (Z0 - R_out) - n_layer*h - safety
z_max = (Z0 + R_out) + n_layer*h + safety

complement_outer_ = (x_adapt_ < x_min) + (x_adapt_ > x_max) +  (y_adapt_ < y_min) + (y_adapt_ > y_max) +  (z_adapt_ < z_min) + (z_adapt_ > z_max)
complement_ = complement_inner_ + complement_outer_

x_adapt_ = x_adapt_[~complement_]
y_adapt_ = y_adapt_[~complement_]
z_adapt_ = z_adapt_[~complement_]
h_adapt_ = h*np.ones(len(x_adapt_))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_adapt_,y_adapt_,z_adapt_)

x_node_ = np.concatenate((x_node_, x_adapt_))
y_node_ = np.concatenate((y_node_, y_adapt_))
z_node_ = np.concatenate((z_node_, z_adapt_))
h_ = np.concatenate((h_, h_adapt_))
# ax.plot(x_adapt1_,y_adapt1_,'.') # plotting

#* 4. box
h = h4
n_layer = 3
extend_mult = 3

x_min, x_max, y_min, y_max, z_min, z_max, x_node_, y_node_, z_node_, h_ = \
    get_nodes_box(h, n_layer, extend_mult, x_min, x_max, y_min, y_max, z_min, z_max, x_node_, y_node_, z_node_, h_, x_min_add=0, x_max_add=h, y_min_add=0, y_max_add=0, z_min_add=0, z_max_add=0 )

#* 5. box
h = h5
n_layer = 3
extend_mult = 3

x_min, x_max, y_min, y_max, z_min, z_max, x_node_, y_node_, z_node_, h_ = \
    get_nodes_box(h, n_layer, extend_mult, x_min, x_max, y_min, y_max, z_min, z_max, x_node_, y_node_, z_node_, h_, x_min_add=0, x_max_add=0, y_min_add=0, y_max_add=0, z_min_add=0, z_max_add=0 )

# #* 6. box
# h = h6
# n_layer = 3
# extend_mult = 3

# x_min, x_max, y_min, y_max, z_min, z_max, x_node_, y_node_, z_node_, h_ = \
#     get_nodes_box(h, n_layer, extend_mult, x_min, x_max, y_min, y_max, z_min, z_max, x_node_, y_node_, z_node_, h_, x_min_add=0, x_max_add=0, y_min_add=0, y_max_add=0, z_min_add=0, z_max_add=0 )


# %% Construct Inner Nodes
h = hmax
safety = 1e-8
x_ = np.linspace(XMIN+h, XMAX-h, NX-1-1)
y_ = np.linspace(YMIN+h, YMAX-h, NY-1-1)
z_ = np.linspace(ZMIN+h, ZMAX-h, NZ-1-1)

X_3d, Y_3d, Z_3d = np.meshgrid(x_,y_,z_)
x_inner_ = X_3d.flatten()
y_inner_ = Y_3d.flatten()
z_inner_ = Z_3d.flatten()

# find unwanted nodes (inner rectangle)
complement_ = (x_inner_ > x_min - safety) *  (x_inner_ < x_max + safety) \
    * (y_inner_ > y_min - safety) *  (y_inner_ < y_max + safety) \
        * (z_inner_ > z_min - safety) *  (z_inner_ < z_max + safety)

x_inner_ = x_inner_[~complement_]
y_inner_ = y_inner_[~complement_]
z_inner_ = z_inner_[~complement_]

n_inner = len(x_inner_)
h_inner_ = h*np.ones(n_inner)

# Concatenating xyz nodes
x_node_ = np.concatenate((x_node_, x_inner_))
y_node_ = np.concatenate((y_node_, y_inner_))
z_node_ = np.concatenate((z_node_, z_inner_))
h_ = np.concatenate((h_, h_inner_))
# n_inner = len(x_inner_)
# print("number of inner nodes: ", n_inner)

n_total = len(x_node_)
print("total number of nodes: ", n_total)
nodes_2d = np.concatenate((x_node_.reshape(-1,1),y_node_.reshape(-1,1),z_node_.reshape(-1,1)),axis=1)
np.save("nodes_2d.npy",nodes_2d)

# h 
# h_ = np.concatenate((h_bound_, h_adapt1_, h_adapt2_, h_adapt3_, h_adapt4_, h_adapt5_, h_inner_))
fig, ax = plt.subplots()
ax.plot(h_,'.')
np.save("h_.npy", h_)

# first idx of inner nodes
idx_inner_1 = n_total - n_inner

# checking created geometry in the midY plane
midy = abs(y_node_ - Y0)<1e-8
fig,ax = plt.subplots()
ax.plot(x_node_[midy], z_node_[midy], '.')
if 'n_sphere' in locals(): # if nodes on the sphere surface are generated in this code
    midy_sphere_ = abs(y_sphere_ - Y0)<1e-8
    ax.quiver(x_sphere_[midy_sphere_],z_sphere_[midy_sphere_], normal_x_sphere_[midy_sphere_],normal_z_sphere_[midy_sphere_])
ax.set_aspect('equal')

# checking created geometry
midx = abs(x_node_ - (X0+hmin))<1e-8
fig,ax = plt.subplots()
ax.plot(y_node_[midx], z_node_[midx], '.')
ax.set_aspect('equal')

# check sphere's volume
solid_ = (nodes_2d[:,0]-X0)**2 + (nodes_2d[:,1]-Y0)**2 + (nodes_2d[:,2]-Z0)**2 <= (RAD+1e-13)**2 
h_solid_ = h_[solid_]
vol = np.sum((h_solid_**3))
vol_true = 4/3*np.pi*RAD**3
print('Vol_true=',vol_true,'\t Vol_computed=', vol)
# check sphere's area
if 'n_sphere' in locals():
    idx_begin = n_bound # sphere coordinates
    idx_end = n_bound + n_sphere
    area = np.sum(h_[idx_begin:idx_end]**2)
    area_true = 4*np.pi*RAD**2
    print('Area_true=',area_true,'\t Area_computed=', area)

# create dictionary of geometry data
geom = {'h':h, 'n_total':n_total, 'n_bound':n_bound,
    'n_east':n_east,'n_north':n_north,'n_west':n_west,'n_south':n_south,\
        'n_top':n_top,'n_bottom':n_bottom, 'RAD':RAD,\
            'X0':X0, 'Y0':Y0, 'Z0':Z0, 'idx_inner_1':idx_inner_1} # dictionary of number of boundary nodes
if 'n_sphere' in locals(): # if nodes on the sphere surface are generated in this code
    geom['n_sphere'] = n_sphere
# save json
with open('geom.txt', 'w') as file:
    json.dump(geom, file)
    
# # Plotting all nodes 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_node_,y_node_,z_node_)

print("geometry creation done.")

# Computing time
elapsed = (time.process_time() - start)
print(f"Total elapsed time =  {elapsed} s" ) 

plt.show(block=True)
# input('press <ENTER> to continue')