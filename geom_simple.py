# 3D-box geometry creation
# * Input: xmax, xmin, ymin, ymax, nx, ny
# * Output: x and y coordinates of the nodes + boundary nodes' normal vector
import numpy as np
import matplotlib.pyplot as plt
import json
import time
start = time.process_time()

# %% Input Parameters

XMIN, XMAX  = 0.0, 1.0
YMIN, YMAX  = 0.0, 1.0
ZMIN, ZMAX  = 0.0, 1.0
lx  = XMAX - XMIN
ly  = YMAX - YMIN
lz  = ZMAX - ZMIN

# uniform spacing in all directions
h   = lx/40
NX  = int(lx/h ) + 1 
NY  = int(ly/h ) + 1
NZ  = int(lz/h ) + 1

# %% Computation (Inner Nodes)

x_ = np.linspace(XMIN+h, XMAX-h, NX-1-1)
y_ = np.linspace(YMIN+h, YMAX-h, NY-1-1)
z_ = np.linspace(ZMIN+h, ZMAX-h, NZ-1-1)

X_3d, Y_3d, Z_3d = np.meshgrid(x_,y_,z_)

x_inner_ = X_3d.flatten()
y_inner_ = Y_3d.flatten()
z_inner_ = Z_3d.flatten()
n_inner = len(x_inner_)
print("number of inner nodes: ", n_inner)

# plotting inner nodes (the boundary nodes will be plotted in this figure later)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_inner_,y_inner_, z_inner_)


# %%  Construct Boundary Nodes

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
normal_z_top_ = 1*np.ones(n_top) 

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

#* Concatenating boundary nodes
x_bound_ = np.concatenate((x_east_,x_north_,x_west_,x_south_,x_top_,x_bottom_))
y_bound_ = np.concatenate((y_east_,y_north_,y_west_,y_south_,y_top_,y_bottom_))
z_bound_ = np.concatenate((z_east_,z_north_,z_west_,z_south_,z_top_,z_bottom_))
normal_x_bound_ = np.concatenate((normal_x_east_,normal_x_north_,normal_x_west_,normal_x_south_,normal_x_top_,normal_x_bottom_))
normal_y_bound_ = np.concatenate((normal_y_east_,normal_y_north_,normal_y_west_,normal_y_south_,normal_y_top_,normal_y_bottom_))
normal_z_bound_ = np.concatenate((normal_z_east_,normal_z_north_,normal_z_west_,normal_z_south_,normal_z_top_,normal_z_bottom_))
normal_2d = np.concatenate((normal_x_bound_.reshape(-1,1),normal_y_bound_.reshape(-1,1),normal_z_bound_.reshape(-1,1)),axis=1)
n_bound = len(x_bound_)
print("number of boundary nodes: ", n_bound)
np.save("normal_2d.npy",normal_2d)

# plotting boundary nodes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_bound_,y_bound_,z_bound_)

#* Concatenating inner and boundary nodes
x_node_ = np.concatenate((x_bound_,x_inner_))
y_node_ = np.concatenate((y_bound_,y_inner_))
z_node_ = np.concatenate((z_bound_,z_inner_))
nodes_2d = np.concatenate((x_node_.reshape(-1,1),y_node_.reshape(-1,1),z_node_.reshape(-1,1)),axis=1)
np.save("nodes_2d.npy",nodes_2d)

idx_inner_1 = len(x_bound_)
n_total = len(x_node_)
h_ = h*np.ones(n_total)
np.save("h_.npy", h_)
fig, ax = plt.subplots()
ax.plot(h_, '.')

# create dictionary of geometry data
geom = {'h':h, 'n_total':n_total, 'n_bound':n_bound,\
    'n_east':n_east,'n_north':n_north,'n_west':n_west,'n_south':n_south,\
        'n_top':n_top,'n_bottom':n_bottom, 'idx_inner_1':idx_inner_1} # dictionary of number of boundary nodes
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




























"""# %% Writing the dat file
print("writing txt...")
file = open("geom.txt","w")

# file.write((f"{h}","\n",f"{n_inner}","\n",f"{n_bound}","\n"))
file.write(f"{h}")
file.write("\n")
file.write(f"{n_east}" + "\t" + f"{n_north}" + "\t" + f"{n_west}" + "\t" + f"{n_south}")
file.write("\n")
file.write(f"{n_inner}")
file.write("\n")

for i in range(n_bound):
    file.write(f"{x_bound_[i]}" + "\t" + f"{y_bound_[i]}" + "\t" + f"{normal_x_bound_[i]}" + "\t" + f"{normal_y_bound_[i]}")
    file.write("\n")
    pass

for i in range(n_inner):
    file.write(f"{x_inner_[i]}" + "\t" + f"{y_inner_[i]}" )
    file.write("\n")
    pass

file.close()"""

