import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import post
import matplotlib.pyplot as plt
import pickle
import json

def get_darcy_drag(eta, inner_2d, X0, Y0, Z0, RAD, I_inner_2d):    
    # return the (solid MASK * eta) 2d sparse matrix (for inner particles only)
    
    # cylinder
    in_solid_ = (inner_2d[:,0]-X0)**2 + (inner_2d[:,1]-Y0)**2 + (inner_2d[:,2]-Z0)**2 <= (RAD+1e-13)**2 
    darcy_drag_ = (1/eta)*in_solid_.astype(float)
    Ddrag_2d = I_inner_2d.multiply(darcy_drag_.reshape(-1,1))
    return Ddrag_2d


def rotinc_pc(contd, poisson_2d, omega, sphere, V0_2d, nu, DT, TF, dx_full_2d, dy_full_2d, dz_full_2d, dxx_2d, dyy_2d, dzz_2d, u_bound_2d, v_bound_2d, w_bound_2d, phi_bound_2d, rhs_u_, rhs_v_, rhs_w_, rhs_phi_, n_bound, brinkman, eta, nodes_2d):
    """Rotational-incremental pressure-correction algorithm for incompressible Navier-Stokes.
    with irreducible splitting error order of O(dt^2)
    PRESSURE BC: dp/dn = del x del x V

    Args:
        V0_2d (2d array): Initial velocity vector [u v]
        nu (float): kinematic viscosity
        DT (float): time increment
        TF (float): final time
        dx_bound_2d (2d sparse matrix): x-derivative operator for boundary particles
        dy_bound_2d (2d sparse matrix): y-derivative operator for boundary particles
        [[Operators below are for INNER PARTICLES ONLY!!]]
        dx_2d (2d sparse matrix): x-derivative operator
        dy_2d (2d sparse matrix): y-derivative operator
        dz_2d (2d sparse matrix): z-derivative operator
        dxx_2d (2d sparse matrix): xx-derivative operator
        dyy_2d (2d sparse matrix): yy-derivative operator
        dzz_2d (2d sparse matrix): zz-derivative operator
        u_bound_2d (2d sparse matrix): boundary operator for velocity 
        v_bound_2d (2d sparse matrix): boundary operator for velocity 
        w_bound_2d (2d sparse matrix): boundary operator for w-velocity 
        Phi_bound_2d (2d sparse matrix): boundary operator for pressure
        rhs_u_ (1d array): rhs for u-velocity
        rhs_v_ (1d array): rhs for v-velocity
        rhs_w_ (1d array): rhs for w-velocity
        rhs_phi_ (1d array): rhs for phi
        n_bound (int): number of boundary particles

    Output:
        u_(1d array): u-velocity at the end of simulation time
        v_(1d array): v-velocity at the end of simulation time
        w_(1d array): w-velocity at the end of simulation time
        p_(1d array): pressure at the end of simulation time
    """
    # ! deriv (INPUT)
    deriv = 'dcpse'
    # upwind: import derivatives
    dx_xneg_dcpse_2d = sparse.load_npz(deriv +'/dx_xneg_'+deriv+'_2d.npz')
    dx_xpos_dcpse_2d = sparse.load_npz(deriv +'/dx_xpos_'+deriv+'_2d.npz')
    dy_yneg_dcpse_2d = sparse.load_npz(deriv +'/dy_yneg_'+deriv+'_2d.npz')
    dy_ypos_dcpse_2d = sparse.load_npz(deriv +'/dy_ypos_'+deriv+'_2d.npz')
    dz_zneg_dcpse_2d = sparse.load_npz(deriv +'/dz_zneg_'+deriv+'_2d.npz')
    dz_zpos_dcpse_2d = sparse.load_npz(deriv +'/dz_zpos_'+deriv+'_2d.npz') 

    info = "direct-solver" # dummy (if direct method is used)
    
    n_total = V0_2d.shape[0]
    n_inner = n_total - n_bound

    #! remember that the derivatives operator are for inner particles only [n_bound:]
    #  except
    dx_2d = dx_full_2d[n_bound:]
    dy_2d = dy_full_2d[n_bound:]
    dz_2d = dz_full_2d[n_bound:]
    
    # Initialize variables
    I_2d = sparse.identity(n_total).tocsr() # identity matrix
    Ddrag_2d = sparse.csr_matrix((n_inner, n_total), dtype=float) # initialize zero sparse matrix
    cd_container = [[],[]]
    u_rot_ = 0
    w_rot_ = 0
    convectionx_solid_ = 0
    convectionz_solid_ = 0

    #* LOG printing
    log_printing = open('current_result\\log.txt','w')
    log_printing.close()
    log_printing = open('current_result\\log.txt','a')
    
    #NOTE : POISSON HAS BEEN CONSTRUCTED OUTSIDE
    # #* construct matrix for pressure-poisson eqn
    # # poisson_2d = dxx_backfor_2d[n_bound:] + dyy_backfor_2d[n_bound:] + dzz_backfor_2d[n_bound:]
    # poisson_2d = dxx_2d + dyy_2d + dzz_2d

    # # stack with boundary condition
    # poisson_2d = sparse.vstack([phi_bound_2d, poisson_2d])
    
    # # factorize the poisson LHS
    # poisson_2d = linalg.factorized(poisson_2d.tocsc()) # pre-factorized: makes LU decomposition
    #NOTE

    if brinkman: # prepare brinkman term and cd computation
        # sphere
        with open('geom.txt','r') as file:
            geom = json.load(file)
        h_ = np.load('h_.npy')
        X0 = geom['X0'] 
        Y0 = geom['Y0'] 
        Z0 = geom['Z0'] 
        RAD = geom['RAD']
        solid_ = (nodes_2d[:,0]-X0)**2 + (nodes_2d[:,1]-Y0)**2 + (nodes_2d[:,2]-Z0)**2 <= (RAD+1e-13)**2 
        Ddrag_2d = get_darcy_drag(eta, nodes_2d[n_bound:], X0, Y0, Z0, RAD, I_2d[n_bound:]) 

        if omega>0:
            u_rot_ = 1/eta* omega* (nodes_2d[solid_,2] -Z0)
            w_rot_ = 1/eta* omega* -(nodes_2d[solid_,0] -X0)

    elif sphere: # prepare non-brinkman cd computation
        # sphere
        with open('geom.txt','r') as file:
            geom = json.load(file)
        h_ = np.load('h_.npy')
        # sphere coordinates
        idx_begin = n_bound - geom['n_sphere']
        idx_end = n_bound
        normal_2d = np.load('normal_2d.npy')
        normal_x_sp_ = normal_2d[idx_begin:,0]
        normal_y_sp_ = normal_2d[idx_begin:,1]
        normal_z_sp_ = normal_2d[idx_begin:,2]
        dx_sp_2d = dx_full_2d[idx_begin:idx_end]
        dy_sp_2d = dy_full_2d[idx_begin:idx_end]
        dz_sp_2d = dz_full_2d[idx_begin:idx_end]        

    # first time-step or contd
    if not contd: # start from time 0
        t_done = 0 # current time in the loop
        u_ = V0_2d[:,0] # initial-condition
        v_ = V0_2d[:,1]
        w_ = V0_2d[:,2]   
    
        # save u(k-1)
        u_old_ = np.copy(u_)
        v_old_ = np.copy(v_)
        w_old_ = np.copy(w_)

        # Compute the first time step using first-order pressure correction scheme
        #! Assuming that the initial velocity are uniform and divergence-free, THUS  p(0) = 0
        # construct matrix corresponding to diffusion (1st order time stepping)
        diff_2d = I_2d[n_bound:] - DT*nu* (dxx_2d + dyy_2d + dzz_2d)
    
        # 1. Semi-implicit convection-diffusion
        # obtain inner velocity and reshape
        in_u_ = u_[n_bound:].reshape(n_inner,1)
        in_v_ = v_[n_bound:].reshape(n_inner,1)
        in_w_ = w_[n_bound:].reshape(n_inner,1)
        # construct the LHS matrix (adding convection)
        in_LHS_2d = diff_2d + DT* (dx_2d.multiply(in_u_) + dy_2d.multiply(in_v_)  + dz_2d.multiply(in_w_) + Ddrag_2d)

        LHS_2d = sparse.vstack([u_bound_2d, in_LHS_2d])
        rhs_            = np.copy(u_)    # construct rhs
        rhs_[:n_bound]  = rhs_u_
        if brinkman:
            rhs_[solid_] += u_rot_*DT
        # u_ = linalg.spsolve(LHS_2d, rhs_)
        # u_, info = linalg.gmres(LHS_2d, rhs_)
        u_, info = linalg.bicgstab(LHS_2d, rhs_, x0=u_, tol=1e-07)
        # print('info_u: ',info)

        LHS_2d = sparse.vstack([v_bound_2d, in_LHS_2d])
        rhs_            = np.copy(v_)    # construct rhs
        rhs_[:n_bound]  = rhs_v_
        # v_ = linalg.spsolve(LHS_2d, rhs_)
        # v_, info = linalg.gmres(LHS_2d, rhs_)
        v_, info = linalg.bicgstab(LHS_2d, rhs_, x0=v_, tol=1e-07)
        # print('info_v: ',info)

        LHS_2d = sparse.vstack([w_bound_2d, in_LHS_2d])
        rhs_            = np.copy(w_)    # construct rhs
        rhs_[:n_bound]  = rhs_w_
        if brinkman:
            rhs_[solid_] += w_rot_*DT
        # w_ = linalg.spsolve(LHS_2d, rhs_)
        # w_, info = linalg.gmres(LHS_2d, rhs_)
        w_, info = linalg.bicgstab(LHS_2d, rhs_, x0=w_, tol=1e-07)
        # print('info_w: ',info)

        if brinkman:
            #? force computation
            uu_solid_ = u_[solid_]/eta - u_rot_
            ww_solid_ = w_[solid_]/eta - w_rot_
            if omega>0:
                convectionx_solid_ = u_[solid_]* (dx_2d[solid_].dot(u_)) \
                                + v_[solid_]* (dy_2d[solid_].dot(u_)) \
                                + w_[solid_]* (dz_2d[solid_].dot(u_))
                convectionz_solid_ = u_[solid_]* (dx_2d[solid_].dot(w_)) \
                                + v_[solid_]* (dy_2d[solid_].dot(w_)) \
                                + w_[solid_]* (dz_2d[solid_].dot(w_))
            h_solid_ = h_[solid_]
            c_x = 2*(4/np.pi)*np.sum((uu_solid_ - convectionx_solid_) *(h_solid_**3))
            c_z = 2*(4/np.pi)*np.sum((ww_solid_ - convectionz_solid_) *(h_solid_**3))
            cd_container[1].append(c_x)
            print(' CD = ', c_x)
            print(' CL = ', c_z)
            print(' CD = ', c_x, file=log_printing)
            print(' CL = ', c_z, file=log_printing)

        # 2. Pressure-correction 
        rhs_    = 1/DT * ( dx_2d.dot(u_) + dy_2d.dot(v_) + dz_2d.dot(w_) )
        # rhs_    = 1/DT * ( dx_xneg_dcpse_2d.dot(u_) + dy_yneg_dcpse_2d.dot(v_) + dz_zneg_dcpse_2d.dot(w_) )
        rhs_    = np.concatenate((rhs_phi_, rhs_)) #! beware here because here we solve for p whereas the input is for phi
        # p_, info = linalg.gmres(poisson_2d, rhs_ )
        # p_, info = linalg.bicgstab(poisson_2d, rhs_ , tol=1e-05)
        # p_ = linalg.spsolve(poisson_2d, rhs_ )
        p_ = poisson_2d(rhs_)
        # p_ = poisson_2d.solve(rhs_)
        # p_ = tobsolve_lu(rhs_, L, U, perm_r, perm_c)
        # print('info: ',info)

        #! here the boundary velocities are not updated by the pressure correction
        u_[n_bound:] = u_[n_bound:] - DT* dx_2d.dot(p_)
        v_[n_bound:] = v_[n_bound:] - DT* dy_2d.dot(p_)
        w_[n_bound:] = w_[n_bound:] - DT* dz_2d.dot(p_)

        if not brinkman and sphere:
            #? force computation
            Fx_rho_ = -p_[idx_begin:idx_end] * -normal_x_sp_ + nu* (2*(dx_sp_2d@u_)*-normal_x_sp_ \
                    + (dy_sp_2d@u_ + dx_sp_2d@v_)*-normal_y_sp_ \
                    + (dz_sp_2d@u_ + dx_sp_2d@w_)*-normal_z_sp_) 
            Fx_rho = Fx_rho_.sum() * min(h_)**2
            cd = 2*Fx_rho*4/np.pi # Johnson
            cd_container[1].append(cd)
            print(f' CD = {cd}')
            print(f' CD = {cd}', file=log_printing)

        t_done += DT
        print(f"t = {t_done} done")
        print(f"t = {t_done} done", file=log_printing)
        cd_container[0].append(t_done)
        # dummy for 1st iter only
        phi_ = np.copy(p_)
    
    else: # continue from previous time step
        # importing velocities and pressure
        V_2d = np.load("contd\\V_2d.npy")
        u_, v_, w_ = V_2d[:,0], V_2d[:,1], V_2d[:,2]
        p_ = np.load("contd\\p_.npy")
        # importing old velocities and pressure
        V_old_2d = np.load("contd\\V_old_2d.npy")
        u_old_, v_old_, w_old_ = V_old_2d[:,0], V_old_2d[:,1], V_old_2d[:,2]
        p_now_ = np.load("contd\\p_now_.npy")   
        
        # import time
        with open('contd\\t_done.pkl','rb') as f:
            t_done = pickle.load(f)

        # computing phi
        divV_   = dx_full_2d.dot(u_) + dy_full_2d.dot(v_) + dz_full_2d.dot(w_)
        phi_ = p_ - p_now_ + nu*divV_
    
    print(' max u = ', max(u_))
    
    #* construct matrix corresponding to diffusion (2nd order SCHEME)
    diff_2d = I_2d[n_bound:] - 2/3*DT*nu* (dxx_2d + dyy_2d + dzz_2d)

    while t_done < TF -1e-10:

        #* 1. (Semi-implicit) convection (implicit) diffusion step
        # obtain inner velocity and reshape
        in_u_ = u_[n_bound:].reshape(n_inner,1)
        in_v_ = v_[n_bound:].reshape(n_inner,1)
        in_w_ = w_[n_bound:].reshape(n_inner,1)
        # construct the LHS matrix (inner nodes part)
        # in_LHS_2d = diff_2d + 2/3*DT* (dx_2d.multiply(in_u_) + dy_2d.multiply(in_v_) + dz_2d.multiply(in_w_) + Ddrag_2d) 
        in_LHS_2d = diff_2d + 2/3*DT* (\
            dx_xneg_dcpse_2d.multiply(np.maximum(in_u_,0)) + dx_xpos_dcpse_2d.multiply(np.minimum(in_u_,0)) +\
            dy_yneg_dcpse_2d.multiply(np.maximum(in_v_,0)) + dy_ypos_dcpse_2d.multiply(np.minimum(in_v_,0)) +\
            dz_zneg_dcpse_2d.multiply(np.maximum(in_w_,0)) + dz_zpos_dcpse_2d.multiply(np.minimum(in_w_,0)) +\
            Ddrag_2d) 
        # NOTE: positive velocity is multiplied with neighbors from the negative regions

        LHS_2d = sparse.vstack([u_bound_2d, in_LHS_2d])
        # construct rhs: u(k), u(k-1), (dp/dx)
        # rhs_u_[:361]  = -0.3*(u_[:361]-1) 
        rhs_    = 4/3*u_[n_bound:] - 1/3*u_old_[n_bound:] - 2/3*DT* dx_2d.dot(p_)
        # rhs_    = 4/3*u_[n_bound:] - 1/3*u_old_[n_bound:] - 2/3*DT* (dx_xneg_dcpse_2d.multiply(np.maximum(np.sign(in_u_),0)) - dx_xpos_dcpse_2d.multiply(np.minimum(np.sign(in_u_),0))).dot(p_)
        rhs_    = np.concatenate((rhs_u_, rhs_))
        if brinkman:
            rhs_[solid_] += u_rot_ *2/3*DT
        # save u(k-1)
        u_old_ = np.copy(u_)
        # solve the linear system
        # u_ = linalg.spsolve(LHS_2d, rhs_)
        # u_, info = linalg.gmres(LHS_2d, rhs_)
        u_, info = linalg.bicgstab(LHS_2d, rhs_, x0=u_, tol=1e-07)
        # print('info_u: ',info)

        LHS_2d = sparse.vstack([v_bound_2d, in_LHS_2d])
        # construct rhs: v(k), v(k-1), (dp/dy)
        rhs_    = 4/3*v_[n_bound:] - 1/3*v_old_[n_bound:] - 2/3*DT* dy_2d.dot(p_)
        # rhs_    = 4/3*v_[n_bound:] - 1/3*v_old_[n_bound:] - 2/3*DT* dy_ypos_dcpse_2d.dot(p_)
        rhs_    = np.concatenate((rhs_v_, rhs_))
        # save v(k-1)
        v_old_ = np.copy(v_)
        # solve the linear system
        # v_ = linalg.spsolve(LHS_2d, rhs_)
        # v_, info = linalg.gmres(LHS_2d, rhs_)
        v_, info = linalg.bicgstab(LHS_2d, rhs_, x0=v_, tol=1e-07)
        # print('info_v: ',info)

        LHS_2d = sparse.vstack([w_bound_2d, in_LHS_2d])
        rhs_    = 4/3*w_[n_bound:] - 1/3*w_old_[n_bound:] - 2/3*DT* dz_2d.dot(p_)  # construct rhs
        # rhs_    = 4/3*w_[n_bound:] - 1/3*w_old_[n_bound:] - 2/3*DT* dz_zpos_dcpse_2d.dot(p_)  # construct rhs
        rhs_    = np.concatenate((rhs_w_, rhs_))
        if brinkman:
            rhs_[solid_] += w_rot_*2/3*DT
        # save w(k-1)
        w_old_ = np.copy(w_)
        # solve the linear system
        # w_ = linalg.spsolve(LHS_2d, rhs_)
        # w_, info = linalg.gmres(LHS_2d, rhs_)
        w_, info = linalg.bicgstab(LHS_2d, rhs_, x0=w_, tol=1e-07)
        # print('info_w: ',info)           

        if brinkman:
            #? force computation
            uu_solid_ = u_[solid_]/eta - u_rot_
            ww_solid_ = w_[solid_]/eta - w_rot_
            if omega>0:
                convectionx_solid_ = u_[solid_]* (dx_2d[solid_].dot(u_)) \
                                + v_[solid_]* (dy_2d[solid_].dot(u_)) \
                                + w_[solid_]* (dz_2d[solid_].dot(u_))
                convectionz_solid_ = u_[solid_]* (dx_2d[solid_].dot(w_)) \
                                + v_[solid_]* (dy_2d[solid_].dot(w_)) \
                                + w_[solid_]* (dz_2d[solid_].dot(w_))
            h_solid_ = h_[solid_]
            c_x = 2*(4/np.pi)*np.sum((uu_solid_ - convectionx_solid_) *(h_solid_**3))
            c_z = 2*(4/np.pi)*np.sum((ww_solid_ - convectionz_solid_) *(h_solid_**3))
            cd_container[1].append(c_x)
            print(' CD = ', c_x)
            print(' CL = ', c_z)
            print(' CD = ', c_x, file=log_printing)
            print(' CL = ', c_z, file=log_printing)

        #* 2. Pressure-correction step
        #! Here we solve for phi (phi = p(k+1)-p(k)+nu*div(V_tilde(k+1))
        # construct rhs: div(V)
        divV_   = dx_full_2d.dot(u_) + dy_full_2d.dot(v_) + dz_full_2d.dot(w_)
        # divV_   = dx_xneg_dcpse_full_2d.dot(u_) + dy_yneg_dcpse_full_2d.dot(v_) + dz_zneg_dcpse_full_2d.dot(w_)
        rhs_    = 3/(2*DT) * ( divV_[n_bound:] )
        rhs_    = np.concatenate((rhs_phi_, rhs_))
        # save p(k)
        p_now_ = np.copy(p_)
        # solve the linear system
        # phi_, info = linalg.gmres(poisson_2d, rhs_ )
        # phi_, info = linalg.bicgstab(poisson_2d, rhs_ , x0=phi_, tol=1e-05)
        # phi_ = linalg.spsolve(poisson_2d, rhs_ )
        phi_ = poisson_2d(rhs_)
        # phi_ = poisson_2d.solve(rhs_)
        # phi_ = tobsolve_lu(rhs_, L, U, perm_r, perm_c)
        p_ =  phi_ + p_now_ - nu*divV_
        # print('info: ',info)

        #! here the boundary velocities are not updated by the pressure correction
        u_[n_bound:] = u_[n_bound:] - 2/3*DT* dx_2d.dot(phi_)
        v_[n_bound:] = v_[n_bound:] - 2/3*DT* dy_2d.dot(phi_)
        w_[n_bound:] = w_[n_bound:] - 2/3*DT* dz_2d.dot(phi_)
        # u_ = u_ - 2/3*DT* dx_full_2d.dot(phi_)
        # v_ = v_ - 2/3*DT* dy_full_2d.dot(phi_)
        # w_ = w_ - 2/3*DT* dz_full_2d.dot(phi_)

        if not brinkman and sphere:
            Fx_rho_ = -p_[idx_begin:idx_end] * -normal_x_sp_ + nu* (2*(dx_sp_2d@u_)*-normal_x_sp_ \
                    + (dy_sp_2d@u_ + dx_sp_2d@v_)*-normal_y_sp_ \
                    + (dz_sp_2d@u_ + dx_sp_2d@w_)*-normal_z_sp_) 
            Fx_rho = Fx_rho_.sum() * min(h_)**2
            cd = 2*Fx_rho*4/np.pi # Johnson
            cd_container[1].append(cd)
            print(f' CD = {cd}')
            print(f' CD = {cd}', file=log_printing)

        print(' max u = ', max(u_))
        # saving velocities and pressure
        V_2d = np.concatenate((u_.reshape(-1,1),v_.reshape(-1,1),w_.reshape(-1,1)),axis=1)
        np.save("current_result\\V_2d.npy", V_2d)
        np.save("current_result\\p_.npy", p_)
        # saving old velocities and pressure
        V_old_2d = np.concatenate((u_old_.reshape(-1,1),v_old_.reshape(-1,1),w_old_.reshape(-1,1)),axis=1)
        np.save("current_result\\V_old_2d.npy", V_old_2d)
        np.save("current_result\\p_now_.npy", p_now_)

        
        # end of time step
        t_done += DT
        cd_container[0].append(t_done)
        print(f"t = {t_done} done")
        print(f"t = {t_done} done", file=log_printing)
        # save t_done
        with open('current_result\\t_done.pkl', 'wb') as f:
            pickle.dump(t_done, f)

        # # Cavity plotting
        # post.get_profile('x', 0.5, 'y', 0.5, u_, nodes_2d)
        # post.get_profile('y', 0.5, 'z', 0.5, w_, nodes_2d) 

        # # Sphere plotting
        # try:
        #     post.get_profile('x', 4.5, 'y', 4.5, u_, nodes_2d)
        #     post.get_profile('y', 4.5, 'z', 4.5, w_, nodes_2d)
        # except:
        #     post.get_profile('x', 4.4, 'y', 4.4, u_, nodes_2d)
        #     post.get_profile('y', 4.4, 'z', 4.4, w_, nodes_2d)

        # plt.show(block=True)

    with open('current_result\\cd_container.txt', 'w') as file:
        json.dump(cd_container, file)

    log_printing.close() # end of log printing

    return u_, v_, w_, p_

def get_dn_operator(idx_begin, idx_end, dx_2d, dy_2d, dz_2d, normal_x_bound_, normal_y_bound_, normal_z_bound_ ):
    """get d()/dn operator at a particular boundary

    Args:
        idx_begin (int): first index of the particle in the boundary
        idx_end (int): index of the last particle in the boundary
        dx_2d (2d sparse): x-deriv operator
        dy_2d (2d sparse): y-deriv operator
        dz_2d (2d sparse): z-deriv operator
        normal_x_bound_ (1d np.array): x-component of boundary normal
        normal_y_bound_ (1d np.array): y-component of boundary normal
        normal_z_bound_ (1d np.array): z-component of boundary normal

    Returns:
        2d sparse scipy array: normal deriv operator
    """
    return dx_2d[idx_begin:idx_end].multiply(normal_x_bound_[idx_begin:idx_end]) \
        + dy_2d[idx_begin:idx_end].multiply(normal_y_bound_[idx_begin:idx_end]) \
            +  dz_2d[idx_begin:idx_end].multiply(normal_z_bound_[idx_begin:idx_end])




# NEED TO BE REPAIRED (UNUSED)
def noninc_pc(contd, V0_2d, nu, DT, TF, dx_2d, dy_2d, dz_2d, dxx_2d, dyy_2d, dzz_2d, u_bound_2d, v_bound_2d, w_bound_2d, p_bound_2d, rhs_u_, rhs_v_,  rhs_w_, rhs_p_, n_bound, brinkman, eta, nodes_2d):
    """Non-incremental pressure-correction algorithm for incompressible Navier-Stokes.
    with irreducible splitting error order of O(dt)
    PRESSURE BC: dp/dn = 0

    Args:
        V0_2d (2d array): Initial velocity vector [u v w]
        nu (float): kinematic viscosity
        DT (float): time increment
        TF (float): final time
        [[Operators below are for INNER PARTICLES ONLY!!]]
        dx_2d (2d sparse matrix): x-derivative operator
        dy_2d (2d sparse matrix): y-derivative operator
        dz_2d (2d sparse matrix): z-derivative operator
        dxx_2d (2d sparse matrix): xx-derivative operator
        dyy_2d (2d sparse matrix): yy-derivative operator
        dzz_2d (2d sparse matrix): zz-derivative operator
        u_bound_2d (2d sparse matrix): boundary operator for u-velocity 
        v_bound_2d (2d sparse matrix): boundary operator for v-velocity 
        w_bound_2d (2d sparse matrix): boundary operator for w-velocity 
        p_bound_2d (2d sparse matrix): boundary operator for pressure
        rhs_u_ (1d array): rhs for u-velocity
        rhs_v_ (1d array): rhs for v-velocity
        rhs_w_ (1d array): rhs for w-velocity
        rhs_p_ (1d array): rhs for p-velocity
        n_bound (int): number of boundary particles

    Output:
        u_(1d array): u-velocity at the end of simulation time
        v_(1d array): v-velocity at the end of simulation time
        w_(1d array): w-velocity at the end of simulation time
        p_(1d array): pressure at the end of simulation time
    """
    info = "direct-solver" # dummy (if direct method is used)
    #! remember that the derivatives operator are for inner particles only [n_bound:]
    n_total = V0_2d.shape[0]
    n_inner = n_total - n_bound
    
    t_done = 0 # current time in the loop
    u_ = V0_2d[:,0] # initial-condition
    v_ = V0_2d[:,1]
    w_ = V0_2d[:,2]

    I_2d = sparse.identity(n_total).tocsr() # identity matrix
    Ddrag_2d = sparse.csr_matrix((n_inner, n_total), dtype=float) # initialize zero sparse matrix
    
    # construct matrix corresponding to diffusion
    diff_2d = I_2d[n_bound:] - DT*nu* (dxx_2d + dyy_2d  + dzz_2d)

    if brinkman:
        # cylinder
        with open('geom.txt','r') as file:
            geom = json.load(file)
        h_ = np.load('h_.npy')
        X0 = geom['X0'] 
        Y0 = geom['Y0'] 
        Z0 = geom['Z0'] 
        RAD = geom['RAD']
        solid_ = (nodes_2d[:,0]-X0)**2 + (nodes_2d[:,1]-Y0)**2 + (nodes_2d[:,2]-Z0)**2 <= (RAD+1e-13)**2 
        Ddrag_2d = get_darcy_drag(eta, nodes_2d[n_bound:], X0, Y0, Z0, RAD, I_2d[n_bound:])        

    # construct matrix for pressure-poisson eqn
    poisson_2d = dxx_2d + dyy_2d + dzz_2d
    poisson_2d = sparse.vstack([p_bound_2d, poisson_2d])
    # poisson_2d = linalg.splu(poisson_2d.tocsc())
    
    if contd:
        # importing velocities and pressure
        V_2d = np.load("contd\\V_2d.npy")
        u_, v_, w_ = V_2d[:,0], V_2d[:,1], V_2d[:,2]
        p_ = np.load("contd\\p_.npy")
        
        # import time
        with open('contd\\t_done.pkl','rb') as f:
            t_done = pickle.load(f)

    while t_done < TF -1e-10:
        
        # 1. Semi-implicit convection-diffusion
        # obtain inner velocity and reshape
        in_u_ = u_[n_bound:].reshape(n_inner,1)
        in_v_ = v_[n_bound:].reshape(n_inner,1)
        in_w_ = w_[n_bound:].reshape(n_inner,1)
        # construct the LHS matrix
        in_LHS_2d = diff_2d + DT* (dx_2d.multiply(in_u_) + dy_2d.multiply(in_v_) + dz_2d.multiply(in_w_) + Ddrag_2d)

        LHS_2d = sparse.vstack([u_bound_2d, in_LHS_2d])
        rhs_            = np.copy(u_)    # construct rhs
        rhs_[:n_bound]  = rhs_u_
        # u_ = linalg.spsolve(LHS_2d, rhs_)
        u_, info = linalg.bicgstab(LHS_2d, rhs_, x0=u_, tol=1e-07)
        print('info_u: ',info)
        
        LHS_2d = sparse.vstack([v_bound_2d, in_LHS_2d])
        rhs_            = np.copy(v_)    # construct rhs
        rhs_[:n_bound]  = rhs_v_
        # v_ = linalg.spsolve(LHS_2d, rhs_)
        v_, info = linalg.bicgstab(LHS_2d, rhs_, x0=v_, tol=1e-07)
        print('info_v: ',info)
        
        LHS_2d = sparse.vstack([w_bound_2d, in_LHS_2d])
        rhs_            = np.copy(w_)    # construct rhs
        rhs_[:n_bound]  = rhs_w_
        # w_ = linalg.spsolve(LHS_2d, rhs_)
        w_, info = linalg.bicgstab(LHS_2d, rhs_, x0=w_, tol=1e-07)
        print('info_w: ',info)

        if brinkman:
            #? force computation
            u_solid_ = u_[solid_]
            h_solid_ = h_[solid_]
            c_x = 2/eta*np.sum(u_solid_*(h_solid_**3))
            print('Cd_1 : ', c_x)

        # 3. Pressure-correction 
        rhs_    = 1/DT * ( dx_2d.dot(u_) + dy_2d.dot(v_) + dz_2d.dot(w_) )
        rhs_    = np.concatenate((rhs_p_, rhs_))
        p_ = linalg.spsolve(poisson_2d, rhs_ )
        # p_, info = linalg.bicgstab(poisson_2d, rhs_, tol=1e-05)
        print('info_p: ',info)
        # p_ = poisson_2d.solve(rhs_)
        
        #! BEWARE: here the boundary velocities are not updated by the pressure
        u_[n_bound:] = u_[n_bound:] - DT* dx_2d.dot(p_)
        v_[n_bound:] = v_[n_bound:] - DT* dy_2d.dot(p_)
        w_[n_bound:] = w_[n_bound:] - DT* dz_2d.dot(p_)

        print('max u = ', max(u_))
        # saving velocities and pressure
        V_2d = np.concatenate((u_.reshape(-1,1),v_.reshape(-1,1),w_.reshape(-1,1)),axis=1)
        np.save("current_result\\V_2d.npy", V_2d)
        np.save("current_result\\p_.npy", p_)
        
        t_done += DT
        print(f"t = {t_done} done")

        # save t_done
        with open('current_result\\t_done.pkl', 'wb') as f:
            pickle.dump(t_done, f)

    return u_, v_, w_, p_

# NEED TO BE REPAIRED (UNUSED)
def stdinc_pc(contd, V0_2d, nu, DT, TF, dx_2d, dy_2d, dz_2d, dxx_2d, dyy_2d, dzz_2d, u_bound_2d, v_bound_2d, w_bound_2d, p_bound_2d, rhs_u_, rhs_v_, rhs_w_, rhs_p_, n_bound, brinkman, eta, nodes_2d):
    """Standard-incremental pressure-correction algorithm for incompressible Navier-Stokes.
    with irreducible splitting error order of O(dt)
    PRESSURE BC: dp/dn = dp(0)/dn

    Args:
        V0_2d (2d array): Initial velocity vector [u v]
        nu (float): kinematic viscosity
        DT (float): time increment
        TF (float): final time
        [[Operators below are for INNER PARTICLES ONLY!!]]
        dx_2d (2d sparse matrix): x-derivative operator
        dy_2d (2d sparse matrix): y-derivative operator
        dz_2d (2d sparse matrix): z-derivative operator
        dxx_2d (2d sparse matrix): xx-derivative operator
        dyy_2d (2d sparse matrix): yy-derivative operator
        dzz_2d (2d sparse matrix): zz-derivative operator
        u_bound_2d (2d sparse matrix): boundary operator for velocity 
        v_bound_2d (2d sparse matrix): boundary operator for velocity 
        w_bound_2d (2d sparse matrix): boundary operator for w-velocity 
        p_bound_2d (2d sparse matrix): boundary operator for pressure
        rhs_u_ (1d array): rhs for u-velocity
        rhs_v_ (1d array): rhs for v-velocity
        rhs_w_ (1d array): rhs for w-velocity
        rhs_p_ (1d array): rhs for p-velocity
        n_bound (int): number of boundary particles

    Output:
        u_(1d array): u-velocity at the end of simulation time
        v_(1d array): v-velocity at the end of simulation time
        w_(1d array): w-velocity at the end of simulation time
        p_(1d array): pressure at the end of simulation time
    """
    info = "direct-solver" # dummy (if direct method is used)
    #! remember that the derivatives operator are for inner particles only [n_bound:]
    n_total = V0_2d.shape[0]
    n_inner = n_total - n_bound

    I_2d = sparse.identity(n_total).tocsr() # identity matrix
    Ddrag_2d = sparse.csr_matrix((n_inner, n_total), dtype=float) # initialize zero sparse matrix
    
    # construct matrix for pressure-poisson eqn
    laplace_2d = dxx_2d + dyy_2d + dzz_2d
    poisson_2d = sparse.vstack([p_bound_2d, laplace_2d])
    # poisson_2d = linalg.splu(poisson_2d.tocsc())

    if brinkman:
        # cylinder
        with open('geom.txt','r') as file:
            geom = json.load(file)
        h_ = np.load('h_.npy')
        X0 = geom['X0'] 
        Y0 = geom['Y0'] 
        Z0 = geom['Z0'] 
        RAD = geom['RAD']
        solid_ = (nodes_2d[:,0]-X0)**2 + (nodes_2d[:,1]-Y0)**2 + (nodes_2d[:,2]-Z0)**2 <= (RAD+1e-13)**2 
        Ddrag_2d = get_darcy_drag(eta, nodes_2d[n_bound:], X0, Y0, Z0, RAD, I_2d[n_bound:])        

    if not contd:
        t_done = 0 # current time in the loop
        u_ = V0_2d[:,0] # initial-condition
        v_ = V0_2d[:,1]
        w_ = V0_2d[:,2]   
    
        # save u(k-1)
        u_old_ = np.copy(u_)
        v_old_ = np.copy(v_)
        w_old_ = np.copy(w_)

        # Compute the first time step using first-order pressure correction scheme
        #! Assuming that the initial velocity are uniform and divergence-free, THUS  p(0) = 0
        # construct matrix corresponding to diffusion (1st order time stepping)
        diff_2d = I_2d[n_bound:] - DT*nu* (dxx_2d + dyy_2d + dzz_2d)
    
        # 1. Semi-implicit convection-diffusion
        # obtain inner velocity and reshape
        in_u_ = u_[n_bound:].reshape(n_inner,1)
        in_v_ = v_[n_bound:].reshape(n_inner,1)
        in_w_ = w_[n_bound:].reshape(n_inner,1)
        # construct the LHS matrix (adding convection)
        in_LHS_2d = diff_2d + DT* (dx_2d.multiply(in_u_) + dy_2d.multiply(in_v_)  + dz_2d.multiply(in_w_) + Ddrag_2d)

        LHS_2d = sparse.vstack([u_bound_2d, in_LHS_2d])
        rhs_            = np.copy(u_)    # construct rhs
        rhs_[:n_bound]  = rhs_u_
        # u_ = linalg.spsolve(LHS_2d, rhs_)
        # u_, info = linalg.gmres(LHS_2d, rhs_)
        u_, info = linalg.bicgstab(LHS_2d, rhs_, x0=u_, tol=1e-07)
        print('info_u: ',info)

        LHS_2d = sparse.vstack([v_bound_2d, in_LHS_2d])
        rhs_            = np.copy(v_)    # construct rhs
        rhs_[:n_bound]  = rhs_v_
        # v_ = linalg.spsolve(LHS_2d, rhs_)
        # v_, info = linalg.gmres(LHS_2d, rhs_)
        v_, info = linalg.bicgstab(LHS_2d, rhs_, x0=v_, tol=1e-07)
        print('info_v: ',info)

        LHS_2d = sparse.vstack([w_bound_2d, in_LHS_2d])
        rhs_            = np.copy(w_)    # construct rhs
        rhs_[:n_bound]  = rhs_w_
        # w_ = linalg.spsolve(LHS_2d, rhs_)
        # w_, info = linalg.gmres(LHS_2d, rhs_)
        w_, info = linalg.bicgstab(LHS_2d, rhs_, x0=w_, tol=1e-07)
        print('info_w: ',info)

        # 2. Pressure-correction 
        rhs_    = 1/DT * ( dx_2d.dot(u_) + dy_2d.dot(v_) + dz_2d.dot(w_) )
        rhs_    = np.concatenate((rhs_p_, rhs_)) #! beware here because here we solve for p whereas the input is for phi
        # p_ = linalg.spsolve(poisson_2d, rhs_ )
        # p_, info = linalg.gmres(poisson_2d, rhs_ )
        p_, info = linalg.bicgstab(poisson_2d, rhs_ , tol=1e-05)
        # p_ = poisson_2d.solve(rhs_)
        print('info: ',info)

        #! BEWARE: here the boundary velocities are not updated by the pressure
        u_[n_bound:] = u_[n_bound:] - DT* dx_2d.dot(p_)
        v_[n_bound:] = v_[n_bound:] - DT* dy_2d.dot(p_)
        w_[n_bound:] = w_[n_bound:] - DT* dz_2d.dot(p_)

        t_done += DT
    
    else:
        # importing velocities and pressure
        V_2d = np.load("contd\\V_2d.npy")
        u_, v_, w_ = V_2d[:,0], V_2d[:,1], V_2d[:,2]
        p_ = np.load("contd\\p_.npy")
        # importing old velocities and pressure
        V_old_2d = np.load("contd\\V_old_2d.npy")
        u_old_, v_old_, w_old_ = V_old_2d[:,0], V_old_2d[:,1], V_old_2d[:,2]
        p_now_ = np.load("contd\\p_now_.npy")   
        
        # import time
        with open('contd\\t_done.pkl','rb') as f:
            t_done = pickle.load(f)

    # construct matrix corresponding to diffusion (2nd order SCHEME)
    diff_2d = I_2d[n_bound:] - 2/3*DT*nu* (dxx_2d + dyy_2d + dzz_2d)

    while t_done < TF -1e-10:

        # 1. Semi-implicit convection-diffusion
        # obtain inner velocity and reshape
        in_u_ = u_[n_bound:].reshape(n_inner,1)
        in_v_ = v_[n_bound:].reshape(n_inner,1)
        in_w_ = w_[n_bound:].reshape(n_inner,1)
        # construct the LHS matrix
        in_LHS_2d = diff_2d + 2/3*DT* (dx_2d.multiply(in_u_) + dy_2d.multiply(in_v_) + dz_2d.multiply(in_w_) + Ddrag_2d)

        LHS_2d = sparse.vstack([u_bound_2d, in_LHS_2d])
        rhs_    = 4/3*u_[n_bound:] - 1/3*u_old_[n_bound:] - 2/3*DT* dx_2d.dot(p_)  # construct rhs
        rhs_    = np.concatenate((rhs_u_, rhs_))
        # save u(k-1)
        u_old_ = np.copy(u_)
        # solve the linear system
        # u_ = linalg.spsolve(LHS_2d, rhs_)
        u_, info = linalg.bicgstab(LHS_2d, rhs_, x0=u_, tol=1e-07)
        print('info_u: ',info)

        LHS_2d = sparse.vstack([v_bound_2d, in_LHS_2d])
        rhs_    = 4/3*v_[n_bound:] - 1/3*v_old_[n_bound:] - 2/3*DT* dy_2d.dot(p_)  # construct rhs
        rhs_    = np.concatenate((rhs_v_, rhs_))
        # save v(k-1)
        v_old_ = np.copy(v_)
        # solve the linear system
        # v_ = linalg.spsolve(LHS_2d, rhs_)
        v_, info = linalg.bicgstab(LHS_2d, rhs_, x0=v_, tol=1e-07)
        print('info_v: ',info)

        LHS_2d = sparse.vstack([w_bound_2d, in_LHS_2d])
        rhs_    = 4/3*w_[n_bound:] - 1/3*w_old_[n_bound:] - 2/3*DT* dz_2d.dot(p_)  # construct rhs
        rhs_    = np.concatenate((rhs_w_, rhs_))
        # save w(k-1)
        w_old_ = np.copy(w_)
        # solve the linear system
        # w_ = linalg.spsolve(LHS_2d, rhs_)
        w_, info = linalg.bicgstab(LHS_2d, rhs_, x0=w_, tol=1e-07)
        print('info_w: ',info)
        
        if brinkman:
            #? force computation
            u_solid_ = u_[solid_]
            h_solid_ = h_[solid_]
            c_x = 2/eta*np.sum(u_solid_*(h_solid_**3))
            print('Cd_1 : ', c_x)

        # 3. Pressure-correction 
        rhs_    = laplace_2d.dot(p_) + 3/(2*DT) * ( dx_2d.dot(u_) + dy_2d.dot(v_) + dz_2d.dot(w_)  )
        rhs_    = np.concatenate((rhs_p_, rhs_))
        # save p(k)
        p_now_ = np.copy(p_)
        # solve the linear system
        # p_ = linalg.spsolve(poisson_2d, rhs_ )
        # p_ = poisson_2d.solve(rhs_)
        p_, info = linalg.bicgstab(poisson_2d, rhs_, tol=1e-05)
        print('info_p: ',info)

        #! BEWARE: here the boundary velocities are not updated by the pressure
        del_p = p_-p_now_ # delta p
        u_[n_bound:] = u_[n_bound:] - 2/3*DT* dx_2d.dot(del_p)
        v_[n_bound:] = v_[n_bound:] - 2/3*DT* dy_2d.dot(del_p)
        w_[n_bound:] = w_[n_bound:] - 2/3*DT* dz_2d.dot(del_p)

        print('max u = ', max(u_))
        # saving velocities and pressure
        V_2d = np.concatenate((u_.reshape(-1,1),v_.reshape(-1,1),w_.reshape(-1,1)),axis=1)
        np.save("current_result\\V_2d.npy", V_2d)
        np.save("current_result\\p_.npy", p_)
        # saving old velocities and pressure
        V_old_2d = np.concatenate((u_old_.reshape(-1,1),v_old_.reshape(-1,1),w_old_.reshape(-1,1)),axis=1)
        np.save("current_result\\V_old_2d.npy", V_old_2d)
        np.save("current_result\\p_now_.npy", p_now_)

        t_done += DT
        print(f"t = {t_done} done")
        # save t_done
        with open('current_result\\t_done.pkl', 'wb') as f:
            pickle.dump(t_done, f)

    return u_, v_, w_, p_

# UNUSED
def tobsolve_lu(b, L, U, perm_r, perm_c): 
    """solve Ax=b given LU decomposition of matrix A
    Pr A Pc = L U
    Pr.t L U Pc.t = A
    
    Ax = b
    L (U Pc.t x) = Pr b
    L y = Pr b
    y = solve(L, Pr b)
    U Pc.t x = y
    x = Pc solve(U, y) 

    Args:
        b (np 1darray): rhs
        L (sparse 2darray): Lower
        U (sparse 2darray): Upper
        perm_r (np 1darray): first permutation
        perm_c (np 1darray): last permutation

    Returns:
        x: 1darray (solution of linear system)
    """
    # y = linalg.spsolve(L, b[perm_r])
    # x = linalg.spsolve(U, y)[perm_c]
    y = linalg.spsolve_triangular(L, b[perm_r], lower=True, unit_diagonal=True)
    x = linalg.spsolve_triangular(U, y, lower=False)[perm_c]

    return x













































































#! NOT USED
#! added nodes_2d to the input argument.. it's used only for post 
def rotinc_pc_contd(V_2d, V_old_2d, p_, p_now_, tcontd, nu, DT, TF, dx_full_2d, dy_full_2d, dz_full_2d, dxx_2d, dyy_2d, dzz_2d, u_bound_2d, v_bound_2d, w_bound_2d, phi_bound_2d, rhs_u_, rhs_v_, rhs_w_, rhs_phi_, n_bound, nodes_2d):
    """CONT'D Rotational-incremental pressure-correction algorithm for incompressible Navier-Stokes.
    with irreducible splitting error order of O(dt^2)
    PRESSURE BC: dp/dn = del x del x V

    Args:
        V0_2d (2d array): Initial velocity vector [u v]
        nu (float): kinematic viscosity
        DT (float): time increment
        TF (float): final time
        dx_bound_2d (2d sparse matrix): x-derivative operator for boundary particles
        dy_bound_2d (2d sparse matrix): y-derivative operator for boundary particles
        [[Operators below are for INNER PARTICLES ONLY!!]]
        dx_2d (2d sparse matrix): x-derivative operator
        dy_2d (2d sparse matrix): y-derivative operator
        dz_2d (2d sparse matrix): z-derivative operator
        dxx_2d (2d sparse matrix): xx-derivative operator
        dyy_2d (2d sparse matrix): yy-derivative operator
        dzz_2d (2d sparse matrix): zz-derivative operator
        u_bound_2d (2d sparse matrix): boundary operator for velocity 
        v_bound_2d (2d sparse matrix): boundary operator for velocity 
        w_bound_2d (2d sparse matrix): boundary operator for w-velocity 
        Phi_bound_2d (2d sparse matrix): boundary operator for pressure
        rhs_u_ (1d array): rhs for u-velocity
        rhs_v_ (1d array): rhs for v-velocity
        rhs_w_ (1d array): rhs for w-velocity
        rhs_phi_ (1d array): rhs for phi
        n_bound (int): number of boundary particles

    Output:
        u_(1d array): u-velocity at the end of simulation time
        v_(1d array): v-velocity at the end of simulation time
        w_(1d array): w-velocity at the end of simulation time
        p_(1d array): pressure at the end of simulation time
    """
    info = "direct-solver" # dummy (if direct method is used)
    #! remember that the derivatives operator are for inner particles only [n_bound:]

    #  except
    dx_2d = dx_full_2d[n_bound:]
    dy_2d = dy_full_2d[n_bound:]
    dz_2d = dz_full_2d[n_bound:]

    n_total = V_2d.shape[0]
    n_inner = n_total - n_bound
    
    I_2d = sparse.identity(n_total).tocsr() # identity matrix
    # construct matrix for pressure-poisson eqn
    poisson_2d = dxx_2d + dyy_2d + dzz_2d
    poisson_2d = sparse.vstack([phi_bound_2d, poisson_2d])
    
    # the contd code started here
    # importing velocitie components
    u_old_, v_old_, w_old_ = V_old_2d[:,0], V_old_2d[:,1], V_old_2d[:,2]
    u_, v_, w_ = V_2d[:,0], V_2d[:,1], V_2d[:,2]

    # computing phi
    divV_   = dx_full_2d.dot(u_) + dy_full_2d.dot(v_) + dz_full_2d.dot(w_)
    phi_ = p_ - p_now_ + nu*divV_
    
    t_done = tcontd # current time in the loop

    
    # poisson_2d = linalg.splu(poisson_2d.tocsc())
    # construct matrix corresponding to diffusion (2nd order SCHEME)
    diff_2d = I_2d[n_bound:] - 2/3*DT*nu* (dxx_2d + dyy_2d + dzz_2d)

    while t_done < TF -1e-10:

        # 1. Semi-implicit convection-diffusion
        # obtain inner velocity and reshape
        in_u_ = u_[n_bound:].reshape(n_inner,1)
        in_v_ = v_[n_bound:].reshape(n_inner,1)
        in_w_ = w_[n_bound:].reshape(n_inner,1)
        # construct the LHS matrix
        in_LHS_2d = diff_2d + 2/3*DT* (dx_2d.multiply(in_u_) + dy_2d.multiply(in_v_) + dz_2d.multiply(in_w_))

        LHS_2d = sparse.vstack([u_bound_2d, in_LHS_2d])
        # construct rhs: u(k), u(k-1), (dp/dx)
        rhs_    = 4/3*u_[n_bound:] - 1/3*u_old_[n_bound:] - 2/3*DT* dx_2d.dot(p_)
        rhs_    = np.concatenate((rhs_u_, rhs_))
        # save u(k-1)
        u_old_ = np.copy(u_)
        # solve the linear system
        # u_ = linalg.spsolve(LHS_2d, rhs_)
        # u_, info = linalg.gmres(LHS_2d, rhs_)
        u_, info = linalg.bicgstab(LHS_2d, rhs_, x0=u_, tol=1e-07)
        print('info_u: ',info)

        LHS_2d = sparse.vstack([v_bound_2d, in_LHS_2d])
        # construct rhs: v(k), v(k-1), (dp/dy)
        rhs_    = 4/3*v_[n_bound:] - 1/3*v_old_[n_bound:] - 2/3*DT* dy_2d.dot(p_)
        rhs_    = np.concatenate((rhs_v_, rhs_))
        # save v(k-1)
        v_old_ = np.copy(v_)
        # solve the linear system
        # v_ = linalg.spsolve(LHS_2d, rhs_)
        # v_, info = linalg.gmres(LHS_2d, rhs_)
        v_, info = linalg.bicgstab(LHS_2d, rhs_, x0=v_, tol=1e-07)
        print('info_v: ',info)

        LHS_2d = sparse.vstack([w_bound_2d, in_LHS_2d])
        rhs_    = 4/3*w_[n_bound:] - 1/3*w_old_[n_bound:] - 2/3*DT* dz_2d.dot(p_)  # construct rhs
        rhs_    = np.concatenate((rhs_w_, rhs_))
        # save w(k-1)
        w_old_ = np.copy(w_)
        # solve the linear system
        # w_ = linalg.spsolve(LHS_2d, rhs_)
        # w_, info = linalg.gmres(LHS_2d, rhs_)
        w_, info = linalg.bicgstab(LHS_2d, rhs_, x0=w_, tol=1e-07)
        print('info_w: ',info)

        # 3. Pressure-correction
        #! Here we solve for phi (phi = p(k+1)-p(k)+nu*div(V_tilde(k+1))
        # construct rhs: div(V)
        divV_   = dx_full_2d.dot(u_) + dy_full_2d.dot(v_) + dz_full_2d.dot(w_)
        rhs_    = 3/(2*DT) * ( divV_[n_bound:] )
        rhs_    = np.concatenate((rhs_phi_, rhs_))
        # save p(k)
        p_now_ = np.copy(p_)
        # solve the linear system
        # phi_ = linalg.spsolve(poisson_2d, rhs_ )
        # phi_, info = linalg.gmres(poisson_2d, rhs_ )
        phi_, info = linalg.bicgstab(poisson_2d, rhs_ , x0=phi_, tol=1e-05)
        print('info: ',info)
        # phi_ = poisson_2d.solve(rhs_)
        p_ =  phi_ + p_now_ - nu*divV_
        # print('phi done')

        #! BEWARE: here the boundary velocities are not updated by the pressure
        u_[n_bound:] = u_[n_bound:] - 2/3*DT* dx_2d.dot(phi_)
        v_[n_bound:] = v_[n_bound:] - 2/3*DT* dy_2d.dot(phi_)
        w_[n_bound:] = w_[n_bound:] - 2/3*DT* dz_2d.dot(phi_)
        # u_ = u_ - 2/3*DT* dx_full_2d.dot(phi_)
        # v_ = v_ - 2/3*DT* dy_full_2d.dot(phi_)
        # w_ = w_ - 2/3*DT* dz_full_2d.dot(phi_)

        # saving velocities and pressure
        V_2d = np.concatenate((u_.reshape(-1,1),v_.reshape(-1,1),w_.reshape(-1,1)),axis=1)
        np.save("cu_result\V_2d.npy", V_2d)
        np.save("cu_result\p_.npy", p_)
        # saving old velocities and pressure
        V_old_2d = np.concatenate((u_old_.reshape(-1,1),v_old_.reshape(-1,1),w_old_.reshape(-1,1)),axis=1)
        np.save("cu_result\V_old_2d.npy", V_old_2d)
        np.save("cu_result\p_now_.npy", p_now_)

        t_done += DT
        print(f"t = {t_done} done")


        # # Cavity plotting
        # post.get_profile('x', 0.5, 'y', 0.5, u_, nodes_2d)
        # post.get_profile('y', 0.5, 'z', 0.5, w_, nodes_2d) 

        # # Sphere velocity profiles
        # if (4.5 % h) < 1e-8:
        #     post.get_profile('x', 4.5, 'y', 4.5, u_, nodes_2d)
        #     post.get_profile('y', 4.5, 'z', 4.5, w_, nodes_2d)
        # else:
        #     post.get_profile('x', 4.4, 'y', 4.4, u_, nodes_2d)
        #     post.get_profile('y', 4.4, 'z', 4.4, w_, nodes_2d)

        # plt.show(block=True)

    return u_, v_, w_, p_