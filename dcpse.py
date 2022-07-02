import numpy as np
from scipy import sparse
from scipy import linalg
from scipy.special import factorial
from scipy.linalg import LinAlgWarning
from matplotlib import pyplot as plt
import json
import warnings

def get_deriv(i, nodes_2d, neighbors_i_, eps, P, typ, i_, j_, dx_=[], dy_=[], dz_=[], dxx_=[], dyy_=[], dzz_=[]):

    typ_all = typ == 'all'

    # define multi-index [Lx2]
    if P == 1:
        alpha_2d = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=int)
    elif P == 2:
        alpha_2d = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [2,0,0], [0,2,0], [0,0,2]], dtype=int)
    elif P == 3:
        alpha_2d = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], \
            [2,0,0], [0,2,0], [0,0,2], [1,1,1], [2,1,0], [2,0,1], [1,2,0], [0,2,1], [1,0,2], [0,1,2], \
                [3,0,0], [0,3,0], [0,0,3]], dtype=int)

    # Define b
    if P == 1:
        b_dx_ = -1*np.array([0, 1, 0, 0], dtype=float)
        b_dy_ = -1*np.array([0, 0, 1, 0], dtype=float)
        b_dz_ = -1*np.array([0, 0, 0, 1], dtype=float)
        if typ_all:
            b_2d = np.concatenate((b_dx_, b_dy_, b_dz_)).reshape((-1,3), order='F')
    elif P == 2:
        b_dx_ = -1*np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        b_dy_ = -1*np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        b_dz_ = -1*np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        b_dxx_ = np.array([0, 0, 0, 0, 0, 0, 0, 2, 0, 0], dtype=float)
        b_dyy_ = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 0], dtype=float)
        b_dzz_ = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2], dtype=float)
        if typ_all:
            b_2d = np.concatenate((b_dx_, b_dy_, b_dz_, b_dxx_, b_dyy_, b_dzz_)).reshape((-1,6), order='F')
    elif P == 3:
        b_dx_ = -1*np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        b_dy_ = -1*np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        b_dz_ = -1*np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        b_dxx_ = np.array([0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        b_dyy_ = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        b_dzz_ = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        if typ_all:
            b_2d = np.concatenate((b_dx_, b_dy_, b_dz_, b_dxx_, b_dyy_, b_dzz_)).reshape((-1,6), order='F')

    # generate (x,y)_i - (x,y)_neighs / eps
    z_eps_2d = (nodes_2d[i,:] - nodes_2d[neighbors_i_,:])/eps
    len_z_eps_ = linalg.norm(z_eps_2d, axis=1)
    # Construct E
    E_diag_ = np.exp(-0.5*(len_z_eps_)**2)
    E_2d = sparse.diags(E_diag_)
    # Construct V 
    V_2d = z_eps_2d[:,0].reshape(-1,1)**alpha_2d[:,0] \
            * z_eps_2d[:,1].reshape(-1,1)**alpha_2d[:,1] \
                * z_eps_2d[:,2].reshape(-1,1)**alpha_2d[:,2]

    # Compute B
    B_2d = E_2d @ V_2d
    # Compute A
    A_2d = B_2d.transpose() @ B_2d

    # Solve system of linear eqns and compute kernels
    # NOTE: in computing kernel for every neighbor of the particle
    #       a_ @ V_2d.t equals [1 x n_poly] x [n_poly x n_neigh]  
    if typ == 'x':
        a_dx_ = linalg.solve(A_2d, b_dx_)
        kernel_dx_ = 1/eps* np.multiply(E_diag_**2, (a_dx_ @ V_2d.transpose()) ) 
    elif typ == 'y':
        a_dy_ = linalg.solve(A_2d, b_dy_)
        kernel_dy_ = 1/eps* np.multiply(E_diag_**2, (a_dy_ @ V_2d.transpose()) ) 
    elif typ == 'z':
        a_dz_ = linalg.solve(A_2d, b_dz_)
        kernel_dz_ = 1/eps* np.multiply(E_diag_**2, (a_dz_ @ V_2d.transpose()) ) 
    elif typ_all:       
        a_2d = linalg.solve(A_2d, b_2d, assume_a='pos')
        a_dx_ = a_2d[:,0]
        a_dy_ = a_2d[:,1]
        a_dz_ = a_2d[:,2]
        kernel_dx_ = 1/eps* np.multiply(E_diag_**2, (a_dx_ @ V_2d.transpose()) )
        kernel_dy_ = 1/eps* np.multiply(E_diag_**2, (a_dy_ @ V_2d.transpose()) )
        kernel_dz_ = 1/eps* np.multiply(E_diag_**2, (a_dz_ @ V_2d.transpose()) )
        if P > 1:
            a_dxx_ = a_2d[:,3]
            a_dyy_ = a_2d[:,4]
            a_dzz_ = a_2d[:,5]
            kernel_dxx_ = (1/eps**2)* np.multiply(E_diag_**2, (a_dxx_ @ V_2d.transpose()) )
            kernel_dyy_ = (1/eps**2)* np.multiply(E_diag_**2, (a_dyy_ @ V_2d.transpose()) )
            kernel_dzz_ = (1/eps**2)* np.multiply(E_diag_**2, (a_dzz_ @ V_2d.transpose()) )
    
    
    # # *Solve using Cholesky decomposition (supposedly faster computation)
    # c_, low = linalg.cho_factor(A_2d)
    
    # if typ == 'x' or typ_all:
    #     a_dx_ = linalg.cho_solve((c_, low),b_dx_)
    #     # a_dx_ = linalg.solve(A_2d, b_dx_, assume_a='pos')
    #     kernel_dx_ = 1/eps* np.multiply(E_diag_**2, (a_dx_ @ V_2d.transpose()) ) 
    # if typ == 'y' or typ_all:
    #     a_dy_ = linalg.cho_solve((c_, low),b_dy_)
    #     kernel_dy_ = 1/eps* np.multiply(E_diag_**2, (a_dy_ @ V_2d.transpose()) ) 
    # if typ == 'z' or typ_all:
    #     a_dz_ = linalg.cho_solve((c_, low),b_dz_)
    #     kernel_dz_ = 1/eps* np.multiply(E_diag_**2, (a_dz_ @ V_2d.transpose()) ) 
    # if P > 1 and typ_all:
    #     a_dxx_ = linalg.cho_solve((c_, low),b_dxx_)
    #     a_dyy_ = linalg.cho_solve((c_, low),b_dyy_)
    #     a_dzz_ = linalg.cho_solve((c_, low),b_dzz_)
    #     kernel_dxx_ = (1/eps**2)* np.multiply(E_diag_**2, (a_dxx_ @ V_2d.transpose()) )
    #     kernel_dyy_ = (1/eps**2)* np.multiply(E_diag_**2, (a_dyy_ @ V_2d.transpose()) )
    #     kernel_dzz_ = (1/eps**2)* np.multiply(E_diag_**2, (a_dzz_ @ V_2d.transpose()) ) 
    
    # store into array for constructing coo sparse matrix
    # dumj_ = neighbors_2d[i, 0:n_neigh_[i]]
    dumj_ = neighbors_i_.copy()
    dumj_ = np.append(dumj_, i)
    dumi_ = i*np.ones_like(dumj_, dtype=int)
    j_ = np.concatenate((j_, dumj_))
    i_ = np.concatenate((i_, dumi_))
    if typ == 'x' or typ_all:
        dumdx_ = np.append(kernel_dx_, (kernel_dx_.sum()) ) #positive for odd |alpha| 
        dx_ = np.concatenate((dx_, dumdx_))
    if typ == 'y' or typ_all:
        dumdy_ = np.append(kernel_dy_, (kernel_dy_.sum()) )
        dy_ = np.concatenate((dy_, dumdy_))
    if typ == 'z' or typ_all:
        dumdz_ = np.append(kernel_dz_, (kernel_dz_.sum()) )
        dz_ = np.concatenate((dz_, dumdz_))
    if P >= 2 and typ_all:
        dumdxx_ = np.append(kernel_dxx_, -(kernel_dxx_.sum()) ) #negative for even |alpha| 
        dumdyy_ = np.append(kernel_dyy_, -(kernel_dyy_.sum()) )
        dumdzz_ = np.append(kernel_dzz_, -(kernel_dzz_.sum()) ) 
        dxx_ = np.concatenate((dxx_, dumdxx_))
        dyy_ = np.concatenate((dyy_, dumdyy_)) 
        dzz_ = np.concatenate((dzz_, dumdzz_))

    # return
    if typ == 'x':
        return i_, j_, dx_
    elif typ == 'y':
        return i_, j_, dy_
    elif typ == 'z':
        return i_, j_, dz_
    elif typ_all and P>1:
        return i_, j_, dx_, dy_, dz_, dxx_, dyy_, dzz_
    else:
        return i_, j_, dx_, dy_, dz_

def dcpse(deriv_contd, P, eps_, nodes_2d, neighbors_2d, n_neigh_, n_bound, upwind):
    """DC-PSE Operator

    Args:
        P (int): order of polynomial basis
        eps (float): core size / just a parameter
        nodes_2d (2d np array): [x|y|z] nodes coordinates
        neighbors_2d (2d np array) : index of the neighbors of each particle
        n_neigh_ (1d np array) : number of neighbors for each particle\
        upwind (bool) : whether or not to compute one-sided derivatives

    Returns:
        dx_dcpse_2d (2D csr matrix) : x-derivative operator 
        dy_dcpse_2d (2D csr matrix) : y-derivative operator 
        dz_dcpse_2d (2D csr matrix) : z-derivative operator 
        if P>=2: also returns
        dxx_dcpse_2d (2D csr matrix) : x^2-derivative operator 
        dyy_dcpse_2d (2D csr matrix) : y^2-derivative operator 
        dzz_dcpse_2d (2D csr matrix) : z^2-derivative operator  
    
    Note that:
        P = 1, gives error order r = 1 for 1st derivatives
        P = 2, gives error order r = 2,1 for 1st, 2nd derivatives, respectively
        P = 3, gives error order r = 3,2 for 1st, 2nd derivatives, respectively
    """
    warnings.filterwarnings(action='error', category=LinAlgWarning) # raise LinAlgWarning: ill-conditioned matrix

    n_total = nodes_2d.shape[0] # total number of nodes

    # load upwinding neighbor lists
    if upwind:
        with open('neighbors_xneg_2d.txt', 'r') as file:
            neighbors_xneg_2d = json.load(file)
        with open('neighbors_xpos_2d.txt', 'r') as file:
            neighbors_xpos_2d = json.load(file)
        with open('neighbors_yneg_2d.txt', 'r') as file:
            neighbors_yneg_2d = json.load(file)
        with open('neighbors_ypos_2d.txt', 'r') as file:
            neighbors_ypos_2d = json.load(file)
        with open('neighbors_zneg_2d.txt', 'r') as file:
            neighbors_zneg_2d = json.load(file)
        with open('neighbors_zpos_2d.txt', 'r') as file:
            neighbors_zpos_2d = json.load(file)
    
    # Initialize vectors to construct derivative kernels
    if deriv_contd:
        # handle continuation of deriv_computation
        j_ = np.load("contd\\j_.npy")
        i_ = np.load("contd\\i_.npy")
        i_contd = int(i_[-1]) + 1
        # print('i_contd', i_contd)
        dx_ = np.load("contd\\dx_.npy")
        dy_ = np.load("contd\\dy_.npy")
        dz_ = np.load("contd\\dz_.npy")
        if P >= 2:
            dxx_ = np.load("contd\\dxx_.npy")
            dyy_ = np.load("contd\\dyy_.npy")
            dzz_ = np.load("contd\\dzz_.npy")
    else:
        # Initialize array to be used in constructing sparse matrix
        i_ = np.array([], dtype=int)
        j_ = np.array([], dtype=int)
        dx_ = np.array([], dtype=float)
        dy_ = np.array([], dtype=float)
        dz_ = np.array([], dtype=float)
        if P >= 2:
            dxx_ = np.array([], dtype=float)
            dyy_ = np.array([], dtype=float)
            dzz_ = np.array([], dtype=float)
        if upwind:
            # initialize upwindings
            i_xneg_, j_xneg_, dx_xneg_ = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
            i_xpos_, j_xpos_, dx_xpos_ = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
            i_yneg_, j_yneg_, dy_yneg_ = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
            i_ypos_, j_ypos_, dy_ypos_ = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
            i_zneg_, j_zneg_, dz_zneg_ = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
            i_zpos_, j_zpos_, dz_zpos_ = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    # iteration printing
    every_percent_ = np.linspace(0, n_total-1, 101, dtype=int)
    every_percent_ = np.append(every_percent_, 0)
    if deriv_contd:
        percent = int(i_contd/(n_total-1)*100) + 1 # next percent
        print('continuing from ',percent-1,"%")
    else:
        percent = 1 # next percent, currently at 0%
    idx_percent = every_percent_[percent]
    # print('idx_percent', idx_percent)

    if deriv_contd:
        i_start = i_contd
    else:
        i_start = 0

    # for every particle i
    for i in range(i_start, n_total):
        # eps
        eps = eps_[i]

        neighbors_i_ = neighbors_2d[i]

        # central difference computing
        if P>1:
            i_, j_, dx_, dy_, dz_, dxx_, dyy_, dzz_ = \
                get_deriv(i, nodes_2d, neighbors_i_, eps, P, 'all', i_, j_, dx_, dy_, dz_, dxx_, dyy_, dzz_)
        else:
            i_, j_, dx_, dy_, dz_ = \
                get_deriv(i, nodes_2d, neighbors_i_, eps, P, 'all', i_, j_, dx_, dy_, dz_)
            

        # upwind difference computing
        if i >= n_bound and upwind:
            try: # try to compute derivatives with order of accuracy P, if the neighbors are not sufficient for the corresponding P, change to 1st order of accuracy
                i_xneg_, j_xneg_, dx_xneg_ = get_deriv(i, nodes_2d, neighbors_xneg_2d[i], eps, P, 'x', i_xneg_, j_xneg_, dx_=dx_xneg_)
            except:
                i_xneg_, j_xneg_, dx_xneg_ = get_deriv(i, nodes_2d, neighbors_xneg_2d[i], eps, 1, 'x', i_xneg_, j_xneg_, dx_=dx_xneg_)
                # print(i,'xneg')
            try:
                i_xpos_, j_xpos_, dx_xpos_ = get_deriv(i, nodes_2d, neighbors_xpos_2d[i], eps, P, 'x', i_xpos_, j_xpos_, dx_=dx_xpos_)
            except:
                i_xpos_, j_xpos_, dx_xpos_ = get_deriv(i, nodes_2d, neighbors_xpos_2d[i], eps, 1, 'x', i_xpos_, j_xpos_, dx_=dx_xpos_)
                # print(i)
            try:
                i_yneg_, j_yneg_, dy_yneg_ = get_deriv(i, nodes_2d, neighbors_yneg_2d[i], eps, P, 'y', i_yneg_, j_yneg_, dy_=dy_yneg_)
            except:
                i_yneg_, j_yneg_, dy_yneg_ = get_deriv(i, nodes_2d, neighbors_yneg_2d[i], eps, 1, 'y', i_yneg_, j_yneg_, dy_=dy_yneg_)
                # print(i,'yneg')
            try:
                i_ypos_, j_ypos_, dy_ypos_ = get_deriv(i, nodes_2d, neighbors_ypos_2d[i], eps, P, 'y', i_ypos_, j_ypos_, dy_=dy_ypos_)
            except:
                i_ypos_, j_ypos_, dy_ypos_ = get_deriv(i, nodes_2d, neighbors_ypos_2d[i], eps, 1, 'y', i_ypos_, j_ypos_, dy_=dy_ypos_)
                # print(i)
            try:
                i_zneg_, j_zneg_, dz_zneg_ = get_deriv(i, nodes_2d, neighbors_zneg_2d[i], eps, P, 'z', i_zneg_, j_zneg_, dz_=dz_zneg_)
            except:
                i_zneg_, j_zneg_, dz_zneg_ = get_deriv(i, nodes_2d, neighbors_zneg_2d[i], eps, 1, 'z', i_zneg_, j_zneg_, dz_=dz_zneg_)
                # print(i,'zneg')
            try:
                i_zpos_, j_zpos_, dz_zpos_ = get_deriv(i, nodes_2d, neighbors_zpos_2d[i], eps, P, 'z', i_zpos_, j_zpos_, dz_=dz_zpos_)
            except:
                i_zpos_, j_zpos_, dz_zpos_ = get_deriv(i, nodes_2d, neighbors_zpos_2d[i], eps, 1, 'z', i_zpos_, j_zpos_, dz_=dz_zpos_)
                # print(i)

        if i == idx_percent: # checking progress and save in-progress computations
            print(percent,"%")
            percent += 1 # next i check
            idx_percent = every_percent_[percent]
            with open("Output.txt", "w") as text_file:
                print(f"i: {i}", file=text_file)    
            # central
            np.save("current_result\\j_.npy", j_)
            np.save("current_result\\i_.npy", i_)
            np.save("current_result\\dx_.npy", dx_)
            np.save("current_result\\dy_.npy", dy_)
            np.save("current_result\\dz_.npy", dz_)
            if P >= 2:
                np.save("current_result\\dxx_.npy", dxx_)
                np.save("current_result\\dyy_.npy", dyy_)
                np.save("current_result\\dzz_.npy", dzz_)
            #! upwind

    # Store the derivatives in the COO matrix and convert the COO matrices to CSR format 
    # central
    dx_dcpse_2d = sparse.coo_matrix((dx_, (i_, j_)), shape=(n_total,n_total)).tocsr()
    dy_dcpse_2d = sparse.coo_matrix((dy_, (i_, j_)), shape=(n_total,n_total)).tocsr()
    dz_dcpse_2d = sparse.coo_matrix((dz_, (i_, j_)), shape=(n_total,n_total)).tocsr()
    if P >= 2:
        dxx_dcpse_2d = sparse.coo_matrix((dxx_, (i_, j_)), shape=(n_total,n_total)).tocsr()
        dyy_dcpse_2d = sparse.coo_matrix((dyy_, (i_, j_)), shape=(n_total,n_total)).tocsr()
        dzz_dcpse_2d = sparse.coo_matrix((dzz_, (i_, j_)), shape=(n_total,n_total)).tocsr()
    # upwind
    if upwind:
        dx_xneg_dcpse_2d = sparse.coo_matrix((dx_xneg_, (i_xneg_, j_xneg_)), shape=(n_total,n_total)).tocsr()[n_bound:] # one-sided derivative is only for inner nodes (bcs boundary nodes is already one sided(in the opposite direction of its outer normal))
        dx_xpos_dcpse_2d = sparse.coo_matrix((dx_xpos_, (i_xpos_, j_xpos_)), shape=(n_total,n_total)).tocsr()[n_bound:]
        dy_yneg_dcpse_2d = sparse.coo_matrix((dy_yneg_, (i_yneg_, j_yneg_)), shape=(n_total,n_total)).tocsr()[n_bound:]
        dy_ypos_dcpse_2d = sparse.coo_matrix((dy_ypos_, (i_ypos_, j_ypos_)), shape=(n_total,n_total)).tocsr()[n_bound:]
        dz_zneg_dcpse_2d = sparse.coo_matrix((dz_zneg_, (i_zneg_, j_zneg_)), shape=(n_total,n_total)).tocsr()[n_bound:]
        dz_zpos_dcpse_2d = sparse.coo_matrix((dz_zpos_, (i_zpos_, j_zpos_)), shape=(n_total,n_total)).tocsr()[n_bound:]

    # Saving and checking
    fcn = nodes_2d[:,0] + nodes_2d[:,1] + nodes_2d[:,2] # test-function 
    #* upwind
    if upwind:
        # save
        sparse.save_npz('dcpse/dx_xneg_dcpse_2d', dx_xneg_dcpse_2d)
        sparse.save_npz('dcpse/dx_xpos_dcpse_2d', dx_xpos_dcpse_2d)
        sparse.save_npz('dcpse/dy_yneg_dcpse_2d', dy_yneg_dcpse_2d)
        sparse.save_npz('dcpse/dy_ypos_dcpse_2d', dy_ypos_dcpse_2d)
        sparse.save_npz('dcpse/dz_zneg_dcpse_2d', dz_zneg_dcpse_2d)
        sparse.save_npz('dcpse/dz_zpos_dcpse_2d', dz_zpos_dcpse_2d)
        # check
        fig2, ax2 = plt.subplots(3,2)
        ax2[0,0].plot(dx_xneg_dcpse_2d.dot(fcn) )
        ax2[0,1].plot(dx_xpos_dcpse_2d.dot(fcn) )
        ax2[1,0].plot(dy_yneg_dcpse_2d.dot(fcn) )
        ax2[1,1].plot(dy_ypos_dcpse_2d.dot(fcn) )
        ax2[2,0].plot(dz_zneg_dcpse_2d.dot(fcn) )
        ax2[2,1].plot(dz_zpos_dcpse_2d.dot(fcn) )

    #* central (ended the fcn w return)   
    if P < 2:
        # save
        sparse.save_npz('dcpse/dx_dcpse_2d', dx_dcpse_2d)
        sparse.save_npz('dcpse/dy_dcpse_2d', dy_dcpse_2d)
        sparse.save_npz('dcpse/dz_dcpse_2d', dz_dcpse_2d)
        # check
        fig2, ax2 = plt.subplots(1,3)
        ax2[0].plot(dx_dcpse_2d.dot(fcn) )
        ax2[1].plot(dy_dcpse_2d.dot(fcn) )
        ax2[2].plot(dz_dcpse_2d.dot(fcn) )
        return dx_dcpse_2d, dy_dcpse_2d, dz_dcpse_2d 
    else:
        # save
        sparse.save_npz('dcpse/dx_dcpse_2d', dx_dcpse_2d)
        sparse.save_npz('dcpse/dy_dcpse_2d', dy_dcpse_2d)
        sparse.save_npz('dcpse/dz_dcpse_2d', dz_dcpse_2d)
        sparse.save_npz('dcpse/dxx_dcpse_2d', dxx_dcpse_2d)
        sparse.save_npz('dcpse/dyy_dcpse_2d', dyy_dcpse_2d)   
        sparse.save_npz('dcpse/dzz_dcpse_2d', dzz_dcpse_2d)
        # check
        fig2, ax2 = plt.subplots(2,3 )
        ax2[0,0].plot(dx_dcpse_2d.dot(fcn) )
        ax2[0,1].plot(dy_dcpse_2d.dot(fcn) )
        ax2[0,2].plot(dz_dcpse_2d.dot(fcn) )
        ax2[1,0].plot(dxx_dcpse_2d.dot(fcn) )
        ax2[1,1].plot(dyy_dcpse_2d.dot(fcn) )        
        ax2[1,2].plot(dzz_dcpse_2d.dot(fcn) )        
        return dx_dcpse_2d, dy_dcpse_2d, dz_dcpse_2d, dxx_dcpse_2d, dyy_dcpse_2d, dzz_dcpse_2d
    
