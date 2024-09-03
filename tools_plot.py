# Functions for plotting the Curves (Pablo) Last update: 2024-08-26
#-------------
# This part gathers the functions used to analyze the data (energies and [m_s,S(S+1),%S_M=1, %S_L=1])
# in order to plot them.
#-------------
import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


##----------------------------BUILDING FUNCTIONS--------------##
################################################################
#---FUNCTION TO BUILD THE NBODY BASIS---------------------------    
################################################################
def build_basis(N_MO,
                index_metal,index_ligand,
                h00,h11,h22,h33,
                K_M0M1,K_L2L3,K_M0L2,K_M0L3,K_M1L2,K_M1L3,
                U_M,U_L,t_M,t_L,t_ML):
    """
    This function updates the one- and two-electron matrices as a function of the parameters.

    Parameters
    ----------
    N_MO : int
        Number of MOs.
    index_metal : list
        Indices of the metal MOs.
    index_ligand : list
        Indices of the ligand MOs.
    h00 : float
        Energy of 1st MO.
    h11 : float
        Energy of 2nd MO.
    h22 : float
        Energy of 3rd MO.
    h33 : float
        Energy of 4st MO.
    K_M0M1 : float
        Exchange integral between both metal MOs.
    K_L2L3 : float
        Exchange integral between both ligand MOs.
    K_M0L2 : float
        Exchange integral between metal M and ligand L1.
    K_M0L3 : float
        Exchange integral between metal M and ligand L2.
    K_M1L2 : float
        Exchange integral between metal M' and ligand L1.
    K_M1L3 : float
        Exchange integral between metal M' and ligand L2.
    U_M : float
        Coulomb integral on metal MOs.
    U_L : float
        Coulomb integral on ligand MOs.
    t_M : float
        Hopping between metal MOs (MC).
    t_L : float
        Hopping between ligand MOs (LC).
    t_ML : float
        Hopping between metal and ligand (MLCT/LMCT).

    Returns
    -------
    h_ : array
        One-electron matrix.
    g_ : array
        Two-electron matrix.

    """
    # Initialization of the 1- and 2-electron matrices
    h_  = np.zeros((N_MO,N_MO))
    g_  = np.zeros((N_MO,N_MO,N_MO,N_MO)) 
    # Energy of each MO
    h_[0,0] = h00   # metal
    h_[1,1] = h11  # metal1
    h_[2,2] = h22 # ligand1
    h_[3,3] = h33 # ligand2
    #------- We do not use Js (Coulomb integrals) so there are only zeroes
    # J_M
    J_M0M1 = 0.
    # J_LL
    J_L2L3 = 0.
    # J_ML
    J_M0L2 = 0.
    J_M0L3 = J_M0L2
    # J_M'L
    J_M1L2 = 0.
    J_M1L3 = J_M1L2
    
    #-------------
    # This part constructs the Hamiltonian matrix
    #-------------
    # Hamiltonian with K (and J)
    for i in range(N_MO):   
        for j in range(N_MO): 
            if j != i :
                if (i in index_metal) and (j in index_metal):
                    g_[i,j,j,i]  = K_M0M1   
                    g_[i,i,j,j]  = J_M0M1
                    
                elif  i in index_ligand and j in index_ligand:
                      g_[i,j,j,i]  = K_L2L3   
                      g_[i,i,j,j]  = J_L2L3
                      
                elif ( (i in index_metal and j in index_ligand) or
                       (j in index_metal and i in index_ligand) ) :  
                    
                    if ( ( i == index_ligand[0] and j == index_metal[0] ) or 
                         ( j == index_ligand[0] and i == index_metal[0] )  ):
                        g_[i,j,j,i]  = K_M0L2
                        g_[i,i,j,j]  = J_M0L2
                        
                    if ( ( i == index_ligand[1] and j == index_metal[0] ) or 
                         ( j == index_ligand[1] and i == index_metal[0] ) ):   
                        g_[i,j,j,i]  = K_M0L3
                        g_[i,i,j,j]  = J_M0L3
                        
                    if ( ( i == index_ligand[0] and j == index_metal[1] ) or 
                         ( j == index_ligand[0] and i == index_metal[1] ) ):
                        g_[i,j,j,i]  = K_M1L2
                        g_[i,i,j,j]  = J_M1L2
                    if ( ( i == index_ligand[1] and j == index_metal[1] ) or 
                         ( j == index_ligand[1] and i == index_metal[1] ) ): 
                        g_[i,j,j,i]  = K_M1L3
                        g_[i,i,j,j]  = J_M1L3
                   
               
    
    # Hamiltonian with U (and t)
    for i in range(N_MO):  
        # Coulombic repulsion
        if i in  index_metal  :   
            g_[i,i,i,i] = U_M
        if i in index_ligand :   
            g_[i,i,i,i] = U_L
        # Hopping
        for j in range(N_MO): 
            if j != i :
                if (i in index_metal) and (j in index_metal): 
                    h_[i,j]  = t_M 
                elif  i in index_ligand and j in index_ligand: 
                    h_[i,j]  = t_L  
                elif ( (i in index_metal and j in index_ligand) or
                        (j in index_metal and i in index_ligand) ) : 
                    h_[i,j] = t_ML
                        
    return h_, g_    

##----------------------------ANALYSIS FUNCTIONS--------------##
################################################################
#---FUNCTION TO CLEAN THE STATES--------------------------------    
################################################################
def cleaner(eigvec_SO, Op_penalty, penalty=1e3,visu=False,warning_CT=True): 
    """
    This function analyzes the 70 eigenstates obtained after diagonalization,
    then select only the 16 states of interest (singly occupied only) by applying
    a penalty operator Op_penalty that penalizes the doubly occupied states by penalty.
    The 16 unpenalized states are the one of interest.
    This produces a list of 16 indices that are then used by curve_analysis_spin().

    Parameters
    ----------
    eigvec_SO : array
        Eigenvectors obtained after diagonalization (70,70).
    Op_penalty : array
        Penalty operator that should be in the form:
            Op_penalty = scipy.sparse.csr_matrix((np.shape(nbody_basis)[0],np.shape(nbody_basis)[0]))
            for p in range(0,N_MO): 
                Op_penalty += 1*((a_dagger_a[2*p,2*p]+a_dagger_a[2*p+1,2*p+1]) == 2)
    penalty : float
        Value of penalty.
        The default is 1e3.
    visu : bool, optional
        True: prints the value of ref_vec.T @ Op_penalty * penalty @ ref_vec for each 16 selected candidate. 
        The default is False.

    Returns
    -------
    index: array
        array of (16) with the indices of the 16 states of interest.

    """
    
    # If not penalty, return the lowest 16 states
    if penalty == 0:
        return [i for i in range(16)]
    # Store the indices
    index = []
    # Loop over the 70 states
    for vec in range(len(eigvec_SO)):
        ref_vec = eigvec_SO[:,vec]
        # Projection of the eigenvector onto the penalty operator
        criteria = ref_vec.T @ Op_penalty * penalty @ ref_vec
        # The criteria should be atleast two times lower than the penalty.
        # Using penalty/2 and not 1 allows for adaptation of the code if one would add some t
        if np.abs(criteria) < penalty/2:
            if visu:
                print(criteria)
            index.append(vec)
            
            # The criteria should always be 0 if there is no CT.
            if warning_CT and (np.abs(criteria) > 0.1):
                raise ValueError(" This code was designed in a vision where CT is impossible."+
                                 "\n If you added some t integral, please modify/check the way the code executes."+
                                 "The vector #"+str(vec)+" has a criteria of "+str(np.abs(criteria))+
                                 "\n If you added some t, please provide: visu=False, warning_CT=False")
    # If there is more than 16 indices, then everything will break.
    if len(index) > 16:
        raise ValueError("You have selected "+str(len(index))+ " states."+
                         " It should be 16. Please check your Op_penalty. Make sure penalty > 0")
    
    return index
################################################################
#---FUNCTION TO ANALYSE THE RESULTS-----------------------------    
################################################################
def curve_analysis_spin(lambd, all_energies, all_proj_SPIN, eigval_SO, eigvec_SO, 
                         S_Z, S2, Proj_triplet_metal, Proj_triplet_ligand,
                         Op_penalty, penalty,use_index=True):
    """
    
    This function treats a problem where the 16 microstates arising from coupling either a S_M = 1 or 0 and a S_L = 1 or 0.
    
    Flow of selections criteria: (x = T or S, y = m (-1), z (0) or p (+1))
        1) S_2: separates QTTy (5 states), Txxy (9 states) and Sxxz (2 states)
        for QTTy:
            S_z: separates QTTmm=QTT(-2), QTTm=QTT(-1), QTTz=QTT(0), QTTp=QTT(+1), QTTpp=QTT(+2)
        for Txxy:
            S_z: separates Txxm (3 states), Txxz (3 states) and Txxp (3 states)
                for each batch (m_s = -1, 0 or +1):
                    a) Proj_triplet_metal: separates TSTy (1 state) and TTxy (2 states)
                    b) Proj_triplet_ligand: separates TTSy and TTTy
        for Sxxz:
            Proj_triplet_metal: separates SSSz from STTz
    
    Ordering:
    There are 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The states are ordered as the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    lambd : int
        index of lambd.
    all_energies : array
        array of (16,#lambd) gathering the energies of each microstate.
    all_proj_SPIN : array
        array of (16,#lambd,4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    eigval_SO : array
        Eigenvalues of the Hamiltonian.
    eigvec_SO : array
        Eigenvectors of the Hamiltonian.
    S_Z : array
        S_Z operator.
    S2 : array
        S^2 operator.
    Proj_triplet_metal : array
        Projector onto %S_M = 1.
    Proj_triplet_ligand : array
        Projector onto %S_L = 1.
    Op_penalty : array
        Penalty operator for doubly occupied states. It penalizes CT states.    
    penalty : int
        Amount of penalty. In general, 1e3 is great.
    use_index : bool, optional
        True: Allow to use the cleaner() fct (selection of the 16 states using Op_penalty). 
        False: Take the lowest 16 states.  
        This should always be activated.
        The default is True.
        
    Returns
    -------
    all_energies : array
        array of (16,#lambd) gathering the energies of each microstate.
    all_proj_SPIN : array
        array of (16,#lambd,4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].

    """
    if use_index:
        index = cleaner(eigvec_SO,Op_penalty,penalty)
        eigval_SOa = [eigval_SO[idx] for idx in index]
        eigvec_SOa = [eigvec_SO[:,idx] for idx in index]
    else:
        eigval_SOa = eigval_SO[:16]
        eigvec_SOa = eigvec_SO.T[:16]
    
    
    ### WE LABEL states
    # 1) S_2: separates QTTy (5 states), Txxy (9 states) and Sxxz (2 states)
    S2_mat = np.zeros(len(eigval_SOa))
    for i in range(len(eigvec_SOa)):
        S2_mat[i] = eigvec_SOa[i].T @ S2 @ eigvec_SOa[i]
    
    # Sort indices
    index_sort_S2 = np.argsort(S2_mat)
    # Update eigval and eigvec list using the sorted indices
    sorted_eigval_global = [eigval_SOa[index] for index in index_sort_S2]
    sorted_eigvec_global = [eigvec_SOa[index] for index in index_sort_S2]
    
    # Quintet list: QTTy
    eigval_Q = sorted_eigval_global[11:]
    eigvec_Q = sorted_eigvec_global[11:]
    # Triplet list: Txxy 
    eigval_T = sorted_eigval_global[2:11]
    eigvec_T = sorted_eigvec_global[2:11]
    # Singlet list: Sxxz
    eigval_S = sorted_eigval_global[:2]
    eigvec_S = sorted_eigvec_global[:2]
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    ### FOR QUINTET: QTTy
    # S_z: separates QTTmm=QTT(-2), QTTm=QTT(-1), QTTz=QTT(0), QTTp=QTT(+1), QTTpp=QTT(+2)
    S_z_mat = np.zeros(len(eigval_Q))
    for i in range(len(eigvec_Q)):    
        S_z_mat[i] = eigvec_Q[i].T @ S_Z @ eigvec_Q[i]
    
    # Sort indices
    index_sort_S_z = np.argsort(S_z_mat)
    # Update eigval and eigvec list using the sorted indices
    sorted_eigval = [eigval_Q[index] for index in index_sort_S_z]
    sorted_eigvec = [eigvec_Q[index] for index in index_sort_S_z]
    # QTTmm, QTTm, QTTz, QTTp, QTTpp
    order_list = [0,1,2,3,4]
    for myvec in range(len(order_list)):
        # Takes the global index of the state from order_list
        global_vec = order_list[myvec]
        # Fill in all_energies
        all_energies[global_vec][lambd] = sorted_eigval[myvec]
        # Fill in all_proj_SPIN
        all_proj_SPIN[global_vec][lambd] = [sorted_eigvec[myvec].T @ S_Z @ sorted_eigvec[myvec], 
                                 sorted_eigvec[myvec].T @ S2 @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_metal @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_ligand @ sorted_eigvec[myvec]]
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    ### FOR TRIPLET: Txxy
    # S_z: separates Txxm (3 states), Txxz (3 states) and Txxp (3 states)
    S_z_mat = np.zeros(len(eigval_T))
    for i in range(len(eigvec_T)):    
        S_z_mat[i] = eigvec_T[i].T @ S_Z @ eigvec_T[i]
    # Sort indices
    index_sort_S_z = np.argsort(S_z_mat)
    # Update eigval and eigvec list using the sorted indices
    sorted_eigval_T = [eigval_T[index] for index in index_sort_S_z]
    sorted_eigvec_T = [eigvec_T[index] for index in index_sort_S_z]
    
    # Txxm: m_s = -1
    eigval_T_m = sorted_eigval_T[:3]
    eigvec_T_m = sorted_eigvec_T[:3]
    # Txxz: m_s = 0
    eigval_T_z = sorted_eigval_T[3:6]
    eigvec_T_z = sorted_eigvec_T[3:6]
    # Txxp: m_s = +1
    eigval_T_p = sorted_eigval_T[6:]
    eigvec_T_p = sorted_eigvec_T[6:]
    #--------------------------------------------------------------------------
    ## FOR Txxm
    # Computer trick to have a more readable/repeatable/less resources consuming code
    eigval = eigval_T_m
    eigvec = eigvec_T_m
    # Proj_triplet_metal: separates TSTy (1 state) and TTxy (2 states)
    Proj_met_mat = np.zeros(len(eigval))
    for i in range(len(eigval)):
        Proj_met_mat[i] = eigvec[i].T @ Proj_triplet_metal @ eigvec[i]
    # Sort indices
    index_sort = np.argsort(Proj_met_mat)
    # Update eigval and eigvec list using the sorted indices
    sorted_eigval_T_m = [eigval[index] for index in index_sort]
    sorted_eigvec_T_m = [eigvec[index] for index in index_sort]  
    # Computer trick to have a more readable/repeatable/less resources consuming code      
    sorted_eigval = sorted_eigval_T_m
    sorted_eigvec = sorted_eigvec_T_m
    
    # TSTm
    myvec = 0
    # Global index of the state
    glob_vec = 12
    # Fill in all_energies
    all_energies[glob_vec][lambd] = sorted_eigval[myvec]
    # Fill in all_proj_SPIN
    all_proj_SPIN[glob_vec][lambd] = [sorted_eigvec[myvec].T @ S_Z @ sorted_eigvec[myvec], 
                             sorted_eigvec[myvec].T @ S2 @ sorted_eigvec[myvec],
                             sorted_eigvec[myvec].T @ Proj_triplet_metal @ sorted_eigvec[myvec],
                             sorted_eigvec[myvec].T @ Proj_triplet_ligand @ sorted_eigvec[myvec]]

    # Extract TTxm from sorted_eigval_T_m
    eigval = sorted_eigval_T_m[1:]
    eigvec = sorted_eigvec_T_m[1:]
    # Proj_triplet_ligand: separates TTSy and TTTy
    Proj_lig_mat = np.zeros(len(eigval))
    for i in range(len(eigval)):
        Proj_lig_mat[i] = eigvec[i].T @ Proj_triplet_ligand @ eigvec[i]
    # Sort indices
    index_sort = np.argsort(Proj_lig_mat)
    # Update eigval and eigvec list using the sorted indices
    sorted_eigval_TT = [eigval[index] for index in index_sort]
    sorted_eigvec_TT = [eigvec[index] for index in index_sort]
    # Computer trick to have a more readable/repeatable/less resources consuming code
    sorted_eigval = sorted_eigval_TT
    sorted_eigvec = sorted_eigvec_TT
    # TTSm, TTTm
    order_list = [9,5]
    for myvec in range(len(order_list)):
        # Takes the global index of the state from order_list
        global_vec = order_list[myvec]
        # Fill in all_energies
        all_energies[global_vec][lambd] = sorted_eigval[myvec]
        # Fill in all_proj_SPIN
        all_proj_SPIN[global_vec][lambd] = [sorted_eigvec[myvec].T @ S_Z @ sorted_eigvec[myvec], 
                                 sorted_eigvec[myvec].T @ S2 @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_metal @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_ligand @ sorted_eigvec[myvec]]
    #--------------------------------------------------------------------------   
    ## FOR Txxz
    # Same comments than for Txxm
    eigval = eigval_T_z
    eigvec = eigvec_T_z
    Proj_met_mat = np.zeros(len(eigval))
    for i in range(len(eigval)):
        Proj_met_mat[i] = eigvec[i].T @ Proj_triplet_metal @ eigvec[i]
    
    index_sort = np.argsort(Proj_met_mat)
    sorted_eigval_T_z = [eigval[index] for index in index_sort]
    sorted_eigvec_T_z = [eigvec[index] for index in index_sort]        
    sorted_eigval = sorted_eigval_T_z
    sorted_eigvec = sorted_eigvec_T_z
    myvec = 0
    glob_vec = 13
    all_energies[glob_vec][lambd] = sorted_eigval[myvec]
    all_proj_SPIN[glob_vec][lambd] = [sorted_eigvec[myvec].T @ S_Z @ sorted_eigvec[myvec], 
                             sorted_eigvec[myvec].T @ S2 @ sorted_eigvec[myvec],
                             sorted_eigvec[myvec].T @ Proj_triplet_metal @ sorted_eigvec[myvec],
                             sorted_eigvec[myvec].T @ Proj_triplet_ligand @ sorted_eigvec[myvec]]
    # TSTm
    eigval = sorted_eigval_T_z[1:]
    eigvec = sorted_eigvec_T_z[1:]
    Proj_lig_mat = np.zeros(len(eigval))
    for i in range(len(eigval)):
        Proj_lig_mat[i] = eigvec[i].T @ Proj_triplet_ligand @ eigvec[i]
    index_sort = np.argsort(Proj_lig_mat)
    sorted_eigval_TT = [eigval[index] for index in index_sort]
    sorted_eigvec_TT = [eigvec[index] for index in index_sort]
    sorted_eigval = sorted_eigval_TT
    sorted_eigvec = sorted_eigvec_TT
    # TTSm, TTTm
    order_list = [10,6]
    for myvec in range(len(order_list)):
        global_vec = order_list[myvec]
        all_energies[global_vec][lambd] = sorted_eigval[myvec]
        all_proj_SPIN[global_vec][lambd] = [sorted_eigvec[myvec].T @ S_Z @ sorted_eigvec[myvec], 
                                 sorted_eigvec[myvec].T @ S2 @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_metal @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_ligand @ sorted_eigvec[myvec]]
    ## FOR Txxp
    # Same comments than for Txxm
    eigval = eigval_T_p
    eigvec = eigvec_T_p
    Proj_met_mat = np.zeros(len(eigval))
    for i in range(len(eigval)):
        Proj_met_mat[i] = eigvec[i].T @ Proj_triplet_metal @ eigvec[i]
    index_sort = np.argsort(Proj_met_mat)
    sorted_eigval_T_p = [eigval[index] for index in index_sort]
    sorted_eigvec_T_p = [eigvec[index] for index in index_sort]        
    sorted_eigval = sorted_eigval_T_p
    sorted_eigvec = sorted_eigvec_T_p
    myvec = 0
    glob_vec = 14
    all_energies[glob_vec][lambd] = sorted_eigval[myvec]
    all_proj_SPIN[glob_vec][lambd] = [sorted_eigvec[myvec].T @ S_Z @ sorted_eigvec[myvec], 
                             sorted_eigvec[myvec].T @ S2 @ sorted_eigvec[myvec],
                             sorted_eigvec[myvec].T @ Proj_triplet_metal @ sorted_eigvec[myvec],
                             sorted_eigvec[myvec].T @ Proj_triplet_ligand @ sorted_eigvec[myvec]]
    # TSTm
    eigval = sorted_eigval_T_p[1:]
    eigvec = sorted_eigvec_T_p[1:]
    
    Proj_lig_mat = np.zeros(len(eigval))
    for i in range(len(eigval)):
        Proj_lig_mat[i] = eigvec[i].T @ Proj_triplet_ligand @ eigvec[i]
    index_sort = np.argsort(Proj_lig_mat)
    sorted_eigval_TT = [eigval[index] for index in index_sort]
    sorted_eigvec_TT = [eigvec[index] for index in index_sort]
    sorted_eigval = sorted_eigval_TT
    sorted_eigvec = sorted_eigvec_TT
    # TTSm, TTTm
    order_list = [11,7]
    for myvec in range(len(order_list)):
        global_vec = order_list[myvec]
        all_energies[global_vec][lambd] = sorted_eigval[myvec]
        all_proj_SPIN[global_vec][lambd] = [sorted_eigvec[myvec].T @ S_Z @ sorted_eigvec[myvec], 
                                 sorted_eigvec[myvec].T @ S2 @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_metal @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_ligand @ sorted_eigvec[myvec]]
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    ### FOR SINGLET: Sxxz
    # Proj_triplet_metal: separates SSSz from STTz
    Double_Proj_mat = np.zeros(len(eigval_S))
    for i in range(len(eigval_S)):
        Double_Proj_mat[i] = (eigvec_S[i].T @ Proj_triplet_metal @ eigvec_S[i] +
                              eigvec_S[i].T @ Proj_triplet_ligand @ eigvec_S[i])/2
    # Sort indices
    index_sort = np.argsort(Double_Proj_mat)
    # Update eigval and eigvec list using the sorted indices
    sorted_eigval_S = [eigval_S[index] for index in index_sort]
    sorted_eigvec_S = [eigvec_S[index] for index in index_sort]
    
    sorted_eigval,sorted_eigvec =  sorted_eigval_S,sorted_eigvec_S
    # SSSz, STTz
    order_list = [15,8]
    for myvec in range(len(order_list)):
        # Takes the global index of the state from order_list
        global_vec = order_list[myvec]
        # Fill in all_energies
        all_energies[global_vec][lambd] = sorted_eigval[myvec]
        # Fill in all_proj_SPIN
        all_proj_SPIN[global_vec][lambd] = [sorted_eigvec[myvec].T @ S_Z @ sorted_eigvec[myvec], 
                                 sorted_eigvec[myvec].T @ S2 @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_metal @ sorted_eigvec[myvec],
                                 sorted_eigvec[myvec].T @ Proj_triplet_ligand @ sorted_eigvec[myvec]]

    return all_energies, all_proj_SPIN

##----------------------------PLOTTING FUNCTIONS--------------##

################################################################
#---PLOTTING of m_s, S2, ProjS_M and ProjS_L for the 16 states--    
################################################################
def plot_ms_S2_ProjSM_ProjSL(lambd, Q, lambd_val,all_proj_SPIN, 
                             K_M0L2, print_var="lambda",
                             txt_fontsize=20,
                             name_file="plot_ms_S2_ProjSM_ProjSL.png",
                             save=False,mlml=None):
    """
    This creates a 16*4 tabular plot:
        horizontal: QTT(-2), QTT(-1), QTT(0), QTT(+1), QTT(+2), TTT(-1), TTT(0), TTT(+1),
                    STT(0), TTS(-1), TTS(0), TTS(+1), TST(-1), TST(0), TST(+1), SSS(0)
        vertical (top -> bottom): %(S_L = 1), %(S_M = 1), S(S+1), m_s
    This helps identifying with a simple glimpse the local and global spin properties of our 16 states of interest.
    Parameters
    ----------
    lambd : float
        Value of lambda.
    Q : float
        Value of Q.
    lambd_val : int
        Index of lambda or Q or mlml in all_proj_SPIN.
    all_proj_SPIN : array
        Array of (16,#,4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    K_M0L2 : float
        Value of K_1 integral.
    print_var : string, optional
        The name of the parameter to go fecth in all_proj_SPIN.
        "lambda" for lambda.
        "Q" for Q.   
        "mlml" for {m_l_M;m_l_M'}.
        The default is "lambda".
    name_file : string
        Name of the file to print.
    save : bool, optional
        True: save the plot.
        False: don't save the plot.
        The default is False.
    mlml : string, optional
        Label of the mlml.
        The default is None. It is only used when print_var == "mlml".
    Returns
    -------
    None.

    """
    # Initialize the subplots
    fig, axs = plt.subplots(nrows=4,figsize=(24,10), sharex=True)
    # Number of significative digits to be shown
    number_float = '%.2f'
    
    # Labels for each state
    label_list = ["$QTT$\n$(-2)$", "$QTT$\n$(-1)$", "$QTT$\n$(0)$", "$QTT$\n$(+1)$", "$QTT$\n$(+2)$", 
                  "$TTT$\n$(-1)$", "$TTT$\n$(0)$", "$TTT$\n$(+1)$", 
                  "$STT$\n$(0)$", 
                  "$TTS$\n$(-1)$", "$TTS$\n$(0)$", "$TTS$\n$(+1)$", 
                  "$TST$\n$(-1)$", "$TST$\n$(0)$", "$TST$\n$(+1)$", 
                  "$SSS$\n$(0)$"]
    # Title of the Figure
    if print_var == "Q":
        fig.suptitle("$\\lambda = " + str(np.round(lambd,7))+"$, $Q = " + str(np.round(Q,7))+
                     "$, $K_1 = "+str(K_M0L2)+"$",fontsize=txt_fontsize,x=0.5,y=.92)
    elif print_var == "lambda":
        fig.suptitle("$\\lambda = " + str(np.round(lambd,7))+"$, $Q = " + str(np.round(Q,7))+
                     "$, $K_1 = "+str(K_M0L2)+"$",fontsize=txt_fontsize,x=0.5,y=.92)
    elif print_var == "mlml" and mlml != None:
        fig.suptitle("$\\lambda = " + str(np.round(lambd,7))+"$, $Q = " + str(np.round(Q,7))+
                     "$, $K_1 = "+str(K_M0L2)+"$, $\\{m_{l_{M}};m_{l_{M'}}\\}$ = "+mlml,fontsize=txt_fontsize,x=0.5,y=.92)
    else:
        raise ValueError('Please select print_var = "Q" or "lambda" or "mlml')
    # Microstates
    x = label_list
    
    # No need to modify this, it helps to know what is plotted where
    spins = ["m_s","S2","S_met","S_lig"]
    #--------------------------------------------------------------------------
    ## m_s data
    # position of the graph
    val,ax = 0,3
    # Matplotlib trick to use colorbar(pcolormesh())
    # I plot a duplicate over a y "range" [m_s,m_s2]
    # This will make the figure continuous on y
    # Disclaimer: THIS COULD BE IMPROVED BUT IT WORKS FOR NOW.
    y = spins[val]
    Y = np.array([y,y+"2"])
    
    z = all_proj_SPIN[:,lambd_val].T[val]
    Z = np.array([z,z])
    
    # Separates the results (from -2 to +2) in shades of gray (-2: black, 0: gray, +2: white)
    plot_colormesh_ms = axs[ax].pcolormesh(x,Y,Z,cmap='gray')
    divider = make_axes_locatable(axs[ax])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    # Plot the colorbars
    fig.colorbar(plot_colormesh_ms, cax=cax, orientation='vertical')
    # Add labelling inside boxes
    for xx in range(len(x)):
        # Text in white for dark background
        if Z[0,xx] < -0.5:
            axs[ax].text(xx, 0.5, number_float % Z[0,xx],
                     horizontalalignment='center',
                     verticalalignment='center',color="white",fontsize=txt_fontsize,
                         )
        # Text in black for light background
        else:
            axs[ax].text(xx, 0.5, number_float % Z[0,xx],
                     horizontalalignment='center',
                     verticalalignment='center',fontsize=txt_fontsize,
                         )
    axs[ax].yaxis.get_label().set_fontsize(txt_fontsize)
    #--------------------------------------------------------------------------
    ## S(S+1) data
    # Same comments than for m_s
    # Exceptions:
    #   - colormesh is (0: red, 2: pink, 3: white, 6: blue) 
    #   - white text for S(S+1) > 4 and for S(S+1) < 1 
    #   - black text for the rest
    val,ax = 1,2
    y = spins[val]
    Y = np.array([y,y+"2"])
    z = all_proj_SPIN[:,lambd_val].T[val]
    Z = np.array([z,z])
    plot_colormesh_ms = axs[ax].pcolormesh(x,Y,Z, cmap='RdBu')
    divider = make_axes_locatable(axs[ax])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot_colormesh_ms, cax=cax, orientation='vertical')
    for xx in range(len(x)):
        if 1.< Z[0,xx] < 4:
            axs[ax].text(xx, 0.5, number_float % Z[0,xx],
                     horizontalalignment='center',
                     verticalalignment='center',fontsize=txt_fontsize,
                         )
        else:
            axs[ax].text(xx, 0.5, number_float % Z[0,xx],
                     horizontalalignment='center',
                     verticalalignment='center',color="white",fontsize=txt_fontsize,
                         )
    axs[ax].yaxis.get_label().set_fontsize(txt_fontsize)
    #--------------------------------------------------------------------------
    # S_m data
    # Same comments than for m_s
    # Exceptions:
    #   - colormesh is (0: purple, 0.5: blue, 1: yellow) 
    #   - white text for S_m < 0.3 
    #   - black text for the rest
    val,ax = 2,1
    y = spins[val]
    Y = np.array([y,y+"2"])
    z = all_proj_SPIN[:,lambd_val].T[val]
    Z = np.array([z,z])
    plot_colormesh_ms = axs[ax].pcolormesh(x,Y,Z)
    divider = make_axes_locatable(axs[ax])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot_colormesh_ms, cax=cax, orientation='vertical')
    for xx in range(len(x)):
        if Z[0,xx] < 0.3:
            axs[ax].text(xx, 0.5, number_float % Z[0,xx],
                     horizontalalignment='center',
                     verticalalignment='center',color="white",fontsize=txt_fontsize,
                         )
        else:
            axs[ax].text(xx, 0.5, number_float % Z[0,xx],
                     horizontalalignment='center',
                     verticalalignment='center',fontsize=txt_fontsize,
                         )
    axs[ax].yaxis.get_label().set_fontsize(txt_fontsize)
    #--------------------------------------------------------------------------
    # S_l data
    # Same comments than for m_s
    # Exceptions:
    #   - colormesh is (0: purple, 0.5: blue, 1: yellow) 
    #   - white text for S_m < 0.3 
    #   - black text for the rest
    val,ax = 3,0
    y = spins[val]
    Y = np.array([y,y+"2"])
    z = all_proj_SPIN[:,lambd_val].T[val]
    Z = np.array([z,z])
    plot_colormesh_ms = axs[ax].pcolormesh(x,Y,Z)
    divider = make_axes_locatable(axs[ax])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot_colormesh_ms, cax=cax, orientation='vertical')
    
    for xx in range(len(x)):
        if Z[0,xx] < 0.3:
            axs[ax].text(xx, 0.5, number_float % Z[0,xx],
                     horizontalalignment='center',
                     verticalalignment='center',color="white",fontsize=txt_fontsize,
                         )
        else:
            axs[ax].text(xx, 0.5, number_float % Z[0,xx],
                     horizontalalignment='center',
                     verticalalignment='center',fontsize=txt_fontsize,
                         )
    axs[ax].yaxis.get_label().set_fontsize(txt_fontsize)
    #--------------------------------------------------------------------------
    # Remove all ticks
    axs[3].set_yticks([])
    axs[2].set_yticks([])
    axs[1].set_yticks([])
    axs[0].set_yticks([])
    
    # Add custom labels for each plots (accepts LateX)
    axs[3].set_ylabel('$m_s$')
    axs[2].set_ylabel('$S(S+1)$')
    axs[1].set_ylabel('$\\%(S_M = 1)$')
    axs[0].set_ylabel('$\\%(S_L = 1)$')
    
    # Save the Figure as pdf if save == True
    if save:
        plt.savefig(name_file, bbox_inches='tight')
    # Show the Figure in Spyder
    plt.show()

################################################################
#---PLOTTING of Energy = f({m_l_M;m_l_M'})----------------------    
################################################################
def plot_graph_mlml(lambd,all_energies,label_y, 
                    ymin,ymax,
                    K_M0L2,K_M1L2,states=[i for i in range(16)],
                    name_file="plot_graph_mlml.png",
                    save=False,
                    size_linewidth=3, size_figsize=(16,10), size_fontsize=16, alpha_t=0.5):
    """
    This plots a graph: Energy as a function of different sets of {m_l_M;m_l_M'}
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz
    
    Parameters
    ----------
    lambd : float
        Lambda value.
    all_energies : array
        Array of (16,len(label_y)) gathering the energies of each microstate.
    label_y : list(string)
        List of the {m_l_M;m_l_M'} sets.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_mlml.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of the lines. 
        The default is 0.5.

    Returns
    -------
    None.

    """
    # Size of figure
    fig, ax = plt.subplots(figsize=size_figsize)
    # Title of Figure
    ax.set_title("Energy as a function of $\\{m_{l_M};m_{l_{M'}}\\}$",pad=20)
    # Add a text box inside the figure
    # textstr = "$\\lambda= "+str(lambd)+"$, $K_1 = "+str(K_M0L2)+"$, $K_1^{\\prime} = "+str(np.round(K_M1L2,7))+"$"
    textstr = "$\\lambda= "+str(lambd)+"$, $K_1 = K_1^{\\prime} = "+str(np.round(K_M1L2,7))+"$"
    ax.text(0.05,0.95, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set the limits of y
    ax.set_ylim([ymin, ymax])
    # Label on the x and y axis (it accepts LateX)
    plt.xlabel("$\\{ m_{l_{M}}; m_{l_{M'}} \\}$",fontsize=38)
    plt.ylabel("Energy ($K_M$ units)",fontsize=34)
    # y: set of {m_l_M;m_l_M'}; x: Energy
    x = all_energies
    y = label_y
    
    
    # Label of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Marker of each state
    markers = ["s","s","s","s","^",
              "s","s","s",
              "o",
              "s","s","s",
              "s","s","s",
              "X"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","solid",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted",
                  "solid"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # This is just to plot (if there is) state 15 before states 12, 13 and 14
    # if 12 in states:
    #     states.remove(12)
    #     states.append(12)
    # if 13 in states:
    #     states.remove(13)
    #     states.append(13)
    # if 14 in states:
    #     states.remove(14)
    #     states.append(14)
    ## PLOT FIRST THE MARKERS
    for i in states:
        if i == 4 or i == 8:
            plt.plot(y,x[i],marker=markers[i], color=colors[i],
                      linewidth=size_linewidth,markersize=10,label='_Hidden')
        elif i == 15:
            plt.plot(y,x[i],marker=markers[i], color=colors[i],linestyle='None',
                     markersize=10,label=label_list[i])
        
        else:
            plt.plot(y,x[i],markers[i], color=colors[i],linewidth=size_linewidth,label='_Hidden')
    ## THEN PLOT THE LINES
    # Store the label of the states that are going to be plotted
    final_label_list = []
    for i in states:
        
        if i == 4 or i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i],marker=markers[i], color=colors[i],
                      linewidth=size_linewidth,alpha=alpha_t,label=label_list[i])
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                      linewidth=size_linewidth,alpha=alpha_t,label='_Hidden')
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], marker=markers[i],color=colors[i],
                      linewidth=size_linewidth,alpha=alpha_t,label=label_list[i])
            final_label_list.append(label_list[i])
    
    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend(handles[1:]+[handles[0]],
                     labels[1:]+[labels[0]],
                      bbox_to_anchor=(1,0.5),loc='center left')
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
    # plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    # Show gridlines
    plt.grid(True)
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()

################################################################
#---PLOTTING of Energy = f(lambda)------------------------------    
################################################################
def plot_graph_lambd(all_energies, lambd_var, 
                     xmin, xmax, ymin, ymax,
                     K_M0L2,K_M1L2,states=[i for i in range(16)],
                     name_file="plot_graph_lambda.png",
                     save=False,s=True,
                     size_linewidth=3,size_figsize=(16,10),size_fontsize=16):
    """
    This plots a graph: Energy as a function of different lambda values.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    all_energies : array
        Array of (16,len(lambd_var)) gathering the energies of each microstate.
    lambd_var : list(string)
        List of the lambda values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_lambda.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=size_figsize)
    # Title of the Figure (it can be commented to not have any title)
    ax.set_title("Energy as a function of $\\lambda$",pad=20)
    # Text box inside the graph
    textstr = "$K_1 = "+str(K_M0L2)+"$, $K_1^{\\prime} = "+str(np.round(K_M1L2,7))+"$"
    ax.text(0.05,0.95, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    
    # Set the limits of y and x
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([xmin, xmax])
    # Set the spacing between the x ticks (optional)
    #ax.set_xticks(np.arange(xmin,xmax,0.1))
    # Label on the x and y axis (it accepts LateX)
    plt.xlabel("$\\lambda$",fontsize=34)
    plt.ylabel("Energy ($K_M$ units)",fontsize=38)
    # y: value of lambd; x: Energy
    y = lambd_var
    x = all_energies
    
    # Labels of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of the states that are going to be plotted
    final_label_list = []
    for i in states:
        # FOR STT(0)
        if i == 8:
            plt.plot(y,x[i], marker='o', color=colors[i],
                     linewidth=size_linewidth)
            final_label_list.append(label_list[i])
        # FOR SSS(0)
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i])
            final_label_list.append(label_list[i])
        # FOR THE REST
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth)
            final_label_list.append(label_list[i])
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    #--------------------------------------------------------------------------    
    # Show or not the gridlines
    plt.grid(True)
    # Save the Figure 
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show the Figure in Spyder
    plt.show()

################################################################
#---PLOTTING of Energy = f(Q)-----------------------------------    
################################################################
def plot_graph_Q(lambd,all_energies,Q_var, 
                 xmin,xmax,ymin,ymax,
                 K_M0L2,K_M0L3,states=[i for i in range(16)],
                 name_file="plot_graph_Q.png",
                 save=False,
                 size_linewidth=2,size_figsize=(16,10),size_fontsize=20):
    """
    This plots a graph: Energy as a function of different Q values, for a given lambda.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    lambd : float
        Value of lambda.
    all_energies : array
        Array of (16,len(Q_var)) gathering the energies of each microstate.
    Q_var : list(string)
        List of the Q values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_Q.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=size_figsize)
    # Title of Figure
    # fig.suptitle("Energy as a function of Q")
    # ax.set_title("Energy as a function of Q",pad=20)
    # Text box inside the Figure
    textstr = "$\\lambda = "+str(lambd)+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
    ax.text(0.05,0.95, textstr,transform=ax.transAxes, fontsize=size_fontsize,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set the limits of x and y
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # Label on the x and y axis (it accepts LateX)
    plt.ylabel("Energy ($K_M$ units)",fontsize=34)
    plt.xlabel("$Q = \\frac{K_1^{\\prime} - K_1}{2(K_M-K_1)}$",fontsize=38)
    # y: value of Q; x: Energy
    x = all_energies
    y = Q_var
    # Font size of Figure
    # plt.rcParams.update({'font.size': size_fontsize})
    plt.rc('axes',titlesize=30)
    plt.rc('xtick',labelsize=28)
    plt.rc('ytick',labelsize=28)
    plt.rc('legend',fontsize=22)
    
    # Labels of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of the states that are going to be plotted
    final_label_list = []
    for i in states:
        # FOR STT(0)
        if i == 8:
            plt.plot(y,x[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,
                     markevery=[j for j in range(len(y)) if j%((len(y)-1)/100) ==1])
            final_label_list.append(label_list[i])
        # FOR SSS(0)
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],
            markevery=[j for j in range(len(y)) if j%((len(y)-1)/100) ==1])
            final_label_list.append(label_list[i])
        # FOR THE REST
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth)
            final_label_list.append(label_list[i])
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    #--------------------------------------------------------------------------           
    # Show gridlines
    plt.grid(True)
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()

################################################################
#---PLOTTING of SM,SL,Sz,S2 = f(Q)------------------------------    
################################################################
def plot_graph_SM_Q(lambd, all_proj_SPIN, Q_var, 
                    xmin, xmax, ymin, ymax, 
                    K_M0L2, K_M0L3, states=[i for i in range(16)],
                    name_file="plot_graph_SM_Q.png",
                    save=False, 
                    size_linewidth=3,size_figsize=(16,10),size_fontsize=16,
                    alpha_t=1,ytick_step=0.1):
    """
    This plots a graph: Projection of the metal spin as a function of different Q values, for a given lambda.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    lambd : float
        Value of lambda.
    all_proj_SPIN : array
        array of (16,len(Q_var),4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    Q_var : list(string)
        List of the Q values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_SM_Q.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of lines. 
        The default is 1.
    ytick_step : float, optional
        Tick step on y. 
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=size_figsize)
    # Title of Figure
    # ax.set_title("$\\%(S_M = 1)$ as a function of Q",pad=20)
    # Text box inside the Figure
    
    textstr = "$\\lambda = "+str(lambd)+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
    ax.text(0.75,0.95, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set x and y axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # Set y ticks
    ax.set_yticks(np.arange(0,ymax+.01,ytick_step))
    # Labels of axis
    plt.xlabel("$Q = \\frac{K_1^{\\prime} - K_1}{2(K_M-K_1)}$",fontsize=38)
    # plt.xlabel("$Q = \\frac{K_1^{\\prime} + K_1}{2K_M}$",fontsize=38)
    # plt.xlabel("$Q = \\frac{K_1^{\\prime} - 3K_1}{2(K_M-2K_1)}$",fontsize=38)
    plt.ylabel("Proportion of local spin triplet \n on the metal",fontsize=34)
    
    # x: Proj_SM; y: variations of Q
    x = all_proj_SPIN[:,:,2] #we want Proj_SM
    y = Q_var
    
    
    # Label of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of shown states
    final_label_list = []
    for i in states:
        if i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],alpha=alpha_t)
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
    
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    # Show gridlines
    plt.grid(True)
    # Add a black horizontal at y = 0.5
    plt.axhline(y = 0.5, color = 'black', linestyle='-')
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()

def plot_graph_SM_Q_Fig3(lambd, all_proj_SPIN, Q_var, 
                    xmin, xmax, ymin, ymax, 
                    K_M0L2, K_M0L3, states=[i for i in range(16)],
                    name_file="plot_graph_SM_Q_Fig3.png",
                    save=False, 
                    size_linewidth=3,size_figsize=(16,10),size_fontsize=16,
                    alpha_t=1,ytick_step=0.1):
    """
    This plots a graph: Projection of the metal spin as a function of different Q values, for a given lambda.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    lambd : float
        Value of lambda.
    all_proj_SPIN : array
        array of (16,len(Q_var),4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    Q_var : list(string)
        List of the Q values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_SM_Q_Fig3.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of lines. 
        The default is 1.
    ytick_step : float, optional
        Tick step on y. 
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=size_figsize)
    # Title of Figure
    # ax.set_title("$\\%(S_M = 1)$ as a function of Q",pad=20)
    # Text box inside the Figure
    
    textstr = "$\\lambda = "+str(lambd)+"$"
    ax.text(0.88,0.95, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set x and y axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # Set y ticks
    ax.set_yticks(np.arange(0,ymax+.01,ytick_step))
    # Labels of axis
    plt.xlabel("$Q = \\frac{K_1^{\\prime} - K_1}{2(K_M-K_1)}$",fontsize=38)
    plt.ylabel("Proportion of local spin triplet \n on the metal",fontsize=34)
    
    # x: Proj_SM; y: variations of Q
    x = all_proj_SPIN[:,:,2] #we want Proj_SM
    y = Q_var
    
    
    # Label of each state
    label_list = ["$QTT$ $(0, \\pm1, \\pm2)$"]*5 +\
                  ["$TTT$ $(0, \\pm1)$"]*3 +\
                  ["$STT$ $(0)$"]+\
                  ["$TTS$ $(0, \\pm1)$"]*3+\
                  ["$TST$ $(0, \\pm1)$"]*3+\
                  ["$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of shown states
    final_label_list = []
    for i in states:
        if i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],alpha=alpha_t)
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
    
    
    # Legend
    fig.legend(labels=final_label_list, 
                    bbox_to_anchor=(0.5,0.97),
                   loc='upper center', 
                   # ncol = 3, 
                   ncol = 3,
                   fontsize=26)
    # Show gridlines
    plt.grid(True)
    # Add a black horizontal at y = 0.5
    plt.axhline(y = 0.5, color = 'black', linestyle='-')
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()

def plot_graph_SL_Q(lambd, all_proj_SPIN, Q_var, 
                    xmin, xmax, ymin, ymax, 
                    K_M0L2, K_M0L3, states=[i for i in range(16)],
                    name_file="plot_graph_SL_Q.png",
                    save=False, 
                    size_linewidth=3,size_figsize=(16,10),size_fontsize=16,
                    alpha_t=1,ytick_step=0.1):
    """
    This plots a graph: Projection of the metal spin as a function of different Q values, for a given lambda.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    lambd : float
        Value of lambda.
    all_proj_SPIN : array
        array of (16,len(Q_var),4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    Q_var : list(string)
        List of the Q values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_SL_Q.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of lines. 
        The default is 1.
    ytick_step : float, optional
        Tick step on y. 
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=size_figsize)
    # Title of Figure
    # ax.set_title("$\\%(S_M = 1)$ as a function of Q",pad=20)
    # Text box inside the Figure
    
    textstr = "$\\lambda = "+str(lambd)+"$"
    # textstr = "$\\lambda = "+str(lambd)+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
    # ax.text(0.75,0.95, textstr,transform=ax.transAxes, fontsize=24,
    ax.text(0.88,0.95, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set x and y axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # Set y ticks
    ax.set_yticks(np.arange(0,ymax+.01,ytick_step))
    # Labels of axis
    plt.xlabel("$Q = \\frac{K_1^{\\prime} - K_1}{2(K_M-K_1)}$",fontsize=38)
    plt.ylabel("Proportion of local spin triplet \n on the ligand",fontsize=34)
    
    # x: Proj_SM; y: variations of Q
    x = all_proj_SPIN[:,:,3] #we want Proj_SM
    y = Q_var
    # Font size
    # plt.rcParams.update({'font.size': size_fontsize})
    plt.rc('axes',titlesize=30)
    plt.rc('xtick',labelsize=28)
    plt.rc('ytick',labelsize=28)
    plt.rc('legend',fontsize=22)
    
    # Label of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of shown states
    final_label_list = []
    for i in states:
        if i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],alpha=alpha_t)
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
    
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    # Show gridlines
    plt.grid(True)
    # Add a black horizontal at y = 0.5
    plt.axhline(y = 0.5, color = 'black', linestyle='-')
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()

def plot_graph_Sz_Q(lambd, all_proj_SPIN, Q_var, 
                    xmin, xmax, ymin, ymax, 
                    K_M0L2, K_M0L3, states=[i for i in range(16)],
                    name_file="plot_graph_Sz_Q.png",
                    save=False, 
                    size_linewidth=3,size_figsize=(16,10),size_fontsize=16,
                    alpha_t=1,ytick_step=1):
    """
    This plots a graph: Projection of the metal spin as a function of different Q values, for a given lambda.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    lambd : float
        Value of lambda.
    all_proj_SPIN : array
        array of (16,len(Q_var),4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    Q_var : list(string)
        List of the Q values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_Sz_Q.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of lines. 
        The default is 1.
    ytick_step : float, optional
        Tick step on y. 
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=size_figsize)
    # Title of Figure
    # ax.set_title("$\\%(S_M = 1)$ as a function of Q",pad=20)
    # Text box inside the Figure
    
    textstr = "$\\lambda = "+str(lambd)+"$"
    # textstr = "$\\lambda = "+str(lambd)+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
    # ax.text(0.75,0.95, textstr,transform=ax.transAxes, fontsize=24,
    ax.text(0.88,0.95, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set x and y axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # Set y ticks
    ax.set_yticks(np.arange(ymin+0.01,ymax+.01,ytick_step))
    # Labels of axis
    plt.xlabel("$Q = \\frac{K_1^{\\prime} - K_1}{2(K_M-K_1)}$",fontsize=38)
    plt.ylabel("$M_{S_{tot}}$",fontsize=34)
    
    # x: Proj_SM; y: variations of Q
    x = all_proj_SPIN[:,:,0] #we want Ms
    y = Q_var
    # Font size
    # plt.rcParams.update({'font.size': size_fontsize})
    plt.rc('axes',titlesize=30)
    plt.rc('xtick',labelsize=28)
    plt.rc('ytick',labelsize=28)
    plt.rc('legend',fontsize=22)
    
    # Label of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of shown states
    final_label_list = []
    for i in states:
        if i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],alpha=alpha_t)
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
    
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    # Show gridlines
    plt.grid(True)
    # Add a black horizontal at y = 0.5
    plt.axhline(y = 0.5, color = 'black', linestyle='-')
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()

def plot_graph_S2_Q(lambd, all_proj_SPIN, Q_var, 
                    xmin, xmax, ymin, ymax, 
                    K_M0L2, K_M0L3, states=[i for i in range(16)],
                    name_file="plot_graph_S2_Q.png",
                    save=False, 
                    size_linewidth=3,size_figsize=(16,10),size_fontsize=16,
                    alpha_t=1,ytick_step=0.1):
    """
    This plots a graph: Projection of the metal spin as a function of different Q values, for a given lambda.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    lambd : float
        Value of lambda.
    all_proj_SPIN : array
        array of (16,len(Q_var),4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    Q_var : list(string)
        List of the Q values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_S2_Q.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of lines. 
        The default is 1.
    ytick_step : float, optional
        Tick step on y. 
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=size_figsize)
    # Title of Figure
    # ax.set_title("$\\%(S_M = 1)$ as a function of Q",pad=20)
    # Text box inside the Figure
    
    textstr = "$\\lambda = "+str(lambd)+"$"
    ax.text(0.88,0.95, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set x and y axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # Set y ticks
    ax.set_yticks(np.arange(0,ymax+.01,0.5))
    # Labels of axis
    plt.xlabel("$Q = \\frac{K_1^{\\prime} - K_1}{2(K_M-K_1)}$",fontsize=38)
    plt.ylabel("$S_{tot}(S_{tot}+1)$",fontsize=34)
    
    # x: Proj_SM; y: variations of Q
    x = all_proj_SPIN[:,:,1] #we want S(S+1)
    y = Q_var
    # Font size
    # plt.rcParams.update({'font.size': size_fontsize})
    plt.rc('axes',titlesize=30)
    plt.rc('xtick',labelsize=28)
    plt.rc('ytick',labelsize=28)
    plt.rc('legend',fontsize=22)
    
    # Label of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of shown states
    final_label_list = []
    for i in states:
        if i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],alpha=alpha_t)
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
    
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    # Show gridlines
    plt.grid(True)
    # Add a black horizontal at y = 0.5
    plt.axhline(y = 0.5, color = 'black', linestyle='-')
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()
    
################################################################
#---PLOTTING of SM,SL,Sz,S2 = f(lambd)--------------------------    
################################################################
def plot_graph_SM_lambd(Q, all_proj_SPIN, lambd_var, 
                        xmin, xmax, ymin, ymax, 
                        K_M0L2, K_M0L3, states=[i for i in range(16)],
                        name_file="plot_graph_SM_lambd.png",
                        save=False, 
                        size_linewidth=3,size_figsize=(16,10),size_fontsize=16,
                        alpha_t=1,ytick_step=0.1):
    """
    This plots a graph: Projection of the metal spin as a function of different lambda values, for a given Q.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    Q : float
        Value of Q.
    all_proj_SPIN : array
        array of (16,len(lambd_var),4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    lambd_var : list(string)
        List of the lambda values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_SM_lambd.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of lines. 
        The default is 1.
    ytick_step : float, optional
        Tick step on y. 
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=(16,10))
    # Title of Figure
    # ax.set_title("$\\%(S_M = 1)$ as a function of $\\lambda$",pad=20)
    # Text box inside the graph
    textstr = "$Q = "+str(np.round(Q,7))+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
    ax.text(0.75,0.9, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set x and y axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    
    # Set y ticks
    ax.set_yticks(np.arange(0,ymax+.01,ytick_step))
    # Labels of x and y axis
    plt.xlabel("$\\lambda$",fontsize=38)
    plt.ylabel("Proportion of local spin triplet \n on the metal",fontsize=34)
    # x: Proj_SM; y: variations of Q
    x = all_proj_SPIN[:,:,2] #we want Proj_SM
    y = lambd_var
    
    
    # Labels of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of shown states
    final_label_list = []
    for i in states:
        if i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],alpha=alpha_t)
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    # Show gridlines
    plt.grid(True)
    # Add a black horizontal at y = 0.5
    plt.axhline(y = 0.5, color = 'black', linestyle='-')
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()
    
def plot_graph_SL_lambd(Q, all_proj_SPIN, lambd_var, 
                        xmin, xmax, ymin, ymax, 
                        K_M0L2, K_M0L3, states=[i for i in range(16)],
                        name_file="plot_graph_SL_lambd.png",
                        save=False, 
                        size_linewidth=3,size_figsize=(16,10),size_fontsize=16,
                        alpha_t=1,ytick_step=0.1):
    """
    This plots a graph: Projection of the metal spin as a function of different lambda values, for a given Q.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    Q : float
        Value of Q.
    all_proj_SPIN : array
        array of (16,len(lambd_var),4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    lambd_var : list(string)
        List of the lambda values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_SL_lambd.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of lines. 
        The default is 1.
    ytick_step : float, optional
        Tick step on y. 
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=(16,10))
    # Title of Figure
    # ax.set_title("$\\%(S_M = 1)$ as a function of $\\lambda$",pad=20)
    # Text box inside the graph
    textstr = "$Q = "+str(np.round(Q,7))+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
    ax.text(0.75,0.9, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set x and y axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    
    # Set y ticks
    ax.set_yticks(np.arange(0,ymax+.01,ytick_step))
    # Labels of x and y axis
    plt.xlabel("$\\lambda$",fontsize=38)
    plt.ylabel("Proportion of local spin triplet \n on the ligand",fontsize=34)
    # x: Proj_SM; y: variations of Q
    x = all_proj_SPIN[:,:,3] #we want Proj_SL
    y = lambd_var
    
    
    # Labels of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of shown states
    final_label_list = []
    for i in states:
        if i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],alpha=alpha_t)
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    # Show gridlines
    plt.grid(True)
    # Add a black horizontal at y = 0.5
    plt.axhline(y = 0.5, color = 'black', linestyle='-')
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()

def plot_graph_Sz_lambd(Q, all_proj_SPIN, lambd_var, 
                        xmin, xmax, ymin, ymax, 
                        K_M0L2, K_M0L3, states=[i for i in range(16)],
                        name_file="plot_graph_Sz_lambd.png",
                        save=False, 
                        size_linewidth=3,size_figsize=(16,10),size_fontsize=16,
                        alpha_t=1,ytick_step=0.5):
    """
    This plots a graph: Projection of the metal spin as a function of different lambda values, for a given Q.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    Q : float
        Value of Q.
    all_proj_SPIN : array
        array of (16,len(lambd_var),4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    lambd_var : list(string)
        List of the lambda values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_Sz_lambd.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of lines. 
        The default is 1.
    ytick_step : float, optional
        Tick step on y. 
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=(16,10))
    # Title of Figure
    # ax.set_title("$\\%(S_M = 1)$ as a function of $\\lambda$",pad=20)
    # Text box inside the graph
    textstr = "$Q = "+str(np.round(Q,7))+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
    ax.text(0.75,0.9, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set x and y axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    
    # Set y ticks
    ax.set_yticks(np.arange(ymin+0.01,ymax+.01,ytick_step))
    # Labels of x and y axis
    plt.xlabel("$\\lambda$",fontsize=38)
    plt.ylabel("$M_{S_{tot}}$",fontsize=34)
    # x: Proj_SM; y: variations of Q
    x = all_proj_SPIN[:,:,0] #we want Ms_tot
    y = lambd_var
    
    
    # Labels of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of shown states
    final_label_list = []
    for i in states:
        if i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],alpha=alpha_t)
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    # Show gridlines
    plt.grid(True)
    # Add a black horizontal at y = 0.5
    plt.axhline(y = 0.5, color = 'black', linestyle='-')
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()

def plot_graph_S2_lambd(Q, all_proj_SPIN, lambd_var, 
                        xmin, xmax, ymin, ymax, 
                        K_M0L2, K_M0L3, states=[i for i in range(16)],
                        name_file="plot_graph_S2_lambd.png",
                        save=False, 
                        size_linewidth=3,size_figsize=(16,10),size_fontsize=16,
                        alpha_t=1,ytick_step=0.5):
    """
    This plots a graph: Projection of the metal spin as a function of different lambda values, for a given Q.
    One can select the states to plot by giving an indices list.
    It can plot a maximum of 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
        m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
    The indices of the states are the following:
    0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
    5,6,7:        TTTm, TTTz, TTTp, 
    8:            STTz,
    9,10,11:      TTSm, TTSz, TTSp, 
    12,13,14:     TSTm, TSTz, TSTp, 
    15:           SSSz

    Parameters
    ----------
    Q : float
        Value of Q.
    all_proj_SPIN : array
        array of (16,len(lambd_var),4) gathering [m_s,S(S+1),%S_M=1,%S_L=1].
    lambd_var : list(string)
        List of the lambda values.
    xmin : float
        Minimum value on x axis.
    xmax : float
        Maximum value on x axis.
    ymin : float
        Minimum value on y axis.
    ymax : float
        Maximum value on y axis.
    K_M0L2 : float
        Value of K_1 integral.
    K_M1L2 : float
        Value of K_1' integral.
    states : list(int), optional
        List of indices of the states to be plotted. 
        The default is [i for i in range(16)].
    name_file : string, optional
        Name of the file to be created if save == True. 
        The default is "plot_graph_S2_lambd.png".
    save : bool, optional
        Save the Figure if True. 
        The default is False.
    size_linewidth : int, optional
        Size of the lines. 
        The default is 2.
    size_figsize : tuple(int,int), optional
        (x,y) size of the Figure in cm. 
        The default is (16,10).
    size_fontsize : int, optional
        Size of the font. 
        The default is 16.
    alpha_t : float, optional
        Transparency of lines. 
        The default is 1.
    ytick_step : float, optional
        Tick step on y. 
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Size of Figure
    fig, ax = plt.subplots(figsize=(16,10))
    # Title of Figure
    # ax.set_title("$\\%(S_M = 1)$ as a function of $\\lambda$",pad=20)
    # Text box inside the graph
    textstr = "$Q = "+str(np.round(Q,7))+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
    ax.text(0.75,0.9, textstr,transform=ax.transAxes, fontsize=24,
            verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
    # Set x and y axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    
    # Set y ticks
    ax.set_yticks(np.arange(0,ymax+.01,0.5))
    # Labels of x and y axis
    plt.xlabel("$\\lambda$",fontsize=38)
    plt.ylabel("$S_{tot}(S_{tot}+1)$",fontsize=34)
    # x: Proj_SM; y: variations of Q
    x = all_proj_SPIN[:,:,1] #we want SS+1
    y = lambd_var
    
    
    # Labels of each state
    label_list = ["$QTT$ $(-2)$", "$QTT$ $(-1)$", "$QTT$ $(0)$", "$QTT$ $(+1)$", "$QTT$ $(+2)$", 
                  "$TTT$ $(-1)$", "$TTT$ $(0)$", "$TTT$ $(+1)$", 
                  "$STT$ $(0)$", 
                  "$TTS$ $(-1)$", "$TTS$ $(0)$", "$TTS$ $(+1)$", 
                  "$TST$ $(-1)$", "$TST$ $(0)$", "$TST$ $(+1)$", 
                  "$SSS$ $(0)$"]
    # Linestyle of each state
    linestyles = ["dashdot","dashed","solid","dotted","dashdot",
                  "dashed","solid","dotted",
                  "solid",
                  "dashed","solid","dotted",
                  "dashed","solid","dotted"]
    # Color of each state
    colors = ["black","black","black","black","black",
              "blue","blue","blue",
              "black",
              "green","green","green",
              "red","red","red",
              "black"]
    # Store the label of shown states
    final_label_list = []
    for i in states:
        if i == 8:
            plt.plot(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
        elif i == 15:
            plt.plot(y,x[i],'x', color=colors[i],alpha=alpha_t)
            final_label_list.append(label_list[i])
        else:
            plt.plot(y,x[i],linestyle=linestyles[i], color=colors[i],
                     linewidth=size_linewidth,alpha=alpha_t)
            final_label_list.append(label_list[i])
    
    # Legend
    plt.legend(labels=final_label_list, bbox_to_anchor=(1,0.5),loc='center left')
    # Show gridlines
    plt.grid(True)
    # Add a black horizontal at y = 0.5
    plt.axhline(y = 0.5, color = 'black', linestyle='-')
    # Save Figure
    if save == True:
        plt.savefig(name_file, bbox_inches='tight')
    # Show Figure in Spyder
    plt.show()



