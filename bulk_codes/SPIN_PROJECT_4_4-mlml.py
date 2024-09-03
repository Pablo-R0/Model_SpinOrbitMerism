import quantnbody as qnb 
import numpy as np  
import scipy
import matplotlib.pyplot as plt
import math    

np.set_printoptions(precision=6) # For nice numpy matrix printing

#%% Graphs Modules 
## Functions for plotting the Curves (Pablo)
from tools_plot import plot_graph_mlml, plot_ms_S2_ProjSM_ProjSL
## Function to analyze the results
from tools_plot import curve_analysis_spin
## Function to build the 1- and 2-electron matrices
from tools_plot import build_basis
#%% 1)  Parameters of the Hamiltonian
#-------------
# This part defines the parameters of the Hamiltonian
#-------------
# Number of MOs and of electrons
N_MO = N_elec = 4
dim_H  = math.comb( 2*N_MO, N_elec ) # Dimension of the many-body space 
# Index of the metal MOs
index_metal  = [ 0, 1 ]
# Index of the ligand MOs
index_ligand = [ 2, 3 ] 
#--- S.O.C parameters
# l_metal is the azimutal quantum number (1:p, 2:d, 3:f, 4:g)
l_metal = 2 
# metal_l are the local angular momentum of each MO on the metal
metal_l = [-2.,-1.0]
#---
# I rename the values to paste them into the code
list_mo_local, l_local, list_l_local = index_metal, l_metal, metal_l

#------------- GLOBAL -------------
# Build the many-body basis            
nbody_basis = qnb.fermionic.tools.build_nbody_basis( N_MO, N_elec )     
# Build the matrix representation of the a_dagger_a operator in the many-body basis                       
a_dagger_a  = qnb.fermionic.tools.build_operator_a_dagger_a( nbody_basis ) 
# Build the matrix representation of the global S2, S_z and S_plus spin operators in the many-body basis  
S2, S_Z, S_p = qnb.fermionic.tools.build_s2_sz_splus_operator( a_dagger_a )
#------------- LOCAL -------------
# Build the matrix representation of the local S2, S_z and S_plus spin operators (metal) in the many-body basis  
S2_metal, S_z_metal, S_plus_metal  = qnb.fermionic.tools.build_local_s2_sz_splus_operator( a_dagger_a, 
                                                                                          index_metal )
# Build the matrix representation of the local S2, S_z and S_plus spin operators (ligand) in the many-body basis  
S2_ligand, S_z_ligand, S_plus_ligand = qnb.fermionic.tools.build_local_s2_sz_splus_operator( a_dagger_a, 
                                                                                            index_ligand )
# Build the matrix representation of the local triplet projection (metal) in the many-body basis  
Proj_triplet_metal = qnb.fermionic.tools.build_spin_subspaces(S2_metal, 2)
# Build the matrix representation of the local triplet projection (ligand) in the many-body basis  
Proj_triplet_ligand = qnb.fermionic.tools.build_spin_subspaces(S2_ligand, 2)

# Build the matrix representation of the penalty operator: penalty applied to the doubly occupied determinants
penalty    = 1e3 
Op_penalty = scipy.sparse.csr_matrix((np.shape(nbody_basis)[0],np.shape(nbody_basis)[0]))
for p in range(0,N_MO): 
    Op_penalty += 1*((a_dagger_a[2*p,2*p]+a_dagger_a[2*p+1,2*p+1]) == 2)

#------------- PARAMETERS OF THE SYSTEM --------------
# Energy of each MO
h00,h11,h22,h33 = 0., 0.5, -1.0, -0.7   # metal, metal', ligand1, ligand2
# K_M, K_LL
K_M0M1, K_L2L3 = 1, 1e-5
# K_ML
K_M0L2 = K_M0L3 = 0.8
# K_M'L
K_M1L2 = K_M1L3 = 1.2
## U and t
U_M, U_L = 10., 10. 
t_M, t_L, t_ML = 0, 0, 0 
# Create the one- and two-electron matrices
h_, g_ = build_basis(N_MO, index_metal, index_ligand,
                     h00,h11,h22,h33,K_M0M1,K_L2L3,
                     K_M0L2,K_M0L3,K_M1L2,K_M1L3,
                     U_M,U_L,t_M,t_L,t_ML)
# Build the matrix representation of the Hamiltonian operator
H       = qnb.fermionic.tools.build_hamiltonian_quantum_chemistry(h_, g_, nbody_basis, a_dagger_a)

# There are 16 states TOTALSPIN/METALSPIN/LIGANDSPIN/M_S : 
    # m_s : mm = -2, m = -1, z = 0, p = +1, pp = +2
# The states are ordered as the following:
# 0,1,2,3,4:    QTTmm, QTTm, QTTz, QTTp, QTTpp, 
# 5,6,7:        TTTm, TTTz, TTTp, 
# 8:            STTz,
# 9,10,11:      TTSm, TTSz, TTSp, 
# 12,13,14:     TSTm, TSTz, TSTp, 
# 15:           SSSz

#%% 1)  CALCULATION DATA for variation of Energy spectrum = f({ml,ml'}) + [m_s, S(S+1),%S_M=1,%S_L=1] = f(state)
#-------------
# This part constructs the data for the variation of the Energy spectrum as function of {ml,ml'} for a given lambd
# It also produces information on the [m_s, S(S+1),%S_M=1,%S_L=1] of each microstate for a given {ml,ml'} and lambd
#-------------
K_M0L2 = K_M0L3 = 0.8
# K_M'L
K_M1L2 = K_M1L3 = 0.8
# Value of lambd
lambd = 0.1
# Create the one- and two-electron matrices
h_, g_ = build_basis(N_MO, index_metal, index_ligand,
                     h00,h11,h22,h33,K_M0M1,K_L2L3,
                     K_M0L2,K_M0L3,K_M1L2,K_M1L3,
                     U_M,U_L,t_M,t_L,t_ML)
# Build the matrix representation of the Hamiltonian operator
H       = qnb.fermionic.tools.build_hamiltonian_quantum_chemistry(h_, g_, nbody_basis, a_dagger_a)

# All possible combinations of {ml,ml'}
comb_metal_l = [[0.,0.],[-2.,-1.],[-2.,0.],[-2.,1.],[-1.,0.],[-2.,2.],[-1.,1.],[-1.,2.],[0.,1.],[0.,2.],[1.,2.]]



# Initialize the array for the eigenvalues
all_energies = np.zeros((16,len(comb_metal_l)))
# Initialize the array for [m_s, S(S+1),%S_M=1,%S_L=1]
all_proj_SPIN = np.zeros((16,len(comb_metal_l),4))


for comb in range(len(comb_metal_l)):
    l_local, list_l_local = l_metal, comb_metal_l[comb]
    H_SO = qnb.fermionic.tools.build_local_spinorbit_lz(a_dagger_a, list_mo_local,l_local, list_l_local)
    H_tot = H + lambd * H_SO
    eigval_SO, eigvec_SO = scipy.linalg.eigh( H_tot.toarray()  ) 
    all_energies, all_proj_SPIN = curve_analysis_spin(comb, all_energies, all_proj_SPIN, eigval_SO, eigvec_SO, 
                                       S_Z, S2, Proj_triplet_metal, Proj_triplet_ligand,
                                       Op_penalty,penalty,use_index=True)

                    

#%% FIGURE SI1: E = f({ml,ml'}) for given lambd
#-------------
# This part plots the data for the variation of the Energy spectrum as function of {ml,ml'} for given lambd
#-------------

# It will generate a pdf Figure named 
name_file = "K1_"+str(K_M0L2)+"_K1p_"+str(np.round(K_M1L2,7))+"_comb_ml_ml_full"+".pdf"

# Indices of states to be plotted
states = [i for i in range(16)]

# Min and Max on y axis
ymin = -6. 
ymax = -0.5 
# Label of each {ml,ml'}
label_y = ["no\nS.O.C","$\\{-2;-1\\}$","$\\{-2;0\\}$","$\\{-2;1\\}$","$\\{-1;0\\}$","$\\{-2;2\\}$","$\\{-1;1\\}$","$\\{-1;2\\}$","$\\{0;1\\}$","$\\{0;2\\}$","$\\{1;2\\}$"]

# Font size
plt.rc('axes',titlesize=30)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=28)
plt.rc('legend',fontsize=22)    
plot_graph_mlml(lambd,all_energies,label_y,
                ymin,ymax,
                K_M0L2,K_M1L2,states,
                name_file,save=True)

#%% 1b)PLOT [m_s, S(S+1),%S_M=1,%S_L=1] = f(state) for given {ml,ml'} and lambd
#-------------
# This part plots information on [m_s, S(S+1),%S_M=1,%S_L=1] of each microstate for a given {ml,ml'} = comb_metal_l[lambd_val] and lambd = epsi
#-------------
Q = (K_M1L2 - K_M0L2) / (2*(K_M0M1 - K_M0L2))
# The index of {ml,ml'} you want to see
mlml_val = 1 # It is the -2,-1
label_y = ["no\nS.O.C","$\\{-2;-1\\}$","$\\{-2;0\\}$","$\\{-2;1\\}$","$\\{-1;0\\}$","$\\{-2;2\\}$","$\\{-1;1\\}$","$\\{-1;2\\}$","$\\{0;1\\}$","$\\{0;2\\}$","$\\{1;2\\}$"]

# Name of the pdf file
name_file = "K1_"+str(K_M0L2)+"_K1p_"+str(np.round(K_M1L2,7))+"_lambd_"+str(lambd)+"ml-2_ml-1"+".pdf"
# Font size
plt.rc('axes',titlesize=30)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.rc('legend',fontsize=22) 
plot_ms_S2_ProjSM_ProjSL(lambd, Q, mlml_val, all_proj_SPIN,
                         K_M0L2, print_var="mlml",
                         name_file=name_file,save=True,mlml=label_y[mlml_val])


