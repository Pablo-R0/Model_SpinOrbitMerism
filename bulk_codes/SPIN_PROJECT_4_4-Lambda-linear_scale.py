import quantnbody as qnb 
import numpy as np  
import scipy
import matplotlib.pyplot as plt
import math    

np.set_printoptions(precision=6) # For nice numpy matrix printing


#%% Graphs Modules 
## Functions for plotting the Curves (Pablo)
from tools_plot import plot_ms_S2_ProjSM_ProjSL,plot_graph_lambd,plot_graph_SM_lambd
## Function to analyze the results
from tools_plot import curve_analysis_spin
## Function to build the 1- and 2-electron matrices
from tools_plot import build_basis
from tools_plot import plot_graph_Sz_lambd,plot_graph_S2_lambd,plot_graph_SL_lambd
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
U_M, U_L = 10., 10. #5., 1.
t_M, t_L, t_ML = 0, 0, 0 #5e-5, 5e-5, 1.
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

#%% 2)  CALCULATION DATA for variation of Energy spectrum = f(lambd) + [m_s, S(S+1),%S_M=1,%S_L=1] = f(state)
#-------------
# This part constructs the data for the variation of the Energy spectrum as function of lambd 
# (defined between xmin and xmax, with a step)
# It also produces information on the [m_s, S(S+1),%S_M=1,%S_L=1] of each microstate for a given lambd
#-------------

# Build the matrix representation of the Spin-Orbit Hamiltonian operator 
H_SO = qnb.fermionic.tools.build_local_spinorbit_lz(a_dagger_a, list_mo_local,l_local, list_l_local)

xmin=0
xmax=5.1
step=(xmax-xmin)/100
lambd_var = np.arange(xmin,xmax,step)


# Initialize the array for the eigenvalues
all_energies = np.zeros((16,len(lambd_var)))
# Initialize the array for [m_s, S(S+1),%S_M=1,%S_L=1]
all_proj_SPIN = np.zeros((len(all_energies),len(lambd_var),4))

for eps in range(len(lambd_var)):
    
    lambd = lambd_var[eps]
    H_tot = H + lambd * H_SO
    eigval_SO, eigvec_SO = scipy.linalg.eigh( H_tot.toarray()  ) 
    # Specific function constructed by Pablo   
    all_energies, all_proj_SPIN = curve_analysis_spin(eps, all_energies, all_proj_SPIN, eigval_SO, eigvec_SO, 
                                        S_Z, S2, Proj_triplet_metal, Proj_triplet_ligand,
                                        Op_penalty, penalty,use_index=True,)     


#%% 2a) PLOT variation of Energy spectrum = f(lambd)
#-------------
# This part plots the data for the variation of the Energy spectrum as function of lambd 
#-------------
# It will generate a pdf Figure named 
name_file = "K1_"+str(K_M0L2)+"_K1p_"+str(np.round(K_M1L2,7))+"_fct_lambd"+".pdf"

# Indices of states to be plotted
# states = [5,6,7,8,12,13,14]
states = [i for i in range(16)]
# Min and Max of the Energy axis
ymin=-20
ymax=15 
save = True
# Font size of Figure
plt.rc('axes',titlesize=30)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.rc('legend',fontsize=22)
plot_graph_lambd(all_energies, lambd_var, 
                 xmin, xmax, ymin, ymax,
                 K_M0L2,K_M1L2,states,
                 name_file,save,)   

#%% 2b) PLOT ProjS_M = f(lambda)
Q = (K_M1L2 - K_M0L2) / (2*(K_M0M1 - K_M0L2))
# Name of the pdf file
name_file = "Q_"+str(np.round(Q,7))+"K1_"+str(K_M0L2)+"_Proj_SM_var_lambda"+str(xmax)+".pdf"

states = [5,6,7,
          8,
          12,13,14]
# states = [i for i in range(16)]
# states = [2]
# Min and Max of y axis
ymin=-0.01
ymax=1.01
# Min and Max of x axis, adding a step (for a more beautiful graph)
step = 0.02
xmin2=xmin - step
xmax2=xmax + step

save = True
# Font size
plt.rc('axes',titlesize=30)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.rc('legend',fontsize=22)
plot_graph_SM_lambd(Q, all_proj_SPIN, lambd_var,
                 xmin, xmax, ymin, ymax,
                 K_M0L2, K_M0L3, states,
                 name_file, save)
#%% 2b) PLOT ProjS_L = f(lambda)
Q = (K_M1L2 - K_M0L2) / (2*(K_M0M1 - K_M0L2))
# Name of the pdf file
name_file = "Q_"+str(np.round(Q,7))+"K1_"+str(K_M0L2)+"_Proj_SL_var_lambda"+str(xmax)+".pdf"

# states = [5,6,7,
#           8,
#           12,13,14]
# states = [i for i in range(16)]
# Min and Max of y axis
ymin=-0.01
ymax=1.01
# Min and Max of x axis, adding a step (for a more beautiful graph)
step = 0.02
xmin2=xmin - step
xmax2=xmax + step

save = True
# Font size
plt.rc('axes',titlesize=30)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.rc('legend',fontsize=22)
plot_graph_SL_lambd(Q, all_proj_SPIN, lambd_var,
                 xmin, xmax, ymin, ymax,
                 K_M0L2, K_M0L3, states,
                 name_file, save)
#%% 2b) PLOT M_stot = f(lambda)
Q = (K_M1L2 - K_M0L2) / (2*(K_M0M1 - K_M0L2))
# Name of the pdf file
name_file = "Q_"+str(np.round(Q,7))+"K1_"+str(K_M0L2)+"_Proj_Sz_var_lambda"+str(xmax)+".pdf"

# states = [5,6,7,
#           8,
#           12,13,14]
states = [i for i in range(16)]
# Min and Max of y axis
ymin=-2.01
ymax=2.01

save = True
# Font size
plt.rc('axes',titlesize=30)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.rc('legend',fontsize=22)
plot_graph_Sz_lambd(Q, all_proj_SPIN, lambd_var,
                 xmin, xmax, ymin, ymax,
                 K_M0L2, K_M0L3, states,
                 name_file, save)

#%% 2b) PLOT S(S+1) = f(lambda)
Q = (K_M1L2 - K_M0L2) / (2*(K_M0M1 - K_M0L2))
# Name of the pdf file
name_file = "Q_"+str(np.round(Q,7))+"K1_"+str(K_M0L2)+"_Proj_S2_var_lambda"+str(xmax)+".pdf"

# states = [5,6,7,
#           8,
#           12,13,14]
states = [i for i in range(16)]
# states = [2]
# Min and Max of y axis
ymin=0
ymax=6.01

save = True
# Font size
plt.rc('axes',titlesize=30)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.rc('legend',fontsize=22)
plot_graph_S2_lambd(Q, all_proj_SPIN, lambd_var,
                 xmin, xmax, ymin, ymax,
                 K_M0L2, K_M0L3, states,
                 name_file, save)


#%% 2c) PLOT [m_s, S(S+1),%S_M=1,%S_L=1] = f(state) for given lambd
#-------------
# This part plots information on [m_s, S(S+1),%S_M=1,%S_L=1] of each microstate for a given lambd = epsi
#-------------

# The value of lambd that you want to see
lamb = lambd_var[-1] #lambd_var[39] #0.3
# This loop searches for the index of epsi in lambd_var
# for var in range(len(lambd_var)):
#     if round(lambd_var[var],7) == lamb:
#         print(var)
#         lamb_val = var
lamb_val = -1
# epsi = 4.04
# lambd_val = 40
# Number of significative digits to be shown
number_float = '%.2f'

# Name of the pdf file
name_file = "K1_"+str(K_M0L2)+"_K1p_"+str(np.round(K_M1L2,7))+"_lambd_"+str(lambd)+".pdf"

save = False
# Font size
plt.rc('axes',titlesize=30)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.rc('legend',fontsize=22) 
plot_ms_S2_ProjSM_ProjSL(Q,lamb,lamb_val,all_proj_SPIN,
                         K_M0L2,print_var="lambda",
                         name_file=name_file,save=save)
