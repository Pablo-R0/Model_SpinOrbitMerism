import quantnbody as qnb 
import numpy as np  
import scipy
import matplotlib.pyplot as plt
import math    

np.set_printoptions(precision=6) # For nice numpy matrix printing
# Font size
plt.rc('axes',titlesize=30)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.rc('legend',fontsize=22)

#%% Graphs Modules 
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
# metal_l = [-1.,1.0]
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
Q = (K_M1L2 - K_M0L2) / (2*(K_M0M1 - K_M0L2))
# Build the matrix representation of the Spin-Orbit Hamiltonian operator 
H_SO = qnb.fermionic.tools.build_local_spinorbit_lz(a_dagger_a, list_mo_local,l_local, list_l_local)

# Log space
lambd_var = np.logspace(-2,2,num=100)

all_energies = np.zeros((16,len(lambd_var)))
all_proj_SPIN = np.zeros((len(all_energies),len(lambd_var),4))

for eps in range(len(lambd_var)):
    lambd = lambd_var[eps]
    H_tot = H + lambd * H_SO
    eigval_SO, eigvec_SO = scipy.linalg.eigh( H_tot.toarray()  ) 
    all_energies, all_proj_SPIN = curve_analysis_spin(eps, all_energies, all_proj_SPIN, eigval_SO, eigvec_SO, 
                                        S_Z, S2, Proj_triplet_metal, Proj_triplet_ligand,
                                        Op_penalty, penalty,use_index=True,)

#%% FIGURE SI6: S(S+1) = f(lambda) : Logscale
from matplotlib.ticker import ScalarFormatter

states = [5,6,7,
          8,
          12,13,14]
states = [1,2,3,5,6,7,8,12,13,14]
# states = [i for i in range(16)]

ytick_step = 0.1
size_linewidth = 3
# Size of Figure
fig = plt.figure(figsize=(16,10))
gs = fig.add_gridspec(nrows=1, ncols=1)
ax = gs.subplots()


# Show gridlines
ax.grid(True)
# Add a black horizontal at y = 0.5
# Save Figure
ax.set_ylim([-0.03, 6.03])
# ax.set_ylim([-0.03, 4.03])
ax.set_yticks(np.arange(0,6+.03,1))
# ax.set_yticks(np.arange(0,4+.03,1))
# Title of Figure
# Text box inside the graph
textstr = "$Q = "+str(np.round(Q,7))+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
# ax.text(0.76,0.95, textstr,transform=ax.transAxes, fontsize=24,
ax.text(0.76,0.95, textstr,transform=ax.transAxes, fontsize=24,
        verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
# Set x and y axis limits
ax.set_xlim([0.1, 100])
# ax.set_xscale('log')
ax.set_xticks([0.1,1,10,100])



# Labels of x and y axis
ax.set_xlabel("$\\lambda$",fontsize=38)
ax.set_ylabel("$S_{tot}(S_{tot}+1)$",fontsize=34)

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


# x: Proj_SM; y: variations of Q
x = all_proj_SPIN[:,:,1] #we want S(S+1)
y = lambd_var


# Store the label of shown states
final_label_list = []
lines = []
for i in states:
    if i == 8:
        line, =ax.semilogx(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                 linewidth=size_linewidth,markersize=7,
                 markevery=[j for j in range(len(y)) if j%5 ==0])
        final_label_list.append(label_list[i])
    elif i == 15:
        line, =ax.semilogx(y,x[i],'x', color=colors[i],markersize=10,
                    markevery=[j for j in range(len(y)) if j%5 ==0])
        final_label_list.append(label_list[i])
    else:
        line, =ax.semilogx(y,x[i],linestyle=linestyles[i], color=colors[i],
                 linewidth=size_linewidth)
        final_label_list.append(label_list[i])
    lines += [line]
# Legend
fig.legend(handles=lines,labels=final_label_list, bbox_to_anchor=(0.9,0.5),loc='center left')

ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xticks([0.01,0.1,1,10,100])
name_file = "Q_"+str(np.round(Q,7))+"K1_"+str(K_M0L2)+"_Proj_S2_var_lambda.pdf"


plt.savefig(name_file, bbox_inches='tight')
# Show Figure in Spyder
plt.show()

#%% FIGURE 6: ProjSM = f(lambda) : Logscale
from matplotlib.ticker import ScalarFormatter

states = [5,6,7,
          8,
          12,13,14]
# states = [i for i in range(16)]

ytick_step = 0.1
size_linewidth = 3
# Size of Figure
fig = plt.figure(figsize=(16,10))
gs = fig.add_gridspec(nrows=1, ncols=1)
ax = gs.subplots()


# Show gridlines
ax.grid(True)
# Add a black horizontal at y = 0.5
plt.axhline(y = 0.5, color = 'black', linestyle='-')
# Save Figure
ax.set_ylim([-0.01, 1.01])
ax.set_yticks(np.arange(0,1+.01,0.1))
# Title of Figure
# Text box inside the graph
textstr = "$Q = "+str(np.round(Q,7))+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
# ax.text(0.76,0.95, textstr,transform=ax.transAxes, fontsize=24,
ax.text(0.76,0.9, textstr,transform=ax.transAxes, fontsize=24,
        verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))
# Set x and y axis limits
ax.set_xlim([0.01, 100])



# Labels of x and y axis
ax.set_xlabel("$\\lambda$",fontsize=38)
ax.set_ylabel("Proportion of local spin triplet \n on the metal",fontsize=34)
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


ax_idx = 0
# x: Proj_SM; y: variations of Q
x = all_proj_SPIN[:,:,2] #we want ProjSM
y = lambd_var


# Store the label of shown states
final_label_list = []
lines = []
for i in states:
    if i == 8:
        line, =ax.semilogx(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                 linewidth=size_linewidth,markersize=7,
                 markevery=[j for j in range(len(y)) if j%5 ==0])
        final_label_list.append(label_list[i])
    elif i == 15:
        line, =ax.semilogx(y,x[i],'x', color=colors[i],markersize=10,
                    markevery=[j for j in range(len(y)) if j%5 ==0])
        final_label_list.append(label_list[i])
    else:
        line, =ax.semilogx(y,x[i],linestyle=linestyles[i], color=colors[i],
                 linewidth=size_linewidth)
        final_label_list.append(label_list[i])
    lines += [line]
# Legend
fig.legend(handles=lines,labels=final_label_list, bbox_to_anchor=(0.9,0.5),loc='center left')

ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xticks([0.01,0.1,1,10,100])
name_file = "Q_"+str(np.round(Q,7))+"K1_"+str(K_M0L2)+"_Proj_SM_var_lambda.pdf"
plt.savefig(name_file, bbox_inches='tight')
# Show Figure in Spyder
plt.show()




#%% NOT in PAPER: E = f(lambda) - Logscale 
from matplotlib.ticker import ScalarFormatter

# states = [5,6,7,
#           8,
#           12,13,14]
states = [1,2,3,5,6,7,8,12,13,14]
# states = [i for i in range(16)]

ytick_step = 0.1
size_linewidth = 3
# Size of Figure
fig = plt.figure(figsize=(16,10))
gs = fig.add_gridspec(nrows=1, ncols=1)
ax = gs.subplots()


# Show gridlines
ax.grid(True)
# Save Figure
ax.set_ylim([-10, 5])
ax.set_ylim([-30, 20])
# Title of Figure
# Text box inside the graph
textstr = "$Q = "+str(np.round(Q,7))+"$, $K_1 = "+str(np.round(K_M0L2,6))+"$"
ax.text(0.76,0.95, textstr,transform=ax.transAxes, fontsize=24,
        verticalalignment='top',bbox=dict(boxstyle='round',facecolor='white'))


# Labels of x and y axis
ax.set_xlabel("$\\lambda$",fontsize=38)
ax.set_ylabel("E ($K_M$ units)",fontsize=34)
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


# x: Proj_SM; y: variations of Q
x = all_energies #all_proj_SPIN[:,:,1] #we want S(S+1)
y = lambd_var


# Store the label of shown states
final_label_list = []
lines = []
for i in states:
    if i == 8:
        line, =ax.semilogx(y,x[i],linestyle=linestyles[i], marker='o', color=colors[i],
                 linewidth=size_linewidth,markersize=7,
                 # markevery=[j for j in range(len(y)) if j%((len(y)-1)/100) ==1])
                 markevery=[j for j in range(len(y)) if j%5 ==0])
        final_label_list.append(label_list[i])
    elif i == 15:
        line, =ax.semilogx(y,x[i],'x', color=colors[i],markersize=10,
                    # markevery=[j for j in range(len(y)) if j%((len(y)-1)/100) ==1])
                    markevery=[j for j in range(len(y)) if j%5 ==0])
        final_label_list.append(label_list[i])
    else:
        line, =ax.semilogx(y,x[i],linestyle=linestyles[i], color=colors[i],
                 linewidth=size_linewidth)
        final_label_list.append(label_list[i])
    lines += [line]
# Legend
fig.legend(handles=lines,labels=final_label_list, bbox_to_anchor=(0.9,0.5),loc='center left')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlim([0.1, 100])
ax.set_xticks([0.01,0.1,1,10,100])
name_file = "Q_"+str(np.round(Q,7))+"K1_"+str(K_M0L2)+"_E_var_lambda"+str(100)+".pdf"
plt.savefig(name_file, bbox_inches='tight')
# Show Figure in Spyder
plt.show()