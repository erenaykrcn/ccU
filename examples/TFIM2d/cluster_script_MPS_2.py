import numpy as np
import qiskit
from qiskit.quantum_info import state_fidelity
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm

import sys
sys.path.append("../../src/brickwall_sparse")
from utils_sparse import get_perms
sys.path.append("../../src/MPS")
from utils_MPS import (random_mps, apply_localGate, apply_two_site_operator, 
						mps_to_state_vector, get_mps_of_sv, mps_fidelity)
from MPS import trotter, ccU

t = 0.25
dt    = 0.01 # Trotter step to be used for the 'quasi'-exact reference
order = 4  # Trotter order to be used for the 'quasi'-exact reference
initial_state_BD, exact_state_BD, ccU_BD = (2**2, 2**11, 2**10) # Bond dimensions


Lx, Ly = (6, 6)
L= Lx*Ly

# construct Hamiltonian
latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J, h, g = (1, 0, 3)
hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamil, k=10)
idx = eigenvalues.argsort()
eigenvalues_sort = eigenvalues[idx]
eigenvectors_sort = eigenvectors[:,idx]
ground_state = eigenvectors_sort[:, 0]

perms_v, perms_h = get_perms(Lx, Ly)

if L==36:
    perms_v, perms_h = (
    [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6, 13, 14, 15, 16, 17, 12, 19, 20, 21, 22, 23, 18, 25, 26, 27, 28, 29, 24, 31, 32, 33, 34, 35, 30]],
    [[0, 6, 12, 18, 24, 30, 1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29, 35], 
    [6, 12, 18, 24, 30, 0, 7, 13, 19, 25, 31, 1, 8, 14, 20, 26, 32, 2, 9, 15, 21, 27, 33, 3, 10, 16, 22, 28, 34, 4, 11, 17, 23, 29, 35, 5]]
    )


perms_extended = [[perms_v[0]]] + [perms_v]*3 + [[perms_v[0]], [perms_h[0]]] +\
                    [perms_h]*3 + [[perms_h[0]], [perms_v[0]]] + [perms_v]*3 + [[perms_v[0]]]
perms_ext_reduced = [perms_v]*3  + [perms_h]*3 + [perms_v]*3


Vlist = []
with h5py.File(f"./results/tfim2d_ccU_SPARSE_103_Lx4Ly4_t0.25_layers15_rS1_niter15_3hloc.hdf5", "r") as f:
    Vlist =  f["Vlist"][:]
    
control_layers = [0, 4, 5, 9, 10, 14]
perms_qc = [[0, 1], [0, 2], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2]]
Xlists_opt = {}
for i in control_layers:
    with h5py.File(f"./results/tfim2d_ccU_SPARSE_103_Lx4Ly4_t0.25_layers15_niter20_rS5_DECOMPOSE_n9_layer{i}.hdf5", "r") as file:
        Xlists_opt[i] = file[f"Xlist_{i}"][:]


import os
initial_mps = []
if os.path.isfile(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t0.25_INITIAL_MPS.h5"):
    with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t0.25_INITIAL_MPS.h5", "r") as f:
        mps_group = f["mps"]
        initial_mps = [mps_group[f"site_{i}"][()] for i in range(L)]
else:
	initial_mps = random_mps(L, max_bond_dim=initial_state_BD)
	with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t0.25_INITIAL_MPS.h5", "w") as f:
		mps_group = f.create_group("mps")
		for i, tensor in enumerate(initial_mps):
			mps_group.create_dataset(f"site_{i}", data=tensor)


exact_mps_forw_input = initial_mps.copy()
exact_mps_forwards = trotter(exact_mps_forw_input, t, L, Lx, Ly, J, g, max_bond_dim=exact_state_BD, trotter_order=order, dt=dt)
with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t0.25_TROTTER_MPS_FORWARDS_Order{order}_dt{dt}.h5", "w") as f:
    mps_group = f.create_group("mps")
    for i, tensor in enumerate(exact_mps_forwards):
        mps_group.create_dataset(f"site_{i}", data=tensor)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)
    f.attrs["order"] = order
    f.attrs["dt"] = dt
with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t0.25_TROTTER_MPS_FORWARDS_Order{order}_dt{dt}.h5", "r") as f:
    mps_group = f["mps"]
    exact_mps_forwards = [mps_group[f"site_{i}"][()] for i in range(L)]


A0 = np.zeros((2, 1, 1), dtype=np.complex128)
A0[1, :, :] = 1
initial_mps_forwards = [A0]+initial_mps
exact_mps_forwards_EXT = [A0]+exact_mps_forwards
mps_ccU_forwards = ccU(initial_mps_forwards, L, Vlist, Xlists_opt, perms_extended, perms_qc, control_layers, max_bond_dim=ccU_BD)
with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}_MPS_103_t0.25_ccU_MPS_FORWARDS.h5", "w") as f:
    mps_group = f.create_group("mps")
    for i, tensor in enumerate(mps_ccU_forwards):
        mps_group.create_dataset(f"site_{i}", data=tensor)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)
    f.attrs["order"] = order
    f.attrs["dt"] = dt
with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}_MPS_103_t0.25_ccU_MPS_FORWARDS.h5", "r") as f:
    mps_group = f["mps"]
    mps_ccU_forwards = [mps_group[f"site_{i}"][()] for i in range(L+1)]
print("ccU forwards fidelity: ", mps_fidelity(exact_mps_forwards_EXT, mps_ccU_forwards))
with open("eval_results.txt", "a") as file:
	file.write("\n ccU forwards fidelity: " + str(mps_fidelity(exact_mps_forwards_EXT, mps_ccU_forwards))+ "\n")


