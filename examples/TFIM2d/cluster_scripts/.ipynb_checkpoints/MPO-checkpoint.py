import os
import numpy as np
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm

import sys
sys.path.append("../../../src/brickwall_sparse")
from utils_sparse import get_perms
sys.path.append("../../../src/MPS")
from utils_MPS import *
from MPS import trotter, ccU

Lx, Ly = (4, 4)
L= Lx*Ly

# construct Hamiltonian
latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J, h, g = (1, 0, 3)
t = 0.125
hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamil, k=10)
idx = eigenvalues.argsort()
eigenvalues_sort = eigenvalues[idx]
eigenvectors_sort = eigenvectors[:,idx]
ground_state = eigenvectors_sort[:, 0]


perms_v, perms_h = ([[0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15],
  [4, 8, 5, 9, 6, 10, 7, 11, 12, 0, 13, 1, 14, 2, 15, 3]],
 [[0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15],
  [1, 2, 5, 6, 9, 10, 13, 14, 3, 0, 7, 4, 11, 8, 15, 12]])
perms_extended = [[perms_v[0]]] + [perms_v]*3 + [[perms_v[0]], [perms_h[0]]] +\
                    [perms_h]*3 + [[perms_h[0]]]
perms_ext_reduced = [perms_v]*3  + [perms_h]*3
control_layers = [0, 4, 5, 9] 			# 4 control layers


Vlist = []
with h5py.File(f"../results/tfim2d_ccU_SPARSE_103_Lx4Ly4_t0.125_layers10_niter10_rS1_2hloc.hdf5", "r") as f:
    Vlist =  f["Vlist"][:]
    

perms_qc = [[0, 1], [0, 2]]
Xlists_opt = {}
for i in control_layers:
    with h5py.File(f"../results/tfim2d_ccU_SPARSE_{J}{h}{g}_Lx{Lx}Ly{Ly}_t{t}_layers{len(Vlist)}_niter15_rS1_DECOMPOSE_n{len(perms_qc)}_layer{i}.hdf5", "r") as file:
        Xlists_opt[i] = file[f"Xlist_{i}"][:]


ccU_BD = 12
initial_mps = [np.array([[[1/np.sqrt(2)]], [[1/np.sqrt(2)]]]) for i in range(L)]
A0 = np.zeros((2, 1, 1), dtype=np.complex128)
A0[0, :, :] = 1
initial_mps_backwards = [A0]+initial_mps

mps_ccU_backwards_MPO = ccU(initial_mps_backwards.copy(), L, Vlist, Xlists_opt, perms_extended, perms_qc,
                             control_layers, max_bond_dim=ccU_BD, swap=False)

mps_ccU_backwards_SWAP = ccU(initial_mps_backwards.copy(), L, Vlist, Xlists_opt, perms_extended, perms_qc, 
                             control_layers, max_bond_dim=ccU_BD, swap=True)

fid = mps_fidelity(mps_ccU_backwards_SWAP, mps_ccU_backwards_MPO)
with open(f"MPO_log.txt", "a") as file:
	file.write(f"Fidelity {fid}")

