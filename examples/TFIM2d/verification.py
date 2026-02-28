import numpy as np
import qiskit
from qiskit.quantum_info import state_fidelity
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py
import scipy.linalg
import time
import rqcopt as oc
from quimb.tensor.tensor_arbgeom_tebd import LocalHamGen, TEBDGen, edge_coloring

import sys
sys.path.append("../../src/brickwall_sparse")
from utils_sparse import construct_ising_local_term, reduce_list, X, I2, get_perms
from ansatz_sparse import ansatz_sparse
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm
from itertools import product
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import time
import tracemalloc
tracemalloc.start()


perms_1, perms_2 = (
    [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6, 13, 14, 15, 16, 17, 12, 19, 20, 21, 22, 23, 18, 25, 26, 27, 28, 29, 24, 31, 32, 33, 34, 35, 30]],
    [[0, 6, 12, 18, 24, 30, 1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29, 35], 
    [6, 12, 18, 24, 30, 0, 7, 13, 19, 25, 31, 1, 8, 14, 20, 26, 32, 2, 9, 15, 21, 27, 33, 3, 10, 16, 22, 28, 34, 4, 11, 17, 23, 29, 35, 5]]
)

chi_overlap = 256
cutoff = 1e-12
BD, chi_overlap1, chi_overlap2, chi_overlap_incr  = (2, 10, 11, 3)
trotter_order, trotter_step = (1, 2)
trotter_order_ref, trotter_step_ref = (2, 2)
nsteps = 2
t = 0.125


J, g = (1, 3)
Lx, Ly = (6, 6)
L = Lx*Ly

# Pauli and identity
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)


import quimb.tensor as qtn
from qiskit.quantum_info import state_fidelity
import quimb.tensor as qtn
import quimb
from quimb.tensor.tensor_2d_tebd import TEBD2D, LocalHam2D
import gc



def trotter(peps, t, L, Lx, Ly, J, g, perms_v, perms_h, dag=False, max_bond_dim=5, 
            dt=0.1, trotter_order=2):
    nsteps = np.abs(int(np.ceil(t/dt)))
    print("Trotter steps ", nsteps)
    dt = t/nsteps
    if trotter_order > 1:
        indices = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2))).indices
        coeffs = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2))).coeffs
    else:
        indices = [0, 1]
        coeffs = [1, 1]
    
    hloc1 = g*(np.kron(X, I2)+np.kron(I2, X))/4
    hloc2 = J*np.kron(Z, Z)
    hlocs = (hloc1, hloc2)
    Vlist_start = []
    for i, c in zip(indices, coeffs):
        Vlist_start.append(-1j*c*dt*hlocs[i])

    for n in range(nsteps):
        for layer, V in enumerate(Vlist_start):
            i = n*len(Vlist_start)+layer
            for perm in perms_h:
                ordering = {(map_[perm[2*j]], map_[perm[2*j+1]]): V for j in range(L//2)}
                start = time.time()
                t = TEBD2D(peps, ham=LocalHam2D(Lx, Ly, ordering, cyclic=True),
                    tau=-1, D=max_bond_dim, chi=1)
                t.sweep(tau=-1)
                peps = t.state
                
            for perm in perms_v:
                ordering = {(map_[perm[2*j]], map_[perm[2*j+1]]): V for j in range(L//2)}
                start = time.time()
                t = TEBD2D(peps, ham=LocalHam2D(Lx, Ly, ordering, cyclic=True),
                    tau=-1, D=max_bond_dim, chi=1)
                t.sweep(tau=-1)
                peps = t.state
    return peps


def ccU(peps, Vlist, perms_extended, control_layers, dagger=False, max_bond_dim=10, lower_max_bond_dim=4, treshold=10):
    for i, V in enumerate(Vlist):
        if dagger or i not in control_layers:
            perms = perms_extended[i]
            for perm in perms:
                ordering = {(map_[perm[2*j]], map_[perm[2*j+1]]): scipy.linalg.logm(V) for j in range(L//2)}
                start = time.time()
                t = TEBD2D(peps, ham=LocalHam2D(Lx, Ly, ordering, cyclic=True),
                    tau=-1, D=max_bond_dim, chi=1)
                t.sweep(tau=-1)
                peps = t.state
    return peps


perms_extended = [[perms_1[0]]] + [perms_1]*3 + [[perms_1[0]], [perms_2[0]]] +\
                    [perms_2]*3 + [[perms_2[0]]]
perms_ext_reduced = [perms_1]*3  + [perms_2]*3
control_layers = [0, 4, 5, 9]           # 4 control layers


with h5py.File(f'./results/tfim2d_ccU_SPARSE_103_Lx4Ly4_t0.125_layers10_niter10_rS1_2hloc.hdf5', 'r') as f:
    Vlist  =  f["Vlist"][:]
Vlist_reduced = []
for i in range(len(Vlist)):
    if i not in control_layers:
        Vlist_reduced.append(Vlist[i])


map_ = {i: (i//Ly, i%Lx) for i in range(L)}
peps = qtn.PEPS.rand(Lx, Ly, bond_dim=1, phys_dim=2, cyclic=True)

ov_tn = peps.make_overlap(
    peps,
    layer_tags=("KET", "BRA"),
)
overlap_approx = ov_tn.contract_compressed(
    optimize="auto-hq",
    max_bond=chi_overlap,
    cutoff=cutoff,
)
norm = np.sqrt(np.abs(overlap_approx))
peps = peps/norm

peps_E = peps.copy()
peps_T = peps.copy()
peps_C = peps.copy()
peps_E = trotter(peps_E.copy(), t, L, Lx, Ly, J, g, perms_1, perms_2,
                     dt=t/trotter_step_ref, max_bond_dim=BD, trotter_order=trotter_order_ref)
peps_aE = ccU(peps_C.copy(), Vlist_reduced, perms_ext_reduced, [], dagger=False,
                 max_bond_dim=BD)
peps_T = trotter(peps_T.copy(), t, L, Lx, Ly, J, g, perms_1, perms_2,
                     dt=t/trotter_step, max_bond_dim=BD, trotter_order=trotter_order)
peps_T.compress_all(max_bond=BD)
peps_E.compress_all(max_bond=BD)
peps_aE.compress_all(max_bond=BD)


ov_tn = peps_E.make_overlap(
    peps_aE,
    layer_tags=("KET", "BRA"),
)
for chi_overlap in range(chi_overlap1, chi_overlap2, chi_overlap_incr):
    overlap_approx = ov_tn.contract_compressed(
        optimize="auto-hq",
        max_bond=chi_overlap,
        cutoff=cutoff,
    )
    with open(f"{L}_PEPS_log.txt", "a") as file:
        file.write("\n Fidelity for TICC: "+str(np.abs(overlap_approx)) + f", BD={BD}, chi_overlap={chi_overlap} \n")

ov_tn = peps_E.make_overlap(
    peps_T,
    layer_tags=("KET", "BRA"),
)
for chi_overlap in range(chi_overlap1, chi_overlap2, chi_overlap_incr):
    overlap_approx = ov_tn.contract_compressed(
        optimize="auto-hq",
        max_bond=chi_overlap,
        cutoff=cutoff,
    )
    with open(f"{L}_PEPS_log.txt", "a") as file:
        file.write(f"Fidelity for Trotter {trotter_order}: "+str(np.abs(overlap_approx)) + f", BD={BD}, chi_overlap={chi_overlap} \n")


