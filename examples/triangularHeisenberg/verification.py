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


chi_overlap = 25
cutoff = 1e-12
BD, chi_overlap1, chi_overlap2, chi_overlap_incr  = (2, 5, 6, 3)
trotter_order, trotter_step = (1, 3)
trotter_order_ref, trotter_step_ref = (2, 6)
nsteps = 2

niter = 40
t = 0.1
layers = 36

J = (1, 1, 1)
h = (3, -1, 1)
Lx, Ly = (6, 6)
L = Lx*Ly
perms_1 = [[i for i in range(36)], 
           [1, 2, 3, 4, 5, 0,
            7, 8, 9, 10, 11, 6,
           13, 14, 15, 16, 17, 12, 
           19, 20, 21, 22, 23, 18,
           25, 26, 27, 28, 29, 24,
           31, 32, 33, 34, 35, 30]]
perms_2 = [[0, 7, 14, 21, 28, 35, 
            1, 8, 15, 22, 29, 30, 
            2, 9, 16, 23, 24, 31, 
            3, 10, 17, 18, 25, 32, 
            4, 11, 12, 19, 26, 33, 
            5, 6, 13, 20, 27, 34], 
           [7, 14, 21, 28, 35, 0,
           8, 15, 22, 29, 30, 1,
           9, 16, 23, 24, 31, 2,
           10, 17, 18, 25, 32, 3,
           11, 12, 19, 26, 33, 4, 
           6, 13, 20, 27, 34, 5]]
perms_3 = [[0, 6, 12, 18, 24, 30,
            1, 7, 13, 19, 25, 31, 
            2, 8, 14, 20, 26, 32,
            3, 9, 15, 21, 27, 33, 
            4, 10, 16, 22, 28, 34, 
            5, 11, 17, 23, 29, 35], 
           [6, 12, 18, 24, 30, 0,
           7, 13, 19, 25, 31, 1,
           8, 14, 20, 26, 32, 2,
           9, 15, 21, 27, 33, 3, 
           10, 16, 22, 28, 34, 4, 
           11, 17, 23, 29, 35, 5]]


# Pauli and identity
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)


def layer_from_flat_perm(perm_row, L):
    return [(perm_row[2*j], perm_row[2*j+1]) for j in range(len(perm_row) // 2)]
layers_raw = [
    perms_1[0], perms_1[1],
    perms_2[0], perms_2[1],
    perms_3[0], perms_3[1],
]
perms_for_trotter = [layer_from_flat_perm(row, L) for row in layers_raw]


import numpy as np
import quimb as qu
import quimb.tensor as qtn


def _edges_from_permutations(perms_1, perms_2, perms_3):
    """
    Take the three [src, tgt] permutation pairs and return a sorted list of
    unique undirected edges (i, j) with i < j.
    """
    edge_set = set()

    for perms in (perms_1, perms_2, perms_3):
        src, tgt = perms
        for a, b in zip(src, tgt):
            if a == b:
                continue
            i, j = sorted((a, b))
            edge_set.add((i, j))

    return sorted(edge_set)


def build_triangular_PEPS(Lx, Ly, bond_dim, phys_dim=2,
                          seed=None, dtype="complex128"):
    """
    Build a random 'PEPS-like' tensor network on a PBC triangular lattice
    using quimb's TN_from_edges_rand.

    Returns:
        tn : TensorNetwork
            A generic quimb tensor network with
            - one tensor per site (labelled by integer 0..Lx*Ly-1)
            - each internal edge with bond dimension 'bond_dim'
            - one physical leg of dimension 'phys_dim' per site.
        perms : (perms_1, perms_2, perms_3)
            The three permutation pairs used to define the triangular NN bonds.
    """

    # 2. convert to a set of unique graph edges
    edges = _edges_from_permutations(perms_1, perms_2, perms_3)

    # 3. build a random TN with that graph geometry
    #    This is a documented quimb function:
    #    TN_from_edges_rand(edges, D, phys_dim=None, seed=None, dtype=..., ...)
    tn = qtn.TN_from_edges_rand(
        edges,
        D=bond_dim,
        phys_dim=phys_dim,
        seed=seed,
        dtype=dtype,
    )

    # At this point:
    # - each node is an integer site index (0..Lx*Ly-1)
    # - each tensor has one physical index + several virtual ones
    # - tags follow the default site_tag_id='I{}' convention.

    return tn, (perms_1, perms_2, perms_3)

import gc



def trotter(peps, t, L, J,  perms, dag=False,
                      max_bond_dim=5, dt=0.1, trotter_order=2, h=h):
    # Number of steps
    nsteps = abs(int(np.ceil(t / dt)))
    dt = t / nsteps

    hloc1 = J[0]*np.kron(X, X) + h[0]/6 * (np.kron(X, I2)+np.kron(I2, X))
    hloc2 = J[1]*np.kron(Y, Y) + h[1]/6 * (np.kron(Y, I2)+np.kron(I2, Y))
    hloc3 = J[2]*np.kron(Z, Z) + h[2]/6 * (np.kron(Z, I2)+np.kron(I2, Z))
    hlocs = (hloc1, hloc2, hloc3)

    # Suzuki splitting
    if trotter_order > 1:
        sm = oc.SplittingMethod.suzuki(len(hlocs), int(np.log(trotter_order)/np.log(2)))
        indices, coeffs = sm.indices, sm.coeffs
    else:
        indices, coeffs = range(len(hlocs)), [1]*len(hlocs)
        
    Vlist_start = []
    for i, c in zip(indices, coeffs):
        Vlist_start.append(-1j*c*dt*hlocs[i])
    

    for n in range(nsteps):
        for layer, V in enumerate(Vlist_start):
            i = n*len(Vlist_start)+layer
            for perm in perms:
                
                edges = [(perm[2*j], perm[2*j+1]) for j in range(len(perm)//2)]
                H2 = {edge: V for edge in edges}
                ham = LocalHamGen(H2=H2, H1=None)
                tebd = TEBDGen(peps, ham=ham, D=max_bond_dim)
                tebd.sweep(tau=-1)
                peps = tebd.state

                del tebd, ham
                gc.collect()
    return peps


def ccU(peps, Vlist, perms_extended, control_layers, dagger=False, max_bond_dim=10):
    for i, V in enumerate(Vlist):
        if dagger or i not in control_layers:
            perms = perms_extended[i]
            for perm in perms:
                edges = [(perm[2*j], perm[2*j+1]) for j in range(len(perm)//2)]
                H2 = {edge: scipy.linalg.logm(V) for edge in edges}
                ham = LocalHamGen(H2=H2, H1=None)
                tebd = TEBDGen(peps, ham=ham, D=max_bond_dim)
                tebd.sweep(tau=-1)
                peps = tebd.state

                del tebd, ham
                gc.collect()
    return peps


if layers==36:
    perms_extended = [[perms_1[0]]] + [[perms_1[0]]]+ [[perms_1[1]]] + [[perms_1[0]], [perms_2[0]]] +\
      [[perms_2[0]]]+ [[perms_2[1]]]  + [[perms_2[0]], [perms_3[0]]] + [[perms_3[0]]]+ [[perms_3[1]]]  + [[perms_3[0]]]
    perms_extended = perms_extended*3
    perms_ext_reduced =  [[perms_1[0]]]+ [[perms_1[1]]]  +  [[perms_2[0]]]+ [[perms_2[1]]]  +  [[perms_3[0]]]+ [[perms_3[1]]] 
    perms_ext_reduced = perms_ext_reduced*3
elif layers==72:
    perms_extended = [[perms_1[0]]] + [perms_1] + [[perms_1[0]], [perms_2[0]]] +\
          [perms_2] + [[perms_2[0]], [perms_3[0]]] +  [perms_3] + [[perms_3[0]]]
    perms_extended = perms_extended*5
    perms_ext_reduced = [perms_1] + [perms_2] + [perms_3]
    perms_ext_reduced = perms_ext_reduced*5

non_control_layers = []
k = 0
while True:
    a = 1 + 4*k
    b = 2 + 4*k
    if a > layers or b > layers:
        break
    non_control_layers.extend([a, b])
    k += 1
control_layers = []
for i in range(len(perms_extended)):
    if i not in non_control_layers:
        control_layers.append(i)


Vlists = {}
for t in [t]:
    with h5py.File(f'./results/tr_Heis_L16_t{t}_layers{layers}_niter{niter}.hdf5') as f:
        Vlists[t]  =  f["Vlist"][:]
Vlist = Vlists[t]
Vlist_reduced = []
for i, V in enumerate(Vlist):
    if i not in control_layers:
        Vlist_reduced.append(V)



from qiskit.quantum_info import state_fidelity
import quimb.tensor as qtn
import quimb
from quimb.tensor.tensor_2d_tebd import TEBD2D, LocalHam2D

bond_dim = 1
phys_dim = 2
peps, (p1, p2, p3) = build_triangular_PEPS(Lx, Ly, bond_dim, phys_dim)
ov_tn = peps.make_overlap(
    peps,
    layer_tags=("KET", "BRA"),
)
overlap_approx = ov_tn.contract_compressed(
    optimize="hyper-compressed",
    max_bond=chi_overlap,
    cutoff=1e-10,
)
norm = np.sqrt(abs(overlap_approx))
peps = peps/np.abs(norm)

peps_E = peps.copy()
peps_T = peps.copy()
peps_C = peps.copy()
peps_E = trotter(peps_E.copy(), t, L,  J, perms_1+perms_2+perms_3,
                     dt=t/trotter_step_ref, max_bond_dim=BD, trotter_order=trotter_order_ref)
peps_aE = ccU(peps_C.copy(), Vlist_reduced, perms_ext_reduced, [], dagger=False,
                 max_bond_dim=BD)
peps_T = trotter(peps_T.copy(), t, L,  J, perms_1+perms_2+perms_3,
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
        optimize="hyper-compressed",
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
        optimize="hyper-compressed",
        max_bond=chi_overlap,
        cutoff=cutoff,
    )
    with open(f"{L}_PEPS_log.txt", "a") as file:
        file.write(f"Fidelity for Trotter {trotter_order}: "+str(np.abs(overlap_approx)) + f", BD={BD}, chi_overlap={chi_overlap} \n")


