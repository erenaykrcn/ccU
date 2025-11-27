import h5py
import qib
import numpy as np
from tenpy.models.lattice import Kagome
from tenpy.networks.site import SpinHalfSite


perms_1 = [[0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23], [1, 2, 3, 4, 5, 0, 10, 11, 12, 13, 14, 9, 19, 20, 21, 22, 23, 18]]
perms_2 = [[0, 6, 9, 15, 18, 24, 2, 7, 11, 16, 20, 25, 4, 8, 13, 17, 22, 26], [6, 9, 15, 18, 24, 0, 7, 11, 16, 20, 25, 2, 8, 13, 17, 22, 26, 4]]
perms_3 = [[0, 1, 3, 7, 10, 15, 5, 8, 12, 16, 19, 24, 14, 17, 21, 25, 23, 26], [1, 0, 7, 10, 15, 3, 8, 12, 16, 19, 24, 5, 17, 21, 25, 14, 26, 23]]

BD = 7
chi_overlap = 12
result_string = f"kagome_Heis_L12_t0.125_layers3.hdf5"
perms_extended = [perms_1]  + [perms_2] + [perms_3]
ref_trotter_order = 4
    
hloc1 = J[0]*np.kron(X, X)
hloc2 = J[1]*np.kron(Y, Y)
hloc3 = J[2]*np.kron(Z, Z)
hlocs = (hloc1, hloc2, hloc3)
#hloc1 = J*np.kron(Z, Z)
#hloc2 = g*(np.kron(X, I2)+np.kron(I2, X))/4
#hlocs = (hloc1, hloc2)

L = 27
lat = Kagome(3, 3, [SpinHalfSite() for _ in range(3)], bc='periodic')

N = lat.N_sites
A = np.zeros((N, N), dtype=int)

J = (1, 1, 1)
h = (0, 0, 0)
"""perms_1 = [[0, 8, 12, 20, 24, 32, 36, 44, 
            2, 9, 14, 21, 26, 33, 38, 45,
           4, 10, 16, 22, 28, 34, 40, 46,
           6, 11, 18, 23, 30, 35, 42, 47], 
           [8, 12, 20, 24, 32, 36, 44, 0, 
            9, 14, 21, 26, 33, 38, 45, 2, 
            10, 16, 22, 28, 34, 40, 46, 4,
            11, 18, 23, 30, 35, 42, 47, 6
           ]]
perms_2 = [[0, 1, 2, 3, 4, 5, 6, 7,
           12, 13, 14, 15, 16, 17, 18, 19,
          24, 25, 26, 27, 28, 29, 30, 31,
          36, 37, 38, 39, 40, 41, 42, 43], 
            [1, 2, 3, 4, 5, 6, 7, 0,
             13, 14, 15, 16, 17, 18, 19, 12,
             25, 26, 27, 28, 29, 30, 31, 24,
             37, 38, 39, 40, 41, 42, 43, 36
            ]]
perms_3 = [[1, 8, 3, 9, 13, 20, 5, 10, 15, 21, 25, 32, 
            7, 11, 17, 22, 27, 33, 37, 44, 
            19, 23, 29, 34, 39, 45, 31, 35, 41, 46, 43, 47], 
           [8, 1, 9, 13, 20, 3, 10, 15, 21, 25, 32, 5, 11, 17, 22, 27, 
            33, 37, 44, 7, 23, 29, 34, 39, 45, 19, 35, 41, 46, 31, 47, 43]]
"""

for perm in perms_1+perms_2+perms_3:
    for i in range(len(perm)//2):
        A[perm[2*i], perm[2*i+1]] = 1
        A[perm[2*i+1], perm[2*i]] = 1


import numpy as np
import scipy.linalg
import time
import rqcopt as oc
from quimb.tensor.tensor_arbgeom_tebd import LocalHamGen, TEBDGen, edge_coloring

# Pauli and identity
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)


def layer_from_flat_perm(perm_row, L):
    """perm_row is a flat list of length L."""
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


def build_triangular_PEPS(bond_dim, phys_dim=2,
                          seed=None, dtype="complex128"):
    edges = _edges_from_permutations(perms_1, perms_2, perms_3)
    tn = qtn.TN_from_edges_rand(
        edges,
        D=bond_dim,
        phys_dim=phys_dim,
        seed=seed,
        dtype=dtype,
    )

    return tn, (perms_1, perms_2, perms_3)

import gc

import gc

def trotter(peps, t, L, J, perms, dag=False,
                      max_bond_dim=5, dt=0.1, trotter_order=2):
    # Number of steps
    nsteps = abs(int(np.ceil(t / dt)))
    dt = t / nsteps

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

import h5py

Vlists = {}
for t in [0.125]:
    with h5py.File(f'../results/{result_string}') as f:
        Vlists[t]  =  f["Vlist"][:]
Vlist = Vlists[0.125]


chi_overlap = 20
peps, (p1, p2, p3) = build_triangular_PEPS(1, 2)
peps_copy_norm = peps.copy()
ov_tn = peps_copy_norm.make_overlap(
    peps_copy_norm,
    layer_tags=("KET", "BRA"),
)
overlap_approx = ov_tn.contract_compressed(
    optimize="auto-hq",
    max_bond=chi_overlap,
    cutoff=1e-10,
)
norm = np.sqrt(abs(overlap_approx))
peps = peps/np.abs(norm)

from qiskit.quantum_info import state_fidelity
import quimb.tensor as qtn
import quimb
from quimb.tensor.tensor_2d_tebd import TEBD2D, LocalHam2D


peps_E = peps.copy()
peps_T = peps.copy()
peps_C = peps.copy()


nsteps = 1
peps_E = trotter(peps_E.copy(), t, L,  J, perms_1+perms_2+perms_3,
                     dt=t/nsteps, max_bond_dim=BD, trotter_order=ref_trotter_order)
peps_aE = ccU(peps_C.copy(), Vlist, perms_extended, [], dagger=False,
                 max_bond_dim=BD)
#peps_T = trotter(peps_T.copy(), t, L,  J, perms_1+perms_2+perms_3,
#                     dt=t/nsteps, max_bond_dim=BD, trotter_order=1)

peps_T.compress_all(max_bond=BD)
peps_E.compress_all(max_bond=BD)
peps_aE.compress_all(max_bond=BD)

ov_tn = peps_E.make_overlap(
    peps_aE,
    layer_tags=("KET", "BRA"),
)

overlap_approx = ov_tn.contract_compressed(
    optimize="hyper-compressed",
    max_bond=chi_overlap,
    cutoff=1e-10,
)

with open(f"PEPS_log.txt", "a") as file:
    file.write("Fidelity for ccU, Heisenberg: "+str(np.abs(overlap_approx))+"\n")