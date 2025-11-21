import numpy as np
import qiskit
from qiskit.quantum_info import state_fidelity
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py

import sys
sys.path.append("../../../src/brickwall_sparse")
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

J, h, g = (1, 0, 3)
Lx, Ly = (4, 4)
L = Lx*Ly
t = 0.125
# construct Hamiltonian
latt = qib.lattice.TriangularLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()

perms_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]]
perms_2 = [[0, 5, 10, 15, 3, 4, 9, 14, 2, 7, 8, 13, 1, 6, 11, 12], [5, 10, 15, 0, 4, 9, 14, 3, 7, 8, 13, 2, 6, 11, 12, 1]]
perms_3 = [[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], [4, 8, 12, 0, 5, 9, 13, 1, 6, 10, 14, 2, 7, 11, 15, 3]]


import numpy as np
import scipy.linalg
import time
import rqcopt as oc

# Pauli and identity
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)

from quimb.tensor.tensor_arbgeom_tebd import LocalHamGen, TEBDGen, edge_coloring

import numpy as np
import quimb as qu
import quimb.tensor as qtn


def triangular_permutations(Lx, Ly):
    """
    Return three nearest-neighbour permutation pairs for a PBC triangular lattice
    on an Lx x Ly grid.

    Each perms_k is [src_list, tgt_list], where src_list[i] -> tgt_list[i].

    For Lx == Ly, this exactly reproduces your 4x4 perms_1, perms_2, perms_3
    (up to the values of Lx, Ly).
    """
    def idx(x, y):
        # row-major site indexing: (x, y) -> i
        return x * Ly + y

    # --- perms_1: vertical neighbours (x, y) -> (x, y+1) ---
    p1_src, p1_tgt = [], []
    for x in range(Lx):
        for y in range(Ly):
            p1_src.append(idx(x, y))
            p1_tgt.append(idx(x, (y + 1) % Ly))

    # --- perms_2: diagonal neighbours (x, y) -> (x+1, y+1) ---
    p2_src, p2_tgt = [], []
    if Lx == Ly:
        # match your 4x4 pattern exactly, generalized to L=Lx=Ly
        L = Lx
        # order offsets by 0, L-1, L-2, ..., 1
        offsets = [0] + list(reversed(range(1, L)))
        for off in offsets:
            for x in range(L):
                y = (x + off) % L
                p2_src.append(idx(x, y))
                p2_tgt.append(idx((x + 1) % L, (y + 1) % L))
    else:
        # simple, consistent ordering for rectangular case
        for x in range(Lx):
            for y in range(Ly):
                p2_src.append(idx(x, y))
                p2_tgt.append(idx((x + 1) % Lx, (y + 1) % Ly))

    # --- perms_3: horizontal neighbours (x, y) -> (x+1, y) ---
    # loop over y then x so that for Lx == Ly we reproduce your 4x4 perms_3.
    p3_src, p3_tgt = [], []
    for y in range(Ly):
        for x in range(Lx):
            p3_src.append(idx(x, y))
            p3_tgt.append(idx((x + 1) % Lx, y))

    perms_1 = [p1_src, p1_tgt]
    perms_2 = [p2_src, p2_tgt]
    perms_3 = [p3_src, p3_tgt]

    return perms_1, perms_2, perms_3



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
    # 1. get the three directional permutations
    perms_1, perms_2, perms_3 = triangular_permutations(Lx, Ly)

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


def trotter(peps, t, L, Lx, Ly, J, g, perms, dag=False,
                      max_bond_dim=5, dt=0.1, trotter_order=2):
    """
    Perform a Trotter‐evolution on `peps` using an arbitrary‐geometry TEBD.
    `perms` is a list of lists of site‐pairs (i,j) specifying each layer’s two‐site interactions.
    """
    # Number of steps
    nsteps = abs(int(np.ceil(t / dt)))
    print("Trotter steps:", nsteps)
    dt = t / nsteps

    # Suzuki splitting
    if trotter_order > 1:
        sm = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2)))
        indices, coeffs = sm.indices, sm.coeffs
    else:
        indices, coeffs = [0, 1], [1, 1]

    
    hloc1 = g*(np.kron(X, I2)+np.kron(I2, X))/4
    hloc2 = J*np.kron(Z, Z)
    hlocs = (hloc1, hloc2)
    Vlist_start = []
    for i, c in zip(indices, coeffs):
        Vlist_start.append(-1j*c*dt*hlocs[i])

    for n in range(nsteps):
        for layer, V in enumerate(Vlist_start):
            i = n*len(Vlist_start)+layer
            for perm in perms:
                ordering = {(perm[2*j], perm[2*j+1]): V for j in range(L//2)}
                start = time.time()
                t = TEBDGen(peps, ham=LocalHam2D(Lx, Ly, ordering, cyclic=True),
                    tau=-1, D=max_bond_dim, 
                            #chi=1
                           )
                t.sweep(tau=-1)
                peps = t.state

    return peps


def layer_from_flat_perm(perm_row, L):
    """perm_row is a flat list of length L."""
    return [(perm_row[2*j], perm_row[2*j+1]) for j in range(L // 2)]
layers_raw = [
    perms_1[0], perms_1[1],
    perms_2[0], perms_2[1],
    perms_3[0], perms_3[1],
]
perms_for_trotter = [layer_from_flat_perm(row, L) for row in layers_raw]


from qiskit.quantum_info import state_fidelity
import quimb.tensor as qtn
import quimb
from quimb.tensor.tensor_2d_tebd import TEBD2D, LocalHam2D

bond_dim = 1
phys_dim = 2
peps, (p1, p2, p3) = build_triangular_PEPS(Lx, Ly, bond_dim, phys_dim)

peps = peps / peps.norm()
peps_E = peps.copy()
peps_T = peps.copy()
peps_C = peps.copy()

map_ = {i: (i//Ly, i%Lx) for i in range(L)}
BD = 3
nsteps = 2
peps_E = trotter(peps_E.copy(), t, L, Lx, Ly, J, g, perms_1+perms_2+perms_3,
                     dt=t/nsteps, max_bond_dim=BD, trotter_order=2)


sv1 =  peps_E.to_dense()
sv1 /= np.linalg.norm(sv1)
sv2 = peps.to_dense()
sv2 /= np.linalg.norm(sv2)

with open(f"PEPS_log.txt", "a") as file:
    file.write("Fidelity for Trotter 2: "+str(np.abs(state_fidelity(
        sv1, scipy.sparse.linalg.expm_multiply(-1j * t * hamil, sv2) ))))








