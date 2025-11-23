import numpy as np
import qiskit
from qiskit.quantum_info import state_fidelity
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py

import sys
sys.path.append("../../../src/brickwall_PEPS")
from ansatz_PEPS import ansatz_PEPS
from optimize_PEPS import optimize_PEPS
sys.path.append("../../../src/brickwall_sparse")
from utils_sparse import construct_ising_local_term

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

from quimb.tensor.tensor_arbgeom_tebd import LocalHamGen, TEBDGen
import gc
import quimb as qu
import quimb.tensor as qtn

BD = 4
chi_overlap = 8
nsteps = 1
niter = 5
n_workers_1, n_workers_2 = (1, 1)


X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)

J, h, g = (1, 0, 3)
Lx, Ly = (3, 3)
L = Lx*Ly
t = 0.125

# construct Hamiltonian
latt = qib.lattice.TriangularLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()

perms_1 = [[0, 1, 3, 4, 6, 7], [1, 2, 4, 5, 7, 8], [2, 0, 5, 3, 8, 6]]
perms_2 = [[0, 3, 1, 4, 2, 5], [3, 6, 4, 7, 5, 8], [6, 0, 7, 1, 8, 2]]
perms_3 = [[0, 4, 1, 5, 2, 3], [4, 8, 5, 6, 3, 7], [8, 0, 6, 1, 7, 2]]
perms_extended = [perms_1]  + [perms_2] + [perms_3]
hloc1 = construct_ising_local_term(J, 0, 0, ndim=2)
hloc2 = g*(np.kron(X, I2)+np.kron(I2, X))/6
hloc = hloc1 + hloc2
V = scipy.linalg.expm(-1j*t*hloc)
Vlist_start = [V, V, V]


def compute_overlap(peps1, peps2, chi_overlap):
    ov_tn = peps1.make_overlap(
            peps2,
            layer_tags=("KET", "BRA"),
    )
    overlap_approx = ov_tn.contract_compressed(
            optimize="hyper-compressed",  # preset strategy name understood via cotengra
            max_bond=chi_overlap,
            cutoff=1e-10,
            # leave strip_exponent=False (default) so we just get a scalar back
    )
    return overlap_approx


def trotter(peps, t, L, Lx, Ly, J, g, perms, dag=False,
                      max_bond_dim=5, dt=0.1, trotter_order=2):
    # Number of steps
    import numpy as np
    nsteps = abs(int(np.ceil(t / dt)))
    dt = t / nsteps

    # Suzuki splitting
    if trotter_order > 1:
        sm = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2)))
        indices, coeffs = sm.indices, sm.coeffs
    else:
        indices, coeffs = [0, 1], [1, 1]
        #indices, coeffs = [0], [1]
    
    hloc1 = g*(np.kron(X, I2)+np.kron(I2, X))/6
    hloc2 = J*np.kron(Z, Z)
    
    hlocs = (hloc1, hloc2)
    #hlocs = (hloc1+hloc2, )
    Vlist_start = []
    for i, c in zip(indices, coeffs):
        Vlist_start.append(-1j*c*dt*hlocs[i])

    for n in range(nsteps):
        for layer, V in enumerate(Vlist_start):
            i = n*len(Vlist_start)+layer
            for perm in perms:
                #ordering = {(perm[2*j], perm[2*j+1]): V for j in range(L//2)}
                #start = time.time()
                
                edges = [(perm[2*j], perm[2*j+1]) for j in range(len(perm) // 2)]
                H2 = {edge: V for edge in edges}
                ham = LocalHamGen(H2=H2, H1=None)
                tebd = TEBDGen(peps, ham=ham, D=max_bond_dim)
                tebd.sweep(tau=-1)
                peps = tebd.state
                #peps /= np.sqrt(compute_overlap(peps, peps, chi_overlap))

                del tebd, ham
                gc.collect()
    return peps

    import numpy as np


def _edges_from_permutations(*perm_groups):
    """
    perm_groups: e.g. (perms_1, perms_2, perms_3),
    each perms_k is a list of layers, each layer is [i0, j0, i1, j1, ...].
    """
    edge_set = set()

    for perms in perm_groups:
        for layer in perms:
            if len(layer) % 2 != 0:
                raise ValueError(f"Layer length must be even, got {len(layer)}")
            for p in range(0, len(layer), 2):
                a = int(layer[p])
                b = int(layer[p + 1])
                if a == b:
                    continue
                i, j = sorted((a, b))
                edge_set.add((i, j))

    return sorted(edge_set)


def build_triangular_PEPS(Lx, Ly, bond_dim, phys_dim=2,
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

bond_dim, phys_dim = (1, 2)
peps, (p1, p2, p3) = build_triangular_PEPS(Lx, Ly, bond_dim, phys_dim)
peps_copy_norm = peps.copy()
ov_tn = peps_copy_norm.make_overlap(
    peps_copy_norm,
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
peps_E = trotter(peps_E.copy(), t, L, Lx, Ly, J, g, perms_1+perms_2+perms_3,
                     dt=t/nsteps, max_bond_dim=BD, trotter_order=2)
peps_E.compress_all(max_bond=BD)
peps_E /= np.sqrt(compute_overlap(peps_E, peps_E, chi_overlap))

reference_states = [peps_E]
initial_states = [peps]

Vlist, f_iter, err_iter = optimize_PEPS(L, reference_states, initial_states, t, 
	Vlist_start, perms_extended, BD, BD+2, log=True, niter=niter, n_workers_1=n_workers_1, n_workers_2=n_workers_2)


with h5py.File(f"./results/triangularTFIM_PEPS_{J}{h}{g}_Lx{Lx}Ly{Ly}_t{t}_layers{len(Vlist_start)}_niter{niter}.hdf5", "w") as f:
    f.create_dataset("Vlist", data=Vlist)
    f.create_dataset("f_iter", data=f_iter)
    f.create_dataset("err_iter", data=err_iter)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)

