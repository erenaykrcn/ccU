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
from optimize_sparse import optimize


layer = 13

L = 10
niter = 30
ts = [0.5, 0.6, 0.75, 0.9]
rS=10
Vlists = {}
for t in ts:
    latt = qib.lattice.IntegerLattice((L, ), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    J, h, g = (1, 0, 3)
    hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()
    hloc = construct_ising_local_term(J, 0, 0, ndim=2) + g*(np.kron(X, I2)+np.kron(I2, X))/2
    V = scipy.linalg.expm(-1j*t*hloc/layer)
    Vlist_reduced = [V]*layer
    perms = [[[i for i in range(L)]] if i%2==0 else [[i for i in range(1, L)]+[0]] for i in range(len(Vlist_reduced))]
    Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_reduced, perms, rS=rS, niter=niter, log=True)
    Vlists[t] = Vlist

with h5py.File(f".results/t{t}_layers{layer}.hdf5", "w") as f:
    for t in ts:
        f.create_dataset(f"Vlist{t}", data=Vlists[t])

