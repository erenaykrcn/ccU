import numpy as np
import qiskit
from qiskit.quantum_info import state_fidelity

from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py

import sys
sys.path.append("../../src/brickwall_sparse")
from utils_sparse import construct_ising_local_term, reduce_list, X, I2, get_perms
from ansatz_sparse import ansatz_sparse
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector

Lx, Ly = (4, 4)
L = Lx*Ly
latt = qib.lattice.TriangularLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J, h = (1, 0)

perms_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]]
perms_2 = [[0, 5, 10, 15, 3, 4, 9, 14, 2, 7, 8, 13, 1, 6, 11, 12], [5, 10, 15, 0, 4, 9, 14, 3, 7, 8, 13, 2, 6, 11, 12, 1]]
perms_3 = [[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], [4, 8, 12, 0, 5, 9, 13, 1, 6, 10, 14, 2, 7, 11, 15, 3]]


from itertools import product
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np


def eval_energy(config):
    e = 0
    for perm in perms_1+perms_2+perms_3:
        for j in range(len(perm)//2):
            e += 1 if config[perm[2*j]]==config[perm[2*j+1]] else -1
    return e


e = 48
gs = '0'*L
gss = []
bitstrings = [''.join(bits) for bits in product('01', repeat=16)]
for bitstring in bitstrings:
    if eval_energy(bitstring) < e:
        gs = bitstring
        e = eval_energy(bitstring)
    if eval_energy(bitstring) == -16:
        gss.append(bitstring)

ground_states = gss
n = len(ground_states[0])
dim = 2**n
state = np.zeros(dim, dtype=complex)
for s in ground_states:
    index = int(s, 2)
    state[index] = 1 / np.sqrt(len(ground_states))
psi = Statevector(state)


gs = np.linspace(0, 1.8, 50)
for g in gs:
    hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamil, k=5, v0=psi.data, which='SA')
    idx = eigenvalues.argsort()
    eigenvalues_sort = eigenvalues[idx]

    with open(f"calc_EVs_SA.txt", "a") as file:
        file.write(f"\n {eigenvalues_sort[0]}, \n")

