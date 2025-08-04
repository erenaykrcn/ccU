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

gs = np.linspace(1.8, 3.5, 50)
for g in gs:
	hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()
	eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamil, k=400)
	idx = eigenvalues.argsort()
	eigenvalues_sort = eigenvalues[idx]
	eigenvectors_sort = eigenvectors[:,idx]

	with open(f"calc_EVs_4x4.txt", "a") as file:
	    file.write(f"\n {eigenvalues_sort[0]} \n")