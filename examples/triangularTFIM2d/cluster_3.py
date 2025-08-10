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
from scipy.linalg import expm

Lx, Ly = (4, 4)
L = Lx*Ly
t = 0.25
# construct Hamiltonian
latt = qib.lattice.TriangularLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J, h, g = (1, 0, 3)
hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamil, k=10)
idx = eigenvalues.argsort()
eigenvalues_sort = eigenvalues[idx]
eigenvectors_sort = eigenvectors[:,idx]
ground_state = eigenvectors_sort[:, 0]

hloc1 = construct_ising_local_term(J, 0, 0, ndim=2)
hloc2 = g*(np.kron(X, I2)+np.kron(I2, X))/6
hloc = hloc1 + hloc2
perms_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]]
perms_2 = [[0, 5, 10, 15, 3, 4, 9, 14, 2, 7, 8, 13, 1, 6, 11, 12], [5, 10, 15, 0, 4, 9, 14, 3, 7, 8, 13, 2, 6, 11, 12, 1]]
perms_3 = [[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], [4, 8, 12, 0, 5, 9, 13, 1, 6, 10, 14, 2, 7, 11, 15, 3]]

Vlist_3 = [expm(-1j*t*hloc1/2) , expm(-1j*t*hloc2), expm(-1j*t*hloc1/2)]
Vlist_2 = [expm(-1j*t*hloc1/4), expm(-1j*t*hloc2/2), expm(-1j*t*hloc1/4)]
Vlist_1 = [expm(-1j*t*hloc1/4), expm(-1j*t*hloc2/2), expm(-1j*t*hloc1/4)]
Vlist_start    = Vlist_1 + Vlist_2 + Vlist_3 + Vlist_2 + Vlist_1
perms_extended = [perms_1]*3 + [perms_2]*3 + [perms_3]*3 + [perms_2]*3 + [perms_1]*3

state = np.array(random_statevector(2**L).data)
print("Trotter error: ", np.linalg.norm(ansatz_sparse(Vlist_start, L, perms_extended, state) - expm_multiply(
    -1j * t * hamil, state), ord=2))

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
I2 = np.array([[1, 0], [0, 1]])

V1 = scipy.linalg.expm(-1j*t*hloc1/2)
V2 = scipy.linalg.expm(-1j*t*hloc2)
V = scipy.linalg.expm(-1j*t*hloc)
YZ = np.kron(Y, Z)

Vlist_start =  [YZ, V, YZ, YZ, V, YZ, YZ, V, YZ]
Vlist_reduced = [V, V, V]

perms_extended = [[perms_1[0]]] + [perms_1] + [[perms_1[0]], [perms_2[0]]] +\
                    [perms_2] + [[perms_2[0]], [perms_3[0]]] + [perms_3] + [[perms_3[0]]] 
perms_ext_reduced = [perms_1]  + [perms_2] + [perms_3]
control_layers = [0, 2, 3, 5, 6, 8]

# 12 layers with 6 being controlled, 9 parameters in total.
state = random_statevector(2**L).data
print("Trotter error of the starting point: ", (np.linalg.norm(ansatz_sparse(Vlist_start, L, perms_extended, state) - expm_multiply(
    1j * t * hamil, state), ord=2) + np.linalg.norm(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state) - expm_multiply(
    -1j * t * hamil, state), ord=2))/2)

print('fidelity: ', (state_fidelity(ansatz_sparse(Vlist_start, L, perms_extended, state), expm_multiply(
    1j * t * hamil, state)) + state_fidelity(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state), expm_multiply(
    -1j * t * hamil, state)))/2)

from optimize_sparse import optimize, err

Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_start, perms_extended, perms_reduced=perms_ext_reduced, 
                                   control_layers=control_layers, rS=1, niter=5)
plt.plot(err_iter)
plt.yscale('log')
print(err_iter[-1])


with h5py.File(f"./triangularTFIM_ccU_SPARSE_{J}{h}{g}_Lx{Lx}Ly{Ly}_t{t}_layers{len(Vlist_start)}_niter5_rS1_2hloc.hdf5", "w") as f:
    f.create_dataset("Vlist", data=Vlist)
    f.create_dataset("f_iter", data=f_iter)
    f.create_dataset("err_iter", data=err_iter)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)