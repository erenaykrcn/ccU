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
from utils_sparse import construct_heisenberg_local_term, reduce_list, X, I2, get_perms
from ansatz_sparse import ansatz_sparse
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm

Lx, Ly = (4, 4)
L = Lx*Ly
t = 0.125
# construct Hamiltonian
latt = qib.lattice.TriangularLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J = (1, 1, 1)
h = (3, -1, 1)
hamil = qib.HeisenbergHamiltonian(field, J, h).as_matrix()
#eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamil, k=10)
#idx = eigenvalues.argsort()
#eigenvalues_sort = eigenvalues[idx]
#eigenvectors_sort = eigenvectors[:,idx]
#ground_state = eigenvectors_sort[:, 0]
perms_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]]
perms_2 = [[0, 5, 10, 15, 3, 4, 9, 14, 2, 7, 8, 13, 1, 6, 11, 12], [5, 10, 15, 0, 4, 9, 14, 3, 7, 8, 13, 2, 6, 11, 12, 1]]
perms_3 = [[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], [4, 8, 12, 0, 5, 9, 13, 1, 6, 10, 14, 2, 7, 11, 15, 3]]


hloc1 = construct_heisenberg_local_term((J[0], 0   ,    0), (0, h[1],       0), ndim=3)
hloc2 = construct_heisenberg_local_term((0   ,    J[1], 0), (0, 0, h[2]      ), ndim=3)
hloc3 = construct_heisenberg_local_term((0   , 0   , J[2]), (h[0], 0,       0), ndim=3)

V1 = scipy.linalg.expm(-1j*t*hloc1/2)
V2 = scipy.linalg.expm(-1j*t*hloc2/2)
V3 = scipy.linalg.expm(-1j*t*hloc3)

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
I2 = np.array([[1, 0], [0, 1]])
XZ = np.kron(X, Z)
XY = np.kron(X, Y)
ZY = np.kron(Z, Y)

Vlist_start = [XZ, V1, XZ]*3 + [XY, V2, XY]*3 + [ZY, V3, ZY]*3 + [XY, V2, XY]*3 + [XZ, V1, XZ]*3
Vlist_reduced = [V1]*3 + [V2]*3  + [V3]*3 + [V2]*3 + [V1]*3 

perms_extended = [[perms_1[0]]] + [perms_1] + [[perms_1[0]], [perms_2[0]]] +\
      [perms_2] + [[perms_2[0]], [perms_3[0]]] +  [perms_3] + [[perms_3[0]]]
perms_extended = perms_extended*5
perms_ext_reduced = [perms_1] + [perms_2] + [perms_3]
perms_ext_reduced = perms_ext_reduced*5
non_control_layers = [i for i in range(1, 45, 3)]
control_layers = []
for i in range(45):
    if i not in non_control_layers:
        control_layers.append(i)

state = random_statevector(2**L).data
print("Trotter error of the starting point: ", (np.linalg.norm(ansatz_sparse(Vlist_start, L, perms_extended, state) - expm_multiply(
    1j * t * hamil, state), ord=2) + np.linalg.norm(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state) - expm_multiply(
    -1j * t * hamil, state), ord=2))/2)
print("Trotter error forwards: ", np.linalg.norm(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state) - expm_multiply(
    -1j * t * hamil, state), ord=2))

print('fidelity: ', (state_fidelity(ansatz_sparse(Vlist_start, L, perms_extended, state), expm_multiply(
    1j * t * hamil, state)) + state_fidelity(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state), expm_multiply(
    -1j * t * hamil, state)))/2)



from optimize_sparse import optimize, err

with h5py.File("./results/triangularXY_ccU_SPARSE_J111_h3-11_Lx4Ly4_t0.125_layers45_niter5_rS1_2hloc.hdf5", "r") as f:
    Vlist_start =  f["Vlist"][:]

Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_start, perms_extended, perms_reduced=perms_ext_reduced, 
                                   control_layers=control_layers, rS=1, niter=5)
plt.plot(err_iter)
plt.yscale('log')
print(err_iter[-1])


with h5py.File(f"./triangularXY_ccU_SPARSE_{J}{h}{g}_Lx{Lx}Ly{Ly}_t{t}_layers{len(Vlist_start)}_niter5_rS1_2hloc.hdf5", "w") as f:
    f.create_dataset("Vlist", data=Vlist)
    f.create_dataset("f_iter", data=f_iter)
    f.create_dataset("err_iter", data=err_iter)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)

