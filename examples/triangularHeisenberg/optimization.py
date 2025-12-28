import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from functools import reduce
import sys
import qib
import scipy
sys.path.append("../../src/brickwall_sparse")
from utils_sparse import construct_ising_local_term, reduce_list, X, I2, get_perms, construct_heisenberg_local_term
from ansatz_sparse import ansatz_sparse
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm
from qiskit.quantum_info import state_fidelity


result_string = None
niter = 40
t = 0.1
layers = 36


perms_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]]
perms_2 = [[0, 5, 10, 15, 3, 4, 9, 14, 2, 7, 8, 13, 1, 6, 11, 12], [5, 10, 15, 0, 4, 9, 14, 3, 7, 8, 13, 2, 6, 11, 12, 1]]
perms_3 = [[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], [4, 8, 12, 0, 5, 9, 13, 1, 6, 10, 14, 2, 7, 11, 15, 3]]

Lx, Ly = (4, 4)
L = Lx*Ly
J = (1, 1, 1)
h = (3, -1, 1)
latt = qib.lattice.TriangularLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
hamil = qib.HeisenbergHamiltonian(field, J, h).as_matrix()


X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
YZ = np.kron(Y, Z)
I2 = np.array([[1, 0], [0, 1]])
XZ = np.kron(X, Z)
XY = np.kron(X, Y)
ZY = np.kron(Z, Y)


state = np.array(random_statevector(2**L).data)
hloc1 = construct_heisenberg_local_term((J[0], 0   ,    0), (0, h[1],       0), ndim=3)
hloc2 = construct_heisenberg_local_term((0   ,    J[1], 0), (0, 0, h[2]   ), ndim=3)
hloc3 = construct_heisenberg_local_term((0   , 0   , J[2]), (h[0], 0,       0), ndim=3)


if layers == 36:
    V1 = scipy.linalg.expm(-1j*t*hloc1)
    V2 = scipy.linalg.expm(-1j*t*hloc2)
    V3 = scipy.linalg.expm(-1j*t*hloc3)
    Vlist_start = [XZ, V1, V1, XZ]*3 + [XY, V2, V2, XY]*3  + [ZY, V3, V3, ZY]*3
    Vlist_reduced = [V1]*6 + [V2]*6 + [V3]*6

    perms_extended = [[perms_1[0]]] + [[perms_1[0]]]+ [[perms_1[1]]] + [[perms_1[0]], [perms_2[0]]] +\
          [[perms_2[0]]]+ [[perms_2[1]]]  + [[perms_2[0]], [perms_3[0]]] + [[perms_3[0]]]+ [[perms_3[1]]]  + [[perms_3[0]]]
    perms_extended = perms_extended*3
    perms_ext_reduced =  [[perms_1[0]]]+ [[perms_1[1]]]  +  [[perms_2[0]]]+ [[perms_2[1]]]  +  [[perms_3[0]]]+ [[perms_3[1]]]
    perms_ext_reduced = perms_ext_reduced*3
    print(len(perms_extended))
elif layers == 72:
    hloc1_1 = construct_heisenberg_local_term((J[0], 0   ,    0), (0, 0,       0), ndim=3)
    hloc1_2 = construct_heisenberg_local_term((0, 0   ,    0), (0, h[1],       0), ndim=3)
    hloc2_1 = construct_heisenberg_local_term((0   ,    J[1], 0), (0, 0, 0   ), ndim=3)
    hloc2_2 = construct_heisenberg_local_term((0   ,    0, 0), (0, 0, h[2]   ), ndim=3)
    hloc3_1 = construct_heisenberg_local_term((0   , 0   , J[2]), (0, 0,       0), ndim=3)
    hloc3_2 = construct_heisenberg_local_term((0   , 0   , 0), (h[0], 0,       0), ndim=3)
    V1_1 = scipy.linalg.expm(-1j*t*hloc1_1)
    V1_2 = scipy.linalg.expm(-1j*t*hloc1_2)
    V2_1 = scipy.linalg.expm(-1j*t*hloc2_1)
    V2_2 = scipy.linalg.expm(-1j*t*hloc2_2)
    V3_1 = scipy.linalg.expm(-1j*t*hloc3_1)
    V3_2 = scipy.linalg.expm(-1j*t*hloc3_2)
    Vlist_start = [XZ, V1_1, V1_1, XZ]*3 + [XZ, V1_2, V1_2, XZ]*3 +\
     [XY, V2_1, V2_1, XY]*3  + [XY, V2_2, V2_2, XY]*3 +\
     [ZY, V3_1, V3_1, ZY]*3 + [ZY, V3_2, V3_2, ZY]*3
    Vlist_reduced = [V1_1]*6 + [V1_2]*6  + [V2_1]*6 + [V2_2]*6 + [V3_1]*6 + [V3_2]*6

    perms_extended = [[perms_1[0]]] + [[perms_1[0]]]+ [[perms_1[1]]] + [[perms_1[0]], [perms_2[0]]] +\
          [[perms_2[0]]]+ [[perms_2[1]]]  + [[perms_2[0]], [perms_3[0]]] + [[perms_3[0]]]+ [[perms_3[1]]]  + [[perms_3[0]]]
    perms_extended = perms_extended*6
    perms_ext_reduced =  [[perms_1[0]]]+ [[perms_1[1]]]  +  [[perms_2[0]]]+ [[perms_2[1]]]  +  [[perms_3[0]]]+ [[perms_3[1]]]
    perms_ext_reduced = perms_ext_reduced*6


non_control_layers = []
k = 0
while True:
    a = 1 + 4*k
    b = 2 + 4*k
    if a > len(perms_extended) or b > len(perms_extended):
        break
    non_control_layers.extend([a, b])
    k += 1
control_layers = []
for i in range(len(perms_extended)):
    if i not in non_control_layers:
        control_layers.append(i)
print(control_layers)


print("Trotter error of the starting point: ", 1-state_fidelity(ansatz_sparse(Vlist_start, L, perms_extended, state), expm_multiply(
    1j * t * hamil, state)))
print("Trotter error of the starting point: ", 1-state_fidelity(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state), expm_multiply(
    -1j * t * hamil, state)))
print("Trotter error of the starting point: ", (np.linalg.norm(ansatz_sparse(Vlist_start, L, perms_extended, state) - expm_multiply(
    1j * t * hamil, state), ord=2) + np.linalg.norm(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state) - expm_multiply(
    -1j * t * hamil, state), ord=2))/2)

from optimize_sparse import optimize
import h5py

if result_string is not None:
    with h5py.File(f'../results/{result_string}') as f:
        Vlist_start_2  =  f["Vlist"][:]
else:
    Vlist_start_2 = Vlist_start

Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_start_2, perms_extended, perms_reduced=perms_ext_reduced,
                                   control_layers=control_layers, rS=1, niter=niter)

with h5py.File(f"../results/kagome_Heis_L{L}_t{t}_layers{len(Vlist)}.hdf5", "w") as f:
    f.create_dataset("Vlist", data=Vlist)
    f.create_dataset("f_iter", data=f_iter)
    f.create_dataset("err_iter", data=err_iter)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)

