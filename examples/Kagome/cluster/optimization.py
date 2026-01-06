import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from functools import reduce
import sys
import scipy
sys.path.append("../../../src/brickwall_sparse")
from utils_sparse import construct_ising_local_term, reduce_list, X, I2, get_perms, construct_heisenberg_local_term
from ansatz_sparse import ansatz_sparse
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm
from qiskit.quantum_info import state_fidelity


result_string = None
niter = 20
t = 0.1
rS = 1
layers = 72


def bonds_from_perms(perms):
    bonds = []
    for p in perms:
        assert len(p) % 2 == 0
        for k in range(0, len(p), 2):
            bonds.append((p[k], p[k+1]))
    return bonds

sx = sp.csr_matrix([[0, 1],
                    [1, 0]], dtype=complex)

sy = sp.csr_matrix([[0, -1j],
                    [1j,  0]], dtype=complex)

sz = sp.csr_matrix([[1,  0],
                    [0, -1]], dtype=complex)

id2 = sp.identity(2, dtype=complex, format='csr')


def two_site_pauli_term(L, i, j, pauli1, pauli2):
    if i == j:
        raise ValueError("i and j must be different")

    if i > j:
        i, j = j, i
        pauli1, pauli2 = pauli2, pauli1

    ops = []
    for site in range(L):
        if site == i:
            ops.append(pauli1)
        elif site == j:
            ops.append(pauli2)
        else:
            ops.append(id2)

    op = ops[0]
    for k in range(1, L):
        op = sp.kron(op, ops[k], format='csr')
    return op


def build_H(L, bonds, J, h, n_neighbours):
    dim = 2**L
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for (i, j) in bonds:
        if J[0] != 0:
            H += J[0] * two_site_pauli_term(L, i, j, sx, sx)
        if J[1] != 0:
            H += J[1] * two_site_pauli_term(L, i, j, sy, sy)
        if J[2] != 0:
            H += J[2] * two_site_pauli_term(L, i, j, sz, sz)

        if h[0] != 0:
            H += h[0]  * (two_site_pauli_term(L, i, j, sx, id2) + two_site_pauli_term(L, i, j, id2, sx))/n_neighbours
        if h[1] != 0:
            H += h[1]  * (two_site_pauli_term(L, i, j, sy, id2) + two_site_pauli_term(L, i, j, id2, sy))/n_neighbours
        if h[2] != 0:
            H += h[2]  * (two_site_pauli_term(L, i, j, sz, id2) + two_site_pauli_term(L, i, j, id2, sz))/n_neighbours
    return H

perms_1 = [[0, 4, 6, 10, 2, 5, 8, 11], [4, 6, 10, 0, 5, 8, 11, 2]]
perms_2 = [[0, 1, 2, 3, 6, 7, 8, 9], [1, 2, 3, 0, 7, 8, 9, 6]]
perms_3 = [[1, 4, 9, 11, 3, 5, 7, 10], [4, 1, 11, 9, 5, 7, 10, 3]]
bonds_1 = bonds_from_perms(perms_1)
bonds_2 = bonds_from_perms(perms_2)
bonds_3 = bonds_from_perms(perms_3)
all_bonds = bonds_1 + bonds_2 + bonds_3
L = 12
J = (1, 1, 1)
h = (3, -1, 1)
hamil = build_H(L, all_bonds, J, h, 4)


X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
YZ = np.kron(Y, Z)
I2 = np.array([[1, 0], [0, 1]])
XZ = np.kron(X, Z)
XY = np.kron(X, Y)
ZY = np.kron(Z, Y)


state = np.array(random_statevector(2**L).data)
hloc1 = construct_heisenberg_local_term((J[0], 0   ,    0), (0, h[1],       0), ndim=2)
hloc2 = construct_heisenberg_local_term((0   ,    J[1], 0), (0, 0, h[2]   ), ndim=2)
hloc3 = construct_heisenberg_local_term((0   , 0   , J[2]), (h[0], 0,       0), ndim=2)


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
elif layers == 72:
    hloc1_1 = construct_heisenberg_local_term((J[0], 0   ,    0), (0, 0,       0), ndim=2)
    hloc1_2 = construct_heisenberg_local_term((0, 0   ,    0), (0, h[1],       0), ndim=2)
    hloc2_1 = construct_heisenberg_local_term((0   ,    J[1], 0), (0, 0, 0   ), ndim=2)
    hloc2_2 = construct_heisenberg_local_term((0   ,    0, 0), (0, 0, h[2]   ), ndim=2)
    hloc3_1 = construct_heisenberg_local_term((0   , 0   , J[2]), (0, 0,       0), ndim=2)
    hloc3_2 = construct_heisenberg_local_term((0   , 0   , 0), (h[0], 0,       0), ndim=2)
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
                                   control_layers=control_layers, rS=rS, niter=niter)

with h5py.File(f"../results/kagome_Heis_L{L}_t{t}_layers{len(Vlist)}_rS{rS}.hdf5", "w") as f:
    f.create_dataset("Vlist", data=Vlist)
    f.create_dataset("f_iter", data=f_iter)
    f.create_dataset("err_iter", data=err_iter)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)

