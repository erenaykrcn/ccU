import qiskit
from qiskit.quantum_info import state_fidelity
import numpy as np
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py
import sys
sys.path.append("../../../src/brickwall_ansatz")
from utils import construct_heisenberg_local_term
sys.path.append("../../../src/brickwall_sparse")
from optimize_sparse import optimize
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from functools import reduce


niter = 20
t = 0.4
rS = 1
result_string = None
custom_result_string = ""
layers = 72
# ext layers: 22

def bonds_from_perms(perms):
    """
    Each row p in perms encodes pairs:
    (p[0], p[1]), (p[2], p[3]), ...
    """
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
    """
    Build the operator:
        I ⊗ ... ⊗ pauli1(at i) ⊗ ... ⊗ pauli2(at j) ⊗ ... I
    """
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
p1, p2, p3, p4, p5, p6 = ([perms_1[0]], [perms_1[1]], [perms_2[0]], [perms_2[1]], [perms_3[0]], [perms_3[1]])
ps = [p1, p2, p3, p4, p5, p6]
bonds_1 = bonds_from_perms(perms_1)
bonds_2 = bonds_from_perms(perms_2)
bonds_3 = bonds_from_perms(perms_3)
all_bonds = bonds_1 + bonds_2 + bonds_3
J = (1, 1, 1)
h = (3, -1, 1)
L = 12
hamil = build_H(L, all_bonds, J, h, 4)
hloc = construct_heisenberg_local_term((J[0], J[1], J[2]), (h[0], h[1], h[2]), ndim=2)
V = scipy.linalg.expm(-1j*t*hloc/(layers//6))
Vlist_reduced = [V for i in range(layers)]
    

if layers==72:
    control = list(range(0, 85, 7))
    perms_reduced = [p1, p2, p3, p4, p5, p6]*12
    perms_ext = [p2] + ps  + [p3] + ps  + [p5]  + ps + [p2]  + ps + [p3] +  ps + [p5] + ps + [p2] +\
        ps  + [p3] + ps  + [p5]  + ps + [p2]  + ps + [p3] +  ps + [p5] + ps + [p2]
    with h5py.File(f"../results/kagome_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_t{t/2}_layers43_rS{rS}_opt_SHORT{custom_result_string}.hdf5", 'r') as f:
        Vlist_start_2  =  f["Vlist"][:]
    Vlist_start = list(Vlist_start_2) + list(Vlist_start_2)[1:]
    Vlist_start[42] = Vlist_start_2[0] @ Vlist_start_2[-1]


if layers==36:
    control = list(range(0, 43, 7))
    perms_reduced = [p1, p2, p3, p4, p5, p6]*6
    perms_ext = [p2] + ps  + [p3] + ps  + [p5]  + ps + [p2]  + ps + [p3] +  ps + [p5] + ps + [p2]
    with h5py.File(f"../results/kagome_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_t{t/2}_layers22_rS{rS}_opt_SHORT{custom_result_string}.hdf5", 'r') as f:
        Vlist_start_2  =  f["Vlist"][:]
    Vlist_start = list(Vlist_start_2) + list(Vlist_start_2)[1:]
    Vlist_start[21] = Vlist_start_2[0] @ Vlist_start_2[-1]

elif layers==18:
    Vlist_start = [np.eye(4), V, V, V, V, V, V, np.eye(4), V, V, V, V, V, V, np.eye(4), V, V, V, V, V, V, np.eye(4)]
    control = [0, 7, 14, 21]
    perms_reduced = ps*3
    perms_ext = [p2] + ps  + [p3] + ps  + [p5]  + ps + [p2]

elif layers==54:
    control = list(range(0, 64, 7))
    print('Control Layers: ', control)
    perms_reduced = [p1, p2, p3, p4, p5, p6]*9
    perms_ext = [p2] + ps  + [p3] + ps  + [p5]  + ps + [p2]  + ps + [p3] +  ps + [p5] + ps + [p2] + ps  + [p3] + ps  + [p5]  + ps + [p2] 
    with h5py.File(f"../results/kagome_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_t{round(t/3, 3)}_layers22_rS{rS}_opt_SHORT{custom_result_string}.hdf5", 'r') as f:
        Vlist_start_2  =  f["Vlist"][:]

    Vlist_start = list(Vlist_start_2) + list(Vlist_start_2)[1:] + list(Vlist_start_2)[1:]
    Vlist_start[21] = Vlist_start_2[0] @ Vlist_start_2[-1]
    Vlist_start[42] = Vlist_start_2[0] @ Vlist_start_2[-1]


if result_string is not None:
    with h5py.File(f'../results/{result_string}') as f:
        Vlist_start  =  f["Vlist"][:]

Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_start, perms_ext, perms_reduced=perms_reduced, 
                                       control_layers=control, rS=rS, niter=niter, log_txt=custom_result_string)


with h5py.File(f"../results/kagome_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_t{t}_layers{len(Vlist)}_rS{rS}_opt_SHORT{custom_result_string}.hdf5", "w") as f:
    f.create_dataset("Vlist", data=Vlist)
    f.create_dataset("f_iter", data=f_iter)
    f.create_dataset("err_iter", data=err_iter)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)