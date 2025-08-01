import numpy as np
import qiskit
from qiskit.quantum_info import state_fidelity
from qiskit_aer import AerSimulator
from qiskit import transpile

from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py

import sys
sys.path.append("../../src/brickwall_sparse")
from ansatz_sparse import ansatz_sparse, construct_ccU
from utils_sparse import get_perms
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector

J, h, g = (1, 0, 1)
Lx, Ly = (4, 4)
L = Lx*Ly
t = .125

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
I2 = np.array([[1, 0], [0, 1]])

perms_v, perms_h = get_perms(Lx, Ly)
"""perms_v, perms_h = (
    [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6, 13, 14, 15, 16, 17, 12, 19, 20, 21, 22, 23, 18, 25, 26, 27, 28, 29, 24, 31, 32, 33, 34, 35, 30]],
    [[0, 6, 12, 18, 24, 30, 1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29, 35], 
    [6, 12, 18, 24, 30, 0, 7, 13, 19, 25, 31, 1, 8, 14, 20, 26, 32, 2, 9, 15, 21, 27, 33, 3, 10, 16, 22, 28, 34, 4, 11, 17, 23, 29, 35, 5]]
)"""


"""
    Compressed-Controlled Time Evolution Operator that we optimized previously.
"""
import h5py
import sys
sys.path.append("../../src/brickwall_sparse")
from ansatz_sparse import ansatz_sparse, construct_ccU


perms_qc = [[0, 1], [0, 2]]
Xlists_opts = {}
Vlists = {}
qc_cUs = {}
for t in [.125]:
    Vlist = []
    with h5py.File(f"./results/tfim2d_ccU_SPARSE_101_Lx4Ly4_t{t}_layers6_niter8_rS1_2hloc.hdf5", "r") as f:
        Vlist =  f["Vlist"][:]
    perms_extended = [[perms_v[0]]] + [perms_v]*1 + [[perms_v[0]], [perms_h[0]]] +\
                            [perms_h]*1 + [[perms_h[0]]]
    perms_ext_reduced = [perms_v]*1  + [perms_h]*1
    control_layers = [0, 2, 3, 5]           # 4 control layers
    perms_qc = [[0, 1], [0, 2]]
    Xlists_opt = {}
    for i in control_layers:
        with h5py.File(f"./results/tfim2d_ccU_SPARSE_{J}{h}{g}_Lx4Ly4_t{t}_layers{len(Vlist)}_niter15_rS1_DECOMPOSE_n{len(perms_qc)}_layer{i}.hdf5", "r") as file:
            Xlists_opt[i] = file[f"Xlist_{i}"][:]

    
    Xlists_opts[t] = Xlists_opt
    Vlists[t] = Vlist
    qc_cUs[t] = construct_ccU(L, Vlist, Xlists_opt, perms_extended, perms_qc, control_layers)


qc = qiskit.QuantumCircuit(L+1, 1)    
for y in range(Ly):
    for x in range(Lx):
        i = Ly * y + x  # qubit index in row-major order
        if (x + y) % 2 == 1:
            qc.x(i)  # Flip qubit to |1⟩ (spin down)
qc_real = qc.copy()
qc_imag = qc.copy()

qc_real.h(L)
qc_real.append(qc_cUs[0.125], [i for i in range(L+1)])
qc_real.h(L)
qc_real.measure(L, 0)

qc_imag.h(L)
qc_imag.append(qc_cUs[0.125], [i for i in range(L+1)])
qc_imag.p(-0.5*np.pi, L)
qc_imag.h(L)
qc_imag.measure(L, 0)

simulator = AerSimulator(method='matrix_product_state')
tcirc = transpile(qc_real, simulator)
result = simulator.run(tcirc).result()
counts_real = result.get_counts(0)

with open(f"mps_result.txt", "a") as file:
    file.write(f"Real Counts: \n")
    file.write(f"{counts_real} \n\n")


simulator = AerSimulator(method='matrix_product_state')
tcirc = transpile(qc_imag, simulator)
result = simulator.run(tcirc).result()
counts_imag = result.get_counts(0)

with open(f"mps_result.txt", "a") as file:
    file.write(f"Imag Counts: \n")
    file.write(f"{counts_imag} \n\n")


