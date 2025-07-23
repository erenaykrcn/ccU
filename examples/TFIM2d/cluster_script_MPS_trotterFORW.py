import numpy as np
import qiskit
from qiskit.quantum_info import state_fidelity
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm

import sys
sys.path.append("../../src/brickwall_sparse")
from utils_sparse import get_perms
sys.path.append("../../src/MPS")
from utils_MPS import (random_mps, apply_localGate, apply_two_site_operator, 
						mps_to_state_vector, get_mps_of_sv, mps_fidelity)
from MPS import trotter, ccU


J, h, g = (1, 0, 3)

t = 0.25
dt    = 0.25/8 # Trotter step to be used for the 'quasi'-exact reference
order = 2  # Trotter order to be used for the 'quasi'-exact reference
initial_state_BD, exact_state_BD, ccU_BD = (2**0, 2**9, 2**9) # Bond dimensions


Lx, Ly = (6, 6)
L= Lx*Ly

#perms_v, perms_h = get_perms(Lx, Ly)
perms_v, perms_h = (
    [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6, 13, 14, 15, 16, 17, 12, 19, 20, 21, 22, 23, 18, 25, 26, 27, 28, 29, 24, 31, 32, 33, 34, 35, 30]],
    [[0, 6, 12, 18, 24, 30, 1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29, 35], 
    [6, 12, 18, 24, 30, 0, 7, 13, 19, 25, 31, 1, 8, 14, 20, 26, 32, 2, 9, 15, 21, 27, 33, 3, 10, 16, 22, 28, 34, 4, 11, 17, 23, 29, 35, 5]]
)

"""
perms_v, perms_h = (
    [[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23],
    [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24],
    [0, 4, 5, 9, 15, 19, 20, 24]],
    [[0, 5, 1, 6, 2, 7, 3, 8, 4, 9, 10, 15, 11, 16, 12, 17, 13, 18, 14, 19], 
    [5, 10, 6, 11, 7, 12, 8, 13, 9, 14, 15, 20, 16, 21, 17, 22, 18, 23, 19, 24],
    [20, 0, 21, 1, 22, 2, 23, 3, 24, 4]
    ]
)"""


perms_extended = [[perms_v[0]]] + [perms_v]*3 + [[perms_v[0]], [perms_h[0]]] +\
                    [perms_h]*3 + [[perms_h[0]], [perms_v[0]]] + [perms_v]*3 + [[perms_v[0]]]
perms_ext_reduced = [perms_v]*3  + [perms_h]*3 + [perms_v]*3



import os
initial_mps = []
if os.path.isfile(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t0.25_INITIAL_MPS_initBD{initial_state_BD}.h5"):
    with open(f"trotter_log{Lx}{Ly}.txt", "a") as file:
        file.write(f"Init MPS loaded \n")

    with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t0.25_INITIAL_MPS_initBD{initial_state_BD}.h5", "r") as f:
        mps_group = f["mps"]
        initial_mps = [mps_group[f"site_{i}"][()] for i in range(L)]
else:
	initial_mps = random_mps(L, max_bond_dim=initial_state_BD)
	with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t0.25_INITIAL_MPS_initBD{initial_state_BD}.h5", "w") as f:
		mps_group = f.create_group("mps")
		for i, tensor in enumerate(initial_mps):
			mps_group.create_dataset(f"site_{i}", data=tensor)


if os.path.isfile(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t{t}_TROTTER_MPS_FORWARDS_Order{order}_dt{dt}_BD{exact_state_BD}_initBD{initial_state_BD}.h5"):
    with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t{t}_TROTTER_MPS_FORWARDS_Order{order}_dt{dt}_BD{exact_state_BD}_initBD{initial_state_BD}.h5", "r") as f:
        mps_group = f["mps"]
        exact_mps_forwards = [mps_group[f"site_{i}"][()] for i in range(L)]
else:
    exact_mps_forw_input = initial_mps.copy()
    exact_mps_forwards = trotter(exact_mps_forw_input, t, L, Lx, Ly, J, g, perms_v, perms_h, max_bond_dim=exact_state_BD, trotter_order=order, dt=dt)
    with h5py.File(f"./MPS/tfim2d_Lx{Lx}Ly{Ly}__MPS_103_t{t}_TROTTER_MPS_FORWARDS_Order{order}_dt{dt}_BD{exact_state_BD}_initBD{initial_state_BD}.h5", "w") as f:
        mps_group = f.create_group("mps")
        for i, tensor in enumerate(exact_mps_forwards):
            mps_group.create_dataset(f"site_{i}", data=tensor)
        f.attrs["L"] = L
        f.attrs["t"] = float(t)
        f.attrs["order"] = order
        f.attrs["dt"] = dt


