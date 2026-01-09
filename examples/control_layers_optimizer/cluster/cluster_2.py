import qiskit
from qiskit.quantum_info import state_fidelity
import numpy as np
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py


I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
ket_0 = np.array([[1],[0]])
ket_1 = np.array([[0],[1]])
rho_0_anc = ket_0 @ ket_0.T
rho_1_anc = ket_1 @ ket_1.T

import sys
sys.path.append("../../../src/brickwall_ansatz")
from optimize import optimize
from utils import construct_heisenberg_local_term
from ansatz import ansatz
import rqcopt as oc
from qiskit.quantum_info import random_statevector
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm

L = 6
latt = qib.lattice.IntegerLattice((L,), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J = (1, 1, 1)
h = (3, -1, 1)
hamil = qib.HeisenbergHamiltonian(field, J, h).as_matrix().toarray()
t = 0.25

hloc1 = construct_heisenberg_local_term((J[0], 0   ,    0), (0, h[1], 0))
hloc2 = construct_heisenberg_local_term((0   , J[1],    0), (0, 0, h[2]))
hloc3 = construct_heisenberg_local_term((0   , 0   , J[2]), (h[0], 0, 0))
hlocs = (hloc1, hloc2, hloc3)

r = 2
Vs = [scipy.linalg.expm(-1j*t*hlocs[0]/r), scipy.linalg.expm(-1j*t*hlocs[1]/r), 
      scipy.linalg.expm(-1j*t*hlocs[2]), 
      scipy.linalg.expm(-1j*t*hlocs[1]/r), scipy.linalg.expm(-1j*t*hlocs[0]/r)
]
cs = [np.eye(4) for i in range(4)]
Vlist_start = [cs[0], Vs[0], Vs[0], Vs[1], Vs[1], cs[1], Vs[2], Vs[2], Vs[1], Vs[1], cs[2], Vs[0], Vs[0], np.eye(4), np.eye(4), cs[3]]
Vlist_reduced = [Vs[0], Vs[0], Vs[1], Vs[1], 
                 Vs[2], Vs[2], Vs[1], Vs[1], 
                 Vs[0], Vs[0], np.eye(4), np.eye(4)]
control_layers = [0, 5, 10, 15] # 4 control layers

perms_extended = [[i for i in range(L)] if i%2==0 else [i for i in range(1, L)]+[0] for i in range(len(Vlist_start))]
perms_ext_reduced = []
for i, perm in enumerate(perms_extended):
    if i not in control_layers:
        perms_ext_reduced.append(perm)

eta, gamma = (3, 4)
niter = 5000
Vlist_ID, f_iter, err_iter = optimize(L, expm(-1j * t * hamil), eta, gamma, 
	Vlist_start, perms_extended, niter=niter, conv_tol=1e-20)


