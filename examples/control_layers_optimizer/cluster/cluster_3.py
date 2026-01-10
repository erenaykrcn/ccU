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
n_control_layers = 5

t = 0.5
niter = 2000
for layers in range(n_control_layers-1, 5*(n_control_layers-1), n_control_layers-1):
	eta = layers//(n_control_layers-1)
	hloc = construct_heisenberg_local_term((J[0], J[1], J[2]), (h[0], h[1], h[2]))
	Vlist_reduced = [scipy.linalg.expm(-1j*t*hloc/(layers//2)) for i in range(layers)]
	Vlist_start = [None for i in range(layers+n_control_layers)]

	control_layers = [] 
	for i in range(n_control_layers):
		c = i*(eta + 1)
		control_layers.append(c)
		Vlist_start[c] = np.eye(4)
		if i < n_control_layers - 1:
			Vlist_start[c+1:c+layers//(n_control_layers-1)+1] = Vlist_reduced[eta*i:eta*(i+1)]

	perms_extended = [[i for i in range(L)] if i%2==0 else [i for i in range(1, L)]+[0] for i in range(len(Vlist_start))]
	perms_ext_reduced = []
	for i, perm in enumerate(perms_extended):
	    if i not in control_layers:
	        perms_ext_reduced.append(perm)

	"""
	print("Trotter error of the starting point: ", 1-np.trace(ansatz(Vlist_start, L, perms_extended).conj().T @ scipy.linalg.expm(
	    1j * t * hamil)).real/2**L)
	print("Trotter error of the starting point: ", 1-np.trace(ansatz(Vlist_reduced, L, perms_ext_reduced).conj().T @ scipy.linalg.expm(
	    -1j * t * hamil)).real/2**L)
	"""

	Vlist_ID, f_iter, err_iter = optimize(L, expm(-1j * t * hamil), eta, n_control_layers-1,
		Vlist_start, perms_extended, niter=niter, conv_tol=1e-15)

	with h5py.File(f"../results/ticc_Heis_cLayers{n_control_layers}_L{L}_t{t}_layers{len(Vlist_start)}.hdf5", "w") as f:
	    f.create_dataset("Vlist", data=Vlist_ID)
	    f.create_dataset("f_iter", data=f_iter)
	    f.create_dataset("err_iter", data=err_iter)
	    f.attrs["L"] = L
	    f.attrs["t"] = float(t)


