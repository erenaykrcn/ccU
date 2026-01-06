import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from functools import reduce
import sys
import scipy
sys.path.append("../../../src/brickwall_sparse")
from utils_sparse import construct_ising_local_term, reduce_list, X, I2, get_perms, construct_heisenberg_local_term
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm
from qiskit.quantum_info import state_fidelity
import qib
sys.path.append("../../../src/brickwall_ansatz")
from optimize import optimize, dynamics_opt
from ansatz import ansatz
import h5py


layers_list = [i for i in range(1, 10)]
for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
	L = 6
	latt = qib.lattice.IntegerLattice((L,), pbc=True)
	field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
	hamil = qib.IsingHamiltonian(field, 1, 0, 1).as_matrix().toarray()
	hloc = construct_ising_local_term(1, 0, 1)

	results = {}
	for layers in layers_list:
		perms = [[[i for i in range(L)]], [[i for i in range(1, L)]+[0]]]*layers
		Vs = [scipy.linalg.expm(-1j*t*hloc/layers)]*layers*2
		Vlist, f_iter, err_iter = dynamics_opt(hamil, t, layers*2, 1, False, Vs, perms, niter=1000)
		results[layers] = Vlist

		with h5py.File(f"./results/rqc_scaling_L{L}_t{t}_layers{layers}.hdf5", "w") as f:
			f.create_dataset("Vlist", data=Vlist)

	L = 8
	latt = qib.lattice.IntegerLattice((L,), pbc=True)
	field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
	hamil = qib.IsingHamiltonian(field, 1, 0, 1).as_matrix().toarray()

	errs = []
	for layers in layers_list:
		perms = [[i for i in range(L)], [i for i in range(1, L)]+[0]]*layers
		state = random_statevector(2**L).data
		err = 1-state_fidelity(ansatz(
	    	results[layers], L, perms)@state, expm_multiply(
		-1j * t * hamil, state) )
		errs.append(err)

	import json
	with open(f"results{t}.json", "w") as f:
		json.dump(errs, f)



