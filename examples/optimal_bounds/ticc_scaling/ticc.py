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


L = 4
latt = qib.lattice.IntegerLattice((L,), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J = (1, 1, 1)
h = (3, -1, 1)
hamil = qib.HeisenbergHamiltonian(field, J, h).as_matrix().toarray()


hloc1 = construct_heisenberg_local_term((J[0], 0   ,    0), (0, h[1], 0))
hloc2 = construct_heisenberg_local_term((0   , J[1],    0), (0, 0, h[2]))
hloc3 = construct_heisenberg_local_term((0   , 0   , J[2]), (h[0], 0, 0))
hlocs = (hloc1, hloc2, hloc3)
Vs = [scipy.linalg.expm(-1j*t*hloc/2) for i, hloc in enumerate(hlocs)]
cs = [np.kron(X,  Z), np.kron(Z@Y, I2), np.kron(X@Z, I2), np.kron(Y, Z)]

Vlist_start = [cs[0], Vs[0], Vs[0], cs[1], Vs[1], Vs[1], cs[2], Vs[2], Vs[2], cs[3]]
Vlist_reduced = [Vs[0], Vs[0], Vs[1], Vs[1], Vs[2], Vs[2]]
control_layers = [0, 3, 6, 9] # 4 control layers
perms_extended = [[[i for i in range(L)]] if i%2==0 else [[i for i in range(1, L)]+[0]] for i in range(len(Vlist_start))]
perms_ext_reduced = []
for i, perm in enumerate(perms_extended):
    if i not in control_layers:
        perms_ext_reduced.append(perm)

state = random_statevector(2**L).data
print("Trotter error of the starting point: ", (np.linalg.norm(ansatz_sparse(Vlist_start, L, perms_extended, state) - expm_multiply(
    1j * t * hamil, state), ord=2) + np.linalg.norm(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state) - expm_multiply(
    -1j * t * hamil, state), ord=2))/2)