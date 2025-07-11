"""
    Attempt for the controlled version:
"""
import numpy as np
import qiskit
import qib
import matplotlib.pyplot as plt
import scipy
import h5py
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector

import sys
sys.path.append("../../src/brickwall_sparse")
from utils_sparse import construct_ising_local_term, get_perms
from ansatz_sparse import ansatz_sparse
from optimize_sparse import optimize

# Optimization parameters! Hard-coded here.
rS    = 20 # how many random state vectors to be sampled over for the cost function.
niter = 30 # number of iterations of the riemannian protocol.

Lx, Ly = (2, 2)
L = Lx*Ly
t = 0.25
latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J, h, g = (1, 0, 3)
hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
I2 = np.array([[1, 0], [0, 1]])

hloc1 = construct_ising_local_term(J, 0, 0, ndim=2)/2
hloc2 = g*(np.kron(X, I2)+np.kron(I2, X))/4

V1 = scipy.linalg.expm(-1j*t*hloc1/4)
V2 = scipy.linalg.expm(-1j*t*hloc2/2)

V3 = scipy.linalg.expm(-1j*t*hloc1/2)
V4 = scipy.linalg.expm(-1j*t*hloc2)
YZ = np.kron(Y, Z)


Vlist_start = [YZ, V1, V2, V1, YZ, YZ, V3, V4, V3, YZ, YZ, V1, V2, V1, YZ]
Vlist_reduced = [V1, V2, V1, V3, V4, V3, V1, V2, V1]

#perms_v, perms_h = get_perms(Lx, Ly)
perms_v, perms_h = ([[0, 1, 2, 3], [1, 0, 3, 2]], [[0, 2, 1, 3], [2, 0, 3, 1]])

perms_extended = [[perms_v[0]]] + [perms_v]*3 + [[perms_v[0]], [perms_h[0]]] +\
                    [perms_h]*3 + [[perms_h[0]], [perms_v[0]]] + [perms_v]*3 + [[perms_v[0]]]
perms_ext_reduced = [perms_v]*3  + [perms_h]*3 + [perms_v]*3
control_layers = [0, 4, 5, 9, 10, 14] # 6 control layers

# 24 layers with 6 being controlled, 15 parameters in total.
state = random_statevector(2**L).data
print("Trotter error of the starting point: ", 
  (
    np.linalg.norm(ansatz_sparse(Vlist_start, L, perms_extended, state) - expm_multiply(
      1j * t * hamil, state), ord=2) +\
    np.linalg.norm(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state) - expm_multiply(
      -1j * t * hamil, state), ord=2)
  )/2
)

"""
Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_start, perms_extended, perms_reduced=perms_ext_reduced, 
                                   control_layers=control_layers, rS=rS, niter=niter)
plt.plot(err_iter)
plt.yscale('log')
plt.savefig(f"./cluster_results/tfim2d_ccU_SPARSE_{J}{h}{g}_Lx{Lx}Ly{Ly}_t{t}_layers{len(Vlist_start)}_rS{rS}_niter{niter}_3hloc.png")
print("Trotter error of final point: ", err_iter[-1])

file_path = f"./cluster_results/tfim2d_ccU_SPARSE_{J}{h}{g}_Lx{Lx}Ly{Ly}_t{t}_layers{len(Vlist_start)}_rS{rS}_niter{niter}_3hloc.hdf5"
print("Saving the result to: ", file_path)
with h5py.File(file_path, "w") as f:
    f.create_dataset("Vlist", data=Vlist)
    f.create_dataset("f_iter", data=f_iter)
    f.create_dataset("err_iter", data=err_iter)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)
"""

