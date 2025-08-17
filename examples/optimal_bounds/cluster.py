import numpy as np
import qiskit
from qiskit.quantum_info import state_fidelity
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py

import sys
sys.path.append("../../src/brickwall_sparse")
from utils_sparse import construct_ising_local_term, reduce_list, X, I2, get_perms
from ansatz_sparse import ansatz_sparse
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from optimize_sparse import optimize


L = 8
ts = list(np.logspace(-4, 4, num=30))
for t in ts:
    latt = qib.lattice.IntegerLattice((L, ), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    J, h, g = (1, 0, 3)
    hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()
    
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    Y = np.array([[0, -1j], [1j, 0]])
    I2 = np.array([[1, 0], [0, 1]])
    
    
    hloc = construct_ising_local_term(J, 0, 0, ndim=2) + g*(np.kron(X, I2)+np.kron(I2, X))/2
    V1 = scipy.linalg.expm(-1j*t*hloc/2)
    V2 = scipy.linalg.expm(-1j*t*hloc)
    Vlist_reduced_3 = [V1, V2, V1]
    perms_3 = [[[i for i in range(L)]] if i%2==0 else [[i for i in range(1, L)]+[0]] for i in range(len(Vlist_reduced_3))]
    
    V1 = scipy.linalg.expm(-1j*t*hloc/4)
    V2 = scipy.linalg.expm(-1j*t*hloc/2)
    Vlist_reduced_6 = [V1, V2, V1]*2
    perms_6 = [[[i for i in range(L)]] if i%2==0 else [[i for i in range(1, L)]+[0]] for i in range(len(Vlist_reduced_6))]
    
    V1 = scipy.linalg.expm(-1j*t*hloc/6)
    V2 = scipy.linalg.expm(-1j*t*hloc/3)
    Vlist_reduced_9 = [V1, V2, V1]*3
    perms_9 = [[[i for i in range(L)]] if i%2==0 else [[i for i in range(1, L)]+[0]] for i in range(len(Vlist_reduced_9))]
    
    V1 = scipy.linalg.expm(-1j*t*hloc/8)
    V2 = scipy.linalg.expm(-1j*t*hloc/4)
    Vlist_reduced_12 = [V1, V2, V1]*4
    perms_12 = [[[i for i in range(L)]] if i%2==0 else [[i for i in range(1, L)]+[0]] for i in range(len(Vlist_reduced_12))]
    

    state = random_statevector(2**L).data
    if t>1:
        niter = 200
    else:
        niter = 50
    rS    = 1


    Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_reduced_3, perms_3, rS=rS, niter=niter)
    with open(f"layers3_results.txt", "a") as file:
        file.write(f"{err_iter[-1]}, \n")
    
    Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_reduced_6, perms_6, rS=rS, niter=niter)
    with open(f"layers6_results.txt", "a") as file:
        file.write(f"{err_iter[-1]}, \n")

    Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_reduced_9, perms_9, rS=rS, niter=niter)
    with open(f"layers9_results.txt", "a") as file:
        file.write(f"{err_iter[-1]}, \n")

    Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_reduced_12, perms_12, rS=rS, niter=niter)
    with open(f"layers12_results.txt", "a") as file:
        file.write(f"{err_iter[-1]}, \n")

