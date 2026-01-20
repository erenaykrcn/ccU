import numpy as np
import qiskit
from qiskit.quantum_info import state_fidelity
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py

import sys
sys.path.append("../../../src/brickwall_ansatz")
from utils import construct_heisenberg_local_term, X, I2
from precision import HP as prec
from ansatz import ansatz
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from optimize import optimize


def random_hermitian(n, normalize=True):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H = (A + A.conj().T) / 2
    evals = np.linalg.eigvalsh(H)
    norm = np.max(np.abs(evals)) if normalize else 1
    return H / norm


N, L = (4, 3)

perms = [[0, 1, 2, 3] if i%2==0 else [1, 2, 3, 0] for i in range(L)]
layers = len(perms)
V = lambda t: scipy.linalg.expm(-1j*t*random_hermitian(4))


for t in [0.1, 0.5, 1]:
  for _ in range(100):
    G = ansatz([V(t) for i in range(layers)], 
      N, perms) # Randomly Chosen Target from the Reachable Manifold of the Ansatz.
    Vlist = [V(t) for i in range(layers)]
    Vlist_trap, f_iter, err_iter = optimize(N, G, len(Vlist), 1,
                                               Vlist, perms, niter=100,
                                               
                                               rho_trust=1e-1, radius_init=0.01, maxradius=0.1,
                                               tcg_abstol=1e-12, tcg_reltol=1e-10, tcg_maxiter=100
                                              )
    
    with open(f"./logs/log_L{L}_N{N}_t{t}.txt", "a") as file:
      file.write(f"{err_iter[-1]} \n")


