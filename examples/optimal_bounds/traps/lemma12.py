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


N, L = (4, 2)
perms = [[0, 1, 2, 3] if i%2==0 else [1, 2, 3, 0] for i in range(L)]
V = lambda t, n=4: scipy.linalg.expm(-1j*t*random_hermitian(n))

num_random_searches = 1000
ts = [10, 1, 0.1, 0.01]
R_ests = np.logspace(-2, 1, 50)

Rt = {t: None for t in ts}

for t in ts:
  print("t: ", t)

  for R_est in R_ests:
    print(f"\n R_est: {R_est} \n")
    Rts = []
    for _ in range(num_random_searches):
      target_list = [V(t) for i in range(L)]
      G = ansatz(target_list, N, perms) # Randomly Chosen Target from the Reachable Manifold of the Ansatz.

      G0 = V(R_est, 2**N) @ G
      
      Vlist_0, f_iter, err_iter = optimize(N, G0, L, 1, target_list, perms, niter=100, log=False,
                                          rho_trust=1e-1, radius_init=0.01, maxradius=0.1,
                                          tcg_abstol=1e-12, tcg_reltol=1e-10, tcg_maxiter=100
                                                )
      dist = np.linalg.norm(ansatz(Vlist_0, N, perms)-G, ord=2) # Usually just below the R_est but same order of mag.

      R = dist
      print("R: ", R)

      Vlist_0, f_iter, err_iter = optimize(N, G, L, 1, Vlist_0, perms, niter=200, log=False,
                                    
                                    rho_trust=1e-1, radius_init=0.01, maxradius=0.1,
                                    tcg_abstol=1e-12, tcg_reltol=1e-10, tcg_maxiter=100
      )
      if err_iter[-1] > 1e-6:
        print("err: ", err_iter[-1])
        print("\n \n TRAP ENCOUNTERED \n \n")
        Rts.append(R)
        with open(f"./logs/lemma12_L{L}_N{N}_t{t}.txt", "a") as file:
          file.write(f"TRAP at R={R}, err={err_iter[-1]} \n")

    if len(Rts):
      Rt[t] = np.min(Rts)
      break

  
with open(f"./logs/lemma12.txt", "a") as file:
  file.write(f"Rt: {Rt} \n")


