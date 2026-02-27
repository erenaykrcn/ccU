"""
    Convergence Guarantee.
"""
import os
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

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from functools import reduce


from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor


def random_hermitian(n, normalize=True):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H = (A + A.conj().T) / 2
    evals = np.linalg.eigvalsh(H)
    norm = np.max(np.abs(evals)) if normalize else 1
    return H / norm


def bonds_from_perms(perms):
    """
    Each row p in perms encodes pairs:
    (p[0], p[1]), (p[2], p[3]), ...
    """
    bonds = []
    for p in perms:
        assert len(p) % 2 == 0
        for k in range(0, len(p), 2):
            bonds.append((p[k], p[k+1]))
    return bonds

sx = sp.csr_matrix([[0, 1],
                    [1, 0]], dtype=complex)
sy = sp.csr_matrix([[0, -1j],
                    [1j,  0]], dtype=complex)
sz = sp.csr_matrix([[1,  0],
                    [0, -1]], dtype=complex)
id2 = sp.identity(2, dtype=complex, format='csr')


def two_site_operator(L, i, j, H):

    def swap_two_qubit_operator(H):
        """
        Swap the two tensor legs of a 4x4 operator.
        Works for dense numpy array.
        """
        H = np.asarray(H)
        H = H.reshape(2, 2, 2, 2)        # (i,j ; i',j')
        H = H.transpose(1, 0, 3, 2)      # swap i<->j and i'<->j'
        return H.reshape(4, 4)

    
    if i == j:
        raise ValueError("i and j must be different")

    if sp.issparse(H):
        H = H.toarray()
    else:
        H = np.asarray(H)

    # If indices are reversed, swap tensor legs of H
    if i > j:
        i, j = j, i
        H = swap_two_qubit_operator(H)

    dim = 2**L
    op = sp.lil_matrix((dim, dim), dtype=complex)

    for basis_in in range(dim):
        bits = list(format(basis_in, f'0{L}b'))
        two_site_in = int(bits[i] + bits[j], 2)

        for two_site_out in range(4):
            amp = H[two_site_out, two_site_in]
            if amp != 0:
                new_bits = bits.copy()
                new_bits[i] = str((two_site_out >> 1) & 1)
                new_bits[j] = str(two_site_out & 1)
                basis_out = int("".join(new_bits), 2)
                op[basis_out, basis_in] += amp

    return op.tocsr()

def build_H(L, bonds, Hloc=None, norm=1):
    dim = 2**L
    H = sp.csr_matrix((dim, dim), dtype=complex)

    if Hloc is None:
        Hloc = random_hermitian(4)*norm
    
    for (i, j) in bonds:
        H += two_site_operator(L, i, j, Hloc)
    return H
V = lambda t: scipy.linalg.expm(-1j*t*random_hermitian(4))


N, L = 4, 3
perms = [[0, 1, 2, 3], [1, 2, 3, 0], [0, 1, 2, 3]]
all_bonds = bonds_from_perms(perms)
ts = np.logspace(-2, 2, 50)


# module globals
_G = {}
def _init_worker(N, perms, L):
    _G["N"] = N
    _G["perms"] = perms
    _G["L"] = L
def _run(t):
    hamil = build_H(N, all_bonds, norm=1)
    U = scipy.linalg.expm(-1j*t*hamil.todense())
    for _ in range(2000):
        Vlist_reduced = [V(t) for i in range(L)]
        r = optimize(N, U, 
                len(Vlist_reduced), 1, Vlist_reduced, perms, niter=3000, conv_tol=1e-12)

        with open(f"./logs/ConvGuar_log_L{L}_N{N}_t{t}.txt", "a") as file:
          file.write(f"{err_iter[-1]} \n")


nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
ctx = get_context("fork")  # best on Linux HPC; if not available, remove
with open(f"./_C_Plog.txt", "a") as file:
    file.write(f"Workers: {nproc}\n ")
with ProcessPoolExecutor(
        max_workers=nproc,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(N, perms, L),
) as ex:
    ex.map(_run, ts)


