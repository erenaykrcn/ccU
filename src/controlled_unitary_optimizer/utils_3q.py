import numpy as np

import sys
sys.path.append("../brickwall_ansatz")
from utils import applyG_tensor, antisymm_to_real, antisymm, partial_trace_keep, real_to_antisymm


I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
ket_0 = np.array([[1],[0]])
ket_1 = np.array([[0],[1]])
rho_0_anc = ket_0 @ ket_0.T
rho_1_anc = ket_1 @ ket_1.T


# Random unitary generator
def random_unitary(n):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)
    return Q


def unflatten(x, nG, nV):
    li = []
    idx = 0
    # Recover Glist gradients (2x2 matrices)
    for _ in range(nG):
        g_flat = x[idx:idx+4]
        li.append(g_flat.reshape(2, 2))
        idx += 4
    # Recover Vlist gradients (4x4 matrices)
    for _ in range(nV):
        v_flat = x[idx:idx+16]
        li.append(v_flat.reshape(4, 4))
        idx += 16
    return li


def make_controlled(U):
    return np.kron(rho_1_anc, U) + np.kron(rho_0_anc, np.eye(U.shape[0]))


def cU_ansatz_bare(Xs, perms):
    Vs = Xs
    L = 3
    ret_tensor = np.eye(2**L, dtype=complex).reshape([2]*2*L)
    for i, V in enumerate(Vs):
        k, l = perms[i]
        ret_tensor = applyG_tensor(V, ret_tensor, k, l)
    return ret_tensor.reshape((2**L, 2**L))

def cU_grad_bare(cU, Xs, perms, flatten=True, unprojected=False):
    L = 3
    Vs = Xs
    
    grad = []
    for i in range(len(Vs)):
        k, l = perms[i]
        G =  np.eye(2**L).reshape([2]*2*L)

        for j in range(i+1, len(Vs)):
            k_, l_ = perms[j]
            G = applyG_tensor(Vs[j], G, k_, l_)
        G = (cU.conj().T @ G.reshape((2**L, 2**L))).reshape([2]*2*L)
        
        for j in range(i):
            k_, l_ = perms[j]
            G = applyG_tensor(Vs[j], G, k_, l_)

        G = partial_trace_keep(G.reshape(2**L, 2**L), [k, l], L)
        if k > l:
            SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            G = SWAP @ G @ SWAP
        
        grad.append(-G.conj().T)
        
    if unprojected:
        return grad
        
    Wlist = Vs
    # Project onto tangent space.
    if flatten:
        return np.concatenate([
            antisymm_to_real(antisymm(Wlist[j].conj().T @ grad[j])).reshape(-1)
            for j in range(len(grad))
        ])
    else:
        return [
            antisymm_to_real(antisymm(Wlist[j].conj().T @ grad[j]))
            for j in range(len(grad))
        ]