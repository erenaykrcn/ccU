import numpy as np
import qiskit
from utils import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm, I2, X, Y, Z,
	project_unitary_tangent, real_to_antisymm, partial_trace_keep_tensor
	)


def ansatz(Vlist, L, perms):
    """
    Tensor-based ansatz construction, applying gates directly
    to a (2,)*2L tensor without forming large matrices.
    """
    assert len(Vlist) == len(perms)
    ret_tensor = np.eye(2**L, dtype=complex).reshape([2]*2*L)
    for i, V in enumerate(Vlist):
        ret_tensor = applyG_block_tensor(V, ret_tensor, L, perms[i])
    return ret_tensor.reshape(2**L, 2**L)


def ansatz_grad(V, L, U_tilde_tensor, perm):
    G = np.zeros_like(V, dtype=complex)
    for i in range(L // 2):
        k, l = perm[2 * i], perm[2 * i + 1]
        U_working = U_tilde_tensor.copy()
        for j in range(i):
            k_, l_ = perm[2 * j], perm[2 * j + 1]
            U_working = applyG_tensor(V, U_working, k_, l_)
        for j in range(i + 1, L // 2):
            k_, l_ = perm[2 * j], perm[2 * j + 1]
            U_working = applyG_tensor(V, U_working, k_, l_)
        T = partial_trace_keep(U_working.reshape(2**L, 2**L), [k, l], L)
        if k > l:
            SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            T = SWAP @ T @ SWAP
        G += T
    return G


def ansatz_grad_vector(Vlist, L, cU, perms, flatten=True, unprojected=False):
    grad = []
    for i, V in enumerate(Vlist):
        U_tilde = np.eye(2**L).reshape([2]*2*L)
        for j in range(i+1, len(perms)):
            U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
        U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
        for j in range(i):
            U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
        grad.append(ansatz_grad(V, L, U_tilde, perms[i]).conj().T)

    if unprojected:
        return grad

    # Project onto tangent space.
    if flatten:
        return np.stack([
            antisymm_to_real(antisymm(Vlist[j].conj().T @ grad[j]))
            for j in range(len(grad))
        ]).reshape(-1)
    else:
        return np.stack([
            antisymm_to_real(antisymm(Vlist[j].conj().T @ grad[j]))
            for j in range(len(grad))
        ])


def construct_ccU(L, eta, Vs, Xlists_opt, perms, perms_qc):
    nlayers = len(Vs)
    qc = qiskit.QuantumCircuit(L+1)
    qc.x(L)
    for i, V in enumerate(Vs):
        layer = i
        if i in range(0, nlayers, eta+1):
            Glist = Xlists_opt[i]
            qc_3 = qiskit.QuantumCircuit(3)
            for j, G in enumerate(Glist):
                qc_3.unitary( G, (3-1-perms_qc[j][1], 3-1-perms_qc[j][0]))  

            for j in range(L//2):
                qc.append(qc_3.to_gate(), [L-perms[layer][2*j]-1, L-perms[layer][2*j+1]-1, L])

        else:
            for j in range(L//2):
                qc.unitary(V, [L-perms[layer][2*j]-1, L-perms[layer][2*j+1]-1])
    qc.x(L)
    return qc


