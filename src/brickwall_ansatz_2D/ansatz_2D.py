import numpy as np
import qiskit
from utils_2D import (
	applyG_tensor, applyG_block_tensor, applyG_block_state,
	partial_trace_keep, antisymm_to_real, antisymm, I2, X, Y, Z,
	project_unitary_tangent, real_to_antisymm, partial_trace_keep_tensor
	)


def ansatz_2D(Vlist, L, perms, reps=1):
    ret_tensor = np.eye(2**L, dtype=complex).reshape([2]*2*L)
    for j, perm in enumerate(perms):
        ret_tensor = applyG_block_tensor(Vlist[j//reps], ret_tensor, L, perm)
    return ret_tensor.reshape(2**L, 2**L)



def ansatz_2D_grad(V, L, U_tilde_tensor, perms):
    G = np.zeros_like(V, dtype=complex)
    for _, perm in enumerate(perms):
        U_working1 = np.eye(2**L).reshape([2]*2*L)
        for j in range(_+1, len(perms)):
            U_working1 = applyG_block_tensor(V, U_working1, L, perms[j])
        U_working1 = (U_tilde_tensor.reshape(2**L, 2**L) @ U_working1.reshape(2**L, 2**L)).reshape([2]*2*L)
        for j in range(_):
            U_working1 = applyG_block_tensor(V, U_working1, L, perms[j])

        for i in range(len(perm) // 2):
            k, l = perm[2 * i], perm[2 * i + 1]
            U_working = U_working1.copy()
            for j in range(i):
                k_, l_ = perm[2 * j], perm[2 * j + 1]
                U_working = applyG_tensor(V, U_working, k_, l_)
            for j in range(i + 1, len(perm) // 2):
                k_, l_ = perm[2 * j], perm[2 * j + 1]
                U_working = applyG_tensor(V, U_working, k_, l_)
            T = partial_trace_keep(U_working.reshape(2**L, 2**L), [k, l], L)
            if k > l:
                SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
                T = SWAP @ T @ SWAP
            G += T
    return G


def ansatz_2D_grad_vector(Vlist, L, cU, perms_extended, reps=1, flatten=True, unprojected=False):
    grad = [None for V in Vlist]

    for i, V in enumerate(Vlist):
        U_tilde = np.eye(2**L).reshape([2]*2*L)
        perms = perms_extended[reps*i:reps*(i+1)]

        for j in range(i+1, len(Vlist)):
            perms_j = perms_extended[reps*j:reps*(j+1)]
            for perm in perms_j:
                U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perm)
        U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
        for j in range(i):
            perms_j = perms_extended[reps*j:reps*(j+1)]
            for perm in perms_j:
                U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perm)
        grad[i] = ansatz_2D_grad(V, L, U_tilde, perms).conj().T

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


def ansatz_sparse(Vlist, L, perms, input_state, reps=1):
    """
    Applies sequence of gates specified by Vlist and perms to ground_state.
    """
    state = input_state.copy()
    for j, perm in enumerate(perms):
        state = applyG_block_state(Vlist[j//reps], state, L, perm)
    return state



