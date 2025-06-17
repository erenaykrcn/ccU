import numpy as np
import qiskit
from utils_2D import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm, I2, X, Y, Z,
	project_unitary_tangent, real_to_antisymm, partial_trace_keep_tensor
	)


def ansatz_2D(Vlist, L, perms):
    assert len(Vlist)%2 == 0
    Vlist_verticals = Vlist[:len(Vlist)//2]
    Vlist_horizontals = Vlist[len(Vlist)//2:]
    perms_verticals = perms[:len(perms)//2]
    perms_horizontals = perms[len(perms)//2:]

    ret_tensor = np.eye(2**L, dtype=complex).reshape([2]*2*L)
    """for i, perm in enumerate(perms):
        ret_tensor = applyG_block_tensor((
                Vlist_verticals[i//(2*alpha)] if (i//alpha)%2==0 else Vlist_horizontals[i//(2*alpha)]
            ), ret_tensor, L, perm)"""
    
    for i in range(len(Vlist)//2):
        for j, perm in enumerate(perms_verticals):
            ret_tensor = applyG_block_tensor(Vlist_verticals[i], ret_tensor, L, perm)
        for j, perm in enumerate(perms_horizontals):
            ret_tensor = applyG_block_tensor(Vlist_horizontals[i], ret_tensor, L, perm)
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


def ansatz_2D_grad_vector(Vlist, L, cU, perms, flatten=True, unprojected=False):
    assert len(Vlist)%2 == 0
    grad_verticals = []
    grad_horizontals = []
    Vlist_verticals = Vlist[:len(Vlist)//2]
    Vlist_horizontals = Vlist[len(Vlist)//2:]
    perms_verticals = perms[:len(perms)//2]
    perms_horizontals = perms[len(perms)//2:]

    grads = (grad_verticals, grad_horizontals)
    Vl = (Vlist_verticals, Vlist_horizontals)
    perms_l = (perms_verticals, perms_horizontals)

    for _ in range(2): # 2D
        for i, V in enumerate(Vl[_]):
            G = np.zeros_like(V, dtype=complex)
            U_tilde = np.eye(2**L).reshape([2]*2*L)
            if _==0:
                for perm in perms_horizontals:
                    U_tilde = applyG_block_tensor(Vlist_horizontals[i], U_tilde, L, perm)
            for j in range(i+1, len(Vl[_])):
                for perm in perms_verticals:
                    U_tilde = applyG_block_tensor(Vlist_verticals[j], U_tilde, L, perm)
                for perm in perms_horizontals:
                    U_tilde = applyG_block_tensor(Vlist_horizontals[j], U_tilde, L, perm)
            U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
            for j in range(i):
                for perm in perms_verticals:
                    U_tilde = applyG_block_tensor(Vlist_verticals[j], U_tilde, L, perm)
                for perm in perms_horizontals:
                    U_tilde = applyG_block_tensor(Vlist_horizontals[j], U_tilde, L, perm)
            if _==1:
                for perm in perms_verticals:
                    U_tilde = applyG_block_tensor(Vlist_verticals[i], U_tilde, L, perm)
            
            grads[_].append(ansatz_2D_grad(V, L, U_tilde, [perm for perm in perms_l[_]]).conj().T)


    grad = grad_verticals+grad_horizontals
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



