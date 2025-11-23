import numpy as np
import qiskit
from utils_PEPS import (
    applyG_block_PEPS, applyG_PEPS,
	antisymm_to_real, antisymm, I2, X, Y, Z,
    partial_inner_product, compute_overlap
	)


def ansatz_PEPS(Vlist, L, perms, state, max_bond_dim, chi_overlap=10):
    for j, V in enumerate(Vlist):
        for perm in perms[j]:
            state = applyG_block_PEPS(V, state, L, perm, max_bond_dim, chi_overlap)
    return state


def ansatz_PEPS_grad(V, L, v, w, perms, max_bond_dim, chi_overlap):
    grad = np.zeros_like(V, dtype=complex)
    for _, perm in enumerate(perms):
        v1, w1 = (v.copy(), w.copy())
        for j in range(_):
            v1 = applyG_block_PEPS(V, v1, L, perms[j], max_bond_dim)
        for j in list(range(_+1, len(perms)))[::-1]: # reverse! 
            w1 = applyG_block_PEPS(V.conj().T, w1, L, perms[j], max_bond_dim) # !! -> conj().T
        
        for i in range(len(perm) // 2):
            v_working = v1.copy()
            k, l = perm[2 * i], perm[2 * i + 1]
            for j in range(i):
                k_, l_ = perm[2 * j], perm[2 * j + 1]
                v_working = applyG_PEPS(V, v_working, L, k_, l_, max_bond_dim)
            for j in range(i + 1, len(perm) // 2):
                k_, l_ = perm[2 * j], perm[2 * j + 1]
                v_working = applyG_PEPS(V, v_working, L, k_, l_, max_bond_dim)

            w1 /= np.sqrt(compute_overlap(w1, w1, chi_overlap))
            v_working /= np.sqrt(compute_overlap(v_working, v_working, chi_overlap))
            T = partial_inner_product(w1, v_working, k, l, chi_overlap).conj() # -> conj!
            #if k > l:
            #    SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            #    T = SWAP @ T @ SWAP
            grad += T
    return grad


def ansatz_PEPS_grad_vector(Vlist, L, reference_state, state, perms_extended, 
    max_bond_dim, chi_overlap,
    flatten=True, unprojected=False):
    #  |   | -           - |    |
    #  | v | - Brickwall - | Uv |
    #  |   | -           - |    |

    dVlist = [None for i in range(len(Vlist))]
    for i, V in enumerate(Vlist):
        perms = perms_extended[i]
        v, w = (state.copy(), reference_state.copy())
        for j in range(i):
            for perm in perms_extended[j]:
                v = applyG_block_PEPS(Vlist[j], v, L, perm, max_bond_dim)
        for j in list(range(i+1, len(Vlist)))[::-1]:
            for perm in list(perms_extended[j])[::-1]:
                w = applyG_block_PEPS(Vlist[j].conj().T, w, L, perm, max_bond_dim)
        grad = ansatz_PEPS_grad(V, L, v, w, perms, max_bond_dim, chi_overlap)
        dVlist[i] = grad

    if unprojected:
        return dVlist
    # Project onto tangent space.
    if flatten:
        return np.stack([
            antisymm_to_real(antisymm(Vlist[j].conj().T @ dVlist[j]))
            for j in range(len(dVlist))
        ]).reshape(-1)
    else:
        return np.stack([
            antisymm_to_real(antisymm(Vlist[j].conj().T @ dVlist[j]))
            for j in range(len(dVlist))
        ])


