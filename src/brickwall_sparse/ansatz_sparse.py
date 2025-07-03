import numpy as np
import qiskit
from utils_sparse import (
	applyG_tensor, applyG_block_tensor, applyG_block_state,
	partial_trace_keep, antisymm_to_real, antisymm, I2, X, Y, Z,
	project_unitary_tangent, real_to_antisymm, partial_trace_keep_tensor,
    partial_inner_product, applyG_state, applyG_block_state
	)


def ansatz_sparse(Vlist, L, perms, state):
    for j, V in enumerate(Vlist):
        for perm in perms[j]:
            state = applyG_block_state(V, state, L, perm)
    return state


def ansatz_sparse_grad(V, L, v, w, perms):
    grad = np.zeros_like(V, dtype=complex)
    for _, perm in enumerate(perms):
        v1, w1 = (v.copy(), w.copy())
        for j in range(_):
            v1 = applyG_block_state(V, v1, L, perms[j])
        for j in list(range(_+1, len(perms)))[::-1]: # reverse! 
            w1 = applyG_block_state(V.conj().T, w1, L, perms[j]) # !! -> conj().T
        
        for i in range(len(perm) // 2):
            v_working = v1.copy()
            k, l = perm[2 * i], perm[2 * i + 1]
            for j in range(i):
                k_, l_ = perm[2 * j], perm[2 * j + 1]
                v_working = applyG_state(V, v_working, L, k_, l_)
            for j in range(i + 1, len(perm) // 2):
                k_, l_ = perm[2 * j], perm[2 * j + 1]
                v_working = applyG_state(V, v_working, L, k_, l_)

            T = partial_inner_product(w1, v_working, k, l).conj() # -> conj!
            if k > l:
                SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
                T = SWAP @ T @ SWAP
            grad += T
    return grad


def ansatz_sparse_grad_vector(Vlist, L, Uv, state, perms_extended, flatten=True, unprojected=False):
    #  |   | -           - |    |
    #  | v | - Brickwall - | Uv |
    #  |   | -           - |    |

    dVlist = [None for i in range(len(Vlist))]
    for i, V in enumerate(Vlist):
        perms = perms_extended[i]
        v, w = (state.copy(), Uv.copy())
        for j in range(i):
            for perm in perms_extended[j]:
                v = applyG_block_state(Vlist[j], v, L, perm)
        for j in list(range(i+1, len(Vlist)))[::-1]:
            for perm in list(perms_extended[j])[::-1]:
                w = applyG_block_state(Vlist[j].conj().T, w, L, perm)
        grad = ansatz_sparse_grad(V, L, v, w, perms)
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



def construct_ccU(L, Vs, Xlists_opt, perms, perms_qc, control_layers):
    qc = qiskit.QuantumCircuit(L+1)
    qc.x(L)
    for i, V in enumerate(Vs):
        layer = i
        if i in control_layers:
            Glist = Xlists_opt[i]
            qc_3 = qiskit.QuantumCircuit(3)
            for j, G in enumerate(Glist):
                qc_3.unitary( G, (3-1-perms_qc[j][1], 3-1-perms_qc[j][0]))

            for perm in perms[layer]:
                for j in range(L//2):
                    qc.append(qc_3.to_gate(), [L-perm[2*j]-1, L-perm[2*j+1]-1, L])
            
        else:
            for perm in perms[layer]:
                for j in range(L//2):
                    qc.unitary(V, [L-perm[2*j]-1, L-perm[2*j+1]-1])
    qc.x(L)
    return qc







