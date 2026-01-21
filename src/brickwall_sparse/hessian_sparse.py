import numpy as np
from ansatz_sparse import ansatz_sparse_grad, ansatz_sparse_grad_vector
from utils_sparse import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm,
	project_unitary_tangent, real_to_antisymm,
	applyG_block_state, applyG_state, partial_inner_product
	)
import os
from multiprocessing import get_context
from concurrent.futures import ThreadPoolExecutor


def _hessian_block(k, j, Vlist, L, Uv, state, perms, unprojected):
    eta = len(Vlist)

    if unprojected:
        Z = np.zeros((4, 4), dtype=complex)
        Z.flat[j] = 1.0
    else:
        Z = np.zeros(16)
        Z[j] = 1
        Z = real_to_antisymm(Z.reshape(4, 4))

    dVZj = ansatz_sparse_hess(
        Vlist, L, Vlist[k] @ Z, k, Uv, state, perms,
        unprojected=unprojected
    )
    block = np.zeros((eta, 16), dtype=float if not unprojected else complex)
    for i in range(eta):
        if unprojected:
            block[i, :] = dVZj[i].reshape(-1)
        else:
            block[i, :] = antisymm_to_real(
                antisymm(Vlist[i].conj().T @ dVZj[i])
            ).reshape(-1)

    return k, j, block



def ansatz_sparse_hessian_matrix(Vlist, L, Uv, state, perms, flatten=True, unprojected=False):
    """
    Construct the Hessian matrix.
    """
    eta = len(Vlist)
    Hess = np.zeros((eta, 16, eta, 16))

    eta = len(Vlist)
    Hess = np.zeros((eta, 16, eta, 16))
    nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    with open(f"./_Hessian_P_log.txt", "a") as file:
        file.write(f"Workers: {nthreads}\n ")

    tasks = [(k, j) for k in range(eta) for j in range(16)]
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        futures = [
            ex.submit(
                _hessian_block, k, j, Vlist, L, Uv, state, perms, unprojected
            )
            for (k, j) in tasks
        ]

        for fut in futures:
            k, j, block = fut.result()
            Hess[:, :, k, j] = block

    return Hess.reshape((eta * 16, eta * 16)) if flatten else Hess



def ansatz_hess_single_layer(V, L, Z, v, w, perm):
	G = np.zeros_like(V, dtype=complex)
	for i in range(len(perm)//2):
		k, l = (perm[2*i], perm[2*i+1])
		for z in range(len(perm)//2):
			v1 = v.copy()
			if z==i:
				continue
			for j in range(i):
				k_, l_ = (perm[2*j], perm[2*j+1])
				if z==j:
					v1 = applyG_state(Z, v1, L, k_, l_)
				else:
					v1 = applyG_state(V, v1, L, k_, l_)

			for j in list(range(i+1, len(perm)//2))[::-1]:
				k_, l_ = (perm[2*j], perm[2*j+1])
				if z==j:
					v1 = applyG_state(Z, v1, L, k_, l_)
				else:
					v1 = applyG_state(V, v1, L, k_, l_) 

			T = partial_inner_product(w, v1, k, l).conj()
			if k > l:
				SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
				T = SWAP @ T @ SWAP
			G += T
	return G


def ansatz_sparse_hess_single_layer(V, L, Z, v, w, perms):
	G = np.zeros_like(V, dtype=complex)
	for dashed_index, perm in enumerate(perms):
		for z_index in range(len(perms)):
			v1, w1 = (v.copy(), w.copy())

			for j in range(dashed_index+1, len(perms))[::-1]:
				if j!= z_index:
					w1 = applyG_block_state(V.conj().T, w1, L, perms[j])
				else:
					w1 = ansatz_sparse_grad_directed(V.conj().T, w1, L, Z.conj().T, [perms[j]])
			for j in range(dashed_index):
				if j!= z_index:
					v1 = applyG_block_state(V, v1, L, perms[j])
				else:
					v1 = ansatz_sparse_grad_directed(V, v1, L, Z, [perms[j]])

			if z_index != dashed_index:
				G += ansatz_sparse_grad(V, L, v1, w1, [perm])
			else:
				G += ansatz_hess_single_layer(V, L, Z, v1, w1, perm)
	return G



def ansatz_sparse_grad_directed(V, state, L, Z, perms):
	s = np.zeros_like(state, dtype=complex)

	for _, perm in enumerate(perms):
		for i in range(len(perm)//2):
			v_working = state.copy()
			for j in range(_):
				v_working = applyG_block_state(V, v_working, L, perms[j])

			k, l = (perm[2*i], perm[2*i+1])

			for j in range(i):
				k_, l_ = (perm[2*j], perm[2*j+1])
				v_working = applyG_state(V, v_working, L, k_, l_) 

			v_working = applyG_state(Z, v_working, L, k, l)

			for j in list(range(i+1, len(perm)//2))[::-1]:
				k_, l_ = (perm[2*j], perm[2*j+1])
				v_working = applyG_state(V, v_working, L, k_, l_)

			for j in range(_+1, len(perms)):
				v_working = applyG_block_state(V, v_working, L, perms[j])
			s += v_working
	return s


def ansatz_sparse_hess(Vlist, L, Z, k, Uv, state, perms_extended, unprojected=False):
	dVlist = [None for i in range(len(Vlist))]

	for i in range(len(Vlist)):
		perms = perms_extended[i]
		v, w = (state.copy(), Uv.copy())
		if i==k:
			continue
		for j in list(range(i+1, len(Vlist)))[::-1]:
			perms_j = perms_extended[j]
			if k!=j:
				for perm in perms_j[::-1]:
					w = applyG_block_state(Vlist[j].conj().T, w, L, perm)
			else:
				w = ansatz_sparse_grad_directed(Vlist[k].conj().T, w, L, Z.conj().T, perms_j[::-1])
		for j in range(i):
			perms_j = perms_extended[j]
			if k!=j:
				for perm in perms_j:
					v = applyG_block_state(Vlist[j], v, L, perm)
			else:
				v = ansatz_sparse_grad_directed(Vlist[k], v, L, Z, perms_j)
		dVi = ansatz_sparse_grad(Vlist[i], L, v, w, perms)
		dVlist[i] = dVi if unprojected else project_unitary_tangent(Vlist[i], dVi)

	i = k
	perms = perms_extended[i]
	for j in list(range(i+1, len(Vlist)))[::-1]:
		for perm in perms_extended[j][::-1]:
			Uv = applyG_block_state(Vlist[j].conj().T, Uv, L, perm)
	for j in range(i):
		for perm in perms_extended[j]:
			state = applyG_block_state(Vlist[j], state, L, perm)
	G = ansatz_sparse_hess_single_layer(Vlist[k], L, Z, state, Uv, perms)
	# Projection.
	if not unprojected:
		V = Vlist[k]
		G = project_unitary_tangent(V, G)
		grad = ansatz_sparse_grad(V, L, state, Uv, perms)
		G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
		if not np.allclose(Z, project_unitary_tangent(V, Z)):
			G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
	dVlist[k] = G

	return dVlist







