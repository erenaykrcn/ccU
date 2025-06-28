import numpy as np
from ansatz_2D import ansatz_2D_grad, ansatz_2D_grad_vector
from utils_2D import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm,
	project_unitary_tangent, real_to_antisymm
	)


def ansatz_2D_hessian_matrix(Vlist, L, cU, perms, reps=1, flatten=True, unprojected=False):
	"""
	Construct the Hessian matrix.
	"""
	eta = len(Vlist)
	Hess = np.zeros((eta, 16, eta, 16))

	for k in range(eta):
		for j in range(16):
			if unprojected:
				Z = np.zeros((4, 4), dtype=complex)
				Z.flat[j] = 1.0
			else:
				Z = np.zeros(16)
				Z[j] = 1
				Z = real_to_antisymm(np.reshape(Z, (4, 4)))
			dVZj = ansatz_2D_hess(Vlist, L, Vlist[k] @ Z, k, cU, perms, reps=reps, unprojected=unprojected)
			for i in range(eta):
				Hess[i, :, k, j] = dVZj[i].reshape(-1) if unprojected else \
						antisymm_to_real(antisymm( Vlist[i].conj().T @ dVZj[i] )).reshape(-1)
	if flatten:
		return Hess.reshape((eta*16, eta*16))
	else:
		return Hess


def ansatz_hess_single_layer(V, L, Z, U_tilde_, perm):
	G = np.zeros_like(V, dtype=complex)
	for i in range(len(perm)//2):
		k, l = (perm[2*i], perm[2*i+1])
		for z in range(len(perm)//2):
			U_tilde = U_tilde_.copy()
			if z==i:
				continue
			for j in range(i):
				k_, l_ = (perm[2*j], perm[2*j+1])
				if z==j:
					U_tilde = applyG_tensor(Z, U_tilde, k_, l_)
				else:
					U_tilde = applyG_tensor(V, U_tilde, k_, l_)

			for j in list(range(i+1, len(perm)//2))[::-1]:
				k_, l_ = (perm[2*j], perm[2*j+1])
				if z==j:
					U_tilde = applyG_tensor(Z, U_tilde, k_, l_) # Reversed left to right multiplication here!
				else:
					U_tilde = applyG_tensor(V, U_tilde, k_, l_) # Reversed left to right multiplication here!

			# Take partial trace wrt all qubits but k, l.
			T = partial_trace_keep(U_tilde.reshape(2**L, 2**L), [k, l], L)
			if k>l:
				SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
				T = SWAP @ T @ SWAP
			G += T
	return G


def ansatz_2D_hess_single_layer(V, L, Z, U_tilde_, perms):
	G = np.zeros_like(V, dtype=complex)
	for dashed_index, perm in enumerate(perms):
		for z_index in range(len(perms)):
			U = np.eye(2**L).reshape([2]*2*L)
			for j in range(dashed_index+1, len(perms)):
				if j!= z_index:
					U = applyG_block_tensor(V, U, L, perms[j])
				else:
					U = ansatz_2D_grad_directed(V, U, L, Z, [perms[j]])
			U = (U_tilde_.reshape(2**L, 2**L) @ U.reshape(2**L, 2**L)).reshape([2]*2*L)
			for j in range(dashed_index):
				if j!= z_index:
					U = applyG_block_tensor(V, U, L, perms[j])
				else:
					U = ansatz_2D_grad_directed(V, U, L, Z, [perms[j]])

			if z_index != dashed_index:
				G += ansatz_2D_grad(V, L, U, [perm])
			else:
				G += ansatz_hess_single_layer(V, L, Z, U, perm)
	return G



def ansatz_2D_grad_directed(V, U_tilde, L, Z, perms):
	G = np.zeros((2**L, 2**L), dtype=complex).reshape([2]*2*L)
	for _, perm in enumerate(perms):
		for i in range(len(perm)//2):
			U_working = U_tilde.copy()
			for j in range(_):
				U_working = applyG_block_tensor(V, U_working, L, perms[j])

			k, l = (perm[2*i], perm[2*i+1])

			for j in range(i):
				k_, l_ = (perm[2*j], perm[2*j+1])
				U_working = applyG_tensor(V, U_working, k_, l_) 

			U_working = applyG_tensor(Z, U_working, k, l)

			for j in list(range(i+1, len(perm)//2))[::-1]:
				k_, l_ = (perm[2*j], perm[2*j+1])
				U_working = applyG_tensor(V, U_working, k_, l_)

			for j in range(_+1, len(perms)):
				U_working = applyG_block_tensor(V, U_working, L, perms[j])
			G += U_working
	return G


def ansatz_2D_hess(Vlist, L, Z, k, cU, perms_extended, reps=1, unprojected=False):
	dVlist = [None for i in range(len(Vlist))]

	for i in range(len(Vlist)):
		perms = perms_extended[reps*i:reps*(i+1)]
		if i==k:
			continue

		U_tilde = np.eye(2**L).reshape([2]*2*L)
		for j in range(i+1, len(Vlist)):
			perms_j = perms_extended[reps*j:reps*(j+1)]
			if k!=j:
				for perm in perms_j:
					U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perm)
			else:
				U_tilde = ansatz_2D_grad_directed(Vlist[k], U_tilde, L, Z, perms_j)
		U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(i):
			perms_j = perms_extended[reps*j:reps*(j+1)]
			if k!=j:
				for perm in perms_j:
					U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perm)
			else:
				U_tilde = ansatz_2D_grad_directed(Vlist[k], U_tilde, L, Z, perms_j)

		dVi = ansatz_2D_grad(Vlist[i], L, U_tilde, perms).conj().T
		dVlist[i] = dVi if unprojected else project_unitary_tangent(Vlist[i], dVi)

	i = k
	perms = perms_extended[reps*i:reps*(i+1)]
	U_tilde = np.eye(2**L).reshape([2]*2*L)
	for j in range(i+1, len(Vlist)):
		for perm in perms_extended[reps*j:reps*(j+1)]:
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perm)
	U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
	for j in range(i):
		for perm in perms_extended[reps*j:reps*(j+1)]:
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perm)
	G = ansatz_2D_hess_single_layer(Vlist[k], L, Z, U_tilde, perms).conj().T
	# Projection.
	if not unprojected:
		V = Vlist[k]
		G = project_unitary_tangent(V, G)
		grad = ansatz_2D_grad(V, L, U_tilde, perms).conj().T
		G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
		if not np.allclose(Z, project_unitary_tangent(V, Z)):
			G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
	dVlist[k] = G

	return dVlist







