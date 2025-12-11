import numpy as np
from ansatz import ansatz_grad, ansatz_grad_vector
from utils import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm,
	project_unitary_tangent, real_to_antisymm
	)


def ansatz_hessian_matrix(Vlist, L, cU, perms, flatten=True, unprojected=False):
	
	eta = len(Vlist)
	Hess = np.zeros((eta, 16, eta, 16), dtype=np.longdouble)

	for k in range(eta):
		for j in range(16):
			if unprojected:
				Z = np.zeros((4, 4), dtype=np.complex128)
				Z.flat[j] = 1.0
			else:
				Z = np.zeros(16)
				Z[j] = 1
				Z = real_to_antisymm(np.reshape(Z, (4, 4)))

			dVZj = ansatz_hess(Vlist, L, Vlist[k] @ Z, k, cU, perms, unprojected=unprojected)
			
			for i in range(eta):
				Hess[i, :, k, j] = dVZj[i].reshape(-1) if unprojected else \
						antisymm_to_real(antisymm( Vlist[i].conj().T @ dVZj[i] )).reshape(-1)

	if flatten:
		return Hess.reshape((eta*16, eta*16))
	else:
		return Hess


def ansatz_hess_single_layer(V, L, Z, U_tilde_, perm):
	
	G = np.zeros_like(V, dtype=np.complex128)
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
				# Partial trace interprets the qubit l as the first qubit of 
				# the resulting two qubit gate. We need to fix that.
				SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
				T = SWAP @ T @ SWAP
			G += T
	return G


def ansatz_grad_directed(V, U_tilde, L, Z, perm):
	assert V.shape == (4, 4)
	assert Z.shape == (4, 4)
	#assert L % 2 == 0

	G = np.zeros((2**L, 2**L), dtype=np.complex128).reshape([2]*2*L)
	for i in range(len(perm)//2):
		U = np.eye(2**L).reshape([2]*2*L)
		k, l = (perm[2*i], perm[2*i+1])

		for j in range(i):
			k_, l_ = (perm[2*j], perm[2*j+1])
			U = applyG_tensor(V, U, k_, l_) 

		U = applyG_tensor(Z, U, k, l)

		for j in list(range(i+1, len(perm)//2))[::-1]:
			k_, l_ = (perm[2*j], perm[2*j+1])
			U = applyG_tensor(V, U, k_, l_)
		G += U
	U_tilde = G.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)
	return U_tilde.reshape([2]*2*L)


def ansatz_hess(Vlist, L, Z, k, cU, perms, unprojected=False):
	dVlist = [None for i in range(len(Vlist))]
	# i runs over Vlist, i>k
	for i in range(k+1, len(Vlist)):
		U_tilde = np.eye(2**L).reshape([2]*2*L)
		for j in range(i+1, len(Vlist)):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(k):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = ansatz_grad_directed(Vlist[k], U_tilde, L, Z, perms[k])
		for j in range(k+1, i):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		dVi = ansatz_grad(Vlist[i], L, U_tilde, perms[i]).conj().T
		dVlist[i] = dVi if unprojected else project_unitary_tangent(Vlist[i], dVi)

	# i runs over Vlist, k>i
	for i in range(k):
		U_tilde = np.eye(2**L).reshape([2]*2*L)
		for j in range(i+1, k):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = ansatz_grad_directed(Vlist[k], U_tilde, L, Z, perms[k])
		for j in range(k+1, len(Vlist)):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(i):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])

		dVi = ansatz_grad(Vlist[i], L, U_tilde, perms[i]).conj().T
		dVlist[i] = dVi if unprojected else project_unitary_tangent(Vlist[i], dVi)

	# i=k case.
	i = k
	U_tilde = np.eye(2**L).reshape([2]*2*L)
	for j in range(i+1, len(Vlist)):
		U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
	U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
	for j in range(i):
		U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
	G = ansatz_hess_single_layer(Vlist[k], L, Z, U_tilde, perms[k]).conj().T
	# Projection.
	if not unprojected:
		V = Vlist[k]
		G = project_unitary_tangent(V, G)
		grad = ansatz_grad(V, L, U_tilde, perms[k]).conj().T
		G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
		if not np.allclose(Z, project_unitary_tangent(V, Z)):
			G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
	dVlist[k] = G

	return dVlist

