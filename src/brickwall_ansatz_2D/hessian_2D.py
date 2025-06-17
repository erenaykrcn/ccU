import numpy as np
from ansatz_2D import ansatz_2D_grad, ansatz_2D_grad_vector
from utils_2D import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm,
	project_unitary_tangent, real_to_antisymm
	)

#TODOs
def ansatz_2D_hessian_matrix(Vlist, L, cU, perms, flatten=True, unprojected=False):
	pass


def ansatz_2D_hess_single_layer(V, L, Z, U_tilde_, perm):
	pass


def ansatz_2D_grad_directed(V, U_tilde, L, Z, perm):
	pass


def ansatz_2D_hess(Vlist, L, Z, k, cU, perms, unprojected=False):
	pass