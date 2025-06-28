import numpy as np
from ansatz_2D import ansatz_2D, ansatz_2D_grad_vector
from hessian_2D import ansatz_2D_hessian_matrix
from rqcopt.trust_region import riemannian_trust_region_optimize
from utils_2D import (polar_decomp, real_to_antisymm, real_to_skew, 
    separability_penalty, grad_separability_penalty, reduce_list)



def err(vlist, U, L, perms, reps=1):
    f_base = -np.trace(U.conj().T @ ansatz_2D(vlist, L, perms, reps=reps)).real
    return f_base


def optimize(L, U, Vlist_start, perms, reps=1, **kwargs):
    # TODO: Add reduced list approach.

    n = len(Vlist_start)
    def f(vlist):
        return -np.trace(U.conj().T @ ansatz_2D(vlist, L, perms, reps=reps)).real

    def gradfunc(vlist):
        gradfunc1 = -ansatz_2D_grad_vector(vlist, L, U, perms, reps=reps, flatten=False)
        return gradfunc1.reshape(-1)

    def hessfunc(vlist):
        hessfunc1 = -ansatz_2D_hessian_matrix(vlist, L, U, perms, reps=reps, flatten=False)
        return hessfunc1.reshape((n * 16, n * 16))

    def errfunc(vlist):
        return np.linalg.norm(ansatz_2D(vlist, L, perms, reps=reps) - U, ord=2)


    kwargs["gfunc"] = errfunc
    Vlist, f_iter, err_iter = riemannian_trust_region_optimize(
        f, retract_unitary_list, gradfunc, hessfunc, np.stack(Vlist_start), **kwargs)
    return Vlist, f_iter, err_iter


def retract_unitary_list(vlist, eta):
    """
    Retraction for unitary matrices, with tangent direction represented as anti-symmetric matrices.
    """
    n = len(vlist)
    eta = np.reshape(eta, (n, 4, 4))
    dvlist = [vlist[j] @ real_to_antisymm(eta[j]) for j in range(n)]
    return np.stack([polar_decomp(vlist[j] + dvlist[j])[0] for j in range(n)])
