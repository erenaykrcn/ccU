import numpy as np
from ansatz_2D import ansatz_2D, ansatz_2D_grad_vector
from hessian_2D import ansatz_2D_hessian_matrix
from rqcopt.trust_region import riemannian_trust_region_optimize
from utils_2D import (polar_decomp, real_to_antisymm, real_to_skew, 
    separability_penalty, grad_separability_penalty, reduce_list)



def err(vlist, U, L, perms):
    f_base = -np.trace(U.conj().T @ ansatz_2D(vlist, L, perms)).real
    return f_base


def optimize(L, U, Vlist_start, perms, perms_reduced=None, control_layers=[], **kwargs):
    n = len(Vlist_start)
    indices = []
    for i in range(len(Vlist_start)):
        if i not in control_layers:
            indices.append(i)

    def f(vlist):
        if len(control_layers)==0:
            return -np.trace(U.conj().T @ ansatz_2D(vlist, L, perms)).real
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)
            return -np.trace(U @ ansatz_2D(vlist, L, perms)).real +\
                -np.trace(U.conj().T @ ansatz_2D(vlist_reduced, L, perms_reduced)).real

    def gradfunc(vlist):
        if len(control_layers)==0:
            return -ansatz_2D_grad_vector(vlist, L, U, perms)
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)
            gradfunc1 = -ansatz_2D_grad_vector(vlist, L, U.conj().T, perms, flatten=False)
            gradfunc2 = -ansatz_2D_grad_vector(vlist_reduced, L, U, perms_reduced, flatten=False)
            for i, index in enumerate(indices):
                gradfunc1[index] += gradfunc2[i]
        return gradfunc1.reshape(-1)

    def hessfunc(vlist):
        if len(control_layers)==0:
            return -ansatz_2D_hessian_matrix(vlist, L, U, perms)
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)
            hessfunc1 = -ansatz_2D_hessian_matrix(vlist, L, U.conj().T, perms, flatten=False)
            hessfunc2 = -ansatz_2D_hessian_matrix(vlist_reduced, L, U, perms_reduced, flatten=False)
            for i, index in enumerate(indices):
                for j, index_ in enumerate(indices):
                    hessfunc1[index, :, index_, :] += hessfunc2[i, :, j, :]
            return hessfunc1.reshape((n * 16, n * 16))

    def errfunc(vlist):
        if len(control_layers)==0:
            return np.linalg.norm(ansatz_2D(vlist, L, perms) - U, ord=2)
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)
            return (np.linalg.norm(ansatz_2D(vlist, L, perms) - U.conj().T, ord=2) +\
                np.linalg.norm(ansatz_2D(vlist_reduced, L, perms_reduced) - U, ord=2))/2


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
