import numpy as np
from ansatz_sparse import ansatz_sparse, ansatz_sparse_grad_vector
from hessian_sparse import ansatz_sparse_hessian_matrix
from rqcopt.trust_region import riemannian_trust_region_optimize
from utils_sparse import (polar_decomp, real_to_antisymm, real_to_skew, 
    separability_penalty, grad_separability_penalty, reduce_list)
from qiskit.quantum_info import random_statevector
from scipy.sparse.linalg import expm_multiply


def err(vlist, Uv, v, L, perms):
    return -np.vdot(Uv, ansatz_sparse(vlist, L, perms, v)).real


def optimize(L, hamil, t, Vlist_start, perms, perms_reduced=None, control_layers=[], rS=1, **kwargs):
    n = len(Vlist_start)
    indices = []
    for i in range(len(Vlist_start)):
        if i not in control_layers:
            indices.append(i)
    random_svs = [random_statevector(2**L).data for i in range(rS)]

    def f(vlist):
        if len(control_layers)==0:
            s = 0
            for v in random_svs:
                s += -np.vdot(expm_multiply(-1j * t * hamil, v), ansatz_sparse(vlist, L, perms, v)).real
            return s
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)
            s = 0
            for v in random_svs:
                s += -np.vdot(expm_multiply(1j * t * hamil, v), ansatz_sparse(vlist, L, perms, v)).real
                s += -np.vdot(expm_multiply(-1j * t * hamil, v), ansatz_sparse(vlist_reduced, L, perms_reduced, v)).real
            return s

    def gradfunc(vlist):
        if len(control_layers)==0:
            g = -ansatz_sparse_grad_vector(vlist, L, expm_multiply(-1j * t * hamil, random_svs[0]), random_svs[0], perms)
            for v in random_svs[1:]:
                g += -ansatz_sparse_grad_vector(vlist, L, expm_multiply(-1j * t * hamil, v), v, perms)
            return g
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)

            gradfunc1 = -ansatz_sparse_grad_vector(vlist, L, expm_multiply(1j * t * hamil, 
                random_svs[0]), random_svs[0], perms, flatten=False)
            for v in random_svs[1:]:
                gradfunc1 += -ansatz_sparse_grad_vector(vlist, L, expm_multiply(1j * t * hamil, v), v, perms, flatten=False)
            gradfunc2 = -ansatz_sparse_grad_vector(vlist_reduced, L, expm_multiply(-1j * t * hamil, 
                random_svs[0]), random_svs[0], perms_reduced, flatten=False)
            for v in random_svs[1:]:
                gradfunc2 += -ansatz_sparse_grad_vector(vlist_reduced, L, expm_multiply(-1j * t * hamil, 
                    v), v, perms_reduced, flatten=False)

            for i, index in enumerate(indices):
                gradfunc1[index] += gradfunc2[i]
        return gradfunc1.reshape(-1)

    def hessfunc(vlist):
        if len(control_layers)==0:
            h = -ansatz_sparse_hessian_matrix(vlist, L, expm_multiply(-1j * t * hamil, random_svs[0]), random_svs[0], perms)
            for v in random_svs[1:]:
                h += -ansatz_sparse_hessian_matrix(vlist, L, expm_multiply(-1j * t * hamil, v), v, perms)
            return h
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)

            hessfunc1 = -ansatz_sparse_hessian_matrix(vlist, L, expm_multiply(1j * t * hamil, 
                random_svs[0]), random_svs[0], perms, flatten=False)
            for v in random_svs[1:]:
                hessfunc1 += -ansatz_sparse_hessian_matrix(vlist, L, expm_multiply(1j * t * hamil, 
                    v), v, perms, flatten=False)

            hessfunc2 = -ansatz_sparse_hessian_matrix(vlist_reduced, L, expm_multiply(-1j * t * hamil, 
                random_svs[0]), random_svs[0], perms_reduced, flatten=False)
            for v in random_svs[1:]:
                hessfunc2 += -ansatz_sparse_hessian_matrix(vlist_reduced, L, expm_multiply(-1j * t * hamil, 
                    v), v, perms_reduced, flatten=False)

            for i, index in enumerate(indices):
                for j, index_ in enumerate(indices):
                    hessfunc1[index, :, index_, :] += hessfunc2[i, :, j, :]
            return hessfunc1.reshape((n * 16, n * 16))

    def errfunc(vlist):
        if len(control_layers)==0:
            e = 0
            for v in random_svs:
                e += np.linalg.norm(ansatz_sparse(vlist, L, perms, v) - expm_multiply(-1j * t * hamil, v), ord=2)
            print("Current error: ", e/rS)
            return e/rS
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)

            e = 0
            for v in random_svs:
                e += np.linalg.norm(ansatz_sparse(vlist, L, perms, v) - expm_multiply(1j * t * hamil, v), ord=2) 
                e += np.linalg.norm(ansatz_sparse(vlist_reduced, L, perms_reduced, v) - expm_multiply(-1j * t * hamil, v), ord=2)
                
            with open("log.txt", "a") as file:
                file.write(f"Error {e/(2*rS)}\n")
            print("Current error: ", e/(2*rS))
            return e/(2*rS)

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
