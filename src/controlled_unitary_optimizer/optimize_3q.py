import numpy as np
import sys
sys.path.append("../brickwall_ansatz")
from rqcopt.trust_region import riemannian_trust_region_optimize
from utils import polar_decomp, real_to_antisymm
from utils_3q import unflatten
from ansatz_3q import ansatz_3q, ansatz_grad_3q



def f(X, cU, perms, bare=False):
    if not bare:
        Xlist = unflatten(X, 3*(len(perms)+1), len(perms))
    else:
        Xlist = unflatten(X, 0, len(perms))
    return -np.trace(cU.conj().T @ ansatz_3q(Xlist, perms)).real

def optimize_3q(L: int, cU, Xlist_start, perms, **kwargs):
    eta = len(perms)
    bare = len(Xlist_start) == eta
    err = lambda x: f(x, cU, perms, bare=bare)
    gradfunc = lambda x: ansatz_grad_3q(cU, unflatten(x, 3*(len(perms)+1) if not bare else 0, len(perms)), perms)
    hessfunc = lambda x: np.zeros((((4*3*(eta+1)) if not bare else 0)+16*eta, ((4*3*(eta+1)) if not bare else 0)+16*eta))   
    
    #hessfunc = lambda glist: cU_hess_matrix(cU, glist, perms)
    retract_unitary_list = lambda x, nu: retract_unitary(x, nu, eta, bare=bare)
    
    # quantify error by spectral norm
    errfunc = lambda x: np.linalg.norm(ansatz_3q(unflatten(x, (3*(len(perms)+1) if not bare else 0), len(perms)), perms) - cU, ord=2)
    kwargs["gfunc"] = errfunc
    X_start = np.concatenate([x.reshape(-1) for x in Xlist_start])
    X, f_iter, err_iter = riemannian_trust_region_optimize(
        err, retract_unitary_list, gradfunc, hessfunc, X_start, **kwargs)
    Xlist = unflatten(X, (3*(len(perms)+1) if not bare else 0), len(perms))
    return Xlist, f_iter, err_iter

def retract_unitary(x, nu, eta, bare=False):
    vlist = unflatten(x, (3*(eta+1) if not bare else 0), eta)
    nu = unflatten(nu,  (3*(eta+1) if not bare else 0), eta)
    n = len(vlist)
    
    dvlist = [vlist[j] @ real_to_antisymm(nu[j]) for j in range(n)]
    flat_vec =  np.concatenate([polar_decomp(vlist[j] + dvlist[j])[0].reshape(-1) for j in range(n)])
    return flat_vec