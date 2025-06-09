import numpy as np
from ansatz import ansatz, ansatz_grad_vector
from hessian import ansatz_hessian_matrix
from rqcopt.trust_region import riemannian_trust_region_optimize
from utils import (polar_decomp, real_to_antisymm, real_to_skew, 
    separability_penalty, grad_separability_penalty, reduce_list)
import scipy
import matplotlib.pyplot as plt


def optimize(L: int, U, eta, gamma, Vlist_start, perms, penalty_weight=0, **kwargs):
    # Here, eta refers to the number of layers in between each controlling layer.
    # Gamma refers to the number of controlling layers. If controlling layers are
    # inserted, we expect the time evolution direction to be reverted.
    n = len(Vlist_start)
    U_back = U.conj().T
    indices = []
    for i in range(gamma):
        indices += list(range(eta*i+1+i, eta*(i+1)+1+i))

    # target function
    def f(vlist):
        f_base = -np.trace(U.conj().T @ ansatz(reduce_list(
             vlist, gamma, eta), L, reduce_list(perms, gamma, eta))).real -\
        np.trace(U_back.conj().T @ ansatz(vlist, L, perms)).real
        

        if penalty_weight != 0:
            penalty = 0
            for i in range(len(vlist)):
                if i not in indices:  # indices used for U
                    penalty += separability_penalty(vlist[i])
            return f_base + penalty_weight * penalty
        else:
            return f_base


    def gradfunc(vlist):
        gradfunc1 = -ansatz_grad_vector(vlist, L, U_back, perms, flatten=False)
        gradfunc2 = -ansatz_grad_vector(reduce_list(
            vlist, gamma, eta), L, U, reduce_list(perms, gamma, eta), flatten=False)
        for i, index in enumerate(indices):
            gradfunc1[index] += gradfunc2[i]

        if penalty_weight != 0:
            # Add penalty gradient
            for i in range(len(vlist)):
                if i not in indices:
                    grad_penalty = grad_separability_penalty(vlist[i])
                    gradfunc1[i] += penalty_weight * grad_penalty

        return gradfunc1.reshape(-1)

    def hessfunc(vlist):
        hessfunc1 = -ansatz_hessian_matrix(vlist, L, U_back, perms, flatten=False)
        hessfunc2 = -ansatz_hessian_matrix(reduce_list(
            vlist, gamma, eta), L, U, reduce_list(perms, gamma, eta), flatten=False)
        for i, index in enumerate(indices):
            for j, index_ in enumerate(indices):
                hessfunc1[index, :, index_, :] += hessfunc2[i, :, j, :]
        return hessfunc1.reshape((n * 16, n * 16))

    # quantify error by spectral norm
    errfunc = lambda vlist: np.linalg.norm(
        ansatz(reduce_list(
            vlist, gamma, eta), L, reduce_list(perms, gamma, eta)) - U, ord=2) + np.linalg.norm(
        ansatz(vlist, L, perms) - U_back, ord=2)
    kwargs["gfunc"] = errfunc
    # perform optimization
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


def dynamics_opt(hamil, t, eta, gamma, bootstrap: bool, Vlist_start=None, coeffs_start=[], perms=None, penalty_weight=1, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the time evolution governed by a Hamiltonian.
    """
    
    # reference global unitary
    L = int(np.log(hamil.shape[0])/np.log(2))
    expiH = scipy.linalg.expm(-1j*hamil*t)
    nlayers = len(Vlist_start)

    # unitaries used as starting point for optimization
    if bootstrap:
        # load optimized unitaries for nlayers - 2 from disk
        with h5py.File(f"dynamics_opt_L{L}_t{t}_layers{nlayers-2}_gamma{gamma}_eta{eta}.hdf5", "r") as f:
            Vlist_start = f["Vlist"][:]
            assert Vlist_start.shape[0] == nlayers - 2
        # pad identity matrices
        id4 = np.identity(4).reshape((1, 4, 4))
        Vlist_start = np.concatenate((id4, Vlist_start, id4), axis=0)
        assert Vlist_start.shape[0] == nlayers
        perms = [[i for i in range(L)] if i % 2 == 1 else np.roll(range(L), 1) for i in range(len(Vlist_start))]
        
    if perms is None:
        perms = [[i for i in range(L)] if i % 2 == 0 else np.roll(range(L), 1) for i in range(nlayers)]
        
    # perform optimization
    Vlist, f_iter, err_iter = optimize(L, expiH, eta, gamma, Vlist_start, perms, penalty_weight=penalty_weight, **kwargs)

    # visualize optimization progress
    print(f"err_iter before: {err_iter[0]}")
    print(f"err_iter after {len(err_iter)-1} iterations: {err_iter[-1]}")
    plt.semilogy(range(len(err_iter)), err_iter)
    plt.xlabel("iteration")
    plt.ylabel("spectral norm error")
    plt.title(f"optimization progress for a quantum circuit with {len(Vlist)} layers")
    plt.show()
    # rescaled and shifted target function
    plt.semilogy(range(len(f_iter)), 1 + np.array(f_iter) / 2**(L+1))
    plt.xlabel("iteration")
    plt.ylabel(r"$1 + f(\mathrm{Vlist})/2^L$")
    plt.title(f"optimization target function for a quantum circuit with {len(Vlist)} layers")
    plt.show()
    print("Last f: ", f_iter[-1])
    # save results to disk
    f_iter = np.array(f_iter)
    err_iter = np.array(err_iter)
    
    return Vlist, f_iter, err_iter



