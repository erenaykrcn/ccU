import numpy as np
from ansatz import ansatz, ansatz_grad_vector
from hessian import ansatz_hessian_matrix
#from rqcopt.trust_region import riemannian_trust_region_optimize
from utils import (polar_decomp, real_to_antisymm, real_to_skew, 
    separability_penalty, grad_separability_penalty, reduce_list)
import scipy
import matplotlib.pyplot as plt
import warnings
from precision import HP as prec


def optimize(L: int, U, eta, gamma, Vlist_start, perms, penalty_weight=0, **kwargs):
    # Here, eta refers to the number of layers in between each controlling layer.
    # Gamma refers to the number of controlling layers. If controlling layers are
    # inserted, we expect the time evolution direction to be reverted.
    n = len(Vlist_start)
    U_back = U.conj().T if gamma>1 else U
    indices = []
    for i in range(gamma):
        indices += list(range(eta*i+1+i, eta*(i+1)+1+i))
    t = kwargs.get("t", "?")

    # target function
    def f(vlist):
        if gamma>1:
            f_base = -np.trace(U.conj().T @ ansatz(reduce_list(
                 vlist, gamma, eta), L, reduce_list(perms, gamma, eta))).real -\
            np.trace(U_back.conj().T @ ansatz(vlist, L, perms)).real
        else:
            f_base = -np.trace(U.conj().T @ ansatz(vlist, L, perms)).real

        if penalty_weight != 0:
            penalty = 0
            for i in range(len(vlist)):
                if i not in indices:  # indices used for U
                    penalty += separability_penalty(vlist[i])
            return f_base + penalty_weight * penalty
        else:
            return f_base


    def gradfunc(vlist):
        gradfunc1 = -ansatz_grad_vector(vlist, L, U_back, perms, flatten=False).real
        magn = [np.linalg.norm(G, ord=2) for G in gradfunc1]

        if gamma>1:
            gradfunc2 = -ansatz_grad_vector(reduce_list(
                vlist, gamma, eta), L, U, reduce_list(perms, gamma, eta), flatten=False).real
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
        
        hessfunc1 = -ansatz_hessian_matrix(vlist, L, U_back, perms, flatten=False).real
        """H = hessfunc1.reshape((n * 16, n * 16))
        H_sym = 0.5 * (H + H.conj().T)
        evals = np.linalg.eigvalsh(H_sym)
        tol = 1e-12
        num_neg  = np.sum(evals < -tol)
        num_pos  = np.sum(evals >  tol)
        num_zero = np.sum(np.abs(evals) <= tol)
        print("min eigenvalue:", evals[0])
        print("max eigenvalue:", evals[-1])
        print("num_neg, num_zero, num_pos:", num_neg, num_zero, num_pos)

        if num_neg > 0 and num_pos > 0:
            print("=> Hessian is indefinite: this point is a saddle (or very close).")
        elif num_neg == 0 and num_pos > 0:
            print("=> Hessian is positive semidefinite: local min / flat directions.")
        elif num_pos == 0 and num_neg > 0:
            print("=> Hessian is negative semidefinite: local max / flat directions.")
        else:
            print("=> Hessian is (numerically) very flat / singular.")"""
        #return np.zeros((n * 16, n * 16))

        if gamma>1:
            hessfunc2 = -ansatz_hessian_matrix(reduce_list(
                vlist, gamma, eta), L, U, reduce_list(perms, gamma, eta), flatten=False)
            for i, index in enumerate(indices):
                for j, index_ in enumerate(indices):
                    hessfunc1[index, :, index_, :] += hessfunc2[i, :, j, :]
        return hessfunc1.reshape((n * 16, n * 16))

    # quantify error by spectral norm
    def errfunc(vlist): 
        """if gamma>1:
            err =  np.linalg.norm(
            ansatz(reduce_list(
                vlist, gamma, eta), L, reduce_list(perms, gamma, eta)) - U, ord=2) + np.linalg.norm(
            ansatz(vlist, L, perms) - U_back, ord=2)
            
        else:
            M = np.asarray(ansatz(vlist, L, perms), dtype=np.complex128)
            err = np.linalg.norm(M - U, ord=2)
        return err"""
        err = f(vlist)
        print("err: ", 1+err/2**(L+1))
        with open(f"./_BRICKWALL_log_layers{n}_t{t}.txt", "a") as file:
            file.write(f"Error {1+err/2**(L+1)}\n")
        return 1+err/2**(L+1)


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

    """# visualize optimization progress
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
    print("Last f: ", f_iter[-1])"""
    # save results to disk
    f_iter = np.array(f_iter)
    err_iter = np.array(err_iter)
    
    return Vlist, f_iter, err_iter




def riemannian_trust_region_optimize(f, retract, gradfunc, hessfunc, x_init, **kwargs):
    """
    Optimization via the Riemannian trust-region (RTR) algorithm.

    Reference:
        Algorithm 10 in:
        P.-A. Absil, R. Mahony, Rodolphe Sepulchre
        Optimization Algorithms on Matrix Manifolds
        Princeton University Press (2008)
    """
    rho_trust   = kwargs.get("rho_trust", 0.125)
    radius_init = kwargs.get("radius_init", 0.01)
    maxradius   = kwargs.get("maxradius",   0.1)
    tol = kwargs.get("conv_tol",   1e-10)
    print("Conv tol. ", tol)
    niter       = kwargs.get("niter", 20)
    gfunc       = kwargs.get("gfunc", None)
    # transfer keyword arguments for truncated_cg
    tcg_kwargs = {}
    for key in ["maxiter", "abstol", "reltol"]:
        if ("tcg_" + key) in kwargs.keys():
            tcg_kwargs[key] = kwargs["tcg_" + key]
    assert 0 <= rho_trust < 0.25
    x = x_init
    radius = radius_init
    f_iter = []
    g_iter = []
    if gfunc is not None:
        g_iter.append(gfunc(x))

    
    patience = 5        # number of consecutive small changes required
    conv_counter = 0
    for k in range(niter):
        grad = gradfunc(x)
        hess = hessfunc(x)
        eta, on_boundary = truncated_cg(grad, hess, radius, **tcg_kwargs)
        x_next = retract(x, eta)
        fx = f(x)
        f_iter.append(fx)
        # Eq. (7.7)
        rho = (f(x_next) - fx) / (np.dot(grad, eta) + 0.5 * np.dot(eta, hess @ eta))
        if rho < 0.25:
            # reduce radius
            radius *= 0.25
            #radius = max(radius * 0.5, 1e-3)
        elif rho > 0.75 and on_boundary:
            # enlarge radius
            radius = min(2 * radius, maxradius)
        #print("Radius ", radius)
        if rho > rho_trust:
            x = x_next
        if gfunc is not None:
            g_iter.append(gfunc(x))

        # ---- convergence check block ----
        if len(g_iter) > 1:
            if abs(g_iter[-1] - g_iter[-2]) < tol:
                conv_counter += 1
            else:
                conv_counter = 0

            if conv_counter >= patience:
                #with open(f"./_optlog_12_L{n}_t{t}_log.txt", "a") as file:
                #    file.write(f"Converged at iteration {k}")
                print(f"Converged at iteration {k}")
                break
    return x, f_iter, g_iter


def truncated_cg(grad, hess, radius, **kwargs):
    """
    Truncated CG (tCG) method for the trust-region subproblem:
        minimize   <grad, z> + 1/2 <z, H z>
        subject to <z, z> <= radius^2

    References:
      - Algorithm 11 in:
        P.-A. Absil, R. Mahony, Rodolphe Sepulchre
        Optimization Algorithms on Matrix Manifolds
        Princeton University Press (2008)
      - Trond Steihaug
        The conjugate gradient method and trust regions in large scale optimization
        SIAM Journal on Numerical Analysis 20, 626-637 (1983)
    """
    maxiter = kwargs.get("maxiter", 2 * len(grad))
    abstol  = kwargs.get("abstol", 1e-8)
    reltol  = kwargs.get("reltol", 1e-6)
    r = grad.copy()
    rsq = np.dot(r, r)
    stoptol = max(abstol, reltol * np.sqrt(rsq))
    z = np.zeros_like(r)
    d = -r
    for j in range(maxiter):
        Hd = hess @ d
        dHd = np.dot(d, Hd)
        t = _move_to_boundary(z, d, radius)
        alpha = rsq / dHd
        if dHd <= 0 or alpha > t:
            # return with move to boundary
            return z + t*d, True
        # update iterates
        r += alpha * Hd
        z += alpha * d
        rsq_next = np.dot(r, r)
        if np.sqrt(rsq_next) <= stoptol:
            # early stopping
            return z, False
        beta = rsq_next / rsq
        d = -r + beta * d
        rsq = rsq_next
    # maxiter reached
    return z, False


def _move_to_boundary(b, d, radius):
    """
    Move to the unit ball boundary by solving
    || b + t*d || == radius
    for t with t > 0.
    """
    dsq = np.dot(d, d)
    if dsq == 0:
        warnings.warn("input vector 'd' is zero")
        return b
    p = np.dot(b, d) / dsq
    q = (np.dot(b, b) - radius**2) / dsq
    t = solve_quadratic_equation(p, q)[1]
    if t < 0:
        warnings.warn("encountered t < 0")
    return t


def solve_quadratic_equation(p, q):
    """
    Compute the two solutions of the quadratic equation x^2 + 2 p x + q == 0.
    """
    if p**2 - q < 0:
        raise ValueError("require non-negative discriminant")
    if p == 0:
        x = np.sqrt(-q)
        return (-x, x)
    x1 = -(p + np.sign(p)*np.sqrt(p**2 - q))
    x2 = q / x1
    return tuple(sorted((x1, x2)))




