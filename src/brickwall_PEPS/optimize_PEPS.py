import numpy as np
from ansatz_PEPS import ansatz_PEPS, ansatz_PEPS_grad_vector
from rqcopt.trust_region import riemannian_trust_region_optimize
from utils_PEPS import (polar_decomp, real_to_antisymm, real_to_skew)
from qiskit.quantum_info import random_statevector
from scipy.sparse.linalg import expm_multiply



def optimize_PEPS(L, reference_states, initial_states, t, Vlist_start, perms, max_bond_dim, chi_overlap,
 perms_reduced=None, control_layers=[], log=False, reference_states_back=None,
 n_workers_1=1, n_workers_2=1, **kwargs):
    def compute_overlap(peps1, peps2):
        ov_tn = peps1.make_overlap(
            peps2,
            layer_tags=("KET", "BRA"),
        )
        overlap_approx = ov_tn.contract_compressed(
            optimize="hyper-compressed",  # preset strategy name understood via cotengra
            max_bond=chi_overlap,
            cutoff=1e-10,
            # leave strip_exponent=False (default) so we just get a scalar back
        )
        return overlap_approx

    n = len(Vlist_start)
    indices = []
    for i in range(len(Vlist_start)):
        if i not in control_layers:
            indices.append(i)

    def f(vlist):
        if len(control_layers)==0:
            s = 0
            for reference_state, initial_state in zip(reference_states, initial_states):
                peps_evolved = ansatz_PEPS(vlist, L, perms, initial_state, max_bond_dim)
                peps_evolved /= np.sqrt(compute_overlap(peps_evolved, peps_evolved))
                s += -np.abs(compute_overlap(reference_state, peps_evolved))
            return s
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)
            s = 0
            for reference_state, initial_state in zip(reference_states, initial_states):
                peps_evolved1 = ansatz_PEPS(vlist, L, perms, initial_state, max_bond_dim)
                peps_evolved1 /= np.sqrt(compute_overlap(peps_evolved1, peps_evolved1))

                peps_evolved2 = ansatz_PEPS(vlist_reduced, L, perms_reduced, initial_state, max_bond_dim)
                peps_evolved2 /= np.sqrt(compute_overlap(peps_evolved2, peps_evolved2))

                s += -np.abs(compute_overlap(reference_state_back, peps_evolved1))
                s += -np.abs(compute_overlap(reference_state, peps_evolved2))
            return s

    def gradfunc(vlist):
        if len(control_layers)==0:
            g = -ansatz_PEPS_grad_vector(vlist, L, reference_states[0], 
                initial_states[0], perms, max_bond_dim, chi_overlap, n_workers_1=n_workers_1, n_workers_2=n_workers_2)
            for reference_state, initial_state in zip(reference_states[1:], initial_states[1:]):
                g += -ansatz_PEPS_grad_vector(vlist, L, reference_state, initial_state, 
                    perms, max_bond_dim, chi_overlap, n_workers_1=n_workers_1, n_workers_2=n_workers_2)
            return g
        else:
            vlist_reduced = []
            for i, V in enumerate(vlist):
                if i not in control_layers:
                    vlist_reduced.append(V)

            gradfunc1 = -ansatz_PEPS_grad_vector(vlist, L, reference_states_back[0], initial_states[0], perms, max_bond_dim, chi_overlap, flatten=False, n_workers_1=n_workers_1, n_workers_2=n_workers_2)
            for reference_state, initial_state in zip(reference_states_back[1:], initial_states[1:]):
                gradfunc1 += -ansatz_PEPS_grad_vector(vlist, L, reference_state, initial_state, perms, max_bond_dim, chi_overlap, flatten=False, n_workers_1=n_workers_1, n_workers_2=n_workers_2)
            gradfunc2 = -ansatz_PEPS_grad_vector(vlist_reduced, L, reference_states[0], initial_states[0], perms_reduced, max_bond_dim, chi_overlap, flatten=False, n_workers_1=n_workers_1, n_workers_2=n_workers_2)
            for reference_state, initial_state in zip(reference_states[1:], initial_states[1:]):
                gradfunc2 += -ansatz_PEPS_grad_vector(vlist_reduced, L, reference_state, initial_state, 
                    perms_reduced, max_bond_dim, chi_overlap, flatten=False, n_workers_1=n_workers_1, n_workers_2=n_workers_2)

            for i, index in enumerate(indices):
                gradfunc1[index] += gradfunc2[i]
        return gradfunc1.reshape(-1)

    def hessfunc(vlist):
            return np.zeros((n * 16, n * 16))

    def errfunc(vlist):
        e = f(vlist)
        with open("log.txt", "a") as file:
            file.write(f"Error {e}\n")
        print("Current error: ", e)
        return e
            
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
