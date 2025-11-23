import scipy
import numpy as np
import scipy.sparse as sp 
import qutip
import qiskit
from qiskit import Aer, execute, transpile
from quimb.tensor.tensor_arbgeom_tebd import LocalHamGen, TEBDGen
import gc

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def compute_overlap(peps1, peps2, chi_overlap):
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


def applyG_PEPS(G, peps, L, k, l, max_bond_dim, chi_overlap=10):
    """
    Applies 2-qubit gate G to qubits k and l of a state vector of length 2^L.
    """
    if k > l:
        k, l = l, k
        SWAP = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
        G = SWAP @ G @ SWAP

    edges = [(k, l), ]
    H2 = {edge: G for edge in edges}
    ham = LocalHamGen(H2=H2, H1=None)
    tebd = TEBDGen(peps, ham=ham, D=max_bond_dim)
    tebd.sweep(tau=-1)
    peps = tebd.state
    #peps /= np.sqrt(compute_overlap(peps, peps, chi_overlap))

    del tebd, ham
    gc.collect()

    return peps


def applyG_block_PEPS(G, state, L, perm, max_bond_dim, chi_overlap=10):
    """
    Applies 2-qubit gate G to each (k, l) in perm on a 2^L state vector.
    """
    assert len(perm) % 2 == 0

    edges = [(perm[2*j], perm[2*j+1]) for j in range(len(perm)//2)]
    H2 = {edge: G for edge in edges}
    ham = LocalHamGen(H2=H2, H1=None)
    tebd = TEBDGen(state, ham=ham, D=max_bond_dim)
    tebd.sweep(tau=-1)
    peps = tebd.state
    #peps /= np.sqrt(compute_overlap(peps, peps, chi_overlap))

    del tebd, ham
    gc.collect()

    return peps


def partial_inner_product(peps1, peps2, k, l, chi_overlap):
    phys_k = f'k{k}'
    phys_l = f'k{l}'
    bra = peps1.conj(mangle_inner=True)
    bra_k = f'{phys_k}_BRA'
    bra_l = f'{phys_l}_BRA'
    bra = bra.reindex({phys_k: bra_k, phys_l: bra_l})
    ov_tn = bra & peps2
    out_inds = (bra_k, bra_l, phys_k, phys_l)

    T = ov_tn.contract_compressed(
            output_inds=out_inds,
            optimize="hyper-compressed",
            max_bond=chi_overlap,
            cutoff=1e-10,
    )

    A = np.asarray(T.data).reshape(2, 2, 2, 2)
    M = A.reshape(4, 4)
    return M


def antisymm_to_real(w):
	return w.real + w.imag


def antisymm(w):
	return 0.5 * (w - w.conj().T)


def symm(w):
	return 0.5 * (w + w.conj().T)

def project_unitary_tangent(u, z):
    return z - u @ symm(u.conj().T @ z)

def real_to_antisymm(r):
	return 0.5*(r - r.T) + 0.5j*(r + r.T)


def polar_decomp(a):
	u, s, vh = np.linalg.svd(a)
	return u @ vh, (vh.conj().T * s) @ vh

def real_to_skew(r, n: int):
    """
    Map a real vector to a skew-symmetric matrix containing the vector entries in its upper-triangular part.
    """
    if len(r) != n * (n - 1) // 2:
        raise ValueError("length of input vector does not match matrix dimension")
    w = np.zeros((n, n))
    # sqrt(2) factor to preserve inner products
    w[np.triu_indices(n, k=1)] = r / np.sqrt(2)
    w -= w.T
    return w


def skew_to_real(w):
    """
    Map a real skew-symmetric matrix to a real vector containing the upper-triangular entries.
    """
    # sqrt(2) factor to preserve inner products
    return np.sqrt(2) * w[np.triu_indices(len(w), k=1)]

