import scipy
import numpy as np
import scipy.sparse as sp 
import qutip
import qiskit
from qiskit import Aer, execute, transpile

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def applyG_tensor(G, U_tensor, k, l):
    """
        Performs a 'left' multiplication of applying G
        two qubit gate to qubits k and l.
    """
    L = U_tensor.ndim // 2
    assert G.shape == (4, 4), "G must be a 2-qubit gate (4x4)"
    assert 0 <= k < L and 0 <= l < L and k != l

    # Ensure k < l for consistency
    if k > l:
        k, l = l, k
        SWAP = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
        G = SWAP @ G @ SWAP

    # Reshape G to tensor form
    G_tensor = G.reshape(2, 2, 2, 2)
    input_axes = [k, l]
    output_axes = [L + i for i in range(L)]

    # Perform contraction: G_{ab,cd} * U_{...a...b..., ...}
    U_tensor = np.tensordot(G_tensor, U_tensor, axes=([2, 3], input_axes))

    # Move axes back into original order
    new_axes = list(range(2))  # G's output legs
    insert_at = input_axes[0]
    remaining_axes = list(range(2 * L))
    for t in sorted(input_axes, reverse=True):
        del remaining_axes[t]
    for i, ax in enumerate(new_axes):
        remaining_axes.insert(insert_at + i, ax)

    U_tensor = np.moveaxis(U_tensor, range(2), input_axes)
    return U_tensor



def applyG_block_tensor(G, U_tensor, L, perm):
    """
    Applies the 2-qubit gate G to every (k, l) in `perm` (length 2n) 
    on the (2,)*2L tensor U_tensor.

    G is a (4, 4) matrix.
    """
    assert len(perm) % 2 == 0
    for j in range(len(perm) // 2):
        k = perm[2 * j]
        l = perm[2 * j + 1]
        U_tensor = applyG_tensor(G, U_tensor, k, l)
    return U_tensor



def applyG_state(G, state, L, k, l):
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

    # Reshape state into a tensor of shape (2,)*L
    state_tensor = state.reshape([2] * L)

    # Transpose to bring qubits k and l to front
    axes = [k, l] + [i for i in range(L) if i != k and i != l]
    inv_axes = np.argsort(axes)
    transposed = np.transpose(state_tensor, axes)

    # Reshape to (4, -1) so we can apply G
    transposed = transposed.reshape(4, -1)
    updated = G @ transposed

    # Reshape back
    updated = updated.reshape([2, 2] + [2] * (L - 2))
    updated = np.transpose(updated, inv_axes)
    return updated.reshape(2**L)


def applyG_block_state(G, state, L, perm):
    """
    Applies 2-qubit gate G to each (k, l) in perm on a 2^L state vector.
    """
    assert len(perm) % 2 == 0
    for j in range(len(perm) // 2):
        k = perm[2 * j]
        l = perm[2 * j + 1]
        state = applyG_state(G, state, L, k, l)
    return state

def partial_inner_product(v, w, k, l):
    assert v.shape == w.shape
    N = int(np.log2(v.size))
    assert 0 <= k < N and 0 <= l < N and k != l

    # Reshape state vectors into tensor of shape (2, 2, ..., 2)
    v_tensor = v.reshape([2] * N)
    w_tensor = w.reshape([2] * N)

    # Determine axes to keep and trace
    keep = sorted([k, l])
    trace = [i for i in range(N) if i not in keep]

    # Move qubits k and l to front
    perm = keep + trace
    v_perm = np.transpose(v_tensor, perm)
    w_perm = np.transpose(w_tensor, perm)

    # Reshape to (4, 2**(N-2))
    v_flat = v_perm.reshape(4, -1)
    w_flat = w_perm.reshape(4, -1)

    # Compute matrix: M_ij = sum_m conj(v_i,m) * w_j,m
    return np.dot(np.conj(v_flat), w_flat.T)


def get_perms(Lx, Ly):
    # permutations specifying gate layout
    # Verticals
    L = Lx*Ly
    perms_vercs = []
    for i in range(Ly):
        start_ind = Lx*i
        perms_verc = []
        for j in range(start_ind, start_ind+Lx):
            perms_verc += [j, (j+Lx)%L]
        perms_vercs.append(perms_verc)

    if Ly % 2 == 0:
        perms_vercs_compr = [perms_vercs[0], perms_vercs[1]]
        for i in range(Ly//2):
            perms_vercs_compr[i] += perms_vercs[i+Ly//2]
    if Ly % 2 == 1:
        perms_vercs_compr = [perms_vercs[0], perms_vercs[1], perms_vercs[-1]]
        if Ly>3:
            for i in range(Ly//2):
                perms_vercs_compr[i] += perms_vercs[i+Ly//2]
    # Horizontals
    perms_horzs = []
    for i in range(Lx):
        start_ind = i
        perms_horz = []
        for j in range(start_ind, L, Lx):
            if start_ind != Lx-1:
                perms_horz += [j, j+1]
            else:
                perms_horz += [j, j+1-Lx]
        perms_horzs.append(perms_horz)
    
    if Lx % 2 == 0:
        perms_horzs_compr = [perms_horzs[0], perms_horzs[1]]
        for i in range(Lx//2):
            perms_horzs_compr[i] += perms_horzs[i+Lx//2]
    if Lx % 2 == 1:
        perms_horzs_compr = [perms_horzs[0], perms_horzs[1], perms_horzs[-1]]
        if Lx>3:
            for i in range(Lx//2):
                perms_horzs_compr[i] += perms_horzs[i+Lx//2]
    
    return perms_vercs_compr, perms_horzs_compr



def reduce_list(vlist):
    vlist_vert_reduced = []
    vlist_horz_reduced = []
    vlist_vert = vlist[:len(vlist)//2]
    vlist_horz = vlist[len(vlist)//2:]

    for i in range(1, len(vlist)//2, 2):
        vlist_vert_reduced += [vlist_vert[i]]
        vlist_horz_reduced += [vlist_horz[i]]
    
    return vlist_vert_reduced + vlist_horz_reduced


def partial_trace_keep(U, keep_qubits, N):
	# Convert NumPy matrix to QuTiP Qobj
	full_dim = [2] * N  # Each qubit has dimension 2
	rho = qutip.Qobj(U, dims=[full_dim, full_dim])

	reduced_rho = rho.ptrace(keep_qubits)
	return reduced_rho.full()


def partial_trace_keep_tensor(rho_tensor, keep, L):
    all_idx = list(range(L))
    traced = [q for q in all_idx if q not in keep]
    
    idx = list(range(2*L))
    i_axes = traced
    j_axes = [L + q for q in traced]
    
    return np.trace(rho_tensor, axis1=i_axes, axis2=j_axes).reshape(4, 4)



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

def separability_penalty(V):
    """
    Penalty measuring entanglement in a 4x4 unitary matrix V.
    0 if V is a tensor product of 2 unitaries (ideally).
    """
    V_reshaped = V.reshape(2, 2, 2, 2)  # reshape for bipartite system
    rho = np.tensordot(V_reshaped, V_reshaped.conj(), axes=([1, 3], 
        [1, 3]))  # partial trace over second qubit
    rho = rho.reshape(4, 4)
    rho = rho / np.trace(rho)
    purity = np.trace(rho @ rho).real
    return 1 - purity  # 0 for product state, >0 for entangled


def grad_separability_penalty(V, eps=1e-6):
    grad = np.zeros_like(V)
    for i in range(4):
        for j in range(4):
            E = np.zeros_like(V)
            E[i, j] = eps
            grad[i, j] = (separability_penalty(V + E) - separability_penalty(V - E)) / (2 * eps)
    
    grad = antisymm_to_real(antisymm(V.conj().T @ grad)) 
    return grad


def construct_heisenberg_local_term(J, h, ndim=1):
    """
    Construct local interaction term of a Heisenberg-type Hamiltonian on a one-dimensional
    lattice for interaction parameters `J` and external field parameters `h`.
    """
    # Pauli matrices
    X = np.array([[ 0.,  1.], [ 1.,  0.]])
    Y = np.array([[ 0., -1j], [ 1j,  0.]])
    Z = np.array([[ 1.,  0.], [ 0., -1.]])
    I = np.identity(2)
    return (  J[0]*np.kron(X, X)
            + J[1]*np.kron(Y, Y)
            + J[2]*np.kron(Z, Z)
            + h[0]*0.5/ndim*(np.kron(X, I) + np.kron(I, X))
            + h[1]*0.5/ndim*(np.kron(Y, I) + np.kron(I, Y))
            + h[2]*0.5/ndim*(np.kron(Z, I) + np.kron(I, Z)))


def construct_ising_local_term(J, h, g, ndim=1):
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return J*np.kron(Z, Z) + g*0.5/ndim*(np.kron(X, I) + np.kron(I, X)) + h*0.5/ndim*(np.kron(Z, I) + np.kron(I, Z))


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

