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


def get_params(eta, gamma):
    """
        Helper function to retrive the Trotter coefficients for the starting 
        point, depending on the number of controlling layers (gamma.)
        This one is for Heisenberg 000 model.
    """
    
    eta_coeffs_dict = {2: [1, 1], 3: [0.5, 1, 0.5], 4: [0.5, 0.5, 0.5, 0.5], 5:[0.25, 0.5, 0.5, 0.5, 0.25]}
    gamma_coeffs_dict = {2: [1, 1], 3: [1, 1, 1], 4: [0.5, 1, 1, 0.5], 5: [0.5, 0.5, 1, 0.5, 0.5], 6: [0.5, 0.5, 1, 0.5, 0.5]}
    gamma_indices_dict = {2: [0, 1], 3: [0,1,2], 4: [0, 1, 2, 0], 5: [0, 1, 2, 1, 0], 6: [0, 1, 2, 1, 0]}
    
    W_dict = {(0, 6): [np.kron(Z, I2), np.kron(I2, I2), np.kron(Z@X, I2), np.kron(I2, I2),
      np.kron(X@Z, I2), np.kron(I2, I2), np.kron(Z, I2)],
              (1, 5): [np.kron(Z, I2), np.kron(I2, I2), np.kron(Z@X, I2), np.kron(X@Z, I2), np.kron(I2, I2), np.kron(Z, I2)],
              (0, 5): [np.kron(Z, I2), np.kron(I2, I2), np.kron(Z@X, I2), np.kron(I2, X@Z), np.kron(I2, I2), np.kron(I2, Z)],
              (1, 4): [np.kron(Z, I2), np.kron(I2, I2), np.kron(Z@X, I2), np.kron(X@Z, I2), np.kron(Z, I2)],
              (0, 4): [np.kron(Z, I2), np.kron(I2, I2), np.kron(Z@X, I2), np.kron(I2, X@Z), np.kron(Z, I2)],
              (1, 3): [np.kron(Z, I2), np.kron(I2, I2), np.kron(Z@X, I2), np.kron(X, I2)],
              (0, 3): [np.kron(Z, I2), np.kron(I2, I2), np.kron(Z@X, I2), np.kron(I2, X)],
              
              (1, 2): [np.kron(Z, I2), np.kron(Z@X, I2), np.kron(X, I2)],
              (0, 2): [np.kron(Z, I2), np.kron(I2, Z@X), np.kron(X, I2)],
    }

    return eta_coeffs_dict[eta], gamma_coeffs_dict[gamma], gamma_indices_dict[gamma], W_dict[(eta%2, gamma)]


def get_params_heis111(eta, gamma):
    """
        Helper function to retrive the Trotter coefficients for the starting 
        point, depending on the number of controlling layers (gamma.)
        This one is for Heisenberg 111 model.
    """
    
    eta_coeffs_dict = {2: [1, 1], 3: [0.5, 1, 0.5], 4: [0.5, 0.5, 0.5, 0.5], 5:[0.25, 0.5, 0.5, 0.5, 0.25]}
    gamma_coeffs_dict = {2: [1, 1], 3: [1, 1, 1], 4: [0.5, 1, 1, 0.5], 5: [0.5, 0.5, 1, 0.5, 0.5], 6: [0.5, 0.5, 1, 0.5, 0.5]}
    gamma_indices_dict = {2: [0, 1], 3: [0,1,2], 4: [0, 1, 2, 0], 5: [0, 1, 2, 1, 0], 6: [0, 1, 2, 1, 0]}
    
    W_dict = {
              (0, 5): [np.kron(X,  Z), np.kron(Z@Y, I2), np.kron(X@Z, I2), np.kron(I2, Z@X),
                        np.kron(I2, Y@Z), np.kron(Z, X)],
              
              (1, 3): [np.kron(X, Z), np.kron(I2,  Z@Y), np.kron(X@Z, I2), np.kron(Z, Y)],
              (0, 3): [np.kron(X,  Z), np.kron(Z@Y, I2), np.kron(X@Z, I2), np.kron(Y, Z)],
    }

    return eta_coeffs_dict[eta], gamma_coeffs_dict[gamma], gamma_indices_dict[gamma], W_dict[(eta%2, gamma)]



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


def reduce_list(vlist, gamma, eta):
    vlist_reduced = []
    for i in range(gamma):
        for j in range(eta*i+1+i, eta*(i+1)+1+i):
            vlist_reduced.append(vlist[j])
    return vlist_reduced



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


def construct_ising_local_term(J, g):
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return J*np.kron(Z, Z) + g*0.5*(np.kron(X, I) + np.kron(I, X))


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



def trotterized_time_evolution(qc, L, hlocs, t, gate, lamb=None):
    # permutations specifying gate layout
    perms1 = [i for i in range(L)]
    perms2 = [i for i in range(1, L)]+[0]
    perm_set = [perms1, perms2]
    perms = perm_set
    
    K_layer = [None for _ in range(L)]
    for j in range(len(perms[0])//2):
        K_layer[perms[0][2*j]] = gate[0]
        K_layer[perms[0][2*j+1]] = gate[1]
        
    K_b = [K_layer, None]
    K_a = [None, K_layer]
    
    Vlists = [[scipy.linalg.expm(-1j*t*hloc) for i in range(len(perm_set))] for hloc in hlocs]
    Vlists_gates = [[] for _ in hlocs]
    for i, Vlist in enumerate(Vlists):
        for V in Vlist:
            qc2 = qiskit.QuantumCircuit(2)
            qc2.unitary(V, [0, 1], label='str')
            Vlists_gates[i].append(qc2)


    for layer in range(len(perms)):
        if K_b[layer] is not None:
            qc.x(L)
            for j in range(L):
                if K_b[layer][j] is not None:
                    qc.append(K_b[layer][j](), [L, L-1-j])
            qc.x(L)

        for Vlist_gates in Vlists_gates:
            qc_gate = Vlist_gates[layer]
            
            for j in range(len(perms[layer])//2):
                    qc.append(qc_gate.to_gate(), [L-(perms[layer][2*j]+1), 
                                                    L-(perms[layer][2*j+1]+1)])
        if K_a[layer] is not None:
            qc.x(L)
            for j in range(L):
                if K_a[layer][j] is not None:
                    qc.append(K_a[layer][j](), [L, L-1-j])
            qc.x(L)


def cU_trotter(t, L, hlocs, cgates, trotter_step=0.1, trotter_degree=2):
    nsteps = 1
    if t > trotter_step:
        nsteps = int(np.ceil(t/trotter_step))
    dt = t/nsteps
    
    # Trotter 2nd Order.
    indices = [0, 1, 0]
    coeffs  = [0.5, 1, 0.5]
    if trotter_degree==1:
        # Trotter 1st Order.
        indices = [0, 1]
        coeffs  = [1, 1]
    elif trotter_degree==3:    
        # Trotter 3rd Order.
        indices = [0, 1, 0, 1, 0]
        coeffs  = [0.25, 0.5, 0.5, 0.5, 0.25]
    elif trotter_degree==4:
        # Trotter 4th order:
        indices = [0, 1, 0, 1, 0, 1, 0, 1, 0]
        coeffs  = [0.125, 0.25,  0.25,  0.25,  0.25, 0.25,  0.25,  0.25,  0.125]

    qc_cU_ins = qiskit.QuantumCircuit(L+1)
    for n in range(nsteps):
        for index, coef in zip(indices, coeffs):
            trotterized_time_evolution(qc_cU_ins, L, hlocs[index], dt * coef, cgates[index])

    return qc_cU_ins







    