import numpy as np
from scipy.linalg import svd as robust_svd


def left_normalize(Ms):
    As = []
    T = np.ones((1, 1))
    for M in Ms:
        M = np.tensordot(T, M, axes=(1, 1)) 
        M = np.transpose(M, [1, 0, 2])
        d, chi1, chi2 = M.shape             
        U, S, Vh = safe_svd(np.reshape(M, [d*chi1, chi2]), full_matrices=False)
        A = np.reshape(U, [d, chi1, -1])   
        As.append(A)                        
        T = np.diag(S) @ Vh                 
    # Keep leftover signs (but no normalization)
    As[0] = As[0]*np.sign(T)
    return As


def right_normalize(Ms):
    Bs = []
    T = np.ones((1, 1))
    for M in Ms[::-1]:
        M = np.tensordot(M, T, axes=(2, 0))
        d, chi1, chi2 = M.shape
        M = np.reshape(M, [chi1, d*chi2])
        U, S, Vh = safe_svd(M, full_matrices=False)
        _, chi_s = U.shape
        Bs.append(Vh.reshape([chi_s, d, chi2]).transpose([1, 0, 2]))
        T = U@np.diag(S)
    Bs[0] = Bs[0] * np.sign(T)
    return Bs[::-1]


def safe_svd(matrix, full_matrices=False):
    try:
        return np.linalg.svd(matrix, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        print("⚠️ Warning: Falling back to scipy SVD due to convergence failure.")
        with open("safe_svd.txt", "a") as file:
            file.write(f"⚠️ Warning: Falling back to scipy SVD due to convergence failure. \n")
        matrix += 1e-12 * np.random.randn(*matrix.shape)
        return robust_svd(matrix, full_matrices=full_matrices)

    
def random_mps(L_plus_1, max_bond_dim=None, anc=None):
    D = max_bond_dim if max_bond_dim is not None else np.inf
    
    d = 2  # Qubit system (dim=2)
    mps = []

    D1 = min(D, d)
    A0 = np.zeros((d, 1, D1), dtype=np.complex128)
    if anc is None:
        A0[0, 0, :] = np.random.randn(D1) + 1j * np.random.randn(D1)  # only |0⟩ component
        A0[1, 0, :] = np.random.randn(D1) + 1j * np.random.randn(D1)  # only |0⟩ component
    elif anc==0:
        A0[0, 0, :] = np.random.randn(D1) + 1j * np.random.randn(D1)  # only |0⟩ component
    else:
        A0[1, 0, :] = np.random.randn(D1) + 1j * np.random.randn(D1)  # only |0⟩ component
        
    A0 /= np.linalg.norm(A0)  # normalize
    mps.append(A0)
        
    Dl = D1
    # Middle sites (1 to L-1)
    for i in range(1, L_plus_1 - 1):
        Dr = min(D, d * Dl)
        A = np.random.randn(d, Dl, Dr) + 1j * np.random.randn(d, Dl, Dr)
        A /= np.linalg.norm(A)
        mps.append(A)
        Dl = Dr

    # Last site (site L): (d=2, Dl, Dr=1)
    A_last = np.random.randn(d, Dl, 1) + 1j * np.random.randn(d, Dl, 1)
    A_last /= np.linalg.norm(A_last)
    mps.append(A_last)
    mps = left_normalize(mps)
    return mps


def mps_fidelity(mps1, mps2):
    assert len(mps1) == len(mps2), "MPSs must have same length"

    # Start with scalar "environment" = 1
    env = np.ones((1, 1))  # shape (1, 1)
    for A, B in zip(mps1, mps2):
        env = np.tensordot(env, A, axes=(0, 1))
        env = np.tensordot(env, B.conj(), axes=([0, 1], [1, 0]))
    # env should now be scalar (1x1)
    inner_product = env[0, 0]
    fidelity = np.linalg.norm(inner_product) ** 2
    return fidelity.real



def get_mps_of_sv(state_vector, max_bond_dim=None, cutoff=1e-10):
    sv = np.asarray(state_vector, dtype=complex)
    N = int(np.log2(sv.size))
    assert 2 ** N == sv.size, "Input size must be a power of 2 (qubit system)"
    
    mps = []
    psi = sv.reshape([2] * N)  # Turn into N-dimensional tensor

    # Left-to-right SVD sweep
    for n in range(N - 1):
        D = psi.shape[0]
        psi = psi.reshape(D, -1)  # Flatten remaining qubits
        # SVD
        U, S, Vh = safe_svd(psi, full_matrices=False)

        # Truncate
        #keep = (S > cutoff)
        keep = np.where(S > cutoff)[0]
        if max_bond_dim is not None:
            keep = keep[:max_bond_dim]
        U = U[:, keep]
        S = S[keep]
        Vh = Vh[keep, :]

        Dl = U.shape[0] // 2
        Dr = U.shape[1]
        A = U.reshape(Dl, 2, Dr).transpose([1, 0, 2])
        mps.append(A)
        
        psi = np.diag(S) @ Vh  # Pass remainder to next step
        psi = psi.reshape(Dr * 2, -1)
        

    A_last = psi.reshape(psi.shape[0] // 2, 2, 1).transpose([1, 0, 2])
    mps.append(A_last)
    return mps


def mps_to_state_vector(mps):
    N = len(mps)
    psi = mps[0]  # shape: (2, 1, D1)

    # Contract left to right
    for i in range(1, N):
        A = mps[i]
        psi = np.tensordot(psi, A, axes=(psi.ndim-1, 1))
        psi = psi.transpose([i for i in range(psi.ndim-3)] + [psi.ndim-2, psi.ndim-3] + [psi.ndim-1])

    psi = np.squeeze(psi)  # removes axes of length 1
    psi = psi.reshape(-1)
    return psi


def apply_two_site_operator(mps, gate, site, max_bond_dim=None, cutoff=1e-10):
    A1 = mps[site]      # shape (2, Dl, Dmid)
    A2 = mps[site + 1]  # shape (2, Dmid, Dr)

    d, Dl, Dmid = A1.shape
    d2, Dmid2, Dr = A2.shape
    assert d == d2 == 2 and Dmid == Dmid2

    # Combine A1 and A2 into one tensor
    theta = np.tensordot(A1, A2, axes=(2, 1))  # (2, Dl, Dmid) x (2, Dmid, Dr) → (2, Dl, 2, Dr)
    theta = np.transpose(theta, (0, 2, 1, 3))  # → (2,2,Dl,Dr)
    theta = theta.reshape(4, Dl * Dr)          # → (4, Dl*Dr)

    # Apply gate
    theta = np.tensordot(gate, theta, axes=(1, 0))  # (4,4) x (4, Dl*Dr)
    theta = theta.reshape(2, 2, Dl, Dr)        # back to (2,2,Dl,Dr)
    theta = np.transpose(theta, (0, 2, 1, 3))  # → (2,Dl,2,Dr)
    theta = theta.reshape(2 * Dl, 2 * Dr)      # → (2*Dl, 2*Dr)

    # SVD
    U, S, Vh = safe_svd(theta, full_matrices=False)
    # Truncate
    keep = np.where(S > cutoff)[0]
    if max_bond_dim is not None:
        keep = keep[:max_bond_dim]
    U = U[:, keep]
    S = S[keep]
    Vh = Vh[keep, :]

    # Rebuild A1 and A2
    new_D = len(S)
    U = U.reshape(2, Dl, new_D)
    Vh = (np.diag(S)@Vh).reshape(new_D, 2, Dr).transpose([1, 0, 2])

    mps[site] = U
    mps[site + 1] = Vh

    return mps


def apply_localGate(mps, op, k, l, max_bond_dim=None):
    SWAP = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    if k == l:
        raise ValueError("Sites k and l must be different")
    if k > l:
        k, l = l, k
        op = SWAP @ op @ SWAP
    
    for i in range(l-1, k, -1):
        mps = apply_two_site_operator(mps, SWAP, i)
    
    # Now sites k and k+1 correspond to original sites k and l
    mps = apply_two_site_operator(mps, op, k, max_bond_dim=max_bond_dim)
    
    for i in range(k+1, l):
        mps = apply_two_site_operator(mps, SWAP, i)
    
    return mps


def decompose_two_site_gate(gate, tol=1e-12):
    gate = gate.reshape([2, 2, 2, 2]).transpose([0, 2, 1, 3]).reshape(4, 4)
    d = 2
    U, S, Vh = np.linalg.svd(gate, full_matrices=False)
    A_list = []
    B_list = []
    for alpha in range(len(S)):
        vecA = U[:, alpha] * S[alpha]
        vecB = Vh[alpha, :]
        A = vecA.reshape(d, d)
        B = vecB.reshape(d, d)
        A_list.append(A)
        B_list.append(B)
    return A_list, B_list


def make_long_range_mpo(L, k, l, gate, tol=1e-12):
    SWAP = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    if k == l:
        raise ValueError("Sites k and l must be different")
    if k > l:
        k, l = l, k
        gate = SWAP @ gate @ SWAP
    d = 2
    Id = np.eye(d, dtype=complex)
    mpo = []
    
    A_list, B_list = decompose_two_site_gate(gate.copy(), tol=tol)
    r = len(A_list) # rank of the gate.
        
    for i in range(L):
        if i < k or i > l:
            # trivial identity tensor, bond dimension 1
            W = Id.reshape(1, 1, d, d)
        elif i == k:
            # left endpoint: shape (1, r, d, d)
            W = np.zeros((1, r, d, d), dtype=complex)
            for alpha, A in enumerate(A_list):
                W[0, alpha, :, :] = A
        elif i == l:
            # right endpoint: shape (r, 1, d, d)
            W = np.zeros((r, 1, d, d), dtype=complex)
            for alpha, B in enumerate(B_list):
                W[alpha, 0, :, :] = B
        else:
            # intermediate sites between k and l: carry label alpha with identity
            W = np.zeros((r, r, d, d), dtype=complex)
            for alpha in range(r):
                W[alpha, alpha, :, :] = Id
        mpo.append(W)
    return mpo

def mpo_to_full_matrix(mpo):
    L = len(mpo)
    M = mpo[0].reshape([mpo[0].shape[1], 2, 2]).transpose([1, 0, 2]).transpose([0, 2, 1])
    for i, W in enumerate(mpo[1:]):
        W = W.transpose([2, 3, 0, 1])
        M = np.tensordot(M, W, axes=((i+1)*2, 2))
    upper = list(range(0, 2*L, 2))
    lower = list(range(1, 2*L, 2))
    perm = upper + lower
    M = M.reshape(2*L*[2])
    T_perm = M.transpose(perm)
    d = M.shape[0]
    T_mat = T_perm.reshape(d**L, d**L)
    return T_mat


def apply_mpo_to_mps(mpo, mps, max_bond_dim=None, cutoff=1e-10):
    L = len(mps)
    assert len(mpo) == L
    new_mps = []

    T = np.ones((1, 1), dtype=complex)
    for M, W in zip(mps, mpo):
        e = np.tensordot(W, M, axes=(3, 0))
        e = np.transpose(e, (2,3,0,1,4))
        e = np.transpose(e, (0, 1, 2, 4, 3))
        s, a, b, c, d = e.shape
        e = e.reshape([s, a*b, c*d])
        
        e = np.tensordot(T, e, axes=(1, 1)).transpose([1, 0, 2])
        s, Dl, Dr = e.shape
        e = e.reshape([s*Dl, Dr])
        U, S, Vh = safe_svd(e)

        keep = np.where(S > cutoff)[0]
        if max_bond_dim is not None:
            keep = keep[:max_bond_dim]
        U = U[:, keep]
        S = S[keep]
        Vh = Vh[keep, :]
        # Rebuild A1 and A2
        new_D = len(S)
        U = U.reshape(s, Dl, new_D)
        T = (np.diag(S)@Vh)

        new_mps.append(U)    
    return left_normalize(new_mps)
    



