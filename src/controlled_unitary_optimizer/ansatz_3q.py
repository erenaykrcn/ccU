import sys
sys.path.append("../brickwall_ansatz")
from utils import applyG_tensor, antisymm_to_real, antisymm, partial_trace_keep, real_to_antisymm
from utils_3q import cU_grad_bare, cU_ansatz_bare

def ansatz_3q(Xs, perms):
    if len(Xs)==len(perms):
        return cU_ansatz_bare(Xs, perms)
    
    Gs, Vs = (Xs[:3*(len(perms)+1)], Xs[3*(len(perms)+1):])
    L = 3
    ret_tensor = np.eye(2**L, dtype=complex).reshape([2]*2*L)
    for i, V in enumerate(Vs):
        for j in range(3):
            ret_tensor = applyG_tensor(np.kron(Gs[3*i+j], I2), ret_tensor, j, (j+1)%3)
        k, l = perms[i]
        ret_tensor = applyG_tensor(V, ret_tensor, k, l)
    for j in range(3):
        ret_tensor = applyG_tensor(np.kron(Gs[3*(len(perms))+j], I2), ret_tensor, j, (j+1)%3)
    return ret_tensor.reshape((2**L, 2**L))

def ansatz_grad_3q(cU, Xs, perms, flatten=True, unprojected=False):
    if len(Xs)==len(perms):
        return cU_grad_bare(cU, Xs, perms, flatten=flatten, unprojected=unprojected)
    L = 3
    Gs, Vs = (Xs[:3*(len(perms)+1)], Xs[3*(len(perms)+1):])

    gradG = []
    for i in range(len(Gs)):
        layer = i//3
        sub_index = i%3
        
        g =  np.eye(2**L).reshape([2]*2*L)
        for j in range(layer, len(perms)):
            g = applyG_tensor(Vs[j], g, perms[j][0], perms[j][1])
            for _ in range(3):
                g = applyG_tensor(np.kron(Gs[3*(j+1)+_], I2), g, _, (_+1)%3)
        g = (cU.conj().T @ g.reshape((2**L, 2**L))).reshape([2]*2*L)
                
        for j in range(layer):
            g = applyG_tensor(Vs[j], g, perms[j][0], perms[j][1])
            for _ in range(3):
                g = applyG_tensor(np.kron(Gs[3*(j+1)+_], I2), g, _, (_+1)%3)
        
        for j in range(3):
            if j!=sub_index:
                g = applyG_tensor(np.kron(Gs[3*layer+j], I2), g, j, (j+1)%3)

        g = partial_trace_keep(g.reshape(2**L, 2**L), [sub_index], L)
        gradG.append(-g.conj().T)
            
    gradV = []
    for i in range(len(Vs)):
        k, l = perms[i]
        G =  np.eye(2**L).reshape([2]*2*L)


        for j in range(i+1, len(Vs)):
            for _ in range(3):
                G = applyG_tensor(np.kron(Gs[3*j+_], I2), G, _, (_+1)%3)
            k_, l_ = perms[j]
            G = applyG_tensor(Vs[j], G, k_, l_)
        for _ in range(3):
            G = applyG_tensor(np.kron(Gs[3*len(Vs)+_], I2), G, _, (_+1)%3)
        G = (cU.conj().T @ G.reshape((2**L, 2**L))).reshape([2]*2*L)
        
        for j in range(i):
            for _ in range(3):
                G = applyG_tensor(np.kron(Gs[3*j+_], I2), G, _, (_+1)%3)
            k_, l_ = perms[j]
            G = applyG_tensor(Vs[j], G, k_, l_)
        
        for _ in range(3):
            G = applyG_tensor(np.kron(Gs[3*i+_], I2), G, _, (_+1)%3)

        G = partial_trace_keep(G.reshape(2**L, 2**L), [k, l], L)
        if k > l:
            SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            G = SWAP @ G @ SWAP
        
        gradV.append(-G.conj().T)
        
    if unprojected:
        return gradG+gradV
        
    Wlist = Gs + Vs
    grad = gradG+gradV
    # Project onto tangent space.
    if flatten:
        return np.concatenate([
            antisymm_to_real(antisymm(Wlist[j].conj().T @ grad[j])).reshape(-1)
            for j in range(len(grad))
        ])
    else:
        return [
            antisymm_to_real(antisymm(Wlist[j].conj().T @ grad[j]))
            for j in range(len(grad))
        ]


