import numpy as np
from utils_MPS import *
import scipy

import sys
sys.path.append("../brickwall_sparse")
from utils_sparse import construct_ising_local_term, reduce_list, X, I2, get_perms, Z
import rqcopt as oc


def trotter(mps, t, L, Lx, Ly, J, g, perms_v, perms_h, dag=False, max_bond_dim=None, dt=0.1, trotter_order=2):
    nsteps = np.abs(int(np.ceil(t/dt)))
    t = t/nsteps
    print("dt", t)
    indices = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2))).indices
    coeffs = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2))).coeffs
    
    hloc1 = g*(np.kron(X, I2)+np.kron(I2, X))/4
    hloc2 = J*np.kron(Z, Z)
    hlocs = (hloc1, hloc2)
    Vlist_start = []
    for i, c in zip(indices, coeffs):
        Vlist_start.append(scipy.linalg.expm(-1j*c*t*hlocs[i]))

    gates = 0
    for n in range(nsteps):
        for layer, V in enumerate(Vlist_start):
            for perm in perms_h:
                for j in range(len(perm)//2):
                    mps = apply_localGate(mps, V, perm[2*j], perm[2*j+1], max_bond_dim=max_bond_dim)
                    gates += 1
                    with open(f"trotter{Lx}{Ly}_log.txt", "a") as file:
                        file.write(f"Gate {gates} \n")

            with open(f"trotter{Lx}{Ly}_log.txt", "a") as file:
                file.write(f"\n Normalization starting \n")
            mps = right_normalize(mps)
            with open(f"trotter_log{Lx}{Ly}.txt", "a") as file:
                file.write(f"Normalization finished \n")

            for perm in perms_v:
                for j in range(len(perm)//2):
                    mps = apply_localGate(mps, V, perm[2*j], perm[2*j+1], max_bond_dim=max_bond_dim)
                    gates += 1
                    with open(f"trotter_log{Lx}{Ly}.txt", "a") as file:
                        file.write(f"Gate {gates} \n")
                    
            with open(f"trotter_log{Lx}{Ly}.txt", "a") as file:
                file.write(f"Normalization starting \n")
            mps = right_normalize(mps)
            with open(f"trotter_log{Lx}{Ly}.txt", "a") as file:
                file.write(f"Normalization finished \n")

            with open(f"trotter_log{Lx}{Ly}.txt", "a") as file:
                file.write(f"Time step {n}/{nsteps}, layer {layer}/{len(Vlist_start)} applied \n")
    return mps


def ccU(mps, L, Vlist, Xlists_opt, perms, perms_qc, control_layers, max_bond_dim=None, swap=False):
    mps = apply_localGate(mps, np.kron(X, I2), 0, 1, max_bond_dim=max_bond_dim)
    for i, V in enumerate(Vlist):
        layer = i
        if i in control_layers:
            for perm in perms[layer]:
                for j in range(L//2):
                    mapp = {0: 0, 1: perm[2*j]+1, 2:perm[2*j+1]+1}
                    for l, G in enumerate(Xlists_opt[i]):
                        if swap:
                            mps = apply_localGate(mps, G, mapp[perms_qc[l][0]], 
                                mapp[perms_qc[l][1]], max_bond_dim=max_bond_dim)
                        else:
                            mpo = make_long_range_mpo(L+1, mapp[perms_qc[l][0]], 
                                mapp[perms_qc[l][1]], G)
                            mps = apply_mpo_to_mps(mpo, mps, max_bond_dim=max_bond_dim)
                            
            #mps = right_normalize(mps)
        else:
            for perm in perms[layer]:
                for j in range(len(perm)//2):
                    if swap:
                        mps = apply_localGate(mps, V, perm[2*j]+1, perm[2*j+1]+1, 
                            max_bond_dim=max_bond_dim)
                    else:
                        mpo = make_long_range_mpo(L+1, perm[2*j]+1, perm[2*j+1]+1, V)
                        mps = apply_mpo_to_mps(mpo, mps, max_bond_dim=max_bond_dim)
            #mps = right_normalize(mps)

        with open(f"ccU_log.txt", "a") as file:
            file.write(f"Layer {i}/{len(Vlist)} applied \n")
    mps = apply_localGate(mps, np.kron(X, I2), 0, 1, max_bond_dim=max_bond_dim)

    return mps
