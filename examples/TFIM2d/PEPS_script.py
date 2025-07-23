import numpy as np
import quimb.tensor as qtn
import quimb
from quimb.tensor.tensor_2d_tebd import TEBD2D, LocalHam2D
import scipy
import h5py
import qib
from scipy.sparse.linalg import expm_multiply

import sys
sys.path.append("../../src/brickwall_sparse")
from utils_sparse import applyG_block_state, get_perms

Lx, Ly = (4, 4)
L = Lx*Ly
latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J, h, g = (1, 0, 3)
hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()

perms_v, perms_h = get_perms(Lx, Ly)
perms_extended = [[perms_v[0]]] + [perms_v]*3 + [[perms_v[0]], [perms_h[0]]] +\
                    [perms_h]*3 + [[perms_h[0]], [perms_v[0]]] + [perms_v]*3 + [[perms_v[0]]]
perms_ext_reduced = [perms_v]*3  + [perms_h]*3 + [perms_v]*3
map_ = {i: (i//Ly, i%Lx) for i in range(L)}


Vlist = []
with h5py.File(f"./results/tfim2d_ccU_SPARSE_103_Lx4Ly4_t0.25_layers15_rS1_niter15_3hloc.hdf5", "r") as f:
    Vlist =  f["Vlist"][:]
control_layers = [0, 4, 5, 9, 10, 14]
perms_qc = [[0, 1], [0, 2], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2]]
Xlists_opt = {}
for i in control_layers:
    with h5py.File(f"./results/tfim2d_ccU_SPARSE_103_Lx4Ly4_t0.25_layers15_niter20_rS5_DECOMPOSE_n9_layer{i}.hdf5", "r") as file:
        Xlists_opt[i] = file[f"Xlist_{i}"][:]


peps = qtn.PEPS.rand(Lx, Ly, bond_dim=1, phys_dim=2, cyclic=True)
peps /= peps.norm()
peps_copy = peps.copy()
sv = peps_copy.to_dense()[:, 0]
sv = expm_multiply(-1j * 0.25 * hamil, sv)

for i, V in enumerate(Vlist):
    if i not in control_layers:
        perms = perms_extended[i]
        for perm in perms:
            t = TEBD2D(peps, ham=LocalHam2D(Lx, Ly, 
                    {(map_[perm[2*j]], map_[perm[2*j+1]]): scipy.linalg.logm(V) for j in range(L//2)}, cyclic=True),
            tau=-1, D=8 if i<10 else 4)
            t.sweep(tau=-1)
            peps = t.state
            peps /= peps.norm()

        with open(f"PEPS_log{Lx}{Ly}.txt", "a") as file:
            file.write(f"Layer {i}/{len(Vlist)} applied \n")

f = quimb.fidelity(peps.to_dense()[:, 0], sv)
with open(f"PEPS_log{Lx}{Ly}.txt", "a") as file:
    file.write(f"Fidelity after identity: {f} \n\n")


