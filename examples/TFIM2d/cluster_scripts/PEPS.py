import sys
sys.path.append("../../src/brickwall_sparse")
from utils_sparse import applyG_block_state, get_perms

import numpy as np
import quimb.tensor as qtn
import quimb
from quimb.tensor.tensor_2d_tebd import TEBD2D, LocalHam2D
import scipy
import h5py
import qib
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply

import time
import tracemalloc
tracemalloc.start()


chi_norm = 256
chi_overlap = 256
BD = 5
nsteps = 2


t = 0.125
Lx, Ly = (6, 6)
L = Lx*Ly
latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J, h, g = (1, 0, 3)

perms_v, perms_h = (
    [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6, 13, 14, 15, 16, 17, 12, 19, 20, 21, 22, 23, 18, 25, 26, 27, 28, 29, 24, 31, 32, 33, 34, 35, 30]],
    [[0, 6, 12, 18, 24, 30, 1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29, 35], 
    [6, 12, 18, 24, 30, 0, 7, 13, 19, 25, 31, 1, 8, 14, 20, 26, 32, 2, 9, 15, 21, 27, 33, 3, 10, 16, 22, 28, 34, 4, 11, 17, 23, 29, 35, 5]]
)

perms_extended = [[perms_v[0]]] + [perms_v]*3 + [[perms_v[0]], [perms_h[0]]] +\
                    [perms_h]*3 + [[perms_h[0]]]
perms_ext_reduced = [perms_v]*3  + [perms_h]*3
control_layers = [0, 4, 5, 9] 			# 4 control layers
Vlist = []
with h5py.File(f"../results/tfim2d_ccU_SPARSE_103_Lx4Ly4_t0.125_layers10_niter10_rS1_2hloc.hdf5", "r") as f:
    Vlist =  f["Vlist"][:]

map_ = {i: (i//Ly, i%Lx) for i in range(L)}

peps = qtn.PEPS.rand(Lx, Ly, bond_dim=1, phys_dim=2, cyclic=True)
peps = peps / peps.norm()
peps_E = peps.copy()
peps_T = peps.copy()
peps_C = peps.copy()


peps_E = trotter(peps_E.copy(), t, L, Lx, Ly, J, g, perms_v, perms_h,
                     dt=t/nsteps, max_bond_dim=BD, lower_max_bond_dim=BD, trotter_order=2)
peps_T = trotter(peps_T.copy(), t, L, Lx, Ly, J, g, perms_v, perms_h,
                     dt=t/nsteps, max_bond_dim=BD, lower_max_bond_dim=BD, trotter_order=1)
peps_aE = ccU(peps_C.copy(), Vlist, perms_extended, control_layers, dagger=False,
                 max_bond_dim=BD, lower_max_bond_dim=BD)



for state in (peps_E, peps_T, peps_aE):
    # normalize in-place using approximate boundary contraction
    state.normalize(
        max_bond=chi_norm,
        cutoff=1e-10,
        mode="mps",
        layer_tags=("KET", "BRA"),
        inplace=True,
    )
ov_tn = peps_E.make_overlap(
    peps_aE,
    layer_tags=("KET", "BRA"),
)
  # max internal bond during compressed contraction
overlap_approx = ov_tn.contract_compressed(
    optimize="auto-hq",  # preset strategy name understood via cotengra
    max_bond=chi_overlap,
    cutoff=1e-10,
    # leave strip_exponent=False (default) so we just get a scalar back
)

with open(f"PEPS_log.txt", "a") as file:
	file.write("Fidelity for ccU: "+str(np.abs(overlap_approx)))


 
for state in (peps_E, peps_T, peps_aE):
    # normalize in-place using approximate boundary contraction
    state.normalize(
        max_bond=chi_norm,
        cutoff=1e-10,
        mode="mps",
        layer_tags=("KET", "BRA"),
        inplace=True,
    )
ov_tn = peps_E.make_overlap(
    peps_T,
    layer_tags=("KET", "BRA"),
)

overlap_approx = ov_tn.contract_compressed(
    optimize="auto-hq",  # preset strategy name understood via cotengra
    max_bond=chi_overlap,
    cutoff=1e-10,
    # leave strip_exponent=False (default) so we just get a scalar back
)

with open(f"PEPS_log.txt", "a") as file:
	file.write("Fidelity for Trotter 1: "+str(np.abs(overlap_approx)))

