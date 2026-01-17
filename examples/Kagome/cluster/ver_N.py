import qib
import numpy as np
from tenpy.models.lattice import Kagome
from tenpy.networks.site import SpinHalfSite
import scipy.linalg
import time
import rqcopt as oc
from quimb.tensor.tensor_arbgeom_tebd import LocalHamGen, TEBDGen, edge_coloring
import quimb as qu
import quimb.tensor as qtn
import gc
from qiskit.quantum_info import state_fidelity
import quimb.tensor as qtn
import quimb
from quimb.tensor.tensor_2d_tebd import TEBD2D, LocalHam2D
import h5py

import sys
sys.path.append("../../../src/brickwall_sparse")
from utils_sparse import construct_heisenberg_local_term

reverse = True
chi_overlap = 25
Lx, BD, chi_overlap1, chi_overlap2, chi_overlap_incr  = (3, 4, 5, 32, 30)
#Lx, BD, chi_overlap1, chi_overlap2, chi_overlap_incr  = (4, 3, 2, 15, 2)
cutoff = 1e-12
layers = 22
t = 0.1
trotter_order, trotter_step = (2, 1)
trotter_order_ref, trotter_step_ref = (2, 6)

rS = 1

L = Lx*Lx*3
lat = Kagome(Lx, Lx, [SpinHalfSite() for _ in range(3)], bc='periodic')
N = lat.N_sites
A = np.zeros((N, N), dtype=int)
J = (1, 1, 1)
h = (3, -1, 1)

if Lx>2:
    perms_1_ = []
    for i in range(Lx):
        perms_1_ += [i*3*Lx+j for j in range(2*Lx)]
    perms_2_ = []
    for i in range(Lx):
        for j in range(Lx):
            perms_2_ += [2*i+j*(Lx*3), 2*i+j*(Lx*3)+(Lx*2-i)]
    perms_1 = [perms_1_, []]
    for i in range(Lx):
        perms_1[1] += list(np.roll(np.array(perms_1_[2*Lx*i:2*Lx*(i+1)]), 1))
    perms_2 = [perms_2_, []]
    for i in range(Lx):
        perms_2[1] += list(np.roll(np.array(perms_2_[2*Lx*i:2*Lx*(i+1)]), 1))
    if Lx== 3:
        perms_3 = [[1, 6, 3, 7, 10, 15, 5, 8, 12, 16, 19, 24, 14, 17, 21, 25, 23, 26],
                   [6, 1, 7, 10, 15, 3, 8, 12, 16, 19, 24, 5, 17, 21, 25, 14, 26, 23]]
    elif Lx==2:
        perms_3 = [[1, 4, 9, 11, 3, 5, 7, 10], [4, 1, 11, 9, 5, 7, 10, 3]]
    elif Lx ==4:
        perms_3 = [[1, 8, 3, 9, 13, 20, 5, 10, 15, 21, 25, 32, 7, 11, 17, 22, 27, 33, 37, 44, 19, 23, 29, 34, 39, 45, 31, 35, 41, 46, 43, 47],
        [8, 1, 9, 13, 20, 23, 10, 15, 21, 25, 32, 5, 11, 17, 22, 27, 33, 37, 44, 7, 23, 29, 34, 39, 45, 19, 35, 41, 46, 31, 47, 43]]
else:
    perms_1 = [[0, 4, 6, 10, 2, 5, 8, 11], [4, 6, 10, 0, 5, 8, 11, 2]]
    perms_2 = [[0, 1, 2, 3, 6, 7, 8, 9], [1, 2, 3, 0, 7, 8, 9, 6]]
    perms_3 = [[1, 4, 9, 11, 3, 5, 7, 10], [4, 1, 11, 9, 5, 7, 10, 3]]
p1, p2, p3, p4, p5, p6 = ([perms_2[0]], [perms_2[1]], [perms_1[0]], [perms_1[1]], [perms_3[0]], [perms_3[1]])
ps = [p1, p2, p3, p4, p5, p6]
for i in range(6):
    print(f"p{i} ", ps[i])

for perm in perms_1+perms_2+perms_3:
    for i in range(len(perm)//2):
        A[perm[2*i], perm[2*i+1]] = 1
        A[perm[2*i+1], perm[2*i]] = 1

# Pauli and identity
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)
def layer_from_flat_perm(perm_row, L):
    return [(perm_row[2*j], perm_row[2*j+1]) for j in range(len(perm_row) // 2)]
layers_raw = [
    perms_1[0], perms_1[1],
    perms_2[0], perms_2[1],
    perms_3[0], perms_3[1],
]
perms_for_trotter = [layer_from_flat_perm(row, L) for row in layers_raw]



def _edges_from_permutations(perms_1, perms_2, perms_3):
    """
    #Take the three [src, tgt] permutation pairs and return a sorted list of
    #unique undirected edges (i, j) with i < j.
    """
    edge_set = set()

    for perms in (perms_1, perms_2, perms_3):
        src, tgt = perms
        for a, b in zip(src, tgt):
            if a == b:
                continue
            i, j = sorted((a, b))
            edge_set.add((i, j))

    return sorted(edge_set)


def build_triangular_PEPS(bond_dim, phys_dim=2,
                          seed=None, dtype="complex128"):
    edges = _edges_from_permutations(perms_1, perms_2, perms_3)
    tn = qtn.TN_from_edges_rand(
        edges,
        D=bond_dim,
        phys_dim=phys_dim,
        seed=seed,
        dtype=dtype,
    )

    return tn, (perms_1, perms_2, perms_3)


def trotter(peps, t, L, J,  perms, dag=False,
                      max_bond_dim=5, dt=0.1, trotter_order=2, h=h):
    # Number of steps
    nsteps = abs(int(np.ceil(t / dt)))
    dt = t / nsteps

    #hloc1 = J[0]*np.kron(X, X) + h[0]/4 * (np.kron(X, I2)+np.kron(I2, X))
    #hloc2 = J[1]*np.kron(Y, Y) + h[1]/4 * (np.kron(Y, I2)+np.kron(I2, Y))
    #hloc3 = J[2]*np.kron(Z, Z) + h[2]/4 * (np.kron(Z, I2)+np.kron(I2, Z))
    #hlocs = (hloc1, hloc2, hloc3)
    hloc1 = construct_heisenberg_local_term((0, 0   ,    0), (0, h[1],    0), ndim=2)
    hloc2 = construct_heisenberg_local_term((J[0], 0   ,    0), (0, 0,    0), ndim=2)
    hloc3 = construct_heisenberg_local_term((0   ,    0, 0), (0, 0, h[2]   ), ndim=2)
    hloc4 = construct_heisenberg_local_term((0   ,    J[1], 0), (0, 0, 0   ), ndim=2)
    hloc5 = construct_heisenberg_local_term((0   , 0   , 0), (h[0], 0,    0), ndim=2)
    hloc6 = construct_heisenberg_local_term((0   , 0   , J[2]), (0, 0,    0), ndim=2)
    hlocs = (hloc1, hloc2, hloc3, hloc4, hloc5, hloc6)
    
    # Suzuki splitting
    if trotter_order > 1:
        sm = oc.SplittingMethod.suzuki(len(hlocs), int(np.log(trotter_order)/np.log(2)))
        indices, coeffs = sm.indices, sm.coeffs
    else:
        indices, coeffs = range(len(hlocs)), [1]*len(hlocs)
        
    Vlist_start = []
    for i, c in zip(indices, coeffs):
        Vlist_start.append(-1j*c*dt*hlocs[i])
    

    for n in range(nsteps):
        for layer, V in enumerate(Vlist_start):
            i = n*len(Vlist_start)+layer
            for perm in perms:
                
                edges = [(perm[2*j], perm[2*j+1]) for j in range(len(perm)//2)]
                H2 = {edge: V for edge in edges}
                ham = LocalHamGen(H2=H2, H1=None)
                tebd = TEBDGen(peps, ham=ham, D=max_bond_dim)
                tebd.sweep(tau=-1)
                peps = tebd.state

                del tebd, ham
                gc.collect()
    return peps


def ccU(peps, Vlist, perms_extended, control_layers, dagger=False, max_bond_dim=10):
    for i, V in enumerate(Vlist):
        if dagger or i not in control_layers:
            perms = perms_extended[i]
            for perm in perms:
                edges = [(perm[2*j], perm[2*j+1]) for j in range(len(perm)//2)]
                H2 = {edge: scipy.linalg.logm(V) for edge in edges}
                ham = LocalHamGen(H2=H2, H1=None)
                tebd = TEBDGen(peps, ham=ham, D=max_bond_dim)
                tebd.sweep(tau=-1)
                peps = tebd.state

                del tebd, ham
                gc.collect()
    return peps

"""
if layers==36:
    perms_extended = [[perms_1[0]]] + [[perms_1[0]]]+ [[perms_1[1]]] + [[perms_1[0]], [perms_2[0]]] +\
      [[perms_2[0]]]+ [[perms_2[1]]]  + [[perms_2[0]], [perms_3[0]]] + [[perms_3[0]]]+ [[perms_3[1]]]  + [[perms_3[0]]]
    perms_extended = perms_extended*3
    perms_ext_reduced =  [[perms_1[0]]]+ [[perms_1[1]]]  +  [[perms_2[0]]]+ [[perms_2[1]]]  +  [[perms_3[0]]]+ [[perms_3[1]]] 
    perms_ext_reduced = perms_ext_reduced*3
elif layers==72:
    perms_extended = [[perms_1[0]]] + [perms_1] + [[perms_1[0]], [perms_2[0]]] +\
	      [perms_2] + [[perms_2[0]], [perms_3[0]]] +  [perms_3] + [[perms_3[0]]]
    perms_extended = perms_extended*5
    perms_ext_reduced = [perms_1] + [perms_2] + [perms_3]
    perms_ext_reduced = perms_ext_reduced*5
non_control_layers = []
k = 0
while True:
    a = 1 + 4*k
    b = 2 + 4*k
    if a > layers or b > layers:
        break
    non_control_layers.extend([a, b])
    k += 1
control_layers = []
for i in range(len(perms_extended)):
    if i not in non_control_layers:
        control_layers.append(i)
"""
if layers==22:
    control_layers = [0, 7, 14, 21]
    perms_ext_reduced = ps*3
    perms_extended = [p2] + ps  + [p3] + ps  + [p5]  + ps + [p2]
elif layers==43:
    control_layers = list(range(0, 43, 7))
    perms_ext_reduced = [p1, p2, p3, p4, p5, p6]*6
    perms_extended = [p2] + ps  + [p3] + ps  + [p5]  + ps + [p2]  + ps + [p3] +  ps + [p5] + ps + [p2]
elif layers==64:
    control_layers = list(range(0, 64, 7))
    perms_ext_reduced = [p1, p2, p3, p4, p5, p6]*9
    perms_extended = [p2] + ps  + [p3] + ps  + [p5]  + ps + [p2]  + ps + [p3] +  ps + [p5] + ps + [p2] + ps  + [p3] + ps  + [p5]  + ps + [p2] 


with h5py.File(
    #f'../results/kagome_Heis_L12_t{t}_layers{layers}.hdf5'
    f'../results/kagome_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L12_t{t}_layers{layers}_rS{rS if rS is not None else ""}_opt_SHORT.hdf5'
    ) as f:
    Vlist  =  f["Vlist"][:]
Vlist_reduced = []
for i, V in enumerate(Vlist):
    if i not in control_layers:
        Vlist_reduced.append(V)


peps, (p1, p2, p3) = build_triangular_PEPS(1, 2)
peps_copy_norm = peps.copy()
ov_tn = peps_copy_norm.make_overlap(
    peps_copy_norm,
    layer_tags=("KET", "BRA"),
)
overlap_approx = ov_tn.contract_compressed(
    optimize="auto-hq",
    max_bond=chi_overlap,
    cutoff=1e-10,
)
norm = np.sqrt(abs(overlap_approx))
peps = peps/np.abs(norm)

ov_tn = peps.make_overlap(
    peps,
    layer_tags=("KET", "BRA"),
)
overlap_approx = ov_tn.contract_compressed(
    optimize="auto-hq",
    max_bond=chi_overlap,
    cutoff=1e-10,
)


peps_E = peps.copy()
peps_T = peps.copy()
peps_C = peps.copy()
peps_E = trotter(peps_E.copy(), -t if reverse else t, L,  J, perms_1+perms_2+perms_3,
                     dt=t/trotter_step_ref, max_bond_dim=BD, trotter_order=trotter_order_ref)
peps_aE = ccU(peps_C.copy(), Vlist if reverse else Vlist_reduced, 
    perms_extended if reverse else perms_ext_reduced, [], dagger=False, max_bond_dim=BD)
peps_T = trotter(peps_T.copy(), -t if reverse else t, L,  J, perms_1+perms_2+perms_3,
                     dt=t/trotter_step, max_bond_dim=BD, trotter_order=trotter_order)
peps_T.compress_all(max_bond=BD)
peps_E.compress_all(max_bond=BD)
peps_aE.compress_all(max_bond=BD)


ov_tn = peps_E.make_overlap(
    peps_aE,
    layer_tags=("KET", "BRA"),
)
for chi_overlap in range(chi_overlap1, chi_overlap2, chi_overlap_incr):
    overlap_approx = ov_tn.contract_compressed(
        optimize="hyper-compressed",
        max_bond=chi_overlap,
        cutoff=cutoff,
    )
    with open(f"{L}_PEPS_log.txt", "a") as file:
        file.write("\n Fidelity for TICC: "+str(np.abs(overlap_approx)) + f", BD={BD}, chi_overlap={chi_overlap} \n")

ov_tn = peps_E.make_overlap(
    peps_T,
    layer_tags=("KET", "BRA"),
)
for chi_overlap in range(chi_overlap1, chi_overlap2, chi_overlap_incr):
    overlap_approx = ov_tn.contract_compressed(
        optimize="hyper-compressed",
        max_bond=chi_overlap,
        cutoff=cutoff,
    )
    with open(f"{L}_PEPS_log.txt", "a") as file:
        file.write(f"Fidelity for Trotter {trotter_order}: "+str(np.abs(overlap_approx)) + f", BD={BD}, chi_overlap={chi_overlap} \n")


