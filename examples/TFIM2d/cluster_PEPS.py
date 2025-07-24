import numpy as np
import quimb.tensor as qtn
import quimb
from quimb.tensor.tensor_2d_tebd import TEBD2D, LocalHam2D
import scipy
import h5py
import qib
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
import rqcopt as oc
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)
import time
import tracemalloc
import os
tracemalloc.start()

#max_bond_dim_T, lower_max_bond_dim_T, treshold_T = (5, 4, 10)
#max_bond_dim_C, lower_max_bond_dim_C, treshold_C = (4, 3, 8)
max_bond_dim_T, lower_max_bond_dim_T, treshold_T = (3, 3, 10)
max_bond_dim_C, lower_max_bond_dim_C, treshold_C = (3, 3, 8)

nsteps, order = (8, 2)                
Vlist = []
with h5py.File(f"./results/tfim2d_ccU_SPARSE_103_Lx4Ly4_t0.25_layers15_rS1_niter15_3hloc.hdf5", "r") as f:
    Vlist =  f["Vlist"][:]
control_layers = [0, 4, 5, 9, 10, 14]
perms_qc = [[0, 1], [0, 2], [1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [1, 2]]
Xlists_opt = {}
for i in control_layers:
    with h5py.File(f"./results/tfim2d_ccU_SPARSE_103_Lx4Ly4_t0.25_layers15_niter20_rS5_DECOMPOSE_n9_layer{i}.hdf5", "r") as file:
        Xlists_opt[i] = file[f"Xlist_{i}"][:]

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
                    [perms_h]*3 + [[perms_h[0]], [perms_v[0]]] + [perms_v]*3 + [[perms_v[0]]]
perms_ext_reduced = [perms_v]*3  + [perms_h]*3 + [perms_v]*3
map_ = {i: (i//Ly, i%Lx) for i in range(L)}


def trotter(peps, t, L, Lx, Ly, J, g, perms_v, perms_h, dag=False, max_bond_dim=5,
            dt=0.1, trotter_order=2, treshold=10, lower_max_bond_dim=4):
    nsteps = np.abs(int(np.ceil(t/dt)))
    t = t/nsteps
    indices = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2))).indices
    coeffs = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2))).coeffs

    hloc1 = g*(np.kron(X, I2)+np.kron(I2, X))/4
    hloc2 = J*np.kron(Z, Z)
    hlocs = (hloc1, hloc2)
    Vlist_start = []
    for i, c in zip(indices, coeffs):
        Vlist_start.append(-1j*c*t*hlocs[i])

    for n in range(nsteps):
        for layer, V in enumerate(Vlist_start):
            i = n*len(Vlist_start)+layer
            for perm in perms_h:
                ordering = {(map_[perm[2*j]], map_[perm[2*j+1]]): V for j in range(L//2)}
                start = time.time()
                t = TEBD2D(peps, ham=LocalHam2D(Lx, Ly, ordering, cyclic=True),
                    tau=-1, D=max_bond_dim if i<treshold else lower_max_bond_dim, chi=1)
                t.sweep(tau=-1)
                peps = t.state

            for perm in perms_v:
                ordering = {(map_[perm[2*j]], map_[perm[2*j+1]]): V for j in range(L//2)}
                start = time.time()
                t = TEBD2D(peps, ham=LocalHam2D(Lx, Ly, ordering, cyclic=True),
                    tau=-1, D=max_bond_dim if i<treshold else lower_max_bond_dim, chi=1)
                t.sweep(tau=-1)
                peps = t.state
            with open(f"trotter_PEPS_log{Lx}{Ly}.txt", "a") as file:
                file.write(f"Time step {n}/{nsteps}, layer {layer}/{len(Vlist_start)} applied \n")
    peps /= peps.norm()
    return peps


def ccU(peps, Vlist, perms_extended, control_layers, dagger=False, max_bond_dim=10, lower_max_bond_dim=4, treshold=10):
    for i, V in enumerate(Vlist):
        if dagger or i not in control_layers:
            perms = perms_extended[i]
            for perm in perms:
                ordering = {(map_[perm[2*j]], map_[perm[2*j+1]]): scipy.linalg.logm(V) for j in range(L//2)}
                start = time.time()
                t = TEBD2D(peps, ham=LocalHam2D(Lx, Ly, ordering, cyclic=True),
                    tau=-1, D=max_bond_dim if i<treshold else lower_max_bond_dim, chi=1)
                t.sweep(tau=-1)
                peps = t.state
                with open(f"ccU_PEPS_log{Lx}{Ly}.txt", "a") as file:
                    file.write(f"Step {i} took {time.time() - start:.2f} seconds")
    peps /= peps.norm()
    return peps


# Get and save initial peps
if not os.path.isfile(f"./PEPS/init_peps.h5") or True: #TODO: remove this later
    peps = qtn.PEPS.rand(Lx, Ly, bond_dim=1, phys_dim=2, cyclic=True)
    peps /= peps.norm()
    with h5py.File(f'./PEPS/init_peps.h5', 'w') as f:
        for site, t in enumerate(peps.tensors):
            dset_name = f"site_{site}"
            dset = f.create_dataset(dset_name, data=t.data)
            inds_ascii = [ind.encode('ascii', 'ignore') for ind in t.inds]
            dset.attrs['inds'] = inds_ascii


# Get and save peps_trotter
if not os.path.isfile(f"./PEPS/trotter_peps_D_init={max_bond_dim_T}_D_late={lower_max_bond_dim_T}_treshold={treshold_T}.h5") or True: #TODO: remove this later
    peps_trotter = trotter(peps.copy(), -0.25, L, Lx, Ly, J, g, perms_v, perms_h, trotter_order=order,
        dt=0.25/nsteps, max_bond_dim=max_bond_dim_T, lower_max_bond_dim=lower_max_bond_dim_T, treshold=treshold_T)
    with h5py.File(f'./PEPS/trotter_peps_D_init={max_bond_dim_T}_D_late={lower_max_bond_dim_T}_treshold={treshold_T}.h5', 'w') as f:
        for site, t in enumerate(peps_trotter.tensors):
            dset_name = f"site_{site}"
            dset = f.create_dataset(dset_name, data=t.data)
            inds_ascii = [ind.encode('ascii', 'ignore') for ind in t.inds]
            dset.attrs['inds'] = inds_ascii

# Get and save peps_ccU
if not os.path.isfile(f'./PEPS/ccU_PEPS_D_init={max_bond_dim_C}_D_late={lower_max_bond_dim_C}_treshold={treshold_C}.h5') or True: #TODO: remove this later
    peps_ccU = ccU(peps.copy(), Vlist, perms_extended, control_layers,
        dagger=True, max_bond_dim=max_bond_dim_C, lower_max_bond_dim=lower_max_bond_dim_C, treshold=treshold_C)
    with h5py.File(f'./PEPS/ccU_PEPS_D_init={max_bond_dim_C}_D_late={lower_max_bond_dim_C}_treshold={treshold_C}.h5', 'w') as f:
        for site, t in enumerate(peps_ccU.tensors):
            dset_name = f"site_{site}"
            dset = f.create_dataset(dset_name, data=t.data)
            inds_ascii = [ind.encode('ascii', 'ignore') for ind in t.inds]
            dset.attrs['inds'] = inds_ascii


#f = np.linalg.norm(peps_ccU.overlap(peps_trotter,  contract='auto-hq'))
block_sites = [(2,2), (2,3), (3,2), (3,3)]
rho_trotter = peps_trotter.partial_trace(block_sites, optimize='auto-hq', max_bond=lower_max_bond_dim_T)
rho_trotter /= np.trace(rho_trotter)
rho_ccU = peps_ccU.partial_trace(block_sites, optimize='auto-hq', max_bond=lower_max_bond_dim_C)
rho_ccU /= np.trace(rho_ccU)

f = np.abs(np.trace(rho_ccU.conj().T @ rho_trotter))

with open(f"combined_PEPS_log{Lx}{Ly}.txt", "a") as file:
    file.write(f"\n Fidelity, Backwards \n D_init={max_bond_dim_T}, D_late={lower_max_bond_dim_T}, treshold={treshold_T}, nsteps={nsteps}, order={order} for Trotter \n")
    file.write(f"D_init={max_bond_dim_C}, D_late={lower_max_bond_dim_C}, treshold={treshold_C} for ccU \n")
    file.write(f"Fidelity: {f}\n \n")
