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


import rqcopt as oc
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)

def trotter(peps, t, L, Lx, Ly, J, g, perms_v, perms_h, dag=False, max_bond_dim=5, 
            dt=0.1, trotter_order=2, treshold=10, lower_max_bond_dim=4, imag=False):
    nsteps = np.abs(int(np.ceil(t/dt)))
    t = t/nsteps
    if trotter_order > 1:
        indices = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2))).indices
        coeffs = oc.SplittingMethod.suzuki(2, int(np.log(trotter_order)/np.log(2))).coeffs
    else:
        indices = [0, 1]
        coeffs = [1, 1]
    
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
                ordering = {(map_[perm[2*j]], map_[perm[2*j+1]]): V for j in range(len(perm)//2)}
                start = time.time()
                t = TEBD2D(peps, ham=LocalHam2D(Lx, Ly, ordering, cyclic=True),
                    tau=1, D=max_bond_dim if i<treshold else lower_max_bond_dim, chi=1)
                t.sweep(tau=1j if imag else -1)
                peps = t.state
                
            for perm in perms_v:
                ordering = {(map_[perm[2*j]], map_[perm[2*j+1]]): V for j in range(len(perm)//2)}
                start = time.time()
                t = TEBD2D(peps, ham=LocalHam2D(Lx, Ly, ordering, cyclic=True),
                    tau=1, D=max_bond_dim if i<treshold else lower_max_bond_dim, chi=1)
                t.sweep(tau=1j if imag else -1)
                peps = t.state
            #with open(f"trotter_PEPS_log{Lx}{Ly}.txt", "a") as file:
            #    file.write(f"Time step {n}/{nsteps}, layer {layer}/{len(Vlist_start)} applied \n")
            #print(f"Time step {n}/{nsteps}, layer {layer}/{len(Vlist_start)} applied")
    print(f"Norm. Trotter")
    #peps.normalize()
    peps = peps / peps.norm()
    print(f"norm finished Trotter")
    return peps


def run_adiabatic(peps_init_state, Lx, Ly, T, S, perms_v, perms_h, J_i=J, h_i=0, g_i=0, J_f=J, h_f=0, g_f=g,
                 lower_max_bond_dim=3, max_bond_dim=3
                 ):
    L = Lx*Ly
    tau = 1/S
    t_s = np.linspace(0, T, S*T)
    sch = lambda t, T: np.sin(np.pi*t/(2*T))**2
    
    peps = peps_init_state.copy()
    for s in range(S*T):
        lamb = sch(t_s[s], T)
        assert lamb <= 1 and lamb >= 0
        J = lamb*J_f + (1-lamb)*J_i
        g = lamb*g_f + (1-lamb)*g_i
        h = lamb*h_f + (1-lamb)*h_i
        
        peps = trotter(peps, tau, L, Lx, Ly, J, g, perms_v, perms_h, max_bond_dim=max_bond_dim, 
            dt=tau, trotter_order=2, lower_max_bond_dim=lower_max_bond_dim)
    return peps

Lx, Ly = (6, 6)
L = Lx*Ly
latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
J, h, g = (1, 0, 1)

#perms_v, perms_h = get_perms(Lx, Ly)
perms_v, perms_h = (
    [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6, 13, 14, 15, 16, 17, 12, 19, 20, 21, 22, 23, 18, 25, 26, 27, 28, 29, 24, 31, 32, 33, 34, 35, 30]],
    [[0, 6, 12, 18, 24, 30, 1, 7, 13, 19, 25, 31, 2, 8, 14, 20, 26, 32, 3, 9, 15, 21, 27, 33, 4, 10, 16, 22, 28, 34, 5, 11, 17, 23, 29, 35], 
    [6, 12, 18, 24, 30, 0, 7, 13, 19, 25, 31, 1, 8, 14, 20, 26, 32, 2, 9, 15, 21, 27, 33, 3, 10, 16, 22, 28, 34, 4, 11, 17, 23, 29, 35, 5]]
)
map_ = {i: (i//Ly, i%Lx) for i in range(L)}

peps = qtn.PEPS.rand(Lx, Ly, bond_dim=1, phys_dim=2, cyclic=True)
peps = peps / peps.norm()
peps_E = peps.copy()


peps_E = trotter(peps_E.copy(), 15, L, Lx, Ly, J, g, perms_v, perms_h,
                     dt=0.1, max_bond_dim=3, lower_max_bond_dim=3, trotter_order=2, imag=True)

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
ZZ1 = J*peps_E.compute_local_expectation({((2, 2), (2, 3)): np.kron(Z, Z)})
ZZ2 = J*peps_E.compute_local_expectation({((2, 2), (3, 2)): np.kron(Z, Z)})
gX = g*peps_E.compute_local_expectation({((2, 2)): X})
e = (ZZ2 + ZZ1 + gX)
with open(f"PEPS-TEBD_log{Lx}{Ly}.txt", "a") as file:
    file.write(f"Energy: {e} \n")



zero = np.array([1, 0])
one = np.array([0, 1])
site_map = {map_[i]: one for i in range(L)}
for y in range(Ly):
    for x in range(Lx):
        i = Ly * y + x
        if (x + y) % 2 == 1:
            site_map[map_[i]] = zero
peps_product1 = qtn.PEPS.product_state(site_map)
site_map = {map_[i]: zero for i in range(L)}
for y in range(Ly):
    for x in range(Lx):
        i = Ly * y + x
        if (x + y) % 2 == 1:
            site_map[map_[i]] = one
peps_product2 = qtn.PEPS.product_state(site_map)
with open(f"PEPS-TEBD_log{Lx}{Ly}.txt", "a") as file:
    file.write(f"Fidelities with Product State: {np.abs(peps_product2.overlap(peps_E))**2} \n")
    file.write(f"{np.abs(peps_product1.overlap(peps_E))**2} \n\n")


peps_A = run_adiabatic(peps_product1, Lx, Ly, 2, 2, perms_v, perms_h, J_i=J, g_i=0, J_f=J, g_f=g,
	lower_max_bond_dim=3, max_bond_dim=3)
with open(f"PEPS-TEBD_log{Lx}{Ly}.txt", "a") as file:
    file.write(f"Fidelities with AQC State: \n")
    file.write(f"{np.abs(peps_A.overlap(peps_E))**2} \n\n")



