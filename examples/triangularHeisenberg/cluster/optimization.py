import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from functools import reduce
import sys
import qib
import h5py
import scipy
sys.path.append("../../../src/brickwall_ansatz")
from utils import construct_heisenberg_local_term
sys.path.append("../../../src/brickwall_sparse")
from optimize_sparse import optimize
from ansatz_sparse import ansatz_sparse
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm
from qiskit.quantum_info import state_fidelity



perms_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]]
perms_2 = [[0, 5, 10, 15, 3, 4, 9, 14, 2, 7, 8, 13, 1, 6, 11, 12], [5, 10, 15, 0, 4, 9, 14, 3, 7, 8, 13, 2, 6, 11, 12, 1]]
perms_3 = [[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], [4, 8, 12, 0, 5, 9, 13, 1, 6, 10, 14, 2, 7, 11, 15, 3]]
p1, p2, p3, p4, p5, p6 = ([perms_1[0]], [perms_1[1]], [perms_2[0]], [perms_2[1]], [perms_3[0]], [perms_3[1]])
ps = [p1, p2, p3, p4, p5, p6]

Lx, Ly = (4, 4)
L = Lx*Ly
J = (1, 1, 1)
h = (3, -1, 1)
latt = qib.lattice.TriangularLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
hamil = qib.HeisenbergHamiltonian(field, J, h).as_matrix()

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
YZ = np.kron(Y, Z)
I2 = np.array([[1, 0], [0, 1]])
XZ = np.kron(X, Z)
XY = np.kron(X, Y)
ZY = np.kron(Z, Y)

state = np.array(random_statevector(2**L).data)
hloc = construct_heisenberg_local_term((J[0], J[1], J[2]), (h[0], h[1], h[2]), ndim=3)


def exec(t, layers, rS=1, result_string=None, custom_result_string='', bootstrap=False, niter=50, hessian = True):
    V = scipy.linalg.expm(-1j*t*hloc/(layers//4))
    Vlist_reduced = [V for i in range(layers)]

    if layers==22:
        Vlist_start = [np.eye(4), V, V, V, V, V, V, np.eye(4), 
            V, V, V, V, V, V, np.eye(4), V, V, V, V, V, V, np.eye(4)]
        control_layers = list(range(0, 22, 7))
        perms_reduced = ps*3
        perms_ext = [p2] + ps + [p3] + ps  + [p5] + ps  + [p2]

    elif layers==43:
        control_layers = list(range(0, 43, 7))
        perms_reduced = ps*6
        perms_ext = [p2] + ps + [p3] + ps  + [p5] + ps  + [p2] + ps + [p3] + ps  + [p5] + ps  + [p2]
        with h5py.File(f"../results/triang_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_L{L}_t0.05_layers22_{custom_result_string}.hdf5", 'r') as f:
                Vlist_start_2  =  f["Vlist"][:]
        Vlist_start = list(Vlist_start_2) + list(Vlist_start_2)[1:]
        Vlist_start[21] = Vlist_start_2[0] @ Vlist_start_2[-1]

    if result_string is not None:
        with h5py.File(f'../results/{result_string}') as f:
            Vlist_start_2  =  f["Vlist"][:]
    else:
        Vlist_start_2 = Vlist_start

    Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_start_2, perms_ext, perms_reduced=perms_reduced,
                                       control_layers=control_layers, rS=rS, niter=niter, log_txt=custom_result_string,
                                       hessian=hessian)

    with h5py.File(f"../results/triang_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_L{L}_t{t}_layers{len(Vlist)}_{custom_result_string}.hdf5", "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        f.attrs["L"] = L
        f.attrs["t"] = float(t)


custom_result_string = "_gamma4_"
bootstrap = False
rS = 1
hessian = True


exec(0.05, 22, rS=rS, result_string="triang_Heis1113-11_L16_L16_t0.05_layers22__gamma4_.hdf5", 
    custom_result_string=custom_result_string, bootstrap=bootstrap, 
    niter=50, hessian=hessian)

exec(0.1, 43, rS=rS, result_string=None, 
    custom_result_string=custom_result_string, bootstrap=bootstrap, 
    niter=20, hessian=hessian)

