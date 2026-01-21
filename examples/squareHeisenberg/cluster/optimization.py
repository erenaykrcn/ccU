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
from utils_sparse import get_perms
from ansatz_sparse import ansatz_sparse
import rqcopt as oc
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import random_statevector
from scipy.linalg import expm
from qiskit.quantum_info import state_fidelity


custom_result_string = "p2"
bootstrap = False
niter = 20
t = 0.1
layers = 12
rS = 1
hessian = True

Lx, Ly = (4, 4)
L = Lx*Ly
J = (1, 1, 1)
h = (3, -1, 1)

result_string = f"square_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_L{L}_t0.1_layers17_None.hdf5"
#result_string = None

latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
hamil = qib.HeisenbergHamiltonian(field, J, h).as_matrix()

perms_1, perms_2 = get_perms(Lx, Ly)
p1, p2, p3, p4 = ([perms_1[0]], [perms_1[1]], [perms_2[0]], [perms_2[1]])
ps = [p1, p2, p3, p4]

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
YZ = np.kron(Y, Z)
I2 = np.array([[1, 0], [0, 1]])
XZ = np.kron(X, Z)
XY = np.kron(X, Y)
ZY = np.kron(Z, Y)

state = np.array(random_statevector(2**L).data)
hloc = construct_heisenberg_local_term((J[0], J[1], J[2]), (h[0], h[1], h[2]), ndim=2)
V = scipy.linalg.expm(-1j*t*hloc/(layers//4))
Vlist_reduced = [V for i in range(layers)]


if bootstrap:
    if layers==24:
        control_layers = list(range(0, 17, 4)) + [17+i for i in range(0, 17, 4)]
        print(control_layers)
        perms_reduced = ps*6
        perms_ext = [p2] + [p1, p2, p3] + [p1] + [p4, p1, p2]  + [p4]  + [p3, p4, p1] + [p3] + [p2, p3, p4] + [p1] +\
            [p2] + [p1, p2, p3] + [p1] + [p4, p1, p2]  + [p4]  + [p3, p4, p1] + [p3] + [p2, p3, p4] + [p1]

        with h5py.File(f"../results/square_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_L{L}_t0.1_layers17_None.hdf5", 'r') as f:
            Vlist_start_2  =  f["Vlist"][:]
        #Vlist_start = list(Vlist_start_2) + list(Vlist_start_2)[1:]
        #Vlist_start[16] = Vlist_start_2[0] @ Vlist_start_2[-1]
        Vlist_start = list(Vlist_start_2) + list(Vlist_start_2)

elif layers==4:
    Vlist_start = [np.eye(4), V, V, np.eye(4), V, V, np.eye(4)]
    control_layers = list(range(0, 7, 3))
    perms_reduced = ps
    perms_ext = [p2] + [p1, p2] + [p4] + [p3, p4]  + [p2]

elif layers==8:
    Vlist_start = [np.eye(4), V, V, V, V, np.eye(4), V, V, V, V, np.eye(4)]
    control_layers = list(range(0, 11, 5))
    perms_reduced = ps*2
    perms_ext = [p2] + ps + [p3] + ps  + [p2]


elif layers==12:
    Vlist_start = [np.eye(4), V, V, V, np.eye(4), V, V, V, np.eye(4), V, V, V, np.eye(4), V, V, V, np.eye(4)]
    control_layers = list(range(0, 17, 4))
    perms_reduced = ps*3
    perms_ext = [p2] + [p1, p2, p3] + [p1] + [p4, p1, p2]  + [p4]  + [p3, p4, p1] + [p3] + [p2, p3, p4] + [p2]

elif layers==24:
    #Vlist_start = [np.eye(4), V, V, V, V, V, V, np.eye(4), V, V, V, V, V,V, np.eye(4), V, V, V, V, V, V,
    #               np.eye(4), V, V, V, V, V,V, np.eye(4), V, V, V, V, V,V, np.eye(4), V, V, V, V, V,V, np.eye(4)]
    control_layers = list(range(0, 31, 5))
    perms_reduced = ps*6
    perms_ext = [p2] + ps  + [p3] + ps  + [p2]  + ps + [p3]  + ps + [p2] +  ps + [p3] + ps + [p1]
    with h5py.File(f"../results/kagome_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_t{t/2}_layers16_rS{rS}_opt_SHORT{custom_result_string}.hdf5", 'r') as f:
        Vlist_start_2  =  f["Vlist"][:]
    Vlist_start = list(Vlist_start_2) + list(Vlist_start_2)[1:]
    Vlist_start[15] = Vlist_start_2[0] @ Vlist_start_2[-1]

elif layers==36:
    control_layers = list(range(0, 46, 5))
    print('Control Layers: ', control)
    perms_reduced = ps*9
    perms_ext = [p2] + ps  + [p3] + ps  + [p2]  + ps + [p3]  + ps + [p2] +  ps + [p3] + ps + [p2] + ps  + [p3] + ps  + [p2]  + ps + [p3] 
    with h5py.File(f"../results/kagome_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_t{round(t/3, 3)}_layers16_rS{rS}_opt_SHORT{custom_result_string}.hdf5", 'r') as f:
        Vlist_start_2  =  f["Vlist"][:]

    Vlist_start = list(Vlist_start_2) + list(Vlist_start_2)[1:] + list(Vlist_start_2)[1:]
    Vlist_start[15] = Vlist_start_2[0] @ Vlist_start_2[-1]
    Vlist_start[31] = Vlist_start_2[0] @ Vlist_start_2[-1]


#print("Trotter error of the starting point: ", 1-state_fidelity(ansatz_sparse(Vlist_start, L, perms_ext, state), expm_multiply(
#    1j * t * hamil, state)))
#print("Trotter error of the starting point: ", 1-state_fidelity(ansatz_sparse(Vlist_reduced, L, perms_reduced, state), expm_multiply(
#    -1j * t * hamil, state)))
#print("Trotter error of the starting point: ", (np.linalg.norm(ansatz_sparse(Vlist_start, L, perms_extended, state) - expm_multiply(
#    1j * t * hamil, state), ord=2) + np.linalg.norm(ansatz_sparse(Vlist_reduced, L, perms_ext_reduced, state) - expm_multiply(
#    -1j * t * hamil, state), ord=2))/2)


if result_string is not None:
    with h5py.File(f'../results/{result_string}') as f:
        Vlist_start_2  =  f["Vlist"][:]
else:
    Vlist_start_2 = Vlist_start

Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_start_2, perms_ext, perms_reduced=perms_reduced,
                                   control_layers=control_layers, rS=rS, niter=niter, log_txt=custom_result_string,
                                   hessian=hessian)

with h5py.File(f"../results/square_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_L{L}_t{t}_layers{len(Vlist)}_{custom_result_string}.hdf5", "w") as f:
    f.create_dataset("Vlist", data=Vlist)
    f.create_dataset("f_iter", data=f_iter)
    f.create_dataset("err_iter", data=err_iter)
    f.attrs["L"] = L
    f.attrs["t"] = float(t)

