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


L = 6
J = (1, 1, 1)
h = (3, -1, 1)

latt = qib.lattice.IntegerLattice((L, ), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
hamil = qib.HeisenbergHamiltonian(field, J, h).as_matrix()

p1, p2 = ([list(range(L))], [list(range(1, L))+[0]])
ps = [p1, p2]

X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
YZ = np.kron(Y, Z)
I2 = np.array([[1, 0], [0, 1]])
XZ = np.kron(X, Z)
XY = np.kron(X, Y)
ZY = np.kron(Z, Y)

state = np.array(random_statevector(2**L).data)
hloc = construct_heisenberg_local_term((J[0], J[1], J[2]), (h[0], h[1], h[2]), ndim=1)

def exec(t, layers, rS=1, result_string=None, custom_result_string='', bootstrap=False, niter=50, hessian = True):
    V = scipy.linalg.expm(-1j*t*hloc/(layers//2))
    Vlist_reduced = [V for i in range(layers)]


    if layers==8:
        Vlist_start = [np.eye(4), V, V, np.eye(4), V, V, np.eye(4), V, V, np.eye(4), V, V, np.eye(4)]
        control_layers = list(range(0, layers+5, 3))
        perms_reduced = ps*4
        perms_ext = [p1] + ps + [p1] + ps  + [p1] + ps  + [p1] + ps  + [p1]
    elif layers==12:
        Vlist_start = [np.eye(4), V, V, V, np.eye(4), V, V, V, np.eye(4), V, V, V, np.eye(4), V, V, V, np.eye(4)]
        control_layers = list(range(0, layers+5, 4))
        ps_ = [p1, p2, p1]

        perms_reduced = ps_*4
        perms_ext = [p1] + ps_ + [p1] + ps_  + [p1] + ps_  + [p1] + ps_  + [p1]


    if result_string is not None:
        with h5py.File(f'../results/{result_string}') as f:
            Vlist_start_2  =  f["Vlist"][:]
    else:
        Vlist_start_2 = Vlist_start

    Vlist, f_iter, err_iter = optimize(L, hamil, t, Vlist_start_2, perms_ext, perms_reduced=perms_reduced,
                                       control_layers=control_layers, rS=rS, niter=niter, log_txt=custom_result_string,
                                       hessian=hessian)

    with h5py.File(f"../results/chain_Heis{J[0]}{J[1]}{J[2]}{h[0]}{h[1]}{h[2]}_L{L}_L{L}_t{t}_layers{len(Vlist)}_{custom_result_string}.hdf5", "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        f.attrs["L"] = L
        f.attrs["t"] = float(t)


custom_result_string = ""
bootstrap = False
niter = 50
rS = 10
hessian = True

exec(0.1, 8, rS=rS, result_string=None, 
    custom_result_string=custom_result_string, bootstrap=bootstrap, 
    niter=niter, hessian=hessian)
exec(0.1, 12, rS=rS, result_string=None, 
    custom_result_string=custom_result_string, bootstrap=bootstrap, 
    niter=niter, hessian=hessian)

exec(0.2, 8, rS=rS, result_string=None, 
    custom_result_string=custom_result_string, bootstrap=bootstrap, 
    niter=niter, hessian=hessian)
exec(0.2, 12, rS=rS, result_string=None, 
    custom_result_string=custom_result_string, bootstrap=bootstrap, 
    niter=niter, hessian=hessian)

exec(0.4, 8, rS=rS, result_string=None, 
    custom_result_string=custom_result_string, bootstrap=bootstrap, 
    niter=niter, hessian=hessian)
exec(0.4, 12, rS=rS, result_string=None, 
    custom_result_string=custom_result_string, bootstrap=bootstrap, 
    niter=niter, hessian=hessian)


