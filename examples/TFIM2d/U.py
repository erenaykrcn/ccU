import qiskit
from qiskit.quantum_info import state_fidelity
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy
import h5py

import sys
sys.path.append("../../src/brickwall_ansatz")
from optimize import optimize, dynamics_opt
from utils import construct_ising_local_term, reduce_list
from ansatz import ansatz
import rqcopt as oc

J, h, g = (1, 0, 1)
t = .25
Lx, Ly = (4, 4)
L = Lx*Ly
latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
hamil = qib.IsingHamiltonian(field, J, h, g).as_matrix()
U = scipy.sparse.linalg.expm(-1j*t*hamil)


