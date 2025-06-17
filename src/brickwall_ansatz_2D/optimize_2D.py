import numpy as np
from ansatz_2D import ansatz_2D


def err(vlist, U, L, perms):
    f_base = -np.trace(U.conj().T @ ansatz_2D(vlist, L, perms)).real
    return f_base