"""
    Adiabatic Evolution Implementation.
"""
from utils_gsp import construct_heisenberg_local_term

import numpy as np
import qiskit
from qiskit import Aer, execute, transpile
from qiskit.converters import circuit_to_dag
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.quantum_info import state_fidelity
import rqcopt as oc
import scipy



def trotter(tau, L, lamb, hloc):
    """
        This Trotterization slowly enables the 
        second permutation as lamb goes 0 -> 1.
    """

    assert lamb <= 1 and lamb >= 0
    qc = qiskit.QuantumCircuit(L)
    
    # permutations specifying gate layout
    perms1 = [i for i in range(L)]
    perms2 = [i for i in range(1, L)]+[0]
    perm_set = [perms1, perms2]
    perms = perm_set

    method_start = oc.SplittingMethod.suzuki(len(perms), 1)
    indices = method_start.indices
    coeffs = method_start.coeffs

    # unitaries used as starting point for optimization
    Vlist_start = []
    perms = []
    
    for i, c in zip(indices, coeffs):
        Vlist_start.append(
            scipy.linalg.expm(-1j*c*tau*hloc *\
                (lamb if i!=0 else 1))
        )
        perms.append(perm_set[i])
    Vlist_gates = []
    for V in Vlist_start:
        qc2 = qiskit.QuantumCircuit(2)
        qc2.unitary(V, [0, 1], label='str')
        Vlist_gates.append(qc2)
    
    for layer, qc_gate in enumerate(Vlist_gates):     
        for j in range(len(perms[layer])//2):
            qc.append(qc_gate.to_gate(), [L-(perms[layer][2*j]+1), 
                                          L-(perms[layer][2*j+1]+1)])
    
    return qc



def run_adiabatic(L, T, S, hloc, init_state, return_state=False, eigenvectors_sort=None):
    tau = 1/S
    t_s = np.linspace(0, T, S*T)
    sch = lambda t, T: np.sin(np.pi*t/(2*T))**2
    
    qc = qiskit.QuantumCircuit(L)
    for s in range(S*T):
        qc.append(trotter(tau, L, sch(t_s[s], T), hloc).to_gate(),
                  [i for i in range(L)])
    
    backend = Aer.get_backend("statevector_simulator")
    qc_ = qiskit.QuantumCircuit(L)
    qc_.initialize(init_state)
    qc_.append(qc.to_gate(), [i for i in range(L)])
    final_state = execute(transpile(qc_), backend).result().get_statevector().data

    if eigenvectors_sort is not None:
        toPlot = [state_fidelity(final_state, eigenvectors_sort[:, i]) for i in range(10)]
        print("Resulting Ground State Fidelity: ", np.sum(np.array([toPlot[i] for i in range(1)])))

    if return_state:
        return qc
    else:
        noise_model = NoiseModel()
        dag = circuit_to_dag(transpile(qc_, basis_gates=noise_model.basis_gates+['unitary', 
            'initialize', 'cx']))
        count_ops = dag.count_ops()
        return [state_fidelity(final_state, eigenvectors_sort[:, i]) for i in range(10)], 
        {"gates": count_ops['unitary']}, final_state
