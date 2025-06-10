import qiskit
from qiskit import Aer, execute, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.converters import circuit_to_dag

import numpy as np
import scipy
import h5py
import rqcopt as oc



def estimate_phases(L, prepared_state, eigenvalues_sort, t, tau,
    shots, depolarizing_error, qc_cU, return_counts=False, mid_cbits=0, 
    noise_model=None, get_cx=False, qasm=False,
    delta_tau=None
    ):
    backend = qiskit.Aer.get_backend("aer_simulator")

    if noise_model is None:
        x1_error = errors.depolarizing_error(depolarizing_error*0.1, 1)
        x2_error = errors.depolarizing_error(depolarizing_error, 2)
        x3_error = errors.depolarizing_error(depolarizing_error, 3)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(x1_error, ['u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(x2_error, ['cu', 'cx','cy', 'cz'])
        noise_model.add_all_qubit_quantum_error(x3_error, ['ccu', 'ccx','ccy', 'ccz'])
        print(noise_model)

    phase_estimates_with_noise = []
    phase_exacts = []
    counts_list = []

    state = prepared_state.copy()
    qc_cU_ins = qiskit.QuantumCircuit(L+1)
    nsteps = int(t/tau)
    print("t: ", t)
    print("nsteps: ", nsteps)
    for n in range(nsteps):
        qc_cU_ins.append(qc_cU.to_gate(), [i for i in range(L+1)])
    qpe_real, qpe_imag = qc_QPE(L, state, qc_cU_ins, mid_cbits=mid_cbits)

    count_ops = 0
    if get_cx:
        dag = circuit_to_dag(transpile(qpe_real, basis_gates=noise_model.basis_gates+['unitary', 'initialize']))
        count_ops = dag.count_ops()

    print("getting counts")
    counts_real = execute(transpile(qpe_real), backend, noise_model=noise_model, shots=shots).result().get_counts()
    counts_imag = execute(transpile(qpe_imag), backend, noise_model=noise_model, shots=shots).result().get_counts()

    if return_counts:
        if get_cx:
            if qasm:
                counts_list.append((counts_real, counts_imag, count_ops, (qpe_real.qasm(), qpe_imag.qasm())))
            else:
                counts_list.append((counts_real, counts_imag, count_ops))
        else:
            counts_list.append((counts_real, counts_imag))
    else:
        
        phase_est_real = (counts_real["0"] if "0" in counts_real.keys() else 0)/shots -\
         (counts_real["1"] if "1" in counts_real.keys() else 0)/shots
        
        phase_est_imag = (counts_imag["0"] if "0" in counts_imag.keys() else 0)/shots -\
         (counts_imag["1"] if "1" in counts_imag.keys() else 0)/shots
        
        phase_est = phase_est_real + 1j*phase_est_imag
        phase_estimates_with_noise.append((t, phase_est))
        exact_phase = np.exp(-1j*t*eigenvalues_sort[0])
        phase_exacts.append((t, exact_phase))

    if return_counts:
        return counts_list
    else:
        return phase_estimates_with_noise, phase_exacts



def qc_QPE(L, qpe_real, qc_cU, mid_cbits=0):
    qpe_real.h(L)
    qpe_imag = qpe_real.copy()
    qpe_imag.append(qc_cU.to_gate(), [i for i in range(L+1)])
    qpe_real.append(qc_cU.to_gate(), [i for i in range(L+1)])
    qpe_imag.p(-0.5*np.pi, L)
    qpe_imag.h(L)
    qpe_real.h(L)
    qpe_real.measure(L, mid_cbits)
    qpe_imag.measure(L, mid_cbits)
    return qpe_real, qpe_imag

