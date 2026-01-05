import qutip as qt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../src/B-Gate")
from MS import averaged_ms_fidelity


def B_qt(phi, theta):
    sx = qt.sigmax()
    sy = qt.sigmay()
    sigma_phi  = np.cos(phi) * sx + np.sin(phi) * sy
    phim = -phi 
    sigma_mphi = np.cos(phim) * sx + np.sin(phim) * sy # spin echo!
    H_eff1 = theta * qt.tensor(sigma_phi, sigma_phi)
    H_eff2 = theta * qt.tensor(sigma_mphi, sigma_mphi)
    U1 = (-1j * H_eff1).expm()
    U2 = (-1j * H_eff2).expm()
    return U2*U1


def run(T1, T2, det_error_frac, noiseless=False, gate='1', mot_deph=np.inf, ndot=1, z_sigma=2*np.pi*10):
    eta=0.06
    Omega=2*np.pi*100e3
    
    if noiseless:
        T1=np.inf
        T2=np.inf
        ndot=0
        det_error_frac = 0
        z_sigma=0
        mot_deph=np.inf,


    if gate=='B':
        reps, phases = 2, [np.arcsin(1/np.sqrt(3)), -np.arcsin(1/np.sqrt(3))]
        theta = np.pi/4
    elif gate=='1':
        reps, phases = 1, [0, np.pi]
        theta = np.pi/4
    else:
        reps, phases = 2, [0, np.pi]
        theta = np.pi/4
        

    delta = np.sqrt(np.pi/np.abs(theta))*Omega*eta * np.sqrt(reps)
    print("Gate time: ", 1e6*2*np.pi/delta*reps, "us")
        
    # The higher this ratio of: "det_error_frac/z_sigma", the better it is to do 2-loop
    F_mean, F_std = averaged_ms_fidelity(
                n_states=10, Nmode=10,
                nsteps=50, n_zshots=1,
        
                ideal_unitary=B_qt(phases[0], theta/2),
                
                eta=eta,
                Omega=Omega,
                delta=delta,
                T1=T1, T2=T2, mot_deph=mot_deph,
                detuning_error=det_error_frac*delta,
                heating_rate=ndot, z_sigma=z_sigma,
                correlated=False,
                
                reps=reps, phases=phases
    )
        
    #print(f"Average gate fidelity: {F_mean:.5f} ± {F_std:.5f}")
    #print(f"Average SU4 fidelity: {F_mean**(2 if gate=='B' else 3):.5f} ± {F_std:.5f}")
    return F_mean**(2 if gate=='B' else 3)



def sweep_T1_T2(
    T2_min=None, T2_max=None,
    det_error_frac_min=None, det_error_frac_max=None,
    mot_deph_min=None, mot_deph_max=None,
    n_T1=20, n_T2=20,
):
    """
    Returns:
        T1_grid, T2_grid : 1D arrays
        R                : 2D array, shape (n_T2, n_T1)
    """

    if det_error_frac_min is not None:
        T1_vals = np.logspace(det_error_frac_min, det_error_frac_max, n_T1)
        T2_vals = np.logspace(mot_deph_min, mot_deph_max, n_T2)
    elif T2_min is not None:
        T1_vals = np.logspace(T2_min, T2_max, n_T1)
        T2_vals = np.logspace(mot_deph_min, mot_deph_max, n_T2)


    R = np.zeros((n_T2, n_T1))

    for i, T2 in enumerate(T2_vals):
        for j, T1 in enumerate(T1_vals):
            
            if det_error_frac_min is not None:
                F1 = 1-run(100, 10, det_error_frac=T1, mot_deph=T2, ndot=1, gate="1")
                F2 = 1-run(100, 10, det_error_frac=T1, mot_deph=T2, ndot=1, gate="2")            
                FB = 1-run(100, 10, det_error_frac=T1, mot_deph=T2, ndot=1, gate="B")
            elif T2_min is not None:
                F1 = 1-run(100, T1, det_error_frac=1e-3, mot_deph=T2, ndot=1, gate="1")
                F2 = 1-run(100, T1, det_error_frac=1e-3, mot_deph=T2, ndot=1, gate="2")            
                FB = 1-run(100, T1, det_error_frac=1e-3, mot_deph=T2, ndot=1, gate="B")

            denom = min(F1, F2)
            R[i, j] = FB / denom

    return T1_vals, T2_vals, R


import json 


n_T1, n_T2 = (100, 100)
T1_vals_, T2_vals_, R1 = sweep_T1_T2(
        det_error_frac_min=-4, det_error_frac_max=-1,
        mot_deph_min=-2, mot_deph_max=1, n_T1=n_T1, n_T2=n_T2)

data = R1.tolist()
with open(f"R1_{n_T1}x{n_T2}.json", "w") as f:
    json.dump(data, f)

T1_vals_, T2_vals_, R2 = sweep_T1_T2(
        T2_min=-2, T2_max=3, 
        mot_deph_min=-2, mot_deph_max=1, n_T1=n_T1, n_T2=n_T2)
data = R2.tolist()
with open(f"R2_{n_T1}x{n_T2}.json", "w") as f:
    json.dump(data, f)


