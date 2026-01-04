"""
Two-ion (2-qubit) Mølmer–Sørensen (MS) gate with common motional mode in QuTiP,
including noise channels:
  - T1 (amplitude damping) on each qubit
  - T2 (pure dephasing) on each qubit
  - detuning error (drive detuning from sideband beat-note)
  - quasi-static Z-offset (constant Z shift per shot, sampled from a distribution)
  - motional heating (phonon creation; optional cooling/thermalization hook shown)

This is a *simulation template*: it models a standard MS interaction on a single mode
using a time-dependent Hamiltonian and Lindblad master equation via qutip.mesolve.

Requires: qutip >= 4.x
"""

import numpy as np
import qutip as qt


# ----------------------------
# Helpers: tensor operators
# ----------------------------
def destroy_mode(N):
    return qt.destroy(N)

def op_on_qubit(op, which, Nmode):
    """
    Place a single-qubit operator on qubit index `which` (0 or 1)
    in a Hilbert space: qubit0 ⊗ qubit1 ⊗ mode
    """
    I2 = qt.qeye(2)
    Im = qt.qeye(Nmode)
    if which == 0:
        return qt.tensor(op, I2, Im)
    elif which == 1:
        return qt.tensor(I2, op, Im)
    else:
        raise ValueError("which must be 0 or 1")


def op_on_mode(op_mode, Nmode):
    I2 = qt.qeye(2)
    return qt.tensor(I2, I2, op_mode)


# ----------------------------
# Build MS Hamiltonian
# ----------------------------
def ms_hamiltonian(
    Nmode,
    eta=0.05,          # Lamb-Dicke parameter
    Omega=2*np.pi*20e3, # carrier Rabi freq (rad/s) effective
    delta=2*np.pi*2e3,  # symmetric detuning from sideband (rad/s)
    detuning_error=0.0, # extra detuning error added to delta (rad/s)
    phase=0.0,          # drive phase
):
    """
    Standard bichromatic MS interaction (interaction picture w.r.t. qubit splitting),
    for a single motional mode:
        H_I(t) = ħ * (ηΩ/2) * S_phi * ( a e^{+i(δ+ε)t} + a† e^{-i(δ+ε)t} )
    where S_phi = Σ_i (σ_i^+ e^{iφ} + σ_i^- e^{-iφ}) = Σ_i (cosφ σ_x + sinφ σ_y)

    We implement it as a QuTiP time-dependent Hamiltonian list.
    """
    a  = op_on_mode(destroy_mode(Nmode), Nmode)
    adag = a.dag()

    sx = qt.sigmax()
    sy = qt.sigmay()

    Sphi = (np.cos(phase) * (op_on_qubit(sx, 0, Nmode) + op_on_qubit(sx, 1, Nmode))
          + np.sin(phase) * (op_on_qubit(sy, 0, Nmode) + op_on_qubit(sy, 1, Nmode)))

    g = 0.5 * eta * Omega  # effective spin-motion coupling (rad/s)

    d = delta + detuning_error

    # Time-dependent coefficients
    def c_plus(t, args):
        return np.cos(d * t)  # real form
    def s_plus(t, args):
        return np.sin(d * t)

    # Rewrite:
    #   a e^{+i d t} + a† e^{-i d t} = (a + a†) cos(dt) + i(a - a†) sin(dt)
    X = (a + adag)
    P = 1j * (a - adag)  # Hermitian quadrature

    H0 = 0 * Sphi  # no static term here (we add Z-offsets separately)
    H = [
        H0,
        [g * Sphi * X, c_plus],
        [g * Sphi * P, s_plus],
    ]
    return H


# ----------------------------
# Noise / collapse operators
# ----------------------------
def collapse_operators(
    Nmode,
    T1=np.inf,
    T2=np.inf,
    heating_rate=0.0,   # phonons/s (i.e., \dot{n})
    n_th=0.0,           # optional thermal occupancy for symmetric heating/cooling
    motional_Tphi=np.inf,
    motional_damping=0.0 # optional mode damping rate (1/s), to a bath with n_th
):
    """
    Returns list of collapse operators for:
      - qubit amplitude damping (T1)
      - qubit pure dephasing (T2, as pure dephasing rate)
      - motional heating (a†) at rate heating_rate
      - optional motional damping/thermalization (a) with motional_damping to bath n_th
    Notes:
      - For qubits: total dephasing 1/T2 = 1/(2T1) + 1/Tphi  => Tphi derived
      - If T2 is given, we implement pure dephasing with rate gamma_phi = 1/Tphi.
    """
    cops = []

    # Qubit T1
    if np.isfinite(T1) and T1 > 0:
        gamma1 = 1.0 / T1
        sm = qt.sigmam()
        for qi in [0, 1]:
            cops.append(np.sqrt(gamma1) * op_on_qubit(sm, qi, Nmode))

    # Qubit T2 as pure dephasing (derive Tphi so T2 matches)
    if np.isfinite(T2) and T2 > 0:
        gamma2 = 1.0 / T2

        gamma1 = 0.0
        if np.isfinite(T1) and T1 > 0:
            gamma1 = 1.0 / T1

        # gamma2 = gamma1/2 + gamma_phi  => gamma_phi = gamma2 - gamma1/2
        gamma_phi = max(0.0, gamma2 - 0.5 * gamma1)
        if gamma_phi > 0:
            sz = qt.sigmaz()
            for qi in [0, 1]:
                cops.append(np.sqrt(gamma_phi/2) * op_on_qubit(sz, qi, Nmode))
                # factor 1/2 because Lindblad with sz gives dephasing rate gamma_phi
                # under common conventions; adjust if you use a different convention.

    # Motional heating: L = sqrt(Γ_h) a†
    if heating_rate and heating_rate > 0:
        a = op_on_mode(destroy_mode(Nmode), Nmode)
        cops.append(np.sqrt(heating_rate) * a.dag())


    # ----------------
    # Motional dephasing (NEW)
    # ----------------
    if np.isfinite(motional_Tphi) and motional_Tphi > 0:
        gamma_phi_m = 1.0 / motional_Tphi
        a = op_on_mode(destroy_mode(Nmode), Nmode)
        n_op = a.dag() * a
        cops.append(np.sqrt(gamma_phi_m) * n_op)


    # Optional motional damping to thermal bath:
    #   Γ (n_th+1) D[a] + Γ n_th D[a†]
    if motional_damping and motional_damping > 0:
        a = op_on_mode(destroy_mode(Nmode), Nmode)
        cops.append(np.sqrt(motional_damping * (n_th + 1.0)) * a)
        if n_th > 0:
            cops.append(np.sqrt(motional_damping * n_th) * a.dag())

    return cops


# ----------------------------
# Quasi-static Z-offset sampling
# ----------------------------
def z_offset_hamiltonian(Nmode, z0_0=0.0, z0_1=0.0):
    """
    Adds a constant Hamiltonian term:
        H_Z = (ħ/2) * (z0_0 σz0 + z0_1 σz1)
    where z0_i are angular frequencies (rad/s).
    """
    sz = qt.sigmaz()
    Hz = 0.5 * (z0_0 * op_on_qubit(sz, 0, Nmode) + z0_1 * op_on_qubit(sz, 1, Nmode))
    return Hz


# ----------------------------
# MS gate run (single shot)
# ----------------------------
def run_ms_gate_single_shot(
    Nmode=15,
    eta=0.05,
    Omega=2*np.pi*20e3,
    delta=2*np.pi*2e3,
    detuning_error=0.0,
    phase=0.0,
    T1=np.inf,
    T2=np.inf,
    heating_rate=0.0,
    mot_deph=np.inf,
    z0_0=0.0,      # quasi-static Z offset for qubit 0 (rad/s)
    z0_1=0.0,      # quasi-static Z offset for qubit 1 (rad/s)
    t_gate=None,
    nsteps=2000,
    psi0=None,
    rho0=None,
    return_states=False,
):
    """
    Simulate one MS gate "shot" for a given (fixed) quasi-static Z-offset and detuning error.

    If t_gate is None, we pick t_gate = 2π/|delta| which closes phase-space for ideal MS.
    """
    if t_gate is None:
        t_gate = 2*np.pi / abs(delta)

    # Initial state default: |00> ⊗ |0_mode>
    if rho0 is None and psi0 is None:
        g = qt.basis(2, 0)
        vac = qt.basis(Nmode, 0)
        psi0 = qt.tensor(g, g, vac)

    # Time grid
    tlist = np.linspace(0.0, t_gate, nsteps)

    # Hamiltonian
    H_ms = ms_hamiltonian(
        Nmode=Nmode, eta=eta, Omega=Omega,
        delta=delta, detuning_error=detuning_error,
        phase=phase
    )
    Hz = z_offset_hamiltonian(Nmode, z0_0=z0_0, z0_1=z0_1)

    # Combine: if H_ms is list, prepend/merge static term
    # H_total = [Hz] + H_ms_time_dependent_terms (H_ms already has a static 0 term)
    H_total = [Hz] + H_ms  # works because H_ms starts with a Qobj (H0)

    # Collapses
    cops = collapse_operators(
        Nmode=Nmode, T1=T1, T2=T2, motional_Tphi=mot_deph,
        heating_rate=heating_rate
    )

    # Solve
    if rho0 is not None:
        result = qt.mesolve(H_total, rho0, tlist, c_ops=cops, e_ops=[])
    else:
        result = qt.mesolve(H_total, psi0, tlist, c_ops=cops, e_ops=[])

    if return_states:
        return tlist, result.states
    return result.states[-1]


# ----------------------------
# Average over quasi-static Z offsets (many shots)
# ----------------------------
def run_ms_gate_with_quasistatic_average(
    nshots=50,
    z_sigma=2*np.pi*200.0,   # rad/s std dev of Z-offset per qubit (example)
    correlated=False,        # if True, same offset applied to both qubits
    seed=0, **kwargs
):
    """
    Monte-Carlo average over quasi-static Z offsets (Gaussian).
    Returns averaged density matrix at t_gate.
    """
    rng = np.random.default_rng(seed)
    rhos = []

    for _ in range(nshots):
        if correlated:
            z = rng.normal(0.0, z_sigma)
            z0_0, z0_1 = z, z
        else:
            z0_0 = rng.normal(0.0, z_sigma)
            z0_1 = rng.normal(0.0, z_sigma)

        rho_f = run_ms_gate_single_shot(z0_0=z0_0, z0_1=z0_1, **kwargs)

        # Ensure density matrix
        if rho_f.isket:
            rho_f = rho_f.proj()
        rhos.append(rho_f)

    return sum(rhos) / len(rhos)

def ideal_ms_unitary(theta=np.pi/4):
    """
    Ideal 2-qubit MS unitary:
        U = exp(-i theta * σx ⊗ σx)
    """
    sx = qt.sigmax()
    H_eff = theta * qt.tensor(sx, sx)
    return (-1j * H_eff).expm()


def random_two_qubit_state(seed=None):
    """
    Haar-random pure state on 2 qubits with correct tensor dims.
    """
    rng = np.random.default_rng(seed)

    # Generate random complex amplitudes
    vec = rng.normal(size=4) + 1j * rng.normal(size=4)
    vec /= np.linalg.norm(vec)

    psi = qt.Qobj(
        vec,
        dims=[[2, 2], [1, 1]]  # <-- critical fix
    )
    return psi


def embed_with_motion(psi_qq, Nmode, nbar=0):
    """
    Embed 2-qubit state into |ψ⟩ ⊗ |nbar⟩
    """
    mot = qt.basis(Nmode, nbar)
    return qt.tensor(psi_qq, mot)


def ms_gate_state_fidelity(
    Nmode=20,
    eta=0.05,
    Omega=2*np.pi*20e3,
    delta=2*np.pi*2e3,
    detuning_error=0.0,
    phase=0.0,
    T1=np.inf,
    T2=np.inf,
    heating_rate=0.0,
    z0_0=0.0,
    z0_1=0.0,
    nsteps=2000,
    seed=None,
    ideal_unitary=None,
    mot_deph=np.inf,
    reps=1, phases=[]
):
    """
    1. Draw random 2-qubit input
    2. Apply ideal MS unitary
    3. Apply noisy MS evolution
    4. Trace out motion
    5. Compute fidelity
    """

    # --- Random input ---
    psi_qq = random_two_qubit_state(seed)
    psi_full = embed_with_motion(psi_qq, Nmode)

    # --- Ideal evolution ---
    if ideal_unitary is None:
        U_ms = ideal_ms_unitary(theta=np.pi/4)
    else:
        U_ms = ideal_unitary
    psi_ideal = U_ms @ psi_qq
    rho_ideal = psi_ideal.proj()
    if reps==1:
        phases = [phase]

    # --- Noisy evolution ---
    for _ in range(reps):
        psi_full = run_ms_gate_single_shot(
            Nmode=Nmode,
            eta=eta,
            Omega=Omega,
            delta=delta,
            detuning_error=detuning_error,
            phase=phases[_],
            T1=T1,
            T2=T2,
            mot_deph=mot_deph,
            heating_rate=heating_rate,
            z0_0=z0_0,
            z0_1=z0_1,
            psi0=psi_full,
            nsteps=nsteps,
        )
    rho_final = psi_full

    # Trace out motion
    rho_qq_noisy = qt.ptrace(rho_final, [0, 1])

    # Fidelity
    F = qt.metrics.fidelity(rho_qq_noisy, rho_ideal)
    return F.real


def averaged_ms_fidelity(
    n_states=20,
    n_zshots=20,
    z_sigma=2*np.pi*100.0,
    correlated=False,
    seed=0, **kwargs
):
    rng = np.random.default_rng(seed)
    fidelities = []

    for k in range(n_states):
        for _ in range(n_zshots):
            if correlated:
                z = rng.normal(0.0, z_sigma)
                z0_0, z0_1 = z, z
            else:
                z0_0 = rng.normal(0.0, z_sigma)
                z0_1 = rng.normal(0.0, z_sigma)

            F = ms_gate_state_fidelity(
                z0_0=z0_0,
                z0_1=z0_1,
                seed=rng.integers(1e9),
                **kwargs
            )
            fidelities.append(F)

    return np.mean(fidelities), np.std(fidelities)


if __name__ == "__main__":
    reps, phases = 2, (0, np.pi)
    
    eta=0.06
    Omega=2*np.pi*100e3
    delta=2*eta*Omega*np.sqrt(reps)
    print("Gate time: ", 1e6*2*np.pi/delta, "us")

    T1=1
    T2=200e-3
    heating_rate=1 

    det_error_frac = 1e-2
    z_sigma=2*np.pi*10 #(Assumes circa 100 Hz B-Field fluct.)
    
    # The lower this ratio det_error_frac/z_sigma, the better it is to do 2-loop
    

    
    F_mean, F_std = averaged_ms_fidelity(
        n_states=10, Nmode=10,
        nsteps=100, n_zshots=20,
        
        eta=eta,
        Omega=Omega,
        delta=delta,
        T1=T1, T2=T2,
        detuning_error=det_error_frac*delta,
        heating_rate=heating_rate, z_sigma=z_sigma,
        correlated=False,

        reps=reps, phases=phases
    )

    print(f"Average MS gate fidelity: {F_mean*100:.5f} ± {F_std:.5f}")
