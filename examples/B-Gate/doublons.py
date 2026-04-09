import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# ============================================================
# Basis (4x4 relevant block):
#   0 -> |D+>
#   1 -> |D->
#   2 -> |t0>
#   3 -> |s>
#
# Full 6-state basis (for compatibility with original code):
#   0 -> |t+>  (decoupled, Pauli)
#   1 -> |t0>
#   2 -> |t->  (decoupled, Pauli)
#   3 -> |D+>
#   4 -> |D->
#   5 -> |s>
# ============================================================


def hamiltonian_4x4(delta, tp, tm, U):
    """
    4x4 Hamiltonian in basis {|D+>, |D->, |t0>, |s>}
    tp = t_+ = t_up + t_down
    tm = t_- = t_up - t_down
    """
    H = np.array([
        [U,      2*delta, -tp,  0  ],
        [2*delta, U,       0,  -tm ],
        [-tp,     0,       0,   0  ],
        [0,      -tm,      0,   0  ]
    ], dtype=np.complex128)
    return H


def delta_ramp(time, T, delta_i, delta_f, kind="linear"):
    x = time / T
    if kind == "linear":
        return delta_i + (delta_f - delta_i) * x
    elif kind == "tanh":
        y = np.tanh(8.0 * (x - 0.5))
        y0, y1 = np.tanh(-4.0), np.tanh(4.0)
        return delta_i + (delta_f - delta_i) * (y - y0) / (y1 - y0)


def compute_propagator(T, delta_i, delta_f, tp, tm, U,
                       ramp_kind="linear", n_steps=3000):
    """
    Compute the full 4x4 propagator by evolving each basis vector.
    Returns U_full (4x4) and U_logical (2x2) in {|t0>, |s>}.
    """
    times = np.linspace(0, T, n_steps)

    # Evolve all 4 basis vectors simultaneously as columns of a matrix
    # dP/dt = -i H(t) P,  P(0) = I
    def rhs(t, P_flat):
        delta = delta_ramp(t, T, delta_i, delta_f, kind=ramp_kind)
        H = hamiltonian_4x4(delta, tp, tm, U)
        P = P_flat.reshape(4, 4)
        dP = -1j * H @ P
        return dP.flatten()

    P0 = np.eye(4, dtype=np.complex128).flatten()

    sol = solve_ivp(
        rhs,
        t_span=(0, T),
        y0=P0,
        t_eval=times,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    U_full = sol.y[:, -1].reshape(4, 4)

    # Extract logical 2x2 block: rows/cols 2,3 = |t0>, |s>
    U_logical = U_full[2:, 2:]
    return U_full, U_logical


def check_unitarity(U, label=""):
    err = np.linalg.norm(U @ U.conj().T - np.eye(len(U)))
    print(f"Unitarity error {label}: {err:.2e}")
    return err


# ============================================================
# Main analysis
# ============================================================
def main():
    Uval = 0
    tbar    = 1.0
    delta_i = -30.0 * tbar
    delta_f = +30.0 * tbar
    T       = 1.0 / tbar   # long enough to be adiabatic

    # --- 3. Scan U at fixed eps: check dynamical phase ---
    print("\n" + "=" * 55)
    print(f"3. Vary eps, U={0}")
    print("=" * 55)
    epss = np.linspace(0.025, 0.175, 1)

    phase_t0 = []
    phase_s  = []

    phase_cross = []
    amp_cross = []
    for eps in epss:
        tup, tdown = tbar+eps, tbar-eps
        tp, tm = tup+tdown, tup-tdown
        _, U_log = compute_propagator(T, delta_i, delta_f, tp, tm, U=Uval)
        phase_t0.append(np.angle(U_log[0, 0]))
        phase_s.append(np.angle(U_log[1, 1]))

        phase_cross.append(np.angle(U_log[0, 1]))
        amp_cross.append(np.abs(U_log[0, 1]))


    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epss/tbar, np.array(phase_t0)/np.pi, 'o-', label=r"$\arg\langle t_0|U|t_0\rangle/\pi$")
    ax.plot(epss/tbar, np.array(phase_s)/np.pi,  's-', label=r"$\arg\langle s|U|s\rangle/\pi$")
    
    ax.plot(epss/tbar, np.array(amp_cross), label=r"$ |\langle t_0|U|s\rangle |$")
    ax.plot(epss/tbar, np.array(phase_cross)/np.pi,  label=r"$ \arg \langle t0|U|s\rangle /\pi$")
    
    ax.set_xlabel(r"$U/\bar{t}$")
    ax.set_ylabel(r"phase / $\pi$")
    ax.set_title(rf"Diagonal phases vs $U$, $\epsilon={eps}$, $T={T:.0f}/\bar{{t}}$")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return np.array(phase_t0)/np.pi, np.array(phase_s)/np.pi, np.array(amp_cross), np.array(phase_cross)/np.pi



def print_complex_matrix(M):
    for row in M:
        parts = []
        for z in row:
            parts.append(f"{z.real:+.4f}{z.imag:+.4f}j")
        print("  [" + ",  ".join(parts) + "]")


import json 

phase_t0, phase_s, amp_cross, phase_cross = main()
data = {
    'phase_t0': phase_t0.tolist(), 'phase_s': phase_s.tolist(), 'amp_cross': amp_cross.tolist(), 
    'phase_cross': phase_cross.tolist()
}
with open(f"./logs/U0_epss_0.025-0.175_T1000_n21.json", "w") as f:
    json.dump(data, f)
