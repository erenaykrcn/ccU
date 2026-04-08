import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# Basis:
#   0 -> |t+>
#   1 -> |t0>
#   2 -> |t->
#   3 -> |D+>
#   4 -> |D->
#   5 -> |s>
#
# Spin-dependent tunneling Hamiltonian:
#
# H =
# [[0, 0, 0, 0,          0,          0],
#  [0, 0, 0, 0,     tu - td,         0],
#  [0, 0, 0, 0,          0,          0],
#  [0, 0, 0, U,       2Δ,    -(tu + td)],
#  [0, tu-td, 0, 2Δ,    U,          0],
#  [0, 0, 0, -(tu+td),  0,          0]]
#
# For tu = td = t, this reduces to the paper's symmetric case.
# ============================================================


def delta_ramp(time, T, delta_i, delta_f, kind="tanh", smoothness=8.0):
    x = time / T
    if kind == "linear":
        return delta_i + (delta_f - delta_i) * x
    elif kind == "tanh":
        y = np.tanh(smoothness * (x - 0.5))
        y0 = np.tanh(-smoothness / 2)
        y1 = np.tanh(+smoothness / 2)
        s = (y - y0) / (y1 - y0)
        return delta_i + (delta_f - delta_i) * s
    else:
        raise ValueError("kind must be 'linear' or 'tanh'")


def hamiltonian_spin_dependent(delta, tu, td, U):
    H = np.zeros((6, 6), dtype=np.complex128)

    # triplets |t+>, |t-> stay decoupled at zero
    # |t0> couples to |D-> with tu - td
    H[1, 4] = tu - td
    H[4, 1] = tu - td

    # doublon block
    H[3, 3] = U
    H[4, 4] = U
    H[3, 4] = 2.0 * delta
    H[4, 3] = 2.0 * delta

    # singlet coupling to |D+>
    H[3, 5] = -(tu + td)
    H[5, 3] = -(tu + td)

    return H


def schrodinger_rhs(time, psi, T, delta_i, delta_f, tu, td, U, ramp_kind):
    delta = delta_ramp(time, T, delta_i, delta_f, kind=ramp_kind)
    H = hamiltonian_spin_dependent(delta, tu, td, U)
    return -1j * H @ psi


def evolve_state(
    psi0,
    T,
    delta_i,
    delta_f,
    tu,
    td,
    U,
    ramp_kind="tanh",
    n_times=2000,
):
    times = np.linspace(0.0, T, n_times)
    sol = solve_ivp(
        fun=lambda time, psi: schrodinger_rhs(
            time, psi, T, delta_i, delta_f, tu, td, U, ramp_kind
        ),
        t_span=(0.0, T),
        y0=psi0,
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return times, sol.y


def basis_vector(i, dim=6):
    v = np.zeros(dim, dtype=np.complex128)
    v[i] = 1.0
    return v


def project_logical(psi):
    """
    Logical subspace = span{|t0>, |s>}
    Return [amp_t0, amp_s]
    """
    return np.array([psi[1], psi[5]], dtype=np.complex128)


def effective_unitary(T, delta_i, delta_f, tu, td, U, ramp_kind="tanh"):
    """
    Evolve |t0> and |s> separately and build the 2x2 map
    on the logical subspace span{|t0>, |s>}.
    """
    e_t0 = basis_vector(1)
    e_s = basis_vector(5)

    _, psi_t0 = evolve_state(e_t0, T, delta_i, delta_f, tu, td, U, ramp_kind=ramp_kind, n_times=1200)
    _, psi_s  = evolve_state(e_s,  T, delta_i, delta_f, tu, td, U, ramp_kind=ramp_kind, n_times=1200)

    final_t0 = project_logical(psi_t0[:, -1])
    final_s  = project_logical(psi_s[:, -1])

    Ueff = np.column_stack([final_t0, final_s])
    return Ueff


def populations(psi_t):
    return np.abs(psi_t) ** 2


def leakage_probability(psi):
    """
    Outside logical subspace {|t0>, |s>}
    """
    return np.abs(psi[0])**2 + np.abs(psi[2])**2 + np.abs(psi[3])**2 + np.abs(psi[4])**2


def run_single_example():
    # -------------------------
    # Parameters
    # -------------------------
    tbar = 1.0
    
    eps = 0.25              # spin-dependent asymmetry
    tu = tbar + eps
    td = tbar - eps

    U = 0.0                 # start from geometric case, then try U != 0
    delta_i = -30.0 * tbar
    delta_f = +30.0 * tbar
    T = 80.0 / tbar
    ramp_kind = "tanh"

    labels = [r"$|t_+\rangle$", r"$|t_0\rangle$", r"$|t_-\rangle$",
              r"$|D_+\rangle$", r"$|D_-\rangle$", r"$|s\rangle$"]

    # Start in |s>
    psi0 = basis_vector(5)
    times, psi_t = evolve_state(
        psi0, T, delta_i, delta_f, tu, td, U, ramp_kind=ramp_kind, n_times=2500
    )
    pops = populations(psi_t)

    psi_final = psi_t[:, -1]
    print("=== single run: initial |s> ===")
    print(f"tu = {tu:.4f}, td = {td:.4f}, U = {U:.4f}")
    for i, lab in enumerate(labels):
        print(f"{lab:>10s}: amp = {psi_final[i]: .6f}, pop = {abs(psi_final[i])**2:.6f}")

    print("\nLogical amplitudes:")
    print(f"<t0|psi(T)> = {psi_final[1]: .8f}")
    print(f"<s |psi(T)> = {psi_final[5]: .8f}")
    print(f"leakage      = {leakage_probability(psi_final):.8e}")

    deltas = np.array([delta_ramp(t, T, delta_i, delta_f, kind=ramp_kind) for t in times])

    plt.figure(figsize=(7, 4))
    plt.plot(times, deltas)
    plt.xlabel("time")
    plt.ylabel(r"$\Delta(t)$")
    plt.title("Bias ramp")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    for i, lab in enumerate(labels):
        plt.plot(times, pops[i], label=lab)
    plt.xlabel("time")
    plt.ylabel("population")
    plt.title(rf"Populations for $t_\uparrow={tu:.2f}$, $t_\downarrow={td:.2f}$, $U={U:.2f}$")
    plt.legend()
    plt.tight_layout()



def run_single_example(eps, U):
    # -------------------------
    # Parameters
    # -------------------------
    tbar = 1

    tu = tbar + eps
    td = tbar - eps

    
    delta_i = -30.0 * tbar
    delta_f = +30.0 * tbar
    T = 2000.0 / tbar
    ramp_kind = "linear"

    labels = [r"$|t_+\rangle$", r"$|t_0\rangle$", r"$|t_-\rangle$",
              r"$|D_+\rangle$", r"$|D_-\rangle$", r"$|s\rangle$"]

    # Start in |s>
    #psi0 = basis_vector(5)

    # Start in t0 + s superpos.
    psi0 = (basis_vector(1) + basis_vector(5)) / np.sqrt(2)

    # Start in t0
    #psi0 = basis_vector(1)

    times, psi_t = evolve_state(
        psi0, T, delta_i, delta_f, tu, td, U, ramp_kind=ramp_kind, n_times=2500
    )
    pops = populations(psi_t)

    psi_final = psi_t[:, -1]
    print("=== single run: initial |s> ===")
    print(f"tu = {tu:.4f}, td = {td:.4f}, U = {U:.4f}")
    for i, lab in enumerate(labels):
        print(f"{lab:>10s}: amp = {psi_final[i]: .6f}, pop = {abs(psi_final[i])**2:.6f}")

    print("\nLogical amplitudes:")
    print(f"<t0|psi(T)> = {psi_final[1]: .8f}")
    print(f"<s |psi(T)> = {psi_final[5]: .8f}")
    print(f"leakage      = {leakage_probability(psi_final):.8e}")
    return psi_final[1], psi_final[5]


epss = np.linspace(0, 1, 30)
for eps in epss:
    psi_t, psi_s = run_single_example(eps, 0)
    with open(f"./logs/epss.txt", "a") as file:
        file.write(f"eps={eps}, ({psi_final[1]: .8f}, {psi_final[5]: .8f}) \n")


Us = np.linspace(0, 10, 30)
for U in Us:
    psi_t, psi_s = run_single_example(0, U)
    with open(f"./logs/Us.txt", "a") as file:
        file.write(f"U={U}, ({psi_final[1]: .8f}, {psi_final[5]: .8f}) \n")


