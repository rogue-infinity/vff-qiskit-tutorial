"""
generate_precomputed.py
Run this script once to generate pre-computed Hamiltonian files and
pre-trained VFF parameters for the three built-in models.

Usage:
    python generate_precomputed.py
"""

import pickle, os, itertools
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp, Operator, Statevector
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.circuit.library import EfficientSU2

os.makedirs("hamiltonians",   exist_ok=True)
os.makedirs("trained_params", exist_ok=True)

# ── Hamiltonian builders ──────────────────────────────────────────────────────

def build_tfim(n, J=1.0, h=0.5):
    terms = [(f"{'I'*(n-2-i)}ZZ{'I'*i}", -J) for i in range(n-1)]
    terms += [(f"{'I'*(n-1-i)}X{'I'*i}", -h) for i in range(n)]
    return SparsePauliOp.from_list(terms)

def build_xy(n, Jx=1.0, Jy=1.0):
    terms = [(f"{'I'*(n-2-i)}XX{'I'*i}", -Jx) for i in range(n-1)]
    terms += [(f"{'I'*(n-2-i)}YY{'I'*i}", -Jy) for i in range(n-1)]
    return SparsePauliOp.from_list(terms)

def build_heisenberg(n, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0):
    terms  = [(f"{'I'*(n-2-i)}XX{'I'*i}", -Jx) for i in range(n-1)]
    terms += [(f"{'I'*(n-2-i)}YY{'I'*i}", -Jy) for i in range(n-1)]
    terms += [(f"{'I'*(n-2-i)}ZZ{'I'*i}", -Jz) for i in range(n-1)]
    if h != 0:
        terms += [(f"{'I'*(n-1-i)}Z{'I'*i}", -h) for i in range(n)]
    return SparsePauliOp.from_list(terms)

# ── VFF ansatz ────────────────────────────────────────────────────────────────

def build_w(n, reps=1):
    return EfficientSU2(n, reps=reps, entanglement="linear", parameter_prefix="θ")

def build_d(n, order=2):
    """D(γ, t): Rz layer + optional ZZ layer. Returns (circuit, gamma_params, t_param)."""
    t     = Parameter("t")
    pairs = list(itertools.combinations(range(n), 2)) if order >= 2 else []
    gamma = ParameterVector("γ", n + len(pairs))
    qc    = QuantumCircuit(n, name="D(γ,t)")
    for k in range(n):
        qc.rz(2 * gamma[k] * t, k)
    for idx, (a, b) in enumerate(pairs):
        qc.rzz(2 * gamma[n + idx] * t, a, b)
    return qc, gamma, t, pairs

def build_vff(n, w_reps=1, d_order=2):
    """Full VFF circuit V = W† D W. Returns (circuit, w_params, gamma_params, t_param)."""
    w               = build_w(n, reps=w_reps)
    d_circ, gamma, t, _ = build_d(n, order=d_order)
    qc = QuantumCircuit(n, name="V(α,Δt)")
    qc.compose(w.inverse(), inplace=True)
    qc.barrier(label="D")
    qc.compose(d_circ, inplace=True)
    qc.barrier(label="W")
    qc.compose(w, inplace=True)
    return qc, list(w.parameters), list(gamma), t

# ── Cost function ─────────────────────────────────────────────────────────────

def hs_cost(params, vff_qc, w_params, gamma_params, t_param, u_mat, delta_t, history=None):
    n_w = len(w_params)
    pd  = {p: params[i]       for i, p in enumerate(w_params)}
    pd.update({p: params[n_w + i] for i, p in enumerate(gamma_params)})
    pd[t_param] = delta_t
    v_mat = Operator(vff_qc.assign_parameters(pd)).data
    d     = u_mat.shape[0]
    cost  = 1.0 - abs(np.trace(u_mat.conj().T @ v_mat))**2 / d**2
    if history is not None:
        history.append(cost)
    return cost

# ── Training ──────────────────────────────────────────────────────────────────

def train_vff(H_spo, delta_t, w_reps=1, d_order=2, seed=42, maxiter=1500):
    n      = H_spo.num_qubits
    H_mat  = H_spo.to_matrix()
    u_mat  = expm(-1j * H_mat * delta_t)
    qc, wp, gp, tp = build_vff(n, w_reps=w_reps, d_order=d_order)
    np.random.seed(seed)
    x0      = np.random.uniform(-np.pi, np.pi, len(wp) + len(gp))
    history = []
    result  = minimize(
        hs_cost, x0,
        args=(qc, wp, gp, tp, u_mat, delta_t, history),
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )
    print(f"  Final cost: {history[-1]:.5f}  ({'converged' if result.success else 'max iter'})")
    theta_opt = result.x[:len(wp)]
    gamma_opt = result.x[len(wp):]
    return theta_opt, gamma_opt, history

# ── Main ──────────────────────────────────────────────────────────────────────

CONFIGS = [
    ("tfim",        build_tfim(4),        0.2),
    ("xy",          build_xy(4),          0.2),
    ("heisenberg",  build_heisenberg(4),  0.2),
]

for name, H_spo, dt in CONFIGS:
    pkl_path = f"hamiltonians/{name}_4q.pkl"
    npz_path = f"trained_params/vff_{name}_4q_dt{dt}.npz"

    # Save Hamiltonian
    with open(pkl_path, "wb") as f:
        pickle.dump(H_spo, f)
    print(f"Saved {pkl_path}")

    # Train and save params
    print(f"Training VFF for {name} (4q, Δt={dt}) ...")
    theta, gamma, hist = train_vff(H_spo, dt)
    np.savez(npz_path, theta=theta, gamma=gamma, history=hist)
    print(f"Saved {npz_path}\n")

print("Done.")
