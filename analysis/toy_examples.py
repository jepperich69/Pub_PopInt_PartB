"""
Integerization demo: Hamilton (largest remainders) vs Entropy (min-KL with exact margins)
+ 'Small cell at risk' scenario
+ Entropy-informed randomized rounding with coverage over replicates

Produces the toy figures of Section 2: Figure 1 = toy_entropy_vs_hamilton_A.png,
Figure 2 = toy_entropy_vs_hamilton_B.png, Figure 3 = toy_small_cell_risk.png, all in
output/figures/, plus toy_metrics_A_B.csv.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Utilities ----------
OUTDIR = Path(__file__).resolve().parents[1] / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
DPI = 220

def enumerate_2x3_tables(row_sums, col_sums):
    """Enumerate all nonnegative integer 2x3 tables with given row/col sums."""
    r1, r2 = int(row_sums[0]), int(row_sums[1])
    c1, c2, c3 = map(int, col_sums)
    feasible = []
    for a in range(c1 + 1):
        d = c1 - a
        for b in range(c2 + 1):
            e = c2 - b
            for c in range(c3 + 1):
                f = c3 - c
                if a + b + c == r1 and d + e + f == r2 and min(a, b, c, d, e, f) >= 0:
                    feasible.append(np.array([[a, b, c], [d, e, f]], dtype=int))
    return feasible

def kl_div(Q, P):
    eps = 1e-12
    return float(np.sum(Q * np.log((Q + eps) / (P + eps))))

def total_variation(Q, P):
    return 0.5 * np.sum(np.abs(Q - P))

def js_divergence(Q, P):
    M = 0.5 * (Q + P)
    return 0.5 * kl_div(Q, M) + 0.5 * kl_div(P, M)

def hamilton_round(X):
    """Largest remainders relative to floors; enforces only the grand total."""
    floors = np.floor(X).astype(int)
    residuals = (X - floors).flatten()
    R = int(round((X - floors).sum()))
    z = np.zeros_like(residuals, dtype=int)
    z[np.argsort(-residuals)[:R]] = 1  # give R extras to largest residuals
    return (floors.flatten() + z).reshape(X.shape)

def best_entropy_integer(X, row_sums, col_sums):
    """
    Among all integer tables that match row/col sums, pick the one minimizing KL(q||p),
    where p is the normalized fractional IPF table.
    """
    feas = enumerate_2x3_tables(row_sums, col_sums)
    p = (X / X.sum()).flatten()
    best_tbl, best_kl = None, None
    for T in feas:
        q = (T / T.sum()).flatten()
        val = kl_div(q, p)
        if best_kl is None or val < best_kl:
            best_tbl, best_kl = T.copy(), val
    return best_tbl, best_kl

def heatmap(ax, M, title, vmin=0.0, vmax=None):
    # One colour scale per figure (vmin/vmax shared by the three panels), so
    # equal counts get equal colours; white labels on the dark end of viridis.
    im = ax.imshow(M, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(['Low','Mid','High'])
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(['Young','Old'])
    ax.set_title(title)
    span = (vmax - vmin) if vmax is not None else 1.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = f"{M[i, j]:.1f}" if M.dtype == float else f"{int(M[i, j])}"
            dark = vmax is not None and (M[i, j] - vmin) / span < 0.5
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if dark else 'black')
    return im

def metrics_row(X, H, E, row_sums, col_sums):
    """Compute margin errors and divergences vs fractional p."""
    p = (X / X.sum()).flatten()
    H_row = H.sum(axis=1); H_col = H.sum(axis=0)
    E_row = E.sum(axis=1); E_col = E.sum(axis=0)
    H_q = (H / H.sum()).flatten(); E_q = (E / E.sum()).flatten()
    return {
        "Row L1 (Hamilton)": int(np.sum(np.abs(H_row - row_sums))),
        "Col L1 (Hamilton)": int(np.sum(np.abs(H_col - col_sums))),
        "Row L1 (Entropy)": int(np.sum(np.abs(E_row - row_sums))),
        "Col L1 (Entropy)": int(np.sum(np.abs(E_col - col_sums))),
        "KL(H||p)": kl_div(H_q, p), "KL(E||p)": kl_div(E_q, p),
        "TV(H,p)": total_variation(H_q, p), "TV(E,p)": total_variation(E_q, p),
        "JS(H,p)": js_divergence(H_q, p), "JS(E,p)": js_divergence(E_q, p),
    }

def sample_entropy_weighted(X, row_sums, col_sums, lam=10.0, n_samples=100, rng=None):
    """
    Entropy-informed randomized rounding: sample feasible integer tables T
    with probability proportional to exp(-lam * KL(q||p)).
    Returns samples, weights (normalized), and coverage stats.
    """
    if rng is None:
        rng = np.random.default_rng(123)
    feas = enumerate_2x3_tables(row_sums, col_sums)
    p = (X / X.sum()).flatten()
    weights = []
    for T in feas:
        q = (T / T.sum()).flatten()
        weights.append(np.exp(-lam * kl_div(q, p)))
    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()
    idx = rng.choice(len(feas), size=n_samples, replace=True, p=probs)
    samples = [feas[i] for i in idx]
    # coverage: probability a cell is >= 1 across samples
    cov = np.mean(np.array([S >= 1 for S in samples], dtype=float), axis=0)
    return samples, probs, cov, feas

# ---------- Scenario 1 & 2 (as before) ----------
# Fractional IPF table (Young/Old × Low/Mid/High)
X = np.array([[1.7, 2.3, 1.0],
              [0.3, 1.7, 3.0]], dtype=float)
N = int(X.sum())  # 10

# A: margins equal to X
row_A = X.sum(axis=1).round().astype(int)   # [5, 5]
col_A = X.sum(axis=0).round().astype(int)   # [2, 4, 4]

# B: shift 1 from Mid to Low (policy controls)
row_B = row_A.copy()
col_B = np.array([3, 3, 4], dtype=int)

results = []

for label, row_sums, col_sums in [("A (baseline margins)", row_A, col_A),
                                  ("B (shifted controls)", row_B, col_B)]:
    H = hamilton_round(X)
    E, _ = best_entropy_integer(X, row_sums, col_sums)
    m = metrics_row(X, H, E, row_sums, col_sums)
    m["Scenario"] = label
    results.append(m)

    # Figures
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    vmax = max(X.max(), H.max(), E.max())
    heatmap(axes[0], X, "Fractional IPF table X", vmax=vmax)
    # deltas for Hamilton under this scenario
    row_delta = tuple((H.sum(axis=1) - row_sums).tolist())
    col_delta = tuple((H.sum(axis=0) - col_sums).tolist())
    heatmap(axes[1], H, f"Hamilton (grand total only)\nmargin error: rows {row_delta}, cols {col_delta}", vmax=vmax)
    heatmap(axes[2], E, "KL optimum, Eq. (2)\nmargin error: 0", vmax=vmax)
    fig.suptitle(label)
    plt.tight_layout()
    fig.savefig(OUTDIR / f"toy_entropy_vs_hamilton_{label.split()[0]}.png", dpi=DPI)


metrics_df = pd.DataFrame(results, columns=[
    "Scenario",
    "Row L1 (Hamilton)", "Col L1 (Hamilton)",
    "Row L1 (Entropy)", "Col L1 (Entropy)",
    "KL(H||p)", "KL(E||p)", "TV(H,p)", "TV(E,p)", "JS(H,p)", "JS(E,p)"
])
metrics_df.to_csv(OUTDIR / "toy_metrics_A_B.csv", index=False)
print("\n=== Metrics (Scenarios A & B) ===")
print(metrics_df.to_string(index=False))

# ---------- Scenario 3: Small cell at risk ----------
# Construct a fractional table with a very small (rare) cell in Young–Low
# Paper's Fig. 3 table: rare Young-Low cell of 0.4; margins are the row and
# column sums (4, 6) and (2, 4, 4), so no re-fit is involved.
X_risk = np.array([[0.4, 2.3, 1.3],
                   [1.6, 1.7, 2.7]], dtype=float)
N_risk = int(round(X_risk.sum()))
row_R = X_risk.sum(axis=1).round().astype(int)   # [4, 6]
col_R = X_risk.sum(axis=0).round().astype(int)   # [2, 4, 4]

# Deterministic solutions
H_risk = hamilton_round(X_risk)
E_risk, E_kl_risk = best_entropy_integer(X_risk, row_R, col_R)

# Show that Young–Low cell (0,0) is often zeroed by Hamilton
rare_cell = (0, 0)
H_has_rare = int(H_risk[rare_cell] >= 1)
E_has_rare = int(E_risk[rare_cell] >= 1)

print("\n=== Scenario C: Small cell at risk ===")
print("Row sums:", row_R.tolist(), "Col sums:", col_R.tolist())
print("Hamilton rare cell >=1?:", bool(H_has_rare))
print("Entropy rare cell >=1?:", bool(E_has_rare))

# Figures for risk scenario
figC, axesC = plt.subplots(1, 3, figsize=(12, 3.6))
vmaxC = max(X_risk.max(), H_risk.max(), E_risk.max())
heatmap(axesC[0], X_risk, "Fractional IPF table (risk)", vmax=vmaxC)
row_deltaC = tuple((H_risk.sum(axis=1) - row_R).tolist())
col_deltaC = tuple((H_risk.sum(axis=0) - col_R).tolist())
heatmap(axesC[1], H_risk, f"Hamilton (grand total only)\nmargin error: rows {row_deltaC}, cols {col_deltaC}", vmax=vmaxC)
heatmap(axesC[2], E_risk, "KL optimum, Eq. (2)\nmargin error: 0", vmax=vmaxC)
figC.suptitle("C (small cell at risk)")
plt.tight_layout()
figC.savefig(OUTDIR / "toy_small_cell_risk.png", dpi=DPI)

# ---------- Entropy-informed randomized rounding around the risk scenario ----------
samples, probs, cov, feas = sample_entropy_weighted(
    X_risk, row_R, col_R, lam=10.0, n_samples=100, rng=np.random.default_rng(42)
)
coverage = pd.DataFrame(cov, index=["Young","Old"], columns=["Low","Mid","High"])

print("\nCoverage (probability cell >=1 across 100 samples):")
print(coverage.round(2).to_string())

# Plot coverage heatmap
fig_cov, ax_cov = plt.subplots(figsize=(4.5, 3.6))
im = ax_cov.imshow(coverage.values, aspect='auto')
ax_cov.set_xticks(range(3)); ax_cov.set_xticklabels(["Low","Mid","High"])
ax_cov.set_yticks(range(2)); ax_cov.set_yticklabels(["Young","Old"])
ax_cov.set_title("Coverage across randomized entropy samples")
for i in range(2):
    for j in range(3):
        ax_cov.text(j, i, f"{coverage.values[i,j]:.2f}", ha='center', va='center')
plt.tight_layout()
fig_cov.savefig(OUTDIR / "toy_entropy_coverage.png", dpi=DPI)

# Save sample frequency of the rare cell specifically
rare_freq = float(coverage.values[rare_cell])
pd.DataFrame({"cell":[str(rare_cell)], "coverage":[rare_freq]}).to_csv(
    OUTDIR / "toy_rare_cell_coverage.csv", index=False
)

# ---------- LaTeX snippet ----------
latex_snippet = r"""
\begin{figure}[t]
\centering
\includegraphics[width=.32\linewidth]{toy_entropy_vs_hamilton_A.png}
\includegraphics[width=.32\linewidth]{toy_entropy_vs_hamilton_B.png}
\includegraphics[width=.32\linewidth]{toy_small_cell_risk.png}
\caption{Toy examples. Left: baseline margins; middle: shifted controls (Hamilton violates columns, entropy matches); right: small-cell-at-risk case.}
\label{fig:toy_examples}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=.5\linewidth]{toy_entropy_coverage.png}
\caption{Entropy-informed randomized rounding: coverage (probability of allocating $\ge 1$) across 100 samples.}
\label{fig:toy_coverage}
\end{figure}
"""
with open(OUTDIR / "toy_figs_latex_snippet.tex", "w", encoding="utf-8") as f:
    f.write(latex_snippet)

print("\nSaved files:")
for p in ["toy_entropy_vs_hamilton_A.png",
          "toy_entropy_vs_hamilton_B.png",
          "toy_small_cell_risk.png",
          "toy_entropy_coverage.png",
          "toy_metrics_A_B.csv",
          "toy_rare_cell_coverage.csv",
          "toy_figs_latex_snippet.tex"]:
    print(" -", OUTDIR / p)
