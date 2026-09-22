"""Numerical checks quoted in the Mathematical Supplement (Sections S2-S6).

A. log L = -N D_KL(q||p) + C(N, xhat) exactly (Eq. S.6 of the supplement).
B. Method-of-types bounds on small tables; the type class of q is the most probable
   type class under q (Cover and Thomas 2006, Thm 11.1.4 and its lemma).
C. Expansions among near tables: gamma_k - 1 = (1-2r)/(2x) + O(1/x^2),
   ML cost log((n+1)/x) = (1-r)/x + O(1/x^2), Delta C_k = 1 - 1/(2x) + O(1/x^2).
D. The residual-space surrogate -log r_k equals gamma_k exactly when n_k = 0.
E. A 3x3 example with residuals consistent with unit row and column margins where the
   log-odds objective (Eq. 12) and the log-residual surrogate (Eq. 8) select different
   permutation tables (Proposition 2.2, overlapping case).
F. Eq. (13) is the I-projection of the secondary margin onto the supported cells.
G. Danish table: N, |K|, N D_KL, C(N, xhat), the crude bound -|K| log(N+1), and the share
   of residual units placed in cells with empty floors (where the surrogate is exact).
H. The two worked examples of Sections S3 and S4.

Needs output/fractional_fit.csv and output/integer_table.csv (stage step3).
Writes output/supplement/summary.txt and echoes it.
"""

from itertools import permutations, product
from math import lgamma, log, floor, exp
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LS = ROOT / "output"
OUT = ROOT / "output" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)
import builtins, io
_buf = io.StringIO()
_print = builtins.print


def print(*a, **k):
    _print(*a, **k)
    _print(*a, **k, file=_buf)

rng = random.Random(7)


def kl_N(counts, x):
    return sum(c * log(c / xk) for c, xk in zip(counts, x) if c)


def loglik(counts, p):
    n = sum(counts)
    if any(c and pk == 0 for c, pk in zip(counts, p)):
        return float("-inf")
    return lgamma(n + 1) - sum(lgamma(c + 1) for c in counts) + sum(c * log(pk) for c, pk in zip(counts, p) if c)


def C_of(counts):
    n = sum(counts)
    return lgamma(n + 1) - sum(lgamma(c + 1) for c in counts) + sum(c * log(c / n) for c in counts if c)


# ---- A. exact identity -----------------------------------------------------------
print("A. log L = -N D + C")
for _ in range(5):
    K = rng.randint(2, 8)
    raw = [rng.random() + 0.05 for _ in range(K)]
    p = [v / sum(raw) for v in raw]
    N = rng.randint(5, 60)
    counts = [0] * K
    for _ in range(N):
        u, acc = rng.random(), 0.0
        for k in range(K):
            acc += p[k]
            if u < acc:
                counts[k] += 1
                break
    x = [N * pk for pk in p]
    lhs = loglik(counts, p)
    rhs = -kl_N(counts, x) + C_of(counts)
    assert abs(lhs - rhs) < 1e-9, (lhs, rhs)
print("   identity holds to 1e-9 on 5 random tables")

# ---- B. type-class bounds ----------------------------------------------------------
print("B. method-of-types bounds")
for K, N in [(2, 7), (3, 6), (4, 5)]:
    raw = [rng.random() + 0.1 for _ in range(K)]
    p = [v / sum(raw) for v in raw]
    x = [N * pk for pk in p]
    types = [c for c in product(range(N + 1), repeat=K) if sum(c) == N]
    assert len(types) <= (N + 1) ** K
    for c in types:
        L = exp(loglik(c, p))
        D = kl_N(c, x) / N
        assert L <= exp(-N * D) + 1e-12
        assert L >= (N + 1) ** (-K) * exp(-N * D) - 1e-12
        # T(q) is the most probable type class under q itself
        q = [ck / N for ck in c]
        Pq = exp(loglik(c, q))
        for c2 in types:
            assert Pq >= exp(loglik(c2, q)) - 1e-12
    print(f"   K={K} N={N}: {len(types)} types, both bounds and the q-mode property hold")

# ---- C. expansions among near tables ----------------------------------------------
print("C. expansions (x, r fixed, error should shrink like 1/x^2)")
r = 0.3
for n in (5, 50, 500, 5000):
    x = n + r
    gamma = (n + 1) * log((n + 1) / x) - n * log(n / x)
    ml = log((n + 1) / x)
    dC = gamma - ml  # per-cell change in C when the cell is raised by one unit
    e1 = gamma - 1 - (1 - 2 * r) / (2 * x)
    e2 = ml - (1 - r) / x
    e3 = dC - (1 - 1 / (2 * x))
    print(f"   x={x:8.1f}: gamma-1={gamma-1:+.3e} approx={(1-2*r)/(2*x):+.3e} err={e1:+.1e} | "
          f"ml={ml:.3e} approx={(1-r)/x:.3e} err={e2:+.1e} | dC-1={dC-1:+.3e} err={e3:+.1e}")

# ---- D. surrogate exact on empty floors -------------------------------------------
print("D. n_k = 0: gamma_k = -log r_k")
for rr in (0.05, 0.3, 0.7, 0.95):
    gamma = 1 * log(1 / rr)
    assert abs(gamma - (-log(rr))) < 1e-12
print("   holds")

# ---- E. Proposition 2.2 example ----------------------------------------------------
print("E. overlapping margins: log-odds vs log-residual pick different tables")


def sinkhorn(M, it=500):
    M = np.array(M, float)
    for _ in range(it):
        M /= M.sum(1, keepdims=True)
        M /= M.sum(0, keepdims=True)
    return M


def best(M, f):
    vals = []
    for perm in permutations(range(3)):
        vals.append((sum(f(M[i, perm[i]]) for i in range(3)), perm))
    vals.sort(reverse=True)
    return vals


found = None
for trial in range(20000):
    M = sinkhorn(np.exp(np.random.default_rng(trial).normal(0, 1.6, (3, 3))))
    if M.min() < 0.02 or M.max() > 0.97:
        continue
    a = best(M, lambda v: log(v / (1 - v)))
    b = best(M, lambda v: log(v))
    if a[0][1] != b[0][1] and a[0][0] - a[1][0] > 0.05 and b[0][0] - b[1][0] > 0.05:
        found = (M, a, b)
        break
M, a, b = found
np.set_printoptions(precision=3, suppress=True)
print(M)
print("   row sums", M.sum(1), "col sums", M.sum(0))
print("   log-odds picks", a[0][1], f"(value {a[0][0]:.3f}, runner-up {a[1][0]:.3f})")
print("   log r    picks", b[0][1], f"(value {b[0][0]:.3f}, runner-up {b[1][0]:.3f})")
# a hand-rounded version with exact unit margins for the text
R3 = np.array([[0.050, 0.575, 0.375],
               [0.362, 0.395, 0.243],
               [0.588, 0.030, 0.382]])
print("   candidate rounded matrix rows", R3.sum(1), "cols", R3.sum(0))
a = best(R3, lambda v: log(v / (1 - v)))
b = best(R3, lambda v: log(v))
print("   rounded: log-odds ->", a[0][1], f"{a[0][0]:.3f} vs {a[1][0]:.3f};  log r ->", b[0][1], f"{b[0][0]:.3f} vs {b[1][0]:.3f}")

# ---- F. Eq. (13) as I-projection ---------------------------------------------------
print("F. I-projection onto the support")

t = np.array([4.0, 3.0, 2.0, 1.0])
supp = np.array([True, True, False, True])
T = t.sum()  # harmonized: the secondary margin sums to the target total
S = t[supp].sum()
proj = np.where(supp, t * T / S, 0.0)


def obj(u):
    return float(np.sum(u * np.log(u / t[supp])))


# stationarity: log(u_k / t_k) is the same constant on the support (Lagrange condition)
ratios = np.log(proj[supp] / t[supp])
assert np.ptp(ratios) < 1e-12
# optimality: every random feasible vector on the support has a larger objective
g = np.random.default_rng(0)
worst = min(obj(T * v / v.sum()) - obj(proj[supp]) for v in g.exponential(1.0, (5000, supp.sum())))
print("   rescaled:", proj[supp], f" log-ratio constant {ratios[0]:.6f} = log(T/S) {np.log(T/S):.6f};"
      f" min objective gap over 5000 feasible vectors {worst:.3e} (>= 0)")
assert worst >= -1e-12

# ---- G. Danish table --------------------------------------------------------------
print("G. Danish table")
import pandas as pd
KEY = ["ZoneID", "AgeID", "NumChildID", "FamID", "GenderID", "IncomeID", "LmaID"]
f = pd.read_csv(LS / "fractional_fit.csv")
g = pd.read_csv(LS / "integer_table.csv")
d = f.merge(g, on=KEY, how="inner", validate="one_to_one")
x = d.x.to_numpy(float)
xh = d.n.to_numpy(float)
N = xh.sum()
Nx = x.sum()
K = len(d)
q = xh / N
p = x / Nx
m = xh > 0
ND = float(np.sum(xh[m] * np.log(q[m] / p[m])))
C = float(lgamma(N + 1) - np.sum([lgamma(v + 1) for v in xh]) + np.sum(xh[m] * np.log(q[m])))
print(f"   |K| = {K:,}   N = {N:,.0f}   sum x = {Nx:,.2f}   positive integer cells = {m.sum():,}")
print(f"   N D_KL(q||p) = {ND:,.1f} nats   D_KL = {ND/N:.3e}")
print(f"   C(N, xhat)   = {C:,.1f} nats   crude bound -|K| log(N+1) = {-K*log(N+1):,.1f}")
print(f"   log L = {-ND + C:,.1f}")
n0 = np.floor(x)
rres = x - n0
z = xh - n0
inside = (z >= 0) & (z <= 1)
R = rres.sum()
print(f"   R = sum r_k = {R:,.1f}; cells with n_k = 0: {(n0==0).sum():,} ({(n0==0).mean()*100:.1f}%)")
print(f"   residual mass in n_k = 0 cells: {rres[n0==0].sum()/R*100:.1f}%")
print(f"   units placed (z=1) in n_k = 0 cells: {((z==1)&(n0==0)).sum():,} of {(z==1).sum():,} = {((z==1)&(n0==0)).sum()/(z==1).sum()*100:.1f}%")
print(f"   floor/ceiling class respected: {inside.all()}")

# ---- H. worked examples quoted in the supplement ----------------------------------
print("H. worked examples")
# S3: x = (10.4, 2.6), N = 13; floor-ceiling tables (11,2) and (10,3)
x2 = [10.4, 2.6]
p2 = [v / 13 for v in x2]
for tab in [(11, 2), (10, 3)]:
    print(f"   table {tab}: N*D = {kl_N(tab, x2):.4f}   log L = {loglik(tab, p2):.4f}")
print(f"   L(11,2)/L(10,3) = {exp(loglik((11,2), p2) - loglik((10,3), p2)):.4f}  (3/11 * 10.4/2.6 = {3/11*10.4/2.6:.4f})")


def gam(n, x):
    return (n + 1) * log((n + 1) / x) - (n * log(n / x) if n else 0.0)


# S4: surrogate vs exact cost, cells a = 1.40 (n=1, r=.40) and b = 30.30 (n=30, r=.30)
for lab, n, x in [("a", 1, 1.40), ("b", 30, 30.30)]:
    print(f"   cell {lab}: x={x}: sigma=-log r = {-log(x-n):.4f}   gamma = {gam(n, x):.4f}   "
          f"gamma approx 1+(1-2r)/(2x) = {1 + (1-2*(x-n))/(2*x):.4f}   phi = {log((n+1)/x):.4f}")

(OUT / "summary.txt").write_text(_buf.getvalue(), encoding="utf-8")
_print("wrote", OUT / "summary.txt")
