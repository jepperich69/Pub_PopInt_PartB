"""Exact KL integerization under the target margins (Zone x Age x Gender) and the
optimality gaps of the floor-ceiling solutions (Sections 2.4 and 3.2 of the paper).

Per zone, the integer tables are compared on
    F(n) = sum_k f_k(n_k),   f_k(m) = m log(m/x_k) - m + x_k,
which equals N * D_KL(q||p) when the totals agree. Tables:
  OPT   exact optimum over all integer tables with the Age x Gender margins exact
  FC    exact optimum within the floor-ceiling class under the same margins
  PRE   the deterministic largest-remainder table         (integer_table.csv)
  REP   the swap-repaired table                           (integer_repaired.csv)
  ALG   the seeded slice-sampling table of Step 2         (tmp_Minimal_Integerized.csv)
Gaps are relative: (F_X - F_OPT) / F_OPT.

Both OPT and FC are solved by marginal analysis (greedy). The target margins form a
single joint margin, so the problem separates into one allocation problem per
Age x Gender slice, and because f_k is convex the unit increments of each cell are
increasing; filling the cheapest increments first is therefore exact (Fox 1966,
Ibaraki and Katoh 1988). The original check used Gurobi with the same unit-increment
formulation and gives the same objective values.

Usage: python kl_gap.py [NZONES]   (default: all 98 zones, largest first)
Outputs: output/kl_gap/per_zone.csv and summary.txt
"""
import heapq
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LS = ROOT / "output"
OUT = LS / "kl_gap"
OUT.mkdir(parents=True, exist_ok=True)
KEY = ["ZoneID", "AgeID", "NumChildID", "FamID", "GenderID", "IncomeID", "LmaID"]
NZONES = int(sys.argv[1]) if len(sys.argv) > 1 else 98
BELOW, ABOVE = 3, 3          # search range: [floor-BELOW, ceil+ABOVE], widened if hit


def f(m, x):
    return x if m == 0 else m * math.log(m / x) - m + x


def F(n, x):
    return float(sum(f(int(a), b) for a, b in zip(n, x)))


def theta(n, x):
    """share of population mass in the right cells, sum_k min(p_k, q_k)."""
    return float(np.minimum(np.asarray(n, float) / np.sum(n), x / x.sum()).sum())


def solve(x, groups, lo, hi, tag):
    """min sum f_k(n_k) s.t. group sums fixed, lo_k <= n_k <= hi_k integer.

    Greedy marginal analysis per group: start every cell at lo_k and add the
    cheapest available unit increment until the group total is reached. Exact
    because the increments f_k(m) - f_k(m-1) are increasing in m.
    """
    t0 = time.time()
    n = np.array(lo, dtype=int)
    for g, idx in groups.items():
        target = round(sum(x[i] for i in idx))
        need = target - int(n[idx].sum())
        assert need >= 0, (tag, g, need)
        heap = []
        for k in idx:
            if n[k] < hi[k]:
                heap.append((f(n[k] + 1, x[k]) - f(n[k], x[k]), int(k)))
        heapq.heapify(heap)
        for _ in range(need):
            assert heap, (tag, g, "range too narrow")
            _, k = heapq.heappop(heap)
            n[k] += 1
            if n[k] < hi[k]:
                heapq.heappush(heap, (f(n[k] + 1, x[k]) - f(n[k], x[k]), k))
    return n, F(n, x), time.time() - t0


fr = pd.read_csv(LS / "fractional_fit.csv")
alg = pd.read_csv(LS / "tmp_Minimal_Integerized.csv").rename(columns={"n": "n_alg"})
rep = pd.read_csv(LS / "integer_repaired.csv").rename(columns={"n": "n_rep"})
pre = pd.read_csv(LS / "integer_table.csv").rename(columns={"n": "n_pre"})
df = fr.merge(alg, on=KEY, how="left").merge(rep, on=KEY, how="left").merge(pre, on=KEY, how="left").fillna(0)
zones = df.groupby("ZoneID")["x"].sum().sort_values(ascending=False).index[:NZONES]

rows = []
for z in zones:
    d = df[(df.ZoneID == z) & (df.x > 0)].reset_index(drop=True)
    x = d.x.to_numpy(); N = round(x.sum())
    groups = {g: idx.to_numpy() for g, idx in d.groupby(["AgeID", "GenderID"]).groups.items()}
    fl = np.floor(x + 1e-12).astype(int); ce = np.ceil(x - 1e-12).astype(int)
    # exact optimum, widen the range until no cell sits on a bound
    below, above = BELOW, ABOVE
    while True:
        lo = np.maximum(fl - below, 0); hi = ce + above
        n_opt, F_opt, dt_opt = solve(x, groups, lo, hi, f"{z}-OPT")
        hit = ((n_opt == lo) & (lo > 0)).sum() + (n_opt == hi).sum()
        if hit == 0: break
        below += 2; above += 2
    n_fc, F_fc, dt_fc = solve(x, groups, fl, ce, f"{z}-FC")
    F_alg = F(d.n_alg.to_numpy(), x); F_rep = F(d.n_rep.to_numpy(), x); F_pre = F(d.n_pre.to_numpy(), x)
    ag = d.groupby(["AgeID", "GenderID"])
    ok_alg = (ag.n_alg.sum().round() == ag.x.sum().round()).all()
    ok_rep = (ag.n_rep.sum().round() == ag.x.sum().round()).all()
    ok_pre = (ag.n_pre.sum().round() == ag.x.sum().round()).all()
    inclass_pre = bool(((d.n_pre >= fl) & (d.n_pre <= ce)).all()); inclass_rep = bool(((d.n_rep >= fl) & (d.n_rep <= ce)).all())
    out = (n_opt < fl).sum(), (n_opt > ce).sum(), int((fl - n_opt).clip(0).sum() + (n_opt - ce).clip(0).sum())
    r = dict(ZoneID=z, N=N, cells=len(x), F_opt=F_opt, F_fc=F_fc, F_alg=F_alg, F_rep=F_rep, F_pre=F_pre,
             gap_fc=(F_fc - F_opt) / F_opt, gap_alg=(F_alg - F_opt) / F_opt, gap_rep=(F_rep - F_opt) / F_opt, gap_pre=(F_pre - F_opt) / F_opt,
             alg_margins_ok=bool(ok_alg), rep_margins_ok=bool(ok_rep), pre_margins_ok=bool(ok_pre), pre_inclass=inclass_pre, rep_inclass=inclass_rep,
             cells_below_floor=int(out[0]), cells_above_ceil=int(out[1]), units_outside=out[2],
             max_dist=int(max((fl - n_opt).max(), (n_opt - ce).max(), 0)),
             theta_opt=theta(n_opt, x), theta_fc=theta(n_fc, x), theta_alg=theta(d.n_alg.to_numpy(), x),
             theta_rep=theta(d.n_rep.to_numpy(), x), theta_pre=theta(d.n_pre.to_numpy(), x),
             kl_opt_per_person=F_opt / N, kl_fc_per_person=F_fc / N,
             t_opt=dt_opt, t_fc=dt_fc, range=(below, above))
    rows.append(r); print({k: (round(v, 9) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "per_zone.csv", index=False)

t = pd.DataFrame(rows)
lines = [f"summary over {len(t)} zones",
         "F_opt-weighted gap FC: %.3e  PRE: %.3e  REP: %.3e  ALG(step2): %.3e" % tuple(
             (t[c] * t.F_opt).sum() / t.F_opt.sum() for c in ("gap_fc", "gap_pre", "gap_rep", "gap_alg")),
         "max gap FC: %.3e  PRE: %.3e  REP: %.3e  ALG: %.3e" % (t.gap_fc.max(), t.gap_pre.max(), t.gap_rep.max(), t.gap_alg.max()),
         "theta (unweighted zone mean): OPT %.4f  FC %.4f  PRE %.4f  REP %.4f  ALG %.4f" % tuple(
             t[c].mean() for c in ("theta_opt", "theta_fc", "theta_pre", "theta_rep", "theta_alg")),
         "units outside class in OPT: %d of %d persons (%d cells below floor, %d above ceiling); max distance %d" % (
             t.units_outside.sum(), t.N.sum(), t.cells_below_floor.sum(), t.cells_above_ceil.sum(), t.max_dist.max()),
         "zones where OPT leaves the floor-ceiling class: %d of %d" % ((t.units_outside > 0).sum(), len(t)),
         "absolute FC gap per person (nats): overall %.2e, worst zone %.2e" % (
             (t.F_fc - t.F_opt).sum() / t.N.sum(), ((t.F_fc - t.F_opt) / t.N).max())]
print("\n" + "\n".join(lines))
(OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
