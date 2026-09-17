"""Recompute Table 1 (error relative to the fractional baseline, 20 largest zones) for R1B.

Two corrections to the submitted table.
(1) Rows. The submitted "Minimal" row is output/tmp_Minimal_Integerized.csv, a seeded
    PPS table (see fig5_theta_violin.py); the deterministic pre-swap table is
    output/integer_table.csv. The submitted "Integerized" row (27.9 / 0.045) predates
    the final pipeline run: output/Table1_ErrorRelativeFraction.tex (2025-09-30) already
    says 20.443 / 0.036 for integer_repaired.csv.
(2) KL direction. The pipeline's _safe_kl(p, q) computes KL(p||q) with eps-clipping, so empty
    integer cells dominate. The paper's objective, Eq. (1), is D_KL(q||p) with q the integer
    shares; this script reports that, with 0 log 0 = 0.

Zones: the 20 in output/tmp_VariationBaseline_Summary.csv (largest by population, written by step3).
Sampling: multinomial, R = 200 per zone, seed 42, zone total = repaired total.
Output: output/table1/table1.csv and per_zone.csv. Paper Table 1 = rows repaired / minimal_det /
sampling with the L2 and KLqp columns.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LS = ROOT / "output"          # pipeline outputs
OUT = LS / "table1"
OUT.mkdir(parents=True, exist_ok=True)
KEY = ["ZoneID", "AgeID", "NumChildID", "FamID", "GenderID", "IncomeID", "LmaID"]

zones = list(pd.read_csv(LS / "tmp_VariationBaseline_Summary.csv").ZoneID)
f = pd.read_csv(LS / "fractional_fit.csv")
tabs = {"repaired": "integer_repaired.csv", "minimal_det": "integer_table.csv",
        "seeded_pps": "tmp_Minimal_Integerized.csv"}
d = f
for name, fn in tabs.items():
    d = d.merge(pd.read_csv(LS / fn).rename(columns={"n": name}), on=KEY, how="inner", validate="one_to_one")


def kl_qp(n, x):
    q = n / n.sum()
    p = x / x.sum()
    m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / p[m])))


def kl_pq_clipped(p, q, eps=1e-12):  # the pipeline's convention, for the record
    p = np.clip(p / p.sum(), eps, 1.0)
    q = np.clip(q / max(q.sum(), eps), eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


rng = np.random.default_rng(42)
rows = []
for z in zones:
    g = d[d.ZoneID == z]
    x = g.x.to_numpy(float)
    p = x / x.sum()
    row = {"ZoneID": z, "N": int(round(g.repaired.sum()))}
    for name in tabs:
        n = g[name].to_numpy(float)
        row[f"{name}_L2"] = float(np.linalg.norm(n - x))
        row[f"{name}_KLqp"] = kl_qp(n, x)
        row[f"{name}_KLpq_clipped"] = kl_pq_clipped(p, n / n.sum())
    T = row["N"]
    l2, kq, kp = [], [], []
    for _ in range(200):
        ns = rng.multinomial(T, p).astype(float)
        l2.append(np.linalg.norm(ns - x))
        kq.append(kl_qp(ns, x))
        kp.append(kl_pq_clipped(p, ns / T))
    for arr, lab in [(np.array(l2), "L2"), (np.array(kq), "KLqp"), (np.array(kp), "KLpq_clipped")]:
        row[f"samp_{lab}_mean"] = arr.mean()
        row[f"samp_{lab}_p05"] = np.percentile(arr, 5)
        row[f"samp_{lab}_p95"] = np.percentile(arr, 95)
    rows.append(row)

pz = pd.DataFrame(rows)
pz.to_csv(OUT / "per_zone.csv", index=False)
mean = pz.drop(columns=["ZoneID", "N"]).mean()
mean.to_csv(OUT / "table1.csv", header=["mean_over_20_zones"])
pd.set_option("display.width", 200)
print(mean.round(4).to_string())
