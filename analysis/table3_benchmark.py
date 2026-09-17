"""Reviewer 1 point 3: benchmark on the paper's own pipeline (output/).

Per municipality (98 zones), compare five integer tables against the fractional
IPF table x:
  det_repaired  paper's method with swap repair      (output/integer_repaired.csv)
  det_minimal   Step 3 pre-swap intermediate         (output/integer_table.csv)
  det_step2     paper's "minimal allocation", 0.977  (output/tmp_Minimal_Integerized.csv)
  hamilton      floor + largest remainders, single zone total, no margins
  trs           floor + residual units drawn without replacement, p ~ r  (R reps)
  multinomial   N draws with replacement, p ~ x                          (R reps)

Metrics per zone: theta = sum min(p, q); L1 on the controlling margin
(Age x Gender) and on the four secondary margins, in persons and as share of
zone population; mass placed in sub-unit cells (x < 1) relative to fractional
mass. Zone means are unweighted (paper convention). Outputs in
output/table3/.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LS = ROOT / "output"          # pipeline outputs
OUT = LS / "table3"
OUT.mkdir(exist_ok=True)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEED = 12345
DIMS = ["AgeID", "NumChildID", "FamID", "GenderID", "IncomeID", "LmaID"]
ANCHOR = ["AgeID", "GenderID"]
SECONDARY = [["AgeID", "IncomeID"], ["AgeID", "FamID"], ["AgeID", "NumChildID"], ["AgeID", "LmaID"]]

log = open(OUT / "run.log", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(v) for v in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


def theta(x, n):
    return float(np.minimum(x / x.sum(), n / n.sum()).sum())


def margin_l1(x, n, codes):
    """codes: integer group id per cell. L1 between fractional and integer group sums."""
    fx = np.bincount(codes, weights=x)
    fn = np.bincount(codes, weights=n)
    return float(np.abs(fx - fn).sum())


def hamilton(x):
    fl = np.floor(x).astype(np.int64)
    need = int(round(x.sum())) - int(fl.sum())
    r = x - fl
    if need > 0:
        fl[np.argsort(-r, kind="stable")[:need]] += 1
    return fl


def trs(x, rng):
    fl = np.floor(x).astype(np.int64)
    need = int(round(x.sum())) - int(fl.sum())
    r = x - fl
    out = fl.copy()
    if need > 0 and r.sum() > 0:
        idx = rng.choice(len(x), size=need, replace=False, p=r / r.sum())
        out[idx] += 1
    return out


def multinomial(x, rng):
    return rng.multinomial(int(round(x.sum())), x / x.sum())


def metrics(x, n, codes_anchor, codes_sec, sub):
    m = {"theta": theta(x, n),
         "anchor_l1": margin_l1(x, n, codes_anchor),
         "sec_l1": sum(margin_l1(x, n, c) for c in codes_sec),
         "sub_mass": float(n[sub].sum())}
    return m


def main():
    t0 = time.time()
    say(f"=== benchmark run {time.strftime('%Y-%m-%d %H:%M:%S')}  R={R}")
    frac = pd.read_csv(LS / "fractional_fit.csv")
    rep = pd.read_csv(LS / "integer_repaired.csv")
    mini = pd.read_csv(LS / "integer_table.csv")
    step2 = pd.read_csv(LS / "tmp_Minimal_Integerized.csv")   # paper's "minimal" (theta 0.977)
    key = ["ZoneID"] + DIMS
    df = frac.merge(rep.rename(columns={"n": "n_rep"}), on=key, how="left") \
             .merge(mini.rename(columns={"n": "n_min"}), on=key, how="left") \
             .merge(step2.rename(columns={"n": "n_s2"}), on=key, how="left")
    df[["n_rep", "n_min", "n_s2"]] = df[["n_rep", "n_min", "n_s2"]].fillna(0).astype(np.int64)
    say("cells", len(df), "zones", df.ZoneID.nunique(),
        "unmatched repaired", int(rep.n.sum() - df.n_rep.sum()),
        "unmatched minimal", int(mini.n.sum() - df.n_min.sum()))
    rng = np.random.default_rng(SEED)
    rows = []
    for zi, (z, g) in enumerate(df.groupby("ZoneID", sort=True)):
        x = g.x.to_numpy(float)
        N = x.sum()
        sub = x < 1
        ca = pd.factorize(pd.MultiIndex.from_frame(g[ANCHOR]))[0]
        cs = [pd.factorize(pd.MultiIndex.from_frame(g[s]))[0] for s in SECONDARY]
        base = {"ZoneID": z, "N": N, "cells": len(x), "sub_cells": int(sub.sum()),
                "sub_mass_frac": float(x[sub].sum())}
        det = {"det_repaired": g.n_rep.to_numpy(np.int64),
               "det_minimal": g.n_min.to_numpy(np.int64),
               "det_step2": g.n_s2.to_numpy(np.int64),
               "hamilton": hamilton(x)}
        for name, n in det.items():
            rows.append({**base, "method": name, "rep": -1, **metrics(x, n, ca, cs, sub)})
        for name, fn in (("trs", trs), ("multinomial", multinomial)):
            for r_ in range(R):
                n = fn(x, rng)
                rows.append({**base, "method": name, "rep": r_, **metrics(x, n, ca, cs, sub)})
        if zi % 10 == 0:
            say(f"zone {zi+1}/98 {z} N={N:.0f} cells={len(x)} t={time.time()-t0:.0f}s")
    per = pd.DataFrame(rows)
    per["anchor_pct"] = 100 * per.anchor_l1 / per.N
    per["sec_pct"] = 100 * per.sec_l1 / per.N
    per["sub_ratio"] = per.sub_mass / per.sub_mass_frac
    per.to_csv(OUT / "per_zone_rep.csv", index=False)

    # zone-level: mean over reps for stochastic methods, plus std of theta
    zone = per.groupby(["ZoneID", "method"]).agg(
        N=("N", "first"), theta=("theta", "mean"), theta_sd=("theta", "std"),
        anchor_pct=("anchor_pct", "mean"), sec_pct=("sec_pct", "mean"),
        anchor_l1=("anchor_l1", "mean"), sec_l1=("sec_l1", "mean"),
        sub_ratio=("sub_ratio", "mean")).reset_index()
    zone["theta_sd"] = zone.theta_sd.fillna(0.0)
    zone.to_csv(OUT / "per_zone.csv", index=False)

    order = ["det_repaired", "det_minimal", "det_step2", "hamilton", "trs", "multinomial"]
    top20 = zone[zone.method == "hamilton"].nlargest(20, "N").ZoneID
    summ = zone.groupby("method").agg(
        theta_mean=("theta", "mean"), theta_p05=("theta", lambda s: s.quantile(.05)),
        theta_p95=("theta", lambda s: s.quantile(.95)), theta_sd_mean=("theta_sd", "mean"),
        anchor_pct=("anchor_pct", "mean"), sec_pct=("sec_pct", "mean"),
        sub_ratio=("sub_ratio", "mean")).reindex(order)
    summ20 = zone[zone.ZoneID.isin(top20)].groupby("method").agg(
        anchor_pct_top20=("anchor_pct", "mean"), sec_pct_top20=("sec_pct", "mean"),
        theta_top20=("theta", "mean")).reindex(order)
    summ = summ.join(summ20)
    # population-weighted theta for the record
    w = zone.pivot(index="ZoneID", columns="method", values="theta")
    Nz = zone.groupby("ZoneID").N.first().reindex(w.index)
    summ["theta_popweighted"] = [(w[m] * Nz).sum() / Nz.sum() for m in order]
    summ.to_csv(OUT / "summary.csv")
    say("\n" + summ.round(4).to_string())
    say(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
