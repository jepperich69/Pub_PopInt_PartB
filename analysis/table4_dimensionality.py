"""
Dimensionality stress test for Reviewer 1, point 5 (TRB-D-25-00887 R1).

Question: does the two-step KL integerization "scale structurally" when the
joint table has many more attributes than the margins control?

Design
------
Start from the real fitted Danish table (98 municipalities, 6 fitting
attributes: Age, Children, Family, Gender, Income, LMA; all six are covered by
the five Age x {.} margins). Add k = 0..4 synthetic *descriptive* attributes
with 5 categories each, not covered by any margin. Category probabilities
depend on (Age, Income) through a Dirichlet perturbation of the base vector
(0.40, 0.30, 0.20, 0.08, 0.02), so each attribute has a rare category (~2%).

    x_full[B, c] = x_B * prod_j p_j(c_j | age_B, income_B)

At k = 4 a municipality of 6,500 blocks has 4.05 million cells.

Methods compared on the full table
----------------------------------
  direct   : the paper's floor + largest-remainder-in-anchor-slices applied to
             the full table without aggregation (no swap repair).
  hier     : hierarchical / aggregate-first. Block totals N_B are the paper's
             final integer table (integer_repaired.csv); within each block the
             N_B units are allocated by floor + largest remainder over the
             sub-cells. By the KL chain rule the within-block problem is a
             single-total problem, independent of the block level.
  hybrid   : same N_B; within block, multinomial sampling (R reps).
  multi    : multinomial sampling of the zone total over the full table (R reps).

Metrics per (zone, k, method)
-----------------------------
  floor_share_full : sum floor(x_full) / N               (method independent)
  theta_block      : sum_B min(N_B, X_B) / N             (fitting-attribute overlap)
  theta_full       : sum min(n, x_full) / N
  kl_full          : KL(n/N || x/N) over the full table (eps-guarded)
  sec_L1           : L1 violation of the four secondary margins (Age x Children,
                     Family, Income, LMA) by the block aggregate
  desc_L1          : L1 error of the descriptive single-attribute marginals
                     (sum over added attributes) relative to fractional, per N
  rare_ratio       : integer count / fractional count for the rare category,
                     pooled over added attributes
  time_s           : wall time of the allocation

Outputs -> output/table4/{per_zone.csv, summary.csv, table4_unweighted.csv, SUMMARY.md}
The paper's Table 4 is table4_unweighted.csv (unweighted zone means, the paper's convention);
summary.csv is population-weighted.
"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LS = os.path.join(ROOT, "output")      # pipeline outputs
DATA = os.path.join(ROOT, "data")    # target margins
OUT = os.path.join(LS, "table4")
os.makedirs(OUT, exist_ok=True)

KEY = ["AgeID", "NumChildID", "FamID", "GenderID", "IncomeID", "LmaID"]
SEC = {"Children": "NumChildID", "Family": "FamID", "Income": "IncomeID", "Lma": "LmaID"}
C = 5                                   # categories per added attribute
BASE = np.array([0.40, 0.30, 0.20, 0.08, 0.02])
ALPHA = 50.0                            # Dirichlet concentration
K_LIST = [0, 1, 2, 3, 4]
R = int(os.environ.get("STRESS_R", 10))  # sampling replications
ZONES = os.environ.get("STRESS_ZONES")   # optional comma list for quick runs
EPS = 1e-12


def kl_norm(n, x):
    p = n / n.sum(); q = x / x.sum()
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], EPS))))


def theta(n, x):
    return float(np.minimum(n, x).sum() / x.sum())


def floor_lr_groups(x, group_id, target_by_group, n_groups):
    """Floor, then largest remainder within groups until each group's target is met.
    Groups whose floor already exceeds the target are trimmed on smallest fractions.
    Returns integer array."""
    n = np.floor(x + 1e-9).astype(np.int64)
    frac = x - n
    flo = np.bincount(group_id, weights=n, minlength=n_groups).astype(np.int64)
    need = target_by_group - flo
    # rank fractions within group: sort by (group, -frac)
    order = np.lexsort((-frac, group_id))
    g_sorted = group_id[order]
    # position within group
    starts = np.searchsorted(g_sorted, np.arange(n_groups), side="left")
    pos = np.arange(len(order)) - starts[g_sorted]
    add_mask = pos < need[g_sorted]            # top-need fractions get +1
    n[order[add_mask]] += 1
    # trimming (rare): groups with need < 0 -> remove units from smallest frac with n>0
    neg = np.where(need < 0)[0]
    for g in neg:
        idx = np.where(group_id == g)[0]
        idx = idx[n[idx] > 0]
        take = int(-need[g])
        rm = idx[np.argsort(frac[idx])[:take]]
        n[rm] -= 1
    return n


def sample_within_groups(x, group_id, N_by_group, n_groups, rng):
    """Multinomial within each group: N_g units over the group's cells with
    probabilities proportional to x. Vectorised via one searchsorted on
    (group + within-group cumulative)."""
    tot = np.bincount(group_id, weights=x, minlength=n_groups)
    p = x / np.maximum(tot[group_id], EPS)
    order = np.argsort(group_id, kind="stable")
    g_sorted = group_id[order]
    cum = np.cumsum(p[order])
    starts = np.searchsorted(g_sorted, np.arange(n_groups), side="left")
    base = np.zeros(len(order)); base[:] = np.concatenate(([0.0], cum))[starts[g_sorted]]
    cum_within = np.minimum(cum - base, 1.0)
    key = g_sorted + cum_within                     # monotone, block-major
    reps = np.repeat(np.arange(n_groups), N_by_group)
    u = rng.random(len(reps))
    q = reps + u
    pos = np.searchsorted(key, q, side="left")
    pos = np.minimum(pos, len(key) - 1)
    n = np.bincount(order[pos], minlength=len(x)).astype(np.int64)
    return n


def main():
    frac = pd.read_csv(os.path.join(LS, "fractional_fit.csv"))
    integ = pd.read_csv(os.path.join(LS, "integer_repaired.csv"))
    df = frac.merge(integ, on=["ZoneID"] + KEY, how="left")
    df["n"] = df["n"].fillna(0).astype(np.int64)
    margins = {}
    for name, col in SEC.items():
        m = pd.read_csv(os.path.join(DATA, f"in_Pop_TargetZoneAge{name}.csv"))
        margins[name] = m.rename(columns={"ZoneID0": "ZoneID"})
    anchor = pd.read_csv(os.path.join(DATA, "in_Pop_TargetZoneAgeGender.csv")).rename(columns={"ZoneID0": "ZoneID"})

    zones = sorted(df.ZoneID.unique())
    if ZONES:
        zones = [int(z) for z in ZONES.split(",")]

    rng_feat = np.random.default_rng(20260914)
    # descriptive-attribute probabilities per (age, income), for up to 4 attributes
    ages = sorted(df.AgeID.unique()); incs = sorted(df.IncomeID.unique())
    P = {}
    for j in range(max(K_LIST)):
        for a in ages:
            for i in incs:
                P[(j, a, i)] = rng_feat.dirichlet(ALPHA * BASE)

    rows = []
    for zi, z in enumerate(zones):
        dz = df[df.ZoneID == z].reset_index(drop=True)
        xB = dz.x.to_numpy(); NB = dz.n.to_numpy(); Nz = int(round(xB.sum()))
        nblk = len(dz)
        age = dz.AgeID.to_numpy(); inc = dz.IncomeID.to_numpy(); gen = dz.GenderID.to_numpy()
        # anchor slice id and target
        a_key = pd.Series(list(zip(age, gen)))
        slices = {k: s for s, k in enumerate(sorted(set(a_key)))}
        slice_id_B = np.array([slices[k] for k in a_key])
        az = anchor[anchor.ZoneID == z]
        tgt_slice = np.zeros(len(slices), dtype=np.int64)
        for _, r in az.iterrows():
            k = (int(r.AgeID), int(r.GenderID))
            if k in slices:
                tgt_slice[slices[k]] = int(round(r.Val))
        # secondary margin targets and block-level column values
        sec_targets = {}
        for name, col in SEC.items():
            mz = margins[name][margins[name].ZoneID == z]
            sec_targets[name] = {(int(r.AgeID), int(r[col])): float(r.Val) for _, r in mz.iterrows()}
        sec_cols = {name: dz[col].to_numpy() for name, col in SEC.items()}

        def sec_L1(Nblk):
            tot = 0.0
            for name, col in SEC.items():
                s = pd.Series(Nblk).groupby([age, sec_cols[name]]).sum()
                for key, val in sec_targets[name].items():
                    tot += abs(s.get(key, 0) - val)
            return tot

        # per-block descriptive probability vectors for each attribute j
        for k in K_LIST:
            S = C ** k
            t0 = time.time()
            # build full table: x_full shape (nblk, S), sub-cell index c -> digits base C
            w = np.ones((nblk, S))
            digits = np.zeros((k, S), dtype=np.int64)
            for j in range(k):
                digits[j] = (np.arange(S) // (C ** j)) % C
            for j in range(k):
                pj = np.stack([P[(j, a, i)] for a, i in zip(age, inc)])   # (nblk, C)
                w *= pj[:, digits[j]]
            x_full = (xB[:, None] * w).ravel()
            block_id = np.repeat(np.arange(nblk), S)
            slice_id = np.repeat(slice_id_B, S)
            t_build = time.time() - t0
            floor_share = float(np.floor(x_full + 1e-9).sum() / Nz)
            cells = len(x_full)

            # descriptive fractional marginals (per attribute, per category)
            def desc_marg(n_full):
                out = np.zeros((k, C))
                nf = n_full.reshape(nblk, S)
                for j in range(k):
                    for c in range(C):
                        out[j, c] = nf[:, digits[j] == c].sum()
                return out
            xd = desc_marg(x_full) if k > 0 else None

            def record(method, n_full, t_s, rep=None):
                Nblk = n_full.reshape(nblk, S).sum(axis=1)
                rec = dict(ZoneID=z, N=Nz, k=k, cells=cells, method=method, rep=rep,
                           floor_share_full=floor_share,
                           theta_block=theta(Nblk, xB), theta_full=theta(n_full, x_full),
                           kl_full=kl_norm(n_full, x_full), sec_L1=sec_L1(Nblk),
                           time_s=t_s)
                if k > 0:
                    nd = desc_marg(n_full)
                    rec["desc_L1"] = float(np.abs(nd - xd).sum() / Nz / k)
                    rec["rare_ratio"] = float(nd[:, C - 1].sum() / max(xd[:, C - 1].sum(), EPS))
                    rec["rare_zero"] = float((nd[:, C - 1] == 0).mean())
                else:
                    rec["desc_L1"] = np.nan; rec["rare_ratio"] = np.nan; rec["rare_zero"] = np.nan
                rows.append(rec)

            # --- direct: floor + LR within anchor slices on the full table
            t0 = time.time()
            n_dir = floor_lr_groups(x_full, slice_id, tgt_slice, len(slices))
            record("direct", n_dir, time.time() - t0)

            # --- hierarchical: paper's N_B, floor + LR within blocks
            t0 = time.time()
            n_hier = floor_lr_groups(x_full, block_id, NB, nblk)
            record("hier", n_hier, time.time() - t0 + 0.0)

            rng = np.random.default_rng(1000 * zi + k)
            # --- hybrid: paper's N_B, multinomial within blocks
            for r in range(R):
                t0 = time.time()
                n_hyb = sample_within_groups(x_full, block_id, NB, nblk, rng)
                record("hybrid", n_hyb, time.time() - t0, rep=r)
            # --- full multinomial
            p = x_full / x_full.sum()
            for r in range(R):
                t0 = time.time()
                n_mul = rng.multinomial(Nz, p).astype(np.int64)
                record("multi", n_mul, time.time() - t0, rep=r)

            print(f"zone {z} ({zi+1}/{len(zones)}) k={k} cells={cells:,} floor={floor_share:.3f} "
                  f"theta_block dir/hier={rows[-2*R-2]['theta_block']:.4f}/{rows[-2*R-1]['theta_block']:.4f} "
                  f"build {t_build:.1f}s", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, "per_zone.csv"), index=False)

    # population-weighted summary across zones (mean over reps first)
    agg = per.groupby(["ZoneID", "k", "method"], as_index=False).mean(numeric_only=True)
    def wavg(g, col):
        return np.average(g[col], weights=g["N"])
    summ = []
    for (k, m), g in agg.groupby(["k", "method"]):
        summ.append(dict(k=k, method=m, cells_total=int(g["cells"].sum()),
                         floor_share_full=wavg(g, "floor_share_full"),
                         theta_block=wavg(g, "theta_block"), theta_full=wavg(g, "theta_full"),
                         kl_full=wavg(g, "kl_full"), sec_L1_total=g["sec_L1"].sum(),
                         desc_L1=wavg(g, "desc_L1") if k > 0 else np.nan,
                         rare_ratio=wavg(g, "rare_ratio") if k > 0 else np.nan,
                         rare_zero=wavg(g, "rare_zero") if k > 0 else np.nan,
                         time_s_total=g["time_s"].sum()))
    summ = pd.DataFrame(summ).sort_values(["k", "method"])
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    with open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("# Dimensionality stress test (R1 point 5)\n\n")
        f.write(f"Zones: {len(zones)}; sampling reps R={R}; categories per added attribute C={C}; base probs {BASE.tolist()}.\n\n")
        f.write("```" + chr(10) + summ.to_string(index=False, float_format=lambda v: f"{v:.4f}") + chr(10) + "```")
        f.write("\n")
    print(summ.to_string())

    # unweighted zone means = the paper's Table 4 (direct, hier = aggregate-first, multi)
    unw = (agg.groupby(["k", "method"])[["floor_share_full", "theta_block", "theta_full"]]
              .mean().round(3).unstack("method"))
    unw.to_csv(os.path.join(OUT, "table4_unweighted.csv"))
    print("\nTable 4 (unweighted zone means):")
    print(unw.to_string())


if __name__ == "__main__":
    main()
