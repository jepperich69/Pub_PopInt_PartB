"""Regenerate the theta violin figure for R1B with the deterministic minimal allocation.

The submitted figure's middle panel ("Minimal") was output/tmp_Minimal_Integerized.csv,
which Step 2 wrote with seeded PPS sampling inside each Age x Gender slice (step2_anchor_pps,
seed 123), not with the largest-remainder rule of the paper. The deterministic pre-swap
table is output/integer_table.csv (Step 3 _Determ, step2_anchor_deterministic).
Per-zone theta values come from output/table3/per_zone_rep.csv (R = 200
multinomial draws per zone, seed 12345); the sampled panel shows the per-zone median.

Outputs: output/figures/fig5_overlap_theta_violin.png and the Table 2 statistics on stdout.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rep = pd.read_csv(ROOT / "output" / "table3" / "per_zone_rep.csv")
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

det_rep = rep[rep.method == "det_repaired"].set_index("ZoneID").theta
det_min = rep[rep.method == "det_minimal"].set_index("ZoneID").theta
det_s2 = rep[rep.method == "det_step2"].set_index("ZoneID").theta
samp_med = rep[rep.method == "multinomial"].groupby("ZoneID").theta.median()
samp_mean = rep[rep.method == "multinomial"].groupby("ZoneID").theta.mean()
zones = det_rep.index
assert len(zones) == 98 and set(zones) == set(det_min.index) == set(samp_med.index)


def stats(s, name):
    s = s.loc[zones]
    print(f"{name:28s} mean {s.mean():.4f} median {s.median():.4f} "
          f"p5 {s.quantile(0.05):.4f} p95 {s.quantile(0.95):.4f} "
          f"| misalloc mean {1 - s.mean():.4f} median {1 - s.median():.4f} "
          f"p5 {1 - s.quantile(0.95):.4f} p95 {1 - s.quantile(0.05):.4f}")


print("Table 2 (aggregated overlap, unweighted zone statistics):")
stats(det_rep, "integerized (swap-repaired)")
stats(det_min, "minimal (deterministic)")
stats(det_s2, "seeded slice sampling")
stats(samp_mean, "sampled (per-zone mean)")
stats(samp_med, "sampled (per-zone median)")

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
data = [det_rep.loc[zones].to_numpy(), det_min.loc[zones].to_numpy(), samp_med.loc[zones].to_numpy()]
ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Integerized θ", "Minimal θ", "Sampled θ (median)"])
ax.set_ylabel("Overlap θ = Σ min(p, q)")
ax.set_title("Share of Population Allocated ‘Information-Theoretically Correct’")
fig.tight_layout()
out = FIG / "fig5_overlap_theta_violin.png"
fig.savefig(out)
print("wrote", out)
out_pdf = out.with_suffix(".pdf")
fig.savefig(out_pdf)
print("wrote", out_pdf)
