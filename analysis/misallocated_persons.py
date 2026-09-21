"""Misallocated persons (Introduction and Sec. 3.2 takeaway (ii)).

Sum over zones of N_z (1 - theta_z) for the swap-repaired table and for the
multinomial mean, from output/overlap_summary_by_zone.csv (fig5 stage). This
is the population-weighted count; the paper's 98.4% / 92.1% are unweighted
zone means, so the two must not be combined.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
b = pd.read_csv(ROOT / "output" / "overlap_summary_by_zone.csv")
N = b["T_int"]; tot = N.sum()
mis_int = (N * (1 - b["theta_int"])).sum()
mis_samp = (N * (1 - b["theta_samp_mean"])).sum()
out = ROOT / "output" / "misallocated_persons.txt"
lines = [
    f"population: {tot:.0f}",
    f"misallocated persons, swap-repaired: {mis_int:.0f} ({100*mis_int/tot:.2f}%)",
    f"misallocated persons, multinomial mean: {mis_samp:.0f} ({100*mis_samp/tot:.2f}%)",
    f"difference: {mis_samp - mis_int:.0f}",
    f"weighted theta: swap-repaired {1-mis_int/tot:.4f}, multinomial {1-mis_samp/tot:.4f}",
]
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
