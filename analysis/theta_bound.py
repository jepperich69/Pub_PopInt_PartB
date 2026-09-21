"""Upper bound on the overlap theta per zone (Sec. 3.2, Table 2 row "Upper bound").

For any integer table with the zone total N, sum_k min(xhat_k, x_k) <=
sum_k floor(x_k) + (sum of the R largest residuals), R = N - sum_k floor(x_k):
a cell at or below its floor contributes at most its floor, a cell above its
floor contributes x_k, and at most R cells can be above their floor. Dividing
by N gives the bound on theta. Reports the unweighted zone mean, median, 5th
and 95th percentiles, min, max and the population-weighted mean.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
frac = pd.read_csv(ROOT / "output" / "fractional_fit.csv")
rows = []
for zone, g in frac.groupby("ZoneID"):
    x = g["x"].to_numpy()
    N = int(round(x.sum()))
    n = np.floor(x + 1e-9)
    r = x - n
    R = N - int(n.sum())
    top = np.sort(r)[::-1][:R].sum() if R > 0 else 0.0
    rows.append(dict(ZoneID=zone, N=N, R=R, theta_max=(n.sum() + top) / N))
df = pd.DataFrame(rows)
out = ROOT / "output" / "theta_bound"
out.mkdir(exist_ok=True)
df.to_csv(out / "theta_bound_by_zone.csv", index=False)
w = df["N"] / df["N"].sum()
summary = dict(
    mean=df.theta_max.mean(), median=df.theta_max.median(),
    p05=df.theta_max.quantile(0.05), p95=df.theta_max.quantile(0.95),
    min=df.theta_max.min(), max=df.theta_max.max(),
    weighted_mean=float((df.theta_max * w).sum()), zones=len(df), R_total=int(df.R.sum()))
with open(out / "summary.txt", "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
