"""Supplement Section S3: does the disagreement between the KL minimizer and the most
probable (exact multinomial) floor/ceiling table vanish as N grows? Single total, K = 6
cells, 2000 random fractional tables per N. Writes output/supplement/kl_vs_ml_ranking.txt."""
from pathlib import Path

from itertools import product
from math import lgamma, log, floor
import random

def kl_N(counts, x):
    return sum(c * log(c / xk) for c, xk in zip(counts, x) if c)

def loglik(counts, p):
    n = sum(counts)
    return lgamma(n + 1) - sum(lgamma(c + 1) for c in counts) + sum(c * log(pk) for c, pk in zip(counts, p) if c)

def fc_tables(x):
    n = [floor(v) for v in x]
    R = round(sum(x) - sum(n))
    for z in product((0, 1), repeat=len(x)):
        if sum(z) == R:
            yield tuple(a + b for a, b in zip(n, z))

rng = random.Random(1)
OUT = Path(__file__).resolve().parents[1] / "output" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)
lines = []
for N in (10, 100, 1000, 10_000, 100_000):
    K = 6
    trials, disagree, ties = 2000, 0, 0
    for _ in range(trials):
        raw = [rng.random() + 0.05 for _ in range(K)]
        p = [v / sum(raw) for v in raw]
        x = [N * v for v in p]
        cands = list(fc_tables(x))
        if len(cands) < 2:
            ties += 1
            continue
        kl_best = min(cands, key=lambda c: kl_N(c, x))
        ml_best = max(cands, key=lambda c: loglik(c, p))
        if kl_best != ml_best:
            disagree += 1
    line = f"N={N:>7} K={K}: KL and exact-ML pick different floor/ceiling tables in {disagree}/{trials-ties} cases"
    print(line)
    lines.append(line)
(OUT / "kl_vs_ml_ranking.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
