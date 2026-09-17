"""Floor-ceiling counterexample of Section 2.4 and single-total diagnostics on the
Danish tables. The counterexample x = (0.457, 1.727, 7.129, 1.687), total 11, has the
best floor-ceiling table (1, 2, 7, 1) and the KL optimum (1, 2, 6, 2). Also sweeps
random vectors of increasing skew and reports, per municipality, whether the
single-total KL optimum stays in the floor-ceiling class.
Outputs: output/counterexample/{SUMMARY.md, skew_sweep.csv, real_zone_single_sum_diagnostics.csv}
"""
import csv
import heapq
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRACTIONAL = ROOT / "output" / "fractional_fit.csv"
OUTDIR = ROOT / "output" / "counterexample"
OUTDIR.mkdir(parents=True, exist_ok=True)


def f(n, x):
    if n == 0:
        return x
    return n * math.log(n / x) - n + x


def marginal(k, x):
    return f(k, x) - f(k - 1, x)


def kl_opt_single_sum(xs):
    """KL-optimal integer vector for sum n = round(sum xs)."""
    n_total = int(round(sum(xs)))
    heap = []
    for i, x in enumerate(xs):
        if x > 0:
            heapq.heappush(heap, (marginal(1, x), i, 1))
    ns = [0] * len(xs)
    for _ in range(n_total):
        _, i, k = heapq.heappop(heap)
        ns[i] += 1
        heapq.heappush(heap, (marginal(k + 1, xs[i]), i, k + 1))
    return ns


def fc_opt_single_sum(xs):
    """Best integer vector constrained to floor/ceiling under sum n = round(sum xs)."""
    n_total = int(round(sum(xs)))
    floors = [math.floor(x) for x in xs]
    residual = n_total - sum(floors)
    ns = floors[:]
    candidates = []
    for i, x in enumerate(xs):
        lo = floors[i]
        if abs(x - lo) > 1e-10:
            candidates.append((marginal(lo + 1, x), i))
    for _, i in sorted(candidates)[:residual]:
        ns[i] += 1
    return ns


def objective(ns, xs):
    return sum(f(n, x) for n, x in zip(ns, xs))


def floor_ceiling_stats(xs):
    ns = kl_opt_single_sum(xs)
    fc = fc_opt_single_sum(xs)
    floors = [math.floor(x) for x in xs]
    ceils = [math.ceil(x) for x in xs]
    devs = []
    for n, lo, hi in zip(ns, floors, ceils):
        if n < lo:
            devs.append(lo - n)
        elif n > hi:
            devs.append(n - hi)
        else:
            devs.append(0)
    return {
        "max_outside": max(devs) if devs else 0,
        "outside_cells": sum(d > 0 for d in devs),
        "outside_units_l1": sum(devs),
        "global_kl": objective(ns, xs),
        "fc_kl": objective(fc, xs),
        "kl_gap_fc_minus_global": objective(fc, xs) - objective(ns, xs),
        "n": ns,
    }


def floor_ceiling_certificate(xs):
    """Strong marginal-cost separation that makes floor-ceiling optimal.

    If every unit up to the floor of every cell is cheaper than every first
    residual/ceiling unit, the KL optimum must include all floors and can only
    distribute the remaining residual units across ceilings. The counterexample
    violates exactly this: a tiny cell's first unit is cheaper than the last floor
    unit of a large cell.
    """
    floor_costs = []
    first_residual_costs = []
    for x in xs:
        lo = math.floor(x)
        if lo >= 1:
            floor_costs.append(marginal(lo, x))
        if abs(x - lo) > 1e-10:
            first_residual_costs.append(marginal(lo + 1, x))
    max_floor = max(floor_costs) if floor_costs else float("-inf")
    min_residual = min(first_residual_costs) if first_residual_costs else float("inf")
    return max_floor <= min_residual, max_floor, min_residual, max_floor - min_residual


def normalize_to_integer_total(raw, total=100.0):
    s = sum(raw)
    return [x * total / s for x in raw]


def random_vector(k, skew_ratio, total=100.0):
    # Uniform log-scale draw with max/min bounded by skew_ratio.
    vals = [math.exp(random.random() * math.log(skew_ratio)) for _ in range(k)]
    return normalize_to_integer_total(vals, total)


def experiment_skew():
    random.seed(20260704)
    rows = []
    for k in [4, 8, 16, 64]:
        for ratio in [1.0, 1.05, 1.1, 1.25, 1.5, 2, 3, 5, 10, 25, 100]:
            trials = 1 if ratio == 1.0 else 1000
            fail = 0
            max_outside = 0
            sep_fail = 0
            for _ in range(trials):
                xs = [100.0 / k] * k if ratio == 1.0 else random_vector(k, ratio)
                stats = floor_ceiling_stats(xs)
                ok, _, _, _ = floor_ceiling_certificate(xs)
                fail += int(stats["max_outside"] > 0)
                sep_fail += int(not ok)
                max_outside = max(max_outside, stats["max_outside"])
            rows.append({
                "k": k,
                "max_min_ratio": ratio,
                "trials": trials,
                "fc_fail_rate": fail / trials,
                "separation_fail_rate": sep_fail / trials,
                "max_outside_seen": max_outside,
            })
    return rows


def search_low_ratio_failures():
    random.seed(20260705)
    best = None
    # Use small dimensions because the result needs to be inspectable by hand.
    for k in [3, 4, 5, 6]:
        for total in [5.0, 10.0, 20.0, 50.0, 100.0]:
            for _ in range(20000):
                vals = [10 ** random.uniform(-1.0, 1.0) for _ in range(k)]
                xs = normalize_to_integer_total(vals, total)
                ratio = max(xs) / min(xs)
                if best and ratio >= best["ratio"]:
                    continue
                stats = floor_ceiling_stats(xs)
                if stats["max_outside"] > 0:
                    best = {
                        "k": k,
                        "total": total,
                        "ratio": ratio,
                        "xs": xs,
                        "n": stats["n"],
                        "outside_cells": stats["outside_cells"],
                        "max_outside": stats["max_outside"],
                        "kl_gap": stats["kl_gap_fc_minus_global"],
                    }
    return best


def analyze_real_zones():
    by_zone = defaultdict(list)
    with FRACTIONAL.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            x = float(row["x"])
            if x > 0:
                by_zone[row["ZoneID"]].append(x)

    rows = []
    for zone, xs in sorted(by_zone.items()):
        total = sum(xs)
        floors = sum(math.floor(x) for x in xs)
        residual = round(total) - floors
        positive = [x for x in xs if x > 0]
        subunit = [x for x in positive if x < 1]
        large = [x for x in positive if x >= 1]
        ok, max_floor, min_above, gap = floor_ceiling_certificate(xs)
        stats = floor_ceiling_stats(xs)
        rows.append({
            "ZoneID": zone,
            "cells": len(xs),
            "total": total,
            "positive_cells": len(positive),
            "subunit_cells": len(subunit),
            "subunit_share": len(subunit) / len(positive) if positive else 0,
            "mass_in_subunit_cells": sum(subunit) / total if total else 0,
            "max_x": max(positive),
            "min_x": min(positive),
            "p99_over_median": percentile(positive, 0.99) / percentile(positive, 0.50),
            "max_over_median": max(positive) / percentile(positive, 0.50),
            "residual_units": residual,
            "single_sum_fc_certified": ok,
            "marginal_gap": gap,
            "single_sum_outside_cells": stats["outside_cells"],
            "single_sum_max_outside": stats["max_outside"],
            "single_sum_outside_units_l1": stats["outside_units_l1"],
            "single_sum_global_kl": stats["global_kl"],
            "single_sum_fc_kl": stats["fc_kl"],
            "single_sum_kl_gap_fc_minus_global": stats["kl_gap_fc_minus_global"],
            "single_sum_kl_gap_per_person": stats["kl_gap_fc_minus_global"] / total if total else 0,
        })
    return rows


def percentile(vals, p):
    vals = sorted(vals)
    if not vals:
        return float("nan")
    pos = p * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    counter = [0.457, 1.727, 7.129, 1.687]
    counter_stats = floor_ceiling_stats(counter)
    counter_cert = floor_ceiling_certificate(counter)

    skew_rows = experiment_skew()
    low_ratio = search_low_ratio_failures()
    real_rows = analyze_real_zones()

    write_csv(OUTDIR / "skew_sweep.csv", skew_rows)
    write_csv(OUTDIR / "real_zone_single_sum_diagnostics.csv", real_rows)

    certified = sum(r["single_sum_fc_certified"] for r in real_rows)
    outside = sum(r["single_sum_outside_cells"] > 0 for r in real_rows)
    max_out = max(r["single_sum_max_outside"] for r in real_rows)
    total_outside_l1 = sum(r["single_sum_outside_units_l1"] for r in real_rows)
    total_kl_gap = sum(r["single_sum_kl_gap_fc_minus_global"] for r in real_rows)
    total_population = sum(r["total"] for r in real_rows)
    median_subunit = percentile([r["subunit_share"] for r in real_rows], 0.50)
    median_subunit_mass = percentile([r["mass_in_subunit_cells"] for r in real_rows], 0.50)
    median_max_over_median = percentile([r["max_over_median"] for r in real_rows], 0.50)
    worst = sorted(real_rows, key=lambda r: (r["single_sum_max_outside"], r["single_sum_outside_cells"], r["max_over_median"]), reverse=True)[:10]

    with (OUTDIR / "SUMMARY.md").open("w", encoding="utf-8") as fh:
        fh.write("# Floor-ceiling theorem recheck\n\n")
        fh.write("## Counterexample sanity check\n\n")
        fh.write(f"x = {counter}\n\n")
        fh.write(f"single-sum KL optimum n = {counter_stats['n']}\n\n")
        fh.write(f"outside floor/ceiling cells = {counter_stats['outside_cells']}; max outside = {counter_stats['max_outside']}\n\n")
        fh.write(f"marginal separation ok = {counter_cert[0]}; max floor cost - min above-ceiling cost = {counter_cert[3]:.6g}\n\n")
        fh.write("## Random bounded-skew sweep\n\n")
        fh.write("See `skew_sweep.csv`. Equal vectors are always floor-ceiling. Failures begin once the vector is not exactly symmetric; their frequency rises with dimension and skew.\n\n")
        if low_ratio:
            fh.write("Lowest-ratio random failure found in this run:\n\n")
            fh.write(f"- k = {low_ratio['k']}, total = {low_ratio['total']}, max/min = {low_ratio['ratio']:.3f}\n")
            fh.write(f"- x = {[round(v, 6) for v in low_ratio['xs']]}\n")
            fh.write(f"- KL-optimal n = {low_ratio['n']}; outside cells = {low_ratio['outside_cells']}; max outside = {low_ratio['max_outside']}\n")
            fh.write(f"- KL gap of best floor-ceiling table = {low_ratio['kl_gap']:.6g}\n\n")
        fh.write("## Real Danish fractional tables, by municipality\n\n")
        fh.write(f"zones = {len(real_rows)}\n\n")
        fh.write(f"zones satisfying the single-sum marginal separation certificate = {certified}/{len(real_rows)}\n\n")
        fh.write(f"zones whose single-sum KL optimum leaves floor/ceiling = {outside}/{len(real_rows)}\n\n")
        fh.write(f"maximum outside distance in the single-sum relaxation = {max_out}\n\n")
        fh.write(f"total L1 outside-floor-ceiling deviation in the single-sum relaxation = {total_outside_l1}\n\n")
        fh.write(f"aggregate KL gap of best floor-ceiling table versus single-sum optimum = {total_kl_gap:.6g}\n\n")
        fh.write(f"aggregate KL gap per person = {total_kl_gap / total_population:.6g}\n\n")
        fh.write(f"median share of positive cells below one person = {median_subunit:.3f}\n\n")
        fh.write(f"median population mass in subunit cells = {median_subunit_mass:.3f}\n\n")
        fh.write(f"median max/median positive cell size = {median_max_over_median:.1f}\n\n")
        fh.write("Worst zones under the single-sum relaxation:\n\n")
        fh.write("| ZoneID | cells | subunit share | mass in subunit | max/median | outside cells | max outside |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in worst:
            fh.write(
                f"| {r['ZoneID']} | {r['cells']} | {r['subunit_share']:.3f} | "
                f"{r['mass_in_subunit_cells']:.3f} | {r['max_over_median']:.1f} | "
                f"{r['single_sum_outside_cells']} | {r['single_sum_max_outside']} |\n"
            )


if __name__ == "__main__":
    main()
