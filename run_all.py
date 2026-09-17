"""Reproduction runner for

    Rich, J. (2026). From Fractions to Individuals: Information-Theoretic
    Integerization for Population Synthesis. Transportation Research Part B
    (manuscript TRB-D-25-00887, revision R1).

Runs the pipeline and the analysis scripts in dependency order and writes every
table and figure of the paper to output/. See README.md for the mapping from
paper objects to scripts and output files.

Usage
    python run_all.py               # everything (about 30-40 minutes on a laptop)
    python run_all.py --quick       # smoke test: Table 4 on three zones, kl_gap on ten
    python run_all.py --only table3 fig5
    python run_all.py --list
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"

# (name, script, paper objects, extra env for --quick)
STAGES = [
    ("step1", "pipeline/step1_generate_seed.py",
     "seed table from the five target margins (uniform-start IPF)", {}),
    ("step2", "pipeline/step2_seeded_pps.py",
     "IPF fit + seeded slice sampling (Table 3 row 'seeded slice sampling')", {}),
    ("step3", "pipeline/step3_integerize.py",
     "IPF fit, floor, largest remainder, swap repair (the paper's method)", {}),
    ("table3", "analysis/table3_benchmark.py",
     "Table 3 benchmark of integerization methods", {}),
    ("table1", "analysis/table1_error_vs_fractional.py",
     "Table 1 error relative to the fractional baseline (20 largest zones)", {}),
    ("fig5", "analysis/fig5_theta_violin.py",
     "Figure 5 theta violins and the Table 2 statistics", {}),
    ("kl_gap", "analysis/kl_gap.py",
     "Sections 2.4 and 3.2: exact KL optimum and floor-ceiling gaps", {"QUICK_ARGS": "10"}),
    ("counterexample", "analysis/floor_ceiling_counterexample.py",
     "Section 2.4 counterexample and single-total diagnostics", {}),
    ("toy", "analysis/toy_examples.py",
     "Figures 1-3 toy examples", {}),
    ("table4", "analysis/table4_dimensionality.py",
     "Table 4 dimensionality experiment (k = 0..4 descriptive attributes)",
     {"STRESS_ZONES": "101000,751000,461000", "STRESS_R": "2"}),
]


def run(name, script, desc, quick_env, quick):
    env = dict(os.environ)
    args = [sys.executable, str(ROOT / script)]
    if quick:
        for k, v in quick_env.items():
            if k == "QUICK_ARGS":
                args.append(v)
            else:
                env[k] = v
    log = OUT / f"{name}.log"
    print(f"[{name}] {desc}\n        -> {script}  (log: output/{name}.log)", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(args, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc != 0:
        print(f"[{name}] FAILED (exit {rc}) after {dt:.0f}s; see output/{name}.log", flush=True)
        sys.exit(rc)
    print(f"[{name}] done in {dt:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="reduced runs for a smoke test")
    ap.add_argument("--only", nargs="+", metavar="STAGE", help="run only these stages (in the standard order)")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    a = ap.parse_args()
    if a.list:
        for name, script, desc, _ in STAGES:
            print(f"{name:15s} {script:45s} {desc}")
        return
    OUT.mkdir(exist_ok=True)
    names = [s[0] for s in STAGES]
    if a.only:
        bad = [n for n in a.only if n not in names]
        if bad:
            sys.exit(f"unknown stage(s): {bad}; use --list")
    t0 = time.time()
    for name, script, desc, quick_env in STAGES:
        if a.only and name not in a.only:
            continue
        run(name, script, desc, quick_env, a.quick)
    print(f"\nall done in {(time.time() - t0) / 60:.1f} min; outputs in {OUT}")


if __name__ == "__main__":
    main()
