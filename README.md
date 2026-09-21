# From Fractions to Individuals: Information-Theoretic Integerization for Population Synthesis

Reproduction code and data for

> Rich, J. (2026). From Fractions to Individuals: Information-Theoretic Integerization
> for Population Synthesis. *Transportation Research Part B*, manuscript TRB-D-25-00887
> (revision R1).

The repository is self-contained. Every table and figure in the paper is produced by
`run_all.py` from the five target-margin files in `data/`. Nothing else is needed.

## Quick start

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # numpy, pandas, matplotlib
python run_all.py                                   # full reproduction, about an hour (Table 4 is 35 min of it)
python run_all.py --quick                           # smoke test, a few minutes
python run_all.py --only table3 fig5                # individual stages
python run_all.py --list
```

Python 3.10 or later. Verified with Python 3.13, numpy 2.5, pandas 3.0, matplotlib 3.11
on Windows 11. No solver licence is required: the exact KL optimum in Section 3.2 is
computed by marginal analysis (see `analysis/kl_gap.py`), which reproduces the Gurobi
solution used during the revision to 1e-11.

All generated files go to `output/` (gitignored). Each stage writes a log,
`output/<stage>.log`.

## Data

`data/` holds the five target margins of the Danish application, one file per
Zone x Age x {Children, Family, Gender, Income, LMA} margin for the 98 municipalities.
They are aggregated control totals and carry no information on individuals.

The seed table of the paper is *constructed* from these margins: `pipeline/step1_generate_seed.py`
runs IPF from a uniform start on the six-way table (Age x Children x Family x Gender x
Income x LMA) within each municipality. The seed is therefore a maximum-entropy table
consistent with the margins, not register data. Steps 2 and 3 then run the paper's IPF fit
on that seed and integerize it. The fractional table has 634,956 cells and 5,932,654 persons.

## What produces what

| Paper object | Stage | Script | Output |
|---|---|---|---|
| Seed table | `step1` | `pipeline/step1_generate_seed.py` | `output/tmp_Pop_SyntheticSeed.csv` |
| Fractional IPF table `x` | `step2`, `step3` | `pipeline/step3_integerize.py` | `output/fractional_fit.csv` |
| Minimal allocation (floor + largest remainder) | `step3` | `pipeline/step3_integerize.py` | `output/integer_table.csv` |
| Swap-repaired table (the paper's method) | `step3` | `pipeline/step3_integerize.py` | `output/integer_repaired.csv` |
| Seeded slice sampling (Table 3 row) | `step2` | `pipeline/step2_seeded_pps.py` | `output/tmp_Minimal_Integerized.csv` |
| Figures 2 and 3 (toy examples A and C; B was dropped in R1) | `toy` | `analysis/toy_examples.py` | `output/figures/toy_entropy_vs_hamilton_A.png`, `toy_small_cell_risk.png` |
| Section 2.4 counterexample | `counterexample` | `analysis/floor_ceiling_counterexample.py` | `output/counterexample/SUMMARY.md` |
| Table 1 (error vs. fractional baseline, 20 largest zones) | `table1` | `analysis/table1_error_vs_fractional.py` | `output/table1/table1.csv` |
| Table 2 (aggregated overlap) and Figure 5 (theta violins) | `fig5` | `analysis/fig5_theta_violin.py` | `output/fig5.log`, `output/figures/fig5_overlap_theta_violin.png` |
| Table 3 (benchmark of integerization methods) | `table3` | `analysis/table3_benchmark.py` | `output/table3/summary.csv` |
| Section 3.2, floor-ceiling gap paragraph | `kl_gap` | `analysis/kl_gap.py` | `output/kl_gap/summary.txt`, `per_zone.csv` |
| Table 4 (dimensionality experiment) | `table4` | `analysis/table4_dimensionality.py` | `output/table4/table4_unweighted.csv` |

Figure 4 (controlled and descriptive attributes) is a TikZ drawing in the manuscript.

Stage order matters: `step1 -> step2 -> step3 -> table3 -> table1 -> fig5`; `kl_gap`,
`counterexample`, `toy` and `table4` need `step3` only. `run_all.py` keeps this order.

## Expected results

Deterministic stages reproduce the paper exactly. Rows that involve sampling (multinomial,
TRS) are seeded and reproduce to the reported precision; the sampling rows of Table 1 can
move in the third digit between numpy/pandas versions because the cell order of the merge
changes the draw sequence.

| Quantity | Paper | `run_all.py` |
|---|---|---|
| theta, swap-repaired / minimal / multinomial (unweighted zone means) | 0.984 / 0.985 / 0.921 | `output/table3/summary.csv`, column `theta_mean` |
| Secondary-margin L1, repaired / minimal (% of population) | 1.39 / 1.62 | `output/table3/summary.csv`, column `sec_pct` |
| Sub-unit mass retained, repaired / multinomial | 0.92 / 1.00 | `output/table3/summary.csv`, column `sub_ratio` |
| Table 1, repaired L2 / KL | 20.4 / 0.0036 | `output/table1/table1.csv` |
| Best floor-ceiling table above the unrestricted optimum | 3.2e-5 (worst zone 1.3e-4) | `output/kl_gap/summary.txt` |
| Largest-remainder / swap-repaired table above the optimum | 3.6% / 6.6% | `output/kl_gap/summary.txt` |
| Optimum leaves the floor-ceiling class | 95 of 98 zones, 658 units, 434 cells below floor, 207 above ceiling | `output/kl_gap/summary.txt` |
| Table 4, floor share at k = 0 ... 4 | 0.95, 0.87, 0.73, 0.53, 0.31 | `output/table4/table4_unweighted.csv` |
| Table 4, theta_B direct at k = 4 / aggregate-first | 0.821 / 0.984 | `output/table4/table4_unweighted.csv` |

In `table4_unweighted.csv` the method `hier` is the paper's *aggregate-first* procedure and
`multi` is multinomial sampling. The script also runs `hybrid` (deterministic block totals,
fixed-seed sampling within blocks), the variant that Section 2.6 mentions; it is not in Table 4.

## Layout

```
data/        five target-margin files (inputs)
pipeline/    step1_generate_seed.py, step2_seeded_pps.py, step3_integerize.py
analysis/    one script per table, figure or reported number
run_all.py   runner
output/      everything generated (gitignored)
```

The pipeline scripts are the ones used for the paper, with their paths made relative to
the repository. `step3_integerize.py` is the paper's method: IPF fit, floor, largest-remainder
allocation within each Age x Gender slice of the target margin, and anchor-preserving swap
repair of the secondary margins. Its post-stage also writes the pipeline's own summary
tables and violin plots, which the analysis scripts recompute in the conventions of the
paper (unweighted zone means, KL direction D(q||p) of Eq. (1)).
