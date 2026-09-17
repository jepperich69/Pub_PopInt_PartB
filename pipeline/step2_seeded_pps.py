
# -*- coding: utf-8 -*-
"""
SPAR-TCS (v8-minimal) — Strip-down for step-by-step integerization, aligned to the cookbook.

Keeps:
  ✓ Loader + structural zeros + ghost support
  ✓ Stage-0 harmonization & support projection
  ✓ Hard-coded fast IPF (NumPy-only)
Adds (minimal, stepwise integerization):
  1) Deterministic split: floor + residuals (Cookbook §1)
  2) Anchor-conditioned PPS sampling (Cookbook §2b, "rand_anchor")
  3) (Optional) swap-repair placeholder (Cookbook §2a)
  4) Verification & diagnostics

Outputs (OUT_DIR):
  fractional_fit.csv           — IPF fractional solution per row
  step1_floor.csv              — floor n, residual target K_add per zone
  step2_anchor_selected.csv    — indices chosen by PPS within anchor slices
  integer_table.csv            — y = n + z (no heavy ILP; exactness is not guaranteed yet)
  residuals_after_step2.csv    — per-margin absolute residuals after the anchor step
  summary.csv                  — run summary & file paths
"""

# ===================== CONFIG =====================
from pathlib import Path
# Repo-relative paths: margins in data/, seed and generated files in output/
ROOT     = Path(__file__).resolve().parents[1]
BASE_DIR = str(ROOT / "data")
OUT_DIR  = str(ROOT / "output")

IPF_TOL = 1e-7
IPF_MAX_ITERS = 200
VERBOSE_IPF = True

# Integerization anchor (first-priority target)
ANCHOR_NAME = "Age×Gender"
ANCHOR_SLICE_DIMS = ("ZoneID","AgeID","GenderID")

# Stage-0 tolerances (logging only)
HARM_ABS_TOL = 5.0
HARM_REL_TOL = 0.005
# =================================================

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Iterable
import numpy as np
import pandas as pd

EPS = 1e-18
EPS_GHOST = 1e-9  # tiny seed mass

# ---------- small IO helpers ----------
def _read_csv(fp: str) -> pd.DataFrame:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing file: {fp}")
    return pd.read_csv(fp)

def _ensure_int_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    return df

def _ensure_float_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    return df

# ---------- problem spec ----------
@dataclass
class ConstraintSpec:
    name: str
    dims: Tuple[str, ...]
    df: pd.DataFrame
    val_col: str = "Val"

@dataclass
class IPFResult:
    w: pd.DataFrame
    converged: bool
    n_iter: int
    max_rel_err: float

# ---------- loader ----------
def load_all(base_dir: str) -> Tuple[pd.DataFrame, List[ConstraintSpec]]:
    start_fp = os.path.join(OUT_DIR, "tmp_Pop_SyntheticSeed.csv")  # written by step1_generate_seed.py
    df = _read_csv(start_fp)

    # Normalize column names
    df = df.rename(columns={"Val": "x", "ZoneID0": "ZoneID"})
    needed = ["ZoneID", "AgeID", "GenderID", "LmaID", "NumChildID", "FamID", "IncomeID", "x"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in seed table")

    # Types
    df = _ensure_int_columns(df, ["ZoneID","AgeID","GenderID","LmaID","NumChildID","FamID","IncomeID"])
    df = _ensure_float_column(df, "x")
    df["x"] = df["x"].clip(lower=1e-12)

    # Constraints
    constraints: List[ConstraintSpec] = []

    def load_constraint(filename: str, dims: List[str], name: str) -> None:
        fp = os.path.join(base_dir, filename)
        cdf = _read_csv(fp).rename(columns={"ZoneID0": "ZoneID"})
        cdf = _ensure_int_columns(cdf, ["ZoneID","AgeID","GenderID","LmaID","NumChildID","FamID","IncomeID"])
        cdf = _ensure_float_column(cdf, "Val")
        keep = dims + ["Val"]
        missing = set(dims) - set(cdf.columns)
        if missing:
            raise ValueError(f"{name}: expected dims {dims} not present, missing {missing}")
        cdf = cdf[keep].copy()
        constraints.append(ConstraintSpec(name=name, dims=tuple(dims), df=cdf, val_col="Val"))

    load_constraint("in_Pop_TargetZoneAgeChildren.csv", ["ZoneID","AgeID","NumChildID"], "Age×Children")
    load_constraint("in_Pop_TargetZoneAgeFamily.csv",   ["ZoneID","AgeID","FamID"],      "Age×Family")
    load_constraint("in_Pop_TargetZoneAgeGender.csv",   ["ZoneID","AgeID","GenderID"],   "Age×Gender")
    load_constraint("in_Pop_TargetZoneAgeIncome.csv",   ["ZoneID","AgeID","IncomeID"],   "Age×Income")
    load_constraint("in_Pop_TargetZoneAgeLma.csv",      ["ZoneID","AgeID","LmaID"],      "Age×Lma")

    return df, constraints

# ---------- structural zeros (drop once) ----------
def build_structural_zero_mask(seed: pd.DataFrame, constraints: List[ConstraintSpec]) -> pd.Series:
    keep = pd.Series(True, index=seed.index)
    for cs in constraints:
        dims = list(cs.dims)
        zero_keys = cs.df.loc[cs.df[cs.val_col] == 0, dims].drop_duplicates()
        if zero_keys.empty:
            continue
        idx = seed.merge(zero_keys.assign(__drop__=True), on=dims, how="left")["__drop__"].fillna(False)
        keep &= ~idx
    return keep

# ---------- ghosts ----------
def _augment_seed_with_ghosts(seed: pd.DataFrame,
                              constraints: List[ConstraintSpec],
                              eps_ghost: float = EPS_GHOST) -> Tuple[pd.DataFrame, List[str]]:
    notes=[]
    seed_cols = [c for c in seed.columns if c != "x"]
    all_dims = [c for c in seed_cols if c != "ZoneID"]

    exist_idx = pd.MultiIndex.from_frame(seed[["ZoneID", *all_dims]])
    new_rows = []

    # collect positive levels per (ZoneID,AgeID) for any margin that has them
    ZA_pos: Dict[Tuple[int,int], Dict[str, List[int]]] = {}
    for cs in constraints:
        if not {"ZoneID","AgeID"}.issubset(cs.dims):
            continue
        pos = cs.df.loc[cs.df[cs.val_col] > 0, ["ZoneID","AgeID", *[d for d in cs.dims if d not in ("ZoneID","AgeID")]]]
        if pos.empty:
            continue
        other = [d for d in cs.dims if d not in ("ZoneID","AgeID")][0]
        for (z,a), g in pos.groupby(["ZoneID","AgeID"], observed=True, sort=False):
            z = int(z); a = int(a)
            ZA_pos.setdefault((z,a), {}).setdefault(other, [])
            levs = sorted({int(v) for v in g[other].dropna().unique().tolist()})
            ZA_pos[(z,a)][other] = sorted(set(ZA_pos[(z,a)][other] + levs))

    # simple bridge ghosts: for each pos slice, ensure 1 joint row exists
    for (z,a), posmap in ZA_pos.items():
        row = {"ZoneID": z, "AgeID": a}
        for d in all_dims:
            if d in posmap and len(posmap[d]) > 0:
                row[d] = posmap[d][0]
            else:
                # use the modal value in seed for that dim (fallback: min)
                m = seed.loc[seed["ZoneID"]==z, d].mode(dropna=True)
                row[d] = int(m.iloc[0]) if not m.empty else int(seed[d].min())
        key = (row["ZoneID"],) + tuple(row[d] for d in all_dims)
        if key not in exist_idx:
            new_rows.append({**row, "x": float(eps_ghost)})

    if new_rows:
        seed = pd.concat([seed, pd.DataFrame(new_rows, columns=["ZoneID", *all_dims, "x"])], ignore_index=True)
        notes.append(f"[Ghosts] Added {len(new_rows)} bridge rows (x={eps_ghost:g}).")

    for c in ["ZoneID", *all_dims]:
        seed[c] = pd.to_numeric(seed[c], errors="raise").astype(int)
    seed["x"] = seed["x"].astype(float)
    return seed, notes

# ---------- Stage 0: Harmonize + Support Projection (per zone) ----------
def _safe_divide(a, b, default=1.0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = np.full_like(a, default, dtype=float)
    m = np.abs(b) > 0
    out[m] = a[m] / b[m]
    return out

def _seed_supported_index(seed_z: pd.DataFrame, dims: List[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(seed_z[dims].drop_duplicates(), names=dims)

def _harmonize_slice_totals(cons_z: List[ConstraintSpec],
                            controlling_name: str,
                            slice_dims: Tuple[str, ...],
                            abs_tol: float,
                            rel_tol: float) -> Tuple[List[ConstraintSpec], List[str]]:
    notes=[]
    ctrl = next(cz for cz in cons_z if cz.name == controlling_name)
    ctrl_df = ctrl.df.copy()
    val_col = ctrl.val_col
    aligned=[]
    for cz in cons_z:
        if cz.name == controlling_name:
            aligned.append(cz); continue
        df = cz.df.copy()
        by_dims = tuple(d for d in slice_dims if d in df.columns)
        if len(by_dims) == 0:
            aligned.append(cz); continue
        T = ctrl_df.groupby(list(by_dims), observed=True, sort=False)[val_col].sum()
        S = df.groupby(list(by_dims), observed=True, sort=False)[val_col].sum().reindex(T.index, fill_value=0.0)
        scale_map = (T / S.replace({0.0: np.nan})).to_dict()
        def _lookup_scale(row):
            k = tuple(row[d] for d in by_dims)
            return scale_map.get(k, 1.0)
        df["__sc__"] = df.apply(_lookup_scale, axis=1)
        df[val_col] = df[val_col] * df["__sc__"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df.drop(columns=["__sc__"], inplace=True)
        aligned.append(ConstraintSpec(cz.name, cz.dims, df, cz.val_col))
    return aligned, notes

def _project_onto_support(cons_z: List[ConstraintSpec],
                          seed_z: pd.DataFrame,
                          controlling_name: str,
                          slice_dims: Tuple[str, ...]) -> Tuple[List[ConstraintSpec], List[str]]:
    notes=[]
    ctrl = next(cz for cz in cons_z if cz.name == controlling_name)
    ctrl_df = ctrl.df.copy()
    val_col = ctrl.val_col
    projected=[]
    for cz in cons_z:
        df = cz.df.copy()
        dims = list(cz.dims)
        by_dims = tuple(d for d in slice_dims if d in dims)
        if len(by_dims) > 0:
            T = ctrl_df.groupby(list(by_dims), observed=True, sort=False)[val_col].sum()
        else:
            T = pd.Series({(): float(ctrl_df[val_col].sum())})
        supp_idx = _seed_supported_index(seed_z, dims)
        idx = df.set_index(dims).index
        is_sup = idx.isin(supp_idx)
        df_sup = df.loc[is_sup].copy()
        df_nsup = df.loc[~is_sup].copy()
        if len(by_dims) > 0 and len(df_sup):
            Ssup = df_sup.groupby(list(by_dims), observed=True, sort=False)[val_col].sum()
            df_sup = df_sup.merge(Ssup.rename("__Ssup__"), on=list(by_dims), how="left")
            df_sup = df_sup.merge(T.rename("__T__"), on=list(by_dims), how="left")
            fac = _safe_divide(df_sup["__T__"].values, df_sup["__Ssup__"].replace({0.0: np.nan}).values, default=0.0)
            df_sup[val_col] = df_sup[val_col] * np.asarray(fac)
            df_sup.drop(columns=["__Ssup__","__T__"], inplace=True)
        elif len(df_sup):
            Ssup_total = df_sup[val_col].sum()
            T_total = float(ctrl_df[val_col].sum())
            fac = (T_total / Ssup_total) if Ssup_total > 0 else 1.0
            df_sup[val_col] = df_sup[val_col] * fac
        df_nsup[val_col] = 0.0
        df_new = pd.concat([df_sup, df_nsup], ignore_index=True)
        projected.append(ConstraintSpec(cz.name, cz.dims, df_new, cz.val_col))
    return projected, notes

def _reconcile_zone_totals_global(cons_z: List[ConstraintSpec],
                                  controlling_name: str) -> Tuple[List[ConstraintSpec], List[str]]:
    notes=[]
    ctrl = next(cz for cz in cons_z if cz.name == controlling_name)
    T = float(ctrl.df[ctrl.val_col].sum())
    adj=[]
    for cz in cons_z:
        if cz.name == controlling_name:
            adj.append(cz); continue
        S = float(cz.df[cz.val_col].sum())
        if S <= 0.0 or T <= 0.0:
            adj.append(cz); continue
        if abs(S - T) > 1e-9:
            f = T / S
            df2 = cz.df.copy()
            df2[cz.val_col] = df2[cz.val_col] * f
            adj.append(ConstraintSpec(cz.name, cz.dims, df2, cz.val_col))
        else:
            adj.append(cz)
    return adj, notes

def stage0_harmonize_zone(seed_z: pd.DataFrame,
                          cons_z: List[ConstraintSpec],
                          controlling_name: str = ANCHOR_NAME,
                          slice_dims: Tuple[str, ...] = ANCHOR_SLICE_DIMS) -> Tuple[pd.DataFrame, List[ConstraintSpec], List[str]]:
    notes=[]
    cons_h, n1 = _harmonize_slice_totals(cons_z, controlling_name, slice_dims, HARM_ABS_TOL, HARM_REL_TOL)
    cons_p, n2 = _project_onto_support(cons_h, seed_z, controlling_name, slice_dims)
    notes.extend(n1); notes.extend(n2)
    return seed_z, cons_p, notes

# ---------- Hard-coded fast IPF (NumPy-only) ----------
class HardIPF:
    def __init__(self, seed_df: pd.DataFrame, constraints: List[ConstraintSpec], weight_col="x"):
        keep_cols = [c for c in seed_df.columns if c not in {"w","n","frac"}]
        self.df = seed_df[keep_cols].copy()
        self.weight = self.df[weight_col].to_numpy(dtype=float)
        self.cols = [c for c in self.df.columns if c != weight_col]
        self.weight_col = weight_col

        # category-code each column
        self.codes: Dict[str, np.ndarray] = {}
        self.levels: Dict[str, np.ndarray] = {}
        for c in self.cols:
            cats, inv = np.unique(self.df[c].to_numpy(), return_inverse=True)
            self.codes[c] = inv.astype(np.int64)
            self.levels[c] = cats

        # build constraint indexers
        self.margins = []
        self.names = []
        for cs in constraints:
            dims = list(cs.dims)
            self.names.append(cs.name)
            shape = tuple(len(self.levels[d]) for d in dims)

            tdf = cs.df[dims + [cs.val_col]].copy()
            ok = np.ones(len(tdf), dtype=bool)
            maps = {}
            for d in dims:
                lvl = self.levels[d]
                mp = {v:i for i,v in enumerate(lvl)}
                maps[d] = mp
                ok &= tdf[d].map(mp).notna().to_numpy()
            if not ok.all():
                tdf = tdf.loc[ok].copy()

            idx = np.ravel_multi_index(
                tuple(tdf[d].map(maps[d]).to_numpy(int) for d in dims),
                dims=shape
            )
            tgt = np.zeros(np.prod(shape), dtype=float)
            np.add.at(tgt, idx, tdf[cs.val_col].to_numpy(dtype=float))
            tgt = tgt.reshape(shape)

            row_flat = np.ravel_multi_index(tuple(self.codes[d] for d in dims), dims=shape)
            self.margins.append((dims, shape, row_flat, tgt))

    @staticmethod
    def _rel_err_safe(tgt: np.ndarray, cur: np.ndarray) -> float:
        denom = np.maximum(np.maximum(np.abs(tgt), np.abs(cur)), EPS)
        return float(np.max(np.abs(tgt - cur) / denom)) if denom.size else 0.0

    def fit(self, tol=1e-7, max_iters=200, verbose=True) -> Tuple[np.ndarray, bool, int, float]:
        w = self.weight.copy()
        for it in range(1, max_iters+1):
            max_rel = 0.0
            for (dims, shape, row_flat, tgt) in self.margins:
                cur = np.bincount(row_flat, weights=w, minlength=np.prod(shape)).reshape(shape)
                scale = np.ones_like(cur, dtype=float)
                mask = cur > 0
                scale[mask] = tgt[mask] / cur[mask]
                w *= scale.reshape(-1)[row_flat]
                rel = self._rel_err_safe(tgt, cur)
                max_rel = max(max_rel, rel)

            if verbose and (it <= 10 or it % 5 == 0 or max_rel < tol):
                parts=[]
                for name, (dims, shape, row_flat, tgt) in zip(self.names, self.margins):
                    cur = np.bincount(row_flat, weights=w, minlength=np.prod(shape)).reshape(shape)
                    parts.append(f"{name}:{self._rel_err_safe(tgt, cur):.2e}")
                print(f"[IPF] iter={it:3d} max_rel_err={max_rel:.3e} | " + ", ".join(parts))

            if max_rel < tol:
                return w, True, it, float(max_rel)
        return w, False, max_iters, float(max_rel)

# ---------- Cookbook Step 1: Deterministic split ----------
def step1_split(frac_df_zone: pd.DataFrame, zone_target: int) -> Tuple[pd.DataFrame, int]:
    """Return a frame with floor 'n' and 'frac', and K_add units to add."""
    df = frac_df_zone.copy()
    df["n"] = np.floor(df["x"] + 1e-9).astype(int)
    df["frac"] = df["x"] - df["n"]
    floor_sum = int(df["n"].sum())
    K_add = int(zone_target - floor_sum)
    if K_add < 0:
        # defloor smallest fracs to match target
        take = -K_add
        idx = df["frac"].nsmallest(take).index
        df.loc[idx, "n"] = (df.loc[idx, "n"] - 1).astype(int)
        K_add = 0
    return df, K_add

# ---------- Helpers: slice indexers and PPS sampling ----------
def _group_keys(df: pd.DataFrame, dims: Iterable[str]) -> pd.Series:
    return df.apply(lambda r: tuple(r[d] for d in dims), axis=1)

def _pps_without_replacement(idx_array: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Simple PPS-WOR via weighted sampling without replacement (NumPy choice)."""
    w = weights.clip(min=1e-12)
    p = w / w.sum()
    k = int(min(k, len(idx_array)))
    if k <= 0:
        return np.array([], dtype=idx_array.dtype)
    chosen = rng.choice(idx_array, size=k, replace=False, p=p)
    return chosen

# ---------- Cookbook Step 2: Anchor-conditioned PPS sampling ----------
def step2_anchor_pps(df_split: pd.DataFrame,
                     cons_z: List[ConstraintSpec],
                     anchor_dims: Tuple[str, ...],
                     K_add: int,
                     seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Within each anchor slice, choose exactly needed residual units via PPS (proportional to 'frac').
    Returns (df_with_z_and_n, chosen_rows_df)
    """
    rng = np.random.default_rng(seed)
    df = df_split.copy()
    df["z"] = 0  # indicator for +1

    # compute residual need per anchor slice: target - floor_sum
    anchor = next(cz for cz in cons_z if cz.name == ANCHOR_NAME)
    dims = list(anchor_dims)
    tgt = anchor.df.groupby(dims, observed=True, sort=False)[anchor.val_col].sum().rename("tgt")
    flr = df.groupby(dims, observed=True, sort=False)["n"].sum().rename("floor")
    need = (tgt - flr).reindex(tgt.index, fill_value=0).clip(lower=0).astype(int)

    chosen_records = []

    # iterate slices
    for key, need_units in need.items():
        if int(need_units) <= 0:
            continue
        # candidates in the slice
        mask = np.ones(len(df), dtype=bool)
        for d, v in zip(dims, key if isinstance(key, tuple) else (key,)):
            mask &= (df[d].to_numpy() == v)
        idx = df.index[mask].to_numpy()
        if len(idx) == 0:
            continue
        weights = df.loc[idx, "frac"].to_numpy()
        # if all weights are zero, use tiny jitter to avoid degeneracy
        if not np.any(weights > 0):
            weights = np.ones_like(weights) * 1e-12
        chosen = _pps_without_replacement(idx, weights, int(need_units), rng)
        df.loc[chosen, "z"] = 1
        for j in chosen:
            rec = {"row_idx": int(j)}
            for d, v in zip(dims, key if isinstance(key, tuple) else (key,)):
                rec[d] = int(v)
            rec["frac"] = float(df.at[j, "frac"])
            chosen_records.append(rec)

    # enforce global K_add if slight drift (rare due to double counting protection)
    picked = int(df["z"].sum())
    if picked != int(K_add):
        # adjust by trimming/adding a few based on frac
        if picked > K_add:
            drop = picked - K_add
            drop_idx = df.index[df["z"]==1][df.loc[df["z"]==1, "frac"].nsmallest(drop).index]
            df.loc[drop_idx, "z"] = 0
        else:
            add = K_add - picked
            cand = df.index[df["z"]==0][df.loc[df["z"]==0, "frac"].nlargest(add).index]
            df.loc[cand, "z"] = 1

    # finalize n
    df["n"] = (df["n"] + df["z"]).astype(int)

    chosen_df = pd.DataFrame(chosen_records)
    return df, chosen_df

# ---------- Verification ----------
def verify_margins(int_df, constraints, col="n", by_zone=True):
    """
    Corrected verification.
    - If by_zone=True: verify each zone separately (strict).
    - Else: aggregate both sides over ZoneID and verify globally (looser).
    Returns (ok, residuals_dict)
    """
    ok = True
    viols = {}

    if by_zone:
        zones = int_df["ZoneID"].unique().tolist()
        for cs in constraints:
            dims = list(cs.dims)
            # group both sides by ZoneID + dims
            cur = int_df.groupby(["ZoneID", *dims], observed=True, sort=False)[col].sum()
            tgt = cs.df.groupby(["ZoneID", *dims], observed=True, sort=False)[cs.val_col].sum()
            # align and compare
            cur = cur.reindex(tgt.index, fill_value=0)
            gap = float((cur - tgt).abs().sum())
            viols[cs.name] = viols.get(cs.name, 0.0) + gap
            ok = ok and (gap == 0.0)
    else:
        for cs in constraints:
            dims = list(cs.dims)
            cur = int_df.groupby(dims, observed=True, sort=False)[col].sum()
            tgt = cs.df.groupby(dims, observed=True, sort=False)[cs.val_col].sum()
            cur = cur.reindex(tgt.index, fill_value=0)
            gap = float((cur - tgt).abs().sum())
            viols[cs.name] = gap
            ok = ok and (gap == 0.0)
    return ok, viols


# ---------- Pipeline (minimal) ----------
def run_pipeline(base_dir: str,
                 out_dir: str,
                 tol=1e-7, max_iters=200,
                 verbose_ipf=True):
    os.makedirs(out_dir, exist_ok=True)

    # --- Load once ---
    seed, constraints = load_all(base_dir)

    # --- STRUCTURAL ZEROS ---
    keep_mask = build_structural_zero_mask(seed, constraints)
    if (~keep_mask).any():
        print(f"[StructZero] Dropped {int((~keep_mask).sum())} seed rows.")
    seed = seed.loc[keep_mask].copy()

    # --- GHOSTS ---
    seed, ghost_notes = _augment_seed_with_ghosts(seed, constraints, eps_ghost=EPS_GHOST)
    for n in ghost_notes:
        print(n)

    # Common zones
    zones = set(seed["ZoneID"].dropna().astype(int).unique().tolist())
    for cs in constraints:
        zones &= set(cs.df["ZoneID"].dropna().astype(int).unique().tolist())
    if not zones:
        raise ValueError("No common ZoneID between seed and constraints.")
    zones = sorted(zones)

    # Trim constraints to common zones; remove ZoneID from dims
    constraints_proc=[]
    for cs in constraints:
        cdf = cs.df[cs.df["ZoneID"].isin(zones)].copy()
        dims_no_zone = tuple(d for d in cs.dims if d != "ZoneID")
        constraints_proc.append(ConstraintSpec(cs.name, dims_no_zone, cdf, cs.val_col))

    if ANCHOR_NAME not in [c.name for c in constraints_proc]:
        raise ValueError(f"Expected anchor '{ANCHOR_NAME}' not found.")

    dim_cols = [c for c in seed.columns if c not in ("ZoneID","x")]

    # --- IPF per zone -> fractional table ---
    frac_tables = []
    ipf_logs = []

    for z in zones:
        seed_z = seed[seed["ZoneID"]==z].copy()
        cons_z = [ConstraintSpec(cs.name, cs.dims, cs.df[cs.df["ZoneID"]==z].copy(), cs.val_col)
                  for cs in constraints_proc]

        # Stage-0 (harmonize + support projection)
        _, cons_z, _ = stage0_harmonize_zone(seed_z, cons_z, controlling_name=ANCHOR_NAME, slice_dims=ANCHOR_SLICE_DIMS)
        cons_z, _ = _reconcile_zone_totals_global(cons_z, controlling_name=ANCHOR_NAME)

        # Determine zone target from anchor & scale seed
        zone_target = float(next(cz for cz in cons_z if cz.name==ANCHOR_NAME).df[cons_z[0].val_col].sum())
        mass0 = float(seed_z["x"].sum())
        if mass0 <= 0:
            raise ValueError(f"[Zone {z}] Seed mass is zero.")
        seed_z["x"] *= (zone_target / max(mass0, EPS))

        hipf = HardIPF(seed_z[["ZoneID", *dim_cols, "x"]], cons_z, weight_col="x")
        w_ret, ok, nit, mre = hipf.fit(tol=tol, max_iters=max_iters, verbose=verbose_ipf)

        out_df = seed_z[["ZoneID", *dim_cols]].copy()

        # interpret w_ret as masses or multipliers
        sum_w = float(np.sum(w_ret))
        if abs(sum_w - zone_target) / max(zone_target, 1.0) < 1e-8:
            x_ipf = w_ret.astype(float)
        else:
            x_ipf = (seed_z["x"].to_numpy() * w_ret).astype(float)

        # tiny renorm to anchor total
        scale = zone_target / max(np.sum(x_ipf), EPS)
        x_ipf *= scale
        out_df["x"] = x_ipf
        frac_tables.append(out_df.assign(ZoneID=z))
        ipf_logs.append({"ZoneID": z, "converged": ok, "n_iter": int(nit), "max_rel_err": float(mre)})

    frac_df = pd.concat(frac_tables, ignore_index=True)
    frac_out = os.path.join(out_dir, "fractional_fit.csv")
    frac_df.to_csv(frac_out, index=False)

    # --- STEP 1: split (floor + residual target) ---
    step1_logs = []
    step1_frames = []

    for z in zones:
        base = frac_df[frac_df["ZoneID"]==z].copy()
        cons_z = [ConstraintSpec(cs.name, cs.dims, cs.df[cs.df["ZoneID"]==z].copy(), cs.val_col)
                  for cs in constraints_proc]
        anchor = next(cz for cz in cons_z if cz.name==ANCHOR_NAME)
        zone_target = int(round(float(anchor.df[anchor.val_col].sum())))

        df_split, K_add = step1_split(base, zone_target)
        step1_logs.append({"ZoneID": z, "zone_target": zone_target, "floor_sum": int(df_split["n"].sum()), "K_add": K_add})
        step1_frames.append(df_split)

    df_step1 = pd.concat(step1_frames, ignore_index=True)
    df_step1.to_csv(os.path.join(out_dir, "step1_floor.csv"), index=False)
    pd.DataFrame(step1_logs).to_csv(os.path.join(out_dir, "step1_log.csv"), index=False)

    # --- STEP 2: anchor-conditioned PPS (no swap/ILP yet) ---
    int_tables = []
    chosen_all = []

    for z in zones:
        df_split = df_step1[df_step1["ZoneID"]==z].copy()
        cons_z = [ConstraintSpec(cs.name, cs.dims, cs.df[cs.df["ZoneID"]==z].copy(), cs.val_col)
                  for cs in constraints_proc]
        K_add = int(pd.read_csv(os.path.join(out_dir, "step1_log.csv")).set_index("ZoneID").loc[z, "K_add"])
        df_after, chosen = step2_anchor_pps(df_split, cons_z, ANCHOR_SLICE_DIMS, K_add, seed=123)
        int_tables.append(df_after[["ZoneID", *dim_cols, "n"]])
        if not chosen.empty:
            chosen["ZoneID"] = z
            chosen_all.append(chosen)

    int_df = pd.concat(int_tables, ignore_index=True)
    # Seeded PPS table (the paper's "seeded slice sampling" row in Table 3).
    # Named tmp_Minimal_Integerized.csv so that step3_integerize.py does not overwrite it.
    int_out = os.path.join(out_dir, "tmp_Minimal_Integerized.csv")
    int_df.to_csv(int_out, index=False)

    if chosen_all:
        pd.concat(chosen_all, ignore_index=True).to_csv(os.path.join(out_dir, "step2_anchor_selected.csv"), index=False)

    # --- Verification (after step 2 only) ---
    ok_all, viols_all = verify_margins(int_df, constraints_proc, "n")
    pd.DataFrame([{"rounding_verified_global": ok_all, **{f"viol_{k}": v for k, v in viols_all.items()}}])\
        .to_csv(os.path.join(out_dir, "residuals_after_step2.csv"), index=False)

    # --- Extra diagnostics ---
    # 1) Pick vs K_add per zone
    picked_per_zone = pd.read_csv(os.path.join(out_dir, "step2_anchor_selected.csv")) \
                         .groupby("ZoneID")["row_idx"].count() if chosen_all else pd.Series()
    kadd_per_zone = pd.read_csv(os.path.join(out_dir, "step1_log.csv")).set_index("ZoneID")["K_add"]
    pick_vs_kadd = pd.DataFrame({"picked": picked_per_zone, "K_add": kadd_per_zone}).fillna(0).astype(int)
    pick_vs_kadd["delta"] = pick_vs_kadd["picked"] - pick_vs_kadd["K_add"]
    pick_vs_kadd.to_csv(os.path.join(out_dir, "pick_vs_kadd_per_zone.csv"), index=True)

    # 2) Margin gaps (integer vs fractional)
    def gap(int_df, frac_df, dims):
        cur = int_df.groupby(dims, observed=True, sort=False)["n"].sum()
        tgt = frac_df.groupby(dims, observed=True, sort=False)["x"].sum()
        cur = cur.reindex(tgt.index, fill_value=0)
        absdiff = (cur - tgt).abs()
        return float(absdiff.sum()), float(absdiff.max())

    dims_list = {
        "Age×Gender": ["ZoneID","AgeID","GenderID"],
        "Age×Children": ["ZoneID","AgeID","NumChildID"],
        "Age×Family": ["ZoneID","AgeID","FamID"],
        "Age×Income": ["ZoneID","AgeID","IncomeID"],
        "Age×Lma": ["ZoneID","AgeID","LmaID"]
    }
    margin_summary = []
    for name, dims in dims_list.items():
        L1, Linf = gap(int_df, frac_df, dims)
        margin_summary.append({"margin": name, "L1_abs_gap_vs_frac": L1, "Linf_abs_gap_vs_frac": Linf})
    pd.DataFrame(margin_summary).to_csv(os.path.join(out_dir, "margin_gaps_vs_fractional_global.csv"), index=False)

    # 3) Ghost-like rows usage
    frac_with_idx = frac_df.copy()
    frac_with_idx["is_ghost_like"] = frac_with_idx["x"] <= 1e-6
    ghost_like = frac_with_idx.merge(int_df, on=["ZoneID","AgeID","NumChildID","FamID","GenderID","IncomeID","LmaID"], how="left")
    ghost_like["n"] = ghost_like["n"].fillna(0).astype(int)
    ghost_like.groupby("is_ghost_like")["n"].sum().rename("integer_mass") \
              .reset_index().to_csv(os.path.join(out_dir, "ghost_like_integer_mass.csv"), index=False)

    # 4) Anchor margin gaps per zone
    dims_anchor = ["ZoneID","AgeID","GenderID"]
    cur_anchor = int_df.groupby(dims_anchor, observed=True, sort=False)["n"].sum()
    tgt_anchor = frac_df.groupby(dims_anchor, observed=True, sort=False)["x"].sum()
    cur_anchor = cur_anchor.reindex(tgt_anchor.index, fill_value=0)
    absdiff_anchor = (cur_anchor - tgt_anchor).abs().reset_index(name="abs_gap")
    anchor_zone_L1 = absdiff_anchor.groupby("ZoneID")["abs_gap"].sum().reset_index().rename(columns={"abs_gap":"L1_abs_gap"})
    anchor_zone_L1.to_csv(os.path.join(out_dir, "anchor_age_gender_gap_per_zone.csv"), index=False)

    # 5) Totals summary
    pd.DataFrame([{
        "total_integer": int(int_df["n"].sum()),
        "total_fractional": float(frac_df["x"].sum()),
        "totals_equal": int(int_df["n"].sum()) == int(round(float(frac_df["x"].sum())))
    }]).to_csv(os.path.join(out_dir, "totals_summary.csv"), index=False)


    # --- Summary ---
    pd.DataFrame([{
        "n_zones": len(zones),
        "ipf_converged_zones": int(pd.DataFrame(ipf_logs)["converged"].sum()) if ipf_logs else 0,
        "ipf_max_rel_err_overall": float(pd.DataFrame(ipf_logs)["max_rel_err"].max() if ipf_logs else 0.0),
        "fractional_fit_path": os.path.abspath(frac_out),
        "integer_table_path": os.path.abspath(int_out),
        "residuals_after_step2_path": os.path.abspath(os.path.join(out_dir, "residuals_after_step2.csv"))
    }]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print("\n=== SPAR-TCS (v8-minimal) Complete ===")
    print(f"Fractional fit : {frac_out}")
    print(f"Step1 floor    : {os.path.join(out_dir, 'step1_floor.csv')}")
    print(f"Integer table  : {int_out}  (after anchor PPS only)")
    print(f"Residuals      : {os.path.join(out_dir, 'residuals_after_step2.csv')}")
    print(f"Summary        : {os.path.join(out_dir, 'summary.csv')}")

# ---------- CLI ----------
if __name__ == "__main__":
    run_pipeline(
        base_dir=BASE_DIR,
        out_dir=OUT_DIR,
        tol=IPF_TOL,
        max_iters=IPF_MAX_ITERS,
        verbose_ipf=VERBOSE_IPF,
    )
