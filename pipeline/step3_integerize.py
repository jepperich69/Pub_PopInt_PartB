
# -*- coding: utf-8 -*-
"""
Intege_Paper_Minimal_SWAP_with_tables_v4.py

SPAR-TCS (v9b-swap) — Minimal pipeline + anchor-preserving swap-repair,
with Windows/OneDrive-safe I/O (no mid-loop re-reads of step1_log.csv).

Cookbook mapping:
  §0 Load + struct-zeros + ghosts
  §Stage-0 Harmonize + support projection
  §IPF (hard-coded, NumPy-only)
  §1 Split (floor + residual K_add)
  §2 Anchor-conditioned PPS rounding
  §3 Swap-repair (anchor-preserving two-cell moves)
  §5 Diagnostics & verification
  §6 Post-stage summaries (violin-like, θ overlap, LaTeX tables)
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

ANCHOR_NAME = "Age×Gender"
ANCHOR_SLICE_DIMS = ("ZoneID","AgeID","GenderID")

HARM_ABS_TOL = 5.0
HARM_REL_TOL = 0.005

ENABLE_SWAPS = True
SWAP_MAX_PASSES = 3
SWAP_MAX_MOVES_PER_SLICE = 2000
GHOST_X_EPS = 1e-6

# Dump per-zone histograms (many PNGs). Default OFF for shareability.
DUMP_ZONE_HISTS = False

# Make cross-zone violin-like summary plots (L2 & KL). Default ON.
MAKE_VIOLIN_SUMMARY = True

# Compute θ-overlap violin plot. Default ON.
MAKE_OVERLAP_VIOLIN = True

# If you have a "minimal" integer solution (slightly worse than swap),
# include it in the θ-overlap violin for comparison. Default ON.
INCLUDE_MINIMAL_IN_OVERLAP = True

# Optional file name for minimal solution (same schema as the repaired integer df).
MINIMAL_CSV_NAME = "tmp_Minimal_Integerized.csv"
# =================================================

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import json

EPS = 1e-18
EPS_GHOST = 1e-9

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

def _latex_escape(s: str) -> str:
    s = str(s)
    rep = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
    }
    for k,v in rep.items():
        s = s.replace(k,v)
    return s

def _write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _fmt_num(x, digits=3):
    try:
        if x is None or (isinstance(x, float) and (not math.isfinite(x))):
            return "--"
        if abs(x) >= 100:
            return f"{x:.1f}"
        return f"{x:.{digits}f}"
    except Exception:
        return "--"

# ---------- problem spec ----------
@dataclass
class ConstraintSpec:
    name: str
    dims: Tuple[str, ...]
    df: pd.DataFrame
    val_col: str = "Val"

# ---------- loader ----------
def load_all(base_dir: str):
    start_fp = os.path.join(OUT_DIR, "tmp_Pop_SyntheticSeed.csv")  # written by step1_generate_seed.py
    df = _read_csv(start_fp).rename(columns={"Val": "x", "ZoneID0": "ZoneID"})
    needed = ["ZoneID","AgeID","GenderID","LmaID","NumChildID","FamID","IncomeID","x"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in seed table")
    df = _ensure_int_columns(df, ["ZoneID","AgeID","GenderID","LmaID","NumChildID","FamID","IncomeID"])
    df = _ensure_float_column(df, "x")
    df["x"] = df["x"].clip(lower=1e-12)

    constraints: List[ConstraintSpec] = []

    def load_constraint(filename: str, dims: List[str], name: str):
        fp = os.path.join(base_dir, filename)
        cdf = _read_csv(fp).rename(columns={"ZoneID0": "ZoneID"})
        cdf = _ensure_int_columns(cdf, ["ZoneID","AgeID","GenderID","LmaID","NumChildID","FamID","IncomeID"])
        cdf = _ensure_float_column(cdf, "Val")
        keep = dims + ["Val"]
        for d in dims:
            if d not in cdf.columns:
                raise ValueError(f"{name}: expected dim '{d}' not present")
        constraints.append(ConstraintSpec(name=name, dims=tuple(dims), df=cdf[keep].copy(), val_col="Val"))

    # NOTE: Uses your naming ('NumChildID', not 'Children').
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
        idx = (
            seed.merge(zero_keys.assign(__drop__=True), on=dims, how="left")["__drop__"]
                .fillna(False)
                .astype(bool)
        )
        keep &= ~idx
    return keep

# ---------- ghosts ----------
def _augment_seed_with_ghosts(seed: pd.DataFrame,
                              constraints: List[ConstraintSpec],
                              eps_ghost: float = EPS_GHOST):
    notes=[]
    seed_cols = [c for c in seed.columns if c != "x"]
    all_dims = [c for c in seed_cols if c != "ZoneID"]
    exist_idx = pd.MultiIndex.from_frame(seed[["ZoneID", *all_dims]])
    new_rows = []

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

    for (z,a), posmap in ZA_pos.items():
        row = {"ZoneID": z, "AgeID": a}
        for d in all_dims:
            if d in posmap and len(posmap[d]) > 0:
                row[d] = posmap[d][0]
            else:
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

# ---------- Stage 0 helpers ----------
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
                            rel_tol: float):
    notes=[]
    ctrl = next(cz for cz in cons_z if cz.name == controlling_name)
    ctrl_df = ctrl.df.copy(); val_col = ctrl.val_col
    aligned=[]
    for cz in cons_z:
        if cz.name == controlling_name:
            aligned.append(cz); continue
        df = cz.df.copy(); by_dims = tuple(d for d in slice_dims if d in df.columns)
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
                          slice_dims: Tuple[str, ...]):
    notes=[]
    ctrl = next(cz for cz in cons_z if cz.name == controlling_name)
    ctrl_df = ctrl.df.copy(); val_col = ctrl.val_col
    projected=[]
    for cz in cons_z:
        df = cz.df.copy(); dims = list(cz.dims)
        by_dims = tuple(d for d in slice_dims if d in dims)
        if len(by_dims) > 0:
            T = ctrl_df.groupby(list(by_dims), observed=True, sort=False)[val_col].sum()
        else:
            T = pd.Series({(): float(ctrl_df[val_col].sum())})
        supp_idx = _seed_supported_index(seed_z, dims)
        idx = df.set_index(dims).index
        is_sup = idx.isin(supp_idx)
        df_sup = df.loc[is_sup].copy(); df_nsup = df.loc[~is_sup].copy()
        if len(by_dims) > 0 and len(df_sup):
            Ssup = df_sup.groupby(list(by_dims), observed=True, sort=False)[val_col].sum()
            df_sup = df_sup.merge(Ssup.rename("__Ssup__"), on=list(by_dims), how="left")
            df_sup = df_sup.merge(T.rename("__T__"), on=list(by_dims), how="left")
            fac = _safe_divide(df_sup["__T__"].values, df_sup["__Ssup__"].replace({0.0: np.nan}).values, default=0.0)
            df_sup[val_col] = df_sup[val_col] * np.asarray(fac)
            df_sup.drop(columns=["__Ssup__","__T__"], inplace=True)
        elif len(df_sup):
            Ssup_total = df_sup[val_col].sum(); T_total = float(ctrl_df[val_col].sum())
            fac = (T_total / Ssup_total) if Ssup_total > 0 else 1.0
            df_sup[val_col] = df_sup[val_col] * fac
        df_nsup[val_col] = 0.0
        df_new = pd.concat([df_sup, df_nsup], ignore_index=True)
        projected.append(ConstraintSpec(cz.name, cz.dims, df_new, cz.val_col))
    return projected, notes

def _reconcile_zone_totals_global(cons_z: List[ConstraintSpec], controlling_name: str):
    ctrl = next(cz for cz in cons_z if cz.name == controlling_name)
    T = float(ctrl.df[ctrl.val_col].sum())
    out=[]
    for cz in cons_z:
        if cz.name == controlling_name:
            out.append(cz); continue
        S = float(cz.df[cz.val_col].sum())
        if S<=0 or T<=0:
            out.append(cz); continue
        if abs(S-T)>1e-9:
            f=T/S; df2 = cz.df.copy(); df2[cz.val_col] = df2[cz.val_col]*f
            out.append(ConstraintSpec(cz.name, cz.dims, df2, cz.val_col))
        else:
            out.append(cz)
    return out, []

# ---------- Hard-coded fast IPF (NumPy-only) ----------
class HardIPF:
    def __init__(self, seed_df: pd.DataFrame, constraints: List[ConstraintSpec], weight_col="x"):
        keep_cols = [c for c in seed_df.columns if c not in {"w","n","frac"}]
        self.df = seed_df[keep_cols].copy()
        self.weight = self.df[weight_col].to_numpy(dtype=float)
        self.cols = [c for c in self.df.columns if c != weight_col]
        self.codes = {}; self.levels = {}
        for c in self.cols:
            cats, inv = np.unique(self.df[c].to_numpy(), return_inverse=True)
            self.codes[c] = inv.astype(np.int64)
            self.levels[c] = cats
        self.margins = []; self.names = []
        for cs in constraints:
            dims = list(cs.dims)
            self.names.append(cs.name)
            shape = tuple(len(self.levels[d]) for d in dims)
            tdf = cs.df[dims + [cs.val_col]].copy()
            ok = np.ones(len(tdf), dtype=bool); maps={}
            for d in dims:
                lvl = self.levels[d]; mp = {v:i for i,v in enumerate(lvl)}; maps[d]=mp
                ok &= tdf[d].map(mp).notna().to_numpy()
            if not ok.all():
                tdf = tdf.loc[ok].copy()
            idx = np.ravel_multi_index(tuple(tdf[d].map(maps[d]).to_numpy(int) for d in dims), dims=shape)
            tgt = np.zeros(np.prod(shape), dtype=float)
            np.add.at(tgt, idx, tdf[cs.val_col].to_numpy(dtype=float))
            tgt = tgt.reshape(shape)
            row_flat = np.ravel_multi_index(tuple(self.codes[d] for d in dims), dims=shape)
            self.margins.append((dims, shape, row_flat, tgt))

    @staticmethod
    def _rel_err_safe(tgt: np.ndarray, cur: np.ndarray) -> float:
        denom = np.maximum(np.maximum(np.abs(tgt), np.abs(cur)), EPS)
        return float(np.max(np.abs(tgt - cur) / denom)) if denom.size else 0.0

    def fit(self, tol=1e-7, max_iters=200, verbose=True):
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
                print(f"[IPF] iter={it:3d} max_rel_err={max_rel:.3e}")
            if max_rel < tol:
                return w, True, it, float(max_rel)
        return w, False, max_iters, float(max_rel)

# ---------- Cookbook Step 1: Deterministic split ----------
def step1_split(frac_df_zone: pd.DataFrame, zone_target: int):
    df = frac_df_zone.copy()
    df["n"] = np.floor(df["x"] + 1e-9).astype(int)
    df["frac"] = df["x"] - df["n"]
    floor_sum = int(df["n"].sum())
    K_add = int(zone_target - floor_sum)
    if K_add < 0:
        take = -K_add
        idx = df["frac"].nsmallest(take).index
        df.loc[idx, "n"] = (df.loc[idx, "n"] - 1).astype(int)
        K_add = 0
    return df, K_add

# ---------- PPS without replacement ----------
def _pps_without_replacement(idx_array: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator):
    w = weights.clip(min=1e-12)
    p = w / w.sum()
    k = int(min(k, len(idx_array)))
    if k <= 0:
        return np.array([], dtype=idx_array.dtype)
    chosen = rng.choice(idx_array, size=k, replace=False, p=p)
    return chosen

# ---------- Cookbook Step 2: Anchor-conditioned PPS sampling ----------
def step2_anchor_deterministic(df_split: pd.DataFrame,
                                cons_z: List[ConstraintSpec],
                                anchor_dims: Tuple[str, ...],
                                K_add: int):
    """Deterministic largest-remainder allocation within anchor slices."""
    df = df_split.copy()
    df["z"] = 0
    
    anchor = next(cz for cz in cons_z if cz.name == ANCHOR_NAME)
    dims = ["AgeID","GenderID"]
    tgt = anchor.df.groupby(dims, observed=True, sort=False)[anchor.val_col].sum().rename("tgt")
    flr = df.groupby(dims, observed=True, sort=False)["n"].sum().rename("floor")
    need = (tgt - flr).reindex(tgt.index, fill_value=0).clip(lower=0).astype(int)
    
    for key, need_units in need.items():
        if int(need_units) <= 0:
            continue
        
        # Find candidates in this slice
        mask = np.ones(len(df), dtype=bool)
        for d, v in zip(dims, key if isinstance(key, tuple) else (key,)):
            mask &= (df[d].to_numpy() == v)
        
        candidates = df.loc[mask].copy()
        if len(candidates) == 0:
            continue
        
        # Sort by fractional part (descending) and take top need_units
        # Use index as tiebreaker for full determinism
        candidates = candidates.sort_values(
            by=["frac"], 
            ascending=False
        ).head(int(need_units))
        
        df.loc[candidates.index, "z"] = 1
    
    df["n"] = (df["n"] + df["z"]).astype(int)
    return df

def step2_anchor_pps(df_split: pd.DataFrame,
                     cons_z: List[ConstraintSpec],
                     anchor_dims: Tuple[str, ...],
                     K_add: int,
                     seed: int = 42):
    rng = np.random.default_rng(seed)
    df = df_split.copy()
    df["z"] = 0  # indicator for +1

    anchor = next(cz for cz in cons_z if cz.name == ANCHOR_NAME)
    dims = ["AgeID","GenderID"]  # Zone already filtered
    tgt = anchor.df.groupby(dims, observed=True, sort=False)[anchor.val_col].sum().rename("tgt")
    flr = df.groupby(dims, observed=True, sort=False)["n"].sum().rename("floor")
    need = (tgt - flr).reindex(tgt.index, fill_value=0).clip(lower=0).astype(int)

    chosen_records = []

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

    df["n"] = (df["n"] + df["z"]).astype(int)
    chosen_df = pd.DataFrame(chosen_records)
    return df, chosen_df

# ---------- Verification ----------
def verify_margins(int_df: pd.DataFrame, constraints: List[ConstraintSpec], col="n", by_zone=True):
    ok=True; viols={}
    if by_zone:
        for cs in constraints:
            dims=list(cs.dims)
            cur=int_df.groupby(["ZoneID",*dims], observed=True, sort=False)[col].sum()
            tgt=cs.df.groupby(["ZoneID",*dims], observed=True, sort=False)[cs.val_col].sum()
            cur=cur.reindex(tgt.index, fill_value=0)
            gap=float((cur-tgt).abs().sum())
            viols[cs.name]=gap; ok=ok and (gap==0.0)
    else:
        for cs in constraints:
            dims=list(cs.dims)
            cur=int_df.groupby(dims, observed=True, sort=False)[col].sum()
            tgt=cs.df.groupby(dims, observed=True, sort=False)[cs.val_col].sum()
            cur=cur.reindex(tgt.index, fill_value=0)
            gap=float((cur-tgt).abs().sum())
            viols[cs.name]=gap; ok=ok and (gap==0.0)
    return ok, viols

# ---------- Step 3: Swap-repair (anchor-preserving) ----------
def _compute_zone_residuals(int_df_zone: pd.DataFrame, cons_z: List[ConstraintSpec]):
    resid = {}
    for cs in cons_z:
        if cs.name == ANCHOR_NAME: 
            continue
        dims=list(cs.dims)  # ["AgeID", OtherDim]
        tgt=cs.df.groupby(dims, observed=True, sort=False)[cs.val_col].sum()
        cur=int_df_zone.groupby(dims, observed=True, sort=False)["n"].sum()
        cur=cur.reindex(tgt.index, fill_value=0)
        resid[cs.name]=(tgt-cur).astype(float)
    return resid

def swap_repair_zone(int_df_zone: pd.DataFrame,
                     frac_df_zone: pd.DataFrame,
                     cons_z: List[ConstraintSpec],
                     max_passes: int = SWAP_MAX_PASSES,
                     max_moves_per_slice: int = SWAP_MAX_MOVES_PER_SLICE,
                     ghost_x_eps: float = GHOST_X_EPS):
    key_cols = ["AgeID","NumChildID","FamID","GenderID","IncomeID","LmaID"]

    # Merge integer counts with fractional x
    merged = int_df_zone.merge(
        frac_df_zone[key_cols + ["x"]],
        on=key_cols, how="left"
    )
    merged["x"] = merged["x"].fillna(0.0)
    merged["delta"] = merged["x"] - merged["n"]
    merged["ghost_blocked"] = merged["x"] <= ghost_x_eps

    # Residuals per non-anchor margin (indexed by ["AgeID", other_dim])
    resid = _compute_zone_residuals(merged, cons_z)

    # Map each non-anchor margin name -> its "other" dimension (besides AgeID)
    margin_dims = {}
    for cs in cons_z:
        if cs.name == ANCHOR_NAME:
            continue
        other = [d for d in cs.dims if d != "AgeID"][0]
        margin_dims[cs.name] = other

    moves_total = 0

    for _pass in range(int(max_passes)):
        moves_in_pass = 0

        # Work slice-by-slice within (AgeID, GenderID)
        for (age, gen), sl_idx in merged.groupby(["AgeID", "GenderID"], sort=False).groups.items():
            # sl_idx are LABELS of 'merged'. Build a slice 'sl' and rebase to a fresh RangeIndex
            idx = np.asarray(list(sl_idx), dtype=merged.index.dtype)
            sl = merged.loc[idx].copy()

            # Keep a pointer back to parent rows and rebase slice index to 0..len-1
            sl["_orig"] = sl.index.to_numpy()
            sl.reset_index(drop=True, inplace=True)

            # Identify donors/receivers within the slice
            donors_mask    = (sl["n"] > 0) & (sl["delta"] < -1e-9)
            receivers_mask = (sl["delta"] >  1e-9) & (~sl["ghost_blocked"])
            if not donors_mask.any() or not receivers_mask.any():
                continue

            # Precompute, for each margin, a map: value -> (row positions within sl)
            per_dim_index = {}
            for mname, dim in margin_dims.items():
                gb = sl.groupby(dim, sort=False, observed=True)
                # gb.indices returns index LABELS of 'sl'; since 'sl' has RangeIndex, labels==positions
                per_dim_index[(mname, "val_to_rows")] = {
                    k: np.asarray(v, dtype=np.int64) for k, v in gb.indices.items()
                }

            slice_moves = 0
            while slice_moves < int(max_moves_per_slice):
                improved = False

                for mname, dim in margin_dims.items():
                    r_series = resid[mname]
                    # extract residuals at the current 'age'
                    try:
                        r_age = r_series.xs(age, level=r_series.index.names.index("AgeID"))
                    except Exception:
                        continue

                    deficits  = [v for v, r in r_age.items() if r >  0.5]
                    surpluses = [v for v, r in r_age.items() if r < -0.5]
                    if not deficits or not surpluses:
                        continue

                    val_to_rows = per_dim_index[(mname, "val_to_rows")]

                    for v_def in deficits:
                        recv_rows = val_to_rows.get(v_def, np.array([], dtype=np.int64))
                        if recv_rows.size == 0:
                            continue

                        # Positional selection within the slice
                        recv_cand = sl.iloc[recv_rows]
                        recv_cand = recv_cand[(recv_cand["delta"] > 1e-9) & (~recv_cand["ghost_blocked"])]
                        if recv_cand.empty:
                            continue

                        # Row position within 'sl' (since sl has RangeIndex, idx is positional)
                        recv_row = int(recv_cand["delta"].idxmax())

                        donor_row  = None
                        best_delta = 0.0
                        for v_sur in surpluses:
                            don_rows = val_to_rows.get(v_sur, np.array([], dtype=np.int64))
                            if don_rows.size == 0:
                                continue

                            don_cand = sl.iloc[don_rows]
                            don_cand = don_cand[(don_cand["delta"] < -1e-9) & (don_cand["n"] > 0)]
                            if don_cand.empty:
                                continue

                            cand_row   = int(don_cand["delta"].idxmin())
                            cand_delta = float(don_cand["delta"].min())
                            if (donor_row is None) or (cand_delta < best_delta):
                                donor_row  = cand_row
                                best_delta = cand_delta

                        if donor_row is None:
                            continue

                        # Map slice rows back to the parent 'merged' via _orig
                        donor_orig = int(sl.at[donor_row, "_orig"])
                        recv_orig  = int(sl.at[recv_row,  "_orig"])

                        # Apply move on parent rows (use .loc with labels from _orig)
                        merged.loc[donor_orig, "n"]     -= 1
                        merged.loc[recv_orig,  "n"]     += 1
                        merged.loc[donor_orig, "delta"] += 1.0
                        merged.loc[recv_orig,  "delta"] -= 1.0

                        # Keep slice view in sync with parent
                        sl.loc[donor_row, "n"]     = merged.at[donor_orig, "n"]
                        sl.loc[recv_row,  "n"]     = merged.at[recv_orig,  "n"]
                        sl.loc[donor_row, "delta"] = merged.at[donor_orig, "delta"]
                        sl.loc[recv_row,  "delta"] = merged.at[recv_orig,  "delta"]

                        # Update residual tracker for this margin
                        resid[mname].loc[(age, sl.at[recv_row,  dim])] -= 1.0
                        resid[mname].loc[(age, sl.at[donor_row, dim])] += 1.0

                        slice_moves += 1
                        moves_total += 1
                        improved = True

                        if slice_moves >= int(max_moves_per_slice):
                            break  # stop this slice for this margin

                if not improved:
                    break  # no progress in this slice

            moves_in_pass += slice_moves

        if moves_in_pass == 0:
            break  # no progress this pass → stop

    out_cols = ["ZoneID","AgeID","NumChildID","FamID","GenderID","IncomeID","LmaID","n"]
    return merged[out_cols].copy(), {"moves_total": float(moves_total)}

# ---------- POST-STAGE: variability baseline vs. integerization ----------

# ---- PATCH START: minimal-aware helpers ----
def _try_load_minimal_df(out_dir: str, default_name: str = None):
    """Try to load a minimal integer solution CSV. Returns DataFrame with KEY_COLS + ['n'] or None."""
    try_paths = []
    if default_name:
        try_paths.append(os.path.join(out_dir, default_name))
    # scan directory for candidates with 'minimal' in the name
    for fn in os.listdir(out_dir):
        if fn.lower().endswith(".csv") and "minimal" in fn.lower():
            p = os.path.join(out_dir, fn)
            if p not in try_paths:
                try_paths.append(p)

    for p in try_paths:
        try:
            df = pd.read_csv(p)
            cols = set(df.columns)
            need = set(KEY_COLS)
            if need.issubset(cols) and ("n" in cols or "n_min" in cols):
                if "n_min" in df.columns and "n" not in df.columns:
                    df = df.rename(columns={"n_min": "n"})
                # Coerce dtypes
                for c in KEY_COLS:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
                df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
                print(f"[Minimal] Loaded minimal integer solution: {p}")
                return df[KEY_COLS + ["n"]].copy()
        except Exception as e:
            print(f"[Minimal] Skipped candidate {p}: {e}")
    print("[Minimal] No valid minimal CSV found.")
    return None
# ---- PATCH END ----
KEY_COLS = ["ZoneID","AgeID","NumChildID","FamID","GenderID","IncomeID","LmaID"]

def _safe_kl(p, q, eps=1e-12):
    # KL(p||q), both p and q must sum to 1; add eps to avoid log(0)
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), eps); q = q / max(q.sum(), eps)
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

def _poststage_make_violin_like_plots(out_dir: str, summary_csv: str = "tmp_VariationBaseline_Summary.csv"):
    """
    Create 'violin-like' comparison plots (using 5/50/95% quantiles) for
    L2 distance and KL divergence, overlaying the integerized solution.
    Files:
      - variation_comparison_violin.png       (L2)
      - variation_comparison_violin_KL.png    (KL)
    """
    path = os.path.join(out_dir, summary_csv)
    if not os.path.exists(path):
        print(f"[PostStage Plots] Summary file not found: {path}")
        return
    summary = pd.read_csv(path)

    # ---- L2 variant ----
    fig = plt.figure(figsize=(10, 6))
    has_min = "E_min_L2" in summary.columns and summary["E_min_L2"].notna().any()
    for i, row in summary.iterrows():
        x = str(int(row["ZoneID"]))
        plt.plot([x, x],
                 [row["E_samp_L2_p05"], row["E_samp_L2_p95"]],
                 linewidth=3)
        if i == 0:
            plt.plot(x, row["E_samp_L2_p50"], "o", label="Sample median")
        else:
            plt.plot(x, row["E_samp_L2_p50"], "o")
    plt.scatter(summary["ZoneID"].astype(int).astype(str),
                summary["E_int_L2"],
                marker="x", s=80, label="Integerized")
    if has_min:
        plt.scatter(summary["ZoneID"].astype(int).astype(str),
                    summary["E_min_L2"],
                    marker="s", s=60, label="Minimal")
    plt.yscale("log")
    plt.ylabel("L2 distance to fractional x (log scale)")
    plt.xlabel("ZoneID")
    plt.title("Integerized vs. sampling baseline error per zone (L2)")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_l2 = os.path.join(out_dir, "variation_comparison_violin.png")
    fig.savefig(out_l2, dpi=300)
    plt.close(fig)

    # ---- KL variant ----
    fig = plt.figure(figsize=(10, 6))
    has_min_kl = "KL_min" in summary.columns and summary["KL_min"].notna().any()
    for i, row in summary.iterrows():
        x = str(int(row["ZoneID"]))
        plt.plot([x, x],
                 [row["KL_samp_p05"], row["KL_samp_p95"]],
                 linewidth=3)
        if i == 0:
            plt.plot(x, row["KL_samp_p50"], "o", label="Sample median")
        else:
            plt.plot(x, row["KL_samp_p50"], "o")
    plt.scatter(summary["ZoneID"].astype(int).astype(str),
                summary["KL_int"],
                marker="x", s=80, label="Integerized")
    if has_min_kl:
        plt.scatter(summary["ZoneID"].astype(int).astype(str),
                    summary["KL_min"],
                    marker="s", s=60, label="Minimal")
    plt.yscale("log")
    plt.ylabel("KL(p||q) to fractional p (log scale)")
    plt.xlabel("ZoneID")
    plt.title("Integerized vs. sampling baseline per zone (KL)")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_kl = os.path.join(out_dir, "variation_comparison_violin_KL.png")
    fig.savefig(out_kl, dpi=300)
    plt.close(fig)

    print(f"[PostStage Plots] Saved:\n  {out_l2}\n  {out_kl}")

def poststage_variation_baseline(
    repaired_all: pd.DataFrame,
    frac_df: pd.DataFrame,
    out_dir: str,
    zones: List[int],
    R: int = 200,
    seed: int = 42,
    top_k_zones: int = 20,
    minimal_df: Optional[pd.DataFrame] = None,
):
    """
    Compare deterministic integerized counts to a multinomial sampling baseline.
    Writes summary CSV + 2 plots to out_dir.
    """
    # Align repaired integers and fractional solution
    minimal_present = minimal_df is not None and set(KEY_COLS).issubset(set(minimal_df.columns)) and ('n' in minimal_df.columns)

    need_cols_int = set(KEY_COLS + ["n"])
    need_cols_frac = set(KEY_COLS + ["x"])
    for need, df, name in [(need_cols_int, repaired_all, "repaired_all"),
                           (need_cols_frac, frac_df, "frac_df")]:
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"[poststage] {name} missing required columns: {sorted(missing)}")

    df = frac_df[KEY_COLS + ["x"]].merge(
        repaired_all[KEY_COLS + ["n"]],
        on=KEY_COLS, how="inner", validate="one_to_one"
    )

    df_min = None
    if minimal_present:
        df_min = frac_df[KEY_COLS + ["x"]].merge(
            minimal_df[KEY_COLS + ["n"]].rename(columns={"n":"n_min"}),
            on=KEY_COLS, how="inner", validate="one_to_one"
        )

    # Choose which zones to analyze (largest K by mass)
    zone_mass = df.groupby("ZoneID")["n"].sum().sort_values(ascending=False)
    chosen_zones = zone_mass.index.tolist()[: min(top_k_zones, len(zone_mass))]
    chosen_zones = [z for z in chosen_zones if z in set(zones)]

    rng = np.random.default_rng(seed)
    rows = []

    for z in chosen_zones:
        sub = df[df["ZoneID"] == z].copy()
        x = sub["x"].to_numpy(float)
        n = sub["n"].to_numpy(float)

        T = int(np.round(n.sum()))
        mass_x = float(x.sum())

        if T <= 0 or mass_x <= 0:
            continue

        p = x / mass_x
        q_int = n / max(T, 1)

        # Metrics for the deterministic integer solution
        e_min_l2 = None; kl_min = None
        if minimal_present and df_min is not None:
            subm = df_min[df_min["ZoneID"] == z].copy()
            if not subm.empty:
                nm = subm["n_min"].to_numpy(float)
                Tm = int(np.round(nm.sum()))
                if Tm > 0:
                    q_min = nm / float(Tm)
                    e_min_l2 = float(np.linalg.norm(nm - x))
                    kl_min   = _safe_kl(p, q_min)

        e_int_l2 = float(np.linalg.norm(n - x))
        kl_int   = _safe_kl(p, q_int)

        # Multinomial sampling baseline
        e_samp_l2 = np.empty(R, dtype=float)
        kl_samp   = np.empty(R, dtype=float)

        for r in range(R):
            ns = rng.multinomial(T, p)
            q  = ns / max(T, 1)
            e_samp_l2[r] = float(np.linalg.norm(ns - x))
            kl_samp[r]   = _safe_kl(p, q)

        def qtl(a, q): return float(np.percentile(a, q))

        rows.append({
            "E_min_L2": e_min_l2 if e_min_l2 is not None else np.nan,
            "KL_min": kl_min if kl_min is not None else np.nan,
            "ZoneID": int(z),
            "T": int(T),
            # L2 distance to fractional x
            "E_int_L2": e_int_l2,
            "E_samp_L2_mean": float(e_samp_l2.mean()),
            "E_samp_L2_p05":  qtl(e_samp_l2, 5),
            "E_samp_L2_p50":  qtl(e_samp_l2, 50),
            "E_samp_L2_p95":  qtl(e_samp_l2, 95),
            # KL divergence KL(p||q)
            "KL_int": kl_int,
            "KL_samp_mean": float(kl_samp.mean()),
            "KL_samp_p05":  qtl(kl_samp, 5),
            "KL_samp_p50":  qtl(kl_samp, 50),
            "KL_samp_p95":  qtl(kl_samp, 95),
        })

    # Save summary
    summary = pd.DataFrame(rows).sort_values(["T","ZoneID"], ascending=[False, True])
    out_csv = os.path.join(out_dir, "tmp_VariationBaseline_Summary.csv")
    summary.to_csv(out_csv, index=False)

    # Make plots
    if MAKE_VIOLIN_SUMMARY and not summary.empty:
        _poststage_make_violin_like_plots(out_dir, summary_csv="tmp_VariationBaseline_Summary.csv")

    # -------- Table 1: Error relative to fractional baseline (averaged across zones) --------
    try:
        if not summary.empty:
            l2_int_mean  = float(summary["E_int_L2"].mean())
            l2_s_mean    = float(summary["E_samp_L2_mean"].mean())
            l2_p05       = float(summary["E_samp_L2_p05"].mean())
            l2_p95       = float(summary["E_samp_L2_p95"].mean())

            kl_int_mean  = float(summary["KL_int"].mean())
            kl_s_mean    = float(summary["KL_samp_mean"].mean())
            kl_p05       = float(summary["KL_samp_p05"].mean())
            kl_p95       = float(summary["KL_samp_p95"].mean())

            tex = []
            tex.append(r"\begin{table}[ht]")
            tex.append(r"\centering")
            tex.append(r"\caption{Error relative to fractional baseline: integerized vs.\ sampling (200 draws). Values are averaged across analyzed zones.}")
            tex.append(r"\label{tab:baseline_errors}")
            tex.append(r"\begin{tabular}{lcc}")
            tex.append(r"\toprule")
            tex.append(r" & L2 distance & KL divergence \\")
            tex.append(r"\midrule")
            row_int = r"Integerized & %s & %s \\" % (_fmt_num(l2_int_mean,3), _fmt_num(kl_int_mean,3))
            tex.append(row_int)
            row_smean = r"Sampling (mean) & %s & %s \\" % (_fmt_num(l2_s_mean,3), _fmt_num(kl_s_mean,3))
            # Optional minimal row
            if "E_min_L2" in summary.columns and summary["E_min_L2"].notna().any():
                l2_min_mean = float(summary["E_min_L2"].mean())
                kl_min_mean = float(summary["KL_min"].mean()) if "KL_min" in summary.columns else float("nan")
                row_min = r"Minimal & %s & %s \\" % (_fmt_num(l2_min_mean,3), _fmt_num(kl_min_mean,3))
                tex.append(row_min)
            tex.append(row_smean)
            row_sp = r"Sampling (p05--p95) & %s -- %s & %s -- %s \\" % (_fmt_num(l2_p05,3), _fmt_num(l2_p95,3), _fmt_num(kl_p05,3), _fmt_num(kl_p95,3))
            tex.append(row_sp)
            tex.append(r"\bottomrule")
            tex.append(r"\end{tabular}")
            tex.append(r"\end{table}")
            _write_text(os.path.join(out_dir, "Table1_ErrorRelativeFraction.tex"), "\n".join(tex))
            print("[LaTeX] Wrote Table1_ErrorRelativeFraction.tex")
    except Exception as e:
        print("[LaTeX] Table1 generation skipped:", e)

    return out_csv

# ---------- θ overlap utilities ----------
def _list_dim_cols(df, zone_col="ZoneID", value_cols=("x","n","val","count")):
    skip = {zone_col, *value_cols}
    return [c for c in df.columns if c not in skip]

def _canonical_cell_id(df, dim_cols):
    return df[dim_cols].astype(str).agg("§".join, axis=1)

def _align_zone(seed_z, int_z, dim_cols, frac_col="x", int_col="n"):
    key_seed = _canonical_cell_id(seed_z, dim_cols)
    key_int  = _canonical_cell_id(int_z,  dim_cols)
    a = seed_z.assign(__key__=key_seed)[dim_cols + ["__key__", frac_col]]
    b = int_z.assign(__key__=key_int)[dim_cols + ["__key__", int_col]]
    m = pd.merge(a, b, on=["__key__"] + dim_cols, how="outer")
    m[frac_col] = m[frac_col].fillna(0.0)
    m[int_col]  = m[int_col].fillna(0.0)
    return m

def _safe_norm(v):
    s = float(v.sum())
    if s <= 0:
        return np.zeros_like(v, dtype=float), 0.0
    return (v / s), s

def _theta_overlap(p, q):
    if p.size == 0:
        return 0.0
    return float(np.minimum(p, q).sum())

def _zone_arrays(seed_z_aligned, frac_col="x", int_col="n"):
    v_p = seed_z_aligned[frac_col].to_numpy(dtype=float, copy=False)
    v_q = seed_z_aligned[int_col].to_numpy(dtype=float, copy=False)
    p, T_seed = _safe_norm(v_p)
    q_share, T_int = _safe_norm(v_q)
    return p, q_share, T_seed, T_int

def _multinomial_sample_shares(p, T, R=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if T <= 0 or p.sum() <= 0:
        return np.zeros((R, p.size), dtype=float)
    draws = rng.multinomial(n=int(T), pvals=p, size=R)
    return (draws / float(T)).astype(float)

def post_stage_overlap(
    seed_df: pd.DataFrame,
    int_df: pd.DataFrame,
    out_dir: str,
    dims: Optional[List[str]] = None,
    zone_col: str="ZoneID",
    frac_col: str="x",
    int_col: str="n",     # integer column for the main (swap-repaired) solution
    R: int=100,
    rng_seed: int=42,
    samples_df: Optional[pd.DataFrame] = None,
    minimal_df: Optional[pd.DataFrame] = None,    # optional "minimal" integer solution
    minimal_int_col: str="n",                     # column name holding counts in minimal_df
):
    os.makedirs(out_dir, exist_ok=True)
    if dims is None:
        dims = _list_dim_cols(seed_df, zone_col=zone_col, value_cols=(frac_col,int_col,"val","count"))

    zones = sorted(set(seed_df[zone_col].dropna().astype(int)).intersection(
                   set(int_df[zone_col].dropna().astype(int))))
    rng = np.random.default_rng(rng_seed)

    rows_overlap, dist_rows = [], []

    if samples_df is not None:
        df_s = samples_df.copy()
        if "rep" not in df_s.columns or "n_sample" not in df_s.columns:
            raise ValueError("samples_df needs columns: rep, n_sample")
        df_s["__key__"] = _canonical_cell_id(df_s[dims], dims)

    for z in zones:
        seed_z = seed_df.loc[seed_df[zone_col]==z, dims + [frac_col]].copy()
        int_z  = int_df.loc[int_df[zone_col]==z,  dims + [int_col]].copy()

        aligned = _align_zone(seed_z, int_z, dims, frac_col=frac_col, int_col=int_col)
        p, q_int_share, T_seed, T_int = _zone_arrays(aligned, frac_col=frac_col, int_col=int_col)
        theta_int = _theta_overlap(p, q_int_share)

        if samples_df is not None:
            sub = df_s[df_s[zone_col]==z].copy()
            if sub.empty:
                T_use = int(round(T_int if T_int>0 else T_seed))
                samp_shares = _multinomial_sample_shares(p, T_use, R=R, rng=rng)
            else:
                aligned_keys = _canonical_cell_id(aligned, dims)
                grp = sub.groupby(["rep","__key__"], as_index=False)["n_sample"].sum()
                mat, reps = [], sorted(grp["rep"].unique())
                for r in reps:
                    g = grp[grp["rep"]==r]
                    g2 = pd.merge(pd.DataFrame({"__key__": aligned_keys}), g, on="__key__", how="left")
                    counts = g2["n_sample"].fillna(0.0).to_numpy(dtype=float, copy=False)
                    _, T_r = _safe_norm(counts)
                    share = counts / T_r if T_r>0 else np.zeros_like(counts)
                    mat.append(share)
                samp_shares = np.vstack(mat) if mat else np.zeros((0, p.size))
        else:
            T_use = int(round(T_int if T_int>0 else T_seed))
            samp_shares = _multinomial_sample_shares(p, T_use, R=R, rng=rng)

        thetas_samp = [ _theta_overlap(p, s) for s in samp_shares ] if samp_shares.size else []
        if thetas_samp:
            p05 = float(np.percentile(thetas_samp, 5))
            p50 = float(np.percentile(thetas_samp, 50))
            p95 = float(np.percentile(thetas_samp, 95))
            mean = float(np.mean(thetas_samp))
        else:
            p05 = p50 = p95 = mean = float("nan")

        row = {
            zone_col: int(z),
            "theta_int": float(theta_int),
            "theta_samp_mean": mean,
            "theta_samp_p05": p05,
            "theta_samp_p50": p50,
            "theta_samp_p95": p95,
            "misalloc_int": float(1.0 - theta_int),
            "misalloc_samp_mean": float(1.0 - mean) if math.isfinite(mean) else float("nan"),
            "misalloc_samp_p05": float(1.0 - p95) if math.isfinite(p95) else float("nan"),
            "misalloc_samp_p50": float(1.0 - p50) if math.isfinite(p50) else float("nan"),
            "misalloc_samp_p95": float(1.0 - p05) if math.isfinite(p05) else float("nan"),
            "T_seed": float(T_seed),
            "T_int": float(T_int)
        }
        # Optional: minimal solution overlap
        if minimal_df is not None and minimal_int_col in minimal_df.columns:
            int_min_z = minimal_df.loc[minimal_df[zone_col]==z, dims + [minimal_int_col]].copy()
            if not int_min_z.empty:
                ali_min = _align_zone(seed_z, int_min_z.rename(columns={minimal_int_col: "n"}), dims, frac_col=frac_col, int_col="n")
                p_min, q_min_share, _, _ = _zone_arrays(ali_min, frac_col=frac_col, int_col="n")
                row["theta_min"] = float(_theta_overlap(p_min, q_min_share))

        rows_overlap.append(row)

        aligned_key = _canonical_cell_id(aligned, dims)
        dist_rows.append(pd.DataFrame({ zone_col:int(z), "CellID":aligned_key, "Method":"Baseline",    "p":p, "q":p }))
        dist_rows.append(pd.DataFrame({ zone_col:int(z), "CellID":aligned_key, "Method":"Integerized", "p":p, "q":q_int_share }))
        if thetas_samp:
            q_med = np.median(samp_shares, axis=0)
            dist_rows.append(pd.DataFrame({ zone_col:int(z), "CellID":aligned_key, "Method":"Sampled(p50)", "p":p, "q":q_med }))

    overlap_df = pd.DataFrame(rows_overlap)
    overlap_path = os.path.join(out_dir, "overlap_summary_by_zone.csv")
    overlap_df.to_csv(overlap_path, index=False)

    dist_path = None
    if dist_rows:
        dist_df = pd.concat(dist_rows, ignore_index=True)
        dist_path = os.path.join(out_dir, "overlap_distributions_long.csv")
        dist_df.to_csv(dist_path, index=False)

    # Aggregated CSV
    def _agg(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        return pd.Series({
            "mean": float(np.nanmean(s.to_numpy(dtype=float))),
            "median": float(np.nanmedian(s.to_numpy(dtype=float))),
            "p05": float(np.nanpercentile(s, 5)),
            "p95": float(np.nanpercentile(s, 95)),
        })

    table = pd.concat({
        "θ Integerized": _agg(overlap_df["theta_int"]),
        "θ Sampled (mean)": _agg(overlap_df["theta_samp_mean"]),
        "Misalloc Integerized (1-θ)": _agg(overlap_df["misalloc_int"]),
        "Misalloc Sampled (mean)": _agg(overlap_df["misalloc_samp_mean"]),
    }, axis=1).T.reset_index().rename(columns={"index":"Metric"})
    table_path = os.path.join(out_dir, "overlap_aggregated_table.csv")
    table.to_csv(table_path, index=False)

    # Violin of θ (Integerized vs Sampled median)
    fig_path = os.path.join(out_dir, "overlap_theta_violin.png")
    try:
        plt.figure(figsize=(7.5, 4.2))
        data = [
            pd.to_numeric(overlap_df["theta_int"], errors="coerce").dropna().to_numpy(dtype=float),
            pd.to_numeric(overlap_df["theta_samp_p50"], errors="coerce").dropna().to_numpy(dtype=float),
        ]
        # Include minimal θ if present
        datasets = data
        labels = ["Integerized θ", "Sampled θ (median)"]
        if "theta_min" in overlap_df.columns and overlap_df["theta_min"].notna().any():
            datasets = [
                data[0],
                pd.to_numeric(overlap_df["theta_min"], errors="coerce").dropna().to_numpy(dtype=float),
                data[1],
            ]
            labels = ["Integerized θ", "Minimal θ", "Sampled θ (median)"]
        plt.violinplot(dataset=datasets, showmeans=True, showmedians=False)
        plt.xticks(range(1, len(labels)+1), labels)
        plt.ylabel("Overlap θ = Σ min(p, q)")
        plt.title("Share of Population Allocated ‘Information-Theoretically Correct’")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
    except Exception as e:
        print("[Overlap] violin skipped:", e)
        fig_path = None

    print("[Overlap] per-zone ->", overlap_path)
    if dist_path: print("[Overlap] dists ->", dist_path)
    print("[Overlap] aggregate ->", table_path)
    if fig_path: print("[Overlap] violin ->", fig_path)

    # -------- Table 2: Aggregated overlap theta and misallocation --------
    try:
        have_min = ("theta_min" in overlap_df.columns)
        def agg(series: pd.Series):
            s = pd.to_numeric(series, errors="coerce")
            return float(np.nanmean(s)), float(np.nanmedian(s)), float(np.nanpercentile(s,5)), float(np.nanpercentile(s,95))

        rows = []
        mean, med, p05, p95 = agg(overlap_df["theta_int"])
        rows.append((r"$\theta$ Integerized", mean, med, p05, p95))
        if have_min:
            mean, med, p05, p95 = agg(overlap_df["theta_min"])
            rows.append((r"$\theta$ Minimal", mean, med, p05, p95))
        mean, med, p05, p95 = agg(overlap_df["theta_samp_mean"])
        rows.append((r"$\theta$ Sampled (mean)", mean, med, p05, p95))

        mean, med, p05, p95 = agg(1.0 - pd.to_numeric(overlap_df["theta_int"], errors="coerce"))
        rows.append((r"Misalloc Integerized ($1-\theta$)", mean, med, p05, p95))
        if have_min:
            mean, med, p05, p95 = agg(1.0 - pd.to_numeric(overlap_df["theta_min"], errors="coerce"))
            rows.append((r"Misalloc Minimal ($1-\theta$)", mean, med, p05, p95))
        mean, med, p05, p95 = agg(1.0 - pd.to_numeric(overlap_df["theta_samp_mean"], errors="coerce"))
        rows.append((r"Misalloc Sampled ($1-\theta$)", mean, med, p05, p95))

        tex = []
        tex.append(r"\begin{table}[t]")
        tex.append(r"\centering")
        tex.append(r"\caption{Aggregated overlap $\theta$ and misallocated share $1-\theta$ across zones. Values are mean and median; parentheses show 5th--95th percentiles across zones.}")
        tex.append(r"\label{tab:theta-aggregate}")
        tex.append(r"\begin{tabular}{lcc}")
        tex.append(r"\toprule")
        tex.append(r"Metric & Mean & Median \\")
        tex.append(r"\midrule")
        for label, mean, med, p05, p95 in rows:
            tex.append(rf"{label} & {_fmt_num(mean,3)} ({_fmt_num(p05,3)}--{_fmt_num(p95,3)}) & {_fmt_num(med,3)} \\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        tex.append(r"\end{table}")
        table2_tex = "\n".join(tex)
        _write_text(os.path.join(out_dir, "Table2_AggregatedOverlap.tex"), table2_tex)
        print("[LaTeX] Wrote Table2_AggregatedOverlap.tex")
    except Exception as e:
        print("[LaTeX] Table2 generation skipped:", e)

    return {
        "overlap_by_zone_csv": overlap_path,
        "distributions_long_csv": dist_path,
        "aggregated_table_csv": table_path,
        "theta_violin_png": fig_path,
    }

# ---------- Diagnostics (saved to OUT_DIR) ----------
def write_diagnostics(out_dir: str, int_df: pd.DataFrame, frac_df: pd.DataFrame, chosen_all_present: bool):
    # 1) Pick vs K_add per zone
    step1_log_path = os.path.join(out_dir, "step1_log.csv")
    step1_log = pd.read_csv(step1_log_path) if os.path.exists(step1_log_path) else pd.DataFrame(columns=["ZoneID","K_add"])
    step2_sel_path = os.path.join(out_dir, "step2_anchor_selected.csv")
    if chosen_all_present and os.path.exists(step2_sel_path):
        picked_per_zone = pd.read_csv(step2_sel_path).groupby("ZoneID")["row_idx"].count()
    else:
        picked_per_zone = pd.Series(dtype=int)
    kadd_per_zone = step1_log.set_index("ZoneID")["K_add"] if "K_add" in step1_log.columns else pd.Series(dtype=int)
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

# ---------- Pipeline (with swap-repair) ----------
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
        cons_z, _ = _harmonize_slice_totals(cons_z, ANCHOR_NAME, ANCHOR_SLICE_DIMS, HARM_ABS_TOL, HARM_REL_TOL)
        cons_z, _ = _project_onto_support(cons_z, seed_z, ANCHOR_NAME, ANCHOR_SLICE_DIMS)
        cons_z, _ = _reconcile_zone_totals_global(cons_z, ANCHOR_NAME)

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
    K_add_map = {}  # in-memory to avoid re-reading file later

    for z in zones:
        base = frac_df[frac_df["ZoneID"]==z].copy()
        cons_z = [ConstraintSpec(cs.name, cs.dims, cs.df[cs.df["ZoneID"]==z].copy(), cs.val_col)
                  for cs in constraints_proc]
        anchor = next(cz for cz in cons_z if cz.name==ANCHOR_NAME)
        zone_target = int(round(float(anchor.df[anchor.val_col].sum())))

        df_split, K_add = step1_split(base, zone_target)
        step1_logs.append({"ZoneID": z, "zone_target": zone_target, "floor_sum": int(df_split["n"].sum()), "K_add": K_add})
        step1_frames.append(df_split)
        K_add_map[int(z)] = int(K_add)

    df_step1 = pd.concat(step1_frames, ignore_index=True)
    df_step1.to_csv(os.path.join(out_dir, "step1_floor.csv"), index=False)
    pd.DataFrame(step1_logs).to_csv(os.path.join(out_dir, "step1_log.csv"), index=False)

    # --- STEP 2: anchor-conditioned PPS ---
    int_tables = []

    for z in zones:
        df_split = df_step1[df_step1["ZoneID"]==z].copy()
        cons_z = [ConstraintSpec(cs.name, cs.dims, cs.df[cs.df["ZoneID"]==z].copy(), cs.val_col)
                  for cs in constraints_proc]
        K_add = K_add_map[int(z)]
        df_after = step2_anchor_deterministic(df_split, cons_z, ANCHOR_SLICE_DIMS, K_add)   
        int_tables.append(df_after[["ZoneID", *dim_cols, "n"]])
            
    int_df = pd.concat(int_tables, ignore_index=True)
    int_out = os.path.join(out_dir, "integer_table.csv")
    int_df.to_csv(int_out, index=False)

    # --- STEP 3: swap-repair (optional) ---
    if ENABLE_SWAPS:
        repaired = []
        for z in zones:
            int_z  = int_df[int_df["ZoneID"]==z].copy()
            frac_z = frac_df[frac_df["ZoneID"]==z].copy()
            cons_z = [ConstraintSpec(cs.name, cs.dims, cs.df[cs.df["ZoneID"]==z].copy(), cs.val_col)
                      for cs in constraints_proc]
            rep_z, stats = swap_repair_zone(int_z, frac_z, cons_z,
                                            max_passes=SWAP_MAX_PASSES,
                                            max_moves_per_slice=SWAP_MAX_MOVES_PER_SLICE,
                                            ghost_x_eps=GHOST_X_EPS)
            repaired.append(rep_z)
        repaired_all = pd.concat(repaired, ignore_index=True)
    else:
        repaired_all = int_df.copy()

    repaired_out = os.path.join(out_dir, "integer_repaired.csv")
    repaired_all.to_csv(repaired_out, index=False)

    # --- Diagnostics ---
#    write_diagnostics(out_dir, repaired_all, frac_df, chosen_all_present=bool(chosen_all))

    # --- Post-stage comparisons ---
    # Try to load a minimal integer solution for comparisons
    minimal_df_runtime = None
    if INCLUDE_MINIMAL_IN_OVERLAP:
        minimal_df_runtime = _try_load_minimal_df(out_dir, default_name=MINIMAL_CSV_NAME)
    post_csv = poststage_variation_baseline(repaired_all, frac_df, out_dir, zones, R=200, seed=42, top_k_zones=20, minimal_df=minimal_df_runtime)

    # --- θ overlap (against baseline and sampled median) ---
    if MAKE_OVERLAP_VIOLIN:
        minimal_df = None
        minimal_path = os.path.join(out_dir, MINIMAL_CSV_NAME)
        if INCLUDE_MINIMAL_IN_OVERLAP and os.path.exists(minimal_path):
            minimal_df = _try_load_minimal_df(out_dir, default_name=MINIMAL_CSV_NAME)
        post_stage_overlap(
            seed_df=frac_df.rename(columns={"x":"x"}),
            int_df=repaired_all.rename(columns={"n":"n"}),
            out_dir=out_dir,
            dims=[c for c in KEY_COLS if c != "ZoneID"],
            zone_col="ZoneID",
            frac_col="x",
            int_col="n",
            R=100,
            rng_seed=42,
            samples_df=None,
            minimal_df=minimal_df,
            minimal_int_col="n"
        )

    # --- Return paths ---
    return {
        "fractional_fit_csv": frac_out,
        "integer_floor_csv": os.path.join(out_dir, "step1_floor.csv"),
        "integer_after_pps_csv": int_out,
        "integer_repaired_csv": repaired_out,
        "variation_summary_csv": post_csv
    }

# ---------- CLI ----------
# ---------- CLI ----------
if __name__ == "__main__":
    paths = run_pipeline(
        base_dir=BASE_DIR,
        out_dir=OUT_DIR,
        tol=IPF_TOL,
        max_iters=IPF_MAX_ITERS,
        verbose_ipf=VERBOSE_IPF,
    )
    print(json.dumps(paths, indent=2))

