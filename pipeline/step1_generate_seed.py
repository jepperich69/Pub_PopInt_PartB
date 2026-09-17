# -*- coding: utf-8 -*-
"""
Baseline-only, NumPy IPF (no pandas)
------------------------------------
Inputs in BASE_DIR:
  - in_Pop_TargetZoneAgeChildren.csv   (ZoneID0/ZoneID, AgeID/Age, NumChildID, Val/val)
  - in_Pop_TargetZoneAgeFamily.csv     (ZoneID0/ZoneID, AgeID/Age, FamilyID/FamID, Val/val)
  - in_Pop_TargetZoneAgeGender.csv     (ZoneID0/ZoneID, AgeID/Age, GenderID, Val/val)
  - in_Pop_TargetZoneAgeIncome.csv     (ZoneID0/ZoneID, AgeID/Age, IncomeID, Val/val)
  - in_Pop_TargetZoneAgeLma.csv        (ZoneID0/ZoneID, AgeID/Age, LmaID, Val/val)

Output:
  - tmp_Pop_SyntheticSeed.csv with columns:
    ZoneID,AgeID,NumChildID,FamilyID,GenderID,IncomeID,LmaID,x
"""

import os, csv, math
from pathlib import Path
import numpy as np
from collections import defaultdict

# ---------------- config ----------------
# Repo-relative paths: margins in data/, generated files in output/
ROOT     = Path(__file__).resolve().parents[1]
BASE_DIR = str(ROOT / "data")
OUT_CSV  = str(ROOT / "output" / "tmp_Pop_SyntheticSeed.csv")

IPF_TOL   = 1e-6      # loosen/tighten for speed/accuracy
IPF_MAXIT = 200
EPS       = 1e-12
VERBOSE   = False     # baseline-only: keep quiet

# Canonical order of dimensions in 6D array
DIM_ORDER = ["AgeID","NumChildID","FamID","GenderID","IncomeID","LmaID"]

# File specs: (filename, dim1, dim2)
CONS_SPECS = [
    ("in_Pop_TargetZoneAgeChildren.csv", "AgeID", "NumChildID"),
    ("in_Pop_TargetZoneAgeFamily.csv",   "AgeID", "FamID"),
    ("in_Pop_TargetZoneAgeGender.csv",   "AgeID", "GenderID"),
    ("in_Pop_TargetZoneAgeIncome.csv",   "AgeID", "IncomeID"),
    ("in_Pop_TargetZoneAgeLma.csv",      "AgeID", "LmaID"),
]

# Column aliases
ALIASES = {
    "ZoneID":   ["ZoneID0","ZoneID","zoneid","ZONEID0","ZONEID"],
    "AgeID":    ["AgeID","Age","age","AGEID","AGE"],
    "NumChildID":["NumChildID","Children","ChildID","NumChildren","numchildid","CHILDREN"],
    "FamID": ["FamilyID","FamID","familyid","famid","FAMILYID","FAMID"],
    "GenderID": ["GenderID","SexID","Gender","genderid","GENDERID","SEXID"],
    "IncomeID": ["IncomeID","IncID","incomeid","INCID","INCOMEID"],
    "LmaID":    ["LmaID","LMAID","LMA_Id","lmaid","LMA_ID"],
    "Val":      ["Val","val","VALUE","value","VAL"],
}

# -------------- tiny CSV helpers --------------
def find_col(header, aliases):
    s = {h.strip(): h.strip() for h in header}
    for a in aliases:
        if a in s: return s[a]
    # case-insensitive match
    lower = {h.lower(): h for h in header}
    for a in aliases:
        if a.lower() in lower: return lower[a.lower()]
    raise KeyError(f"None of aliases {aliases} found in columns {list(header)}")

def load_margin_rows(path, d1, d2):
    """Yield rows: (zone, v1, v2, val) using alias resolution."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        rdr = csv.DictReader(f)
        h = rdr.fieldnames or []
        c_zone = find_col(h, ALIASES["ZoneID"])
        c_d1   = find_col(h, ALIASES[d1])
        c_d2   = find_col(h, ALIASES[d2])
        c_val  = find_col(h, ALIASES["Val"])
        for row in rdr:
            try:
                z  = row[c_zone]
                v1 = row[c_d1]
                v2 = row[c_d2]
                v  = float(row[c_val])
            except Exception:
                # skip malformed rows
                continue
            yield z, v1, v2, v

# -------------- data assembly --------------
def gather_data():
    """
    Returns:
      zones -> {
        'domains': {dim: sorted list of category labels},
        'margins': list of (dims_tuple, entries) where
                   dims_tuple like ("AgeID","NumChildID"),
                   entries is list of (v1_label, v2_label, val)
      }
    """
    zones = defaultdict(lambda: {"domains": {d:set() for d in DIM_ORDER},
                                 "margins": []})
    for fname, d1, d2 in CONS_SPECS:
        path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input: {path}")
        by_zone = defaultdict(list)
        for z, v1, v2, val in load_margin_rows(path, d1, d2):
            by_zone[z].append((v1, v2, val))
        # update zone structures
        for z, entries in by_zone.items():
            # add to domains
            for v1, v2, _ in entries:
                zones[z]["domains"][d1].add(v1)
                zones[z]["domains"][d2].add(v2)
            zones[z]["margins"].append(((d1, d2), entries))
    # finalize domains as sorted lists
    for z, obj in zones.items():
        for d in DIM_ORDER:
            if not obj["domains"][d]:
                obj["domains"][d].add("0")  # ensure non-empty
            obj["domains"][d] = sorted(obj["domains"][d], key=lambda x: (str(x)))
    return zones

# -------------- IPF core (NumPy) --------------
def build_index_maps(domains):
    """Return maps label->index and sizes in DIM_ORDER."""
    idx = {}
    sizes = []
    for d in DIM_ORDER:
        labels = domains[d]
        idx[d] = {lab:i for i,lab in enumerate(labels)}
        sizes.append(len(labels))
    return idx, tuple(sizes)

def build_target_array(entries, i1, i2, size1, size2):
    """Build 2D target array from entries list of (lab1,lab2,val)."""
    T = np.zeros((size1, size2), dtype=float)
    for v1, v2, val in entries:
        j1 = i1.get(v1)
        j2 = i2.get(v2)
        if j1 is None or j2 is None:
            continue
        T[j1, j2] += float(val)
    return T

def ipf_numpy(domains, margins):
    """
    domains: {dim: [labels]}
    margins: list of ((d1,d2), entries)
    Returns 6D array x consistent with all 2D margins (within tol).
    """
    idx_map, shape = build_index_maps(domains)
    x = np.ones(shape, dtype=float)

    # prebuild target arrays and axes info
    tasks = []
    for (d1, d2), entries in margins:
        axis1 = DIM_ORDER.index(d1)
        axis2 = DIM_ORDER.index(d2)
        T = build_target_array(entries,
                               idx_map[d1], idx_map[d2],
                               shape[axis1], shape[axis2])
        tasks.append((axis1, axis2, T))

    # if a target is all zeros, keep it but avoid division explosions
    for it in range(1, IPF_MAXIT + 1):
        max_rel = 0.0
        for a1, a2, T in tasks:
            # sum over all other axes
            axes = tuple(i for i in range(x.ndim) if i not in (a1, a2))
            cur = x.sum(axis=axes)
            # ratio on 2D
            denom = np.maximum(cur, EPS)
            R = np.divide(T, denom, out=np.ones_like(T), where=denom>0)
            # broadcast multiply along a1,a2 planes
            # build slicer of shape (1,...,s1,...,s2,...) via expand_dims
            R_expand = R
            for ax in sorted(axes):
                R_expand = np.expand_dims(R_expand, axis=ax if ax < a1 else (ax if ax < a2 else ax))
            # The above is fiddly; do it more robustly with reshape:
            reshape = [1]*x.ndim
            reshape[a1] = R.shape[0]
            reshape[a2] = R.shape[1]
            x *= R.reshape(reshape)

        # convergence check (worst relative error over all margins)
        for a1, a2, T in tasks:
            axes = tuple(i for i in range(x.ndim) if i not in (a1, a2))
            cur = x.sum(axis=axes)
            num = np.abs(T - cur)
            den = np.maximum(np.abs(T), 1.0)  # safe denom
            rel = float(np.max(num/den)) if T.size else 0.0
            if rel > max_rel: max_rel = rel
        if max_rel <= IPF_TOL:
            if VERBOSE:
                print(f"[IPF] converged in {it} iters, max_rel={max_rel:.2e}")
            break
    return x, idx_map, shape

# -------------- writer --------------
def write_zone_to_csv(writer, zone, x, idx_map):
    # inverse maps
    inv = {d: [lab for lab,_i in sorted((lab,i) for lab,i in idx_map[d].items())] for d in DIM_ORDER}
    A, C, F, G, I, L = [len(inv[d]) for d in DIM_ORDER]
    # iterate all cells (fast-ish; still linear in array size)
    for ia in range(A):
        for ic in range(C):
            for iff in range(F):
                for ig in range(G):
                    for ii in range(I):
                        for il in range(L):
                            val = float(x[ia, ic, iff, ig, ii, il])
                            if val <= 0.0:
                                continue  # skip exact zeros to reduce file size (comment out to keep all)
                            writer.writerow([
                                zone,
                                inv["AgeID"][ia],
                                inv["NumChildID"][ic],
                                inv["FamID"][iff],
                                inv["GenderID"][ig],
                                inv["IncomeID"][ii],
                                inv["LmaID"][il],
                                f"{val:.12g}",
                            ])

# -------------- main --------------
def main():
    zones = gather_data()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["ZoneID","AgeID","NumChildID","FamID","GenderID","IncomeID","LmaID","x"])
        for z, obj in zones.items():
            if VERBOSE:
                print(f"Zone {z}: sizes " +
                      ", ".join(f"{d}={len(obj['domains'][d])}" for d in DIM_ORDER))
            x, idx_map, _shape = ipf_numpy(obj["domains"], obj["margins"])
            write_zone_to_csv(wr, z, x, idx_map)
    if VERBOSE:
        print(f"[OK] Wrote: {OUT_CSV}")

if __name__ == "__main__":
    main()
