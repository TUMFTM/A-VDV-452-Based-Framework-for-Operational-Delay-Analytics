#!/usr/bin/env python3
"""Anonymize depot (Betriebshof) locations and names in the shipped SOLL data.

The existing `pseudonymize_data.py` replaces `unternehmen` (operator name)
with "Operator A", "Operator B", ... but depot stops in `data/soll_stops.csv`
still carry the real operator's name and abbreviation directly
(`ort_name`/`ort_ref_ort_name`/`ort_kuerzel`/`ort_ref_ort_kuerzel`, e.g.
"Betriebshof STW" / "STWM_E"), and their exact real-world coordinates.

This script:

1. Finds every depot row (`ort_name`/`ort_ref_ort_name` containing
   "Betriebshof", case-insensitive).
2. Renames the depot's name/short-code fields to match the already-shipped
   operator pseudonym for that row's `unternehmen` (e.g. "Betriebshof STW" ->
   "Operator A Depot"), preserving the existing Einfahrt/Ausfahrt ("E"/"A")
   suffix distinction.
3. Moves the depot's published coordinate 500 m along the deadhead leg
   toward the trip's other stop (every deadhead trip in this data has
   exactly two stops: the depot and one productive stop), so the exact
   depot location is never published. If the whole leg is under 500 m, the
   depot point collapses onto the other stop's coordinate instead.

Idempotent: rows already carrying a pseudonymized depot name are left alone.

    python scripts/anonymize_depots.py            # apply in place
    python scripts/anonymize_depots.py --check     # report only, no writes
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOLL_CSV = ROOT / "data" / "soll_stops.csv"

DEPOT_RE = re.compile(r"betriebshof", re.IGNORECASE)
PSEUDO_DEPOT_RE = re.compile(r"^Operator [A-Z]+ Depot")

EARTH_R_M = 6371008.8
CUT_M = 500.0


def _vdv_to_decimal(v: pd.Series) -> pd.Series:
    v = pd.to_numeric(v, errors="coerce")
    sign = (v < 0).map({True: -1.0, False: 1.0})
    v = v.abs()
    frac_sec = v % 1000
    v = (v - frac_sec) // 1000
    seconds = v % 100
    v = (v - seconds) // 100
    minutes = v % 100
    degrees = (v - minutes) // 100
    return sign * (degrees + minutes / 60 + (seconds + frac_sec / 1000) / 3600)


def _decimal_to_vdv(d: float) -> int:
    sign = -1 if d < 0 else 1
    d = abs(d)
    degrees = int(d)
    minutes_f = (d - degrees) * 60
    minutes = int(minutes_f)
    seconds_f = (minutes_f - minutes) * 60
    seconds = int(seconds_f)
    frac_sec = int(round((seconds_f - seconds) * 1000))
    if frac_sec >= 1000:  # rounding carry
        frac_sec -= 1000
        seconds += 1
    if seconds >= 60:
        seconds -= 60
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        degrees += 1
    packed = degrees * 10_000_000 + minutes * 100_000 + seconds * 1000 + frac_sec
    return sign * packed


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_M * math.asin(math.sqrt(a))


def _bearing_deg(lon1, lat1, lon2, lat2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return math.degrees(math.atan2(y, x))


def _destination(lon1, lat1, bearing_deg, dist_m):
    p1, l1 = math.radians(lat1), math.radians(lon1)
    brng = math.radians(bearing_deg)
    ang = dist_m / EARTH_R_M
    p2 = math.asin(math.sin(p1) * math.cos(ang) + math.cos(p1) * math.sin(ang) * math.cos(brng))
    l2 = l1 + math.atan2(
        math.sin(brng) * math.sin(ang) * math.cos(p1),
        math.cos(ang) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(l2), math.degrees(p2)  # lon, lat


def _operator_letter(unternehmen: str) -> str:
    m = re.match(r"^Operator ([A-Z]+)$", str(unternehmen).strip())
    if not m:
        raise ValueError(f"unternehmen {unternehmen!r} is not pseudonymized yet; "
                          f"run pseudonymize_data.py first")
    return m.group(1)


def anonymize(check_only: bool = False) -> dict:
    df = pd.read_csv(SOLL_CSV, dtype=str)
    n0 = len(df)

    is_depot = (
        df["ort_name"].str.contains(DEPOT_RE, na=False)
        | df["ort_ref_ort_name"].str.contains(DEPOT_RE, na=False)
    )
    already = df.loc[is_depot, "ort_ref_ort_name"].map(
        lambda v: bool(PSEUDO_DEPOT_RE.match(str(v)))
    )
    report: dict = {"depot_rows": int(is_depot.sum())}
    if is_depot.sum() == 0:
        report["status"] = "no depot rows found"
        return report
    if already.all():
        report["status"] = "already anonymized"
        return report

    df["stop_seq_i"] = df["stop_seq"].astype(int)
    df["lon"] = _vdv_to_decimal(df["ort_pos_laenge"])
    df["lat"] = _vdv_to_decimal(df["ort_pos_breite"])

    grp_keys = ["um_uid", "betriebstag", "frt_fid"]
    sizes = df.groupby(grp_keys)["stop_seq_i"].transform("count")
    if (sizes[is_depot] != 2).any():
        raise RuntimeError(
            "assumption violated: not every deadhead trip has exactly 2 stops; "
            "inspect before proceeding"
        )

    idx_min = df.groupby(grp_keys)["stop_seq_i"].idxmin()
    idx_max = df.groupby(grp_keys)["stop_seq_i"].idxmax()
    other_of = {}
    for lo, hi in zip(idx_min, idx_max):
        other_of[lo] = hi
        other_of[hi] = lo

    moved = 0
    max_shift, min_shift = 0.0, float("inf")
    for i in df.index[is_depot]:
        j = other_of[i]
        lon_d, lat_d = df.at[i, "lon"], df.at[i, "lat"]
        lon_o, lat_o = df.at[j, "lon"], df.at[j, "lat"]
        dist = _haversine_m(lon_d, lat_d, lon_o, lat_o)
        if dist <= CUT_M:
            new_lon, new_lat = lon_o, lat_o
            shift = dist
        else:
            brg = _bearing_deg(lon_d, lat_d, lon_o, lat_o)
            new_lon, new_lat = _destination(lon_d, lat_d, brg, CUT_M)
            shift = CUT_M
        df.at[i, "lon"], df.at[i, "lat"] = new_lon, new_lat
        max_shift, min_shift = max(max_shift, shift), min(min_shift, shift)
        moved += 1

    if not check_only:
        depot_idx = df.index[is_depot]
        df.loc[depot_idx, "ort_pos_laenge"] = df.loc[depot_idx, "lon"].map(_decimal_to_vdv).astype(str)
        df.loc[depot_idx, "ort_pos_breite"] = df.loc[depot_idx, "lat"].map(_decimal_to_vdv).astype(str)
        df.loc[depot_idx, "geom"] = [
            f"POINT ({lon} {lat})" for lon, lat in zip(df.loc[depot_idx, "lon"], df.loc[depot_idx, "lat"])
        ]

    def _suffix(name: str) -> str:
        parts = str(name).strip().split()
        return parts[-1] if parts and len(parts[-1]) == 1 and parts[-1].isalpha() else ""

    renamed = {}
    for i in df.index[is_depot]:
        letter = _operator_letter(df.at[i, "unternehmen"])
        pseudo_ref = f"Operator {letter} Depot"
        suf_name = _suffix(df.at[i, "ort_name"])
        suf_kuerzel = str(df.at[i, "ort_kuerzel"]).rsplit("_", 1)[-1] if "_" in str(df.at[i, "ort_kuerzel"]) else ""
        pseudo_name = f"{pseudo_ref} {suf_name}".strip() if suf_name else pseudo_ref
        pseudo_kuerzel_ref = f"OP{letter}"
        pseudo_kuerzel = f"{pseudo_kuerzel_ref}_{suf_kuerzel}" if suf_kuerzel else pseudo_kuerzel_ref
        renamed.setdefault(df.at[i, "ort_ref_ort_name"], pseudo_ref)
        if not check_only:
            df.at[i, "ort_ref_ort_name"] = pseudo_ref
            df.at[i, "ort_name"] = pseudo_name
            df.at[i, "ort_ref_ort_kuerzel"] = pseudo_kuerzel_ref
            df.at[i, "ort_kuerzel"] = pseudo_kuerzel
            if pd.notna(df.at[i, "ort_druckname"]):
                df.at[i, "ort_druckname"] = pseudo_name

    report.update(
        depot_stops_moved=moved,
        shift_m={"min": round(min_shift, 1), "max": round(max_shift, 1)},
        depot_names_renamed=renamed,
    )

    if not check_only:
        df = df.drop(columns=["stop_seq_i", "lon", "lat"])
        df.to_csv(SOLL_CSV, index=False)

    assert len(df) == n0, "row count changed"
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only, no writes")
    args = ap.parse_args()
    rep = anonymize(check_only=args.check)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
