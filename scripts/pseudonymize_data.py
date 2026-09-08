#!/usr/bin/env python3
"""Pseudonymize the shipped sample data IN PLACE for publication.

Replaces the only two direct identifiers with deterministic, stable pseudonyms:

  data/vp.csv          vehicle_id      -> V0001, V0002, ...   (sorted unique)
  data/soll_stops.csv  unternehmen     -> "Operator A", ...   (sorted unique)

Everything else is left untouched: stop coordinates, scheduled/actual times,
geometry, trip_id / route_id / entity_id (entity_id is a per-feed message index,
not the vehicle), and `fremdunternehmer` (a true/false flag, not a company name).

The mapping preserves the many-to-one structure of the data exactly (a bijection
on the identifier space), so vehicle matching and every downstream statistic are
unchanged - only the labels differ. The real->pseudonym mapping is written to a
gitignored file and is NOT part of the published package.

Idempotent: if the files already carry pseudonyms it makes no changes.

    python scripts/pseudonymize_data.py            # pseudonymize in place
    python scripts/pseudonymize_data.py --check     # report only, no writes
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VP_CSV = ROOT / "data" / "vp.csv"
SOLL_CSV = ROOT / "data" / "soll_stops.csv"
MAP_PATH = ROOT / "data" / ".pseudonym_map.json"   # gitignored, never shipped

VEHICLE_RE = re.compile(r"^V\d{4,}$")
OPERATOR_RE = re.compile(r"^Operator [A-Z]+$")


def _operator_label(i: int) -> str:
    """0->A, 1->B, ... 25->Z, 26->AA, ... (Excel-column style)."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = string.ascii_uppercase[r] + s
    return f"Operator {s}"


def _already_pseudo(series: pd.Series, pattern: re.Pattern) -> bool:
    vals = series.dropna().astype(str)
    vals = vals[vals.str.strip().ne("") & vals.str.lower().ne("nan")]
    return len(vals) > 0 and vals.map(lambda v: bool(pattern.match(v))).all()


def pseudonymize(check_only: bool = False) -> dict:
    report: dict = {}
    mapping: dict = {"vehicle_id": {}, "unternehmen": {}}

    # ---- vp.csv : vehicle_id ----
    vp = pd.read_csv(VP_CSV, dtype=str)
    n_vp = len(vp)
    if _already_pseudo(vp["vehicle_id"], VEHICLE_RE):
        report["vehicle_id"] = "already pseudonymized"
    else:
        reals = sorted(v for v in vp["vehicle_id"].dropna().unique()
                       if str(v).strip() and str(v).lower() != "nan")
        vmap = {r: f"V{i + 1:04d}" for i, r in enumerate(reals)}
        mapping["vehicle_id"] = vmap
        report["vehicle_id"] = f"{len(vmap)} vehicles -> V0001..V{len(vmap):04d}"
        if not check_only:
            vp["vehicle_id"] = vp["vehicle_id"].map(lambda v: vmap.get(v, v))
            vp.to_csv(VP_CSV, index=False)
    assert len(vp) == n_vp, "row count changed for vp.csv"

    # ---- soll_stops.csv : unternehmen ----
    soll = pd.read_csv(SOLL_CSV, dtype=str)
    n_soll = len(soll)
    if _already_pseudo(soll["unternehmen"], OPERATOR_RE):
        report["unternehmen"] = "already pseudonymized"
    else:
        reals = sorted(v for v in soll["unternehmen"].dropna().unique()
                       if str(v).strip() and str(v).lower() != "nan")
        omap = {r: _operator_label(i) for i, r in enumerate(reals)}
        mapping["unternehmen"] = omap
        report["unternehmen"] = f"{len(omap)} operators -> " + ", ".join(omap.values())
        if not check_only:
            soll["unternehmen"] = soll["unternehmen"].map(
                lambda v: omap.get(v, v) if pd.notna(v) else v)
            soll.to_csv(SOLL_CSV, index=False)
    assert len(soll) == n_soll, "row count changed for soll_stops.csv"

    if not check_only and (mapping["vehicle_id"] or mapping["unternehmen"]):
        MAP_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
        report["mapping_written"] = str(MAP_PATH.relative_to(ROOT)) + " (gitignored)"

    report["rows"] = {"vp.csv": n_vp, "soll_stops.csv": n_soll}
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only, no writes")
    args = ap.parse_args()
    rep = pseudonymize(check_only=args.check)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
