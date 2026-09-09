"""VDV 452 (.x10) → soll_stops.csv import.

Reads a raw VDV 452 drop (a directory of ~88 fixed-structure ``.x10`` files,
``atr;``/``rec;`` layout) and reconstructs the per-stop scheduled-timetable table
(``soll_stops.csv``) that the delay pipeline consumes — one row per
(Umlauf, Fahrt, stop) with scheduled passing time, stop attributes, operator and
segment travel time/distance.

The join runs entirely in memory (no database): REC_FRT ⋈ LID_VERLAUF for the
per-stop sequence, REC_ORT for stop attributes and coordinates, REC_FRT_HZT for
planned dwell, SEL_FZT_FELD / REC_SEL for per-segment travel time and distance,
MENGE_FZG_TYP for the vehicle type, and MENGE_TAGESART / FIRMENKALENDER for the
operating day — written to the ``soll_stops.csv`` schema the pipeline consumes.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. VDV .x10 reader (atr;/rec; structure, cp1252/latin1 auto-detect)
# --------------------------------------------------------------------------- #

_VDV_MARKERS = ("REC_UMLAUF.X10", "REC_FRT.X10", "LID_VERLAUF.X10")


def read_vdv_file(filename: str, base_dir: Path) -> pd.DataFrame:
    """Read one VDV-x10 file (``atr;`` header + ``rec;`` data) into a DataFrame.

    All values come back as stripped strings; callers coerce as needed.
    """
    path = Path(base_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"VDV file not found: {path}")

    with open(path, "r", encoding="latin1", errors="ignore") as f:
        header = [f.readline() for _ in range(40)]
    encoding = "cp1252" if any("1252" in l for l in header) else "latin1"

    columns = None
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            if line.startswith("atr;"):
                columns = [c.strip() for c in line.split(";")[1:] if c.strip() != ""]
                break
    if not columns:
        raise ValueError(f"No 'atr;' header in {path}")

    records = []
    ncol = len(columns)
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            if line.startswith("rec;"):
                vals = [c.strip().strip('"').strip() for c in line.rstrip("\n").split(";")[1:]]
                # rec lines may carry a trailing empty field; pad/trim to columns
                if len(vals) < ncol:
                    vals += [""] * (ncol - len(vals))
                elif len(vals) > ncol:
                    vals = vals[:ncol]
                records.append(vals)

    return pd.DataFrame(records, columns=columns)


def _holds_vdv_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {p.name.upper() for p in path.iterdir() if p.is_file()}
    return any(m in names for m in _VDV_MARKERS)


def resolve_vdv_dir(vdv_dir: str | Path) -> Path:
    """Return the directory that actually contains the .x10 files (accepts the
    drop dir itself or a parent holding exactly one drop)."""
    vdv_dir = Path(vdv_dir)
    if not vdv_dir.exists():
        raise FileNotFoundError(f"VDV dir does not exist: {vdv_dir}")
    if _holds_vdv_files(vdv_dir):
        return vdv_dir
    nested = sorted(p for p in vdv_dir.iterdir() if _holds_vdv_files(p))
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        raise ValueError(f"{vdv_dir} holds {len(nested)} VDV drops — name the one you mean.")
    raise FileNotFoundError(f"No VDV .x10 files under {vdv_dir}.")


# --------------------------------------------------------------------------- #
# 2. helpers
# --------------------------------------------------------------------------- #

def vdv_coord_to_decimal(value) -> pd.Series:
    """Packed VDV452 ``DDD MM SS.fff`` integer → decimal degrees.

    VDV coordinates are NOT 1e-7 degrees; they pack degrees/minutes/seconds.
    """
    v = pd.to_numeric(value, errors="coerce")
    sign = np.where(v < 0, -1.0, 1.0)
    v = v.abs()
    frac_sec = v % 1000
    v = (v - frac_sec) / 1000
    seconds = v % 100
    v = (v - seconds) / 100
    minutes = v % 100
    degrees = (v - minutes) / 100
    return sign * (degrees + minutes / 60 + (seconds + frac_sec / 1000) / 3600)


def _sec_to_uhrzeit(sec: float) -> str | None:
    """Seconds since service-day start → 'HH:MM:SS' (hours may exceed 24)."""
    if pd.isna(sec):
        return None
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


#: Final column order — must match the shipped data/soll_stops.csv schema.
SOLL_COLUMNS = [
    "um_uid", "betriebstag", "frt_fid", "uhrzeit", "basis_version", "li_lfd_nr",
    "li_nr", "str_li_var", "onr_typ_nr", "ort_nr", "znr_nr", "anr_nr",
    "einfangbereich", "li_knoten", "einsteigeverbot", "aussteigeverbot",
    "zone_wabe_nr", "kurzstrecke", "produktiv", "innerortsverbot", "bedarfshalt",
    "nicht_veroeffentlicht", "frt_start", "fgr_nr", "fzg_typ_nr", "fzg_typ_text",
    "fremdunternehmer_nr", "basis_version_ort", "onr_typ_nr_ort", "ort_name",
    "ort_ref_ort", "ort_ref_ort_typ", "ort_ref_ort_langnr", "ort_ref_ort_kuerzel",
    "ort_ref_ort_name", "zone_wabe_nr_ort", "ort_pos_laenge", "ort_pos_breite",
    "ort_pos_hoehe", "ort_richtung", "ort_druckname", "richtungswechsel",
    "ort_kuerzel", "tz_zeitzone", "hst_nr_national", "hst_nr_international",
    "frt_hzt_zeit", "unternehmen", "fremdunternehmer", "fahrzeit_sek",
    "distanz_m", "tagesart_nr", "geom", "stop_seq",
]

#: ORT_NRs treated as the operator's own depots (fremdunternehmer=false when a
#: run starts there). Operator-specific; override for other networks.
OWN_DEPOT_ORT_NRS = {"1", "101", "102"}


# --------------------------------------------------------------------------- #
# 3. main build
# --------------------------------------------------------------------------- #

def build_soll_stops(vdv_dir: str | Path, betriebstag: str | None = None) -> pd.DataFrame:
    """Reconstruct the per-stop scheduled timetable from a VDV 452 drop.

    ``betriebstag`` (YYYY-MM-DD): stamp every row with this operating day. If
    omitted, each Fahrt's day is derived from FIRMENKALENDER via its TAGESART_NR
    (first matching calendar day).
    """
    d = resolve_vdv_dir(vdv_dir)

    rec_frt = read_vdv_file("REC_FRT.X10", d)
    lid = read_vdv_file("LID_VERLAUF.X10", d)
    rec_ort = read_vdv_file("REC_ORT.X10", d)
    rec_frt_hzt = read_vdv_file("REC_FRT_HZT.X10", d)
    sel_fzt = read_vdv_file("SEL_FZT_FELD.X10", d)
    rec_sel = read_vdv_file("REC_SEL.X10", d)
    menge_fzg = read_vdv_file("MENGE_FZG_TYP.X10", d)
    firmencal = read_vdv_file("FIRMENKALENDER.X10", d)

    # betriebstag per TAGESART_NR (first calendar day), unless overridden
    if betriebstag is None:
        cal = (firmencal.sort_values("BETRIEBSTAG")
               .drop_duplicates("TAGESART_NR").set_index("TAGESART_NR")["BETRIEBSTAG"])
        tag_to_day = cal.to_dict()
    else:
        day = str(pd.to_datetime(betriebstag).date())
        tag_to_day = None

    # --- fahrten (+ vehicle type) ---
    frt = rec_frt.merge(menge_fzg[["FZG_TYP_NR", "FZG_TYP_TEXT"]], on="FZG_TYP_NR", how="left")
    frt_cols = ["LI_NR", "STR_LI_VAR", "FRT_FID", "FRT_START", "FGR_NR", "UM_UID",
                "TAGESART_NR", "FZG_TYP_NR", "FZG_TYP_TEXT", "FREMDUNTERNEHMER_NR"]
    frt = frt[frt_cols].copy()
    frt["FRT_START"] = pd.to_numeric(frt["FRT_START"], errors="coerce")

    # --- expand each Fahrt to its LID_VERLAUF stop sequence ---
    exp = frt.merge(lid, on=["LI_NR", "STR_LI_VAR"], how="inner", suffixes=("", "_LID"))
    exp["LI_LFD_NR"] = pd.to_numeric(exp["LI_LFD_NR"], errors="coerce")
    exp = exp.sort_values(["UM_UID", "FRT_START", "FRT_FID", "LI_LFD_NR"]).reset_index(drop=True)

    # --- stop attributes (suffix _ORT for collisions: BASIS_VERSION/ONR_TYP_NR/ZONE_WABE_NR) ---
    exp = exp.merge(rec_ort, on="ORT_NR", how="left", suffixes=("", "_ORT"))

    # --- planned dwell (Haltezeit) per stop ---
    rfh = rec_frt_hzt[["FRT_FID", "LI_LFD_NR", "FRT_HZT_ZEIT"]].copy()
    rfh["LI_LFD_NR"] = pd.to_numeric(rfh["LI_LFD_NR"], errors="coerce")
    exp = exp.merge(rfh, on=["FRT_FID", "LI_LFD_NR"], how="left")

    # --- segment travel time (SEL_FZT_FELD) + distance (REC_SEL) via next stop ---
    # vectorized: next stop within each Fahrt, then merge the segment tables
    exp["_next_ort"] = exp.groupby("FRT_FID")["ORT_NR"].shift(-1)
    fzt = (sel_fzt[["FGR_NR", "ORT_NR", "SEL_ZIEL", "SEL_FZT"]]
           .drop_duplicates(["FGR_NR", "ORT_NR", "SEL_ZIEL"])
           .rename(columns={"SEL_ZIEL": "_next_ort"}))
    exp = exp.merge(fzt, on=["FGR_NR", "ORT_NR", "_next_ort"], how="left")
    sel = (rec_sel[["ORT_NR", "SEL_ZIEL", "SEL_LAENGE"]]
           .drop_duplicates(["ORT_NR", "SEL_ZIEL"])
           .rename(columns={"SEL_ZIEL": "_next_ort"}))
    exp = exp.merge(sel, on=["ORT_NR", "_next_ort"], how="left")
    exp["fahrzeit_sek"] = pd.to_numeric(exp["SEL_FZT"], errors="coerce")
    exp["distanz_m"] = pd.to_numeric(exp["SEL_LAENGE"], errors="coerce")
    exp["FRT_HZT_ZEIT"] = pd.to_numeric(exp["FRT_HZT_ZEIT"], errors="coerce")

    # --- scheduled passing time: cumulative from FRT_START (fahrzeit else dwell) ---
    inc = exp["fahrzeit_sek"].fillna(exp["FRT_HZT_ZEIT"]).fillna(0.0)
    cum = inc.groupby(exp["FRT_FID"]).transform(lambda s: s.shift(1, fill_value=0.0).cumsum())
    uhrzeit_sec = (exp["FRT_START"] + cum)
    # vectorized 'HH:MM:SS' (hours may exceed 24); NaN -> None
    _sec = uhrzeit_sec.round().astype("Int64")
    _h = (_sec // 3600); _m = (_sec % 3600) // 60; _s = _sec % 60
    exp["uhrzeit"] = (_h.astype("string").str.zfill(2) + ":" +
                      _m.astype("string").str.zfill(2) + ":" +
                      _s.astype("string").str.zfill(2))
    exp.loc[_sec.isna(), "uhrzeit"] = None

    # --- betriebstag ---
    if betriebstag is None:
        exp["betriebstag"] = exp["TAGESART_NR"].astype(str).map(
            {str(k): v for k, v in tag_to_day.items()})
        exp["betriebstag"] = pd.to_datetime(exp["betriebstag"], errors="coerce").dt.date.astype(str)
    else:
        exp["betriebstag"] = day

    # --- operator per Umlauf (from the first stop of the earliest Fahrt) ---
    ort_name_by = {}
    for r in rec_ort.itertuples(index=False):
        nm = (getattr(r, "ORT_REF_ORT_NAME", "") or getattr(r, "ORT_NAME", "") or "").strip()
        ort_name_by[str(r.ORT_NR)] = nm
    first = (exp.sort_values(["FRT_START", "LI_LFD_NR"])
             .groupby("UM_UID").first().reset_index())
    um_operator = {}
    for r in first.itertuples(index=False):
        start_ort = str(r.ORT_NR)
        um_operator[str(r.UM_UID)] = (
            ort_name_by.get(start_ort, ""),
            "false" if start_ort in OWN_DEPOT_ORT_NRS else "true",
        )
    exp["unternehmen"] = exp["UM_UID"].astype(str).map(lambda u: um_operator.get(u, ("", "false"))[0])
    exp["fremdunternehmer"] = exp["UM_UID"].astype(str).map(lambda u: um_operator.get(u, ("", "false"))[1])

    # --- geometry (WKT POINT, WGS84) from packed VDV coords (vectorized) ---
    lon = vdv_coord_to_decimal(exp["ORT_POS_LAENGE"])
    lat = vdv_coord_to_decimal(exp["ORT_POS_BREITE"])
    ok = lon.notna() & lat.notna()
    exp["geom"] = ("POINT (" + lon.astype("string") + " " + lat.astype("string") + ")").where(ok, None)

    # --- within-Umlauf stop sequence (1-based, over FRT_START then LI_LFD_NR) ---
    exp["stop_seq"] = exp.groupby("UM_UID").cumcount() + 1

    # --- assemble to the shipped schema (lowercase; TAGESART_NR -> tagesart_nr) ---
    exp = exp.rename(columns={c: c.lower() for c in exp.columns})
    exp = exp.loc[:, ~exp.columns.duplicated()]          # guard against case collisions
    for col in SOLL_COLUMNS:
        if col not in exp.columns:
            exp[col] = None
    return exp[SOLL_COLUMNS].copy()


def write_soll_stops_csv(vdv_dir: str | Path, out_csv: str | Path,
                         betriebstag: str | None = None) -> Path:
    """Build soll_stops from a VDV drop and write it as CSV."""
    df = build_soll_stops(vdv_dir, betriebstag=betriebstag)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


# Developed with the support of Claude Opus 4.8 (Anthropic).
