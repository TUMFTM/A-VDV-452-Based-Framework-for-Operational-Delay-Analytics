"""Teaching/inspection figure: how a stop delay is constructed.

For a small subsample — a few consecutive stops of one Fahrt of the configured
Umlauf/day — this draws the pieces the delay reconstruction combines:

* the scheduled stop positions (SOLL) and the route segments between them,
* the raw GTFS-RT vehicle-position (VP) points of the matched vehicle nearby,
* the reconstructed arrival point that set each stop's timestamp,
* the resulting delay (``delay_s`` = actual − scheduled), annotated per stop.

This is a lightweight PNG inspection aid, deliberately distinct from the paper's
high-resolution result figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import get


def _vdv_lonlat(series_lon: pd.Series, series_lat: pd.Series) -> tuple[pd.Series, pd.Series]:
    from .vdv452 import vdv_coord_to_decimal
    return vdv_coord_to_decimal(series_lon), vdv_coord_to_decimal(series_lat)


def viz_delay(cfg, n_stops: int = 6, out_dir: str | Path | None = None) -> Path:
    """Render the delay-construction figure for the configured umlauf/day.

    Runs the pipeline for the configured tag if its arrivals CSV is missing.
    """
    raw = cfg.raw
    umlauf = str(get(raw, "run.umlauf_id"))
    tag = cfg.tag
    export_dir = Path(get(raw, "paths.export_dir", "export"))
    out_dir = Path(out_dir) if out_dir else export_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arrivals_csv = export_dir / f"stops_arrivals_anchor_interp_{umlauf}_{tag}.csv"
    if not arrivals_csv.exists():
        from .pipeline import run_pipeline
        run_pipeline(cfg)

    arr = pd.read_csv(arrivals_csv, low_memory=False)

    # choose one Fahrt with several halts that actually have a delay
    arr = arr.sort_values(["frt_start", "stop_seq"])
    have = arr.dropna(subset=["delay_s"])
    if have.empty:
        raise RuntimeError("No stops with delay_s to visualize.")
    frt = have["frt_fid"].value_counts().index[0]
    sub = arr[arr["frt_fid"] == frt].sort_values("stop_seq")
    sub = sub[sub["delay_s"].notna()].head(n_stops).copy()
    if len(sub) < 2:
        sub = arr[arr["frt_fid"] == frt].sort_values("stop_seq").head(n_stops).copy()

    slon, slat = _vdv_lonlat(sub["ort_pos_laenge"], sub["ort_pos_breite"])
    sub["s_lon"], sub["s_lat"] = slon.to_numpy(), slat.to_numpy()

    # matched-vehicle VP points inside the subsample bbox + time window
    vp_pts = None
    va_csv = export_dir / f"vehicle_assignment_{umlauf}_{tag}.csv"
    try:
        veh = str(pd.read_csv(va_csv)["vehicle_id"].iloc[0])
        vp = pd.read_csv(cfg.p("vp_csv"), low_memory=False)
        vp = vp[vp["vehicle_id"].astype(str) == veh]
        pad = 0.004
        vp = vp[(vp["lon"] >= sub["s_lon"].min() - pad) & (vp["lon"] <= sub["s_lon"].max() + pad)
                & (vp["lat"] >= sub["s_lat"].min() - pad) & (vp["lat"] <= sub["s_lat"].max() + pad)]
        vp_pts = vp
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(9, 7))
    # route segments between consecutive scheduled stops
    ax.plot(sub["s_lon"], sub["s_lat"], "-", color="#8c9bad", lw=1.6, zorder=2,
            label="route segment (SOLL)")
    # raw VP points of the matched vehicle
    if vp_pts is not None and len(vp_pts):
        ax.scatter(vp_pts["lon"], vp_pts["lat"], s=10, c="#9aa7b3", alpha=.5,
                   zorder=3, label=f"GTFS-RT VP (vehicle {veh})")
    # reconstructed arrival points (the VP anchor that set the timestamp)
    if {"anchor_ist_lon", "anchor_ist_lat"} <= set(sub.columns):
        ai = sub.dropna(subset=["anchor_ist_lon", "anchor_ist_lat"])
        ax.scatter(ai["anchor_ist_lon"], ai["anchor_ist_lat"], s=70, marker="x",
                   c="#2f6fb2", zorder=5, label="reconstructed arrival")
    # scheduled stop nodes
    ax.scatter(sub["s_lon"], sub["s_lat"], s=150, marker="s", facecolor="white",
               edgecolor="#222", linewidths=1.6, zorder=6, label="scheduled stop (SOLL)")

    # annotate delay per stop
    for r in sub.itertuples(index=False):
        d = getattr(r, "delay_s", None)
        lbl = getattr(r, "ort_name", "")
        txt = lbl if pd.isna(d) else f"{lbl}\nΔ {int(d):+d}s"
        col = "#1b7a37" if (pd.notna(d) and d <= 0) else ("#b02a37" if pd.notna(d) else "#555")
        ax.annotate(txt, (r.s_lon, r.s_lat), xytext=(7, 6), textcoords="offset points",
                    fontsize=8, color=col, fontweight="bold", zorder=7)

    ax.set_title(f"Delay construction — Umlauf {umlauf}, Fahrt {frt}, {tag}\n"
                 "delay = reconstructed arrival − scheduled", fontsize=11)
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    out = out_dir / f"delay_construction_{umlauf}_{tag}.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


# Developed with the support of Claude Opus 4.8 (Anthropic).
