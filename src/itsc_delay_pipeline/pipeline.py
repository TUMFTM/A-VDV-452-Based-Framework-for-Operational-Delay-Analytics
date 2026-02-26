from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from .config import Config, get
from .io import read_soll_stops, read_vp, read_segments_geojson
from .matching import (
    ensure_gdf,
    normalize_time_utc,
    to_metric,
    AnchorVehicleParams,
    match_vehicles_via_anchors,
    compute_arrivals_anchor_interp,
    fill_edge_gaps_two_stage,
)
from .osrm import OsrmConfig, build_soll_segments
from .viz import export_map


def run_pipeline(cfg: Config) -> dict:
    raw = cfg.raw
    local_tz = str(get(raw, "arrivals.local_tz", get(raw, "arrivals.LOCAL_TZ", "Europe/Berlin")))

    export_dir = Path(get(raw, "paths.export_dir", "export"))
    export_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load inputs ----
    soll_stops = read_soll_stops(cfg.p("soll_stops_csv"))
    vp = read_vp(cfg.p("vp_csv"))

    if "betriebstag" in soll_stops.columns:
        soll_stops = soll_stops[soll_stops["betriebstag"].astype(str) == str(cfg.tag)]

    if "um_uid" in soll_stops.columns:
        soll_stops = soll_stops[soll_stops["um_uid"].astype(str) == str(cfg.umlauf_id)]

    if len(soll_stops) == 0:
        raise ValueError(
            f"No SOLL stops after filtering. Check run.tag={cfg.tag} (betriebstag) and "
            f"run.umlauf_id={cfg.umlauf_id} (um_uid)."
        )
    # Normalize & sort
    soll_stops = ensure_gdf(soll_stops, crs="EPSG:4326")
    soll_stops["ts_soll"] = normalize_time_utc(soll_stops["ts_soll"], assume_local_if_naive=True, local_tz=local_tz)
    # 2026-02-25 12:00 Europe/Berlin
    # ---- Optional: restrict VP to the relevant time window of this umlauf/day ----
    # This prevents "VP window empty" due to unrelated days in vp.csv (and speeds things up).
    vp_gate_min = float(raw.get("vp_load", {}).get("time_gate_load_min", 10))
    tmin = soll_stops["ts_soll"].min() - pd.Timedelta(minutes=vp_gate_min)
    tmax = soll_stops["ts_soll"].max() + pd.Timedelta(minutes=vp_gate_min)

    vp = vp[(vp["ts"] >= tmin) & (vp["ts"] <= tmax)].copy()

    if len(vp) == 0:
        raise ValueError(
            f"VP empty after time-window filtering: [{tmin} .. {tmax}]. "
            "Your vp.csv likely does not contain positions for this run.tag day/time window."
        )   
    soll_stops = soll_stops.sort_values(["frt_fid", "stop_seq"]).reset_index(drop=True)

    vp = ensure_gdf(vp, crs="EPSG:4326")
    vp["ts"] = normalize_time_utc(vp["ts"], assume_local_if_naive=False, local_tz=local_tz)

    # ---- Build / load segments ----
    osrm_base = str(get(raw, "osrm.base_url", "")).strip()
    if osrm_base:
        osrm_cfg = OsrmConfig(
            base_url=osrm_base,
            profile=str(get(raw, "osrm.profile", "driving")),
            timeout_s=int(get(raw, "osrm.timeout_s", 20)),
        )
        soll_segments = build_soll_segments(soll_stops.to_crs("EPSG:4326"), osrm_cfg)
    else:
        soll_segments = read_segments_geojson(cfg.p("fallback_segments_geojson"))

    soll_segments = ensure_gdf(soll_segments, crs="EPSG:4326")

    # ---- Anchor vehicle matching (best vehicle) ----
    metric_crs = soll_stops.estimate_utm_crs()
    soll_m = soll_stops.to_crs(metric_crs)
    vp_m = vp.to_crs(metric_crs)

    av = raw.get("anchor_vehicle_match", {})
    av_params = AnchorVehicleParams(
        anchor_time_buf=timedelta(minutes=float(av.get("anchor_time_buf_min", 8))),
        anchor_dist_m=float(av.get("anchor_dist_m", 200.0)),
        k_nearest=int(av.get("k_nearest", 10)),
        w_dt_s=float(av.get("w_dt_s", 1.0)),
        w_dist_m=float(av.get("w_dist_m", 1.0)),
        dt_scale_s=float(av.get("dt_scale_s", 60.0)),
        dist_scale_m=float(av.get("dist_scale_m", 50.0)),
        max_dt_s=float(av.get("max_dt_s", 600)),
        max_dist_m=float(av.get("max_dist_m", 200.0)),
    )

    best_vehicle_auto, df_anchor, agg_vehicle = match_vehicles_via_anchors(soll_m, vp_m, av_params)

    manual_map = av.get("manual_vehicle_by_umlauf", {}) or {}
    best_vehicle_final = best_vehicle_auto
    if str(cfg.umlauf_id) in manual_map:
        best_vehicle_final = str(manual_map[str(cfg.umlauf_id)])

    if best_vehicle_final is None:
        raise ValueError("No vehicle found by anchor-matching. Provide a manual override in config.")

    # ---- Arrivals ----
    arr = raw.get("arrivals", {})
    time_gate = timedelta(minutes=float(arr.get("time_gate_min", 10)))
    stop_anchor_max_dt = timedelta(minutes=float(arr.get("stop_anchor_max_dt_min", 6)))

    arrivals = compute_arrivals_anchor_interp(
        soll_stops_wgs=soll_stops,
        soll_segments_wgs=soll_segments,
        vp_wgs=vp,
        vehicle_id=str(best_vehicle_final),
        time_gate=time_gate,
        likelihood_max_dist_m=float(arr.get("likelihood_max_dist_m", 100.0)),
        sigma_dist_m=float(arr.get("sigma_dist_m", 25.0)),
        stop_anchor_dist_m=float(arr.get("stop_anchor_dist_m", 75.0)),
        stop_anchor_max_dt=stop_anchor_max_dt,
        local_tz=local_tz,
    )

    final_stops = (
        soll_stops.merge(arrivals, on=["frt_fid", "stop_seq"], how="left")
        .rename(columns={"ts_soll": "ankunft_soll"})
    )

    # ---- Edge gap fill ----
    eg = raw.get("edge_gap_fill", {})
    only_first_last = bool(eg.get("only_first_last", True))
    stages = []
    for name, minutes, dist in eg.get("stages", [["anchor_edge_narrow", 3, 100.0], ["anchor_edge_wide", 10, 300.0]]):
        stages.append((str(name), timedelta(minutes=float(minutes)), float(dist)))

    final_stops = fill_edge_gaps_two_stage(
        final_stops_wgs=final_stops,
        vp_wgs=vp,
        vehicle_id=str(best_vehicle_final),
        stages=stages,
        only_first_last=only_first_last,
        local_tz=local_tz,
    )

    # ---- Exports ----
    tag = cfg.tag
    umlauf_id = cfg.umlauf_id

    csv_out = export_dir / f"stops_arrivals_anchor_interp_{umlauf_id}_{tag}.csv"
    geom_drop = [c for c in ["geometry", "geom"] if c in final_stops.columns]
    final_stops.drop(columns=geom_drop, errors="ignore").to_csv(csv_out, index=False)

    viz_cfg = raw.get("viz", {})
    html_out = export_dir / f"stops_arrivals_anchor_interp_{umlauf_id}_{tag}.html"
    export_map(
        final_stops_wgs=final_stops,
        soll_segments_wgs=soll_segments,
        vp_wgs=vp,
        vehicle_id=str(best_vehicle_final),
        out_html=html_out,
        max_plot_vp=int(viz_cfg.get("max_plot_vp", 40000)),
        local_tz=local_tz,
    )

    # debug outputs
    df_anchor_out = export_dir / f"anchor_candidates_{umlauf_id}_{tag}.csv"
    df_anchor.to_csv(df_anchor_out, index=False)

    agg_out = export_dir / f"anchor_vehicle_ranking_{umlauf_id}_{tag}.csv"
    agg_vehicle.to_csv(agg_out, index=False)

    return {
        "best_vehicle": str(best_vehicle_final),
        "csv": str(csv_out),
        "html": str(html_out),
        "anchors_csv": str(df_anchor_out),
        "vehicle_ranking_csv": str(agg_out),
    }
