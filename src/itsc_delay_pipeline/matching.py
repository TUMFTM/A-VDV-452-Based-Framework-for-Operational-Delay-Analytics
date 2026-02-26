from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString


def ensure_gdf(df, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if isinstance(df, gpd.GeoDataFrame):
        if df.geometry is None:
            if "geometry" in df.columns:
                df = df.set_geometry("geometry")
            elif "geom" in df.columns:
                df = df.set_geometry("geom")
            else:
                raise ValueError(f"GeoDataFrame without geometry/geom. Columns: {list(df.columns)}")
        if df.crs is None:
            df = df.set_crs(crs)
        return df

    if "geometry" in df.columns:
        return gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
    if "geom" in df.columns:
        return gpd.GeoDataFrame(df, geometry="geom", crs=crs)

    raise ValueError(f"No geometry/geom field present. Columns: {list(df.columns)}")


def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or len(gdf) == 0:
        return gdf
    gdf = ensure_gdf(gdf)
    return gdf.to_crs(gdf.estimate_utm_crs())


def normalize_time_utc(series: pd.Series, assume_local_if_naive: bool, local_tz: str) -> pd.Series:
    """Return tz-aware UTC series.

    - SOLL is often tz-naive -> interpret as local_tz -> convert to UTC
    - VP is usually already UTC -> assume_local_if_naive=False
    """
    s = pd.to_datetime(series, errors="coerce")
    tz = getattr(s.dt, "tz", None)
    if tz is None:
        if assume_local_if_naive:
            return s.dt.tz_localize(local_tz).dt.tz_convert("UTC")
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def time_window(soll_fahrt: gpd.GeoDataFrame, time_gate: timedelta) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return (
        soll_fahrt["ts_soll"].min() - time_gate,
        soll_fahrt["ts_soll"].max() + time_gate,
    )


def likelihood_match_fahrt_metric(
    soll_segments_fahrt_wgs: gpd.GeoDataFrame,
    vp_wgs: gpd.GeoDataFrame,
    vehicle_id: str,
    t_min,
    t_max,
    *,
    max_dist_m: float,
    sigma_m: float,
    local_tz: str,
) -> gpd.GeoDataFrame:
    segs = ensure_gdf(soll_segments_fahrt_wgs, crs="EPSG:4326")
    vp0 = ensure_gdf(vp_wgs, crs="EPSG:4326")

    vp0 = vp0.copy()
    vp0["ts"] = normalize_time_utc(vp0["ts"], assume_local_if_naive=False, local_tz=local_tz)

    ist = vp0[(vp0["vehicle_id"] == vehicle_id) & (vp0["ts"] >= t_min) & (vp0["ts"] <= t_max)].copy()
    if ist.empty or segs.empty:
        return gpd.GeoDataFrame(columns=["ts", "geometry", "s_hat", "confidence"], geometry="geometry", crs=vp0.crs)

    segs_m = to_metric(segs)
    metric_crs = segs_m.crs
    ist_m = ist.to_crs(metric_crs)

    sidx = segs_m.sindex
    gc_seg = segs_m.geometry.name
    gc_ist = ist_m.geometry.name

    rows = []
    for idx, r in ist_m.iterrows():
        pt: Point = r[gc_ist]
        if pt is None or pt.is_empty:
            continue

        buf = pt.buffer(max_dist_m)
        cand_idx = list(sidx.intersection(buf.bounds)) if sidx is not None else []
        if not cand_idx:
            continue

        cand = segs_m.iloc[cand_idx].copy()
        cand["dist_m"] = cand[gc_seg].distance(pt)
        cand = cand[cand["dist_m"] <= max_dist_m].copy()
        if cand.empty:
            continue

        s_list, w_list = [], []
        for _, seg in cand.iterrows():
            line: LineString = seg[gc_seg]
            proj_len = float(line.project(pt))
            proj_pt = line.interpolate(proj_len)
            dist_m = float(pt.distance(proj_pt))

            geom_len = float(line.length) if float(line.length) > 0 else 1.0
            frac = float(np.clip(proj_len / geom_len, 0.0, 1.0))
            seg_len = float(seg["cum_end_m"] - seg["cum_start_m"])
            s_abs = float(seg["cum_start_m"] + frac * seg_len)

            w = float(np.exp(-0.5 * (dist_m / sigma_m) ** 2))
            s_list.append(s_abs)
            w_list.append(w)

        if not s_list:
            continue

        s_arr = np.asarray(s_list, dtype=float)
        w_arr = np.asarray(w_list, dtype=float)
        s_hat = float((s_arr * w_arr).sum() / w_arr.sum())
        conf = float(w_arr.sum())

        rows.append(
            {
                "ts": ist.loc[idx, "ts"],
                "geometry": ist.loc[idx, ist.geometry.name],
                "s_hat": s_hat,
                "confidence": conf,
                "vp_id": ist.loc[idx, "vp_id"] if "vp_id" in ist.columns else np.nan,
            }
        )

    if not rows:
        return gpd.GeoDataFrame(columns=["ts", "geometry", "s_hat", "confidence"], geometry="geometry", crs=vp0.crs)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=vp0.crs)


def project_stop_to_route_metric(stop_point_wgs: Point, segs_fahrt_wgs: gpd.GeoDataFrame) -> dict:
    segs = ensure_gdf(segs_fahrt_wgs, crs="EPSG:4326")
    if segs.empty or stop_point_wgs is None or stop_point_wgs.is_empty:
        return {"s_stop": np.nan, "dist_m": np.nan}

    segs_m = to_metric(segs)
    metric_crs = segs_m.crs
    stop_m = gpd.GeoSeries([stop_point_wgs], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    gc = segs_m.geometry.name

    best = None
    for _, seg in segs_m.iterrows():
        line: LineString = seg[gc]
        proj_len = float(line.project(stop_m))
        proj_pt = line.interpolate(proj_len)
        dist_m = float(stop_m.distance(proj_pt))

        geom_len = float(line.length) if float(line.length) > 0 else 1.0
        frac = float(np.clip(proj_len / geom_len, 0.0, 1.0))
        seg_len = float(seg["cum_end_m"] - seg["cum_start_m"])
        s_abs = float(seg["cum_start_m"] + frac * seg_len)

        if best is None or dist_m < best["dist_m"]:
            best = {"s_stop": s_abs, "dist_m": dist_m}

    return best if best is not None else {"s_stop": np.nan, "dist_m": np.nan}


def detect_anchors_strict(
    soll_fahrt_wgs: gpd.GeoDataFrame,
    soll_stops_proj: pd.DataFrame,
    traj_wgs: gpd.GeoDataFrame,
    *,
    dist_m: float,
    max_dt: timedelta,
) -> dict:
    anchors: Dict[int, Dict[str, Any]] = {}

    if traj_wgs is None or len(traj_wgs) == 0 or soll_fahrt_wgs is None or len(soll_fahrt_wgs) == 0:
        return anchors

    metric_crs = soll_fahrt_wgs.estimate_utm_crs()
    stops_m = soll_fahrt_wgs.to_crs(metric_crs).copy()
    traj_m = traj_wgs.to_crs(metric_crs).copy()

    gc_stop = stops_m.geometry.name
    gc_traj = traj_m.geometry.name
    sidx = traj_m.sindex

    stop_geom_by_seq = stops_m.set_index("stop_seq")[gc_stop].to_dict() if "stop_seq" in stops_m.columns else {}
    gc_traj_wgs = traj_wgs.geometry.name

    for _, stop in soll_stops_proj.iterrows():
        seq = int(stop.stop_seq)
        ts_soll = stop.ts_soll

        pt = stop_geom_by_seq.get(seq, None)
        if pt is None or pt.is_empty or pd.isna(ts_soll):
            continue

        buf = pt.buffer(dist_m)
        idx = list(sidx.intersection(buf.bounds)) if sidx is not None else []
        if not idx:
            continue

        cand = traj_m.iloc[idx].copy()
        cand["dist_m"] = cand[gc_traj].distance(pt)
        cand = cand[cand["dist_m"] <= dist_m].copy()
        if cand.empty:
            continue

        cand["dt_s"] = (cand["ts"] - ts_soll).abs().dt.total_seconds()
        cand = cand[cand["dt_s"] <= max_dt.total_seconds()].copy()
        if cand.empty:
            continue

        cand = cand.sort_values(["dt_s", "dist_m"], ascending=[True, True])
        best = cand.iloc[0]
        best_idx = best.name

        pt_wgs = traj_wgs.loc[best_idx, gc_traj_wgs]
        ist_lon = float(pt_wgs.x) if pt_wgs is not None and (not pt_wgs.is_empty) else np.nan
        ist_lat = float(pt_wgs.y) if pt_wgs is not None and (not pt_wgs.is_empty) else np.nan

        anchors[seq] = {
            "ts": best["ts"],
            "s": float(best["s_hat_dir"]),
            "vp_id": best["vp_id"] if "vp_id" in best.index else np.nan,
            "ist_lon": ist_lon,
            "ist_lat": ist_lat,
        }

    return anchors


def interpolate_between_anchors(stop_seq: int, s_stop: float, anchors: dict, soll_stops_proj: pd.DataFrame):
    seqs = soll_stops_proj["stop_seq"].values
    idx = np.where(seqs == stop_seq)[0][0]

    if idx == 0 or idx == len(seqs) - 1:
        return None

    prev_seq, next_seq = seqs[idx - 1], seqs[idx + 1]
    if int(prev_seq) not in anchors or int(next_seq) not in anchors:
        return None

    a0, a1 = anchors[int(prev_seq)], anchors[int(next_seq)]
    denom = (a1["s"] - a0["s"])
    if denom == 0:
        return None

    ratio = (s_stop - a0["s"]) / denom
    ratio = np.clip(ratio, 0, 1)

    return a0["ts"] + ratio * (a1["ts"] - a0["ts"])


# PATCH: replace your current trajectory_interpolation() with this version

def trajectory_interpolation(traj: pd.DataFrame, s_stop: float):
    # robust: ensure numeric s and tz-aware timestamps
    s = pd.to_numeric(traj.get("s_hat_dir"), errors="coerce").to_numpy(dtype=float)
    ts = pd.to_datetime(traj.get("ts"), utc=True, errors="coerce")

    if len(s) == 0 or ts.isna().all():
        return None, None

    # pandas Series -> numpy datetime64[ns] -> int64 nanoseconds
    t_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")

    # keep only valid pairs
    ok = np.isfinite(s) & np.isfinite(t_ns.astype(float))
    s = s[ok]
    t_ns = t_ns[ok]

    if len(s) < 2:
        return None, None

    # IMPORTANT: np.interp needs ascending x (s)
    order = np.argsort(s)
    s = s[order]
    t_ns = t_ns[order]

    s_min, s_max = float(np.nanmin(s)), float(np.nanmax(s))

    # interpolation inside observed s-range
    if s_min <= s_stop <= s_max:
        ts_hat_ns = int(np.interp(float(s_stop), s, t_ns))
        return pd.to_datetime(ts_hat_ns, utc=True), "trajectory_interp"

    # short extrapolation at edges (bounded + sanity speed)
    # choose first two or last two points (already sorted by s)
    if s_stop < s_min:
        i0, i1 = 0, 1
    else:
        i0, i1 = -2, -1

    ds = float(s[i1] - s[i0])
    dt_s = float((t_ns[i1] - t_ns[i0]) / 1e9)

    if ds > 0 and dt_s > 0:
        v = ds / dt_s  # m/s
        if 0.5 < v < 30:
            dt_extra_s = (float(s_stop) - float(s[i0])) / v
            if abs(dt_extra_s) <= 600:
                ts_hat_ns = int(t_ns[i0] + dt_extra_s * 1e9)
                return pd.to_datetime(ts_hat_ns, utc=True), "trajectory_extrapol"

    return None, None


def _cast_vehicle_id_like(vp_df: gpd.GeoDataFrame | None, vehicle_id: str) -> str:
    if vp_df is None or "vehicle_id" not in vp_df.columns:
        return str(vehicle_id)
    return str(vehicle_id)


@dataclass
class AnchorVehicleParams:
    anchor_time_buf: timedelta
    anchor_dist_m: float
    k_nearest: int
    w_dt_s: float
    w_dist_m: float
    dt_scale_s: float
    dist_scale_m: float
    max_dt_s: float
    max_dist_m: float


def build_anchors_from_soll(soll_m: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    soll_m = soll_m.copy().sort_values(["frt_fid", "stop_seq"]).reset_index(drop=True)
    anchors = []
    gc = soll_m.geometry.name

    for frt, grp in soll_m.groupby("frt_fid"):
        g = grp.sort_values("stop_seq")
        if len(g) == 0:
            continue
        first, last = g.iloc[0], g.iloc[-1]
        anchors.append({"frt_fid": frt, "anchor_type": "start", "stop_seq": int(first["stop_seq"]), "ts_soll": first["ts_soll"], "geometry": first[gc]})
        anchors.append({"frt_fid": frt, "anchor_type": "end",   "stop_seq": int(last["stop_seq"]),  "ts_soll": last["ts_soll"],  "geometry": last[gc]})

    return gpd.GeoDataFrame(anchors, geometry="geometry", crs=soll_m.crs)


def anchor_candidates(anchor_row: pd.Series, vp_m: gpd.GeoDataFrame, sidx, params: AnchorVehicleParams) -> pd.DataFrame:
    pt = anchor_row["geometry"]
    ts = anchor_row["ts_soll"]
    if pt is None or pt.is_empty or pd.isna(ts):
        return pd.DataFrame()

    buf = pt.buffer(params.anchor_dist_m)
    idx = list(sidx.intersection(buf.bounds)) if sidx is not None else []
    if not idx:
        return pd.DataFrame()

    cand = vp_m.iloc[idx].copy()
    cand["dist_m"] = cand.geometry.distance(pt)
    cand = cand[cand["dist_m"] <= params.anchor_dist_m].copy()
    if cand.empty:
        return pd.DataFrame()

    t0 = ts - params.anchor_time_buf
    t1 = ts + params.anchor_time_buf
    cand = cand[(cand["ts"] >= t0) & (cand["ts"] <= t1)].copy()
    if cand.empty:
        return pd.DataFrame()

    cand["dt_s"] = (cand["ts"] - ts).dt.total_seconds().abs()
    cand = cand[(cand["dt_s"] <= params.max_dt_s) & (cand["dist_m"] <= params.max_dist_m)].copy()
    if cand.empty:
        return pd.DataFrame()

    cand = cand.sort_values(["dist_m", "dt_s"]).head(params.k_nearest).copy()
    return cand[[c for c in ["vehicle_id", "ts", "dist_m", "dt_s"] if c in cand.columns]]


def pick_best_vehicle_for_anchor(cand: pd.DataFrame, params: AnchorVehicleParams):
    if cand is None or len(cand) == 0:
        return None, pd.DataFrame()

    c = cand.copy()
    c["score_pt"] = (
        params.w_dt_s * (c["dt_s"] / params.dt_scale_s) +
        params.w_dist_m * (c["dist_m"] / params.dist_scale_m)
    )

    best_per_vehicle = (
        c.sort_values(["vehicle_id", "score_pt", "dt_s", "dist_m"])
        .groupby("vehicle_id", as_index=False)
        .head(1)
        .sort_values("score_pt")
        .reset_index(drop=True)
    )

    best_vehicle = best_per_vehicle.iloc[0]["vehicle_id"] if len(best_per_vehicle) else None
    return best_vehicle, best_per_vehicle


def match_vehicles_via_anchors(soll_m: gpd.GeoDataFrame, vp_m: gpd.GeoDataFrame, params: AnchorVehicleParams):
    anchors = build_anchors_from_soll(soll_m)
    sidx = vp_m.sindex

    rows = []
    for _, a in anchors.iterrows():
        cand = anchor_candidates(a, vp_m, sidx, params)
        best_vid, best_tbl = pick_best_vehicle_for_anchor(cand, params)
        rows.append({
            "frt_fid": a["frt_fid"],
            "anchor_type": a["anchor_type"],
            "stop_seq": a["stop_seq"],
            "ts_soll": a["ts_soll"],
            "best_vehicle_id": best_vid,
            "n_cand_points": int(len(cand)) if cand is not None else 0,
            "n_cand_vehicles": int(cand["vehicle_id"].nunique()) if cand is not None and len(cand) else 0,
            "best_score": float(best_tbl.iloc[0]["score_pt"]) if len(best_tbl) else np.nan,
            "best_dt_s": float(best_tbl.iloc[0]["dt_s"]) if len(best_tbl) else np.nan,
            "best_dist_m": float(best_tbl.iloc[0]["dist_m"]) if len(best_tbl) else np.nan,
        })

    df_anchor = pd.DataFrame(rows)

    agg = (
        df_anchor.dropna(subset=["best_vehicle_id"])
        .groupby("best_vehicle_id")
        .agg(
            anchor_hits=("best_vehicle_id", "size"),
            frt_covered=("frt_fid", "nunique"),
            score_mean=("best_score", "mean"),
            dt_mean_s=("best_dt_s", "mean"),
            dist_mean_m=("best_dist_m", "mean"),
        )
        .reset_index()
        .sort_values(["anchor_hits", "frt_covered"], ascending=[False, False])
    )

    best_day_vehicle = agg.iloc[0]["best_vehicle_id"] if len(agg) else None
    return best_day_vehicle, df_anchor, agg


def compute_arrivals_anchor_interp(
    soll_stops_wgs: gpd.GeoDataFrame,
    soll_segments_wgs: gpd.GeoDataFrame,
    vp_wgs: gpd.GeoDataFrame,
    vehicle_id: str,
    *,
    manual_vehicle_id: str | None = None,
    manual_vehicle_by_frt: dict | None = None,
    time_gate: timedelta,
    likelihood_max_dist_m: float,
    sigma_dist_m: float,
    stop_anchor_dist_m: float,
    stop_anchor_max_dt: timedelta,
    local_tz: str,
) -> pd.DataFrame:

    manual_vehicle_by_frt = manual_vehicle_by_frt or {}

    soll_stops0 = ensure_gdf(soll_stops_wgs, crs="EPSG:4326").copy()
    soll_segments0 = ensure_gdf(soll_segments_wgs, crs="EPSG:4326").copy()
    vp0 = ensure_gdf(vp_wgs, crs="EPSG:4326").copy()

    soll_stops0["ts_soll"] = normalize_time_utc(soll_stops0["ts_soll"], assume_local_if_naive=True, local_tz=local_tz)
    vp0["ts"] = normalize_time_utc(vp0["ts"], assume_local_if_naive=False, local_tz=local_tz)

    soll_stops0 = soll_stops0.sort_values(["frt_fid", "stop_seq"]).reset_index(drop=True)
    geom_col_soll = soll_stops0.geometry.name

    all_rows: List[dict] = []

    for frt_fid, soll_fahrt in soll_stops0.groupby("frt_fid"):
        segs_fahrt = soll_segments0[soll_segments0["frt_fid"] == str(frt_fid)].copy()
        if segs_fahrt.empty:
            continue

        vid = vehicle_id
        if str(frt_fid) in manual_vehicle_by_frt and manual_vehicle_by_frt[str(frt_fid)] is not None:
            vid = manual_vehicle_by_frt[str(frt_fid)]
        if manual_vehicle_id is not None:
            vid = manual_vehicle_id

        vid = _cast_vehicle_id_like(vp0, vid)

        proj_rows = []
        for _, row in soll_fahrt.sort_values("stop_seq").iterrows():
            pt = row[geom_col_soll]
            res = project_stop_to_route_metric(pt, segs_fahrt)
            proj_rows.append({
                "stop_seq": int(row["stop_seq"]),
                "s_stop": float(res["s_stop"]),
                "ts_soll": row["ts_soll"],
                "fahrzeit_sek": row.get("fahrzeit_sek", np.nan),
            })
        soll_stops_proj = pd.DataFrame(proj_rows).sort_values("stop_seq").reset_index(drop=True)

        t_min, t_max = time_window(soll_fahrt, time_gate)
        ist_on_route = likelihood_match_fahrt_metric(
            segs_fahrt, vp0, str(vid), t_min, t_max,
            max_dist_m=likelihood_max_dist_m,
            sigma_m=sigma_dist_m,
            local_tz=local_tz,
        )
        if ist_on_route.empty:
            continue

        traj = ist_on_route.sort_values("s_hat").reset_index(drop=True)
        traj["s_hat_dir"] = traj["s_hat"].cummax()

        anchors = detect_anchors_strict(
            soll_fahrt_wgs=soll_fahrt,
            soll_stops_proj=soll_stops_proj,
            traj_wgs=traj,
            dist_m=stop_anchor_dist_m,
            max_dt=stop_anchor_max_dt,
        )

        for _, stop in soll_stops_proj.iterrows():
            stop_seq = int(stop["stop_seq"])
            s_stop = float(stop["s_stop"])

            if stop_seq in anchors:
                ts_hat, src = anchors[stop_seq]["ts"], "anchor"
                anchor_ist_lon = anchors[stop_seq].get("ist_lon", np.nan)
                anchor_ist_lat = anchors[stop_seq].get("ist_lat", np.nan)
                anchor_vp_id = anchors[stop_seq].get("vp_id", pd.NA)
            else:
                ts_hat = interpolate_between_anchors(stop_seq, s_stop, anchors, soll_stops_proj)
                src = "anchor_interp" if ts_hat is not None else None
                anchor_ist_lon, anchor_ist_lat, anchor_vp_id = np.nan, np.nan, pd.NA

            if ts_hat is None:
                ts_hat, src2 = trajectory_interpolation(traj, s_stop)
                src = src2
                anchor_ist_lon, anchor_ist_lat, anchor_vp_id = np.nan, np.nan, pd.NA

            all_rows.append({
                "frt_fid": str(frt_fid),
                "stop_seq": stop_seq,
                "ankunft_ist": ts_hat,
                "arrival_source": src,
                "vehicle_id": str(vid),
                "anchor_ist_lon": anchor_ist_lon,
                "anchor_ist_lat": anchor_ist_lat,
                "anchor_vp_id": anchor_vp_id,
            })

    return pd.DataFrame(all_rows)


def fill_edge_gaps_two_stage(
    final_stops_wgs: gpd.GeoDataFrame,
    vp_wgs: gpd.GeoDataFrame,
    vehicle_id: str,
    *,
    stages: list[tuple[str, timedelta, float]],
    only_first_last: bool,
    local_tz: str,
) -> gpd.GeoDataFrame:

    final_stops = ensure_gdf(final_stops_wgs, crs="EPSG:4326").copy()
    vp0 = ensure_gdf(vp_wgs, crs="EPSG:4326").copy()

    final_stops["ankunft_soll"] = normalize_time_utc(final_stops["ankunft_soll"], assume_local_if_naive=True, local_tz=local_tz)
    final_stops["ankunft_ist"] = normalize_time_utc(final_stops["ankunft_ist"], assume_local_if_naive=False, local_tz=local_tz)
    vp0["ts"] = normalize_time_utc(vp0["ts"], assume_local_if_naive=False, local_tz=local_tz)

    vp0 = vp0[vp0["vehicle_id"] == str(vehicle_id)].copy()
    if vp0.empty:
        return final_stops

    metric_crs = final_stops.estimate_utm_crs()
    stops_m = final_stops.to_crs(metric_crs)
    vp_m = vp0.to_crs(metric_crs)

    sidx = vp_m.sindex
    gc_stop = stops_m.geometry.name
    gc_vp = vp_m.geometry.name

    if only_first_last:
        edge_idx = []
        for frt, grp in final_stops.groupby("frt_fid"):
            g = grp.sort_values("stop_seq")
            if len(g) == 0:
                continue
            edge_idx.append(g.index[0])
            edge_idx.append(g.index[-1])
        edge_mask = final_stops.index.isin(edge_idx)
    else:
        edge_mask = np.ones(len(final_stops), dtype=bool)

    gap_mask = final_stops["ankunft_ist"].isna() & edge_mask
    gaps = final_stops.loc[gap_mask].copy()
    if gaps.empty:
        return final_stops

    for idx, row in gaps.iterrows():
        ts_soll = row["ankunft_soll"]
        if pd.isna(ts_soll):
            continue

        pt = stops_m.loc[idx, gc_stop]
        if pt is None or pt.is_empty:
            continue

        for stage_name, time_buf, dist_m in stages:
            t0 = ts_soll - time_buf
            t1 = ts_soll + time_buf

            buf = pt.buffer(dist_m)
            cand_idx = list(sidx.intersection(buf.bounds)) if sidx is not None else []
            if not cand_idx:
                continue

            cand = vp_m.iloc[cand_idx].copy()
            cand["dist_m"] = cand[gc_vp].distance(pt)
            cand = cand[cand["dist_m"] <= dist_m].copy()
            if cand.empty:
                continue

            cand = cand[(cand["ts"] >= t0) & (cand["ts"] <= t1)].copy()
            if cand.empty:
                continue

            cand["dt_s"] = (cand["ts"] - ts_soll).abs().dt.total_seconds()
            cand = cand.sort_values(["dt_s", "dist_m"])
            best = cand.iloc[0]

            final_stops.loc[idx, "ankunft_ist"] = best["ts"]
            prev_src = final_stops.loc[idx, "arrival_source"]
            final_stops.loc[idx, "arrival_source"] = stage_name if pd.isna(prev_src) else f"{prev_src}|{stage_name}"
            break

    return final_stops
