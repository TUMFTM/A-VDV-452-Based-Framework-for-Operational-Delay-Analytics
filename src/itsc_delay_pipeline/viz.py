from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import folium
from folium.plugins import Fullscreen, MousePosition, MeasureControl

from .matching import normalize_time_utc


def export_map(
    *,
    final_stops_wgs: gpd.GeoDataFrame,
    soll_segments_wgs: gpd.GeoDataFrame,
    vp_wgs: gpd.GeoDataFrame,
    vehicle_id: str,
    out_html: str | Path,
    max_plot_vp: int,
    local_tz: str,
):
    out_html = Path(out_html)

    stops_wgs = final_stops_wgs.to_crs("EPSG:4326").copy()
    segs_wgs = soll_segments_wgs.to_crs("EPSG:4326").copy()
    vp_best = vp_wgs[vp_wgs["vehicle_id"] == str(vehicle_id)].to_crs("EPSG:4326").copy()

    gc = stops_wgs.geometry.name
    center_lat = float(stops_wgs[gc].y.mean())
    center_lon = float(stops_wgs[gc].x.mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)
    Fullscreen(position="topright").add_to(m)
    MousePosition().add_to(m)
    MeasureControl(position="topright", primary_length_unit="meters").add_to(m)

    # SOLL Segmente
    for _, r in segs_wgs.sort_values(["frt_fid", "edge_idx"]).iterrows():
        ls = r[segs_wgs.geometry.name]
        if ls is None or ls.is_empty:
            continue
        coords = [(lat, lon) for lon, lat in list(ls.coords)]
        folium.PolyLine(coords, weight=3, opacity=0.35, color="#777777").add_to(m)

    fg_anchor = folium.FeatureGroup(name="Stops: anchor", show=True)
    fg_interp = folium.FeatureGroup(name="Stops: anchor_interp", show=True)
    fg_traj = folium.FeatureGroup(name="Stops: traj fallback", show=False)
    fg_edge = folium.FeatureGroup(name="Stops: edge fill", show=True)
    fg_none = folium.FeatureGroup(name="Stops: no arrival", show=False)

    def _src_has(src, key):
        return isinstance(src, str) and (key in src)

    for _, r in stops_wgs.sort_values(["frt_fid", "stop_seq"]).iterrows():
        pt = r[gc]
        if pt is None or pt.is_empty:
            continue

        src = r.get("arrival_source", None)
        tip = (
            f"frt={r['frt_fid']} seq={r['stop_seq']} | "
            f"soll={r.get('ankunft_soll')} | ist={r.get('ankunft_ist')} | src={src}"
        )

        if src == "anchor":
            color = "#2ca02c"; layer = fg_anchor
        elif src == "anchor_interp":
            color = "#ff7f0e"; layer = fg_interp
        elif src in ("trajectory_interp", "trajectory_extrapol"):
            color = "#1f77b4"; layer = fg_traj
        elif _src_has(src, "anchor_edge_"):
            color = "#9467bd"; layer = fg_edge
        else:
            color = "#d62728"; layer = fg_none

        folium.CircleMarker(
            [float(pt.y), float(pt.x)],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=tip,
        ).add_to(layer)

    for fg in [fg_anchor, fg_interp, fg_traj, fg_edge, fg_none]:
        fg.add_to(m)

    # IST points sample
    vp_best = vp_best.copy()
    vp_best["ts"] = normalize_time_utc(vp_best["ts"], assume_local_if_naive=False, local_tz=local_tz)

    fg_ist = folium.FeatureGroup(name=f"IST VP (vehicle={vehicle_id}) sample", show=False)
    step = max(1, int(len(vp_best) / max_plot_vp)) if len(vp_best) > max_plot_vp else 1

    gcv = vp_best.geometry.name
    for _, r in vp_best.sort_values("ts").iloc[::step].iterrows():
        pt = r[gcv]
        if pt is None or pt.is_empty:
            continue
        folium.CircleMarker(
            [float(pt.y), float(pt.x)],
            radius=2,
            color="#9467bd",
            fill=True,
            fill_opacity=0.6,
            tooltip=f"ts={r.get('ts')}",
        ).add_to(fg_ist)

    fg_ist.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return out_html
