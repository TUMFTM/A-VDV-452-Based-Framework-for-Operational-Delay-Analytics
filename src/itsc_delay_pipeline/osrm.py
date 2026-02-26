from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import requests
import geopandas as gpd
from shapely.geometry import LineString

from .matching import ensure_gdf


@dataclass
class OsrmConfig:
    base_url: str
    profile: str = "driving"
    timeout_s: int = 20


def osrm_segments_between(p1, p2, cfg: OsrmConfig):
    url = (
        f"{cfg.base_url}/route/v1/{cfg.profile}/"
        f"{p1[0]},{p1[1]};{p2[0]},{p2[1]}"
        "?steps=true&geometries=geojson&overview=false"
    )
    r = requests.get(url, timeout=cfg.timeout_s)
    r.raise_for_status()
    data = r.json()

    segments = []
    for leg in data["routes"][0]["legs"]:
        for step in leg["steps"]:
            segments.append(
                {
                    "geometry": LineString(step["geometry"]["coordinates"]),
                    "length_m": float(step["distance"]),
                }
            )
    return segments


def build_soll_segments(soll_stops_wgs84: gpd.GeoDataFrame, cfg: OsrmConfig) -> gpd.GeoDataFrame:
    """Build route segments (LineStrings) between consecutive stops using OSRM step geometries."""
    stops = ensure_gdf(soll_stops_wgs84, crs="EPSG:4326")

    rows = []
    for frt_fid, grp in stops.sort_values(["frt_fid", "stop_seq"]).groupby("frt_fid"):
        coords = [(float(p.x), float(p.y)) for p in grp.geometry]
        if len(coords) < 2:
            continue

        cum_dist = 0.0
        edge_idx = 0
        for p1, p2 in zip(coords[:-1], coords[1:]):
            for seg in osrm_segments_between(p1, p2, cfg):
                rows.append(
                    {
                        "frt_fid": str(frt_fid),
                        "edge_idx": edge_idx,
                        "geometry": seg["geometry"],
                        "cum_start_m": cum_dist,
                        "cum_end_m": cum_dist + seg["length_m"],
                    }
                )
                cum_dist += seg["length_m"]
                edge_idx += 1

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
