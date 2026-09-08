from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt


def _geometry_from_lonlat(df: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat"):
    if lon_col in df.columns and lat_col in df.columns:
        return gpd.GeoSeries([Point(float(x), float(y)) for x, y in zip(df[lon_col], df[lat_col])], crs="EPSG:4326")
    return None


def _geometry_from_wkt(df: pd.DataFrame, wkt_col: str = "wkt"):
    if wkt_col in df.columns:
        geoms = df[wkt_col].apply(lambda s: wkt.loads(s) if isinstance(s, str) and s.strip() else None)
        return gpd.GeoSeries(geoms, crs="EPSG:4326")
    return None


def read_points_csv(
    path: str | Path,
    *,
    time_col: str,
    id_cols: list[str],
    wkt_col: str | None = None,  # <-- NEU
):
    df = pd.read_csv(path)

    # NEU: wenn wkt_col angegeben ist, nutzen wir den explizit
    if wkt_col is not None and wkt_col in df.columns:
        df["wkt"] = df[wkt_col]  # standardisiere intern auf "wkt"

    geom = _geometry_from_lonlat(df)
    if geom is None:
        geom = _geometry_from_wkt(df)
    if geom is None:
        raise ValueError(f"{path}: Provide lon/lat or wkt column.")

    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    if time_col in gdf.columns:
        gdf[time_col] = pd.to_datetime(gdf[time_col], errors="coerce")
    else:
        gdf[time_col] = pd.NaT

    if id_cols:
        for c in id_cols:
            if c in gdf.columns:
                gdf[c] = gdf[c].astype(str)

    return gdf


def read_soll_stops(path: str | Path) -> gpd.GeoDataFrame:
    """Read scheduled stops.

    Required columns: frt_fid, stop_seq, and either ts_soll OR (betriebstag + uhrzeit)
    Geometry: lon/lat or wkt
    """
    gdf = read_points_csv(path, time_col="ts_soll", id_cols=["frt_fid", "um_uid"], wkt_col="geom")

    # If ts_soll missing/empty, try to build from betriebstag + uhrzeit
    if gdf["ts_soll"].isna().all():
        if not {"betriebstag", "uhrzeit"}.issubset(set(gdf.columns)):
            raise ValueError("SOLL stops: need ts_soll or (betriebstag + uhrzeit).")
        base = pd.to_datetime(gdf["betriebstag"], errors="coerce").dt.normalize()
        # allow >24h strings like "1 day, 01:40:00"
        u = gdf["uhrzeit"].astype(str)
        td = pd.to_timedelta(u, errors="coerce")
        # if plain HH:MM:SS, to_timedelta works too
        gdf["ts_soll"] = base + td

    for c in ["frt_fid", "stop_seq"]:
        if c not in gdf.columns:
            raise ValueError(f"SOLL stops: missing '{c}'.")

    gdf["frt_fid"] = gdf["frt_fid"].astype(str)
    gdf["stop_seq"] = pd.to_numeric(gdf["stop_seq"], errors="coerce").astype("Int64")
    gdf = gdf.dropna(subset=["stop_seq"]).copy()

    return gdf


def read_vp(path: str | Path) -> gpd.GeoDataFrame:
    gdf = read_points_csv(path, time_col="time", id_cols=["vehicle_id"])
    gdf = gdf.rename(columns={"time": "ts"})
    if "vehicle_id" not in gdf.columns:
        raise ValueError("VP CSV: missing 'vehicle_id'.")
    gdf["vehicle_id"] = gdf["vehicle_id"].astype(str)
    return gdf


def read_segments_geojson(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    required = {"frt_fid", "edge_idx", "cum_start_m", "cum_end_m"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"Segments file missing columns: {sorted(missing)}")

    gdf["frt_fid"] = gdf["frt_fid"].astype(str)
    gdf["edge_idx"] = pd.to_numeric(gdf["edge_idx"], errors="coerce").astype("Int64")
    return gdf
