"""Build straight stop-to-stop fallback segments without a routing service."""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString


def build(soll_csv: Path, out_geojson: Path, tags: set[str], umlauf_ids: set[str]) -> int:
    cols = ["betriebstag", "um_uid", "frt_fid", "stop_seq", "geom"]
    parts = []
    for chunk in pd.read_csv(soll_csv, usecols=cols, chunksize=200_000):
        keep = chunk["betriebstag"].astype(str).isin(tags)
        keep &= chunk["um_uid"].astype(str).isin(umlauf_ids)
        parts.append(chunk.loc[keep])
    stops = pd.concat(parts, ignore_index=True).sort_values(["betriebstag", "frt_fid", "stop_seq"])

    rows = []
    for (tag, frt), group in stops.groupby(["betriebstag", "frt_fid"], sort=False):
        points = [wkt.loads(value) for value in group["geom"]]
        if len(points) < 2:
            continue
        metric = gpd.GeoSeries([], crs="EPSG:4326")
        cumulative = 0.0
        for edge_idx, (a, b) in enumerate(zip(points[:-1], points[1:])):
            line = LineString([a, b])
            length = float(gpd.GeoSeries([line], crs="EPSG:4326").to_crs("EPSG:25832").length.iloc[0])
            rows.append({"betriebstag": str(tag), "frt_fid": str(frt), "edge_idx": edge_idx,
                         "cum_start_m": cumulative, "cum_end_m": cumulative + length,
                         "geometry": line})
            cumulative += length

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(out_geojson, driver="GeoJSON")
    print(f"wrote {len(out):,} fallback segments for {out.frt_fid.nunique():,} Fahrten: {out_geojson}")
    return len(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soll-csv", type=Path, default=Path("data/soll_stops.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/fallback/soll_segments.geojson"))
    parser.add_argument("--tags", default="2026-01-08,2026-01-15,2026-01-22,2026-01-29")
    parser.add_argument("--umlaeufe", default="1749")
    args = parser.parse_args()
    return 0 if build(args.soll_csv, args.out, set(args.tags.split(",")), set(args.umlaeufe.split(","))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
