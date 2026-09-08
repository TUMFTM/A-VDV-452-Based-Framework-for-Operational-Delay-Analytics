# 2026-02-03 11:02 Europe/Berlin

from datetime import timedelta
import pandas as pd
import geopandas as gpd
from helpers.umlauf_exporter_db import get_connection


def load_vp_for_umlauf(
    tag: str,
    ts_start,
    ts_end=None,
    source: str = "muenster",
    vehicle_id: str | None = None,
    statement_timeout_s: int = 600,
):
    """
    Lädt Vehicle Positions aus public_transport.gtfs_rt_vp_raw.

    Performance-Fix:
    - KEIN time::date Filter (Index-Killer)
    - clamp ts_start/ts_end auf [tag 00:00, tag+1d) in UTC
    - optional vehicle_id Filter
    - liefert vp_id (DB PK) mit
    """

    # Default window
    if ts_end is None:
        ts_end = ts_start + timedelta(hours=6)

    # --- clamp auf den Betriebstag (UTC) ---
    day_start = pd.Timestamp(tag).tz_localize("UTC")
    day_end = day_start + timedelta(days=1)

    ts_start = pd.Timestamp(ts_start)
    ts_end = pd.Timestamp(ts_end)

    if ts_start < day_start:
        ts_start = day_start
    if ts_end > day_end:
        ts_end = day_end

    # Guard: leeres Fenster vermeiden
    if ts_end <= ts_start:
        return gpd.GeoDataFrame(columns=["vp_id", "vehicle_id", "ts", "geometry"], geometry="geometry", crs="EPSG:4326")

    sql = """
        SELECT
            id AS vp_id,
            vehicle_id,
            time AS ts,
            geometry
        FROM public_transport.gtfs_rt_vp_raw
        WHERE source = %(source)s
          AND time >= %(ts_start)s
          AND time <  %(ts_end)s
    """

    params = {
        "source": source,
        "ts_start": ts_start,
        "ts_end": ts_end,
    }

    if vehicle_id is not None:
        sql += " AND vehicle_id = %(vehicle_id)s"
        params["vehicle_id"] = vehicle_id

    sql += " ORDER BY time, id;"

    engine = get_connection()
    with engine.connect() as conn:
        # wichtig: konsistente TZ, sonst kann "tag" vs. "time" verwirren
        conn.exec_driver_sql("SET TIME ZONE 'UTC';")
        conn.exec_driver_sql(f"SET statement_timeout TO '{int(statement_timeout_s)}s';")
        vp = gpd.read_postgis(sql, conn, params=params, geom_col="geometry")

    return vp
