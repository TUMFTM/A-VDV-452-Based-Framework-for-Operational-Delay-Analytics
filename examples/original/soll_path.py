# 2026-02-02 15:32 Europe/Berlin
"""
SOLL-Pfad-Generator (ohne Corridore) + SOLL-Loader Fix
------------------------------------------------------

Fix:
- Query filtert wie in DBeaver: um_uid + betriebstag::date
- betriebstag wird aus der DB gelesen (nicht überschrieben)

Enthält weiterhin:
- load_soll_haltestellen(...)
- build_soll_path_for_umlauf(...)
- visualize_soll_path(...)
"""

from pathlib import Path
from datetime import timedelta
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import folium

from helpers.umlauf_exporter_db import get_connection


# -------------------------------------------------------------------
# Hilfsfunktion: robuste SOLL-Zeit
# -------------------------------------------------------------------
def _parse_ts_soll(betriebstag, uhrzeit):
    """
    Kombiniert Betriebstag + Uhrzeit robust.

    uhrzeit kann sein:
    - 'HH:MM:SS'
    - '1 day, 1:40:00'  (String)
    - pandas.Timedelta
    - NaN
    """
    if pd.isna(uhrzeit) or pd.isna(betriebstag):
        return pd.NaT

    base = pd.Timestamp(betriebstag).normalize()

    # 1) echter Timedelta
    if isinstance(uhrzeit, pd.Timedelta):
        return base + uhrzeit

    # 2) String → Timedelta (z. B. "1 day, 1:40:00")
    if isinstance(uhrzeit, str) and "day" in uhrzeit:
        try:
            return base + pd.to_timedelta(uhrzeit)
        except Exception:
            return pd.NaT

    # 3) klassisches HH:MM:SS
    try:
        t = pd.to_datetime(uhrzeit, format="%H:%M:%S").time()
        return pd.Timestamp.combine(base, t)
    except Exception:
        return pd.NaT


# -------------------------------------------------------------------
# 1) SOLL-Haltestellen laden (FIXED)
# -------------------------------------------------------------------
def load_soll_haltestellen(tag: str, umlauf_id: str) -> gpd.GeoDataFrame:
    """
    Lädt die SOLL-Haltestellen für einen Umlauf aus public_transport.umlaeufe_swms.

    ✔ Filter: um_uid + betriebstag::date (wie DBeaver)
    ✔ Unterstützt >24h-Zeiten (Timedelta / "1 day, ...")
    ✔ Entfernt Join-Duplikate (safe keys)
    ✔ ts_soll auf Sekunden normiert
    """

    engine = get_connection()
    tag_date = pd.to_datetime(tag).date()

    sql = """
        SELECT
            um_uid,
            betriebstag::date AS betriebstag,
            frt_fid,
            li_lfd_nr,
            li_nr,
            str_li_var,
            onr_typ_nr,
            ort_nr,
            produktiv,
            fgr_nr,
            fzg_typ_nr,
            fzg_typ_text,
            unternehmen,
            ort_ref_ort_name,
            ort_pos_laenge,
            ort_pos_breite,
            frt_hzt_zeit,
            fahrzeit_sek,
            distanz_m,
            tagesart_nr,
            stop_seq,
            uhrzeit,
            geom
        FROM public_transport.umlaeufe_swms
        WHERE um_uid = %(umlauf_id)s
          AND betriebstag::date = %(tag)s
        ORDER BY frt_fid, li_lfd_nr, stop_seq;
    """

    gdf = gpd.read_postgis(
        sql,
        engine,
        geom_col="geom",
        params={"umlauf_id": str(umlauf_id), "tag": tag_date},
    )

    if gdf.empty:
        raise RuntimeError(
            f"Keine SOLL-Daten gefunden: um_uid={umlauf_id}, betriebstag={tag_date}"
        )

    # Sanity: wirklich genau ein Betriebstag
    uniq_days = pd.to_datetime(gdf["betriebstag"]).dt.date.unique()
    if len(uniq_days) != 1 or uniq_days[0] != tag_date:
        raise RuntimeError(
            f"DB liefert mehrere/andere betriebstage für um_uid={umlauf_id}: {list(uniq_days)}"
        )

    # Zeit bauen
    gdf["ts_soll"] = gdf.apply(
        lambda r: _parse_ts_soll(r["betriebstag"], r["uhrzeit"]),
        axis=1
    )
    gdf["ts_soll"] = gdf["ts_soll"].dt.floor("s")

    bad = gdf[gdf["ts_soll"].isna()][["frt_fid", "li_lfd_nr", "stop_seq", "uhrzeit", "betriebstag"]]
    if not bad.empty:
        print("⚠️ Ungültige SOLL-Zeiten gefunden (Beispiele):")
        print(bad.head(10).to_string(index=False))

    # Typen normalisieren (nicht erzwingen, falls NULLs)
    for c in ["frt_fid", "li_lfd_nr", "stop_seq"]:
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce").astype("Int64")

    # Duplikate entfernen (safe keys)
    before = len(gdf)
    dedup_keys = ["um_uid", "betriebstag", "frt_fid", "li_lfd_nr", "stop_seq", "uhrzeit"]
    dedup_keys = [k for k in dedup_keys if k in gdf.columns]
    gdf = gdf.drop_duplicates(subset=dedup_keys).reset_index(drop=True)
    after = len(gdf)

    if before != after:
        print(
            f"⚠️ load_soll_haltestellen: {before - after} Duplikate entfernt "
            f"(um_uid={umlauf_id}, betriebstag={tag_date})"
        )

    gdf = gdf.sort_values(["stop_seq"]).reset_index(drop=True)

    print(
        f"✅ SOLL geladen: rows={len(gdf)} | fahrten={gdf['frt_fid'].nunique(dropna=True)} | "
        f"li_lfd_nr={gdf['li_lfd_nr'].nunique(dropna=True)} | "
        f"ts_soll span={gdf['ts_soll'].min()} → {gdf['ts_soll'].max()}"
    )

    return gdf


# -------------------------------------------------------------------
# 2) Zeitbasierte Interpolation zwischen zwei Stopps
# -------------------------------------------------------------------
def interpolate_segment_time(A_geom, B_geom, tA, tB, dt):
    """
    Interpoliert zwischen zwei Stopps ausschließlich anhand der Soll-Uhrzeit.
    dt in Sekunden.
    """
    if pd.isna(tA) or pd.isna(tB):
        return []

    duration = (tB - tA).total_seconds()
    if duration <= 0:
        return []

    line = LineString([A_geom, B_geom])
    total_length = line.length
    steps = int(duration // dt)

    records = []
    for k in range(steps + 1):
        t = tA + timedelta(seconds=k * dt)
        frac = (t - tA).total_seconds() / duration
        geom = line.interpolate(frac * total_length)

        records.append({
            "ts_soll": t,
            "geom": geom,
            "type": "interp",
            "segment_frac": frac
        })

    return records


# -------------------------------------------------------------------
# 3) Super-Sollpfad für einen Umlauf
# -------------------------------------------------------------------
def build_soll_path_for_umlauf(tag: str, umlauf_id: str, dt: int = 5) -> gpd.GeoDataFrame:
    """
    Baut den vollständigen Sollpfad über alle Fahrten eines Umlaufs.
    dt in Sekunden.
    """
    stops = load_soll_haltestellen(tag, umlauf_id)
    records = []

    # Achtung: ursprüngliches Notebook nutzt li_lfd_nr als "Linienlauf"
    for li_lfd, group in stops.groupby("li_lfd_nr", dropna=True):
        group = group.sort_values("stop_seq").reset_index(drop=True)

        for i in range(len(group) - 1):
            A = group.loc[i]
            B = group.loc[i + 1]

            # Stop
            records.append({
                "ts_soll": A.ts_soll,
                "geom": A.geom,
                "type": "stop",
                "li_lfd_nr": li_lfd,
                "stop_seq": A.stop_seq,
                "frt_fid": A.frt_fid,
                "um_uid": A.um_uid,
                "betriebstag": A.betriebstag,
            })

            # Interpolation
            interps = interpolate_segment_time(A.geom, B.geom, A.ts_soll, B.ts_soll, dt)
            for r in interps:
                r.update({
                    "li_lfd_nr": li_lfd,
                    "stop_seq": A.stop_seq,
                    "frt_fid": A.frt_fid,
                    "um_uid": A.um_uid,
                    "betriebstag": A.betriebstag,
                })
                records.append(r)

        # letzter Stop
        last = group.iloc[-1]
        records.append({
            "ts_soll": last.ts_soll,
            "geom": last.geom,
            "type": "stop",
            "li_lfd_nr": li_lfd,
            "stop_seq": last.stop_seq,
            "frt_fid": last.frt_fid,
            "um_uid": last.um_uid,
            "betriebstag": last.betriebstag,
        })

    gdf = gpd.GeoDataFrame(records, geometry="geom", crs=stops.crs)
    return gdf.sort_values("ts_soll").reset_index(drop=True)


# -------------------------------------------------------------------
# 4) Visualisierung
# -------------------------------------------------------------------
def visualize_soll_path(gdf: gpd.GeoDataFrame, export_dir, umlauf_id):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if gdf.empty:
        raise RuntimeError("visualize_soll_path: gdf ist leer.")

    center = [gdf.geometry.iloc[0].y, gdf.geometry.iloc[0].x]
    m = folium.Map(location=center, zoom_start=13)

    coords = [(p.y, p.x) for p in gdf.geometry]
    folium.PolyLine(coords, color="blue", weight=3).add_to(m)

    for _, row in gdf.iterrows():
        color = "red" if row.get("type") == "stop" else "gray"
        pt = row.geom
        folium.CircleMarker(
            [pt.y, pt.x],
            radius=6 if row.get("type") == "stop" else 3,
            color=color,
            fill=True,
            fill_opacity=0.8,
            tooltip=f"{row.get('type')} | seq={row.get('stop_seq')} | {row.get('ts_soll')}"
        ).add_to(m)

    html_path = export_dir / f"soll_path_{umlauf_id}.html"
    m.save(str(html_path))
    return html_path
