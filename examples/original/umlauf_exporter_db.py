# 2025-11-13 13:45 Europe/Berlin
"""
🚌 helpers.umlauf_exporter_db (angepasst)
----------------------------------------
Exportiert vollständige Umläufe (VDV 452) direkt in eine PostgreSQL-Datenbank.

- Betriebstag wird explizit übergeben (nicht mehr aus Tagesart abgeleitet)
- Berechnet Fahr- und Haltezeiten
- ergänzt Fahrzeugtyp & Fremdunternehmerstatus
- erstellt Tabelle automatisch, falls sie nicht existiert
- hängt neue Daten an bestehende Tabelle an

Aufruf:
    from helpers.umlauf_exporter_db import export_umlauf
"""

from pathlib import Path
import pandas as pd
from datetime import timedelta
import os
import sqlalchemy
from sqlalchemy import inspect

from functools import lru_cache
import sqlalchemy

# ------------------------------------------------------------
# 🔌 Datenbankverbindung
# ------------------------------------------------------------
from functools import lru_cache
from pathlib import Path
import importlib.util

DB_STMT_TIMEOUT_MS = 30000

def _mk_engine(uri: str) -> sqlalchemy.Engine:
    return sqlalchemy.create_engine(
        uri,
        pool_pre_ping=True,
        connect_args={"options": f"-c statement_timeout={DB_STMT_TIMEOUT_MS}"}
    )


@lru_cache(maxsize=1)
def get_connection() -> sqlalchemy.Engine:
    """
    Gibt eine einzige Engine im gesamten Prozess zurück.
    Verhindert, dass hunderte Connection-Pools entstehen.
    """
    SCRIPT_DIR = Path(__file__).resolve().parent

    # 1️⃣ Umgebungsvariable DB_DSN
    dsn = os.getenv("DB_DSN")
    if dsn:
        return _mk_engine(dsn)

    # 2️⃣ Einzelvariablen
    user, pw, host, port, name = (
        os.getenv("DB_USER"),
        os.getenv("DB_PASSWORD"),
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        os.getenv("DB_NAME")
    )
    port = port or "5432"

    if all([user, pw, host, name]):
        return _mk_engine(f"postgresql://{user}:{pw}@{host}:{port}/{name}")

    # 3️⃣ Fallback: config.py
    for p in [SCRIPT_DIR, *SCRIPT_DIR.parents, Path.cwd()]:
        cfg = p / "config.py"
        if cfg.exists():
            spec = importlib.util.spec_from_file_location("config", str(cfg))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return _mk_engine(
                f"postgresql://{mod.username}:{mod.password}@{mod.url}:{getattr(mod, 'port', '5432')}/{mod.db_name}"
            )

    raise RuntimeError("❌ Keine DB-Konfiguration gefunden.")


# ------------------------------------------------------------
# 📦 Hauptfunktion: export_umlauf
# ------------------------------------------------------------
def export_umlauf(
    umlauf_id: str,
    rec_umlauf: pd.DataFrame,
    rec_frt: pd.DataFrame,
    lid_verlauf: pd.DataFrame,
    rec_ort: pd.DataFrame,
    rec_frt_hzt: pd.DataFrame,
    sel_fzt_feld: pd.DataFrame,
    rec_sel: pd.DataFrame,
    menge_tagesart: pd.DataFrame,
    menge_fzg_typ: pd.DataFrame,
    firmencal: pd.DataFrame,
    betriebstag: str | None = None  # <--- ✅ NEU: explizite Übergabe
) -> str | None:
    """
    Exportiert einen vollständigen Umlauf mit Fahrplanzeiten, Orten,
    Fahrzeugtyp und Fremdunternehmerstatus direkt in die PostgreSQL-Datenbank.
    """

    # ------------------------------------------------------------
    # 1️⃣ Betriebstag bestimmen (explizit übergeben)
    # ------------------------------------------------------------
    umlauf = rec_umlauf.loc[rec_umlauf["UM_UID"] == str(umlauf_id)]
    if umlauf.empty:
        print(f"⚠️ Umlauf {umlauf_id} nicht gefunden.")
        return None

    umlauf = umlauf.merge(menge_tagesart, on="TAGESART_NR", how="left")
    tagesart = umlauf.iloc[0]["TAGESART_NR"]

    # Wenn Betriebstag explizit übergeben wurde → übernehmen
    if betriebstag is not None:
        datum = pd.to_datetime(betriebstag).date()
    else:
        # Fallback: falls Funktion ohne Argument aufgerufen wird
        betriebstage = firmencal.loc[firmencal["TAGESART_NR"] == tagesart]
        datum = pd.to_datetime(betriebstage.iloc[0]["BETRIEBSTAG"]).date() if not betriebstage.empty else None

    if datum is None:
        print(f"⚠️ Kein Betriebstag für Umlauf {umlauf_id} gefunden.")
        return None

    # ------------------------------------------------------------
    # 2️⃣ Fahrten selektieren & Fahrzeugtyp anreichern
    # ------------------------------------------------------------
    fahrten = (
        rec_frt.loc[rec_frt["UM_UID"] == str(umlauf_id)]
        .sort_values("FRT_START", key=lambda s: s.astype(float))
    )
    fahrten = fahrten.merge(
        menge_fzg_typ[["FZG_TYP_NR", "FZG_TYP_TEXT"]],
        on="FZG_TYP_NR",
        how="left"
    )
    fahrten["FRT_START"] = fahrten["FRT_START"].astype(float)

    # ------------------------------------------------------------
    # 3️⃣ Fahrten expandieren (Stops & Orte)
    # ------------------------------------------------------------
    expanded_list = []
    for _, fahrt in fahrten.iterrows():
        verlauf = lid_verlauf.loc[
            (lid_verlauf["LI_NR"] == fahrt["LI_NR"]) &
            (lid_verlauf["STR_LI_VAR"] == fahrt["STR_LI_VAR"])
        ].copy()
        if verlauf.empty:
            continue

        verlauf["FRT_FID"] = fahrt["FRT_FID"]
        verlauf["FRT_START"] = fahrt["FRT_START"]
        verlauf["FGR_NR"] = fahrt["FGR_NR"]
        verlauf["UM_UID"] = umlauf_id
        verlauf["BETRIEBSTAG"] = datum  # ✅ richtiger Tag
        verlauf["FZG_TYP_NR"] = fahrt["FZG_TYP_NR"]
        verlauf["FZG_TYP_TEXT"] = fahrt["FZG_TYP_TEXT"]
        verlauf["FREMDUNTERNEHMER_NR"] = fahrt["FREMDUNTERNEHMER_NR"]
        expanded_list.append(verlauf)

    if not expanded_list:
        print(f"⚠️ Keine Fahrten für Umlauf {umlauf_id} gefunden.")
        return None

    lid_expanded = pd.concat(expanded_list, ignore_index=True)
    lid_expanded = lid_expanded.merge(rec_ort, on="ORT_NR", how="left", suffixes=("", "_ORT"))
    lid_expanded = lid_expanded.merge(
        rec_frt_hzt[["FRT_FID", "LI_LFD_NR", "FRT_HZT_ZEIT"]],
        on=["FRT_FID", "LI_LFD_NR"],
        how="left"
    )

    # ------------------------------------------------------------
    # Restlicher Code (Unternehmen, Fahrzeit, Bereinigung, Upload)
    # ------------------------------------------------------------
    # 🔧 Ab hier bleibt alles exakt wie in deiner bestehenden Version —
    # es wird nur der Betriebstag korrekt gesetzt.

    # [ ... unverändert weiter bis zum Upload ... ]



    # ------------------------------------------------------------
    # 4️⃣ Unternehmerlogik (erster Halt)
    # ------------------------------------------------------------
    eigene_betriebe = {"1", "101", "102"}
    ort2unternehmen = {
        str(r["ORT_NR"]): (r.get("ORT_REF_ORT_NAME") or r.get("ORT_NAME") or "").strip()
        for _, r in rec_ort.iterrows()
    }

    erste_fahrt = fahrten.sort_values("FRT_START", key=lambda s: s.astype(float)).iloc[0]
    verlauf_erste = lid_verlauf.loc[
        (lid_verlauf["LI_NR"] == erste_fahrt["LI_NR"]) &
        (lid_verlauf["STR_LI_VAR"] == erste_fahrt["STR_LI_VAR"])
    ].sort_values("LI_LFD_NR", key=lambda s: s.astype(float))

    if not verlauf_erste.empty:
        start_ort = str(verlauf_erste.iloc[0]["ORT_NR"])
        unternehmen_name = ort2unternehmen.get(start_ort, "")
        fremd_flag = "False" if start_ort in eigene_betriebe else "True"
    else:
        unternehmen_name, fremd_flag = "", "False"

    lid_expanded["UNTERNEHMEN"] = unternehmen_name
    lid_expanded["FREMDUNTERNEHMER"] = fremd_flag

    # ------------------------------------------------------------
    # 5️⃣ Fahrzeit & Distanz pro Segment
    # ------------------------------------------------------------
    def get_segment_values(row):
        try:
            next_lfd = str(int(row["LI_LFD_NR"]) + 1)
        except ValueError:
            return None, None
        same_fahrt = lid_expanded[
            (lid_expanded["FRT_FID"] == row["FRT_FID"]) &
            (lid_expanded["LI_LFD_NR"] == next_lfd)
        ]
        if same_fahrt.empty:
            return None, None
        ziel = same_fahrt.iloc[0]["ORT_NR"]
        fgr = row["FGR_NR"]
        ort = row["ORT_NR"]

        fzt = sel_fzt_feld.loc[
            (sel_fzt_feld["FGR_NR"] == fgr) &
            (sel_fzt_feld["ORT_NR"] == ort) &
            (sel_fzt_feld["SEL_ZIEL"] == ziel)
        ]
        dist = rec_sel.loc[
            (rec_sel["ORT_NR"] == ort) &
            (rec_sel["SEL_ZIEL"] == ziel)
        ]
        return (
            fzt.iloc[0]["SEL_FZT"] if not fzt.empty else None,
            dist.iloc[0]["SEL_LAENGE"] if not dist.empty else None,
        )

    lid_expanded[["FAHRZEIT_SEK", "DISTANZ_M"]] = lid_expanded.apply(
        lambda r: pd.Series(get_segment_values(r)), axis=1
    )

    # ------------------------------------------------------------
    # 6️⃣ Fahrplanzeiten berechnen
    # ------------------------------------------------------------
    def sec_to_time(sec: float) -> str:
        if pd.isna(sec):
            return None
        return str(timedelta(seconds=int(sec)))

    fahrplan = []
    keep_cols = lid_expanded.columns
    for fid, group in lid_expanded.groupby("FRT_FID", sort=False):
        group = group.copy()[keep_cols]
        start_sec = group["FRT_START"].astype(float).iloc[0]
        laufzeit = start_sec
        times = []
        for _, row in group.iterrows():
            times.append(sec_to_time(laufzeit))
            if not pd.isna(row["FAHRZEIT_SEK"]):
                laufzeit += float(row["FAHRZEIT_SEK"])
            elif not pd.isna(row["FRT_HZT_ZEIT"]):
                laufzeit += float(row["FRT_HZT_ZEIT"])
        group["UHRZEIT"] = times
        fahrplan.append(group)

    lid_final = pd.concat(fahrplan, ignore_index=True)

    # ------------------------------------------------------------
    # 🧹 Doppelte Fahrten (Ein-/Ausfahrt) bereinigen
    # ------------------------------------------------------------
    dupe_keys = ["FRT_START", "ORT_NR", "BETRIEGSTAG", "FGR_NR", "FZG_TYP_NR"]
    dupe_keys = [c for c in dupe_keys if c in lid_final.columns]
    before = len(lid_final)
    lid_final = lid_final.drop_duplicates(subset=dupe_keys, keep="first").reset_index(drop=True)
    after = len(lid_final)
    if after < before:
        print(f"🧹 {before - after} doppelte Zeilen entfernt (Ein-/Ausfahrten)")

    # 2025-11-12 13:45 Europe/Berlin
    # ------------------------------------------------------------
    # 7️⃣ Exportieren → PostgreSQL (DDL-frei, mit finalem Typen-Fix)
    # ------------------------------------------------------------
    from sqlalchemy import inspect

    engine = get_connection()
    schema = "public_transport"
    table_name = "umlaeufe_swms"

    # ✅ Prüfen, ob Tabelle existiert (DDL-frei)
    insp = inspect(engine)
    if not insp.has_table(table_name, schema=schema):
        raise RuntimeError(f"❌ Tabelle {schema}.{table_name} existiert nicht. Bitte manuell anlegen (kein CREATE erlaubt).")

    # 🧹 Pflichtspalten ergänzen
    if "BETRIEBSTAG" not in lid_final.columns:
        lid_final["BETRIEBSTAG"] = datum
    if "UM_UID" not in lid_final.columns:
        lid_final["UM_UID"] = umlauf_id
    if "TAGESART_NR" not in lid_final.columns:
        lid_final["TAGESART_NR"] = int(tagesart)

    # ❌ Geometriespalten entfernen (Trigger übernimmt später)
    for col in ["geom", "geom_wkt"]:
        if col in lid_final.columns:
            lid_final = lid_final.drop(columns=[col])

    # 🔡 Spaltennamen vereinheitlichen
    lid_final.columns = lid_final.columns.str.lower()

    # 🧹 Finales Cleaning (Typenabgleich DB-kompatibel)
    numeric_cols = [
        "li_lfd_nr", "li_nr", "str_li_var", "onr_typ_nr", "ort_nr", "znr_nr", "anr_nr",
        "einfangbereich", "li_knoten", "einsteigeverbot", "aussteigeverbot",
        "produktiv", "innerortsverbot", "nicht_veroeffentlicht",
        "fgr_nr", "fzg_typ_nr", "frt_fid",
        "ort_pos_laenge", "ort_pos_breite", "ort_pos_hoehe",
        "frt_start", "frt_hzt_zeit", "fahrzeit_sek", "distanz_m",
        "zone_wabe_nr", "zone_wabe_nr_ort", "richtungswechsel", "tagesart_nr"
    ]

    for col in lid_final.columns:
        if col in numeric_cols:
            # Kommas entfernen, leere Strings, Whitespaces und "nan" → None
            lid_final[col] = (
                lid_final[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace(r"^\s*$", None, regex=True)
                .replace("nan", None)
            )
            lid_final[col] = pd.to_numeric(lid_final[col], errors="coerce")
        else:
            lid_final[col] = (
                lid_final[col]
                .astype(str)
                .replace(r"^\s*$", None, regex=True)
                .replace("nan", None)
            )

    # 🚫 NaN → None (für SQL NULL)
    lid_final = lid_final.where(pd.notna(lid_final), None)

    # 🧠 Debug-Ausgabe
    print("\n📊 Datentypen vor Upload:")
    print(lid_final.dtypes.sort_index())

    print("\n🔍 Beispielzeilen:")
    print(lid_final.head(3).to_string(index=False))

    # ✅ Upload starten
    print(f"\n⬆️ Lade {len(lid_final)} Zeilen in {schema}.{table_name} …")

    try:
        lid_final.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists="append",
            index=False,
            chunksize=5000,
            method="multi"
        )
        print(f"✅ Umlauf {umlauf_id} → {schema}.{table_name}")
    except Exception as e:
        print(f"❌ Fehler beim Upload in {schema}.{table_name}: {e}")
        raise
