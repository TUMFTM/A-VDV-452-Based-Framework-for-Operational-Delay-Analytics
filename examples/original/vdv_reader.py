# 2025-11-07 10:05 Europe/Berlin
"""
📄 vdv_reader.py
----------------
Hilfsmodul zum Einlesen von VDV-x10-Dateien (z. B. REC_FRT.x10, REC_UMLAUF.x10).
Funktioniert header-basiert (liest „atr;“ und „rec;“-Zeilen) und
behandelt automatisch Codierung (Windows-1252 / Latin-1).

Rückgabe:
    pd.DataFrame mit Spalten aus der 'atr;'-Zeile.
"""

from pathlib import Path
import pandas as pd

def read_vdv_file(filename: str | Path, base_dir: Path) -> pd.DataFrame:
    """Liest eine VDV-x10-Datei (mit 'atr;' und 'rec;'-Struktur) in ein DataFrame ein."""
    path = base_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {path}")

    # 1️⃣ Encoding erkennen
    with open(path, "r", encoding="latin1", errors="ignore") as f:
        header_lines = [f.readline() for _ in range(30)]
    encoding = "cp1252" if any("1252" in l for l in header_lines) else "latin1"

    # 2️⃣ Spaltennamen aus 'atr;' extrahieren
    columns = None
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            if line.startswith("atr;"):
                columns = [c.strip() for c in line.split(";")[1:]]
                break
    if not columns:
        print(f"⚠️ Keine 'atr;' Zeile in {filename}")
        return pd.DataFrame()

    # 3️⃣ 'rec;'-Zeilen lesen
    records = []
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            if line.startswith("rec;"):
                records.append([c.strip().strip('"') for c in line.split(";")[1:]])

    df = pd.DataFrame(records, columns=columns)
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)

    print(f"📂 Eingelesen: {filename} ({len(df):,} Zeilen, {len(df.columns)} Spalten, Encoding='{encoding}')")
    return df
