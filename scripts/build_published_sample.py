#!/usr/bin/env python3
"""Build a small, publishable one-week sample from local full extracts."""
from __future__ import annotations

import argparse
import csv
import gzip
from datetime import date, timedelta
from pathlib import Path


def _subset(src: Path, dst: Path, selected: set[str], date_column: str, timestamp: bool = False) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with src.open("r", newline="", encoding="utf-8") as inp, gzip.open(dst, "wt", newline="", encoding="utf-8") as out:
        reader = csv.DictReader(inp)
        if reader.fieldnames is None or date_column not in reader.fieldnames:
            raise ValueError(f"{src}: missing {date_column!r}")
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            day = row[date_column][:10] if timestamp else row[date_column]
            if day in selected:
                writer.writerow(row)
                rows += 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2026-01-12", help="First operating day (YYYY-MM-DD).")
    p.add_argument("--days", type=int, default=7, help="Number of consecutive operating days.")
    p.add_argument("--source-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("data/sample"))
    args = p.parse_args()
    if args.days < 1:
        p.error("--days must be positive")
    start = date.fromisoformat(args.start)
    selected = {(start + timedelta(days=i)).isoformat() for i in range(args.days)}
    n_soll = _subset(args.source_dir / "soll_stops.csv", args.output_dir / "soll_stops.csv.gz", selected, "betriebstag")
    n_vp = _subset(args.source_dir / "vp.csv", args.output_dir / "vp.csv.gz", selected, "time", timestamp=True)
    print(f"days={min(selected)}..{max(selected)} soll_rows={n_soll} vp_rows={n_vp}")


if __name__ == "__main__":
    main()
