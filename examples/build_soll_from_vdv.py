"""Example: build the scheduled-timetable CSV from a raw VDV 452 drop.

Usage:
    python examples/build_soll_from_vdv.py /path/to/VDV_drop [YYYY-MM-DD]

Equivalent CLI: `itsc-delay build-soll --vdv-dir <dir> --out data/soll_stops.csv`.
"""
from __future__ import annotations

import sys

from itsc_delay_pipeline.vdv452 import write_soll_stops_csv


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_soll_from_vdv.py <vdv_dir> [betriebstag YYYY-MM-DD]")
    vdv_dir = sys.argv[1]
    betriebstag = sys.argv[2] if len(sys.argv) > 2 else None
    out = write_soll_stops_csv(vdv_dir, "data/soll_stops.csv", betriebstag=betriebstag)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()


# Developed with the support of Claude Opus 4.8 (Anthropic).
