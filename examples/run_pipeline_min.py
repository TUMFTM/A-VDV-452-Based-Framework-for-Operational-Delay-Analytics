"""Example: run the delay pipeline for one Umlauf/day and inspect the result.

Reads `config.yaml`, matches the vehicle, reconstructs arrivals and computes the
per-stop delay, then prints the matched vehicle (with its confidence gate) and a
short delay summary. Runs as-is on the shipped pseudonymized sample data.

Equivalent CLI: `itsc-delay run -c config.yaml --tag 2026-01-08`.
"""
from __future__ import annotations

import pandas as pd

from itsc_delay_pipeline.config import load_config
from itsc_delay_pipeline.pipeline import run_pipeline


def main() -> None:
    cfg = load_config("config.yaml")
    cfg.raw.setdefault("run", {})["tag"] = cfg.raw["run"].get("tag", "2026-01-08")

    out = run_pipeline(cfg)
    print("outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")

    arr = pd.read_csv(out["csv"], low_memory=False)
    delays = arr["delay_s"].dropna()
    print(f"\nstops with a delay: {len(delays)}  |  median {delays.median():.0f} s  |  mean {delays.mean():.0f} s")


if __name__ == "__main__":
    main()


# Developed with the support of Claude Opus 4.8 (Anthropic).
