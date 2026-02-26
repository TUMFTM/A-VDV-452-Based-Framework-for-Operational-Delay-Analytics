from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict

    @property
    def tag(self) -> str:
        return str(self.raw["run"]["tag"])

    @property
    def umlauf_id(self) -> str:
        # keep as string to avoid dtype issues in joins
        return str(self.raw["run"]["umlauf_id"])

    def p(self, key: str) -> Path:
        return Path(self.raw["paths"][key])


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return Config(raw=raw)


def get(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
