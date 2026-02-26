# 2026-02-26 13:00 Europe/Berlin
# src/itsc_delay_pipeline/cli.py

from __future__ import annotations

import typer

from .config import load_config
from .pipeline import run_pipeline
from .visualize_results import visualize_results

app = typer.Typer(add_completion=False)


def _normalize_tags_arg(tags: list[str] | None, tags_csv: str | None) -> list[str] | None:
    if tags and len(tags) > 0:
        return [t.strip() for t in tags if t.strip()]
    if tags_csv:
        return [t.strip() for t in tags_csv.split(",") if t.strip()]
    return None


def _get_tags_from_cfg(cfg) -> list[str]:
    run_cfg = (cfg.raw or {}).get("run", {}) if hasattr(cfg, "raw") else {}
    tags = run_cfg.get("tags")
    if isinstance(tags, list) and tags:
        return [str(t) for t in tags]
    tag = run_cfg.get("tag")
    return [str(tag)] if tag else []


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c"),
    tags: list[str] = typer.Option(None, "--tag", "-t", help="Repeatable. Example: -t 2026-01-08 -t 2026-01-15"),
    tags_csv: str = typer.Option(None, "--tags", help="Comma-separated. Example: --tags 2026-01-08,2026-01-15"),
):
    """Run the pipeline for one or multiple days (tags)."""
    cfg = load_config(config)

    override = _normalize_tags_arg(tags, tags_csv)
    tag_list = override if override is not None else _get_tags_from_cfg(cfg)
    if not tag_list:
        raise typer.BadParameter("No tag(s) provided. Use run.tags in config or pass --tag/--tags.")

    for tag in tag_list:
        cfg.raw.setdefault("run", {})
        cfg.raw["run"]["tag"] = str(tag)

        typer.echo(f"=== PIPELINE | tag={tag} ===")
        out = run_pipeline(cfg)
        for k, v in out.items():
            typer.echo(f"{k}: {v}")


@app.command(name="visualize-results")
def visualize_results_cmd(
    config: str = typer.Option(..., "--config", "-c"),
    tags: list[str] = typer.Option(None, "--tag", "-t", help="Optional override for visualize_results.tags (repeatable)."),
    tags_csv: str = typer.Option(None, "--tags", help="Optional override for visualize_results.tags (comma-separated)."),
):
    """Generate paper figures from arrivals CSVs."""
    cfg = load_config(config)

    override = _normalize_tags_arg(tags, tags_csv)
    if override is not None:
        cfg.raw.setdefault("visualize_results", {})
        cfg.raw["visualize_results"]["tags"] = override

    out = visualize_results(raw_cfg=cfg.raw)
    for k, v in out.items():
        typer.echo(f"{k}: {v}")


@app.command(name="run-all")
def run_all_cmd(
    config: str = typer.Option(..., "--config", "-c"),
    tags: list[str] = typer.Option(None, "--tag", "-t", help="Repeatable. Runs pipeline for each tag, then figures."),
    tags_csv: str = typer.Option(None, "--tags", help="Comma-separated tags override."),
):
    """Run pipeline for multiple days AND generate figures (one call)."""
    cfg = load_config(config)

    override = _normalize_tags_arg(tags, tags_csv)
    tag_list = override if override is not None else _get_tags_from_cfg(cfg)
    if not tag_list:
        raise typer.BadParameter("No tag(s) provided. Use run.tags in config or pass --tag/--tags.")

    # 1) pipeline for each day
    for tag in tag_list:
        cfg.raw.setdefault("run", {})
        cfg.raw["run"]["tag"] = str(tag)

        typer.echo(f"=== PIPELINE | tag={tag} ===")
        out1 = run_pipeline(cfg)
        for k, v in out1.items():
            typer.echo(f"{k}: {v}")

    # 2) figures across all days (ensure tags list is set)
    cfg.raw.setdefault("visualize_results", {})
    cfg.raw["visualize_results"]["tags"] = tag_list

    typer.echo("=== FIGURES ===")
    out2 = visualize_results(raw_cfg=cfg.raw)
    for k, v in out2.items():
        typer.echo(f"{k}: {v}")


def main():
    app()


if __name__ == "__main__":
    main()