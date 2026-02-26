"""
src/itsc_delay_pipeline/visualize_results.py

Generates paper-style plots from arrivals CSVs.

Paper figures (single files):
1) Boxplots: absolute delay by scheduled hour (multi-day, QC-filtered).
2) Split violins: scheduled vs. actual layover times by propulsion type (multi-day, QC-filtered).
3) Heatmap: median signed delay along ONE selected vehicle schedule (um_uid) across days (trip-gap separators).

Batch exports (multiple files, based on the selected umlauf_id as seed):
4) For EACH unique recurring trip signature AND EACH Fahrzeitgruppe (FGR) with sufficient occurrences:
   - Fig4: deviation from schedule along trip progression with buffer-event markers.
   - Fig5: actual travel time curves (cumulative IST), with optional delay cap.

Called by CLI:
    itsc-delay visualize-results -c config.yaml
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# seaborn is used for heatmap + split violin
import seaborn as sns


# -------------------------
# Helpers: config access
# -------------------------
def _get(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# -------------------------
# Time parsing helpers
# -------------------------
def _to_dt_utc(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    try:
        return pd.to_datetime(s, errors="coerce", utc=True, format="ISO8601")
    except TypeError:
        return pd.to_datetime(s, errors="coerce", utc=True)


def _hour_local(ts_utc: pd.Series, tz: str) -> pd.Series:
    return ts_utc.dt.tz_convert(tz).dt.hour


def _extract_um_uid_from_name(p: Path) -> str:
    m = re.search(r"stops_arrivals_anchor_interp_(\d+)_", p.name)
    return m.group(1) if m else p.stem


def _extract_tag_from_name(p: Path) -> str | None:
    m = re.search(r"_(\d{4}-\d{2}-\d{2})\.csv$", p.name)
    return m.group(1) if m else None


def _calc_match_rate(stops: pd.DataFrame) -> tuple[float, int, int, int]:
    soll_dt = _to_dt_utc(stops.get("ankunft_soll", pd.Series([], dtype="string")))
    ist_dt = _to_dt_utc(stops.get("ankunft_ist", pd.Series([], dtype="string")))

    den = int(soll_dt.notna().sum())
    num = int((soll_dt.notna() & ist_dt.notna()).sum())
    rate = (num / den) if den else np.nan

    raw_ist = stops.get("ankunft_ist")
    if raw_ist is None:
        parsing_fails = 0
    else:
        raw_ist_s = raw_ist.astype("string")
        parsing_fails = int(raw_ist_s.notna().sum() - ist_dt.notna().sum())

    return rate, den, num, parsing_fails


def _safe_tag(s) -> str:
    t = str(s) if s is not None else "NA"
    t = t.replace("-", "_")
    t = re.sub(r"[^0-9A-Za-z_]+", "", t)
    return t or "NA"


# -------------------------
# Paper styling
# -------------------------
from dataclasses import dataclass


@dataclass(frozen=True)
class PaperStyle:
    fig_dpi: int = 1000
    paper_fig_w_in: float = 4.49
    ax_label_fs: int = 11
    tick_fs: int = 10
    title_fs: int = 8
    legend_fs: int = 11


_STYLE = PaperStyle()


def _init_paper_style(*, fig_dpi: int, paper_fig_w_in: float) -> None:
    """
    IMPORTANT: Don't store custom keys in mpl.rcParams (matplotlib rejects unknown keys).
    We keep our own style state in _STYLE.
    """
    global _STYLE
    _STYLE = PaperStyle(fig_dpi=int(fig_dpi), paper_fig_w_in=float(paper_fig_w_in))

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": _STYLE.ax_label_fs,
            "axes.titlesize": _STYLE.title_fs,
            "axes.labelsize": _STYLE.ax_label_fs,
            "xtick.labelsize": _STYLE.tick_fs,
            "ytick.labelsize": _STYLE.tick_fs,
            "legend.fontsize": _STYLE.legend_fs,
            "figure.dpi": 120,
            "savefig.dpi": _STYLE.fig_dpi,
            "savefig.bbox": "tight",
        }
    )


def _paper_subplots(height_in: float):
    return plt.subplots(figsize=(_STYLE.paper_fig_w_in, float(height_in)))


def _apply_paper_axes(ax: plt.Axes) -> plt.Axes:
    ax.tick_params(axis="both", labelsize=_STYLE.tick_fs)
    ax.xaxis.label.set_size(_STYLE.ax_label_fs)
    ax.yaxis.label.set_size(_STYLE.ax_label_fs)
    ax.title.set_size(_STYLE.title_fs)
    return ax

def _save_tiff(fig: plt.Figure, out_dir: Path, name: str, *, fig_dpi: int) -> Path:
    """
    Save figure as high-resolution TIFF in out_dir with given name (without extension).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tiff"
    fig.savefig(path, format="tiff", dpi=fig_dpi)
    return path
# -------------------------
# File collection + QC
# -------------------------
def _collect_arrival_files(arrivals_dir: Path, tags: list[str]) -> list[Path]:
    files: list[Path] = []
    for tag in tags:
        files.extend(sorted(arrivals_dir.glob(f"stops_arrivals_anchor_interp_*_{tag}.csv")))
    return files


def _qc_filter_files(
    files: list[Path],
    *,
    match_threshold: float,
) -> tuple[pd.DataFrame, list[Path], float]:
    qc_rows: list[dict] = []
    used_files: list[Path] = []

    sum_den_all = 0
    sum_num_all = 0

    for fp in files:
        um_uid = _extract_um_uid_from_name(fp)
        tag = _extract_tag_from_name(fp)

        try:
            stops = pd.read_csv(fp, dtype={"stop_seq": "Int64"}, low_memory=False)
            rate, den, num, parsing_fails = _calc_match_rate(stops)

            if np.isfinite(rate):
                sum_den_all += int(den)
                sum_num_all += int(num)

            used = bool(np.isfinite(rate) and rate >= match_threshold)

            qc_rows.append(
                {
                    "um_uid": str(um_uid),
                    "tag": tag,
                    "file": fp.name,
                    "match_rate": rate,
                    "den_soll": den,
                    "num_ist": num,
                    "parsing_fails_ist": parsing_fails,
                    "used": used,
                }
            )

            if used:
                used_files.append(fp)

        except Exception as e:
            qc_rows.append(
                {
                    "um_uid": str(um_uid),
                    "tag": tag,
                    "file": fp.name,
                    "match_rate": np.nan,
                    "den_soll": np.nan,
                    "num_ist": np.nan,
                    "parsing_fails_ist": np.nan,
                    "used": False,
                    "error": repr(e),
                }
            )

    qc = pd.DataFrame(qc_rows)
    overall_match = (sum_num_all / sum_den_all) if sum_den_all else np.nan
    return qc, used_files, overall_match


# -------------------------
# Figure 1: boxplot delay by hour (all QC-used files)
# -------------------------
def _figure1_delay_by_hour(
    used_files: list[Path],
    *,
    hour_tz: str,
    order_hours: list[int],
    out_dir: Path,
    fig_dpi: int,
) -> Path:
    rows: list[pd.DataFrame] = []

    for fp in used_files:
        stops = pd.read_csv(fp, dtype={"stop_seq": "Int64"}, low_memory=False)

        soll_dt = _to_dt_utc(stops.get("ankunft_soll"))
        ist_dt = _to_dt_utc(stops.get("ankunft_ist"))
        ok = soll_dt.notna() & ist_dt.notna()
        if not ok.any():
            continue

        delay_s = (ist_dt - soll_dt).dt.total_seconds()
        delay_s = pd.to_numeric(delay_s, errors="coerce").where(ok)
        abs_diff_s = delay_s.abs()

        tmp = (
            pd.DataFrame({"hour": _hour_local(soll_dt, hour_tz).where(ok), "abs_diff_s": abs_diff_s})
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["hour", "abs_diff_s"])
            .copy()
        )
        tmp["hour"] = tmp["hour"].astype(int)
        rows.append(tmp)

    if not rows:
        raise RuntimeError("Figure 1: No plot data after QC/parsing.")

    all_df = pd.concat(rows, ignore_index=True)

    data = []
    for h in order_hours:
        v = all_df.loc[all_df["hour"] == h, "abs_diff_s"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        data.append(v)

    ymax = 1.0
    for v in data:
        if v.size:
            ymax = max(ymax, float(np.nanpercentile(v, 95)))

    fig, ax = _paper_subplots(height_in=3.5)
    _apply_paper_axes(ax)

    bp = ax.boxplot(data, patch_artist=True, widths=0.45, showfliers=False, whis=(5, 95))

    # Keep exactly like your notebook styling (incl. colors)
    for b in bp["boxes"]:
        b.set(facecolor="#1f77b4", alpha=0.85, linewidth=1.2)
    for w in bp["whiskers"]:
        w.set(color="black", linewidth=1.1)
    for c in bp["caps"]:
        c.set(color="black", linewidth=1.1)
    for m in bp["medians"]:
        m.set(color="orange", linewidth=1.3)

    xpos = np.arange(1, len(order_hours) + 1)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{h:02d}" for h in order_hours], rotation=0)
    ax.tick_params(axis="x", which="both", length=0)

    ax.set_xlabel(f"Hours ({hour_tz})")
    ax.set_ylabel("Absolute delay (s)")
    ax.set_title("Distribution of absolute delays by scheduled hour")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1)

    fig.tight_layout()
    out = _save_tiff(fig, out_dir, "fig1_boxplot_delay_by_hour", fig_dpi=fig_dpi)
    plt.close(fig)
    return out


# -------------------------
# Figure 2: layover times by propulsion type (all QC-used files)
# -------------------------
def _compute_layovers_by_li_reset(stops: pd.DataFrame, *, max_pause_s: float) -> pd.DataFrame:
    d = stops.copy()
    for c in ["stop_seq", "li_lfd_nr", "ankunft_soll", "ankunft_ist"]:
        if c not in d.columns:
            d[c] = pd.NA

    d["stop_seq"] = pd.to_numeric(d["stop_seq"], errors="coerce")
    d["li_lfd_nr"] = pd.to_numeric(d["li_lfd_nr"], errors="coerce")
    d = d.dropna(subset=["stop_seq", "li_lfd_nr", "ankunft_soll"]).copy()
    if d.empty:
        return pd.DataFrame()

    d["stop_seq"] = d["stop_seq"].astype(int)
    d["li_lfd_nr"] = d["li_lfd_nr"].astype(int)

    d["soll_dt"] = _to_dt_utc(d["ankunft_soll"])
    d["ist_dt"] = _to_dt_utc(d["ankunft_ist"])
    d = d.dropna(subset=["soll_dt"]).copy()
    d = d.sort_values("stop_seq").reset_index(drop=True)
    if len(d) < 2:
        return pd.DataFrame()

    is_reset = d["li_lfd_nr"].diff() < 0
    idx = np.where(is_reset.to_numpy())[0]
    if len(idx) == 0:
        return pd.DataFrame()

    prev = d.iloc[idx - 1].reset_index(drop=True)
    nxt = d.iloc[idx].reset_index(drop=True)

    pause_soll_s = (nxt["soll_dt"] - prev["soll_dt"]).dt.total_seconds()

    # "early starts/ends excluded" logic (exactly like notebook):
    # effective start = max(ist_next, soll_next)
    # effective end   = max(ist_prev, soll_prev)
    ist_next_raw = nxt["ist_dt"]
    soll_next = nxt["soll_dt"]
    ist_prev = prev["ist_dt"]
    soll_prev = prev["soll_dt"]

    start_eff_next = ist_next_raw.where(ist_next_raw >= soll_next, soll_next)
    end_eff_prev = ist_prev.where(ist_prev >= soll_prev, soll_prev)

    pause_ist_eff_s = (start_eff_next - end_eff_prev).dt.total_seconds()

    out = pd.DataFrame(
        {
            "pause_idx": np.arange(1, len(prev) + 1),
            "stop_seq_prev": prev["stop_seq"].to_numpy(),
            "stop_seq_next": nxt["stop_seq"].to_numpy(),
            "li_prev": prev["li_lfd_nr"].to_numpy(),
            "li_next": nxt["li_lfd_nr"].to_numpy(),
            "pause_soll_s": pause_soll_s.to_numpy(dtype=float),
            "pause_ist_eff_s": pause_ist_eff_s.to_numpy(dtype=float),
        }
    )

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["pause_soll_s", "pause_ist_eff_s"]).copy()
    out = out[out["pause_soll_s"].between(0, max_pause_s)].copy()
    out = out[out["pause_ist_eff_s"].between(0, max_pause_s)].copy()
    return out.reset_index(drop=True)


def _figure2_layovers_by_propulsion(
    used_files: list[Path],
    *,
    veh_col_candidates: list[str],
    e_types: set[str],
    max_pause_s: float,
    out_dir: Path,
    fig_dpi: int,
) -> Path:
    pauses_all: list[pd.DataFrame] = []
    for fp in used_files:
        stops = pd.read_csv(fp, low_memory=False)
        pauses = _compute_layovers_by_li_reset(stops, max_pause_s=max_pause_s)
        if pauses.empty:
            continue
        pauses["tag"] = _extract_tag_from_name(fp)
        pauses["um_uid"] = str(_extract_um_uid_from_name(fp))
        pauses_all.append(pauses)

    pauses_day = pd.concat(pauses_all, ignore_index=True) if pauses_all else pd.DataFrame()
    if pauses_day.empty:
        raise RuntimeError("Figure 2: no layovers found after QC.")

    # map um_uid -> vehicle type (mode)
    need_um = set(pauses_day["um_uid"].astype(str).unique())
    um_to_type: dict[str, str] = {}

    for fp in used_files:
        um_uid = str(_extract_um_uid_from_name(fp))
        if um_uid not in need_um or um_uid in um_to_type:
            continue

        head = pd.read_csv(fp, nrows=1)
        cols = [c for c in veh_col_candidates if c in head.columns]
        if not cols:
            continue
        usecol = cols[0]

        tmp = pd.read_csv(fp, usecols=[usecol])
        s = tmp[usecol].astype("string").dropna()
        if len(s):
            um_to_type[um_uid] = str(s.value_counts().index[0])

    if not um_to_type:
        sample_cols = pd.read_csv(used_files[0], nrows=1).columns.tolist()
        raise KeyError(
            "Figure 2: No vehicle type column found.\n"
            f"Candidates: {veh_col_candidates}\n"
            f"Example columns: {sample_cols}"
        )

    veh_map = pd.DataFrame({"um_uid": list(um_to_type.keys()), "fzg_typ_text": list(um_to_type.values())})

    p = pauses_day.copy()
    p["um_uid"] = p["um_uid"].astype(str)
    veh_map["um_uid"] = veh_map["um_uid"].astype(str)
    p = p.merge(veh_map, on="um_uid", how="left").dropna(subset=["fzg_typ_text"]).copy()

    p["fzg_typ_text"] = p["fzg_typ_text"].astype("string")
    p["vehicle"] = np.where(p["fzg_typ_text"].isin(list(e_types)), "Electric", "Diesel")

    p["pause_soll_s"] = pd.to_numeric(p["pause_soll_s"], errors="coerce")
    p["pause_ist_eff_s"] = pd.to_numeric(p["pause_ist_eff_s"], errors="coerce")
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["pause_soll_s", "pause_ist_eff_s", "vehicle"]).copy()
    p = p[(p["pause_soll_s"].between(0, max_pause_s)) & (p["pause_ist_eff_s"].between(0, max_pause_s))].copy()

    # build plot_df for seaborn split-violin (Scheduled vs Actual)
    plot_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "vehicle": p["vehicle"],
                    "status": "Scheduled",
                    "duration_min": p["pause_soll_s"].to_numpy(dtype=float) / 60.0,
                }
            ),
            pd.DataFrame(
                {
                    "vehicle": p["vehicle"],
                    "status": "Actual",
                    "duration_min": p["pause_ist_eff_s"].to_numpy(dtype=float) / 60.0,
                }
            ),
        ],
        ignore_index=True,
    )
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["duration_min", "vehicle", "status"]).copy()

    fig, ax = _paper_subplots(height_in=3.5)
    _apply_paper_axes(ax)

    palette = {"Scheduled": "#1f77b4", "Actual": "#ff7f0e"}
    sns.violinplot(
        data=plot_df,
        x="vehicle",
        y="duration_min",
        hue="status",
        hue_order=["Scheduled", "Actual"],
        split=True,
        inner="quartile",
        cut=0,
        linewidth=0.9,
        width=0.78,
        palette=palette,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Layover time (min)")
    ax.set_title("Scheduled vs actual layover times (early starts/ends excluded)")
    ax.grid(True, axis="y", alpha=0.3)

    ymax = float(np.nanpercentile(plot_df["duration_min"].to_numpy(dtype=float), 99)) if len(plot_df) else 1.0
    ax.set_ylim(0, max(1.0, ymax * 1.15))

    # legend below
    if ax.get_legend() is not None:
        ax.legend_.set_title("")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = _save_tiff(fig, out_dir, "fig2_violin_layovers_by_propulsion", fig_dpi=fig_dpi)
    plt.close(fig)
    return out


# -------------------------
# Figure 3: heatmap (selected umlauf only, across days)
# -------------------------
def _build_heatmap_source_df(used_files: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for fp in used_files:
        tag = _extract_tag_from_name(fp)
        uid_raw = _extract_um_uid_from_name(fp)
        umlauf_id = int(uid_raw) if str(uid_raw).isdigit() else None

        stops = pd.read_csv(fp, low_memory=False)
        if "ankunft_soll" not in stops.columns or "ankunft_ist" not in stops.columns:
            continue

        soll_dt = _to_dt_utc(stops["ankunft_soll"])
        ist_dt = _to_dt_utc(stops["ankunft_ist"])
        ok = soll_dt.notna() & ist_dt.notna()
        if not ok.any():
            continue

        delta_s = (ist_dt - soll_dt).dt.total_seconds()
        delta_s = pd.to_numeric(delta_s, errors="coerce").where(ok)

        keep = pd.DataFrame(
            {
                "umlauf_id": umlauf_id,
                "tag": tag,
                "stop_seq": pd.to_numeric(stops.get("stop_seq"), errors="coerce"),
                "delta_time_sec_signed": delta_s,
                "ankunft_ist": stops.get("ankunft_ist"),
            }
        )
        if "frt_fid" in stops.columns:
            keep["frt_fid"] = stops["frt_fid"]

        parts.append(keep)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _figure3_heatmap(
    heat_df: pd.DataFrame,
    *,
    umlauf_id: int,
    clip_min: float,
    clip_max: float,
    pause_gap_rows: int,
    pause_gap_step: float,
    y_tick_step: int,
    out_dir: Path,
    fig_dpi: int,
) -> Path:
    d = heat_df[heat_df["umlauf_id"] == umlauf_id].copy()
    if d.empty:
        raise RuntimeError(f"Figure 3: no heatmap rows for umlauf_id={umlauf_id} (after QC).")

    d = d[d["ankunft_ist"].notna()].copy()
    d["stop_seq"] = pd.to_numeric(d["stop_seq"], errors="coerce")
    d = d.dropna(subset=["stop_seq"]).copy()
    d["stop_seq"] = d["stop_seq"].astype(int)

    d["delta_time_sec_signed"] = pd.to_numeric(d["delta_time_sec_signed"], errors="coerce")
    d = d.dropna(subset=["delta_time_sec_signed"]).copy()
    d = d[(d["delta_time_sec_signed"] >= clip_min) & (d["delta_time_sec_signed"] <= clip_max)].copy()
    if d.empty:
        raise RuntimeError(f"Figure 3: no rows in [{clip_min}, {clip_max}] seconds for umlauf_id={umlauf_id}.")

    med = d.groupby(["stop_seq", "tag"], as_index=False)["delta_time_sec_signed"].median()
    pivot = med.pivot(index="stop_seq", columns="tag", values="delta_time_sec_signed").sort_index()

    max_stop = int(pivot.index.max()) if len(pivot.index) else 0
    pivot = pivot.reindex(list(range(0, max_stop + 1)))

    # detect trip boundaries via frt_fid changes (mode per stop_seq)
    boundary_stop_seqs: list[int] = []
    if "frt_fid" in d.columns:
        map_df = d[["stop_seq", "frt_fid"]].dropna().copy()
        if not map_df.empty:

            def _mode_safe(x):
                m = x.mode()
                return m.iloc[0] if len(m) else np.nan

            seq2frt = (
                map_df.groupby("stop_seq")["frt_fid"].apply(_mode_safe).reset_index().sort_values("stop_seq")
            )
            seq2frt["frt_prev"] = seq2frt["frt_fid"].shift(1)
            seq2frt["is_boundary"] = (seq2frt["frt_fid"] != seq2frt["frt_prev"]) & seq2frt["frt_prev"].notna()
            boundary_stop_seqs = seq2frt.loc[seq2frt["is_boundary"], "stop_seq"].astype(int).tolist()

    # insert pause-gap rows (exact logic from your notebook)
    gap_midline_positions: list[float] = []
    if boundary_stop_seqs:
        new_index: list[float] = []
        for s in range(0, max_stop + 1):
            new_index.append(float(s))
            if s in boundary_stop_seqs:
                gap_start_pos = len(new_index)
                for k in range(1, pause_gap_rows + 1):
                    new_index.append(float(s) + k * float(pause_gap_step))
                gap_end_pos = len(new_index) - 1
                gap_midline_positions.append((gap_start_pos + gap_end_pos) / 2.0)

        pivot = pivot.reindex(new_index)

    # plot: swapped axes (Date on y, Position on x) like your preferred variant
    pivot2 = pivot.T  # rows: tag, cols: stop_seq(+gaps)
    fig = plt.figure(figsize=(_STYLE.paper_fig_w_in, 3.5))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[30, 1.4], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    hm = sns.heatmap(
        pivot2,
        ax=ax,
        cmap="RdBu_r",
        vmin=clip_min,
        vmax=clip_max,
        center=0,
        linewidths=0.0,
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"orientation": "horizontal"},
    )

    # y labels: dd.mm.yy if parseable
    y_labels_raw = pivot2.index.astype(str)
    y_dt = pd.to_datetime(y_labels_raw, errors="coerce")
    fmt = "%d.%m.%y"
    y_labels = [dt.strftime(fmt) if pd.notna(dt) else s for dt, s in zip(y_dt, y_labels_raw)]
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_yticklabels(y_labels, rotation=0, ha="right")
    ax.tick_params(axis="y", pad=2)

    # horizontal day separators
    for y in range(1, pivot2.shape[0]):
        ax.axhline(y=y, color="black", linewidth=0.8, alpha=0.4)

    # colorbar label
    cbar = hm.collections[0].colorbar
    cbar.set_label("Delay (s)")
    cax.xaxis.set_label_position("bottom")
    cax.xaxis.set_ticks_position("bottom")

    # spines
    for side in ["left", "bottom", "right", "top"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)

    # x ticks: integers every y_tick_step
    xvals = np.array(pivot2.columns, dtype=float)
    int_mask = np.isclose(xvals, np.round(xvals))
    int_pos = np.where(int_mask)[0]
    int_labels = xvals[int_mask].astype(int)

    want = set(range(0, max_stop + 1, int(y_tick_step)))
    keep_mask = np.array([lab in want for lab in int_labels], dtype=bool)
    int_pos = int_pos[keep_mask]
    int_labels = int_labels[keep_mask]

    ax.set_xticks(int_pos)
    ax.set_xticklabels(int_labels, rotation=0)

    # gap midlines (vertical now)
    for x in gap_midline_positions:
        ax.axvline(x=x, color="black", linewidth=2.0, zorder=50)

    # labels
    ax.tick_params(axis="both", labelsize=_STYLE.tick_fs)
    cax.tick_params(axis="x", labelsize=_STYLE.tick_fs)
    ax.set_xlabel("Position in schedule")
    ax.set_ylabel("Date")
    ax.set_title(f"Delay along service day – vehicle schedule {umlauf_id}")

    fig.tight_layout()
    out = _save_tiff(fig, out_dir, f"fig3_heatmap_schedule_{umlauf_id}", fig_dpi=fig_dpi)
    plt.close(fig)
    return out


# -------------------------
# Figure 4: deviation plot (selected umlauf only; one recurring trip)
# -------------------------
_FGR_COL_CANDIDATES = ["fgr_nr", "fgr", "fgr_id", "fahrzeitgruppe", "fahrzeitgruppen", "fgr_nr_int"]


def _detect_fgr_col(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)
    for c in _FGR_COL_CANDIDATES:
        if c in cols:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"fgr", "fgr_nr", "fgr-id", "fgr_id"}:
            return c
        if "fahrzeitgr" in lc:
            return c
    return None


def _filter_stops_to_fgr(stops: pd.DataFrame, fgr_selected: float | int | None) -> pd.DataFrame:
    if fgr_selected is None or (isinstance(fgr_selected, float) and np.isnan(fgr_selected)):
        return stops
    col = _detect_fgr_col(stops)
    if col is None:
        return stops
    s = stops.copy()
    v = pd.to_numeric(s[col], errors="coerce")
    if v.notna().any():
        s["_fgr_num"] = v
        return s[s["_fgr_num"] == float(fgr_selected)].drop(columns=["_fgr_num"])
    return s[s[col].astype("string") == str(fgr_selected)]


def _ensure_trip_cols(d: pd.DataFrame) -> pd.DataFrame:
    need = ["stop_seq", "li_lfd_nr", "fgr_nr", "ort_ref_ort_name", "ankunft_soll", "ankunft_ist"]
    out = d.copy()
    for c in need:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def _split_into_fahrten_local(stops: pd.DataFrame, *, only_produktiv: bool, split_on_fgr_change: bool) -> pd.DataFrame:
    d = _ensure_trip_cols(stops)

    if only_produktiv and "produktiv" in d.columns:
        pr = pd.to_numeric(d["produktiv"], errors="coerce")
        d = d[pr == 1].copy()

    d["stop_seq"] = pd.to_numeric(d["stop_seq"], errors="coerce")
    d["li_lfd_nr"] = pd.to_numeric(d["li_lfd_nr"], errors="coerce")
    d["fgr_nr"] = pd.to_numeric(d["fgr_nr"], errors="coerce")

    d = d.dropna(subset=["stop_seq", "li_lfd_nr", "ort_ref_ort_name"]).copy()
    if d.empty:
        return d.assign(fahrt_idx=pd.Series(dtype="Int64"))

    d["stop_seq"] = d["stop_seq"].astype(int)
    d["li_lfd_nr"] = d["li_lfd_nr"].astype(int)
    d = d.sort_values(["stop_seq"]).reset_index(drop=True)

    li = d["li_lfd_nr"].to_numpy()
    li_prev = np.r_[li[0], li[:-1]]
    back_jump = li < li_prev

    if split_on_fgr_change:
        fgr = d["fgr_nr"].to_numpy(dtype=float)
        fgr_prev = np.r_[fgr[0], fgr[:-1]]
        fgr_change = (fgr != fgr_prev)
    else:
        fgr_change = np.zeros(len(d), dtype=bool)

    new_trip = (back_jump | fgr_change)
    new_trip[0] = True
    d["fahrt_idx"] = np.cumsum(new_trip).astype(int)
    return d


def _fahrt_signature(fahrt_stops: pd.DataFrame) -> tuple[float, tuple[str, ...]]:
    fgr = fahrt_stops["fgr_nr"].dropna()
    fgr_val = float(fgr.iloc[0]) if len(fgr) else np.nan

    names = fahrt_stops["ort_ref_ort_name"].astype("string").fillna("").tolist()
    cleaned: list[str] = []
    last = None
    for n in names:
        if last is None or n != last:
            cleaned.append(n)
        last = n

    return fgr_val, tuple(cleaned)


def _build_edges_for_occurrence(
    stops: pd.DataFrame,
    *,
    fahrt_idx: int,
    fgr_selected: float | int | None,
    only_produktiv: bool,
    split_on_fgr_change: bool,
) -> pd.DataFrame:
    stops = _filter_stops_to_fgr(stops, fgr_selected=fgr_selected)

    d = _split_into_fahrten_local(stops, only_produktiv=only_produktiv, split_on_fgr_change=split_on_fgr_change)
    d = d[d["fahrt_idx"] == int(fahrt_idx)].copy()
    if len(d) < 2:
        return pd.DataFrame()

    d["ank_soll_dt"] = _to_dt_utc(d["ankunft_soll"])
    d["ank_ist_dt"] = _to_dt_utc(d["ankunft_ist"])
    d = d.sort_values("stop_seq").reset_index(drop=True)

    a = d.iloc[:-1].copy()
    b = d.iloc[1:].copy()

    # neighbor edges only, and remove pause edges where li resets (li decreases)
    is_neighbor = (
        pd.to_numeric(b["stop_seq"], errors="coerce").to_numpy()
        == pd.to_numeric(a["stop_seq"], errors="coerce").to_numpy() + 1
    )
    li_a = pd.to_numeric(a["li_lfd_nr"], errors="coerce").to_numpy()
    li_b = pd.to_numeric(b["li_lfd_nr"], errors="coerce").to_numpy()
    is_pause_edge = li_b < li_a
    mask = is_neighbor & (~is_pause_edge)

    a = a.loc[mask].reset_index(drop=True)
    b = b.loc[mask].reset_index(drop=True)
    if a.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "edge_idx": np.arange(len(a), dtype=int),
            "li_lfd_nr": pd.to_numeric(a["li_lfd_nr"], errors="coerce").astype("Int64"),
            "halt_from": a["ort_ref_ort_name"].astype("string"),
            "halt_to": b["ort_ref_ort_name"].astype("string"),
        }
    )

    out["soll_s"] = (b["ank_soll_dt"] - a["ank_soll_dt"]).dt.total_seconds().to_numpy()
    out["ist_s"] = (b["ank_ist_dt"] - a["ank_ist_dt"]).dt.total_seconds().to_numpy()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["halt_from", "halt_to", "soll_s"]).copy()
    out = out[(out["soll_s"] >= 0)].copy()
    out = out[(out["ist_s"].isna()) | (out["ist_s"] >= 0)].copy()
    return out


def _build_wide_selected(
    occ_df: pd.DataFrame,
    *,
    arrivals_dir: Path,
    fgr_selected: float | int | None,
    only_produktiv: bool,
    split_on_fgr_change: bool,
) -> pd.DataFrame:
    wide = None
    occ2 = occ_df.drop_duplicates(subset=["tag", "um_uid", "file", "fahrt_idx"]).reset_index(drop=True)

    for _, r in occ2.iterrows():
        fp = arrivals_dir / str(r["file"])
        fahrt_idx = int(r["fahrt_idx"])
        tag = _safe_tag(r.get("tag", "NA"))
        um_uid = str(r.get("um_uid", "NA"))

        stops = pd.read_csv(fp, dtype={"stop_seq": "Int64"}, low_memory=False)
        edges = _build_edges_for_occurrence(
            stops,
            fahrt_idx=fahrt_idx,
            fgr_selected=fgr_selected,
            only_produktiv=only_produktiv,
            split_on_fgr_change=split_on_fgr_change,
        )
        if edges.empty:
            continue

        col_name = f"ist_{tag}_{um_uid}__f{fahrt_idx}"
        edges = edges.rename(columns={"ist_s": col_name})

        key_cols = ["edge_idx", "li_lfd_nr", "halt_from", "halt_to"]
        base_cols = key_cols + ["soll_s"]

        if wide is None:
            wide = edges[base_cols + [col_name]].copy()
        else:
            wide = wide.merge(edges[key_cols + ["soll_s"]], on=key_cols, how="outer", suffixes=("", "__dup"))
            if "soll_s__dup" in wide.columns:
                wide["soll_s"] = wide["soll_s"].combine_first(wide["soll_s__dup"])
                wide = wide.drop(columns=["soll_s__dup"])
            wide = wide.merge(edges[key_cols + [col_name]], on=key_cols, how="outer")

    if wide is None or wide.empty:
        return pd.DataFrame()

    wide["li_lfd_nr"] = pd.to_numeric(wide["li_lfd_nr"], errors="coerce")
    wide = wide.sort_values(["edge_idx", "li_lfd_nr", "halt_from", "halt_to"]).reset_index(drop=True)
    return wide


def _compute_fastest_soll_all_fgr(
    *,
    trips_all: pd.DataFrame,
    signature: tuple[str, ...],
    arrivals_dir: Path,
    only_produktiv: bool,
    split_on_fgr_change: bool,
) -> pd.DataFrame:
    # compute fastest "soll_s" per edge over all FGR occurrences of same signature
    fgrs = trips_all.loc[trips_all["signature"] == signature, "fgr_nr"].dropna().unique().tolist()
    fgrs = sorted([float(x) for x in fgrs])

    soll_rows = []
    for f in fgrs:
        occ = trips_all[(trips_all["signature"] == signature) & (trips_all["fgr_nr"] == f)].copy()
        if occ.empty:
            continue
        wide = _build_wide_selected(
            occ,
            arrivals_dir=arrivals_dir,
            fgr_selected=f,
            only_produktiv=only_produktiv,
            split_on_fgr_change=split_on_fgr_change,
        )
        if wide.empty:
            continue
        tmp = wide[["edge_idx", "li_lfd_nr", "halt_from", "halt_to", "soll_s"]].copy()
        soll_rows.append(tmp)

    if not soll_rows:
        return pd.DataFrame(columns=["edge_idx", "li_lfd_nr", "halt_from", "halt_to", "soll_fastest_all_fgr_s"])

    soll_all = pd.concat(soll_rows, ignore_index=True)
    fastest = (
        soll_all.groupby(["edge_idx", "li_lfd_nr", "halt_from", "halt_to"], as_index=False)
        .agg(soll_fastest_all_fgr_s=("soll_s", "min"))
        .copy()
    )
    return fastest


def _plot_fig4_deviation_with_buffer_markers(
    wide_selected: pd.DataFrame,
    *,
    delay_cap_min: float,
    out_dir: Path,
    fig_dpi: int,
    title: str,
    save_name: str,
    soll_col: str = "soll_s",
    fastest_soll_col: str = "soll_fastest_all_fgr_s",
    ist_prefix: str = "ist_",
) -> Path:
    d = wide_selected.copy()

    ist_cols = [c for c in d.columns if str(c).startswith(ist_prefix)]
    if not ist_cols:
        raise KeyError(f"Figure 4: no IST columns found (prefix '{ist_prefix}').")

    d["li_lfd_nr"] = pd.to_numeric(d["li_lfd_nr"], errors="coerce")
    d = d.dropna(subset=["li_lfd_nr"]).sort_values(["li_lfd_nr", "edge_idx"]).reset_index(drop=True)

    d[soll_col] = pd.to_numeric(d[soll_col], errors="coerce")
    d[fastest_soll_col] = pd.to_numeric(d[fastest_soll_col], errors="coerce")
    for c in ist_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    x = d["li_lfd_nr"].to_numpy(dtype=float)

    soll_seg_min = d[soll_col].to_numpy(dtype=float) / 60.0
    soll_cum_min = np.cumsum(np.nan_to_num(soll_seg_min, nan=0.0))

    # IST cumulative curves (segment times -> cumulative)
    cum_list = []
    for c in ist_cols:
        seg_min = d[c].to_numpy(dtype=float) / 60.0
        ok = np.isfinite(seg_min)
        cum = np.cumsum(np.where(ok, seg_min, 0.0))
        cum = np.where(ok, cum, np.nan)
        cum_list.append(cum)
    Y = np.column_stack(cum_list)

    cap = float(delay_cap_min) if delay_cap_min is not None else float("inf")
    if np.isfinite(cap):
        delay = Y - soll_cum_min[:, None]
        Y = np.where(delay <= cap, Y, np.nan)

    mean_cum = np.nanmean(Y, axis=1)
    std_cum = np.nanstd(Y, axis=1, ddof=1)
    max_cum = np.nanmax(Y, axis=1)
    max_cum = np.where(np.isfinite(max_cum), max_cum, np.nan)

    mean_delta = mean_cum - soll_cum_min
    max_delta = max_cum - soll_cum_min
    lo = mean_delta - std_cum
    hi = mean_delta + std_cum

    # buffer per segment = scheduled - fastest(all fgr)
    fast_seg_min = d[fastest_soll_col].to_numpy(dtype=float) / 60.0
    buffer_seg_min = (soll_seg_min - fast_seg_min)
    buffer_seg_min = np.where(np.isfinite(buffer_seg_min), buffer_seg_min, np.nan)
    buffer_seg_min = np.clip(buffer_seg_min, 0.0, None)

    buf_mask = np.isfinite(buffer_seg_min) & (buffer_seg_min > 0)
    x_buf = x[buf_mask]
    buf_vals = buffer_seg_min[buf_mask]

    # marker y slightly below 0 (scaled)
    y_hi = np.nanmax([np.nanmax(hi), 1.0])
    y_lo = np.nanmin([np.nanmin(lo), -1.0])
    y_span = y_hi - y_lo
    marker_y = -0.06 * y_span
    y_buf = np.full_like(x_buf, float(marker_y), dtype=float)

    # marker sizes scaled by buffer
    size_min, size_max = 25.0, 220.0
    if len(buf_vals) > 0:
        bmin = float(np.nanmin(buf_vals))
        bmax = float(np.nanmax(buf_vals))
        if np.isfinite(bmin) and np.isfinite(bmax) and bmax > bmin:
            s = size_min + (buf_vals - bmin) * (size_max - size_min) / (bmax - bmin)
        else:
            s = np.full_like(buf_vals, 55.0, dtype=float)
    else:
        s = np.array([], dtype=float)

    fig, ax = _paper_subplots(height_in=5)
    _apply_paper_axes(ax)

    ax.plot(x, mean_delta, linewidth=2.5, label="mean delay")
    ax.plot(x, max_delta, linewidth=2.0, linestyle=":", label="max delay")
    ax.axhline(0.0, linewidth=1.2, alpha=0.6, linestyle="--", label="scheduled time")

    # SD band ALWAYS green (wie Notebook)
    ax.fill_between(x, lo, hi, alpha=0.20, color="tab:green", label="mean delay ± 1 σ")

    if len(x_buf) > 0:
        ax.scatter(x_buf, y_buf, s=s, marker="v", alpha=0.35, label="Buffer time added")
        ax.scatter(x_buf, np.zeros_like(x_buf), s=120, marker="|", alpha=0.35)

    ax.set_xlabel("Stops along trip")
    ax.set_ylabel("Deviation from scheduled time [min]")
    ax.set_title(title + (f" (cap {cap:.0f}m)" if np.isfinite(cap) else ""))
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(0, float(np.nanmax(x)) if len(x) else 1.0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = _save_tiff(fig, out_dir, save_name, fig_dpi=fig_dpi)
    plt.close(fig)
    return out

def _plot_fig5_actual_travel_time_curves_capped(
    wide_selected: pd.DataFrame,
    *,
    delay_cap_min: float | None,
    out_dir: Path,
    fig_dpi: int,
    title: str,
    save_name: str,
    ist_prefix: str = "ist_",
    soll_col: str = "soll_s",
    max_curves: int | None = None,
) -> Path:
    d = wide_selected.copy()
    d["li_lfd_nr"] = pd.to_numeric(d["li_lfd_nr"], errors="coerce")
    d = d.dropna(subset=["li_lfd_nr"]).sort_values(["li_lfd_nr", "edge_idx"]).reset_index(drop=True)

    ist_cols = [c for c in d.columns if str(c).startswith(ist_prefix)]
    if not ist_cols:
        raise KeyError(f"Figure 5: no IST columns found (prefix '{ist_prefix}').")

    d[soll_col] = pd.to_numeric(d[soll_col], errors="coerce")
    for c in ist_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    soll_seg_min = d[soll_col].to_numpy(dtype=float) / 60.0
    soll_cum_min = np.cumsum(np.nan_to_num(soll_seg_min, nan=0.0))

    # IST cumulative curves (segment times -> cumulative)
    cum_list = []
    for c in ist_cols:
        seg_min = d[c].to_numpy(dtype=float) / 60.0
        ok = np.isfinite(seg_min)
        cum = np.cumsum(np.where(ok, seg_min, 0.0))
        cum = np.where(ok, cum, np.nan)
        cum_list.append(cum)
    Y = np.column_stack(cum_list)

    # Cap: (cum IST - cum SOLL) > cap => NaN
    if delay_cap_min is not None and np.isfinite(float(delay_cap_min)):
        delay = Y - soll_cum_min[:, None]
        Y = np.where(delay <= float(delay_cap_min), Y, np.nan)

    x = d["li_lfd_nr"].to_numpy(dtype=float)

    cols_to_plot = ist_cols
    if max_curves is not None and len(cols_to_plot) > int(max_curves):
        idx = np.linspace(0, len(cols_to_plot) - 1, int(max_curves)).round().astype(int)
        cols_to_plot = [cols_to_plot[i] for i in idx]

    fig, ax = _paper_subplots(height_in=4)
    _apply_paper_axes(ax)

    for c in cols_to_plot:
        j = ist_cols.index(c)
        ax.plot(x, Y[:, j], linewidth=1.0, alpha=0.20)

    ax.plot(x, soll_cum_min, linewidth=1.8, linestyle="--", alpha=0.7, label="scheduled (cum)")
    ax.set_xlabel("Stops along trip")
    ax.set_ylabel("Cumulative travel time [min]")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = _save_tiff(fig, out_dir, save_name, fig_dpi=fig_dpi)
    plt.close(fig)
    return out

def _fig4_fig5_batch_for_selected_um_uid(
    used_files: list[Path],
    *,
    arrivals_dir: Path,
    umlauf_id: str,
    min_occurrences: int,
    only_produktiv: bool,
    split_on_fgr_change: bool,
    fig4_delay_cap_min: float,
    fig5_delay_cap_min: float | None,
    start_row: int,
    max_plots: int | None,
    out_dir: Path,
    fig_dpi: int,
) -> dict:
    """
    Builds seed signatures from the selected um_uid, matches them across ALL QC-used files,
    builds catalog over (signature, fgr_nr), then plots:
      - Figure 4: deviation + buffer markers
      - Figure 5: actual travel time curves capped
    for ALL catalog rows (optionally limited by start_row/max_plots).
    """
    seed_files = [fp for fp in used_files if str(_extract_um_uid_from_name(fp)) == str(umlauf_id)]
    if not seed_files:
        raise RuntimeError(f"Figure 4/5: no QC-used files for selected umlauf_id={umlauf_id}.")

    # --- seed extraction (only selected um_uid)
    seed_rows: list[dict] = []
    for fp in seed_files:
        tag = _extract_tag_from_name(fp)
        um_uid = _extract_um_uid_from_name(fp)

        stops = pd.read_csv(fp, dtype={"stop_seq": "Int64"}, low_memory=False)
        d = _split_into_fahrten_local(stops, only_produktiv=only_produktiv, split_on_fgr_change=split_on_fgr_change)
        if d.empty:
            continue

        for fahrt_idx, g in d.groupby("fahrt_idx", sort=True):
            if len(g) < 2:
                continue
            fgr_val, sig = _fahrt_signature(g)
            if len(sig) < 2:
                continue
            seed_rows.append(
                {
                    "tag": tag,
                    "um_uid": str(um_uid),
                    "file": fp.name,
                    "fahrt_idx": int(fahrt_idx),
                    "fgr_nr": fgr_val,
                    "signature": sig,
                    "signature_str": " → ".join(sig),
                    "start_stop": sig[0],
                    "end_stop": sig[-1],
                }
            )

    fahrten_seed = pd.DataFrame(seed_rows)
    if fahrten_seed.empty:
        raise RuntimeError(f"Figure 4/5: seed extraction empty for umlauf_id={umlauf_id}.")

    seed_signatures = set(fahrten_seed["signature"].tolist())

    # --- global matching across all used_files
    all_rows: list[dict] = []
    for fp in used_files:
        tag = _extract_tag_from_name(fp)
        um_uid = _extract_um_uid_from_name(fp)

        stops = pd.read_csv(fp, dtype={"stop_seq": "Int64"}, low_memory=False)
        d = _split_into_fahrten_local(stops, only_produktiv=only_produktiv, split_on_fgr_change=split_on_fgr_change)
        if d.empty:
            continue

        for fahrt_idx, g in d.groupby("fahrt_idx", sort=True):
            if len(g) < 2:
                continue
            fgr_val, sig = _fahrt_signature(g)
            if sig not in seed_signatures:
                continue
            all_rows.append(
                {
                    "tag": tag,
                    "um_uid": str(um_uid),
                    "file": fp.name,
                    "fahrt_idx": int(fahrt_idx),
                    "fgr_nr": fgr_val,
                    "signature": sig,
                    "signature_str": " → ".join(sig),
                    "start_stop": sig[0],
                    "end_stop": sig[-1],
                }
            )

    trips_all = pd.DataFrame(all_rows)
    if trips_all.empty:
        raise RuntimeError("Figure 4/5: no trips found matching seed signatures (across QC-used files).")

    # --- catalog by (fgr_nr, signature)
    catalog = (
        trips_all.groupby(["fgr_nr", "signature"], as_index=False)
        .agg(
            occurrences=("um_uid", "size"),
            n_umlaeufe=("um_uid", "nunique"),
            signature_str=("signature_str", "first"),
            start_stop=("start_stop", "first"),
            end_stop=("end_stop", "first"),
        )
        .sort_values(["occurrences", "n_umlaeufe"], ascending=[False, False])
        .reset_index(drop=True)
    )
    catalog = catalog[catalog["occurrences"] >= int(min_occurrences)].reset_index(drop=True)
    if catalog.empty:
        raise RuntimeError(f"Figure 4/5: catalog empty after min_occurrences={min_occurrences}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_csv = out_dir / f"fig45_catalog_umuid_{umlauf_id}.csv"
    catalog.to_csv(catalog_csv, index=False)

    # --- plotting loop
    start_row = int(max(0, start_row))
    end_row = len(catalog) if max_plots is None else min(len(catalog), start_row + int(max_plots))
    rows = list(range(start_row, end_row))

    fig4_paths: list[str] = []
    fig5_paths: list[str] = []

    for i in rows:
        row = catalog.iloc[i]
        signature = row["signature"]
        fgr = float(row["fgr_nr"]) if pd.notna(row["fgr_nr"]) else np.nan

        occ_sel = trips_all[
            (trips_all["signature"] == signature)
            & (pd.to_numeric(trips_all["fgr_nr"], errors="coerce") == fgr)
        ].copy()
        if occ_sel.empty:
            continue

        wide_selected = _build_wide_selected(
            occ_sel,
            arrivals_dir=arrivals_dir,
            fgr_selected=fgr,
            only_produktiv=only_produktiv,
            split_on_fgr_change=split_on_fgr_change,
        )
        if wide_selected.empty:
            continue

        # fastest scheduled across all FGR for buffer markers (same signature, all fgr)
        fastest = _compute_fastest_soll_all_fgr(
            trips_all=trips_all,
            signature=signature,
            arrivals_dir=arrivals_dir,
            only_produktiv=only_produktiv,
            split_on_fgr_change=split_on_fgr_change,
        )
        wide_selected = wide_selected.merge(
            fastest,
            on=["edge_idx", "li_lfd_nr", "halt_from", "halt_to"],
            how="left",
        )
        wide_selected["soll_fastest_all_fgr_s"] = np.where(
            wide_selected["soll_s"].notna() & wide_selected["soll_fastest_all_fgr_s"].notna(),
            np.minimum(wide_selected["soll_fastest_all_fgr_s"], wide_selected["soll_s"]),
            wide_selected["soll_fastest_all_fgr_s"],
        )

        # FIGURE 4
        name4 = f"fig4_delay_row{i}_fgr{int(fgr) if np.isfinite(fgr) else 'NA'}"
        title4 = f"Deviation from schedule (um_uid={umlauf_id} | fgr={fgr:.0f} | row={i})"
        p4 = _plot_fig4_deviation_with_buffer_markers(
            wide_selected,
            delay_cap_min=float(fig4_delay_cap_min),
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            title=title4,
            save_name=name4,
        )
        fig4_paths.append(str(p4))

        # FIGURE 5
        name5 = f"fig5_curves_row{i}_fgr{int(fgr) if np.isfinite(fgr) else 'NA'}"
        title5 = f"Actual travel time curves (um_uid={umlauf_id} | fgr={fgr:.0f} | row={i})"
        p5 = _plot_fig5_actual_travel_time_curves_capped(
            wide_selected,
            delay_cap_min=fig5_delay_cap_min,
            out_dir=out_dir,
            fig_dpi=fig_dpi,
            title=title5,
            save_name=name5,
            max_curves=None,
        )
        fig5_paths.append(str(p5))

    return {
        "catalog_csv": str(catalog_csv),
        "fig4_n": len(fig4_paths),
        "fig5_n": len(fig5_paths),
        "fig4_paths": fig4_paths,
        "fig5_paths": fig5_paths,
    }


# -------------------------
# Public entry point
# -------------------------
def visualize_results(*, raw_cfg: dict) -> dict:
    """
        Entry point used by CLI (see src/itsc_delay_pipeline/cli.py).

        Required config fields (defaults provided where reasonable):
        paths.export_dir                   (arrivals CSV dir)
        visualize_results.tags             (list of YYYY-MM-DD)
        run.umlauf_id                      (seed vehicle schedule for Fig3 and for Fig4/Fig5 batch seed)

        Optional visualize_results fields:
        out_dir, fig_dpi, paper_fig_w_in
        match_threshold, hour_tz, order_hours
        max_layover_min, veh_col_candidates, e_types
        heatmap.clip_min, heatmap.clip_max, heatmap.y_tick_step, heatmap.pause_gap_rows
        fig45.min_occurrences, fig45.fig4_delay_cap_min, fig45.fig5_delay_cap_min
        fig45.start_row, fig45.max_plots
        fig45.only_produktiv, fig45.split_on_fgr_change
    """
    # ---- config
    tags = _get(raw_cfg, "visualize_results.tags", None)
    if not isinstance(tags, list) or not tags:
        raise ValueError("visualize_results.tags must be a non-empty list (e.g., ['2026-01-08','2026-01-15']).")

    arrivals_dir = Path(_get(raw_cfg, "paths.export_dir", "export"))
    out_dir = Path(_get(raw_cfg, "visualize_results.out_dir", str(arrivals_dir / "Plots")))

    fig_dpi = int(_get(raw_cfg, "visualize_results.fig_dpi", 1000))
    paper_fig_w_in = float(_get(raw_cfg, "visualize_results.paper_fig_w_in", 4.49))
    _init_paper_style(fig_dpi=fig_dpi, paper_fig_w_in=paper_fig_w_in)

    match_threshold = float(_get(raw_cfg, "visualize_results.match_threshold", 0.95))
    hour_tz = str(_get(raw_cfg, "visualize_results.hour_tz", "Europe/Berlin"))
    order_hours = _get(raw_cfg, "visualize_results.order_hours", list(range(5, 21)))
    order_hours = [int(x) for x in order_hours]

    max_layover_min = float(_get(raw_cfg, "visualize_results.max_layover_min", 45))
    max_pause_s = max_layover_min * 60.0

    veh_col_candidates = _get(
        raw_cfg,
        "visualize_results.veh_col_candidates",
        ["fzg_typ_text", "fzg_typ", "vehicle_type", "vehicle_type_text", "vehicle_type_name"],
    )
    e_types = set(_get(raw_cfg, "visualize_results.e_types", ["E-Gelenkbus", "E-Solo"]))

    umlauf_id = str(_get(raw_cfg, "run.umlauf_id", "")).strip()
    if not umlauf_id:
        raise ValueError("run.umlauf_id is required for Fig. 3 and Fig. 4 (selected vehicle schedule).")

    heat_clip_min = float(_get(raw_cfg, "visualize_results.heatmap.clip_min", -200))
    heat_clip_max = float(_get(raw_cfg, "visualize_results.heatmap.clip_max", 800))
    heat_y_tick_step = int(_get(raw_cfg, "visualize_results.heatmap.y_tick_step", 50))
    heat_gap_rows = int(_get(raw_cfg, "visualize_results.heatmap.pause_gap_rows", 6))
    heat_gap_step = float(_get(raw_cfg, "visualize_results.heatmap.pause_gap_step", 0.25))

    fig45_min_occ = int(_get(raw_cfg, "visualize_results.fig45.min_occurrences", 2))
    fig45_fig4_cap = float(_get(raw_cfg, "visualize_results.fig45.fig4_delay_cap_min", 30))
    fig45_fig5_cap = _get(raw_cfg, "visualize_results.fig45.fig5_delay_cap_min", None)
    fig45_fig5_cap = None if fig45_fig5_cap is None else float(fig45_fig5_cap)

    fig45_start_row = int(_get(raw_cfg, "visualize_results.fig45.start_row", 0))
    fig45_max_plots = _get(raw_cfg, "visualize_results.fig45.max_plots", None)
    fig45_max_plots = None if fig45_max_plots is None else int(fig45_max_plots)

    fig45_only_produktiv = bool(_get(raw_cfg, "visualize_results.fig45.only_produktiv", True))
    fig45_split_on_fgr_change = bool(_get(raw_cfg, "visualize_results.fig45.split_on_fgr_change", True))

    # ---- collect + QC
    files = _collect_arrival_files(arrivals_dir, tags)
    if not files:
        raise FileNotFoundError(f"No arrivals CSVs found in {arrivals_dir} for tags={tags}")

    qc, used_files, overall_match = _qc_filter_files(files, match_threshold=match_threshold)
    if not used_files:
        raise RuntimeError(f"No files passed QC (match_rate >= {match_threshold:.0%}).")

    # ---- generate EXACTLY four figures
    fig1 = _figure1_delay_by_hour(used_files, hour_tz=hour_tz, order_hours=order_hours, out_dir=out_dir, fig_dpi=fig_dpi)
    fig2 = _figure2_layovers_by_propulsion(
        used_files,
        veh_col_candidates=veh_col_candidates,
        e_types=e_types,
        max_pause_s=max_pause_s,
        out_dir=out_dir,
        fig_dpi=fig_dpi,
    )

    heat_df = _build_heatmap_source_df(used_files)
    fig3 = _figure3_heatmap(
        heat_df,
        umlauf_id=int(umlauf_id),
        clip_min=heat_clip_min,
        clip_max=heat_clip_max,
        pause_gap_rows=heat_gap_rows,
        pause_gap_step=heat_gap_step,
        y_tick_step=heat_y_tick_step,
        out_dir=out_dir,
        fig_dpi=fig_dpi,
    )

    fig45 = _fig4_fig5_batch_for_selected_um_uid(
        used_files,
        arrivals_dir=arrivals_dir,
        umlauf_id=umlauf_id,
        min_occurrences=fig45_min_occ,
        only_produktiv=fig45_only_produktiv,
        split_on_fgr_change=fig45_split_on_fgr_change,
        fig4_delay_cap_min=fig45_fig4_cap,
        fig5_delay_cap_min=fig45_fig5_cap,
        start_row=fig45_start_row,
        max_plots=fig45_max_plots,
        out_dir=out_dir,
        fig_dpi=fig_dpi,
    )

    qc_out = out_dir / "qc_overview.csv"
    qc.to_csv(qc_out, index=False)

    return {
        "arrivals_dir": str(arrivals_dir),
        "out_dir": str(out_dir),
        "tags": ",".join([str(t) for t in tags]),
        "qc_csv": str(qc_out),
        "n_files_total": str(len(files)),
        "n_files_used": str(len(used_files)),
        "overall_match_rate_all_csvs": f"{overall_match:.6f}" if np.isfinite(overall_match) else "nan",
        "fig1": str(fig1),
        "fig2": str(fig2),
        "fig3": str(fig3),
        "fig45_catalog_csv": fig45["catalog_csv"],
        "fig4_n": str(fig45["fig4_n"]),
        "fig5_n": str(fig45["fig5_n"]),
    }
    