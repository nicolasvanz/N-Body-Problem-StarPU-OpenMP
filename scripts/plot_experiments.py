#!/usr/bin/env python3
"""Generate experiment plots from results/experiments sessions."""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODE_LABELS = {
    "mpi": "full-mpi",
    "ms": "master-slave",
}
DEFAULT_INSTANCE_HOURLY_COST_USD = 0.9776
PLOT_FONT_SCALE = 1.25


EXPECTED_RESULTS_COLUMNS = {
    "timestamp",
    "mode",
    "nodes",
    "np",
    "paired_nodes",
    "exp",
    "n_bodies",
    "parts",
    "gpu_lanes",
    "rep_kind",
    "rep_idx",
    "time_us",
    "run_ok",
    "notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready plots from all experiment sessions."
    )
    parser.add_argument(
        "--results-root",
        default="results/experiments",
        help="Root directory containing experiment session folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: results/plots/<timestamp>.",
    )
    parser.add_argument(
        "--rep-kind",
        choices=("all", "timed", "warmup"),
        default="all",
        help="Which repetition kind to include in the main analysis.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="pdf",
        help="Figure output format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for raster outputs.",
    )
    parser.add_argument(
        "--include-sessions",
        default="",
        help="Comma-separated session IDs to include (optional).",
    )
    parser.add_argument(
        "--exclude-sessions",
        default="",
        help="Comma-separated session IDs to exclude (optional).",
    )
    parser.add_argument(
        "--latest-link",
        default="results/plots/latest",
        help="Symlink to point at latest output directory.",
    )
    parser.add_argument(
        "--instance-hourly-cost",
        type=float,
        default=DEFAULT_INSTANCE_HOURLY_COST_USD,
        help="Per-instance hourly cost in USD (used for execution cost plot).",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def fmt_factor(value: float) -> str:
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


def factor_sort_value(label: str) -> float:
    # Backward-compatible with older labels that may still include an "x" suffix.
    return float(label.rstrip("x"))


def pair_label(nodes: int, paired_nodes: int) -> str:
    lo = min(nodes, paired_nodes)
    hi = max(nodes, paired_nodes)
    return f"{lo} \u2194 {hi}"


def sort_pair_labels(labels: Iterable[str]) -> list[str]:
    def key_fn(label: str) -> tuple[int, int]:
        try:
            norm = label.replace("\u2194", "<->")
            left, right = [part.strip() for part in norm.split("<->")]
            return (int(left), int(right))
        except Exception:
            return (10_000, 10_000)

    return sorted(set(labels), key=key_fn)


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_results_frame(
    results_root: Path, include_sessions: set[str], exclude_sessions: set[str]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not results_root.exists():
        raise SystemExit(f"results root not found: {results_root}")

    for session_dir in sorted(results_root.iterdir()):
        if not session_dir.is_dir():
            continue

        session = session_dir.name
        if include_sessions and session not in include_sessions:
            continue
        if session in exclude_sessions:
            continue

        csv_path = session_dir / "results.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as exc:
            print(f"[plot] skip session {session}: failed to read results.csv ({exc})")
            continue

        missing = sorted(EXPECTED_RESULTS_COLUMNS - set(df.columns))
        if missing:
            print(
                f"[plot] skip session {session}: missing required columns: {', '.join(missing)}"
            )
            continue

        df = df[df["run_ok"] == "1"].copy()
        if df.empty:
            print(f"[plot] skip session {session}: no successful rows (run_ok=1)")
            continue

        df = ensure_numeric(
            df,
            ["nodes", "paired_nodes", "exp", "n_bodies", "parts", "gpu_lanes", "time_us"],
        )
        df = df.dropna(
            subset=["nodes", "paired_nodes", "exp", "parts", "gpu_lanes", "time_us"]
        ).copy()
        if df.empty:
            print(f"[plot] skip session {session}: no parseable numeric rows")
            continue

        df["nodes"] = df["nodes"].astype(int)
        df["paired_nodes"] = df["paired_nodes"].astype(int)
        df["exp"] = df["exp"].astype(int)
        df["parts"] = df["parts"].astype(int)
        df["gpu_lanes"] = df["gpu_lanes"].astype(int)

        df = df[df["gpu_lanes"] > 0].copy()
        if df.empty:
            print(f"[plot] skip session {session}: no rows with gpu_lanes > 0")
            continue

        df["parts_factor"] = df["parts"] / df["gpu_lanes"]
        df = df[df["parts_factor"].map(math.isfinite) & (df["parts_factor"] > 0)].copy()
        if df.empty:
            print(f"[plot] skip session {session}: invalid parts_factor values")
            continue

        df["parts_factor_label"] = df["parts_factor"].map(lambda x: fmt_factor(float(x)))
        df["mode"] = df["mode"].str.lower()
        df = df[df["mode"].isin(MODE_LABELS)].copy()
        if df.empty:
            print(f"[plot] skip session {session}: no supported mode rows (mpi/ms)")
            continue
        df["rep_kind"] = df["rep_kind"].str.lower()
        df = df[df["rep_kind"].isin(["warmup", "timed"])].copy()
        if df.empty:
            print(f"[plot] skip session {session}: no warmup/timed rows")
            continue

        df["mode_label"] = df["mode"].map(MODE_LABELS)
        df["time_s"] = df["time_us"] / 1_000_000.0
        df["paired_label"] = [
            pair_label(n, p) for n, p in zip(df["nodes"], df["paired_nodes"], strict=True)
        ]
        df["session"] = session
        frames.append(df)

    if not frames:
        raise SystemExit("[plot] no valid results.csv sessions found")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        by=["mode", "nodes", "exp", "parts", "rep_kind", "session"], ignore_index=True
    )
    return out


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            [
                "mode",
                "mode_label",
                "nodes",
                "paired_nodes",
                "paired_label",
                "exp",
                "n_bodies",
                "parts",
                "gpu_lanes",
                "parts_factor",
                "parts_factor_label",
                "rep_kind",
            ],
            dropna=False,
        )["time_s"]
        .agg(
            samples="count",
            median_s="median",
            mean_s="mean",
            min_s="min",
            max_s="max",
            stddev_s="std",
        )
        .reset_index()
    )
    grouped["stddev_s"] = grouped["stddev_s"].fillna(0.0)
    return grouped


def case_medians(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "mode",
                "mode_label",
                "nodes",
                "paired_nodes",
                "paired_label",
                "exp",
                "parts",
                "gpu_lanes",
                "parts_factor",
                "parts_factor_label",
            ],
            dropna=False,
        )["time_s"]
        .median()
        .reset_index(name="time_s_median")
    )


def save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Facet figures with a shared legend are laid out in move_facet_legend_below.
    # Re-running tight_layout here would shift axes and break legend placement.
    if fig.legends:
        fig.savefig(path, dpi=dpi)
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


def move_facet_legend_below(grid: sns.FacetGrid, ncol: int | None = None) -> None:
    handles = []
    labels = []

    # Collect legend entries from all subplot axes.
    for ax in grid.axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h and l:
            handles.extend(h)
            labels.extend(l)
        # Remove any axes-level legends that might overlap data.
        if ax.legend_ is not None:
            ax.legend_.remove()

    # If seaborn created a FacetGrid-level legend, remove it and use it as fallback.
    if grid._legend is not None:
        if not handles:
            fallback_handles = getattr(grid._legend, "legendHandles", None)
            fallback_labels = [t.get_text() for t in grid._legend.get_texts()]
            if fallback_handles and fallback_labels:
                handles = list(fallback_handles)
                labels = fallback_labels
        grid._legend.remove()

    if not labels or not handles:
        return

    # De-duplicate while preserving order.
    dedup_handles = []
    dedup_labels = []
    seen = set()
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        dedup_handles.append(h)
        dedup_labels.append(l)

    if ncol is None:
        ncol = max(1, min(len(dedup_labels), 4))

    # Start from a compact layout; adjust bottom only if overlap is detected.
    grid.fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))

    # Center legend under the subplot area (not full figure/page width).
    axes = [ax for ax in grid.axes.flat if ax is not None]
    if axes:
        min_x = min(ax.get_position().x0 for ax in axes)
        max_x = max(ax.get_position().x1 for ax in axes)
        center_x = (min_x + max_x) / 2.0
    else:
        center_x = 0.5

    legend_y = 0.004
    legend = grid.fig.legend(
        dedup_handles,
        dedup_labels,
        loc="lower center",
        bbox_to_anchor=(center_x, legend_y),
        bbox_transform=grid.fig.transFigure,
        ncol=ncol,
        frameon=False,
        title="",
    )

    # Guarantee no overlap with subplot decorations (ticks/labels) using tight bboxes.
    # Increase bottom margin only as much as required.
    required_gap = 0.006
    axes = [ax for ax in grid.axes.flat if ax is not None]
    for _ in range(4):
        grid.fig.canvas.draw()
        renderer = grid.fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer).transformed(
            grid.fig.transFigure.inverted()
        )
        tight_boxes = [
            ax.get_tightbbox(renderer=renderer).transformed(grid.fig.transFigure.inverted())
            for ax in axes
        ]
        if not tight_boxes:
            break
        min_tight_y0 = min(b.y0 for b in tight_boxes)
        overlap = (legend_bbox.y1 + required_gap) - min_tight_y0
        if overlap <= 0:
            break
        new_bottom = min(0.35, grid.fig.subplotpars.bottom + overlap + 0.004)
        grid.fig.subplots_adjust(bottom=new_bottom)


def remove_plot_borders(grid: sns.FacetGrid) -> None:
    for ax in grid.axes.flat:
        if ax is None:
            continue
        for spine in ax.spines.values():
            spine.set_visible(False)


def annotate_catplot_bars(
    grid: sns.FacetGrid,
    value_fmt: str,
    frac_offset: float = 0.012,
    rotation_deg: float = 45.0,
) -> None:
    for ax in grid.axes.flat:
        if ax is None:
            continue
        y0, y1 = ax.get_ylim()
        yrange = max(1e-12, y1 - y0)
        dy = yrange * frac_offset
        for patch in ax.patches:
            h = patch.get_height()
            if not math.isfinite(h) or h <= 0:
                continue
            x = patch.get_x() + patch.get_width() / 2.0
            ax.text(
                x,
                h + dy,
                value_fmt.format(h),
                ha="left",
                va="bottom",
                fontsize=10,
                rotation=rotation_deg,
                rotation_mode="anchor",
            )
        # Ensure labels are not clipped by top border.
        ax.set_ylim(y0, y1 + dy * 2.5)


def plot_runtime_by_mode_paired(df: pd.DataFrame, output: Path, fmt: str, dpi: int) -> None:
    if df.empty:
        return
    pair_order = sort_pair_labels(df["paired_label"].tolist())
    exp_order = sorted(df["exp"].unique())
    factor_order = sorted(df["parts_factor_label"].unique(), key=factor_sort_value)

    grid = sns.catplot(
        data=df,
        kind="bar",
        x="paired_label",
        y="time_s_median",
        hue="mode_label",
        col="exp",
        row="parts_factor_label",
        order=pair_order,
        col_order=exp_order,
        row_order=factor_order,
        height=3.5,
        aspect=1.15,
        errorbar=None,
    )
    grid.set_axis_labels("nodes", "Execution time (s)")
    grid.set_titles(
        row_template="Partitions / GPUs = {row_name}",
        col_template="n = 2^{col_name} bodies",
    )
    annotate_catplot_bars(grid, "{:.2f}")
    remove_plot_borders(grid)
    move_facet_legend_below(grid, ncol=2)
    save_fig(grid.fig, output / f"01_runtime_by_mode_paired.{fmt}", dpi)


def plot_execution_cost(
    df: pd.DataFrame, output: Path, fmt: str, dpi: int, instance_hourly_cost: float
) -> None:
    if df.empty:
        return
    if instance_hourly_cost <= 0:
        raise SystemExit("[plot] --instance-hourly-cost must be > 0")

    data = df.copy()
    exp_order = sorted(data["exp"].unique())
    factor_order = sorted(data["parts_factor_label"].unique(), key=factor_sort_value)
    pair_order = sort_pair_labels(data["paired_label"].tolist())
    data["execution_cost_usd"] = (
        data["time_s_median"] * data["nodes"] * instance_hourly_cost / 3600.0
    )

    grid = sns.catplot(
        data=data,
        kind="bar",
        x="paired_label",
        y="execution_cost_usd",
        hue="mode_label",
        col="exp",
        row="parts_factor_label",
        col_order=exp_order,
        row_order=factor_order,
        order=pair_order,
        height=3.5,
        aspect=1.15,
        errorbar=None,
    )
    grid.set_axis_labels("nodes", "Execution cost (USD)")
    grid.set_titles(
        row_template="Partitions / GPUs = {row_name}",
        col_template="n = 2^{col_name} bodies",
    )
    annotate_catplot_bars(grid, "{:.3f}")
    remove_plot_borders(grid)
    move_facet_legend_below(grid, ncol=2)
    save_fig(grid.fig, output / f"02_execution_cost.{fmt}", dpi)


def plot_scaling_with_nodes(df: pd.DataFrame, output: Path, fmt: str, dpi: int) -> None:
    if df.empty:
        return
    data = df.copy()
    exp_order = sorted(data["exp"].unique())
    factor_order = sorted(data["parts_factor_label"].unique(), key=factor_sort_value)

    # Normalize each series by its smallest-lane runtime to show scale-up.
    base = (
        data.sort_values("gpu_lanes")
        .groupby(["mode_label", "exp", "parts_factor_label"], as_index=False)
        .first()[["mode_label", "exp", "parts_factor_label", "time_s_median"]]
        .rename(columns={"time_s_median": "base_time_s"})
    )
    data = data.merge(base, on=["mode_label", "exp", "parts_factor_label"], how="left")
    data["speedup"] = data["base_time_s"] / data["time_s_median"]
    data["efficiency_pct"] = (data["speedup"] / data["gpu_lanes"]) * 100.0

    grid = sns.relplot(
        data=data,
        kind="line",
        x="gpu_lanes",
        y="speedup",
        hue="mode_label",
        style="mode_label",
        markers=True,
        dashes=False,
        estimator=None,
        col="exp",
        row="parts_factor_label",
        col_order=exp_order,
        row_order=factor_order,
        height=3.4,
        aspect=1.15,
    )
    grid.set_axis_labels("GPU lanes", "Speedup")
    grid.set_titles(
        row_template="Partitions / GPUs = {row_name}",
        col_template="n = 2^{col_name} bodies",
    )
    # Show integer speedup ticks (odd and even values) when range is reasonable.
    for ax in grid.axes.flat:
        if ax is None:
            continue
        y0, y1 = ax.get_ylim()
        lo = max(0, int(math.floor(y0)))
        hi = int(math.ceil(y1))
        if hi - lo <= 20:
            ax.set_yticks(list(range(lo, hi + 1)))
    remove_plot_borders(grid)
    move_facet_legend_below(grid, ncol=2)
    save_fig(grid.fig, output / f"03_scaling_with_nodes.{fmt}", dpi)


def plot_warmup_vs_timed_delta(
    all_runs: pd.DataFrame, output: Path, fmt: str, dpi: int
) -> None:
    warm = (
        all_runs[all_runs["rep_kind"] == "warmup"]
        .groupby(
            ["mode", "mode_label", "nodes", "paired_label", "exp", "parts_factor_label"],
            dropna=False,
        )["time_s"]
        .median()
        .reset_index(name="warmup_median")
    )
    timed = (
        all_runs[all_runs["rep_kind"] == "timed"]
        .groupby(
            ["mode", "mode_label", "nodes", "paired_label", "exp", "parts_factor_label"],
            dropna=False,
        )["time_s"]
        .median()
        .reset_index(name="timed_median")
    )
    merged = warm.merge(
        timed,
        on=["mode", "mode_label", "nodes", "paired_label", "exp", "parts_factor_label"],
        how="inner",
    )
    if merged.empty:
        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.text(
            0.5,
            0.5,
            "No warmup+timed overlap available for delta plot.",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.axis("off")
        save_fig(fig, output / f"04_warmup_vs_timed_delta.{fmt}", dpi)
        return

    merged["delta_pct"] = (
        (merged["warmup_median"] - merged["timed_median"]) / merged["timed_median"] * 100.0
    )
    merged["case"] = (
        merged["mode_label"]
        + " n"
        + merged["nodes"].astype(str)
        + " e"
        + merged["exp"].astype(str)
        + " "
        + merged["parts_factor_label"]
    )
    merged = merged.sort_values(by=["mode", "nodes", "exp", "parts_factor_label"])

    fig, ax = plt.subplots(figsize=(max(10, len(merged) * 0.45), 4.8))
    sns.barplot(
        data=merged,
        x="case",
        y="delta_pct",
        hue="mode_label",
        ax=ax,
        errorbar=None,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Case")
    ax.set_ylabel("(warmup - timed) / timed (%)")
    for tick in ax.get_xticklabels():
        tick.set_rotation(75)
        tick.set_ha("right")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, title="")
    save_fig(fig, output / f"04_warmup_vs_timed_delta.{fmt}", dpi)


def update_latest_symlink(latest_link: Path, target_dir: Path) -> None:
    latest_link.parent.mkdir(parents=True, exist_ok=True)
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    rel_target = Path(
        os.path.relpath(target_dir.resolve(), latest_link.parent.resolve())
    )
    latest_link.symlink_to(rel_target)
    print(f"[plot] updated {latest_link} -> {rel_target}")


def main() -> int:
    args = parse_args()
    sns.set_theme(
        style="whitegrid",
        font_scale=PLOT_FONT_SCALE,
        rc={
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        },
    )

    include_sessions = parse_csv_list(args.include_sessions)
    exclude_sessions = parse_csv_list(args.exclude_sessions)
    results_root = Path(args.results_root)

    all_runs = load_results_frame(results_root, include_sessions, exclude_sessions)
    if args.rep_kind != "all":
        plot_runs = all_runs[all_runs["rep_kind"] == args.rep_kind].copy()
    else:
        plot_runs = all_runs.copy()

    if plot_runs.empty:
        raise SystemExit(f"[plot] no rows after rep_kind filter: {args.rep_kind}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = Path("results/plots") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregated = aggregate_metrics(plot_runs)
    aggregated.to_csv(out_dir / "aggregated_metrics.csv", index=False)
    print(f"[plot] wrote {out_dir / 'aggregated_metrics.csv'}")

    medians = case_medians(plot_runs)
    plot_runtime_by_mode_paired(medians, out_dir, args.format, args.dpi)
    plot_execution_cost(
        medians, out_dir, args.format, args.dpi, args.instance_hourly_cost
    )
    plot_scaling_with_nodes(medians, out_dir, args.format, args.dpi)

    update_latest_symlink(Path(args.latest_link), out_dir)
    print(f"[plot] done: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
