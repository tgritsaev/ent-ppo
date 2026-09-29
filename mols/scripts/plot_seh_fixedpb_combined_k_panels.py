"""Paper plotting style adapted to explicit reproduction roots.

Lines show seed means; shaded bands show the seed min–max range.
Only points with --min-seeds observations are plotted; no endpoint stretching.
"""
import argparse
import csv
import json
import sys
import yaml
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D


METRIC_KEYS = ("elbo", "iw_eubo")
EUBO_SOURCE = "iw_eubo"
FIGURE_NOTE = ""
BASELINE_LABELS = {
    "DB", "TB", "SubTB", "VarGrad", "TRPO (ours)", "TRPO (GFN-PG regime)"
}

COLORS = {
    "DB": "tab:blue",
    "TB": "tab:purple",
    "SubTB": "tab:brown",
    "VarGrad": "navy",
    "TRPO (ours)": "black",
    "TRPO (GFN-PG regime)": "black",
    r"Ent-PPO ($K = 1$) / VPG (GAE)": "tab:green",
    r"Ent-PPO ($K = 2$)": "tab:orange",
    r"Ent-PPO ($K = 4$)": "tab:red",
    r"Ent-PPO ($K = 8$)": "royalblue",
}


def set_publication_style() -> None:
    """Style copied from paper_images_PPO.py without importing its data-loading code."""
    plt.rcParams.update(
        {
            "savefig.format": "eps",
            "ps.useafm": True,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
            "figure.figsize": (6.8, 4.2),
            "figure.dpi": 120,
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "lines.linewidth": 1.2,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def format_k(x: float, _: object) -> str:
    if x >= 10_000:
        scale = x / 10_000
        scale_str = f"{int(scale)}" if scale.is_integer() else f"{scale:.1f}"
        return rf"${scale_str} \times 10^4$"
    return f"{int(x)}"


def parse_validation_log(
    path: Path,
    label: str,
    panel: str,
    seed: int,
    run_batch_size: int,
) -> list[dict[str, float | int | str]]:
    line_re = re.compile(r"validation - iteration (\d+) : (.*)$")
    metric_re = re.compile(r"([A-Za-z0-9_./-]+):\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
    rows = []
    for line in path.read_text(encoding="utf8", errors="replace").splitlines():
        match = line_re.search(line)
        if not match:
            continue
        metrics = {name: float(value) for name, value in metric_re.findall(match.group(2))}
        row = {
            "panel": panel,
            "label": label,
            "seed": seed,
            "iteration": int(match.group(1)),
            "num_trajectories": int(match.group(1)) * run_batch_size,
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(EUBO_SOURCE if key == "iw_eubo" else key, "")
        rows.append(row)
    return rows


def collect_run(
    run_dir: Path,
    label: str,
    panel: str,
    max_seed: int | None,
    run_batch_size: int,
) -> list[dict[str, float | int | str]]:
    rows = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        if max_seed is not None and seed > max_seed:
            continue
        log_path = seed_dir / "train.log"
        if log_path.exists():
            rows.extend(parse_validation_log(log_path, label, panel, seed, run_batch_size))
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], row["label"], row["num_trajectories"])].append(row)

    summary = []
    for (panel, label, num_trajectories), group in sorted(
        grouped.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1]), int(item[0][2])),
    ):
        out = {
            "panel": panel,
            "label": label,
            "iteration": int(round(num_trajectories / np.mean([float(row["num_trajectories"]) / float(row["iteration"]) for row in group]))),
            "num_trajectories": num_trajectories,
            "num_seeds": len({row["seed"] for row in group}),
        }
        for metric in METRIC_KEYS:
            values = [float(row[metric]) for row in group if row[metric] != ""]
            if values:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values))
                out[f"{metric}_min"] = float(np.min(values))
                out[f"{metric}_max"] = float(np.max(values))
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
                out[f"{metric}_min"] = ""
                out[f"{metric}_max"] = ""
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(
    axis,
    rows,
    labels,
    metric: str,
    batch_size: int,
    max_trajectories: int,
    legend_handles: dict[str, object],
    ymin: float | None,
    ymax: float | None,
    align_final_k8_to_xmax: bool,
    align_final_k4_to_xmax: bool,
) -> None:
    mean_key = f"{metric}_mean"
    final_x_by_label = {
        label: max(
            [int(row["num_trajectories"]) for row in rows if row["label"] == label],
            default=0,
        )
        for label in labels
    }
    for label in labels:
        label_rows = [row for row in rows if row["label"] == label and row[mean_key] != ""]
        label_rows = [row for row in label_rows if int(row["num_trajectories"]) <= max_trajectories]
        if not label_rows:
            continue
        x = [
            max_trajectories
            if (
                (
                    align_final_k8_to_xmax
                    and label == r"Ent-PPO ($K = 8$)"
                    and int(row["num_trajectories"]) == final_x_by_label[label]
                )
                or (
                    align_final_k4_to_xmax
                    and label == r"Ent-PPO ($K = 4$)"
                    and int(row["num_trajectories"]) == final_x_by_label[label]
                )
            )
            else int(row["num_trajectories"])
            for row in label_rows
        ]
        y = [float(row[mean_key]) for row in label_rows]
        y_low = [float(row[f"{metric}_min"]) for row in label_rows]
        y_high = [float(row[f"{metric}_max"]) for row in label_rows]
        color = COLORS[label]
        (line,) = axis.plot(
            x,
            y,
            label=label,
            color=color,
            alpha=0.9,
            linestyle="--" if label in BASELINE_LABELS else "-",
        )
        axis.fill_between(
            x,
            y_low,
            y_high,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        legend_handles.setdefault(label, line)
    axis.set_xlabel("Number of trajectories")
    axis.set_xlim(left=0, right=max_trajectories)
    if ymin is not None or ymax is not None:
        axis.set_ylim(bottom=ymin, top=ymax)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(FuncFormatter(format_k))
    axis.tick_params(axis="x", labelsize=6.5)
    axis.grid(True, alpha=0.25)


def add_elbo_zoom(
    axis,
    rows,
    labels,
    batch_size: int,
    max_trajectories: int,
    top_n: int,
    ymin: float | None = None,
    ymax: float | None = None,
    align_final_k8_to_xmax: bool = False,
    align_final_k4_to_xmax: bool = False,
    xmin: int | None = None,
    selection_threshold: float | None = None,
) -> None:
    min_zoom_x = xmin if xmin is not None else max_trajectories / 2
    final_x_by_label = {
        label: max(
            [int(row["num_trajectories"]) for row in rows if row["label"] == label],
            default=0,
        )
        for label in labels
    }
    scored_labels = []
    for label in labels:
        label_rows = [
            row
            for row in rows
            if row["label"] == label
            and row["elbo_mean"] != ""
            and min_zoom_x <= int(row["num_trajectories"]) <= max_trajectories
        ]
        if not label_rows:
            continue
        values = [float(row["elbo_mean"]) for row in label_rows]
        threshold = selection_threshold if selection_threshold is not None else ymin
        if threshold is not None and max(values) <= threshold:
            continue
        score = float(np.mean(values))
        scored_labels.append((score, label))
    ranked_labels = sorted(scored_labels, reverse=True)
    # A threshold selects every qualifying mean curve. Without one, retain
    # the historical top-N selection. A y minimum remains a backward-compatible
    # selection threshold when no separate threshold was supplied.
    select_all = selection_threshold is not None or ymin is not None
    zoom_labels = [label for _, label in (ranked_labels if select_all else ranked_labels[:top_n])]
    if not zoom_labels:
        return

    inset = axis.inset_axes([0.47, 0.14, 0.50, 0.38])
    y_values = []
    mean_values = []
    endpoints = []
    for label in labels:
        if label not in zoom_labels:
            continue
        label_rows = [
            row
            for row in rows
            if row["label"] == label
            and row["elbo_mean"] != ""
            and min_zoom_x <= int(row["num_trajectories"]) <= max_trajectories
        ]
        if not label_rows:
            continue
        x = [
            max_trajectories
            if (
                (
                    align_final_k8_to_xmax
                    and label == r"Ent-PPO ($K = 8$)"
                    and int(row["num_trajectories"]) == final_x_by_label[label]
                )
                or (
                    align_final_k4_to_xmax
                    and label == r"Ent-PPO ($K = 4$)"
                    and int(row["num_trajectories"]) == final_x_by_label[label]
                )
            )
            else int(row["num_trajectories"])
            for row in label_rows
        ]
        y = [float(row["elbo_mean"]) for row in label_rows]
        y_low = [float(row["elbo_min"]) for row in label_rows]
        y_high = [float(row["elbo_max"]) for row in label_rows]
        color = COLORS[label]
        inset.plot(
            x,
            y,
            color=color,
            alpha=0.95,
            linewidth=1.0,
            linestyle="--" if label in BASELINE_LABELS else "-",
        )
        inset.fill_between(
            x,
            y_low,
            y_high,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        mean_values.extend(y)
        endpoints.append((x[-1], y[-1], color))
        y_values.extend(y_low)
        y_values.extend(y_high)

    if y_values:
        y_min, y_max = min(y_values), max(y_values)
        if ymin is not None:
            # Keep this a zoom on the mean curves; broad uncertainty bands
            # are clipped at its bounds rather than expanding the inset.
            y_min, y_max = ymin, max(mean_values)
        margin = max((y_max - y_min) * 0.18, 0.08)
        inset.set_ylim(ymin if ymin is not None else y_min - margin, ymax if ymax is not None else y_max + margin)
    for rank, (x_end, y_end, color) in enumerate(sorted(endpoints, key=lambda item: item[1])):
        offset = (rank - (len(endpoints) - 1) / 2) * 6
        inset.annotate(
            f"{y_end:.2f}",
            xy=(x_end, y_end),
            xytext=(1.02, (rank + 0.5) / len(endpoints)) if ymin is not None else (4, offset),
            textcoords="axes fraction" if ymin is not None else "offset points",
            fontsize=6,
            color=color,
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.18},
            zorder=10,
        )
    inset.set_xlim(min_zoom_x, max_trajectories)
    inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
    inset.xaxis.set_major_formatter(FuncFormatter(format_k))
    inset.tick_params(axis="x", labelsize=4.5)
    inset.tick_params(axis="y", labelsize=6)
    inset.grid(True, alpha=0.25)


def make_plot(
    summary_rows: list[dict[str, float | int | str]],
    output_path: Path,
    batch_size: int,
    max_trajectories: int,
    elbo_ymin: float | None = None,
    elbo_ymax: float | None = None,
    eubo_ymin: float | None = None,
    eubo_ymax: float | None = None,
    elbo_zoom_top_n: int = 4,
    right_elbo_zoom_top_n: int | None = None,
    include_k8: bool = True,
    left_elbo_zoom_ymin: float | None = None,
    left_elbo_zoom_ymax: float | None = None,
    right_elbo_zoom_ymin: float | None = None,
    right_elbo_zoom_ymax: float | None = None,
    include_entppo_k1: bool = True,
    include_entppo_k2: bool = True,
    include_right_tb: bool = True,
    align_final_k8_to_xmax: bool = False,
    align_final_k4_to_xmax: bool = False,
    elbo_zoom_xmin: int | None = None,
    elbo_zoom_threshold: float | None = None,
    match_eubo_ylims: bool = False,
    include_elbo_zoom: bool = True,
) -> None:
    set_publication_style()
    left_labels = [
        "DB",
        "TB",
        "SubTB",
        r"Ent-PPO ($K = 4$)",
    ]
    if include_entppo_k1:
        left_labels.insert(3, r"Ent-PPO ($K = 1$) / VPG (GAE)")
    if include_entppo_k2:
        insert_idx = 4 if include_entppo_k1 else 3
        left_labels.insert(insert_idx, r"Ent-PPO ($K = 2$)")
    right_labels = ["DB", "SubTB", r"Ent-PPO ($K = 4$)"]
    if include_right_tb:
        right_labels.insert(1, "TB")
    if include_k8:
        left_labels.append(r"Ent-PPO ($K = 8$)")
        right_labels.append(r"Ent-PPO ($K = 8$)")
    if any(row["label"] == "VarGrad" for row in summary_rows):
        left_labels.insert(3, "VarGrad")
        right_labels.insert(3, "VarGrad")
    for label in ("TRPO (ours)", "TRPO (GFN-PG regime)"):
        if any(row["label"] == label for row in summary_rows):
            left_labels.append(label)
            right_labels.append(label)
    panels = {
        "left": left_labels,
        "right": right_labels,
    }

    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.0), squeeze=False)
    axes = axes.flatten()
    legend_handles: dict[str, object] = {}
    metric_specs = [
        ("elbo", "ELBO"),
        ("iw_eubo", "Dataset-based EUBO"),
    ]
    axis_index = 0
    for panel_idx, panel in enumerate(("left", "right")):
        panel_rows = [row for row in summary_rows if row["panel"] == panel]
        for metric, ylabel in metric_specs:
            axis = axes[axis_index]
            ymin = elbo_ymin if metric == "elbo" else eubo_ymin
            ymax = elbo_ymax if metric == "elbo" else eubo_ymax
            plot_panel(
                axis,
                panel_rows,
                panels[panel],
                metric,
                batch_size,
                max_trajectories,
                legend_handles,
                ymin,
                ymax,
                align_final_k8_to_xmax,
                align_final_k4_to_xmax,
            )
            axis.set_ylabel(ylabel)
            axis.set_title("")
            if metric == "elbo" and include_elbo_zoom:
                add_elbo_zoom(
                    axis,
                    panel_rows,
                    panels[panel],
                    batch_size,
                    max_trajectories,
                    right_elbo_zoom_top_n if panel == "right" and right_elbo_zoom_top_n is not None else elbo_zoom_top_n,
                    ymin=right_elbo_zoom_ymin if panel == "right" else left_elbo_zoom_ymin,
                    ymax=right_elbo_zoom_ymax if panel == "right" else left_elbo_zoom_ymax,
                    align_final_k8_to_xmax=align_final_k8_to_xmax,
                    align_final_k4_to_xmax=align_final_k4_to_xmax,
                    xmin=elbo_zoom_xmin,
                    selection_threshold=elbo_zoom_threshold,
                )
            axis_index += 1

    if match_eubo_ylims:
        axes[1].set_ylim(axes[3].get_ylim())

    baseline_order = ["DB", "TB", "VarGrad", "SubTB", "TRPO (ours)", "TRPO (GFN-PG regime)"]
    ordered_labels = baseline_order + [label for label in left_labels if label not in baseline_order]
    legend_keys = [label for label in ordered_labels if label in legend_handles]
    handles = [
        Line2D(
            [0], [0],
            color=COLORS[label],
            linestyle=(0, (3, 2)) if label in BASELINE_LABELS else "-",
            linewidth=3.0,
        )
        for label in legend_keys
    ]
    trpo_count = sum(label.startswith("TRPO") for label in legend_keys)
    labels = ["TRPO" if label.startswith("TRPO") and trpo_count == 1 else label
              for label in legend_keys]
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        columnspacing=0.8,
        handletextpad=0.35,
        handlelength=2.8,
    )
    for line in legend.get_lines():
        line.set_linewidth(3.0)

    fig.text(0.5, 0.01, FIGURE_NOTE, ha="center", fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 0.86], w_pad=0.7)
    fig.savefig(output_path, format=output_path.suffix.lstrip(".") or "pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global EUBO_SOURCE, FIGURE_NOTE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["seh", "qm9"], required=True)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="Task directory containing db/, tb/, subtb/, and ent_ppo_*/.")
    parser.add_argument("--trpo-root", type=Path, help="Optional task directory with TRPO runs.")
    parser.add_argument("--trpo-regimes", nargs="+", choices=["ours", "gfn_pg"], default=["ours", "gfn_pg"])
    parser.add_argument("--vargrad-root", type=Path, help="Optional task directory containing vargrad/ and vargrad_k4/.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-name", default=None)
    parser.add_argument("--max-seed", type=int, default=2)
    parser.add_argument("--min-seeds", type=int, default=3,
                        help="Minimum seeds at each plotted point (default: all three).")
    parser.add_argument("--max-trajectories", type=int, default=150000)
    parser.add_argument("--elbo-zoom-ymin", type=float, default=None,
                        help="ELBO inset lower bound; include every mean curve reaching it in the zoom interval.")
    parser.add_argument("--elbo-zoom-ymax", type=float, default=None,
                        help="ELBO inset upper bound.")
    parser.add_argument("--right-elbo-zoom-ymin", type=float, default=None,
                        help="Optional lower bound overriding --elbo-zoom-ymin for the right ELBO inset.")
    parser.add_argument("--elbo-zoom-threshold", type=float, default=None,
                        help="Include every curve whose mean ELBO exceeds this value in the inset.")
    parser.add_argument("--elbo-ylim", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Shared y-axis limits for both main ELBO panels.")
    parser.add_argument("--eubo-ylim", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Shared y-axis limits for both EUBO panels.")
    parser.add_argument("--match-eubo-ylims", action="store_true",
                        help="Use the right EUBO panel's y-axis limits for both EUBO panels.")
    parser.add_argument("--no-elbo-zoom", action="store_true",
                        help="Do not draw ELBO inset panels.")
    parser.add_argument("--no-x-cut", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_seeds <= args.max_seed + 1:
        parser.error("Require 1 <= min-seeds <= max-seed + 1")
    batch = 256 if args.task == "seh" else 128
    EUBO_SOURCE = "iw_eubo_893" if args.task == "seh" else "iw_eubo"
    specs = []
    for panel in ("left", "right"):
        for alg in ("db", "tb", "subtb"):
            specs.append((panel, {"db": "DB", "tb": "TB", "subtb": "SubTB"}[alg],
                          args.project_root / (alg + ("_k4" if panel == "right" else ""))))
        for k in ((1, 2, 4, 8) if panel == "left" else (4, 8)):
            label = rf"Ent-PPO ($K = {k}$)" + (" / VPG (GAE)" if k == 1 else "")
            specs.append((panel, label, args.project_root /
                f"ent_ppo_bs{batch}_clip0.2_gae0.7_vs8_pu{k}_vu4_kl1_vl1_vlr0.333333_norm0"))
        if args.vargrad_root:
            specs.append((panel, "VarGrad", args.vargrad_root / ("vargrad_k4" if panel == "right" else "vargrad")))
    if args.trpo_root:
        for regime, label in (("ours", "TRPO (ours)"), ("gfn_pg", "TRPO (GFN-PG regime)")):
            if regime not in args.trpo_regimes:
                continue
            matches = sorted(p for p in args.trpo_root.glob(f"trpo_{regime}_*") if p.is_dir())
            if len(matches) > 1:
                raise ValueError(f"Multiple TRPO configurations for {regime}; use a root with one per regime")
            if matches:
                specs.append(("left", label, matches[0]))
    rows, progress = [], []
    for panel, label, directory in specs:
        for seed in range(args.max_seed + 1):
            log = directory / f"seed_{seed}" / "train.log"
            cfg_path = log.with_name("config.yaml")
            parsed = []
            if log.exists():
                cfg = yaml.safe_load(cfg_path.read_text())
                temp = cfg["cond"]["temperature"]
                if temp["sample_dist"] != "constant" or temp["dist_params"] != [16.0]:
                    raise ValueError(f"Expected fixed beta=16: {cfg_path}")
                if cfg["algo"]["num_from_policy"] != batch:
                    raise ValueError(f"Unexpected training batch size: {cfg_path}")
                parsed = parse_validation_log(log, label, panel, seed, batch)
                if any(row[key] == "" for row in parsed for key in METRIC_KEYS):
                    raise ValueError(f"Missing ELBO or {EUBO_SOURCE}: {log}")
                # Resumed logs may repeat a validation step: use the latest record.
                parsed = list({row["iteration"]: row for row in parsed}.values())
                rows.extend(parsed)
            progress.append(dict(panel=panel, label=label, seed=seed, log=str(log),
                                 last_iteration=max((r["iteration"] for r in parsed), default=0)))
    summary = [r for r in summarize(rows) if r["num_seeds"] >= args.min_seeds]
    if not summary:
        raise SystemExit("No validation points meet --min-seeds; no figure generated.")
    xmax = max(r["num_trajectories"] for r in summary) if args.no_x_cut else args.max_trajectories
    counts = sorted({r["num_seeds"] for r in summary if r["num_trajectories"] <= xmax})
    coverage = str(counts[0]) if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
    snapshot = "partial training snapshot" if any(p["last_iteration"] < 1000 for p in progress) else "1000 iterations complete"
    FIGURE_NOTE = (f"{args.task.upper()} · fixed β=16 · mean and min–max, n={coverage} seeds per point · "
                   f"baseline K=1 (left), K=4 (right) · {snapshot}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", rows,
              ["panel", "label", "seed", "iteration", "num_trajectories", *METRIC_KEYS])
    write_csv(args.output_dir / "progress.csv", progress,
              ["panel", "label", "seed", "log", "last_iteration"])
    write_csv(args.output_dir / "summary.csv", summary,
              ["panel", "label", "iteration", "num_trajectories", "num_seeds",
               *[f"{m}_{s}" for m in METRIC_KEYS for s in ("mean", "std", "min", "max")]])
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "command": sys.argv, "eubo_source": EUBO_SOURCE, "beta": 16,
        "batch_size": batch, "band": "min-max", "min_seeds": args.min_seeds,
        "endpoint_stretching": False, "elbo_zoom_ymin": args.elbo_zoom_ymin,
        "elbo_zoom_ymax": args.elbo_zoom_ymax,
        "right_elbo_zoom_ymin": args.right_elbo_zoom_ymin,
        "elbo_zoom_threshold": args.elbo_zoom_threshold,
        "elbo_ylim": args.elbo_ylim, "eubo_ylim": args.eubo_ylim,
        "match_eubo_ylims": args.match_eubo_ylims,
        "include_elbo_zoom": not args.no_elbo_zoom,
        "sources": [str(p) for _, _, p in specs],
    }, indent=2) + "\n")
    output = args.output_dir / (args.plot_name or f"{args.task}_fixedpb_combined_k_panels.pdf")
    for path in (output, output.with_suffix(".png")):
        make_plot(summary, path, batch, xmax,
                  elbo_ymin=args.elbo_ylim[0] if args.elbo_ylim else None,
                  elbo_ymax=args.elbo_ylim[1] if args.elbo_ylim else None,
                  eubo_ymin=args.eubo_ylim[0] if args.eubo_ylim else None,
                  eubo_ymax=args.eubo_ylim[1] if args.eubo_ylim else None,
                  match_eubo_ylims=args.match_eubo_ylims,
                  include_elbo_zoom=not args.no_elbo_zoom,
                  elbo_zoom_threshold=args.elbo_zoom_threshold,
                  left_elbo_zoom_ymin=args.elbo_zoom_ymin,
                  left_elbo_zoom_ymax=args.elbo_zoom_ymax,
                  right_elbo_zoom_ymin=(args.right_elbo_zoom_ymin
                                         if args.right_elbo_zoom_ymin is not None
                                         else args.elbo_zoom_ymin),
                  right_elbo_zoom_ymax=args.elbo_zoom_ymax)
        print(path)
    print(f"{len(summary)} plotted points; n={coverage}; EUBO={EUBO_SOURCE}")


if __name__ == "__main__":
    main()
