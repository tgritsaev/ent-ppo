"""Plot a snapshot of molecular validation curves, without smoothing or extrapolation."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

VALIDATION = re.compile(r"validation - iteration (\d+) : (.*)$")
METRIC = re.compile(r"([\w/]+):\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|[-+]?inf|nan)(?=\s|$)")
COLORS = {"TRPO": "black", "VarGrad": "#8B4513", "DB": "#0072B2", "TB": "#CC79A7", "SubTB": "#D55E00", 1: "#009E73", 2: "#E69F00", 4: "#56B4E9", 8: "#333333"}


def scalar(config, key):
    match = re.search(rf"^\s*{key}:\s*(\d+)\s*$", config, re.M)
    if match is None:
        raise ValueError(f"Missing numeric config key {key}")
    return int(match[1])


def load_runs(root, seed):
    runs = []
    for task in ("seh", "qm9"):
        for log in sorted((root / task).glob(f"*/seed_{seed}/train.log")):
            name = log.parent.parent.name
            ppo = name.startswith("ent_ppo_")
            method = "Ent-PPO" if ppo else {"db": "DB", "tb": "TB", "subtb": "SubTB", "vargrad": "VarGrad", "trpo": "TRPO"}[name.split("_")[0]]
            k = int(re.search(r"_pu(\d+)", name)[1]) if ppo else int(re.search(r"_k(\d+)", name)[1]) if "_k" in name else 1
            config = (log.parent / "config.yaml").read_text()
            batch = scalar(config, "num_from_policy")
            budget = scalar(config, "num_training_steps")
            eubo_key = "iw_eubo_893" if task == "seh" else "iw_eubo"
            points = {}
            for line in log.read_text().splitlines():
                match = VALIDATION.search(line)
                if match is None:
                    continue
                metrics = {key: float(value) for key, value in METRIC.findall(match[2])}
                if "elbo" not in metrics or eubo_key not in metrics:
                    raise ValueError(f"Missing ELBO/{eubo_key} in {log}, iteration {match[1]}")
                iteration = int(match[1])
                points[iteration] = {"iteration": iteration, "reward_evaluations": iteration * batch, "elbo": metrics["elbo"], "eubo": metrics[eubo_key]}
            if not points:
                continue
            runs.append(dict(task=task, method=method, k=k, seed=seed, batch_size=batch, budget=budget,
                             eubo_key=eubo_key, points=[points[i] for i in sorted(points)], log=str(log)))
    return runs


def panel(ax, runs, task, baseline_k, metric):
    selected = [r for r in runs if r["task"] == task and (r["method"] == "Ent-PPO" or r["k"] == baseline_k)]
    selected.sort(key=lambda r: ({"DB": 0, "TB": 1, "SubTB": 2, "VarGrad": 3, "TRPO": 4, "Ent-PPO": 5}[r["method"]], r["k"]))
    for r in selected:
        points = r["points"]
        partial = points[-1]["iteration"] < r["budget"]
        label = f'{r["method"]} K={r["k"]}' + (f' ({points[-1]["iteration"]} it.)' if partial else "")
        color = COLORS[r["k"]] if r["method"] == "Ent-PPO" else COLORS[r["method"]]
        ax.plot([p["reward_evaluations"] for p in points], [p[metric] for p in points], color=color,
                linestyle="-" if r["method"] == "Ent-PPO" else "--", linewidth=1.7, label=label,
                marker="o" if partial else None, markevery=[len(points)-1] if partial else None, markersize=4)
    task_label = "sEH" if task == "seh" else "QM9"
    ax.set_title(f"{task_label} · baselines K={baseline_k}", fontsize=11)
    ax.set_ylabel("ELBO ↑" if metric == "elbo" else "Dataset-based EUBO ↓")
    ax.set_xlabel("Training reward evaluations")
    ax.set_xlim(0, max(r["budget"] * r["batch_size"] for r in selected))
    ax.xaxis.set_major_formatter(EngFormatter(sep=""))
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=.85)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    runs = load_runs(args.log_root, args.seed)
    if not runs:
        raise ValueError("No validation curves found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    footer = "Raw validation curves · endpoint circles mark partial runs · sEH EUBO uses 893 molecules · no uncertainty bands (one seed)"
    for task in ("seh", "qm9", "combined"):
        if task == "combined":
            fig, axes = plt.subplots(2, 4, figsize=(19, 8), constrained_layout=True)
            for row, t in enumerate(("seh", "qm9")):
                for group, k in enumerate((1, 4)):
                    for col, metric in enumerate(("elbo", "eubo")):
                        panel(axes[row, group*2+col], runs, t, k, metric)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for row, k in enumerate((1, 4)):
                for col, metric in enumerate(("elbo", "eubo")):
                    panel(axes[row, col], runs, task, k, metric)
        fig.suptitle(f"ELBO / EUBO · seed {args.seed} · snapshot {stamp}\n{footer}", fontsize=11)
        for extension in ("png", "pdf"):
            fig.savefig(args.output_dir / f"{task}_elbo_eubo_seed{args.seed}.{extension}", dpi=180)
        plt.close(fig)
    with (args.output_dir / "validation_curves.csv").open("w", newline="") as f:
        fields = ["task", "method", "k", "seed", "eubo_key", "iteration", "reward_evaluations", "elbo", "eubo"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            for point in r["points"]:
                writer.writerow({**{key: r[key] for key in fields[:5]}, **point})
    with (args.output_dir / "run_progress.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "method", "k", "last_validation_iteration", "budget", "log"])
        for r in runs:
            writer.writerow([r["task"], r["method"], r["k"], r["points"][-1]["iteration"], r["budget"], r["log"]])
    print(f"Plotted {len(runs)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
