"""Aggregate final Ising metrics from run_ising_paper.py into a CSV table.

Use --legacy-correlation only on logs from before the 1/(2L) normalization fix.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import math

KEYS = {
    "elbo": "elbo/elbo",
    "eubo": "eubo/eubo",
    "corr_error": "physics/corr_error",
    "mag_error": "physics/mag_error",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root", type=Path)
    p.add_argument("--legacy-correlation", action="store_true")
    a = p.parse_args()
    manifest = json.loads((a.run_root / "manifest.json").read_text())
    if manifest.get("smoke"):
        p.error("Smoke runs must not be aggregated as paper results.")
    groups = defaultdict(list)
    seen = set()
    for run in manifest["runs"]:
        identity = (run["method"], run["beta"], run["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate run: {identity}")
        seen.add(identity)
        directory = Path(run["output_dir"])
        config = json.loads((directory / "metrics/config.json").read_text())
        records = [
            json.loads(line)
            for line in (directory / "metrics/log.jsonl").read_text().splitlines()
            if line.strip()
        ]
        final = [r for r in records if "final/elbo/elbo" in r]
        if len(final) != 1 or final[0]["step"] != config["num_train_steps"]:
            raise ValueError(f"Missing/duplicate/incomplete final evaluation: {directory}")
        values = {key: float(final[0][f"final/{path}"]) for key, path in KEYS.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Nonfinite metrics: {directory}")
        if a.legacy_correlation:
            values["corr_error"] *= 2
        groups[(run["method"], run["beta"])].append(values)
    rows = []
    for (method, beta), values in groups.items():
        row = dict(method=method, beta=beta, n_seeds=len(values))
        for key in KEYS:
            v = [x[key] for x in values]
            row[key + "_mean"] = statistics.mean(v)
            row[key + "_std"] = statistics.stdev(v) if len(v) > 1 else ""
        rows.append(row)
    if not rows:
        p.error("No runs in manifest.")
    output = a.run_root / "final_summary.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
