"""Print or execute the 42 completed Ising table configurations (no scheduler needed)."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("db", "tb", "vargrad", "subtb", "trpo", "ppo_k1", "ppo_k2")


def command(method, beta, seed, output, smoke=False):
    script = (
        "TRPO_ising_bitseq.py"
        if method == "trpo"
        else "PPO_ising_bitseq.py" if method.startswith("ppo") else "balance_ising_bitseq.py"
    )
    overrides = [
        f"seed={seed}",
        f"environment.beta={beta}",
        f"hydra.run.dir={output}",
        f'logging.log_dir={output / "metrics"}',
        f'logging.checkpoint_dir={output / "checkpoints"}',
    ]
    if method.startswith("ppo"):
        k = int(method[-1])
        # K=1 table results came from the earlier clip=0.02 sweep.
        overrides += [f"agent.ppo_policy_epochs={k}", f"agent.clip_eps={0.02 if k == 1 else 0.2}"]
    elif method != "trpo":
        overrides += [f"agent.objective={method}"]
    if smoke:
        overrides += [
            "num_train_steps=2",
            "num_envs=4",
            "network.num_layers=1",
            "network.embedding_size=8",
            "network.hidden_size=8",
            "network.intermediate_size=8",
            "network.num_heads=2",
            "metrics.n_rounds=1",
            "metrics.batch_size=4",
            "metrics.final_batch_size=4",
            "metrics.compute_sinkhorn=false",
            "metrics.sinkhorn_chunk_size=4",
            "logging.eval_each=1",
            "logging.track_each=1",
            "logging.tqdm_print_rate=1",
            "logging.metric_buffer_size=10",
        ]
        if method.startswith("ppo"):
            overrides += ["agent.ppo_baseline_num_splits=2"]
    return [sys.executable, str(ROOT / "baselines" / script), *overrides]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--betas", nargs="+", choices=["0.4407", "0.6"], default=["0.4407", "0.6"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--output", type=Path, default=ROOT / "runs/ising_paper")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Execute sequentially; otherwise only print commands.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny networks/batches for integration checks, not paper results.",
    )
    a = p.parse_args()
    runs = []
    for method in a.methods:
        for beta in a.betas:
            for seed in a.seeds:
                out = a.output.resolve() / method / f"beta_{beta}" / f"seed_{seed}"
                cmd = command(method, beta, seed, out, a.smoke)
                print(shlex.join(cmd), flush=True)
                runs.append(
                    dict(
                        method=method,
                        beta=float(beta),
                        seed=seed,
                        output_dir=str(out),
                        command=cmd,
                    )
                )
    if not a.execute:
        return
    # Refuse to overwrite existing outputs, including a partially completed run.
    existing = [r["output_dir"] for r in runs if Path(r["output_dir"]).exists()]
    if existing:
        p.error(f"Output directories already exist: {existing}")
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "manifest.json").write_text(
        json.dumps(dict(smoke=a.smoke, runs=runs), indent=2) + "\n"
    )
    for run in runs:
        subprocess.run(run["command"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
