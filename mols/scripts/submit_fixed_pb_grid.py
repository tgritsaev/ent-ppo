import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def add_common(cmd: list[str], args: argparse.Namespace, task: str, project: str, batch_size: int) -> list[str]:
    cmd.extend(
        [
            "--task",
            task,
            "--project",
            project,
            "--batch-size",
            str(batch_size),
            "--valid-batch-size",
            str(batch_size),
            "--num-training-steps",
            str(args.num_training_steps),
            "--validate-every",
            str(args.validate_every),
            "--valid-num-eval-trajectories",
            str(args.valid_num_eval_trajectories),
            "--time",
            args.time,
            "--log-root",
            args.log_root,
            "--env",
            args.env,
        ]
    )
    if task == "qm9":
        cmd.extend(["--qm9-h5-path", args.qm9_h5_path, "--qm9-model-path", args.qm9_model_path])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seh-project", default="seh_fixedpb_grid")
    parser.add_argument("--qm9-project", default="qm9_fixedpb_grid")
    parser.add_argument("--log-root", default="./runs")
    parser.add_argument("--env", default=sys.prefix)
    parser.add_argument("--seh-batch-size", type=int, default=256)
    parser.add_argument("--qm9-batch-size", type=int, default=128)
    parser.add_argument("--num-training-steps", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--valid-num-eval-trajectories", type=int, default=2048)
    parser.add_argument("--ent-ppo-value-num-splits", type=int, default=4)
    parser.add_argument("--ent-ppo-ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--time", default="48:00:00")
    parser.add_argument("--qm9-h5-path", default="qm9.h5")
    parser.add_argument("--qm9-model-path", default="mxmnet_gap_model.pt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script = Path(__file__).with_name("launch_seh_metrics_sbatch.py")
    commands: list[list[str]] = []
    for task, project, batch_size in (
        ("seh", args.seh_project, args.seh_batch_size),
        ("qm9", args.qm9_project, args.qm9_batch_size),
    ):
        for baseline_k in (1, 4):
            cmd = [
                sys.executable,
                str(script),
                "--algs",
                "tb",
                "db",
                "subtb",
                "--baseline-k",
                str(baseline_k),
                "--random-action-schedule",
                "zero",
            ]
            commands.append(add_common(cmd, args, task, project, batch_size))
        for k in args.ent_ppo_ks:
            cmd = [
                sys.executable,
                str(script),
                "--algs",
                "ent_ppo",
                "--ent-ppo-policy-updates",
                str(k),
                "--ent-ppo-value-updates",
                str(k),
                "--ent-ppo-value-num-splits",
                str(args.ent_ppo_value_num_splits),
            ]
            commands.append(add_common(cmd, args, task, project, batch_size))

    status_dir = Path(args.log_root) / "fixedpb_grid_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    manifest = status_dir / f"submit_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with manifest.open("w", encoding="utf8") as handle:
        for cmd in commands:
            handle.write(" ".join(cmd) + "\n")

    for cmd in commands:
        subprocess.run(cmd, check=True)
    print(f"Submitted command groups: {len(commands)}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
