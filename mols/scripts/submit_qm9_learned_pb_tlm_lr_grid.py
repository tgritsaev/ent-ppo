import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def add_common(cmd: list[str], args: argparse.Namespace, project: str) -> list[str]:
    cmd.extend(
        [
            "--task",
            "qm9",
            "--project",
            project,
            "--batch-size",
            str(args.batch_size),
            "--valid-batch-size",
            str(args.batch_size),
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
            "--qm9-h5-path",
            args.qm9_h5_path,
            "--qm9-model-path",
            args.qm9_model_path,
            "--random-action-schedule",
            "zero",
        ]
    )
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def multiplier_suffix(multiplier: float) -> str:
    if multiplier == 1.0:
        return "blr1"
    if multiplier == 0.1:
        return "blr01"
    return f"blr{multiplier:g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-project", default="qm9_learnedpb_eps0_baselines")
    parser.add_argument("--tlm-project-prefix", default="qm9_learnedpb_eps0_tlm")
    parser.add_argument("--log-root", default="./runs")
    parser.add_argument("--env", default=sys.prefix)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-training-steps", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--valid-num-eval-trajectories", type=int, default=2048)
    parser.add_argument("--ent-ppo-value-num-splits", type=int, default=4)
    parser.add_argument("--ent-ppo-ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--baseline-backward-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--tlm-backward-lr-multipliers", nargs="+", type=float, default=[1.0, 0.1])
    parser.add_argument("--time", default="48:00:00")
    parser.add_argument("--qm9-h5-path", default="qm9.h5")
    parser.add_argument("--qm9-model-path", default="mxmnet_gap_model.pt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script = Path(__file__).with_name("launch_seh_metrics_sbatch.py")
    commands: list[list[str]] = []

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
            "--backward-approach",
            "naive",
            "--backward-lr-multiplier",
            str(args.baseline_backward_lr_multiplier),
        ]
        commands.append(add_common(cmd, args, args.baseline_project))

    for multiplier in args.tlm_backward_lr_multipliers:
        project = f"{args.tlm_project_prefix}_{multiplier_suffix(multiplier)}"
        for k in args.ent_ppo_ks:
            cmd = [
                sys.executable,
                str(script),
                "--algs",
                "ent_ppo",
                "--backward-approach",
                "tlm",
                "--backward-lr-multiplier",
                str(multiplier),
                "--ent-ppo-policy-updates",
                str(k),
                "--ent-ppo-value-updates",
                str(k),
                "--ent-ppo-value-num-splits",
                str(args.ent_ppo_value_num_splits),
            ]
            commands.append(add_common(cmd, args, project))

    status_dir = Path(args.log_root) / f"{args.tlm_project_prefix}_status"
    manifest = status_dir / f"submit_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
    else:
        status_dir.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf8") as handle:
            for cmd in commands:
                handle.write(" ".join(cmd) + "\n")

    for cmd in commands:
        subprocess.run(cmd, check=True)

    print(f"Submitted command groups: {len(commands)}")
    if not args.dry_run:
        print(f"Manifest: {manifest}")
    print(f"Baseline project: {Path(args.log_root) / args.baseline_project}")
    for multiplier in args.tlm_backward_lr_multipliers:
        project = f"{args.tlm_project_prefix}_{multiplier_suffix(multiplier)}"
        print(f"TLM project: {Path(args.log_root) / project}")


if __name__ == "__main__":
    main()
