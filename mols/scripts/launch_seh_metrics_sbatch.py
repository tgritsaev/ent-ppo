import argparse
import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path

from slurm_options import add_slurm_options
from trpo_options import add_trpo_options, trpo_name, trpo_cli


ALGS = ["tb", "db", "subtb", "ent_ppo", "vargrad", "trpo"]
DEFAULT_ALGS = ["tb", "db", "subtb", "ent_ppo"]
SEEDS = [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_slurm_options(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", choices=["seh", "qm9"], default="seh")
    parser.add_argument("--project", default=f"seh_metrics_256_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--algs", nargs="+", default=DEFAULT_ALGS, choices=ALGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--valid-batch-size", type=int, default=256)
    parser.add_argument("--valid-num-eval-trajectories", type=int, default=2048)
    parser.add_argument("--valid-num-eval-dataset-trajectories", type=int, default=None)
    parser.add_argument("--valid-use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--valid-ema-tau", type=float, default=0.95)
    parser.add_argument("--debug-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--slow-stage-seconds", type=float, default=60.0)
    parser.add_argument("--log-generated-objs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-backward-steps", type=int, default=None)
    parser.add_argument("--baseline-k", type=int, default=1)
    parser.add_argument("--random-action-schedule", choices=["zero", "linear_half"], default="zero")
    parser.add_argument("--random-action-prob", type=float, default=0.05)
    parser.add_argument("--backward-approach", choices=["uniform", "naive", "tlm"], default="uniform")
    parser.add_argument("--backward-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--backward-lr-decay", type=float, default=20_000)
    parser.add_argument("--backward-ema-tau", type=float, default=0.95)
    parser.add_argument("--use-backward-ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-training-steps", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--ent-ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-ppo-gae-lambda", type=float, default=0.7)
    parser.add_argument("--ent-ppo-policy-updates", type=int, default=4)
    parser.add_argument("--ent-ppo-backward-updates", type=int, default=None)
    parser.add_argument("--ent-ppo-value-updates", type=int, default=4)
    parser.add_argument("--ent-ppo-value-num-splits", type=int, default=8)
    parser.add_argument("--ent-ppo-kl-coeff", type=float, default=1.0)
    parser.add_argument("--ent-ppo-value-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--ent-ppo-value-lr-multiplier", type=float, default=1 / 3)
    parser.add_argument("--ent-ppo-normalize-advantages", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time", default="48:00:00")
    parser.add_argument("--log-root", default="./runs")
    parser.add_argument("--env", default=sys.prefix)
    parser.add_argument("--seh-large-test-mols-path", default="data/binary_test_mols.pkl")
    parser.add_argument("--qm9-h5-path", default="qm9.h5")
    parser.add_argument("--qm9-model-path", default="mxmnet_gap_model.pt")
    parser.add_argument("--qm9-rdkit-conformer-timeout-seconds", type=int, default=0)
    add_trpo_options(parser)
    args = parser.parse_args()
    if "trpo" in args.algs and (args.baseline_k != 1 or args.backward_approach != "uniform" or args.random_action_schedule != "zero"):
        parser.error("TRPO requires baseline-k=1, uniform Pb, and zero exploration")

    repo_root = str(Path(os.getcwd()).resolve())
    project_root = Path(args.log_root) / args.project
    slurm_log_root = project_root / "slurm"
    if not args.dry_run:
        slurm_log_root.mkdir(parents=True, exist_ok=True)

    job_count = 0
    for alg in args.algs:
        for seed in args.seeds:
            ent_ppo_suffix = ""
            if alg == "ent_ppo":
                ent_ppo_suffix = (
                    f"-clip{args.ent_ppo_clip_eps:g}"
                    f"-gae{args.ent_ppo_gae_lambda:g}"
                    f"-vs{args.ent_ppo_value_num_splits}"
                    f"-pu{args.ent_ppo_policy_updates}"
                    + (
                        f"-bu{args.ent_ppo_backward_updates}"
                        if args.ent_ppo_backward_updates is not None
                        and args.ent_ppo_backward_updates != args.ent_ppo_policy_updates
                        else ""
                    )
                    + (
                    f"-vu{args.ent_ppo_value_updates}"
                    f"-kl{args.ent_ppo_kl_coeff:g}"
                    f"-vl{args.ent_ppo_value_loss_multiplier:g}"
                    f"-vlr{args.ent_ppo_value_lr_multiplier:g}"
                    f"-norm{int(args.ent_ppo_normalize_advantages)}"
                    )
                )
            backward_suffix = f"-pb{args.backward_approach}" if args.backward_approach != "uniform" else ""
            baseline_suffix = f"-k{args.baseline_k}" if alg != "ent_ppo" and args.baseline_k != 1 else ""
            random_suffix = ""
            random_action_schedule = args.random_action_schedule if alg != "ent_ppo" else "zero"
            if alg != "ent_ppo" and random_action_schedule == "linear_half":
                random_suffix = f"-eps{args.random_action_prob:g}-linhalf"
            job_name = (
                f"{args.task}-{alg}-bs{args.batch_size}{baseline_suffix}{ent_ppo_suffix}"
                f"{backward_suffix}{random_suffix}-s{seed}"
            )
            if alg == "trpo":
                job_name = f"{args.task}-{trpo_name(args)}-bs{args.batch_size}-s{seed}"
            sbatch = [
                "sbatch",
                f"--job-name={job_name}",
                f"--time={args.time}",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={args.cpus_per_task}",
                "--gpus-per-task=1",
                f"--mem={args.mem}",
                f"--output={slurm_log_root / (job_name + '-%j.out')}",
                f"--error={slurm_log_root / (job_name + '-%j.err')}",
            ]
            if args.partition:
                sbatch.append(f"--partition={args.partition}")
            if args.account:
                sbatch.append(f"--account={args.account}")
            if args.exclude:
                sbatch.append(f"--exclude={args.exclude}")
            valid_num_eval_dataset_trajectories = (
                args.valid_num_eval_dataset_trajectories
                if args.valid_num_eval_dataset_trajectories is not None
                else args.valid_num_eval_trajectories
            )

            python_cmd = (
                f"cd {shlex.quote(os.getcwd())} && "
                "GIT_CONFIG_COUNT=1 "
                "GIT_CONFIG_KEY_0=safe.directory "
                f"GIT_CONFIG_VALUE_0={shlex.quote(repo_root)} "
                f"{shlex.quote(args.env)}/bin/python scripts/run_single_seh_metrics.py "
                f"--task {args.task} "
                f"--alg {shlex.quote(alg)} "
                f"--seed {seed} "
                f"--batch-size {args.batch_size} "
                f"--valid-batch-size {args.valid_batch_size} "
                f"--valid-num-eval-trajectories {args.valid_num_eval_trajectories} "
                f"--valid-num-eval-dataset-trajectories {valid_num_eval_dataset_trajectories} "
                f"{'--valid-use-ema' if args.valid_use_ema else '--no-valid-use-ema'} "
                f"--valid-ema-tau {args.valid_ema_tau} "
                f"{'--debug-timing' if args.debug_timing else '--no-debug-timing'} "
                f"--slow-stage-seconds {args.slow_stage_seconds} "
                f"{'--log-generated-objs' if args.log_generated_objs else '--no-log-generated-objs'} "
                + (f"--max-backward-steps {args.max_backward_steps} " if args.max_backward_steps is not None else "")
                + (
                f"--baseline-k {args.baseline_k} "
                f"--random-action-schedule {random_action_schedule} "
                f"--random-action-prob {args.random_action_prob} "
                f"--backward-approach {args.backward_approach} "
                f"--backward-lr-multiplier {args.backward_lr_multiplier} "
                f"--backward-lr-decay {args.backward_lr_decay} "
                f"--backward-ema-tau {args.backward_ema_tau} "
                f"{'--use-backward-ema' if args.use_backward_ema else '--no-use-backward-ema'} "
                f"--num-training-steps {args.num_training_steps} "
                f"--validate-every {args.validate_every} "
                f"--ent-ppo-clip-eps {args.ent_ppo_clip_eps} "
                f"--ent-ppo-gae-lambda {args.ent_ppo_gae_lambda} "
                f"--ent-ppo-policy-updates {args.ent_ppo_policy_updates} "
                + (
                    f"--ent-ppo-backward-updates {args.ent_ppo_backward_updates} "
                    if args.ent_ppo_backward_updates is not None
                    else ""
                )
                + (
                f"--ent-ppo-value-updates {args.ent_ppo_value_updates} "
                f"--ent-ppo-value-num-splits {args.ent_ppo_value_num_splits} "
                f"--ent-ppo-kl-coeff {args.ent_ppo_kl_coeff} "
                f"--ent-ppo-value-loss-multiplier {args.ent_ppo_value_loss_multiplier} "
                f"--ent-ppo-value-lr-multiplier {args.ent_ppo_value_lr_multiplier} "
                f"{'--ent-ppo-normalize-advantages' if args.ent_ppo_normalize_advantages else '--no-ent-ppo-normalize-advantages'} "
                f"--project {shlex.quote(args.project)} "
                f"--log-root {shlex.quote(args.log_root)} "
	                f"--seh-large-test-mols-path {shlex.quote(args.seh_large_test_mols_path)} "
	                f"--qm9-h5-path {shlex.quote(args.qm9_h5_path)} "
	                f"--qm9-model-path {shlex.quote(args.qm9_model_path)} "
	                f"--qm9-rdkit-conformer-timeout-seconds {args.qm9_rdkit_conformer_timeout_seconds} "
                    + (trpo_cli(args) if alg == "trpo" else "")
	                )
	                )
	            )
            command = " ".join(shlex.quote(part) for part in sbatch) + " --wrap=" + shlex.quote(python_cmd)
            if args.dry_run:
                print(command)
            else:
                subprocess.run([*sbatch, "--wrap", python_cmd], check=True)
            job_count += 1

    print(f"Total jobs {'to print' if args.dry_run else 'submitted'}: {job_count}")
    print(f"Project dir: {project_root}")


if __name__ == "__main__":
    main()
