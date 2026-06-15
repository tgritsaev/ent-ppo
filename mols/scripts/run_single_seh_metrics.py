import argparse
import datetime
from pathlib import Path

import torch

from gflownet.algo.config import Backward, TBVariant
from gflownet.config import Config, init_empty
from gflownet.tasks.qm9 import QM9GapTrainer
from gflownet.tasks.seh_frag import SEHFragTrainer


TB_VARIANTS = {
    "tb": TBVariant.TB,
    "db": TBVariant.DB,
    "subtb": TBVariant.SubTB1,
}


def make_config(args: argparse.Namespace) -> Config:
    if args.alg == "ent_ppo" and args.backward_approach == "naive":
        raise ValueError("EntPPO supports backward_approach='uniform' or 'tlm', not 'naive'.")
    if args.alg == "ent_ppo" and args.random_action_schedule != "zero":
        raise ValueError("Random-action schedules are intended for TB/DB/SubTB baselines, not EntPPO.")
    cfg = init_empty(Config())
    alg_dir = args.alg
    pb_suffix = f"_pb-{args.backward_approach}" if args.backward_approach != "uniform" else ""
    random_suffix = ""
    if args.random_action_schedule == "linear_half":
        random_suffix = f"_eps{args.random_action_prob:g}_linhalf"
    if args.alg == "ent_ppo":
        alg_dir = (
            f"{args.alg}_bs{args.batch_size}"
            f"_clip{args.ent_ppo_clip_eps:g}"
            f"_gae{args.ent_ppo_gae_lambda:g}"
            f"_vs{args.ent_ppo_value_num_splits}"
            f"_pu{args.ent_ppo_policy_updates}"
            f"_vu{args.ent_ppo_value_updates}"
            f"_kl{args.ent_ppo_kl_coeff:g}"
            f"_vl{args.ent_ppo_value_loss_multiplier:g}"
            f"_vlr{args.ent_ppo_value_lr_multiplier:g}"
            f"_norm{int(args.ent_ppo_normalize_advantages)}"
            f"{pb_suffix}"
        )
    elif args.baseline_k != 1 or args.backward_approach != "uniform":
        alg_dir = f"{args.alg}_k{args.baseline_k}{pb_suffix}{random_suffix}"
    else:
        alg_dir = f"{args.alg}{random_suffix}"
    cfg.log_dir = str(Path(args.log_root) / args.project / alg_dir / f"seed_{args.seed}")
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.overwrite_existing_exp = True
    cfg.seed = args.seed
    cfg.num_workers = 0
    cfg.debug_timing = args.debug_timing
    cfg.slow_stage_seconds = args.slow_stage_seconds
    cfg.log_generated_objs = args.log_generated_objs
    cfg.print_every = args.print_every
    cfg.num_training_steps = args.num_training_steps
    cfg.validate_every = args.validate_every
    cfg.num_validation_gen_steps = 1
    cfg.num_final_gen_steps = 0
    cfg.pickle_mp_messages = False

    cfg.opt.learning_rate = 1e-4
    cfg.opt.weight_decay = 1e-8
    cfg.opt.momentum = 0.9
    cfg.opt.adam_eps = 1e-8
    cfg.opt.lr_decay = 20_000
    cfg.opt.clip_grad_type = "norm"
    cfg.opt.clip_grad_param = 10

    cfg.algo.method = "EntPPO" if args.alg == "ent_ppo" else "TB"
    cfg.algo.num_from_policy = args.batch_size
    cfg.algo.num_from_dataset = 0
    cfg.algo.valid_num_from_policy = args.valid_batch_size
    cfg.algo.valid_num_from_dataset = 64
    cfg.algo.valid_num_eval_trajectories = args.valid_num_eval_trajectories
    cfg.algo.valid_num_eval_dataset_trajectories = (
        args.valid_num_eval_dataset_trajectories
        if args.valid_num_eval_dataset_trajectories is not None
        else args.valid_num_eval_trajectories
    )
    cfg.algo.valid_use_ema = args.valid_use_ema
    cfg.algo.valid_ema_tau = args.valid_ema_tau
    cfg.algo.max_backward_steps = args.max_backward_steps
    cfg.algo.max_nodes = 9
    cfg.algo.max_len = 128
    cfg.algo.illegal_action_logreward = -75
    if args.random_action_schedule == "linear_half":
        cfg.algo.train_random_action_prob = args.random_action_prob
        cfg.algo.train_random_action_prob_anneal_steps = max(1, args.num_training_steps // 2)
    else:
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.train_random_action_prob_anneal_steps = None
    cfg.algo.valid_random_action_prob = 0.0
    cfg.algo.sampling_tau = 0.0
    cfg.algo.backward_approach = args.backward_approach
    cfg.algo.backward_learning_rate_multiplier = args.backward_lr_multiplier
    cfg.algo.backward_lr_decay = args.backward_lr_decay
    cfg.algo.backward_ema_tau = args.backward_ema_tau
    cfg.algo.use_backward_ema = args.use_backward_ema

    cfg.algo.tb.cum_subtb = True
    cfg.algo.tb.epsilon = None
    cfg.algo.tb.bootstrap_own_reward = False
    cfg.algo.tb.Z_learning_rate = 1e-3
    cfg.algo.tb.Z_lr_decay = 50_000
    cfg.algo.tb.do_parameterize_p_b = args.backward_approach != "uniform"
    cfg.algo.tb.do_sample_p_b = True
    cfg.algo.tb.do_length_normalize = False
    cfg.algo.tb.gradient_steps = args.baseline_k if args.alg != "ent_ppo" else 1
    if cfg.algo.tb.do_parameterize_p_b:
        cfg.algo.tb.backward_policy = Backward.Free
    if args.alg in TB_VARIANTS:
        cfg.algo.tb.variant = TB_VARIANTS[args.alg]

    cfg.algo.ent_ppo.clip_eps = args.ent_ppo_clip_eps
    cfg.algo.ent_ppo.gae_lambda = args.ent_ppo_gae_lambda
    cfg.algo.ent_ppo.policy_updates = args.ent_ppo_policy_updates
    cfg.algo.ent_ppo.backward_updates = args.ent_ppo_backward_updates
    cfg.algo.ent_ppo.value_updates = args.ent_ppo_value_updates
    cfg.algo.ent_ppo.value_num_splits = args.ent_ppo_value_num_splits
    cfg.algo.ent_ppo.value_learning_rate_multiplier = args.ent_ppo_value_lr_multiplier
    cfg.algo.ent_ppo.value_loss_multiplier = args.ent_ppo_value_loss_multiplier
    cfg.algo.ent_ppo.kl_coeff = args.ent_ppo_kl_coeff
    cfg.algo.ent_ppo.normalize_advantages = args.ent_ppo_normalize_advantages
    cfg.algo.ent_ppo.do_sample_p_b = True

    cfg.model.num_emb = 128
    cfg.model.num_layers = 4
    cfg.replay.use = False
    cfg.cond.temperature.sample_dist = "constant"
    cfg.cond.temperature.dist_params = [16.0]
    if args.task == "qm9":
        cfg.task.qm9.h5_path = args.qm9_h5_path
        cfg.task.qm9.model_path = args.qm9_model_path
        cfg.task.qm9.rdkit_conformer_timeout_seconds = args.qm9_rdkit_conformer_timeout_seconds
        cfg.cond.temperature.sample_dist = "uniform"
        cfg.cond.temperature.dist_params = [0.5, 32.0]
        cfg.cond.temperature.num_thermometer_dim = 32
    else:
        cfg.task.seh.large_test_mols_path = args.seh_large_test_mols_path
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["seh", "qm9"], default="seh")
    parser.add_argument("--alg", choices=["tb", "db", "subtb", "ent_ppo"], required=True)
    parser.add_argument("--seed", type=int, required=True)
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
    parser.add_argument("--print-every", type=int, default=10)
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
    parser.add_argument("--project", default=f"seh_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--log-root", default="./runs")
    parser.add_argument("--qm9-h5-path", default="qm9.h5")
    parser.add_argument("--qm9-model-path", default="mxmnet_gap_model.pt")
    parser.add_argument("--qm9-rdkit-conformer-timeout-seconds", type=int, default=0)
    parser.add_argument("--seh-large-test-mols-path", default="")
    args = parser.parse_args()

    cfg = make_config(args)
    print(
        f"RUN task={args.task} alg={args.alg} seed={args.seed} batch_size={args.batch_size} "
        f"baseline_k={args.baseline_k} backward_approach={args.backward_approach} "
        f"random_action_schedule={args.random_action_schedule} random_action_prob={cfg.algo.train_random_action_prob} "
        f"random_action_anneal_steps={cfg.algo.train_random_action_prob_anneal_steps} "
        f"num_from_policy={cfg.algo.num_from_policy} "
        f"gradient_steps={cfg.algo.tb.gradient_steps} "
        f"ent_ppo_policy_updates={cfg.algo.ent_ppo.policy_updates} "
        f"ent_ppo_backward_updates={cfg.algo.ent_ppo.backward_updates or cfg.algo.ent_ppo.policy_updates} "
        f"debug_timing={cfg.debug_timing} log_generated_objs={cfg.log_generated_objs} "
        f"slow_stage_seconds={cfg.slow_stage_seconds} max_backward_steps={cfg.algo.max_backward_steps} "
        f"log_dir={cfg.log_dir}",
        flush=True,
    )
    trainer_cls = QM9GapTrainer if args.task == "qm9" else SEHFragTrainer
    trainer = trainer_cls(cfg)
    trainer.run()
    print(f"RUN_DONE log_dir={cfg.log_dir}", flush=True)


if __name__ == "__main__":
    main()
