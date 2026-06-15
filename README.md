# Proximal Policy Optimization for Amortized Discrete Sampling

This repository contains the code for the paper Proximal Policy Optimization for Amortized Discrete Sampling (Poster at ICML 2026, SPIGM Workshop).

This publication snapshot currently contains the molecule experiments under `mols/`. The `mols/` code is based on [`recursionpharma/gflownet`](https://github.com/recursionpharma/gflownet) and includes the code for sEH and QM9 experiments.

## Layout

```text
mols/   PyTorch GFlowNet molecule-generation experiments for sEH and QM9
```

Important entry points:

- `mols/scripts/run_single_seh_metrics.py`: run one sEH or QM9 job locally.
- `mols/scripts/launch_seh_metrics_sbatch.py`: submit one algorithm group to Slurm.
- `mols/scripts/submit_fixed_pb_grid.py`: submit the fixed-backward-policy paper grid.
- `mols/scripts/submit_qm9_learned_pb_tlm_lr_grid.py`: submit the QM9 learned-backward-policy/TLM grid.

## Experiments on molecules

### Installation

Run from the repository root:

```bash
cd mols
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

The QM9 experiments expect these files in `mols/` unless another path is passed on the command line:

```text
qm9.h5
mxmnet_gap_model.pt
```

They are included in this repository.

### Quick runs

Run these from `mols/` after installation. The commands write logs under `./runs` by default unless `--log-root` is changed.

```bash
python scripts/run_single_seh_metrics.py \
  --task seh \
  --alg tb \
  --seed 0 \
  --num-training-steps 100
```

```bash
python scripts/run_single_seh_metrics.py \
  --task seh \
  --alg ent_ppo \
  --seed 0 \
  --ent-ppo-policy-updates 4 \
  --ent-ppo-value-updates 4 \
  --num-training-steps 100
```

```bash
python scripts/run_single_seh_metrics.py \
  --task qm9 \
  --alg tb \
  --seed 0 \
  --qm9-h5-path qm9.h5 \
  --qm9-model-path mxmnet_gap_model.pt \
  --num-training-steps 100
```

### Reproduce experiments from the paper

This submits the main fixed-`Pb` runs for sEH and QM9: TB, DB, SubTB baselines with `K=1` and `K=4`, and Ent-PPO with `K=1,2,4,8`.

The command is a Slurm launcher. Replace `/path/to/env` with the Python environment that has `mols` installed.

```bash
cd mols
python scripts/submit_fixed_pb_grid.py \
  --seh-project seh_fixedpb_eps_grid \
  --qm9-project qm9_fixedpb_eps_grid \
  --log-root ./runs \
  --env /path/to/env \
  --num-training-steps 1000 \
  --validate-every 25 \
  --valid-num-eval-trajectories 2048 \
  --ent-ppo-value-num-splits 4 \
  --qm9-h5-path qm9.h5 \
  --qm9-model-path mxmnet_gap_model.pt
```

Use `--dry-run` to print the generated `sbatch` commands without submitting jobs.