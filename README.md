# Proximal Policy Optimization for Amortized Discrete Sampling

> [**Proximal Policy Optimization for Amortized Discrete Sampling**](https://github.com/tgritsaev/ent-ppo/),            
> Anna Zykova-Myzina*, Timofei Gritsaev*, Daniil Tiapkin†, Nikita Morozov†,
> *ICML 2026, SPIGM Workshop ([arXiv 2606.15793](https://arxiv.org/abs/2606.15793))*  

## Contact

Feel free to contact us if you have any questions about the paper!

- Anna Zykova-Myzina [azykova.myzina@gmail.com](mailto:azykova.myzina@gmail.com)
- Timofei Gritsaev [tgritsaev@gmail.com](mailto:tgritsaev@gmail.com)

## Abstract

This paper explores policy gradient algorithms for training stochastic policies to
sample from structured discrete probability distributions under the Generative Flow
Network (GFlowNet) framework. Building on extensive theoretical connections
between GFlowNets and entropy-regularized reinforcement learning, we derive
equivalents of standard policy gradient algorithms for training GFlowNets, as well
as experimentally explore their various methodological aspects, including baseline
training and advantage estimation. Most importantly, our work is the first to derive
and successfully apply proximal policy optimization to GFlowNets, showing its
improved convergence speed and data efficiency compared to standard GFlowNet
training objectives on benchmarks ranging from synthetic energies to molecular
graph generation.

## Structure

The structure is the following:
- **Synthetic problems: Hypergrid, TFBind8, and String QM9.** The code for experiments on synthetic problems is located in `gfnx/` 
and includes Hypergrid, TFBind8, and String QM9. The implementation is based on the JAX-based [`gfnx`](https://github.com/d-tiapkin/gfnx) library.
- **Molecular-graph generation: sEH and QM9.** The code for experiments on graph molecular problems is located in `mols/` and includes sEH and QM9 experiments. 
 The implementation is based on the [code of recursionpharma](https://github.com/recursionpharma/gflownet).

## Experiments on synthetic problems

Synthetic experiments cover three environments from the paper: **Hypergrid**, **TFBind8**, and **String QM9**. 
The code lives in `gfnx/` and uses the JAX-based [`gfnx`](https://github.com/d-tiapkin/gfnx) library.

### Installation

Requires Python 3.10+. For GPU/TPU support, follow the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) before running the commands below.

```bash
cd gfnx
pip install -e .[baselines]
```

### Quick run

All scripts must be run from the `gfnx/`.

Run a short Ent-PPO training on Hypergrid:

```bash
cd gfnx
python baselines/PPO_hypergrid.py num_train_steps=1000
```

### Reproduce experiments from the paper

The paper trains all methods for 31 250 iterations on Hypergrid, 200 000 on TFBind8, and 50 000 on String QM9, with 3 seeds each. All experiments run on CPU.

To reproduce a single run, use the default config and override the seed:

```bash
# VPG GAE on Hypergrid
python baselines/GAE_hypergrid.py seed=1

# VPG Reward-to-go on TFBind8
python baselines/RTG_pg_tfbind.py seed=2

# Ent-PPO on String QM9
python baselines/PPO_qm9_small.py seed=3
```

Online experiment tracking is disabled by default. To enable Comet ML logging, set the `COMET_API_KEY` environment variable and pass `writer.writer_type=comet_ml logging.use_writer=true` on the command line.

## Experiments on molecular-graph generation

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

# Citation

If you find Ent-PPO useful or relevant to your research, please kindly cite our papers:

```
@inproceedings{
  zykova-myzina2026proximal,
  title={Proximal Policy Optimization for Amortized Discrete Sampling},
  author={Anna Zykova-Myzina and Timofei Gritsaev and Daniil Tiapkin and Nikita Morozov},
  booktitle={ICML 2026 Workshop on Structured Probabilistic Inference {\&} Generative Modeling},
  year={2026},
  url={https://openreview.net/forum?id=ODbzTJmgp3}
}
```
