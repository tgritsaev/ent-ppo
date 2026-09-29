"""Launch both TRPO critic regimes on sEH and QM9 (seed 0 by default)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from slurm_options import add_slurm_options, slurm_cli
from trpo_options import add_trpo_options, trpo_cli
import shlex


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', choices=['seh', 'qm9'], default=['seh', 'qm9'])
    p.add_argument('--regimes', nargs='+', choices=['ours', 'gfn_pg'], default=['ours', 'gfn_pg'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--log-root', default='./runs')
    p.add_argument('--project-prefix', default='trpo_fixedpb')
    p.add_argument('--env', default=sys.prefix)
    p.add_argument('--num-training-steps', type=int, default=1000)
    p.add_argument('--validate-every', type=int, default=25)
    p.add_argument('--valid-num-eval-trajectories', type=int, default=2048)
    p.add_argument('--time', default='48:00:00')
    p.add_argument('--seh-large-test-mols-path', default='data/binary_test_mols.pkl')
    p.add_argument('--qm9-h5-path', default='qm9.h5')
    p.add_argument('--qm9-model-path', default='mxmnet_gap_model.pt')
    p.add_argument('--dry-run', action='store_true')
    add_slurm_options(p)
    add_trpo_options(p)
    args = p.parse_args()
    commands=[]
    launcher=Path(__file__).with_name('launch_seh_metrics_sbatch.py')
    for task in args.tasks:
        for regime in args.regimes:
            args.trpo_critic_regime=regime
            batch=256 if task=='seh' else 128
            c=[sys.executable,str(launcher),'--task',task,'--algs','trpo',
               '--project',f'{args.project_prefix}_{task}','--log-root',args.log_root,
               '--env',args.env,'--seeds',*map(str,args.seeds),
               '--batch-size',str(batch),'--valid-batch-size',str(batch),
               '--num-training-steps',str(args.num_training_steps),'--validate-every',str(args.validate_every),
               '--valid-num-eval-trajectories',str(args.valid_num_eval_trajectories),'--time',args.time,
               '--backward-approach','uniform','--random-action-schedule','zero',
               '--seh-large-test-mols-path',args.seh_large_test_mols_path,
               '--qm9-h5-path',args.qm9_h5_path,'--qm9-model-path',args.qm9_model_path,
               *shlex.split(trpo_cli(args)), *slurm_cli(args)]
            if args.dry_run:c.append('--dry-run')
            commands.append(c)
    if not args.dry_run:
        status=Path(args.log_root)/'trpo_grid_status';status.mkdir(parents=True,exist_ok=True)
        (status/f'{datetime.now():%Y%m%d_%H%M%S}.json').write_text(json.dumps(commands,indent=2)+'\n')
    for c in commands:subprocess.run(c,check=True)


if __name__=='__main__':main()
