# Paper-style reproduction plots

Run from `mols/` after installation; add plotting dependencies with
`python -m pip install matplotlib pyyaml`.
The plotting style is adapted from the original
`gfn-mols/scripts/plot_seh_fixedpb_combined_k_panels.py`; colors match
`paper_images_PPO.py`. Historical data paths are replaced by an explicit task root.

```bash
python scripts/plot_seh_fixedpb_combined_k_panels.py \
  --task seh --project-root ./runs/seh_fixedpb_grid \
  --vargrad-root ./runs/seh_fixedpb_grid --trpo-root ./runs/seh_fixedpb_grid \
  --trpo-regimes gfn_pg --output-dir ./plots/seh --max-seed 2 --no-x-cut

python scripts/plot_seh_fixedpb_combined_k_panels.py \
  --task qm9 --project-root ./runs/qm9_fixedpb_grid \
  --vargrad-root ./runs/qm9_fixedpb_grid --trpo-root ./runs/qm9_fixedpb_grid \
  --trpo-regimes gfn_pg --output-dir ./plots/qm9 --max-seed 2 --no-x-cut
```

The task root contains `db/seed_0/train.log`, `db_k4/seed_0/train.log`,
and the other directories produced by the fixed-Pb sweep. Each existing log
must have its `config.yaml`. The script checks constant beta=16 and training
batch size (sEH 256, QM9 128), rejecting old variable-temperature QM9 runs.

Each figure has four panels: ELBO/EUBO for baselines K=1 against Ent-PPO
K=1,2,4,8, then ELBO/EUBO for baselines K=4 against Ent-PPO K=4,8.
Lines show the seed mean; shading spans the seed minimum and maximum, as in the paper.
The default requires all three seeds at every plotted point; curves stop at
the last shared validation step. There is no smoothing or endpoint stretching.
sEH uses reward-weighted `iw_eubo_893`; QM9 uses `iw_eubo`.

The commands include VarGrad and TRPO from the standard sweep. For runs in
separate projects, point `--vargrad-root` and `--trpo-root` to those directories.
Add `--min-seeds 1` only for a preliminary snapshot
with incomplete seed coverage; the figure notes the seed-count range.
A single seed has no estimated variability. Missing runs are omitted, not zero-filled.

Outputs include PDF, PNG, per-seed values, plotted means/minima/maxima/std/counts,
per-run progress, and a manifest with the exact command, sources and EUBO key.
Read `progress.csv` to distinguish missing runs from completed runs.

Use `--elbo-zoom-ymin 30` to start both ELBO insets at 30 and include every
mean curve that reaches 30 in the zoomed interval, instead of only the top four.

The legend uses one row, ordered DB, TB, VarGrad, SubTB, TRPO, then Ent-PPO
by K. A single TRPO variant is displayed as `TRPO`. Use
`--elbo-ylim -20 40 --match-eubo-ylims` to fix both main ELBO axes and match
both EUBO axes to the right panel.

Use `--eubo-ylim 30 100 --elbo-zoom-ymin 36 --elbo-zoom-ymax 38` to set
shared EUBO bounds and explicit bounds for both ELBO insets.

Use `--no-elbo-zoom` to omit both ELBO insets. Baseline curves and their
legend handles use the same dashed-line style.

VarGrad uses navy and TRPO uses black, following the shared plotting palette.
Ent-PPO K=8 uses royal blue.
TRPO is shown only in the left pair of panels because the right pair compares
the K=4 baselines.

Use `--elbo-zoom-threshold 40` to include every curve whose mean ELBO exceeds
40 in the zoom interval without forcing the inset's lower y-axis bound to 40.
Use `--right-elbo-zoom-ymin 40` to clip only the right inset below 40.
