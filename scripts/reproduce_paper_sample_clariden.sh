#!/usr/bin/env bash
# docs/reproduce_paper_sample.md on CSCS Clariden, with every input read from the local
# copies under /capstor/store/cscs/swissai/ab016 instead of downloading them. Needs Unix
# group ab016, a Slurm account and `uv` on PATH.
#
#   bash scripts/reproduce_paper_sample_clariden.sh env     # once: .venv in this checkout (Step 1)
#   bash scripts/reproduce_paper_sample_clariden.sh prep    # WORKDIR, datastores, graph (Steps 2, 5, 6)
#   bash scripts/reproduce_paper_sample_clariden.sh train   # optional: 1-epoch training (Step 7)
#   bash scripts/reproduce_paper_sample_clariden.sh eval    # published checkpoint -> test forecasts (Step 9)
#   bash scripts/reproduce_paper_sample_clariden.sh plot    # after eval: WORKDIR/sample_summary.png
#
# Local inputs (Step 0 and 3 of the guide):
#   ab016/COSMO/cosmo_sample.zarr       COSMO sample 2016-01/02 (same as the ETH Research Collection sample)
#   ab016/COSMO/era_subset.zarr         ERA5 boundary 2015-11 to 2020-10 (scripts/era_download.py output)
#   ab016/sadamov/cosmo_model.ckpt      published checkpoint (Zenodo record 15131838)
# IFS forecasts for the sample window are not on Clariden (fetching them from WeatherBench 2 is far too
# slow here), so the evaluation uses the ERA5 boundary config; the guide's Step 9 uses IFS.
# Outputs go to WORKDIR (default /iopsstor/scratch/cscs/$USER/neural_lam_sample, purgeable).

set -euo pipefail

SRCDIR="$(cd "$(dirname "$0")/.." && pwd)"
AB016=/capstor/store/cscs/swissai/ab016
WORKDIR="${WORKDIR:-/iopsstor/scratch/cscs/$USER/neural_lam_sample}"
ACCOUNT="${ACCOUNT:-ab016}"
PARTITION="${PARTITION:-debug}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

mkdir -p "$WORKDIR/logs"

# submit <name> <walltime>: the job body is read from stdin.
submit() {
    sbatch --account="$ACCOUNT" --partition="$PARTITION" --time="$2" --nodes=1 --ntasks=1 \
        --cpus-per-task=64 --mem=400G --job-name="nlam_sample_$1" \
        --output="$WORKDIR/logs/$1_%j.log" <<EOF
#!/bin/bash
set -euo pipefail
export TMPDIR="$WORKDIR/tmp" WANDB_MODE=offline PYTHONUNBUFFERED=1
mkdir -p "\$TMPDIR"
$(cat)
echo "done: $1"
EOF
}

case "${1:-}" in
env)
    submit env 00:45:00 <<EOF
cd "$SRCDIR"
uv python install 3.12
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url "$TORCH_INDEX"
uv pip install --python .venv/bin/python ".[graph]" "zarr<3" \
    "mllam-data-prep @ git+https://github.com/sadamov/mllam-data-prep.git@building-ml-lams" \
    "weather-model-graphs @ git+https://github.com/joeloskarsson/weather-model-graphs.git@building-ml-lams"
.venv/bin/python -c "import torch, neural_lam, mllam_data_prep, weather_model_graphs; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
EOF
    ;;
prep)
    submit prep 01:00:00 <<EOF
source "$SRCDIR/.venv/bin/activate"
cd "$WORKDIR"
for f in cosmo_interior_config_sample.yaml cosmo_era5_config_sample.yaml cosmo_ifs_config_sample.yaml \
         cosmo_build_graphs_sample.sh cosmo_model_config_era5_sample.yaml cosmo_model_config_ifs_sample.yaml; do
    cp "$SRCDIR/scripts/\$f" .
done
rm -rf cosmo_land_sea_mask.zarr && cp -r "$SRCDIR/scripts/artifacts/cosmo_land_sea_mask.zarr" .
# Statistics the published checkpoint was trained with (docs/cosmo_dataset_reference.md); the yaml
# sibling only lends its file stem, its content is not read.
rm -rf cosmo_train_stats.zarr && cp -r "$SRCDIR/scripts/artifacts/cosmo_train_stats.zarr" .
cp cosmo_interior_config_sample.yaml cosmo_train_stats.yaml
sed 's|^  config_path: cosmo_interior_config_sample.yaml\$|&\n  overload_stats_path: cosmo_train_stats.yaml|' \
    cosmo_model_config_era5_sample.yaml > cosmo_model_config_era5_published.yaml
mkdir -p figures && cp "$SRCDIR/figures/earth_texture.jpeg" figures/
ln -sfn "$AB016/COSMO/cosmo_sample.zarr" cosmo_sample.zarr
ln -sfn "$AB016/COSMO/era_subset.zarr" cosmo_era5_sample.zarr
ln -sfn "$AB016/sadamov/cosmo_model.ckpt" cosmo_model.ckpt
python -m mllam_data_prep cosmo_interior_config_sample.yaml
python -m mllam_data_prep cosmo_era5_config_sample.yaml
bash cosmo_build_graphs_sample.sh
EOF
    ;;
train)
    submit train 01:00:00 <<EOF
source "$SRCDIR/.venv/bin/activate"
cd "$WORKDIR"
export CUDA_VISIBLE_DEVICES=0   # one GPU, as in the guide (multi-GPU evaluation skews the metrics)
python -m neural_lam.train_model --config_path cosmo_model_config_era5_sample.yaml \
    --model hi_lam --graph_name rectangular_hierarchical --hidden_dim 300 --hidden_dim_grid 150 \
    --time_delta_enc_dim 32 --processor_layers 2 --batch_size 1 --min_lr 0.001 --epochs 1 \
    --val_interval 10 --val_steps_to_log 1 2 3 4 --ar_steps_eval 4 --precision bf16-mixed \
    --plot_vars T_2M --num_workers 8 --num_nodes 1
EOF
    ;;
eval)
    submit eval 01:00:00 <<EOF
source "$SRCDIR/.venv/bin/activate"
cd "$WORKDIR"
export CUDA_VISIBLE_DEVICES=0   # one GPU, as in the guide (multi-GPU evaluation skews the metrics)
python -m neural_lam.train_model --config_path cosmo_model_config_era5_published.yaml \
    --model hi_lam --epochs 1 --eval test --n_example_pred 1 --ar_steps_eval 24 \
    --val_steps_to_log 1 6 12 18 24 --hidden_dim 300 --hidden_dim_grid 150 --time_delta_enc_dim 32 \
    --processor_layers 2 --num_nodes 1 --batch_size 1 --plot_vars T_2M U_10M --precision bf16-mixed \
    --graph_name rectangular_hierarchical --load cosmo_model.ckpt \
    --save_eval_to_zarr_path cosmo_sample_forecasts.zarr
EOF
    ;;
plot)
    "$SRCDIR/.venv/bin/python" - "$WORKDIR" <<'PY'
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

workdir = Path(sys.argv[1])
fc = xr.open_zarr(workdir / "cosmo_sample_forecasts.zarr")["state"].isel(start_time=0)
pred = fc.sel(state_feature="T_2M").isel(elapsed_forecast_duration=-1)
# The zarr's `time` coordinate belongs to the last start time; valid time = start_time + lead.
valid = pred["start_time"].values + pred["elapsed_forecast_duration"].values
truth = xr.open_zarr(workdir / "cosmo_sample.zarr")["T_2M"].sel(time=valid).transpose(*pred.dims)
hours = int(pred["elapsed_forecast_duration"].values / 3.6e12)
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
kw = dict(origin="lower", cmap="RdYlBu_r", vmin=float(truth.min()), vmax=float(truth.max()))
for ax, field, title in [(axes[0], pred, f"neural-LAM T_2M, +{hours} h"), (axes[1], truth, "COSMO analysis")]:
    im = ax.imshow(field.values.T, **kw)
    ax.set_title(f"{title}, valid {str(valid)[:16]}")
    fig.colorbar(im, ax=ax, label="K")
im = axes[2].imshow((pred - truth).values.T, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
rmse = float(((pred - truth) ** 2).mean() ** 0.5)
axes[2].set_title(f"forecast - analysis (RMSE {rmse:.2f} K)")
fig.colorbar(im, ax=axes[2], label="K")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(workdir / "sample_summary.png", dpi=100)
print(workdir / "sample_summary.png")
PY
    ;;
*)
    sed -n '2,17p' "$0"
    exit 1
    ;;
esac
