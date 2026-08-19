#!/usr/bin/env bash

# NOTE: paths below are hardcoded for one machine and user. See
# verification/README.md before running this anywhere else.

set -euo pipefail

REPO_DIR="/users/sadamov/pyprojects/neural-lam-dev"
SCRIPT_DIR="${REPO_DIR}/verification/scripts"
LOGS_DIR="${REPO_DIR}/logs"
ACCOUNT="${SLURM_ACCOUNT:-ab016}"
mkdir -p "${LOGS_DIR}"

scripts=(
    "verification_gridded_danra_histograms.py"
    "verification_gridded_danra_maps.py"
    "verification_gridded_danra_spectra.py"
    "verification_gridded_danra_vertical.py"
    "verification_sparse_danra.py"
)

for script in "${scripts[@]}"; do
    job_name="verify_${script%.py}"
    echo "Submitting ${job_name}"
    sbatch \
        --job-name="${job_name}" \
        --account="${ACCOUNT}" \
        --partition=normal \
        --time=12:00:00 \
        --nodes=1 \
        --ntasks=1 \
        --mem=444G \
        --output="${LOGS_DIR}/${job_name}_%j.out" \
        --error="${LOGS_DIR}/${job_name}_%j.err" \
        --wrap="source ${REPO_DIR}/.venv/bin/activate && cd ${REPO_DIR} && python -u ${SCRIPT_DIR}/${script}"
done

# Submit metrics for both full domain and Denmark crop in parallel
for variant in "" "--denmark"; do
    if [[ -z "${variant}" ]]; then
        job_name="verify_verification_gridded_danra_metrics"
    else
        job_name="verify_verification_gridded_danra_metrics_denmark"
    fi
    echo "Submitting ${job_name}"
    sbatch \
        --job-name="${job_name}" \
        --account="${ACCOUNT}" \
        --partition=normal \
        --time=12:00:00 \
        --nodes=1 \
        --ntasks=1 \
        --mem=444G \
        --output="${LOGS_DIR}/${job_name}_%j.out" \
        --error="${LOGS_DIR}/${job_name}_%j.err" \
        --wrap="source ${REPO_DIR}/.venv/bin/activate && cd ${REPO_DIR} && python -u ${SCRIPT_DIR}/verification_gridded_danra_metrics.py ${variant}"
done
