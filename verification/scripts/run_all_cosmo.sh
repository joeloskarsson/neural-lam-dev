#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/users/sadamov/pyprojects/neural-lam-dev"
SCRIPT_DIR="${REPO_DIR}/verification/scripts"
LOGS_DIR="${REPO_DIR}/logs"
mkdir -p "${LOGS_DIR}"

submit_script() {
    local script="$1"
    local job_name="verify_${script%.py}"
    echo "Submitting ${job_name}"
    sbatch \
        --job-name="${job_name}" \
        --account=a122 \
        --partition=normal \
        --time=12:00:00 \
        --nodes=1 \
        --ntasks=1 \
        --mem=444G \
        --output="${LOGS_DIR}/${job_name}_%j.out" \
        --error="${LOGS_DIR}/${job_name}_%j.err" \
        --wrap="source ${REPO_DIR}/.venv/bin/activate && cd ${REPO_DIR} && python -u ${SCRIPT_DIR}/${script}"
}

submit_script "verification_gridded_cosmo.py"
submit_script "verification_gridded_cosmo_metrics.py"

if [[ -e "${SCRIPT_DIR}/../../cosmo_observations.zarr" ]]; then
    submit_script "verification_sparse_cosmo.py"
    submit_script "verification_sparse_cosmo_metrics.py"
else
    echo "Skipping verification_sparse_cosmo.py: cosmo_observations.zarr is missing"
    echo "Skipping verification_sparse_cosmo_metrics.py: cosmo_observations.zarr is missing"
fi
