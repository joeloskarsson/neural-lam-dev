#!/usr/bin/env bash

# NOTE: paths below are hardcoded for one machine and user. See
# verification/README.md before running this anywhere else.

set -euo pipefail

cd "/users/sadamov/pyprojects/neural-lam-dev"
bash verification/scripts/run_all_danra.sh "$@"
