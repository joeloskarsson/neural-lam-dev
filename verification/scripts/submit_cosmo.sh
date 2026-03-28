#!/usr/bin/env bash

set -euo pipefail

cd "/users/sadamov/pyprojects/neural-lam-dev"
bash verification/scripts/run_all_cosmo.sh "$@"
