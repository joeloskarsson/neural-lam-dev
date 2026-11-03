# COSMO Sample on CSCS Clariden

The lightweight reproduction of [reproduce_paper_sample.md](reproduce_paper_sample.md) with every input
read from the local copies on Clariden, so nothing is downloaded except Python packages.
`scripts/reproduce_paper_sample_clariden.sh` runs the steps of that guide as Slurm jobs; what each step
does is explained there.

Needs Unix group `ab016` (data), a Slurm account (`ab016` by default, `ACCOUNT=...` to change) and `uv`
on `PATH`. Outputs go to `WORKDIR` (default `/iopsstor/scratch/cscs/$USER/neural_lam_sample`, purged
without notice).

## Inputs

| Input | Path under `/capstor/store/cscs/swissai/ab016` |
| --- | --- |
| COSMO sample (Step 0) | `COSMO/cosmo_sample.zarr` |
| ERA5 boundary (Step 3) | `COSMO/era_subset.zarr` (2015-11 to 2020-10; the config selects the sample window) |
| IFS boundary (Step 3) | `COSMO/cosmo_ifs_sample.zarr` (`scripts/ifs_download.py` output for the sample window) |
| Published checkpoint (Step 0) | `sadamov/cosmo_model.ckpt` (same file as Zenodo record 15131838) |

## Run

Each step submits one job and returns. Wait for it (`squeue --me`, logs in `WORKDIR/logs/`) before
starting the next.

```zsh
git clone --branch research https://github.com/joeloskarsson/neural-lam-dev.git && cd neural-lam-dev
bash scripts/reproduce_paper_sample_clariden.sh env     # Step 1, once
bash scripts/reproduce_paper_sample_clariden.sh prep    # Steps 2, 5 and 6 (~11 min)
bash scripts/reproduce_paper_sample_clariden.sh train   # Step 7, optional (~15 min)
bash scripts/reproduce_paper_sample_clariden.sh eval    # Step 9: published checkpoint, IFS boundaries (~6 min)
bash scripts/reproduce_paper_sample_clariden.sh plot    # after eval, on the login node: WORKDIR/sample_summary.png
```

## Expected result
