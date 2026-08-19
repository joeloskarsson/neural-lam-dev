# Verification scripts

Scripts that produce the figures, tables and metrics in the paper. Each script
is a standalone `# %%`-cell file, converted from the original notebooks.

## Before you run anything: change the paths

**These scripts were written for one machine (CSCS Alps) and one user. Every
path below is hardcoded and will not resolve anywhere else.** There is no
auto-detection and no config file, so the scripts fail at the first
`open_zarr` unless you edit them first.

Three kinds of path need changing:

**1. Repo location.** `run_all_cosmo.sh` and `run_all_danra.sh` set it as a
variable; `submit_cosmo.sh` and `submit_danra.sh` hardcode the same path in a
bare `cd`:

```bash
REPO_DIR="/users/sadamov/pyprojects/neural-lam-dev"   # run_all_*.sh
cd "/users/sadamov/pyprojects/neural-lam-dev"         # submit_*.sh
```

**2. Input datasets**, the `PATH_*` constants in the first ~80 lines of each
script. Relative names such as `danra_test_gt.zarr` or `cosmo_e_forecast.zarr`
are resolved from the repo root, where they exist only as symlinks into
project storage, so they must be repointed at your own copies. A few are
absolute and hardcoded outright:

| Path | Used by |
| --- | --- |
| `/capstor/.../a122/sadamov/cosmo_e_forecast.zarr` | `verification_gridded_cosmo_metrics.py`, `..._wind_sal_tmp.py` |
| `/capstor/.../a122/sadamov/cosmo_e_forecast_fixed.zarr` | `verification_gridded_cosmo.py`, `..._metrics.py`, `..._wind_sal_tmp.py` |
| `/capstor/.../a122/sadamov/lam_model_forecasts` | `verification_gridded_cosmo_metrics.py`, `..._wind_sal_tmp.py` |
| `/capstor/.../a122/sadamov/danra_station_observations.zarr` | `verification_sparse_danra.py`, `fetch_danra_obs.py`, `temp_wind_speed_map_6h_danra.py` |

**3. Scratch/temp directories.** Several scripts write dask spill and temp
files to a hardcoded `/iopsstor/scratch/cscs/sadamov`. Grep for it:

```bash
grep -rn '/iopsstor/scratch/cscs/sadamov' verification/scripts/
```

A quick way to find everything that needs attention:

```bash
grep -rnE '^(PATH_[A-Z_]*|path_obs|REPO_DIR)\s*=' verification/scripts/
grep -rnoE '"/(users|capstor|iopsstor)[^"]*"' verification/scripts/
```

## SLURM account

`run_all_*.sh` default to `--account=ab016`. Override without editing:

```bash
SLURM_ACCOUNT=your_account ./run_all_cosmo.sh
```

## Station observations

DANRA station data is fetched from the DMI open-data API by
`fetch_danra_obs.py` and cached to the `danra_station_observations.zarr` path
above. COSMO station data (`cosmo_observations.zarr`) is not public; contact
the authors. Note that HTTPS from CSCS compute nodes needs
`export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')`,
as there is no system CA bundle there.

## Outputs

Results are written to `verification/<domain>/<gridded|sparse|case_study>/`,
each figure as `.pdf` plus a `.npz` sidecar holding the plotted arrays. These
are gitignored (see the repo `.gitignore`), so **the outputs are not committed
and only exist on the machine that produced them.** Copy anything the paper
needs out of the tree before it is lost.
