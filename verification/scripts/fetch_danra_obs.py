"""Fetch DMI station observations for DANRA sparse verification.

Run this on the login node (which has internet access) before submitting the
SLURM job for verification_sparse_danra.py. The resulting zarr is then read by
the main script without any network access required.

Usage:
    cd /users/sadamov/pyprojects/neural-lam-dev
    source .venv/bin/activate
    python verification/scripts/fetch_danra_obs.py
"""

# Standard library
import shutil
from pathlib import Path

# Third-party
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr

# ── Config (must match verification_sparse_danra.py) ──────────────────────────
PATH_OBS = (
    "/capstor/store/cscs/swissai/a122/sadamov/danra_station_observations.zarr"
)
PATH_ML = "48h_eval_test_7deg_rect_hi4_2867.zarr"
PLOT_TIME = "2020-02-09T12:00:00"

url_obs = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"
api_limit = 300_000
api_time_step = np.timedelta64(7, "D")

VARIABLES_OBS = [
    "wind_speed_past1h",
    "wind_dir_past1h",
    "temp_dry",
    "pressure_at_sea",
]

# ── Check if existing zarr already covers PLOT_TIME ───────────────────────────
plot_time_np = np.datetime64(PLOT_TIME)
if Path(PATH_OBS).exists():
    _ds = xr.open_zarr(PATH_OBS)
    _t0, _t1 = _ds.time.values[0], _ds.time.values[-1]
    _ds.close()
    if _t0 <= plot_time_np <= _t1:
        print(f"Obs zarr already covers {_t0} to {_t1}. Nothing to do.")
        raise SystemExit(0)
    print(
        f"Obs zarr covers {_t0} to {_t1} — does not include {PLOT_TIME}. "
        "Deleting and re-fetching."
    )
    shutil.rmtree(PATH_OBS)

# ── Determine fetch window from ML data ───────────────────────────────────────
ds_ml_raw = xr.open_zarr(PATH_ML, decode_timedelta=True)
ELAPSED_FORECAST_DURATION = list(range(16))
START_TIMES = ["2019-10-30T00:00", "2020-10-23T12:00"]

ds_ml_raw = ds_ml_raw.sel(start_time=slice(*START_TIMES))
forecast_times = (
    ds_ml_raw.start_time.values[:, None]
    + ds_ml_raw.elapsed_forecast_duration.values[ELAPSED_FORECAST_DURATION]
)
datetime_start = forecast_times[0, 0]
datetime_end = forecast_times[-1, -1]
print(f"Fetching obs from {datetime_start} to {datetime_end}")

# ── Fetch from DMI API ────────────────────────────────────────────────────────
dfs = {}
for var in VARIABLES_OBS:
    print(f"\nFetching variable: {var}")
    fetch_start_time = datetime_start
    var_dfs = []
    while fetch_start_time <= datetime_end:
        fetch_end_time = fetch_start_time + api_time_step
        start_str = np.datetime_as_string(fetch_start_time, unit="s")
        end_str = np.datetime_as_string(fetch_end_time, unit="s")
        datetime_range = f"{start_str}Z/{end_str}Z"
        print(f"  {datetime_range}")
        fetch_start_time = fetch_end_time + np.timedelta64(1, "h")

        params = {
            "datetime": datetime_range,
            "parameterId": var,
            "bbox": "7,54,16,58",
            "limit": api_limit,
        }
        response = requests.get(url_obs, params=params)
        response.raise_for_status()
        data = response.json()
        assert data["numberReturned"] < api_limit, "Hit API limit"
        gdf = gpd.GeoDataFrame.from_features(data["features"])
        gdf["time"] = pd.to_datetime(gdf["observed"], utc=True)
        df_pivot = gdf.pivot(index="time", columns="stationId", values="value")
        var_dfs.append(df_pivot)
    dfs[var] = pd.concat(var_dfs)

# ── Find common stations across all variables ─────────────────────────────────
common_stations = sorted(
    set.intersection(*[set(dfs[v].columns) for v in VARIABLES_OBS])
)
print(f"\nStations with data for all variables: {len(common_stations)}")
dfs_filtered = {v: dfs[v][common_stations] for v in VARIABLES_OBS}
gdf_filtered = gdf[gdf["stationId"].isin(common_stations)]

# ── Align to model forecast times ─────────────────────────────────────────────
model_times = np.unique(forecast_times.flatten())
model_times_utc = pd.to_datetime(model_times).tz_localize("UTC")
dfs_filtered = {
    k: v[v.index.isin(model_times_utc)] for k, v in dfs_filtered.items()
}
gdf_filtered = gdf_filtered.set_index("time")
gdf_filtered = gdf_filtered[gdf_filtered.index.isin(model_times_utc)]

# ── Build xarray Dataset ──────────────────────────────────────────────────────
ds_obs = xr.Dataset(
    {
        var: (["time", "stationId"], dfs_filtered[var].values)
        for var in VARIABLES_OBS
    },
    coords={
        "time": model_times,
        "stationId": dfs_filtered[VARIABLES_OBS[0]].columns,
        "lat": (
            "stationId",
            gdf_filtered.groupby("stationId")["geometry"].first().y,
        ),
        "lon": (
            "stationId",
            gdf_filtered.groupby("stationId")["geometry"].first().x,
        ),
    },
)
ds_obs = ds_obs.sortby("time")

# ── Unit conversions and wind components ──────────────────────────────────────
ds_obs["u10m"] = -ds_obs["wind_speed_past1h"] * np.cos(
    np.radians(90 - ds_obs["wind_dir_past1h"])
)
ds_obs["v10m"] = -ds_obs["wind_speed_past1h"] * np.sin(
    np.radians(90 - ds_obs["wind_dir_past1h"])
)
ds_obs = ds_obs.drop_vars(["wind_speed_past1h", "wind_dir_past1h"])
# Convert units and rename to match SynopProcessor output expected by the script
ds_obs["temp_dry"] = ds_obs["temp_dry"] + 273.15  # °C → K
ds_obs["pressure_at_sea"] = ds_obs["pressure_at_sea"] * 100  # hPa → Pa
ds_obs = ds_obs.rename_vars(
    {"temp_dry": "t2m", "pressure_at_sea": "pres_seasurface"}
)

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {PATH_OBS}")
ds_obs.to_zarr(PATH_OBS, mode="w")
print("Done.")
