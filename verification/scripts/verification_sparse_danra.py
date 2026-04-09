# %% [markdown]
# ## Verification of Two Model Forecasts vs Observations (Measurements)

# %% [markdown]
# Documentation about DMI dat and Scores can be found here:
# - https://opendatadocs.dmi.govcloud.dk/APIs/Meteorological_Observation_API
# - https://scores.readthedocs.io/en/stable/index.html

# %% [markdown]
# Please define the following variables to define the data you want to
# verify. Each model var must have a corresponding obs var after processing.

# Standard library
# %%
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import xarray as xr
from dask.diagnostics import ProgressBar
from debug_utils import (
    align_on_common_coord,
    filter_forecast_dataset_by_time_coverage,
    parse_debug_runtime_args,
    resolve_requested_time,
    select_available_times,
    select_valid_positions,
    subset_dataset_for_debug,
)
from plot_styles import (
    DPI,
    FONT_SIZES,
    SCALE_INTERP_DANRA,
    SCALE_METRICS,
    apply_style,
    scaled_fonts,
)
from scipy.spatial import cKDTree
from scipy.stats import kurtosis, skew, wasserstein_distance
from scores.categorical import ThresholdEventOperator as TEO
from scores.continuous import mae, mean_error, mse, rmse
from scores.continuous.correlation import pearsonr

# %%
# Constants for station obs
path_obs = (
    "/capstor/store/cscs/swissai/a122/sadamov/danra_station_observations.zarr"
)
url_obs = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"
api_limit = 300000
api_time_step = np.timedelta64(7, "D")

# %%
# DEFAULTS
# This config will be applied to the data before any plotting. The data will be
# sliced and indexed according to the values in this config. The whole analysis
# plotting will be done on the reduced data.

# IF YOUR DATA HAS DIFFERENT DIMENSIONS OR NAMES, PLEASE ADJUST THE CELLS BELOW
# MAKE SURE THE XARRAY DATASETS LOOK OKAY BEFORE RUNNING CHAPTER 1-4

# This path should point to the NWP forecast data in zarr format
PATH_NWP = "danra_test_nwp_forecasts.zarr"
# This path should point to the ML forecast data in zarr format
# (e.g. produced by neural-lam in `eval` mode)
PATH_ML = "48h_eval_test_7deg_rect_hi4_2867.zarr"
# This path should point to the observations data in zarr format
PATH_OBS = path_obs

# All variables should map to the same values.
# For obs please also theck the pre-processing in the cell below
# Often some data wrangling is required to match the variables
# OBS and ML variables should match and always be present
VARIABLES_ML = {
    "t2m": "temperature_2m",
    "u10m": "wind_u_10m",
    "v10m": "wind_v_10m",
    "pres_seasurface": "seasurface_pressure",
}
# The function calls are flexible to allow for missing nwp data
VARIABLES_NWP = {
    "t2m": "temperature_2m",
    "u10m": "wind_u_10m",
    "v10m": "wind_v_10m",
    "pres_seasurface": "seasurface_pressure",
}
VARIABLES_OBS = [
    "wind_speed_past1h",
    "wind_dir_past1h",
    "temp_dry",
    "pressure_at_sea",
    # "pressure",
]
# Similar mapping to ML/NWP
OBS_VAR_MAPPING = {
    "t2m": "temperature_2m",
    "u10m": "wind_u_10m",
    "v10m": "wind_v_10m",
    "pres_seasurface": "seasurface_pressure",
}

# Add units dictionary after the imports
# units from zarr archives are not reliable and should rather be defined here
VARIABLE_UNITS = {
    # Surface and near-surface variables
    "temperature_2m": "K",
    "wind_u_10m": "m/s",
    "wind_v_10m": "m/s",
    "wind_speed": "m/s",
    "seasurface_pressure": "Pa",
}

THRESHOLDS_WIND = [2.5, 5, 10]  # m/s

# lead time in steps for the forecast - [0] refers to the first forecast step
# at t+1
# this should be a list of integers
ELAPSED_FORECAST_DURATION = list(range(16))
ELAPSED_FORECAST_DURATION_SHORT = list(range(6))  # 18 h, matching NWP forecasts
ELAPSED_FORECAST_DURATION_PLOT = [1, 3, 5]  # ELS to plot

# Select specific start_times for the forecast. This is the start and end of
# a slice in xarray. The start_time is included, the end_time is excluded.
# This should be a list of two strings in the format "YYYY-MM-DDTHH:MM:SS"
# Should be handy to evaluate certain dates, e.g. for a case study of a storm
# START_TIMES = ["2020-02-07T00:00:00", "2020-02-10T00:00:00"]  # 7 init times
START_TIMES = [
    "2019-10-30T00:00",
    "2020-10-23T12:00",
]  # Full year, 720 init times


# Select specific plot times for the forecast (will be used to create maps for
# all variables)
# This only affect chapter one with the plotting of the maps
# Map creation takes a lot of time so this is limited to a single time step
# Simply rerun these cells and chapter one for more time steps
PLOT_TIME = "2020-02-09T12:00:00"

# Define Thresholds for the ETS metric (Equitable Threat Score)
# These are calculated for wind and precipitation if available
# The score creates contingency tables for different thresholds
# The ETS is calculated for each threshold and the results are plotted
# The default thresholds are [0.1, 1, 5] for precipitation and
# [2.5, 5, 10] for wind
THRESHOLDS_PRECIPITATION = [0.1, 1, 5]  # mm/h
THRESHOLDS_WIND = [2.5, 5, 10]  # m/s

# Define the metrics to compute for the verification
# Some additional verifications will always be computed if the respective vars
# are available in the data
METRICS = [
    # "MAE",
    "RMSE",
    # "MSE",
    "ME",
    "STDEV_ERR",
    # "RelativeMAE",
    # "RelativeRMSE",
    # "PearsonR",
    "Wasserstein",
]

# Map projection settings for plotting of the ml data
PROJECTION = ccrs.LambertConformal(
    central_longitude=25.0,
    central_latitude=56.7,
    standard_parallels=[56.7, 56.7],
    globe=ccrs.Globe(
        semimajor_axis=6367470.0,
        semiminor_axis=6367470.0,
    ),
)

# For some chapters a random seed is required to reproduce the results
RANDOM_SEED = 42


# Takes a long time, but if you see NaN in your output, you can set this to True
# This will check if there are any missing values in the data further below
CHECK_MISSING = True
# In this script missing data is allowed as observations often have
# missing values
# All time steps with missing values will be omitted from the verification
# - Scores/Xarray masked arrays are created and false values are omitted
#   by default
# - For Scipy metrics, we need to convert to numpy arrays and change
#   nan-policy to 'omit'.
# - Fore the wasserstein metric and the wind vector without internal
#   nan-handling policy,
#   we need to remove the missing values for obs, ml, nwp before
#   calculating the metric


# Larger font sizes for sparse error/observation maps (+50%)
ERROR_MAP_FONT_SIZES = scaled_fonts(
    SCALE_INTERP_DANRA
)  # interpolated comparison maps

DEBUG_MODE, DEBUG_FRACTION, DEBUG_MIN_SIZE = parse_debug_runtime_args(
    "Run sparse DANRA verification.",
)

USE_EXAMPLE_DEBUG_DATA = False
if USE_EXAMPLE_DEBUG_DATA:
    # This path should point to the NWP forecast data in zarr format
    # PATH_NWP = "danra_test_nwp_forecasts.zarr"
    PATH_NWP = "danra_ex_fc/danra_nwp_ex_forecasts.zarr"
    # This path should point to the ML forecast data in zarr format
    # (e.g. produced by neural-lam in `eval` mode)
    # PATH_ML = "48h_eval_test_7deg_rect_hi4_2867.zarr"
    PATH_ML = "danra_ex_fc/danra_ml_ex_forecasts.zarr"

    START_TIMES = ["2020-01-02T00:00", "2020-01-04T00:00"]
    PLOT_TIME = "2020-01-02T00:00:00"

# %% [markdown]
# ### Read in datasets

# %%
ds_nwp = xr.open_zarr(PATH_NWP, decode_timedelta=True)
ds_nwp = subset_dataset_for_debug(
    ds_nwp,
    label="NWP zarr",
    dims_to_trim=("analysis_time", "x", "y"),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
)
ds_nwp = ds_nwp.sel(analysis_time=slice(*START_TIMES))
ds_nwp = ds_nwp[VARIABLES_NWP.keys()].rename(VARIABLES_NWP)
ds_nwp = ds_nwp.rename_dims(
    {
        "analysis_time": "start_time",
    }
)
ds_nwp = ds_nwp.rename_vars(
    {
        "analysis_time": "start_time",
        "lon": "longitude",
        "lat": "latitude",
    }
)
forecast_times = (
    ds_nwp.start_time.values[:, None] + ds_nwp.elapsed_forecast_duration.values
)
ds_nwp = ds_nwp.assign_coords(
    forecast_time=(
        ("start_time", "elapsed_forecast_duration"),
        forecast_times,
    )
)

ds_nwp = ds_nwp.drop("time")

# The NWP data starts at lead time 3h
ds_nwp = select_valid_positions(
    ds_nwp,
    dim="elapsed_forecast_duration",
    positions=ELAPSED_FORECAST_DURATION_SHORT,
    label="NWP",
)

ds_nwp = ds_nwp.transpose("start_time", "elapsed_forecast_duration", "x", "y")
ds_nwp = ds_nwp[
    [
        "start_time",
        "elapsed_forecast_duration",
        "x",
        "y",
        *VARIABLES_NWP.values(),
    ]
]

ds_nwp

# %%
ds_ml = xr.open_zarr(PATH_ML, decode_timedelta=True)
ds_ml = subset_dataset_for_debug(
    ds_ml,
    label="ML zarr",
    dims_to_trim=("start_time", "x", "y"),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
)
ds_ml = ds_ml.sel(state_feature=list(VARIABLES_ML.keys()))
ds_ml = ds_ml.sel(start_time=slice(*START_TIMES))
for feature in ds_ml.state_feature.values:
    ds_ml[VARIABLES_ML[feature]] = ds_ml["state"].sel(state_feature=feature)
forecast_times = (
    ds_ml.start_time.values[:, None] + ds_ml.elapsed_forecast_duration.values
)
ds_ml = ds_ml.assign_coords(
    forecast_time=(
        ("start_time", "elapsed_forecast_duration"),
        forecast_times,
    )
)
ds_ml = ds_ml.drop_vars(["state", "state_feature", "time"])
ds_ml = ds_ml.transpose("start_time", "elapsed_forecast_duration", "x", "y")
ds_ml = ds_ml[
    [
        "start_time",
        "elapsed_forecast_duration",
        "x",
        "y",
        *VARIABLES_ML.values(),
    ]
]
ds_ml = ds_ml.assign_coords(
    {
        "latitude": ds_nwp.latitude,
        "longitude": ds_nwp.longitude,
    }
)

ds_ml = select_valid_positions(
    ds_ml,
    dim="elapsed_forecast_duration",
    positions=ELAPSED_FORECAST_DURATION,
    label="ML",
)

ds_ml, ds_nwp = align_on_common_coord(
    ds_ml,
    ds_nwp,
    coord_name="start_time",
    left_label="ML",
    right_label="NWP",
)

ds_ml

# %% [markdown]
# ### Retrieve observations

# %%
# From first forecasted time in first forecast, to last forecasted time in
# last forecast
datetime_start = ds_ml.forecast_time.values[0, 0]
datetime_end = ds_ml.forecast_time.values[-1, -1]
print(f"from {datetime_start} to {datetime_end}")

# %%
# Delete obs zarr if it doesn't cover PLOT_TIME (case study) or the full
# forecast window, so it gets re-fetched below.
if Path(PATH_OBS).exists() and PLOT_TIME is not None:
    # Standard library
    import shutil

    _existing_obs = xr.open_zarr(PATH_OBS)
    _obs_start = _existing_obs.time.values[0]
    _obs_end = _existing_obs.time.values[-1]
    _existing_obs.close()
    _plot_time_np = np.datetime64(PLOT_TIME)
    if _obs_start > _plot_time_np or _obs_end < _plot_time_np:
        print(
            f"Obs zarr covers {_obs_start} to {_obs_end}, "
            f"but PLOT_TIME {PLOT_TIME} is not covered. "
            "Deleting zarr for re-fetch over full forecast window."
        )
        shutil.rmtree(PATH_OBS)

# %%
dfs = {}
if not Path(PATH_OBS).exists():
    for var in VARIABLES_OBS:
        print(f"Fetching variable {var}")
        fetch_start_time = datetime_start

        var_dfs = []
        while fetch_start_time <= datetime_end:
            fetch_end_time = fetch_start_time + api_time_step

            fetch_start_str = np.datetime_as_string(fetch_start_time, unit="s")
            fetch_end_str = np.datetime_as_string(fetch_end_time, unit="s")
            fetch_datetime_range = f"{fetch_start_str}Z/{fetch_end_str}Z"
            print(f"fetching {fetch_datetime_range}")
            fetch_start_time = fetch_end_time + np.timedelta64(1, "h")

            params = {
                "datetime": fetch_datetime_range,
                "parameterId": var,
                "bbox": "7,54,16,58",  # Bounding box for Denmark
                "limit": api_limit,
            }

            response = requests.get(url_obs, params=params)
            response.raise_for_status()
            data = response.json()
            assert (
                data["numberReturned"] < api_limit
            ), "Hit API limit for returned values"
            gdf = gpd.GeoDataFrame.from_features(data["features"])
            gdf["time"] = pd.to_datetime(gdf["observed"], utc=True)
            df_pivot = gdf.pivot(
                index="time", columns="stationId", values="value"
            )
            var_dfs.append(df_pivot)

        var_df = pd.concat(var_dfs)
        dfs[var] = var_df

# %%
if not Path(PATH_OBS).exists():
    # Get the set of stations for each variable
    station_sets = [set(dfs[var].columns) for var in VARIABLES_OBS]

    # Find common stations across all variables using set intersection
    common_stations = set.intersection(*station_sets)

    # Convert to sorted list if needed
    common_stations = sorted(list(common_stations))

    # Print the number of common stations and their IDs
    print(
        f"Number of stations with data for all variables:"
        f" {len(common_stations)}"
    )
    print("\nStation IDs:")
    print(common_stations)

    # Filter your DataFrames to keep only common stations
    dfs_filtered = {}
    for var in VARIABLES_OBS:
        dfs_filtered[var] = dfs[var][common_stations]
    # filter gdf by stations as well
    gdf_filtered = gdf[gdf["stationId"].isin(common_stations)]

# %%
if not Path(PATH_OBS).exists():
    # Convert model times to UTC datetime (assuming ds_model is available)
    model_times = np.unique(ds_ml.forecast_time.values.flatten())
    model_times_utc = pd.to_datetime(model_times).tz_localize("UTC")
    print(model_times_utc)
    dfs_filtered = {
        k: v[v.index.isin(model_times_utc)] for k, v in dfs_filtered.items()
    }
    gdf_filtered = gdf_filtered.set_index("time")
    gdf_filtered = gdf_filtered[gdf_filtered.index.isin(model_times_utc)]

# %%
if not Path(PATH_OBS).exists():
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
    ds_obs


# %% [markdown]
# Conversion of wind speed and direction to u and v.


# %%
class SynopProcessor:
    def __init__(self):
        self.var_mapping = {
            "temp_dry": "t2m",
            "pressure_at_sea": "pres_seasurface",
            # "pressure": "pres0m",
        }

    def calculate_wind_components(
        self, ds, speed_var="wind_speed_past1h", dir_var="wind_dir_past1h"
    ):
        """Calculate u and v wind components from speed and direction."""
        ds = ds.copy()
        ds["u10m"] = -ds[speed_var] * np.cos(np.radians(90 - ds[dir_var]))
        ds["v10m"] = -ds[speed_var] * np.sin(np.radians(90 - ds[dir_var]))
        ds = ds.drop_vars([speed_var, dir_var])
        return ds

    def rename_variables(self, ds):
        """Rename variables according to mapping."""
        return ds.rename_vars(self.var_mapping)

    def convert_units(self, ds):
        """Convert temperature to Kelvin and pressure to Pa."""
        ds = ds.copy()
        # Convert temperature from Celsius to Kelvin
        if "temp_dry" in ds:
            ds["temp_dry"] = ds["temp_dry"] + 273.15

        # Convert pressure from hPa to Pa
        for pressure_var in ["pressure", "pressure_at_sea"]:
            if pressure_var in ds:
                ds[pressure_var] = ds[pressure_var] * 100
        return ds

    def process_dataset(self, ds):
        """Process the entire dataset with unit conversions, wind calculation
        and variable renaming."""
        ds = self.convert_units(ds)
        ds = self.calculate_wind_components(ds)
        ds = self.rename_variables(ds)
        return ds


# %%
if not Path(PATH_OBS).exists():
    processor = SynopProcessor()
    ds_obs = processor.process_dataset(ds_obs)

if not Path(PATH_OBS).exists():
    # %% [markdown]
    # Plotting of selected variable (?)

    # %%
    date = ds_obs.isel(time=0)["time"].values
    formatted_datetime = pd.to_datetime(date)
    formatted_date = formatted_datetime.strftime("%Y-%m-%d")
    hour = formatted_datetime.strftime("%H")

    # %%
    print(ds_obs)

    # %%
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND)

    # Scatter plot of station locations
    ax.scatter(
        ds_obs.lon,
        ds_obs.lat,
        transform=ccrs.PlateCarree(),
        s=20,
        c=ds_obs.t2m.isel(time=0),
        marker="o",
    )

    # Set plot title and labels
    plt.title("Station Locations with Coastlines")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")

# %%
if not Path(PATH_OBS).exists():
    ds_obs.to_zarr(PATH_OBS, mode="w")

# %% [markdown]
# ### Read in obs data

# %%
ds_obs = xr.open_zarr(PATH_OBS, decode_timedelta=True)
ds_obs = subset_dataset_for_debug(
    ds_obs,
    label="observation zarr",
    dims_to_trim=("stationId",),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
)
ds_obs = ds_obs.rename_vars(OBS_VAR_MAPPING)
ds_obs = ds_obs.rename_dims(
    {
        "stationId": "station",
    }
)
ds_obs = ds_obs.rename_vars(
    {
        "stationId": "station",
        "lat": "latitude",
        "lon": "longitude",
    }
)

# Don't need any of the below
# ds_obs = ds_obs.sel(time=np.unique(ds_ml.forecast_time.values.flatten()))
# ds_obs = ds_obs.where(ds_obs != 32767, np.nan)
# ds_obs = calculate_wind_components(ds_obs)
# ds_obs["temperature_2m"] += 273.15  # Convert to Kelvin
# ds_obs["surface_pressure"] *= 100  # Convert to Pa
ds_obs = select_available_times(
    ds_obs,
    ds_ml.forecast_time.values.flatten(),
    label="observations",
)
ds_ml = filter_forecast_dataset_by_time_coverage(ds_ml, ds_obs, label="ML")
ds_nwp, ds_ml = align_on_common_coord(
    ds_nwp,
    ds_ml,
    coord_name="start_time",
    left_label="NWP",
    right_label="ML",
)
ds_obs = select_available_times(
    ds_obs,
    ds_ml.forecast_time.values.flatten(),
    label="observations synchronized",
)
ds_obs

# %% [markdown]
# ### Define plotting functions

# %%
# Create directories for plots and tables
Path("verification/danra/case_study").mkdir(parents=True, exist_ok=True)
Path("verification/danra/sparse").mkdir(parents=True, exist_ok=True)
Path("verification/danra/gridded").mkdir(parents=True, exist_ok=True)

# Colorblind-friendly color palette
COLORS = {
    "gt": "#000000",  # Black
    "ml": "#E69F00",  # Orange
    "nwp": "#56B4E9",  # Light blue
    "error": "#CC79A7",  # Pink
}

# Line styles and markers for accessibility
LINE_STYLES = {
    "gt": ("solid", "o"),
    "ml": ("dashed", "s"),
    "nwp": ("dotted", "^"),
}

# Set global font sizes
apply_style(SCALE_METRICS)

# Colorblind-friendly colormap for 2D plots
COLORMAP = "viridis"


def get_colormap_for_variable(variable_name: str) -> str:
    """Return an appropriate colormap for a given physical variable."""
    v = variable_name.lower()
    if any(k in v for k in ("wind_u", "wind_v", "vertical_velocity")):
        return "RdBu_r"
    if "precipitation" in v or "tot_prec" in v:
        return "Blues"
    if any(k in v for k in ("temperature", "radiation", "heat_flux")):
        return "magma"
    if "humidity" in v:
        return "Greens"
    if "wind_speed" in v:
        return "YlOrRd"
    if any(k in v for k in ("pressure", "geopotential")):
        return "viridis"
    return "viridis"


def save_plot(fig, name, time=None, remove_title=True, dpi=300):
    """Save figure as PDF and accompanying .npz with all plotted data.

    Args:
        fig: matplotlib figure object
        name (str): base name for the plot file
        time (datetime, optional): timestamp to append to filename
        remove_title (bool): remove suptitle/title hierarchically if True
        dpi (int): resolution for the saved figure, defaults to 300
    """
    if time is not None:
        name = f"{name}_{time.dt.strftime('%Y%m%d_%H').values}"

    # Sanitize filename by replacing problematic characters
    safe_name = name.replace("/", "_per_")

    # Normalize the path and ensure plots directory exists
    full_name = f"observations_{safe_name}"
    if (
        full_name.startswith("observations_comparison_")
        or full_name.startswith("observations_interpolated_comparison_panel_")
        or full_name.startswith("observations_mean_error_map_")
    ):
        plot_dir = Path("verification/danra/case_study")
    elif full_name.startswith("observations_"):
        plot_dir = Path("verification/danra/sparse")
    else:
        plot_dir = Path("verification/danra/gridded")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Remove titles if requested
    if remove_title:
        if hasattr(fig, "texts") and fig.texts:  # Check for suptitle
            fig.suptitle("")
        ax = fig.gca()
        if ax.get_title():
            ax.set_title("")

    pdf_path = plot_dir / f"observations_{safe_name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)

    # --- persist plotted data as .npz ---
    npz_payload = {}
    for ax_idx, ax in enumerate(fig.get_axes()):
        for line_idx, line in enumerate(ax.get_lines()):
            label = line.get_label()
            if label and not label.startswith("_"):
                key = f"ax{ax_idx}_{label.replace(' ', '_')}"
            else:
                key = f"ax{ax_idx}_line{line_idx}"
            npz_payload[f"{key}_x"] = np.asarray(line.get_xdata(), dtype=float)
            npz_payload[f"{key}_y"] = np.asarray(line.get_ydata(), dtype=float)
        for img_idx, img in enumerate(ax.get_images()):
            npz_payload[f"ax{ax_idx}_img{img_idx}"] = np.asarray(
                img.get_array()
            )
        for col_idx, col in enumerate(ax.collections):
            arr = col.get_array()
            if arr is not None:
                label = col.get_label() or ""
                key = (
                    f"ax{ax_idx}_{label.replace(' ', '_')}"
                    if label and not label.startswith("_")
                    else f"ax{ax_idx}_col{col_idx}"
                )
                npz_payload[key] = np.asarray(arr, dtype=float)
    np.savez_compressed(
        plot_dir / f"observations_{safe_name}.npz", **npz_payload
    )


def export_table(df, name, caption=""):
    """Helper function to export tables consistently"""
    # Export to LaTeX with caption
    latex_str = df.to_latex(
        float_format="%.4f", caption=caption, label=f"tab:{name}"
    )
    with open(f"verification/danra/sparse/observations_{name}.tex", "w") as f:
        f.write(latex_str)

    # Export to CSV
    df.to_csv(f"verification/danra/sparse/observations_{name}.csv")


# %% [markdown]
# ### Check missing values

# %% [markdown]
# Check for missing data in any of the variables. If you have missing data
# and don't treat it properly, the corresponding time steps will be removed
# from the evaluation. This can lead to a bias in the evaluation.


# %%
def analyze_missing_data(ds_obs):
    """
    Create a 2D table of missing data percentages by variable and month.

    Args:
        ds_obs (xarray.Dataset): Observation dataset
    """
    # Convert time to pandas datetime for month extraction
    times = pd.DatetimeIndex(ds_obs.time.values)

    # Calculate total stations with missing data
    stations_with_missing = len(
        [
            station
            for station in ds_obs.station
            if np.isnan(ds_obs.sel(station=station)).any()
        ]
    )

    # Initialize results dictionary
    missing_by_month = {}

    # Calculate percentages for each variable and month
    for var in ds_obs.data_vars:
        missing_by_month[var] = {}
        for month in range(1, 13):
            month_mask = times.month == month
            if not any(month_mask):
                continue

            total_elements = ds_obs.sizes["station"] * month_mask.sum()
            n_missing = (
                np.isnan(ds_obs[var].sel(time=ds_obs.time[month_mask]))
                .sum()
                .values
            )
            missing_by_month[var][month] = (n_missing / total_elements) * 100

    # Create DataFrame
    df = pd.DataFrame(missing_by_month)

    # Format percentages to 2 decimal places
    df = df.round(2)

    # Print summary and table
    print(
        f"\nMissing Data Analysis: {stations_with_missing} out of"
        f" {ds_obs.sizes['station']} stations affected"
    )
    print("Percentage of Missing Values by Variable and Month:")
    print("=" * 50)
    print(df)

    return df


# Call the function
if CHECK_MISSING:
    analyze_missing_data(ds_obs)

# %%
assert ds_obs.sizes["time"] == len(
    np.unique(ds_ml.forecast_time.values.flatten())
), (
    f"Number of time steps do not match: {ds_obs.sizes['time']} !="
    f" {len(np.unique(ds_ml.forecast_time.values.flatten()))}"
)
assert ds_ml.sizes["start_time"] == ds_nwp.sizes["start_time"]


# %% [markdown]
# ## Interpolation to station positions
# Interpolation from the gridded model data to the station locations is
# done using the nearest neighbor method. Since we are only looking at
# surface level data, this is a reasonable approach. However, if you are
# working with data at different levels, you may want to consider a more
# sophisticated interpolation method.

# %% [markdown]
# #### Interpolate in projection


# %%
def interpolate_to_obs(
    ds_model_1, ds_model_2, ds_obs, vars_plot, neighbors=None
):
    """Optimized function to interpolate model datasets to observation
    points using xarray/dask."""

    print("Starting optimized interpolation...")

    # Project before computing distances and weights
    source_crs = ccrs.PlateCarree()
    # Transform the coordinates

    # Extract observation coordinates

    points_obs = PROJECTION.transform_points(
        source_crs,
        ds_obs.longitude.values,
        ds_obs.latitude.values,
    )[:, :2]

    # Extract model coordinates from 2D lat/lon arrays
    model_lats = ds_model_1.latitude
    model_lons = ds_model_1.longitude

    points_model = PROJECTION.transform_points(
        source_crs,
        model_lons.values.ravel(),
        model_lats.values.ravel(),
    )[:, :2]

    # Build KDTree
    print("Building KD-tree and finding neighbors...")
    k = 4 if neighbors is None else neighbors
    kdtree = cKDTree(points_model)
    distances, flat_indices = kdtree.query(points_obs, k=k)

    # Convert flat indices back to 2D indices
    _, ny = ds_model_1.x.size, ds_model_1.y.size
    x_indices = flat_indices // ny
    y_indices = flat_indices % ny

    # Precompute weights with proper dimensions
    weights = xr.DataArray(
        1.0 / (distances + 1e-10), dims=["station", "neighbor"]
    )
    weights = weights / weights.sum("neighbor")

    def interpolate_variable(var):
        print(f"Processing variable: {var}")

        # Select nearest neighbors for both datasets
        data_1 = ds_model_1[var].isel(
            x=xr.DataArray(x_indices, dims=["station", "neighbor"]),
            y=xr.DataArray(y_indices, dims=["station", "neighbor"]),
        )
        data_2 = ds_model_2[var].isel(
            x=xr.DataArray(x_indices, dims=["station", "neighbor"]),
            y=xr.DataArray(y_indices, dims=["station", "neighbor"]),
        )

        # Create interpolated datasets using weighted sum along
        # neighbor dimension
        result_1 = (data_1 * weights).sum(dim="neighbor")
        result_2 = (data_2 * weights).sum(dim="neighbor")

        return var, result_1, result_2

    # Process all variables
    results = {}
    for var in vars_plot:
        var_name, data_1, data_2 = interpolate_variable(var)
        results[var_name] = (data_1, data_2)

    # Create output datasets
    ds_interp_1 = xr.Dataset(
        {var: results[var][0] for var in vars_plot},
        coords={
            "start_time": ds_model_1.start_time,
            "elapsed_forecast_duration": ds_model_1.elapsed_forecast_duration,
            "station": ds_obs.station,
            "forecast_time": ds_model_1.forecast_time,
            "latitude": ds_obs.latitude,
            "longitude": ds_obs.longitude,
        },
    )

    ds_interp_2 = xr.Dataset(
        {var: results[var][1] for var in vars_plot},
        coords={
            "start_time": ds_model_2.start_time,
            "elapsed_forecast_duration": ds_model_2.elapsed_forecast_duration,
            "station": ds_obs.station,
            "forecast_time": ds_model_2.forecast_time,
            "latitude": ds_obs.latitude,
            "longitude": ds_obs.longitude,
        },
    )

    return ds_interp_1, ds_interp_2


ds_ml_interp, ds_nwp_interp = interpolate_to_obs(
    ds_ml, ds_nwp, ds_obs, VARIABLES_ML.values(), neighbors=4
)

with ProgressBar():
    print("Computing interpolated datasets...")
    ds_ml_interp = ds_ml_interp.compute()
    print("ML interpolation done.")
    ds_nwp_interp = ds_nwp_interp.compute()
    print("NWP interpolation done.")

# %% [markdown]
# ### 1. Maps
#
# **Random Time Selection:** A random time step is selected to avoid bias
# in the comparison, ensuring that the assessment is representative of
# typical model performance.
#
# **Consistent Color Scales:** By setting the same minimum and maximum
# values across all datasets for each variable, we ensure that color
# differences in the plots reflect true discrepancies, not artifacts of
# scaling.
#
# **Spatial Patterns:** The plots reveal how the ML model and NWP model
# represent geographical features like weather fronts, high and low-pressure
# systems, and temperature gradients. Visual comparisons can immediately
# highlight areas where the models perform well or poorly, guiding further
# investigation.
#
# **Edge Effects:** Near the boundaries, artifacts may occur as the model
# does not calculate a loss in the boundary region.

# %%
# Define the map extent based on the station data
extent = [
    ds_obs.longitude.min().values - 0.1,
    ds_obs.longitude.max().values + 0.1,
    ds_obs.latitude.min().values - 0.1,
    ds_obs.latitude.max().values + 0.1,
]
if PLOT_TIME is None:
    plot_time = None
else:
    plot_time = resolve_requested_time(
        ds_ml,
        "start_time",
        PLOT_TIME,
        label="plot time",
        allow_nearest=DEBUG_MODE,
    )


# %%
def add_map_features(axes):
    n_rows, _ = axes.shape
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            ax.coastlines(resolution="50m")
            ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.7)
            gl = ax.gridlines(
                draw_labels=True,
                dms=True,
                x_inline=False,
                y_inline=False,
                rotate_labels=False,
            )

            # Turn off all labels by default
            gl.top_labels = False
            gl.bottom_labels = False
            gl.left_labels = False
            gl.right_labels = False

            # Enable left labels only for leftmost column
            if j == 0:
                gl.left_labels = True

            # Enable bottom labels only for last row
            if i == n_rows - 1:
                gl.bottom_labels = True


def plot_comparison_maps(ds_obs, ds_ml, ds_nwp, plot_time=None, variables=None):
    """
    Plot comparison between observations, NWP and ML data for each
    variable and forecast step.

    Args:
        ds_obs (xarray.Dataset): Observations dataset
        ds_ml (xarray.Dataset): ML forecast data
        ds_nwp (xarray.Dataset): NWP forecast data
        plot_time (str): Time for plot title
        variables (list): List of variables to plot. If None, plots all
            common variables
    """
    # Convert plot_time to pandas datetime if it's a string
    if isinstance(plot_time, str):
        plot_time = pd.to_datetime(plot_time)

    u_var, v_var = "wind_u_10m", "wind_v_10m"
    pressure_var = "seasurface_pressure"
    has_wind = u_var in ds_ml.data_vars and v_var in ds_ml.data_vars

    # Get common variables if not specified; replace u/v with wind_speed
    if variables is None:
        variables = list(set(ds_nwp.data_vars).intersection(ds_ml.data_vars))
    if has_wind:
        variables = [v for v in variables if v not in (u_var, v_var)]
        if "wind_speed" not in variables:
            variables.insert(0, "wind_speed")

    def _get_wind_speed(ds_slice):
        return np.sqrt(
            ds_slice[u_var].values ** 2 + ds_slice[v_var].values ** 2
        )

    def _get_data(ds_slice, var):
        if var == "wind_speed":
            if "wind_speed" in ds_slice.data_vars:
                return ds_slice["wind_speed"].values
            return _get_wind_speed(ds_slice)
        return ds_slice[var].values

    # Get number of forecast steps
    n_steps = len(ds_ml.elapsed_forecast_duration)

    # Compute figure width from map aspect ratio to avoid column gaps
    # Standard library
    import math as _math

    lon_span = extent[1] - extent[0]
    lat_span = extent[3] - extent[2]
    lat_mid = (extent[2] + extent[3]) / 2
    _map_aspect = lon_span * _math.cos(_math.radians(lat_mid)) / lat_span
    _fig_w = max(15.0, min(27.0, 3 * 6 * _map_aspect + 3))

    for var in variables:
        fig = plt.figure(figsize=(_fig_w, 6 * n_steps), dpi=DPI)
        axes = np.array(
            [
                [
                    plt.subplot(
                        n_steps, 3, i * 3 + j + 1, projection=PROJECTION
                    )
                    for j in range(3)
                ]
                for i in range(n_steps)
            ]
        )

        for ax_row in axes:
            for ax in ax_row:
                ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Collect data for global min/max
        arrays_for_minmax = []
        for step_idx, step in enumerate(ds_ml.elapsed_forecast_duration):
            forecast_time = ds_ml.forecast_time.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )
            obs_slice = ds_obs.sel(time=forecast_time)
            nwp_slice = ds_nwp.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )
            ml_slice = ds_ml.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )
            arrays_for_minmax.extend(
                [
                    _get_data(obs_slice, var),
                    _get_data(nwp_slice, var),
                    _get_data(ml_slice, var),
                ]
            )

        vmin = float(np.nanmin([np.nanmin(a) for a in arrays_for_minmax]))
        vmax = float(np.nanmax([np.nanmax(a) for a in arrays_for_minmax]))

        _cmap = get_colormap_for_variable(var)
        if var == "wind_speed":
            vmin = 0.0
        scatter_kw = dict(
            cmap=_cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree()
        )
        mesh_kw = dict(
            cmap=_cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
            rasterized=True,
        )

        def _add_pressure_contour(ax, ds_slice):
            if pressure_var in ds_slice.data_vars:
                p = ds_slice[pressure_var].values / 100  # Pa → hPa
                p_min = float(np.nanmin(p))
                p_max = float(np.nanmax(p))
                levels = np.arange(
                    np.floor(p_min / 4) * 4,
                    np.ceil(p_max / 4) * 4 + 4,
                    4,
                )
                cs = ax.contour(
                    ds_slice.longitude.values,
                    ds_slice.latitude.values,
                    p,
                    levels=levels,
                    colors="k",
                    linewidths=0.7,
                    alpha=0.6,
                    transform=ccrs.PlateCarree(),
                )
                # For each level find midpoint of longest in-extent
                # segment, transform to projection coords, and pass
                # as manual positions so every line gets an inline label
                lon0, lon1, lat0, lat1 = extent
                manual_positions = []
                for lvl_idx in range(len(cs.levels)):
                    best, best_n = None, 0
                    for seg in cs.allsegs[lvl_idx]:
                        mask = (
                            (seg[:, 0] >= lon0)
                            & (seg[:, 0] <= lon1)
                            & (seg[:, 1] >= lat0)
                            & (seg[:, 1] <= lat1)
                        )
                        if mask.sum() > best_n:
                            best_n = mask.sum()
                            best = seg[mask]
                    if best is not None and best_n > 0:
                        mid = best[best_n // 2]
                        proj_xy = ax.projection.transform_point(
                            mid[0], mid[1], ccrs.PlateCarree()
                        )
                        manual_positions.append(proj_xy)
                if manual_positions:
                    ax.clabel(
                        cs,
                        fmt="%d",
                        fontsize=6,
                        inline=True,
                        manual=manual_positions,
                    )

        for step_idx, step in enumerate(ds_ml.elapsed_forecast_duration):
            forecast_time = ds_ml.forecast_time.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )
            forecast_hours = int(step.values / 1e9 / 3600)

            obs_slice = ds_obs.sel(time=forecast_time)
            nwp_slice = ds_nwp.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )
            ml_slice = ds_ml.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )

            # Observations (scatter — no pressure contour for point data)
            axes[step_idx, 0].scatter(
                ds_obs.longitude,
                ds_obs.latitude,
                c=_get_data(obs_slice, var),
                **scatter_kw,
            )

            # NWP (gridded pcolormesh + pressure isolines)
            axes[step_idx, 1].pcolormesh(
                ds_nwp.longitude,
                ds_nwp.latitude,
                _get_data(nwp_slice, var),
                **mesh_kw,
            )
            _add_pressure_contour(axes[step_idx, 1], nwp_slice)

            # ML (gridded pcolormesh + pressure isolines)
            im2 = axes[step_idx, 2].pcolormesh(
                ds_ml.longitude,
                ds_ml.latitude,
                _get_data(ml_slice, var),
                **mesh_kw,
            )
            _add_pressure_contour(axes[step_idx, 2], ml_slice)

            if step_idx == 0:
                axes[step_idx, 0].set_title(f"Observations\n+{forecast_hours}h")
                axes[step_idx, 1].set_title(f"NWP\n+{forecast_hours}h")
                axes[step_idx, 2].set_title(f"ML\n+{forecast_hours}h")
            else:
                axes[step_idx, 0].set_title(f"+{forecast_hours}h")
                axes[step_idx, 1].set_title(f"+{forecast_hours}h")
                axes[step_idx, 2].set_title(f"+{forecast_hours}h")

        add_map_features(axes)

        plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.105, wspace=0.003)

        cbar_ax = fig.add_axes([0.15, 0.00, 0.7, 0.02])
        cbar = plt.colorbar(  # noqa: F841
            im2,
            cax=cbar_ax,
            orientation="horizontal",
            label=f"({VARIABLE_UNITS.get(var, var)})",
        )

        plt.suptitle(
            f"{var} Comparison at {plot_time.dt.date.values!s}"
            f" - {plot_time.dt.hour.values!s} UTC",
            y=0.95,
        )

        plt.show()
        save_plot(fig, f"comparison_{var}_multi_step", time=plot_time)
        plt.close()


# Call the function
plot_comparison_maps(
    ds_obs,
    ds_ml.isel(elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT),
    ds_nwp.isel(elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT),
    plot_time=plot_time,
)


# %% [markdown]
# After interpolation all three datasets contain the same number of
# station-level data points.
# These datapoints are visualised on a map to show the spatial distribution
# of the stations.


# %%
# Visualization of the interpolated model data
def plot_comparison_interpolated(
    ds_obs, ds_ml_interp, ds_nwp_interp, var_plot, plot_time=None
):
    """Plot comparison between observations and interpolated model datasets
    across forecast steps."""
    if isinstance(plot_time, str):
        plot_time = pd.to_datetime(plot_time)

    n_steps = len(ds_nwp_interp.elapsed_forecast_duration)

    # Increased top margin for suptitle
    # Standard library
    import math as _math

    lon_span = extent[1] - extent[0]
    lat_span = extent[3] - extent[2]
    lat_mid = (extent[2] + extent[3]) / 2
    _map_aspect = lon_span * _math.cos(_math.radians(lat_mid)) / lat_span
    _fig_w = max(15.0, min(27.0, 3 * 6 * _map_aspect + 3))
    fig = plt.figure(figsize=(_fig_w, 6 * n_steps), dpi=DPI)
    axes = np.array(
        [
            [
                plt.subplot(n_steps, 3, i * 3 + j + 1, projection=PROJECTION)
                for j in range(3)
            ]
            for i in range(n_steps)
        ]
    )

    # Set extent for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

    arrays_for_minmax = []
    for step_idx, step in enumerate(ds_nwp_interp.elapsed_forecast_duration):
        forecast_time = ds_nwp_interp.forecast_time.sel(
            start_time=plot_time, elapsed_forecast_duration=step
        )
        arrays_for_minmax.extend(
            [
                ds_obs[var_plot].sel(time=forecast_time).values,
                ds_nwp_interp[var_plot]
                .sel(start_time=plot_time, elapsed_forecast_duration=step)
                .values,
                ds_ml_interp[var_plot]
                .sel(start_time=plot_time, elapsed_forecast_duration=step)
                .values,
            ]
        )

    vmin = min(np.nanmin(arr) for arr in arrays_for_minmax)
    vmax = max(np.nanmax(arr) for arr in arrays_for_minmax)

    for step_idx, step in enumerate(ds_nwp_interp.elapsed_forecast_duration):
        forecast_time = ds_nwp_interp.forecast_time.sel(
            start_time=plot_time, elapsed_forecast_duration=step
        )
        forecast_hours = int(step.values / 1e9 / 3600)

        for ax_idx, (ax, data, title) in enumerate(
            zip(
                axes[step_idx],
                [ds_obs, ds_nwp_interp, ds_ml_interp],
                ["Observations", "NWP", "ML"],
            )
        ):
            _cmap = get_colormap_for_variable(var_plot)
            if ax_idx == 0:
                scatter = ax.scatter(
                    ds_obs.longitude,
                    ds_obs.latitude,
                    c=data[var_plot].sel(time=forecast_time),
                    cmap=_cmap,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=True,
                )
            else:
                scatter = ax.scatter(
                    ds_obs.longitude,
                    ds_obs.latitude,
                    c=data[var_plot].sel(
                        start_time=plot_time, elapsed_forecast_duration=step
                    ),
                    cmap=_cmap,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=True,
                )

            ax.add_feature(cfeature.COASTLINE, edgecolor="grey")
            ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="grey")

            # Configure gridlines - only on edges
            gl = ax.gridlines(
                draw_labels=True,
                alpha=0.2,
                x_inline=False,
                y_inline=False,
                rotate_labels=False,
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = ax_idx == 0  # Only leftmost column
            gl.bottom_labels = step_idx == n_steps - 1  # Only bottom row

            if step_idx == 0:
                ax.set_title(f"{title}\n+{forecast_hours}h")
            else:
                ax.set_title(f"+{forecast_hours}h")

    # Adjust subplot spacing
    plt.subplots_adjust(
        top=0.9,
        bottom=0.05,
        hspace=0.15,
        wspace=0.003,
    )

    # Add colorbar with adjusted position
    cbar_ax = fig.add_axes([0.15, 0.0, 0.7, 0.02])
    plt.colorbar(
        scatter,
        cax=cbar_ax,
        orientation="horizontal",
        label=f"({VARIABLE_UNITS[var_plot]})",
    )

    # Adjusted suptitle position
    plt.suptitle(
        f"{var_plot} Comparison at {plot_time.dt.date.values!s}"
        f" - {plot_time.dt.hour.values!s} UTC",
        y=0.95,  # Higher position
        fontsize=FONT_SIZES["suptitle"],
    )
    return fig


apply_style(SCALE_INTERP_DANRA)

# Call the function for each variable
for var in VARIABLES_ML.values():
    fig = plot_comparison_interpolated(
        ds_obs,
        ds_ml_interp.isel(
            elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT
        ),
        ds_nwp_interp.isel(
            elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT
        ),
        var,
        plot_time=plot_time,
    )
    plt.show()
    save_plot(fig, f"interpolated_comparison_panel_{var}", time=plot_time)


# %%
def plot_mean_error_maps(ds_obs, ds_ml, ds_nwp, var=None):
    """Plot mean error maps comparing ML and NWP predictions against
    ground truth."""
    # Setup
    variables = [var] if var else VARIABLES_ML.values()
    n_forecast_times = len(ds_ml.elapsed_forecast_duration)

    for var in variables:
        # Configure plot layout
        n_cols = 2 if (ds_nwp is not None and var in ds_nwp) else 1

        # Increased figure height for colorbar space
        fig = plt.figure(figsize=(7.5 * n_cols, 6 * n_forecast_times), dpi=DPI)
        axes = np.array(
            [
                [
                    plt.subplot(
                        n_forecast_times,
                        n_cols,
                        i * n_cols + j + 1,
                        projection=PROJECTION,
                    )
                    for j in range(n_cols)
                ]
                for i in range(n_forecast_times)
            ]
        )
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Set extent for all subplots
        for ax_row in axes:
            for ax in ax_row:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
        # Combined error calculation
        arrays_for_minmax = []
        for idx, forecast_time in enumerate(ds_ml.elapsed_forecast_duration):
            forecast_hours = int(forecast_time.values / 1e9 / 3600)
            obs_times = ds_ml.forecast_time.sel(
                elapsed_forecast_duration=forecast_time
            )

            # Calculate both errors at once
            errors = {}
            errors["ml"] = mean_error(
                ds_ml[var].sel(
                    elapsed_forecast_duration=forecast_time,
                ),
                ds_obs[var].sel(time=obs_times),
                preserve_dims=["station"],
            )
            arrays_for_minmax.append(errors["ml"])

            if ds_nwp is not None and var in ds_nwp:
                errors["nwp"] = mean_error(
                    ds_nwp[var].sel(
                        elapsed_forecast_duration=forecast_time,
                    ),
                    ds_obs[var].sel(time=obs_times),
                    preserve_dims=["station"],
                )
                arrays_for_minmax.append(errors["nwp"])

            # Calculate global min/max with symmetric limits around 0
            abs_max = max(abs(np.nanmax(arr)) for arr in arrays_for_minmax)
            abs_max = max(
                abs_max, abs(min(np.nanmin(arr) for arr in arrays_for_minmax))
            )
            vmin, vmax = -abs_max, abs_max

            if ds_nwp is not None and var in ds_nwp:
                # Plot NWP errors
                axes[idx, 0].scatter(
                    ds_obs.longitude,
                    ds_obs.latitude,
                    c=errors["nwp"],
                    transform=ccrs.PlateCarree(),
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=True,
                )
                axes[idx, 0].set_title(
                    f"Mean NWP Error\n+{forecast_hours}h"
                    if idx == 0
                    else f"+{forecast_hours}h"
                )

            # Plot ML errors
            col = 1 if ds_nwp is not None and var in ds_nwp else 0
            scatter = axes[idx, col].scatter(
                ds_obs.longitude,
                ds_obs.latitude,
                c=errors["ml"],
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            axes[idx, col].set_title(
                f"Mean ML Error\n+{forecast_hours}h"
                if idx == 0
                else f"+{forecast_hours}h"
            )

            # Add map features
            for ax in axes[idx]:
                ax.add_feature(cfeature.COASTLINE, edgecolor="grey")
                ax.add_feature(
                    cfeature.BORDERS, linestyle=":", edgecolor="grey"
                )

                gl = ax.gridlines(
                    draw_labels=True,
                    alpha=0.2,
                    x_inline=False,
                    y_inline=False,
                    rotate_labels=False,
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.left_labels = ax == axes[idx, 0]
                gl.bottom_labels = idx == n_forecast_times - 1

        # Adjust subplot spacing
        plt.subplots_adjust(
            top=0.9,
            bottom=0.05,
            hspace=0.105,
            wspace=0.0105,
        )

        # Add colorbar with adjusted position
        cbar_ax = fig.add_axes([0.15, 0.0, 0.7, 0.02])
        plt.colorbar(
            scatter,
            cax=cbar_ax,
            orientation="horizontal",
            label=f"Mean Error ({VARIABLE_UNITS[var]})",
        )

        # Add title
        plt.suptitle(
            f"Mean Error in {var}",
            y=0.95,
        )
        plt.show()
        save_plot(fig, f"mean_error_map_{var}", time=plot_time)


# mean error maps use base font (not scaled like interp comparison)
apply_style()
fig = plot_mean_error_maps(
    ds_obs,
    ds_ml_interp.isel(elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT),
    ds_nwp_interp.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT
    ),
)
apply_style()  # restore base


# %% [markdown]
# ### 2. Histograms
# By examining these distributions, we can assess whether the ML model and
# NWP model accurately capture the variability and frequency of different
# atmospheric states.
#
# **Distribution Shape:** The histograms show whether the models replicate
# the skewness, kurtosis, and overall shape of the ground truth data
# distributions.
#
# **Extreme Values:** Identifying how the models handle extreme conditions,
# such as unusually high or low temperatures, is crucial for weather
# prediction and risk assessment.
#
# **Normalization Needs:** Differences in scale between variables suggest
# that normalization may be necessary for accurate comparisons.


# %%
def plot_interpolated_histograms(ds_obs, ds_ml_interp, ds_nwp_interp):
    """Plot histograms for interpolated station data comparing obs, NWP and ML.

    Args:
        ds_obs: xarray Dataset containing observations
        ds_ml_interp: xarray Dataset containing interpolated ML predictions
        ds_nwp_interp: xarray Dataset containing interpolated NWP predictions
    """
    for variable_name in VARIABLES_ML.values():
        if variable_name not in ds_obs:
            continue

        fig, ax = plt.subplots(figsize=(16, 7), dpi=DPI)

        # Convert to numpy arrays
        data_obs = ds_obs[variable_name].values.flatten()
        data_ml = ds_ml_interp[variable_name].values.flatten()
        data_nwp = ds_nwp_interp[variable_name].values.flatten()

        # Special handling of bins for temperature, that takes
        # discretized values
        num_bins = 100 if variable_name == "temperature_2m" else 300

        # Create histograms for observations
        ax.hist(
            data_obs,
            bins=num_bins,
            alpha=0.7,
            density=True,
            color=COLORS["gt"],
            label="Observations",
            histtype="stepfilled",
            linewidth=0,
        )

        # Plot NWP interpolated data
        ax.hist(
            data_nwp,
            bins=num_bins,
            alpha=0.5,
            density=True,
            color=COLORS["nwp"],
            label="NWP Interpolated",
            histtype="stepfilled",
            linewidth=0,
        )

        # Create histogram for ML interpolated data
        ax.hist(
            data_ml,
            bins=num_bins,
            alpha=0.5,
            density=True,
            color=COLORS["ml"],
            label="ML Interpolated",
            histtype="stepfilled",
            linewidth=0,
        )

        # Add labels and title
        units = VARIABLE_UNITS[variable_name]
        ax.set_title(
            f"Distribution of {variable_name} at Station Locations", pad=20
        )
        ax.set_xlabel(f"({units})")

        # Place legend in top left
        ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        # Adjust axis limits
        current_ylim = ax.get_ylim()
        ax.set_ylim(0, current_ylim[1] * 1.3)

        # Calculate statistics
        stats_data = {
            "Obs": [
                f"{skew(data_obs, nan_policy='omit'):.2f}",
                f"{kurtosis(data_obs, nan_policy='omit'):.2f}",
            ],
            "NWP": [
                f"{skew(data_nwp, nan_policy='omit'):.2f}",
                f"{kurtosis(data_nwp, nan_policy='omit'):.2f}",
            ],
            "ML": [
                f"{skew(data_ml, nan_policy='omit'):.2f}",
                f"{kurtosis(data_ml, nan_policy='omit'):.2f}",
            ],
        }

        # Create and position table
        col_labels = ["Skewness", "Kurtosis"]
        row_labels = list(stats_data.keys())
        cell_text = [
            [stats_data[row][i] for i in range(2)] for row in row_labels
        ]

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="upper right",
            bbox=[
                0.68,
                0.6,
                0.3,
                0.35,
            ],  # Adjust position (x, y, width, height)
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(22)

        for (row, col), cell in table._cells.items():
            cell.set_text_props(wrap=True)
            cell.set_facecolor("white")
            cell.set_alpha(0.9)
            cell.set_edgecolor("#D3D3D3")

            if col == -1:
                cell.set_text_props(horizontalalignment="right")
            else:
                cell.set_text_props(horizontalalignment="center")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        save_plot(fig, f"histogram_interpolated_{variable_name}")
        plt.close()


# Call the function after interpolation
plot_interpolated_histograms(
    ds_obs,
    # Use values in short time span, up to 18h
    ds_ml_interp.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
    ),
    ds_nwp_interp.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
    ),
)


# %% [markdown]
# ### 3. Various Verification Metrics
# The final chapter consolidates various statistical metrics to provide a broad
# evaluation of the ML model's performance. By considering multiple metrics, we
# gain a nuanced understanding of both the strengths and weaknesses of
# the model.
#
# **Metric Diversity:** Including ME, STDEV-ERR, MAE, RMSE, MSE, Pearson
# correlation, covers different aspects of model performance, from average
# errors to spatial pattern accuracy.
#
# **ME** (Mean Error): Indicates the average discrepancy between the model
# and ground truth values. A positive value indicates that the model tends
# to overestimate, while a negative value suggests underestimation. Also
# called Bias.
#
# **STDEV-ERR** (Standard Deviation of Errors): Shows the variability of
# errors, highlighting whether the model is consistent in its predictions.
#
# **MAE, MSE and RMSE:** Offer insights into the average magnitude of
# errors, with
# RMSE emphasizing larger discrepancies. The colors indicating high errors are
# only implemented for these three metrics with standardization.
#
# **Pearson Correlation:** Assesses the linear relationship, indicating whether
# the model captures variability even if biases exist.
#
# **Wasserstein Distance:** Provides a holistic view of distributional
# similarity
# across variables. Same as chapter 3.
#
# **Holistic Assessment:** The combination of metrics provides a comprehensive
# performance profile, essential for model validation and comparison. More
# complex metrics are explained in more detail.


# %%
def calculate_metrics_by_efd(
    ds_obs,
    ds_ml,
    ds_nwp=None,
    metrics_to_compute=None,
    prefix="metrics",
):
    """Calculate metrics for each Lead Time for station data using xarray."""
    if metrics_to_compute is None:
        metrics_to_compute = METRICS

    variables = list(ds_obs.data_vars)
    elapsed_forecast_durations = ds_ml.elapsed_forecast_duration
    elapsed_forecast_durations_hours = elapsed_forecast_durations.values.astype(
        "timedelta64[s]"
    ) / np.timedelta64(1, "h")

    metrics_by_efd = {}
    combined_metrics = {}

    for efd, lt_hours in zip(
        elapsed_forecast_durations, elapsed_forecast_durations_hours
    ):
        print(f"\nCalculating metrics for lead time: {lt_hours.item():.1f}h")

        ds_ml_lead = ds_ml.sel(elapsed_forecast_duration=efd)
        comp_for_nwp = (ds_nwp is not None) and (
            efd.values in ds_nwp.elapsed_forecast_duration
        )
        if comp_for_nwp:
            ds_nwp_lead = ds_nwp.sel(elapsed_forecast_duration=efd)
            ds_nwp_lead["start_time"] = ds_nwp_lead.forecast_time
            ds_nwp_lead = ds_nwp_lead.rename_dims(
                {"start_time": "time"}
            ).rename_vars({"start_time": "time"})

        ds_ml_lead["start_time"] = ds_ml_lead.forecast_time
        ds_ml_lead = ds_ml_lead.rename_dims({"start_time": "time"}).rename_vars(
            {"start_time": "time"}
        )

        forecast_times = ds_ml_lead.forecast_time.values.flatten()
        ds_obs_lead = ds_obs.sel(time=forecast_times)
        metrics_dict = {}

        for var in variables:
            print(f"Processing {var}")

            # Get data as xarray DataArrays
            y_true = ds_obs_lead[var]
            y_pred_ml = ds_ml_lead[var]

            # Create masks for each dataset
            mask_true = xr.where(~np.isnan(y_true), True, False)
            mask_ml = xr.where(~np.isnan(y_pred_ml), True, False)

            if comp_for_nwp and var in ds_nwp:
                y_pred_nwp = ds_nwp_lead[var]
                mask_nwp = xr.where(~np.isnan(y_pred_nwp), True, False)
                # Combine masks
                valid_mask = mask_true & mask_ml & mask_nwp
            else:
                valid_mask = mask_true & mask_ml

            y_true = y_true.where(valid_mask)
            y_pred_ml = y_pred_ml.where(valid_mask)
            if ds_nwp is not None and var in ds_nwp:
                y_pred_nwp = y_pred_nwp.where(valid_mask)

            metrics_dict[var] = {}

            # Calculate ML metrics using xarray
            if "MAE" in metrics_to_compute:
                metrics_dict[var]["MAE ML"] = mae(y_pred_ml, y_true).values
            if "RMSE" in metrics_to_compute:
                metrics_dict[var]["RMSE ML"] = rmse(y_pred_ml, y_true).values
            if "MSE" in metrics_to_compute:
                metrics_dict[var]["MSE ML"] = mse(y_pred_ml, y_true).values
            if "ME" in metrics_to_compute:
                metrics_dict[var]["ME ML"] = mean_error(
                    y_pred_ml, y_true
                ).values
            if "STDEV_ERR" in metrics_to_compute:
                metrics_dict[var]["STDEV_ERR ML"] = (
                    (y_pred_ml - y_true).std().values
                )
            if "RelativeMAE" in metrics_to_compute:
                rel_mae = (
                    abs(y_pred_ml - y_true) / (abs(y_true) + 1e-6)
                ).mean()
                metrics_dict[var]["RelativeMAE ML"] = rel_mae.values
            if "RelativeRMSE" in metrics_to_compute:
                rel_rmse = np.sqrt(
                    ((y_pred_ml - y_true) ** 2 / (y_true**2 + 1e-6)).mean()
                )
                metrics_dict[var]["RelativeRMSE ML"] = rel_rmse.values
            if "PearsonR" in metrics_to_compute:
                metrics_dict[var]["PearsonR ML"] = pearsonr(
                    y_pred_ml, y_true
                ).values
            if "Wasserstein" in metrics_to_compute:
                # Use the valid_mask directly instead of checking for NaN
                # values again
                pred_vals = y_pred_ml.values[valid_mask.values]
                true_vals = y_true.values[valid_mask.values]
                metrics_dict[var]["Wasserstein ML"] = wasserstein_distance(
                    pred_vals, true_vals
                )

            # Calculate NWP metrics if available
            if comp_for_nwp and var in ds_nwp:
                if "MAE" in metrics_to_compute:
                    metrics_dict[var]["MAE NWP"] = mae(
                        y_pred_nwp, y_true
                    ).values
                if "RMSE" in metrics_to_compute:
                    metrics_dict[var]["RMSE NWP"] = rmse(
                        y_pred_nwp, y_true
                    ).values
                if "MSE" in metrics_to_compute:
                    metrics_dict[var]["MSE NWP"] = mse(
                        y_pred_nwp, y_true
                    ).values
                if "ME" in metrics_to_compute:
                    metrics_dict[var]["ME NWP"] = mean_error(
                        y_pred_nwp, y_true
                    ).values
                if "STDEV_ERR" in metrics_to_compute:
                    metrics_dict[var]["STDEV_ERR NWP"] = (
                        (y_pred_nwp - y_true).std().values
                    )
                if "RelativeMAE" in metrics_to_compute:
                    rel_mae = (
                        abs(y_pred_nwp - y_true) / (abs(y_true) + 1e-6)
                    ).mean()
                    metrics_dict[var]["RelativeMAE NWP"] = rel_mae.values
                if "RelativeRMSE" in metrics_to_compute:
                    rel_rmse = np.sqrt(
                        (
                            (y_pred_nwp - y_true) ** 2 / (y_true**2 + 1e-6)
                        ).mean()
                    )
                    metrics_dict[var]["RelativeRMSE NWP"] = rel_rmse.values
                if "PearsonR" in metrics_to_compute:
                    metrics_dict[var]["PearsonR NWP"] = pearsonr(
                        y_pred_nwp, y_true
                    ).values
                if "Wasserstein" in metrics_to_compute:
                    # For Wasserstein, we need to convert to numpy arrays
                    pred_vals = y_pred_nwp.values[valid_mask.values]
                    true_vals = y_true.values[valid_mask.values]
                    metrics_dict[var]["Wasserstein NWP"] = wasserstein_distance(
                        pred_vals, true_vals
                    )

            # Store combined metrics
            for metric_name, value in metrics_dict[var].items():
                key = f"{var}_{metric_name}"
                if key not in combined_metrics:
                    combined_metrics[key] = []
                combined_metrics[key].append(value)

        metrics_by_efd[lt_hours.item()] = pd.DataFrame.from_dict(
            metrics_dict, orient="index"
        )

    # Create combined metrics DataFrame
    elapsed_forecast_durations_hours_float = [  # noqa: F841
        x.item() for x in elapsed_forecast_durations_hours
    ]
    # combined_df = pd.DataFrame(
    #    combined_metrics, index=elapsed_forecast_durations_hours_float
    # )
    # combined_df.index.name = "Forecast Hours"

    return metrics_by_efd  # , combined_df


# %%
# metrics_by_efd, combined_metrics = calculate_metrics_by_efd(
# export_table(combined_metrics, "combined_metrics")
metrics_by_efd = calculate_metrics_by_efd(
    ds_obs=ds_obs,
    ds_ml=ds_ml_interp,
    ds_nwp=ds_nwp_interp,
    metrics_to_compute=METRICS,
)

# %%
# Plot evolution of a specific metric over lead time
elapsed_forecast_durations = list(metrics_by_efd.keys())
metrics_to_compute = METRICS
plot_counter = 0
for variable in VARIABLES_ML.values():
    for metric in metrics_to_compute:
        try:
            # Skip if any scores are missing
            ml_scores = [
                df.loc[variable, f"{metric} ML"]
                for df in metrics_by_efd.values()
            ]

            # Convert lead times from hours to timedelta
            hours = [
                x / np.timedelta64(1, "h")
                for x in ds_ml.elapsed_forecast_duration.values
            ]
            nwp_hours = [
                x / np.timedelta64(1, "h")
                for x in ds_nwp.elapsed_forecast_duration.values
            ]

            fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

            # Plot ML scores
            ax.plot(
                hours,
                ml_scores,
                label="ML",
                color=COLORS["ml"],
                linestyle=LINE_STYLES["ml"][0],
                marker=LINE_STYLES["ml"][1],
            )

            # Plot NWP scores if they exist and are not all NaN
            nwp_metric_name = f"{metric} NWP"
            nwp_scores = [
                df.loc[variable, nwp_metric_name]
                for df in metrics_by_efd.values()
                if nwp_metric_name in df  # Only for lead times where it exists
            ]

            if nwp_scores and not all(pd.isna(nwp_scores)):
                ax.plot(
                    nwp_hours,
                    nwp_scores,
                    label="NWP",
                    color=COLORS["nwp"],
                    linestyle=LINE_STYLES["nwp"][0],
                    marker=LINE_STYLES["nwp"][1],
                )

            ax.set_xlabel("Lead Time (hours)")
            ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
            ax.set_ylabel(f"{metric} ({VARIABLE_UNITS[variable]})")
            ax.set_title(f"{metric} Evolution for {variable}")
            ax.grid(True, alpha=0.3)
            # Only add legend for the first plot
            if plot_counter == 0:
                ax.legend()

            plot_counter += 1

            plt.tight_layout()
            plt.show()
            save_plot(fig, f"{metric}_{variable}_evolution")
            plt.close()

        except (KeyError, ValueError) as e:
            print(f"Skipping {metric} for {variable}: {e!s}")
            continue


# %% [markdown]
# #### Equitable Threat Score (Traditional Version)
# Range: [-1/3, 1], where:
# - 1 = perfect score
# - 0 = no skill compared to random chance
# - -1/3 = worst possible performance
#
# **Key Properties:**
# - Measures how well predicted events correspond to observed events,
#   accounting for hits due to random chance
# - Particularly useful for rare events (like precipitation above a high
#   threshold)
# - More equitable than simple Threat Score by accounting for hits due to
#   random chance
#
# **Advantages:**
# - Well-established metric in meteorological verification
# - Reference point at 0 makes interpretation clear
# - Penalizes both misses and false alarms
# - Accounts for random chance, making it more robust than basic threat scores
#
# #### Frequency Bias Index
# Range: 0 to infinity, where:
# - 1 = no bias
# - < 1 = underforecasting
# - > 1 = overforecasting
#
# **Key Properties:**
# - FBI measures the ratio of observed to forecasted events, indicating
#   whether the model tends to over- or underforecast
# - It's particularly useful for understanding systematic biases in event
#   frequency
#
# **Advantages:**
# - Provides a clear indication of over- or underforecasting
# - Easy to interpret: 1 indicates no bias, while values above or below 1
#   show the direction and magnitude of the bias


# %%
def calculate_meteoswiss_metrics(ds_obs, ds_ml, ds_nwp=None):
    """Calculate MeteoSwiss verification metrics (FBI and ETS) for station
    data."""
    metrics_by_var = {}

    # Get available variables in each dataset
    gt_ml_vars = set(ds_obs.variables) & set(ds_ml.variables)
    nwp_vars = set(ds_nwp.variables) if ds_nwp is not None else set()

    all_variables = {
        "precipitation": {
            "thresholds": THRESHOLDS_PRECIPITATION,
            "unit": "mm/h",
        },
        "wind_u_10m": {"thresholds": THRESHOLDS_WIND, "unit": "m/s"},
        "wind_v_10m": {"thresholds": THRESHOLDS_WIND, "unit": "m/s"},
    }

    # Filter variables that exist in gt and ml
    all_variables = {k: v for k, v in all_variables.items() if k in gt_ml_vars}

    # Initialize metrics structure
    for var_name in all_variables:
        metrics_by_var[var_name] = {}
        for thr in all_variables[var_name]["thresholds"]:
            metric_key = f"{thr}{all_variables[var_name]['unit']}"
            metrics_by_var[var_name][metric_key] = {
                "FBI_ML": [],
                "ETS_ML": [],
                "FBI_NWP": [] if ds_nwp is not None else None,
                "ETS_NWP": [] if ds_nwp is not None else None,
            }

    for efd in ds_ml.elapsed_forecast_duration.values:
        try:
            print(
                f"\nCalculating metrics for lead time:"
                f" {efd / np.timedelta64(1, 'h'):.1f}h"
            )

            ds_ml_lead = ds_ml.sel(elapsed_forecast_duration=efd)
            ds_nwp_lead = (
                ds_nwp.sel(elapsed_forecast_duration=efd)
                if ds_nwp is not None
                else None
            )
            forecast_times = ds_ml_lead.forecast_time  # No .values
            ds_obs_lead = ds_obs.sel(time=forecast_times)

            for var_name, var_config in all_variables.items():
                print(f"Processing {var_name}")
                try:
                    # Get data and create masks
                    y_true = ds_obs_lead[var_name]
                    y_ml = ds_ml_lead[var_name]
                    y_nwp = (
                        ds_nwp_lead[var_name]
                        if ds_nwp_lead is not None and var_name in nwp_vars
                        else None
                    )

                    # Create masks
                    mask_true = xr.where(~np.isnan(y_true), True, False)
                    mask_ml = xr.where(~np.isnan(y_ml), True, False)
                    if y_nwp is not None:
                        mask_nwp = xr.where(~np.isnan(y_nwp), True, False)
                        valid_mask = mask_true & mask_ml & mask_nwp
                    else:
                        valid_mask = mask_true & mask_ml

                    # Apply masks
                    y_true = y_true.where(valid_mask)
                    y_ml = y_ml.where(valid_mask)
                    if y_nwp is not None:
                        y_nwp = y_nwp.where(valid_mask)

                    for thr in var_config["thresholds"]:
                        metric_key = f"{thr}{var_config['unit']}"

                        # Calculate ML metrics using TEO
                        event_operator = TEO(default_event_threshold=thr)
                        ml_contingency = (
                            event_operator.make_contingency_manager(
                                y_ml, y_true
                            )
                        )

                        fbi_ml = ml_contingency.frequency_bias().values
                        ets_ml = ml_contingency.equitable_threat_score().values

                        metrics_by_var[var_name][metric_key]["FBI_ML"].append(
                            fbi_ml
                        )
                        metrics_by_var[var_name][metric_key]["ETS_ML"].append(
                            ets_ml
                        )

                        # Calculate NWP metrics if available
                        if y_nwp is not None:
                            nwp_contingency = (
                                event_operator.make_contingency_manager(
                                    y_nwp, y_true
                                )
                            )
                            fbi_nwp = nwp_contingency.frequency_bias().values
                            ets_nwp = (
                                nwp_contingency.equitable_threat_score().values
                            )

                            metrics_by_var[var_name][metric_key][
                                "FBI_NWP"
                            ].append(fbi_nwp)
                            metrics_by_var[var_name][metric_key][
                                "ETS_NWP"
                            ].append(ets_nwp)

                except Exception as e:
                    print(f"Error processing {var_name}: {e!s}")
                    continue

        except Exception as e:
            print(f"Error processing lead time {efd}: {e!s}")
            continue

    return metrics_by_var


# %%
def plot_meteoswiss_metrics_evolution(metrics_by_var, var_name, metric_prefix):
    """Plot evolution of MeteoSwiss metrics over forecast time with three
    fixed viridis colors."""
    # Get thresholds for this variable
    thresholds = list(metrics_by_var[var_name].keys())

    # Get three fixed colors from viridis (start, middle, end)
    colors = [plt.cm.viridis(x) for x in [0, 0.5, 0.99]]

    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    hours = [
        x / np.timedelta64(1, "h")
        for x in ds_ml_interp.isel(
            elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
        ).elapsed_forecast_duration.values
    ]

    # For each threshold
    for i, threshold in enumerate(thresholds):
        # Get ML and NWP metrics for this threshold
        ml_metric = f"{metric_prefix}_ML"
        nwp_metric = f"{metric_prefix}_NWP"

        # Extract values
        if ml_metric in metrics_by_var[var_name][threshold]:
            ml_values = metrics_by_var[var_name][threshold][ml_metric]
            ax.plot(
                hours,
                ml_values,
                linestyle=LINE_STYLES["ml"][0],
                marker=LINE_STYLES["ml"][1],
                color=colors[i],
                label=f"ML {threshold}",
                markevery=2,
            )

        if nwp_metric in metrics_by_var[var_name][threshold]:
            nwp_values = metrics_by_var[var_name][threshold][nwp_metric]
            ax.plot(
                hours,
                nwp_values,
                linestyle=LINE_STYLES["nwp"][0],
                marker=LINE_STYLES["nwp"][1],
                color=colors[i],
                label=f"NWP {threshold}",
                markevery=2,
            )

    ax.set_xlabel("Lead Time (hours)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
    ax.set_ylabel(f"{metric_prefix}")
    ax.set_title(f"{metric_prefix} for {var_name}")
    ax.grid(True, alpha=0.3)
    if metric_prefix == "FBI":
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    save_plot(fig, f"{metric_prefix.lower()}_{var_name}_evolution")


# %%
# Calculate MeteoSwiss metrics
def reshape_meteoswiss_metrics(metrics_dict, forecast_hours):
    """Reshape MeteoSwiss metrics into a DataFrame with variables/thresholds
    as columns and forecast hours as rows."""

    # Initialize DataFrame with MultiIndex columns
    columns = []
    values = []
    for var in metrics_dict:
        for threshold in metrics_dict[var]:
            for metric in metrics_dict[var][threshold]:
                col_name = f"{var}_{threshold}_{metric}"
                columns.append(col_name)
                # Extract values and convert from numpy arrays to floats
                metric_values = [
                    float(val) for val in metrics_dict[var][threshold][metric]
                ]
                values.append(metric_values)

    # Create DataFrame
    df = pd.DataFrame(np.array(values).T, columns=columns, index=forecast_hours)
    df.index.name = "forecast_hour"

    return df


# # Use it after calculate_meteoswiss_metrics
print("Calculating MeteoSwiss metrics...")
# Use only short time interval
meteoswiss_metrics = calculate_meteoswiss_metrics(
    ds_obs,
    ds_ml_interp.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
    ),
    ds_nwp_interp.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
    ),
)
metrics_df = reshape_meteoswiss_metrics(
    meteoswiss_metrics,
    ds_ml.isel(elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT)
    .elapsed_forecast_duration.values.astype("timedelta64[h]")
    .astype(int),
)

# Now export the reshaped DataFrame
export_table(
    metrics_df, "meteoswiss_metrics", caption="MeteoSwiss verification metrics"
)

# %%
if "precipitation" in ds_obs:
    plot_meteoswiss_metrics_evolution(
        meteoswiss_metrics, "precipitation", "ETS"
    )
    plot_meteoswiss_metrics_evolution(
        meteoswiss_metrics, "precipitation", "FBI"
    )

if "wind_u_10m" in ds_obs:
    plot_meteoswiss_metrics_evolution(meteoswiss_metrics, "wind_u_10m", "ETS")
    plot_meteoswiss_metrics_evolution(meteoswiss_metrics, "wind_u_10m", "FBI")

if "wind_v_10m" in ds_obs:
    plot_meteoswiss_metrics_evolution(meteoswiss_metrics, "wind_v_10m", "ETS")
    plot_meteoswiss_metrics_evolution(meteoswiss_metrics, "wind_v_10m", "FBI")


# %% [markdown]
# The wind vector RMSE takes into account the magnitude and direction of
# the wind, providing a more comprehensive measure of error than scalar
# metrics.


# %%
def wind_vector_rmse(u_pred, v_pred, u_true, v_true):
    """Calculate RMSE based on wind vector differences."""
    u_diff = u_true - u_pred
    v_diff = v_true - v_pred
    vector_diff = np.sqrt(u_diff**2 + v_diff**2)
    return float(np.sqrt(np.mean(vector_diff**2)))


def calculate_wind_vector_metrics(ds_obs, ds_ml_interp, ds_nwp_interp=None):
    """Calculate wind vector metrics for station data.

    Args:
        ds_obs: xarray Dataset containing observations
        ds_ml_interp: xarray Dataset containing interpolated ML predictions
        ds_nwp_interp: xarray Dataset containing interpolated NWP predictions
    """

    # Get lead times
    elapsed_forecast_durations = ds_ml_interp.elapsed_forecast_duration
    forecast_hours = [
        float(efd / np.timedelta64(1, "h"))
        for efd in elapsed_forecast_durations
    ]

    # Initialize lists to store RMSE values over time
    ml_rmse_over_time = []
    nwp_rmse_over_time = []

    for efd in elapsed_forecast_durations:
        # Get ML data for this forecast lead time
        ds_ml_lead = ds_ml_interp.sel(elapsed_forecast_duration=efd)

        # Get observation times corresponding to these forecast times
        forecast_times = ds_ml_lead.forecast_time.values
        ds_obs_lead = ds_obs.sel(time=forecast_times)

        # Get wind components for observations and ML
        u_true = ds_obs_lead["wind_u_10m"].values
        v_true = ds_obs_lead["wind_v_10m"].values
        u_ml = ds_ml_lead["wind_u_10m"].values
        v_ml = ds_ml_lead["wind_v_10m"].values

        # Create valid mask
        valid_mask = (
            ~np.isnan(u_true)
            & ~np.isnan(v_true)
            & ~np.isnan(u_ml)
            & ~np.isnan(v_ml)
        )

        if np.any(valid_mask):
            # Calculate ML wind vector RMSE
            wind_rmse_ml = wind_vector_rmse(
                u_ml[valid_mask],
                v_ml[valid_mask],
                u_true[valid_mask],
                v_true[valid_mask],
            )
            ml_rmse_over_time.append(wind_rmse_ml)
        else:
            ml_rmse_over_time.append(np.nan)

        # Calculate NWP RMSE if available
        if (
            ds_nwp_interp is not None
            and "wind_u_10m" in ds_nwp_interp
            and "wind_v_10m" in ds_nwp_interp
            and efd.values in ds_nwp_interp.elapsed_forecast_duration
        ):
            ds_nwp_lead = ds_nwp_interp.sel(elapsed_forecast_duration=efd)
            u_nwp = ds_nwp_lead["wind_u_10m"].values
            v_nwp = ds_nwp_lead["wind_v_10m"].values

            valid_mask_nwp = valid_mask & ~np.isnan(u_nwp) & ~np.isnan(v_nwp)

            if np.any(valid_mask_nwp):
                wind_rmse_nwp = wind_vector_rmse(
                    u_nwp[valid_mask_nwp],
                    v_nwp[valid_mask_nwp],
                    u_true[valid_mask_nwp],
                    v_true[valid_mask_nwp],
                )
                nwp_rmse_over_time.append(wind_rmse_nwp)
            else:
                nwp_rmse_over_time.append(np.nan)
        else:
            nwp_rmse_over_time.append(np.nan)

    # Create result DataFrame
    time_series_df = pd.DataFrame(
        {
            "Lead Time": forecast_hours,
            "ML Vector RMSE": ml_rmse_over_time,
            "NWP Vector RMSE": nwp_rmse_over_time,
        }
    )

    return time_series_df


# %%
def plot_wind_vector_rmse(time_series_df):
    """Plot wind vector RMSE evolution over forecast time.

    Args:
        time_series_df: Time series DataFrame from calculate_wind_vector_metrics
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    # Plot ML RMSE
    ax.plot(
        time_series_df["Lead Time"],
        time_series_df["ML Vector RMSE"],
        color=COLORS["ml"],
        linestyle=LINE_STYLES["ml"][0],
        marker=LINE_STYLES["ml"][1],
        label="ML",
    )

    # Plot NWP RMSE if available
    if not all(np.isnan(time_series_df["NWP Vector RMSE"])):
        ax.plot(
            time_series_df["Lead Time"],
            time_series_df["NWP Vector RMSE"],
            color=COLORS["nwp"],
            linestyle=LINE_STYLES["nwp"][0],
            marker=LINE_STYLES["nwp"][1],
            label="NWP",
        )

    ax.set_xlabel("Lead Time (hours)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
    ax.set_ylabel("Wind Vector RMSE (m/s)")
    ax.set_title("Wind Vector RMSE Evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
    save_plot(fig, "wind_vector_rmse_evolution")


# %%
# Calculate wind vector metrics
if "wind_u_10m" in ds_obs and "wind_v_10m" in ds_obs:
    print("Calculating wind vector metrics...")
    wind_timeseries = calculate_wind_vector_metrics(
        ds_obs, ds_ml_interp, ds_nwp_interp
    )
    plot_wind_vector_rmse(wind_timeseries)
    export_table(
        wind_timeseries,
        "wind_vector_metrics_timeseries",
        caption="Wind vector RMSE over forecast lead time",
    )
else:
    print("Wind components not found in the data")
