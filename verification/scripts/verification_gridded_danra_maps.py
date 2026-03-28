# %% [markdown]
# ##  Gridded Model Verification
#
# This script verifies output from a ML-based foundation model versus a
# traditional NWP system for the atmospheric system. The defaults set at
# the top of this script are tailored to the Alps-Clariden HPC system at CSCS.
# - The NWP-model is called DANRA-forecasts and is initialised with the
#   analysis. Only surface level data is available, and up to 18 h lead time.
# - The ML-model is called Neural-LAM and is initialised from the DANRA
#   reanalysis.
# - The Ground Truth is the same deterministic DANRA reanalysis as was used
#   to train the ML-model.
# - The boundary data for both models is IFS HRES from ECMWF, where the
#   NWP-model got 6 hourly boundary updates (?) and the ML model 12 hourly.

# Standard library
# %%
import random
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from debug_utils import (
    align_on_common_coord,
    assert_time_coverage,
    filter_forecast_dataset_by_time_coverage,
    parse_debug_runtime_args,
    resolve_requested_time,
    select_available_times,
    select_valid_positions,
    subset_dataset_for_debug,
)
from plot_styles import DPI, apply_style
from scores.continuous import mean_error

# %% [markdown]
# **--------> Enter all your user settings in the cell below. <--------**

# %%
# DEFAULTS
# This config will be applied to the data before any plotting. The data will be
# sliced and indexed according to the values in this config. The whole analysis
# plotting will be done on the reduced data.

# IF YOUR DATA HAS DIFFERENT DIMENSIONS OR NAMES, PLEASE ADJUST THE CELLS BELOW
# MAKE SURE THE XARRAY DATASETS LOOK OKAY BEFORE RUNNING CHAPTER 1-4

# This path should point to the data that was used to train the model
# (default is mdp-datastore)
PATH_GROUND_TRUTH = "danra_test_gt.zarr"
# This path should point to the NWP forecast data in zarr format
PATH_NWP = "danra_test_nwp_forecasts.zarr"
# This path should point to the ML forecast data in zarr format
# (e.g. produced by neural-lam in `eval` mode)
PATH_ML = "48h_eval_test_7deg_rect_hi4_2867.zarr"
# This path should point to the boundary data in zarr format
# (default is MDP-datastore)
# PATH_BOUNDARY = "ifs_test_boundary.zarr"
PATH_BOUNDARY = None

# lead time in steps for the forecast - [0] refers to the first forecast step
# at t+1
# this should be a list of integers
ELAPSED_FORECAST_DURATION = list(range(16))
ELAPSED_FORECAST_DURATION_SHORT = list(range(6))  # 18 h, matching NWP forecasts
ELAPSED_FORECAST_DURATION_PLOT = [1, 3, 5]  # ELS to plot
ELAPSED_FORECAST_DURATION_VERTICAL = [0, 3, 7, 11, 15]  # For vertical profiles

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

# Selection spatial grid in projection
# This is used to slice the data to a specific region
# This is in projection of the ground truth data
# The default is the whole domain [None, None]
X = [None, None]
Y = [None, None]

# Map projection settings for plotting
# This is the projection of the ground truth data
PROJECTION = ccrs.LambertConformal(
    central_longitude=25.0,
    central_latitude=56.7,
    standard_parallels=[56.7, 56.7],
    globe=ccrs.Globe(
        semimajor_axis=6367470.0,
        semiminor_axis=6367470.0,
    ),
)

# Define how variables map between different data sources

# Define here which of the variables are available in the ground truth data
# The keys are the names of the variables in the ground truth data
# The values are the conventional names, used in this notebook
VARIABLES_GROUND_TRUTH = {
    # Surface and near-surface variables
    "t2m": "temperature_2m",
    "u10m": "wind_u_10m",
    "v10m": "wind_v_10m",
    "pres_seasurface": "pressure_sea_level",
    "pres0m": "surface_pressure",
    "swavr0m": "surface_net_shortwave_radiation",
    "lwavr0m": "surface_net_longwave_radiation",
    # Upper air variables - U component
    # "u100": "wind_u_100hPa",
    "u200": "wind_u_200hPa",
    # "u400": "wind_u_400hPa",
    # "u600": "wind_u_600hPa",
    "u700": "wind_u_700hPa",
    # "u850": "wind_u_850hPa",
    # "u925": "wind_u_925hPa",
    # "u1000": "wind_u_1000hPa",
    # Upper air variables - V component
    # "v100": "wind_v_100hPa",
    "v200": "wind_v_200hPa",
    # "v400": "wind_v_400hPa",
    # "v600": "wind_v_600hPa",
    "v700": "wind_v_700hPa",
    # "v850": "wind_v_850hPa",
    # "v925": "wind_v_925hPa",
    # "v1000": "wind_v_1000hPa",
    # Upper air variables - Pressure
    # "z100": "geopotential_100hPa",
    # "z200": "geopotential_200hPa",
    # "z400": "geopotential_400hPa",
    # "z600": "geopotential_600hPa",
    "z700": "geopotential_700hPa",
    # "z850": "geopotential_850hPa",
    # "z925": "geopotential_925hPa",
    # "z1000": "geopotential_1000hPa",
    # Upper air variables - Temperature
    # "t100": "temperature_100hPa",
    # "t200": "temperature_200hPa",
    # "t400": "temperature_400hPa",
    # "t600": "temperature_600hPa",
    "t700": "temperature_700hPa",
    # "t850": "temperature_850hPa",
    # "t925": "temperature_925hPa",
    # "t1000": "temperature_1000hPa",
    # Upper air variables - Relative Humidity
    # "r100": "relative_humidity_100hPa",
    "r200": "relative_humidity_200hPa",
    # "r400": "relative_humidity_400hPa",
    # "r600": "relative_humidity_600hPa",
    "r700": "relative_humidity_700hPa",
    # "r850": "relative_humidity_850hPa",
    # "r925": "relative_humidity_925hPa",
    # "r1000": "relative_humidity_1000hPa",
    # Upper air variables - Vertical velocity
    # "tw100": "vertical_velocity_100hPa",
    "tw200": "vertical_velocity_200hPa",
    # "tw400": "vertical_velocity_400hPa",
    # "tw600": "vertical_velocity_600hPa",
    "tw700": "vertical_velocity_700hPa",
    # "tw850": "vertical_velocity_850hPa",
    # "tw925": "vertical_velocity_925hPa",
    # "tw1000": "vertical_velocity_1000hPa",
}

REQUIRED_LEVELS = [
    100,
    200,
    400,
    600,
    700,
    850,
    925,
    1000,
]

# Since the default ground_truth is the datastore that was used for model
# training the variables are identical to the VARIABLES_GROUND_TRUTH
VARIABLES_ML = VARIABLES_GROUND_TRUTH

# For the NWP-Forecast only a limited set of variables is available
# These variables are mapped to the same conventional names
# The script is flexible and will only calculate the NWP-metrics for the
# variables that are available
# The script will not break if some of the variables are not available
VARIABLES_NWP = {
    "t2m": "temperature_2m",
    "u10m": "wind_u_10m",
    "v10m": "wind_v_10m",
    "pres_seasurface": "seasurface_pressure",
}

# These variables are only used for chapter 1, the mapplots.
# They will be plotted for the ground truth, NWP and ML
VARIABLES_BOUNDARY = {
    # Surface and near-surface variables
    "mean_sea_level_pressure": "pressure_sea_level",
    "2m_temperature": "temperature_2m",
    "10m_u_component_of_wind": "wind_u_10m",
    "10m_v_component_of_wind": "wind_v_10m",
    "surface_pressure": "surface_pressure",
    # Upper air variables - U component
    "u_component_of_wind100": "wind_u_100hPa",
    "u_component_of_wind200": "wind_u_200hPa",
    "u_component_of_wind400": "wind_u_400hPa",
    "u_component_of_wind600": "wind_u_600hPa",
    "u_component_of_wind700": "wind_u_700hPa",
    "u_component_of_wind850": "wind_u_850hPa",
    "u_component_of_wind925": "wind_u_925hPa",
    "u_component_of_wind1000": "wind_u_1000hPa",
    # Upper air variables - V component
    "v_component_of_wind100": "wind_v_100hPa",
    "v_component_of_wind200": "wind_v_200hPa",
    "v_component_of_wind400": "wind_v_400hPa",
    "v_component_of_wind600": "wind_v_600hPa",
    "v_component_of_wind700": "wind_v_700hPa",
    "v_component_of_wind850": "wind_v_850hPa",
    "v_component_of_wind925": "wind_v_925hPa",
    "v_component_of_wind1000": "wind_v_1000hPa",
    # Upper air variables - Temperature
    "temperature100": "temperature_100hPa",
    "temperature200": "temperature_200hPa",
    "temperature400": "temperature_400hPa",
    "temperature600": "temperature_600hPa",
    "temperature700": "temperature_700hPa",
    "temperature850": "temperature_850hPa",
    "temperature925": "temperature_925hPa",
    "temperature1000": "temperature_1000hPa",
    # Upper air variables - Vertical velocity
    "vertical_velocity100": "vertical_velocity_100hPa",
    "vertical_velocity200": "vertical_velocity_200hPa",
    "vertical_velocity400": "vertical_velocity_400hPa",
    "vertical_velocity600": "vertical_velocity_600hPa",
    "vertical_velocity700": "vertical_velocity_700hPa",
    "vertical_velocity850": "vertical_velocity_850hPa",
    "vertical_velocity925": "vertical_velocity_925hPa",
    "vertical_velocity1000": "vertical_velocity_1000hPa",
    # Upper air variables - Geopotential
    "geopotential100": "geopotential_100hPa",
    "geopotential200": "geopotential_200hPa",
    "geopotential400": "geopotential_400hPa",
    "geopotential600": "geopotential_600hPa",
    "geopotential700": "geopotential_700hPa",
    "geopotential850": "geopotential_850hPa",
    "geopotential925": "geopotential_925hPa",
    "geopotential1000": "geopotential_1000hPa",
}


# These variables will be used as `basename` for the vertical profiles.
# Since the input of the zarr archives is expected to have data vars that
# are 2D in space
# we need some base_name prefix to create the 3D variables
VARIABLES_3D = [
    "wind_u",
    "wind_v",
    "geopotential",
    "temperature",
    "relative_humidity",
    "vertical_velocity",
]

# Add units dictionary after the imports
# units from zarr archives are not reliable and should rather be defined here
VARIABLE_UNITS = {
    # Surface and near-surface variables
    "temperature_2m": "K",
    "wind_u_10m": "m/s",
    "wind_v_10m": "m/s",
    "pressure_sea_level": "Pa",
    "surface_pressure": "Pa",
    "precipitation": "mm/h",
    # "surface_sensible_heat_flux": "W/m²",
    "surface_net_shortwave_radiation": "W/m²",
    "surface_net_longwave_radiation": "W/m²",
    # Upper air variables
    "wind_u_level": "m/s",
    "wind_v_level": "m/s",
    "geopotential_level": "m²/s²",
    "temperature_level": "K",
    "relative_humidity_level": "%",
    "vertical_velocity_level": "Pa/s",
}

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
    "RelativeMAE",
    # "RelativeRMSE",
    # "PearsonR",
    # "Wasserstein",
]

# This setting is relevant for the mapplots in chapter 1
# Higher levels of ZOOM will zoom in on the map, cropping the boundary
BORDER_WIDTH = 300000  # in m
ZOOM = 1  # Unused

# For some chapters a random seed is required to reproduce the results
RANDOM_SEED = 42


# Subsample the data for faster plotting, 0.1 refers to 10% of the ml/nwp data
# sampled along the start_time, x and y dimensions. If you calculate the FSS
# metrics you would want to limit subsampling to the time-dimensions! There is a
# trade-off between speed and accuracy, that each user has to find.
SUBSAMPLE_FRACTION = 1.0

# If the data should be loaded into memory. Makes following calculations faster
# but requires enough memory to hold the data.
PRECOMPUTE_DATA = False

# Subsample the data for FSS threshold calculation, 1e7 refers to the
# number of elements
# This is not critical, as it is only used to calculate the 90% threshold
# for the FSS based on the ground truth data
SUBSAMPLE_FSS_THRESHOLD = 1e7

# Takes a long time, but if you see NaN in your output, you can set this to True
# This will check if there are any missing values in the data further below
# THIS NOTEBOOK WILL ONLY WORK RELIABLY IF THERE ARE NO MISSING VALUES
# If there are missing values, you have to interpolate them or remove them
CHECK_MISSING = False


DEBUG_MODE, DEBUG_FRACTION, DEBUG_MIN_SIZE = parse_debug_runtime_args(
    "Run gridded DANRA map verification.",
)

# %%
# Create directories for plots and tables
Path("verification/danra/gridded").mkdir(parents=True, exist_ok=True)
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
apply_style()

# Colorblind-friendly colormap for 2D plots
COLORMAP = "viridis"


def get_colormap_for_variable(variable_name: str) -> str:
    """Return an appropriate colormap for a given physical variable.

    Uses substring matching on the variable name (case-insensitive) to select
    a colormap.  Mirrors the SwissClim convention where possible.
    """
    v = variable_name.lower()

    # Diverging / signed wind components & vertical velocity
    if any(k in v for k in ("wind_u", "wind_v", "vertical_velocity")):
        return "RdBu_r"
    # Precipitation / moisture
    if "precipitation" in v or "tot_prec" in v:
        return "Blues"
    # Temperature / radiation
    if any(k in v for k in ("temperature", "radiation", "heat_flux")):
        return "magma"
    # Humidity / vegetation
    if "humidity" in v:
        return "Greens"
    # Wind speed (scalar)
    if "wind_speed" in v:
        return "YlOrRd"
    # Pressure / geopotential (keep viridis)
    if any(k in v for k in ("pressure", "geopotential")):
        return "viridis"
    return "viridis"


# First, collect all the base variables and units we need to extend
base_level_vars = {}
for base_var, unit in VARIABLE_UNITS.items():
    if "_level" in base_var:
        base_level_vars[base_var] = unit

# Then create the level-specific entries
for level in REQUIRED_LEVELS:
    for base_var, unit in base_level_vars.items():
        VARIABLE_UNITS[f"{base_var[:-len('_level')]}_{level}hPa"] = unit
print(f"All units: {VARIABLE_UNITS}")


def save_plot(fig, name, time=None, remove_title=True, dpi=300):
    """Save figure as PDF and accompanying .npz with all plotted data.

    Args:
        fig: matplotlib figure object
        name (str): base name for the plot file
        time (datetime, optional): timestamp to append to filename
        remove_title (bool): remove suptitle/title hierarchically if True
        dpi (int): resolution for the saved figure, defaults to 300
    """
    is_case_study = time is not None
    if time is not None:
        name = f"{name}_{time.dt.strftime('%Y%m%d_%H').values}"

    # Sanitize filename by replacing problematic characters
    safe_name = name.replace("/", "_per_")

    # Time-stamped plots are case-study snapshots; aggregate plots go to gridded
    if is_case_study:
        plot_dir = Path("verification/danra/case_study")
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

    pdf_path = plot_dir / f"{safe_name}.pdf"
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
        # Also capture images (pcolormesh / imshow) as 2-D arrays
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
    np.savez_compressed(plot_dir / f"{safe_name}.npz", **npz_payload)


def export_table(df, name, caption=""):
    """Helper function to export tables consistently"""
    # Export to LaTeX with caption
    latex_str = df.to_latex(
        float_format="%.4f", caption=caption, label=f"tab:{name}"
    )
    with open(f"verification/danra/gridded/{name}.tex", "w") as f:
        f.write(latex_str)

    # Export to CSV
    df.to_csv(f"verification/danra/gridded/{name}.csv")


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
ds_ml = ds_ml.sel(y=slice(*Y), x=slice(*X))
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
ds_ml = select_valid_positions(
    ds_ml,
    dim="elapsed_forecast_duration",
    positions=ELAPSED_FORECAST_DURATION,
    label="ML",
)

ds_ml

# %%
ds_gt = xr.open_zarr(PATH_GROUND_TRUTH, decode_timedelta=True)
ds_gt = ds_gt.set_index(grid_index=["y", "x"]).unstack("grid_index")
ds_gt = subset_dataset_for_debug(
    ds_gt,
    label="ground-truth grid",
    dims_to_trim=("x", "y"),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
)
ds_gt = ds_gt.sel(y=slice(*Y), x=slice(*X))
ds_gt = ds_gt.sel(state_feature=list(VARIABLES_ML.keys()))
ds_gt = ds_gt.sel(split_name="test").drop_dims(
    [
        "forcing_feature",
        "static_feature",
        "split_part",
    ]
)
for feature in ds_gt.state_feature.values:
    ds_gt[VARIABLES_ML[feature]] = ds_gt["state"].sel(state_feature=feature)
ds_gt = ds_gt.drop_vars(
    [
        "state",
        "state_feature",
        "state_feature_units",
        "state_feature_long_name",
        "state_feature_source_dataset",
        "state__train__diff_mean",
        "state__train__diff_std",
        "state__train__mean",
        "state__train__std",
    ]
)
ds_gt = ds_gt.transpose("time", "x", "y")
ds_gt = ds_gt[
    [
        "time",
        "x",
        "y",
        *VARIABLES_GROUND_TRUTH.values(),
    ]
]
time_subset = np.concatenate(
    (ds_ml.forecast_time.values.flatten(), ds_ml.start_time.values.flatten())
)
ds_gt = select_available_times(ds_gt, time_subset, label="ground-truth")
ds_ml = filter_forecast_dataset_by_time_coverage(ds_ml, ds_gt, label="ML")
time_subset = np.concatenate(
    (ds_ml.forecast_time.values.flatten(), ds_ml.start_time.values.flatten())
)
ds_gt = select_available_times(ds_gt, time_subset, label="ground-truth aligned")
ds_gt


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
ds_nwp = ds_nwp.sel(y=slice(*Y), x=slice(*X), analysis_time=slice(*START_TIMES))
ds_nwp = ds_nwp[VARIABLES_NWP.keys()].rename(VARIABLES_NWP)
ds_nwp = ds_nwp.rename_dims(
    {
        "analysis_time": "start_time",
    }
)
ds_nwp = ds_nwp.rename_vars(
    {
        "analysis_time": "start_time",
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

# Do not need these, remove so no issues later
ds_nwp = ds_nwp.drop(["time"])

# The NWP data starts at lead time 0 = start_time
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

ds_ml, ds_nwp = align_on_common_coord(
    ds_ml,
    ds_nwp,
    coord_name="start_time",
    left_label="ML",
    right_label="NWP",
)
time_subset = np.concatenate(
    (ds_ml.forecast_time.values.flatten(), ds_ml.start_time.values.flatten())
)
ds_gt = select_available_times(
    ds_gt, time_subset, label="ground-truth synchronized"
)

ds_nwp

# %% [markdown]
# Check for missing data in any of the variables. If you have missing
# data, you need to handle it before running the verification.

# %%
if CHECK_MISSING:
    missing_counts = dask.compute(
        {var: ds_gt[var].isnull().sum().values for var in ds_gt.data_vars},
        {var: ds_nwp[var].isnull().sum().values for var in ds_nwp.data_vars},
        {var: ds_ml[var].isnull().sum().values for var in ds_ml.data_vars},
    )
    # Unpack results
    gt_missing, nwp_missing, ml_missing = missing_counts

    # Print results
    print("Ground Truth")
    for var, count in gt_missing.items():
        print(f"{var}: {count} missing values")

    print("\nNWP Model")
    for var, count in nwp_missing.items():
        print(f"{var}: {count} missing values")

    print("\nML Model")
    for var, count in ml_missing.items():
        print(f"{var}: {count} missing values")


# %%
assert ds_gt.sizes["x"] == ds_ml.sizes["x"]
assert ds_gt.sizes["x"] == ds_nwp.sizes["x"]
assert ds_gt.sizes["y"] == ds_ml.sizes["y"]
assert ds_gt.sizes["y"] == ds_nwp.sizes["y"]
assert_time_coverage(
    ds_gt,
    ds_ml.forecast_time.values.flatten(),
    label="Ground truth",
)
# Since nwp has less lead times below might not hold
# assert ds_gt.sizes["time"] == len(
#    np.unique(ds_nwp.forecast_time.values.flatten())
# )

# Generate random indices for each dimension
rng = np.random.RandomState(RANDOM_SEED)
sampled_start_time_indices = np.sort(
    rng.choice(
        len(ds_ml.start_time),
        size=int(len(ds_ml.start_time) * SUBSAMPLE_FRACTION),
        replace=False,
    )
)

# with (
#     LocalCluster(
#         n_workers=4,
#         threads_per_worker=16,
#         memory_limit="96GB",
#         local_directory="/iopsstor/scratch/cscs/sadamov",  # noqa: E501
#         # Use fast local storage for spilling
#         dashboard_address=None,
#     ) as cluster
# ):
#     with Client(cluster) as client:
ds_ml_sampled = ds_ml.isel(start_time=sampled_start_time_indices)
ds_nwp_sampled = ds_nwp.isel(start_time=sampled_start_time_indices)
ds_gt_sampled = ds_gt.sel(time=ds_ml_sampled.forecast_time)
if PRECOMPUTE_DATA:
    with ProgressBar():
        print("Computing ML data")
        ds_ml_sampled = ds_ml_sampled.compute()
        print("Computing NWP data")
        ds_nwp_sampled = ds_nwp_sampled.compute()
        print("Computing GT data")
        ds_gt_sampled = ds_gt_sampled.compute()


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
# Get coordinates
if hasattr(ds_gt, "longitude") and hasattr(ds_gt, "latitude"):
    lons = ds_gt.longitude.values
    lats = ds_gt.latitude.values
elif hasattr(ds_gt, "lon") and hasattr(ds_gt, "lat"):
    lons = ds_gt.lon.values
    lats = ds_gt.lat.values

lon_min = lons.min()
lon_max = lons.max()
lat_min = lats.min()
lat_max = lats.max()

# Transform domain bounds to rotated coordinates
transformer = PROJECTION.transform_points(
    ccrs.PlateCarree(),
    np.array([lon_min, lon_max]),
    np.array([lat_min, lat_max]),
)

# Get rotated coordinate bounds
rot_lon_min, rot_lon_max = transformer[:, 0].min(), transformer[:, 0].max()
rot_lat_min, rot_lat_max = transformer[:, 1].min(), transformer[:, 1].max()

# %% [markdown]
# Load in boundary

# %%
ds_boundary = None
if PATH_BOUNDARY:
    ds_boundary = xr.open_zarr(PATH_BOUNDARY, decode_timedelta=True)
    ds_boundary = subset_dataset_for_debug(
        ds_boundary,
        label="boundary zarr",
        dims_to_trim=(
            "time",
            "analysis_time",
            "elapsed_forecast_duration",
            "latitude",
            "longitude",
        ),
        debug_mode=DEBUG_MODE,
        debug_fraction=DEBUG_FRACTION,
        debug_min_size=DEBUG_MIN_SIZE,
    )

    temporal_dim = "time" if "time" in ds_boundary.dims else "analysis_time"
    forecast_duration_dim = (
        "elapsed_forecast_duration"
        if "elapsed_forecast_duration" in ds_boundary.dims
        else None
    )
    dims_to_transpose = [
        dim
        for dim in [
            temporal_dim,
            forecast_duration_dim,
            "latitude",
            "longitude",
        ]
        if dim is not None
    ]

    ds_boundary = ds_boundary.sel(
        forcing_feature=list(VARIABLES_BOUNDARY.keys())
    )
    ds_boundary = ds_boundary.sel(split_name="test").drop_dims(
        [
            "split_part",
            "static_feature",
        ]
    )
    for feature in ds_boundary.forcing_feature.values:
        ds_boundary[VARIABLES_BOUNDARY[feature]] = ds_boundary["forcing"].sel(
            forcing_feature=feature
        )
    ds_boundary = ds_boundary.drop_vars(
        [
            "forcing",
            "forcing_feature",
            "forcing_feature_units",
            "forcing_feature_long_name",
            "forcing_feature_source_dataset",
            "forcing__train__diff_mean",
            "forcing__train__diff_std",
            "forcing__train__mean",
            "forcing__train__std",
        ]
    )
    ds_boundary = ds_boundary.set_index(grid_index=["latitude", "longitude"])
    ds_boundary = ds_boundary.unstack("grid_index")
    ds_boundary = ds_boundary.transpose(*dims_to_transpose)
    longitude_new = np.where(
        ds_boundary["longitude"] > 180,
        ds_boundary["longitude"] - 360,
        ds_boundary["longitude"],
    )
    ds_boundary = ds_boundary.assign_coords(longitude=longitude_new).sortby(
        [
            "longitude",
            "latitude",
        ]
    )

    lon_mesh, lat_mesh = np.meshgrid(
        ds_boundary.longitude, ds_boundary.latitude
    )

ds_boundary


# %%
def create_comparison_maps(
    ds_gt,
    ds_ml,
    ds_nwp,
    ds_boundary=None,
    var=None,
    plot_time=None,
    step_size_gt=pd.Timedelta("1h"),
    random_seed=42,
    zoom_factor=None,
):
    # Handle variable selection
    variables = [var] if var else VARIABLES_GROUND_TRUTH.values()

    # Select time
    if plot_time is None:
        random.seed(random_seed)
        time_index = random.randint(0, len(ds_gt.time) - 1)
        time_selected = ds_ml.time[time_index].values
    else:
        time_selected = plot_time

    # Get number of forecast dimensions
    n_elapsed_forecast_durations = len(ds_ml.elapsed_forecast_duration)

    # Pre-select start_time once to avoid repeated sel in every loop iteration
    ds_ml_start = ds_ml.sel(start_time=time_selected)
    if ds_nwp is not None:
        ds_nwp_start = ds_nwp.sel(start_time=time_selected)

    for var in variables:
        # Determine number of columns based on NWP data availability
        n_cols = 3 if (ds_nwp is not None and var in ds_nwp) else 2

        # Create figure with n_elapsed_forecast_durations rows and
        # n_cols columns
        fig = plt.figure(
            figsize=(9 * n_cols, 6 * n_elapsed_forecast_durations), dpi=DPI
        )
        axes = np.array(
            [
                [
                    plt.subplot(
                        n_elapsed_forecast_durations,
                        n_cols,
                        i * n_cols + j + 1,
                        projection=PROJECTION,
                    )
                    for j in range(n_cols)
                ]
                for i in range(n_elapsed_forecast_durations)
            ]
        )

        # Initialize arrays for global min/max
        arrays_for_minmax = []

        # First pass: collect all data for global min/max calculation
        for elapsed_forecast_dimension in ds_ml.elapsed_forecast_duration:
            # Select data for current forecast time
            ds_ml_time = ds_ml_start.sel(
                elapsed_forecast_duration=elapsed_forecast_dimension,
            )
            ds_gt_time = ds_gt.sel(time=ds_ml_time.forecast_time)

            # Add ground truth and ML prediction to min/max arrays
            arrays_for_minmax.extend(
                [
                    ds_gt_time[var].values,
                    ds_ml_time[var].values,
                ]
            )

            # Add NWP data if available
            if ds_nwp is not None and var in ds_nwp:
                ds_nwp_time = ds_nwp_start.sel(
                    elapsed_forecast_duration=elapsed_forecast_dimension,
                )
                arrays_for_minmax.append(ds_nwp_time[var].values)

            # Add boundary data if available
            if ds_boundary is not None and var in ds_boundary:
                if "elapsed_forecast_duration" in ds_boundary:
                    # Determine number of steps based on time alignment
                    steps = (
                        2
                        if (
                            ds_boundary.sel(
                                analysis_time=time_selected - step_size_gt,
                                method="pad",
                            ).analysis_time.values
                            == time_selected - step_size_gt
                        )
                        else 1
                    )

                    # Get boundary data for current time
                    ds_boundary_var = ds_boundary[var].sel(
                        analysis_time=time_selected - steps * step_size_gt,
                        method="pad",
                    )
                    # Update forecast times
                    forecast_times = (
                        ds_boundary_var.analysis_time.values
                        + ds_boundary_var.elapsed_forecast_duration.values
                    )
                    ds_boundary_var[
                        "elapsed_forecast_duration"
                    ] = forecast_times
                    ds_boundary_var = ds_boundary_var.sel(
                        elapsed_forecast_duration=ds_ml_time.forecast_time,
                        method="pad",
                    )
                else:
                    ds_boundary_var = ds_boundary[var].sel(
                        time=ds_ml_time.forecast_time, method="pad"
                    )
                arrays_for_minmax.append(ds_boundary_var.values)

        # Calculate global min/max without allocating a combined array
        vmin = np.nanmin([np.nanmin(arr) for arr in arrays_for_minmax])
        vmax = np.nanmax([np.nanmax(arr) for arr in arrays_for_minmax])

        # Choose variable-specific colormap
        var_cmap = get_colormap_for_variable(var)

        # Use log scale for precipitation (avoid zero/negative in LogNorm)
        var_norm = None
        if "precipitation" in var.lower():
            log_vmin = max(vmin, 0.01)  # lower bound for log scale
            log_vmax = max(vmax, log_vmin * 10)
            var_norm = mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)

        # Second pass: create plots
        for dim_idx, elapsed_forecast_dimension in enumerate(
            ds_ml.elapsed_forecast_duration
        ):
            # Select data for current forecast time
            ds_ml_time = ds_ml_start.sel(
                elapsed_forecast_duration=elapsed_forecast_dimension,
            )
            ds_gt_time = ds_gt.sel(time=ds_ml_time.forecast_time)

            # Calculate forecast hours for titles
            forecast_hours = int(elapsed_forecast_dimension.values / 1e9 / 3600)

            # Plot boundary conditions if available
            if ds_boundary is not None and var in ds_boundary:
                if "elapsed_forecast_duration" in ds_boundary:
                    steps = (
                        2
                        if (
                            ds_boundary.sel(
                                analysis_time=time_selected - step_size_gt,
                                method="pad",
                            ).analysis_time.values
                            == time_selected - step_size_gt
                        )
                        else 1
                    )

                    ds_boundary_var = ds_boundary[var].sel(
                        analysis_time=time_selected - steps * step_size_gt,
                        method="pad",
                    )
                    forecast_times = (
                        ds_boundary_var.analysis_time.values
                        + ds_boundary_var.elapsed_forecast_duration.values
                    )
                    ds_boundary_var[
                        "elapsed_forecast_duration"
                    ] = forecast_times
                    ds_boundary_var = ds_boundary_var.sel(
                        elapsed_forecast_duration=ds_ml_time.forecast_time,
                        method="pad",
                    )
                else:
                    ds_boundary_var = ds_boundary[var].sel(
                        time=ds_ml_time.forecast_time, method="pad"
                    )

                for ax in axes[dim_idx]:
                    ax.contourf(
                        lon_mesh,
                        lat_mesh,
                        ds_boundary_var.values,
                        transform=ccrs.PlateCarree(),
                        cmap=var_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        alpha=0.5,
                        levels=20,
                    )
                    if zoom_factor is not None:
                        set_map_extent(ax, zoom_factor, ds_boundary_var)

            # Plot ground truth
            im0 = plot_field(
                axes[dim_idx, 0],
                ds_gt_time[var],
                vmin,
                vmax,
                cmap=var_cmap,
                norm=var_norm,
            )

            # Set titles
            if dim_idx == 0:
                axes[dim_idx, 0].set_title(f"Ground Truth\n+{forecast_hours}h")
                if ds_nwp is not None and var in ds_nwp:
                    axes[dim_idx, 1].set_title(f"NWP\n+{forecast_hours}h")
                    axes[dim_idx, 2].set_title(f"ML\n+{forecast_hours}h")
                else:
                    axes[dim_idx, 1].set_title(f"ML\n+{forecast_hours}h")
            else:
                axes[dim_idx, 0].set_title(f"+{forecast_hours}h")
                if ds_nwp is not None and var in ds_nwp:
                    axes[dim_idx, 1].set_title(f"+{forecast_hours}h")
                    axes[dim_idx, 2].set_title(f"+{forecast_hours}h")
                else:
                    axes[dim_idx, 1].set_title(f"+{forecast_hours}h")

            # Plot NWP and ML predictions
            col = 1
            if ds_nwp is not None and var in ds_nwp:
                ds_nwp_time = ds_nwp_start.sel(
                    elapsed_forecast_duration=elapsed_forecast_dimension,
                )
                plot_field(
                    axes[dim_idx, col],
                    ds_nwp_time[var],
                    vmin,
                    vmax,
                    cmap=var_cmap,
                    norm=var_norm,
                )
                col += 1

            plot_field(
                axes[dim_idx, col],
                ds_ml_time[var],
                vmin,
                vmax,
                cmap=var_cmap,
                norm=var_norm,
            )

        # Add common features and colorbar
        add_map_features(axes)
        add_colorbar(fig, im0, var)

        # Adjust layout and add title
        plt.subplots_adjust(
            top=0.92,
            bottom=0.05,
            hspace=0.050,
            wspace=0.05,
        )
        title = (  # noqa: E501
            f"{var} starting at {str(time_selected.dt.date.values)}"  # noqa: E501
            f" - {time_selected.dt.hour.values:02d} UTC"
        )
        plt.suptitle(title, y=0.98)

        # Show and save plot
        plt.show()
        save_plot(fig, f"map_{var}_multi_efd", time_selected, dpi=DPI)
        plt.close()


def old_set_map_extent(ax, zoom_factor=None, boundary_data=None):
    """Set the map extent based on zoom factor and boundary data."""
    if zoom_factor is not None and boundary_data is not None:
        # Get the boundary extent
        lon = (
            boundary_data.longitude
            if hasattr(boundary_data, "longitude")
            else boundary_data.lon
        )
        lat = (
            boundary_data.latitude
            if hasattr(boundary_data, "latitude")
            else boundary_data.lat
        )

        # Calculate center
        lon_center = (lon.max() + lon.min()) / 2
        lat_center = (lat.max() + lat.min()) / 2

        # Calculate ranges
        lon_range = (lon.max() - lon.min()) / zoom_factor
        lat_range = (lat.max() - lat.min()) / zoom_factor

        # Set new extent
        ax.set_extent(
            [
                lon_center - lon_range / 2,
                lon_center + lon_range / 2,
                lat_center - lat_range / 2,
                lat_center + lat_range / 2,
            ],
            crs=ccrs.PlateCarree(),
        )


def set_map_extent(ax, zoom_factor=None, boundary_data=None):
    """Set the map extent based on zoom factor and boundary data."""
    # Set new extent
    ax.set_extent(
        [
            ds_ml.x.min().values - BORDER_WIDTH,
            ds_ml.x.max().values + BORDER_WIDTH,
            ds_ml.y.min().values - BORDER_WIDTH,
            ds_ml.y.max().values + BORDER_WIDTH,
        ],
        crs=PROJECTION,
    )


def plot_field(ax, data, vmin, vmax, cmap="viridis", norm=None):
    kwargs = dict(
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="auto",
        rasterized=True,
    )
    if norm is not None:
        kwargs["norm"] = norm
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    return ax.pcolormesh(
        data.longitude if hasattr(data, "longitude") else data.lon,
        data.latitude if hasattr(data, "latitude") else data.lat,
        data.values,
        **kwargs,
    )


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


def add_colorbar(fig, im, var):
    cbar_ax = fig.add_axes([0.2, 0.0, 0.6, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(VARIABLE_UNITS[var])


# %%
step_size_gt = pd.Timedelta(ds_gt.time.diff("time").min().values, "h")
ds_ml = ds_ml.assign_coords(
    {
        "lon": (("x", "y"), ds_nwp.lon.values),
        "lat": (("x", "y"), ds_nwp.lat.values),
    }
)
if PLOT_TIME is None:
    time_selected = None
else:
    time_selected = resolve_requested_time(
        ds_ml,
        dim="start_time",
        requested_time=PLOT_TIME,
        label="plot",
        allow_nearest=True,
    )
create_comparison_maps(
    ds_gt=ds_gt,
    ds_ml=ds_ml.isel(elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT),
    ds_nwp=ds_nwp.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT
    ),
    ds_boundary=ds_boundary,
    plot_time=time_selected,
    step_size_gt=step_size_gt,
    zoom_factor=ZOOM,
)


# %% [markdown]
# #### Mean Error Plot Across All Start_Times

# %%
def create_error_maps(ds_gt, ds_ml, ds_nwp=None, var=None):
    """Create average error maps for model outputs organized in subfigures."""
    variables = [var] if var else VARIABLES_GROUND_TRUTH.values()

    n_elapsed_forecast_durations = len(ds_ml.elapsed_forecast_duration)

    for var in variables:
        n_cols = 2 if (ds_nwp is not None and var in ds_nwp) else 1

        fig = plt.figure(
            figsize=(7.5 * n_cols, 6 * n_elapsed_forecast_durations), dpi=DPI
        )

        axes = np.array(
            [
                [
                    plt.subplot(
                        n_elapsed_forecast_durations,
                        n_cols,
                        i * n_cols + j + 1,
                        projection=PROJECTION,
                    )
                    for j in range(n_cols)
                ]
                for i in range(n_elapsed_forecast_durations)
            ]
        )

        arrays_for_minmax = []
        for dim_idx, elapsed_forecast_dimension in enumerate(
            ds_ml.elapsed_forecast_duration
        ):
            forecast_hours = int(elapsed_forecast_dimension.values / 1e9 / 3600)

            ds_ml_time = ds_ml.isel(
                start_time=0, elapsed_forecast_duration=dim_idx
            )
            lons = (
                ds_ml_time.longitude
                if hasattr(ds_ml_time, "longitude")
                else ds_ml_time.lon
            )
            lats = (
                ds_ml_time.latitude
                if hasattr(ds_ml_time, "latitude")
                else ds_ml_time.lat
            )

            # Calculate errors using mean_error function
            obs_times = ds_ml.forecast_time.sel(
                elapsed_forecast_duration=elapsed_forecast_dimension
            )

            errors = {}
            errors["ml"] = mean_error(
                ds_ml[var].sel(
                    elapsed_forecast_duration=elapsed_forecast_dimension
                ),
                ds_gt[var].sel(time=obs_times),
                preserve_dims=["x", "y"],
            )
            arrays_for_minmax.append(errors["ml"])

            if ds_nwp is not None and var in ds_nwp:
                errors["nwp"] = mean_error(
                    ds_nwp[var].sel(
                        elapsed_forecast_duration=elapsed_forecast_dimension
                    ),
                    ds_gt[var].sel(time=obs_times),
                    preserve_dims=["x", "y"],
                )
                arrays_for_minmax.append(errors["nwp"])

            # Calculate symmetric bounds
            abs_max = max(abs(np.nanmax(arr)) for arr in arrays_for_minmax)
            abs_max = max(
                abs_max, abs(min(np.nanmin(arr) for arr in arrays_for_minmax))
            )
            vmin, vmax = -abs_max, abs_max

            if ds_nwp is not None and var in ds_nwp:
                plot_error_field(
                    axes[dim_idx, 0], lons, lats, errors["nwp"], vmin, vmax
                )
                axes[dim_idx, 0].set_title(
                    f"Mean NWP Error\n+{forecast_hours}h"
                    if dim_idx == 0
                    else f"+{forecast_hours}h"
                )

            col_idx = 1 if ds_nwp is not None and var in ds_nwp else 0
            im = plot_error_field(
                axes[dim_idx, col_idx], lons, lats, errors["ml"], vmin, vmax
            )
            axes[dim_idx, col_idx].set_title(
                f"Mean ML Error\n+{forecast_hours}h"
                if dim_idx == 0
                else f"+{forecast_hours}h"
            )

        add_map_features(axes)
        add_error_colorbar(fig, im, var)

        plt.subplots_adjust(
            top=0.9,
            bottom=0.05,
            hspace=0.030,
            wspace=0.05,
        )

        title = f"Mean Error in {var}"
        plt.suptitle(title, y=0.98)

        plt.show()
        save_plot(fig, f"mean_errormap_{var}_multi_efd", dpi=DPI)
        plt.close()


def plot_error_field(ax, lons, lats, data, vmin, vmax):
    return ax.pcolormesh(
        lons,
        lats,
        data,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )


def add_error_colorbar(fig, im, var):
    cbar_ax = fig.add_axes([0.2, 0.0, 0.6, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"Error in {VARIABLE_UNITS[var]}")


# %%
if "lon" not in ds_ml.coords:
    ds_ml = ds_ml.assign_coords(
        {
            "lon": (("x", "y"), ds_nwp.lon.values),
            "lat": (("x", "y"), ds_nwp.lat.values),
        }
    )
if PLOT_TIME is None:
    time_selected = None
else:
    time_selected = resolve_requested_time(
        ds_ml,
        dim="start_time",
        requested_time=PLOT_TIME,
        label="plot",
        allow_nearest=True,
    )
create_error_maps(
    ds_gt=ds_gt,
    ds_ml=ds_ml.isel(elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT),
    ds_nwp=ds_nwp.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT
    ),
)
