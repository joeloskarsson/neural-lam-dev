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
import argparse
import os
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import dask
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

# %%
from plot_styles import DPI, SCALE_METRICS, apply_style
from scipy.stats import wasserstein_distance
from scores.categorical import ThresholdEventOperator as TEO
from scores.continuous import mae, mean_error, mse, rmse
from scores.continuous.correlation import pearsonr
from scores.spatial import fss_2d

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

# X and Y are set dynamically from --denmark CLI flag (see _parse_runtime_args)

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
    # "pres_seasurface": "pressure_sea_level",
    # "pres0m": "surface_pressure",
    # "swavr0m": "surface_net_shortwave_radiation",
    # "lwavr0m": "surface_net_longwave_radiation",
    # Upper air variables - U component
    # "u100": "wind_u_100hPa",
    # "u200": "wind_u_200hPa",
    # "u400": "wind_u_400hPa",
    # "u600": "wind_u_600hPa",
    # "u700": "wind_u_700hPa",
    # "u850": "wind_u_850hPa",
    # "u925": "wind_u_925hPa",
    # "u1000": "wind_u_1000hPa",
    # Upper air variables - V component
    # "v100": "wind_v_100hPa",
    # "v200": "wind_v_200hPa",
    # "v400": "wind_v_400hPa",
    # "v600": "wind_v_600hPa",
    # "v700": "wind_v_700hPa",
    # "v850": "wind_v_850hPa",
    # "v925": "wind_v_925hPa",
    # "v1000": "wind_v_1000hPa",
    # Upper air variables - Pressure
    # "z100": "geopotential_100hPa",
    # "z200": "geopotential_200hPa",
    # "z400": "geopotential_400hPa",
    # "z600": "geopotential_600hPa",
    # "z700": "geopotential_700hPa",
    # "z850": "geopotential_850hPa",
    # "z925": "geopotential_925hPa",
    # "z1000": "geopotential_1000hPa",
    # Upper air variables - Temperature
    # "t100": "temperature_100hPa",
    # "t200": "temperature_200hPa",
    # "t400": "temperature_400hPa",
    # "t600": "temperature_600hPa",
    # "t700": "temperature_700hPa",
    # "t850": "temperature_850hPa",
    # "t925": "temperature_925hPa",
    # "t1000": "temperature_1000hPa",
    # Upper air variables - Relative Humidity
    # "r100": "relative_humidity_100hPa",
    # "r200": "relative_humidity_200hPa",
    # "r400": "relative_humidity_400hPa",
    # "r600": "relative_humidity_600hPa",
    # "r700": "relative_humidity_700hPa",
    # "r850": "relative_humidity_850hPa",
    # "r925": "relative_humidity_925hPa",
    # "r1000": "relative_humidity_1000hPa",
    # Upper air variables - Vertical velocity
    # "tw100": "vertical_velocity_100hPa",
    # "tw200": "vertical_velocity_200hPa",
    # "tw400": "vertical_velocity_400hPa",
    # "tw600": "vertical_velocity_600hPa",
    # "tw700": "vertical_velocity_700hPa",
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
    # "RelativeMAE",
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


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_runtime_args():
    parser = argparse.ArgumentParser(
        description="Run gridded DANRA verification metrics.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on a small deterministic subset of each opened zarr dataset.",
    )
    parser.add_argument(
        "--debug-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of each supported dimension to keep in debug mode"
            " (0, 1]."
        ),
    )
    parser.add_argument(
        "--debug-min-size",
        type=int,
        default=None,
        help=(
            "Minimum number of points to keep per trimmed dimension in debug"
            " mode."
        ),
    )
    parser.add_argument(
        "--denmark",
        action="store_true",
        help=(
            "Crop to the Denmark sub-region (x: -1080000 to -600000,"
            " y: -160000 to 220000) and write outputs to"
            " verification/danra/gridded_denmark/."
        ),
    )
    args, _ = parser.parse_known_args()

    debug_mode = args.debug or _env_flag("VERIFICATION_DEBUG", default=False)
    debug_fraction = args.debug_fraction
    if debug_fraction is None:
        debug_fraction = float(
            os.environ.get("VERIFICATION_DEBUG_FRACTION", "0.02")
        )
    debug_min_size = args.debug_min_size
    if debug_min_size is None:
        debug_min_size = int(os.environ.get("VERIFICATION_DEBUG_MIN_SIZE", "2"))

    if not 0 < debug_fraction <= 1:
        raise ValueError("--debug-fraction must be in the interval (0, 1].")
    if debug_min_size < 1:
        raise ValueError("--debug-min-size must be at least 1.")

    if args.denmark:
        x = [-1080000, -600000]
        y = [-160000, 220000]
        output_subdir = "gridded_denmark"
    else:
        x = [None, None]
        y = [None, None]
        output_subdir = "gridded"

    return debug_mode, debug_fraction, debug_min_size, x, y, output_subdir


(
    DEBUG_MODE,
    DEBUG_FRACTION,
    DEBUG_MIN_SIZE,
    X,
    Y,
    OUTPUT_SUBDIR,
) = _parse_runtime_args()


def subset_dataset_for_debug(ds, label, dims_to_trim):
    if not DEBUG_MODE:
        return ds

    indexers = {}
    for dim in dims_to_trim:
        if dim not in ds.sizes:
            continue
        size = ds.sizes[dim]
        keep = min(
            size, max(DEBUG_MIN_SIZE, int(np.ceil(size * DEBUG_FRACTION)))
        )
        if keep < size:
            indexers[dim] = slice(0, keep)

    if not indexers:
        print(
            f"Debug mode active for {label}, but no matching"  # noqa: E501
            " dimensions were reduced."
        )
        return ds

    trimmed = ds.isel(indexers)
    size_summary = ", ".join(
        f"{dim}={trimmed.sizes[dim]}/{ds.sizes[dim]}" for dim in indexers
    )
    print(f"Debug mode active for {label}: {size_summary}")
    return trimmed


def select_valid_positions(ds, dim, positions, label):
    valid_positions = [
        position for position in positions if position < ds.sizes[dim]
    ]
    if not valid_positions:
        raise ValueError(
            f"No valid positions left for {label} on dimension"  # noqa: E501
            f" '{dim}' with size {ds.sizes[dim]}."
        )
    if len(valid_positions) != len(positions):
        print(
            f"Adjusted {label} {dim} selection from {len(positions)}"  # noqa: E501
            f" to {len(valid_positions)} positions."
        )
    return ds.isel({dim: valid_positions})


def select_available_times(ds, requested_times, label):
    requested_index = pd.Index(np.unique(np.asarray(requested_times).ravel()))
    available_index = pd.Index(ds.time.values)
    matching_times = requested_index.intersection(available_index)
    if matching_times.empty:
        raise ValueError(f"No overlapping times found for {label}.")
    if len(matching_times) != len(requested_index):
        print(
            f"Adjusted {label} time selection from"
            f" {len(requested_index)} to"
            f" {len(matching_times)} available timestamps."
        )
    return ds.sel(time=matching_times.values)


def filter_ml_start_times_with_gt(ds_ml, ds_gt, label):
    available_times = pd.Index(ds_gt.time.values)
    forecast_times = ds_ml.forecast_time.values
    valid_mask = np.isin(forecast_times, available_times).all(axis=1)
    if not valid_mask.any():
        raise ValueError(
            f"No {label} start times have complete ground-truth coverage."
        )
    if not valid_mask.all():
        print(
            f"Adjusted {label} start_time selection from"  # noqa: E501
            f" {len(valid_mask)} to {int(valid_mask.sum())} with complete"
            " ground-truth coverage."
        )
    valid_start_times = ds_ml.start_time.values[valid_mask]
    return ds_ml.sel(start_time=valid_start_times)


def align_on_common_start_times(ds_left, ds_right, left_label, right_label):
    common_start_times = pd.Index(ds_left.start_time.values).intersection(
        pd.Index(ds_right.start_time.values)
    )
    if common_start_times.empty:
        raise ValueError(
            f"No overlapping start times found between"  # noqa: E501
            f" {left_label} and {right_label}."
        )
    if len(common_start_times) != len(ds_left.start_time) or len(
        common_start_times
    ) != len(ds_right.start_time):
        print(
            f"Aligned {left_label} and {right_label} to"  # noqa: E501
            f" {len(common_start_times)} common start times."
        )
    common_values = common_start_times.values
    return ds_left.sel(start_time=common_values), ds_right.sel(
        start_time=common_values
    )


USE_EXAMPLE_DEBUG_DATA = False
if USE_EXAMPLE_DEBUG_DATA:
    # This path should point to the data that was used to train the model
    # (default is mdp-datastore)
    # PATH_GROUND_TRUTH = "danra_test_gt.zarr"
    PATH_GROUND_TRUTH = "danra_ex_fc/danra_gt.zarr"
    # This path should point to the NWP forecast data in zarr format
    # PATH_NWP = "danra_test_nwp_forecasts.zarr"
    PATH_NWP = "danra_ex_fc/danra_nwp_ex_forecasts.zarr"
    # This path should point to the ML forecast data in zarr format
    # (e.g. produced by neural-lam in `eval` mode)
    # PATH_ML = "48h_eval_test_7deg_rect_hi4_2867.zarr"
    PATH_ML = "danra_ex_fc/danra_ml_ex_forecasts.zarr"
    # This path should point to the boundary data in zarr format
    # (default is MDP-datastore)
    # PATH_BOUNDARY = "ifs_test_boundary.zarr"
    PATH_BOUNDARY = "danra_ex_fc/danra_ifs_boundary.zarr"

    START_TIMES = ["2020-01-02T00:00", "2020-01-04T00:00"]
    PLOT_TIME = "2020-01-02T00:00:00"

    VARIABLES_GROUND_TRUTH = {
        # Surface and near-surface variables
        "t2m": "temperature_2m",
        "u10m": "wind_u_10m",
        "v10m": "wind_v_10m",
    }
    VARIABLES_ML = VARIABLES_GROUND_TRUTH

# %%
# Create directories for plots and tables
Path(f"verification/danra/{OUTPUT_SUBDIR}").mkdir(parents=True, exist_ok=True)

# Colorblind-friendly color palette
COLORS = {
    "gt": "#000000",  # Black
    "ml": "#E69F00",  # Orange
    "nwp": "#56B4E9",  # Light blue
    "per": "#CC79A7",  # Pink
}

# Line styles and markers for accessibility
LINE_STYLES = {
    "gt": ("solid", "o"),
    "ml": ("dashed", "s"),
    "nwp": ("dotted", "^"),
    "per": ("dashdot", "v"),
}

# Set global font sizes
apply_style(SCALE_METRICS)

# Colorblind-friendly colormap for 2D plots
COLORMAP = "viridis"

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
    if time is not None:
        name = f"{name}_{time.dt.strftime('%Y%m%d_%H').values}"

    # Sanitize filename by replacing problematic characters
    safe_name = name.replace("/", "_per_")

    # Normalize the path and ensure plots directory exists
    plot_dir = Path(f"verification/danra/{OUTPUT_SUBDIR}")
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
    with open(f"verification/danra/{OUTPUT_SUBDIR}/{name}.tex", "w") as f:
        f.write(latex_str)

    # Export to CSV
    df.to_csv(f"verification/danra/{OUTPUT_SUBDIR}/{name}.csv")


# %%
ds_ml = xr.open_zarr(PATH_ML, decode_timedelta=True)
ds_ml = subset_dataset_for_debug(
    ds_ml,
    label="ML zarr",
    dims_to_trim=("start_time", "x", "y"),
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
ds_gt = subset_dataset_for_debug(
    ds_gt,
    label="ground-truth zarr",
    dims_to_trim=(),
)
ds_gt = ds_gt.set_index(grid_index=["y", "x"]).unstack("grid_index")
ds_gt = subset_dataset_for_debug(
    ds_gt,
    label="ground-truth grid",
    dims_to_trim=("x", "y"),
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
ds_ml = filter_ml_start_times_with_gt(ds_ml, ds_gt, label="ML")
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

ds_ml, ds_nwp = align_on_common_start_times(
    ds_ml,
    ds_nwp,
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
# assert ds_gt.sizes["time"] == len(
#    np.unique(ds_ml.forecast_time.values.flatten())
# )
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
# ### 5. Various Verification Metrics
# The final chapter consolidates various statistical metrics to provide a broad
# evaluation of the ML model's performance. By considering multiple metrics, we
# gain a nuanced understanding of both the strengths and weaknesses of
# the model.
#
# **Metric Diversity:** Including MAE, RMSE, MSE, Pearson correlation, and the
# Fractions Skill Score (FSS) covers different aspects of model
# performance, from
# average errors to spatial pattern accuracy.
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
# **FSS:** Evaluates spatial accuracy, which is particularly important for
# predicting localized weather events.
#
# **Wasserstein Distance:** Provides a holistic view of distributional
# similarity
# across variables. Same as chapter 3.
#
# **Holistic Assessment:** The combination of metrics provides a comprehensive
# performance profile, essential for model validation and comparison. More
# complex metrics are explained in more detail.

# %% [markdown]
# #### Fractions Skill Score
# Range: 0 to 1, where:
# - 1 = perfect score
# - 0 = no skill compared to random chance
#
# **Key Properties:**
# - FSS measures the spatial agreement between two fields, accounting for
#   the spatial scale of the features
# - It's particularly useful for assessing the spatial distribution of
#   precipitation, cloud cover, or other fields with spatial structure
# - FSS is sensitive to the threshold used to define the presence of a
#   feature, so it's important to choose an appropriate threshold
# - The FSS can be calculated for different spatial scales by defining the
#   window size
#
# **Advantages:**
# - More meaningful than simple correlation for spatial fields
# - Accounts for the spatial scale of features
# - Provides a single value for the entire field comparison

# %%
# These helper functions are only used to calculate the FSS threshold
max_spatial_dim = np.maximum(ds_gt.x.size, ds_gt.y.size)
window_size = (int(max_spatial_dim // 100),) * 2
n_points = int(
    np.minimum(
        1e7,
        ds_ml[list(VARIABLES_GROUND_TRUTH.values())[0]]
        .isel(elapsed_forecast_duration=0)
        .size,
    )
)
print(f"Using window size for FSS: {window_size}")
print(f"Using n_points for FSS: {n_points}")


# %%
def calculate_metrics_by_efd(
    ds_gt,
    ds_ml,
    ds_nwp=None,
    metrics_to_compute=None,
    window_size=3,
    prefix="metrics",
):
    """Calculate metrics for each Lead Time for gridded data."""
    if isinstance(window_size, (int, float)):
        window_size = (int(window_size), int(window_size))

    if metrics_to_compute is None:
        metrics_to_compute = METRICS

    variables = list(ds_gt.data_vars)
    elapsed_forecast_durations = ds_ml.elapsed_forecast_duration
    elapsed_forecast_durations_hours = elapsed_forecast_durations.values.astype(
        "timedelta64[s]"
    ) / np.timedelta64(1, "h")

    metrics_by_efd = {}
    combined_metrics = {}

    # Added persistence forecast
    per_fc = ds_gt.sel(time=ds_ml.start_time)

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

        forecast_times = ds_ml_lead.forecast_time  # No .values
        ds_gt_lead = ds_gt.sel(time=forecast_times)

        metrics_dict = {}

        # print(f"ds_ml_lead: {ds_ml_lead}")
        # print(ds_ml_lead.start_time)
        # print(f"ds_nwp_lead: {ds_nwp_lead}")
        # print(ds_nwp_lead.start_time)
        # print(f"per_fc: {per_fc}")

        for var in variables:
            print(f"Processing {var}")

            # Get data as xarray DataArrays
            y_true = ds_gt_lead[var]
            y_pred_ml = ds_ml_lead[var]
            y_pred_per = per_fc[var]

            if comp_for_nwp and var in ds_nwp_lead:
                y_pred_nwp = ds_nwp_lead[var]

            metrics_dict[var] = {}

            # Calculate ML metrics
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
            if "FSS" in metrics_to_compute:
                quantile_90 = y_true.quantile(0.9).values
                metrics_dict[var]["FSS ML"] = fss_2d(
                    y_pred_ml.compute(),
                    y_true.compute(),
                    event_threshold=quantile_90,
                    window_size=window_size,
                    spatial_dims=["y", "x"],
                ).values
            if "Wasserstein" in metrics_to_compute:
                pred_vals = y_pred_ml.values
                true_vals = y_true.values
                metrics_dict[var]["Wasserstein ML"] = wasserstein_distance(
                    pred_vals, true_vals
                )

            # Persistence
            if "RMSE" in metrics_to_compute:
                metrics_dict[var]["RMSE Persistence"] = rmse(
                    y_pred_per, y_true
                ).values

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
                if "FSS" in metrics_to_compute:
                    metrics_dict[var]["FSS NWP"] = fss_2d(
                        y_pred_nwp.compute(),
                        y_true.compute(),
                        event_threshold=quantile_90,
                        window_size=window_size,
                        spatial_dims=["y", "x"],
                    ).values
                if "Wasserstein" in metrics_to_compute:
                    pred_vals = y_pred_nwp.values
                    true_vals = y_true.values
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
    # elapsed_forecast_durations_hours_float = [
    #    x.item() for x in elapsed_forecast_durations_hours
    # ]
    # combined_df = pd.DataFrame(
    #    combined_metrics, index=elapsed_forecast_durations_hours_float
    # )
    # combined_df.index.name = "Forecast Hours"
    # export_table(combined_df, f"{prefix}", "Combined metrics")

    return metrics_by_efd


# %%
# with (
#     LocalCluster(
#         n_workers=1,
#         threads_per_worker=32,
#         memory_limit="48GB",
#         local_directory="/iopsstor/scratch/cscs/sadamov",
#         dashboard_address=None,
#     ) as cluster
# ):
#     with Client(cluster) as client:
metrics_by_efd = calculate_metrics_by_efd(
    ds_gt=ds_gt,
    ds_ml=ds_ml_sampled,
    ds_nwp=ds_nwp_sampled,
    prefix="combined_metrics",
)
metrics_by_efd

# %%
# Plot evolution of a specific metric over lead time
elapsed_forecast_durations = list(metrics_by_efd.keys())
for i, variable in enumerate(VARIABLES_GROUND_TRUTH.values()):
    for metric in METRICS:
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

            fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

            # Plot ML scores
            ax.plot(
                hours,
                ml_scores,
                label="ML" if i == 0 else "",
                color=COLORS["ml"],
                linestyle=LINE_STYLES["ml"][0],
                marker=LINE_STYLES["ml"][1],
            )

            nwp_metric_name = f"{metric} NWP"
            nwp_scores = [
                df.loc[variable, nwp_metric_name]
                for df in metrics_by_efd.values()
                if nwp_metric_name in df
            ]

            # Plot NWP scores if they exist and are not all NaN
            if nwp_scores and not all(pd.isna(nwp_scores)):
                hours_nwp = [
                    x / np.timedelta64(1, "h")
                    for x in ds_nwp.elapsed_forecast_duration.values
                ]

                ax.plot(
                    hours_nwp,
                    nwp_scores,
                    label="NWP" if i == 0 else "",
                    color=COLORS["nwp"],
                    linestyle=LINE_STYLES["nwp"][0],
                    marker=LINE_STYLES["nwp"][1],
                )

            # Persistence
            per_metric_name = f"{metric} Persistence"
            per_scores = [
                df.loc[variable, per_metric_name]
                for df in metrics_by_efd.values()
                if per_metric_name in df
            ]

            # Plot NWP scores if they exist and are not all NaN
            if per_scores and not all(pd.isna(per_scores)):
                hours_per = [
                    x / np.timedelta64(1, "h")
                    for x in ds_ml.elapsed_forecast_duration.values  # noqa: E501
                    # From ml data
                ]

                ax.plot(
                    hours_per,
                    per_scores,
                    label="Persistence" if i == 0 else "",
                    color=COLORS["per"],
                    linestyle=LINE_STYLES["per"][0],
                    marker=LINE_STYLES["per"][1],
                )

            ax.set_xlabel("Lead Time (h)")
            ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
            ax.set_ylabel(f"{metric} ({VARIABLE_UNITS[variable]})")
            ax.set_title(f"{metric} Evolution for {variable}")
            ax.grid(True, alpha=0.3)

            # Only show legend for first variable
            if i == 0:
                ax.legend()

            plt.tight_layout()
            plt.show()
            save_plot(fig, f"{metric}_{variable}_evolution")
            plt.close()

        except (KeyError, ValueError) as e:
            print(f"Skipping {metric} for {variable}: {str(e)}")
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
# Set display options for all float values
pd.set_option("display.float_format", lambda x: "{:.4f}".format(x))


# %%
def calculate_meteoswiss_metrics(ds_gt, ds_ml, ds_nwp):
    """Calculate MeteoSwiss verification metrics"""
    metrics_by_var = {}

    # Get available variables in each dataset
    gt_ml_vars = set(ds_gt.variables) & set(ds_ml.variables)
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
                f"\nCalculating metrics for lead time:"  # noqa: E501
                f" {efd / np.timedelta64(1, 'h'):.1f}h"
            )

            ds_ml_lead = ds_ml.sel(elapsed_forecast_duration=efd)
            ds_nwp_lead = (
                ds_nwp.sel(elapsed_forecast_duration=efd)
                if ds_nwp is not None
                else None
            )
            forecast_times = ds_ml_lead.forecast_time  # No .values
            ds_gt_lead = ds_gt.sel(time=forecast_times)

            for var_name, var_config in all_variables.items():
                print(f"Processing {var_name}")
                try:
                    # Get data
                    y_true = ds_gt_lead[var_name]
                    y_ml = ds_ml_lead[var_name]
                    y_nwp = (
                        ds_nwp_lead[var_name]
                        if ds_nwp_lead is not None and var_name in nwp_vars
                        else None
                    )

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
                    print(f"Error processing {var_name}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error processing lead time {efd}: {str(e)}")
            continue

    return metrics_by_var


def plot_metrics_evolution(
    metrics_by_var,
    elapsed_forecast_durations,
    var_name,
    metric_name,
    var_index=0,
):
    """Plot evolution of FBI/ETS metrics over lead times for gridded data."""
    try:
        if not metrics_by_var or var_name not in metrics_by_var:
            print(f"No metrics data available for {var_name}")
            return

        # Convert timedelta to hours for x-axis
        forecast_hours = [
            efd / np.timedelta64(1, "h") for efd in elapsed_forecast_durations
        ]

        # Get three fixed colors from viridis
        colors = [plt.cm.viridis(x) for x in [0, 0.5, 0.99]]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

        # Plot for each threshold
        for i, (threshold, metrics) in enumerate(
            metrics_by_var[var_name].items()
        ):
            # Plot ML metrics
            if metrics[f"{metric_name}_ML"]:
                ax.plot(
                    forecast_hours,
                    metrics[f"{metric_name}_ML"],
                    linestyle=LINE_STYLES["ml"][0],
                    marker=LINE_STYLES["ml"][1],
                    color=colors[i],
                    label=f"ML {threshold}" if var_index == 0 else "",
                )

            # Plot NWP metrics if available
            if metrics[f"{metric_name}_NWP"]:
                ax.plot(
                    forecast_hours,
                    metrics[f"{metric_name}_NWP"],
                    linestyle=LINE_STYLES["nwp"][0],
                    marker=LINE_STYLES["nwp"][1],
                    color=colors[i],
                    label=f"NWP {threshold}" if var_index == 0 else "",
                )

        ax.set_xlabel("Lead Time (h)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Evolution for {var_name}")
        ax.grid(True, alpha=0.3)

        # Only show legend for first variable
        if var_index == 0:
            ax.legend()

        plt.tight_layout()
        plt.show()
        save_plot(fig, f"{metric_name.lower()}_{var_name}_evolution")
        plt.close()

    except Exception as e:
        print(f"Error plotting {var_name} - {metric_name}: {str(e)}")
        plt.close("all")


# %%
def reshape_metrics_by_var(metrics_by_var, elapsed_forecast_durations):
    """Reshape the metrics dictionary into a DataFrame with thresholds
    as columns."""
    # Convert forecast durations to hours for index
    forecast_hours = [
        efd / np.timedelta64(1, "h") for efd in elapsed_forecast_durations
    ]

    # Create a list to store all data
    data = []

    # Iterate through each forecast hour
    for idx, hour in enumerate(forecast_hours):
        row_data = {"forecast_hour": hour}

        # Iterate through each variable and its thresholds
        for var_name, thresholds in metrics_by_var.items():
            for threshold, metrics in thresholds.items():
                # Add ML metrics if they exist and have values
                if metrics["FBI_ML"] is not None:
                    row_data[f"{var_name}_{threshold}_FBI_ML"] = metrics[
                        "FBI_ML"
                    ][idx]
                    row_data[f"{var_name}_{threshold}_ETS_ML"] = metrics[
                        "ETS_ML"
                    ][idx]

                # Add NWP metrics if they exist and have values
                if metrics["FBI_NWP"] is not None:
                    row_data[f"{var_name}_{threshold}_FBI_NWP"] = metrics[
                        "FBI_NWP"
                    ][idx]
                    row_data[f"{var_name}_{threshold}_ETS_NWP"] = metrics[
                        "ETS_NWP"
                    ][idx]

        data.append(row_data)

    # Create DataFrame from collected data
    df = pd.DataFrame(data)
    df.set_index("forecast_hour", inplace=True)
    return df


# Modify the main execution block:
try:
    print("Calculating MeteoSwiss metrics...")
    metrics_by_var = calculate_meteoswiss_metrics(
        ds_gt,
        ds_ml_sampled.isel(
            elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
        ),
        ds_nwp_sampled.isel(
            elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
        ),
    )

    # Reshape and export metrics to table
    elapsed_forecast_durations = ds_ml.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
    ).elapsed_forecast_duration.values
    metrics_df = reshape_metrics_by_var(
        metrics_by_var, elapsed_forecast_durations
    )

    export_table(
        metrics_df,
        "meteoswiss_metrics",
        "MeteoSwiss verification metrics (FBI and ETS)",
    )

    # Plot metrics for each variable
    for i, var_name in enumerate(metrics_by_var):
        for metric_name in ["FBI", "ETS"]:
            plot_metrics_evolution(
                metrics_by_var,
                elapsed_forecast_durations,
                var_name,
                metric_name,
                var_index=i,
            )

except Exception as e:
    print(f"Error in main execution: {str(e)}")


# %% [markdown]
# The wind vector RMSE takes into account the magnitude and direction of
# the wind, providing a more comprehensive measure of error than scalar
# metrics.

# %%
def wind_vector_rmse(u_pred, v_pred, u_true, v_true):
    """Calculate RMSE based on wind vector differences."""
    rmse_u = rmse(u_true, u_pred)
    rmse_v = rmse(v_true, v_pred)
    rmse_wind = np.sqrt(rmse_u**2 + rmse_v**2)
    return float(rmse_wind)


# Initialize results dictionary
results_wind = {"vector_metrics": {}}
elapsed_forecast_durations = ds_ml_sampled.elapsed_forecast_duration.values

# Lists to store RMSE values over time
ml_rmse_over_time = []
per_rmse_over_time = []
nwp_rmse_over_time = []
forecast_hours = [
    efd / np.timedelta64(1, "h") for efd in elapsed_forecast_durations
]

if "wind_u_10m" in ds_gt and "wind_v_10m" in ds_gt:

    # Persistence
    u_per = ds_gt["wind_u_10m"].sel(time=ds_ml_sampled.start_time)
    v_per = ds_gt["wind_v_10m"].sel(time=ds_ml_sampled.start_time)

    for efd in elapsed_forecast_durations:
        print(f"Calculating wind vector RMSE for EFD: {efd}")
        # Get corresponding forecast times
        forecast_times = ds_ml_sampled.sel(
            elapsed_forecast_duration=efd
        ).forecast_time  # No .values

        # Get wind components for specific forecast time
        u_true = ds_gt["wind_u_10m"].sel(time=forecast_times)
        v_true = ds_gt["wind_v_10m"].sel(time=forecast_times)
        u_ml = ds_ml_sampled["wind_u_10m"].sel(elapsed_forecast_duration=efd)
        v_ml = ds_ml_sampled["wind_v_10m"].sel(elapsed_forecast_duration=efd)

        # Calculate ML RMSE
        wind_rmse_ml = wind_vector_rmse(u_ml, v_ml, u_true, v_true)
        ml_rmse_over_time.append(wind_rmse_ml)

        # Calculate Per RMSE
        wind_rmse_per = wind_vector_rmse(u_per, v_per, u_true, v_true)
        per_rmse_over_time.append(wind_rmse_per)

        # Calculate NWP RMSE if available
        if (
            "wind_u_10m" in ds_nwp
            and "wind_v_10m" in ds_nwp
            and efd in ds_nwp_sampled.elapsed_forecast_duration
        ):
            u_nwp = ds_nwp_sampled["wind_u_10m"].sel(
                elapsed_forecast_duration=efd
            )
            v_nwp = ds_nwp_sampled["wind_v_10m"].sel(
                elapsed_forecast_duration=efd
            )
            wind_rmse_nwp = wind_vector_rmse(u_nwp, v_nwp, u_true, v_true)
            nwp_rmse_over_time.append(wind_rmse_nwp)
        else:
            nwp_rmse_over_time.append(np.nan)

    # Create time series DataFrame
    time_series_df = pd.DataFrame(
        {
            "Lead Time": forecast_hours,
            "ML RMSE": ml_rmse_over_time,
            "Per RMSE": per_rmse_over_time,
            "NWP RMSE": nwp_rmse_over_time,
        }
    )

    # Plot RMSE over time
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ML model
    ax.plot(
        forecast_hours,
        ml_rmse_over_time,
        linestyle=LINE_STYLES["ml"][0],
        marker=LINE_STYLES["ml"][1],
        label="ML",
        color=COLORS["ml"],
    )

    # Plot Per model
    ax.plot(
        forecast_hours,
        per_rmse_over_time,
        linestyle=LINE_STYLES["per"][0],
        marker=LINE_STYLES["per"][1],
        label="Persistence",
        color=COLORS["per"],
    )

    # Plot NWP model if available
    if not all(np.isnan(nwp_rmse_over_time)):
        ax.plot(
            forecast_hours,
            nwp_rmse_over_time,
            linestyle=LINE_STYLES["nwp"][0],
            marker=LINE_STYLES["nwp"][1],
            label="NWP",
            color=COLORS["nwp"],
        )

    ax.set_xlabel("Lead Time (h)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
    ax.set_ylabel("Wind Vector RMSE (m/s)")
    ax.set_title("Wind Vector RMSE Evolution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    save_plot(fig, "wind_vector_rmse_evolution")

    export_table(
        time_series_df,
        "wind_vector_metrics_timeseries",
        caption="Wind vector RMSE over forecast duration",
    )
