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
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import dask
from debug_utils import (
    align_on_common_coord,
    assert_time_coverage,
    filter_forecast_dataset_by_time_coverage,
    parse_debug_runtime_args,
    select_available_times,
    select_valid_positions,
    subset_dataset_for_debug,
)
from plot_styles import (
    DPI,
    FONT_SIZES,
    SCALE_SPECTRA,
    SCALE_WAVENUMBER,
    apply_style,
    scaled_fonts,
)

try:
    # Third-party
    from IPython.display import display
except ImportError:
    display = None
# Third-party
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

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
    "Run gridded DANRA spectra verification.",
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

# Grid spacing — fixes 4Δx marker position when data is subsampled in debug mode
GRID_SPACING_KM: float = 2.5  # DANRA km grid

# Derived font dicts for section overrides
SPECTRA_FONT_SIZES = scaled_fonts(SCALE_SPECTRA)
WAVENUMBER_FONT_SIZES = scaled_fonts(SCALE_WAVENUMBER)

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
        VARIABLE_UNITS[f"{base_var[: -len('_level')]}_{level}hPa"] = unit
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
    (
        ds_ml.forecast_time.values.flatten(),
        ds_ml.start_time.values.flatten(),
    )
)
ds_gt = select_available_times(ds_gt, time_subset, label="ground-truth")
ds_ml = filter_forecast_dataset_by_time_coverage(ds_ml, ds_gt, label="ML")
time_subset = np.concatenate(
    (
        ds_ml.forecast_time.values.flatten(),
        ds_ml.start_time.values.flatten(),
    )
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
    (
        ds_ml.forecast_time.values.flatten(),
        ds_ml.start_time.values.flatten(),
    )
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
#         local_directory="/iopsstor/scratch/cscs/sadamov",
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
# ### 4. Energy Spectra
#
# This chapter examines how energy is distributed across different spatial
# scales in the atmosphere by computing and comparing the energy spectra of
# both models. This analysis is critical in understanding the models'
# capabilities to simulate
# atmospheric processes ranging from large-scale weather systems to small-scale
# turbulence.
#
# **FFT Computation:** The Fast Fourier Transform (FFT) is used to
# transform spatial data into the frequency domain, revealing how different
# scales contribute to the overall energy. The energy spectra are averaged
# over latitudes.
#
# **Scale Representation:** The energy spectra show whether the ML model
# captures the correct amount of energy at various spatial scales.
#
# **Effective Resolution:** Identifying the effective resolution helps
# understand the smallest scales that the model can reliably simulate.
#
# **Numerical Artifacts:** Limitations in numerical precision can introduce
# artifacts in the spectra, especially at the smallest scales.

# %% [markdown]
# To interpret the Log-Spectral Distance (LSD) metric:
# The LSD quantifies the difference between two spectra, with lower values
# indicating better similarity (area between the two spectra).
#
# Lower values indicate better similarity between spectra
# - LSD = 0 means identical spectra
#
# Typical values depend on the specific application, but generally:
# - LSD < 1: Good similarity
# - 1 < LSD < 2: Moderate differences
# - LSD > 2: Significant differences


# %%
def calculate_energy_spectra(data, dx_m=None):
    """Calculate the energy spectra of the given data using 2D FFT.

    Parameters
    ----------
    data : xarray.DataArray
        The data for which the energy spectra should be calculated.
        Expected dimensions must include 'x' and 'y', other dimensions
        will be handled automatically.

    Returns
    -------
    wavenumber : np.ndarray
        The isotropic wavenumbers in m⁻¹.
    power : np.ndarray
        The power spectrum averaged over all non-spatial dimensions.
    effective_resolution : float
        The effective resolution of the model in m⁻¹.
    """

    # Get grid spacing and convert to metres regardless of coordinate units.
    # Heuristic:
    #   |dx| < 1      → rotated/geographic degrees (e.g. COSMO RotatedPole)
    #                    → 1° ≈ 111 km on the great circle
    #   1 ≤ |dx| < 1000 → kilometres
    #   |dx| ≥ 1000   → metres (LambertConformal and similar projections)
    def _to_metres(d: float) -> float:
        if d < 1.0:
            return d * 111_000.0  # degrees → m
        if d < 1_000.0:
            return d * 1_000.0  # km → m
        return d  # already m

    if dx_m is not None:
        dx = dy = float(dx_m)
    else:
        dx = _to_metres(abs(float(data.x[1] - data.x[0])))
        dy = _to_metres(abs(float(data.y[1] - data.y[0])))

    # Identify spatial dimensions
    spatial_dims = ["y", "x"]

    # Move spatial dimensions to the end
    other_dims = [dim for dim in data.dims if dim not in spatial_dims]
    var_data = data.transpose(*other_dims, *spatial_dims).values

    # Reshape the array to combine all non-spatial dimensions
    shape = var_data.shape
    ny, nx = shape[-2:]

    if len(shape) > 2:
        var_data = var_data.reshape(-1, ny, nx)
    else:
        var_data = var_data[np.newaxis, :, :]

    # Compute 2D FFT
    fft_data = np.fft.fft2(var_data, axes=(-2, -1))
    power_spectrum = (np.abs(fft_data) ** 2).mean(axis=0)

    # Rest of the function remains the same
    # Get wavenumbers
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)

    # Create 2D wavenumber grid
    kxx, kyy = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kxx**2 + kyy**2)

    # Create wavenumber bins for azimuthal averaging
    k_bins = np.logspace(
        np.log10(k_mag[k_mag > 0].min()), np.log10(k_mag.max()), num=50
    )

    # Perform azimuthal averaging
    k_averaged = []
    power_averaged = []

    for i in range(len(k_bins) - 1):
        k_mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        if k_mask.any():
            k_averaged.append(np.mean(k_mag[k_mask]))
            power_averaged.append(np.mean(power_spectrum[k_mask]))

    # Convert to arrays
    k_averaged = np.array(k_averaged)
    power_averaged = np.array(power_averaged)

    # Effective resolution: practical minimum resolvable scale =
    # 4 * grid spacing
    effective_resolution = 1 / (4 * dx)

    # Remove first and last two wavenumbers
    return k_averaged[2:-2], power_averaged[2:-2], effective_resolution


def calculate_all_spectra(ds_gt, ds_ml, ds_nwp, variables):
    """Calculate and cache all energy spectra, including temporal evolution.

    Parameters
    ----------
    ds_gt : xarray.Dataset
        Ground truth dataset
    ds_ml : xarray.Dataset
        Machine learning model dataset
    ds_nwp : xarray.Dataset
        Numerical weather prediction model dataset
    variables : list
        List of variables to process

    Returns
    -------
    dict
        Nested dictionary containing wavenumbers, spectra and effective
        resolution
        for each dataset and variable, including temporal evolution
    """
    spectra_cache = {
        "gt": {},
        "ml": {"static": {}, "temporal": {}},
        "nwp": {"static": {}, "temporal": {}} if ds_nwp is not None else None,
    }

    for var in variables:
        if var not in ds_gt or var not in ds_ml:
            continue
        print(f"Calculating spectra for {var}...")

        _dx = (
            GRID_SPACING_KM * 1_000.0
        )  # fixed spacing avoids debug-mode artefact

        # Calculate static spectra for ground truth
        k_gt, spec_gt, eff_res = calculate_energy_spectra(ds_gt[var], dx_m=_dx)
        spectra_cache["gt"][var] = (k_gt, spec_gt, eff_res)

        # Calculate temporal spectra for ML model
        forecast_times = ds_ml.elapsed_forecast_duration.values
        ml_temporal = {}
        for time in forecast_times:
            data = ds_ml[var].sel(elapsed_forecast_duration=time)
            k, spec, _ = calculate_energy_spectra(data, dx_m=_dx)
            ml_temporal[time] = (k, spec)

        spectra_cache["ml"]["temporal"][var] = ml_temporal
        spectra_cache["ml"]["static"][var] = calculate_energy_spectra(
            ds_ml[var], dx_m=_dx
        )

        # Calculate temporal spectra for NWP model if available
        if ds_nwp is not None and var in ds_nwp:
            forecast_times_nwp = ds_nwp.elapsed_forecast_duration.values
            nwp_temporal = {}
            for time in forecast_times_nwp:
                data = ds_nwp[var].sel(elapsed_forecast_duration=time)
                k, spec, _ = calculate_energy_spectra(data)
                nwp_temporal[time] = (k, spec)

            spectra_cache["nwp"]["temporal"][var] = nwp_temporal
            spectra_cache["nwp"]["static"][var] = calculate_energy_spectra(
                ds_nwp[var], dx_m=_dx
            )

    return spectra_cache


def plot_energy_spectra(spectra_cache, var, level=None, show_legend=False):
    """Plot energy spectra comparison using pre-calculated spectra.

    Parameters
    ----------
    spectra_cache : dict
        Cache containing pre-calculated spectra
    var : str
        Variable name to plot
    level : float, optional
        Vertical level in hPa
    show_legend : bool, optional
        Whether to show the legend (default: False)
    """
    k_gt, spec_gt, eff_res = spectra_cache["gt"][var]
    k_ml, spec_ml, _ = spectra_cache["ml"]["static"][var]

    # Convert wavenumber (1/m) to wavelength (km)
    wl_gt = 1.0 / k_gt / 1000.0
    wl_ml = 1.0 / k_ml / 1000.0
    eff_res_wl = 1.0 / eff_res / 1000.0

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=DPI)

    # Plot ground truth spectrum
    ax.loglog(
        wl_gt,
        spec_gt,
        color=COLORS["gt"],
        label="Ground Truth",
        linestyle=LINE_STYLES["gt"][0],
        marker=LINE_STYLES["gt"][1],
        markevery=5,
    )

    # Plot NWP spectrum if available
    if spectra_cache["nwp"] and var in spectra_cache["nwp"]["static"]:
        k_nwp, spec_nwp, _ = spectra_cache["nwp"]["static"][var]
        wl_nwp = 1.0 / k_nwp / 1000.0
        ax.loglog(
            wl_nwp,
            spec_nwp,
            color=COLORS["nwp"],
            label="NWP Model Prediction",
            linestyle=LINE_STYLES["nwp"][0],
            marker=LINE_STYLES["nwp"][1],
            markevery=3,
        )

    # Plot ML spectrum
    ax.loglog(
        wl_ml,
        spec_ml,
        color=COLORS["ml"],
        label="ML Model Prediction",
        linestyle=LINE_STYLES["ml"][0],
        marker=LINE_STYLES["ml"][1],
        markevery=4,
    )

    # Plot effective resolution (4Δx) as wavelength
    ax.axvline(
        eff_res_wl,
        color="salmon",
        linestyle="--",
        label="4Δx",
    )
    ax.text(
        eff_res_wl,
        0.02,
        "4Δx",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        color="salmon",
        fontsize=FONT_SIZES.get("stats", 17),
    )

    # Reference scale lines (Thunis & Bornstein 1996) — boundaries and
    # mid-band labels
    _SCALE_BOUNDARIES_KM = [2000.0, 200.0, 20.0, 2.0]
    _SCALE_BAND_LABELS = ["Meso-α", "Meso-β", "Meso-γ"]
    for wl_ref in _SCALE_BOUNDARIES_KM:
        ax.axvline(
            wl_ref, color="gray", linestyle=":", linewidth=1.0, alpha=0.7
        )
    for i, label in enumerate(_SCALE_BAND_LABELS):
        wl_mid = np.sqrt(_SCALE_BOUNDARIES_KM[i] * _SCALE_BOUNDARIES_KM[i + 1])
        ax.text(
            wl_mid,
            1.0,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES.get("stats", 14),
            color="gray",
        )

    # Invert x-axis: large wavelengths (synoptic) on left, small
    # (convective) on right
    ax.invert_xaxis()

    # Add LSD metric
    spec_nwp = (
        spectra_cache["nwp"]["static"][var][1]
        if spectra_cache["nwp"] and var in spectra_cache["nwp"]["static"]
        else None
    )
    add_lsd_to_plot(ax, spec_gt, spec_ml, spec_nwp)

    # Customize plot
    ax.set_xlabel("Wavelength (km)")
    unit = VARIABLE_UNITS.get(var, "")
    ax.set_ylabel(f"Energy Density ({unit}^2 m)")
    title = f"Energy Spectra Comparison for {var}"
    if level is not None:
        title += f" at Level {level} hPa"
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Save and display plot
    plot_name = f"energy_spectra_{var}"
    if level is not None:
        plot_name += f"_level_{level}"
    plt.tight_layout()
    plt.show()
    save_plot(fig, plot_name)
    plt.close()


def display_lsd_table(spectra_cache, variables, name, caption=""):
    """Display and export LSD metrics table using pre-calculated spectra.

    Parameters
    ----------
    spectra_cache : dict
        Cache containing pre-calculated spectra
    variables : list
        List of variables to analyze
    name : str
        Name for exported files
    caption : str, optional
        Caption for LaTeX table
    """
    lsd_data = {var: {"ML": None, "NWP": None} for var in variables}

    for var in variables:
        if (
            var not in spectra_cache["gt"]
            or var not in spectra_cache["ml"]["static"]
        ):
            continue

        spec_gt = spectra_cache["gt"][var][1]
        spec_ml = spectra_cache["ml"]["static"][var][1]
        spec_nwp = (
            spectra_cache["nwp"]["static"][var][1]
            if spectra_cache["nwp"] and var in spectra_cache["nwp"]["static"]
            else None
        )

        lsd_ml, lsd_nwp = calculate_log_spectral_distance(
            spec_gt, spec_ml, spec_nwp
        )
        lsd_data[var]["ML"] = lsd_ml
        if lsd_nwp is not None:
            lsd_data[var]["NWP"] = lsd_nwp

    df = pd.DataFrame(lsd_data).T

    # Display styled table
    styled_df = df.style.format(
        lambda x: f"{x:.3f}" if pd.notnull(x) else "-"
    ).map(
        lambda x: (
            f"color: {'green' if x < 1 else 'orange' if x < 2 else 'red'}"
            if pd.notnull(x)
            else ""
        )
    )
    if display is not None:
        display(styled_df)
    else:
        print(df.to_string(float_format=lambda x: f"{x:.3f}"))

    # Export raw dataframe
    export_table(df, name, caption)


def plot_wavenumber_evolution(
    spectra_cache,
    ds_ml,
    ds_nwp,
    variables,
    wavenumbers=None,
    level=None,
):
    """Plot wavenumber evolution using pre-calculated temporal spectra.

    Parameters
    ----------
    spectra_cache : dict
        Cache containing pre-calculated spectra including temporal evolution
    ds_ml : xarray.Dataset
        Machine learning model dataset (used only for time coordinates)
    ds_nwp : xarray.Dataset
        Numerical weather prediction model dataset
    variables : list
        List of variables to analyze
    wavenumbers : list, optional
        List of wavenumbers to analyze. If None, three representative scales
        (large / medium / small) are auto-selected from the data's wavenumber
        range matching typical synoptic, mesoscale, and regional thresholds.
    level : float, optional
        Vertical level in hPa
    """
    # Meteorological scale thresholds (Orlanski 1975; Thunis & Bornstein 1996).
    # These apply to both the COSMO Alpine domain (~700 km) and the DANRA
    # Danish domain (~800 km) and are the same in both scripts.
    _WAVEBAND_TARGETS_KM = [
        500.0,
        100.0,
        10.0,
    ]  # Meso-03b1 / Meso-03b2 / Meso-03b3 targets
    _SCALE_LABELS = ["Meso-α", "Meso-β", "Meso-γ"]
    forecast_times = ds_ml.elapsed_forecast_duration.values

    for var in variables:
        if (
            var not in spectra_cache["gt"]
            or var not in spectra_cache["ml"]["temporal"]
        ):
            continue

        if wavenumbers is None:
            k_gt_vals = spectra_cache["gt"][var][0]
            # Convert nominal wavelengths (km) to wavenumbers (m⁻¹), then
            # clamp to the available wavenumber range of this dataset.
            target_ks = [
                float(
                    np.clip(
                        1.0 / (lam_km * 1_000.0),
                        k_gt_vals.min(),
                        k_gt_vals.max(),
                    )
                )
                for lam_km in _WAVEBAND_TARGETS_KM
            ]
        else:
            target_ks = wavenumbers

        fig, axes = plt.subplots(1, len(target_ks), figsize=(24, 8), dpi=DPI)

        for idx, target_k in enumerate(target_ks):
            spectra = {
                "gt": [],
                "ml": [],
                "nwp": [] if spectra_cache["nwp"] else None,
            }

            # Get ground truth power for target wavenumber
            k_gt, spec_gt, _ = spectra_cache["gt"][var]
            idx_k_gt = np.abs(k_gt - target_k).argmin()
            gt_power = spec_gt[idx_k_gt]

            for time in forecast_times:
                # Get ML spectra from cache
                k_ml, spec_ml = spectra_cache["ml"]["temporal"][var][time]
                idx_k_ml = np.abs(k_ml - target_k).argmin()
                spectra["ml"].append(spec_ml[idx_k_ml])

                # Get NWP spectra from cache if available
                if (
                    spectra_cache["nwp"]
                    and var in spectra_cache["nwp"]["temporal"]
                    and time in spectra_cache["nwp"]["temporal"][var]
                ):
                    k_nwp, spec_nwp = spectra_cache["nwp"]["temporal"][var][
                        time
                    ]
                    idx_k_nwp = np.abs(k_nwp - target_k).argmin()
                    spectra["nwp"].append(spec_nwp[idx_k_nwp])
                else:
                    spectra["nwp"].append(np.nan)

                spectra["gt"].append(gt_power)

            # Plot evolution for current wavenumber
            hours = forecast_times / np.timedelta64(1, "h")

            # Export table
            export_table(
                pd.DataFrame(
                    {
                        key.upper(): values
                        for key, values in spectra.items()
                        if values
                    },
                    index=hours,
                ).rename_axis(
                    index="Lead Time (hours)",
                    columns=f"Energy Density (k={target_k:.2e})",
                ),
                f"wavenumber_evolution_{var}_k{target_k:.2e}",
            )

            # Plot spectra
            for key, values in spectra.items():
                if not values:
                    continue

                if not np.isnan(values).all():
                    axes[idx].plot(
                        hours,
                        values,
                        color=COLORS[key],
                        linestyle=LINE_STYLES[key][0],
                        marker=LINE_STYLES[key][1],
                        label=key.upper(),
                    )

            # Add titles and labels — show the nominal target wavelength so
            # both COSMO and DANRA use the same labelling convention.
            if wavenumbers is None:
                wl_km = _WAVEBAND_TARGETS_KM[idx]
            else:
                wl_km = 1.0 / target_k / 1_000.0
            scale_lbl = _SCALE_LABELS[idx] if idx < len(_SCALE_LABELS) else ""
            axes[idx].set_title(f"{scale_lbl}\n\u03bb = {wl_km:.0f} km")
            axes[idx].xaxis.set_major_locator(mticker.MultipleLocator(6))
            if idx == len(target_ks) // 2:  # Only middle plot gets x-label
                axes[idx].set_xlabel("Lead Time (hours)")
            if idx == 0:
                axes[idx].set_ylabel("Energy Density")
                axes[idx].legend()

        plt.tight_layout()
        plt.show()
        save_plot(fig, f"wavenumber_evolution_{var}", remove_title=False)


def calculate_log_spectral_distance(true_spectrum, ml_spectrum, nwp_spectrum):
    """
    Calculate the Log Spectral Distance between three power spectra
    """
    eps = 1e-10
    log_spec1 = np.log10(true_spectrum + eps)
    log_spec2 = np.log10(ml_spectrum + eps)
    lsd_ml = np.sqrt(np.mean((log_spec1 - log_spec2) ** 2))
    if nwp_spectrum is None:
        return lsd_ml, None
    log_spec3 = np.log10(nwp_spectrum + eps)
    lsd_nwp = np.sqrt(np.mean((log_spec1 - log_spec3) ** 2))
    return lsd_ml, lsd_nwp


def add_lsd_to_plot(ax, true_spectrum, ml_spectrum, nwp_spectrum):
    """
    Add LSD metric as text box to spectrum plot
    """
    if nwp_spectrum is None:
        lsd_nwp = None
        lsd_ml, _ = calculate_log_spectral_distance(
            true_spectrum, ml_spectrum, None
        )
        textstr = f"LSD ML = {lsd_ml:.4f}"
    else:
        lsd_ml, lsd_nwp = calculate_log_spectral_distance(
            true_spectrum, ml_spectrum, nwp_spectrum
        )
        textstr = f"LSD NWP = {lsd_nwp:.4f}, LSD ML = {lsd_ml:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.1,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )


# %%
# Calculate all spectra once and cache them
spectra_cache_short = calculate_all_spectra(
    ds_gt_sampled,
    ds_ml_sampled.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
    ),
    ds_nwp_sampled.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SHORT
    ),
    VARIABLES_GROUND_TRUTH.values(),
)

# %%
# Energy spectra
apply_style(SCALE_SPECTRA)
for i, var in enumerate(VARIABLES_GROUND_TRUTH.values()):
    plot_energy_spectra(spectra_cache_short, var, show_legend=(i == 0))

# %%
# Display and export LSD metrics table
display_lsd_table(
    spectra_cache_short,
    VARIABLES_GROUND_TRUTH.values(),
    name="lsd_metrics",
    caption=(
        "Log Spectral Distance (LSD) metrics comparison between ML and NWP"
        " models"
    ),
)


# %%
# Re-compute to get all lead times for ML model
spectra_cache = calculate_all_spectra(
    ds_gt_sampled,
    ds_ml_sampled,
    ds_nwp_sampled,
    VARIABLES_GROUND_TRUTH.values(),
)

# %%
# Wavenumber evolution (+1 pt larger than spectra base)
apply_style(SCALE_WAVENUMBER)
plot_wavenumber_evolution(
    spectra_cache, ds_ml, ds_nwp, variables=VARIABLES_GROUND_TRUTH.values()
)
apply_style()  # restore base


# %%
