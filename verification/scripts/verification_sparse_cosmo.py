# %% [markdown]
# ##  Sparse Model Verification
#
# This script verifies output from a ML-based foundation model versus a
# traditional NWP system for the atmospheric system. The defaults set at
# the top of this script are tailored to the Alps-Clariden HPC system at CSCS.
# - The NWP-model is called COSMO-E and is initialised with the ensemble mean
#   of the analysis. Only surface level data is available in the archive at
#   MeteoSwiss.
# - The ML-model is called Neural-LAM and is initialised with the deterministic
#   analysis.
# - The Ground Truth are surface level observations from MeteoSwiss.
#
# For more info about the COSMO model see:
# - https://www.cosmo-model.org/content/model/cosmo/coreDocumentation/cosmo_io_guide_6.00.pdf  # noqa: E501  # noqa: E501
# - https://www.research-collection.ethz.ch/handle/20.500.11850/720460

# Standard library
# %%
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    SCALE_CASE_STUDY,
    SCALE_ERROR_COSMO,
    apply_style,
    scaled_fonts,
)
from scipy.spatial import cKDTree
from scipy.stats import kurtosis, skew
from scores.continuous import mean_error

# %%
# DEFAULTS
# This config will be applied to the data before any plotting. The data will be
# sliced and indexed according to the values in this config. The whole analysis
# plotting will be done on the reduced data.

# IF YOUR DATA HAS DIFFERENT DIMENSIONS OR NAMES, PLEASE ADJUST THE CELLS BELOW
# MAKE SURE THE XARRAY DATASETS LOOK OKAY BEFORE RUNNING CHAPTER 1-4

# This path should point to the NWP forecast data in zarr format
PATH_NWP = "cosmo_e_forecast.zarr"
# This path should point to the ML forecast data in zarr format
# (e.g. produced by neural-lam in `eval` mode)
PATH_ML = "preds_7_19_margin_interior_lr_0001_ar_12.zarr"
# This path should point to the observations data in zarr format
PATH_OBS = "cosmo_observations.zarr"

# All variables should map to the same values.
# For obs please also theck the pre-processing in the cell below
# Often some data wrangling is required to match the variables
# OBS and ML variables should match and always be present
VARIABLES_ML = {
    "T_2M": "temperature_2m",
    "U_10M": "wind_u_10m",
    "V_10M": "wind_v_10m",
    "PS": "surface_pressure",
    "TOT_PREC": "precipitation",
}
# The function calls are flexible to allow for missing nwp data
VARIABLES_NWP = {
    "temperature_2m": "temperature_2m",
    "wind_u_10m": "wind_u_10m",
    "wind_v_10m": "wind_v_10m",
    "surface_pressure": "surface_pressure",
    "precipitation_1hr": "precipitation",
}
VARIABLES_OBS = [
    "air_temperature",
    "wind_speed",
    "wind_direction",
    "air_pressure",
    "precipitation",
]

# Add units dictionary after the imports
# units from zarr archives are not reliable and should rather be defined here
VARIABLE_UNITS = {
    # Surface and near-surface variables
    "temperature_2m": "K",
    "wind_u_10m": "m/s",
    "wind_v_10m": "m/s",
    "wind_speed": "m/s",
    "surface_pressure": "Pa",
    "precipitation": "mm/h",
}

# lead time in steps for the forecast - [0] refers to the first forecast step
# at t+1
# this should be a list of integers
ELAPSED_FORECAST_DURATION = [0, 23, 47, 71, 95, 119]

# Subset of lead time steps used for map plots (→ 24h, 72h, 120h)
# Positional indices into ELAPSED_FORECAST_DURATION
ELAPSED_FORECAST_DURATION_PLOT = [1, 3, 5]

# Select specific start_times for the forecast. This is the start and end of
# a slice in xarray. The start_time is included, the end_time is excluded.
# This should be a list of two strings in the format "YYYY-MM-DDTHH:MM:SS"
# Should be handy to evaluate certain dates, e.g. for a case study of a storm
START_TIMES = ["2019-10-31T00:00:00", "2020-10-23T13:00:00"]  # Full year
# START_TIMES = ["2020-02-08T00:00:00", "2020-02-15T00:00:00"]  # Ciara/Sabine

# Select specific plot times for the forecast (will be used to create maps for
# all variables)
# This only affect chapter one with the plotting of the maps
# Map creation takes a lot of time so this is limited to a single time step
# Simply rerun these cells and chapter one for more time steps
PLOT_TIME = "2020-02-07T00:00:00"

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
PROJECTION = ccrs.RotatedPole(
    pole_longitude=190,
    pole_latitude=43,
    central_rotated_longitude=10,
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


# Font sizes for comparison maps (obs/NWP/ML side-by-side, +50%)
COMPARISON_FONT_SIZES = scaled_fonts(
    SCALE_CASE_STUDY
)  # obs case-study scatter maps
ERROR_MAP_FONT_SIZES = scaled_fonts(SCALE_ERROR_COSMO)  # mean error maps (+80%)

DEBUG_MODE, DEBUG_FRACTION, DEBUG_MIN_SIZE = parse_debug_runtime_args(
    "Run sparse COSMO verification.",
)


# %%
# Create directories for plots and tables
Path("verification/cosmo/case_study").mkdir(parents=True, exist_ok=True)
Path("verification/cosmo/sparse").mkdir(parents=True, exist_ok=True)
Path("verification/cosmo/gridded").mkdir(parents=True, exist_ok=True)

# Accessible color palette
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
        plot_dir = Path("verification/cosmo/case_study")
    elif full_name.startswith("observations_"):
        plot_dir = Path("verification/cosmo/sparse")
    else:
        plot_dir = Path("verification/cosmo/gridded")
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
    with open(f"verification/cosmo/sparse/observations_{name}.tex", "w") as f:
        f.write(latex_str)

    # Export to CSV
    df.to_csv(f"verification/cosmo/sparse/observations_{name}.csv")


# %%
ds_nwp = xr.open_zarr(PATH_NWP, decode_timedelta=True)
ds_nwp = subset_dataset_for_debug(
    ds_nwp,
    label="NWP zarr",
    dims_to_trim=("time", "x", "y"),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
)
ds_nwp = ds_nwp.sel(time=slice(*START_TIMES))
ds_nwp = ds_nwp[VARIABLES_NWP.keys()].rename(VARIABLES_NWP)
ds_nwp = ds_nwp.rename_dims(
    {
        "lead_time": "elapsed_forecast_duration",
        "time": "start_time",
    }
)
ds_nwp = ds_nwp.rename_vars(
    {
        "lead_time": "elapsed_forecast_duration",
        "time": "start_time",
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

# # Calculate hourly values by taking differences along
# # elapsed_forecast_duration
ds_nwp["precipitation"] = ds_nwp.precipitation.diff(
    dim="elapsed_forecast_duration"
)
# The NWP data starts at lead time 0 = start_time
ds_nwp = ds_nwp.drop_isel(elapsed_forecast_duration=0)
ds_nwp = select_valid_positions(
    ds_nwp,
    dim="elapsed_forecast_duration",
    positions=ELAPSED_FORECAST_DURATION,
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


# %%
OBS_VAR_MAPPING = {
    "air_temperature": "temperature_2m",
    "air_pressure": "surface_pressure",
    "precipitation": "precipitation",
}


def calculate_wind_components(
    ds, speed_var="wind_speed", dir_var="wind_direction"
):
    """Calculate u and v wind components from speed and direction.

    Args:
        ds: xarray Dataset containing wind speed and direction
        speed_var: name of wind speed variable
        dir_var: name of wind direction variable (meteorological
            convention, 0=N, 90=E)

    Returns:
        ds: Dataset with new wind_u_10m (+ = eastward) and wind_v_10m
            (+ = northward)
    """
    ds = ds.copy()
    ds["wind_u_10m"] = -ds[speed_var] * np.sin(np.radians(ds[dir_var]))
    ds["wind_v_10m"] = -ds[speed_var] * np.cos(np.radians(ds[dir_var]))
    ds = ds.drop_vars([speed_var, dir_var])
    return ds


ds_obs = xr.open_zarr(PATH_OBS, decode_timedelta=True)
ds_obs = subset_dataset_for_debug(
    ds_obs,
    label="observation zarr",
    dims_to_trim=("station",),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
)
ds_obs = ds_obs[VARIABLES_OBS].rename_vars(OBS_VAR_MAPPING)
ds_obs = select_available_times(
    ds_obs,
    ds_ml.forecast_time.values.flatten(),
    label="observations",
)
ds_obs = ds_obs.where(ds_obs != 32767, np.nan)
ds_obs = calculate_wind_components(ds_obs)
ds_obs["temperature_2m"] += 273.15  # Convert to Kelvin
ds_obs["surface_pressure"] *= 100  # Convert to Pa
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
# Interpolation from the gridded model data to the station locations is
# done using the nearest neighbor method. Since we are only looking at
# surface level data, this is a reasonable approach. However, if you are
# working with data at different levels, you may want to consider a more
# sophisticated interpolation method.


# %%
def interpolate_to_obs(
    ds_model_1, ds_model_2, ds_obs, vars_plot, neighbors=None
):
    """Interpolate model datasets to observation points using xarray/dask
    in rotated pole coordinates."""

    print("Starting optimized interpolation in rotated pole projection...")

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

    # Build KDTree with rotated coordinates
    print("Building KD-tree and finding neighbors...")
    k = 4 if neighbors is None else neighbors
    kdtree = cKDTree(points_model)
    distances, flat_indices = kdtree.query(points_obs, k=k)

    # Convert flat indices back to 2D indices
    _, ny = ds_model_1.x.size, ds_model_1.y.size
    x_indices = flat_indices // ny
    y_indices = flat_indices % ny

    # Compute weights
    weights = xr.DataArray(
        1.0 / (distances + 1e-10), dims=["station", "neighbor"]
    )
    weights = weights / weights.sum("neighbor")

    def interpolate_variable(var):
        print(f"Processing variable: {var}")
        data_1 = ds_model_1[var].isel(
            x=xr.DataArray(x_indices, dims=["station", "neighbor"]),
            y=xr.DataArray(y_indices, dims=["station", "neighbor"]),
        )
        data_2 = ds_model_2[var].isel(
            x=xr.DataArray(x_indices, dims=["station", "neighbor"]),
            y=xr.DataArray(y_indices, dims=["station", "neighbor"]),
        )
        return (
            var,
            (data_1 * weights).sum(dim="neighbor"),
            (data_2 * weights).sum(dim="neighbor"),
        )

    # Process all variables
    results = {var: interpolate_variable(var)[1:] for var in vars_plot}

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


# Use the function
ds_ml_interp, ds_nwp_interp = interpolate_to_obs(
    ds_ml, ds_nwp, ds_obs, VARIABLES_ML.values(), neighbors=4
)

# from dask.distributed import Client, LocalCluster
# with (
#     LocalCluster(
#         n_workers=4,
#         threads_per_worker=32,
#         memory_limit="96GB",
#         local_directory="/iopsstor/scratch/cscs/sadamov",
#         # Use fast local storage for spilling
#         dashboard_address=None,
#     ) as cluster
# ):
#     with Client(cluster) as client:
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
# Define the map extent based on the ML data
extent = [
    ds_ml.longitude.min().values + 6.2,
    ds_ml.longitude.max().values - 6.7,
    ds_ml.latitude.min().values + 3.4,
    ds_ml.latitude.max().values - 2.4,
]
if PLOT_TIME is None:
    plot_time = None
else:
    plot_time = resolve_requested_time(
        ds_ml, "start_time", PLOT_TIME, label="plot time"
    )


# %%
def add_map_features(axes):
    n_rows, _ = axes.shape
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            ax.coastlines(resolution="50m")
            ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.7)
            gl = ax.gridlines(
                draw_labels=True, dms=True, x_inline=False, y_inline=False
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
    Plot comparison between observations, NWP and ML data for each variable
    and forecast step.

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
    pressure_var = "surface_pressure"
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

    def _get_data(ds_slice, var, is_obs=False):
        if var == "wind_speed":
            if is_obs and "wind_speed" in ds_slice.data_vars:
                return ds_slice["wind_speed"].values
            return _get_wind_speed(ds_slice)
        return ds_slice[var].values

    # Get number of forecast steps
    n_steps = len(ds_ml.elapsed_forecast_duration)

    for var in variables:
        # Create figure with n_steps rows and 3 columns
        fig = plt.figure(
            figsize=(27, 6 * n_steps),
            dpi=DPI,
        )
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

        # Set extent for all subplots
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
                    _get_data(obs_slice, var, is_obs=True),
                    _get_data(nwp_slice, var),
                    _get_data(ml_slice, var),
                ]
            )

        vmin = float(np.nanmin([np.nanmin(a) for a in arrays_for_minmax]))
        vmax = float(np.nanmax([np.nanmax(a) for a in arrays_for_minmax]))

        _cmap = get_colormap_for_variable(var)
        if var == "wind_speed":
            vmin = 0.0

        # Use log scale for precipitation (avoid zero/negative in LogNorm)
        var_norm = None
        if "precipitation" in var.lower() or "tot_prec" in var.lower():
            log_vmin = max(vmin, 0.01)
            log_vmax = max(vmax, log_vmin * 10)
            var_norm = mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)

        norm_kw = (
            {"norm": var_norm}
            if var_norm is not None
            else {"vmin": vmin, "vmax": vmax}
        )
        scatter_kw = dict(cmap=_cmap, transform=ccrs.PlateCarree(), **norm_kw)
        mesh_kw = dict(
            cmap=_cmap,
            transform=ccrs.PlateCarree(),
            shading="auto",
            rasterized=True,
            **norm_kw,
        )

        def _add_pressure_contour(ax, ds_slice):
            if pressure_var in ds_slice.data_vars:
                p = ds_slice[pressure_var].values / 100  # Pa → hPa
                p_min = float(np.nanmin(p))
                p_max = float(np.nanmax(p))
                # Surface pressure over Alps: use 60 hPa interval to
                # show terrain outline without clutter
                levels = np.arange(
                    np.floor(p_min / 60) * 60,
                    np.ceil(p_max / 60) * 60 + 60,
                    60,
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
                ax.clabel(cs, fmt="%d", fontsize=6, inline=True)

        # Plot for each forecast step
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
                c=_get_data(obs_slice, var, is_obs=True),
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

            # Titles
            if step_idx == 0:
                axes[step_idx, 0].set_title(f"Observations\n+{forecast_hours}h")
                axes[step_idx, 1].set_title(f"NWP\n+{forecast_hours}h")
                axes[step_idx, 2].set_title(f"ML\n+{forecast_hours}h")
            else:
                axes[step_idx, 0].set_title(f"+{forecast_hours}h")
                axes[step_idx, 1].set_title(f"+{forecast_hours}h")
                axes[step_idx, 2].set_title(f"+{forecast_hours}h")

        add_map_features(axes)

        plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.105, wspace=0.03)

        cbar_ax = fig.add_axes([0.15, 0.0, 0.7, 0.02])
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
    fig = plt.figure(figsize=(27, 6 * n_steps), dpi=DPI)
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

    # Use log scale for precipitation (avoid zero/negative in LogNorm)
    var_norm = None
    if "precipitation" in var_plot.lower() or "tot_prec" in var_plot.lower():
        log_vmin = max(vmin, 0.01)
        log_vmax = max(vmax, log_vmin * 10)
        var_norm = mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)

    norm_kw = (
        {"norm": var_norm}
        if var_norm is not None
        else {"vmin": vmin, "vmax": vmax}
    )

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
                    rasterized=True,
                    **norm_kw,
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
                    rasterized=True,
                    **norm_kw,
                )

            ax.add_feature(cfeature.COASTLINE, edgecolor="grey")
            ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="grey")

            # Configure gridlines - only on edges
            gl = ax.gridlines(
                draw_labels=True, alpha=0.2, x_inline=False, y_inline=False
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
        hspace=0.075,
        wspace=0.03,
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
    )
    return fig


# Apply larger font sizes for comparison maps (obs/NWP/ML side-by-side)
apply_style(SCALE_CASE_STUDY)

# Call the function for each variable
for var in VARIABLES_ML.values():
    fig = plot_comparison_interpolated(
        ds_obs,
        ds_ml_interp,
        ds_nwp_interp,
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
                    f"NWP\n+{forecast_hours}h"
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
                f"ML \n+{forecast_hours}h"
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
                    draw_labels=True, alpha=0.2, x_inline=False, y_inline=False
                )
                gl.top_labels = False
                gl.right_labels = False
                gl.left_labels = ax == axes[idx, 0]
                gl.bottom_labels = idx == n_forecast_times - 1

        # Adjust subplot spacing
        plt.subplots_adjust(
            top=0.9,
            bottom=0.05,
            hspace=0.053,
            wspace=0.03,
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


apply_style(SCALE_ERROR_COSMO)
fig = plot_mean_error_maps(
    ds_obs,
    ds_ml_interp.isel(elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT),
    ds_nwp_interp.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT
    ),
)

# Restore default font sizes after error map plots
apply_style()


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

        # Create histograms for observations
        ax.hist(
            data_obs,
            bins=200,
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
            bins=200,
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
            bins=200,
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
        table.set_fontsize(FONT_SIZES["stats"])

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
plot_interpolated_histograms(ds_obs, ds_ml_interp, ds_nwp_interp)
