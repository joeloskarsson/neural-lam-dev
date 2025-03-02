# %% [markdown]
# ## Verification of Two Model Forecasts vs Observations (Measurements)

# %%
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from scipy.spatial import cKDTree
from scipy.stats import kurtosis, skew, wasserstein_distance
from scores.categorical import ThresholdEventOperator as TEO
from scores.continuous import (
    mae,
    mean_error,
    mse,
    rmse,
)
from scores.continuous.correlation import pearsonr

# %%
# This path should point to the NWP forecast data in zarr format
PATH_NWP = "/capstor/store/cscs/swissai/a01/sadamov/cosmo_e_forecast.zarr"
# This path should point to the ML forecast data in zarr format (e.g. produced by neural-lam in `eval` mode)
PATH_ML = "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/eval_results/preds_7_19_margin_interior_lr_0001_ar_12.zarr"
# This path should point to the observations data in zarr format
PATH_OBS = "/capstor/store/cscs/swissai/a01/sadamov/cosmo_observations.zarr"

VARIABLES_ML = {
    "T_2M": "temperature_2m",
    "U_10M": "wind_u_10m",
    "V_10M": "wind_v_10m",
    "PS": "surface_pressure",
    "TOT_PREC": "precipitation",
}

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
    "surface_pressure": "Pa",
    "precipitation": "mm/h",
}

# elapsed forecast duration in steps for the forecast - [0] refers to the first forecast step at t+1
# this should be a list of integers
ELAPSED_FORECAST_DURATION = list(range(0, 120, 1))
# Select specific start_times for the forecast. This is the start and end of
# a slice in xarray. The start_time is included, the end_time is excluded.
# This should be a list of two strings in the format "YYYY-MM-DDTHH:MM:SS"
# Should be handy to evaluate certain dates, e.g. for a case study of a storm
START_TIMES = ["2019-10-31T00:00:00", "2020-10-23T13:00:00"]  # Full year
# START_TIMES = ["2020-02-08T00:00:00", "2020-02-15T00:00:00"]  # Ciara/Sabine

# Select specific plot times for the forecast (will be used to create maps for all variables)
# This only affect chapter one with the plotting of the maps
# Map creation takes a lot of time so this is limited to a single time step
# Simply rerun these cells and chapter one for more time steps
PLOT_TIME = "2020-02-10T12:00:00"

# Define Thresholds for the ETS metric (Equitable Threat Score)
# These are calculated for wind and precipitation if available
# The score creates contingency tables for different thresholds
# The ETS is calculated for each threshold and the results are plotted
# The default thresholds are [0.1, 1, 5] for precipitation and [2.5, 5, 10] for wind
THRESHOLDS_PRECIPITATION = [0.1, 1, 5]  # mm/h
THRESHOLDS_WIND = [2.5, 5, 10]  # m/s

# Define the metrics to compute for the verification
# Some additional verifications will always be computed if the repsective vars
# are available in the data
METRICS = [
    "MAE",
    "RMSE",
    "MSE",
    "ME",
    "STDEV_ERR",
    "RelativeMAE",
    "RelativeRMSE",
    "PearsonR",
    "Wasserstein",
]

# This setting is relevant for the mapplots in chapter 1
# Higher levels of ZOOM will zoom in on the map, cropping the boundary
ZOOM = 2  # Halves the extent of the mapplot

# Map projection settings for plotting
# This is the projection of the ground truth data
PROJECTION = ccrs.RotatedPole(
    pole_longitude=190,
    pole_latitude=43,
    central_rotated_longitude=10,
)

# For some chapters a random seed is required to reproduce the results
RANDOM_SEED = 42

# The DPI used in all plots in the notebook, export to pdf will always be 300 DPI
DPI = 100

# Subsample the data for faster plotting, 10 refers to every 10th element
# This is used to create the histograms in chapter 2 (along space and time)
# and in chapter 3 for the energy spectra (along time)
# There is a trade-off between speed and accuracy, that each user has to find
SUBSAMPLE_HISTOGRAM = None

# Takes a long time, but if you see NaN in your output, you can set this to True
# This will check if there are any missing values in the data further below
CHECK_MISSING = True
# In this script missing data is allowed as observations often have missing values
# All time steps with missing values will be omitted from the verification
# - Scores/Xarray masked arrays are created and false values are omitted by default
# - For Scipy metrics, we need to convert to numpy arrays and change nan-policy to 'omit'.
# - Fore the wasserstein metric and the wind vector without internal nan-handling policy,
#  we need to remove the missing values for obs, ml, nwp before calculating the metric

# Font sizes for consistent plotting (different fig-sizes wil require different font sizes)
FONT_SIZES = {
    "axes": 13,  # Axis labels and titles
    "ticks": 13,  # Tick labels
    "legend": 13,  # Legend text
    "cbar": 13,  # Colorbar labels
    "suptitle": 15,  # Figure suptitle
    "title": 13,  # Axes titles
    "stats": 13,  # Statistics text in plots
}


# %%

# Create directories for plots and tables
Path("plots").mkdir(exist_ok=True)
Path("tables").mkdir(exist_ok=True)

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
plt.rcParams.update({
    "font.size": FONT_SIZES["axes"],
    "axes.titlesize": FONT_SIZES["axes"],
    "axes.labelsize": FONT_SIZES["axes"],
    "xtick.labelsize": FONT_SIZES["ticks"],
    "ytick.labelsize": FONT_SIZES["ticks"],
    "legend.fontsize": FONT_SIZES["legend"],
    "figure.titlesize": FONT_SIZES["suptitle"],
})

# Colorblind-friendly colormap for 2D plots
COLORMAP = "viridis"


def save_plot(fig, name, time=None, remove_title=True, dpi=300):
    """Helper function to save plots consistently

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
    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)

    # Remove titles if requested
    if remove_title:
        if hasattr(fig, "texts") and fig.texts:  # Check for suptitle
            fig.suptitle("")
        ax = fig.gca()
        if ax.get_title():
            ax.set_title("")

    pdf_path = plot_dir / f"observations_{safe_name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)


def export_table(df, name, caption=""):
    """Helper function to export tables consistently"""
    # Export to LaTeX with caption
    latex_str = df.to_latex(
        float_format="%.4f", caption=caption, label=f"tab:{name}"
    )
    with open(f"tables/observations_{name}.tex", "w") as f:
        f.write(latex_str)

    # Export to CSV
    df.to_csv(f"tables/observations_{name}.csv")


# %%
ds_nwp = xr.open_zarr(PATH_NWP)
ds_nwp = ds_nwp.sel(time=slice(*START_TIMES))
ds_nwp = ds_nwp[VARIABLES_NWP.keys()].rename(VARIABLES_NWP)
ds_nwp = ds_nwp.rename_dims({
    "lead_time": "elapsed_forecast_duration",
    "time": "start_time",
})
ds_nwp = ds_nwp.rename_vars({
    "lead_time": "elapsed_forecast_duration",
    "time": "start_time",
    "lon": "longitude",
    "lat": "latitude",
})
forecast_times = (
    ds_nwp.start_time.values[:, None] + ds_nwp.elapsed_forecast_duration.values
)
ds_nwp = ds_nwp.assign_coords(
    forecast_time=(
        ("start_time", "elapsed_forecast_duration"),
        forecast_times,
    )
)

# # Calculate hourly values by taking differences along elapsed_forecast_duration
ds_nwp["precipitation"] = ds_nwp.precipitation.diff(
    dim="elapsed_forecast_duration"
)
# The NWP data starts at elapsed forecast duration 0 = start_time
ds_nwp = ds_nwp.drop_isel(elapsed_forecast_duration=0).isel(
    elapsed_forecast_duration=ELAPSED_FORECAST_DURATION
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
ds_ml = xr.open_zarr(PATH_ML)
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
ds_ml = ds_ml.assign_coords({
    "latitude": ds_nwp.latitude,
    "longitude": ds_nwp.longitude,
})

ds_ml = ds_ml.isel(elapsed_forecast_duration=ELAPSED_FORECAST_DURATION)

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
    """Calculate u and v wind components from speed and direction."""
    ds = ds.copy()
    ds["wind_u_10m"] = ds[speed_var] * np.cos(np.radians(90 - ds[dir_var]))
    ds["wind_v_10m"] = ds[speed_var] * np.sin(np.radians(90 - ds[dir_var]))
    ds = ds.drop_vars([speed_var, dir_var])
    return ds


ds_obs = xr.open_zarr(PATH_OBS)
ds_obs = ds_obs[VARIABLES_OBS].rename_vars(OBS_VAR_MAPPING)
ds_obs = ds_obs.sel(time=np.unique(ds_ml.forecast_time.values.flatten()))
ds_obs = ds_obs.where(ds_obs != 32767, np.nan)
ds_obs = calculate_wind_components(ds_obs)
ds_obs["temperature_2m"] += 273.15  # Convert to Kelvin
ds_obs["surface_pressure"] *= 100  # Convert to Pa
ds_obs


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
    stations_with_missing = len([
        station
        for station in ds_obs.station
        if np.isnan(ds_obs.sel(station=station)).any()
    ])

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
        f"\nMissing Data Analysis: {stations_with_missing} out of {ds_obs.sizes['station']} stations affected"
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
    f"Number of time steps do not match: {ds_obs.sizes['time']} != {len(np.unique(ds_ml.forecast_time.values.flatten()))}"
)
assert ds_ml.sizes["start_time"] == ds_nwp.sizes["start_time"]

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
    plot_time = ds_ml.sel(start_time=PLOT_TIME).start_time


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


def plot_comparison_maps(ds_obs, ds_nwp, ds_ml, plot_time=None, variables=None):
    """
    Plot comparison between observations, NWP and ML data for each variable and forecast step.

    Args:
        ds_obs (xarray.Dataset): Observations dataset
        ds_nwp (xarray.Dataset): NWP forecast data
        ds_ml (xarray.Dataset): ML forecast data
        plot_time (str): Time for plot title
        variables (list): List of variables to plot. If None, plots all common variables
    """
    # Convert plot_time to pandas datetime if it's a string
    if isinstance(plot_time, str):
        plot_time = pd.to_datetime(plot_time)

    # Get common variables if not specified
    if variables is None:
        variables = list(set(ds_nwp.data_vars).intersection(ds_ml.data_vars))

    # Get number of forecast steps
    n_steps = len(ds_ml.elapsed_forecast_duration)

    for var in variables:
        # Create figure with n_steps rows and 3 columns
        fig = plt.figure(
            figsize=(15, 3 * n_steps),
            dpi=100,  # Increased height multiplier
        )
        axes = np.array([
            [
                plt.subplot(n_steps, 3, i * 3 + j + 1, projection=PROJECTION)
                for j in range(3)
            ]
            for i in range(n_steps)
        ])

        # Set extent for all subplots
        for ax_row in axes:
            for ax in ax_row:
                ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Initialize arrays for global min/max
        arrays_for_minmax = []

        # Collect data for min/max calculation across all timesteps
        for step_idx, step in enumerate(ds_ml.elapsed_forecast_duration):
            forecast_time = ds_ml.forecast_time.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )

            obs_data = ds_obs.sel(time=forecast_time)[var]
            nwp_data = ds_nwp.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )[var]
            ml_data = ds_ml.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )[var]

            arrays_for_minmax.extend([
                obs_data.values,
                nwp_data.values,
                ml_data.values,
            ])

        # Calculate global min/max
        vmin = min(np.nanmin(arr) for arr in arrays_for_minmax)
        vmax = max(np.nanmax(arr) for arr in arrays_for_minmax)

        # Plot for each forecast step
        for step_idx, step in enumerate(ds_ml.elapsed_forecast_duration):
            forecast_time = ds_ml.forecast_time.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )
            forecast_hours = int(step.values / 1e9 / 3600)  # Convert to hours

            # Plot observations
            obs_data = ds_obs.sel(time=forecast_time)[var]
            axes[step_idx, 0].scatter(
                ds_obs.longitude,
                ds_obs.latitude,
                c=obs_data,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )

            # Plot NWP data
            nwp_data = ds_nwp.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )[var]
            axes[step_idx, 1].pcolormesh(
                ds_nwp.longitude,
                ds_nwp.latitude,
                nwp_data,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                shading="auto",
                rasterized=True,
            )

            # Plot ML data
            ml_data = ds_ml.sel(
                start_time=plot_time, elapsed_forecast_duration=step
            )[var]
            im2 = axes[step_idx, 2].pcolormesh(
                ds_ml.longitude,
                ds_ml.latitude,
                ml_data,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                shading="auto",
                rasterized=True,
            )

            # Add titles for first row
            if step_idx == 0:
                axes[step_idx, 0].set_title(f"Observations\n+{forecast_hours}h")
                axes[step_idx, 1].set_title(f"NWP\n+{forecast_hours}h")
                axes[step_idx, 2].set_title(f"ML\n+{forecast_hours}h")
            else:
                axes[step_idx, 0].set_title(f"+{forecast_hours}h")
                axes[step_idx, 1].set_title(f"+{forecast_hours}h")
                axes[step_idx, 2].set_title(f"+{forecast_hours}h")

        # Add map features
        add_map_features(axes)

        # Adjust subplot spacing
        plt.subplots_adjust(
            top=0.9,
            bottom=0.05,
            hspace=0.15,
            wspace=0.03,
        )

        # Add colorbar with adjusted position
        cbar_ax = fig.add_axes([0.15, 0.0, 0.7, 0.02])
        plt.colorbar(
            im2,
            cax=cbar_ax,
            orientation="horizontal",
            label=f"[{VARIABLE_UNITS[var]}]",
        )

        # Set overall title with adjusted position
        plt.suptitle(
            f"{var} Comparison at {str(plot_time.dt.date.values)} - {str(plot_time.dt.hour.values)} UTC",
            y=0.95,
        )

        plt.show()
        save_plot(fig, f"comparison_{var}_multi_step", time=plot_time)
        plt.close()


# Call the function
# plot_comparison_maps(ds_obs, ds_nwp, ds_ml, plot_time=plot_time)


# %% [markdown]
def interpolate_to_obs(
    ds_model_1, ds_model_2, ds_obs, vars_plot, neighbors=None
):
    """Optimized function to interpolate model datasets to observation points using xarray/dask."""

    print("Starting optimized interpolation...")

    # Extract observation coordinates
    points_obs = np.column_stack([
        ds_obs.latitude.values,
        ds_obs.longitude.values,
    ])

    # Extract model coordinates from 2D lat/lon arrays
    model_lats = ds_model_1.latitude
    model_lons = ds_model_1.longitude

    points_model = np.column_stack([
        model_lats.values.ravel(),
        model_lons.values.ravel(),
    ])

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

        # Create interpolated datasets using weighted sum along neighbor dimension
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


ds_nwp_interp, ds_ml_interp = interpolate_to_obs(
    ds_nwp,
    ds_ml,
    ds_obs,
    VARIABLES_ML.values(),
)

with ProgressBar():
    print("Computing interpolated datasets...")
    ds_nwp_interp = ds_nwp_interp.compute()
    print("NWP interpolation done.")
    ds_ml_interp = ds_ml_interp.compute()
    print("ML interpolation done.")


# %%
# %% [markdown]
# ### Histograms of Interpolated Station Data


def plot_interpolated_histograms(
    ds_obs, ds_nwp_interp, ds_ml_interp, subsample=10
):
    """Plot histograms for interpolated station data comparing observations, NWP and ML predictions.

    Args:
        ds_obs: xarray Dataset containing observations
        ds_nwp_interp: xarray Dataset containing interpolated NWP predictions
        ds_ml_interp: xarray Dataset containing interpolated ML predictions
        subsample: int, subsampling factor for faster plotting
    """
    # Sample data for faster plotting
    ds_obs_sampled = ds_obs.isel(
        time=slice(None, None, subsample), station=slice(None, None, subsample)
    )
    ds_nwp_sampled = ds_nwp_interp.isel(
        start_time=slice(None, None, subsample),
        station=slice(None, None, subsample),
    )
    ds_ml_sampled = ds_ml_interp.isel(
        start_time=slice(None, None, subsample),
        station=slice(None, None, subsample),
    )

    for variable_name in VARIABLES_ML.values():
        if variable_name not in ds_obs:
            continue

        fig, ax = plt.subplots(figsize=(11, 7), dpi=DPI)

        # Convert to numpy arrays
        data_obs = ds_obs_sampled[variable_name].values.flatten()
        data_ml = ds_ml_sampled[variable_name].values.flatten()
        data_nwp = ds_nwp_sampled[variable_name].values.flatten()

        # Create histograms for observations
        ax.hist(
            data_obs,
            bins=300,
            density=True,
            color=COLORS["gt"],
            label="Observations",
            histtype="stepfilled",
            linewidth=0,
        )

        # Plot NWP interpolated data
        ax.hist(
            data_nwp,
            bins=300,
            alpha=0.8,
            density=True,
            color=COLORS["nwp"],
            label="NWP Interpolated",
            histtype="stepfilled",
            linewidth=0,
        )

        # Create histogram for ML interpolated data
        ax.hist(
            data_ml,
            bins=300,
            alpha=0.8,
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
        ax.set_xlabel(f"{units}")

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
            bbox=[0.72, 0.78, 0.25, 0.18],
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(13)

        for (row, col), cell in table._cells.items():
            cell.set_text_props(wrap=True)
            cell.set_facecolor("white")
            cell.set_alpha(0.9)
            cell.set_edgecolor("#D3D3D3")

            if col == -1:
                cell.set_width(0.15)
                cell.set_text_props(horizontalalignment="right")
            else:
                cell.set_width(0.12)
                cell.set_text_props(horizontalalignment="center")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        save_plot(fig, f"histogram_interpolated_{variable_name}")
        plt.close()


# Call the function after interpolation
plot_interpolated_histograms(
    ds_obs, ds_nwp_interp, ds_ml_interp, subsample=SUBSAMPLE_HISTOGRAM
)


# %% [markdown]
# Visualization of the interpolated model data
def plot_comparison_single_var_panel(
    ds_obs, ds_interp_1, ds_interp_2, var_plot, plot_time=None
):
    """Plot comparison between observations and interpolated model datasets across forecast steps."""
    if isinstance(plot_time, str):
        plot_time = pd.to_datetime(plot_time)

    n_steps = len(ds_interp_1.elapsed_forecast_duration)

    # Increased top margin for suptitle
    fig = plt.figure(figsize=(15, 3 * n_steps + 0.5), dpi=100)
    axes = np.array([
        [
            plt.subplot(n_steps, 3, i * 3 + j + 1, projection=PROJECTION)
            for j in range(3)
        ]
        for i in range(n_steps)
    ])

    # Set extent for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

    arrays_for_minmax = []
    for step_idx, step in enumerate(ds_interp_1.elapsed_forecast_duration):
        forecast_time = ds_interp_1.forecast_time.sel(
            start_time=plot_time, elapsed_forecast_duration=step
        )
        arrays_for_minmax.extend([
            ds_obs[var_plot].sel(time=forecast_time).values,
            ds_interp_1[var_plot]
            .sel(start_time=plot_time, elapsed_forecast_duration=step)
            .values,
            ds_interp_2[var_plot]
            .sel(start_time=plot_time, elapsed_forecast_duration=step)
            .values,
        ])

    vmin = min(np.nanmin(arr) for arr in arrays_for_minmax)
    vmax = max(np.nanmax(arr) for arr in arrays_for_minmax)

    for step_idx, step in enumerate(ds_interp_1.elapsed_forecast_duration):
        forecast_time = ds_interp_1.forecast_time.sel(
            start_time=plot_time, elapsed_forecast_duration=step
        )
        forecast_hours = int(step.values / 1e9 / 3600)

        for ax_idx, (ax, data, title) in enumerate(
            zip(
                axes[step_idx],
                [ds_obs, ds_interp_1, ds_interp_2],
                ["Observations", "NWP", "ML"],
            )
        ):
            if ax_idx == 0:
                scatter = ax.scatter(
                    ds_obs.longitude,
                    ds_obs.latitude,
                    c=data[var_plot].sel(time=forecast_time),
                    cmap=COLORMAP,
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
                    cmap=COLORMAP,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    rasterized=True,
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
        hspace=0.15,
        wspace=0.03,
    )

    # Add colorbar with adjusted position
    cbar_ax = fig.add_axes([0.15, 0.0, 0.7, 0.02])
    plt.colorbar(
        scatter,
        cax=cbar_ax,
        orientation="horizontal",
        label=VARIABLE_UNITS[var_plot],
    )

    # Adjusted suptitle position
    plt.suptitle(
        f"{var_plot} Comparison at {str(plot_time.dt.date.values)} - {str(plot_time.dt.hour.values)} UTC",
        y=0.95,  # Higher position
        fontsize=14,
    )
    return fig


# Call the function for each variable
# for var in VARIABLES_ML.values():
#     fig = plot_comparison_single_var_panel(
#         ds_obs,
#         ds_nwp_interp,
#         ds_ml_interp,
#         var,
#         plot_time=plot_time,
#     )
#     plt.show()
#     save_plot(fig, f"interpolated_comparison_panel_{var}", time=plot_time)


# %% [markdown]
# For the verification with scores the data must contain lat and lon as xarray dimensions.
# Here we use masked arrays for this purpose. There might be a better way to do this.


def calculate_metrics_by_efd(
    ds_obs,
    ds_nwp=None,
    ds_ml=None,
    metrics_to_compute=None,
    subsample_points=1e7,
    prefix="metrics",
):
    """Calculate metrics for each Elapsed Forecast Duration for station data using xarray."""
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
        print(
            f"\nCalculating metrics for elapsed forecast duration: {lt_hours.item():.1f}h"
        )

        ds_ml_lead = ds_ml.sel(elapsed_forecast_duration=efd)
        if ds_nwp is not None:
            ds_nwp_lead = ds_nwp.sel(elapsed_forecast_duration=efd)
            ds_nwp_lead["start_time"] = ds_nwp_lead.forecast_time
            ds_nwp_lead = ds_nwp_lead.rename_dims({
                "start_time": "time"
            }).rename_vars({"start_time": "time"})

        ds_ml_lead["start_time"] = ds_ml_lead.forecast_time
        ds_ml_lead = ds_ml_lead.rename_dims({"start_time": "time"}).rename_vars({
            "start_time": "time"
        })

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

            if ds_nwp is not None and var in ds_nwp:
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
                # Use the valid_mask directly instead of checking for NaN values again
                pred_vals = y_pred_ml.values[valid_mask.values]
                true_vals = y_true.values[valid_mask.values]
                metrics_dict[var]["Wasserstein ML"] = wasserstein_distance(
                    pred_vals, true_vals
                )

            # Calculate NWP metrics if available
            if ds_nwp is not None and var in ds_nwp:
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
                        ((y_pred_nwp - y_true) ** 2 / (y_true**2 + 1e-6)).mean()
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
    elapsed_forecast_durations_hours_float = [
        x.item() for x in elapsed_forecast_durations_hours
    ]
    combined_df = pd.DataFrame(
        combined_metrics, index=elapsed_forecast_durations_hours_float
    )
    combined_df.index.name = "Forecast Hours"

    return metrics_by_efd, combined_df


# %%

metrics_by_efd, combined_metrics = calculate_metrics_by_efd(
    ds_obs=ds_obs,
    ds_nwp=ds_nwp_interp,
    ds_ml=ds_ml_interp,
    metrics_to_compute=METRICS,
)
export_table(combined_metrics, "combined_metrics")


# %%
# Plot evolution of a specific metric over elapsed forecast duration
elapsed_forecast_durations = list(metrics_by_efd.keys())
metrics_to_compute = METRICS
for variable in VARIABLES_ML.values():
    for metric in metrics_to_compute:
        try:
            # Skip if any scores are missing
            ml_scores = [
                df.loc[variable, f"{metric} ML"]
                for df in metrics_by_efd.values()
            ]
            nwp_scores = [
                df.loc[variable, f"{metric} NWP"]
                for df in metrics_by_efd.values()
            ]

            # Convert elapsed forecast durations from hours to timedelta
            hours = [
                x / np.timedelta64(1, "h")
                for x in ds_ml.elapsed_forecast_duration.values
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
            if not all(pd.isna(nwp_scores)):
                ax.plot(
                    hours,
                    nwp_scores,
                    label="NWP",
                    color=COLORS["nwp"],
                    linestyle=LINE_STYLES["nwp"][0],
                    marker=LINE_STYLES["nwp"][1],
                )

            ax.set_xlabel("Elapsed Forecast Duration [h]")
            ax.set_ylabel(f"{metric} [{VARIABLE_UNITS[variable]}]")
            ax.set_title(f"{metric} Evolution for {variable}")
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.show()
            save_plot(fig, f"{metric}_{variable}_evolution")
            plt.close()

        except (KeyError, ValueError) as e:
            print(f"Skipping {metric} for {variable}: {str(e)}")
            continue


# %%
def calculate_meteoswiss_metrics(
    ds_obs, ds_ml_interp, ds_nwp_interp=None, subsample=None
):
    """Calculate MeteoSwiss verification metrics for station data.

    Args:
        ds_obs: xarray Dataset containing observations
        ds_ml_interp: xarray Dataset containing interpolated ML predictions
        ds_nwp_interp: xarray Dataset containing interpolated NWP predictions
        subsample: int, subsampling factor for faster processing
    """

    # Sample data if requested
    if subsample:
        ds_obs_sample = ds_obs.isel(
            time=slice(None, None, subsample),
            station=slice(None, None, subsample),
        )
        ds_ml_sample = ds_ml_interp.isel(
            start_time=slice(None, None, subsample),
            station=slice(None, None, subsample),
        )
        if ds_nwp_interp is not None:
            ds_nwp_sample = ds_nwp_interp.isel(
                start_time=slice(None, None, subsample),
                station=slice(None, None, subsample),
            )
    else:
        ds_obs_sample = ds_obs
        ds_ml_sample = ds_ml_interp
        ds_nwp_sample = ds_nwp_interp if ds_nwp_interp is not None else None

    # Initialize results dictionary
    metrics = {}

    # Get elapsed forecast durations
    elapsed_forecast_durations = ds_ml_sample.elapsed_forecast_duration

    for efd in elapsed_forecast_durations:
        efd_key = float(efd / np.timedelta64(1, "h"))
        metrics[efd_key] = {}

        # Get ML data for this forecast lead time
        ds_ml_lead = ds_ml_sample.sel(elapsed_forecast_duration=efd)
        # Get data as xarray DataArrays

        # Get NWP data if available
        if ds_nwp_sample is not None:
            ds_nwp_lead = ds_nwp_sample.sel(elapsed_forecast_duration=efd)

        # Get observation times corresponding to these forecast times
        forecast_times = ds_ml_lead.forecast_time.values
        ds_obs_lead = ds_obs_sample.sel(time=forecast_times)

        # Process each variable
        for var in ds_obs_lead.data_vars:
            # Skip if variable not in ML dataset
            if var not in ds_ml_lead:
                continue
            metrics[efd_key][var] = {}

            y_true = ds_obs_lead[var]
            y_pred_ml = ds_ml_lead[var]

            # Create masks for each dataset
            mask_true = xr.where(~np.isnan(y_true), True, False)
            mask_ml = xr.where(~np.isnan(y_pred_ml), True, False)

            if ds_nwp is not None and var in ds_nwp:
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

            if len(y_true) == 0:
                print(f"No valid data for {var} at forecast hour {efd_key}")
                continue

            # Calculate threshold-based metrics for precipitation and wind
            if var == "precipitation" and len(y_true) > 0:
                for threshold in THRESHOLDS_PRECIPITATION:
                    try:
                        event_operator = TEO(default_event_threshold=threshold)
                        ml_contingency = (
                            event_operator.make_contingency_manager(
                                y_pred_ml, y_true
                            )
                        )
                        metrics[efd_key][var][f"FBI_{threshold}mm_ML"] = (
                            ml_contingency.frequency_bias()
                        )
                        metrics[efd_key][var][f"ETS_{threshold}mm_ML"] = (
                            ml_contingency.equitable_threat_score()
                        )
                        print(
                            "DEBUG",
                            metrics[efd_key][var][f"FBI_{threshold}mm_ML"],
                        )
                    except Exception as e:
                        print(
                            f"Error calculating threshold metrics for {var} at {threshold}mm: {e}"
                        )

                    if y_pred_nwp is not None:
                        try:
                            nwp_contingency = (
                                event_operator.make_contingency_manager(
                                    y_pred_nwp, y_true
                                )
                            )
                            metrics[efd_key][var][f"FBI_{threshold}mm_NWP"] = (
                                nwp_contingency.frequency_bias()
                            )
                            metrics[efd_key][var][f"ETS_{threshold}mm_NWP"] = (
                                nwp_contingency.equitable_threat_score()
                            )
                        except Exception as e:
                            print(
                                f"Error calculating threshold metrics for NWP {var} at {threshold}mm: {e}"
                            )

            if var in ["wind_u_10m", "wind_v_10m"] and len(y_true) > 0:
                for threshold in THRESHOLDS_WIND:
                    try:
                        event_operator = TEO(default_event_threshold=threshold)
                        ml_contingency = (
                            event_operator.make_contingency_manager(
                                y_pred_ml, y_true
                            )
                        )
                        metrics[efd_key][var][f"FBI_{threshold}ms_ML"] = (
                            ml_contingency.frequency_bias().values
                        )
                        metrics[efd_key][var][f"ETS_{threshold}ms_ML"] = (
                            ml_contingency.equitable_threat_score().values
                        )
                    except Exception as e:
                        print(
                            f"Error calculating threshold metrics for {var} at {threshold}m/s: {e}"
                        )

                    if y_pred_nwp is not None:
                        try:
                            nwp_contingency = (
                                event_operator.make_contingency_manager(
                                    y_pred_nwp, y_true
                                )
                            )
                            metrics[efd_key][var][f"FBI_{threshold}ms_NWP"] = (
                                nwp_contingency.frequency_bias().values
                            )
                            metrics[efd_key][var][f"ETS_{threshold}ms_NWP"] = (
                                nwp_contingency.equitable_threat_score().values
                            )
                        except Exception as e:
                            print(
                                f"Error calculating threshold metrics for NWP {var} at {threshold}m/s: {e}"
                            )

    # Convert to DataFrame for easier handling
    metrics_by_var = {}
    for efd_key in metrics:
        metrics_by_var[efd_key] = {}
        for var in metrics[efd_key]:
            metrics_by_var[efd_key][var] = pd.DataFrame(
                metrics[efd_key][var], index=[0]
            ).T

    return metrics_by_var


# %%
def plot_meteoswiss_metrics_evolution(metrics_by_var, var_name, metric_prefix):
    """Plot evolution of MeteoSwiss metrics over forecast time with three fixed viridis colors."""
    forecast_hours = sorted(list(metrics_by_var.keys()))

    # Get three fixed colors from viridis (start, middle, end)
    colors = [plt.cm.viridis(x) for x in [0, 0.5, 0.99]]

    # Find all metrics matching the prefix
    all_metrics = []
    for hour in forecast_hours:
        if var_name in metrics_by_var[hour]:
            for metric in metrics_by_var[hour][var_name].index:
                if metric_prefix in metric:
                    all_metrics.append(metric)

    # Sort metrics by threshold value numerically
    def get_threshold(metric):
        return float(metric.split("_")[1].replace("mm", "").replace("ms", ""))

    all_metrics = sorted(list(set(all_metrics)), key=get_threshold)

    if not all_metrics:
        print(
            f"No metrics found with prefix {metric_prefix} for variable {var_name}"
        )
        return

    # Group by model (ML/NWP)
    ml_metrics = [m for m in all_metrics if m.endswith("ML")]
    nwp_metrics = [m for m in all_metrics if m.endswith("NWP")]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Plot ML metrics
    for i, metric in enumerate(ml_metrics):
        values = []
        for hour in forecast_hours:
            if (
                var_name in metrics_by_var[hour]
                and metric in metrics_by_var[hour][var_name].index
            ):
                values.append(metrics_by_var[hour][var_name].loc[metric, 0])
            else:
                values.append(np.nan)

        threshold = metric.split("_")[1].replace("mm", "").replace("ms", "")
        ax.plot(
            forecast_hours,
            values,
            linestyle=LINE_STYLES["ml"][0],
            marker=LINE_STYLES["ml"][1],
            color=colors[i],
            label=f"ML {threshold}{'mm' if var_name == 'precipitation' else 'm/s'}",
        )

    # Plot NWP metrics
    for i, metric in enumerate(nwp_metrics):
        values = []
        for hour in forecast_hours:
            if (
                var_name in metrics_by_var[hour]
                and metric in metrics_by_var[hour][var_name].index
            ):
                values.append(metrics_by_var[hour][var_name].loc[metric, 0])
            else:
                values.append(np.nan)

        threshold = metric.split("_")[1].replace("mm", "").replace("ms", "")
        ax.plot(
            forecast_hours,
            values,
            linestyle=LINE_STYLES["nwp"][0],
            marker=LINE_STYLES["nwp"][1],
            color=colors[i],
            label=f"NWP {threshold}{'mm' if var_name == 'precipitation' else 'm/s'}",
        )

    ax.set_xlabel("Forecast Lead Time (hours)")
    ax.set_ylabel(f"{metric_prefix}")
    ax.set_title(f"{metric_prefix} for {var_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
    save_plot(fig, f"{metric_prefix.lower()}_{var_name}_evolution")


# %%
# Calculate MeteoSwiss metrics
def reshape_meteoswiss_metrics(metrics_dict):
    """Reshape MeteoSwiss metrics into a single DataFrame with proper column structure"""
    # Create a list to store all data
    data = []

    # Iterate through each forecast hour
    for hour in metrics_dict:
        row_data = {"forecast_hour": hour}
        # Iterate through each variable
        for var in metrics_dict[hour]:
            # Get metrics for this variable
            var_metrics = metrics_dict[hour][var]
            # Add each metric to the row data
            for metric_name in var_metrics.index:
                row_data[f"{var}_{metric_name}"] = var_metrics.loc[
                    metric_name, 0
                ]
        data.append(row_data)

    # Create DataFrame from collected data
    df = pd.DataFrame(data)
    df.set_index("forecast_hour", inplace=True)
    return df


# Use it after calculate_meteoswiss_metrics
print("Calculating MeteoSwiss metrics...")
meteoswiss_metrics = calculate_meteoswiss_metrics(
    ds_obs, ds_ml_interp, ds_nwp_interp, subsample=SUBSAMPLE_HISTOGRAM
)
metrics_df = reshape_meteoswiss_metrics(meteoswiss_metrics)

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


# %%
def wind_vector_rmse(u_true, v_true, u_pred, v_pred):
    """Calculate RMSE based on wind vector differences."""
    u_diff = u_true - u_pred
    v_diff = v_true - v_pred
    vector_diff = np.sqrt(u_diff**2 + v_diff**2)
    return float(np.sqrt(np.mean(vector_diff**2)))


def calculate_wind_vector_metrics(
    ds_obs, ds_ml_interp, ds_nwp_interp=None, subsample=None
):
    """Calculate wind vector metrics for station data.

    Args:
        ds_obs: xarray Dataset containing observations
        ds_ml_interp: xarray Dataset containing interpolated ML predictions
        ds_nwp_interp: xarray Dataset containing interpolated NWP predictions
        subsample: int, subsampling factor for faster processing
    """
    # Sample data if requested
    if subsample:
        ds_obs_sample = ds_obs.isel(
            time=slice(None, None, subsample),
            station=slice(None, None, subsample),
        )
        ds_ml_sample = ds_ml_interp.isel(
            start_time=slice(None, None, subsample),
            station=slice(None, None, subsample),
        )
        if ds_nwp_interp is not None:
            ds_nwp_sample = ds_nwp_interp.isel(
                start_time=slice(None, None, subsample),
                station=slice(None, None, subsample),
            )
    else:
        ds_obs_sample = ds_obs
        ds_ml_sample = ds_ml_interp
        ds_nwp_sample = ds_nwp_interp if ds_nwp_interp is not None else None

    # Check if wind components exist
    if "wind_u_10m" not in ds_obs_sample or "wind_v_10m" not in ds_obs_sample:
        print("Wind components not found in observation dataset")
        return None

    if "wind_u_10m" not in ds_ml_sample or "wind_v_10m" not in ds_ml_sample:
        print("Wind components not found in ML dataset")
        return None

    # Get elapsed forecast durations
    elapsed_forecast_durations = ds_ml_sample.elapsed_forecast_duration
    forecast_hours = [
        float(efd / np.timedelta64(1, "h"))
        for efd in elapsed_forecast_durations
    ]

    # Initialize lists to store RMSE values over time
    ml_rmse_over_time = []
    nwp_rmse_over_time = []

    for efd in elapsed_forecast_durations:
        # Get ML data for this forecast lead time
        ds_ml_lead = ds_ml_sample.sel(elapsed_forecast_duration=efd)

        # Get observation times corresponding to these forecast times
        forecast_times = ds_ml_lead.forecast_time.values
        ds_obs_lead = ds_obs_sample.sel(time=forecast_times)

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
                u_true[valid_mask],
                v_true[valid_mask],
                u_ml[valid_mask],
                v_ml[valid_mask],
            )
            ml_rmse_over_time.append(wind_rmse_ml)
        else:
            ml_rmse_over_time.append(np.nan)

        # Calculate NWP RMSE if available
        if (
            ds_nwp_sample is not None
            and "wind_u_10m" in ds_nwp_sample
            and "wind_v_10m" in ds_nwp_sample
        ):
            ds_nwp_lead = ds_nwp_sample.sel(elapsed_forecast_duration=efd)
            u_nwp = ds_nwp_lead["wind_u_10m"].values
            v_nwp = ds_nwp_lead["wind_v_10m"].values

            valid_mask_nwp = valid_mask & ~np.isnan(u_nwp) & ~np.isnan(v_nwp)

            if np.any(valid_mask_nwp):
                wind_rmse_nwp = wind_vector_rmse(
                    u_true[valid_mask_nwp],
                    v_true[valid_mask_nwp],
                    u_nwp[valid_mask_nwp],
                    v_nwp[valid_mask_nwp],
                )
                nwp_rmse_over_time.append(wind_rmse_nwp)
            else:
                nwp_rmse_over_time.append(np.nan)
        else:
            nwp_rmse_over_time.append(np.nan)

    # Create result DataFrame
    time_series_df = pd.DataFrame({
        "Elapsed Forecast Duration": forecast_hours,
        "ML Vector RMSE": ml_rmse_over_time,
        "NWP Vector RMSE": nwp_rmse_over_time,
    })

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
        time_series_df["Elapsed Forecast Duration"],
        time_series_df["ML Vector RMSE"],
        color=COLORS["ml"],
        linestyle=LINE_STYLES["ml"][0],
        marker=LINE_STYLES["ml"][1],
        label="ML",
    )

    # Plot NWP RMSE if available
    if not all(np.isnan(time_series_df["NWP Vector RMSE"])):
        ax.plot(
            time_series_df["Elapsed Forecast Duration"],
            time_series_df["NWP Vector RMSE"],
            color=COLORS["nwp"],
            linestyle=LINE_STYLES["nwp"][0],
            marker=LINE_STYLES["nwp"][1],
            label="NWP",
        )

    ax.set_xlabel("Forecast Lead Time (hours)")
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
        ds_obs, ds_ml_interp, ds_nwp_interp, subsample=SUBSAMPLE_HISTOGRAM
    )
    plot_wind_vector_rmse(wind_timeseries)
    export_table(
        wind_timeseries,
        "wind_vector_metrics_timeseries",
        caption="Wind vector RMSE over forecast lead time",
    )
else:
    print("Wind components not found in the data")
# %%
