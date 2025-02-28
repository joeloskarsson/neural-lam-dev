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
from scipy.interpolate import RBFInterpolator
from scipy.stats import wasserstein_distance
from scores.continuous import (
    mae,
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
ELAPSED_FORECAST_DURATION = list(range(0, 120, 24))
# Select specific start_times for the forecast. This is the start and end of
# a slice in xarray. The start_time is included, the end_time is excluded.
# This should be a list of two strings in the format "YYYY-MM-DDTHH:MM:SS"
# Should be handy to evaluate certain dates, e.g. for a case study of a storm
START_TIMES = ["2020-02-13T00:00:00", "2020-02-15T00:00:00"]
# Select specific plot times for the forecast (will be used to create maps for all variables)
# This only affect chapter one with the plotting of the maps
# Map creation takes a lot of time so this is limited to a single time step
# Simply rerun these cells and chapter one for more time steps
PLOT_TIME = "2020-02-13T00:00:00"

# Define Thresholds for the ETS metric (Equitable Threat Score)
# These are calculated for wind and precipitation if available
# The score creates contingency tables for different thresholds
# The ETS is calculated for each threshold and the results are plotted
# The default thresholds are [0.1, 1, 5] for precipitation and [2.5, 5, 10] for wind
THRESHOLDS_PRECIPITATION = [0.1, 1, 5]  # mm/h
THRESHOLDS_WIND = [2.5, 5, 10]  # m/s

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
SUBSAMPLE_HISTOGRAM = 10

# Subsample the data for FSS threshold calculation, 1e7 refers to the number of elements
# This is not critical, as it is only used to calculate the 90% threshold
# for the FSS based on the ground truth data
SUBSAMPLE_FSS_THRESHOLD = 1e7

# Takes a long time, but if you see NaN in your output, you can set this to True
# This will check if there are any missing values in the data further below
CHECK_MISSING = False

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

    pdf_path = plot_dir / f"{safe_name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)


def export_table(df, name, caption=""):
    """Helper function to export tables consistently"""
    # Export to LaTeX with caption
    latex_str = df.to_latex(
        float_format="%.4f", caption=caption, label=f"tab:{name}"
    )
    with open(f"tables/{name}.tex", "w") as f:
        f.write(latex_str)

    # Export to CSV
    df.to_csv(f"tables/{name}.csv")


# %%
ds_nwp = xr.open_zarr(PATH_NWP)
ds_nwp = ds_nwp.sel(time=slice(*START_TIMES))
# The NWP data starts at elapsed forecast duration 0 = start_time
ds_nwp = ds_nwp.drop_isel(lead_time=0).isel(lead_time=ELAPSED_FORECAST_DURATION)
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

# Get the first timestep of precipitation (already hourly)
precip_first = ds_nwp.precipitation.isel(elapsed_forecast_duration=0)
# Calculate hourly values by taking differences along elapsed_forecast_duration
precip_hourly = ds_nwp.precipitation.diff(dim="elapsed_forecast_duration")

# Combine first timestep with hourly differences
precip_combined = xr.concat(
    [precip_first.expand_dims("elapsed_forecast_duration"), precip_hourly],
    dim="elapsed_forecast_duration",
)
# Replace the accumulated precipitation with hourly values

ds_nwp["precipitation"] = precip_combined
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
# Check number of NaN values per variable
total_elements = ds_obs.sizes["time"] * ds_obs.sizes["station"]
nan_stats = {}

for var in ds_obs.data_vars:
    n_nans = np.isnan(ds_obs[var]).sum().values
    percent_nans = (n_nans / total_elements) * 100
    nan_stats[var] = {
        "total_nans": n_nans,
        "percent_nans": f"{percent_nans:.2f}%",
    }

# Create a formatted table
df_nans = pd.DataFrame.from_dict(nan_stats, orient="index")
print("\nNaN Statistics per Variable:")
print("----------------------------")
print(df_nans.to_string())


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
            im0 = axes[step_idx, 0].scatter(
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
            im1 = axes[step_idx, 1].pcolormesh(
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
plot_comparison_maps(ds_obs, ds_nwp, ds_ml, plot_time=plot_time)


# %% [markdown]
def interpolate_to_obs(
    ds_model_1, ds_model_2, ds_obs, vars_plot, neighbors=None
):
    """Interpolate both model datasets to observation points."""
    # Extract observation coordinates
    lats_obs = ds_obs.latitude.values
    lons_obs = ds_obs.longitude.values
    points_obs = np.column_stack((lats_obs, lons_obs))

    # Extract model coordinates (assuming same grid for both models)
    lats_model = ds_model_1.latitude.values
    lons_model = ds_model_1.longitude.values
    points_model = np.column_stack((lats_model.flatten(), lons_model.flatten()))

    interpolated_data_1 = {}
    interpolated_data_2 = {}

    for var in vars_plot:
        data_1 = []
        data_2 = []

        for t in ds_model_1.start_time:
            for f in ds_model_1.elapsed_forecast_duration:
                # Extract 2D fields for current time and forecast step
                field_1 = (
                    ds_model_1[var]
                    .sel(start_time=t, elapsed_forecast_duration=f)
                    .values
                )
                field_2 = (
                    ds_model_2[var]
                    .sel(start_time=t, elapsed_forecast_duration=f)
                    .values
                )

                # Model 1 interpolation
                data_slice_1 = field_1.flatten()
                valid_mask_1 = ~np.isnan(data_slice_1)
                if np.any(valid_mask_1):
                    rbf_1 = RBFInterpolator(
                        points_model[valid_mask_1],
                        data_slice_1[valid_mask_1],
                        neighbors=neighbors,
                        kernel="linear",
                    )
                    stations_1 = rbf_1(points_obs)
                else:
                    stations_1 = np.full(len(points_obs), np.nan)

                # Model 2 interpolation
                data_slice_2 = field_2.flatten()
                valid_mask_2 = ~np.isnan(data_slice_2)
                if np.any(valid_mask_2):
                    rbf_2 = RBFInterpolator(
                        points_model[valid_mask_2],
                        data_slice_2[valid_mask_2],
                        neighbors=neighbors,
                        kernel="linear",
                    )
                    stations_2 = rbf_2(points_obs)
                else:
                    stations_2 = np.full(len(points_obs), np.nan)

                data_1.append(stations_1)
                data_2.append(stations_2)

        # Reshape data to (start_time, elapsed_forecast_duration, station)
        data_1 = np.array(data_1).reshape(
            len(ds_model_1.start_time),
            len(ds_model_1.elapsed_forecast_duration),
            len(ds_obs.station),
        )
        data_2 = np.array(data_2).reshape(
            len(ds_model_2.start_time),
            len(ds_model_2.elapsed_forecast_duration),
            len(ds_obs.station),
        )

        # Create DataArrays with proper dimensions
        interpolated_data_1[var] = xr.DataArray(
            data_1,
            dims=["start_time", "elapsed_forecast_duration", "station"],
            coords={
                "start_time": ds_model_1.start_time,
                "elapsed_forecast_duration": ds_model_1.elapsed_forecast_duration,
                "station": ds_obs.station,
                "forecast_time": ds_model_1.forecast_time,
            },
        )
        interpolated_data_2[var] = xr.DataArray(
            data_2,
            dims=["start_time", "elapsed_forecast_duration", "station"],
            coords={
                "start_time": ds_model_2.start_time,
                "elapsed_forecast_duration": ds_model_2.elapsed_forecast_duration,
                "station": ds_obs.station,
                "forecast_time": ds_model_2.forecast_time,
            },
        )

    return xr.Dataset(interpolated_data_1), xr.Dataset(interpolated_data_2)


# Use the function
ds_nwp_interp, ds_ml_interp = interpolate_to_obs(
    ds_nwp, ds_ml, ds_obs, VARIABLES_ML.values(), neighbors=10
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
for var in VARIABLES_ML.values():
    fig = plot_comparison_single_var_panel(
        ds_obs,
        ds_nwp_interp,
        ds_ml_interp,
        var,
        plot_time=plot_time,
    )
    plt.show()
    save_plot(fig, f"interpolated_comparison_panel_{var}", time=plot_time)
    plt.close()


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
        metrics_to_compute = [
            "MAE",
            "RMSE",
            "MSE",
            "RelativeMAE",
            "RelativeRMSE",
            "PearsonR",
            "Wasserstein",
        ]

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

            # Apply masks to data
            y_true = y_true.where(valid_mask)
            y_pred_ml = y_pred_ml.where(valid_mask)
            if ds_nwp is not None and var in ds_nwp:
                y_pred_nwp = y_pred_nwp.where(valid_mask)

            # Log the percentage of valid data points
            total_points = valid_mask.size
            valid_points = valid_mask.sum().values
            print(
                f"Valid data points: {valid_points}/{total_points} ({100 * valid_points / total_points:.2f}%)"
            )

            metrics_dict[var] = {}

            # Calculate ML metrics using xarray
            if "MAE" in metrics_to_compute:
                metrics_dict[var]["MAE ML"] = mae(y_true, y_pred_ml).values
            if "RMSE" in metrics_to_compute:
                metrics_dict[var]["RMSE ML"] = rmse(y_true, y_pred_ml).values
            if "MSE" in metrics_to_compute:
                metrics_dict[var]["MSE ML"] = mse(y_true, y_pred_ml).values
            if "RelativeMAE" in metrics_to_compute:
                rel_mae = (
                    abs(y_true - y_pred_ml) / (abs(y_true) + 1e-6)
                ).mean()
                metrics_dict[var]["Relative MAE ML"] = rel_mae.values
            if "RelativeRMSE" in metrics_to_compute:
                rel_rmse = np.sqrt(
                    ((y_true - y_pred_ml) ** 2 / (y_true**2 + 1e-6)).mean()
                )
                metrics_dict[var]["Relative RMSE ML"] = rel_rmse.values
            if "PearsonR" in metrics_to_compute:
                metrics_dict[var]["Pearson R ML"] = pearsonr(
                    y_true, y_pred_ml
                ).values
            if "Wasserstein" in metrics_to_compute:
                # For Wasserstein, we need to convert to numpy arrays
                true_vals = y_true.values[~np.isnan(y_true.values)]
                pred_vals = y_pred_ml.values[~np.isnan(y_pred_ml.values)]
                metrics_dict[var]["Wasserstein ML"] = wasserstein_distance(
                    true_vals, pred_vals
                )

            # Calculate NWP metrics if available
            if ds_nwp is not None and var in ds_nwp:
                if "MAE" in metrics_to_compute:
                    metrics_dict[var]["MAE NWP"] = mae(
                        y_true, y_pred_nwp
                    ).values
                if "RMSE" in metrics_to_compute:
                    metrics_dict[var]["RMSE NWP"] = rmse(
                        y_true, y_pred_nwp
                    ).values
                if "MSE" in metrics_to_compute:
                    metrics_dict[var]["MSE NWP"] = mse(
                        y_true, y_pred_nwp
                    ).values
                if "RelativeMAE" in metrics_to_compute:
                    rel_mae = (
                        abs(y_true - y_pred_nwp) / (abs(y_true) + 1e-6)
                    ).mean()
                    metrics_dict[var]["Relative MAE NWP"] = rel_mae.values
                if "RelativeRMSE" in metrics_to_compute:
                    rel_rmse = np.sqrt(
                        ((y_true - y_pred_nwp) ** 2 / (y_true**2 + 1e-6)).mean()
                    )
                    metrics_dict[var]["Relative RMSE NWP"] = rel_rmse.values
                if "PearsonR" in metrics_to_compute:
                    metrics_dict[var]["Pearson R NWP"] = pearsonr(
                        y_true, y_pred_nwp
                    ).values
                if "Wasserstein" in metrics_to_compute:
                    # For Wasserstein, we need to convert to numpy arrays
                    nwp_vals = y_pred_nwp.values[~np.isnan(y_pred_nwp.values)]
                    metrics_dict[var]["Wasserstein NWP"] = wasserstein_distance(
                        true_vals, nwp_vals
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

    # Export tables
    export_table(
        combined_df,
        f"observations_{prefix}_combined",
        caption="Combined verification metrics for all forecast hours",
    )

    return metrics_by_efd, combined_df


metrics_by_efd, combined_metrics = calculate_metrics_by_efd(
    ds_obs=ds_obs,
    ds_nwp=ds_nwp_interp,
    ds_ml=ds_ml_interp,
    metrics_to_compute=[
        "MAE",
        "RMSE",
        "MSE",
        "RelativeMAE",
        "RelativeRMSE",
        "PearsonR",
        "Wasserstein",
    ],
)

# %%
