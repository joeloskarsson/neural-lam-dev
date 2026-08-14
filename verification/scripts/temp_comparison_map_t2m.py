"""Temporary script: 3-col physical field comparison map for temperature_2m."""

# Standard library
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
    subset_dataset_for_debug,
)
from plot_styles import DPI, SCALE_MAPS_COSMO, apply_style

# ── Config ────────────────────────────────────────────────────────────────────
PATH_GROUND_TRUTH = "cosmo.datastore.zarr"
PATH_NWP = "cosmo_e_forecast.zarr"
PATH_ML = "preds_7_19_margin_interior_lr_0001_ar_12.zarr"

ELAPSED_FORECAST_DURATION = [0, 23, 47, 71, 95, 119]
ELAPSED_FORECAST_DURATION_PLOT = [1, 3, 5]

START_TIMES = ["2019-10-31T00:00:00", "2020-10-23T13:00:00"]
PLOT_TIME = "2020-02-07T00:00:00"
X = [None, None]
Y = [None, None]

PROJECTION = ccrs.RotatedPole(
    pole_longitude=190,
    pole_latitude=43,
    central_rotated_longitude=10,
)

VARIABLES_GROUND_TRUTH = {"T_2M": "temperature_2m"}
VARIABLES_ML = VARIABLES_GROUND_TRUTH
VARIABLES_NWP = {"temperature_2m": "temperature_2m"}
VARIABLE_UNITS = {"temperature_2m": "K"}

SUBSAMPLE_FRACTION = 0.2
PRECOMPUTE_DATA = True
RANDOM_SEED = 42

DEBUG_MODE, DEBUG_FRACTION, DEBUG_MIN_SIZE = parse_debug_runtime_args(
    "Temp 3-col comparison map for t2m."
)

Path("verification/cosmo/case_study").mkdir(parents=True, exist_ok=True)

apply_style()

# ── Data loading ──────────────────────────────────────────────────────────────
ds_ml = xr.open_zarr(PATH_ML, decode_timedelta=True)
ds_ml = subset_dataset_for_debug(
    ds_ml,
    label="ML zarr",
    dims_to_trim=("x", "y"),
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
    forecast_time=(("start_time", "elapsed_forecast_duration"), forecast_times)
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
ds_ml = ds_ml.isel(elapsed_forecast_duration=ELAPSED_FORECAST_DURATION)

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
_dims_to_drop = [
    d
    for d in ["forcing_feature", "static_feature", "split_part"]
    if d in ds_gt.dims
]
ds_gt = ds_gt.sel(split_name="test").drop_dims(_dims_to_drop)
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
    ],
    errors="ignore",
)
ds_gt = ds_gt.transpose("time", "x", "y")
ds_gt = ds_gt[["time", "x", "y", *VARIABLES_GROUND_TRUTH.values()]]
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

zarr_path = (
    "/capstor/store/cscs/swissai/a122/sadamov/cosmo_e_forecast_fixed.zarr"
)
ds_nwp = xr.open_zarr(zarr_path, decode_timedelta=True)
ds_nwp = ds_nwp.assign_coords(
    forecast_time=(
        ("start_time", "elapsed_forecast_duration"),
        ds_nwp.start_time.values[:, None]
        + ds_nwp.elapsed_forecast_duration.values,
    )
)
ds_nwp = subset_dataset_for_debug(
    ds_nwp,
    label="NWP zarr",
    dims_to_trim=("start_time", "x", "y"),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
)
ds_nwp = ds_nwp.sel(start_time=slice(*START_TIMES), x=slice(*X), y=slice(*Y))
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

assert_time_coverage(
    ds_gt, ds_ml.forecast_time.values.flatten(), label="Ground truth for ML"
)
assert_time_coverage(
    ds_gt, ds_nwp.forecast_time.values.flatten(), label="Ground truth for NWP"
)

if PRECOMPUTE_DATA:
    with ProgressBar():
        print("Computing ML data")
        ds_ml = ds_ml.compute()
        print("Computing NWP data")
        ds_nwp = ds_nwp.compute()
        print("Computing GT data")
        ds_gt = ds_gt.compute()

# Assign lon/lat to ML dataset from NWP
ds_ml = ds_ml.assign_coords(
    {
        "lon": (("x", "y"), ds_nwp.lon.values),
        "lat": (("x", "y"), ds_nwp.lat.values),
    }
)

step_size_gt = pd.Timedelta(ds_gt.time.diff("time").min().values, "h")

time_selected = resolve_requested_time(
    ds_ml,
    dim="start_time",
    requested_time=PLOT_TIME,
    label="plot",
    allow_nearest=DEBUG_MODE,
)


# ── Helper functions ──────────────────────────────────────────────────────────
def get_colormap_for_variable(variable_name):
    v = variable_name.lower()
    if any(k in v for k in ("wind_u", "wind_v", "vertical_velocity")):
        return "RdBu_r"
    if "precipitation" in v:
        return "Blues"
    if any(k in v for k in ("temperature", "radiation", "heat_flux")):
        return "magma"
    if "humidity" in v:
        return "Greens"
    if "wind_speed" in v:
        return "YlOrRd"
    return "viridis"


def plot_field(ax, data, vmin, vmax, cmap="viridis", norm=None):
    kwargs = dict(
        transform=ccrs.PlateCarree(), cmap=cmap, shading="auto", rasterized=True
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
                draw_labels=True, dms=True, x_inline=False, y_inline=False
            )
            gl.top_labels = False
            gl.bottom_labels = False
            gl.left_labels = False
            gl.right_labels = False
            if j == 0:
                gl.left_labels = True
            if i == n_rows - 1:
                gl.bottom_labels = True


def add_colorbar(fig, im, var):
    cbar_ax = fig.add_axes([0.2, 0.0, 0.6, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"({VARIABLE_UNITS[var]})")


def save_plot(fig, name, time=None):
    plot_dir = Path("verification/cosmo/case_study")
    plot_dir.mkdir(parents=True, exist_ok=True)
    if time is not None:
        name = f"{name}_{time.dt.strftime('%Y%m%d_%H').values}"
    if hasattr(fig, "texts") and fig.texts:
        fig.suptitle("")
    fig.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight", dpi=DPI)
    fig.savefig(plot_dir / f"{name}.png", bbox_inches="tight", dpi=DPI)
    print(f"Saved {name}")


# ── Plot ──────────────────────────────────────────────────────────────────────
apply_style(SCALE_MAPS_COSMO * 0.735)

var = "temperature_2m"
ds_ml_plot = ds_ml.isel(
    elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT
)
ds_nwp_plot = ds_nwp.isel(
    elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT
)

n_elapsed = len(ds_ml_plot.elapsed_forecast_duration)
n_cols = 3

fig = plt.figure(figsize=(7.5 * n_cols, 6 * n_elapsed), dpi=DPI)
axes = np.array(
    [
        [
            plt.subplot(
                n_elapsed, n_cols, i * n_cols + j + 1, projection=PROJECTION
            )
            for j in range(n_cols)
        ]
        for i in range(n_elapsed)
    ]
)

ds_ml_start = ds_ml_plot.sel(start_time=time_selected)
ds_nwp_start = ds_nwp_plot.sel(start_time=time_selected)

arrays_for_minmax = []
for elapsed in ds_ml_plot.elapsed_forecast_duration:
    ds_ml_time = ds_ml_start.sel(elapsed_forecast_duration=elapsed)
    ds_gt_time = ds_gt.sel(time=ds_ml_time.forecast_time)
    ds_nwp_time = ds_nwp_start.sel(elapsed_forecast_duration=elapsed)
    arrays_for_minmax.extend(
        [
            ds_gt_time[var].values,
            ds_ml_time[var].values,
            ds_nwp_time[var].values,
        ]
    )

vmin = np.nanmin([np.nanmin(a) for a in arrays_for_minmax])
vmax = np.nanmax([np.nanmax(a) for a in arrays_for_minmax])
var_cmap = get_colormap_for_variable(var)

im0 = None
for dim_idx, elapsed in enumerate(ds_ml_plot.elapsed_forecast_duration):
    ds_ml_time = ds_ml_start.sel(elapsed_forecast_duration=elapsed)
    ds_gt_time = ds_gt.sel(time=ds_ml_time.forecast_time)
    ds_nwp_time = ds_nwp_start.sel(elapsed_forecast_duration=elapsed)
    forecast_hours = int(elapsed.values / 1e9 / 3600)

    im0 = plot_field(
        axes[dim_idx, 0], ds_gt_time[var], vmin, vmax, cmap=var_cmap
    )
    plot_field(axes[dim_idx, 1], ds_nwp_time[var], vmin, vmax, cmap=var_cmap)
    plot_field(axes[dim_idx, 2], ds_ml_time[var], vmin, vmax, cmap=var_cmap)

    if dim_idx == 0:
        axes[dim_idx, 0].set_title(f"Ground Truth\n+{forecast_hours}h")
        axes[dim_idx, 1].set_title(f"NWP\n+{forecast_hours}h")
        axes[dim_idx, 2].set_title(f"ML\n+{forecast_hours}h")
    else:
        for ax in axes[dim_idx]:
            ax.set_title(f"+{forecast_hours}h")

add_map_features(axes)
add_colorbar(fig, im0, var)

title = (
    f"{var} starting at {time_selected.dt.date.values!s}"
    f" - {time_selected.dt.hour.values:02d} UTC"
)
plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.005, wspace=0.03)
plt.suptitle(title, y=0.98)

save_plot(fig, f"map_{var}_multi_efd", time_selected)
plt.close()

print("Done.")
