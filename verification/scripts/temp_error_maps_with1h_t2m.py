"""Temporary script: with1h mean error maps for temperature_2m only."""

# Standard library
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from debug_utils import (
    align_on_common_coord,
    assert_time_coverage,
    filter_forecast_dataset_by_time_coverage,
    parse_debug_runtime_args,
    select_available_times,
    subset_dataset_for_debug,
)
from plot_styles import DPI, SCALE_MAPS_COSMO, apply_style
from scores.continuous import mean_error

# ── Config ────────────────────────────────────────────────────────────────────
PATH_GROUND_TRUTH = "cosmo.datastore.zarr"
PATH_NWP = "cosmo_e_forecast.zarr"
PATH_ML = "preds_7_19_margin_interior_lr_0001_ar_12.zarr"

ELAPSED_FORECAST_DURATION = [0, 23, 47, 71, 95, 119]
ELAPSED_FORECAST_DURATION_PLOT_WITH1H = [0, 1, 3, 5]

START_TIMES = ["2019-10-31T00:00:00", "2020-10-23T13:00:00"]
X = [None, None]
Y = [None, None]

PROJECTION = ccrs.RotatedPole(
    pole_longitude=190,
    pole_latitude=43,
    central_rotated_longitude=10,
)

VARIABLES_GROUND_TRUTH = {
    "T_2M": "temperature_2m",
}
VARIABLES_ML = VARIABLES_GROUND_TRUTH
VARIABLES_NWP = {
    "temperature_2m": "temperature_2m",
}

VARIABLE_UNITS = {
    "temperature_2m": "K",
}

SUBSAMPLE_FRACTION = 0.2
PRECOMPUTE_DATA = True
RANDOM_SEED = 42

DEBUG_MODE, DEBUG_FRACTION, DEBUG_MIN_SIZE = parse_debug_runtime_args(
    "Temp with1h error maps for t2m."
)

Path("verification/cosmo/gridded").mkdir(parents=True, exist_ok=True)

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

rng = np.random.RandomState(RANDOM_SEED)
sampled_start_time_indices = np.sort(
    rng.choice(
        len(ds_ml.start_time),
        size=max(1, int(len(ds_ml.start_time) * SUBSAMPLE_FRACTION)),
        replace=False,
    )
)
ds_ml_sampled = ds_ml.isel(start_time=sampled_start_time_indices)
ds_nwp_sampled = ds_nwp.isel(start_time=sampled_start_time_indices)

if PRECOMPUTE_DATA:
    with ProgressBar():
        print("Computing ML data")
        ds_ml_sampled = ds_ml_sampled.compute()
        print("Computing NWP data")
        ds_nwp_sampled = ds_nwp_sampled.compute()
        print("Computing GT data")
        ds_gt = ds_gt.compute()


# ── Helper functions ──────────────────────────────────────────────────────────
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


def add_error_colorbar(fig, im, var):
    cbar_ax = fig.add_axes([0.15, 0.0, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"Mean Error ({VARIABLE_UNITS[var]})")


def save_plot(fig, name):
    plot_dir = Path("verification/cosmo/gridded")
    plot_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "texts") and fig.texts:
        fig.suptitle("")
    fig.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight", dpi=DPI)
    fig.savefig(plot_dir / f"{name}.png", bbox_inches="tight", dpi=DPI)
    print(f"Saved {name}")


def create_error_maps(ds_gt, ds_ml, ds_nwp=None, var=None, name_prefix=""):
    variables = [var] if var else list(VARIABLES_GROUND_TRUTH.values())
    n_elapsed = len(ds_ml.elapsed_forecast_duration)

    for var in variables:
        n_cols = 2 if (ds_nwp is not None and var in ds_nwp) else 1

        fig = plt.figure(figsize=(7.5 * n_cols, 6 * n_elapsed), dpi=DPI)
        axes = np.array(
            [
                [
                    plt.subplot(
                        n_elapsed,
                        n_cols,
                        i * n_cols + j + 1,
                        projection=PROJECTION,
                    )
                    for j in range(n_cols)
                ]
                for i in range(n_elapsed)
            ]
        )

        arrays_for_minmax = []
        for dim_idx, elapsed in enumerate(ds_ml.elapsed_forecast_duration):
            forecast_hours = int(elapsed.values / 1e9 / 3600)
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

            obs_times = ds_ml.forecast_time.sel(
                elapsed_forecast_duration=elapsed
            )
            errors = {}
            errors["ml"] = mean_error(
                ds_ml[var].sel(elapsed_forecast_duration=elapsed),
                ds_gt[var].sel(time=obs_times),
                preserve_dims=["x", "y"],
            )
            arrays_for_minmax.append(errors["ml"])

            if ds_nwp is not None and var in ds_nwp:
                errors["nwp"] = mean_error(
                    ds_nwp[var].sel(elapsed_forecast_duration=elapsed),
                    ds_gt[var].sel(time=obs_times),
                    preserve_dims=["x", "y"],
                )
                arrays_for_minmax.append(errors["nwp"])

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
                    f"NWP\n+{forecast_hours}h"
                    if dim_idx == 0
                    else f"+{forecast_hours}h"
                )

            col_idx = 1 if ds_nwp is not None and var in ds_nwp else 0
            im = plot_error_field(
                axes[dim_idx, col_idx], lons, lats, errors["ml"], vmin, vmax
            )
            axes[dim_idx, col_idx].set_title(
                f"ML\n+{forecast_hours}h"
                if dim_idx == 0
                else f"+{forecast_hours}h"
            )

        add_map_features(axes)
        add_error_colorbar(fig, im, var)
        title = f"Mean Error in {var}"
        plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.005, wspace=0.03)
        plt.suptitle(title, y=0.95)
        save_plot(fig, f"{name_prefix}mean_errormap_{var}_multi_efd")
        plt.close()


# ── Plot ──────────────────────────────────────────────────────────────────────
if "lon" not in ds_ml_sampled.coords:
    ds_ml_sampled = ds_ml_sampled.assign_coords(
        {
            "lon": (("x", "y"), ds_nwp.lon.values),
            "lat": (("x", "y"), ds_nwp.lat.values),
        }
    )

apply_style(SCALE_MAPS_COSMO * 1.05)

print("Plotting with1h error map: temperature_2m")
create_error_maps(
    ds_gt=ds_gt,
    ds_ml=ds_ml_sampled.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT_WITH1H
    ),
    ds_nwp=ds_nwp_sampled.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_PLOT_WITH1H
    ),
    var="temperature_2m",
    name_prefix="with1h_",
)

print("Done.")
