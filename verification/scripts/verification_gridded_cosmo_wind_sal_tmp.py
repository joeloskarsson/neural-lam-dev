# Temporary script: runs only wind vector RMSE and SAL sections from
# verification_gridded_cosmo_metrics.py — can run in parallel with
# the MeteoSwiss metrics job.

# Standard library
from pathlib import Path

# Third-party
import cartopy.crs as ccrs  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from debug_utils import (
    align_on_common_coord,
    parse_debug_runtime_args,
    subset_dataset_for_debug,
)
from plot_styles import DPI, SCALE_METRICS, apply_style
from pysteps.verification.salscores import sal

# ── Config (keep in sync with main script) ───────────────────────────────────

PATH_GROUND_TRUTH = "cosmo.datastore.zarr"
PATH_NWP_RAW = "/capstor/store/cscs/swissai/a122/sadamov/cosmo_e_forecast.zarr"
PATH_NWP = (
    "/capstor/store/cscs/swissai/a122/sadamov/cosmo_e_forecast_fixed.zarr"
)
PATH_ML = (
    "/capstor/store/cscs/swissai/a122/sadamov/lam_model_forecasts"
    "/preds_7_19_margin_interior_lr_0001_ar_12.zarr"
)

ELAPSED_FORECAST_DURATION = list(range(120))
ELAPSED_FORECAST_DURATION_SAL = [0, 23, 47, 71, 95, 119]
START_TIMES = ["2019-10-31T00:00:00", "2020-10-23T13:00:00"]

X = [None, None]
Y = [None, None]

VARIABLES_GROUND_TRUTH = {
    "T_2M": "temperature_2m",
    "U_10M": "wind_u_10m",
    "V_10M": "wind_v_10m",
    "PS": "surface_pressure",
    "TOT_PREC": "precipitation",
}
VARIABLES_ML = VARIABLES_GROUND_TRUTH
VARIABLES_NWP = {
    "wind_u_10m": "wind_u_10m",
    "wind_v_10m": "wind_v_10m",
    "precipitation_1hr": "precipitation",
    "surface_pressure": "surface_pressure",
    "temperature_2m": "temperature_2m",
}

SUBSAMPLE_FRACTION = 1.0
PRECOMPUTE_DATA = False
RANDOM_SEED = 42

COLORS = {
    "gt": "#000000",
    "ml": "#E69F00",
    "nwp": "#56B4E9",
    "per": "#CC79A7",
}
LINE_STYLES = {
    "gt": ("solid", "o"),
    "ml": ("dashed", "s"),
    "nwp": ("dotted", "^"),
    "per": ("dashdot", "v"),
}

DEBUG_MODE, DEBUG_FRACTION, DEBUG_MIN_SIZE = parse_debug_runtime_args(
    "Wind vector RMSE and SAL only.",
)

Path("verification/cosmo/gridded").mkdir(parents=True, exist_ok=True)
apply_style(SCALE_METRICS)

# ── Helpers ───────────────────────────────────────────────────────────────────


def save_plot(fig, name, dpi=300):
    plot_dir = Path("verification/cosmo/gridded")
    safe_name = name.replace("/", "_per_")
    if hasattr(fig, "texts") and fig.texts:
        fig.suptitle("")
    ax = fig.gca()
    if ax.get_title():
        ax.set_title("")
    fig.savefig(plot_dir / f"{safe_name}.pdf", bbox_inches="tight", dpi=dpi)
    npz_payload = {}
    for ax_idx, ax in enumerate(fig.get_axes()):
        for line_idx, line in enumerate(ax.get_lines()):
            label = line.get_label()
            key = (
                f"ax{ax_idx}_{label.replace(' ', '_')}"
                if label and not label.startswith("_")
                else f"ax{ax_idx}_line{line_idx}"
            )
            npz_payload[f"{key}_x"] = np.asarray(line.get_xdata(), dtype=float)
            npz_payload[f"{key}_y"] = np.asarray(line.get_ydata(), dtype=float)
    np.savez_compressed(plot_dir / f"{safe_name}.npz", **npz_payload)


def export_table(df, name, caption=""):
    df.to_latex(
        f"verification/cosmo/gridded/{name}.tex",
        float_format="%.4f",
        caption=caption,
        label=f"tab:{name}",
    )
    df.to_csv(f"verification/cosmo/gridded/{name}.csv")


# ── Load datasets (same logic as main script) ─────────────────────────────────

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
ds_gt = ds_gt.sel(time=np.unique(time_subset))

zarr_path = PATH_NWP.replace(".zarr", "_processed_120h.zarr")
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
    dims_to_trim=("x", "y"),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
)
ds_nwp = ds_nwp.sel(start_time=slice(*START_TIMES), x=slice(*X), y=slice(*Y))
ds_ml, ds_nwp = align_on_common_coord(
    ds_ml, ds_nwp, coord_name="start_time", left_label="ML", right_label="NWP"
)
ds_ml = subset_dataset_for_debug(
    ds_ml,
    label="ML zarr (start_time)",
    dims_to_trim=("start_time",),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
)
ds_nwp = subset_dataset_for_debug(
    ds_nwp,
    label="NWP zarr (start_time)",
    dims_to_trim=("start_time",),
    debug_mode=DEBUG_MODE,
    debug_fraction=DEBUG_FRACTION,
    debug_min_size=DEBUG_MIN_SIZE,
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
        ds_ml_sampled = ds_ml_sampled.compute()
        ds_nwp_sampled = ds_nwp_sampled.compute()

# ── SAL scores ────────────────────────────────────────────────────────────────

if "precipitation" in ds_gt and "precipitation" in ds_ml_sampled:
    sal_lead_times = ds_ml_sampled.isel(
        elapsed_forecast_duration=ELAPSED_FORECAST_DURATION_SAL
    ).elapsed_forecast_duration.values
    sal_hours = [efd / np.timedelta64(1, "h") for efd in sal_lead_times]

    SAL_COLORS = {
        "S": plt.cm.Set2(0),
        "A": plt.cm.Set2(1),
        "L": plt.cm.Set2(2),
        "SAL": plt.cm.Set2(3),
    }

    sal_results = {
        "S_ML": [],
        "A_ML": [],
        "L_ML": [],
        "SAL_ML": [],
        "S_NWP": [],
        "A_NWP": [],
        "L_NWP": [],
        "SAL_NWP": [],
    }
    has_nwp_prec = (
        ds_nwp_sampled is not None and "precipitation" in ds_nwp_sampled
    )

    for efd in sal_lead_times:
        print(
            f"Computing SAL for lead time {efd / np.timedelta64(1, 'h'):.0f}h"
        )
        forecast_times = ds_ml_sampled.sel(
            elapsed_forecast_duration=efd
        ).forecast_time.values

        s_ml_list, a_ml_list, l_ml_list, sal_ml_list = [], [], [], []
        s_nwp_list, a_nwp_list, l_nwp_list, sal_nwp_list = [], [], [], []

        for fc_time, st in zip(forecast_times, ds_ml_sampled.start_time.values):
            pred = (
                ds_ml_sampled["precipitation"]
                .sel(elapsed_forecast_duration=efd)
                .sel(start_time=st)
                .values
            )
            obs = ds_gt["precipitation"].sel(time=fc_time).values

            result = sal(pred, obs, thr_factor=0.067, thr_quantile=0.9)
            if all(v is not None for v in result):
                s_ml_list.append(result[0])
                a_ml_list.append(result[1])
                l_ml_list.append(result[2])
                sal_ml_list.append(
                    abs(result[0]) + abs(result[1]) + abs(result[2])
                )

            if has_nwp_prec:
                pred_nwp = (
                    ds_nwp_sampled["precipitation"]
                    .sel(elapsed_forecast_duration=efd)
                    .sel(start_time=st)
                    .values
                )
                result_nwp = sal(
                    pred_nwp, obs, thr_factor=0.067, thr_quantile=0.9
                )
                if all(v is not None for v in result_nwp):
                    s_nwp_list.append(result_nwp[0])
                    a_nwp_list.append(result_nwp[1])
                    l_nwp_list.append(result_nwp[2])
                    sal_nwp_list.append(
                        abs(result_nwp[0])
                        + abs(result_nwp[1])
                        + abs(result_nwp[2])
                    )

        sal_results["S_ML"].append(
            np.nanmean(s_ml_list) if s_ml_list else np.nan
        )
        sal_results["A_ML"].append(
            np.nanmean(a_ml_list) if a_ml_list else np.nan
        )
        sal_results["L_ML"].append(
            np.nanmean(l_ml_list) if l_ml_list else np.nan
        )
        sal_results["SAL_ML"].append(
            np.nanmean(sal_ml_list) if sal_ml_list else np.nan
        )
        sal_results["S_NWP"].append(
            np.nanmean(s_nwp_list) if s_nwp_list else np.nan
        )
        sal_results["A_NWP"].append(
            np.nanmean(a_nwp_list) if a_nwp_list else np.nan
        )
        sal_results["L_NWP"].append(
            np.nanmean(l_nwp_list) if l_nwp_list else np.nan
        )
        sal_results["SAL_NWP"].append(
            np.nanmean(sal_nwp_list) if sal_nwp_list else np.nan
        )

    sal_df = pd.DataFrame(sal_results, index=sal_hours)
    sal_df.index.name = "Lead Time (hours)"
    export_table(
        sal_df, "sal_precipitation", caption="SAL scores for precipitation"
    )

    fig, ax = plt.subplots(figsize=(16, 7), dpi=DPI)
    components = [
        ("S_ML", "S_NWP", "Structure"),
        ("A_ML", "A_NWP", "Amplitude"),
        ("L_ML", "L_NWP", "Location"),
        ("SAL_ML", "SAL_NWP", "Combined"),
    ]
    for key_ml, key_nwp, name in components:
        color = SAL_COLORS[key_ml.split("_")[0]]
        ax.plot(
            sal_hours,
            sal_results[key_ml],
            label=f"{name} (ML)",
            color=color,
            linestyle=LINE_STYLES["ml"][0],
            marker=LINE_STYLES["ml"][1],
        )
        if has_nwp_prec and not all(np.isnan(sal_results[key_nwp])):
            ax.plot(
                sal_hours,
                sal_results[key_nwp],
                label=f"{name} (NWP)",
                color=color,
                linestyle=LINE_STYLES["nwp"][0],
                marker=LINE_STYLES["nwp"][1],
            )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("SAL Score")
    ax.set_title("SAL Scores for Precipitation (tp01)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax.grid(True, alpha=0.3)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )
    plt.tight_layout()
    save_plot(fig, "sal_precipitation_evolution")
    plt.close()
    print("SAL done.")
