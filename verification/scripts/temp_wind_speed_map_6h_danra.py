"""Temporary script: DANRA wind speed + MSLP case-study map, +6 h row only."""

# Standard library
import math
import os
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from plot_styles import DPI, SCALE_METRICS, apply_style

# ── Config ────────────────────────────────────────────────────────────────────
PATH_NWP = "danra_test_nwp_forecasts.zarr"
# Override with the extracted subset when the full ML zarr is not unpacked
PATH_ML = os.environ.get("PATH_ML", "48h_eval_test_7deg_rect_hi4_2867.zarr")
PATH_OBS = (
    "/capstor/store/cscs/swissai/a122/sadamov/danra_station_observations.zarr"
)

VARIABLES_ML = {
    "t2m": "temperature_2m",
    "u10m": "wind_u_10m",
    "v10m": "wind_v_10m",
    "pres_seasurface": "seasurface_pressure",
}
VARIABLES_NWP = VARIABLES_ML
OBS_VAR_MAPPING = VARIABLES_ML

PLOT_TIME = "2020-02-09T12:00:00"
# Index into elapsed_forecast_duration; matches the first row (+6 h) of the
# ELAPSED_FORECAST_DURATION_PLOT = [1, 3, 5] maps in verification_sparse_danra
EFD_INDEX = 1

PROJECTION = ccrs.LambertConformal(
    central_longitude=25.0,
    central_latitude=56.7,
    standard_parallels=[56.7, 56.7],
    globe=ccrs.Globe(
        semimajor_axis=6367470.0,
        semiminor_axis=6367470.0,
    ),
)

U_VAR, V_VAR = "wind_u_10m", "wind_v_10m"
PRESSURE_VAR = "seasurface_pressure"

apply_style(SCALE_METRICS)

# ── Data loading ──────────────────────────────────────────────────────────────
ds_nwp = xr.open_zarr(PATH_NWP, decode_timedelta=True)
ds_nwp = ds_nwp.sel(analysis_time=[PLOT_TIME])
ds_nwp = ds_nwp[VARIABLES_NWP.keys()].rename(VARIABLES_NWP)
ds_nwp = ds_nwp.rename_dims({"analysis_time": "start_time"})
ds_nwp = ds_nwp.rename_vars(
    {"analysis_time": "start_time", "lon": "longitude", "lat": "latitude"}
)
ds_nwp = ds_nwp.assign_coords(
    forecast_time=(
        ("start_time", "elapsed_forecast_duration"),
        ds_nwp.start_time.values[:, None]
        + ds_nwp.elapsed_forecast_duration.values,
    )
)
ds_nwp = ds_nwp.drop_vars("time")
ds_nwp = ds_nwp.isel(elapsed_forecast_duration=[EFD_INDEX])
ds_nwp = ds_nwp.transpose("start_time", "elapsed_forecast_duration", "x", "y")

ds_ml = xr.open_zarr(PATH_ML, decode_timedelta=True)
ds_ml = ds_ml.sel(state_feature=list(VARIABLES_ML.keys()))
ds_ml = ds_ml.sel(start_time=[PLOT_TIME])
for feature in ds_ml.state_feature.values:
    ds_ml[VARIABLES_ML[feature]] = ds_ml["state"].sel(state_feature=feature)
ds_ml = ds_ml.assign_coords(
    forecast_time=(
        ("start_time", "elapsed_forecast_duration"),
        ds_ml.start_time.values[:, None]
        + ds_ml.elapsed_forecast_duration.values,
    )
)
ds_ml = ds_ml.drop_vars(["state", "state_feature", "time"])
ds_ml = ds_ml.isel(elapsed_forecast_duration=[EFD_INDEX])
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
    {"latitude": ds_nwp.latitude, "longitude": ds_nwp.longitude}
)

assert (
    ds_ml.elapsed_forecast_duration.values
    == ds_nwp.elapsed_forecast_duration.values
).all(), "ML and NWP lead times do not match"

ds_obs = xr.open_zarr(PATH_OBS, decode_timedelta=True)
ds_obs = ds_obs.rename_vars(OBS_VAR_MAPPING)
ds_obs = ds_obs.rename_dims({"stationId": "station"})
ds_obs = ds_obs.rename_vars(
    {"stationId": "station", "lat": "latitude", "lon": "longitude"}
)
ds_obs = ds_obs.sel(time=np.unique(ds_ml.forecast_time.values.flatten()))

ds_ml = ds_ml.compute()
ds_nwp = ds_nwp.compute()
ds_obs = ds_obs.compute()

plot_time = ds_ml.start_time[0]
extent = [
    ds_obs.longitude.min().values - 0.1,
    ds_obs.longitude.max().values + 0.1,
    ds_obs.latitude.min().values - 0.1,
    ds_obs.latitude.max().values + 0.1,
]


# ── Helpers (mirrored from verification_sparse_danra.py) ──────────────────────
def wind_speed(ds_slice):
    return np.sqrt(ds_slice[U_VAR].values ** 2 + ds_slice[V_VAR].values ** 2)


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
            gl.top_labels = False
            gl.bottom_labels = i == n_rows - 1
            gl.left_labels = j == 0
            gl.right_labels = False


def add_pressure_contour(ax, ds_slice):
    p = ds_slice[PRESSURE_VAR].values / 100  # Pa → hPa
    levels = np.arange(
        np.floor(float(np.nanmin(p)) / 4) * 4,
        np.ceil(float(np.nanmax(p)) / 4) * 4 + 4,
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
            manual_positions.append(
                ax.projection.transform_point(
                    mid[0], mid[1], ccrs.PlateCarree()
                )
            )
    if manual_positions:
        ax.clabel(
            cs, fmt="%d", fontsize=6, inline=True, manual=manual_positions
        )


# ── Plot ──────────────────────────────────────────────────────────────────────
lon_span = extent[1] - extent[0]
lat_span = extent[3] - extent[2]
lat_mid = (extent[2] + extent[3]) / 2
map_aspect = lon_span * math.cos(math.radians(lat_mid)) / lat_span
bottom = 0.22
fig_w = 3 * 6 * (0.9 - bottom) * map_aspect / 0.775

fig = plt.figure(figsize=(fig_w, 6), dpi=DPI)
axes = np.array(
    [[plt.subplot(1, 3, j + 1, projection=PROJECTION) for j in range(3)]]
)
for ax in axes[0]:
    ax.set_extent(extent, crs=ccrs.PlateCarree())

step = ds_ml.elapsed_forecast_duration[0]
forecast_time = ds_ml.forecast_time.isel(
    start_time=0, elapsed_forecast_duration=0
)
forecast_hours = int(step.values / 1e9 / 3600)

obs_slice = ds_obs.sel(time=forecast_time)
nwp_slice = ds_nwp.isel(start_time=0, elapsed_forecast_duration=0)
ml_slice = ds_ml.isel(start_time=0, elapsed_forecast_duration=0)

vmax = float(
    np.nanmax(
        [np.nanmax(wind_speed(s)) for s in (obs_slice, nwp_slice, ml_slice)]
    )
)
mesh_kw = dict(
    cmap="YlOrRd",
    vmin=0.0,
    vmax=vmax,
    transform=ccrs.PlateCarree(),
    shading="auto",
    rasterized=True,
)

axes[0, 0].scatter(
    ds_obs.longitude,
    ds_obs.latitude,
    c=wind_speed(obs_slice),
    cmap="YlOrRd",
    vmin=0.0,
    vmax=vmax,
    transform=ccrs.PlateCarree(),
)
axes[0, 1].pcolormesh(
    ds_nwp.longitude, ds_nwp.latitude, wind_speed(nwp_slice), **mesh_kw
)
add_pressure_contour(axes[0, 1], nwp_slice)
im2 = axes[0, 2].pcolormesh(
    ds_ml.longitude, ds_ml.latitude, wind_speed(ml_slice), **mesh_kw
)
add_pressure_contour(axes[0, 2], ml_slice)

axes[0, 0].set_title(f"Observations\n+{forecast_hours}h")
axes[0, 1].set_title(f"NWP\n+{forecast_hours}h")
axes[0, 2].set_title(f"ML\n+{forecast_hours}h")

add_map_features(axes)

plt.subplots_adjust(top=0.9, bottom=bottom, hspace=0.105, wspace=0.003)
cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
plt.colorbar(im2, cax=cbar_ax, orientation="horizontal", label="(m/s)")

plot_dir = Path("verification/danra/case_study")
plot_dir.mkdir(parents=True, exist_ok=True)
name = (
    "observations_comparison_wind_speed_single_step_"
    f"{plot_time.dt.strftime('%Y%m%d_%H').values}"
)
fig.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight", dpi=DPI)
np.savez(
    plot_dir / f"{name}.npz",
    obs_wind_speed=wind_speed(obs_slice),
    obs_longitude=ds_obs.longitude.values,
    obs_latitude=ds_obs.latitude.values,
    nwp_wind_speed=wind_speed(nwp_slice),
    ml_wind_speed=wind_speed(ml_slice),
    nwp_seasurface_pressure=nwp_slice[PRESSURE_VAR].values,
    ml_seasurface_pressure=ml_slice[PRESSURE_VAR].values,
    longitude=ds_ml.longitude.values,
    latitude=ds_ml.latitude.values,
)
print(f"Saved {plot_dir / name}.pdf")
