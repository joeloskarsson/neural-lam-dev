# COSMO Dataset Reference

Reference for the COSMO dataset behind the published checkpoint
([Zenodo](https://zenodo.org/records/15131838)): what each variable is, how the 57 datastore state
features map onto it, and how to use the training normalisation statistics that ship with this repo.

The underlying data is hourly KENDA operational analysis from MeteoSwiss. Sample dataset and full
citation: [ETHZ Research Collection](https://www.research-collection.ethz.ch/entities/researchdata/3e24bbad-2885-4855-a100-78d18ed321f9).
Publisher MeteoSwiss, licensed In Copyright - Non-Commercial Use Permitted. The statistics in
`scripts/artifacts/` are aggregates derived from that dataset and inherit its terms, which are
narrower than this repository's MIT licence.

## Overview

Hourly representations of the atmospheric state over Switzerland and the Alpine region, created by
the MeteoSwiss data assimilation system KENDA (operational analysis, i.e. initial fields for the
COSMO forecasting model). Prepared for machine learning applications in weather forecasting.

- **Size**: 9.3 TB
- **Time Period**: 2015-11-28T00 to 2020-10-29T04 (43133 hourly steps)
- **Resolution**: 2.2 km (0.02° on rotated grid)
- **Format**: Zarr (Chunked Dask Arrays)

## Projection

Rotated Latitude-Longitude grid.

```python
projection:
    class_name: RotatedPole
    kwargs:
        central_rotated_longitude: 10.0
        pole_longitude: 190.0
        pole_latitude: 43.0
```

**Important**: Vector variables (`U`, `V`, `U_10M`, `V_10M`) are defined relative to the **rotated
grid**, not true North/East.

## Dimensions and coordinates

- **time**: 43133 steps (Hourly, 2015-11-28T00 to 2020-10-29T04)
- **y**: 390 (Latitude index, 0 = southmost, 389 = northmost)
- **x**: 582 (Longitude index, 0 = westmost, 581 = eastmost)
- **z**: 60 levels (Vertical model levels, 1 = top, 60 = surface)
- **lat**: (y, x) float64 - Geographic Latitude
- **lon**: (y, x) float64 - Geographic Longitude
- **time**: datetime64[ns] - Valid time

## Variables

The raw dataset contains 25 atmospheric variables. The tables below are a glossary ordered
alphabetically. They are **not** the datastore feature ordering, see
[Datastore features](#datastore-features) for that.

### Surface / 2D, static

| Variable | Description | Unit | Dimensions | ECMWF Equiv. (ID) | Notes |
|----------|-------------|------|------------|-------------------|-------|
| **HSURF** | Geometrical height of surface topography | m | (y, x) | | ERA5 does not provide geometric height of the surface directly |

### Surface / 2D, time-dependent

| Variable | Description | Unit | Dimensions | ECMWF Equiv. (ID) | Notes |
|----------|-------------|------|------------|-------------------|-------|
| **ALHFL_S** | Average latent heat flux (surface) | W/m² | (time, y, x) | `slhf` (147) | COSMO: Positive upwards, **averaged**. ECMWF: Positive **downwards, accumulated**. |
| **ASHFL_S** | Average sensible heat flux (surface) | W/m² | (time, y, x) | `sshf` (146) | COSMO: Positive upwards, **averaged**. ECMWF: Positive **downwards, accumulated**. |
| **ASOB_S** | Average solar radiation budget (surface) | W/m² | (time, y, x) | `ssr` (176) | COSMO: W/m² **averaged**. ECMWF: **J**/m² **accumulated**. |
| **ATHB_S** | Average thermal radiation budget (surface) | W/m² | (time, y, x) | `str` (177) | COSMO: W/m² **averaged**. ECMWF: **J**/m² **accumulated**. |
| **CLCT** | Total cloud cover | % | (time, y, x) | `tcc` (164) | Fraction of sky covered by clouds. **Range: 0-100**. ECMWF `tcc` is **0-1**. |
| **PMSL** | Surface pressure on mean sea level | Pa | (time, y, x) | `msl` (151) | Reduced pressure. |
| **PS** | Surface pressure | Pa | (time, y, x) | `sp` (134) | Actual pressure at surface height `HSURF`. |
| **TD_2M** | Dew-point in 2m | K | (time, y, x) | `2d` (168) | Diagnostic variable calculated from 2m humidity. |
| **TOT_PREC** | Total precipitation | kg/m² (mm) | (time, y, x) | `tp` (228) | Accumulated amount over output interval (1h). ECMWF in **m**. |
| **TQV** | Total Column Integrated Water Vapour | kg/m² | (time, y, x) | `tcwv` (137) | Precipitable water. |
| **T_2M** | Temperature in 2m | K | (time, y, x) | `2t` (167) | Diagnostic variable calculated from surface and lowest level. |
| **U_10M** | Zonal wind in 10m | m/s | (time, y, x) | `10u` (165) | COSMO: Along **rotated** x-axis. |
| **V_10M** | Meridional wind in 10m | m/s | (time, y, x) | `10v` (166) | COSMO: Along **rotated** y-axis. |
| **VMAX_10M** | Maximal windspeed in 10m | m/s | (time, y, x) | `10fg` (49) | Max value during output interval. |
| **W_SNOW** | Water content of snow | kg/m² (mm) | (time, y, x) | `sd` (141) | Water equivalent of snow depth. ECMWF `sd` is in **m** of water equiv. |

### 3D, static

| Variable | Description | Unit | Dimensions | ECMWF Equiv. (ID) | Notes |
|----------|-------------|------|------------|-------------------|-------|
| **FI** | Geopotential | m²/s² | (z, y, x) | `z` (129) | `FI = g * h`. COSMO: Static field (constant on model levels) - *Missing metadata in Zarr*. ECMWF: **Time-dependent** field. |
| **P0FL** | Base State Pressure | Pa | (z, y, x) | `pres` (54) | **Static**. Reference pressure at full levels. (Missing metadata in Zarr.) |

### 3D, time-dependent

| Variable | Description | Unit | Dimensions | ECMWF Equiv. (ID) | Notes |
|----------|-------------|------|------------|-------------------|-------|
| **PP** | Pressure Perturbation | Pa | (time, z, y, x) | `pres` (54) | Deviation from `P0FL`. **Full Pressure `pres` (54) = P0FL + PP**. |
| **QV** | Specific water vapor content | kg/kg | (time, z, y, x) | `q` (133) | Mass of water vapor per unit mass of moist air. |
| **RELHUM** | Relative humidity | % | (time, z, y, x) | `r` (157) | **Metadata Error**: Zarr attributes incorrectly label this as "Temperature". Values are correct (0-100%). |
| **T** | Temperature | K | (time, z, y, x) | `t` (130) | 3D Air Temperature. |
| **U** | Zonal wind speed | m/s | (time, z, y, x) | `u` (131) | COSMO: Along **rotated** x-axis. ECMWF: Along **unrotated** x-axis. |
| **V** | Meridional wind speed | m/s | (time, z, y, x) | `v` (132) | COSMO: Along **rotated** y-axis. ECMWF: Along **unrotated** y-axis. |
| **W** | Vertical wind speed | m/s | (time, z, y, x) | `w` (135) | COSMO: Positive upwards. ECMWF: `w` Omega (**Pa**/s), positive **downwards**. |

## Technical Notes & Gotchas

1.  **Pressure Reconstruction**: The 3D pressure field is split into a static base state (`P0FL`) and
    a time-dependent perturbation (`PP`). You must sum them to get the full pressure:
    $P(t,z,y,x) = P0FL(z,y,x) + PP(t,z,y,x)$.
2.  **Rotated Grid**: Wind components `U` and `V` are aligned with the rotated grid. To rotate them to
    true North/East, you need to use the grid rotation parameters.
3.  **Vertical Coordinates**: The `z` coordinate is a model level index, not height. The actual height
    of each level depends on the surface topography (`HSURF`) and the vertical coordinate parameters
    (SLEVE system). `FI` (Geopotential) gives information about the height of levels (approx
    $h = FI/g$). Note that `FI` and `P0FL` are provided as static fields in this dataset. On model
    levels, `FI` is constant (terrain-following), whereas on pressure levels (ECMWF standard),
    pressure is constant and Geopotential varies.
4.  **Fluxes and Accumulations**: Variables ending in `_S` (like `ALHFL_S`) and `TOT_PREC` represent
    averages or accumulations over the output interval (1 hour), not instantaneous values. Note that
    COSMO fluxes are generally **positive upwards**, whereas ECMWF fluxes are often positive
    downwards.
5.  **Units**: Temperatures are in Kelvin. Precipitation is in kg/m² (equivalent to mm).

## Datastore features

`python -m mllam_data_prep scripts/cosmo_interior_config.yaml` turns the raw data above into the
datastore the model consumes. Feature order follows the order the inputs appear in that config, with
each 3D variable expanded over its 8 levels before moving to the next variable. This ordering is
recorded in the `state_feature` coordinate of the datastore and is what the columns of the statistics
arrays refer to.

`lev_N` is the COSMO **model level index**, 1 = model top to 60 = lowest level above ground. It is
neither a height nor a pressure, see note 3 above.

| idx | feature | variable |
|-----|---------|----------|
| 0-7 | `U_lev_{6,12,20,27,31,39,45,60}` | zonal wind, rotated x-axis, m/s |
| 8-15 | `V_lev_{...}` | meridional wind, rotated y-axis, m/s |
| 16-23 | `PP_lev_{...}` | pressure perturbation from `P0FL`, Pa |
| 24-31 | `T_lev_{...}` | air temperature, K |
| 32-39 | `RELHUM_lev_{...}` | relative humidity, %, 0-100 |
| 40-47 | `W_lev_{...}` | vertical wind, m/s, positive upwards |
| 48 | `T_2M` | 2 m temperature, K |
| 49 | `U_10M` | 10 m zonal wind, rotated x-axis, m/s |
| 50 | `V_10M` | 10 m meridional wind, rotated y-axis, m/s |
| 51 | `PMSL` | mean sea level pressure, Pa |
| 52 | `PS` | surface pressure at `HSURF`, Pa |
| 53 | `TOT_PREC` | precipitation accumulated over 1 h, kg/m² |
| 54 | `ASHFL_S` | sensible heat flux, W/m², positive upwards, 1 h average |
| 55 | `ASOB_S` | solar radiation budget, W/m², 1 h average |
| 56 | `ATHB_S` | thermal radiation budget, W/m², 1 h average |

Forcing features are derived from time and lat/lon by mllam-data-prep and involve no COSMO data:
`toa_radiation`, `hour_of_day_sin`, `hour_of_day_cos`, `day_of_year_sin`, `day_of_year_cos`.

Static features are `HSURF` from the COSMO data and `lsm` from
`scripts/artifacts/cosmo_land_sea_mask.zarr`.

Note that the datastore's `state_feature_long_name` and `state_feature_units` coordinates are empty
strings, so none of the above is recoverable from the datastore itself. That is what this page is for.

## Training statistics

The published checkpoint's normalisation statistics are **not** stored in the checkpoint. They are
registered as non-persistent buffers in
[`neural_lam/models/ar_model.py`](../neural_lam/models/ar_model.py) and recomputed from whatever
datastore you point at, so running the checkpoint against a different datastore silently renormalises
the model.

The statistics the checkpoint was trained with are committed at
`scripts/artifacts/cosmo_train_stats.zarr`, computed over the training split
2015-11-28T00 to 2019-09-30T00 with `dims: [grid_index, time]`. They are stored against their feature
names rather than as bare arrays, so ordering cannot be misapplied.

To use them with a different datastore, point at them with `overload_stats_path`:

```yaml
datastore:
  kind: mdp
  config_path: your_own_config.yaml
  overload_stats_path: artifacts/cosmo_train_stats.yaml
```

The yaml and the zarr must stay siblings with the same stem, because `MDPDatastore` derives the zarr
path from the config filename. If the zarr is missing, the datastore falls through to
`mdp.create_dataset` and tries to build it from raw data, which will fail.

## Transferring the checkpoint to another domain

Restoring the statistics is necessary but not sufficient. These all degrade forecasts silently and
none of them is repaired by rescaling:

1. **Model levels are configuration-dependent.** `lev_6` is an index into this COSMO configuration's
   60 levels. A model with a different level count or different SLEVE parameters puts index 6 at a
   different altitude.
2. **`PP` is relative to this configuration's reference atmosphere.** `P0FL` is not in the datastore,
   so a different base state makes `PP` incomparable.
3. **Winds are on the rotated grid** given above, not true North/East.
4. **`TOT_PREC` and the `A*_S` fluxes** are 1 h accumulations and averages of analyses, not
   instantaneous fields and not forecast-lead accumulations. COSMO fluxes are positive upwards.
5. **Ranges differ from ERA5 conventions**: `RELHUM` is 0-100, `CLCT` is 0-100, `lsm` is 0-1.
6. **Statistics are computed over the `train` split only.** A short train window gives a much smaller
   standard deviation than the 4-year one used here, which inflates standardised anomalies.

The boundary side needs no special handling. The ERA5 and IFS datastores are built from public
WeatherBench2 subsets by `scripts/era_download.py` and `scripts/ifs_download.py`, so running steps 3
and 6 of [reproduce_paper.md](reproduce_paper.md) reproduces the boundary statistics the checkpoint
was trained with.
