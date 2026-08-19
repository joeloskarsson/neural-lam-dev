# Standard library
import argparse
import os

# Third-party
import numpy as np
import pandas as pd


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_debug_runtime_args(
    description, default_fraction=0.02, default_min_size=2
):
    parser = argparse.ArgumentParser(description=description)
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
            "Fraction of each supported dimension to keep"
            " in debug mode (0, 1]."
        ),
    )
    parser.add_argument(
        "--debug-min-size",
        type=int,
        default=None,
        help=(
            "Minimum number of points to keep per trimmed dimension"
            " in debug mode."
        ),
    )
    args, _ = parser.parse_known_args()

    debug_mode = args.debug or _env_flag("VERIFICATION_DEBUG", default=False)
    debug_fraction = args.debug_fraction
    if debug_fraction is None:
        debug_fraction = float(
            os.environ.get("VERIFICATION_DEBUG_FRACTION", str(default_fraction))
        )
    debug_min_size = args.debug_min_size
    if debug_min_size is None:
        debug_min_size = int(
            os.environ.get("VERIFICATION_DEBUG_MIN_SIZE", str(default_min_size))
        )

    if not 0 < debug_fraction <= 1:
        raise ValueError("--debug-fraction must be in the interval (0, 1].")
    if debug_min_size < 1:
        raise ValueError("--debug-min-size must be at least 1.")

    return debug_mode, debug_fraction, debug_min_size


def subset_dataset_for_debug(
    ds, label, dims_to_trim, debug_mode, debug_fraction, debug_min_size
):
    if not debug_mode:
        return ds

    indexers = {}
    for dim in dims_to_trim:
        if dim not in ds.sizes:
            continue
        size = ds.sizes[dim]
        keep = min(
            size, max(debug_min_size, int(np.ceil(size * debug_fraction)))
        )
        if keep < size:
            indexers[dim] = slice(0, keep)

    if not indexers:
        print(
            f"Debug mode active for {label}, but no matching"
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
            f"No valid positions left for {label} on dimension"
            f" '{dim}' with size {ds.sizes[dim]}."
        )
    if len(valid_positions) != len(positions):
        print(
            f"Adjusted {label} {dim} selection from {len(positions)}"
            f" to {len(valid_positions)} positions."
        )
    return ds.isel({dim: valid_positions})


def select_available_times(ds, requested_times, label, time_dim="time"):
    requested_index = pd.Index(np.unique(np.asarray(requested_times).ravel()))
    available_index = pd.Index(ds[time_dim].values)
    matching_times = requested_index.intersection(available_index)
    if matching_times.empty:
        raise ValueError(f"No overlapping times found for {label}.")
    if len(matching_times) != len(requested_index):
        print(
            f"Adjusted {label} time selection from"
            f" {len(requested_index)} to {len(matching_times)}"
            " available timestamps."
        )
    return ds.sel({time_dim: matching_times.values})


def filter_forecast_dataset_by_time_coverage(
    ds_forecast,
    ds_reference,
    label,
    forecast_time_dim="forecast_time",
    reference_time_dim="time",
    start_time_dim="start_time",
):
    available_times = pd.Index(ds_reference[reference_time_dim].values)
    forecast_times = ds_forecast[forecast_time_dim].values
    valid_mask = np.isin(forecast_times, available_times).all(axis=1)
    if not valid_mask.any():
        raise ValueError(
            f"No {label} start times have complete reference coverage."
        )
    if not valid_mask.all():
        print(
            f"Adjusted {label} {start_time_dim} selection from"
            f" {len(valid_mask)} to {int(valid_mask.sum())}"
            " with complete reference coverage."
        )
    valid_start_times = ds_forecast[start_time_dim].values[valid_mask]
    return ds_forecast.sel({start_time_dim: valid_start_times})


def align_on_common_coord(
    ds_left, ds_right, coord_name, left_label, right_label
):
    common_values = pd.Index(ds_left[coord_name].values).intersection(
        pd.Index(ds_right[coord_name].values)
    )
    if common_values.empty:
        raise ValueError(
            f"No overlapping {coord_name} values found between"
            f" {left_label} and {right_label}."
        )
    if len(common_values) != len(ds_left[coord_name]) or len(
        common_values
    ) != len(ds_right[coord_name]):
        print(
            f"Aligned {left_label} and {right_label} to"
            f" {len(common_values)} common {coord_name} values."
        )
    selected = common_values.values
    return ds_left.sel({coord_name: selected}), ds_right.sel(
        {coord_name: selected}
    )


def assert_time_coverage(ds_reference, requested_times, label, time_dim="time"):
    requested_index = pd.Index(np.unique(np.asarray(requested_times).ravel()))
    available_index = pd.Index(ds_reference[time_dim].values)
    missing = requested_index.difference(available_index)
    if not missing.empty:
        raise AssertionError(
            f"{label} is missing {len(missing)} required {time_dim} values."
        )


def resolve_requested_time(ds, dim, requested_time, label, allow_nearest=True):
    if requested_time is None:
        return None

    index = pd.Index(ds[dim].values)
    requested_timestamp = pd.Timestamp(requested_time)
    if requested_timestamp in index:
        return ds.sel({dim: requested_timestamp})[dim]

    if not allow_nearest:
        raise KeyError(
            f"{label} time {requested_time} not found in dimension '{dim}'."
        )

    nearest_position = index.get_indexer(
        [requested_timestamp], method="nearest"
    )[0]
    if nearest_position < 0:
        raise KeyError(
            f"{label} time {requested_time} not found in dimension '{dim}'."
        )

    nearest_value = index[nearest_position]
    print(
        f"Adjusted {label} time from {requested_timestamp} to"
        f" nearest available {nearest_value}."
    )
    return ds.sel({dim: nearest_value})[dim]
