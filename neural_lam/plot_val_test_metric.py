# Standard library
import pickle
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# Configuration
# X-axis range (hours) applied to every plot.
LEAD_TIME_RANGE = (0, 48)

# Each entry: {"file": path, "color": <group-label>, "ls": <group-label>}
# color  → arbitrary group label (e.g. level count); same label = same color
# ls     → arbitrary group label (e.g. graph type);  same label = same linestyle
# Markers are always unique per model (no grouping needed).
#
# Colors are drawn in first-seen order from COLORS.
# Line styles are drawn in first-seen order from LINE_STYLES.
METRICS_FILES = {
    # 7° grid – rectangular graph
    "7° Rect HiLAM L2": {
        "file": "/home/moentu/VSCode/Python/neural-lam/7deg_rect_hi2_9966.pkl",
        "color": "L2",
        "ls": "rect",
    },
    "7° Rect HiLAM L3": {
        "file": "/home/moentu/VSCode/Python/neural-lam/7deg_rect_hi3_6182.pkl",
        "color": "L3",
        "ls": "rect",
    },
    "7° Rect HiLAM L4": {
        "file": "/home/moentu/VSCode/Python/neural-lam/7deg_rect_hi4_6457.pkl",
        "color": "L4",
        "ls": "rect",
    },
    "7° Rect GraphLAM L3": {
        "file": "/home/moentu/VSCode/Python/neural-lam/7deg_rect_ms3_4965.pkl",
        "color": "L3",
        "ls": "rect",
    },
    "7° Rect GraphLAM L4": {
        "file": "/home/moentu/VSCode/Python/neural-lam/7deg_rect_ms4_2431.pkl",
        "color": "L4",
        "ls": "rect",
    },
    # 7° grid – triangular graph
    "7° Tri HiLAM L3": {
        "file": "/home/moentu/VSCode/Python/neural-lam/7deg_tri_hi3_2904.pkl",
        "color": "L3",
        "ls": "tri",
    },
    "7° Tri HiLAM L4": {
        "file": "/home/moentu/VSCode/Python/neural-lam/7deg_tri_hi4_3092.pkl",
        "color": "L4",
        "ls": "tri",
    },
    "7° Tri GraphLAM L3": {
        "file": "/home/moentu/VSCode/Python/neural-lam/7deg_tri_ms3_0169.pkl",
        "color": "L3",
        "ls": "tri",
    },
}

VARIABLES = {
    # ------ COSMO ------
    # Surface / near-surface
    # "U_10M": "10m_u_component_of_wind",
    # "V_10M": "10m_v_component_of_wind",
    # "T_2M": "2m_temperature",
    # "PMSL": "mean_sea_level_pressure",
    # "PS": "surface_pressure",
    # "TOT_PREC": "total_precipitation",
    # "ASOB_S": "surface_net_shortwave_radiation",
    # "ATHB_S": "surface_net_longwave_radiation",
    # ------ DANRA / ERA5 (7deg pkl files) ------
    # Surface / near-surface
    "u10m": "10m U-wind",
    "v10m": "10m V-wind",
    "t2m": "2m Temperature",
    "pres_seasurface": "Mean Sea-Level Pressure",
    "pres0m": "Surface Pressure",
    "swavr0m": "Surface Net SW Radiation",
    "lwavr0m": "Surface Net LW Radiation",
    # # Upper-air: geopotential
    # "z100": "Z 100 hPa",
    # "z200": "Z 200 hPa",
    # "z400": "Z 400 hPa",
    "z600": "Z 600 hPa",
    # "z700": "Z 700 hPa",
    # "z850": "Z 850 hPa",
    # "z925": "Z 925 hPa",
    # "z1000": "Z 1000 hPa",
    # # Upper-air: temperature
    # "t100": "T 100 hPa",
    # "t200": "T 200 hPa",
    # "t400": "T 400 hPa",
    # "t600": "T 600 hPa",
    # "t700": "T 700 hPa",
    # "t850": "T 850 hPa",
    # "t925": "T 925 hPa",
    # "t1000": "T 1000 hPa",
    # # Upper-air: relative humidity
    # "r100": "RH 100 hPa",
    # "r200": "RH 200 hPa",
    # "r400": "RH 400 hPa",
    # "r600": "RH 600 hPa",
    # "r700": "RH 700 hPa",
    # "r850": "RH 850 hPa",
    # "r925": "RH 925 hPa",
    # "r1000": "RH 1000 hPa",
    # # Upper-air: U-wind
    # "u100": "U 100 hPa",
    # "u200": "U 200 hPa",
    # "u400": "U 400 hPa",
    # "u600": "U 600 hPa",
    # "u700": "U 700 hPa",
    # "u850": "U 850 hPa",
    # "u925": "U 925 hPa",
    # "u1000": "U 1000 hPa",
    # # Upper-air: V-wind
    # "v100": "V 100 hPa",
    # "v200": "V 200 hPa",
    # "v400": "V 400 hPa",
    # "v600": "V 600 hPa",
    # "v700": "V 700 hPa",
    # "v850": "V 850 hPa",
    # "v925": "V 925 hPa",
    # "v1000": "V 1000 hPa",
    # # Upper-air: vertical velocity (omega)
    # "tw100": "W 100 hPa",
    # "tw200": "W 200 hPa",
    # "tw400": "W 400 hPa",
    # "tw600": "W 600 hPa",
    # "tw700": "W 700 hPa",
    # "tw850": "W 850 hPa",
    # "tw925": "W 925 hPa",
    # "tw1000": "W 1000 hPa",
}

# Colorblind-friendly palette (Okabe-Ito / Wong Nature Methods 2011).
# Ordered for line plots on white: high-contrast colours first, yellow and
# black last (yellow near-invisible on white; black reserved for references).
COLORS = {
    "skyblue": "#56B4E9",  # Okabe-Ito 2 – sky blue, lighter
    "orange": "#E69F00",  # Okabe-Ito 1 – warm, high contrast
    "green": "#009E73",  # Okabe-Ito 3 – bluish green
    "red": "#D55E00",  # Okabe-Ito 6 – vermilion  (separates blue/green)
    "purple": "#CC79A7",  # Okabe-Ito 7 – reddish purple
    "blue": "#0072B2",  # Okabe-Ito 5 – deep blue, high contrast
    "brown": "#8C510A",  # extra (not in original 8)
    "grey": "#999999",  # utility
    "yellow": "#F0E442",  # Okabe-Ito 4 – last: near-invisible on white
    "black": "#000000",  # utility / reference lines
}

# Line styles that are distinguishable
LINE_STYLES = [
    "solid",  # ___________
    "dashed",  # - - - - - -
    "dashdot",  # -.-.-.-.-.-
    "dotted",  # .............
    (0, (3, 1)),  # ...  ...  ...
    (0, (5, 1)),  # ....    ....
    (0, (1, 1)),  # . . . . . .
    (0, (3, 1, 1, 1)),  # -..-..-..
    (0, (3, 1, 1, 1, 1, 1)),  # -..-..-.
    (0, (1, 2, 5, 2)),  # Complex dash
]

# Distinct markers
MARKERS = [
    "o",  # Circle
    "s",  # Square
    "D",  # Diamond
    "^",  # Triangle up
    "v",  # Triangle down
    "P",  # Plus (filled)
    "X",  # X (filled)
    "p",  # Pentagon
    "*",  # Star
    "h",  # Hexagon
]

FONT_SIZES = {
    "axes": 11,
    "ticks": 11,
    "legend": 10,
    "title": 11,
}

UNIT_LOOKUP = {
    "m s**-1": "m / s",
    "W m**-2": "W / m^2",
    "m**-2": "W / m^2",
    "m**2 s**-2": "m^2 / s^2",
}


def create_style_dict(metrics_files):
    """Assign styles from the group labels in each METRICS_FILES entry.

    - **Color**     : If all ``color`` group labels are numeric, a perceptually
                      uniform sequential colormap (viridis) is used so that
                      differences in value are visible as a gradient. Otherwise,
                      colors are drawn in first-seen order from COLORS.
    - **Linestyle** : same ``ls`` group label → same linestyle from LINE_STYLES,
                      assigned in first-seen label order.
    - **Marker**    : always unique per model, cycling through MARKERS.
    """
    color_values = list(COLORS.values())

    # Collect unique color group labels in first-seen order
    color_labels_seen = []
    for entry in metrics_files.values():
        cg = entry["color"]
        if cg not in color_labels_seen:
            color_labels_seen.append(cg)

    # If all color group labels are numeric, use a sequential gradient
    try:
        numeric_labels = sorted(color_labels_seen, key=lambda x: float(x))
        n = len(numeric_labels)
        # Sample viridis at evenly-spaced positions, avoiding the very dark end
        cmap = plt.get_cmap("viridis")
        gradient_colors = [
            cmap(0.15 + 0.7 * i / max(n - 1, 1)) for i in range(n)
        ]
        color_map = {
            lbl: gradient_colors[i] for i, lbl in enumerate(numeric_labels)
        }
    except (ValueError, TypeError):
        # Non-numeric labels: fall back to discrete palette
        color_map = {
            lbl: color_values[i % len(color_values)]
            for i, lbl in enumerate(color_labels_seen)
        }

    # Build linestyle map in first-seen order
    ls_map = {}
    for entry in metrics_files.values():
        lg = entry["ls"]
        if lg not in ls_map:
            ls_map[lg] = LINE_STYLES[len(ls_map) % len(LINE_STYLES)]

    styles = {}
    for i, (model_name, entry) in enumerate(metrics_files.items()):
        styles[model_name] = {
            "color": color_map[entry["color"]],
            "linestyle": ls_map[entry["ls"]],
            "marker": MARKERS[i % len(MARKERS)],
        }
    return styles


# Update plot_kwargs in plot_metrics function
def get_plot_kwargs(style, model_name, lead_time_hrs):
    """Get consistent plot kwargs for all plots"""
    # Indices where lead time falls exactly on a 6-hour boundary;
    # drop the last one so no marker appears flush at the right edge.
    mark_indices = [i for i, h in enumerate(lead_time_hrs) if float(h) % 6 == 0]
    if len(mark_indices) > 1:
        mark_indices = mark_indices[:-1]
    return {
        "label": model_name,
        "color": style["color"],
        "linestyle": style["linestyle"],
        "marker": style["marker"],
        "markersize": 4,
        "markevery": mark_indices or 1,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "linewidth": 1.5,
    }


def load_metrics(file_path):
    """Load metrics from pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_plot(fig, name, time=None, output_dir=None, plot_data=None):
    """Save plots to consistent location.

    Always produces two PDF variants:
      * ``<name>.pdf``            – with legend
      * ``<name>_no_legend.pdf``  – without legend

    When *plot_data* is supplied (a plain dict of str -> array) it is also
    persisted as ``<name>.npz`` for later programmatic access.
    """
    if time is not None:
        name = f"{name}_{time.dt.strftime('%Y%m%d_%H').values}"
    if output_dir is None:
        output_dir = "plots"

    # Accept either a Figure object or the pyplot module
    if hasattr(fig, "gcf"):
        fig = fig.gcf()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- version WITHOUT legend (default) ---
    # Temporarily hide every legend on every axes, save, then restore.
    legend_states = []
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is not None:
            legend_states.append((leg, leg.get_visible()))
            leg.set_visible(False)
    fig.savefig(output_path / f"{name}.pdf", bbox_inches="tight", dpi=300)
    for leg, was_visible in legend_states:
        leg.set_visible(was_visible)

    # --- version WITH legend ---
    fig.savefig(
        output_path / f"{name}_with_legend.pdf", bbox_inches="tight", dpi=300
    )

    # --- numerical data as .npz ---
    if plot_data is not None:
        np.savez(output_path / f"{name}.npz", **plot_data)

    plt.close()


def plot_metrics(
    metrics_files,
    metric_name="rmse",
    variables=None,
    combined=False,
    output_dir=None,
    wind_pair_vars=None,
    max_lead_time=None,
    legend_placement="best",
):
    """
    Unified plotting function with consistent styling

    wind_pair_vars is map from a variable name to two wind variables to
    derive it from, e.g. {"wv10m": ("u10m", "v10m")}.

    max_lead_time should be None (keep all lead times) or a np.timedelta64
    """
    plt.style.use("default")

    if wind_pair_vars is None:
        wind_pair_vars = {}
    else:
        # Make sure all of wind pair variables are also in variables list
        for wp_var in wind_pair_vars:
            if wp_var not in variables:
                variables.append(wp_var)

    metrics_dict = {
        model_name: load_metrics(entry["file"]).sel(
            lead_time=slice(None, max_lead_time)
        )
        for model_name, entry in metrics_files.items()
    }

    if output_dir is not None:
        Path(output_dir).mkdir(exist_ok=True)

    # Create style dictionary from the explicit group keys in metrics_files
    PLOT_STYLES = create_style_dict(metrics_files)

    if combined:
        n_cols = 2
        n_rows = (len(variables) + n_cols - 1) // n_cols
        _ = plt.figure(figsize=(15, 4 * n_rows))

    combined_plot_data = {}  # accumulates data for the combined .npz

    for idx, var in enumerate(variables, 1):
        if combined:
            ax = plt.subplot(n_rows, n_cols, idx)
        else:
            _, ax = plt.subplots(figsize=(5, 3), dpi=100)

        var_plot_data = {}  # per-variable data for .npz
        max_lead_time_hrs_var = 0.0
        all_var_y = []  # collect for tight y-limit

        for model_name, metrics in metrics_dict.items():
            lead_time_hrs = metrics.lead_time.dt.total_seconds() / 3600

            if var in wind_pair_vars:
                # Derive this metric value from pair of wind fields
                field1, field2 = wind_pair_vars[var]
                metric_values = np.sqrt(
                    metrics[metric_name].sel(variable=field1) ** 2
                    + metrics[metric_name].sel(variable=field2) ** 2
                )

                # Same unit as one of the fields above
                var_unit = str(
                    metrics["variable_units"].sel(variable=field1).values
                )
            else:
                metric_values = metrics[metric_name].sel(variable=var)
                var_unit = str(
                    metrics["variable_units"].sel(variable=var).values
                )

            if var_unit in UNIT_LOOKUP:
                var_unit = UNIT_LOOKUP[var_unit]

            lth = np.asarray(lead_time_hrs)
            mv = np.asarray(metric_values)
            # Clip data line at LEAD_TIME_RANGE[1]; xlim extends 2 h further
            # so the last point isn't flush against the frame.
            plot_mask = lth <= LEAD_TIME_RANGE[1]

            style = PLOT_STYLES[model_name]
            plot_kwargs = get_plot_kwargs(style, model_name, lth[plot_mask])

            ax.plot(
                lth[plot_mask],
                mv[plot_mask],
                **plot_kwargs,
            )

            y_mask = (lth >= LEAD_TIME_RANGE[0]) & plot_mask
            vals = mv[y_mask]
            all_var_y.extend(vals[~np.isnan(vals)].tolist())

            # Collect data for .npz export
            safe_key = model_name.replace(" ", "_").replace("/", "_")
            var_plot_data[f"{safe_key}_lead_times_h"] = np.asarray(
                lead_time_hrs
            )
            var_plot_data[f"{safe_key}_values"] = np.asarray(metric_values)
            max_lead_time_hrs_var = max(
                max_lead_time_hrs_var, float(lead_time_hrs.max())
            )

        # Merge into the combined pool (prefixed with variable name)
        for k, v in var_plot_data.items():
            combined_plot_data[f"{var}_{k}"] = v

        # --- tight y-limits (5 % pad above/below data range) ---
        if all_var_y:
            y_lo, y_hi = min(all_var_y), max(all_var_y)
            y_pad = (y_hi - y_lo) * 0.05
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        # --- 6-hour x-axis ticks (diurnal-cycle-friendly) ---
        tick_positions = np.arange(
            LEAD_TIME_RANGE[0], LEAD_TIME_RANGE[1] + 6, 6
        )
        ax.set_xticks(tick_positions)
        ax.set_xlim(LEAD_TIME_RANGE[0], LEAD_TIME_RANGE[1] + 2)

        # Common styling
        ax.set_xlabel("Lead Time (h)", fontsize=FONT_SIZES["axes"])
        if var_unit:
            ylabel = f"{metric_name.upper()} (${var_unit}$)"
        else:
            ylabel = f"{metric_name.upper()}"
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axes"])
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(
            axis="both", which="major", labelsize=FONT_SIZES["ticks"]
        )

        # For individual plots always show the legend; for combined only on the
        # first panel.
        if not combined or idx == 1:
            ax.legend(
                frameon=True,
                facecolor="white",
                edgecolor="black",
                fontsize=FONT_SIZES["legend"],
                loc=legend_placement,
            )

        if not combined:
            plt.tight_layout()
            save_plot(
                plt,
                f"{var}_{metric_name}",
                output_dir=output_dir,
                plot_data=var_plot_data,
            )

    if combined:
        plt.tight_layout()
        save_plot(
            plt,
            f"combined_{metric_name}",
            output_dir=output_dir,
            plot_data=combined_plot_data,
        )


def main():
    variables = list(VARIABLES.keys())
    print("Plotting metrics...")
    plot_metrics(
        METRICS_FILES, metric_name="rmse", variables=variables, combined=False
    )
    plot_metrics(
        METRICS_FILES, metric_name="rmse", variables=variables, combined=True
    )
    # MAE is also available in the metrics file


if __name__ == "__main__":
    main()
