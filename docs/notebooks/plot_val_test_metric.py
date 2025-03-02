import pickle
from pathlib import Path

import matplotlib.pyplot as plt

# Configuration
METRICS_FILES = {
    # "no boundary": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135811-taoe2q20/files/test_metrics.pkl",
    # "3.6 ERA margin": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135710-5ikae3ta/files/test_metrics.pkl",
    "7.19 ERA margin": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135748-rx3r2qc1/files/test_metrics.pkl",
    "7.19 ERA margin -- dynamic time embedding": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250228_193408-rbp9ta35/files/test_metrics.pkl",
    # "7.19 ERA margin - no future boundary": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_203137-pevaufbw/files/test_metrics.pkl",
    # "7.19 ERA margin with interior": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_094539-ve7jxmni/files/test_metrics.pkl",
    # "10.79 ERA margin": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_194442-1mrvluca/files/test_metrics.pkl",
    # "14.39 ERA margin": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135712-d82t2arn/files/test_metrics.pkl",
    # SUBSAMPLING
    # "7.19 ERA margin with 3h subsample": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135759-liu0vlsa/files/test_metrics.pkl",
    # FINETUNING
    # "7.19 ERA margin with interior - LR 1e-4": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135732-szuf2x6a/files/test_metrics.pkl",
    # "7.19 ERA margin with interior - LR 1e-4 - AR 12": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135723-klfmyn8q/files/test_metrics.pkl",
    # "7.19 ERA margin with interior - LR 1e-4 - AR 6": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135727-bkrji8ip/files/test_metrics.pkl",
    # IFS BOUNDARY
    # "7.19 IFS margin with interior": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122208-9h4qffmp/files/test_metrics.pkl",
    # "7.19 IFS margin with interior -- LR 0.0001": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122212-mllsk83b/files/test_metrics.pkl",
    # "7.19 IFS margin with interior -- LR 0.0001 -- AR 12": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122219-0c4wc8gs/files/test_metrics.pkl",
    # "7.19 IFS margin with interior -- LR 0.0001 -- AR 6": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122215-4euaxytl/files/test_metrics.pkl",
    "7.19 IFS margin": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122204-2po0e9pl/files/test_metrics.pkl",
    "7.19 IFS margin -- dynamic time embedding": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250228_193433-dokp0m5e/files/test_metrics.pkl",
    # IFS BOUNDARY BUGFIX
    "7.19 IFS margin BUGFIX time embed": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250227_173313-wvauu4xx/files/test_metrics.pkl",
    # "7.19 IFS margin with interior BUGFIX": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250227_173315-ogt0lmto/files/test_metrics.pkl",
}

VARIABLES = {
    "U_10M": "10m_u_component_of_wind",
    "V_10M": "10m_v_component_of_wind",
    "T_2M": "2m_temperature",
    "PMSL": "mean_sea_level_pressure",
    "PS": "surface_pressure",
    "TOT_PREC": "total_precipitation",
    "ASOB_S": "surface_net_shortwave_radiation",
    "ATHB_S": "surface_net_longwave_radiation",
}

# Colorblind-friendly palette (based on Wong's Nature Methods 2011 & Okabe-Ito)
COLORS = {
    "blue": "#56B4E9",  # Sky blue
    "orange": "#E69F00",  # Orange
    "green": "#009E73",  # Bluish green
    "red": "#D55E00",  # Vermillion
    "purple": "#CC79A7",  # Reddish purple
    "yellow": "#F0E442",  # Yellow
    "grey": "#999999",  # Grey
    "black": "#000000",  # Black
    "cyan": "#0072B2",  # Blue
    "brown": "#8C510A",  # Brown
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
    "*",  # Star
    "D",  # Diamond
    "^",  # Triangle up
    "P",  # Plus (filled)
    "X",  # X (filled)
    "v",  # Triangle down
    "p",  # Pentagon
    "h",  # Hexagon
]


def create_style_dict(metrics_dict):
    """Create style dictionary for experiments using distinct visual elements"""
    styles = {}
    for i, model_name in enumerate(metrics_dict.keys()):
        styles[model_name] = {
            "color": list(COLORS.values())[i % len(COLORS)],
            "linestyle": LINE_STYLES[i % len(LINE_STYLES)],
            "marker": MARKERS[i % len(MARKERS)],
        }
    return styles


# Update plot_kwargs in plot_metrics function
def get_plot_kwargs(style, model_name):
    """Get consistent plot kwargs for all plots"""
    return {
        "label": model_name,
        "color": style["color"],
        "linestyle": style["linestyle"],
        "marker": style["marker"],
        "markersize": 6,
        "markevery": 12,
        "markerfacecolor": "white",
        "markeredgewidth": 1.5,
        "linewidth": 2,
    }


OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_metrics(file_path):
    """Load metrics from pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_plot(fig, name, time=None):
    """Save plots to consistent location."""
    if time is not None:
        name = f"{name}_{time.dt.strftime('%Y%m%d_%H').values}"
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_metrics(
    metrics_dict, metric_name="rmse", variables=None, combined=False
):
    """Unified plotting function with consistent styling"""
    plt.style.use("default")

    # Create style dictionary based on number of experiments
    PLOT_STYLES = create_style_dict(metrics_dict)

    if combined:
        n_cols = 2
        n_rows = (len(variables) + n_cols - 1) // n_cols
        _ = plt.figure(figsize=(15, 4 * n_rows))

    for idx, var in enumerate(variables, 1):
        if combined:
            ax = plt.subplot(n_rows, n_cols, idx)
        else:
            _, ax = plt.subplots(figsize=(10, 6), dpi=100)

        for model_name, metrics in metrics_dict.items():
            lead_time_hrs = metrics.lead_time.dt.total_seconds() / 3600
            style = PLOT_STYLES[model_name]
            plot_kwargs = get_plot_kwargs(style, model_name)

            ax.plot(
                lead_time_hrs,
                metrics[metric_name].sel(variable=var),
                **plot_kwargs,
            )

        # Common styling
        ax.set_xlabel("Lead Time (hours)", fontsize=10 if combined else 12)
        ax.set_ylabel(f"{metric_name.upper()}", fontsize=10 if combined else 12)
        ax.set_title(
            f"{var}"
            if combined
            else f"{var} {metric_name.upper()} vs Forecast Lead Time",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(
            axis="both", which="major", labelsize=9 if combined else 10
        )

        if not combined or idx == 1:
            ax.legend(
                frameon=True, facecolor="white", edgecolor="black", fontsize=10
            )

        if not combined:
            plt.tight_layout()
            save_plot(plt, f"{var}_{metric_name}")

    if combined:
        plt.tight_layout()
        save_plot(plt, f"combined_{metric_name}")


def main():
    metrics_dict = {
        model_name: load_metrics(file_path)
        for model_name, file_path in METRICS_FILES.items()
    }

    variables = list(VARIABLES.keys())
    print("Plotting metrics...")
    # plot_metrics(
    #     metrics_dict, metric_name="rmse", variables=variables, combined=False
    # )
    plot_metrics(
        metrics_dict, metric_name="rmse", variables=variables, combined=True
    )
    # MAE is also available in the metrics file


if __name__ == "__main__":
    main()
