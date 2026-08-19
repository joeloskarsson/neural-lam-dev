# Standard library
import importlib.util
import pickle
import re
from pathlib import Path

# Load plot_metrics directly from its file to avoid triggering
# neural_lam/__init__.py (which imports torch, lightning, etc.)
REPO_DIR = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "plot_val_test_metric",
    REPO_DIR / "neural_lam" / "plot_val_test_metric.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
plot_metrics = _mod.plot_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Contact Joel Oskarsson to get access to the files on his Google Drive
DESIGN_STUDY_DIR = Path(__file__).resolve().parent
METRICS_DIR = DESIGN_STUDY_DIR / "metrics_gdrive"
OUTPUT_BASE = DESIGN_STUDY_DIR / "plots"

# Key surface variables to plot per dataset type
DANRA_VARS = [
    "t2m",
    "u10m",
    "v10m",
    "pres_seasurface",
    "pres0m",
    "lwavr0m",
    "swavr0m",
    "z600",
]
COSMO_VARS = ["T_2M", "U_10M", "V_10M", "PS", "TOT_PREC", "ASOB_S", "ATHB_S"]

# Derived wind-speed variable (DANRA only)
DANRA_WIND_PAIRS = {"wv10m": ("u10m", "v10m")}

# Per-subfolder label configs (relative paths from the subfolder root).
# Subfolders not listed here get auto-generated labels from pkl filenames.
STUDY_CONFIGS = {
    "danra_boundary_handling": {
        "No future boundary": "metrics/no_future_boundary_2182.pkl",
        "Future boundary": "metrics/standard_6182.pkl",
        "Future boundary + Time emb.": "metrics/dynamic_time_delta_7784.pkl",
    },
    "danra_boundary_overlap": {
        "Overlap, IFS": "metrics/7deg_overlap_ifs_5059.pkl",
        "No overlap, IFS": "metrics/7deg_standard_ifs_2300.pkl",
        "Overlap, ERA5": "metrics/7deg_overlap_era_7347.pkl",
        "No overlap, ERA5": "metrics/7deg_standard_era_6182.pkl",
    },
    "danra_boundary_width": {
        "800 km": "metrics/7deg_standard_era_6182.pkl",
        "400 km": "metrics/3deg_standard_era_8882.pkl",
    },
    "danra_graphs": {
        "Rect. Hi. 2 lev.": "metrics/7deg_rect_hi2_9966.pkl",
        "Rect. Hi. 3 lev.": "metrics/7deg_rect_hi3_6182.pkl",
        "Rect. Hi. 4 lev.": "metrics/7deg_rect_hi4_6457.pkl",
        "Rect. M.S. 3 lev.": "metrics/7deg_rect_ms3_4965.pkl",
        "Rect. M.S. 4 lev.": "metrics/7deg_rect_ms4_2431.pkl",
        "Tri. Hi. 3 lev.": "metrics/7deg_tri_hi3_2904.pkl",
        "Tri. Hi. 4 lev.": "metrics/7deg_tri_hi4_3092.pkl",
        "Tri. M.S. 3 lev.": "metrics/7deg_tri_ms3_0169.pkl",
    },
    "danra_test_rmse": {
        "ML": "metrics/ml_model_4827.pkl",
    },
    # -------------------------------------------------------------------------
    # COSMO subfolders — absolute paths into wandb run directories
    # -------------------------------------------------------------------------
    "cosmo_no_finetune": {
        "7.19° - no finetune": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250304_111043-qvzebmz8/files/test_metrics.pkl"
        ),
    },
    "cosmo_boundary_width": {
        "no boundary": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135811-taoe2q20/files/test_metrics.pkl"
        ),
        "3.6°": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135710-5ikae3ta/files/test_metrics.pkl"
        ),
        "7.19°": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135748-rx3r2qc1/files/test_metrics.pkl"
        ),
        "7.19° with 3h subsample": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135759-liu0vlsa/files/test_metrics.pkl"
        ),
        "10.79°": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_194442-1mrvluca/files/test_metrics.pkl"
        ),
        "14.39°": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135712-d82t2arn/files/test_metrics.pkl"
        ),
    },
    "cosmo_boundary_handling": {
        "7.19°": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135748-rx3r2qc1/files/test_metrics.pkl"
        ),
        "7.19° -- time embed": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250228_193408-rbp9ta35/files/test_metrics.pkl"
        ),
        "7.19° - no future": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_203137-pevaufbw/files/test_metrics.pkl"
        ),
        "7.19 IFS": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250227_173313-wvauu4xx/files/test_metrics.pkl"
        ),
        "7.19 with interior IFS": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250227_173315-ogt0lmto/files/test_metrics.pkl"
        ),
        "7.19 IFS margin": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_122204-2po0e9pl/files/test_metrics.pkl"
        ),
        "7.19 IFS margin -- dynamic time embedding": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250228_193433-dokp0m5e/files/test_metrics.pkl"
        ),
        "7.19 IFS margin with interior": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_094539-ve7jxmni/files/test_metrics.pkl"
        ),
        "7.19 IFS margin with interior -- LR 0.0001": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_122208-9h4qffmp/files/test_metrics.pkl"
        ),
        "7.19 IFS margin with interior -- LR 0.0001 -- AR 12": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_122219-0c4wc8gs/files/test_metrics.pkl"
        ),
        "7.19 IFS margin with interior -- LR 0.0001 -- AR 6": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_122215-4euaxytl/files/test_metrics.pkl"
        ),
        "7.19 IFS margin -- LR 0.0001": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_122212-mllsk83b/files/test_metrics.pkl"
        ),
    },
    "cosmo_overlapping_interior": {
        "7.19°": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135748-rx3r2qc1/files/test_metrics.pkl"
        ),
        "7.19° with interior": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_094539-ve7jxmni/files/test_metrics.pkl"
        ),
    },
    "cosmo_finetuning_strategy": {
        "7.19°": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135748-rx3r2qc1/files/test_metrics.pkl"
        ),
        "LR 1e-4": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135732-szuf2x6a/files/test_metrics.pkl"
        ),
        "LR 1e-4 - AR 12": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135723-klfmyn8q/files/test_metrics.pkl"
        ),
        "LR 1e-4 - AR 6": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135727-bkrji8ip/files/test_metrics.pkl"
        ),
    },
    "cosmo_graphs": {
        "7.19°": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250223_135748-rx3r2qc1/files/test_metrics.pkl"
        ),
        "Triangular": (
            "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam"
            "/wandb/run-20250305_143223-sc0ind3i/files/test_metrics.pkl"
        ),
    },
}

# Regex to strip the 4-digit random hash suffix from pkl stems
_HASH_RE = re.compile(r"_\d{4}$")


def stem_to_name(stem: str) -> str:
    """Convert a pkl filename stem to a readable model label."""
    clean = _HASH_RE.sub("", stem)
    # Replace degree marker
    clean = clean.replace("7deg", "7°").replace("3deg", "3°")
    # Capitalise individual tokens but preserve degree symbol
    tokens = clean.replace("_", " ").split()
    return " ".join(
        t.upper() if t in ("era", "ifs") else t.capitalize() for t in tokens
    )


def color_group(stem: str) -> str:
    """Derive a color-group key from the stem (stripped of hash)."""
    clean = _HASH_RE.sub("", stem)
    # Graph entries: colour encodes graph architecture — four groups so that
    # rect-hi, tri-hi, rect-ms and tri-ms are each visually distinct.
    m = re.search(r"_(hi|ms)\d", clean)
    if m:
        layout = "tri" if "tri" in clean else "rect"
        return (
            f"{layout}_{m.group(1)}"  # "rect_hi", "tri_hi", "rect_ms", "tri_ms"
        )
    # Combine spatial resolution prefix with boundary driver so that e.g.
    # 3deg_*_era and 7deg_*_era get different colours (boundary-width study),
    # while same-resolution IFS vs ERA models still differ in colour.
    driver = "ifs" if "ifs" in clean else ("era" if "era" in clean else None)
    deg_m = re.match(r"(\d+deg)_", clean)
    if deg_m and driver:
        return f"{deg_m.group(1)}_{driver}"
    if driver:
        return driver.upper()
    return clean  # unique per model


def ls_group(stem: str) -> str:
    """Derive a linestyle-group key from the stem."""
    clean = _HASH_RE.sub("", stem)
    # Graph entries: linestyle encodes level count ("L2", "L3", "L4").
    m = re.search(r"_(hi|ms)(\d)", clean)
    if m:
        return f"L{m.group(2)}"
    if "rect" in clean:
        return "rect"
    if "tri" in clean:
        return "tri"
    # Boundary overlap
    if "overlap" in clean:
        return "overlap"
    # Spatial resolution
    if "3deg" in clean:
        return "3deg"
    # Boundary handling variants — each gets a distinct linestyle so that
    # no-future / standard / time-delta are visually separable without
    # relying solely on colour.
    if "no_future" in clean:
        return "no_future"
    if "dynamic" in clean or "time_delta" in clean:
        return "time_delta"
    return "default"


def build_metrics_files(pkl_files: list[Path]) -> dict:
    metrics_files = {}
    for pkl in sorted(pkl_files):
        name = stem_to_name(pkl.stem)
        metrics_files[name] = {
            "file": str(pkl),
            "color": color_group(pkl.stem),
            "ls": ls_group(pkl.stem),
        }
    return metrics_files


def build_metrics_files_from_labels(label_to_path: dict) -> dict:
    """Build metrics_files from {label: absolute_path}, deriving color/ls."""
    return {
        label: {
            "file": abs_path,
            "color": color_group(Path(abs_path).stem),
            "ls": ls_group(Path(abs_path).stem),
        }
        for label, abs_path in label_to_path.items()
    }


def detect_variables(pkl_path: Path) -> list[str]:
    """Return the list of variables available in a pkl file."""
    with open(pkl_path, "rb") as fh:
        ds = pickle.load(fh)
    return [str(v) for v in ds.variable.values]


def write_make_plots(
    output_dir: Path,
    metrics_files: dict,
    key_vars: list[str],
    wind_pairs: dict,
) -> None:
    """Write a self-contained make_plots.py to output_dir."""
    lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated by run_plots.py — reproduces plots here."""',
        "",
        "import importlib.util",
        "from pathlib import Path",
        "",
        "REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent",
        "_spec = importlib.util.spec_from_file_location(",
        '    "plot_val_test_metric",',
        '    REPO_DIR / "neural_lam" / "plot_val_test_metric.py",',
        ")",
        "_mod = importlib.util.module_from_spec(_spec)",
        "_spec.loader.exec_module(_mod)",
        "plot_metrics = _mod.plot_metrics",
        "",
        "METRICS_FILES = {",
    ]
    for name, entry in metrics_files.items():
        lines.append(
            f'    {name!r}: {{"file": {entry["file"]!r}, '
            f'"color": {entry["color"]!r}, "ls": {entry["ls"]!r}}},'
        )
    lines += [
        "}",
        "",
        f"VARIABLES = {key_vars!r}",
        f"WIND_PAIRS = {wind_pairs!r}",
        "",
        "OUTPUT_DIR = str(Path(__file__).resolve().parent)",
        "",
        "plot_metrics(",
        "    METRICS_FILES,",
        '    metric_name="rmse",',
        "    variables=VARIABLES,",
        "    wind_pair_vars=WIND_PAIRS,",
        "    combined=True,",
        "    output_dir=OUTPUT_DIR,",
        ")",
        "plot_metrics(",
        "    METRICS_FILES,",
        '    metric_name="rmse",',
        "    variables=VARIABLES,",
        "    wind_pair_vars=WIND_PAIRS,",
        "    combined=False,",
        "    output_dir=OUTPUT_DIR,",
        ")",
    ]
    (output_dir / "make_plots.py").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for subfolder in sorted(METRICS_DIR.iterdir()):
    if not subfolder.is_dir():
        continue

    pkl_files = sorted(subfolder.rglob("*.pkl"))
    if not pkl_files:
        continue

    print(f"\n{'=' * 60}")

    cfg = STUDY_CONFIGS.get(subfolder.name)
    if cfg is not None:
        label_to_path = {
            label: path
            if Path(path).is_absolute()
            else str((subfolder / path).resolve())
            for label, path in cfg.items()
        }
        metrics_files = build_metrics_files_from_labels(label_to_path)
        pkl_files = [Path(e["file"]) for e in metrics_files.values()]
        print(
            f"Study: {subfolder.name}  ({len(pkl_files)} model(s),"
            " labels from STUDY_CONFIGS)"
        )
    else:
        metrics_files = build_metrics_files(pkl_files)
        print(
            f"Study: {subfolder.name}  ({len(pkl_files)} model(s),"
            " labels auto-generated)"
        )

    # Detect dataset type from first pkl
    available = detect_variables(pkl_files[0])
    if "t2m" in available:
        key_vars = [v for v in DANRA_VARS if v in available]
        wind_pairs = DANRA_WIND_PAIRS
        dataset = "DANRA"
    else:
        key_vars = [v for v in COSMO_VARS if v in available]
        wind_pairs = {}
        dataset = "COSMO"

    if not key_vars:
        print("  No recognised surface variables found — skipping.")
        print(f"  Available sample: {available[:8]}")
        continue

    print(f"  Dataset : {dataset}")
    print(f"  Models  : {list(metrics_files.keys())}")
    print(f"  Vars    : {key_vars + list(wind_pairs.keys())}")

    output_dir = OUTPUT_BASE / subfolder.name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_metrics(
            metrics_files,
            metric_name="rmse",
            variables=key_vars,
            wind_pair_vars=wind_pairs,
            combined=True,
            output_dir=str(output_dir),
        )
        plot_metrics(
            metrics_files,
            metric_name="rmse",
            variables=key_vars,
            wind_pair_vars=wind_pairs,
            combined=False,
            output_dir=str(output_dir),
        )
        write_make_plots(output_dir, metrics_files, key_vars, wind_pairs)
        print(f"  -> {output_dir}/")
    except Exception as exc:
        # Standard library
        import traceback

        print(f"  ERROR: {exc}")
        traceback.print_exc()

print("\nDone.")
