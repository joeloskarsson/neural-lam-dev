"""
Centralised visual style for all verification scripts.

Change only ``BASE_FONT`` to scale every font in every figure uniformly.
Per-context scale factors (SCALE_*) adjust specific plot sections on top of
the base, and are expressed as multipliers so their effect is always legible.

Usage in any script::

    from plot_styles import (
        FONT_SIZES, DPI, FIG_W_FULL, FIG_W_MEDIUM,
        HSPACE_MAP, WSPACE_MAP, WSPACE_SPARSE,
        scaled_fonts, apply_style,
        SCALE_VERTICAL, SCALE_CASE_STUDY,
        SCALE_ERROR_COSMO, SCALE_ERROR_DANRA, SCALE_INTERP_DANRA,
        SCALE_METRICS, SCALE_MAPS_COSMO,
    )

Then call ``apply_style()`` once at the top of each script.
For section-specific overrides, wrap the relevant plotting calls with::

    plt.rcParams.update({...scaled_fonts(SCALE_X)...})
    plot_something(...)
    apply_style()          # restore base
"""

# Third-party
import matplotlib.pyplot as plt

# ── Primary control knob ──────────────────────────────────────────────────────
# Change this single value to scale every font proportionally across all
# verification scripts and figure types.
BASE_FONT: int = 20
# ─────────────────────────────────────────────────────────────────────────────

# Base font-size dict — used as rcParams throughout all scripts.
# All values derived from BASE_FONT so one edit propagates everywhere.
FONT_SIZES: dict = {
    "suptitle": BASE_FONT + 2,  # figure-level super-title
    "title": BASE_FONT + 1,  # per-axes title
    "axes": BASE_FONT,  # x/y axis labels
    "ticks": BASE_FONT - 2,  # tick labels
    "legend": BASE_FONT - 2,  # legend entries
    "cbar": BASE_FONT - 2,  # colour-bar label
    "stats": BASE_FONT - 3,  # in-plot stats / table text
}

# ── Per-context scale factors ─────────────────────────────────────────────────
# Multiply FONT_SIZES values by these before calling apply_style(scale=…).
# They are exposed here so a single edit in this file adjusts every script.
SCALE_VERTICAL = 1.20  # vertical-profile plots (slightly larger)
SCALE_SPECTRA = 1.10  # energy-spectra (+2 pt on top of base)
SCALE_WAVENUMBER = (
    1.15  # wavenumber-evolution lines (+3 pt on top of base, +2 vs spectra)
)
SCALE_METRICS = 1.05  # lead-time metric plots (+1 pt on top of base)
SCALE_MAPS_COSMO = (
    1.15  # COSMO physical-variable gridded maps (+3 pt on top of base)
)
SCALE_CASE_STUDY = 0.75  # obs case-study scatter maps (dense layout)
SCALE_ERROR_COSMO = 0.90  # COSMO obs-station error maps (-10 % of base)
SCALE_ERROR_DANRA = 1.20  # DANRA obs-station error maps (+60 % of ~20 pt)
SCALE_INTERP_DANRA = 1.60  # DANRA interpolated model-comparison maps (+60 %)

# ── Output resolution ─────────────────────────────────────────────────────────
DPI: int = 300

# ── Figure widths (inches) ────────────────────────────────────────────────────
# Designed to match typical paper column widths at the stated DPI.
FIG_W_FULL: float = 14.0  # three-column map grids, wide error maps
FIG_W_MEDIUM: float = 10.0  # spectra, metrics, vertical profiles
FIG_W_NARROW: float = 7.0  # single-panel or narrow two-column layouts

# ── Subplot spacing ───────────────────────────────────────────────────────────
HSPACE_MAP: float = 0.035  # gridded / error-map rows (tight for paper)
WSPACE_MAP: float = 0.050  # gridded map columns
WSPACE_SPARSE: float = 0.010  # sparse-comparison columns (narrowed)


# ── Helpers ───────────────────────────────────────────────────────────────────


def scaled_fonts(scale: float) -> dict:
    """Return a FONT_SIZES dict with every value multiplied by *scale*."""
    return {k: max(6, round(v * scale)) for k, v in FONT_SIZES.items()}


def _rcparams_from(fs: dict) -> dict:
    return {
        "font.size": fs["axes"],
        "axes.titlesize": fs["title"],
        "axes.labelsize": fs["axes"],
        "xtick.labelsize": fs["ticks"],
        "ytick.labelsize": fs["ticks"],
        "legend.fontsize": fs["legend"],
        "figure.titlesize": fs["suptitle"],
    }


def apply_style(scale: float = 1.0) -> None:
    """Apply base rcParams globally; optionally scale all fonts by *scale*."""
    plt.rcParams.update(_rcparams_from(scaled_fonts(scale)))
