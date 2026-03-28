# Standard library
import re
from pathlib import Path

# Third-party
import fitz
import matplotlib

matplotlib.use("Agg")

# Third-party
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Style constants (mirrors plot_val_test_metric.py)
# ---------------------------------------------------------------------------
FONT_SIZES = {"axes": 11, "ticks": 11, "legend": 10, "title": 11}

NEW_COLORS = [
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermilion  (separates blue/green)
    "#CC79A7",  # reddish purple
    "#0072B2",  # deep blue
    "#8C510A",  # brown
    "#999999",  # grey
    "#F0E442",  # yellow
    "#000000",  # black
]
LINE_STYLES = [
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    (0, (3, 1)),
    (0, (5, 1)),
    (0, (1, 1)),
    (0, (3, 1, 1, 1)),
]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "p", "*", "h"]

# ---------------------------------------------------------------------------
# Label-to-group helpers (mirror run_plots.py color_group / ls_group,
# but operate on the extracted legend text rather than pkl stems)
# ---------------------------------------------------------------------------


def label_color_group(label: str) -> str:
    """Derive a color-group key from a legend label string."""
    low = label.lower()
    if "ifs" in low:
        return "ifs"
    if "era" in low:
        return "era"
    return label  # unique key → sequential fallback


def label_ls_group(label: str) -> str:
    """Derive a linestyle-group key from a legend label string."""
    low = label.lower()
    if "no overlap" in low or "no-overlap" in low:
        return "no_overlap"
    if "overlap" in low:
        return "overlap"
    return label  # unique key → sequential fallback


# ---------------------------------------------------------------------------
# SVG path parsing
# ---------------------------------------------------------------------------


def _parse_ml_path(d: str) -> list[tuple[float, float]]:
    """
    Parse matplotlib's space-separated polyline: M x0 y0 x1 y1 x2 y2 …
    All numbers are treated as alternating x/y coordinates.
    """
    nums = [float(v) for v in re.findall(r"[-\d.eE+]+", d)]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _parse_legend_h_path(d: str) -> tuple[float, float, float] | None:
    """
    Parse a horizontal legend segment:  M x y H x1 [x2 …]
    Returns (x_start, x_end, y) in path coordinates, or None.
    """
    m = re.match(r"M\s*([\d.]+)\s+([\d.]+)\s*H([\s\d.]+)", d.strip())
    if not m:
        return None
    x0 = float(m.group(1))
    y0 = float(m.group(2))
    x_vals = [float(v) for v in m.group(3).split()]
    if not x_vals:
        return None
    return x0, x_vals[-1], y0


# ---------------------------------------------------------------------------
# Extract everything from one PDF
# ---------------------------------------------------------------------------


def _is_number(s: str) -> bool:
    try:
        float(s.replace("−", "-"))
        return True
    except ValueError:
        return False


def _axis_scale(positions: list[float], values: list[float]):
    p1, p2 = positions[0], positions[1]
    v1, v2 = values[0], values[1]
    scale = (p2 - p1) / (v2 - v1)
    origin = p1 - v1 * scale
    return origin, scale


def extract_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    page_h = page.rect.height
    svg = page.get_svg_image()

    # --- text blocks: full bbox (x0, y0, x1, y1, text), y=0 at top ---
    text_blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if txt:
                bb = line["bbox"]
                text_blocks.append((bb[0], bb[1], bb[2], bb[3], txt))

    doc.close()

    # --- axis tick labels ---
    # x-ticks: near bottom row (y_screen ≈ page_h − 37), x0 > 40
    # Use center-x of the label bbox as the tick mark position — avoids the
    # label-width bias where "0" is narrower than "10", "20", … which would
    # otherwise cause a growing rightward drift in extracted x-coordinates.
    x_tick_y_screen = page_h - 37
    x_ticks = sorted(
        ((x0 + x1) / 2, float(txt.replace("−", "-")))
        for x0, y0, x1, y1, txt in text_blocks
        if abs(y0 - x_tick_y_screen) < 15 and _is_number(txt) and x0 > 40
    )
    # y-ticks: near left margin (x0 < 40), not at x-tick row.
    # Use center-y of the label bbox as the tick mark position — same reasoning
    # as center-x for x-ticks: the top edge has a constant half-text-height
    # offset that shifts ALL extracted data_y values downward by ~h/(2*|scale|).
    y_ticks = sorted(
        ((y0 + y1) / 2, float(txt.replace("−", "-")))
        for x0, y0, x1, y1, txt in text_blocks
        if x0 < 40 and _is_number(txt) and abs(y0 - x_tick_y_screen) > 15
    )

    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError(f"Not enough ticks in {pdf_path.name}")

    # x: positions are screen-x pixels; values are hours
    x_orig, x_scale = _axis_scale(
        [t[0] for t in x_ticks], [t[1] for t in x_ticks]
    )
    # y: positions are screen-y pixels (0=top); values are RMSE
    # The y-tick pixel positions increase downward, data values increase upward
    # → scale will be negative (higher y_screen = lower RMSE)
    y_orig, y_scale = _axis_scale(
        [t[0] for t in y_ticks], [t[1] for t in y_ticks]
    )

    def svg_to_data(path_x, path_y):
        """Convert SVG path coords (y=0 at bottom) to data coords."""
        screen_x = path_x
        screen_y = page_h - path_y  # flip: SVG paths have y=0 at bottom
        data_x = (screen_x - x_orig) / x_scale
        data_y = (screen_y - y_orig) / y_scale
        return data_x, data_y

    # --- parse colored paths from SVG ---
    data_lines = []  # long polyline paths = RMSE curves
    legend_lines = []  # short H-command paths = legend samples

    for svg_line in svg.split("\n"):
        svg_line = svg_line.strip()
        if 'fill="none"' not in svg_line or "stroke=" not in svg_line:
            continue
        sw_m = re.search(r'stroke-width="([\d.]+)"', svg_line)
        if not sw_m or float(sw_m.group(1)) < 1.4:
            continue
        color_m = re.search(r'stroke="(#[0-9a-fA-F]{6})"', svg_line)
        if not color_m:
            continue
        color = color_m.group(1).lower()
        if color in ("#000000", "#b0b0b0", "#ffffff"):
            continue
        d_m = re.search(r'\bd="([^"]+)"', svg_line)
        if not d_m:
            continue
        d = d_m.group(1)
        dash_m = re.search(r'stroke-dasharray="([^"]+)"', svg_line)
        dash = dash_m.group(1) if dash_m else ""

        # Try legend (H command) first
        h_seg = _parse_legend_h_path(d)
        if h_seg is not None:
            x0, x1, path_y = h_seg
            screen_y = page_h - path_y
            x_centre = (x0 + x1) / 2.0
            legend_lines.append(
                {
                    "color": color,
                    "dash": dash,
                    "screen_y": screen_y,
                    "x_centre": x_centre,
                }
            )
            continue

        # Try data polyline (many space-separated coordinates)
        coords = _parse_ml_path(d)
        if len(coords) < 5:
            continue
        dxs, dys = zip(*[svg_to_data(px, py) for px, py in coords])
        data_lines.append(
            {
                "color": color,
                "dash": dash,
                "data_x": list(dxs),
                "data_y": list(dys),
            }
        )

    # --- axis labels ---
    ylabel = next(
        (txt for x0, y0, x1, y1, txt in text_blocks if x0 < 12), "RMSE"
    )
    xlabel = next(
        (txt for x0, y0, x1, y1, txt in text_blocks if "Lead Time" in txt),
        "Lead Time (hours)",
    )

    # --- legend: match legend_lines (by screen_y) to text blocks ---
    # Legend text blocks: not y-tick labels, not x-tick labels, not axis labels
    leg_texts = [
        (y0, txt)
        for x0, y0, x1, y1, txt in text_blocks
        if x0 > 40  # not left-margin y-tick
        and abs(y0 - x_tick_y_screen) > 10  # not bottom x-tick row
        and "Lead Time" not in txt
        and not (x0 < 12)  # not rotated y-label
    ]
    # Sort legend text top→bottom
    leg_texts.sort(key=lambda t: t[0])

    # Sort legend line segments top→bottom (screen_y=0 at top)
    legend_lines.sort(key=lambda ll: ll["screen_y"])

    # Match positionally: both lists are sorted top→bottom; zip avoids
    # proximity-threshold mis-assignments when entries are close together.
    label_map: dict[str, str] = {
        ll["color"]: leg_texts[i][1]
        for i, ll in enumerate(legend_lines)
        if i < len(leg_texts)
    }

    # Order data_lines by their first appearance in legend_lines
    color_order = {ll["color"]: idx for idx, ll in enumerate(legend_lines)}

    return {
        "data_lines": sorted(
            data_lines, key=lambda dl: color_order.get(dl["color"], 99)
        ),
        "label_map": label_map,
        "has_legend": len(legend_lines) > 0,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "x_max_data": max(t[1] for t in x_ticks),
        "y_tick_max": max(t[1] for t in y_ticks),
    }


# ---------------------------------------------------------------------------
# Replot with new style
# ---------------------------------------------------------------------------


def replot(extracted: dict, out_path: Path) -> None:
    data_lines = extracted["data_lines"]
    label_map = extracted["label_map"]

    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)

    # Build ordered, deduplicated group lists — same logic as
    # plot_val_test_metric.py.
    # Color tracks ERA5/IFS; linestyle tracks overlap/no-overlap.
    # Falls back to sequential assignment when neither keyword is present.
    labels_ordered = [
        label_map.get(dl["color"], f"Model {i + 1}")
        for i, dl in enumerate(data_lines)
    ]
    color_groups = list(
        dict.fromkeys(label_color_group(lbl) for lbl in labels_ordered)
    )
    ls_groups = list(
        dict.fromkeys(label_ls_group(lbl) for lbl in labels_ordered)
    )

    all_y = []
    for i, dl in enumerate(data_lines):
        label = labels_ordered[i]
        ci = color_groups.index(label_color_group(label))
        li = ls_groups.index(label_ls_group(label))
        color = NEW_COLORS[ci % len(NEW_COLORS)]
        ls = LINE_STYLES[li % len(LINE_STYLES)]
        marker = MARKERS[ci % len(MARKERS)]
        x = np.array(dl["data_x"])
        y = np.array(dl["data_y"])
        all_y.extend(y.tolist())

        # Markers at 6-hour gridline boundaries; drop the last so no marker
        # appears flush at the right edge.
        mark_idx = [j for j, xv in enumerate(x) if round(xv) % 6 == 0]
        if len(mark_idx) > 1:
            mark_idx = mark_idx[:-1]

        ax.plot(
            x,
            y,
            color=color,
            linestyle=ls,
            linewidth=1.5,
            marker=marker,
            markersize=4,
            markevery=mark_idx or 1,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=label,
        )

    # X-axis: 6-hour ticks, range 0 – min(data_max, 48)
    x_max = min(extracted["x_max_data"], 48)
    ax.set_xticks(np.arange(0, x_max + 6, 6))
    ax.set_xlim(0, x_max + 2)  # 2 h right margin so last data point isn't flush

    # Y-axis: data-driven lower bound; upper bound is the max of the extracted
    # data OR the original plot's highest tick label — whichever is larger.
    # This prevents the top from being clipped when pixel-extraction yields
    # values slightly below the true peak.
    if all_y:
        y_lo = min(all_y)
        y_hi = max(max(all_y), extracted["y_tick_max"])
        y_pad = (y_hi - y_lo) * 0.08
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.set_xlabel(extracted["xlabel"], fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(extracted["ylabel"], fontsize=FONT_SIZES["axes"])
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["ticks"])
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=FONT_SIZES["legend"],
        loc="upper left",
    )

    plt.tight_layout()

    # With-legend variant
    with_leg = out_path.with_name(
        out_path.stem + "_with_legend" + out_path.suffix
    )
    fig.savefig(with_leg, bbox_inches="tight", dpi=300)

    # Main file: no legend
    leg = ax.get_legend()
    if leg is not None:
        leg.set_visible(False)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Contact Joel Oskarsson to get access to the files on his Google Drive
COSMO_DIR = Path(__file__).parent / "metrics_gdrive"
PLOTS_DIR = Path(__file__).parent / "plots"

errors = []
for study_dir in sorted(COSMO_DIR.iterdir()):
    if not study_dir.is_dir() or not study_dir.name.startswith("cosmo_"):
        continue
    pdfs = sorted(
        p for p in study_dir.glob("*.pdf") if "_no_legend" not in p.stem
    )
    if not pdfs:
        continue
    out_dir = PLOTS_DIR / study_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 50}\nStudy: {study_dir.name}")

    # First pass: extract all data and find the study-wide labels
    # (only one file per study — typically T_2M — carries the legend)
    all_data: dict[Path, dict] = {}
    for pdf in pdfs:
        try:
            all_data[pdf] = extract_pdf(pdf)
        except (ValueError, RuntimeError, OSError) as e:
            errors.append((pdf, e))
            print(f"  {pdf.name} … ERROR (extract): {e}")

    ref_labels: list[str] = []
    for data in all_data.values():
        if data["has_legend"] and data["label_map"]:
            ref_labels = list(data["label_map"].values())
            break

    # Second pass: propagate study labels to all files, then replot
    for pdf, data in sorted(all_data.items()):
        print(f"  {pdf.name} … ", end="", flush=True)
        try:
            # If this file had no legend, assign ref_labels by line index
            if not data["has_legend"] and ref_labels:
                for i, dl in enumerate(data["data_lines"]):
                    if i < len(ref_labels):
                        data["label_map"][dl["color"]] = ref_labels[i]
            n = len(data["data_lines"])
            labels = list(data["label_map"].values())
            replot(data, out_dir / pdf.name)
            print(f"ok ({n} lines) labels={labels}")
        except (KeyError, ValueError, RuntimeError, OSError) as e:
            errors.append((pdf, e))
            print(f"ERROR: {e}")

if errors:
    print(f"\n{len(errors)} failures:")
    for p, e in errors:
        print(f"  {p.parent.name}/{p.name}: {e}")
else:
    print("\nAll PDFs restyled successfully.")
