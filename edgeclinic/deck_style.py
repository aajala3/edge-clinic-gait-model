"""Figure style matching the project deck ("The AI Edge presentation.pptx", theme "The School v1").

Colours are taken from the deck's theme and the colours its slides use; fonts follow the theme
(Georgia headings, Arial text). Content slides are white, 16:9 (13.33 x 7.5 in).

Use:
    with deck_style.context():
        fig, ax = plt.subplots(...)
        ...
        deck_style.save(fig, 'name')      # PNG 600 dpi + PDF + SVG, inside the context

Fonts are resolved when a file is written, so save inside the context.
Colour pairs used together were checked for colour-vision deficiency (OKLab distance under simulated
protan / deutan / tritan vision, Machado 2009): the marks of the predicted-vs-actual figure and the
Walk / Turn / Standing shading all clear a distance of 8 (normal vision >= 15).
New module: no cache key depends on it.
"""

from contextlib import contextmanager

import matplotlib.pyplot as plt

from . import paths

# ---- deck palette ---------------------------------------------------------------------------
DARK_TEAL = "#0E4F66"     # theme dk2: headings, primary data marks
NAVY = "#1F3A52"          # section headers, strong text
TEXT = "#404040"          # body text
TEXT_2 = "#5F6B76"        # secondary text, axis ticks
GREY = "#7A8590"          # reference lines, neutral marks
TEAL = "#5EC4B6"          # theme accent1
LIGHT_TEAL = "#9ED9D0"    # theme accent2
STEEL = "#4F8FA8"         # theme accent3
CORAL = "#E07A5F"         # theme accent6: highlight / fitted line
AMBER = "#FBAE40"         # deck highlight (use alone; too close to coral to pair with it)
MINT = "#DCF1EE"          # theme lt2: tints
CARD = "#F4F6F9"          # card background
RULE = "#D9D9D9"          # deck rules
GRID = "#E6E9ED"          # gridlines
WHITE = "#FFFFFF"         # slide background

EVENT_COLORS = {"Walk": TEAL, "Turn": CORAL, "Standing": GREY}

HEADING_FONT = "Georgia"
TEXT_FONT = "Arial"

RC = {
    "font.family": "sans-serif", "font.sans-serif": [TEXT_FONT, "DejaVu Sans"],
    "font.serif": [HEADING_FONT, "DejaVu Serif"],
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "axes.edgecolor": GREY, "axes.linewidth": 1.0, "axes.labelcolor": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 8, "ytick.major.pad": 8,
    "axes.labelsize": 18, "axes.labelpad": 12, "axes.titlesize": 18, "axes.titlecolor": DARK_TEAL,
    "legend.frameon": False, "legend.fontsize": 13, "legend.labelcolor": TEXT,
    "lines.solid_capstyle": "round",
    "svg.fonttype": "none", "pdf.fonttype": 42,
}


@contextmanager
def context():
    """Matplotlib settings for deck figures (draw and save inside it)."""
    with plt.rc_context(RC):
        yield


def title_block(fig, title: str, subtitle: str | None = None, x: float = 0.075, top: float = 0.965) -> None:
    """Figure title in the deck heading font (Georgia, dark teal) with an optional subtitle in Arial."""
    fig.text(x, top, title, fontsize=21, color=DARK_TEAL, ha="left", va="top", family="serif")
    if subtitle:
        fig.text(x, top - 0.045, subtitle, fontsize=14, color=TEXT_2, ha="left", va="top", linespacing=1.45)


def save(fig, name: str, folder=None) -> list:
    """Write <name>.png (600 dpi), .pdf and .svg (vector, editable text) to results/figures. Call inside context()."""
    folder = paths.FIGURES_DIR if folder is None else folder
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        p = folder / f"{name}.{ext}"
        fig.savefig(p, facecolor=WHITE, bbox_inches="tight", pad_inches=0.25, **kw)
        out.append(p)
    return out
