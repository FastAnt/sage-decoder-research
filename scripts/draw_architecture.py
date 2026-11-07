#!/usr/bin/env python3
"""Draw the SAGE decoder and GPT-6 Astra's edits for one track.

Requires only Matplotlib and the Python standard library. Run from anywhere,
including from the repository root via ``python scripts/draw_architecture.py``.
The scripts/ entry is a symlink to this self-contained implementation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent

INK = "#263746"
MUTED = "#637381"
LINE = "#AAB5BE"
BLUE = "#477B9F"
RED = "#AD5555"
RED_PALE = "#FAEDED"
PATCH = "#76629B"
PATCH_PALE = "#F2EEF8"
PANEL = "#F7F9FB"


def draw_architecture():
    """Save an exactly 16 x 7 inch figure as PNG (200 dpi) and SVG."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "svg.fonttype": "none",
        "hatch.linewidth": 0.7,
        "axes.unicode_minus": True,
    })
    fig = plt.figure(figsize=(16, 7), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    def label(x, y, text, size=11, color=INK, weight="normal", ha="center"):
        return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                       ha=ha, va="center", zorder=6)

    def rounded(x, y, width, height, fill=PANEL, edge=LINE, radius=0.07):
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=1.0, facecolor=fill, edgecolor=edge, zorder=2,
        )
        ax.add_patch(patch)
        return patch

    def arrow(left, right):
        ax.add_patch(FancyArrowPatch(
            (left + 0.035, 3.55), (right - 0.035, 3.55),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.1,
            color=MUTED, zorder=4,
        ))

    def badge(x, y, width, text, size=8.5, height=0.25):
        rounded(x, y, width, height, fill=PATCH_PALE, edge=PATCH, radius=0.045)
        label(x + width / 2, y + height / 2, text, size=size, color=PATCH)

    def module(x, width, name, detail=None, size=10.5):
        rounded(x, 3.10, width, 0.90)
        label(x + width / 2, 3.64 if detail else 3.55, name, size=size,
              weight="medium")
        if detail:
            label(x + width / 2, 3.36, detail, size=9, color=MUTED)

    def stage(x, width, index, dim, count, per_block, removed):
        rounded(x, 2.72, width, 1.83, edge="#C1CCD5")
        label(x + width / 2, 4.28, f"stage {index}", size=12, weight="bold")
        label(x + width / 2, 4.02, f"dim {dim} · {count} blocks", size=9.5,
              color=MUTED)
        square = 0.40
        gap = 0.08 if count == 6 else 0.14
        row_width = count * square + (count - 1) * gap
        start = x + (width - row_width) / 2
        for block in range(count):
            bx = start + block * (square + gap)
            by = 3.35
            deleted = block in removed
            ax.add_patch(Rectangle(
                (bx, by), square, square, linewidth=1.0,
                facecolor=RED_PALE if deleted else BLUE,
                edgecolor=RED if deleted else BLUE,
                hatch="////" if deleted else None, zorder=3,
            ))
            # Keep the index above the square so the X cannot obscure it.
            label(bx + square / 2, 3.84, str(block), size=8.5, color=MUTED)
            if deleted:
                inset = 0.065
                ax.plot([bx + inset, bx + square - inset],
                        [by + inset, by + square - inset],
                        color=RED, linewidth=1.9, zorder=5)
                ax.plot([bx + inset, bx + square - inset],
                        [by + square - inset, by + inset],
                        color=RED, linewidth=1.9, zorder=5)
                label(bx + square / 2, 3.20, "removed", size=6.8, color=RED)
        label(x + width / 2, 2.92, f"{per_block} M / block", size=9, color=MUTED)

    label(0.40, 6.55, "SAGE decoder", size=23, weight="bold", ha="left")
    label(0.40, 6.13, "Architecture and edits by GPT-6 Astra · one track",
          size=13, color=MUTED, ha="left")
    label(0.40, 5.50, "Original decoder: 52.32 M parameters", size=12, ha="left")
    label(15.60, 5.50, "5 of 10 Swin-V2 blocks removed", size=12,
          color=RED, weight="medium", ha="right")

    # All nodes share a horizontal flow axis. Stage squares are geometrically
    # square because the drawing coordinates and figure dimensions match.
    module(0.35, 0.72, "input", "latent", size=10.5)
    label(0.71, 2.85, "16 channels", size=9, color=MUTED)
    module(1.30, 0.98, "input_proj")

    stages = [
        (2.50, 1.55, 0, 1024, 2, "13.664", {1}),
        (4.27, 3.25, 1, 512, 6, "3.425", {1, 4, 5}),
        (7.74, 1.55, 2, 256, 2, "0.861", {1}),
    ]
    for spec in stages:
        stage(*spec)

    label(5.895, 4.98, "3 Swin-V2 stages", size=11, color=MUTED)
    ax.plot([2.50, 2.50, 9.29, 9.29], [4.66, 4.76, 4.76, 4.66],
            color=LINE, linewidth=0.9, zorder=1)

    module(9.51, 1.15, "patch_expands", "2.62 M", size=9.5)
    module(10.88, 0.57, "norm", size=9.5)
    module(11.67, 1.27, "patch_unembed", "65 k", size=9.5)
    module(13.16, 1.22, "postnet", "10.8 k")
    label(13.77, 2.85, "residual", size=9, color=MUTED)
    label(13.77, 2.62, "4 → 64 → 4", size=9, color=MUTED)

    rounded(14.60, 3.10, 1.05, 0.90, fill="white")
    label(15.125, 3.80, "waveform", size=10)
    wave_x = [14.72 + i * 0.045 for i in range(19)]
    wave_y = [3.42 + v for v in
              [0, .04, -.03, .10, -.12, .07, -.04, .15, -.16,
               .09, -.06, .12, -.10, .05, -.04, .08, -.05, .02, 0]]
    ax.plot(wave_x, wave_y, color=BLUE, linewidth=1.3, zorder=4)

    nodes = [(0.35, 0.72), (1.30, 0.98), (2.50, 1.55), (4.27, 3.25),
             (7.74, 1.55), (9.51, 1.15), (10.88, 0.57), (11.67, 1.27),
             (13.16, 1.22), (14.60, 1.05)]
    for (x, width), (next_x, _) in zip(nodes, nodes[1:]):
        arrow(x + width, next_x)

    # These badges identify the supplied patch locations collectively, not an
    # invented assignment of the 14 gains or 4 rescalings to individual tensors.
    centers = [x + width / 2 for x, width, *_ in stages]
    for center in centers:
        ax.plot([center, center], [2.72, 2.40], color=PATCH, linewidth=0.9,
                zorder=1)
    ax.plot([centers[0], centers[-1]], [2.40, 2.40], color=PATCH, linewidth=0.9)
    ax.plot([centers[1], centers[1]], [2.40, 2.25], color=PATCH, linewidth=0.9)
    badge(centers[1] - 0.92, 2.00, 1.84, "gain × c   |   σ × c")
    label(centers[1], 1.82, "stage projections", size=9, color=PATCH)
    for x, width in [(9.51, 1.15), (11.67, 1.27)]:
        center = x + width / 2
        ax.plot([center, center], [3.10, 2.70], color=PATCH, linewidth=0.9,
                zorder=1)
        badge(x + 0.13, 2.45, width - 0.26, "gain × c")
        badge(x + 0.13, 2.13, width - 0.26, "σ × c")

    label(8, 1.40,
          "0.6 kB weight patch to the kept decoder: 14 per-tensor gain coefficients"
          " + 4 singular-value rescalings",
          size=11, weight="medium")

    # Legend: the same fills, hatching, X and badge outline used in the diagram.
    ax.add_patch(Rectangle((4.02, 0.87), 0.22, 0.22,
                           facecolor=BLUE, edgecolor=BLUE))
    label(4.36, 0.98, "Kept block", size=10, ha="left")
    ax.add_patch(Rectangle((6.03, 0.87), 0.22, 0.22, facecolor=RED_PALE,
                           edgecolor=RED, hatch="////"))
    ax.plot([6.065, 6.215], [0.905, 1.055], color=RED, linewidth=1.3)
    ax.plot([6.065, 6.215], [1.055, 0.905], color=RED, linewidth=1.3)
    label(6.37, 0.98, "Removed block", size=10, ha="left")
    badge(8.55, 0.855, 0.88, "gain × c", size=8)
    label(9.55, 0.98, "Patched weights (gain / σ)", size=10, ha="left")

    ax.plot([0.40, 15.60], [0.68, 0.68], color="#DCE2E7", linewidth=0.8)
    label(8, 0.36,
          "After removal: 27.52 M params (−47%)   ·   15.8 GFLOP/s (−45%)",
          size=13, weight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Do not use bbox_inches='tight': preserve the requested 16 x 7 inch canvas.
    fig.savefig(OUTPUT_DIR / "architecture.png", dpi=200, facecolor="white")
    fig.savefig(OUTPUT_DIR / "architecture.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw_architecture()
