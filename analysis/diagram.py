"""Schematic of the stochastic looped transformer (figures/model_diagram.pdf)."""
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import style
from style import BLUE, INK, INK2, MAGENTA, ORANGE, AQUA, plt

FIG = Path(__file__).resolve().parents[1] / "figures"


def box(ax, x, y, w, h, text, fc, ec=INK2, fs=8, bold=False, left=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + (0.12 if left else w / 2), y + h / 2, text, ha="left" if left else "center", va="center",
            fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, col=INK2, style_="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style_, mutation_scale=9, color=col, lw=1.1,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def main():
    style.apply()
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    ax.set_xlim(0, 12.85)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    ax.grid(False)
    # inputs
    box(ax, 0.05, 3.05, 1.8, 0.6, "input tokens x\n(or graph vertices)", "#f0efec", fs=7.5)
    box(ax, 0.05, 1.55, 1.8, 0.6, "seed  r_t ~ N(0, I)\nfresh each loop", "#fbe3ec", fs=7.5)
    box(ax, 2.2, 3.05, 1.2, 0.6, "E(x)", "#f0efec")
    box(ax, 2.2, 1.55, 1.2, 0.6, "W_r r_t", "#fbe3ec")
    ax.add_patch(plt.Circle((4.1, 2.6), 0.22, fc="white", ec=INK2, lw=1.0))
    ax.text(4.1, 2.6, "+", ha="center", va="center", fontsize=12)
    arrow(ax, (1.8, 3.35), (2.2, 3.35)); arrow(ax, (1.8, 1.85), (2.2, 1.85))
    arrow(ax, (3.4, 3.35), (3.93, 2.75)); arrow(ax, (3.4, 1.85), (3.93, 2.45))
    # shared block
    box(ax, 4.6, 2.05, 2.15, 1.1, "Block_θ\nL layers, weight-tied\n+ LayerNorm", "#d8e6f8", fs=7.5)
    arrow(ax, (4.32, 2.6), (4.6, 2.6))
    box(ax, 7.0, 2.3, 0.9, 0.6, "H(t)", "#d8e6f8")
    arrow(ax, (6.75, 2.6), (7.0, 2.6))
    # loop back
    arrow(ax, (7.45, 2.3), (4.1, 2.38), col=BLUE, rad=-0.55)
    ax.text(5.75, 1.25, "loop  t = 1 … τ   (H(t-1) re-enters)", ha="center", fontsize=7.5, color=BLUE)
    # heads
    box(ax, 8.35, 3.15, 1.6, 0.55, "readout → ŷ_t", "#dff3ea")
    box(ax, 8.35, 1.55, 1.6, 0.75, "pool(sg H(t)) →\nhazard λ_t", "#fde8dc")
    arrow(ax, (7.9, 2.75), (8.35, 3.4)); arrow(ax, (7.9, 2.45), (8.35, 1.95))
    # right panels
    box(ax, 10.25, 2.55, 2.5, 1.55,
        "Training\n• τ ~ π_φ(·|x), loss at τ only\n• q-norm over m seeds\n• REINFORCE for φ\n• E[τ] = B (aug. Lagrangian)",
        "white", ec=MAGENTA, fs=6.8, left=True)
    box(ax, 10.25, 0.25, 2.5, 2.1,
        "Inference (compute counted)\n• arg-max / on-line median stop\n• sampled τ (Las Vegas)\n• depth vote: mode over t of ŷ_t\n   weighted by π(t|x)\n• seed vote / best-of-k\n   (k × compute)",
        "white", ec=ORANGE, fs=6.8, left=True)
    arrow(ax, (9.95, 3.4), (10.25, 3.4)); arrow(ax, (9.95, 1.9), (10.25, 1.6))
    ax.text(0.1, 0.55, "Same parameters for every variant (W_r and the stopping head always exist).\n"
            "Stochasticity: what to compute (seed r)  ×  when to stop (τ).",
            fontsize=7.5, color=INK2, va="center")
    style.save(fig, FIG / "model_diagram")


if __name__ == "__main__":
    main()
