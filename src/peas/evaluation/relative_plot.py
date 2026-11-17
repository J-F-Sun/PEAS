"""Compare per-position relative L2 errors on the paper evaluation window."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import isotonic_regression

from .probability_plot import OUT, ROOT, SCENES, STYLES
from .archive import read, errors, positions
AUDIT=OUT


COMPARISON = {
    "PEAS": ("PEAS", STYLES["PEAS"]["color"], "o", "-"),
    "AMS (mean of 128)": ("Monte-Carlo", STYLES["AMS (mean of 128)"]["color"], "s", "-."),
    "MCvD-Transformer": ("MCvD-Transformer", STYLES["MCvD-Transformer"]["color"], "^", "--"),
    "Frozen-PINN": ("Frozen-PINN", STYLES["Frozen-PINN"]["color"], "D", ":"),
}
ABLATIONS = {
    "full": ("Full PEAS", STYLES["PEAS"]["color"], "o", "-"),
    "no_reformulation": ("w/o reformulation", "#B64342", "X", ":"),
    "no_steady": ("w/o steady-state factorization", "#9A4D8E", "s", "-."),
    "no_features": ("w/o feature mapping", "#42949E", "^", "--"),
}
ABLATION_OUT = ROOT / "outputs/ablations/paper_t0_20"
SCENE_CAPTIONS = {
    "canonical": "unbounded domain",
    "eccentric": "spherical confinement",
    "pipe": "cylindrical confinement",
}


def load_scores(methods, *, comparison, ablation_version="current"):
    data={}
    for scene in SCENES:
        ids,dist,order=positions(scene);matrix=[];stds=[];mean_stds=[]
        for method in methods:
            d=read(scene,method,"comparison" if comparison else "ablations",version=ablation_version)
            e=errors(d["prediction"],d["reference"])[:,order]
            matrix.append(e.mean(0));stds.append(e.std(0,ddof=1) if len(e)>1 else np.zeros(len(ids)))
            mean_stds.append(e.mean(1).std(ddof=1) if len(e)>1 else 0.)
        matrix=np.array(matrix);data[scene]=(dist,matrix,matrix.mean(1),np.array(stds),np.array(mean_stds))
    return data


def add_means(ax, means, stds, methods, x, *, size):
    order = np.argsort(means, kind="stable")
    sorted_means = means[order]
    points_per_pixel = 72 / ax.figure.dpi
    heights = ax.transData.transform(np.column_stack((np.full(len(means), x), sorted_means)))[:, 1]
    heights *= points_per_pixel

    marker_spacing = np.sqrt(size) + .55
    breaks = np.flatnonzero(np.diff(heights) >= marker_spacing) + 1
    marker_offsets = np.zeros(len(means))
    for group in np.split(np.arange(len(means)), breaks):
        marker_offsets[group] = (np.arange(len(group)) - (len(group) - 1) / 2) * marker_spacing

    # Separate labels in display space while preserving their vertical order.
    label_spacing = 7.8 * np.arange(len(means))
    label_heights = isotonic_regression(heights - label_spacing).x + label_spacing
    text_x = 4.3 + marker_offsets.max()
    # Reserve the measured label width plus a 3-point right margin.
    from matplotlib.font_manager import FontProperties
    renderer = ax.figure.canvas.get_renderer()
    font = FontProperties(family="serif", size=6.0)
    labels = [f"{mean:.3f}" for mean in means]
    label_width = max(renderer.get_text_width_height_descent(label, font, False)[0]
                      for label in labels)
    reserved = label_width + (text_x + 3.0) / points_per_pixel
    left = ax.get_xlim()[0]
    assert reserved < ax.bbox.width
    ax.set_xlim(left, left + (x-left) * ax.bbox.width / (ax.bbox.width-reserved))
    points_per_x = ax.bbox.width * points_per_pixel / np.diff(ax.get_xlim()).item()
    styles = list(methods.values())
    for mean, method, marker_offset, label_offset in zip(sorted_means, order, marker_offsets, label_heights - heights):
        _, color, marker, _ = styles[method]
        marker_x = x + marker_offset / points_per_x
        ax.scatter(marker_x, mean, color=color, marker=marker,
                   s=size, linewidths=0, zorder=4)
        ax.annotate(f"{mean:.3f}", (x, mean), xytext=(text_x, label_offset), textcoords="offset points",
                    color=color, fontsize=6.0, va="center", ha="left", zorder=5)


def draw_panel(ax, data, methods, *, show_distance, stacked):
    from matplotlib.ticker import LogLocator, NullFormatter

    distances, values, means, point_stds, mean_stds = data
    x = np.arange(len(distances))
    mean_slot = len(distances) + 2.7
    marker_size = 3.55 if stacked else 2.85
    for style, point_errors in zip(methods.values(), values):
        label, color, marker, line = style
        width = 1.25 if label in {"PEAS", "Full PEAS"} else .82
        ax.plot(x, point_errors, color=color, marker=marker, ls=line,
                ms=marker_size, lw=width, mew=0, label=label, zorder=3)
    ax.axvline(len(distances) + .24, color="#9DA3AA", ls=(0, (2, 2)), lw=.53, zorder=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(values.min() * .66, values.max() * 1.46)
    ax.set_xlim(-.65, mean_slot + 12.8)
    ax.grid(axis="y", which="major", color="#B0B0B0", lw=.35, alpha=.45)
    ax.tick_params(axis="both", which="major", length=2.6, pad=2)
    ax.tick_params(axis="y", which="minor", length=1.4, color="#666666")
    ticks = np.array([0, 4, 8, 12, 16, 19] if stacked else [0, 6, 12, 19])
    labels = [f"{distances[i]:.2f}".rstrip("0").rstrip(".") for i in ticks]
    ax.set_xticks([*ticks, mean_slot], [*labels, "Avg."])
    ax.set_ylabel(r"$\text{Rel-}L^2$ (%)", labelpad=2)
    if show_distance:
        ax.set_xlabel(r"$d$", labelpad=1)
    add_means(ax, means, mean_stds, methods, mean_slot, size=20 if stacked else 15)


def render(plt, data, methods, path, *, stacked):
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.65)) if stacked else plt.subplots(1, 3, figsize=(7.15, 2.42))
    if stacked:
        fig.subplots_adjust(left=.158, right=.965, top=.91, bottom=.133, hspace=.65)
    else:
        fig.subplots_adjust(left=.087, right=.983, top=.88, bottom=.30, wspace=.32)
    for i, (scene, name) in enumerate(SCENES.items()):
        show_distance = not stacked or i == 2
        ax = axes[i]
        draw_panel(ax, data[scene], methods, show_distance=show_distance, stacked=stacked)
        ax.annotate(f"({chr(97 + i)}) {SCENE_CAPTIONS[scene]}", (.5, 0), xycoords="axes fraction",
                    xytext=(0, -30 if show_distance else -17), textcoords="offset points",
                    va="top", ha="center", fontsize=8, annotation_clip=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.52, .987), ncol=2 if stacked else 4,
               handlelength=1.6, handletextpad=.43, columnspacing=1.15,
               fontsize=7.45 if stacked else 8)
    if stacked:
        fig.align_ylabels(axes)
    layout = {"figure_size_inches": fig.get_size_inches().tolist(),
              "axes_width_height_points": [[ax.get_position().width*fig.get_figwidth()*72,
                                            ax.get_position().height*fig.get_figheight()*72] for ax in axes],
              "axes_bounds_figure_fraction": [list(ax.get_position().bounds) for ax in axes],
              "uncertainty": "none", "mean_label_decimals": 3}
    (AUDIT / f"layout_{path.name}.json").write_text(json.dumps(layout, indent=2)+"\n", encoding="utf-8")
    for extension in ("pdf", "png", "svg"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=360, facecolor="white",
                    bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


