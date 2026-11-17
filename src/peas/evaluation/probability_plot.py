"""Draw the two paper layouts from the verified [0,20] comparison sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import t as student_t
from .archive import read, METHODS
from peas.common.paths import repository_root

ROOT = repository_root()
OUT = ROOT / "outputs/figures"
SCENES = {"canonical": "Unbounded", "eccentric": "Sph. conf.", "pipe": "Cylind. conf."}
# Closest archived distances to 4, 6 and 10, using the same points in both layouts.
POSITIONS = {"canonical": (3, 5, 9), "eccentric": (8, 10, 18), "pipe": (8, 11, 20)}
STYLES = {
    "Reference": {"color": "#262626", "ls": (0, (3.2, 2.4)), "lw": 0.92, "zorder": 6},
    "PEAS": {"color": "#007BFF", "ls": "-", "lw": 1.55, "zorder": 5},
    "AMS (mean of 128)": {"color": "#626B72", "ls": "-.", "lw": 0.94, "zorder": 3},
    "MCvD-Transformer": {"color": "#C24E38", "ls": "-", "lw": 0.84, "zorder": 3},
    "Frozen-PINN": {"color": "#884591", "ls": (0, (1.35, 1.35)), "lw": 1.2, "zorder": 4},
}
NEAR_MIDDLE_FAR = {4: "Near", 6: "Middle", 10: "Far"}


def load_curves():
    data,selection={},[]
    for scene in SCENES:
        grid=read(scene,"PEAS");config=json.loads((ROOT/f"configs/scenarios/{scene}.json").read_text())
        rel=grid["samples"]-config["domain"].get("rx_center",[0,0,0]);dist=np.linalg.norm(rel,axis=1)
        indexes=[list(grid["ids"]).index(f"{scene}_test_{number:03d}") for number in POSITIONS[scene]]
        curves={"Reference":grid["reference"][indexes]};intervals={}
        for method in METHODS:
            result=read(scene,method);selected=result["prediction"][:,indexes];mean=selected.mean(0)
            curves[method]=mean
            if "run_predictions" in result:selected=result["run_predictions"][:,indexes]
            half=student_t.ppf(.975,len(selected)-1)*selected.std(0,ddof=1)/np.sqrt(len(selected))
            intervals[method]=(mean-half,mean+half)
        data[scene]=(grid["times"],curves,rel[indexes],dist[indexes],intervals)
    return data,selection


def number(value):
    return f"{value:g}"


def render(plt, data, *, with_errors):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    columns = ((0, False), (0, True), (2, False), (2, True)) if with_errors else (
        (0, False), (1, False), (2, False))
    ncols = len(columns)
    fig, axes = plt.subplots(3, ncols, figsize=(7.15, 5.0), sharex=True, sharey="col")
    fig.subplots_adjust(left=.122, right=.985, bottom=.125, top=.905 if with_errors else .865,
                        hspace=.49, wspace=.53 if with_errors else .45)
    legend_labels = {"AMS (mean of 128)": "Monte-Carlo"}
    handles = [Line2D([], [], label=legend_labels.get(name, name), **style) for name, style in STYLES.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .99),
               ncol=5, handlelength=2.2, handletextpad=.45, columnspacing=1.2)
    error_ceilings = {1: 0., 3: 0.}

    for row, (scene, label) in enumerate(SCENES.items()):
        times, curves, positions, distances, intervals = data[scene]
        coord = axes[row, 0].get_position()
        fig.text(.040 if with_errors else .015, (coord.y0 + coord.y1) / 2, label, fontweight="bold",
                 rotation=90, va="center", ha="center")
        for col, (index, error) in enumerate(columns):
            ax = axes[row, col]
            reference = curves["Reference"][index]
            for method, style in STYLES.items():
                if error and method == "Reference":
                    continue
                values = curves[method][index]
                if error:
                    values = np.abs(values - reference)
                    error_ceilings[col] = max(error_ceilings[col], values.max())
                elif with_errors and method in intervals:
                    lower, upper = (bound[index] for bound in intervals[method])
                    ax.fill_between(times, lower, upper, color=style["color"], linewidth=0,
                                    alpha=.23 if method == "PEAS" else .15, zorder=2)
                ax.plot(times, values, **style)
            if error:
                ax.set_yscale("symlog", linthresh=1e-6, linscale=.65)
                ax.set_ylim(bottom=0, auto=True)
                ax.set_yticks([0, 1e-5, 1e-3, 1e-1] if index == 0 else [0, 1e-5, 1e-3],
                              labels=(["0", r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$"] if index == 0
                                      else ["0", r"$10^{-5}$", r"$10^{-3}$"]))
                ax.set_ylabel(r"$|\Delta q(t)|$", labelpad=2)
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.set_ylabel(r"$q(t)$", labelpad=2)
            ax.set_xlim(0, 20)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.tick_params(axis="x", labelbottom=row == 2)
            ax.grid(axis="y", color="#B0B0B0", lw=.38, alpha=.38)
            ax.set_axisbelow(True)
            if row == 0:
                title = NEAR_MIDDLE_FAR[(4, 6, 10)[index]]
                if with_errors:
                    title += " · Error" if error else " · Probability"
                ax.set_title(title, pad=5, fontweight="bold")
            if row == 2:
                ax.set_xlabel(r"$t$", labelpad=1)
            coords = ",".join(number(v) for v in positions[index])
            panel = chr(ord("a") + row * ncols + col)
            caption = rf"({panel}) $\mathrm{{TX}}=({coords}),\ d={distances[index]:.2f}$"
            ax.annotate(caption, (.5, 0), xycoords="axes fraction",
                        xytext=(0, -28 if row == 2 else -9), textcoords="offset points",
                        ha="center", va="top", fontsize=7.35, annotation_clip=False)

    if with_errors:
        for col, highest in error_ceilings.items():
            axes[0, col].set_ylim(0, highest * 1.24)
    stem = "exp1_probability_error_3x4" if with_errors else "experiment1_probability_3x3"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=360, facecolor="white",
                    **({"bbox_inches": "tight", "pad_inches": .025} if with_errors else {}))
    plt.close(fig)


