"""Four publication figures assembled from immutable research results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.optimize import brentq

import analyze_theory_upgrade as theory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prl_manuscript_revision"
FIG = ROOT / "figures/prl_manuscript_revision"
COLORS = {"charge_density_wave": "#3068ab", "left": "#21866e", "right": "#c44956",
          "block1": "#8064b0", "block3": "#c1893f"}
NAMES = {"charge_density_wave": "CDW", "left": "Left", "right": "Right",
         "block1": r"Block $L/16$", "block3": r"Block $3L/16$"}
INPUTS = ["data/prl_p3_new_sizes/states.csv", "data/prl_theory_upgrade/widths.csv",
          "data/prl_theory_upgrade/roots.csv", "data/prl_theory_upgrade/reference_checks.csv",
          "data/prl_theory_upgrade/weak_skin_response.csv", "data/prl_theory_upgrade/spatial_response_checks.csv",
          "data/prl_theory_upgrade/summary.json", "scripts/analyze_theory_upgrade.py"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / (name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def kind(row):
    name = row["initial"]
    if name.startswith("block_cell"):
        return "block1" if int(name[10:]) == int(row["length"]) // 16 else "block3"
    return name


def style(axis):
    axis.set_facecolor("white")
    axis.grid(False, which="both")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.65)
        spine.set_color("#202124")
    axis.tick_params(which="major", direction="in", length=3.2, width=.65, pad=4,
                     top=True, right=True, color="#202124")
    axis.tick_params(which="minor", direction="in", length=1.8, width=.5, top=True, right=True)


def panel(axis, letter, y=-.52):
    axis.text(.5, y, f"({letter})", transform=axis.transAxes, ha="center",
              va="center", fontsize=9, clip_on=False)


def save(fig, stem):
    assert all(not axis.get_title() for axis in fig.axes)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / (stem + "." + ext), dpi=600, facecolor="white")
    plt.close(fig)


def cdw_projection_bound(time, skin):
    c = theory.constants(skin)
    decay = np.exp(-2 * c["a"] * time)
    k = c["rho"] * decay / (1 - c["b"] * c["rho"] / (2 * c["a"]) * (1 - decay))
    return k / np.sqrt(1 + k * k)


def occupations():
    fig = plt.figure(figsize=(3.4, 2.8))
    sketch = fig.add_axes((0, 0, 1, 1))
    sketch.set(xlim=(0, 12), ylim=(0, 9.88), aspect="equal")
    sketch.axis("off")
    textures = {}

    def sphere(x, y, radius, color, zorder=8):
        if color not in textures:
            # Orthographic surface normals give fixed, reproducible studio lighting.
            u, v = np.meshgrid(np.linspace(-1.015, 1.015, 160), np.linspace(-1.015, 1.015, 160))
            radial = np.sqrt(u * u + v * v)
            z = np.sqrt(np.clip(1 - radial * radial, 0, 1))
            normals = np.stack((u, v, z), axis=-1)
            light = np.array((-.43, .57, .70))
            light /= np.linalg.norm(light)
            halfway = light + np.array((0, 0, 1))
            halfway /= np.linalg.norm(halfway)
            diffuse = np.clip(normals @ light, 0, 1)
            specular = np.clip(normals @ halfway, 0, 1) ** 42
            shade = .23 + .61 * diffuse + .16 * z
            rgb = np.asarray(to_rgb(color)) * shade[..., None] + .52 * specular[..., None]
            image = np.dstack((np.clip(rgb, 0, 1), np.clip((1 - radial) * 80, 0, 1)))
            textures[color] = image
        for scale, alpha in ((1.4, .035), (1.0, .045)):
            sketch.add_patch(Ellipse((x + radius * .15, y - radius * .70),
                                    radius * 2.1 * scale, radius * .46 * scale,
                                    facecolor="#647e90", edgecolor="none", alpha=alpha, zorder=zorder - 2))
        sketch.imshow(textures[color], extent=(x - radius, x + radius, y - radius, y + radius),
                      origin="lower", interpolation="bilinear", zorder=zorder)

    def cylinder(start, end, width, color, zorder=3):
        sketch.plot(*zip(start, end), color="#637e91", lw=width + .5, solid_capstyle="round", zorder=zorder)
        sketch.plot(*zip(start, end), color=color, lw=width, solid_capstyle="round", zorder=zorder + .1)
        sketch.plot([start[0], end[0]], [start[1] + .035, end[1] + .035], color="white", alpha=.8,
                    lw=width * .28, solid_capstyle="round", zorder=zorder + .2)

    def arrow(start, end, color, width=.8, scale=7, **kwargs):
        sketch.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                         mutation_scale=scale, linewidth=width,
                                         color=color, **kwargs))

    a_color, b_color, loss_color = "#389ac3", "#d66c7a", "#b48b34"
    sketch.text(.18, 9.32, "Optical superlattice", fontsize=7.2, color="#344d5e", va="center")
    # Crossed lattice beams are apparatus elements, not a colored figure background.
    for origin, target in (((.65, 8.55), (4.70, 6.35)), ((3.25, 8.55), (1.10, 6.35))):
        ox, oy = origin
        tx, ty = target
        sketch.add_patch(Polygon([(ox-.12, oy), (ox+.12, oy), (tx+.30, ty), (tx-.30, ty)],
                                 facecolor="#389ac3", alpha=.10, edgecolor="none", zorder=0))
        sketch.plot([ox, tx], [oy, ty], color="#7bb6c8", lw=.55, alpha=.7, zorder=1)
        sketch.add_patch(Ellipse(origin, .55, .24, angle=-22 if ox < 1 else 32,
                                 facecolor="#e5eef3", edgecolor="#6e99ae", lw=.6, zorder=4))
    sketch.text(4.60, 5.00, r"$H_{\rm eff}=H-ic_*I$", ha="center", fontsize=7.5, color="#344d5e")

    sites = [(.80 + 2.30 * cell + .92 * sublattice,
              6.23 + .12 * cell + .05 * sublattice)
             for cell in range(4) for sublattice in range(2)]
    for cell in range(4):
        a, b = sites[2 * cell:2 * cell + 2]
        floor = [(a[0] - .41, a[1] - .24), (b[0] + .40, b[1] - .24),
                 (b[0] + .55, b[1] - .55), (a[0] - .26, a[1] - .55)]
        sketch.add_patch(Polygon(floor, closed=True, facecolor="#e5eef3", alpha=.5,
                                 edgecolor="#b7cbd7", linewidth=.45, zorder=0))
        cylinder(a, b, 3.4, "#baceda")
        if cell < 3:
            next_a = sites[2 * cell + 2]
            cylinder(b, next_a, 1.8, "#d4e0e7")
            for start, end, color, width in ((b, next_a, "#248aae", 1.2),
                                            (next_a, b, "#747f9a", .9)):
                sketch.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6,
                                                 connectionstyle="arc3,rad=-.65", shrinkA=7, shrinkB=7,
                                                 linewidth=width, color=color, zorder=6))
    for site, (x, y) in enumerate(sites):
        sphere(x, y, .31, a_color if site % 2 == 0 else b_color)
    sketch.text(1.23, 7.01, r"$t_1=1/2$", ha="center", fontsize=7, color="#344d5e")
    sketch.text(2.42, 7.11, r"$e^{g}$", ha="center", fontsize=7.5, color="#187b9f")
    sketch.text(2.42, 5.59, r"$e^{-g}$", ha="center", fontsize=7.5, color="#5b6381")
    sketch.text(.80, 5.58, r"$A_j$", ha="center", fontsize=7, color="#254b62")
    sketch.text(1.72, 5.58, r"$B_j$", ha="center", fontsize=7, color="#813d4b")
    sketch.text(9.02, 6.38, "OBC", fontsize=6.7, color="#536675")

    # All physical arrows describe particle loss; the positive imaginary site is a relative shift.
    for site in (3, 5):
        x, y = sites[site]
        arrow((x+.05, y+.35), (x+.45, 8.70), loss_color, width=.7, scale=6,
              linestyle=(0, (2.5, 2)), zorder=2)
    sketch.text(6.00, 9.32, "Site loss", ha="center", fontsize=7.2, color=loss_color, va="center")
    sketch.plot([4.45, 6.78], [8.79, 8.79], color=loss_color, lw=.55)

    # Number-resolved imaging objective and detector, in oblique view.
    sketch.add_patch(Polygon([(9.94, 7.24), (11.06, 7.24), (10.98, 7.99), (10.02, 7.99)],
                             facecolor="#e7eef2", edgecolor="#7d96a6", lw=.65, zorder=3))
    sketch.add_patch(Ellipse((10.5, 7.99), .96, .26, facecolor="#f6f9fa", edgecolor="#7d96a6", lw=.65, zorder=4))
    sketch.add_patch(Ellipse((10.5, 7.24), 1.12, .32, facecolor="#c7dee7", edgecolor="#6e99ae", lw=.7, zorder=4))
    sketch.add_patch(Ellipse((10.5, 7.24), .84, .18, facecolor="white", edgecolor="#7fa8ba", lw=.5, zorder=5))
    sketch.add_patch(Rectangle((9.97, 8.37), 1.06, .41, facecolor="#f4f7f9", edgecolor="#667f91", lw=.7))
    for pixel in range(5):
        sketch.add_patch(Rectangle((10.05+.18*pixel, 8.48), .11, .16, facecolor="#a6bccb", edgecolor="none"))
    sketch.plot([10.5, 10.5], [8.15, 8.37], color="#839aa9", lw=.6)
    for start, end in (((7.70, 6.97), (10.04, 7.23)), ((8.85, 6.82), (10.96, 7.23))):
        sketch.plot(*zip(start, end), color="#93aab8", lw=.55, ls=(0, (2, 2)), zorder=1)
    sketch.text(10.50, 9.32, r"$N_f=N$", ha="center", va="center", fontsize=8, color="#344d5e")
    sketch.text(10.50, 6.67, "Imaging", ha="center", va="center", fontsize=7, color="#344d5e")
    arrow((3.45, 3.07), (3.12, 5.90), "#7b8e9a", width=.75, scale=6,
          connectionstyle="arc3,rad=-.18", zorder=2)
    sketch.text(2.62, 4.08, "Load", ha="center", fontsize=7, color="#536675")

    order = ("charge_density_wave", "left", "right")
    for name, y in zip(order, (2.59, 1.77, .95)):
        sketch.text(.13, y, NAMES[name], va="center", fontsize=7.8, color=COLORS[name])
        xs = np.linspace(1.57, 5.31, 8)
        cylinder((xs[0], y), (xs[-1], y), .9, "#dae5ec", zorder=1)
        for site, x in enumerate(xs):
            occupied = site % 2 == 0 if name == "charge_density_wave" else site < 4 if name == "left" else site >= 4
            sphere(x, y, .14, COLORS[name] if occupied else "#e1e9ee")
        arrow((5.59, y), (6.12, y), "#7b8e9a", width=.6, scale=5.5)
    sketch.plot([6.19, 6.27, 6.27, 6.19], [2.59, 2.59, .95, .95], color="#a1b4c0", lw=.6)
    sketch.text(6.40, 1.77, r"$M(t)$", va="center", fontsize=7.8, color="#344d5e")

    aux_color = "#8064b0"
    ends, aux = ((8.04, 1.25), (10.88, 1.25)), (9.46, 3.59)
    for end in ends:
        cylinder(end, aux, 1.5, "#c3bad4", zorder=2)
    sphere(*ends[0], .25, a_color)
    sphere(*ends[1], .25, b_color)
    sphere(*aux, .24, aux_color)
    arrow((9.46, 3.85), (9.46, 4.38), loss_color, width=.8, scale=6.5)
    sketch.text(9.85, 4.27, r"$\kappa$", va="center", fontsize=8, color=loss_color)
    sketch.text(10.18, 4.27, "loss", va="center", fontsize=7, color=loss_color)
    sketch.text(9.97, 3.60, r"$d_j$", va="center", fontsize=8, color=aux_color)
    sketch.text(8.14, 2.44, r"$v_A$", ha="center", fontsize=8, color="#344d5e")
    sketch.text(10.77, 2.44, r"$v_B$", ha="center", fontsize=8, color="#344d5e")
    sketch.text(8.04, .54, r"$A_{j+1}$", ha="center", fontsize=7, color="#254b62")
    sketch.text(10.88, .54, r"$B_j$", ha="center", fontsize=7, color="#813d4b")
    for x, letter in ((3.6, "a"), (9.46, "b")):
        sketch.text(x, .02, f"({letter})", ha="center", va="bottom", fontsize=8)
    assert not sketch.get_title()
    fig.savefig(FIG / "fig1_occupations.pdf", facecolor="white")
    fig.savefig(FIG / "fig1_occupations.png", dpi=600, facecolor="white")
    plt.close(fig)
    return {"panels": 2, "schematic_sites": 8, "model_sites": 8,
            "model_unit_cells": 4, "intracell_bonds": 4, "intercell_bonds": 3,
            "initial_occupied_sites": {"CDW": [0, 2, 4, 6], "Left": list(range(4)), "Right": list(range(4, 8))},
            "model_labels": ["H_eff=H-ic_*I", "t1=1/2", "e^g", "e^-g", "OBC"],
            "apparatus": ["open optical superlattice", "site-selective particle loss", "number-resolved imaging and final N screening",
                          "phase-controlled Raman coupling to a common lossy auxiliary mode"],
            "shared_channel_phase": "v_B=i sgn(g) v_A; reservoir bandwidth kappa requires calibration",
            "physical_gain": False, "experimental_data": False,
            "rendering": "deterministic shaded atoms and auxiliary mode, vector optics, lenses and detector; white background",
            "figure_size_inches": [3.4, 2.8], "png_dpi": 600,
            "scope": "proposed apparatus and initial occupations, not built hardware or a finite-size simulation"}


def flagship(states, widths, export_inputs=True):
    selected = {row["initial"]: row for row in states
                if row["family_key"] == "L192_g0p25" and float(row["physical_time"]) == 93.125}
    order = ("charge_density_wave", "left", "right")
    assert set(selected) == set(order)
    target_entropy = float(selected["right"]["W_entropy"])
    assert all(abs(float(selected[name]["W_entropy"]) - target_entropy) < 1e-14 for name in order)
    ratio = float(selected["right"]["ratio"])
    rows = [row for row in widths if row["initial"] in ("left", "right")]
    if export_inputs:
        write_csv("fig2_same_protocol", [selected[name] for name in order])
        write_csv("fig2_selection_times", rows)
    else:
        assert read_csv("data/prl_manuscript_revision/fig2_same_protocol.csv") == [selected[name] for name in order]
        assert read_csv("data/prl_manuscript_revision/fig2_selection_times.csv") == rows
    bounds = {g: brentq(lambda t: cdw_projection_bound(t, g) - .05, 0, 20) for g in (0., .25)}

    markers = {"charge_density_wave": "o", "left": "s", "right": "D"}
    with plt.rc_context({"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 7.8,
                         "ytick.labelsize": 7.8, "text.color": "#202124"}):
        fig, (distance, entropy, times) = plt.subplots(1, 3, figsize=(7.1, 2.5),
                                                      gridspec_kw={"width_ratios": (1, 1, 1.16)})
        fig.subplots_adjust(left=.066, right=.986, top=.89, bottom=.245, wspace=.44)
        for axis in (distance, entropy, times):
            axis.set_facecolor("white")
            axis.grid(False, which="both")
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(.65)
                spine.set_color("#202124")
            axis.tick_params(which="major", direction="in", length=3.2, width=.65, pad=4,
                             top=True, right=True, color="#202124")
            axis.tick_params(which="minor", direction="in", length=1.8, width=.5, top=True, right=True)
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        # Categorical snapshot points preserve the full numerical range, including near-zero CDW.
        for position, name in enumerate(order):
            for axis, key in ((distance, "distance_to_W"), (entropy, "entropy")):
                value = float(selected[name][key])
                axis.vlines(position, 0, value, colors=COLORS[name], linewidth=1.0, zorder=2)
                axis.plot(position, value, marker=markers[name], ms=5.6, markeredgewidth=.7,
                          color=COLORS[name], linestyle="none", zorder=4)
        for axis in (distance, entropy):
            axis.set(xlim=(-.45, 2.45), xticks=np.arange(3), xticklabels=[NAMES[name] for name in order])
            axis.tick_params(axis="x", top=False, length=0, pad=6)
        distance.set(ylabel=r"$d(P,P_W)$", ylim=(-.045, 1.16), yticks=(0, .25, .5, .75, 1))
        distance.axhline(.05, color="#65696e", ls=(0, (1.3, 2)), lw=.85, zorder=1)
        distance.legend(handles=[Line2D([], [], color="#65696e", ls=(0, (1.3, 2)), lw=.85,
                                         label=r"$\epsilon=0.05$")], loc="upper left", bbox_to_anchor=(.015, .875),
                        frameon=False, fontsize=7.3, handlelength=1.8, handletextpad=.5, borderaxespad=.4)
        distance.text(.04, .14, r"$2.0\times10^{-107}$", transform=distance.transAxes,
                      ha="left", fontsize=7.4, color=COLORS[order[0]])
        distance.text(1, .142, "0.0504", ha="center", fontsize=7.6, color=COLORS[order[1]])
        distance.text(2, 1.044, "1.000", ha="center", fontsize=7.6, color=COLORS[order[2]])
        distance.text(.045, .96, r"$r_\sigma=3.70\times10^{-107}$", transform=distance.transAxes,
                      va="top", fontsize=7.8)

        entropy.set(ylabel=r"$S_A$ (nat)", ylim=(0, .355), yticks=(0, .1, .2, .3))
        entropy.axhline(target_entropy, color="#51565b", ls=(0, (4, 2.5)), lw=.9, zorder=1)
        entropy.legend(handles=[Line2D([], [], color="#51565b", ls=(0, (4, 2.5)), lw=.9,
                                        label=r"$S_A(P_W)$")], loc="upper left", frameon=False, fontsize=7.5,
                       handlelength=1.9, handletextpad=.5, borderaxespad=.5)
        for position, name in enumerate(order):
            value = float(selected[name]["entropy"])
            entropy.annotate(f"{value:.4f}", (position, value), xytext=(0, 7), textcoords="offset points",
                             ha="center", fontsize=7.6, color=COLORS[name])

        for name in ("left", "right"):
            for g, linestyle in (("0", "-"), ("0.25", (0, (4, 2.4)))):
                local = sorted([r for r in rows if r["initial"] == name and r["skin"] == g],
                               key=lambda r: int(r["length"]))
                times.plot([int(r["length"]) for r in local], [float(r["t05"]) for r in local],
                           color=COLORS[name], ls=linestyle, marker=markers[name], ms=4.2, lw=1.25,
                           markeredgewidth=.8, markerfacecolor=COLORS[name] if g == "0" else "white", zorder=3)
        times.set(xlabel=r"$L$", ylabel=r"$t_{0.05}$", xlim=(121, 263), ylim=(-2, 162),
                  xticks=(128, 192, 256), yticks=(0, 40, 80, 120, 160))
        times.xaxis.set_minor_locator(AutoMinorLocator(2))
        family_handles = [Line2D([], [], color=COLORS[name], marker=markers[name], lw=1.1, ms=4,
                                 label=NAMES[name]) for name in ("left", "right")]
        family_legend = times.legend(handles=family_handles, loc="upper left", frameon=False,
                                     fontsize=7.1, handlelength=1.7, handletextpad=.5, borderaxespad=.45)
        times.add_artist(family_legend)
        parameter_handles = [Line2D([], [], color="#51565b", marker="o", lw=1.1, ms=3.8,
                                    ls=linestyle, markerfacecolor=face, label=label)
                             for linestyle, face, label in (("-", "#51565b", r"$g=0$"),
                                                            ((0, (4, 2.4)), "white", r"$g=1/4$"))]
        times.legend(handles=parameter_handles, loc="lower right", bbox_to_anchor=(.98, .055),
                     frameon=False, fontsize=7.1, handlelength=2.0, handletextpad=.5, borderaxespad=.25)
        times.axhline(max(bounds.values()), color=COLORS[order[0]], lw=.95, zorder=1)
        times.text(126, 10.5, "CDW bound", fontsize=7.3, color=COLORS[order[0]])

        left, right = distance.get_position(), entropy.get_position()
        fig.text((left.x0 + right.x1) / 2, .962, r"$L=192,\quad g=1/4,\quad t=93.125$",
                 ha="center", va="center", fontsize=8)
        for axis, letter in zip((distance, entropy, times), "abc"):
            position = axis.get_position()
            fig.text((position.x0 + position.x1) / 2, .04, f"({letter})", ha="center", fontsize=9)
        assert all(not axis.get_title() for axis in fig.axes)
        fig.savefig(FIG / "fig2_geometry_memory.pdf", facecolor="white")
        fig.savefig(FIG / "fig2_geometry_memory.png", dpi=600, facecolor="white")
        plt.close(fig)
    return {"snapshot": {"length": 192, "skin": .25, "time": 93.125, "singular_ratio": ratio,
                          "right_input_overlap": float(selected["right"]["alpha"])},
            "cdw_G_sufficient_times": bounds, "panels": 3,
            "presentation": {"figure_size_inches": [7.1, 2.5], "png_dpi": 600,
                             "background": "white", "grid": False,
                             "snapshot_style": "points with thin stems, no bars or filled regions",
                             "palette": {NAMES[name]: COLORS[name] for name in order},
                             "initial_markers": {NAMES[name]: markers[name] for name in order},
                             "skin_encoding": "g=0: solid and filled; g=1/4: dashed and open",
                             "panel_label_position": "bottom, aligned figure coordinates"},
            "scope": "a,b use independent W-referenced actual states; c uses G threshold roots and analytic CDW G bound"}


def geometry(response, weak, export_inputs=True):
    if export_inputs:
        write_csv("fig3_reflected_geometry", response)
        write_csv("fig3_weak_skin", weak)
    else:
        assert read_csv("data/prl_manuscript_revision/fig3_reflected_geometry.csv") == response
        assert read_csv("data/prl_manuscript_revision/fig3_weak_skin.csv") == weak
    fig = plt.figure(figsize=(3.4, 3.3))
    axes = [fig.add_axes((.20, .67, .765, .305)), fig.add_axes((.20, .18, .765, .305))]
    markers = {"left": "s", "right": "D"}
    for name in ("left", "right"):
        for g, linestyle in ((0., "-"), (.25, (0, (4, 2.4)))):
            local = sorted([r for r in response if r["initial"] == name and float(r["skin"]) == g], key=lambda r: int(r["length"]))
            axes[0].plot([int(r["length"]) for r in local], [float(r["log_delta"]) / np.log(10) for r in local],
                         color=COLORS[name], ls=linestyle, marker=markers[name], ms=4.1, lw=1.2,
                         markeredgewidth=.8, markerfacecolor=COLORS[name] if g == 0 else "white")
    axes[0].set(ylabel=r"$\log_{10}\delta$", xlabel=r"$L$", xlim=(209, 527), ylim=(-414, -130),
                xticks=(224, 320, 512), yticks=(-400, -300, -200))
    family_handles = [Line2D([], [], color=COLORS[name], marker=markers[name], lw=1.1, ms=3.8,
                             label=NAMES[name]) for name in ("left", "right")]
    family_legend = axes[0].legend(handles=family_handles, loc="lower left", frameon=False,
                                   fontsize=7.1, handlelength=1.7, handletextpad=.5, borderaxespad=.55)
    axes[0].add_artist(family_legend)
    parameter_handles = [Line2D([], [], color="#51565b", marker="o", lw=1.0, ms=3.4,
                                ls=linestyle, markerfacecolor=face, label=label)
                         for linestyle, face, label in (("-", "#51565b", r"$g=0$"),
                                                        ((0, (4, 2.4)), "white", r"$g=1/4$"))]
    axes[0].legend(handles=parameter_handles, loc="upper right", frameon=False, fontsize=7.1,
                   handlelength=2.1, handletextpad=.5, borderaxespad=.6)
    for name in ("left", "right"):
        for chi, linestyle in ((2., "-"), (4., (0, (4, 2.4)))):
            local = sorted([r for r in weak if r["initial"] == name and float(r["chi"]) == chi], key=lambda r: int(r["length"]))
            axes[1].plot([int(r["length"]) for r in local],
                         [float(r["log_ratio_to_zero"]) / chi for r in local],
                         color=COLORS[name], ls=linestyle, marker=markers[name], lw=1.2,
                         ms=3.0 if chi == 2 else 4.6, markeredgewidth=.8,
                         markerfacecolor=COLORS[name] if chi == 2 else "none", zorder=3)
    for limit in (.5, -.5):
        axes[1].axhline(limit, color="#65696e", ls=(0, (1.3, 2)), lw=.8, zorder=1)
    response_handles = [Line2D([], [], color="#51565b", marker="o", lw=1.0, ms=size,
                               ls=linestyle, markerfacecolor=face, label=label)
                        for linestyle, face, size, label in (("-", "#51565b", 3.0, r"$\chi=2$"),
                                                             ((0, (4, 2.4)), "none", 4.6, r"$\chi=4$"))]
    response_handles.append(Line2D([], [], color="#65696e", ls=(0, (1.3, 2)), lw=.8,
                                   label=r"$L\to\infty$"))
    axes[1].legend(handles=response_handles, loc="center right", frameon=False, fontsize=7.1,
                   handlelength=2.1, handletextpad=.5, borderaxespad=.65)
    axes[1].set(xlabel=r"$L$", ylabel=r"$\ln[\delta(\chi/L)/\delta(0)]/\chi$",
                xlim=(209, 527), ylim=(-.62, .62), xticks=(224, 320, 512),
                yticks=(-.5, 0, .5), yticklabels=(r"$-1/2$", "0", r"$+1/2$"))
    for axis in axes:
        style(axis)
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.set_xlabel(axis.get_xlabel(), fontsize=8.5, labelpad=4)
        axis.set_ylabel(axis.get_ylabel(), fontsize=8.5, labelpad=6)
        axis.tick_params(labelsize=7.4)
    for axis, letter, y in zip(axes, "ab", (.535, .04)):
        position = axis.get_position()
        fig.text((position.x0 + position.x1) / 2, y, f"({letter})", ha="center", fontsize=9)
    save(fig, "fig3_reflection_response")
    return {"panels": 2, "lengths": [224, 256, 320, 384, 512],
            "presentation": {"figure_size_inches": [3.4, 3.3], "png_dpi": 600,
                             "background": "white", "grid": False,
                             "initial_markers": {NAMES[name]: markers[name] for name in markers},
                             "skin_encoding": "g=0: solid and filled; g=1/4: dashed and open",
                             "response_encoding": "chi=2: solid and filled; chi=4: dashed and open",
                             "asymptotic_limits": [-.5, .5], "reference_lines": "dotted",
                             "panel_label_position": "bottom, aligned figure coordinates"},
            "scope": "static reflected delta; not an absolute exponent or whitened overlap formula"}


def window(roots, references, widths, export_inputs=True):
    if export_inputs:
        write_csv("fig4_threshold_scans", roots)
        write_csv("fig4_independent_points", references)
        write_csv("fig4_widths", widths)
    else:
        assert read_csv("data/prl_manuscript_revision/fig4_threshold_scans.csv") == roots
        assert read_csv("data/prl_manuscript_revision/fig4_independent_points.csv") == references
        assert read_csv("data/prl_manuscript_revision/fig4_widths.csv") == widths
    fig = plt.figure(figsize=(3.4, 3.3))
    axes = [fig.add_axes((.20, .755, .765, .22)), fig.add_axes((.20, .18, .765, .335))]
    markers = {"left": "s", "right": "D", "block1": "^", "block3": "v"}
    groups = (("left", "0", "#555555"), ("left", "0.25", COLORS["left"]), ("right", "0.25", COLORS["right"]))
    centers = {(r["family_key"], r["initial"]): float(r["root_time"]) for r in roots if r["level"] == "0.05"}
    for name, g, color in groups:
        for length in (128, 160, 192, 224, 256):
            local = sorted([r for r in roots if r["initial"] == name and r["skin"] == g and int(r["length"]) == length],
                           key=lambda r: float(r["root_time"]))
            center = centers[(local[0]["family_key"], name)]
            axes[0].plot([float(r["root_time"]) - center for r in local], [float(r["level"]) for r in local],
                         color=color, alpha=.7, lw=.85, marker=markers[name], ms=2.7,
                         markeredgewidth=.55, markerfacecolor=color if g == "0" else "white",
                         ls="-" if g == "0" else (0, (4, 2.4)),
                         label=NAMES[name] + (r", $g=0$" if g == "0" else r", $g=1/4$") if length == 128 else None)
    independent = [r for r in references if r["initial"] in ("left", "right")]
    reference_points = axes[0].scatter(
        [float(r["duration"]) - centers[(r["family_key"], r["initial"])] for r in independent],
        [float(r["distance_to_G"]) for r in independent], s=12, facecolors="none", edgecolors="#202124",
        linewidths=.65, zorder=5)
    axes[0].axhline(.484, color="#65696e", ls=(0, (1.3, 2)), lw=.8, zorder=1)
    axes[0].text(.97, .38, r"$d=0.484$", transform=axes[0].transAxes, ha="right", va="center",
                 fontsize=6.8, color="#51565b")
    axes[0].set(xlabel=r"$t-t_{0.05}$", ylabel=r"$d(P,P_G)$", xlim=(-1.6, .75), ylim=(-.035, 1.03),
                xticks=(-1.5, -1, -.5, 0, .5), yticks=(0, .5, 1))
    group_legend = axes[0].legend(loc="upper right", frameon=False, fontsize=6.3,
                                   handlelength=1.9, handletextpad=.5, borderaxespad=.15, labelspacing=.15)
    axes[0].add_artist(group_legend)
    axes[0].legend(handles=[reference_points], labels=["Independent"], loc="lower left", frameon=False,
                   fontsize=6.5, handletextpad=.4, borderaxespad=.5)
    zoom = axes[1].inset_axes((.17, .33, .79, .405))
    for name in ("left", "right", "block1", "block3"):
        for g, linestyle in (("0", "-"), ("0.25", (0, (4, 2.4)))):
            local = sorted([r for r in widths if kind(r) == name and r["skin"] == g], key=lambda r: int(r["length"]))
            for axis, size in ((axes[1], 3.8), (zoom, 2.7)):
                axis.plot([int(r["length"]) for r in local], [float(r["width_30_to_01"]) for r in local],
                          color=COLORS[name], ls=linestyle, marker=markers[name], ms=size, lw=.85,
                          markeredgewidth=.6, markerfacecolor=COLORS[name] if g == "0" else "white")
    for g, linestyle in (("0", "-"), ("0.25", (0, (4, 2.4)))):
        bound = theory.width_bound(g)
        axes[1].axhline(bound, color="#51565b", ls=linestyle, lw=.85, zorder=1)
        label = r"$g=0$" if g == "0" else r"$g=1/4$"
        axes[1].text(256, bound + .055, label + f": {bound:.3f}", ha="right", fontsize=6.8)
    axes[1].set(xlabel=r"$L$", ylabel=r"$t_{0.01}-t_{0.3}$", xlim=(121, 263), ylim=(1.19, 2.72),
                xticks=(128, 192, 256), yticks=(1.5, 2, 2.5))
    zoom.set(xlim=(121, 263), ylim=(1.275, 1.32), xticks=(128, 192, 256), yticks=(1.28, 1.30, 1.32))
    style(zoom)
    zoom.tick_params(labelsize=6.0, length=2, width=.55, pad=2)
    zoom.tick_params(axis="x", labeltop=True, labelbottom=False)
    zoom.xaxis.set_minor_locator(AutoMinorLocator(2))
    zoom.yaxis.set_minor_locator(AutoMinorLocator(2))
    axes[1].indicate_inset_zoom(zoom, edgecolor="#92969b", linewidth=.45, alpha=.8)
    family_handles = [Line2D([], [], color=COLORS[name], marker=markers[name], lw=1.0, ms=3.5,
                             label=NAMES[name]) for name in markers]
    fig.legend(handles=family_handles, loc="upper center", bbox_to_anchor=(.58, .602), ncol=2,
               frameon=False, fontsize=6.8, handlelength=1.7, handletextpad=.5, columnspacing=1.3,
               borderaxespad=0, labelspacing=.25)
    for axis in axes:
        style(axis)
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.set_xlabel(axis.get_xlabel(), fontsize=8.5, labelpad=4)
        axis.set_ylabel(axis.get_ylabel(), fontsize=8.5, labelpad=6)
        axis.tick_params(labelsize=7.4)
    for axis, letter, y in zip(axes, "ab", (.635, .04)):
        position = axis.get_position()
        fig.text((position.x0 + position.x1) / 2, y, f"({letter})", ha="center", fontsize=9)
    save(fig, "fig4_continuous_window")
    return {"panels": 2, "independent_displayed_points": len(independent),
            "total_independently_bracketed_roots": 48,
            "presentation": {"figure_size_inches": [3.4, 3.3], "png_dpi": 600,
                             "background": "white", "grid": False, "filled_regions": False,
                             "initial_markers": {NAMES[name]: markers[name] for name in markers},
                             "skin_encoding": "g=0: solid and filled; g=1/4: dashed and open",
                             "contraction_boundary": .484, "contraction_boundary_style": "dotted line",
                             "width_bounds": {g: theory.width_bound(g) for g in ("0", "0.25")},
                             "width_inset_ylim": [1.275, 1.32], "inset_uses_same_data": True,
                             "panel_label_position": "bottom, aligned figure coordinates"},
            "scope": "six-threshold lines guide the eye; dotted boundary lies inside the proved cone; large angles diagnostic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model-only", action="store_true", help="Redraw Fig. 1 while preserving all data figures and CSV inputs")
    selection.add_argument("--flagship-only", action="store_true", help="Redraw Fig. 2 while preserving other figures and all CSV inputs")
    selection.add_argument("--mechanism-only", action="store_true", help="Redraw Figs. 3 and 4 while preserving Figs. 1 and 2 and all CSV inputs")
    args = parser.parse_args()
    strengthened = ROOT / "data/prl_figure_strengthening/summary.json"
    if not args.model_only and strengthened.exists() and json.loads(strengthened.read_text(encoding="utf-8"))["status"] == "complete":
        import plot_strengthened_figures
        targets = (["fig2_geometry_memory"] if args.flagship_only else
                   ["fig3_reflection_response", "fig4_continuous_window"] if args.mechanism_only else None)
        details=plot_strengthened_figures.render(targets)
        if (ROOT/"data/prl_priority_strengthening/time_response.csv").exists() and (targets is None or "fig3_reflection_response" in targets):
            import plot_priority_strengthening
            details.update(plot_priority_strengthening.render()["figures"])
        print(json.dumps(details, ensure_ascii=False))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                         "axes.linewidth": .7, "legend.fontsize": 7})
    before = {relative: digest(ROOT / relative) for relative in INPUTS}
    if args.model_only or args.flagship_only or args.mechanism_only:
        targets = (["fig3_reflection_response", "fig4_continuous_window"] if args.mechanism_only else
                   ["fig1_occupations" if args.model_only else "fig2_geometry_memory"])
        previous = json.loads((OUT / "figure_provenance.json").read_text(encoding="utf-8"))
        assert all(previous["source_inputs"].get(path) == value for path, value in before.items())
        assert all(digest(ROOT / row["path"]) == row["sha256"] for row in previous["outputs"]
                   if Path(row["path"]).stem not in targets)
        figures = previous["figures"]
        if args.mechanism_only:
            figures[targets[0]] = geometry(read_csv(INPUTS[5]), read_csv(INPUTS[4]), export_inputs=False)
            figures[targets[1]] = window(read_csv(INPUTS[2]), read_csv(INPUTS[3]), read_csv(INPUTS[1]), export_inputs=False)
        else:
            figures[targets[0]] = occupations() if args.model_only else flagship(read_csv(INPUTS[0]), read_csv(INPUTS[1]), export_inputs=False)
    else:
        figures = {
            "fig1_occupations": occupations(),
            "fig2_geometry_memory": flagship(read_csv(INPUTS[0]), read_csv(INPUTS[1])),
            "fig3_reflection_response": geometry(read_csv(INPUTS[5]), read_csv(INPUTS[4])),
            "fig4_continuous_window": window(read_csv(INPUTS[2]), read_csv(INPUTS[3]), read_csv(INPUTS[1]))
        }
    assert before == {relative: digest(ROOT / relative) for relative in INPUTS}
    receipt = {"status": "generated", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": digest(Path(__file__)), "source_inputs": {**previous["source_inputs"], **before} if args.model_only else before, "figures": figures,
               "outputs": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                           for stem in figures for p in (FIG / (stem + ".pdf"), FIG / (stem + ".png"))],
               "new_scientific_computations": False,
               "panel_titles": False, "panel_label_position": "below axes",
               "redrawn_figures": targets if args.model_only or args.flagship_only or args.mechanism_only else list(figures),
               "scope": "four-figure editorial redraw from frozen data; schematic occupations and analytic bounds explicitly identified"}
    (OUT / "figure_provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt["figures"], ensure_ascii=False))


if __name__ == "__main__":
    main()
