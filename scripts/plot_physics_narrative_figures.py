"""Expose physical time control using the frozen model and kernel records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import plot_prl_closure_upgrade as base

base.OUT = base.ROOT / "data/prl_physics_narrative_duel_20261013/figures"
LEFT, RIGHT = base.GREEN, base.RED


def point_time(row):
    return float(row.get("native_time") or row["kernel_time"])


def records_for(rows, *, side, bias=None, length=None, threshold=.05):
    return [row for row in rows if row["initial"] == side
            and float(row["level"]) == threshold
            and (bias is None or float(row["skin"]) == bias)
            and (length is None or int(row["length"]) == length)]


def legend_sides():
    return [Line2D([], [], marker=marker, color=color, ms=4.5, lw=.85, label=side)
            for side, color, marker in (("left", LEFT, "s"), ("right", RIGHT, "D"))]


def reflection_and_control(fixed):
    fig = plt.figure(figsize=(3.4, 4.2))
    mechanism = fig.add_axes((.13, .66, .83, .32))
    mechanism.set_xlim(-.13, 1.06)
    mechanism.set_ylim(-.05, 1.08)
    mechanism.axis("off")
    positions = np.linspace(0, .91, 400)
    shapes = (("$h<0$", base.RED, positions**2 * np.exp(-18 * positions), .08, r"$L^{-3}$"),
              ("$g=0$", base.BLACK, positions**2 * np.exp(-positions**2 / .067), .42, r"$L^{-3/2}$"),
              ("$h>0$", base.BLUE, np.exp(-(positions - .65)**2 / .014), .76, r"$L^{-1/2}$"))
    for label, color, packet, floor, power in shapes:
        mechanism.plot(positions, floor + .20 * packet / packet.max(), color=color, lw=1.3)
        mechanism.plot((0, .94), (floor, floor), color=base.BLACK, lw=.65)
        mechanism.plot((0, 0), (floor, floor + .23), color=base.BLACK, lw=.65)
        mechanism.text(-.105, floor + .11, label, rotation=90, ha="center", va="center", fontsize=8)
        mechanism.text(.965, floor + .12, power, ha="right", va="center", fontsize=8.3)
    mechanism.text(.92, 1.047, "schematic", ha="right", fontsize=8)
    fig.text(.545, .642, "distance from physical end, $j$", ha="center", fontsize=8.2)
    fig.text(.545, .597, "(a)", ha="center", fontsize=8.6)
    axis = fig.add_axes((.22, .15, .74, .405))
    base.axes_style(axis)
    metadata = []
    for side, color, marker in (("left", LEFT, "s"), ("right", RIGHT, "D")):
        for size in (1280, 4096):
            rows = sorted(records_for(fixed, side=side, length=size), key=lambda row: float(row["skin"]))
            if [float(row["skin"]) for row in rows] != [-.1, -.05, .05, .1]:
                raise ValueError("Directional panel needs all four frozen biases")
            xs, ys = [], []
            for row in rows:
                native = bool(row["native_time"])
                x, time = float(row["skin"]), point_time(row)
                xs.append(x)
                ys.append(time / size)
                axis.plot(x, time / size, marker=marker, ms=4.8, mec=color,
                          mfc=color if native else "white", mew=.95, ls="none", zorder=4)
                metadata.append(dict(initial=side, length=size, g=x, epsilon=.05,
                                     time=time, plotted_value=time / size, target="G_invar",
                                     evidence="original_model" if native else "finite_kernel_prediction"))
            axis.plot(xs, ys, color=color, lw=.85, ls="-" if size == 1280 else (0, (3, 2)))
    axis.set_xlabel("$g$")
    axis.set_ylabel(r"$t_{0.05}/L$")
    axis.set_xlim(-.112, .112)
    axis.set_ylim(.517, .568)
    axis.set_xticks((-.1, 0, .1))
    axis.set_yticks((.52, .54, .56))
    side_legend = axis.legend(handles=legend_sides(), loc="upper center", ncol=2, frameon=False,
                             borderpad=.15, handlelength=1.2, columnspacing=.8)
    axis.add_artist(side_legend)
    axis.legend(handles=[Line2D([], [], color=base.BLACK, marker="o", ms=4, lw=.8,
                               mfc=base.BLACK if size == 1280 else "white",
                               ls="-" if size == 1280 else (0, (3, 2)), label=f"$L={size}$")
                         for size in (1280, 4096)],
                loc="lower center", ncol=2, frameon=False, borderpad=.15,
                handlelength=1.3, columnspacing=.7)
    fig.text(.59, .019, "(b)", ha="center", fontsize=8.6)
    return base.save(fig, "fig3_reflection_response", dict(no_fit=True, data_points=metadata,
                     line_interpretation="Connections between four discrete frozen biases guide the eye.",
                     overlap_schematic_not_particle_density=True, h_definition="s*g, s=+1 left, -1 right",
                     numerical_derivative_test_claimed=False))


def wait_and_window(fixed, windows):
    fig = plt.figure(figsize=(3.4, 4.2))
    wait = fig.add_axes((.235, .64, .73, .34))
    width = fig.add_axes((.235, .15, .73, .315))
    for axis in (wait, width):
        base.axes_style(axis)
        axis.set_xlim(700, 4350)
        axis.set_xticks((1000, 2000, 4000))
        axis.set_xlabel("$L$")
    metadata, width_records = [], []
    for side, color, marker in (("left", LEFT, "s"), ("right", RIGHT, "D")):
        rows = sorted(records_for(fixed, side=side, bias=.1), key=lambda row: int(row["length"]))
        xs, ys = [], []
        for row in rows:
            size, time, native = int(row["length"]), point_time(row), bool(row["native_time"])
            xs.append(size)
            ys.append(time)
            wait.plot(size, time, marker=marker, ms=4.5, mec=color,
                      mfc=color if native else "white", mew=.9, ls="none", zorder=4)
            metadata.append(dict(initial=side, length=size, g=.1, epsilon=.05,
                                 target="G_invar", time=time,
                                 evidence="original_model" if native else "finite_kernel_prediction"))
        wait.plot(xs, ys, color=color, lw=.85)
        selected = sorted((row for row in windows if row["initial"] == side and float(row["skin"]) == .1),
                          key=lambda row: int(row["length"]))
        xs, ys = [], []
        for row in selected:
            size, native = int(row["length"]), bool(row["native_window"])
            value = float(row["native_window"] or row["kernel_window"])
            xs.append(size)
            ys.append(value)
            width.plot(size, value, marker=marker, ms=4.5, mec=color,
                       mfc=color if native else "white", mew=.9, ls="none", zorder=4)
            width_records.append(dict(initial=side, length=size, g=.1, lower_threshold=.01,
                                      upper_threshold=.05, target="G_invar", window=value,
                                      asymptotic_window=float(row["asymptotic_window"]),
                                      evidence="original_model" if native else "finite_kernel_prediction"))
        width.plot(xs, ys, color=color, lw=.8)
        width.axhline(float(selected[0]["asymptotic_window"]), color=color, ls=(0, (3, 2)), lw=.9)
    wait.set_ylim(300, 2450)
    wait.set_yticks((500, 1500, 2500))
    wait.set_ylim(300, 2600)
    wait.set_ylabel(r"$t_{0.05}$")
    wait.legend(handles=legend_sides(), loc="upper left", frameon=False, borderpad=.2,
                handlelength=1.2, labelspacing=.3)
    wait.text(.97, .04, "$g=0.1$", transform=wait.transAxes, ha="right", va="bottom", fontsize=8)
    width.set_ylim(.6069, .61005)
    width.set_yticks((.607, .608, .609, .610))
    width.set_ylabel(r"$t_{0.01}-t_{0.05}$")
    width.legend(handles=legend_sides(), loc="lower right", frameon=False, borderpad=.2,
                 handlelength=1.2, labelspacing=.3)
    fig.text(.60, .522, "(a)", ha="center", fontsize=8.6)
    fig.text(.60, .019, "(b)", ha="center", fontsize=8.6)
    return base.save(fig, "fig4_continuous_window", dict(no_fit=True, wait_data=metadata,
                     window_data=width_records, fixed_bias=.1,
                     line_interpretation="Sample connections; horizontal dashed lines are analytic window limits.",
                     fidelity_numeric_test_claimed=False))


def supplementary_checks(fixed, weak, theory):
    fig = plt.figure(figsize=(7.1, 2.8))
    center = fig.add_axes((.09, .23, .39, .72))
    response = fig.add_axes((.64, .23, .34, .72))
    for axis in (center, response):
        base.axes_style(axis)
    center.set_xlim(math.log(840), math.log(4400))
    xline = np.linspace(*center.get_xlim(), 200)
    centers = []
    for bias, color, marker in ((-.1, base.RED, "s"), (0., base.BLACK, "D"), (.1, base.BLUE, "o")):
        coefficient = theory[("left", bias, .05)]
        linear, constant, logarithm = (float(coefficient[field]) for field in
                                      ("linear_time_coefficient", "absolute_center", "log_time_coefficient"))
        center.plot(xline, logarithm * xline, color=color, lw=1.1)
        rows = records_for(fixed, side="left", bias=bias) if bias else [
            row for row in weak if row["initial"] == "left" and float(row["chi"]) == 0
            and float(row["level"]) == .05]
        for row in sorted(rows, key=lambda row: int(row["length"])):
            size = int(row["length"])
            native = bool(row["native_time"]) if bias else True
            time = point_time(row) if bias else float(row["actual_time"])
            value = time - linear * size - constant
            center.plot(math.log(size), value, marker=marker, ms=4.5, mec=color,
                        mfc=color if native else "white", mew=.9, ls="none")
            centers.append(dict(length=size, g=bias, initial="left", epsilon=.05, target="G_invar",
                                time=time, plotted_value=value, predicted=logarithm * math.log(size),
                                evidence="original_model" if native else "finite_kernel_prediction"))
    center.set_xlabel(r"$\log L$")
    center.set_ylabel(r"$t_{0.05}-TL-c_s$")
    center.set_xticks((7, 7.5, 8))
    center.set_ylim(-10, -.6)
    center.legend(handles=[Line2D([], [], marker=marker, ms=4, color=color, label=label)
                           for color, marker, label in ((base.RED, "s", "$h=-0.1$"),
                                                        (base.BLACK, "D", "$g=0$"),
                                                        (base.BLUE, "o", "$h=0.1$"))],
                  loc="center right", bbox_to_anchor=(.97, .42), frameon=False,
                  handlelength=1.2, borderpad=.2, labelspacing=.3)
    summaries = []
    for side, sign, marker in (("left", 1, "s"), ("right", -1, "D")):
        for size in (896, 1024, 1152, 1280):
            rows = [row for row in weak if row["initial"] == side and int(row["length"]) == size]
            if len(rows) != 9:
                raise ValueError("Nine weak-bias conditions are required")
            values = [math.sqrt(size) * (sign * float(row["actual_response"]) + 1 / (4 * base.MU0)) for row in rows]
            mean, low, high = float(np.mean(values)), min(values), max(values)
            offset = -8 if side == "left" else 8
            response.vlines(size + offset, low, high, color=base.GREEN, lw=.9)
            response.plot(size + offset, mean, color=base.GREEN, marker=marker, ms=4.5, ls="none")
            summaries.append(dict(initial=side, length=size, cases=9, mean=mean, minimum=low,
                                  maximum=high, range_is_deterministic=True, x_offset=offset,
                                  evidence="original_model", target="G_invar"))
    response.axhline(base.A, color=base.BLACK, lw=.9, ls=(0, (3, 2)))
    response.text(1260, base.A + .0003, "$A$", ha="right", va="bottom", fontsize=8)
    response.set_xlim(840, 1330)
    response.set_xticks((900, 1100, 1300))
    response.set_ylim(.308, .338)
    response.set_yticks((.31, .32, .33))
    response.set_xlabel("$L$")
    response.set_ylabel(r"$\sqrt{L}[s\partial_\chi t+1/(4\mu_0)]$")
    response.legend(handles=[Line2D([], [], color=base.GREEN, marker=marker, ls="none", ms=4, label=side)
                             for side, marker in (("left", "s"), ("right", "D"))],
                    loc="upper left", ncol=2, frameon=False, handlelength=.8, columnspacing=.8, borderpad=.1)
    fig.text(.285, .025, "(a)", ha="center", fontsize=8.6)
    fig.text(.80, .025, "(b)", ha="center", fontsize=8.6)
    value = base.save(fig, "fig_s_narrative_checks", dict(no_fit=True, center_data=centers,
                      response_summary=summaries, analytic_A=base.A,
                      range_is_not_statistical_uncertainty=True))
    value.update(width_inches=7.1, height_inches=2.8)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base.OUT)
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()
    base.OUT = args.output_dir.resolve()
    base.INSTALL = not args.no_install
    public_theory = base.ROOT / "data/prl_finite_bias_closure_20261013/public/theory_reference.json"
    if public_theory.is_file():
        base.THEORY = public_theory
    base.style()
    fixed = base.table("figure_fixed_records.csv")
    windows = base.table("figure_fixed_windows.csv")
    weak = base.table("figure_weak_records.csv")
    theory = base.load_theory()
    result = dict(figure3=reflection_and_control(fixed), figure4=wait_and_window(fixed, windows),
                  supplementary=supplementary_checks(fixed, weak, theory))
    for name, before in base.INPUTS.items():
        if base.digest(base.ROOT / name) != before:
            raise ValueError("A frozen scientific input changed: " + name)
    result.update(input_sha256=base.INPUTS, source_sha256=base.digest(Path(__file__)),
                  no_new_computations_or_fidelity_tests=True, no_fit=True,
                  line_interpretation="All nonhorizontal connections join saved discrete samples.")
    base.OUT.mkdir(parents=True, exist_ok=True)
    (base.OUT / "figure_narrative_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="pass", figures=3, frozen_inputs=len(base.INPUTS))))


if __name__ == "__main__":
    main()
