"""Draw manuscript-size panels from the frozen closure tables without fitting."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/prl_finite_bias_closure_20261013"
OUT = ROOT / "data/prl_closure_editorial_upgrade_20261013"
THEORY = ROOT / "research_doc/01_当前研究/有限尺寸与固定偏置闭合_20261013/推导与复核/fixed_bias/fixed_bias_predictions_reviewed.json"
INPUTS: dict[str, str] = {}
INSTALL = True
RED, BLUE, GREEN, BLACK = "#d40000", "#0048d8", "#008c35", "#111111"
MU0 = math.sqrt(7) / 2
A = .3196734354771544


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_key(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def table(name: str) -> list[dict]:
    path = DATA / "comparisons" / name
    INPUTS[input_key(path)] = digest(path)
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_theory() -> dict:
    INPUTS[input_key(THEORY)] = digest(THEORY)
    value = json.loads(THEORY.read_text(encoding="utf-8"))
    return {(row["initial"], float(row["g"]), float(row["epsilon"])): row for row in value["records"]}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.3,
                         "mathtext.fontset": "dejavusans", "axes.labelsize": 8.6,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "path",
                         "axes.linewidth": .7, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "lines.linewidth": 1.1})


def axes_style(axis) -> None:
    axis.set_facecolor("white")
    axis.grid(False)
    axis.tick_params(direction="in", top=True, right=True, length=3, width=.7, pad=3)
    for spine in axis.spines.values():
        spine.set_color("black")


def save(fig, name: str, metadata: dict) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.get_window_extent(renderer)
    escaped = []
    outside_ticks = set()
    for axis in fig.axes:
        for coordinate, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            for tick in coordinate.get_major_ticks() + coordinate.get_minor_ticks():
                if not min(limits) <= tick.get_loc() <= max(limits):
                    outside_ticks.update((tick.label1, tick.label2))
    for item in fig.findobj(match=Text):
        if item.get_text() and item.get_visible() and item not in outside_ticks:
            box = item.get_window_extent(renderer)
            if box.x0 < bounds.x0 - .5 or box.x1 > bounds.x1 + .5 or box.y0 < bounds.y0 - .5 or box.y1 > bounds.y1 + .5:
                escaped.append(item.get_text())
    if escaped:
        raise RuntimeError(f"Labels outside {name}: {escaped}")
    if any(axis.get_title() for axis in fig.axes):
        raise RuntimeError("Panel titles are disallowed")
    files = []
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        path = OUT / f"{name}.{extension}"
        meta = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        fig.savefig(path, dpi=420, metadata=meta)
        files.append(dict(path=input_key(path), sha256=digest(path), bytes=path.stat().st_size))
        for folder in (() if not INSTALL else (ROOT / "figures/prl_manuscript_revision", ROOT / "main/figures/prl_manuscript_revision",
                       ROOT / "main_zh/figures/prl_manuscript_revision", ROOT / "supplement/figures/prl_manuscript_revision",
                       ROOT / "supplment_zh/figures/prl_manuscript_revision")):
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, folder / path.name)
    plt.close(fig)
    return dict(files=files, width_inches=3.4, height_inches=4.2,
                minimum_label_pt=8, white_background=True, no_panel_titles=True,
                no_gradients_shadows_glows=True, panel_letters_below_each_panel=True,
                labels_within_canvas=True, **metadata)


def centers(fixed: list[dict], weak: list[dict], theory: dict) -> dict:
    fig = plt.figure(figsize=(3.4, 4.2))
    mechanism = fig.add_axes((.13, .66, .83, .32))
    mechanism.set_xlim(-.13, 1.06)
    mechanism.set_ylim(-.05, 1.08)
    mechanism.axis("off")
    positions = np.linspace(0, .91, 400)
    shapes = (("$h<0$", RED, positions**2 * np.exp(-18 * positions), .08, r"$L^{-3}$"),
              ("$g=0$", BLACK, positions**2 * np.exp(-positions**2 / .067), .42, r"$L^{-3/2}$"),
              ("$h>0$", BLUE, np.exp(-(positions - .65)**2 / .014), .76, r"$L^{-1/2}$"))
    for label, color, packet, base, power in shapes:
        mechanism.plot(positions, base + .20 * packet / packet.max(), color=color, lw=1.3)
        mechanism.plot((0, .94), (base, base), color=BLACK, lw=.65)
        mechanism.plot((0, 0), (base, base + .23), color=BLACK, lw=.65)
        mechanism.text(-.105, base + .11, label, rotation=90, ha="center", va="center", fontsize=8)
        mechanism.text(.965, base + .12, power, ha="right", va="center", fontsize=8.3)
    mechanism.text(.92, 1.047, "schematic", ha="right", fontsize=8)
    fig.text(.545, .642, "distance from physical end, $j$", ha="center", fontsize=8.2)
    fig.text(.545, .597, "(a)", ha="center", fontsize=8.6)
    axis = fig.add_axes((.22, .15, .74, .405))
    axes_style(axis)
    axis.set_xlim(math.log(840), math.log(4400))
    xline = np.linspace(*axis.get_xlim(), 250)
    plotted = []
    for skin, color, marker, label in ((-.1, RED, "s", "$h=-0.1$"), (0., BLACK, "D", "$g=0$"),
                                        (.1, BLUE, "o", "$h=0.1$")):
        coeff = theory[("left", skin, .05)]
        linear, constant, logarithm = (float(coeff[field]) for field in ("linear_time_coefficient", "absolute_center", "log_time_coefficient"))
        axis.plot(xline, logarithm * xline, color=color, lw=1.1)
        source = [row for row in fixed if row["initial"] == "left" and float(row["skin"]) == skin and float(row["level"]) == .05] if skin else [row for row in weak if row["initial"] == "left" and float(row["chi"]) == 0 and float(row["level"]) == .05]
        source = sorted(source, key=lambda row: int(row["length"]))
        for row in source:
            size = int(row["length"])
            original = bool(row.get("native_time")) if skin else True
            time = float(row.get("native_time") or row.get("kernel_time")) if skin else float(row["actual_time"])
            y = time - linear * size - constant
            axis.plot(math.log(size), y, marker=marker, ms=4.5, color=color, mec=color,
                      mfc=color if original else "white", mew=.9, ls="none")
            plotted.append(dict(length=size, g=skin, initial="left", epsilon=.05,
                                target="G_invar", evidence="original_model" if original else "finite_kernel_prediction",
                                time=time, removed_linear_coefficient=linear, removed_absolute_constant=constant,
                                x_logL=math.log(size), y=y, theory_q_logL=logarithm * math.log(size),
                                sharp_residual=y-logarithm * math.log(size)))
    axis.set_xlabel(r"$\log L$")
    axis.set_ylabel(r"$t_{0.05}-TL-c_s$")
    axis.set_xticks((7, 7.5, 8))
    axis.set_ylim(-10, -.6)
    axis.legend(handles=[Line2D([], [], marker=mark, ms=4, color=color, label=label)
                         for color, mark, label in ((RED, "s", "$h=-0.1$"), (BLACK, "D", "$g=0$"), (BLUE, "o", "$h=0.1$"))],
                loc="center right", bbox_to_anchor=(.96, .42), frameon=False, handlelength=1.2,
                borderpad=.25, labelspacing=.3)
    fig.text(.59, .019, "(b)", ha="center", fontsize=8.6)
    return save(fig, "fig3_reflection_response", dict(no_fit=True, data_points=plotted,
                schematic_packet_curves_not_sampled_density=True,
                logarithmic_axis_uses_actual_logL=True,
                zero_bias_larger_size_prediction_not_claimed=True))


def windows(fixed_windows: list[dict], weak: list[dict]) -> dict:
    fig = plt.figure(figsize=(3.4, 4.2))
    axis = fig.add_axes((.235, .64, .73, .34))
    axes_style(axis)
    plotted = []
    for skin, color, marker in ((-.1, RED, "s"), (.1, BLUE, "o")):
        rows = sorted((row for row in fixed_windows if row["initial"] == "left" and float(row["skin"]) == skin), key=lambda row: int(row["length"]))
        xs, ys = [], []
        for row in rows:
            size = int(row["length"])
            original = bool(row["native_window"])
            value = float(row["native_window"] or row["kernel_window"])
            xs.append(size)
            ys.append(value)
            axis.plot(size, value, marker=marker, ms=4.5, mec=color, mfc=color if original else "white",
                      mew=.9, color=color, ls="none", zorder=4)
            plotted.append(dict(length=size, g=skin, initial="left", target="G_invar",
                                evidence="original_model" if original else "finite_kernel_prediction",
                                window=value, asymptotic_window=float(row["asymptotic_window"])))
        axis.plot(xs, ys, color=color, lw=.8)
        axis.axhline(float(rows[0]["asymptotic_window"]), color=color, ls=(0, (3, 2)), lw=.9)
    axis.set_xlim(700, 4350)
    axis.set_xticks((1000, 2000, 4000))
    axis.set_ylim(.6069, .61005)
    axis.set_yticks((.607, .608, .609, .610))
    axis.set_xlabel("$L$")
    axis.set_ylabel(r"$t_{0.01}-t_{0.05}$")
    axis.legend(handles=[Line2D([], [], color=color, marker=marker, ms=4, label=label)
                         for color, marker, label in ((RED, "s", "$h=-0.1$"), (BLUE, "o", "$h=0.1$"))],
                loc="lower right", frameon=False, handlelength=1.3, borderpad=.2, labelspacing=.3)
    fig.text(.60, .522, "(a)", ha="center", fontsize=8.6)
    axis = fig.add_axes((.235, .15, .73, .315))
    axes_style(axis)
    summaries = []
    for side, sign, marker in (("left", 1, "s"), ("right", -1, "D")):
        for size in (896, 1024, 1152, 1280):
            rows = [row for row in weak if row["initial"] == side and int(row["length"]) == size]
            if len(rows) != 9:
                raise RuntimeError("Weak response panel requires all nine conditions per side and size")
            values = [math.sqrt(size) * (sign * float(row["actual_response"]) + 1 / (4 * MU0)) for row in rows]
            mean, low, high = float(np.mean(values)), min(values), max(values)
            offset = -8 if side == "left" else 8
            axis.vlines(size + offset, low, high, color=GREEN, linewidth=.9)
            axis.plot(size + offset, mean, color=GREEN, marker=marker, ms=4.6, mew=.9,
                      mfc=GREEN, ls="none")
            summaries.append(dict(initial=side, length=size, cases=9, target="G_invar",
                                  mean=mean, minimum=low, maximum=high,
                                  x_offset_only_for_legibility=offset,
                                  evidence="original_model", range_is_deterministic=True))
    axis.axhline(A, color=BLACK, lw=.9, ls=(0, (3, 2)))
    axis.text(1258, A + .0002, "$A$", ha="right", va="bottom", fontsize=8.2)
    axis.set_xlim(840, 1330)
    axis.set_xticks((900, 1100, 1300))
    axis.set_ylim(.308, .338)
    axis.set_yticks((.31, .32, .33))
    axis.set_xlabel("$L$")
    axis.set_ylabel(r"$\sqrt{L}[s\partial_\chi t+1/(4\mu_0)]$", labelpad=4)
    axis.legend(handles=[Line2D([], [], color=GREEN, marker=marker, ls="none", ms=4,
                                mfc=GREEN, label=side)
                         for side, marker in (("left", "s"), ("right", "D"))],
                loc="upper center", bbox_to_anchor=(.53, 1.04), ncol=2, frameon=False,
                handlelength=.8, borderpad=.2, labelspacing=.3, columnspacing=1.)
    fig.text(.60, .019, "(b)", ha="center", fontsize=8.6)
    return save(fig, "fig4_continuous_window", dict(no_fit=True, window_data=plotted,
                response_summary=summaries, analytic_A=A,
                all_response_points_original_model=True, thresholds_fixed=True))


def main(argv: list[str] | None = None) -> None:
    global ROOT, DATA, THEORY, OUT, INSTALL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="Folder containing the three frozen comparisons/*.csv inputs")
    parser.add_argument("--theory-file", type=Path, default=THEORY)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--no-install", action="store_true",
                        help="Write to output-dir only; leave all manuscript figure folders unchanged")
    args = parser.parse_args(argv)
    ROOT, DATA, THEORY, OUT = (path.resolve() for path in (args.project, args.data_dir, args.theory_file, args.output_dir))
    INSTALL = not args.no_install
    INPUTS.clear()
    style()
    fixed = table("figure_fixed_records.csv")
    fixed_windows = table("figure_fixed_windows.csv")
    weak = table("figure_weak_records.csv")
    theory = load_theory()
    results = dict(figure3=centers(fixed, weak, theory), figure4=windows(fixed_windows, weak))
    for name, before in INPUTS.items():
        if digest(ROOT / name) != before:
            raise RuntimeError("A frozen input changed: " + name)
    results.update(input_sha256=INPUTS, source_sha256=digest(Path(__file__)),
                   no_original_data_or_proof_modifications=True,
                   evidence_key="Filled: original model. Open at L2048/4096: finite-kernel prediction.",
                   fidelity_curves_or_new_fidelity_checks=0)
    (OUT / "figure_upgrade_manifest.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "manuscript_figures_created", "figures": 2, "inputs": len(INPUTS)}, indent=2))


if __name__ == "__main__":
    main()
