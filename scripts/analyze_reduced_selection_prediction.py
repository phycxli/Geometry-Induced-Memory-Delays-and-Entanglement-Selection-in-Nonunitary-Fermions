"""Analyze frozen low-rank forecasts, finite OBC Gram formulas and precision."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits

import run_reduced_selection_prediction as study
import analyze_spatial_selection_mechanism as prior_analysis

ROOT, OUT, FIG = study.ROOT, study.OUT, study.FIG
audit, previous, p1 = study.audit, study.previous, study.p1
COLORS = {0.: "#247ba0", .25: "#c04c48", -.25: "#27866d"}
LABELS = dict(charge_density_wave="CDW", left="Left", right="Right", block_cell2="Block cell 2", block_cell7="Block cell 7", center_block="Center")


def spatial():
    locked = {(int(r["length"]), float(r["skin"]), r["initial"]): r for r in audit.read_csv(OUT / "locked_spatial_predictions.csv")}
    combined = {path.stem: study.read_json(path) for path in (study.PRIOR / "static").glob("*.json")}
    combined.update({path.stem: study.read_json(path) for path in (OUT / "static").glob("*.json")})
    rows, overlaps = [], []
    for static in combined.values():
        if static["gamma"] != 2:
            continue
        for initial, state in static["states"].items():
            values = {k: float(state[k]) for k in ("alpha", "cross_block_min", "rho", "overlap_lower", "overlap_upper") if k in state}
            overlaps.append(dict(length=static["length"], skin=static["skin"], initial=initial, **values))
        for initial in ("left", "right"):
            key = static["length"], static["skin"], initial
            if key not in locked:
                continue
            exact_text = static["states"][initial]["cross_block_min"]
            folded_results = [study.spatial_gram(static["length"], static["skin"], initial, nodes, dps, True)
                              for nodes, dps in ((64, 160), (96, 200))]
            folded = [x[0] for x in folded_results]
            with mp.workdps(220):
                exact = mp.mpf(exact_text)
                graph = study.spatial.decode(static["input_graph"])
                m = static["length"] // 4
                block = mp.matrix([[graph[m+i, m-1-j] if initial == "left" else graph[m-1-i, m+j]
                                    for j in range(m)] for i in range(m)])
                matrix_error = p1.mp_norm(block - folded_results[-1][1]) / p1.mp_norm(block)
                assert matrix_error < mp.mpf("1e-8")
                change = study.spatial.relative(*folded)
                error = study.spatial.relative(exact, folded[-1])
                assert change < mp.mpf("1e-8") and error < mp.mpf("1e-8"), (key, change, error)
                bulk = float(locked[key]["bulk_cross_min"])
                true = float(exact)
                rows.append(dict(length=key[0], skin=key[1], initial=initial,
                                 finite_cross_min=true, bulk_cross_min=bulk, folded_cross_min=float(folded[-1]),
                                 folded_relative_error=float(error), folded_matrix_relative_error=float(matrix_error), quadrature_relative_change=float(change),
                                 bulk_over_finite=bulk / true, bulk_log10_error=math.log10(bulk / true),
                                 bulk_exponential_rate=float(locked[key]["bulk_exponential_rate"]),
                                 effective_finite_rate=-math.log(true) / key[0],
                                 interpretation="held-out finite sizes; effective rate is not an asymptotic proof"))
            study.emit(dict(stage="spatial_analysis", length=key[0], skin=key[1], initial=initial))
    assert len(rows) == 12
    slopes = []
    for skin in (-.25, 0., .25):
        for initial in ("left", "right"):
            subset = sorted([r for r in overlaps if r["skin"] == skin and r["initial"] == initial], key=lambda r: r["length"])
            for a, b in zip(subset, subset[1:]):
                slopes.append(dict(skin=skin, initial=initial, length_a=a["length"], length_b=b["length"],
                                   slope_alpha=-math.log(b["alpha"] / a["alpha"]) / (b["length"] - a["length"]),
                                   slope_cross=-math.log(b["cross_block_min"] / a["cross_block_min"]) / (b["length"] - a["length"])))
    study.write_csv("spatial_gram_checks.csv", rows)
    study.write_csv("spatial_overlap.csv", overlaps)
    study.write_csv("spatial_effective_slopes.csv", slopes)
    study.write_json(OUT / "spatial_summary.json", dict(held_out_rows=12,
                     max_folded_relative_error=max(r["folded_relative_error"] for r in rows),
                     max_folded_matrix_relative_error=max(r["folded_matrix_relative_error"] for r in rows),
                     bulk_ratio_min=min(r["bulk_over_finite"] for r in rows), bulk_ratio_max=max(r["bulk_over_finite"] for r in rows),
                     interpretation="exact finite-chain Gram formula verified; dropping reflections tested without refitting; no sharp OBC asymptotic theorem"))
    spatial_plot(rows, overlaps)


def summarize():
    study.check_lock()
    receipts = [study.read_json(path) for path in sorted((OUT / "cases").glob("*.json"))]
    assert len(receipts) == len(study.cases()) == 29
    forecasts = {(r["case_id"], r["initial"]): r for r in audit.read_csv(OUT / "locked_predictions.csv")}
    memory_forecasts = {(r["case_id"], r["initial_b"]): r for r in audit.read_csv(OUT / "locked_memory_predictions.csv")}
    rows, pairs, precision, spectra, density, raw_checks = [], [], [], [], [], []
    for receipt in receipts:
        case = receipt["case"]
        key = study.spatial.identifier(case["length"], 2., case["skin"])
        with gzip.open(OUT / "predictions" / (key + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
            predictions_raw = {(r["case_id"], r["initial"]): r for r in json.load(stream)}
        with gzip.open(OUT / "cases" / (case["case_id"] + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
            reference_raw = json.load(stream)[-1]
        with np.load(OUT / "cases" / (case["case_id"] + ".npz")) as actual, np.load(OUT / "predictions" / (key + ".npz")) as predicted:
            for state in receipt["rows"]:
                forecast = forecasts[state["case_id"], state["initial"]]
                q = actual["Q_" + state["initial"]]
                phase = np.tile([1., 1.j], case["length"] // 2)
                qr = phase[:, None] * predicted[state["case_id"] + "_" + state["initial"]]
                eps = float(forecast["projector_remainder"])
                error = audit.projector_error(q, qr)
                row = {**state, **{k: float(forecast[k]) for k in ("projector_remainder", "distance_lower", "distance_upper", "entropy_error_lower", "entropy_error_upper", "distance_to_G", "small_system_condition", "Ft_remainder")},
                       "rank": int(float(forecast["rank"])), "rank_target_passed": forecast["rank_target_passed"] == "True",
                       "guaranteed_selected": forecast["guaranteed_selected"] == "True", "guaranteed_unselected": forecast["guaranteed_unselected"] == "True",
                       "prediction_projector_error": error, "prediction_entropy_error": abs(state["entropy"] - float(forecast["entropy"])),
                       "projector_remainder_violation": max(0., error - eps)}
                row.update(distance_interval_violation=max(row["distance_lower"] - state["distance_to_W"], state["distance_to_W"] - row["distance_upper"], 0.),
                           entropy_interval_violation=max(row["entropy_error_lower"] - state["entropy_error"], state["entropy_error"] - row["entropy_error_upper"], 0.),
                           wrong_selected=row["guaranteed_selected"] and not state["controlled"], wrong_unselected=row["guaranteed_unselected"] and state["controlled"])
                rows.append(row)
                values = np.sum(abs(q)**2, axis=1)
                assert abs(values.sum() - case["length"] // 2) < 1e-10 and values.min() > -1e-12 and values.max() < 1 + 1e-12
                density.extend(dict(case_id=case["case_id"], initial=state["initial"], site=i, density=float(x)) for i, x in enumerate(values))
                cs = np.linalg.eigvalsh(q[:case["length"] // 2] @ q[:case["length"] // 2].conj().T)
                spectra.extend(dict(case_id=case["case_id"], initial=state["initial"], mode=i, occupation=float(x)) for i, x in enumerate(cs))
                with mp.workdps(220):
                    p_raw = predictions_raw[case["case_id"], state["initial"]]["values"]
                    entropy_error = abs(mp.mpf(reference_raw["states"][state["initial"]]["entropy"]) - mp.mpf(p_raw["entropy"]))
                    budget = study.entropy_budget(mp.mpf(p_raw["projector_remainder"]), case["length"] // 2)
                    assert entropy_error <= budget + mp.mpf("1e-10")
                    raw_checks.append(dict(case_id=case["case_id"], initial=state["initial"], entropy_prediction_error=float(entropy_error),
                                           entropy_remainder_bound=float(budget), reference_dps=reference_raw["dps"]))
            cdw = next(r for r in receipt["rows"] if r["initial"] == "charge_density_wave")
            for state in receipt["rows"][1:]:
                forecast = memory_forecasts[case["case_id"], state["initial"]]
                lower = float(forecast["memory_lower"])
                distance = audit.projector_error(actual["Q_charge_density_wave"], actual["Q_" + state["initial"]])
                entropy_difference = abs(cdw["entropy"] - state["entropy"])
                pairs.append({**case, "initial_a": "charge_density_wave", "initial_b": state["initial"], "pair_projector_distance": distance,
                              "pair_entropy_difference": entropy_difference, "memory_lower": lower, "guaranteed_memory": forecast["guaranteed_memory"] == "True",
                              "memory_retained": distance > .1 or entropy_difference > .1, "bound_violation": max(0., lower - distance)})
        precision.append({**case, "seconds": receipt["seconds"], "reference_dps": receipt["rows"][0]["reference_dps"], **receipt["reference_convergence"]})
    assert len(rows) == 83 and len(pairs) == 54
    study.write_csv("actual_entanglement.csv", rows)
    study.write_csv("initial_memory.csv", pairs)
    study.write_csv("local_density.csv", density)
    study.write_csv("correlation_spectra.csv", spectra)
    study.write_csv("reference_precision_checks.csv", precision)
    study.write_csv("high_precision_entropy_checks.csv", raw_checks)
    study.write_csv("double_precision_failures.csv", [r for r in rows if r["double_status"] != "reference_validated"])
    study.write_csv("rank_target_failures.csv", [r for r in rows if not r["rank_target_passed"]])
    metrics = []
    for role in ("all", "unseen_size", "unseen_offset", "persistence_extension"):
        subset = [r for r in rows if role == "all" or r["role"] == role]
        local_pairs = [r for r in pairs if role == "all" or r["role"] == role]
        metrics.append(dict(role=role, states=len(subset), pairs=len(local_pairs), actual_selected=sum(r["controlled"] for r in subset),
                            guaranteed_selected=sum(r["guaranteed_selected"] for r in subset), guaranteed_unselected=sum(r["guaranteed_unselected"] for r in subset),
                            undecided=sum(not r["guaranteed_selected"] and not r["guaranteed_unselected"] for r in subset),
                            wrong_selected=sum(r["wrong_selected"] for r in subset), wrong_unselected=sum(r["wrong_unselected"] for r in subset),
                            actual_memory=sum(r["memory_retained"] for r in local_pairs), guaranteed_memory=sum(r["guaranteed_memory"] for r in local_pairs),
                            rank_target_passed=sum(r["rank_target_passed"] for r in subset), ranks=dict(sorted(Counter(r["rank"] for r in subset).items())),
                            double_failed=sum(r["double_status"] != "reference_validated" for r in subset), stable_failed=sum(r["stable_status"] != "reference_validated" for r in subset),
                            max_projector_prediction_error=max(r["prediction_projector_error"] for r in subset)))
    study.write_csv("prediction_metrics.csv", metrics)
    intervals = []
    for initial in ("left", "right"):
        for length in (64, 72, 80):
            for skin in (0., .25):
                subset = [r for r in rows if (r["length"], r["skin"], r["initial"]) == (length, skin, initial)]
                if length == 64 and initial == "left":
                    subset += [{**r, "physical_time": float(r["physical_time"]), "controlled": r["controlled"] == "True"}
                               for r in audit.read_csv(study.PRIOR / "actual_entanglement.csv") if (int(r["length"]), float(r["skin"]), r["initial"]) == (length, skin, initial)]
                if subset:
                    intervals.append(dict(length=length, skin=skin, initial=initial, **prior_analysis.trajectory_interval(subset)))
    study.write_csv("selection_time_intervals.csv", intervals)
    summary = dict(date="2026-10-04", conditions=29, states=83, pairs=54, static_families=11, metrics=metrics, selection_intervals=intervals,
                   max_distance_interval_violation=max(r["distance_interval_violation"] for r in rows),
                   max_entropy_interval_violation=max(r["entropy_interval_violation"] for r in rows),
                   max_memory_lower_violation=max(r["bound_violation"] for r in pairs), max_projector_remainder_violation=max(r["projector_remainder_violation"] for r in rows),
                   max_double_projector_error=max(r["double_projector_error"] for r in rows), max_double_entropy_error=max(r["double_entropy_error"] for r in rows),
                   static80_max_projector_error=max(r["static80_projector_error"] for r in rows), static80_max_entropy_error=max(r["static80_entropy_error"] for r in rows),
                   static120_max_projector_error=max(r["static120_projector_error"] for r in rows), static120_max_entropy_error=max(r["static120_entropy_error"] for r in rows),
                   static80_seconds_total=sum(r["static80_seconds"] for r in rows), static120_seconds_total=sum(r["static120_seconds"] for r in rows),
                   reference_digits=sorted({r["reference_dps"] for r in rows}),
                   interpretation="full static-spectrum preparation plus bounded low-rank time evolution; fixed local grid, no universal scaling law")
    study.write_json(OUT / "summary.json", summary)
    dynamic_plot(rows, pairs)
    study.emit(dict(stage="summarize", **metrics[0]))


def sensitivity():
    rows = []
    for length in (40, 80):
        for skin in (-.25, 0., .25):
            for initial in ("left", "right"):
                with mp.workdps(200):
                    value, matrix = study.spatial_gram(length, skin, initial, 96, 200)
                    eigenvalues, vectors = mp.eigsy(matrix)
                    vector = vectors[:, 0]
                    mean_index = mp.fsum(i * abs(vector[i])**2 for i in range(vector.rows))
                    sign = 1 if initial == "left" else -1
                    derivative = sign * (1 + 2 * mean_index)
                    h = mp.mpf(".000001")
                    a = study.spatial_gram(length, mp.mpf(str(skin)) - h, initial, 96, 200)[0]
                    b = study.spatial_gram(length, mp.mpf(str(skin)) + h, initial, 96, 200)[0]
                    finite_difference = mp.log(b / a) / (2 * h)
                    error = abs(derivative - finite_difference)
                    assert 1 <= sign * derivative <= 2 * vector.rows - 1
                    assert error < mp.mpf("1e-6")
                    rows.append(dict(length=length, skin=skin, initial=initial, cross_min=float(value),
                                     mean_cell_index=float(mean_index), log_derivative_skin=float(derivative),
                                     log_derivative_finite_difference=float(finite_difference), derivative_error=float(error),
                                     normalized_absolute_derivative=float(abs(derivative) / (2 * vector.rows - 1)),
                                     interpretation="retrospective check of exact finite-chain identity; not held-out prediction"))
                study.emit(dict(stage="skin_sensitivity", length=length, skin=skin, initial=initial))
    study.write_csv("skin_sensitivity_checks.csv", rows)


def figure_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    FIG.mkdir(parents=True, exist_ok=True)
    return plt


def save(plt, fig, name):
    fig.savefig(FIG / (name + ".png"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def spatial_plot(checks, overlap):
    plt = figure_setup()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for initial, marker in (("left", "o"), ("right", "s")):
        subset = sorted([r for r in overlap if r["skin"] == 0 and r["initial"] == initial], key=lambda r: r["length"])
        axes[0].plot([r["length"] for r in subset], [math.log10(r["alpha"]) for r in subset], marker + "-", label=LABELS[initial])
        for skin, color in COLORS.items():
            local = [r for r in checks if r["skin"] == skin and r["initial"] == initial]
            axes[1].plot([r["length"] for r in local], [r["bulk_over_finite"] for r in local], marker + "-", color=color, label=f"{LABELS[initial]}, g={skin:g}")
    for initial, marker in (("left", "o"), ("right", "s"), ("center_block", "D"), ("block_cell2", "^"), ("block_cell7", "v")):
        subset = sorted([r for r in overlap if r["length"] == 40 and r["initial"] == initial], key=lambda r: r["skin"])
        axes[2].plot([r["skin"] for r in subset], [math.log10(r["alpha"]) for r in subset], marker + "-", label=LABELS[initial])
    axes[0].set(xlabel="Sites L", ylabel=r"$\log_{10}\alpha_\infty$", title="(a) Extreme-block overlap, g=0")
    axes[1].set(xlabel="Sites L", ylabel="Bulk / finite cross-block minimum", title="(b) Frozen bulk approximation", yscale="log")
    axes[2].set(xlabel="Skin parameter g", ylabel=r"$\log_{10}\alpha_\infty$", title="(c) Occupied-cell placement, L=40")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.18)
    save(plt, fig, "finite_boundary_overlap")


def dynamic_plot(rows, pairs):
    plt = figure_setup()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), layout="constrained")
    for length, marker in ((72, "o"), (80, "s")):
        for skin, color in ((0., COLORS[0.]), (.25, COLORS[.25])):
            for initial, ls in (("left", "-"), ("right", "--")):
                subset = sorted([r for r in rows if (r["length"], r["skin"], r["initial"]) == (length, skin, initial)], key=lambda r: r["physical_time"])
                label = f"L={length}, g={skin:g}, {LABELS[initial]}"
                axes[0,0].semilogy([r["physical_time"] for r in subset], [max(1e-12, r["distance_to_W"]) for r in subset], marker + ls, color=color, label=label)
                axes[0,1].plot([r["physical_time"] for r in subset], [r["rank"] for r in subset], marker + ls, color=color)
                if initial == "left":
                    local = sorted([r for r in pairs if (r["length"], r["skin"], r["initial_b"]) == (length, skin, initial)], key=lambda r: r["physical_time"])
                    axes[1,0].semilogy([r["physical_time"] for r in local], [max(1e-12, r["pair_projector_distance"]) for r in local], marker + "-", color=color)
                    axes[1,0].semilogy([r["physical_time"] for r in local], [max(1e-12, r["memory_lower"]) for r in local], "--", color=color)
    x = np.arange(len(rows))
    for field, color, label in (("double_projector_error", "#c04c48", "Double step QR"), ("static80_projector_error", "#247ba0", "Static 80 digits"), ("static120_projector_error", "#27866d", "Static 120 digits")):
        axes[1,1].semilogy(x, [max(1e-16, r[field]) for r in rows], ".", color=color, label=label)
    axes[0,0].axhline(.05, color="black", ls=":")
    failed = [r for r in rows if r["role"] == "unseen_size" and r["initial"] != "charge_density_wave" and not r["rank_target_passed"]]
    axes[0,1].scatter([r["physical_time"] for r in failed], [r["rank"] for r in failed], marker="x", color="black", s=55, zorder=4, label="Remainder > 1e-5")
    axes[0,1].legend(fontsize=8, loc="center right")
    axes[1,0].axhline(.1, color="black", ls=":")
    axes[1,1].axhline(1e-6, color="black", ls=":")
    axes[0,0].set(xlabel="Time t", ylabel="Output projector distance", title="(a) Independent new-size truth")
    axes[0,1].set(xlabel="Time t", ylabel="Chosen rank", yticks=study.RANKS, title="(b) Frozen rank policy")
    axes[1,0].set(xlabel="Time t", ylabel="CDW-left projector memory", title="(c) Dashed: locked lower bounds")
    axes[1,1].set(xlabel="State index (fixed case order)", ylabel="Projector error", title="(d) Precision benchmark")
    axes[0,0].legend(fontsize=7, ncol=2); axes[1,1].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.18)
    save(plt, fig, "unseen_size_reduction_precision")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), layout="constrained", sharey=True)
    for ax, skin in zip(axes, (-.25, 0., .25)):
        for initial, color in (("charge_density_wave", "#247ba0"), ("block_cell2", "#c04c48"), ("block_cell7", "#27866d")):
            subset = sorted([r for r in rows if (r["length"], r["skin"], r["initial"]) == (40, skin, initial)], key=lambda r: r["physical_time"])
            ax.semilogy([r["physical_time"] for r in subset], [max(1e-12, r["distance_to_W"]) for r in subset], "o-", color=color, label=LABELS[initial])
            ax.fill_between([r["physical_time"] for r in subset], [max(1e-12, r["distance_lower"]) for r in subset], [max(1e-12, r["distance_upper"]) for r in subset], color=color, alpha=.15)
        ax.axhline(.05, color="black", ls=":")
        ax.set(xlabel="Time t", ylabel="Output projector distance", title=f"L=40, g={skin:g}"); ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    save(plt, fig, "translated_block_predictions")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")
    axes[0].loglog([max(1e-16, r["projector_remainder"]) for r in rows], [max(1e-16, r["prediction_projector_error"]) for r in rows], "o", ms=4)
    axes[0].plot([1e-16, 1], [1e-16, 1], "k--", lw=1)
    axes[0].set(xlabel="Locked projector remainder", ylabel="Measured low-rank error", title="(a) Remainder coverage (floor 1e-16)")
    calibration = audit.read_csv(OUT / "development/rank_calibration.csv")
    for length, skin, time in ((48, .25, 22.5), (64, 0., 30), (64, 0., 35)):
        subset = [r for r in calibration if (int(r["length"]), float(r["skin"]), float(r["physical_time"])) == (length, skin, time)]
        axes[1].semilogy([int(float(r["rank"])) for r in subset], [max(1e-16, float(r["measured_projector_error"])) for r in subset], "o-", label=f"L={length}, g={skin:g}, t={time:g}")
    axes[1].axhline(1e-5, color="black", ls=":")
    axes[1].set(xlabel="Static SVD rank", ylabel="Measured projector error", xticks=study.RANKS, title="(b) Known-input rank calibration")
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.18)
    save(plt, fig, "remainder_and_rank_calibration")


def local_links(pending=False):
    allowed = {OUT / "delivery_verification.json", OUT / "artifact_manifest.csv"} if pending else set()
    count = 0
    for path in [ROOT / "README.md", *sorted((ROOT / "research_doc").rglob("*.md"))]:
        content = path.read_text(encoding="utf-8-sig")
        assert "\ufffd" not in content
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            target = target.split("#", 1)[0]
            if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target): continue
            resolved = (path.parent / target.strip("<>")).resolve()
            assert resolved.exists() or resolved in allowed, (path, target)
            count += 1
    return count


def verify():
    study.prepare(); study.check_lock()
    lock = study.read_json(OUT / "prediction_lock.json")
    inputs = audit.read_csv(OUT / "input_manifest.csv")
    for entry in inputs + lock["inputs"]:
        assert audit.sha256(ROOT / entry["path"]) == entry["sha256"], entry["path"]
    for name, sha in study.read_json(OUT / "spatial_lock_hash.json").items():
        assert audit.sha256(OUT / name) == sha
    spatial_lock = study.read_json(OUT / "spatial_hypothesis_lock.json")
    static_files = list((OUT / "static").glob("*.json"))
    assert len(static_files) == 11
    for path in static_files:
        value = study.read_json(path)
        assert value["started_utc"] > spatial_lock["locked_utc"]
        assert max(value["reference_convergence"][k] for k in ("relative_scalar", "compression_relative")) <= 1e-8
        assert max(value["reference_convergence"][k] for k in ("matrix_absolute", "gain_entropy")) <= 1e-10
    receipts = [study.read_json(path) for path in (OUT / "cases").glob("*.json")]
    assert len(receipts) == 29
    for value in receipts:
        assert value["started_utc"] > lock["locked_utc"]
        assert previous.accepted_precision(value["reference_convergence"])
        assert value["prediction_lock_sha256"] == audit.sha256(OUT / "prediction_lock.json")
        assert value["config_sha256"] == lock["config_sha256"]
        assert audit.sha256(OUT / "code" / (value["script_sha256"].lower() + ".py")) == value["script_sha256"]
        with gzip.open(OUT / "cases" / (value["case"]["case_id"] + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        assert [x["dps"] for x in raw] == value["precision_digits"]
        with np.load(OUT / "cases" / (value["case"]["case_id"] + ".npz")) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            for state in value["rows"]:
                for suffix in ("", "_static80", "_static120"):
                    q = arrays["Q_" + state["initial"] + suffix]
                    assert audit.opnorm(q.conj().T @ q - np.eye(q.shape[1])) < 1e-10
    summary = study.read_json(OUT / "summary.json")
    assert summary["states"] == 83 and summary["pairs"] == 54
    for key in ("max_distance_interval_violation", "max_entropy_interval_violation", "max_memory_lower_violation", "max_projector_remainder_violation"):
        assert summary[key] < 1e-9, key
    assert all(r["wrong_selected"] == r["wrong_unselected"] == r["stable_failed"] == 0 for r in summary["metrics"])
    actual = audit.read_csv(OUT / "actual_entanglement.csv")
    lookup = {(r["case_id"], r["initial"]): r for v in receipts for r in v["rows"]}
    for row in actual:
        assert all(float(row[k]) == lookup[row["case_id"], row["initial"]][k] for k in ("entropy", "distance_to_W", "alpha", "ratio"))
    assert len(audit.read_csv(OUT / "double_precision_failures.csv")) == summary["metrics"][0]["double_failed"]
    assert len(audit.read_csv(OUT / "rank_target_failures.csv")) == 16
    gram = audit.read_csv(OUT / "spatial_gram_checks.csv")
    assert len(gram) == 12 and max(float(r["folded_relative_error"]) for r in gram) < 1e-8
    assert max(float(r["folded_matrix_relative_error"]) for r in gram) < 1e-8
    sensitivity_rows = audit.read_csv(OUT / "skin_sensitivity_checks.csv")
    assert len(sensitivity_rows) == 12 and max(float(r["derivative_error"]) for r in sensitivity_rows) < 1e-6
    for source in study.read_json(OUT / "literature/source_index.json"):
        for suffix, key in (("pdf", "pdf_sha256"), ("txt", "text_sha256")):
            assert audit.sha256(OUT / "literature" / (source["key"] + "." + suffix)) == source[key]
    links = local_links(True)
    study.write_json(OUT / "delivery_verification.json", dict(status="passed", utc=datetime.now(timezone.utc).isoformat(),
                     immutable_inputs=len(inputs), frozen_prediction_inputs=len(lock["inputs"]), conditions=29, states=83, pairs=54,
                     static_families=11, spatial_holdout_rows=12, local_links_checked=links,
                     spatial_truth_started_after_lock=True, dynamic_truth_started_after_lock=True, both_locks_unchanged=True,
                     reference_convergence_passed=True, remainder_coverage_passed=True, source_pdfs=2,
                     stable80_and120_all_passed=True, double_failures_retained=summary["metrics"][0]["double_failed"],
                     qualification="converged arbitrary precision; no interval arithmetic or asymptotic proof"))
    study.emit(dict(stage="verify", status="passed", states=83, pairs=54, links=links))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("spatial", "sensitivity", "summarize", "verify"))
    args = parser.parse_args()
    study.prepare()
    own = Path(__file__)
    (OUT / "code" / (audit.sha256(own).lower() + ".py")).write_bytes(own.read_bytes())
    with threadpool_limits(limits=1): globals()[args.stage]()
    study.write_json(OUT / ("environment_analysis_" + args.stage + ".json"), dict(utc=datetime.now(timezone.utc).isoformat(),
                     command=__import__("sys").argv, analysis_source_sha256=audit.sha256(own), compute_source_sha256=study.SOURCE_SHA,
                     config_sha256=audit.sha256(OUT / "config.json")))
    if args.stage == "verify":
        files = [p for directory in (OUT, FIG) for p in directory.rglob("*") if p.is_file() and p.name != "artifact_manifest.csv"]
        files.extend([own, study.SOURCE, ROOT / "scripts/collect_reduced_prediction_literature.py", ROOT / "README.md", *sorted((ROOT / "research_doc").rglob("*.md"))])
        study.write_csv("artifact_manifest.csv", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=audit.sha256(p)) for p in sorted(files)])
        assert local_links() == study.read_json(OUT / "delivery_verification.json")["local_links_checked"]


if __name__ == "__main__":
    main()
