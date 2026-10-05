"""Audit sixth-round frozen predictions, costs and finite-chain intervals."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits

import run_directional_selection_prediction as study
import analyze_spatial_selection_mechanism as old_analysis

ROOT, OUT, FIG = study.ROOT, study.OUT, study.FIG
audit, previous, p1 = study.audit, study.previous, study.p1


def summarize():
    study.check_hashes("prediction_lock_hash.json")
    forecasts = {(r["case_id"], r["initial"]): r for r in audit.read_csv(OUT / "locked_predictions.csv")}
    memories = {(r["case_id"], r["initial_b"]): r for r in audit.read_csv(OUT / "locked_memory_predictions.csv")}
    rows, pairs, costs, precision, raw_checks, density, spectra = [], [], [], [], [], [], []
    receipts = [study.read_json(path) for path in sorted((OUT / "cases").glob("*.json"))]
    assert len(receipts) == len(study.cases()) == 19
    for receipt in receipts:
        case = receipt["case"]
        family = study.spatial.identifier(case["length"], 2., case["skin"])
        with gzip.open(OUT / "predictions" / (family + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
            raw_predictions = {(v["case_id"], v["initial"]): v for v in json.load(stream)}
        with gzip.open(OUT / "cases" / (case["case_id"] + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
            raw_truth = json.load(stream)[-1]
        with np.load(OUT / "cases" / (case["case_id"] + ".npz")) as arrays:
            for state in receipt["rows"]:
                forecast = forecasts[state["case_id"], state["initial"]]
                raw_prediction = raw_predictions[state["case_id"], state["initial"]]
                with mp.workdps(260):
                    qa = study.spatial.decode(raw_truth["states"][state["initial"]]["Q"])
                    qr = study.spatial.decode(raw_prediction["Q"])
                    orthogonal_error = p1.mp_norm(qr.T * qr - mp.eye(qr.cols))
                    assert orthogonal_error < mp.mpf("1e-30")
                    error = previous.mp_opnorm(qa - qr * (qr.T * qa))
                    eps = mp.mpf(raw_prediction["values"]["projector_remainder"])
                    assert error <= eps + mp.mpf("1e-30"), (case["case_id"], state["initial"], error, eps)
                    entropy_error = abs(mp.mpf(raw_prediction["values"]["entropy"]) - mp.mpf(raw_truth["states"][state["initial"]]["entropy"]))
                    entropy_bound = study.base.entropy_budget(eps, case["length"] // 2)
                    assert entropy_error <= entropy_bound + mp.mpf("1e-30")
                    raw_checks.append(dict(case_id=case["case_id"], initial=state["initial"], projector_error=float(error),
                                           projector_bound=float(eps), entropy_error=float(entropy_error), entropy_bound=float(entropy_bound),
                                           predicted_orthogonality=float(orthogonal_error), projector_error_mp=str(error),
                                           projector_bound_mp=str(eps), entropy_error_mp=str(entropy_error)))
                fields = ("projector_remainder", "legacy_remainder", "directional_remainder", "transverse_remainder", "small_core_order",
                          "small_core_smin", "distance_lower", "distance_upper", "entropy_error_lower", "entropy_error_upper", "distance_to_G")
                row = {**state, **{k: float(forecast[k]) for k in fields}, "rank": int(float(forecast["rank"])),
                       "prediction_projector_error": float(error), "prediction_entropy_error": float(entropy_error),
                       "rank_target_passed": forecast["rank_target_passed"] == "True",
                       "guaranteed_selected": forecast["guaranteed_selected"] == "True",
                       "guaranteed_unselected": forecast["guaranteed_unselected"] == "True"}
                row.update(distance_interval_violation=max(0, row["distance_lower"] - row["distance_to_W"], row["distance_to_W"] - row["distance_upper"]),
                           entropy_interval_violation=max(0, row["entropy_error_lower"] - row["entropy_error"], row["entropy_error"] - row["entropy_error_upper"]),
                           wrong_selected=row["guaranteed_selected"] and not row["controlled"],
                           wrong_unselected=row["guaranteed_unselected"] and row["controlled"])
                rows.append(row)
                costs.append(dict(case_id=case["case_id"], initial=state["initial"], role=case["role"],
                                  **{k: float(forecast[k]) for k in ("propagation_seconds", "remainder_seconds", "entropy_seconds")},
                                  reference_qr_seconds=state["reference_qr_seconds"], reference_entropy_seconds=state["reference_entropy_seconds"],
                                  static80_seconds=state["static80_seconds"], static120_seconds=state["static120_seconds"]))
                q = arrays["Q_" + state["initial"]]
                occupation = np.sum(abs(q)**2, axis=1)
                assert abs(sum(occupation) - case["length"] // 2) < 1e-10
                density.extend(dict(case_id=case["case_id"], initial=state["initial"], site=i, density=float(x)) for i, x in enumerate(occupation))
                spectrum = np.linalg.eigvalsh(q[:case["length"] // 2] @ q[:case["length"] // 2].conj().T)
                spectra.extend(dict(case_id=case["case_id"], initial=state["initial"], mode=i, occupation=float(x)) for i, x in enumerate(spectrum))
            cdw = next(r for r in receipt["rows"] if r["initial"] == "charge_density_wave")
            for state in receipt["rows"][1:]:
                forecast = memories[case["case_id"], state["initial"]]
                with mp.workdps(260):
                    a = study.spatial.decode(raw_truth["states"]["charge_density_wave"]["Q"])
                    b = study.spatial.decode(raw_truth["states"][state["initial"]]["Q"])
                    distance = previous.mp_opnorm(a - b * (b.T * a))
                lower = float(forecast["memory_lower"])
                entropy_difference = abs(cdw["entropy"] - state["entropy"])
                pairs.append({**case, "initial_a": "charge_density_wave", "initial_b": state["initial"],
                              "pair_projector_distance": float(distance), "pair_entropy_difference": entropy_difference,
                              **{k: float(forecast[k]) for k in ("memory_lower", "direct_lower", "radial_lower", "reduced_pair_distance")},
                              "guaranteed_memory": forecast["guaranteed_memory"] == "True",
                              "memory_retained": distance > .1 or entropy_difference > .1,
                              "projector_memory_retained": distance > .1, "bound_violation": max(0, lower - float(distance))})
        precision.append({**case, "seconds": receipt["seconds"], "precision_digits": json.dumps(receipt["precision_digits"]), **receipt["reference_convergence"]})
        study.emit(dict(stage="analysis", case_id=case["case_id"]))
    study.write_csv("actual_entanglement.csv", rows)
    study.write_csv("initial_memory.csv", pairs)
    study.write_csv("prediction_costs.csv", costs)
    study.write_csv("reference_precision_checks.csv", precision)
    study.write_csv("high_precision_remainder_checks.csv", raw_checks)
    study.write_csv("local_density.csv", density)
    study.write_csv("correlation_spectra.csv", spectra)
    study.write_csv("double_precision_failures.csv", [r for r in rows if r["double_status"] != "reference_validated"])
    with (OUT / "rank_target_failures.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(r for r in rows if not r["rank_target_passed"])
    intervals = []
    for length in (80, 96, 112):
        for skin in ((0., .25) if length == 80 else (0., .05, .25)):
            for initial in ("left", "right"):
                local = [r for r in rows if (r["length"], r["skin"], r["initial"]) == (length, skin, initial)]
                if length == 80:
                    local += [{**r, "physical_time": float(r["physical_time"]), "controlled": r["controlled"] == "True"}
                              for r in audit.read_csv(study.PRIOR / "actual_entanglement.csv")
                              if (int(r["length"]), float(r["skin"]), r["initial"]) == (length, skin, initial)]
                intervals.append(dict(length=length, skin=skin, initial=initial, **old_analysis.trajectory_interval(local)))
    study.write_csv("selection_time_intervals.csv", intervals)
    metrics = []
    for role in ("all", "window_and_persistence", "unseen_size"):
        local = [r for r in rows if role == "all" or r["role"] == role]
        memory = [r for r in pairs if role == "all" or r["role"] == role]
        metrics.append(dict(role=role, states=len(local), pairs=len(memory), actual_selected=sum(r["controlled"] for r in local),
                            guaranteed_selected=sum(r["guaranteed_selected"] for r in local),
                            guaranteed_unselected=sum(r["guaranteed_unselected"] for r in local),
                            undecided=sum(not r["guaranteed_selected"] and not r["guaranteed_unselected"] for r in local),
                            wrong_selected=sum(r["wrong_selected"] for r in local), wrong_unselected=sum(r["wrong_unselected"] for r in local),
                            rank_target_passed=sum(r["rank_target_passed"] for r in local), ranks=dict(sorted(Counter(r["rank"] for r in local).items())),
                            actual_memory=sum(r["memory_retained"] for r in memory), actual_projector_memory=sum(r["projector_memory_retained"] for r in memory),
                            guaranteed_memory=sum(r["guaranteed_memory"] for r in memory),
                            radial_memory=sum(r["radial_lower"] > .1 for r in memory),
                            double_failed=sum(r["double_status"] != "reference_validated" for r in local),
                            stable_failed=sum(r["stable_status"] != "reference_validated" for r in local)))
    study.write_csv("prediction_metrics.csv", metrics)
    statics = [study.read_json(p) for p in (OUT / "static").glob("*.json")]
    cost_summary = dict(static_new_preparation_seconds=sum(v["seconds"] for v in statics if not v["reused_fifth_static"]),
                        reduced_propagation_seconds=sum(r["propagation_seconds"] for r in costs),
                        reduced_remainder_seconds=sum(r["remainder_seconds"] for r in costs),
                        reduced_entropy_seconds=sum(r["entropy_seconds"] for r in costs),
                        reference_exponential_seconds=sum(c["exponential_seconds"] for v in receipts for c in v["reference_costs"]),
                        reference_full_svd_seconds=sum(c["full_svd_seconds"] for v in receipts for c in v["reference_costs"]),
                        reference_case_total_seconds=sum(v["seconds"] for v in receipts),
                        accounting="summed CPU-job wall times, concurrent processes; reduced costs at final precision include rank search")
    summary = dict(date="2026-10-04", conditions=len(receipts), states=len(rows), pairs=len(pairs), static_families=len(statics),
                   metrics=metrics, selection_intervals=intervals, costs=cost_summary,
                   max_prediction_projector_error=max(r["prediction_projector_error"] for r in rows),
                   max_prediction_entropy_error=max(r["prediction_entropy_error"] for r in rows),
                   max_projector_remainder_violation=max(max(0, r["projector_error"] - r["projector_bound"]) for r in raw_checks),
                   max_distance_interval_violation=max(r["distance_interval_violation"] for r in rows),
                   max_entropy_interval_violation=max(r["entropy_interval_violation"] for r in rows),
                   max_memory_lower_violation=max(r["bound_violation"] for r in pairs),
                   max_double_projector_error=max(r["double_projector_error"] for r in rows),
                   max_double_entropy_error=max(r["double_entropy_error"] for r in rows),
                   static80_max_projector_error=max(r["static80_projector_error"] for r in rows),
                   static80_max_entropy_error=max(r["static80_entropy_error"] for r in rows),
                   static120_max_projector_error=max(r["static120_projector_error"] for r in rows),
                   static120_max_entropy_error=max(r["static120_entropy_error"] for r in rows),
                   reference_digits=sorted({r["reference_dps"] for r in rows}),
                   interpretation="finite-size sparse test; sampled persistence, no continuum transition or universal exponent")
    study.write_json(OUT / "summary.json", summary)
    study.emit(dict(stage="summary", **metrics[0]))


def spatial():
    locked = {(int(r["length"]), float(r["skin"]), r["initial"]): r for r in audit.read_csv(OUT / "locked_spatial_predictions.csv")}
    rows, distributions = [], []
    for path in sorted((OUT / "static").glob("L*.json")):
        static = study.read_json(path)
        if static["length"] == 80:
            continue
        for initial in ("left", "right"):
            key = static["length"], static["skin"], initial
            with mp.workdps(260):
                graph = study.spatial.decode(static["input_graph"])
                m = static["length"] // 4
                matrix = mp.matrix([[graph[m + i, m - 1 - j] if initial == "left" else graph[m - 1 - i, m + j]
                                     for j in range(m)] for i in range(m)])
                eigenvalues, vectors = mp.eigsy(matrix)
                exact = eigenvalues[0]
                cross = mp.mpf(static["states"][initial]["cross_block_min"])
                assert study.spatial.relative(cross, exact) < mp.mpf("1e-10")
                predicted = next(r for r in study.read_json(OUT / "spatial/locked_inverse_trace.json")
                                 if (r["length"], r["skin"], r["initial"]) == key)
                lower, upper = mp.mpf(predicted["lower_mp"]), mp.mpf(predicted["upper_mp"])
                assert lower <= exact * (1 + mp.mpf("1e-8")) and upper >= exact * (1 - mp.mpf("1e-8"))
                value, gram = study.base.spatial_gram(key[0], key[1], initial, 96, 240, True)
                matrix_error = p1.mp_norm(matrix - gram) / p1.mp_norm(matrix)
                error = study.spatial.relative(value, exact)
                _, bulk = study.base.spatial_gram(key[0], key[1], initial, 96, 240, False)
                bulk_min = mp.eigsy(bulk, eigvals_only=True)[0]
                extremal_mean = mp.fsum(i * abs(vectors[i, 0])**2 for i in range(m))
                derivative = (1 if initial == "left" else -1) * (1 + 2 * extremal_mean)
                rows.append(dict(length=key[0], skin=key[1], initial=initial, exact_cross_min=float(exact),
                                 cross_lower=float(lower), cross_upper=float(upper), relative_width=float(upper / lower - 1),
                                 lower_relative_error=float((exact - lower) / exact), upper_relative_error=float((upper - exact) / exact),
                                 gram_relative_error=float(error), gram_matrix_relative_error=float(matrix_error),
                                 old_bulk_over_finite=float(bulk_min / exact), extremal_mean_cell=float(extremal_mean),
                                 exact_log_skin_derivative=float(derivative),
                                 trace_log_skin_derivative=float(locked[key]["inverse_trace_log_skin_derivative"]),
                                 derivative_difference=float(abs(derivative - mp.mpf(str(locked[key]["inverse_trace_log_skin_derivative"]))))))
                distribution = [mp.mpf(x) for x in predicted["inverse_diagonal_distribution"]]
                distributions.extend(dict(length=key[0], skin=key[1], initial=initial, cell_index=i,
                                           extremal_probability=float(abs(vectors[i, 0])**2), inverse_trace_probability=float(distribution[i])) for i in range(m))
            study.emit(dict(stage="spatial_check", length=key[0], skin=key[1], initial=initial))
    assert len(rows) == 12
    study.write_csv("spatial_inverse_trace_checks.csv", rows)
    study.write_csv("spatial_extremal_distributions.csv", distributions)
    study.write_json(OUT / "spatial_summary.json", dict(rows=len(rows), covered_rows=len(rows),
                     min_relative_width=min(r["relative_width"] for r in rows), max_relative_width=max(r["relative_width"] for r in rows),
                     max_gram_relative_error=max(r["gram_relative_error"] for r in rows),
                     max_gram_matrix_relative_error=max(r["gram_matrix_relative_error"] for r in rows),
                     old_bulk_ratio_min=min(r["old_bulk_over_finite"] for r in rows), old_bulk_ratio_max=max(r["old_bulk_over_finite"] for r in rows),
                     max_skin_derivative_difference=max(r["derivative_difference"] for r in rows)))


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)
    rows = audit.read_csv(OUT / "actual_entanglement.csv")
    pairs = audit.read_csv(OUT / "initial_memory.csv")
    colors = {0.: "#247ba0", .05: "#27866d", .25: "#c04c48"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for ax, length in zip(axes, (80, 96, 112)):
        for skin in ((0., .25) if length == 80 else (0., .05, .25)):
            for initial, marker, linestyle in (("left", "o", "-"), ("right", "s", "--")):
                local = sorted([r for r in rows if (int(r["length"]), float(r["skin"]), r["initial"]) == (length, skin, initial)], key=lambda r: float(r["physical_time"]))
                ax.semilogy([float(r["physical_time"]) for r in local], [max(1e-14, float(r["distance_to_W"])) for r in local], marker=marker,
                            linestyle=linestyle, color=colors[skin], label=f"g={skin:g}, {initial}")
                plotted = np.array([max(1e-14, float(r["distance_to_W"])) for r in local])
                lower = np.array([max(1e-14, float(r["distance_lower"])) for r in local])
                upper = np.array([max(1e-14, float(r["distance_upper"])) for r in local])
                ax.errorbar([float(r["physical_time"]) for r in local], plotted,
                            yerr=np.stack((np.maximum(0, plotted-lower), np.maximum(0, upper-plotted))),
                            fmt="none", ecolor=colors[skin], capsize=3, alpha=.65)
        ax.axhline(.05, color="black", ls=":")
        ax.set(xlabel="Time t (lines guide the eye)", ylabel="Distance to output W", title=f"L={length}, pointwise frozen bounds")
        ax.grid(alpha=.15); ax.legend(fontsize=8)
    save(fig, "directional_selection_new_sizes")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    development = audit.read_csv(OUT / "development/states.csv")
    axes[0].loglog([max(1e-30, float(r["old_remainder"])) for r in development],
                   [max(1e-30, float(r["new_remainder_at_old_rank"])) for r in development], "o", ms=4)
    axes[0].plot([1e-30, 1], [1e-30, 1], "k--")
    axes[0].set(xlabel="Legacy remainder at original rank", ylabel="New remainder at same rank", title="Observed development states")
    axes[1].loglog([max(1e-30, float(r["projector_remainder"])) for r in rows],
                   [max(1e-30, float(r["prediction_projector_error"])) for r in rows], "o", ms=4)
    axes[1].plot([1e-30, 1], [1e-30, 1], "k--")
    axes[1].set(xlabel="Frozen remainder", ylabel="High-precision measured error", title="57 unseen states")
    axes[2].scatter([float(r["radial_lower"]) for r in pairs], [float(r["direct_lower"]) for r in pairs], c=[colors[float(r["skin"])] for r in pairs], s=25)
    axes[2].plot([0, 1], [0, 1], "k--"); axes[2].axhline(.1, color="black", ls=":")
    axes[2].set(xlabel="Distance-to-G memory lower bound", ylabel="Direct pair-projector lower bound", title="38 new pairs")
    for ax in axes: ax.grid(alpha=.15)
    save(fig, "remainder_and_direct_memory")
    spatial_rows = audit.read_csv(OUT / "spatial_inverse_trace_checks.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for initial, marker in (("left", "o"), ("right", "s")):
        for skin, color in colors.items():
            local = [r for r in spatial_rows if r["initial"] == initial and float(r["skin"]) == skin]
            local.sort(key=lambda r: int(r["length"]))
            axes[0].plot([int(r["length"]) for r in local], [math.log10(float(r["exact_cross_min"])) for r in local], marker + "-", color=color, label=f"{initial}, g={skin:g}")
            axes[1].semilogy([int(r["length"]) for r in local], [float(r["relative_width"]) for r in local], marker + "-", color=color)
            axes[2].semilogy([int(r["length"]) for r in local], [float(r["old_bulk_over_finite"]) for r in local], marker + "-", color=color)
    axes[0].set(xlabel="Sites L", ylabel="log10 finite cross-block minimum", title="Independent static truth")
    axes[1].set(xlabel="Sites L", ylabel="Upper/lower - 1", title="Locked inverse-trace interval width")
    axes[2].set(xlabel="Sites L", ylabel="Old bulk / finite minimum", title="Historical bulk failure persists")
    axes[0].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.15)
    save(fig, "finite_boundary_inverse_trace")


def local_links():
    count = 0
    for path in [ROOT / "README.md", *sorted((ROOT / "research_doc").rglob("*.md"))]:
        content = path.read_text(encoding="utf-8-sig")
        assert "\ufffd" not in content
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            target = target.split("#", 1)[0]
            if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            assert (path.parent / target.strip("<>")).resolve().exists(), (path, target)
            count += 1
    return count


def benchmark():
    """Separate final-rank work from legacy-helper and rank-search overhead."""
    study.check_hashes("prediction_lock_hash.json")
    rows = []
    selected = [r for r in audit.read_csv(OUT / "locked_predictions.csv")
                if float(r["physical_time"]) == int(r["length"]) * .5]
    for row in selected:
        static = study.read_json(OUT / "static" / (study.spatial.identifier(int(row["length"]), 2., float(row["skin"])) + ".json"))
        with mp.workdps(static["dps"]):
            tick = time.perf_counter()
            ctx = study.context(static, row["initial"])
            context_seconds = time.perf_counter() - tick
            matrices, u, vh, sig, rinv, mu, residuals = ctx
            rank, n = int(float(row["rank"])), len(mu)
            duration = mp.mpf(row["physical_time"])
            tick = time.perf_counter()
            decay = [mp.exp(-x * duration) for x in mu]
            if rank:
                a = mp.matrix([[decay[i] * u[i, j] * mp.sqrt(sig[j]) for j in range(rank)] for i in range(n)])
                bt = mp.matrix([[mp.sqrt(sig[i]) * vh[i, j] * decay[j] for j in range(n)] for i in range(rank)])
                v = bt * rinv
                h = mp.eye(rank) + v * matrices["Ccross"] * a
                hinv = mp.inverse(h)
                left = matrices["Bperp"] * a * hinv
                lu, lr = mp.qr(left, mode="skinny")
                rv, rr = mp.qr(v.T, mode="skinny")
                ku, ks, kvh = mp.svd(lr * rr.T)
                uu, vv = lu * ku, rv * kvh.T
                cosine = mp.diag([1 / mp.sqrt(1 + x*x) for x in ks])
                sine = mp.diag([x / mp.sqrt(1 + x*x) for x in ks])
                q = matrices["G"] + matrices["G"] * vv * (cosine - mp.eye(rank)) * vv.T + matrices["Gperp"] * uu * sine * vv.T
            else:
                q = matrices["G"]
            propagation_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            if rank:
                smin = study.small_core_smin(matrices["G"], matrices["minus"] * a, v)[0]
            else:
                smin = mp.mpf(1)
            weighted = mp.matrix([[residuals[rank][i, j] * decay[i] * decay[j] for j in range(n)] for i in range(n)])
            delta = matrices["minus"] * weighted * rinv
            numerator = p1.mp_norm(delta)
            directional = min(mp.mpf(1), numerator / smin)
            transverse_norm = p1.mp_norm(delta - q * (q.T * delta))
            transverse = min(mp.mpf(1), transverse_norm / (smin - numerator)) if numerator < smin else mp.mpf(1)
            remainder_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            entropy, _ = previous.correlation_entropy(q)
            entropy_seconds = time.perf_counter() - tick
            assert abs(entropy - mp.mpf(row["entropy"])) < mp.mpf("1e-10")
            assert abs(directional - mp.mpf(row["directional_remainder"])) < mp.mpf("1e-10")
            assert abs(transverse - mp.mpf(row["transverse_remainder"])) < mp.mpf("1e-10")
            rows.append(dict(case_id=row["case_id"], length=int(row["length"]), skin=float(row["skin"]), initial=row["initial"], rank=rank,
                             context_seconds=context_seconds, propagation_seconds=propagation_seconds,
                             directional_remainder_seconds=remainder_seconds, entropy_seconds=entropy_seconds,
                             dps=static["dps"], scope="single locked rank, no rank search, no old norm-bound computation"))
    assert len(rows) == 18
    study.write_csv("component_benchmark.csv", rows)
    study.write_json(OUT / "component_benchmark_summary.json", dict(states=len(rows),
                     **{key: sum(r[key] for r in rows) for key in ("context_seconds", "propagation_seconds", "directional_remainder_seconds", "entropy_seconds")},
                     precision_digits=200, scope="local per-component pilot at t=L/2; not HPC or optimized throughput",
                     legacy_prediction_timing_note="prediction_costs propagation_seconds includes base.reduced_at_rank legacy norm-bound work and rank search; use this benchmark for separated pure components"))


def core_edges():
    rows = []
    with mp.workdps(120):
        n = 6
        g = mp.matrix(2 * n, n)
        g[:n, :] = mp.eye(n)
        for kind, rank in (("zero_rank", 0), ("thin_core", 2), ("full_ambient_core", 4),
                           ("dependent_columns", 3), ("smin_above_one", 6)):
            y, v = mp.matrix(2 * n, rank), mp.matrix(rank, n)
            for i in range(y.rows):
                for j in range(rank):
                    y[i, j] = mp.mpf(((i + 2*j) % 7) - 2) / 13
            for i in range(rank):
                for j in range(n):
                    v[i, j] = mp.mpf(((2*i + j) % 5) - 1) / 17
            if kind == "dependent_columns":
                y = mp.matrix(2 * n, rank)
                for i in range(rank):
                    for j in range(n):
                        v[i, j] = i + 1
            if kind == "smin_above_one":
                y = g.copy()
                v = mp.eye(n)
            reduced, order, residual = study.small_core_smin(g, y, v)
            full = previous.mp_smin(g + y*v)
            error = abs(reduced - full) / full
            assert error < mp.mpf("1e-90")
            if kind == "smin_above_one":
                assert abs(reduced - 2) < mp.mpf("1e-90")
            rows.append(dict(kind=kind, n=n, rank=rank, core_order=order, core_smin=str(reduced),
                             full_smin=str(full), relative_error=str(error), reconstruction_residual=str(residual)))
    study.write_json(OUT / "core_edge_checks.json", dict(status="passed", cases=rows,
                     scope="post-lock algebra audit of fixed formula; no forecast adjustment or holdout claim"))


def memory_diagnostic():
    rows = []
    for receipt_path in sorted((OUT / "cases").glob("*.json")):
        receipt = study.read_json(receipt_path)
        case = receipt["case"]
        family = study.spatial.identifier(case["length"], 2., case["skin"])
        with gzip.open(OUT / "predictions" / (family + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
            predictions = {r["initial"]: r for r in json.load(stream) if r["case_id"] == case["case_id"]}
        with gzip.open(OUT / "cases" / (case["case_id"] + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
            truth = json.load(stream)[-1]
        with mp.workdps(260):
            a, b = predictions["left"], predictions["right"]
            values_a = {k: mp.mpf(v) for k, v in a["values"].items()}
            values_b = {k: mp.mpf(v) for k, v in b["values"].items()}
            bound = study.pair_bound(values_a, values_b, study.spatial.decode(a["Q"]), study.spatial.decode(b["Q"]))
            qa = study.spatial.decode(truth["states"]["left"]["Q"])
            qb = study.spatial.decode(truth["states"]["right"]["Q"])
            actual = previous.mp_opnorm(qa - qb * (qb.T * qa))
            assert bound["memory_lower"] <= actual + mp.mpf("1e-30")
            rows.append({**case, "initial_a": "left", "initial_b": "right", "actual_pair_distance": float(actual),
                         **{k: float(v) for k, v in bound.items()},
                         "actual_projector_memory": actual > mp.mpf(".1"),
                         "direct_guarantee": bound["direct_lower"] > mp.mpf(".1"),
                         "radial_guarantee": bound["radial_lower"] > mp.mpf(".1"),
                         "scope": "retrospective additional pairs, excluded from 38 locked memory forecasts"})
    study.write_csv("retrospective_left_right_memory.csv", rows)
    study.write_json(OUT / "retrospective_memory_summary.json", dict(pairs=len(rows),
                     actual_projector_memory=sum(r["actual_projector_memory"] for r in rows),
                     direct_guarantees=sum(r["direct_guarantee"] for r in rows), radial_guarantees=sum(r["radial_guarantee"] for r in rows),
                     direct_added_guarantees=sum(r["direct_guarantee"] and not r["radial_guarantee"] for r in rows),
                     scope="post-lock additional-pair mechanism diagnostic; not additional held-out forecast validation"))


def verify():
    study.prepare()
    study.check_hashes("spatial_lock_hash.json")
    study.check_hashes("prediction_lock_hash.json")
    lock = study.read_json(OUT / "prediction_lock.json")
    inputs = audit.read_csv(OUT / "input_manifest.csv")
    for entry in inputs + lock["inputs"]:
        assert audit.sha256(ROOT / entry["path"]) == entry["sha256"], entry["path"]
    spatial_lock = study.read_json(OUT / "spatial_hypothesis_lock.json")
    assert lock["script_sha256"] == spatial_lock["script_sha256"] == audit.sha256(study.SOURCE)
    assert lock["config_sha256"] == spatial_lock["config_sha256"] == audit.sha256(OUT / "config.json")
    receipts = [study.read_json(path) for path in (OUT / "cases").glob("*.json")]
    for value in receipts:
        assert value["started_utc"] > lock["locked_utc"]
        assert value["prediction_lock_sha256"] == audit.sha256(OUT / "prediction_lock.json")
        assert value["script_sha256"] == lock["script_sha256"]
        assert value["config_sha256"] == lock["config_sha256"]
        assert previous.accepted_precision(value["reference_convergence"])
        assert len(value["precision_digits"]) >= 2
    for path in (OUT / "static").glob("*.json"):
        value = study.read_json(path)
        assert value["started_utc"] > spatial_lock["locked_utc"]
        assert max(value["reference_convergence"][k] for k in ("relative_scalar", "compression_relative")) <= 1e-8
        assert max(value["reference_convergence"][k] for k in ("matrix_absolute", "gain_entropy")) <= 1e-10
    assert len(list((OUT / "static").glob("*.json"))) == 8
    assert study.read_json(OUT / "core_edge_checks.json")["status"] == "passed"
    assert study.read_json(OUT / "spatial_summary.json")["covered_rows"] == 12
    for value in study.read_json(OUT / "literature/source_index.json"):
        assert audit.sha256(OUT / "literature" / (value["key"] + ".pdf")) == value["pdf_sha256"]
        assert audit.sha256(OUT / "literature" / (value["key"] + ".txt")) == value["text_sha256"]
    result = study.read_json(OUT / "summary.json")
    assert (result["conditions"], result["states"], result["pairs"]) == (19, 57, 38)
    metrics = result["metrics"][0]
    assert metrics["wrong_selected"] == metrics["wrong_unselected"] == 0
    for key in ("max_projector_remainder_violation", "max_distance_interval_violation", "max_entropy_interval_violation", "max_memory_lower_violation"):
        assert result[key] <= 1e-10, (key, result[key])
    links = local_links()
    source = Path(__file__)
    (OUT / "code" / (audit.sha256(source).lower() + ".py")).write_bytes(source.read_bytes())
    manifest = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=audit.sha256(p))
                for folder in (OUT, FIG) for p in sorted(folder.rglob("*")) if p.is_file() and p.name not in ("artifact_manifest.csv", "delivery_verification.json", "environment_analysis_verify.json")]
    docs = [ROOT / "README.md", *sorted((ROOT / "research_doc").rglob("*.md"))]
    scripts = [study.SOURCE, source, ROOT / "scripts/collect_directional_prediction_literature.py"]
    manifest.extend(dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=audit.sha256(p)) for p in docs + scripts)
    assert len({r["path"] for r in manifest}) == len(manifest)
    study.write_csv("artifact_manifest.csv", manifest)
    study.write_json(OUT / "delivery_verification.json", dict(status="passed", utc=datetime.now(timezone.utc).isoformat(),
                     historical_inputs=len(inputs), locked_inputs=len(lock["inputs"]), manifest_files=len(manifest), local_links=links,
                     static_families=8, new_conditions=len(receipts), new_states=57, new_pairs=38,
                     historical_failures_preserved=True, high_precision_remainder_coverage_checked=True,
                     spatial_lock_utc=spatial_lock["locked_utc"], prediction_lock_utc=lock["locked_utc"],
                     no_HPC_execution=True, no_asymptotic_or_continuous_time_claim=True))
    study.emit(dict(stage="verification", files=len(manifest), links=links, status="passed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("summarize", "spatial", "plots", "benchmark", "core_edges", "memory_diagnostic", "verify"))
    args = parser.parse_args()
    tick = time.perf_counter()
    with threadpool_limits(limits=1):
        globals()[args.stage]()
    info = audit.environment("analysis_" + args.stage)
    info.update(elapsed_seconds=time.perf_counter() - tick)
    study.write_json(OUT / ("environment_analysis_" + args.stage + ".json"), info)


if __name__ == "__main__":
    main()
