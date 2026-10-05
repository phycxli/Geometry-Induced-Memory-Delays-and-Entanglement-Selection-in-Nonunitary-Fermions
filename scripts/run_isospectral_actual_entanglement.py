"""Round-two OBC isospectral and finite-time selection pilot with precision gates."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import mpmath as mp
import numpy as np
import psutil
import scipy
from scipy import linalg
from threadpoolctl import threadpool_limits

import audit_numerical_precision as audit

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "data/prl_p0_precision"
OUT = ROOT / "data/prl_p1_isospectral_actual"
FIG = ROOT / "figures/prl_p1_isospectral_actual"
INITIALS = audit.INITIALS


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def log(stage, value):
    value = {"utc": datetime.now(timezone.utc).isoformat(), **value}
    with (OUT / f"events_{stage}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)


def case_key(length, gamma, skin, steps):
    return f"L{length}_gamma{str(gamma).replace('.', 'p')}_g{str(skin).replace('.', 'p')}_T{steps}"


def all_cases():
    cases = {}

    def add(length, gamma, skin, steps, group):
        key = case_key(length, gamma, skin, steps)
        if key not in cases:
            cases[key] = dict(case_id=key, length=length, gamma=gamma, skin=skin, steps=steps,
                              dt=0.05, physical_time=steps * 0.05, boundary="open", t1=0.5, t2=1.0, groups=[])
        cases[key]["groups"].append(group)

    for skin in (0.0, 0.05, 0.25):
        add(32, 0.5, skin, 300, "pilot")
    for length in (16, 32):
        for gamma in (1.6, 2.0):
            for skin in (0.0, 0.25):
                for steps in (100, 300):
                    add(length, gamma, skin, steps, "controls")
    for length in (16, 32):
        for skin in (0.0, 0.25):
            for steps in (20, 40, 60, 80, 100, 150, 200, 300):
                add(length, 2.0, skin, steps, "time")
            for steps in (100, 300):
                add(length, 0.5, skin, steps, "time")
    return list(cases.values())


def configuration():
    return dict(study_date="2026-10-04", calculation_version=1, cases=all_cases(), initials=list(INITIALS),
                precision_digits=[48, 80], adaptive_precision_digits=120,
                precision_thresholds={"reference_projector": 1e-10, "reference_entropy": 1e-10,
                                      "reference_relative_scalar": 1e-8, "double_projector": 1e-6,
                                      "double_entropy": 1e-6, "orthogonality": 1e-10, "similarity_relative": 1e-12},
                physical_tolerances={"selection_projector": 0.05, "selection_absolute_entropy": 0.05,
                                     "memory_pair_projector": 0.1, "memory_pair_absolute_entropy": 0.1},
                time_definition="first sampled time passing both tolerances and staying passed at every later sampled point; right-censored if no event",
                scope="P1 audited actual-state pilot and P2 initial mechanism/time pilot; no thermodynamic phase claims, no HPC production",
                why_time_grid="conditional P2 pilot after controls identify a selection point; fixed grid declared before results")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(exist_ok=True)
    (OUT / "code").mkdir(exist_ok=True)
    current = configuration()
    path = OUT / "config.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == current, "frozen round-two configuration changed"
    else:
        write_json(path, current)
    manifest = OUT / "input_manifest.csv"
    if not manifest.exists():
        original = [ROOT / row["path"] for row in audit.read_csv(P0 / "baseline_manifest.csv")
                    if row["path"] != "README.md" and not row["path"].startswith("research_doc/")]
        inputs = [*original, *[p for p in P0.rglob("*") if p.is_file()], ROOT / "scripts/audit_numerical_precision.py"]
        write_csv("input_manifest.csv", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                                              sha256=audit.sha256(p)) for p in sorted(set(inputs))])
    for row in audit.read_csv(manifest):
        assert audit.sha256(ROOT / row["path"]) == row["sha256"], f"input changed: {row['path']}"
    source = Path(__file__).read_bytes()
    (OUT / "code" / (hashlib.sha256(source).hexdigest() + ".py")).write_bytes(source)


def encode_matrix(matrix):
    return [[str(matrix[i, j]) for j in range(matrix.cols)] for i in range(matrix.rows)]


def mp_norm(matrix):
    return mp.sqrt(mp.fsum(abs(x) ** 2 for x in matrix))


def mp_distance(q, w):
    difference = q * q.H - w * w.H
    difference = (difference + difference.H) / 2
    return max(abs(value) for value in mp.eighe(difference, eigvals_only=True))


def mp_graph(singular, vh, q0):
    n = q0.cols
    a, b = vh[:n, :] * q0, vh[n:, :] * q0
    overlap = mp.svd(a, compute_uv=False)
    alpha = overlap[n - 1]
    if alpha <= 0:
        raise ArithmeticError("input overlap is singular; graph representation unavailable")
    ba_inverse = mp.matrix(b.rows, n)
    for i in range(b.rows):
        solution = mp.lu_solve(a.T, mp.matrix([b[i, j] for j in range(n)]))
        for j in range(n):
            ba_inverse[i, j] = solution[j]
    k = mp.matrix(b.rows, n)
    for i in range(b.rows):
        for j in range(n):
            k[i, j] = singular[n + i] * ba_inverse[i, j] / singular[j]
    kappa = mp.svd(k, compute_uv=False)[0]
    ratio = singular[n] / singular[n - 1]
    eta = ratio / alpha
    sharp = eta * mp.sqrt(max(mp.mpf(0), 1 - alpha**2))
    return dict(alpha=alpha, overlap_condition=overlap[0] / alpha, ratio=ratio, eta=eta,
                eta_sharp=sharp, graph_norm=kappa, graph_distance=kappa / mp.sqrt(1 + kappa**2),
                bound_distance=sharp / mp.sqrt(1 + sharp**2)), k


def mp_reference(case, dps):
    from mietf_skin.gaussian import initial_slater_state

    with mp.workdps(dps):
        h = audit.mp_hamiltonian(case)
        matrix = mp.expm(-mp.j * h * mp.mpf(str(case["physical_time"])))
        scale = mp_norm(matrix)
        matrix /= scale
        w, singular, vh = mp.svd(matrix)
        n = case["length"] // 2
        wtop = w[:, :n]
        wentropy = audit.mp_entropy(wtop)
        raw = dict(dps=dps, M=encode_matrix(matrix), W=encode_matrix(wtop), VH=encode_matrix(vh),
                   singular_values=[str(x) for x in singular], log_fro_scale=str(mp.log(scale)), states={})
        converted = {"M": audit.mp_to_numpy(matrix), "W": audit.mp_to_numpy(wtop),
                     "V": audit.mp_to_numpy(vh.H[:, :n]), "s": np.array([float(x) for x in singular])}
        diagnostics = []
        for initial in INITIALS:
            q0 = mp.matrix(initial_slater_state(case["length"], pattern=initial).tolist())
            q, _ = mp.qr(matrix * q0, mode="skinny")
            ent = audit.mp_entropy(q)
            distance = mp_distance(q, wtop)
            graph, k = mp_graph(singular, vh, q0)
            m = n
            restricted = (q[:m, :] * q[:m, :].H - wtop[:m, :] * wtop[:m, :].H)
            restricted = (restricted + restricted.H) / 2
            delta = mp.fsum(abs(value) for value in mp.eighe(restricted, eigvals_only=True))
            x = min(delta / m, mp.mpf("0.5"))
            binary_bound = m * (-x * mp.log(x) - (1 - x) * mp.log(1 - x)) if x else mp.mpf(0)
            row = {**case, "initial": initial, "reference_dps": dps, "entropy": float(ent),
                   "W_entropy": float(wentropy), "entropy_error": float(abs(ent - wentropy)),
                   "distance_to_W": float(distance), "restricted_trace_distance": float(delta),
                   "binary_entropy_bound": float(binary_bound),
                   "graph_identity_error": float(abs(distance - graph["graph_distance"])),
                   **{name: float(value) for name, value in graph.items()}}
            diagnostics.append(row)
            converted["Q_" + initial] = audit.mp_to_numpy(q)
            raw["states"][initial] = dict(entropy=str(ent), distance_to_W=str(distance),
                                          graph={key: str(value) for key, value in graph.items()},
                                          Q=encode_matrix(q), K=encode_matrix(k))
        q_first, q_second = (converted["Q_" + initial] for initial in INITIALS)
        pair = dict(two_initial_projector_distance=audit.projector_error(q_first, q_second),
                    two_initial_entropy_difference=abs(diagnostics[0]["entropy"] - diagnostics[1]["entropy"]))
        return diagnostics, pair, converted, raw


def double_reference(case):
    from mietf_skin.gaussian import initial_slater_state
    from mietf_skin.ssh import ssh_no_click_hamiltonian

    h = ssh_no_click_hamiltonian(case["length"] // 2, case["t1"], case["t2"], case["gamma"], case["skin"], "open")
    step = linalg.expm(-1j * h * case["dt"])
    half = linalg.expm(-1j * h * case["dt"] / 2)
    matrix = linalg.expm(-1j * h * case["physical_time"])
    matrix = audit.normalize(matrix)
    w, singular, _ = linalg.svd(matrix, lapack_driver="gesvd")
    values = dict(M_double=matrix, W_double=w[:, :case["length"] // 2], s_double=singular)
    for initial in INITIALS:
        for method, propagator, count in (("step", step, case["steps"]), ("half", half, 2 * case["steps"])):
            q = initial_slater_state(case["length"], pattern=initial)
            for _ in range(count):
                q = linalg.qr(propagator @ q, mode="economic")[0]
            values[f"Q_{initial}_{method}"] = q
    return values


def relative_change(first, second):
    if first == second == 0:
        return 0.0
    return abs(first - second) / max(abs(first), abs(second))


def run_case(case, stage):
    path = OUT / "cases" / (case["case_id"] + ".json")
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        assert previous["config_sha256"] == audit.sha256(OUT / "config.json")
        return previous
    tick = time.perf_counter()
    double = double_reference(case)
    previous = None
    convergence = None
    references, precision_checks = [], []
    digits = [48, 80]
    for dps in digits:
        rows, pair, converted, raw = mp_reference(case, dps)
        references.append(raw)
        if previous is not None:
            old_rows, _, old = previous
            convergence = dict(M=float(linalg.norm(converted["M"] - old["M"])),
                               W=audit.projector_error(converted["W"], old["W"]),
                               Q=max(audit.projector_error(converted["Q_" + initial], old["Q_" + initial]) for initial in INITIALS),
                               S=max(abs(a["entropy"] - b["entropy"]) for a, b in zip(rows, old_rows)),
                               scalar_relative=max(relative_change(a[name], b[name]) for a, b in zip(rows, old_rows)
                                                   for name in ("ratio", "alpha", "graph_norm")))
            converged = max(convergence[name] for name in ("M", "W", "Q", "S")) <= 1e-10 and convergence["scalar_relative"] <= 1e-8
            if not converged and dps == 80:
                digits.append(120)
            if not converged and dps == 120:
                raise ArithmeticError(f"reference did not converge: {case['case_id']} {convergence}")
        precision_checks.append(dict(dps=dps, convergence=convergence))
        previous = rows, pair, converted
        log(stage, dict(case_id=case["case_id"], dps=dps, seconds=time.perf_counter() - tick,
                        reference_d=rows[0]["distance_to_W"], left_d=rows[1]["distance_to_W"], convergence=convergence))
    tolerances = configuration()["physical_tolerances"]
    for row in rows:
        initial = row["initial"]
        q = converted["Q_" + initial]
        w_error = audit.projector_error(converted["W"], double["W_double"])
        row.update(reference_status="converged", W_double_projector_error=w_error,
                   W_double_entropy_error=abs(row["W_entropy"] - audit.entropy(double["W_double"])),
                   double_sigma_next_relative_error=relative_change(float(double["s_double"][case["length"] // 2]), float(converted["s"][case["length"] // 2])))
        for method in ("step", "half"):
            dq = double[f"Q_{initial}_{method}"]
            row[f"double_{method}_projector_error"] = audit.projector_error(q, dq)
            row[f"double_{method}_entropy_error"] = abs(row["entropy"] - audit.entropy(dq))
            row[f"double_{method}_orthogonality"] = audit.opnorm(dq.conj().T @ dq - np.eye(dq.shape[1]))
        row["actual_double_status"] = "reference_validated" if row["double_step_projector_error"] <= 1e-6 and row["double_step_entropy_error"] <= 1e-6 else "reference_failed"
        row["W_double_status"] = "reference_validated" if row["W_double_projector_error"] <= 1e-6 and row["W_double_entropy_error"] <= 1e-6 else "reference_failed"
        row["controlled"] = row["distance_to_W"] <= tolerances["selection_projector"] and row["entropy_error"] <= tolerances["selection_absolute_entropy"]
        row["sufficient_bound_certified"] = row["bound_distance"] <= tolerances["selection_projector"]
        row["entropy_close_only"] = row["entropy_error"] <= tolerances["selection_absolute_entropy"] and not row["controlled"]
        assert row["graph_identity_error"] < 1e-10
        assert row["distance_to_W"] <= row["bound_distance"] + 1e-10
        assert row["entropy_error"] <= row["binary_entropy_bound"] + 1e-10
    pair.update(memory_retained=pair["two_initial_projector_distance"] > tolerances["memory_pair_projector"] or
                pair["two_initial_entropy_difference"] > tolerances["memory_pair_absolute_entropy"],
                controlled_initial_count=sum(row["controlled"] for row in rows))
    result = dict(case=case, rows=rows, memory=pair, reference_convergence=convergence,
                  precision_checks=precision_checks, config_sha256=audit.sha256(OUT / "config.json"),
                  script_sha256=audit.sha256(Path(__file__)), seconds=time.perf_counter() - tick)
    np.savez_compressed(path.with_suffix(".npz"), **double, **converted)
    with gzip.open(path.with_suffix(".raw.json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(references, stream, ensure_ascii=False)
    write_json(path, result)
    return result


def isospectral_stage():
    from mietf_skin.ssh import ssh_no_click_hamiltonian
    from mietf_skin.isospectral import trace_moments, spectral_hausdorff_distance

    rows, spectra = [], []
    families = sorted({(case["length"], case["gamma"], case["skin"]) for case in all_cases()})
    for length, gamma, skin in families:
        base = ssh_no_click_hamiltonian(length // 2, 0.5, 1.0, gamma, 0.0, "open")
        h = ssh_no_click_hamiltonian(length // 2, 0.5, 1.0, gamma, skin, "open")
        diagonal = np.repeat(np.exp(skin * np.arange(length // 2)), 2)
        similar = diagonal[:, None] * base / diagonal[None, :]
        dh = float(linalg.norm(h - similar, "fro") / linalg.norm(h, "fro"))
        eig, eigenvectors = linalg.eig(h)
        original = linalg.eigvals(base)
        residual = max(linalg.norm(h @ v - ev * v) / (linalg.norm(h) * linalg.norm(v))
                       for ev, v in zip(eig, eigenvectors.T))
        t = 15.0
        propagator = linalg.expm(-1j * h * t)
        base_m = linalg.expm(-1j * base * t)
        map_error = linalg.norm(propagator - diagonal[:, None] * base_m / diagonal[None, :], "fro") / linalg.norm(propagator, "fro")
        row = dict(length=length, gamma=gamma, skin=skin, H_similarity_relative_fro=dh,
                   unnormalized_M_similarity_relative_fro=float(map_error), physical_time=t,
                   trace_moment_max_abs_drift=float(np.max(np.abs(trace_moments(h, 4) - trace_moments(base, 4)))),
                   eigenvalue_hausdorff_drift=spectral_hausdorff_distance(eig, original), eigenpair_relative_residual=float(residual),
                   D_condition=float(diagonal.max() / diagonal.min()),
                   spectrum_claim="analytic OBC similarity; H and unnormalized exp(-iHt); GBZ beta radius exp(g)")
        assert dh <= 1e-12 and map_error <= 1e-12
        rows.append(row)
        spectra.extend(dict(length=length, gamma=gamma, skin=skin, index=i, real=float(value.real), imag=float(value.imag)) for i, value in enumerate(eig))
    write_csv("isospectral_checks.csv", rows)
    write_csv("hamiltonian_spectra.csv", spectra)


def summarize_stage():
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((OUT / "cases").glob("*.json"))]
    assert len(results) == len(all_cases()) == 49
    rows = [row for result in results for row in result["rows"]]
    memories = [{**result["case"], **result["memory"]} for result in results]
    singular_checks, gauge_checks = independent_summary_checks(results)
    write_csv("singular_precision_diagnostics.csv", singular_checks)
    write_csv("actual_state_gauge_checks.csv", gauge_checks)
    for result in results:
        for row in result["rows"]:
            row["observed_reference_entropy_change"] = result["reference_convergence"]["S"]
            row["observed_reference_projector_change"] = result["reference_convergence"]["Q"]
            m = row["length"] // 2
            x = min(row["bound_distance"], 0.5)
            row["entropy_bound_from_overlap"] = m * (-x * math.log(x) - (1 - x) * math.log1p(-x)) if x else 0.0
            row["both_tolerances_bound_certified"] = row["sufficient_bound_certified"] and row["entropy_bound_from_overlap"] <= 0.05
    write_csv("actual_entanglement.csv", rows)
    write_csv("initial_memory.csv", memories)
    write_csv("reference_precision_checks.csv", [{**result["case"], "seconds": result["seconds"],
                                                  **result["reference_convergence"], "dps": result["rows"][0]["reference_dps"]} for result in results])
    failed = [row for row in rows if row["actual_double_status"] == "reference_failed" or row["W_double_status"] == "reference_failed"]
    write_csv("double_precision_failed_states.csv", failed)
    comparisons = []
    for row in rows:
        if row["skin"] == 0:
            continue
        base = next((other for other in rows if all(other[key] == row[key] for key in ("length", "gamma", "steps", "initial")) and other["skin"] == 0), None)
        if base:
            comparisons.append({key: row[key] for key in ("case_id", "length", "gamma", "skin", "steps", "physical_time", "initial")} |
                               dict(base_entropy=base["entropy"], actual_entropy=row["entropy"], entropy_difference=row["entropy"] - base["entropy"],
                                    absolute_difference=abs(row["entropy"] - base["entropy"]),
                                    numerical_budget=max(1e-10, base["observed_reference_entropy_change"] + row["observed_reference_entropy_change"]),
                                    budget_meaning="reference convergence resolution, not a rigorous error enclosure",
                                    reference_basis="converged multiprecision actual states"))
    write_csv("isospectral_actual_comparisons.csv", comparisons)
    times = []
    for length in (16, 32):
        for skin in (0.0, 0.25):
            for initial in INITIALS:
                trajectory = sorted([row for row in rows if row["length"] == length and row["skin"] == skin and row["gamma"] == 2.0 and row["initial"] == initial], key=lambda row: row["steps"])
                event_index = next((i for i, row in enumerate(trajectory) if all(r["controlled"] for r in trajectory[i:])), None)
                times.append(dict(length=length, skin=skin, gamma=2.0, initial=initial,
                                  first_sustained_sample_time=trajectory[event_index]["physical_time"] if event_index is not None else None,
                                  bracket_lower=trajectory[event_index - 1]["physical_time"] if event_index is not None and event_index > 0 else (None if event_index is not None else trajectory[-1]["physical_time"]),
                                  bracket_upper=trajectory[event_index]["physical_time"] if event_index is not None else None,
                                  right_censored=event_index is None, left_censored=event_index == 0, observation_end=15.0,
                                  meaning="sampled persistence only; no guarantee between samples or beyond t=15"))
    write_csv("selection_time_pilot.csv", times)
    summary = dict(matrix_conditions=len(results), actual_states=len(rows), highprecision_evaluations=sum(len(r["precision_checks"]) for r in results),
                   controlled_states=sum(row["controlled"] for row in rows), memory_retained_conditions=sum(row["memory_retained"] for row in memories),
                   one_initial_controlled_and_memory_retained=sum(row["controlled_initial_count"] == 1 and row["memory_retained"] for row in memories),
                   entropy_close_only_states=sum(row["entropy_close_only"] for row in rows), double_failed_states=len(failed),
                   projector_bound_certified_states=sum(row["sufficient_bound_certified"] for row in rows),
                   both_tolerances_bound_certified_states=sum(row["both_tolerances_bound_certified"] for row in rows),
                   controlled_without_projector_bound=sum(row["controlled"] and not row["sufficient_bound_certified"] for row in rows),
                   double_ratio_relative_error_above_one_percent=sum(row["ratio_relative_error_to_reference"] > 0.01 for row in singular_checks),
                   gauge_state_checks=len(gauge_checks), max_gauge_projector_error=max(row["projector_error"] for row in gauge_checks),
                   max_graph_identity_error=max(row["graph_identity_error"] for row in rows),
                   max_reference_Q_change=max(result["reference_convergence"]["Q"] for result in results),
                   max_double_actual_P_error=max(row["double_step_projector_error"] for row in rows),
                   max_double_actual_S_error=max(row["double_step_entropy_error"] for row in rows),
                   max_double_W_P_error=max(row["W_double_projector_error"] for row in rows),
                   max_double_W_S_error=max(row["W_double_entropy_error"] for row in rows),
                   max_QR_orthogonality=max(row[f"double_{method}_orthogonality"] for row in rows for method in ("step", "half")),
                   max_binary_entropy_bound_violation=max(row["entropy_error"] - row["binary_entropy_bound"] for row in rows),
                   min_resolved_ratio=min(row["ratio"] for row in rows),
                   max_reference_scalar_relative_change=max(result["reference_convergence"]["scalar_relative"] for result in results),
                   max_local_seconds=max(result["seconds"] for result in results),
                   scope="audited finite L=16,32, t<=15 P1/P2 pilot; no long-time or thermodynamic phase; deterministic grid has no sampling CI")
    write_json(OUT / "summary.json", summary)
    plot_results(rows, memories, comparisons, times)
    log("summarize", summary)


def independent_summary_checks(results):
    by_parameters = {(r["case"]["length"], r["case"]["gamma"], r["case"]["skin"], r["case"]["steps"]): r for r in results}
    singular_checks, gauge_checks = [], []
    for result in results:
        case = result["case"]
        with np.load(OUT / "cases" / (case["case_id"] + ".npz")) as archive:
            n = case["length"] // 2
            reference, double = archive["s"], archive["s_double"]
            ratio, double_ratio = reference[n] / reference[n - 1], double[n] / double[n - 1]
            singular_checks.append({**case, "sigma_N_over_sigma_1": float(reference[n - 1] / reference[0]),
                                    "sigma_next_over_sigma_1": float(reference[n] / reference[0]),
                                    "ratio_reference": float(ratio), "ratio_double": float(double_ratio),
                                    "sigma_N_relative_error_to_reference": float(abs(double[n - 1] / reference[n - 1] - 1)),
                                    "sigma_next_relative_error_to_reference": float(abs(double[n] / reference[n] - 1)),
                                    "ratio_relative_error_to_reference": float(abs(double_ratio / ratio - 1)),
                                    "reference_dps": result["rows"][0]["reference_dps"]})
            if case["skin"] == 0:
                continue
            base = by_parameters[(case["length"], case["gamma"], 0.0, case["steps"])]
            diagonal = np.repeat(np.exp(case["skin"] * np.arange(n)), 2)
            with np.load(OUT / "cases" / (base["case"]["case_id"] + ".npz")) as original:
                for initial in INITIALS:
                    mapped = linalg.qr(diagonal[:, None] * original["Q_" + initial], mode="economic")[0]
                    q = archive["Q_" + initial]
                    distance = audit.projector_error(mapped, q)
                    assert distance < 1e-10, f"product-state gauge identity failed: {case['case_id']} {initial}"
                    gauge_checks.append({**case, "initial": initial, "projector_error": distance,
                                         "entropy_error": abs(audit.entropy(mapped) - audit.entropy(q)),
                                         "identity": "Ran Q_t(g) = D_g Ran Q_t(0) for fixed coordinate product initials"})
    return singular_checks, gauge_checks


def plot_results(rows, memories, comparisons, times):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 190})
    colors = {"charge_density_wave": "#188c81", "left": "#b33c43"}
    labels = {"charge_density_wave": "CDW", "left": "Left occupied"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for initial in INITIALS:
        subset = sorted([r for r in rows if r["length"] == 32 and r["gamma"] == 0.5 and r["steps"] == 300 and r["initial"] == initial], key=lambda r: r["skin"])
        axes[0].plot([r["skin"] for r in subset], [r["entropy"] for r in subset], "o-", color=colors[initial], label=labels[initial])
        axes[1].plot([r["skin"] for r in subset], [r["distance_to_W"] for r in subset], "o-", color=colors[initial], label=labels[initial])
    subset = sorted([r for r in rows if r["length"] == 32 and r["gamma"] == 0.5 and r["steps"] == 300 and r["initial"] == INITIALS[0]], key=lambda r: r["skin"])
    axes[0].plot([r["skin"] for r in subset], [r["W_entropy"] for r in subset], "s--", color="#d99d28", label="output subspace")
    spectra = audit.read_csv(OUT / "hamiltonian_spectra.csv")
    for skin, marker, color in ((0.0, "o", "#188c81"), (0.25, "x", "#b33c43")):
        subset = [r for r in spectra if int(r["length"]) == 32 and float(r["gamma"]) == 0.5 and float(r["skin"]) == skin]
        axes[2].scatter([float(r["real"]) for r in subset], [float(r["imag"]) for r in subset], marker=marker, color=color, label=f"g={skin}")
    axes[0].set(title="(a) Actual isospectral entanglement", xlabel="Skin strength g", ylabel="Half-chain entropy (nats)")
    axes[1].set(title="(b) Whole occupied-space error", xlabel="Skin strength g", ylabel="Distance to output subspace", ylim=(0, 1.05))
    axes[1].axhline(0.05, color="black", ls="--", lw=1)
    axes[2].set(title="(c) Same OBC eigenvalue spectrum", xlabel="Re E", ylabel="Im E")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(FIG / "actual_isospectral_counterexample.png")
    fig.savefig(FIG / "actual_isospectral_counterexample.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), constrained_layout=True)
    for length, ax in zip((16, 32), axes[0]):
        for initial in INITIALS:
            for skin, style in ((0.0, "-"), (0.25, "--")):
                subset = sorted([r for r in rows if r["length"] == length and r["gamma"] == 2 and r["skin"] == skin and r["initial"] == initial], key=lambda r: r["steps"])
                ax.semilogy([r["physical_time"] for r in subset], [r["distance_to_W"] for r in subset], "o" + style, color=colors[initial], ms=4, label=f"{labels[initial]}, g={skin}")
        ax.axhline(0.05, color="black", ls=":", label="selection tolerance")
        ax.set(title=f"Actual selection, L={length}, gamma=2", xlabel="Physical time", ylabel="Projector distance to output subspace")
    for length, ax in zip((16, 32), axes[1]):
        for skin, color in ((0.0, "#188c81"), (0.25, "#b33c43")):
            subset = sorted([r for r in memories if r["length"] == length and r["gamma"] == 2 and r["skin"] == skin], key=lambda r: r["steps"])
            ax.semilogy([r["physical_time"] for r in subset], [r["two_initial_projector_distance"] for r in subset], "o-", color=color, ms=4, label=f"g={skin}")
        ax.axhline(0.1, color="black", ls=":", label="pair memory tolerance")
        ax.set(title=f"Direct initial-state dependence, L={length}", xlabel="Physical time", ylabel="Two-initial projector distance")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(FIG / "selection_and_memory_time_pilot.png")
    fig.savefig(FIG / "selection_and_memory_time_pilot.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for initial in INITIALS:
        subset = [r for r in rows if r["initial"] == initial]
        axes[0].loglog([r["bound_distance"] for r in subset], [r["distance_to_W"] for r in subset], "o", color=colors[initial], ms=4, label=labels[initial])
        axes[1].loglog([r["double_step_projector_error"] for r in subset], [r["distance_to_W"] for r in subset], "o", color=colors[initial], ms=4, label=labels[initial])
    limits = (1e-20, 2)
    axes[0].plot(limits, limits, "k--", lw=1, label="bound equality")
    axes[0].set(title="(a) Overlap-aware sufficient bound", xlabel="Bound on projector distance", ylabel="Reference projector distance", xlim=limits, ylim=limits)
    axes[1].axvline(1e-6, color="black", ls="--", lw=1, label="numerical tolerance")
    axes[1].set(title="(b) Actual QR precision audit", xlabel="Double QR vs reference error", ylabel="Physical distance to output subspace")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(FIG / "overlap_mechanism_and_precision.png")
    fig.savefig(FIG / "overlap_mechanism_and_precision.pdf")
    plt.close(fig)


def verify_stage():
    for row in audit.read_csv(OUT / "input_manifest.csv"):
        assert audit.sha256(ROOT / row["path"]) == row["sha256"], f"input altered: {row['path']}"
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["matrix_conditions"] == 49 and summary["actual_states"] == 98
    assert summary["max_reference_Q_change"] <= 1e-10 and summary["max_reference_scalar_relative_change"] <= 1e-8
    assert summary["max_gauge_projector_error"] <= 1e-10
    for case in all_cases():
        path = OUT / "cases" / (case["case_id"] + ".json")
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row["case"] == case
        assert row["config_sha256"] == audit.sha256(OUT / "config.json")
        executed_source = OUT / "code" / (row["script_sha256"].lower() + ".py")
        assert audit.sha256(executed_source) == row["script_sha256"]
        with np.load(path.with_suffix(".npz")) as archive:
            assert all(np.all(np.isfinite(archive[key])) for key in archive.files)
        with gzip.open(path.with_suffix(".raw.json.gz"), "rt", encoding="utf-8") as stream:
            references = json.load(stream)
            assert len(references) >= 2 and references[-1]["dps"] == row["rows"][0]["reference_dps"]
            assert [r["dps"] for r in references] == [p["dps"] for p in row["precision_checks"]]
    links = 0
    for path in [ROOT / "README.md", *sorted((ROOT / "research_doc").rglob("*.md"))]:
        content = path.read_text(encoding="utf-8-sig")
        assert "\ufffd" not in content
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            target = target.split("#", 1)[0]
            if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            assert (path.parent / target.strip("<>")).exists(), f"broken link: {path} {target}"
            links += 1
    write_json(OUT / "delivery_verification.json", dict(status="passed", conditions=49, states=98,
               immutable_inputs=len(audit.read_csv(OUT / "input_manifest.csv")), local_links_checked=links,
               raw_precision_cases_checked=49, numpy_arrays_finite=True))
    log("verify", dict(status="passed", conditions=49, states=98, local_links=links))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "controls", "time", "summarize", "verify"))
    args = parser.parse_args()
    prepare()
    tick = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="prl_p1_baseline_") as directory:
        with zipfile.ZipFile(P0 / "baseline_snapshot.zip") as archive:
            for member in archive.namelist():
                if member.startswith("src/"):
                    archive.extract(member, directory)
        sys.path.insert(0, str(Path(directory) / "src"))
        with threadpool_limits(limits=1):
            if args.stage in ("pilot", "controls", "time"):
                if args.stage == "time":
                    accepted = [json.loads(path.read_text(encoding="utf-8")) for path in (OUT / "cases").glob("*.json")]
                    assert any(row["controlled"] for item in accepted for row in item["rows"]), "time pilot requires a validated control point"
                isospectral_stage()
                for case in all_cases():
                    if args.stage in case["groups"]:
                        run_case(case, args.stage)
            elif args.stage == "summarize":
                summarize_stage()
            else:
                verify_stage()
    info = audit.environment(args.stage)
    info.update(round="P1/P2 pilot", elapsed_seconds=time.perf_counter() - tick,
                audit_script_sha256=audit.sha256(Path(__file__)), config_sha256=audit.sha256(OUT / "config.json"),
                peak_rss_mb=getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**20)
    write_json(OUT / f"environment_{args.stage}.json", info)
    if args.stage == "verify":
        paths = [p for d in (OUT, FIG) for p in d.rglob("*") if p.is_file() and p.name != "artifact_manifest.csv"]
        paths += [Path(__file__), ROOT / "README.md", *sorted((ROOT / "research_doc").rglob("*.md"))]
        write_csv("artifact_manifest.csv", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=audit.sha256(p)) for p in sorted(paths)])


if __name__ == "__main__":
    main()
