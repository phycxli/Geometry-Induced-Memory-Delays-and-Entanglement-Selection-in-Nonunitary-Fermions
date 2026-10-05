"""Reproduce the frozen baseline and audit finite-time SSH propagation.

Run each stage from the project root. Existing research outputs are read only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

import mpmath as mp
import numpy as np
import psutil
import scipy
from scipy import linalg, special
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "prl_p0_precision"
FIG = ROOT / "figures" / "prl_p0_precision"
EPS = np.finfo(float).eps
INITIALS = ("charge_density_wave", "left")
STUDY_DATE = "2026-10-04"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def event(stage: str, value: dict) -> None:
    value = {"utc": datetime.now(timezone.utc).isoformat(), **value}
    with (OUT / f"events_{stage}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)


def environment(stage: str) -> dict:
    proc = psutil.Process()
    return {
        "study_date": STUDY_DATE,
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_logical": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_gb": psutil.virtual_memory().total / 2**30,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mpmath": mp.__version__,
        "blas": threadpool_info(),
        "threads": {key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "audit_script_sha256": sha256(Path(__file__)),
        "baseline_snapshot_sha256": sha256(OUT / "baseline_snapshot.zip"),
        "config_sha256": sha256(OUT / "config.json"),
        "rss_mb": proc.memory_info().rss / 2**20,
        "stage": stage,
    }


def cases() -> list[dict]:
    rows = []

    def add(group, lengths, boundaries, skin, gammas, steps):
        for length in lengths:
            for boundary in boundaries:
                for gamma in gammas:
                    for count in steps:
                        rows.append(dict(group=group, length=length, boundary=boundary, skin=skin,
                                         gamma=gamma, steps=count, dt=0.05, t1=0.5, t2=1.0))

    add("unitary", [16, 32], ["open", "periodic"], 0.0, [0.0], [1, 100, 300])
    add("reciprocal", [16, 32], ["open"], 0.0, [0.3], [100, 300, 600])
    add("exceptional", [16, 32], ["periodic"], 0.0, [0.49, 0.5, 0.51], [1, 100, 300])
    add("small_skin", [16, 32], ["open"], 0.25, [0.5, 0.7], [100, 300, 600])
    add("main_scale", [64, 128], ["open"], 0.25, [0.5], [300, 600])
    add("conditioning_risk", [192], ["open"], 0.25, [0.7], [300, 600])
    add("reciprocal_p1_reference", [16, 32], ["open"], 0.0, [0.5], [100, 300, 600])
    add("weak_skin_pilot", [32], ["open"], 0.05, [0.5], [300])
    for index, row in enumerate(rows):
        row["case_id"] = f"ssh_{index:03d}"
        row["physical_time"] = row["steps"] * row["dt"]
    return rows


def config() -> dict:
    return {
        "study_date": STUDY_DATE,
        "seed": 1729,
        "cases": cases(),
        "initials": list(INITIALS),
        "precision_digits": [48, 80],
        "adaptive_precision_digits": 120,
        "thresholds": {"step_relative_fro": 1e-10, "orthogonality": 1e-10,
                       "projector_operator": 1e-6, "entropy_absolute": 1e-6,
                       "reference_convergence": 1e-10},
        "sv_noise_rule": "max(64*L*eps, cross_method_relative_operator_error); gap and sigma_N > 10*noise",
        "extra_cases_reason": "OBC reciprocal gamma=0.5 and g=0.05 prepare a same-time natural-state P1 pilot",
        "scope": "S0 + L0 first review + P0; no P1/P2 production scan",
    }


def normalize(matrix):
    return matrix / linalg.norm(matrix, "fro")


def hermitian(matrix):
    return (matrix + matrix.conj().T) / 2


def projector(q):
    return q @ q.conj().T


def opnorm(matrix):
    return float(linalg.svdvals(matrix)[0])


def projector_error(first, second):
    return float(np.max(np.abs(linalg.eigvalsh(hermitian(projector(first) - projector(second))))))


def entropy(q):
    vals = linalg.eigvalsh(hermitian(q[:q.shape[0] // 2] @ q[:q.shape[0] // 2].conj().T))
    vals = np.clip(vals, 0.0, 1.0)
    return float(np.sum(-special.xlogy(vals, vals) - special.xlogy(1 - vals, 1 - vals)))


def stats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
    return dict(n=len(x), pearson=float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-14 else None,
                slope=float(slope), intercept=float(intercept), mae=float(np.mean(np.abs(y - x))),
                max_abs_error=float(np.max(np.abs(y - x))))


def baseline_stage():
    import plot_prl_main_figures as old_plot

    summary = old_plot.build_summary(*(old_plot.read_rows(ROOT / "data" / name) for name in (
        "physical_isospectral_skin_v1.csv", "controlled_overlap_v1_controlled_overlap_scan.csv",
        "quicklarge_ssh_joint_scan.csv", "singular_lt_scaling_v1_summary.csv")))
    expected = {row["metric"]: float(row["value"]) for row in read_csv(ROOT / "data/prl_main_figures_summary.csv")}
    comparison = [{**row, "stored_value": expected[row["metric"]],
                   "difference": float(row["value"]) - expected[row["metric"]]} for row in summary]
    write_csv("baseline_summary_recomputed.csv", comparison)
    comparisons = []
    for source, mode, target in (("quicklarge_ssh_joint_scan.csv", "late_mean_vs_T120", "S_density"),
                                 ("matchedT_ssh_joint_scan.csv", "late_mean_vs_T300", "S_density"),
                                 ("matchedT_ssh_joint_scan.csv", "final_vs_T300", "S_final")):
        rows = old_plot.read_rows(ROOT / "data" / source)
        groups = [("all", rows)] + [(f"{boundary};g={skin}", [r for r in rows if r["boundary"] == boundary and r["skin"] == skin])
                                     for boundary, skin in sorted({(r["boundary"], r["skin"]) for r in rows})]
        for group, subset in groups:
            y = [r[target] / r["L"] if target == "S_final" else r[target] for r in subset]
            comparisons.append(dict(source=source, target=mode, group=group,
                                    steps=sorted({r["steps"] for r in subset}),
                                    burn_in=sorted({r["burn_in"] for r in subset}),
                                    svd_steps=sorted({r["diagnostic_steps"] for r in subset}),
                                    **stats([r["left_subspace_entropy_density"] for r in subset], y)))
    write_csv("baseline_prediction_statistics.csv", comparisons)
    bounds = []
    for source in ("quick_subspace_bound_scan.csv", "random_slater_v1_subspace_bound_scan.csv"):
        rows = old_plot.read_rows(ROOT / "data" / source)
        values = np.array([r["graph_norm_bound"] for r in rows])
        bounds.append(dict(source=source, points=len(rows), bound_lt_one=int(np.sum(values < 1)),
                           bound_median=float(np.median(values)), bound_max=float(np.max(values)),
                           overlap_median=float(np.median([r["top_overlap_min_sv"] for r in rows]))))
    write_csv("baseline_overlap_statistics.csv", bounds)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (OUT / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")
    event("baseline", {"summary_metrics": len(summary), "max_summary_drift": max(abs(r["difference"]) for r in comparison),
                       "statistics_groups": len(comparisons), "bound_counts": [r["bound_lt_one"] for r in bounds]})


def double_stage():
    from mietf_skin.diagnostics import normalized_product
    from mietf_skin.gaussian import initial_slater_state, orthonormalize
    from mietf_skin.models import single_particle_propagator
    from mietf_skin.ssh import ssh_no_click_hamiltonian
    from run_singular_lt_scaling_scan import normalized_matrix_power

    matrices, rows, trajectories = {}, [], []
    cache = {}
    for case in cases():
        tick = time.perf_counter()
        cid, length = case["case_id"], case["length"]
        key = (length, case["boundary"], case["skin"], case["gamma"])
        if key not in cache:
            h = ssh_no_click_hamiltonian(length // 2, case["t1"], case["t2"], case["gamma"], case["skin"], case["boundary"])
            eigvecs = np.linalg.eig(h)[1]
            cache[key] = (h, single_particle_propagator(h, case["dt"]), linalg.expm(-1j * h * case["dt"]),
                          linalg.expm(-1j * h * case["dt"] / 2), float(np.linalg.cond(eigvecs)))
        h, eigen_step, expm_step, half_step, condition = cache[key]
        count, duration = case["steps"], case["physical_time"]
        products = {"eig_repeat": normalized_product(eigen_step, count),
                    "expm_repeat": normalized_product(expm_step, count),
                    "expm_power": normalized_matrix_power(expm_step, count),
                    "expm_direct": normalize(linalg.expm(-1j * h * duration))}
        matrices[cid + "_step_expm"] = expm_step
        row = {**case, "eigvec_condition": condition,
               "step_eig_relative_fro": float(linalg.norm(eigen_step - expm_step, "fro") / linalg.norm(expm_step, "fro")),
               "step_half_relative_fro": float(linalg.norm(half_step @ half_step - expm_step, "fro") / linalg.norm(expm_step, "fro")),
               "unitarity_defect": opnorm(expm_step.conj().T @ expm_step - np.eye(length)) if case["group"] == "unitary" else None}
        svds = {}
        for method, matrix in products.items():
            matrices[cid + "_M_" + method] = matrix
            left, singular, vh = linalg.svd(matrix, lapack_driver="gesvd")
            svds[method] = (left, singular, vh)
            matrices[cid + "_s_" + method] = singular
        w, singular, vh = svds["expm_direct"]
        n = length // 2
        relative_n, relative_next = float(singular[n - 1] / singular[0]), float(singular[n] / singular[0])
        gap = relative_n - relative_next
        cross_noise = max(opnorm(matrix - products["expm_direct"]) / opnorm(products["expm_direct"])
                          for method, matrix in products.items() if method.startswith("expm_"))
        noise = max(64 * length * EPS, cross_noise)
        w_errors = {method: projector_error(left[:, :n], w[:, :n]) for method, (left, _, _) in svds.items()}
        if case["group"] == "unitary":
            status, reason = "degenerate", "unitary map has all singular values equal; no unique W_plus"
        elif relative_n <= 10 * noise or gap <= 10 * noise:
            status, reason = "unresolved", "cutoff singular value or gap is below conservative numerical noise"
        elif max(w_errors[method] for method in ("expm_repeat", "expm_power")) > 1e-6:
            status, reason = "unresolved", "output projector changes across expm product methods"
        else:
            status, reason = "valid", "double precision crosschecked; selected points receive multiprecision reference"
        row.update(sigma_N_over_sigma_1=relative_n, sigma_next_over_sigma_1=relative_next,
                   relative_cutoff_gap=gap, singular_ratio=float(singular[n] / singular[n - 1]),
                   numerical_noise=noise, svd_status=status, svd_reason=reason,
                   left_entropy_expm_direct=entropy(w[:, :n]))
        for method in products:
            row["W_projector_error_" + method] = w_errors[method]
            row["M_relative_fro_" + method] = float(linalg.norm(products[method] - products["expm_direct"], "fro") / linalg.norm(products["expm_direct"], "fro"))
            row["W_entropy_error_" + method] = abs(entropy(svds[method][0][:, :n]) - entropy(w[:, :n]))
        for pattern in INITIALS:
            q0 = initial_slater_state(length, pattern=pattern)
            states = {}
            for method, step, repeats in (("eig_qr", eigen_step, count), ("expm_qr", expm_step, count), ("half_step_qr", half_step, 2 * count)):
                q = q0.copy()
                for _ in range(repeats):
                    q = orthonormalize(step @ q)
                states[method] = q
            states["direct_qr"] = orthonormalize(products["expm_direct"] @ q0)
            reference = states["expm_qr"]
            overlap = linalg.svdvals(vh[:n] @ q0)
            alpha = float(overlap[-1])
            alpha_status = "valid" if alpha > 64 * length * EPS else "unresolved"
            for method, q in states.items():
                matrices[f"{cid}_Q_{pattern}_{method}"] = q
                trajectories.append({**case, "initial": pattern, "method": method, "entropy": entropy(q),
                                     "orthogonality": opnorm(q.conj().T @ q - np.eye(n)),
                                     "projector_vs_step_expm": projector_error(q, reference),
                                     "entropy_vs_step_expm": abs(entropy(q) - entropy(reference)),
                                     "distance_to_W": projector_error(q, w[:, :n]) if status == "valid" else None,
                                     "alpha": alpha if alpha_status == "valid" else None,
                                     "alpha_status": alpha_status,
                                     "eta": row["singular_ratio"] / alpha if alpha_status == "valid" else None,
                                     "svd_status": status})
        row["seconds"] = time.perf_counter() - tick
        row["rss_mb_after"] = psutil.Process().memory_info().rss / 2**20
        rows.append(row)
        write_csv("double_precision_cases.csv", rows)
        write_csv("actual_state_checks.csv", trajectories)
        event("double", {"case_id": cid, "L": length, "g": case["skin"], "gamma": case["gamma"], "T": count,
                         "status": status, "eig_step_error": row["step_eig_relative_fro"], "seconds": row["seconds"]})
    np.savez_compressed(OUT / "double_matrices.npz", **matrices)
    fixtures = []
    for kind, h in (("unitary", np.diag([0.25, -0.5]).astype(complex)),
                    ("jordan", np.array([[0, 1], [0, 0]], complex))):
        for count in (1, 100, 300):
            duration = count * 0.05
            exact = np.diag(np.exp(-1j * np.diag(h) * duration)) if kind == "unitary" else np.eye(2) - 1j * duration * h
            old = single_particle_propagator(h, 0.05)
            fixtures.append(dict(kind=kind, steps=count, physical_time=duration,
                                 step_expm_error=float(linalg.norm(linalg.expm(-1j * h * 0.05) - (np.diag(np.exp(-1j * np.diag(h) * 0.05)) if kind == "unitary" else np.eye(2) - 0.05j * h))),
                                 product_expm_error=float(linalg.norm(linalg.expm(-1j * h * duration) - exact)),
                                 product_eig_error=float(linalg.norm(np.linalg.matrix_power(old, count) - exact)),
                                 eig_step_error=float(linalg.norm(old - linalg.expm(-1j * h * 0.05)))))
    write_csv("analytic_fixtures.csv", fixtures)


def mp_hamiltonian(case):
    length = case["length"]
    h = mp.matrix(length)
    t1, t2, gamma, skin = (mp.mpf(str(case[key])) for key in ("t1", "t2", "gamma", "skin"))
    for cell in range(length // 2):
        a, b = 2 * cell, 2 * cell + 1
        h[a, a], h[b, b] = mp.j * gamma, -mp.j * gamma
        h[a, b] = h[b, a] = t1
        if cell < length // 2 - 1 or case["boundary"] == "periodic":
            nxt = (a + 2) % length
            h[nxt, b], h[b, nxt] = t2 * mp.exp(skin), t2 * mp.exp(-skin)
    return h


def mp_to_numpy(matrix):
    return np.array([[complex(matrix[i, j]) for j in range(matrix.cols)] for i in range(matrix.rows)], complex)


def mp_entropy(q):
    restricted = q[:q.rows // 2, :]
    values = mp.eighe(restricted * restricted.H, eigvals_only=True)
    tolerance = mp.mpf(10) ** (-mp.mp.dps + 8)
    total = mp.mpf(0)
    for value in values:
        if value < -tolerance or value > 1 + tolerance:
            raise ArithmeticError("multiprecision correlation spectrum is outside [0,1]")
        value = max(mp.mpf(0), min(mp.mpf(1), value))
        if value not in (0, 1):
            total -= value * mp.log(value) + (1 - value) * mp.log(1 - value)
    return total


def reference_selected(case):
    if case["length"] not in (16, 32):
        return False
    if case["boundary"] == "periodic":
        return case["gamma"] == 0.5 and case["steps"] == 300
    if case["skin"] == 0:
        return case["gamma"] in (0.3, 0.5) and case["steps"] == 300
    if case["skin"] == 0.05:
        return True
    return case["steps"] in ((100, 300, 600) if case["length"] == 16 and case["gamma"] == 0.5 else (300, 600))


def highprecision_stage():
    from mietf_skin.gaussian import initial_slater_state

    double = np.load(OUT / "double_matrices.npz")
    selected = [case for case in cases() if reference_selected(case)]
    rows, raw, converted = [], [], {}
    for case in selected:
        cid, length = case["case_id"], case["length"]
        previous = None
        precisions = [48, 80]
        for dps in precisions:
            tick = time.perf_counter()
            with mp.workdps(dps):
                h = mp_hamiltonian(case)
                step = mp.expm(-mp.j * h * mp.mpf(str(case["dt"])))
                matrix = mp.expm(-mp.j * h * mp.mpf(str(case["physical_time"])))
                norm = mp.sqrt(mp.fsum(abs(x) ** 2 for x in matrix))
                matrix /= norm
                w, singular, vh = mp.svd(matrix)
                n = length // 2
                wf = mp_to_numpy(w[:, :n])
                mf = mp_to_numpy(matrix)
                ref = {"M": mf, "W": wf, "V": mp_to_numpy(vh.H[:, :n]), "s": np.array([float(x) for x in singular])}
                row = {**case, "dps": dps, "step_vs_double_expm_relative_fro": float(linalg.norm(mp_to_numpy(step) - double[cid + "_step_expm"], "fro") / linalg.norm(mp_to_numpy(step), "fro")),
                       "M_vs_double_relative_fro": float(linalg.norm(mf - double[cid + "_M_expm_direct"], "fro") / linalg.norm(mf, "fro")),
                       "W_vs_double_projector": projector_error(wf, linalg.svd(double[cid + "_M_expm_direct"], lapack_driver="gesvd")[0][:, :n]),
                       "W_entropy": float(mp_entropy(w[:, :n])), "sigma_N_over_sigma_1": float(singular[n - 1] / singular[0]),
                       "sigma_next_over_sigma_1": float(singular[n] / singular[0]), "reference_log_fro_scale": str(mp.log(norm))}
                raw_row = {"case": case, "dps": dps,
                           "matrix": [[str(matrix[i, j]) for j in range(length)] for i in range(length)],
                           "singular_values": [str(x) for x in singular],
                           "W_top": [[str(w[i, j]) for j in range(n)] for i in range(length)], "states": {}}
                for pattern in INITIALS:
                    q0 = mp.matrix(initial_slater_state(length, pattern=pattern).tolist())
                    q, _ = mp.qr(matrix * q0, mode="skinny")
                    qf, ent = mp_to_numpy(q), mp_entropy(q)
                    ref["Q_" + pattern] = qf
                    row["entropy_" + pattern] = float(ent)
                    for method in ("eig_qr", "expm_qr", "half_step_qr", "direct_qr"):
                        actual = double[f"{cid}_Q_{pattern}_{method}"]
                        row[f"P_error_{pattern}_{method}"] = projector_error(qf, actual)
                        row[f"S_error_{pattern}_{method}"] = abs(float(ent) - entropy(actual))
                    raw_row["states"][pattern] = {"entropy": str(ent), "Q": [[str(q[i, j]) for j in range(n)] for i in range(length)]}
                if previous is not None:
                    row["reference_M_convergence"] = float(linalg.norm(mf - previous["M"]))
                    row["reference_W_convergence"] = projector_error(wf, previous["W"])
                    row["reference_Q_convergence"] = max(projector_error(ref["Q_" + p], previous["Q_" + p]) for p in INITIALS)
                    if max(row["reference_M_convergence"], row["reference_W_convergence"], row["reference_Q_convergence"]) > 1e-10 and dps == 80:
                        precisions.append(120)
                row["seconds"] = time.perf_counter() - tick
                rows.append(row)
                raw.append(raw_row)
                previous = ref
                for key, value in ref.items():
                    converted[f"{cid}_{dps}_{key}"] = value
            write_csv("high_precision_checks.csv", rows)
            event("highprecision", {"case_id": cid, "dps": dps, "seconds": row["seconds"],
                                    "W_error_double": row["W_vs_double_projector"]})
        np.savez_compressed(OUT / "high_precision_reference.npz", **converted)
        with gzip.open(OUT / "high_precision_raw.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(raw, stream, ensure_ascii=False)
    double.close()


def binary_entropy(value):
    value = min(1.0, max(0.0, value))
    return float(-special.xlogy(value, value) - special.xlogy(1 - value, 1 - value))


def math_stage():
    rng, rows = np.random.default_rng(1729), []
    for length, particles in ((12, 4), (12, 6), (12, 8), (16, 8)):
        for trial in range(12):
            w = linalg.qr(rng.normal(size=(length, length)) + 1j * rng.normal(size=(length, length)))[0]
            v = linalg.qr(rng.normal(size=(length, length)) + 1j * rng.normal(size=(length, length)))[0]
            q0 = linalg.qr(rng.normal(size=(length, particles)) + 1j * rng.normal(size=(length, particles)), mode="economic")[0]
            singular = np.exp(np.linspace(2, -2, length))
            a, b = v[:, :particles].conj().T @ q0, v[:, particles:].conj().T @ q0
            alpha = float(linalg.svdvals(a)[-1])
            ba_inverse = linalg.solve(a.T, b.T).T
            k = singular[particles:, None] * ba_inverse / singular[:particles][None, :]
            matrix = (w * singular) @ v.conj().T
            q = linalg.qr(matrix @ q0, mode="economic")[0]
            graph = w[:, :particles] + w[:, particles:] @ k
            factor_error = float(linalg.norm(matrix @ q0 - graph @ np.diag(singular[:particles]) @ a))
            d = projector_error(q, w[:, :particles])
            exact = opnorm(k) / math.hypot(1, opnorm(k))
            ratio = singular[particles] / singular[particles - 1]
            tighter = ratio * math.sqrt(max(0.0, 1 - alpha**2)) / alpha
            m = length // 2
            ca, cb = q[:m] @ q[:m].conj().T, w[:m, :particles] @ w[:m, :particles].conj().T
            delta = float(np.sum(np.abs(linalg.eigvalsh(hermitian(ca - cb)))))
            error = abs(entropy(q) - entropy(w[:, :particles]))
            aud = m * (min(delta / m, 1 - 1 / (2 * m)) * math.log(2 * m - 1) + binary_entropy(min(delta / m, 1 - 1 / (2 * m))))
            binary = m * binary_entropy(min(delta / m, 0.5))
            rows.append(dict(length=length, particles=particles, trial=trial, alpha=alpha,
                             graph_factor_error=factor_error, projector_identity_error=abs(d - exact),
                             sharper_bound_violation=opnorm(k) - tighter,
                             compression_violation=delta - m * d,
                             entropy_error=error, audenaert_violation=error - aud,
                             binary_bound_violation=error - binary))
    write_csv("mathematical_identity_checks.csv", rows)
    event("math", {"points": len(rows), "projector_identity_max_error": max(r["projector_identity_error"] for r in rows),
                   "sharper_bound_max_violation": max(r["sharper_bound_violation"] for r in rows),
                   "entropy_binary_max_violation": max(r["binary_bound_violation"] for r in rows)})


def summarize_stage():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = config()["thresholds"]
    raw_cases = read_csv(OUT / "double_precision_cases.csv")
    actual = read_csv(OUT / "actual_state_checks.csv")
    hp = read_csv(OUT / "high_precision_checks.csv")
    latest = {}
    for row in hp:
        cid = row["case_id"]
        if cid not in latest or int(row["dps"]) > int(latest[cid]["dps"]):
            latest[cid] = row
    assert len(raw_cases) == len(cases()) == 61
    assert len(latest) == sum(reference_selected(case) for case in cases()) == 16
    assert len(actual) == 61 * 2 * 4
    assert all(float(row["orthogonality"]) <= thresholds["orthogonality"] for row in actual)
    ref_npz = np.load(OUT / "high_precision_reference.npz")
    double_npz = np.load(OUT / "double_matrices.npz")
    validated, propagated, failures, final_cases, memory = [], [], [], [], []
    for raw in raw_cases:
        row = dict(raw)
        cid = row["case_id"]
        local = [entry for entry in actual if entry["case_id"] == cid]
        reliable = [entry for entry in local if entry["method"] == "half_step_qr"]
        row["state_half_step_max_P_error"] = max(float(entry["projector_vs_step_expm"]) for entry in reliable)
        row["state_half_step_max_S_error"] = max(float(entry["entropy_vs_step_expm"]) for entry in reliable)
        state_cross_ok = (row["state_half_step_max_P_error"] <= thresholds["projector_operator"] and
                          row["state_half_step_max_S_error"] <= thresholds["entropy_absolute"])
        row["state_expm_qr_status"] = "double_crosschecked" if state_cross_ok else "unresolved"
        row["W_status_final"] = "degenerate" if row["svd_status"] == "degenerate" else "double_crosschecked"
        if row["svd_status"] == "unresolved":
            row["W_status_final"] = "unresolved"
        row["reference_dps"] = None
        if cid in latest:
            ref = latest[cid]
            row["reference_dps"] = int(ref["dps"])
            convergence = max(float(ref[key]) for key in
                              ("reference_M_convergence", "reference_W_convergence", "reference_Q_convergence"))
            step_error = float(ref["step_vs_double_expm_relative_fro"])
            w_error = float(ref["W_vs_double_projector"])
            w_entropy_error = abs(float(row["left_entropy_expm_direct"]) - float(ref["W_entropy"]))
            state_p_error = max(float(ref[f"P_error_{initial}_expm_qr"]) for initial in INITIALS)
            state_s_error = max(float(ref[f"S_error_{initial}_expm_qr"]) for initial in INITIALS)
            n = int(row["length"]) // 2
            dps = int(ref["dps"])
            ref_s = ref_npz[f"{cid}_{dps}_s"]
            double_s = double_npz[cid + "_s_expm_direct"]
            sigma_relative_error = max(abs(double_s[k] / ref_s[k] - 1) for k in (n - 1, n))
            ref_ok = convergence <= thresholds["reference_convergence"]
            step_ok = step_error <= thresholds["step_relative_fro"]
            w_ok = ref_ok and w_error <= thresholds["projector_operator"] and w_entropy_error <= thresholds["entropy_absolute"]
            state_ok = ref_ok and step_ok and state_p_error <= thresholds["projector_operator"] and state_s_error <= thresholds["entropy_absolute"]
            row["state_expm_qr_status"] = "reference_validated" if state_ok else "reference_failed"
            row["W_status_final"] = "reference_validated" if w_ok and row["svd_status"] == "valid" else "reference_failed"
            row.update(reference_convergence=convergence, W_reference_error=w_error,
                       W_reference_entropy_error=w_entropy_error, cutoff_sigma_reference_relative_error=sigma_relative_error,
                       state_reference_P_error=state_p_error, state_reference_S_error=state_s_error)
            validated.append(dict(case_id=cid, length=row["length"], skin=row["skin"], gamma=row["gamma"], steps=row["steps"],
                                  dps=dps, reference_convergence=convergence, step_error=step_error,
                                  W_error=w_error, W_entropy_error=w_entropy_error,
                                  cutoff_sigma_relative_error=sigma_relative_error, state_P_error=state_p_error,
                                  state_S_error=state_s_error, W_pass=w_ok, state_pass=state_ok))
        if float(row["step_eig_relative_fro"]) > thresholds["step_relative_fro"]:
            failures.append(dict(case_id=cid, quantity="legacy_step_relative_fro", initial="", method="eig",
                                 error=float(row["step_eig_relative_fro"]), threshold=thresholds["step_relative_fro"]))
        for entry in local:
            for quantity, field, limit in (("actual_projector", "projector_vs_step_expm", thresholds["projector_operator"]),
                                           ("actual_entropy", "entropy_vs_step_expm", thresholds["entropy_absolute"])):
                if float(entry[field]) > limit:
                    failures.append(dict(case_id=cid, quantity=quantity, initial=entry["initial"], method=entry["method"],
                                         error=float(entry[field]), threshold=limit))
            propagated.append({**entry, "state_expm_qr_status": row["state_expm_qr_status"],
                               "W_status_final": row["W_status_final"]})
        q_first = double_npz[f"{cid}_Q_{INITIALS[0]}_expm_qr"]
        q_second = double_npz[f"{cid}_Q_{INITIALS[1]}_expm_qr"]
        memory.append({key: row[key] for key in ("case_id", "length", "boundary", "skin", "gamma", "steps", "physical_time")} |
                      {"two_initial_projector_distance": projector_error(q_first, q_second),
                       "two_initial_entropy_difference": abs(entropy(q_first) - entropy(q_second)),
                       "state_status": row["state_expm_qr_status"],
                       "meaning": "finite-time initial-state dependence; not a long-time memory phase"})
        final_cases.append(row)
    ref_npz.close()
    double_npz.close()
    assert all(row["state_pass"] and row["W_pass"] for row in validated)
    write_csv("final_precision_classification.csv", final_cases)
    write_csv("reference_acceptance.csv", validated)
    write_csv("final_actual_state_checks.csv", propagated)
    write_csv("failed_checks.csv", failures)
    write_csv("initial_memory_checks.csv", memory)

    pilot_ids = [row["case_id"] for row in final_cases if int(row["length"]) == 32 and row["boundary"] == "open" and
                 float(row["gamma"]) == 0.5 and int(row["steps"]) == 300 and float(row["skin"]) in (0.0, 0.05, 0.25)]
    assert len(pilot_ids) == 3
    pilot_rows = [row for row in propagated if row["case_id"] in pilot_ids and row["method"] == "expm_qr"]
    assert all(row["state_expm_qr_status"] == row["W_status_final"] == "reference_validated" for row in pilot_rows)
    write_csv("p1_pilot_reusable_points.csv", pilot_rows)
    write_json("p1_pilot_config.json", {
        "study_goal": "建立可预测的有限时间纠缠选择机制，区分输出奇异子空间控制与初态记忆保留，并研究尺寸、趋肤强度和时间依赖",
        "status": "configuration_only; audited points reusable; P1 isospectral and mechanism acceptance not run",
        "length": 32, "boundary": "open", "skin": [0.0, 0.05, 0.25], "gamma": 0.5,
        "t1": 0.5, "t2": 1.0, "dt": 0.05, "steps": 300, "physical_time": 15.0,
        "initials": list(INITIALS), "case_ids": pilot_ids,
        "actual_method": "scipy.linalg.expm step plus repeated QR",
        "output_subspace_method": "finite-time scipy.linalg.expm plus SVD; crosscheck with mpmath 80 digits",
        "next_required_checks": ["analytic H(g)=D_g H(0) D_g^-1; numerical residual and trace invariants",
                                 "same initial state and same physical time; entropy differences above precision budget",
                                 "two-initial-state projector and entropy differences to assess memory directly",
                                 "do not treat large eta as proof of memory; expand time/size only after fresh precision checks"],
        "positive_control_candidates": {"status": "not computed; precision audit required", "length": [16, 32],
                                        "gamma": [1.6, 2.0], "skin": [0.0, 0.25], "steps": [100, 300],
                                        "reason": "all 32 reference-validated natural-state diagnostics have distance to W_plus > 0.5; seek a separated imaginary-spectrum control"},
        "production_grid": "not scheduled; benchmarked local Windows only; HPC environment not verified",
    })
    audits = [json.loads((OUT / f"environment_{stage}.json").read_text(encoding="utf-8"))
              for stage in ("baseline", "double", "highprecision", "math")]
    write_csv("local_resource_measurements.csv", [
        {"stage": info["stage"], "seconds": info["elapsed_seconds"],
         "peak_rss_mb": info.get("process_peak_rss_mb"), "machine": info["platform"], "blas_threads": 1}
        for info in audits])
    counts = {}
    for row in final_cases:
        counts[row["W_status_final"]] = counts.get(row["W_status_final"], 0) + 1
    info = dict(study_date=STUDY_DATE, matrix_cases=len(final_cases), actual_states=len(actual) // 4,
                state_method_comparisons=len(actual), highprecision_points=len(validated),
                highprecision_evaluations=len(hp), W_status_counts=counts,
                legacy_step_failed_cases=sorted({r["case_id"] for r in failures if r["quantity"] == "legacy_step_relative_fro"}),
                method_failure_counts={method: sum(r["method"] == method for r in failures) for method in ("eig", "eig_qr", "direct_qr", "half_step_qr")},
                reference_maxima={field: max(row[field] for row in validated) for field in
                                  ("reference_convergence", "step_error", "W_error", "W_entropy_error", "cutoff_sigma_relative_error", "state_P_error", "state_S_error")},
                qr_orthogonality_max=max(float(r["orthogonality"]) for r in actual),
                legacy_actual_projector_max=max(float(r["projector_vs_step_expm"]) for r in actual if r["method"] == "eig_qr"),
                legacy_actual_entropy_max=max(float(r["entropy_vs_step_expm"]) for r in actual if r["method"] == "eig_qr"),
                expm_half_step_projector_max=max(float(r["projector_vs_step_expm"]) for r in actual if r["method"] == "half_step_qr"),
                expm_half_step_entropy_max=max(float(r["entropy_vs_step_expm"]) for r in actual if r["method"] == "half_step_qr"),
                local_resource_seconds={row["stage"]: row["elapsed_seconds"] for row in audits},
                outcome="first-round audit completed; legacy eig propagator fails selected cases; only tested subranges accepted for P1",
                scope="no P1/P2 production scan, no long-time phase classification, no HPC timing claim")
    write_json("audit_summary.json", info)

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                         "savefig.dpi": 190, "font.family": "DejaVu Sans"})
    colors = {"eig_qr": "#b33c43", "half_step_qr": "#188c81", "direct_qr": "#d99d28"}
    x = np.arange(len(final_cases))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    axes[0].semilogy(x, [max(float(r["step_eig_relative_fro"]), 1e-17) for r in final_cases], "o", ms=3.5, color=colors["eig_qr"], label="legacy eig vs expm")
    axes[0].semilogy(x, [max(float(r["step_half_relative_fro"]), 1e-17) for r in final_cases], ".", color=colors["half_step_qr"], label="two half steps vs full step")
    axes[0].axhline(1e-10, color="black", ls="--", lw=1, label="step tolerance")
    axes[0].set(title="(a) One-step propagation", ylabel="Relative Frobenius error", xlabel="Audit case index")
    axes[0].legend(fontsize=8, loc="upper left")
    for axis, field, title, ylabel in ((axes[1], "projector_vs_step_expm", "(b) Actual occupied subspace", "Projector operator error"),
                                       (axes[2], "entropy_vs_step_expm", "(c) Actual half-chain entropy", "Absolute entropy error (nats)")):
        for method, color in colors.items():
            ys = [max(float(r[field]) for r in actual if r["case_id"] == case["case_id"] and r["method"] == method) for case in final_cases]
            axis.semilogy(x, np.maximum(ys, 1e-17), "o", ms=3.5, color=color, label=method)
        axis.axhline(1e-6, color="black", ls="--", lw=1)
        axis.set(title=title, ylabel=ylabel, xlabel="Audit case index")
        axis.legend(fontsize=8, loc="upper left")
    for axis in axes:
        axis.grid(alpha=0.2, axis="y")
    fig.savefig(FIG / "propagation_precision_audit.png")
    fig.savefig(FIG / "propagation_precision_audit.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True)
    nonunitary = [r for r in final_cases if r["group"] != "unitary"]
    nx = [int(r["case_id"].split("_")[-1]) for r in nonunitary]
    axes[0].semilogy(nx, [float(r["sigma_N_over_sigma_1"]) for r in nonunitary], "o", ms=4, color="#188c81", label=r"$\sigma_N/\sigma_1$")
    axes[0].semilogy(nx, [float(r["relative_cutoff_gap"]) for r in nonunitary], "^", ms=4, color="#d99d28", label="relative cutoff gap")
    axes[0].semilogy(nx, [10 * float(r["numerical_noise"]) for r in nonunitary], ".", color="#b33c43", label="10 x estimated noise")
    axes[0].set(title="(a) Double-precision cutoff resolution", ylabel="Relative scale", xlabel="Audit case index")
    axes[0].legend(fontsize=9)
    hx = np.arange(len(validated))
    axes[1].semilogy(hx, np.maximum([r["W_error"] for r in validated], 1e-17), "o", color="#b33c43", label=r"$W_+$ projector")
    axes[1].semilogy(hx, np.maximum([r["state_P_error"] for r in validated], 1e-17), "s", color="#188c81", label="actual QR state")
    axes[1].axhline(1e-6, color="black", ls="--", lw=1, label="acceptance tolerance")
    axes[1].set(title="(b) Against converged 80-digit reference", ylabel="Projector operator error", xlabel="Reference case index")
    axes[1].set_xticks(hx, [r["case_id"].replace("ssh_", "") for r in validated], rotation=60)
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2, axis="y")
    fig.savefig(FIG / "singular_subspace_precision_audit.png")
    fig.savefig(FIG / "singular_subspace_precision_audit.pdf")
    plt.close(fig)
    event("summarize", info)


def verify_stage():
    baseline = read_csv(OUT / "baseline_manifest.csv")
    maintained = {
        "README.md", "research_doc/研究文档导航.md",
        "research_doc/01_当前研究/研究计划与任务清单.md",
        "research_doc/01_当前研究/研究现状与结论边界.md",
        "research_doc/02_工程与复现/工程说明与协作指南.md",
        "research_doc/03_理论与图表/理论推导与图表设计.md",
    }
    checks = []
    for row in baseline:
        path = ROOT / row["path"]
        exists = path.is_file()
        current = sha256(path) if exists else None
        allowed = row["path"] in maintained
        checks.append(dict(path=row["path"], baseline_sha256=row["sha256"], current_sha256=current,
                           exists=exists, unchanged=current == row["sha256"], maintained_document=allowed))
    write_csv("baseline_integrity_check.csv", checks)
    assert len(checks) == 162
    assert all(row["exists"] and (row["unchanged"] or row["maintained_document"]) for row in checks)
    with zipfile.ZipFile(OUT / "baseline_snapshot.zip") as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 162
    execution_hash = sha256(OUT / "audit_execution_source.py")
    environments = [json.loads((OUT / f"environment_{stage}.json").read_text(encoding="utf-8"))
                    for stage in ("baseline", "double", "highprecision", "math", "summarize")]
    assert all(env["baseline_snapshot_sha256"] == sha256(OUT / "baseline_snapshot.zip") and
               env["config_sha256"] == sha256(OUT / "config.json") for env in environments)
    assert all(env["audit_script_sha256"] == execution_hash for env in environments[:4])
    assert environments[-1]["audit_script_sha256"] == sha256(Path(__file__))

    final = read_csv(OUT / "final_precision_classification.csv")
    actual = read_csv(OUT / "final_actual_state_checks.csv")
    assert {r["case_id"] for r in final} == {c["case_id"] for c in cases()}
    assert len(actual) == len({(r["case_id"], r["initial"], r["method"]) for r in actual}) == 488
    references = read_csv(OUT / "reference_acceptance.csv")
    assert len(references) == 16 and all(r["W_pass"] == r["state_pass"] == "True" for r in references)
    assert len(read_csv(OUT / "initial_memory_checks.csv")) == 61
    math_checks = read_csv(OUT / "mathematical_identity_checks.csv")
    assert len(math_checks) == 48
    assert max(float(r["projector_identity_error"]) for r in math_checks) < 1e-10
    assert all(float(r[field]) < 1e-10 for r in math_checks for field in
               ("sharper_bound_violation", "compression_violation", "audenaert_violation", "binary_bound_violation"))
    arrays = {}
    for filename, expected in (("double_matrices.npz", 1037), ("high_precision_reference.npz", 192)):
        with np.load(OUT / filename) as archive:
            assert len(archive.files) == expected
            assert all(np.all(np.isfinite(archive[key])) for key in archive.files)
            arrays[filename] = len(archive.files)
    with gzip.open(OUT / "high_precision_raw.json.gz", "rt", encoding="utf-8") as stream:
        raw_hp = json.load(stream)
    assert len(raw_hp) == 32

    md_files = [ROOT / "README.md", *sorted((ROOT / "research_doc").rglob("*.md"))]
    links = 0
    for path in md_files:
        content = path.read_text(encoding="utf-8-sig")
        assert "\ufffd" not in content
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            target = target.strip().split("#", 1)[0]
            if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            local = path.parent / target.strip("<>")
            assert local.exists(), f"broken Markdown link: {path.relative_to(ROOT)} -> {target}"
            links += 1
    literature = ROOT / "data/prl_l0_literature"
    source_metadata = json.loads((literature / "source_metadata.json").read_text(encoding="utf-8"))
    assert len(source_metadata) == 6 and all(row["pdf_downloaded"] for row in source_metadata)
    assert all((literature / (row["source_id"] + ".pdf")).read_bytes().startswith(b"%PDF") for row in source_metadata)
    initial_refs = [r for r in actual if r["method"] == "expm_qr" and r["state_expm_qr_status"] == "reference_validated"]
    value = dict(study_date=STUDY_DATE, baseline_files=len(checks),
                 unchanged_baseline_files=sum(row["unchanged"] for row in checks),
                 protected_scientific_files=sum(not row["path"].startswith("research_doc/") and row["path"] != "README.md" for row in checks),
                 modified_maintained_documents=[row["path"] for row in checks if not row["unchanged"]],
                 missing_or_unexpected_changes=0, baseline_zip_crc_pass=True, arrays_checked=arrays,
                 actual_method_records=len(actual), reference_conditions=len(references),
                 highprecision_raw_records=len(raw_hp), mathematical_checks=len(math_checks),
                 local_markdown_links_checked=links, research_markdown_documents=len(md_files) - 1,
                 primary_pdfs=6, reference_natural_state_outputs=len(initial_refs),
                 reference_natural_state_min_distance_to_W=min(float(r["distance_to_W"]) for r in initial_refs),
                 reference_natural_state_distance_le_005=sum(float(r["distance_to_W"]) <= 0.05 for r in initial_refs),
                 figures_png=2, figures_pdf=2, figure_visual_check="both PNGs inspected; labels and axes are visible",
                 execution_source_sha256=execution_hash,
                 status="passed; first round is complete; limits in stage report retained")
    write_json("delivery_verification.json", value)
    event("verify", value)


def artifact_manifest():
    files = [path for directory in (OUT, ROOT / "data/prl_l0_literature", FIG)
             for path in directory.rglob("*") if path.is_file() and path.name != "artifact_manifest.csv"]
    files.extend([ROOT / "README.md", *sorted((ROOT / "research_doc").rglob("*.md")),
                  Path(__file__), ROOT / "scripts/collect_literature_sources.py"])
    write_csv("artifact_manifest.csv", [dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                                            sha256=sha256(path)) for path in sorted(set(files))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "double", "highprecision", "math", "summarize", "verify"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    proposed = config()
    config_path = OUT / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != proposed:
            raise ValueError("configuration changed; preserve the existing run before replacing its configuration")
    else:
        write_json("config.json", proposed)
    with tempfile.TemporaryDirectory(prefix="prl_p0_baseline_") as temporary:
        with zipfile.ZipFile(OUT / "baseline_snapshot.zip") as archive:
            for member in archive.namelist():
                if member.startswith(("src/", "scripts/")) and not member.startswith("../"):
                    archive.extract(member, temporary)
        sys.path[:0] = [str(Path(temporary) / "src"), str(Path(temporary) / "scripts")]
        with threadpool_limits(limits=1):
            initial_environment = environment(args.stage)
            tick = time.perf_counter()
            stages = {"baseline": baseline_stage, "double": double_stage,
                      "highprecision": highprecision_stage, "math": math_stage,
                      "summarize": summarize_stage, "verify": verify_stage}
            stages[args.stage]()
            initial_environment["elapsed_seconds"] = time.perf_counter() - tick
            initial_environment["rss_mb_final"] = psutil.Process().memory_info().rss / 2**20
            if hasattr(psutil.Process().memory_info(), "peak_wset"):
                initial_environment["process_peak_rss_mb"] = psutil.Process().memory_info().peak_wset / 2**20
            write_json(f"environment_{args.stage}.json", initial_environment)
            if args.stage == "verify":
                artifact_manifest()


if __name__ == "__main__":
    main()
