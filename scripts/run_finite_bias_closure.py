"""Separate frozen predictions and complete-mode checks for finite-bias closure."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mpmath as mp
import numpy as np
from scipy.linalg import norm

import explore_asymptotic_time as spectral
import research_increment_center as center
import run_research_increment as previous

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prl_finite_bias_closure_20261013"
DOCS = ROOT / "research_doc/01_当前研究/有限尺寸与固定偏置闭合_20261013"
LENGTHS = (1152, 1280)
PRECISIONS = ((1580, 1620), (1750, 1790))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def initialize():
    target = OUT / "protected_baseline.json"
    if target.exists():
        return load(target)
    old = load(previous.OUT / "final_delivery_manifest.json")
    paths = {ROOT / row["path"] for row in old["files"]}
    paths.add(previous.OUT / "final_delivery_manifest.json")
    paths.update(ROOT / relative for relative in load(previous.OUT / "protected_baseline.json")["files"])
    rows = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(paths)]
    value = dict(created_utc=datetime.now(timezone.utc).isoformat(), files=rows,
                 scope="Earlier manuscripts, proofs, locks, data and delivery records are preserved.")
    save(target, value)
    return dict(status="initialized", protected_files=len(rows))


def verify():
    baseline = load(OUT / "protected_baseline.json")
    mismatches = [row["path"] for row in baseline["files"] if
                  not (ROOT / row["path"]).is_file() or digest(ROOT / row["path"]) != row["sha256"]]
    value = dict(status="pass" if not mismatches else "fail", files=len(baseline["files"]),
                 mismatches=mismatches, checked_utc=datetime.now(timezone.utc).isoformat())
    save(OUT / "protected_verification.json", value)
    if mismatches:
        raise ValueError(mismatches)
    return value


def freeze():
    target = OUT / "prediction_lock.json"
    if target.exists():
        return load(target)
    theory = center.prefactor_record(100)
    with mp.workdps(100):
        beta = mp.mpf(theory["gaussian_beta"])
        mu0 = mp.mpf(theory["parameters"]["mu0"])
        correction = 2 * mp.sqrt(2 * beta / mp.pi) / mu0
        value = dict(locked_utc=datetime.now(timezone.utc).isoformat(), lengths=LENGTHS,
                     precisions=PRECISIONS, chi_values=[-4, 0, 4], levels=[.3, .05, .01],
                     theory=theory, correction_coefficient=str(correction),
                     prediction="t0_analytic - s*chi/(4*mu0) + s*A*chi/sqrt(L)",
                     proof_status_at_freeze="Next-order boundary-wavepacket derivation in progress; numerical predictions are frozen independently of final proof approval.",
                     source_sha256=digest(Path(__file__)),
                     dependencies={"explore_asymptotic_time.py": digest(Path(spectral.__file__)),
                                   "research_increment_center.py": digest(Path(center.__file__))},
                     gates=dict(cache_maximum_absolute_change=2e-10, invariance_residual=1e-10,
                                root_residual=2e-8, time_finite_difference_error=3e-6,
                                chi_finite_difference_error=3e-6, principal_gap_minimum=1e-6,
                                absolute_center_error=.015, bias_shift_error=.03,
                                refined_bias_shift_error=.003, refined_susceptibility_error=.0005),
                     no_fit=True, prior_failures_unchanged=True,
                     scope="Independent new sizes; neither implementation nor prediction gates alone establish a theorem.")
    save(target, value)
    return value


def prepare(index):
    lock = load(OUT / "prediction_lock.json")
    size_index, precision_index = divmod(index, 2)
    length, digits = lock["lengths"][size_index], lock["precisions"][size_index][precision_index]
    target = OUT / "cache" / f"L{length}_dps{digits}.npz"
    receipt_path = target.with_suffix(".json")
    if target.exists() and receipt_path.exists():
        record = load(receipt_path)
        if digest(target) != record["sha256"]:
            raise ValueError("Existing preparation no longer matches its receipt.")
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    raw, _ = spectral.reflected_spectral(dict(key=f"L{length}", length=length,
                                             gamma=2., t1=.5, t2=1.), digits)
    arrays = spectral.prior.convert_spectral(raw)
    np.savez_compressed(target, **arrays)
    record = dict(length=length, dps=digits, sha256=digest(target),
                  seconds=time.perf_counter() - tick, prediction_lock_sha256=digest(OUT / "prediction_lock.json"),
                  source_sha256=digest(Path(__file__)),
                  states={side: {key: val for key, val in data.items() if key != "F"}
                          for side, data in raw["states"].items()},
                  method=raw["kernel"], all_finite_modes_retained=True,
                  precision_policy="Complete independent arbitrary-precision static construction, then float64 modes and signed logarithms. Full decimal matrices are reproducible from this source.",
                  finished_utc=datetime.now(timezone.utc).isoformat())
    save(receipt_path, record)
    return record


def graph_for(data, initial, chi):
    return previous.GaugedGraph(data["plus"], data["minus"], data["mu"],
                               data[initial + "_log"], data[initial + "_sign"], chi)


def validate(index):
    lock = load(OUT / "prediction_lock.json")
    length = lock["lengths"][index]
    paths = [OUT / "cache" / f"L{length}_dps{digits}.npz" for digits in lock["precisions"][index]]
    with np.load(paths[0]) as low, np.load(paths[1]) as high:
        changes = {}
        for key in high.files:
            if not np.array_equal(np.isfinite(low[key]), np.isfinite(high[key])):
                raise ArithmeticError(f"Nonfinite pattern changed: {key}")
            finite = np.isfinite(high[key])
            changes[key] = float(np.max(np.abs(high[key][finite] - low[key][finite]))) if np.any(finite) else 0.
        data = {key: high[key].copy() for key in high.files}
    if max(changes.values()) > lock["gates"]["cache_maximum_absolute_change"]:
        raise ArithmeticError("Static constructions disagree.")
    theory = lock["theory"]
    constants = {(r["initial"], float(r["epsilon"])): float(r["constant_time_center"])
                 for r in theory["center_constants"]}
    tau, logterm = float(theory["time_linear_coefficient"]), float(theory["time_log_coefficient"])
    mu0, correction = float(theory["parameters"]["mu0"]), float(lock["correction_coefficient"])
    rows = []
    for initial in ("left", "right"):
        sign = 1 if initial == "left" else -1
        for chi in lock["chi_values"]:
            graph = graph_for(data, initial, chi)
            invariant = float(norm(graph.perp.T @ graph.generator @ graph.gain, 2))
            chi_step = .002
            displaced = [graph_for(data, initial, chi + delta) for delta in (-chi_step, chi_step)]
            for level in lock["levels"]:
                prediction = (tau * length + logterm * np.log(length) + constants[initial, level]
                              - sign * chi / (4 * mu0) + sign * correction * chi / np.sqrt(length))
                duration, bracket = previous.stable_root(graph, level, prediction)
                response = asdict(graph.response(duration))
                time_step = .0002
                time_fd = (graph.distance(duration + time_step) - graph.distance(duration - time_step)) / (2 * time_step)
                chi_fd = (displaced[1].distance(duration) - displaced[0].distance(duration)) / (2 * chi_step)
                susceptibility_target = -sign / (4 * mu0) + sign * correction / np.sqrt(length)
                row = dict(length=length, initial=initial, chi=chi, level=level, time=duration,
                           prediction=prediction, actual_minus_prediction=duration - prediction,
                           susceptibility_target=susceptibility_target,
                           susceptibility_error=response["susceptibility"] - susceptibility_target,
                           bracket=bracket, invariance_residual=invariant,
                           time_fd_error=abs(time_fd - response["distance_time_derivative"]),
                           chi_fd_error=abs(chi_fd - response["distance_chi_derivative"]), **response)
                gates = lock["gates"]
                row["implementation_pass"] = bool(abs(response["distance"] - level) < gates["root_residual"]
                    and invariant < gates["invariance_residual"]
                    and row["time_fd_error"] < gates["time_finite_difference_error"]
                    and row["chi_fd_error"] < gates["chi_finite_difference_error"]
                    and response["principal_gap"] > gates["principal_gap_minimum"])
                row["absolute_center_pass"] = bool(abs(row["actual_minus_prediction"]) < gates["absolute_center_error"])
                row["refined_susceptibility_pass"] = bool(abs(row["susceptibility_error"]) < gates["refined_susceptibility_error"])
                rows.append(row)
    for row in rows:
        sign = 1 if row["initial"] == "left" else -1
        zero = next(r["time"] for r in rows if r["initial"] == row["initial"] and r["level"] == row["level"] and r["chi"] == 0)
        shift_target = -sign * row["chi"] / (4 * mu0) + sign * correction * row["chi"] / np.sqrt(length)
        row["bias_shift_error"] = row["time"] - zero - shift_target
        row["bias_shift_pass"] = bool(abs(row["bias_shift_error"]) < lock["gates"]["bias_shift_error"])
        row["refined_bias_shift_pass"] = bool(abs(row["bias_shift_error"]) < lock["gates"]["refined_bias_shift_error"])
    value = dict(length=length, rows=rows, cache_changes=changes,
                 input_sha256=[digest(p) for p in paths], prediction_lock_sha256=digest(OUT / "prediction_lock.json"),
                 source_sha256=digest(Path(__file__)), no_fit=True,
                 status="pass" if all(r["implementation_pass"] and r["absolute_center_pass"]
                    and r["refined_bias_shift_pass"] and r["refined_susceptibility_pass"] for r in rows) else "failed_gate",
                 finished_utc=datetime.now(timezone.utc).isoformat())
    save(OUT / "validation" / f"L{length}.json", value)
    return dict(length=length, status=value["status"], roots=len(rows),
                maximum_center_error=max(abs(r["actual_minus_prediction"]) for r in rows),
                maximum_shift_error=max(abs(r["bias_shift_error"]) for r in rows),
                maximum_susceptibility_error=max(abs(r["susceptibility_error"]) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "verify", "freeze", "prepare", "validate"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    value = globals()[args.action](args.index) if args.action in ("prepare", "validate") else globals()[args.action]()
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
