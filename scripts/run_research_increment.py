"""New fixed-bias diagnostics using complete, previously precision-checked modes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prl_research_increment_20261013"
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.linalg import norm, solve, svd
from scipy.optimize import brentq
from mietf_skin.response import GaugedGraph, contraction_constants

LENGTHS = (128, 192, 256, 320, 384, 448, 512, 640, 768)
SKINS = (0.0, -0.01, 0.01, -0.025, 0.025, -0.05, 0.05)
LEVELS = (0.3, 0.05, 0.01)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cache_for(length):
    folder = "prl_priority_strengthening" if length <= 512 else "prl_asymptotic_time_exploration"
    return ROOT / "data" / folder / "cache" / f"L{length}.npz"


def initialize():
    if (OUT / "research_config.json").exists():
        return json.loads((OUT / "research_config.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "data/prl_claude_editorial_duel/final_delivery_manifest.json").read_text(encoding="utf-8"))
    protected = {row["path"]: row["sha256"] for row in previous["frozen_files"]}
    for path in (ROOT / "src/mietf_skin").glob("*.py"):
        protected[path.relative_to(ROOT).as_posix()] = digest(path)
    old_lock = ROOT / "data/prl_claude_editorial_duel/science_lock.json"
    protected[old_lock.relative_to(ROOT).as_posix()] = digest(old_lock)
    for length in LENGTHS:
        path = cache_for(length)
        protected[path.relative_to(ROOT).as_posix()] = digest(path)
    for relative, expected in protected.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Baseline differs: {relative}")
    save(OUT / "protected_baseline.json", dict(files=protected, previous_decision="DECLINE", created_utc=datetime.now(timezone.utc).isoformat()))
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), lengths=list(LENGTHS),
                  skins=list(SKINS), levels=list(LEVELS), initials=["left", "right"],
                  target="distance to the full growing invariant space G_g",
                  definition="chi=g*L; reported chi susceptibility must be multiplied by L for a g derivative",
                  cache_policy="All finite-chain modes retained; caches reused from two-precision preparations. This scan does not recompute those high-precision inputs.",
                  acceptance=dict(root_residual=2e-8, invariance_residual=2e-8,
                                  frame_orthogonality=2e-12, distance_fd_absolute=3e-6),
                  novelty_scope="Finite-size fixed-g diagnostics, not a fixed-g asymptotic theorem.")
    save(OUT / "research_config.json", config)
    return config


def verify_protected():
    record = json.loads((OUT / "protected_baseline.json").read_text(encoding="utf-8"))
    mismatches = [path for path, checksum in record["files"].items() if digest(ROOT / path) != checksum]
    value = dict(status="pass" if not mismatches else "fail", files=len(record["files"]), mismatches=mismatches,
                 checked_utc=datetime.now(timezone.utc).isoformat())
    save(OUT / "protected_verification.json", value)
    if mismatches:
        raise ValueError(mismatches)
    return value


def stable_root(graph, level, seed):
    width = 1.0
    previous = []
    for _ in range(13):
        lo, hi = max(0.1, seed - width), seed + width
        try:
            left, right = graph.distance(lo), graph.distance(hi)
        except (ArithmeticError, np.linalg.LinAlgError) as error:
            previous.append(str(error))
            width *= 1.5
            continue
        if left > level > right:
            answer = brentq(lambda t: graph.distance(t) - level, lo, hi, xtol=2e-10)
            return answer, [lo, hi]
        width *= 1.5
    raise ArithmeticError(f"No stable crossing: seed={seed}, errors={previous[-2:]}")


def scan(index, folder="fixed_bias"):
    config = json.loads((OUT / "research_config.json").read_text(encoding="utf-8"))
    length = LENGTHS[index // 2]
    initial = ("left", "right")[index % 2]
    source = cache_for(length)
    arrays = np.load(source)
    rows, failures, zero = [], [], {}
    tick = time.perf_counter()
    for skin in SKINS:
        chi = skin * length
        graph = GaugedGraph(arrays["plus"], arrays["minus"], arrays["mu"],
                            arrays[initial + "_log"], arrays[initial + "_sign"], chi)
        invariant = float(norm(graph.perp.T @ graph.generator @ graph.gain, 2))
        orthogonal = float(norm(graph.gain.T @ graph.gain - np.eye(graph.n), 2))
        constants = contraction_constants(2.0, 0.5, 1.0, skin)
        for level in LEVELS:
            candidate = dict(length=length, initial=initial, skin=skin, chi=chi, level=level)
            try:
                reference = zero.get(level, 0.54 * length - 2.0)
                seed = reference + (-1 if initial == "left" else 1) * chi / (4 * np.sqrt(1.75))
                duration, bracket = stable_root(graph, level, seed)
                response = asdict(graph.response(duration))
                if skin == 0:
                    zero[level] = duration
                step = 2e-4
                distance_time_fd = (graph.distance(duration + step) - graph.distance(duration - step)) / (2 * step)
                shifted = []
                for offset in (-step, step):
                    other = GaugedGraph(arrays["plus"], arrays["minus"], arrays["mu"],
                                        arrays[initial + "_log"], arrays[initial + "_sign"], chi + offset)
                    shifted.append(other.distance(duration))
                distance_chi_fd = (shifted[1] - shifted[0]) / (2 * step)
                time_fd_error = abs(distance_time_fd - response["distance_time_derivative"])
                chi_fd_error = abs(distance_chi_fd - response["distance_chi_derivative"])
                exponent = arrays[initial + "_log"] - duration * (arrays["mu"][:, None] + arrays["mu"][None, :])
                ft = arrays[initial + "_sign"] * np.exp(exponent)
                upper = graph.r + graph.cross @ ft
                matrix = graph.graph(duration)
                solve_residual = float(norm(matrix @ upper - graph.bottom @ ft, "fro") / max(norm(graph.bottom @ ft, "fro"), 1e-300))
                singular_upper = svd(upper, compute_uv=False, check_finite=False)
                denominator_condition = float(singular_upper[0] / singular_upper[-1])
                k = response["tangent"]
                contraction_margin = 2 * constants["a"] - constants["b"] * k
                lipschitz_g = (length / 2 - 1) * np.sqrt(1 + k * k) / contraction_margin
                _, singular, vh = svd(matrix, check_finite=False)
                selected = graph.gain @ vh.T[:, 0]
                s = 1 if initial == "left" else -1
                margin = (graph.n - 1) / 2 - s * (np.arange(length) // 2 - (graph.n - 1) / 2)
                row = dict(**candidate, time=duration, time_per_length=duration / length,
                           zero_time=zero[level], shift_per_length=(duration - zero[level]) / length,
                           derivative_g=length * response["susceptibility"],
                           normalized_derivative_g=response["susceptibility"],
                           **response, invariance_residual=invariant, frame_orthogonality=orthogonal,
                           denominator_condition=denominator_condition, solve_residual=solve_residual,
                           gauge_condition=float(np.exp(abs(skin) * (length / 2 - 1))),
                           chi_distance_fd_error=chi_fd_error, time_distance_fd_error=time_fd_error,
                           contraction_margin=contraction_margin, uniform_lipschitz_g=lipschitz_g,
                           selected_endpoint_mean=float(np.abs(selected)**2 @ margin), bracket=bracket)
                gates = config["acceptance"]
                row["accepted"] = bool(abs(response["distance"] - level) <= gates["root_residual"]
                                       and invariant <= gates["invariance_residual"]
                                       and orthogonal <= gates["frame_orthogonality"]
                                       and chi_fd_error <= gates["distance_fd_absolute"]
                                       and time_fd_error <= gates["distance_fd_absolute"]
                                       and contraction_margin > 0
                                       and abs(row["derivative_g"]) <= lipschitz_g + 1e-7)
                rows.append(row)
            except (ArithmeticError, np.linalg.LinAlgError, ValueError) as error:
                failures.append(dict(**candidate, error=type(error).__name__, message=str(error)))
        print(json.dumps(dict(length=length, initial=initial, skin=skin, rows=len(rows),
                              failures=len(failures), elapsed=time.perf_counter() - tick)), flush=True)
    target = OUT / folder / f"L{length}_{initial}.json"
    value = dict(status="completed", rows=rows, failures=failures,
                 accepted=sum(row["accepted"] for row in rows), seconds=time.perf_counter() - tick,
                 source_sha256=digest(Path(__file__)), input_sha256=digest(source),
                 config_sha256=digest(OUT / "research_config.json"), slurm_job=os.getenv("SLURM_JOB_ID"),
                 scope=config["novelty_scope"], finished_utc=datetime.now(timezone.utc).isoformat())
    save(target, value)
    return dict(status=value["status"], result=str(target.relative_to(ROOT)), rows=len(rows),
                accepted=value["accepted"], failures=len(failures), seconds=value["seconds"])


def summarize():
    import csv
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((OUT / "fixed_bias").glob("*.json"))]
    rows = [row for record in records for row in record["rows"]]
    fields = sorted({key for row in rows for key in row if key != "bracket"})
    target = OUT / "fixed_bias_summary.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    value = dict(tasks=len(records), expected_tasks=2 * len(LENGTHS), roots=len(rows),
                 accepted=sum(row["accepted"] for row in rows),
                 failed_roots=[item for record in records for item in record["failures"]],
                 rejected_diagnostics=[row for row in rows if not row["accepted"]],
                 sign_exceptions=[row for row in rows if row["accepted"] and
                                  row["susceptibility"] * (1 if row["initial"] == "left" else -1) >= 0],
                 status="numerical_diagnostics", scope="No claim of fixed-g asymptotic coefficient or direction theorem.")
    save(OUT / "fixed_bias_summary.json", value)
    return {key: value[key] for key in ("tasks", "expected_tasks", "roots", "accepted", "status")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "scan", "summarize", "verify"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-folder", choices=("fixed_bias", "local_reference"), default="fixed_bias")
    args = parser.parse_args()
    functions = {"initialize": initialize, "scan": lambda: scan(args.index, args.output_folder), "summarize": summarize, "verify": verify_protected}
    print(json.dumps(functions[args.stage](), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
