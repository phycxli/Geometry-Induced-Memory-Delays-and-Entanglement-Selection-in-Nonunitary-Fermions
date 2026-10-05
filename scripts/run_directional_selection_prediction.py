"""Sixth round: directional remainders, direct memory and finite boundaries."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits

import run_reduced_selection_prediction as base

spatial, previous, audit, p1 = base.spatial, base.previous, base.audit, base.p1
ROOT, PRIOR = base.ROOT, base.OUT
OUT = ROOT / "data/prl_p2_directional_remainder"
FIG = ROOT / "figures/prl_p2_directional_remainder"
SOURCE = Path(__file__)
SOURCE_SHA = audit.sha256(SOURCE)
RANKS = (0, 1, 2, 4, 6, 8, 12, 16)
DIGITS = (160, 200, 240, 280, 320)
write_json, emit, read_json = base.write_json, base.emit, base.read_json


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cases():
    rows = []
    for length in (80, 96, 112):
        skins = (0., .25) if length == 80 else (0., .05, .25)
        for skin in skins:
            times = ([42.5, 50., 55.] if skin == 0 else [37.5, 42.5, 50., 55.]) if length == 80 else [length * .5, length * .6]
            for duration in times:
                rows.append(dict(case_id=p1.case_key(length, 2., skin, round(duration / .05)), length=length,
                                 gamma=2., skin=skin, physical_time=duration, steps=round(duration / .05), dt=.05,
                                 t1=.5, t2=1., boundary="open", initials=["charge_density_wave", "left", "right"],
                                 role="window_and_persistence" if length == 80 else "unseen_size"))
    return sorted(rows, key=lambda r: (r["length"], r["skin"], r["physical_time"]))


def families():
    return sorted({(r["length"], r["gamma"], r["skin"]) for r in cases()})


def configuration():
    return dict(version=1, date="2026-10-04", cases=cases(), families=[list(x) for x in families()],
                ranks=list(RANKS), model_projector_target=1e-5, precision_digits=list(DIGITS),
                selection_tolerances=dict(projector=.05, absolute_entropy=.05),
                memory_tolerances=dict(projector=.1, absolute_entropy=.1),
                reference_goals=dict(projector=1e-10, entropy=1e-10, relative_scalar=1e-8),
                numerical_goals=dict(projector=1e-6, entropy=1e-6),
                rank_policy="first frozen rank with min(legacy,directional,transverse) remainder <=1e-5; otherwise rank16 with uncertainty",
                core="thin SVD of [G^T Y,v^T], no singular-value truncation; core I+R T R^T of order min(n,2r)",
                directional="||Xminus D(F-Fr)D Rinv||F / smin(Zr); optional projected residual / (smin(Zr)-||DeltaZ||F)",
                memory="max(distance of reduced projectors, absolute distance-to-G difference), minus both remainders",
                spatial="reflected finite-chain Gram inverse trace bounds: 1/tr(J) <= delta <= tr(J)/tr(J^2), J=M^-1",
                precision_policy="two-precision independent matrix-exponential truth; stable80/120 benchmark recorded even if failed",
                development_scope="all 83 fifth-round states now observed; not a holdout",
                prior_plan_sha256=audit.sha256(PRIOR / "next_round_plan.json"),
                scope="local Windows sparse pilot; no HPC production, continuum-time certificate or asymptotic exponent")


def prepare():
    for name in ("code", "development", "spatial", "static", "predictions", "cases", "failures", "literature"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    path = OUT / "config.json"
    if path.exists():
        assert read_json(path) == configuration(), "configuration already frozen"
    else:
        write_json(path, configuration())
    path = OUT / "input_manifest.csv"
    if not path.exists():
        inputs = {ROOT / row["path"] for row in audit.read_csv(PRIOR / "input_manifest.csv")}
        inputs.update(p for folder in (PRIOR, base.FIG) for p in folder.rglob("*") if p.is_file())
        inputs.update((base.SOURCE, ROOT / "scripts/analyze_reduced_selection_prediction.py",
                       ROOT / "research_doc/04_历史研究/08_低秩预测与稳定算法.md"))
        write_csv("input_manifest.csv", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                                              sha256=audit.sha256(p)) for p in sorted(inputs)])
    for row in audit.read_csv(path):
        assert audit.sha256(ROOT / row["path"]) == row["sha256"], row["path"]
    snapshot = OUT / "code" / (SOURCE_SHA.lower() + ".py")
    if not snapshot.exists():
        snapshot.write_bytes(SOURCE.read_bytes())


def context(static, initial):
    ctx = base.compression_context(static, initial)
    matrices, u, vh, sig, rinv, mu, residuals = ctx
    f = spatial.decode(static["states"][initial]["F"])
    for rank in RANKS:
        if rank not in residuals:
            residuals[rank] = f - u[:, :rank] * mp.diag(sig[:rank]) * vh[:rank, :]
    return ctx


def small_core_smin(g, y, v):
    rank, n = v.rows, v.cols
    if not rank:
        return mp.mpf(1), 0, mp.mpf(0)
    c = g.T * y
    umat = mp.matrix(n, 2 * rank)
    umat[:, :rank], umat[:, rank:] = c, v.T
    j, singular, vh = mp.svd(umat)
    ru = mp.diag(singular) * vh
    kernel = mp.matrix(2 * rank)
    kernel[:rank, rank:] = mp.eye(rank)
    kernel[rank:, :rank] = mp.eye(rank)
    kernel[rank:, rank:] = y.T * y
    core = mp.eye(j.cols) + ru * kernel * ru.T
    core = (core + core.T) / 2
    eigenvalues = mp.eigsy(core, eigvals_only=True)
    value = min(mp.mpf(1), eigenvalues[0]) if j.cols < n else eigenvalues[0]
    if value <= 0:
        raise ArithmeticError("nonpositive small core; increase precision")
    residual = p1.mp_norm(umat - j * ru) / max(mp.mpf(1), p1.mp_norm(umat))
    return mp.sqrt(value), j.cols, residual


def at_rank(ctx, duration, rank, core_check=False):
    tick = time.perf_counter()
    values, q = base.reduced_at_rank(ctx, duration, rank)
    propagation_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    matrices, u, vh, sig, rinv, mu, residuals = ctx
    n = len(mu)
    decay = [mp.exp(-x * duration) for x in mu]
    if rank:
        a = mp.matrix([[decay[i] * u[i, j] * mp.sqrt(sig[j]) for j in range(rank)] for i in range(n)])
        bt = mp.matrix([[mp.sqrt(sig[i]) * vh[i, j] * decay[j] for j in range(n)] for i in range(rank)])
        v, y = bt * rinv, matrices["minus"] * a
        smin, order, residual = small_core_smin(matrices["G"], y, v)
        zr = matrices["G"] + y * v
    else:
        smin, order, residual = mp.mpf(1), 0, mp.mpf(0)
        zr = matrices["G"]
    weighted = mp.matrix([[residuals[rank][i, j] * decay[i] * decay[j] for j in range(n)] for i in range(n)])
    delta = matrices["minus"] * weighted * rinv
    numerator = p1.mp_norm(delta)
    directional = min(mp.mpf(1), numerator / smin)
    transverse_numerator = p1.mp_norm(delta - q * (q.T * delta))
    transverse = min(mp.mpf(1), transverse_numerator / (smin - numerator)) if numerator < smin else mp.mpf(1)
    legacy = values["projector_remainder"]
    values.update(legacy_remainder=legacy, directional_remainder=directional, transverse_remainder=transverse,
                  projector_remainder=min(legacy, directional, transverse), column_residual_fro=numerator,
                  transverse_residual_fro=transverse_numerator, small_core_smin=smin, small_core_order=order,
                  small_core_reconstruction_residual=residual)
    check_seconds = 0.
    if core_check:
        check_tick = time.perf_counter()
        full = previous.mp_smin(zr)
        error = abs(full - smin) / full
        assert error < mp.mpf("1e-10"), (rank, full, smin, error)
        values["core_full_smin_relative_error"] = error
        check_seconds = time.perf_counter() - check_tick
    costs = dict(propagation_seconds=propagation_seconds, remainder_seconds=time.perf_counter() - tick - check_seconds,
                 independent_core_check_seconds=check_seconds)
    return values, q, costs


def prediction(static, initial, duration, ctx=None, core_check=False):
    ctx = ctx or context(static, initial)
    attempts, costs = [], dict(propagation_seconds=0., remainder_seconds=0., independent_core_check_seconds=0.)
    for rank in RANKS:
        result, q, local_costs = at_rank(ctx, duration, rank, core_check)
        attempts.append({k: float(v) for k, v in result.items()})
        for key in costs:
            costs[key] += local_costs[key]
        if result["projector_remainder"] <= mp.mpf(".00001"):
            break
    tick = time.perf_counter()
    entropy, _ = previous.correlation_entropy(q)
    costs["entropy_seconds"] = time.perf_counter() - tick
    z = mp.mpf(static["q"]) * mp.exp(-2 * min(ctx[5]) * duration)
    p = mp.mpf(static["p"])
    ew = min(mp.mpf(1), z / (p - z)) if z < p else mp.mpf(1)
    eps, dg = result["projector_remainder"], result["distance_to_G"]
    error = abs(entropy - mp.mpf(static["gain_entropy"]))
    budget = base.entropy_budget(eps, static["length"] // 2) + base.entropy_budget(ew, static["length"] // 2)
    values = {**result, "entropy": entropy, "output_to_G_bound": ew, "entropy_budget": budget,
              "distance_lower": max(0, dg - eps - ew), "distance_upper": min(1, dg + eps + ew),
              "entropy_error_lower": max(0, error - budget), "entropy_error_upper": error + budget}
    return values, q, attempts, costs


def pair_bound(a, b, qa, qb):
    direct = p1.mp_distance(qa, qb)
    radial = abs(a["distance_to_G"] - b["distance_to_G"])
    eps = a["projector_remainder"] + b["projector_remainder"]
    return dict(reduced_pair_distance=direct, radial_lower=max(0, radial - eps),
                direct_lower=max(0, direct - eps), memory_lower=max(0, max(direct, radial) - eps))


def development_job(family):
    key = spatial.identifier(*family)
    path = OUT / "development" / (key + ".json")
    if path.exists():
        return dict(stage="development", family=key, status="reused")
    static = read_json(PRIOR / "static" / (key + ".json"))
    planned = [case for case in base.cases() if (case["length"], case["gamma"], case["skin"]) == tuple(family)]
    contexts, rows, pairs, attempts = {}, [], [], []
    old = {(r["case_id"], r["initial"]): r for r in audit.read_csv(PRIOR / "locked_predictions.csv")}
    with mp.workdps(static["dps"]):
        for initial in {name for case in planned for name in case["initials"]}:
            contexts[initial] = context(static, initial)
        for case in planned:
            with gzip.open(PRIOR / "cases" / (case["case_id"] + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
                truth = json.load(stream)[-1]
            local = {}
            for initial in case["initials"]:
                values, q, trials, costs = prediction(static, initial, mp.mpf(str(case["physical_time"])), contexts[initial], True)
                exact = spatial.decode(truth["states"][initial]["Q"])
                error = p1.mp_distance(q, exact)
                assert error <= values["projector_remainder"] + mp.mpf("1e-30"), (key, initial, error)
                old_rank = int(float(old[case["case_id"], initial]["rank"]))
                fixed, _, _ = at_rank(contexts[initial], mp.mpf(str(case["physical_time"])), old_rank)
                rows.append({**case, "initial": initial, **{k: float(v) for k, v in values.items()}, **costs,
                             "measured_projector_error": float(error), "old_rank": old_rank,
                             "old_remainder": float(old[case["case_id"], initial]["projector_remainder"]),
                             "new_remainder_at_old_rank": float(fixed["projector_remainder"]),
                             "target_passed": values["projector_remainder"] <= mp.mpf(".00001"),
                             "core_check_max_relative_error": max(a["core_full_smin_relative_error"] for a in trials),
                             "scope": "observed fifth-round development, not holdout"})
                local[initial] = values, q, exact
                attempts.extend(dict(case_id=case["case_id"], initial=initial, **t) for t in trials)
            a, qa, exact_a = local["charge_density_wave"]
            for initial in case["initials"][1:]:
                b, qb, exact_b = local[initial]
                bound = pair_bound(a, b, qa, qb)
                actual = p1.mp_distance(exact_a, exact_b)
                assert bound["memory_lower"] <= actual + mp.mpf("1e-30")
                pairs.append({**case, "initial_a": "charge_density_wave", "initial_b": initial,
                              **{k: float(v) for k, v in bound.items()}, "actual_pair_distance": float(actual)})
    write_json(path, dict(rows=rows, pairs=pairs, attempts=attempts, source_sha256=SOURCE_SHA))
    return dict(stage="development", family=key, states=len(rows), target_passed=sum(r["target_passed"] for r in rows))


def finish_development():
    values = [read_json(path) for path in sorted((OUT / "development").glob("L*.json"))]
    rows = [r for v in values for r in v["rows"]]
    pairs = [r for v in values for r in v["pairs"]]
    assert len(rows) == 83 and len(pairs) == 54
    write_csv("development/states.csv", rows)
    write_csv("development/memory.csv", pairs)
    write_csv("development/attempts.csv", [r for v in values for r in v["attempts"]])
    result = dict(status="passed", states=len(rows), pairs=len(pairs), old_target_passed=67,
                  fixed_old_rank_target_passed=sum(r["new_remainder_at_old_rank"] <= 1e-5 for r in rows),
                  target_passed=sum(r["target_passed"] for r in rows),
                  target_passed_at_rank_le8=sum(r["target_passed"] and r["rank"] <= 8 for r in rows),
                  max_core_full_smin_relative_error=max(r["core_check_max_relative_error"] for r in rows),
                  max_projector_error=max(r["measured_projector_error"] for r in rows),
                  guaranteed_direct_memory=sum(r["memory_lower"] > .1 for r in pairs),
                  guaranteed_radial_memory=sum(r["radial_lower"] > .1 for r in pairs),
                  ranks=list(RANKS), scope="development only; rank policy frozen before any new truth")
    write_json(OUT / "development/summary.json", result)
    emit(result)


def inverse_trace_bounds(matrix):
    inverse = mp.inverse(matrix)
    trace = mp.fsum(inverse[i, i] for i in range(inverse.rows))
    square_trace = mp.fsum(inverse[i, j] * inverse[j, i] for i in range(inverse.rows) for j in range(inverse.rows))
    lower, upper = 1 / trace, trace / square_trace
    distribution = [inverse[i, i] / trace for i in range(inverse.rows)]
    return lower, upper, distribution


def spatial_lock():
    path = OUT / "spatial_hypothesis_lock.json"
    if path.exists():
        check_hashes("spatial_lock_hash.json")
        return
    assert not list((OUT / "static").glob("L96*.json")) and not list((OUT / "static").glob("L112*.json"))
    rows, raw = [], []
    for length in (96, 112):
        for skin in (0., .05, .25):
            for initial in ("left", "right"):
                results = []
                for nodes, digits in ((64, 200), (96, 240)):
                    _, matrix = base.spatial_gram(length, skin, initial, nodes, digits, True)
                    with mp.workdps(digits):
                        results.append(inverse_trace_bounds(matrix))
                with mp.workdps(260):
                    lo, hi, distribution = results[-1]
                    change = max(spatial.relative(results[0][i], results[1][i]) for i in (0, 1))
                    assert change < mp.mpf("1e-8")
                    mean = mp.fsum(i * x for i, x in enumerate(distribution))
                    row = dict(length=length, skin=skin, initial=initial, cross_lower=float(lo), cross_upper=float(hi),
                               relative_width=float(hi / lo - 1), quadrature_precision_change=float(change),
                               inverse_trace_log_skin_derivative=float((1 if initial == "left" else -1) * (1 + 2 * mean)),
                               inverse_weight_mean_cell=float(mean))
                    rows.append(row)
                    raw.append({**row, "lower_mp": str(lo), "upper_mp": str(hi), "inverse_diagonal_distribution": [str(x) for x in distribution]})
                emit(dict(stage="spatial_lock", length=length, skin=skin, initial=initial, relative_width=row["relative_width"]))
    write_csv("locked_spatial_predictions.csv", rows)
    write_json(OUT / "spatial/locked_inverse_trace.json", raw)
    write_json(path, dict(locked_utc=datetime.now(timezone.utc).isoformat(), rows=len(rows), script_sha256=SOURCE_SHA,
                         config_sha256=audit.sha256(OUT / "config.json"), method=configuration()["spatial"],
                         interpretation="finite m Gram inversion, not an asymptotic exponent or constant-time law"))
    write_json(OUT / "spatial_lock_hash.json", {name: audit.sha256(OUT / name) for name in
               ("spatial_hypothesis_lock.json", "locked_spatial_predictions.csv", "spatial/locked_inverse_trace.json")})


def static_job(family):
    key = spatial.identifier(*family)
    path = OUT / "static" / (key + ".json")
    if path.exists():
        return dict(stage="static", family=key, status="reused")
    check_hashes("spatial_lock_hash.json")
    tick, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    if family[0] == 80:
        source = PRIOR / "static" / (key + ".raw.json.gz")
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            values = json.load(stream)
        change = read_json(PRIOR / "static" / (key + ".json"))["reference_convergence"]
        reused = True
    else:
        values, reused = [], False
        for digits in DIGITS:
            static_tick = time.perf_counter()
            value = spatial.static_reference(family, digits)
            spatial_seconds = time.perf_counter() - static_tick
            compression_tick = time.perf_counter()
            value = base.decompose(value, ["charge_density_wave", "left", "right"])
            value["costs"] = dict(spectrum_graph_overlap_entropy_seconds=spatial_seconds,
                                  initial_graph_svd_seconds=time.perf_counter() - compression_tick)
            values.append(value)
            if len(values) > 1:
                change = spatial.static_change(values[-2], values[-1])
                with mp.workdps(340):
                    changes = [spatial.relative(mp.mpf(a), mp.mpf(b)) for name in ("charge_density_wave", "left", "right")
                               for a, b in zip(values[-2]["states"][name]["compression"]["singular_values"][:17],
                                               values[-1]["states"][name]["compression"]["singular_values"][:17])]
                change["compression_relative"] = float(max(changes))
                if max(change["relative_scalar"], change["compression_relative"]) <= 1e-8 and max(change["matrix_absolute"], change["gain_entropy"]) <= 1e-10:
                    break
        else:
            raise ArithmeticError("new-size static data did not converge")
    with gzip.open(path.with_suffix(".raw.json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(values, stream)
    write_json(path, {**values[-1], "reference_convergence": change, "precision_digits": [v["dps"] for v in values],
                     "started_utc": started, "seconds": time.perf_counter() - tick, "reused_fifth_static": reused,
                     "script_sha256": SOURCE_SHA, "config_sha256": audit.sha256(OUT / "config.json"),
                     "spatial_lock_sha256": audit.sha256(OUT / "spatial_hypothesis_lock.json")})
    return dict(stage="static", family=key, dps=values[-1]["dps"], seconds=time.perf_counter() - tick)


def prediction_job(family):
    key = spatial.identifier(*family)
    path = OUT / "predictions" / (key + ".json")
    if path.exists():
        return dict(stage="prediction", family=key, status="reused")
    with gzip.open(OUT / "static" / (key + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
        statics = json.load(stream)[-2:]
    planned = [r for r in cases() if (r["length"], r["gamma"], r["skin"]) == tuple(family)]
    contexts, rows, pairs, raw, arrays = [], [], [], [], {}
    for static in statics:
        with mp.workdps(static["dps"]):
            contexts.append({name: context(static, name) for name in planned[0]["initials"]})
    for case in planned:
        locals_at_precision = []
        for static, ctx in zip(statics, contexts):
            local = {}
            with mp.workdps(static["dps"]):
                for initial in case["initials"]:
                    local[initial] = prediction(static, initial, mp.mpf(str(case["physical_time"])), ctx[initial])
            locals_at_precision.append(local)
        with mp.workdps(340):
            for initial in case["initials"]:
                low = locals_at_precision[0][initial][0]
                values, q, attempts, costs = locals_at_precision[1][initial]
                change = max(abs(low[k] - values[k]) for k in ("distance_to_G", "entropy", "projector_remainder", "distance_lower", "distance_upper", "entropy_error_upper"))
                assert change <= mp.mpf("1e-10") and low["rank"] == values["rank"]
                numeric = {k: float(v) for k, v in values.items()}
                rows.append({**case, "initial": initial, **numeric, **costs, "prediction_precision_change": float(change),
                             "prediction_dps": statics[-1]["dps"], "rank_target_passed": numeric["projector_remainder"] <= 1e-5,
                             "guaranteed_selected": numeric["distance_upper"] <= .05 and numeric["entropy_error_upper"] <= .05,
                             "guaranteed_unselected": numeric["distance_lower"] > .05 or numeric["entropy_error_lower"] > .05})
                arrays[case["case_id"] + "_" + initial] = audit.mp_to_numpy(q)
                raw.append(dict(case_id=case["case_id"], initial=initial, Q=p1.encode_matrix(q),
                                values={k: str(v) for k, v in values.items()}, attempts=attempts))
            for initial in case["initials"][1:]:
                pair_values = []
                for local in locals_at_precision:
                    a, qa = local["charge_density_wave"][:2]
                    b, qb = local[initial][:2]
                    pair_values.append(pair_bound(a, b, qa, qb))
                pair_change = max(abs(pair_values[0][k] - pair_values[1][k]) for k in pair_values[0])
                assert pair_change <= mp.mpf("1e-10")
                bound = {k: float(v) for k, v in pair_values[-1].items()}
                pairs.append({**case, "initial_a": "charge_density_wave", "initial_b": initial, **bound,
                              "guaranteed_memory": bound["memory_lower"] > .1, "prediction_precision_change": float(pair_change)})
    write_json(path, dict(family=family, rows=rows, pairs=pairs, script_sha256=SOURCE_SHA,
                         static_sha256=audit.sha256(OUT / "static" / (key + ".json"))))
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    with gzip.open(path.with_suffix(".raw.json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(raw, stream)
    return dict(stage="prediction", family=key, states=len(rows), target_passed=sum(r["rank_target_passed"] for r in rows))


def check_hashes(filename):
    for name, digest in read_json(OUT / filename).items():
        assert audit.sha256(OUT / name) == digest, name


def lock():
    if (OUT / "prediction_lock.json").exists():
        check_hashes("prediction_lock_hash.json")
        return
    assert not list((OUT / "cases").glob("*.json"))
    values = [read_json(path) for path in sorted((OUT / "predictions").glob("*.json"))]
    rows, pairs = [r for v in values for r in v["rows"]], [r for v in values for r in v["pairs"]]
    assert len(values) == 8 and len(rows) == 57 and len(pairs) == 38
    write_csv("locked_predictions.csv", rows)
    write_csv("locked_memory_predictions.csv", pairs)
    inputs = [p for folder in ("static", "predictions", "development") for p in (OUT / folder).rglob("*") if p.is_file()]
    value = dict(locked_utc=datetime.now(timezone.utc).isoformat(), states=len(rows), pairs=len(pairs),
                 ranks=list(RANKS), config_sha256=audit.sha256(OUT / "config.json"), script_sha256=SOURCE_SHA,
                 inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=audit.sha256(p)) for p in sorted(inputs)],
                 policy="no rank, grid, residual, memory or tolerance retuning after new truth")
    write_json(OUT / "prediction_lock.json", value)
    write_json(OUT / "prediction_lock_hash.json", {name: audit.sha256(OUT / name) for name in
               ("prediction_lock.json", "locked_predictions.csv", "locked_memory_predictions.csv")})
    emit(dict(stage="lock", states=len(rows), pairs=len(pairs), utc=value["locked_utc"]))


def reference(case, digits):
    with mp.workdps(digits):
        tick = time.perf_counter()
        matrix = mp.expm(previous.generator(case) * mp.mpf(str(case["physical_time"])))
        scale = p1.mp_norm(matrix)
        matrix /= scale
        exp_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        w, s, vh = mp.svd(matrix)
        n = case["length"] // 2
        top, bottom = w[:, :n], w[:, n:]
        svd_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        wentropy, _ = previous.correlation_entropy(top)
        wentropy_seconds = time.perf_counter() - tick
        phase = np.tile([1., 1.j], n)
        arrays = dict(M=phase[:, None] * audit.mp_to_numpy(matrix) * phase.conj()[None, :],
                      W=phase[:, None] * audit.mp_to_numpy(top), s=np.array([float(x) for x in s]))
        raw = dict(dps=digits, M=p1.encode_matrix(matrix), W=p1.encode_matrix(top), singular_values=[str(x) for x in s],
                   log_fro_scale=str(mp.log(scale)), W_entropy=str(wentropy), states={},
                   costs=dict(exponential_seconds=exp_seconds, full_svd_seconds=svd_seconds, W_entropy_seconds=wentropy_seconds))
        rows = []
        for initial in case["initials"]:
            tick = time.perf_counter()
            q0 = base.q0_matrix(case["length"], initial)
            q = mp.qr(matrix * q0, mode="skinny")[0]
            qr_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            entropy, _ = previous.correlation_entropy(q)
            entropy_seconds = time.perf_counter() - tick
            distance = previous.mp_opnorm(bottom.T * q)
            alpha = previous.mp_smin(vh[:n, :] * q0)
            ratio = s[n] / s[n - 1]
            values = dict(entropy=entropy, distance_to_W=distance, alpha=alpha, ratio=ratio)
            rows.append({**case, "initial": initial, "reference_dps": digits, **{k: float(v) for k, v in values.items()},
                         "W_entropy": float(wentropy), "entropy_error": float(abs(entropy - wentropy)),
                         "controlled": distance <= mp.mpf(".05") and abs(entropy - wentropy) <= mp.mpf(".05"),
                         "reference_qr_seconds": qr_seconds, "reference_entropy_seconds": entropy_seconds})
            arrays["Q_" + initial] = phase[:, None] * audit.mp_to_numpy(q)
            raw["states"][initial] = {**{k: str(v) for k, v in values.items()}, "Q": p1.encode_matrix(q)}
        return rows, arrays, raw


def case_job(case):
    check_hashes("prediction_lock_hash.json")
    path = OUT / "cases" / (case["case_id"] + ".json")
    if path.exists():
        return dict(stage="reference", case_id=case["case_id"], status="reused")
    tick, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    double = base.double_reference(case)
    refs = []
    for digits in DIGITS:
        refs.append(reference(case, digits))
        change = spatial.reference_change(refs[-2], refs[-1]) if len(refs) > 1 else None
        with path.with_suffix(".events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(dps=digits, seconds=time.perf_counter() - tick, convergence=change)) + "\n")
        if change is not None and previous.accepted_precision(change):
            break
    else:
        raise ArithmeticError("independent exponential reference failed convergence")
    rows, arrays, raw = refs[-1]
    static = read_json(OUT / "static" / (spatial.identifier(case["length"], 2., case["skin"]) + ".json"))
    phase = np.tile([1., 1.j], case["length"] // 2)
    stable_raw = {}
    for row in rows:
        initial = row["initial"]
        qdouble = double["Q_" + initial + "_double"]
        row.update(double_projector_error=audit.projector_error(arrays["Q_" + initial], qdouble),
                   double_entropy_error=abs(row["entropy"] - audit.entropy(qdouble)))
        row["double_status"] = "reference_validated" if max(row["double_projector_error"], row["double_entropy_error"]) <= 1e-6 else "reference_failed"
        stable_raw[initial] = []
        for stable_digits in (80, 120):
            q, entropy, seconds = base.stable_static_q(static, initial, mp.mpf(str(case["physical_time"])), stable_digits)
            row[f"static{stable_digits}_projector_error"] = audit.projector_error(arrays["Q_" + initial], phase[:, None] * audit.mp_to_numpy(q))
            with mp.workdps(340):
                row[f"static{stable_digits}_entropy_error"] = float(abs(entropy - mp.mpf(raw["states"][initial]["entropy"])))
            row[f"static{stable_digits}_seconds"] = seconds
            with mp.workdps(stable_digits):
                stable_raw[initial].append(dict(dps=stable_digits, Q=p1.encode_matrix(q), entropy=str(entropy), seconds=seconds))
        row["stable_status"] = "reference_validated" if max(row[f"static{d}_{k}_error"] for d in (80, 120) for k in ("projector", "entropy")) <= 1e-6 else "reference_failed"
    np.savez_compressed(path.with_suffix(".npz"), **double, **arrays)
    with gzip.open(path.with_suffix(".raw.json.gz"), "wt", encoding="utf-8") as stream:
        json.dump([r[2] for r in refs], stream)
    with gzip.open(path.with_suffix(".stable.raw.json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(stable_raw, stream)
    result = dict(case=case, rows=rows, precision_digits=[r[2]["dps"] for r in refs], reference_convergence=change,
                  started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - tick,
                  reference_costs=[r[2]["costs"] for r in refs], script_sha256=SOURCE_SHA,
                  config_sha256=audit.sha256(OUT / "config.json"), prediction_lock_sha256=audit.sha256(OUT / "prediction_lock.json"))
    write_json(path, result)
    return dict(stage="reference", case_id=case["case_id"], dps=digits, seconds=result["seconds"],
                double_failed=sum(r["double_status"] != "reference_validated" for r in rows),
                stable_failed=sum(r["stable_status"] != "reference_validated" for r in rows))


def parallel(function, jobs, workers, source_directory):
    failures = []
    with ProcessPoolExecutor(max_workers=workers, initializer=base.worker_init, initargs=(source_directory,)) as pool:
        pending = {pool.submit(function, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                emit(future.result())
            except Exception as error:
                key = job["case_id"] if isinstance(job, dict) else spatial.identifier(*job)
                value = dict(job=key, function=function.__name__, error_type=type(error).__name__, error=str(error),
                             traceback=traceback.format_exc(), utc=datetime.now(timezone.utc).isoformat(), source_sha256=SOURCE_SHA)
                write_json(OUT / "failures" / (function.__name__ + "_" + key + "_" + SOURCE_SHA[:12] + ".json"), value)
                failures.append(value)
                emit(value)
    assert not failures, f"{len(failures)} failed jobs preserved"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "spatial_lock", "static", "lock", "validate"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert 1 <= args.workers <= 8
    prepare()
    tick = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="prl_direction_baseline_") as directory:
        with zipfile.ZipFile(ROOT / "data/prl_p0_precision/baseline_snapshot.zip") as archive:
            for name in archive.namelist():
                if name.startswith("src/"):
                    archive.extract(name, directory)
        source_directory = str(Path(directory) / "src")
        sys.path.insert(0, source_directory)
        with threadpool_limits(limits=1):
            if args.stage == "development":
                used = sorted({(r["length"], r["gamma"], r["skin"]) for r in base.cases()})
                parallel(development_job, used, args.workers, source_directory)
                finish_development()
            elif args.stage == "spatial_lock":
                assert (OUT / "development/summary.json").exists()
                spatial_lock()
            elif args.stage == "static":
                parallel(static_job, families(), args.workers, source_directory)
            elif args.stage == "lock":
                parallel(prediction_job, families(), args.workers, source_directory)
                lock()
            else:
                check_hashes("prediction_lock_hash.json")
                parallel(case_job, cases(), args.workers, source_directory)
    info = audit.environment(args.stage)
    info.update(stage=args.stage, script_sha256=SOURCE_SHA, workers=args.workers,
                elapsed_seconds=time.perf_counter() - tick, scope="local Windows; no HPC execution")
    write_json(OUT / ("environment_" + args.stage + ".json"), info)


if __name__ == "__main__":
    main()
