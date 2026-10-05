"""P2: preregistered spectral envelopes, early forecasts, and held-out validation."""

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import mpmath as mp
import numpy as np
import psutil
from scipy import linalg, optimize, special
from threadpoolctl import threadpool_limits

import audit_numerical_precision as audit
import run_isospectral_actual_entanglement as p1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prl_p2_prediction"
FIG = ROOT / "figures/prl_p2_prediction"
PRIOR = ROOT / "data/prl_p1_isospectral_actual"
SOURCE_SHA = audit.sha256(Path(__file__))
INITIALS = audit.INITIALS
TIMES = (5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def emit(value):
    print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **value}, ensure_ascii=True, allow_nan=False), flush=True)


def cases():
    values = []
    for length in (16, 24, 32, 40, 48):
        for skin in (0.0, 0.25):
            for duration in TIMES:
                role = "early" if length <= 32 and duration <= 10 else ("time" if length <= 32 else "size")
                values.append(make_case(length, skin, duration, role))
    for length in (24, 32, 40, 48):
        for skin in (0.05, 0.1):
            for duration in (10.0, 15.0, 20.0, 25.0):
                values.append(make_case(length, skin, duration, "weak_skin"))
    return values


def make_case(length, skin, duration, role):
    steps = round(duration / 0.05)
    return dict(case_id=p1.case_key(length, 2.0, skin, steps), length=length, gamma=2.0,
                skin=skin, physical_time=duration, steps=steps, dt=0.05, t1=0.5, t2=1.0,
                boundary="open", role=role)


def configuration():
    return dict(study_date="2026-10-04", version=1, planned_input_sha256=audit.sha256(PRIOR / "next_round_plan.json"),
                product_cases=cases(), initials=list(INITIALS), precision_digits=[80, 120], adaptive_digits=160,
                reference_goals=dict(projector=1e-10, entropy=1e-10, relative_scalar=1e-8),
                numerical_goals=dict(projector=1e-6, entropy=1e-6, orthogonality=1e-10),
                tolerances=dict(projector=0.05, absolute_entropy=0.05, memory_projector=0.1, memory_entropy=0.1),
                training=dict(lengths=[16, 24, 32], skin=[0.0, 0.25], times=[5.0, 7.5, 10.0]),
                general=dict(lengths=[16, 32], skin=[0.0, 0.25], times=[5.0, 15.0], random_seeds=list(range(20)),
                             ground_definition="lowest N eigenvectors of OBC reciprocal Hermitian SSH with gamma=g=0",
                             random_definition="QR of independent standard complex Gaussian L by N; numpy.default_rng(seed)",
                             fixed_initial="same Q0 at every g and t for a given L and initial id"),
                physical_frame="M_real=T^dagger exp(-iHt) T; T=diag(1,i,1,i,...); local unitary frame",
                general_precision="binary64 initial matrices reorthogonalized at reference precision; raw initial matrices saved",
                previously_seen_late_conditions=[c["case_id"] for c in cases() if c["role"] == "time" and (PRIOR / "cases" / (c["case_id"] + ".json")).exists()],
                scope="finite gamma=2 OBC pilot; no thermodynamic phase, no HPC production; empirical and sufficient forecasts separate")


def prepare():
    for name in ("cases", "code", "static", "general", "initial_states", "failures"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    config = configuration()
    if (OUT / "config.json").exists():
        assert json.loads((OUT / "config.json").read_text(encoding="utf-8")) == config
    else:
        write_json(OUT / "config.json", config)
    if not (OUT / "input_manifest.csv").exists():
        inputs = {ROOT / row["path"] for row in audit.read_csv(PRIOR / "input_manifest.csv")}
        inputs.update(p for p in PRIOR.rglob("*") if p.is_file())
        inputs.update((ROOT / "scripts/run_isospectral_actual_entanglement.py",
                       ROOT / "research_doc/04_历史研究/04_基线复核与精度校验.md",
                       ROOT / "research_doc/04_历史研究/05_实际纠缠与有限时间选择.md"))
        write_csv("input_manifest.csv", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                                              sha256=audit.sha256(p)) for p in sorted(inputs)])
    for row in audit.read_csv(OUT / "input_manifest.csv"):
        assert audit.sha256(ROOT / row["path"]) == row["sha256"], f"immutable input changed: {row['path']}"
    assert audit.sha256(Path(__file__)) == SOURCE_SHA
    (OUT / "code" / (SOURCE_SHA.lower() + ".py")).write_bytes(Path(__file__).read_bytes())


def worker_init(source_directory):
    sys.path.insert(0, source_directory)
    threadpool_limits(limits=1)


def generator(case):
    n = case["length"]
    j = mp.matrix(n)
    t1, t2, gamma, skin = (mp.mpf(str(case[k])) for k in ("t1", "t2", "gamma", "skin"))
    for a in range(0, n, 2):
        j[a, a], j[a + 1, a + 1] = gamma, -gamma
        j[a, a + 1], j[a + 1, a] = t1, -t1
        if a + 2 < n:
            j[a + 2, a + 1], j[a + 1, a + 2] = t2 * mp.exp(skin), -t2 * mp.exp(-skin)
    return j


def product_initial(length, initial):
    q = mp.matrix(length, length // 2)
    occupied = range(0, length, 2) if initial == INITIALS[0] else range(length // 2)
    for column, site in enumerate(occupied):
        q[site, column] = 1
    return q


def mp_opnorm(value):
    return mp.svd(value, compute_uv=False)[0]


def mp_smin(value):
    return mp.svd(value, compute_uv=False)[min(value.rows, value.cols) - 1]


def correlation_entropy(q):
    left = q[:q.rows // 2, :]
    spectrum = mp.eighe(left * left.H, eigvals_only=True)
    tolerance = mp.mpf(10) ** (-mp.mp.dps + 8)
    entropy = mp.mpf(0)
    values = []
    for x in spectrum:
        if x < -tolerance or x > 1 + tolerance:
            raise ArithmeticError("correlation spectrum outside [0,1]")
        x = min(mp.mpf(1), max(mp.mpf(0), x))
        values.append(float(x))
        if x not in (0, 1):
            entropy -= x * mp.log(x) + (1 - x) * mp.log(1 - x)
    return entropy, values


def reference(case, dps, general=False):
    with mp.workdps(dps):
        matrix = mp.expm(generator(case) * mp.mpf(str(case["physical_time"])))
        scale = p1.mp_norm(matrix)
        matrix /= scale
        w, singular, vh = mp.svd(matrix)
        n = case["length"] // 2
        wtop = w[:, :n]
        wentropy, _ = correlation_entropy(wtop)
        phase = np.tile([1.0, 1.0j], n)
        converted = dict(M=phase[:, None] * audit.mp_to_numpy(matrix) * phase.conj()[None, :],
                         W=phase[:, None] * audit.mp_to_numpy(wtop), s=np.array([float(x) for x in singular]))
        raw = dict(dps=dps, frame="local_unitary_real_generator", M=p1.encode_matrix(matrix), W=p1.encode_matrix(wtop),
                   VH=p1.encode_matrix(vh), singular_values=[str(x) for x in singular], log_fro_scale=str(mp.log(scale)), states={})
        if general:
            with np.load(OUT / "initial_states" / f"L{case['length']}.npz") as initial_archive:
                inputs = {k: initial_archive[k].copy() for k in initial_archive.files}
        else:
            inputs = {name: None for name in INITIALS}
        rows = []
        for initial, given in inputs.items():
            if given is None:
                q0 = product_initial(case["length"], initial)
            else:
                q0 = mp.matrix((phase.conj()[:, None] * given).tolist())
                q0 = mp.qr(q0, mode="skinny")[0]
            q = mp.qr(matrix * q0, mode="skinny")[0]
            ent, occupation_spectrum = correlation_entropy(q)
            distance = p1.mp_distance(q, wtop)
            graph, k = p1.mp_graph(singular, vh, q0)
            restricted = q[:n, :] * q[:n, :].H - wtop[:n, :] * wtop[:n, :].H
            restricted = (restricted + restricted.H) / 2
            delta = mp.fsum(abs(x) for x in mp.eighe(restricted, eigvals_only=True))
            x = min(delta / n, mp.mpf("0.5"))
            binary_bound = n * (-x * mp.log(x) - (1 - x) * mp.log(1 - x)) if x else mp.mpf(0)
            density = [float(mp.fsum(abs(q[i, j])**2 for j in range(n))) for i in range(case["length"])]
            row = {**case, "initial": initial, "reference_dps": dps, "entropy": float(ent), "W_entropy": float(wentropy),
                   "entropy_error": float(abs(ent - wentropy)), "distance_to_W": float(distance),
                   "restricted_trace_distance": float(delta), "binary_entropy_bound": float(binary_bound),
                   "graph_identity_error": float(abs(distance - graph["graph_distance"])),
                   "right_half_particles": sum(density[n:]),
                   **{key: float(value) for key, value in graph.items()}}
            assert row["graph_identity_error"] < 1e-10
            assert row["distance_to_W"] <= row["bound_distance"] + 1e-10
            assert row["entropy_error"] <= row["binary_entropy_bound"] + 1e-10
            rows.append(row)
            converted["Q_" + initial] = phase[:, None] * audit.mp_to_numpy(q)
            raw["states"][initial] = dict(Q=p1.encode_matrix(q), K=p1.encode_matrix(k), entropy=str(ent),
                                          distance_to_W=str(distance), graph={key: str(value) for key, value in graph.items()},
                                          density=density, correlation_spectrum=occupation_spectrum)
        return rows, converted, raw


def double_states(case, general=False):
    if not general:
        return p1.double_reference(case)
    from mietf_skin.ssh import ssh_no_click_hamiltonian

    h = ssh_no_click_hamiltonian(case["length"] // 2, 0.5, 1.0, 2.0, case["skin"], "open")
    matrix = audit.normalize(linalg.expm(-1j * h * case["physical_time"]))
    w, s, _ = linalg.svd(matrix, lapack_driver="gesvd")
    values = dict(M_double=matrix, W_double=w[:, :case["length"] // 2], s_double=s)
    with np.load(OUT / "initial_states" / f"L{case['length']}.npz") as archive:
        inputs = {k: archive[k].copy() for k in archive.files}
    for method, dt, steps in (("step", case["dt"], case["steps"]), ("half", case["dt"] / 2, case["steps"] * 2)):
        step = linalg.expm(-1j * h * dt)
        for initial, q0 in inputs.items():
            q = q0.copy()
            for _ in range(steps):
                q = linalg.qr(step @ q, mode="economic")[0]
            values[f"Q_{initial}_{method}"] = q
    return values


def convergence(old, new):
    first, arrays_first, raw_first = old
    second, arrays_second, raw_second = new
    with mp.workdps(160):
        scalar_changes = []
        for initial in raw_second["states"]:
            for key in ("ratio", "alpha", "graph_norm"):
                a = mp.mpf(raw_first["states"][initial]["graph"][key])
                b = mp.mpf(raw_second["states"][initial]["graph"][key])
                scalar_changes.append(float(abs(a - b) / max(abs(a), abs(b))) if a or b else 0.0)
        entropy_change = max(float(abs(mp.mpf(raw_second["states"][initial]["entropy"]) -
                                       mp.mpf(raw_first["states"][initial]["entropy"]))) for initial in raw_second["states"])
    return dict(M=float(linalg.norm(arrays_first["M"] - arrays_second["M"])),
                W=audit.projector_error(arrays_first["W"], arrays_second["W"]),
                Q=max(audit.projector_error(arrays_first["Q_" + r["initial"]], arrays_second["Q_" + r["initial"]]) for r in second),
                S=entropy_change, scalar_relative=max(scalar_changes))


def accepted_precision(value):
    return max(value[k] for k in ("M", "W", "Q", "S")) <= 1e-10 and value["scalar_relative"] <= 1e-8


def run_case(case, general=False):
    directory = OUT / ("general" if general else "cases")
    path = directory / (case["case_id"] + ".json")
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["config_sha256"] == audit.sha256(OUT / "config.json")
        return dict(case_id=case["case_id"], status="reused", seconds=0)
    tick = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    prior_path = PRIOR / "cases" / path.name
    if not general and prior_path.exists():
        old = json.loads(prior_path.read_text(encoding="utf-8"))
        value = {**old, "case": case, "rows": [{**r, **case} for r in old["rows"]],
                 "config_sha256": audit.sha256(OUT / "config.json"), "reused_from": prior_path.relative_to(ROOT).as_posix(),
                 "source_receipt_sha256": audit.sha256(prior_path), "current_wrapper_sha256": SOURCE_SHA}
        write_json(path, value)
        return dict(case_id=case["case_id"], status="prior_reference_reused", seconds=time.perf_counter() - tick)
    double = double_states(case, general)
    previous, checks, references = None, [], []
    for dps in (80, 120, 160):
        current = reference(case, dps, general)
        references.append(current[2])
        change = convergence(previous, current) if previous is not None else None
        checks.append(dict(dps=dps, convergence=change))
        with path.with_suffix(".events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), dps=dps,
                                         seconds=time.perf_counter() - tick, convergence=change)) + "\n")
        previous = current
        if change is not None and accepted_precision(change):
            break
    else:
        raise ArithmeticError(f"reference failed to converge: {case['case_id']} {change}")
    rows, converted, _ = current
    for row in rows:
        initial = row["initial"]
        row.update(reference_status="converged", W_double_projector_error=audit.projector_error(converted["W"], double["W_double"]),
                   W_double_entropy_error=abs(row["W_entropy"] - audit.entropy(double["W_double"])))
        for method in ("step", "half"):
            q = double[f"Q_{initial}_{method}"]
            row[f"double_{method}_projector_error"] = audit.projector_error(converted["Q_" + initial], q)
            row[f"double_{method}_entropy_error"] = abs(row["entropy"] - audit.entropy(q))
            row[f"double_{method}_orthogonality"] = audit.opnorm(q.conj().T @ q - np.eye(q.shape[1]))
        row["actual_double_status"] = "reference_validated" if row["double_step_projector_error"] <= 1e-6 and row["double_step_entropy_error"] <= 1e-6 else "reference_failed"
        row["controlled"] = row["distance_to_W"] <= 0.05 and row["entropy_error"] <= 0.05
    value = dict(case=case, rows=rows, precision_checks=checks, reference_convergence=change,
                 config_sha256=audit.sha256(OUT / "config.json"), script_sha256=SOURCE_SHA,
                 seconds=time.perf_counter() - tick, general=general,
                 started_utc=started_utc, completed_utc=datetime.now(timezone.utc).isoformat(),
                 prediction_lock_sha256=audit.sha256(OUT/"prediction_lock.json") if case["role"] != "early" else None,
                 initial_array_sha256=audit.sha256(OUT/"initial_states"/f"L{case['length']}.npz") if general else None)
    np.savez_compressed(path.with_suffix(".npz"), **double, **converted)
    with gzip.open(path.with_suffix(".raw.json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(references, stream)
    write_json(path, value)
    return dict(case_id=case["case_id"], status="converged", dps=dps, seconds=value["seconds"],
                max_actual_double_error=max(r["double_step_projector_error"] for r in rows))


def static_reference(length, skin, dps):
    with mp.workdps(dps):
        n = length // 2
        coupling = mp.matrix(n)
        for j in range(n):
            coupling[j, j] = mp.mpf("0.5")
            if j:
                coupling[j, j - 1] = 1
        u, s, vt = mp.svd(coupling)
        v = vt.T
        gamma = mp.mpf(2)
        mu = [mp.sqrt(gamma**2 - x**2) for x in s]
        xmatrix, ymatrix = mp.matrix(length), mp.matrix(length)
        diagonal = [mp.exp(mp.mpf(str(skin)) * (j // 2)) for j in range(length)]
        for mode in range(n):
            ratio = s[mode] / (gamma + mu[mode])
            a, b = 1 / mp.sqrt(1 + ratio**2), ratio / mp.sqrt(1 + ratio**2)
            for j in range(n):
                xmatrix[2*j, mode], xmatrix[2*j+1, mode] = u[j, mode]*a, -v[j, mode]*b
                xmatrix[2*j, n+mode], xmatrix[2*j+1, n+mode] = u[j, mode]*b, -v[j, mode]*a
                ymatrix[mode, 2*j], ymatrix[mode, 2*j+1] = gamma/mu[mode]*a*u[j, mode], gamma/mu[mode]*b*v[j, mode]
                ymatrix[n+mode, 2*j], ymatrix[n+mode, 2*j+1] = -gamma/mu[mode]*b*u[j, mode], -gamma/mu[mode]*a*v[j, mode]
        for column in range(length):
            norm = mp.sqrt(mp.fsum(abs(diagonal[row] * xmatrix[row, column])**2 for row in range(length)))
            for row in range(length):
                xmatrix[row, column] *= diagonal[row] / norm
                ymatrix[column, row] *= norm / diagonal[row]
        plus, minus = xmatrix[:, :n], xmatrix[:, n:]
        yplus, yminus = ymatrix[:n, :], ymatrix[n:, :]
        gain = mp.qr(plus, mode="skinny")[0]
        input_gain = mp.qr(yplus.H, mode="skinny")[0]
        geometry = dict(a=mp_smin(gain.H * plus), b=mp_opnorm(minus - gain * gain.H * minus),
                        c=mp_opnorm(gain.H * minus), p=mp_smin(plus) * mp_smin(yplus),
                        q=mp_opnorm(minus) * mp_opnorm(yminus), min_gain_rate=min(mu))
        case = make_case(length, skin, 5.0, "static")
        eigen_residual = p1.mp_norm(generator(case)*xmatrix - xmatrix*mp.diag([*mu, *[-x for x in mu]])) / p1.mp_norm(xmatrix)
        inverse_residual = p1.mp_norm(xmatrix*ymatrix - mp.eye(length))
        assert max(eigen_residual, inverse_residual) < mp.mpf("1e-50")
        states = {}
        for initial in INITIALS:
            q0 = product_initial(length, initial)
            aa, bb = yplus*q0, yminus*q0
            f = mp.matrix(n)
            for i in range(n):
                solution = mp.lu_solve(aa.T, mp.matrix([bb[i,j] for j in range(n)]))
                for j in range(n):
                    f[i,j] = solution[j]
            states[initial] = dict(alpha_infinite=str(mp_smin(input_gain.H*q0)), F_norm=str(mp_opnorm(f)),
                                   log_abs_F=[[float(mp.log(abs(f[i,j]))) if f[i,j] else None for j in range(n)] for i in range(n)],
                                   F=p1.encode_matrix(f))
        return dict(length=length, skin=skin, dps=dps, mu=[str(x) for x in mu],
                    geometry={k:str(x) for k,x in geometry.items()}, states=states,
                    eigen_residual=str(eigen_residual), inverse_residual=str(inverse_residual),
                    gain_entropy=str(audit.mp_entropy(gain)))


def run_static(family):
    length, skin = family
    identifier = f"L{length}_g{str(skin).replace('.', 'p')}"
    path = OUT / "static" / (identifier + ".json")
    if path.exists():
        return dict(static=identifier, status="reused")
    tick = time.perf_counter()
    references = [static_reference(length, skin, dps) for dps in (80,120)]
    with mp.workdps(160):
        changes = []
        for initial in INITIALS:
            for key in ("alpha_infinite", "F_norm"):
                a, b = (mp.mpf(r["states"][initial][key]) for r in references)
                changes.append(float(abs(a-b)/max(abs(a),abs(b))))
        relative = max(changes)
    assert relative <= 1e-8
    result = {**references[-1], "reference_relative_change":relative, "script_sha256":SOURCE_SHA,
              "config_sha256":audit.sha256(OUT/"config.json"), "seconds":time.perf_counter()-tick}
    with gzip.open(path.with_suffix(".raw.json.gz"),"wt",encoding="utf-8") as stream:
        json.dump(references,stream)
    write_json(path,result)
    return dict(static=identifier,status="converged",seconds=result["seconds"])


def run_parallel(function, jobs, workers, source_directory, general=False):
    failures = []
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(source_directory,)) as executor:
        pending = {executor.submit(function,job,general) if function is run_case else executor.submit(function,job):job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                emit(future.result())
            except Exception as error:
                identifier = job.get("case_id") if isinstance(job,dict) else f"static_{job[0]}_{job[1]}"
                failure = dict(identifier=identifier,error_type=type(error).__name__,error=str(error),script_sha256=SOURCE_SHA)
                failures.append(failure)
                write_json(OUT/"failures"/(identifier+".json"),failure)
                emit(failure)
    assert not failures, f"{len(failures)} calculations failed; see failures directory"


def binary_entropy(x):
    return -x*math.log(x)-(1-x)*math.log1p(-x) if 0<x<1 else 0.0


def physical_envelope(static, initial, duration, weighted):
    geometry = {k:float(v) for k,v in static["geometry"].items()}
    state = static["states"][initial]
    mu = np.array([float(x) for x in static["mu"]])
    if weighted:
        logs = np.array([[(-np.inf if x is None else x) for x in row] for row in state["log_abs_F"]])
        logf = float(special.logsumexp(2*logs-2*(mu[:,None]+mu[None,:])*duration)/2)
    else:
        logf = math.log(float(state["F_norm"]))-2*geometry["min_gain_rate"]*duration
    limit = geometry["a"]/geometry["c"] if geometry["c"] else math.inf
    if logf >= math.log(limit):
        dqg = 1.0
    else:
        f = math.exp(logf)
        k = geometry["b"]*f/(geometry["a"]-geometry["c"]*f)
        dqg = k/math.hypot(1,k)
    z = geometry["q"]*math.exp(-2*geometry["min_gain_rate"]*duration)
    dwg = min(1.0,z/(geometry["p"]-z)) if z<geometry["p"] else 1.0
    d = min(1.0,dqg+dwg)
    return dict(distance_bound=d, entropy_bound=static["length"]//2*binary_entropy(min(d,0.5)),
                actual_to_gain_bound=dqg, output_to_gain_bound=dwg)


def empirical_value(model,length,skin,duration):
    endpoints = []
    for endpoint in (0.0,0.25):
        item = model[str(endpoint)]
        endpoints.append((float(np.polyval(item["intercept_size_fit"],length)),float(np.polyval(item["rate_size_fit"],length))))
    weight = skin/0.25
    intercept,rate = ((1-weight)*endpoints[0][i]+weight*endpoints[1][i] for i in range(2))
    assert rate>0
    logk = intercept-rate*duration
    d = 1/math.sqrt(1+math.exp(-2*logk)) if logk>=0 else math.exp(logk)/math.sqrt(1+math.exp(2*logk))
    return dict(distance_prediction=d,entropy_budget_proxy=length//2*binary_entropy(min(d,0.5)),rate=rate,log_intercept=intercept)


def lock_predictions():
    path = OUT/"prediction_lock.json"
    if path.exists():
        assert audit.sha256(path)==json.loads((OUT/"prediction_lock_hash.json").read_text())["sha256"]
        return
    assert not list((OUT/"cases").glob("L40*.json")) and not list((OUT/"cases").glob("L48*.json")), "holdout truth was already generated"
    early=[]
    for case in cases():
        if case["role"]=="early":
            item=json.loads((OUT/"cases"/(case["case_id"]+".json")).read_text())
            early.extend(item["rows"])
    model={}
    fits=[]
    for initial in INITIALS:
        model[initial]={}
        for skin in (0.0,0.25):
            individual=[]
            for length in (16,24,32):
                data=sorted([r for r in early if r["initial"]==initial and r["skin"]==skin and r["length"]==length],key=lambda r:r["physical_time"])
                assert len(data)==3
                slope,intercept=np.polyfit([r["physical_time"] for r in data],np.log([r["graph_norm"] for r in data]),1)
                individual.append(dict(length=length,intercept=float(intercept),rate=float(-slope)))
                fits.append(dict(initial=initial,skin=skin,**individual[-1]))
            model[initial][str(skin)]=dict(intercept_size_fit=np.polyfit([x["length"] for x in individual],[x["intercept"] for x in individual],1).tolist(),
                                         rate_size_fit=np.polyfit([x["length"] for x in individual],[x["rate"] for x in individual],1).tolist())
    training_paths=[OUT/"cases"/(c["case_id"]+".json") for c in cases() if c["role"]=="early"]
    static_paths=sorted((OUT/"static").glob("*.json"))
    value=dict(locked_utc=datetime.now(timezone.utc).isoformat(),script_sha256=SOURCE_SHA,config_sha256=audit.sha256(OUT/"config.json"),
               models=["global_gap_envelope","mode_weighted_envelope","early_log_graph_forecast"],empirical_model=model,training_fits=fits,
               inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=audit.sha256(p)) for p in [*training_paths,*static_paths]],
               policy="never change predictors after new late/size/weak-skin truth is generated",
               empirical_definition="OLS log(kappa)=intercept-rate*t at 5,7.5,10; OLS intercept/rate vs L at 16,24,32; linear g interpolation",
               physical_definition="static Hamiltonian modes and initial overlap only; global norm or weighted Frobenius spectral envelope; no future M or SVD",
               entropy_policy="physical envelopes imply rigorous mathematical entropy bounds; empirical entropy budget is only a proxy",
               prior_late_data="four t=15 L16/32 endpoint-g conditions previously seen; they are not fresh time holdouts",
               scope="bounds assume accurate static coefficients; numerical error budgets are empirical, not interval arithmetic")
    write_json(path,value)
    predictions=[]
    time_predictions=[]
    for length,skin in sorted({(c["length"],c["skin"]) for c in cases()}):
        static=json.loads((OUT/"static"/f"L{length}_g{str(skin).replace('.', 'p')}.json").read_text())
        for initial in INITIALS:
            for name in value["models"]:
                def evaluate(duration):
                    if name=="early_log_graph_forecast":
                        return empirical_value(model[initial],length,skin,duration)
                    return physical_envelope(static,initial,duration,name=="mode_weighted_envelope")
                for case in [c for c in cases() if c["length"]==length and c["skin"]==skin]:
                    result=evaluate(case["physical_time"])
                    d=result.get("distance_bound",result.get("distance_prediction"))
                    ent=result.get("entropy_bound",result.get("entropy_budget_proxy"))
                    predictions.append({**case,"initial":initial,"model":name,**result,
                                        "predicted_projector_pass":d<=0.05,"predicted_joint_budget_pass":d<=0.05 and ent<=0.05})
                dtarget=optimize.brentq(lambda d:length//2*binary_entropy(d)-0.05,1e-15,0.5,xtol=1e-14)
                def root(target):
                    function=lambda t:(evaluate(t).get("distance_bound",evaluate(t).get("distance_prediction"))-target)
                    return 0.0 if function(0)<=0 else float(optimize.brentq(function,0,500))
                time_predictions.append(dict(length=length,skin=skin,initial=initial,model=name,
                                             projector_time_forecast=root(0.05),joint_budget_time_forecast=root(min(0.05,dtarget))))
    write_csv("locked_predictions.csv",predictions)
    write_csv("locked_selection_time_forecasts.csv",time_predictions)
    write_json(OUT/"prediction_lock_hash.json",dict(sha256=audit.sha256(path),predictions_sha256=audit.sha256(OUT/"locked_predictions.csv"),
                                                  time_forecasts_sha256=audit.sha256(OUT/"locked_selection_time_forecasts.csv")))
    emit(dict(stage="lock",models=3,predictions=len(predictions),time_forecasts=len(time_predictions),locked_utc=value["locked_utc"]))


def check_lock():
    hashes=json.loads((OUT/"prediction_lock_hash.json").read_text())
    assert audit.sha256(OUT/"prediction_lock.json")==hashes["sha256"]
    assert audit.sha256(OUT/"locked_predictions.csv")==hashes["predictions_sha256"]
    assert audit.sha256(OUT/"locked_selection_time_forecasts.csv")==hashes["time_forecasts_sha256"]


def save_general_initials():
    from mietf_skin.ssh import ssh_no_click_hamiltonian

    for length in (16,32):
        path=OUT/"initial_states"/f"L{length}.npz"
        if path.exists():
            assert audit.sha256(path)==json.loads(path.with_suffix(".json").read_text())["sha256"]
            continue
        h=ssh_no_click_hamiltonian(length//2,0.5,1.0,0.0,0.0,"open")
        _,vectors=linalg.eigh(h)
        values={"hermitian_ground":vectors[:,:length//2]}
        for seed in range(20):
            rng=np.random.default_rng(seed)
            matrix=(rng.normal(size=(length,length//2))+1j*rng.normal(size=(length,length//2)))/math.sqrt(2)
            values[f"random_{seed:02d}"]=linalg.qr(matrix,mode="economic")[0]
        np.savez_compressed(path,**values)
        write_json(path.with_suffix(".json"),dict(length=length,sha256=audit.sha256(path),definition=configuration()["general"],
                                                 max_orthogonality=max(audit.opnorm(q.conj().T@q-np.eye(length//2)) for q in values.values())))


def calibrate():
    case=make_case(16,0.0,5.0,"calibration")
    first=reference(case,80)
    second=reference(case,120)
    change=convergence(first,second)
    old=json.loads((PRIOR/"cases"/(case["case_id"]+".json")).read_text())
    max_s=max(abs(a["entropy"]-b["entropy"]) for a,b in zip(second[0],old["rows"]))
    with np.load(PRIOR/"cases"/(case["case_id"]+".npz")) as archive:
        max_p=max(audit.projector_error(second[1]["Q_"+initial],archive["Q_"+initial]) for initial in INITIALS)
    assert max_s<1e-10 and max_p<1e-10 and accepted_precision(change)
    write_json(OUT/"frame_calibration.json",dict(case=case,reference_convergence=change,prior_entropy_error=max_s,prior_projector_error=max_p,
                                                local_unitary_equivalence="passed",script_sha256=SOURCE_SHA))
    emit(dict(stage="calibration",status="passed",prior_entropy_error=max_s,prior_projector_error=max_p))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("calibrate","early","lock","validate","general"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    assert 1<=args.workers<=8
    prepare()
    tick=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="prl_p2_baseline_") as directory:
        with zipfile.ZipFile(ROOT/"data/prl_p0_precision/baseline_snapshot.zip") as archive:
            for member in archive.namelist():
                if member.startswith("src/"):
                    archive.extract(member,directory)
        source_directory=str(Path(directory)/"src")
        sys.path.insert(0,source_directory)
        with threadpool_limits(limits=1):
            if args.stage=="calibrate":
                calibrate()
            elif args.stage=="early":
                assert (OUT/"frame_calibration.json").exists()
                run_parallel(run_case,[c for c in cases() if c["role"]=="early"],args.workers,source_directory)
            elif args.stage=="lock":
                families=sorted({(c["length"],c["skin"]) for c in cases()})
                run_parallel(run_static,families,args.workers,source_directory)
                lock_predictions()
            elif args.stage=="validate":
                check_lock()
                run_parallel(run_case,[c for c in cases() if c["role"]!="early"],args.workers,source_directory)
            else:
                check_lock()
                save_general_initials()
                general_cases=[make_case(length,skin,t,"general") for length in (16,32) for skin in (0.0,0.25) for t in (5.0,15.0)]
                run_parallel(run_case,general_cases,args.workers,source_directory,general=True)
    info=audit.environment(args.stage)
    info.update(script_sha256=SOURCE_SHA,config_sha256=audit.sha256(OUT/"config.json"),workers=args.workers,
                elapsed_seconds=time.perf_counter()-tick,scope="local Windows execution; worker times saved per receipt")
    write_json(OUT/f"environment_{args.stage}.json",info)


if __name__=="__main__":
    main()
