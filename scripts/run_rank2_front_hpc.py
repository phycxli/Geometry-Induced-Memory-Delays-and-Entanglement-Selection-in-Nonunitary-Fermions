"""Round nine: signed two-mode geometry and independently frozen fine times."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import multiprocessing
import os
import time
import traceback
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import run_fine_front_hpc as previous

old = previous.old
mp, np, ROOT = old.mp, old.np, old.ROOT
PRIOR = previous.OUT
OUT = ROOT / "data/prl_p3_rank2_front"
CONFIG = OUT / "config.json"
SOURCE = Path(__file__)
PREVIOUS_SOURCE = previous.SOURCE
read_json, write_json, write_csv = old.read_json, old.write_json, old.write_csv
read_gzip, write_gzip, sha256, emit = old.read_gzip, old.write_gzip, old.audit.sha256, old.emit
ORIGINAL_CHECK = previous.check_lock
EXCLUDED = {"deployment.tar.gz", "final_sync.tar.gz", "results.tar.gz", "operations.jsonl", "local_inventory.json"}


def config():
    return read_json(CONFIG)


def setup():
    previous.OUT, previous.CONFIG, previous.SOURCE = OUT, CONFIG, SOURCE
    previous.check_lock = check_lock
    previous.setup()
    for folder in ("roots", "response"):
        (OUT / folder).mkdir(exist_ok=True)


def initialize():
    setup()
    if CONFIG.exists():
        assert read_json(OUT / "bootstrap_lock.json")["source_sha256"] == sha256(SOURCE)
        return dict(status="reused")
    history = {ROOT / r["path"] for r in old.audit.read_csv(PRIOR / "input_manifest.csv")}
    history.update(p for folder in (PRIOR, ROOT / "figures/prl_p3_fine_front") for p in folder.rglob("*")
                   if p.is_file() and p.name not in EXCLUDED)
    history.update((PREVIOUS_SOURCE, ROOT / "scripts/run_fine_front_batch.py",
                    ROOT / "scripts/analyze_fine_front_hpc.py",
                    ROOT / "research_doc/04_历史研究/11_精细交叉与平移记忆.md"))
    write_csv(OUT / "input_manifest.csv", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                                               sha256=sha256(p)) for p in sorted(history)])
    families = read_json(PRIOR / "config.json")["families"]
    value = dict(version=1, date="2026-10-04", families=families, cases=[], pilots=[],
                 objective="explain rank1 threshold failures and validate a signed rank2 candidate on unseen times",
                 ranks=list(old.RANKS), fixed_candidate_rank=2, model_projector_target=1e-5,
                 selection_tolerances=dict(projector=.05, absolute_entropy=.05),
                 memory_tolerances=dict(projector=.1, absolute_entropy=.1),
                 root_abs_error_gate=.02, local_grid_gap=.02,
                 grid_offsets=["-0.035", "-0.015", "0.005", "0.025", "0.045"],
                 root_policy="local rank2 branch around the historical rank1 root; not first passage",
                 batch_size=4, workers=4,
                 scope="real-gauge gamma2 OBC SSH; no second-model, asymptotic, Born or experimental claim")
    write_json(CONFIG, value)
    write_json(OUT / "bootstrap_config.json", value)
    kernels = [SOURCE, PREVIOUS_SOURCE, *previous.KERNELS[1:]]
    write_json(OUT / "kernel_manifest.json", [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p))
                                             for p in sorted(set(kernels))])
    write_json(OUT / "bootstrap_lock.json", dict(locked_utc=old.now(), source_sha256=sha256(SOURCE),
               initial_config_sha256=sha256(CONFIG), input_manifest_sha256=sha256(OUT / "input_manifest.csv"),
               grid_offsets=value["grid_offsets"], root_abs_error_gate=.02, batch_size=4, workers=4,
               policy="all eighth-round truth is development; keep thresholds and freeze new times before truth"))
    return dict(status="initialized", families=len(families), historical_inputs=len(history))


def check_source():
    assert read_json(OUT / "bootstrap_lock.json")["source_sha256"] == sha256(SOURCE)
    assert read_json(OUT / "bootstrap_lock.json")["initial_config_sha256"] == sha256(OUT / "bootstrap_config.json")
    for row in read_json(OUT / "kernel_manifest.json"):
        assert sha256(ROOT / row["path"]) == row["sha256"], row["path"]


def check_lock():
    ORIGINAL_CHECK()
    check_source()
    assert sha256(CONFIG) == read_json(OUT / "prediction_lock.json")["config_sha256"]


def rank2_geometry(ctx, duration, supplied=None):
    n = len(ctx["mu"])
    if supplied is None:
        decay = [mp.exp(-x * duration) for x in ctx["mu"]]
        a = mp.matrix([[decay[i] * ctx["u"][i, j] * mp.sqrt(ctx["sig"][j]) for j in range(2)]
                       for i in range(n)])
        bt = mp.matrix([[mp.sqrt(ctx["sig"][i]) * ctx["vh"][i, j] * decay[j] for j in range(n)]
                        for i in range(2)])
        v = bt * ctx["rinv"]
    else:
        a, v = supplied["a"][:, :2], supplied["v"][:2, :]
    b = ctx["matrices"]["Bperp"] * a
    h = mp.eye(2) + v * (ctx["matrices"]["Ccross"] * a)
    inverse = mp.inverse(h)
    c = b.T * b
    k = inverse * (v * v.T) * inverse.T
    trace = sum((c * k)[i, i] for i in range(2))
    determinant = mp.det(c) * mp.det(k)
    discriminant = trace * trace - 4 * determinant
    assert trace >= 0 and discriminant >= 0, (trace, discriminant)
    maximum = (trace + mp.sqrt(discriminant)) / 2
    distance = mp.sqrt(maximum / (1 + maximum))
    diagonal_h = mp.diag([1 / h[i, i] for i in range(2)])
    kd = diagonal_h * (v * v.T) * diagonal_h.T
    td = sum((c * kd)[i, i] for i in range(2))
    dd = td * td - 4 * mp.det(c) * mp.det(kd)
    assert dd >= 0
    maximum_d = (td + mp.sqrt(dd)) / 2
    no_mixing = max(c[i, i] * k[i, i] for i in range(2))
    values = dict(distance_candidate=distance, kappa_candidate=mp.sqrt(maximum), det_h=mp.det(h),
                  gram_trace=trace, gram_determinant=determinant,
                  diagonal_h_candidate=mp.sqrt(maximum_d / (1 + maximum_d)),
                  no_gram_mixing_candidate=mp.sqrt(no_mixing / (1 + no_mixing)))
    return values, b, inverse, v


def development_job(family):
    check_source()
    path = OUT / "development" / (family["key"] + ".json")
    if path.exists():
        assert read_json(path)["source_sha256"] == sha256(SOURCE)
        return dict(status="reused", family=family["key"])
    tick = time.perf_counter()
    static = read_gzip(old.OUT / "static" / (family["key"] + ".raw.json.gz"))[-1]
    observed = [r for r in old.audit.read_csv(PRIOR / "actual_entanglement.csv") if r["family_key"] == family["key"]]
    rows, checks = [], []
    with mp.workdps(static["dps"]):
        contexts = old.context(static)
        selected = {min((r for r in observed if r["initial"] == name),
                        key=lambda r: abs(float(r["distance_to_W"]) - .05))["case_id"] + "|" + name
                    for name in old.initials(family["length"])[1:]}
        selected.update(r["case_id"] + "|" + r["initial"] for r in observed if r["rank1_misclassified"] == "True")
        for actual in observed:
            ctx, duration = contexts[actual["initial"]], mp.mpf(actual["physical_time"])
            started = time.perf_counter()
            values, b, inverse, v = rank2_geometry(ctx, duration)
            seconds = time.perf_counter() - started
            d1, d2, truth = mp.mpf(actual["rank1_distance_candidate"]), values["distance_candidate"], mp.mpf(actual["distance_to_W"])
            row = dict(family_key=family["key"], length=family["length"], case_id=actual["case_id"], initial=actual["initial"],
                       physical_time=actual["physical_time"], actual_distance=float(truth), rank1_distance=float(d1),
                       rank2_distance=float(d2), rank1_error=float(abs(d1 - truth)), rank2_error=float(abs(d2 - truth)),
                       rank1_misclassified=(d1 <= mp.mpf(".05")) != (truth <= mp.mpf(".05")),
                       rank2_misclassified=(d2 <= mp.mpf(".05")) != (truth <= mp.mpf(".05")),
                       correction=float(d2 - d1), corrected_fraction=float((d2 - d1) / (truth - d1)) if truth != d1 else None,
                       det_h=float(values["det_h"]), diagonal_h_distance=float(values["diagonal_h_candidate"]),
                       no_gram_mixing_distance=float(values["no_gram_mixing_candidate"]), compact_seconds=seconds)
            rows.append(row)
            if actual["case_id"] + "|" + actual["initial"] in selected:
                started = time.perf_counter()
                full = old.previous.mp_opnorm(b * inverse * v)
                full_distance = full / mp.sqrt(1 + full * full)
                full_seconds = time.perf_counter() - started
                error = abs(d2 - full_distance)
                assert error < mp.mpf("1e-70"), (family["key"], actual["initial"], error)
                checks.append(dict(case_id=actual["case_id"], initial=actual["initial"], error_mp=str(error),
                                   compact_seconds=seconds, full_seconds=full_seconds, speedup=full_seconds / seconds))
    write_json(path, dict(status="passed", source_sha256=sha256(SOURCE), family=family, rows=rows, checks=checks,
               precision_digits=static["dps"], started_policy="observed eighth-round values are development only",
               actual_csv_sha256=sha256(PRIOR / "actual_entanglement.csv"), completed_utc=old.now(), seconds=time.perf_counter() - tick))
    return dict(status="development_completed", family=family["key"], states=len(rows), kernel_checks=len(checks))


def local_root(ctx, center):
    lo, hi = mp.mpf(str(center)) - mp.mpf(".25"), mp.mpf(str(center)) + mp.mpf(".25")
    f = lambda t: rank2_geometry(ctx, t)[0]["distance_candidate"]
    assert f(lo) > mp.mpf(".05") and f(hi) < mp.mpf(".05"), "rank2 branch not bracketed"
    for _ in range(38):
        middle = (lo + hi) / 2
        if f(middle) > mp.mpf(".05"):
            lo = middle
        else:
            hi = middle
    return (lo + hi) / 2, lo, hi


def roots_job(family):
    check_source()
    development = read_json(OUT / "development" / (family["key"] + ".json"))
    assert development["status"] == "passed"
    path = OUT / "roots" / (family["key"] + ".json")
    if path.exists():
        assert read_json(path)["source_sha256"] == sha256(SOURCE)
        return dict(status="reused", family=family["key"])
    historical = [r for r in read_json(PRIOR / "front_hypothesis_lock.json")["candidates"] if r["family_key"] == family["key"]]
    statics = read_gzip(old.OUT / "static" / (family["key"] + ".raw.json.gz"))[-2:]
    evaluations = []
    for static in statics:
        with mp.workdps(static["dps"]):
            contexts = old.context(static)
            evaluations.append({r["initial"]: local_root(contexts[r["initial"]], r["candidate_time"]) for r in historical})
    rows = []
    with mp.workdps(360):
        for hypothesis in historical:
            name = hypothesis["initial"]
            value, lo, hi = evaluations[-1][name]
            change = abs(value - evaluations[0][name][0])
            assert change < mp.mpf("1e-8")
            rows.append(dict(family_key=family["key"], length=family["length"], group=hypothesis["group"], initial=name,
                             rank1_candidate_time=hypothesis["candidate_time"], rank2_candidate_time=float(value),
                             root_lower_mp=str(lo), root_upper_mp=str(hi), precision_change_mp=str(change)))
    write_json(path, dict(status="passed", source_sha256=sha256(SOURCE), rows=rows,
                         precision_digits=[s["dps"] for s in statics], completed_utc=old.now(),
                         policy="static local crossing candidate; no new truth or first-time guarantee"))
    return dict(status="roots_completed", family=family["key"], roots=len(rows))


def plan():
    check_source()
    if (OUT / "front_hypothesis_lock.json").exists():
        assert sha256(CONFIG) == read_json(OUT / "front_hypothesis_lock.json")["config_sha256"]
        return dict(status="reused", conditions=len(config()["cases"]))
    assert not list((OUT / "cases").glob("*.json"))
    cfg, cases, candidates, checks = config(), [], [], []
    seen = {(c["family_key"], Decimal(str(c["physical_time"]))) for folder in (old.OUT, PRIOR)
            for c in read_json(folder / "config.json")["cases"]}
    for family in cfg["families"]:
        roots = read_json(OUT / "roots" / (family["key"] + ".json"))
        development = read_json(OUT / "development" / (family["key"] + ".json"))
        assert roots["status"] == development["status"] == "passed"
        checks.extend(development["checks"])
        for group in ("edge", "translated"):
            names = ["charge_density_wave", "left", "right"] if group == "edge" else ["charge_density_wave", f"block_cell{family['length']//16}", f"block_cell{3*family['length']//16}"]
            times = {}
            for row in (r for r in roots["rows"] if r["group"] == group):
                value = Decimal(str(row["rank2_candidate_time"]))
                center = (value / Decimal(".02")).to_integral_value(rounding=ROUND_FLOOR) * Decimal(".02")
                planned = [center + Decimal(x) for x in cfg["grid_offsets"]]
                assert all((family["key"], t) not in seen for t in planned), "time was already observed"
                candidates.append({**row, "candidate_time": row["rank2_candidate_time"], "times": [str(t) for t in planned],
                                   "root_abs_error_gate": cfg["root_abs_error_gate"], "maximum_local_grid_gap": .02})
                for duration in planned:
                    times.setdefault(duration, []).append(row["initial"])
            for duration, targets in sorted(times.items()):
                identifier = family["key"] + "_" + group + "_t" + str(duration).replace(".", "p")
                cases.append(dict(**family, family_key=family["key"], case_id=identifier, group=group,
                                  t1=.5, t2=1., boundary="open", skin=float(old.skin_mp(family)),
                                  physical_time=str(duration), time_fraction=str(duration/family["length"]),
                                  dt=.005, steps=int(duration/Decimal(".005")), initials=names,
                                  pairs=previous.pair_list(names), target_initials=sorted(set(targets)),
                                  role="new_time_validation", digits=[200, 240, 280, 320]))
    cfg["cases"] = cases
    write_json(CONFIG, cfg)
    write_json(OUT / "development/kernel_checks.json", dict(status="passed", source_sha256=sha256(SOURCE), rows=checks,
                                                            policy="2x2 expression versus full rectangular graph SVD on development"))
    write_json(OUT / "front_hypothesis_lock.json", dict(locked_utc=old.now(), source_sha256=sha256(SOURCE),
               config_sha256=sha256(CONFIG), candidates=candidates, states=sum(len(c["initials"]) for c in cases),
               policy="signed rank2 formula and new time grid frozen before truth; rank1 is a matched baseline"))
    return dict(status="new_times_frozen", conditions=len(cases), states=sum(len(c["initials"]) for c in cases), candidates=len(candidates))


def fixed_prediction(ctx, duration):
    tc = old.time_context(ctx, duration)
    values, q, factors = previous.at_rank(tc, 2)
    geometry = rank2_geometry(ctx, duration, tc)[0]
    assert abs(values["distance_to_G"] - geometry["distance_candidate"]) < mp.mpf("1e-70")
    entropy = old.previous.correlation_entropy(q)[0]
    static = ctx["static"]
    z = mp.mpf(static["q"]) * mp.exp(-2 * min(ctx["mu"]) * duration)
    p = mp.mpf(static["p"])
    ew = min(mp.mpf(1), z / (p - z)) if z < p else mp.mpf(1)
    eps, distance = values["projector_remainder"], values["distance_to_G"]
    error = abs(entropy - mp.mpf(static["gain_entropy"]))
    budget = old.base.entropy_budget(eps, static["length"] // 2) + old.base.entropy_budget(ew, static["length"] // 2)
    values.update(geometry, entropy=entropy, output_to_G_bound=ew, entropy_budget=budget,
                  distance_lower=max(0, distance - eps - ew), distance_upper=min(1, distance + eps + ew),
                  entropy_error_lower=max(0, error - budget), entropy_error_upper=error + budget)
    return values, q, [], {}, factors


def prediction_job(family):
    check_source()
    path = OUT / "predictions" / (family["key"] + ".json")
    if path.exists() and read_json(path).get("fixed_rank2_complete"):
        assert read_json(path)["source_sha256"] == sha256(SOURCE)
        return dict(status="reused", family=family["key"])
    previous.prediction_job(family)
    adaptive = read_json(path)
    indexed = {(r["case_id"], r["initial"]): r for r in adaptive["rows"]}
    pairs = {(r["case_id"], r["initial_a"], r["initial_b"]): r for r in adaptive["pairs"]}
    statics = read_gzip(old.OUT / "static" / (family["key"] + ".raw.json.gz"))[-2:]
    contexts = []
    for static in statics:
        with mp.workdps(static["dps"]):
            contexts.append(old.context(static))
    raw_rows, raw_pairs = [], []
    for case in (c for c in config()["cases"] if c["family_key"] == family["key"]):
        local = []
        for static, ctx in zip(statics, contexts):
            with mp.workdps(static["dps"]):
                local.append({name: fixed_prediction(ctx[name], mp.mpf(case["physical_time"])) for name in case["initials"]})
        with mp.workdps(360):
            for name in case["initials"]:
                low, high = local[0][name][0], local[1][name][0]
                change = max(abs(low[k] - high[k]) for k in high)
                assert change < mp.mpf("1e-10"), (case["case_id"], name, change)
                row = indexed[case["case_id"], name]
                row.update({"rank2_" + k: float(v) for k, v in high.items()})
                row.update(rank2_precision_change=float(change),
                           rank2_target_passed=high["projector_remainder"] <= mp.mpf("1e-5"),
                           rank2_guaranteed_selected=high["distance_upper"] <= mp.mpf(".05") and high["entropy_error_upper"] <= mp.mpf(".05"),
                           rank2_guaranteed_unselected=high["distance_lower"] > mp.mpf(".05") or high["entropy_error_lower"] > mp.mpf(".05"))
                raw_rows.append(dict(case_id=case["case_id"], initial=name, Q=old.p1.encode_matrix(local[1][name][1]),
                                     values={k: str(v) for k, v in high.items()}))
            for na, nb in case["pairs"]:
                values = []
                for static, data in zip(statics, local):
                    with mp.workdps(static["dps"]):
                        values.append(previous.bound_pair(data[na], data[nb]))
                change = max(abs(values[0][k] - values[1][k]) for k in values[0] if k != "compact_pair_seconds")
                assert change < mp.mpf("1e-10")
                pair = pairs[case["case_id"], na, nb]
                pair.update({"rank2_" + k: float(v) for k, v in values[-1].items()})
                pair.update(rank2_guaranteed_memory=values[-1]["memory_lower"] > mp.mpf(".1"),
                            rank2_guaranteed_entropy_memory=values[-1]["entropy_memory_lower"] > mp.mpf(".1"))
                raw_pairs.append(dict(case_id=case["case_id"], initial_a=na, initial_b=nb,
                                      values={k: str(v) for k, v in values[-1].items()}))
    write_gzip(path.with_suffix(".rank2.raw.json.gz"), dict(rows=raw_rows, pairs=raw_pairs))
    adaptive.update(fixed_rank2_complete=True, fixed_rank2_source_sha256=sha256(SOURCE), completed_utc=old.now())
    write_json(path, adaptive)
    return dict(status="predictions_complete", family=family["key"], states=len(adaptive["rows"]))


def lock():
    check_source()
    if (OUT / "prediction_lock.json").exists():
        check_lock()
        return dict(status="reused")
    for family in config()["families"]:
        assert read_json(OUT / "predictions" / (family["key"] + ".json"))["fixed_rank2_complete"]
    previous.lock()
    path = OUT / "prediction_lock.json"
    value = read_json(path)
    extra = [OUT / "bootstrap_lock.json", OUT / "bootstrap_config.json", OUT / "input_manifest.csv"]
    extra.extend(p for directory in ("roots", "development") for p in (OUT / directory).glob("*.json"))
    inputs = {r["path"]: r for r in value["inputs"]}
    inputs.update({p.relative_to(ROOT).as_posix(): dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p)) for p in extra})
    value["inputs"] = [inputs[k] for k in sorted(inputs)]
    value["policy"] = "rank1, signed rank2 and adaptive certificates locked before every new reference"
    write_json(path, value)
    write_json(OUT / "prediction_lock_hash.json", {name: sha256(OUT / name) for name in
               ("prediction_lock.json", "locked_predictions.csv", "locked_memory_predictions.csv")})
    return dict(status="predictions_locked", states=value["states"], pairs=value["pairs"])


def audit_case(case):
    path = OUT / "audits" / (case["case_id"] + ".json")
    if path.exists() and read_json(path).get("fixed_rank2_checks_complete"):
        return dict(status="reused", case_id=case["case_id"])
    previous.audit_case(case)
    source = OUT / "cases" / path.name
    truth = read_gzip(source.with_suffix(".raw.json.gz"))[-1]
    prediction = OUT / "predictions" / (case["family_key"] + ".rank2.raw.json.gz")
    reduced = read_gzip(prediction)
    states = {r["initial"]: r for r in reduced["rows"] if r["case_id"] == case["case_id"]}
    checks, pair_checks = [], []
    with mp.workdps(max(280, truth["dps"])):
        allowance = mp.mpf("1e-35")
        for name in case["initials"]:
            actual, predicted = truth["states"][name], states[name]
            values = {k: mp.mpf(v) for k, v in predicted["values"].items()}
            q, qr = old.spatial.decode(actual["Q"]), old.spatial.decode(predicted["Q"])
            error = old.pair_distance(q, qr)
            entropy_error = abs(mp.mpf(actual["entropy"]) - values["entropy"])
            actual_distance = mp.mpf(actual["distance_to_W"])
            actual_entropy_error = abs(mp.mpf(actual["entropy"]) - mp.mpf(truth["W_entropy"]))
            gates = dict(remainder_covered=error <= values["projector_remainder"] + allowance,
                         entropy_remainder_covered=entropy_error <= old.base.entropy_budget(values["projector_remainder"], q.cols) + allowance,
                         distance_interval_covered=values["distance_lower"] - allowance <= actual_distance <= values["distance_upper"] + allowance,
                         entropy_interval_covered=values["entropy_error_lower"] - allowance <= actual_entropy_error <= values["entropy_error_upper"] + allowance,
                         bases_orthogonal=old.p1.mp_norm(qr.T * qr - mp.eye(qr.cols)) <= allowance)
            assert all(gates.values()), (case["case_id"], name, gates)
            checks.append(dict(case_id=case["case_id"], initial=name, **gates, projector_error_mp=str(error),
                               entropy_error_mp=str(entropy_error), remainder_mp=str(values["projector_remainder"])))
        for pair in (r for r in reduced["pairs"] if r["case_id"] == case["case_id"]):
            na, nb = pair["initial_a"], pair["initial_b"]
            va, vb = truth["states"][na], truth["states"][nb]
            distance = old.pair_distance(old.spatial.decode(va["Q"]), old.spatial.decode(vb["Q"]))
            entropy = abs(mp.mpf(va["entropy"]) - mp.mpf(vb["entropy"]))
            values = {k: mp.mpf(v) for k, v in pair["values"].items()}
            gates = dict(projector_interval_covered=values["memory_lower"] - allowance <= distance <= values["memory_upper"] + allowance,
                         entropy_interval_covered=values["entropy_memory_lower"] - allowance <= entropy <= values["entropy_memory_upper"] + allowance)
            assert all(gates.values()), (case["case_id"], na, nb, gates)
            pair_checks.append(dict(case_id=case["case_id"], initial_a=na, initial_b=nb, **gates))
    certificate = read_json(path)
    certificate.update(rank2_rows=checks, rank2_pairs=pair_checks, fixed_rank2_checks_complete=True,
                       rank2_prediction_raw_sha256=sha256(prediction), verified_utc=old.now())
    write_json(path, certificate)
    return dict(status="both_models_audited", case_id=case["case_id"], states=len(checks), pairs=len(pair_checks))


def execute(stage, index):
    setup()
    check_source()
    info, started = old.environment(stage), time.perf_counter()
    snapshot = OUT / "code" / (sha256(SOURCE).lower() + ".py")
    if not snapshot.exists():
        snapshot.write_bytes(SOURCE.read_bytes())
    key = str(index)
    try:
        with old.threadpool_limits(limits=1):
            if stage in ("development", "roots", "prediction"):
                family = config()["families"][index]
                key = family["key"]
                value = {"development": development_job, "roots": roots_job, "prediction": prediction_job}[stage](family)
            elif stage == "plan":
                value = plan()
            elif stage == "lock":
                value = lock()
            else:
                case = config()["cases"][index]
                key = case["case_id"]
                value = previous.reference_job(case) if stage == "reference" else audit_case(case)
        emit(dict(stage=stage, **value))
        return value
    except Exception as error:
        write_json(OUT / "failures" / (stage + "_" + key + "_" + old.now().replace(":", "-") + ".json"),
                   dict(stage=stage, key=key, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                        source_sha256=sha256(SOURCE), utc=old.now(), environment=info))
        raise
    finally:
        info.update(seconds=time.perf_counter() - started, source_sha256=sha256(SOURCE), config_sha256=sha256(CONFIG))
        write_json(OUT / "hpc" / ("environment_" + stage + "_" + key + ".json"), info)


def batch_worker(stage, index, cpu):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    execute(stage, index)
    return dict(index=index, affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)


def batch(stage, index):
    setup()
    check_lock()
    indices = list(range(index * 4, min(index * 4 + 4, len(config()["cases"]))))
    cpus = sorted(os.sched_getaffinity(0))
    assert indices and len(cpus) >= len(indices)
    started = time.perf_counter()
    receipt = dict(stage=stage, batch_index=index, case_indices=indices, started_utc=old.now(),
                   source_sha256=sha256(SOURCE), affinity=cpus, workers=len(indices),
                   job_id=os.environ.get("SLURM_JOB_ID"), node=os.environ.get("SLURM_JOB_NODELIST"))
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(indices), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(batch_worker, stage, case, cpu) for case, cpu in zip(indices, cpus)]
            receipt["rows"] = [f.result() for f in futures]
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        receipt.update(completed_utc=old.now(), seconds=time.perf_counter() - started)
        write_json(OUT / "hpc" / f"batch_{stage}_{index}.json", receipt)
    emit(dict(stage="batch_" + stage, batch_index=index, conditions=len(indices), status=receipt["status"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("initialize", "development", "roots", "plan", "prediction", "lock", "reference", "audit", "batch_reference", "batch_audit"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "initialize":
        emit(initialize())
    elif args.stage.startswith("batch_"):
        batch(args.stage[6:], args.index)
    else:
        execute(args.stage, args.index)


if __name__ == "__main__":
    main()
