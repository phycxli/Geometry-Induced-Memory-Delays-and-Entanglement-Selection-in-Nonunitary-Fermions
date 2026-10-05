"""Seventh round: frozen selection fronts and memory controls on Slurm CPUs."""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import run_directional_selection_prediction as prior

base, spatial, previous, audit, p1 = prior.base, prior.spatial, prior.previous, prior.audit, prior.p1
ROOT = prior.ROOT
OUT = ROOT / "data/prl_p3_selection_front"
SOURCE = Path(__file__)
DIGITS = (200, 240, 280, 320)
RANKS = prior.RANKS
CONFIG = OUT / "config.json"
ORIGINAL_GENERATOR = previous.generator
ORIGINAL_SOLVE = spatial.right_solve


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_gzip(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def emit(value):
    print(json.dumps({"utc": now(), **value}, ensure_ascii=True, allow_nan=False), flush=True)


def skin_mp(family):
    value = mp.mpf(str(family["skin_value"]))
    return value / family["length"] if family["skin_mode"] == "gL" else value


def family_key(length, mode, value):
    return f"L{length}_{mode}{value}".replace(".", "p").replace("-", "m")


def initials(length):
    return ["charge_density_wave", "left", "right", f"block_cell{length // 16}", f"block_cell{3 * length // 16}"]


def memory_pairs(length):
    names = initials(length)
    return [[names[0], x] for x in names[1:]] + [[names[1], names[2]], [names[3], names[4]]]


def candidate_configuration():
    families, cases = [], []
    for length in (96, 112, 128, 144):
        protocols = [("g", "0"), ("g", "0.05"), ("g", "0.25")] if length < 128 else [("gL", "0"), ("gL", "2"), ("gL", "4"), ("g", "0.25")]
        fractions = ("0.525", "0.55", "0.575", "0.65") if length < 128 else ("0.45", "0.5", "0.525", "0.55", "0.575", "0.6", "0.65")
        for mode, value in protocols:
            key = family_key(length, mode, value)
            family = dict(key=key, length=length, gamma=2.0, skin_mode=mode, skin_value=value)
            families.append(family)
            for fraction in fractions:
                duration = str(Decimal(fraction) * length)
                case = dict(**family, family_key=key, case_id=key + "_tau" + fraction.replace(".", "p"),
                            t1=0.5, t2=1.0, boundary="open", skin=float(value) / length if mode == "gL" else float(value),
                            time_fraction=fraction, physical_time=duration, dt=0.05, steps=round(float(duration) / .05),
                            initials=initials(length), pairs=memory_pairs(length), role="development_window" if length < 128 else "new_size")
                if (length, mode, value, fraction) in ((128, "gL", "0", "0.5"), (144, "g", "0.25", "0.55")):
                    case["role"] = "precision_cost_pilot"
                cases.append(case)
    return dict(version=1, date="2026-10-04", objective="bounded sampled selection fronts, weak-skin crossover, direct projector and entropy memory",
                families=families, cases=cases, ranks=list(RANKS), precision_digits=list(DIGITS), model_projector_target=1e-5,
                selection_tolerances=dict(projector=.05, absolute_entropy=.05), memory_tolerances=dict(projector=.1, absolute_entropy=.1),
                reference_goals=dict(projector=1e-10, entropy=1e-10, relative_scalar=1e-8),
                pairs_policy="CDW against all four blocks, left versus right, and the two translated blocks, frozen before truth",
                translated_policy="complete-cell blocks start at floor(L/16) and floor(3L/16) cells; coordinate products fixed across g,t",
                rank_policy="first frozen rank passing min(directional,transverse)<=1e-5; otherwise rank16 and retain uncertainty",
                numerical_policy="two arbitrary-precision references; no double-precision reference or interval-arithmetic claim",
                pilot_policy="two pilot conditions are development and excluded from new-size validation counts; their other times remain new",
                spatial_policy="finite reflected Gram at g=0, exact diagonal-congruence inverse-trace transfer to every g; scalar size extrapolation separately falsifiable",
                production_gate="both independent 200/240-digit pilot conditions must converge before nonpilot reference jobs",
                prior_plan_sha256=audit.sha256(prior.OUT / "next_round_plan.json"),
                scope="gamma2 static local OBC SSH family; no second-model, continuous cutoff, asymptotic theorem or Born-trajectory claim")


def size_candidate():
    training, predictions = [], []
    for initial, skin in itertools.product(("left", "right"), (0., .25)):
        rows = {}
        for folder in (spatial.OUT, base.OUT, prior.OUT):
            for path in (folder / "static").glob("*.json"):
                value = read_json(path)
                if value["gamma"] == 2. and value["skin"] == skin and value["length"] >= 64:
                    with mp.workdps(240):
                        delta = mp.mpf(value["states"][initial]["cross_block_min"])
                        rows[value["length"]] = dict(length=value["length"], skin=skin, initial=initial,
                                                    log_delta=float(mp.log(delta)), delta_mp=str(delta), source=path.relative_to(ROOT).as_posix(), sha256=audit.sha256(path))
        ordered = [rows[x] for x in sorted(rows)]
        training.extend(ordered)
        sizes = np.array([r["length"] for r in ordered], float)
        logs = np.array([r["log_delta"] for r in ordered])
        design = np.column_stack((sizes, np.log(sizes), np.ones(len(sizes))))
        coefficients = np.linalg.lstsq(design, logs, rcond=None)[0]
        linear = np.polyfit(sizes[-3:], logs[-3:], 1)
        for length in (128, 144):
            value = float(np.array([length, np.log(length), 1]) @ coefficients)
            alternative = float(np.polyval(linear, length))
            predictions.append(dict(length=length, skin=skin, initial=initial, predicted_log_delta=value,
                                    alternative_log_delta=alternative, sensitivity_abs_log=abs(value-alternative),
                                    coefficients=coefficients.tolist(), training_sizes=sizes.tolist(),
                                    status="empirical_candidate_not_a_bound", falsification_abs_log_threshold=.1))
    return dict(training=training, predictions=predictions, method="log(delta)=a*L+b*log(L)+c; compare last-three linear fit; no fitted tolerance")


def initialize():
    for name in ("code", "development", "spatial", "static", "predictions", "cases", "pilot", "failures", "hpc"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    value = candidate_configuration()
    if CONFIG.exists():
        assert read_json(CONFIG) == value, "existing seventh-round configuration differs"
        return
    inputs = {ROOT / row["path"] for row in audit.read_csv(prior.OUT / "input_manifest.csv")}
    inputs.update(p for folder in (prior.OUT, prior.FIG) for p in folder.rglob("*") if p.is_file())
    inputs.update([prior.SOURCE, ROOT / "scripts/analyze_directional_selection_prediction.py", ROOT / "research_doc/04_历史研究/09_方向余项与新尺寸验证.md"])
    write_csv(OUT / "input_manifest.csv", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=audit.sha256(p)) for p in sorted(inputs)])
    write_json(CONFIG, value)
    write_json(OUT / "spatial/size_candidate_lock.json", dict(locked_utc=now(), config_sha256=audit.sha256(CONFIG), **size_candidate()))
    emit(dict(stage="initialize", families=len(value["families"]), cases=len(value["cases"]), historic_inputs=len(inputs)))


def config():
    return read_json(CONFIG)


def cached_right_solve(numerator, denominator):
    result = mp.matrix(numerator.rows, denominator.rows)
    # Reuse pivoted LU for every right-hand side at the same extra precision as lu_solve.
    with mp.extraprec(10):
        factors, pivots = mp.mp.LU_decomp(denominator.T.copy())
        for row in range(numerator.rows):
            rhs = mp.matrix([numerator[row, column] for column in range(numerator.cols)])
            answer = mp.mp.U_solve(factors, mp.mp.L_solve(factors, rhs, pivots))
            for column in range(result.cols):
                result[row, column] = answer[column]
    return result


def generator(case):
    return ORIGINAL_GENERATOR({**case, "skin": str(skin_mp(case))}) if "skin_mode" in case else ORIGINAL_GENERATOR(case)


def augment(static, names):
    length, n = static["length"], static["length"] // 2
    matrices = {k: spatial.decode(v) for k, v in static["matrices"].items()}
    full = mp.matrix(length)
    full[:, :n], full[:, n:] = matrices["plus"], matrices["minus"]
    inverse = mp.inverse(full)
    yplus, yminus = inverse[:n, :], inverse[n:, :]
    input_gain = mp.qr(yplus.T, mode="skinny")[0]
    graph = spatial.decode(static["input_graph"])
    rho = previous.mp_opnorm(graph)
    for name in names:
        if name in static["states"]:
            continue
        q0 = base.q0_matrix(length, name)
        cells = sorted({site // 2 for site in base.occupied(length, name)})
        empty = [site for site in range(n) if site not in cells]
        block = mp.matrix([[graph[i, j] for j in cells] for i in empty])
        delta, alpha = previous.mp_smin(block), previous.mp_smin(input_gain.T * q0)
        lower = min(1, delta) / ((1+rho)*mp.sqrt(1+rho*rho))
        upper = min(1, delta)
        assert lower <= alpha*(1+mp.mpf("1e-40")) and alpha <= upper*(1+mp.mpf("1e-40"))
        static["states"][name] = dict(alpha=str(alpha), F=p1.encode_matrix(cached_right_solve(yminus*q0, yplus*q0)),
                                     cross_block_min=str(delta), overlap_lower=str(lower), overlap_upper=str(upper), occupied_cells=cells)
        static["states"][name]["F_norm"] = str(previous.mp_opnorm(spatial.decode(static["states"][name]["F"])))
    return base.decompose(static, names)


def static_change(a, b, names):
    values = spatial.static_change(a, b)
    with mp.workdps(360):
        extra = []
        for name in names:
            sa, sb = a["states"][name], b["states"][name]
            extra.extend(spatial.relative(mp.mpf(sa[k]), mp.mpf(sb[k])) for k in ("alpha", "F_norm", "cross_block_min") if k in sa)
            fa, fb = spatial.decode(sa["F"]), spatial.decode(sb["F"])
            extra.append(p1.mp_norm(fa-fb)/max(p1.mp_norm(fa), p1.mp_norm(fb)))
            ca, cb = sa["compression"]["singular_values"], sb["compression"]["singular_values"]
            extra.extend(spatial.relative(mp.mpf(x), mp.mpf(y)) for x, y in zip(ca[:17], cb[:17]))
        values["additional_relative"] = float(max(extra))
    return values


def static_job(family):
    path = OUT / "static" / (family["key"] + ".json")
    if path.exists():
        assert read_json(path)["config_sha256"] == audit.sha256(CONFIG)
        return dict(status="reused", family=family["key"])
    assert (OUT / "spatial_lock.json").exists()
    tick, started, values, attempts = time.perf_counter(), now(), [], []
    for digits in DIGITS:
        local = time.perf_counter()
        with mp.workdps(digits):
            value = spatial.static_reference((family["length"], 2., skin_mp(family)), digits)
            value = augment(value, initials(family["length"]))
            value["skin"] = float(skin_mp(family))
            value["family"] = family
        values.append(value)
        change = static_change(values[-2], values[-1], initials(family["length"])) if len(values) > 1 else None
        attempts.append(dict(dps=digits, seconds=time.perf_counter()-local, convergence=change))
        write_json(path.with_suffix(".progress.json"), dict(started_utc=started, attempts=attempts))
        if change and max(change["relative_scalar"], change["additional_relative"]) <= 1e-8 and max(change["matrix_absolute"], change["gain_entropy"]) <= 1e-10:
            break
    else:
        raise ArithmeticError("static precision did not converge")
    write_gzip(path.with_suffix(".raw.json.gz"), values)
    write_json(path, {**value, "precision_digits": [v["dps"] for v in values], "reference_convergence": change, "attempts": attempts,
                      "started_utc": started, "completed_utc": now(), "seconds": time.perf_counter()-tick,
                      "config_sha256": audit.sha256(CONFIG), "spatial_lock_sha256": audit.sha256(OUT / "spatial_lock.json")})
    return dict(status="static_converged", family=family["key"], dps=digits, seconds=time.perf_counter()-tick)


def context(static, names=None):
    matrices = {k: spatial.decode(v) for k, v in static["matrices"].items()}
    shared = dict(matrices=matrices, rinv=mp.inverse(matrices["R"]), mu=[mp.mpf(v) for v in static["mu"]], static=static)
    contexts = {}
    for name in names or initials(static["length"]):
        state = static["states"][name]
        decomposition = state["compression"]
        contexts[name] = dict(**shared, f=spatial.decode(state["F"]), u=spatial.decode(decomposition["U"]),
                              vh=spatial.decode(decomposition["VH"]), sig=[mp.mpf(v) for v in decomposition["singular_values"]])
    return contexts


def time_context(ctx, duration):
    matrices, n = ctx["matrices"], len(ctx["mu"])
    decay = [mp.exp(-x*duration) for x in ctx["mu"]]
    maximum = min(RANKS[-1], n)
    a = mp.matrix([[decay[i]*ctx["u"][i,j]*mp.sqrt(ctx["sig"][j]) for j in range(maximum)] for i in range(n)])
    bt = mp.matrix([[mp.sqrt(ctx["sig"][i])*ctx["vh"][i,j]*decay[j] for j in range(n)] for i in range(maximum)])
    weighted = mp.matrix([[ctx["f"][i,j]*decay[i]*decay[j] for j in range(n)] for i in range(n)])
    return dict(ctx=ctx, a=a, v=bt*ctx["rinv"], y=matrices["minus"]*a,
                delta0=matrices["minus"]*weighted*ctx["rinv"])


def at_rank(tc, rank):
    ctx, matrices = tc["ctx"], tc["ctx"]["matrices"]
    y, v, a = tc["y"][:, :rank], tc["v"][:rank, :], tc["a"][:, :rank]
    if rank:
        c = matrices["Ccross"]*a
        h = mp.eye(rank)+v*c
        left = matrices["Bperp"]*a*mp.inverse(h)
        lu, lr = mp.qr(left, mode="skinny")
        rv, rr = mp.qr(v.T, mode="skinny")
        ku, ks, kvh = mp.svd(lr*rr.T)
        uu, vv = lu*ku, rv*kvh.T
        cosine = mp.diag([1/mp.sqrt(1+x*x) for x in ks])
        sine = mp.diag([x/mp.sqrt(1+x*x) for x in ks])
        q = matrices["G"]+matrices["G"]*vv*(cosine-mp.eye(rank))*vv.T+matrices["Gperp"]*uu*sine*vv.T
        kappa = ks[0]
        smin, order, reconstruction = prior.small_core_smin(matrices["G"], y, v)
        delta = tc["delta0"]-y*v
    else:
        q, kappa, smin, order, reconstruction = matrices["G"], mp.mpf(0), mp.mpf(1), 0, mp.mpf(0)
        delta = tc["delta0"]
    numerator = p1.mp_norm(delta)
    directional = min(mp.mpf(1), numerator/smin)
    transverse_numerator = p1.mp_norm(delta-q*(q.T*delta))
    transverse = min(mp.mpf(1), transverse_numerator/(smin-numerator)) if numerator < smin else mp.mpf(1)
    return dict(rank=rank, kappa=kappa, distance_to_G=kappa/mp.sqrt(1+kappa*kappa),
                projector_remainder=min(directional, transverse), directional_remainder=directional, transverse_remainder=transverse,
                column_residual_fro=numerator, small_core_smin=smin, small_core_order=order,
                small_core_reconstruction_residual=reconstruction), q


def rank1_scalar(ctx, duration):
    n=len(ctx["mu"])
    decay=[mp.exp(-x*duration) for x in ctx["mu"]]
    amplitude=mp.sqrt(ctx["sig"][0])
    a=mp.matrix([decay[i]*ctx["u"][i,0]*amplitude for i in range(n)])
    bt=mp.matrix([[amplitude*ctx["vh"][0,j]*decay[j] for j in range(n)]])
    v=bt*ctx["rinv"]
    c=ctx["matrices"]["Ccross"]*a
    b=ctx["matrices"]["Bperp"]*a
    h=1+(v*c)[0]
    numerator=p1.mp_norm(b)*p1.mp_norm(v)
    return numerator/mp.sqrt(h*h+numerator*numerator)


def rank1_front(ctx):
    length=ctx["static"]["length"]
    lo,hi=mp.mpf(".30")*length,mp.mpf(".75")*length
    flo,fhi=rank1_scalar(ctx,lo),rank1_scalar(ctx,hi)
    if flo<=mp.mpf(".05"):
        return dict(status="left_censored_candidate",lower=None,upper=float(lo))
    if fhi>mp.mpf(".05"):
        return dict(status="right_censored_candidate",lower=float(hi),upper=None)
    for _ in range(32):
        middle=(lo+hi)/2
        if rank1_scalar(ctx,middle)>mp.mpf(".05"):
            lo=middle
        else:
            hi=middle
    return dict(status="bracketed_crossing_candidate",lower=float(lo),upper=float(hi),
                interpretation="one rank1 projector crossing, without a residual or a continuous first-time guarantee")


def predict(ctx, duration):
    tick = time.perf_counter()
    tc = time_context(ctx, duration)
    preparation_seconds = time.perf_counter()-tick
    tick, attempts = time.perf_counter(), []
    for rank in RANKS:
        values, q = at_rank(tc, rank)
        attempts.append({k: float(v) for k, v in values.items()})
        if values["projector_remainder"] <= mp.mpf("1e-5"):
            break
    rank_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    entropy = previous.correlation_entropy(q)[0]
    entropy_seconds = time.perf_counter()-tick
    static = ctx["static"]
    z = mp.mpf(static["q"])*mp.exp(-2*min(ctx["mu"])*duration)
    p = mp.mpf(static["p"])
    ew = min(mp.mpf(1), z/(p-z)) if z < p else mp.mpf(1)
    eps, distance = values["projector_remainder"], values["distance_to_G"]
    error = abs(entropy-mp.mpf(static["gain_entropy"]))
    budget = base.entropy_budget(eps, static["length"]//2)+base.entropy_budget(ew, static["length"]//2)
    values.update(entropy=entropy, output_to_G_bound=ew, entropy_budget=budget,rank1_distance_candidate=rank1_scalar(ctx,duration),
                  distance_lower=max(0,distance-eps-ew), distance_upper=min(1,distance+eps+ew),
                  entropy_error_lower=max(0,error-budget), entropy_error_upper=error+budget)
    costs = dict(time_context_seconds=preparation_seconds, rank_search_seconds=rank_seconds, entropy_seconds=entropy_seconds)
    return values, q, attempts, costs


def pair_distance(qa, qb):
    return previous.mp_opnorm(qb-qa*(qa.T*qb))


def pair_bound(a, b, qa, qb):
    direct, radial = pair_distance(qa,qb), abs(a["distance_to_G"]-b["distance_to_G"])
    eps = a["projector_remainder"]+b["projector_remainder"]
    n = qa.rows//2
    entropy_error = base.entropy_budget(a["projector_remainder"],n)+base.entropy_budget(b["projector_remainder"],n)
    entropy_difference = abs(a["entropy"]-b["entropy"])
    return dict(reduced_pair_distance=direct, radial_lower=max(0,radial-eps), direct_lower=max(0,direct-eps),
                memory_lower=max(0,max(direct,radial)-eps), memory_upper=min(1,direct+eps),
                reduced_entropy_difference=entropy_difference, entropy_memory_lower=max(0,entropy_difference-entropy_error),
                entropy_memory_upper=entropy_difference+entropy_error)


def prediction_job(family):
    path = OUT / "predictions" / (family["key"]+".json")
    if path.exists():
        return dict(status="reused", family=family["key"])
    tick = time.perf_counter()
    statics = read_gzip(OUT / "static" / (family["key"]+".raw.json.gz"))[-2:]
    planned = [case for case in config()["cases"] if case["family_key"] == family["key"]]
    contexts, rows, pairs, raw, arrays = [], [], [], [], {}
    for static in statics:
        with mp.workdps(static["dps"]):
            contexts.append(context(static))
    for case in planned:
        local = []
        for static, contexts_at_precision in zip(statics, contexts):
            with mp.workdps(static["dps"]):
                local.append({name: predict(contexts_at_precision[name], mp.mpf(case["physical_time"])) for name in case["initials"]})
        with mp.workdps(360):
            for name in case["initials"]:
                low, (values,q,attempts,costs) = local[0][name][0], local[1][name]
                change = max(abs(low[k]-values[k]) for k in ("entropy","distance_to_G","projector_remainder","distance_lower","distance_upper","entropy_error_lower","entropy_error_upper","rank1_distance_candidate"))
                assert change <= mp.mpf("1e-10") and low["rank"] == values["rank"], (case["case_id"],name,change)
                numeric = {k:float(v) for k,v in values.items()}
                rows.append({**case,"initial":name,**numeric,**costs,"prediction_dps":statics[-1]["dps"],"prediction_precision_change":float(change),
                             "rank_target_passed":numeric["projector_remainder"]<=1e-5,
                             "projector_selected":numeric["distance_upper"]<=.05,"entropy_selected":numeric["entropy_error_upper"]<=.05,
                             "guaranteed_selected":numeric["distance_upper"]<=.05 and numeric["entropy_error_upper"]<=.05,
                             "guaranteed_unselected":numeric["distance_lower"]>.05 or numeric["entropy_error_lower"]>.05})
                arrays[case["case_id"]+"_"+name] = audit.mp_to_numpy(q)
                raw.append(dict(case_id=case["case_id"],initial=name,Q=p1.encode_matrix(q),values={k:str(v) for k,v in values.items()},attempts=attempts))
            for na,nb in case["pairs"]:
                pair_values=[]
                for static,values_at_precision in zip(statics,local):
                    with mp.workdps(static["dps"]):
                        pair_values.append(pair_bound(*[values_at_precision[x][0] for x in (na,nb)], *[values_at_precision[x][1] for x in (na,nb)]))
                pair_change=max(abs(pair_values[0][k]-pair_values[1][k]) for k in pair_values[0])
                assert pair_change<=mp.mpf("1e-10"), (case["case_id"],na,nb,pair_change)
                numeric={k:float(v) for k,v in pair_values[-1].items()}
                pairs.append({**case,"initial_a":na,"initial_b":nb,**numeric,"prediction_precision_change":float(pair_change),
                              "guaranteed_memory":numeric["memory_lower"]>.1,"guaranteed_entropy_memory":numeric["entropy_memory_lower"]>.1})
    np.savez_compressed(path.with_suffix(".npz"),**arrays)
    write_gzip(path.with_suffix(".raw.json.gz"),raw)
    with mp.workdps(statics[-1]["dps"]):
        fronts=[dict(initial=name,**rank1_front(contexts[-1][name])) for name in planned[0]["initials"]]
    write_json(path,dict(family=family,rows=rows,pairs=pairs,rank1_front_candidates=fronts,seconds=time.perf_counter()-tick,config_sha256=audit.sha256(CONFIG),
                        static_sha256=audit.sha256(OUT / "static" / (family["key"]+".json"))))
    return dict(status="prediction_converged",family=family["key"],states=len(rows),pairs=len(pairs),seconds=time.perf_counter()-tick)


def spatial_job(length):
    path=OUT / "spatial" / f"L{length}.json"
    if path.exists():
        return dict(status="reused",length=length)
    tick,rows,raw=time.perf_counter(),[],[]
    for initial in ("left","right"):
        matrices=[]
        for nodes,digits in ((96,240),(128,280)):
            _,matrix=base.spatial_gram(length,0.,initial,nodes,digits,True)
            with mp.workdps(digits):
                matrices.append(mp.inverse(matrix))
        for family in [x for x in config()["families"] if x["length"]==length]:
            results=[]
            for inverse,digits in zip(matrices,(240,280)):
                with mp.workdps(digits):
                    sign=1 if initial=="left" else -1
                    skin=skin_mp(family)
                    j=mp.matrix([[mp.exp(-sign*skin*(i+k+1))*inverse[i,k] for k in range(inverse.cols)] for i in range(inverse.rows)])
                    tr=mp.fsum(j[i,i] for i in range(j.rows))
                    tr2=mp.fsum(x*x for x in j)
                    results.append(dict(lower=1/tr,upper=tr/tr2,tr_inverse=tr,tr_inverse_square=tr2))
            with mp.workdps(360):
                change=max(spatial.relative(results[0][k],results[1][k]) for k in results[0])
                assert change<=mp.mpf("1e-8"),(length,initial,change)
                value=results[-1]
                rows.append(dict(family_key=family["key"],length=length,skin=float(skin_mp(family)),initial=initial,
                                 lower=float(value["lower"]),upper=float(value["upper"]),relative_width=float(value["upper"]/value["lower"]-1),precision_change=float(change)))
                raw.append(dict(**rows[-1],values={k:str(v) for k,v in value.items()}))
    write_json(path,dict(rows=rows,raw=raw,seconds=time.perf_counter()-tick,config_sha256=audit.sha256(CONFIG)))
    return dict(status="spatial_prediction",length=length,rows=len(rows),seconds=time.perf_counter()-tick)


def spatial_lock():
    path=OUT / "spatial_lock.json"
    if path.exists():
        return
    assert not list((OUT / "static").glob("*.json"))
    inputs=[OUT / "spatial/size_candidate_lock.json",OUT / "spatial/L128.json",OUT / "spatial/L144.json"]
    rows=[r for p in inputs[1:] for r in read_json(p)["rows"]]
    write_csv(OUT / "locked_spatial_predictions.csv",rows)
    write_json(path,dict(locked_utc=now(),rows=len(rows),config_sha256=audit.sha256(CONFIG),
                         inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=audit.sha256(p)) for p in inputs],
                         method=config()["spatial_policy"],source_sha256=audit.sha256(SOURCE)))
    emit(dict(stage="spatial_lock",rows=len(rows)))


def lock():
    path=OUT / "prediction_lock.json"
    if path.exists():
        check_lock()
        return
    assert not list((OUT / "cases").glob("*.json"))
    files=[OUT / "predictions" / (f["key"]+".json") for f in config()["families"]]
    predictions=[read_json(p) for p in files]
    rows=[r for v in predictions for r in v["rows"]]
    pairs=[r for v in predictions for r in v["pairs"]]
    assert len(rows)==len(config()["cases"])*5 and len(pairs)==len(config()["cases"])*6
    write_csv(OUT / "locked_predictions.csv",rows)
    write_csv(OUT / "locked_memory_predictions.csv",pairs)
    sources=[p for folder in ("static","predictions","pilot","code") for p in (OUT / folder).rglob("*") if p.is_file()]
    sources.extend([CONFIG,OUT / "spatial_lock.json",SOURCE])
    value=dict(locked_utc=now(),states=len(rows),pairs=len(pairs),pilot_case_ids=[x["case_id"] for x in config()["cases"] if x["role"]=="precision_cost_pilot"],
               config_sha256=audit.sha256(CONFIG),source_sha256=audit.sha256(SOURCE),
               inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=audit.sha256(p)) for p in sorted(sources)],
               policy="all scalar tolerances, pairs, grid and ranks frozen; pilot truth excluded from new-size validation")
    write_json(path,value)
    write_json(OUT / "prediction_lock_hash.json",{p.name:audit.sha256(p) for p in (path,OUT / "locked_predictions.csv",OUT / "locked_memory_predictions.csv")})
    emit(dict(stage="lock",states=len(rows),pairs=len(pairs),utc=value["locked_utc"]))


def check_lock():
    for name,digest in read_json(OUT / "prediction_lock_hash.json").items():
        assert audit.sha256(OUT / name)==digest,name
    assert audit.sha256(SOURCE)==read_json(OUT / "prediction_lock.json")["source_sha256"]


def reference_job(case,pilot=False):
    path=OUT / ("pilot" if pilot else "cases") / (case["case_id"]+".json")
    if path.exists():
        return dict(status="reused",case_id=case["case_id"])
    if not pilot:
        check_lock()
        for planned in [r for r in config()["cases"] if r["role"]=="precision_cost_pilot"]:
            assert previous.accepted_precision(read_json(OUT / "pilot" / (planned["case_id"]+".json"))["reference_convergence"])
    tick,started,refs,attempts=time.perf_counter(),now(),[],[]
    for digits in DIGITS:
        local=time.perf_counter()
        refs.append(prior.reference(case,digits))
        change=spatial.reference_change(refs[-2],refs[-1]) if len(refs)>1 else None
        attempts.append(dict(dps=digits,seconds=time.perf_counter()-local,convergence=change))
        write_json(path.with_suffix(".progress.json"),dict(started_utc=started,attempts=attempts))
        emit(dict(stage="pilot" if pilot else "reference",case_id=case["case_id"],dps=digits,convergence=change))
        if change and previous.accepted_precision(change):
            break
    else:
        write_gzip(path.with_suffix(".failed.raw.json.gz"),[r[2] for r in refs])
        raise ArithmeticError("independent exponential reference failed convergence")
    rows,arrays,raw=refs[-1]
    pairs=[]
    with mp.workdps(raw["dps"]):
        for na,nb in case["pairs"]:
            distance=pair_distance(spatial.decode(raw["states"][na]["Q"]),spatial.decode(raw["states"][nb]["Q"]))
            difference=abs(mp.mpf(raw["states"][na]["entropy"])-mp.mpf(raw["states"][nb]["entropy"]))
            pairs.append(dict(case_id=case["case_id"],initial_a=na,initial_b=nb,pair_distance=float(distance),absolute_entropy_difference=float(difference),
                              actual_memory=distance>mp.mpf(".1"),actual_entropy_memory=difference>mp.mpf(".1")))
    np.savez_compressed(path.with_suffix(".npz"),**arrays)
    write_gzip(path.with_suffix(".raw.json.gz"),[r[2] for r in refs])
    write_json(path,dict(case=case,rows=rows,pairs=pairs,precision_digits=[r[2]["dps"] for r in refs],reference_convergence=change,
                         attempts=attempts,started_utc=started,completed_utc=now(),seconds=time.perf_counter()-tick,
                         config_sha256=audit.sha256(CONFIG),pilot=pilot,
                         prediction_lock_sha256=None if pilot else audit.sha256(OUT / "prediction_lock.json")))
    return dict(status="reference_converged",case_id=case["case_id"],states=len(rows),dps=digits,seconds=time.perf_counter()-tick)


def benchmark(backend_label):
    path=OUT / "development" / ("benchmark_"+backend_label+".json")
    if path.exists():
        initial=path.with_name(path.stem+"_initial_with_extra_core_check.json")
        if not initial.exists():
            initial.write_bytes(path.read_bytes())
    static=read_json(prior.OUT / "static/L96_gamma2p0_g0p25.json")
    digits=static["dps"]
    rows=[]
    with mp.workdps(digits):
        n=48
        denominator=mp.matrix([[mp.mpf(1)/(i+j+1) for j in range(n)] for i in range(n)])
        numerator=mp.eye(n)
        tick=time.perf_counter()
        original=ORIGINAL_SOLVE(numerator,denominator)
        original_seconds=time.perf_counter()-tick
        tick=time.perf_counter()
        optimized=cached_right_solve(numerator,denominator)
        optimized_seconds=time.perf_counter()-tick
        lu_change=p1.mp_norm(original-optimized)/p1.mp_norm(original)
        assert lu_change<=mp.mpf("1e-30")
        contexts=context(static,["left","right","charge_density_wave"])
        for name,rank in (("left",4),("right",6),("charge_density_wave",0)):
            oldctx=prior.context(static,name)
            duration=mp.mpf(48)
            tick=time.perf_counter()
            old,qo,costs=prior.at_rank(oldctx,duration,rank,False)
            old_seconds=time.perf_counter()-tick
            tick=time.perf_counter()
            tc=time_context(contexts[name],duration)
            new,qn=at_rank(tc,rank)
            new_seconds=time.perf_counter()-tick
            change=max(abs(old[k]-new[k]) for k in ("distance_to_G","directional_remainder","transverse_remainder","small_core_smin"))
            projector=pair_distance(qo,qn)
            assert max(change,projector)<=mp.mpf("1e-10")
            rows.append(dict(initial=name,rank=rank,old_seconds=old_seconds,optimized_seconds=new_seconds,speedup=old_seconds/new_seconds,
                             absolute_scalar_change=float(change),projector_change=float(projector),old_legacy_remainder=float(old["legacy_remainder"]),
                             directional_transverse_remainder=float(new["projector_remainder"])))
    value=dict(backend=mp.libmp.BACKEND,label=backend_label,dps=digits,rows=rows,
               lu=dict(dimension=n,original_seconds=original_seconds,optimized_seconds=optimized_seconds,speedup=original_seconds/optimized_seconds,relative_change=float(lu_change)),
               environment=environment("benchmark"),scope="same precision, outputs, known static inputs; old legacy bound omitted in optimized method")
    write_json(path,value)
    return dict(status="benchmark_passed",backend=mp.libmp.BACKEND,label=backend_label)


def translated_trace_job(family):
    static=read_json(OUT / "static" / (family["key"]+".json"))
    checks=[]
    with mp.workdps(static["dps"]):
        graph=spatial.decode(static["input_graph"])
        for name in initials(static["length"])[3:]:
            cells=static["states"][name]["occupied_cells"]
            empty=[i for i in range(graph.rows) if i not in cells]
            k=mp.matrix([[graph[i,j] for j in cells] for i in empty])
            # Invert K before forming J to avoid solving the squared-condition Gram matrix.
            inverse=mp.inverse(k)
            j=inverse*inverse.T
            tr=mp.fsum(j[i,i] for i in range(j.rows))
            tr2=mp.fsum(x*x for x in j)
            lower,upper=1/mp.sqrt(tr),mp.sqrt(tr/tr2)
            delta=mp.mpf(static["states"][name]["cross_block_min"])
            assert lower<=delta*(1+mp.mpf("1e-20")) and delta<=upper*(1+mp.mpf("1e-20"))
            checks.append(dict(family_key=family["key"],initial=name,length=family["length"],skin=float(skin_mp(family)),
                               lower_mp=str(lower),upper_mp=str(upper),delta_mp=str(delta),relative_width=float(upper/lower-1)))
    write_json(OUT / "spatial" / (family["key"]+"_translated.json"),dict(rows=checks,scope="static diagnostic before dynamic truth; not a separate blind spatial prediction"))
    return dict(status="translated_trace_checked",family=family["key"],rows=len(checks))


def audit_case(case):
    check_lock()
    directory="pilot" if case["role"]=="precision_cost_pilot" else "cases"
    reference_path=OUT / directory / (case["case_id"]+".json")
    value=read_json(reference_path)
    raw=read_gzip(reference_path.with_suffix(".raw.json.gz"))[-1]
    reduced=read_gzip(OUT / "predictions" / (case["family_key"]+".raw.json.gz"))
    reduced={r["initial"]:r for r in reduced if r["case_id"]==case["case_id"]}
    rows=[]
    with mp.workdps(max(280,raw["dps"])):
        for initial in case["initials"]:
            rv,av=reduced[initial]["values"],raw["states"][initial]
            q,qr=spatial.decode(av["Q"]),spatial.decode(reduced[initial]["Q"])
            distance=pair_distance(q,qr)
            entropy_error=abs(mp.mpf(av["entropy"])-mp.mpf(rv["entropy"]))
            actual_distance=mp.mpf(av["distance_to_W"])
            actual_entropy_error=abs(mp.mpf(av["entropy"])-mp.mpf(raw["W_entropy"]))
            epsilon=mp.mpf(rv["projector_remainder"])
            eb=base.entropy_budget(epsilon,case["length"]//2)
            orthogonality=max(p1.mp_norm(x.T*x-mp.eye(x.cols)) for x in (q,qr))
            allowance=mp.mpf("1e-35")
            checks=dict(remainder_covered=distance<=epsilon+allowance,entropy_remainder_covered=entropy_error<=eb+allowance,
                        distance_interval_covered=mp.mpf(rv["distance_lower"])-allowance<=actual_distance<=mp.mpf(rv["distance_upper"])+allowance,
                        entropy_interval_covered=mp.mpf(rv["entropy_error_lower"])-allowance<=actual_entropy_error<=mp.mpf(rv["entropy_error_upper"])+allowance,
                        bases_orthogonal=orthogonality<=allowance)
            assert all(checks.values()),(case["case_id"],initial,checks)
            rows.append(dict(case_id=case["case_id"],initial=initial,**checks,projector_error_mp=str(distance),remainder_mp=str(epsilon),
                             entropy_error_mp=str(entropy_error),orthogonality_fro_mp=str(orthogonality)))
    target=OUT / "audits" / (case["case_id"]+".json")
    write_json(target,dict(rows=rows,reference_sha256=audit.sha256(reference_path),reference_raw_sha256=audit.sha256(reference_path.with_suffix(".raw.json.gz")),
                           prediction_raw_sha256=audit.sha256(OUT / "predictions" / (case["family_key"]+".raw.json.gz")),
                           source_sha256=audit.sha256(SOURCE),config_sha256=audit.sha256(CONFIG),
                           precision_digits=max(280,raw["dps"]),absolute_roundoff_allowance="1e-35",verified_utc=now()))
    return dict(status="high_precision_audit_passed",case_id=case["case_id"],states=len(rows))


def environment(stage):
    return dict(utc=now(),stage=stage,python=sys.version,executable=sys.executable,platform=platform.platform(),hostname=socket.gethostname(),
                numpy=np.__version__,mpmath=mp.__version__,mp_backend=mp.libmp.BACKEND,
                slurm={k:os.environ.get(k) for k in ("SLURM_JOB_ID","SLURM_ARRAY_JOB_ID","SLURM_ARRAY_TASK_ID","SLURM_CPUS_PER_TASK","SLURM_JOB_NODELIST")},
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
                threads={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","MPMATH_NOGMPY")},
                threadpools=threadpool_info(),source_sha256=audit.sha256(SOURCE),config_sha256=audit.sha256(CONFIG))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("initialize","spatial","spatial_lock","static","prediction","lock","pilot","reference","benchmark","translated","audit"))
    parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--label",default="gmpy")
    args=parser.parse_args()
    if args.stage=="initialize":
        initialize()
        return
    for name in ("code","development","spatial","static","predictions","cases","pilot","failures","hpc"):
        (OUT / name).mkdir(parents=True,exist_ok=True)
    spatial.right_solve=cached_right_solve
    previous.generator=generator
    info=environment(args.stage)
    tick=time.perf_counter()
    snapshot=OUT / "code" / (audit.sha256(SOURCE).lower()+".py")
    if not snapshot.exists():
        snapshot.write_bytes(SOURCE.read_bytes())
    job=str(args.index)
    try:
        with threadpool_limits(limits=1):
            if args.stage=="spatial": result=spatial_job((128,144)[args.index])
            elif args.stage=="spatial_lock": result=spatial_lock()
            elif args.stage=="lock": result=lock()
            elif args.stage=="benchmark":
                job=args.label
                result=benchmark(args.label)
            elif args.stage in ("static","prediction","translated"):
                family=config()["families"][args.index]
                job=family["key"]
                result={"static":static_job,"prediction":prediction_job,"translated":translated_trace_job}[args.stage](family)
            elif args.stage=="audit":
                case=config()["cases"][args.index]
                job=case["case_id"]
                result=audit_case(case)
            else:
                selected=[c for c in config()["cases"] if (c["role"]=="precision_cost_pilot")== (args.stage=="pilot")]
                case=selected[args.index]
                job=case["case_id"]
                result=reference_job(case,args.stage=="pilot")
        emit(dict(stage=args.stage,**(result or {})))
    except Exception as error:
        failure=dict(stage=args.stage,job=job,utc=now(),error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),environment=info)
        write_json(OUT / "failures" / (args.stage+"_"+job+"_"+now().replace(":","-")+".json"),failure)
        raise
    finally:
        info["seconds"]=time.perf_counter()-tick
        write_json(OUT / "hpc" / ("environment_"+args.stage+"_"+job+".json"),info)


if __name__=="__main__":
    main()
