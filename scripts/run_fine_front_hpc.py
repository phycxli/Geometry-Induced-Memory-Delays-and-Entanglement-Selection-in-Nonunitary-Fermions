"""Eighth round: prelocked fine fronts, translated blocks and size pilots."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
import traceback
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import run_selection_front_hpc as old

mp, np = old.mp, old.np
ROOT = old.ROOT
OUT = ROOT / "data/prl_p3_fine_front"
CONFIG = OUT / "config.json"
SOURCE = Path(__file__)
read_json, write_json, write_csv = old.read_json, old.write_json, old.write_csv
read_gzip, write_gzip, emit = old.read_gzip, old.write_gzip, old.emit
sha256 = old.audit.sha256
KERNELS = [SOURCE, *[ROOT / "scripts" / name for name in (
    "run_selection_front_hpc.py", "run_directional_selection_prediction.py",
    "run_reduced_selection_prediction.py", "run_spatial_selection_mechanism.py",
    "run_finite_time_selection_prediction.py", "run_isospectral_actual_entanglement.py",
    "audit_numerical_precision.py")]]


def config():
    return read_json(CONFIG)


def pair_list(names):
    return [list(pair) for pair in itertools.combinations(names, 2)]


def setup():
    old.spatial.right_solve = old.cached_right_solve
    old.previous.generator = old.generator
    for name in ("code", "predictions", "cases", "pilots", "audits", "failures", "hpc", "development"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def initialize():
    setup()
    if CONFIG.exists():
        return emit(dict(stage="initialize", status="reused", cases=len(config()["cases"])))
    candidates = old.audit.read_csv(old.OUT / "rank1_front_candidate_checks.csv")
    families = [f for f in old.config()["families"] if f["length"] >= 128]
    cases, fronts = [], []
    for family in families:
        length = family["length"]
        for group, names in (("edge", ["charge_density_wave", "left", "right"]),
                             ("translated", ["charge_density_wave", f"block_cell{length//16}", f"block_cell{3*length//16}"])):
            times = {}
            for initial in names[1:]:
                candidate = next(r for r in candidates if r["family_key"] == family["key"] and r["initial"] == initial)
                root = Decimal(candidate["root_time_candidate"])
                center = (root / Decimal("0.05")).to_integral_value(rounding=ROUND_FLOOR)*Decimal("0.05")
                planned = [center + Decimal(x) for x in ("-0.10", "0", "0.05", "0.15")]
                fronts.append(dict(family_key=family["key"], length=length, group=group, initial=initial,
                                   candidate_time=float(root), times=[str(t) for t in planned],
                                   root_abs_error_gate=0.1, maximum_local_grid_gap=0.1,
                                   source=candidate, status="frozen_rank1_candidate"))
                for duration in planned:
                    times.setdefault(duration, []).append(initial)
            if group == "translated":
                for fraction in ("0.30", "0.43"):
                    times.setdefault(Decimal(fraction)*length, []).extend(names[1:])
            for duration, targets in sorted(times.items()):
                key = family["key"] + "_" + group + "_t" + str(duration).replace(".", "p")
                cases.append(dict(**family, family_key=family["key"], case_id=key, group=group,
                                  t1=0.5, t2=1.0, boundary="open", skin=float(old.skin_mp(family)),
                                  physical_time=str(duration), time_fraction=str(duration/length), dt=0.05,
                                  steps=round(float(duration)/.05), initials=names, pairs=pair_list(names),
                                  target_initials=sorted(set(targets)), role="new_time_validation", digits=[200, 240, 280, 320]))
    pilots = []
    for length, mode, value, fraction in ((160, "gL", "0", "0.525"), (192, "g", "0.25", "0.55")):
        family = dict(key=old.family_key(length, mode, value), length=length, gamma=2., skin_mode=mode, skin_value=value)
        names = ["charge_density_wave", "left", "right"]
        pilots.append(dict(**family, family_key=family["key"], case_id=family["key"]+"_pilot",
                           t1=.5, t2=1., boundary="open", skin=float(old.skin_mp(family)),
                           physical_time=str(Decimal(fraction)*length), time_fraction=fraction, dt=.05,
                           steps=round(float(Decimal(fraction)*length)/.05), initials=names, pairs=pair_list(names),
                           role="precision_cost_pilot", digits=[240, 280, 320, 360]))
    immutable = {ROOT / r["path"] for r in old.audit.read_csv(old.OUT / "input_manifest.csv")}
    immutable.update(p for directory in (old.OUT, ROOT / "figures/prl_p3_selection_front") for p in directory.rglob("*")
                     if p.is_file() and p.name not in ("operations.jsonl", "local_inventory.json", "results.tar.gz", "deployment.tar.gz", "final_sync.tar.gz"))
    immutable.update(KERNELS[1:])
    immutable.add(ROOT / "research_doc/04_历史研究/10_超算前沿与弱趋肤交叉.md")
    write_csv(OUT / "input_manifest.csv", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(immutable)])
    value = dict(version=1, date="2026-10-04", families=families, cases=cases, pilots=pilots,
                 objective="resolve weak-skin projector shifts and translated-block memory before new truth",
                 ranks=list(old.RANKS), model_projector_target=1e-5,
                 selection_tolerances=dict(projector=.05, absolute_entropy=.05),
                 memory_tolerances=dict(projector=.1, absolute_entropy=.1),
                 root_abs_error_gate=.1, local_grid_gap=.1,
                 scope="finite sampled SSH OBC fronts; no continuous first-crossing, asymptotic theorem or second-model claim")
    write_json(CONFIG, value)
    write_json(OUT / "front_hypothesis_lock.json", dict(locked_utc=old.now(), source_sha256=sha256(SOURCE),
               config_sha256=sha256(CONFIG), candidates=fronts,
               policy="all seventh-round values are development; rank1 candidates and time grid frozen before new truth"))
    write_json(OUT / "kernel_manifest.json", [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p)) for p in KERNELS])
    write_json(OUT / "pilot_hypothesis_lock.json", dict(locked_utc=old.now(), pilots=pilots,
               config_sha256=sha256(CONFIG), policy="size pilots are precision/cost development, never dynamics holdouts"))
    emit(dict(stage="initialize", families=len(families), cases=len(cases), states=sum(len(c["initials"]) for c in cases), historical_inputs=len(immutable)))


def at_rank(tc, rank):
    ctx, matrices = tc["ctx"], tc["ctx"]["matrices"]
    y, v, a = tc["y"][:, :rank], tc["v"][:rank, :], tc["a"][:, :rank]
    if rank:
        h = mp.eye(rank) + v*(matrices["Ccross"]*a)
        left = matrices["Bperp"]*a*mp.inverse(h)
        lu, lr = mp.qr(left, mode="skinny")
        rv, rr = mp.qr(v.T, mode="skinny")
        ku, ks, kvh = mp.svd(lr*rr.T)
        uu, vv = lu*ku, rv*kvh.T
        cosine = [1/mp.sqrt(1+x*x) for x in ks]
        sine = [x/mp.sqrt(1+x*x) for x in ks]
        ga, ha = matrices["G"]*vv, matrices["Gperp"]*uu
        q = matrices["G"] + (ga*(mp.diag(cosine)-mp.eye(rank)) + ha*mp.diag(sine))*vv.T
        kappa = ks[0]
        smin, order, reconstruction = old.prior.small_core_smin(matrices["G"], y, v)
        delta = tc["delta0"] - y*v
        factors = dict(g=ga, h=ha, cosine=cosine, sine=sine)
    else:
        q, kappa, smin, order, reconstruction = matrices["G"], mp.mpf(0), mp.mpf(1), 0, mp.mpf(0)
        delta, factors = tc["delta0"], None
    numerator = old.p1.mp_norm(delta)
    directional = min(mp.mpf(1), numerator/smin)
    transverse_numerator = old.p1.mp_norm(delta-q*(q.T*delta))
    transverse = min(mp.mpf(1), transverse_numerator/(smin-numerator)) if numerator < smin else mp.mpf(1)
    values = dict(rank=rank, kappa=kappa, distance_to_G=kappa/mp.sqrt(1+kappa*kappa),
                  projector_remainder=min(directional, transverse), directional_remainder=directional,
                  transverse_remainder=transverse, column_residual_fro=numerator, small_core_smin=smin,
                  small_core_order=order, small_core_reconstruction_residual=reconstruction)
    return values, q, factors


def predict(ctx, duration):
    tick = time.perf_counter()
    tc = old.time_context(ctx, duration)
    prep = time.perf_counter()-tick
    attempts, tick = [], time.perf_counter()
    for rank in old.RANKS:
        values, q, factors = at_rank(tc, rank)
        attempts.append({k:float(v) for k,v in values.items()})
        if values["projector_remainder"] <= mp.mpf("1e-5"):
            break
    rank_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    entropy = old.previous.correlation_entropy(q)[0]
    entropy_seconds = time.perf_counter()-tick
    static = ctx["static"]
    z = mp.mpf(static["q"])*mp.exp(-2*min(ctx["mu"])*duration)
    p = mp.mpf(static["p"])
    ew = min(mp.mpf(1), z/(p-z)) if z < p else mp.mpf(1)
    eps, distance = values["projector_remainder"], values["distance_to_G"]
    error = abs(entropy-mp.mpf(static["gain_entropy"]))
    budget = old.base.entropy_budget(eps, static["length"]//2)+old.base.entropy_budget(ew, static["length"]//2)
    values.update(entropy=entropy, output_to_G_bound=ew, entropy_budget=budget,
                  rank1_distance_candidate=old.rank1_scalar(ctx, duration),
                  distance_lower=max(0,distance-eps-ew), distance_upper=min(1,distance+eps+ew),
                  entropy_error_lower=max(0,error-budget), entropy_error_upper=error+budget)
    return values, q, attempts, dict(time_context_seconds=prep, rank_search_seconds=rank_seconds, entropy_seconds=entropy_seconds), factors


def compact_pair(first, second):
    active = [(f, sign) for f, sign in ((first, 1), (second, -1)) if f is not None]
    if not active:
        return mp.mpf(0), 0
    order = sum(2*f["g"].cols for f, _ in active)
    span = mp.matrix(active[0][0]["g"].rows, order)
    core, offset = mp.matrix(order), 0
    # P(Q)-P(G) is supported by the rotated gain and transverse directions.
    for f, sign in active:
        rank = f["g"].cols
        span[:,offset:offset+rank], span[:,offset+rank:offset+2*rank] = f["g"], f["h"]
        for j, (c,s) in enumerate(zip(f["cosine"], f["sine"])):
            a,b = offset+j, offset+rank+j
            core[a,a], core[b,b] = -sign*s*s, sign*s*s
            core[a,b] = core[b,a] = sign*c*s
        offset += 2*rank
    _, triangular = mp.qr(span, mode="skinny")
    small = triangular*core*triangular.T
    small = (small+small.T)/2
    eigenvalues = mp.eigsy(small, eigvals_only=True)
    return max(abs(x) for x in eigenvalues), order


def bound_pair(a, b):
    va, vb = a[0], b[0]
    tick = time.perf_counter()
    direct, order = compact_pair(a[4], b[4])
    seconds = time.perf_counter()-tick
    eps = va["projector_remainder"]+vb["projector_remainder"]
    radial = abs(va["distance_to_G"]-vb["distance_to_G"])
    difference = abs(va["entropy"]-vb["entropy"])
    eb = old.base.entropy_budget(va["projector_remainder"], a[1].cols)+old.base.entropy_budget(vb["projector_remainder"], b[1].cols)
    return dict(reduced_pair_distance=direct, radial_lower=max(0,radial-eps), direct_lower=max(0,direct-eps),
                memory_lower=max(0,max(direct,radial)-eps), memory_upper=min(1,direct+eps),
                reduced_entropy_difference=difference, entropy_memory_lower=max(0,difference-eb),
                entropy_memory_upper=difference+eb, compact_core_order=order, compact_pair_seconds=seconds)


def kernel_checks():
    path = OUT / "development/kernel_checks.json"
    if path.exists():
        assert read_json(path)["source_sha256"] == sha256(SOURCE)
        return dict(status="reused")
    family = config()["families"][0]
    static = read_gzip(old.OUT / "static" / (family["key"]+".raw.json.gz"))[-1]
    checks = []
    with mp.workdps(static["dps"]):
        contexts = old.context(static, ["charge_density_wave", "left", "right"])
        for duration in (mp.mpf("64"), mp.mpf("66.5"), mp.mpf("67.5")):
            local = {name:predict(ctx,duration) for name,ctx in contexts.items()}
            for na,nb in pair_list(list(local)):
                a,b = local[na],local[nb]
                tick = time.perf_counter()
                exact = old.pair_distance(a[1],b[1])
                full_seconds = time.perf_counter()-tick
                tick = time.perf_counter()
                reduced,order = compact_pair(a[4],b[4])
                compact_seconds = time.perf_counter()-tick
                error = abs(exact-reduced)
                assert error < mp.mpf("1e-80"), (na,nb,duration,error)
                checks.append(dict(time=float(duration),initial_a=na,initial_b=nb,error_mp=str(error),core_order=order,
                                   full_seconds=full_seconds,compact_seconds=compact_seconds,speedup=full_seconds/compact_seconds))
        zero,order = compact_pair(None,None)
        assert zero==0 and order==0
        identical,order = compact_pair(local["right"][4],local["right"][4])
        assert identical < mp.mpf("1e-80")
    write_json(path,dict(status="passed",source_sha256=sha256(SOURCE),rows=checks,
                        zero_rank_passed=True,identical_subspace_passed=True,verified_utc=old.now()))
    return dict(status="compact_kernel_passed",checks=len(checks))


def prediction_job(family):
    target = OUT / "predictions" / (family["key"]+".json")
    if target.exists():
        assert read_json(target)["source_sha256"]==sha256(SOURCE)
        return dict(status="reused",family=family["key"])
    tick, started = time.perf_counter(), old.now()
    statics = read_gzip(old.OUT / "static" / (family["key"]+".raw.json.gz"))[-2:]
    planned = [c for c in config()["cases"] if c["family_key"]==family["key"]]
    contexts, rows, pairs, raw = [], [], [], []
    for static in statics:
        with mp.workdps(static["dps"]):
            contexts.append(old.context(static))
    for case in planned:
        local=[]
        for static, ctx in zip(statics, contexts):
            with mp.workdps(static["dps"]):
                local.append({name:predict(ctx[name],mp.mpf(case["physical_time"])) for name in case["initials"]})
        with mp.workdps(360):
            for name in case["initials"]:
                low,(values,q,attempts,costs,_) = local[0][name][0],local[1][name]
                change=max(abs(low[k]-values[k]) for k in ("entropy","distance_to_G","projector_remainder","distance_lower","distance_upper","entropy_error_lower","entropy_error_upper","rank1_distance_candidate"))
                assert change<=mp.mpf("1e-10") and low["rank"]==values["rank"],(case["case_id"],name,change)
                numeric={k:float(v) for k,v in values.items()}
                rows.append({**case,"initial":name,**numeric,**costs,"prediction_precision_change":float(change),
                             "rank_target_passed":numeric["projector_remainder"]<=1e-5,
                             "guaranteed_selected":numeric["distance_upper"]<=.05 and numeric["entropy_error_upper"]<=.05,
                             "guaranteed_unselected":numeric["distance_lower"]>.05 or numeric["entropy_error_lower"]>.05})
                raw.append(dict(case_id=case["case_id"],initial=name,Q=old.p1.encode_matrix(q),values={k:str(v) for k,v in values.items()},attempts=attempts))
            for na,nb in case["pairs"]:
                pair_values=[]
                for static,local_at_precision in zip(statics,local):
                    with mp.workdps(static["dps"]):
                        pair_values.append(bound_pair(local_at_precision[na],local_at_precision[nb]))
                change=max(abs(pair_values[0][k]-pair_values[1][k]) for k in pair_values[0] if k!="compact_pair_seconds")
                assert change<=mp.mpf("1e-10"),(case["case_id"],na,nb,change)
                numeric={k:float(v) for k,v in pair_values[-1].items()}
                pairs.append({**case,"initial_a":na,"initial_b":nb,**numeric,"prediction_precision_change":float(change),
                              "guaranteed_memory":numeric["memory_lower"]>.1,"guaranteed_entropy_memory":numeric["entropy_memory_lower"]>.1})
        write_json(target.with_suffix(".progress.json"),dict(started_utc=started,completed_cases=len(rows)//3,planned_cases=len(planned)))
    write_gzip(target.with_suffix(".raw.json.gz"),raw)
    write_json(target,dict(family=family,rows=rows,pairs=pairs,source_sha256=sha256(SOURCE),started_utc=started,
                           completed_utc=old.now(),seconds=time.perf_counter()-tick,config_sha256=sha256(CONFIG),
                           static_source_sha256=sha256(old.OUT / "static" / (family["key"]+".raw.json.gz"))))
    return dict(status="prediction_converged",family=family["key"],states=len(rows),seconds=time.perf_counter()-tick)


def lock():
    if (OUT / "prediction_lock.json").exists():
        check_lock()
        return dict(status="reused")
    assert not list((OUT / "cases").glob("*.json"))
    assert read_json(OUT / "development/kernel_checks.json")["status"]=="passed"
    predictions=[read_json(OUT / "predictions" / (f["key"]+".json")) for f in config()["families"]]
    rows=[r for v in predictions for r in v["rows"]]
    pairs=[r for v in predictions for r in v["pairs"]]
    assert len(rows)==len(config()["cases"])*3 and len(pairs)==len(rows)
    write_csv(OUT / "locked_predictions.csv",rows)
    write_csv(OUT / "locked_memory_predictions.csv",pairs)
    inputs=[CONFIG,OUT / "front_hypothesis_lock.json",OUT / "kernel_manifest.json",OUT / "development/kernel_checks.json"]
    inputs.extend(p for p in (OUT / "predictions").glob("*") if p.is_file())
    inputs.extend(old.OUT / "static" / (f["key"]+ext) for f in config()["families"] for ext in (".json",".raw.json.gz"))
    write_json(OUT / "prediction_lock.json",dict(locked_utc=old.now(),source_sha256=sha256(SOURCE),states=len(rows),pairs=len(pairs),
               config_sha256=sha256(CONFIG),inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in sorted(set(inputs))]))
    write_json(OUT / "prediction_lock_hash.json",{name:sha256(OUT / name) for name in ("prediction_lock.json","locked_predictions.csv","locked_memory_predictions.csv")})
    return dict(status="predictions_locked",states=len(rows),pairs=len(pairs))


def check_lock():
    for name,digest in read_json(OUT / "prediction_lock_hash.json").items():
        assert sha256(OUT / name)==digest,name
    assert sha256(SOURCE)==read_json(OUT / "prediction_lock.json")["source_sha256"]
    for row in read_json(OUT / "kernel_manifest.json"):
        assert sha256(ROOT / row["path"])==row["sha256"],row["path"]


def reference_job(case,pilot=False):
    path=OUT / ("pilots" if pilot else "cases") / (case["case_id"]+".json")
    if path.exists():
        assert read_json(path)["source_sha256"]==sha256(SOURCE)
        return dict(status="reused",case_id=case["case_id"])
    if not pilot:
        check_lock()
    tick,started,refs,attempts=time.perf_counter(),old.now(),[],[]
    for digits in case["digits"]:
        local=time.perf_counter()
        refs.append(old.prior.reference(case,digits))
        change=old.spatial.reference_change(refs[-2],refs[-1]) if len(refs)>1 else None
        attempts.append(dict(dps=digits,seconds=time.perf_counter()-local,convergence=change))
        write_json(path.with_suffix(".progress.json"),dict(started_utc=started,attempts=attempts))
        emit(dict(stage="pilot" if pilot else "reference",case_id=case["case_id"],dps=digits,convergence=change))
        if change and old.previous.accepted_precision(change):
            break
    else:
        write_gzip(path.with_suffix(".failed.raw.json.gz"),[r[2] for r in refs])
        raise ArithmeticError("independent exponential reference did not converge")
    rows,arrays,raw=refs[-1]
    pairs=[]
    with mp.workdps(raw["dps"]):
        for na,nb in case["pairs"]:
            distance=old.pair_distance(old.spatial.decode(raw["states"][na]["Q"]),old.spatial.decode(raw["states"][nb]["Q"]))
            difference=abs(mp.mpf(raw["states"][na]["entropy"])-mp.mpf(raw["states"][nb]["entropy"]))
            pairs.append(dict(case_id=case["case_id"],initial_a=na,initial_b=nb,pair_distance=float(distance),absolute_entropy_difference=float(difference),
                              actual_memory=distance>mp.mpf(".1"),actual_entropy_memory=difference>mp.mpf(".1")))
    np.savez_compressed(path.with_suffix(".npz"),**arrays)
    write_gzip(path.with_suffix(".raw.json.gz"),[r[2] for r in refs])
    write_json(path,dict(case=case,rows=rows,pairs=pairs,precision_digits=[r[2]["dps"] for r in refs],reference_convergence=change,
                         attempts=attempts,started_utc=started,completed_utc=old.now(),seconds=time.perf_counter()-tick,
                         config_sha256=sha256(CONFIG),source_sha256=sha256(SOURCE),pilot=pilot,
                         prediction_lock_sha256=None if pilot else sha256(OUT / "prediction_lock.json")))
    return dict(status="reference_converged",case_id=case["case_id"],states=len(rows),dps=digits,seconds=time.perf_counter()-tick)


def audit_case(case):
    check_lock()
    path=OUT / "cases" / (case["case_id"]+".json")
    raw=read_gzip(path.with_suffix(".raw.json.gz"))[-1]
    prediction_path=OUT / "predictions" / (case["family_key"]+".raw.json.gz")
    reduced={r["initial"]:r for r in read_gzip(prediction_path) if r["case_id"]==case["case_id"]}
    checks=[]
    with mp.workdps(max(280,raw["dps"])):
        allowance=mp.mpf("1e-35")
        for name in case["initials"]:
            av,rv=raw["states"][name],reduced[name]["values"]
            q,qr=old.spatial.decode(av["Q"]),old.spatial.decode(reduced[name]["Q"])
            error=old.pair_distance(q,qr)
            eps=mp.mpf(rv["projector_remainder"])
            entropy_error=abs(mp.mpf(av["entropy"])-mp.mpf(rv["entropy"]))
            actual_d=mp.mpf(av["distance_to_W"])
            actual_e=abs(mp.mpf(av["entropy"])-mp.mpf(raw["W_entropy"]))
            orth=max(old.p1.mp_norm(x.T*x-mp.eye(x.cols)) for x in (q,qr))
            gates=dict(remainder_covered=error<=eps+allowance,
                       entropy_remainder_covered=entropy_error<=old.base.entropy_budget(eps,q.cols)+allowance,
                       distance_interval_covered=mp.mpf(rv["distance_lower"])-allowance<=actual_d<=mp.mpf(rv["distance_upper"])+allowance,
                       entropy_interval_covered=mp.mpf(rv["entropy_error_lower"])-allowance<=actual_e<=mp.mpf(rv["entropy_error_upper"])+allowance,
                       bases_orthogonal=orth<=allowance)
            assert all(gates.values()),(case["case_id"],name,gates)
            checks.append(dict(case_id=case["case_id"],initial=name,**gates,projector_error_mp=str(error),entropy_error_mp=str(entropy_error),
                               remainder_mp=str(eps),orthogonality_fro_mp=str(orth)))
    write_json(OUT / "audits" / path.name,dict(rows=checks,reference_sha256=sha256(path),reference_raw_sha256=sha256(path.with_suffix(".raw.json.gz")),
               prediction_raw_sha256=sha256(prediction_path),precision_digits=max(280,raw["dps"]),absolute_roundoff_allowance="1e-35",verified_utc=old.now()))
    return dict(status="audit_passed",case_id=case["case_id"],states=len(checks))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("initialize","kernel","prediction","lock","reference","pilot","audit"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    if args.stage=="initialize":
        initialize()
        return
    setup()
    info=old.environment(args.stage)
    info.update(source_sha256=sha256(SOURCE),config_sha256=sha256(CONFIG))
    snapshot=OUT / "code" / (sha256(SOURCE).lower()+".py")
    if not snapshot.exists():
        snapshot.write_bytes(SOURCE.read_bytes())
    tick,job=time.perf_counter(),str(args.index)
    try:
        with old.threadpool_limits(limits=1):
            if args.stage=="kernel":result=kernel_checks()
            elif args.stage=="lock":result=lock()
            elif args.stage=="prediction":
                family=config()["families"][args.index]
                job=family["key"]
                result=prediction_job(family)
            else:
                case=config()["pilots" if args.stage=="pilot" else "cases"][args.index]
                job=case["case_id"]
                result=audit_case(case) if args.stage=="audit" else reference_job(case,args.stage=="pilot")
        emit(dict(stage=args.stage,**result))
    except Exception as error:
        write_json(OUT / "failures" / (args.stage+"_"+job+"_"+old.now().replace(":","-")+".json"),
                   dict(stage=args.stage,job=job,utc=old.now(),error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),environment=info))
        raise
    finally:
        info["seconds"]=time.perf_counter()-tick
        write_json(OUT / "hpc" / ("environment_"+args.stage+"_"+job+".json"),info)


if __name__=="__main__":
    main()
