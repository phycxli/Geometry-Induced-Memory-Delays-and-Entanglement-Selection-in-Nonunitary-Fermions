"""Round ten: reflected geometry and frozen L160/192 mechanism tests."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import multiprocessing
import os
import time
import traceback
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import run_rank2_front_hpc as ninth

fine, old = ninth.previous, ninth.old
mp, np, ROOT = old.mp, old.np, old.ROOT
OUT = ROOT / "data/prl_p3_new_sizes"
CONFIG, SOURCE = OUT / "config.json", Path(__file__)
PRIOR = ROOT / "data/prl_p3_rank2_front"
STATIC_PRIOR = ROOT / "data/prl_p3_selection_front/static"
read_json, write_json, write_csv = old.read_json, old.write_json, old.write_csv
read_gzip, write_gzip, sha256, emit = old.read_gzip, old.write_gzip, old.audit.sha256, old.emit
DIGITS = (240, 280, 320, 360)
RANKS = (0, 1, 2, 4, 6, 8, 12, 16, 24)


def config():
    return read_json(CONFIG)


def setup():
    old.OUT, old.CONFIG, old.DIGITS, old.RANKS = OUT, CONFIG, DIGITS, RANKS
    fine.OUT, fine.CONFIG, fine.SOURCE = OUT, CONFIG, SOURCE
    fine.check_lock = check_lock
    fine.setup()
    for name in ("static", "spatial", "roots", "response", "code"):
        (OUT / name).mkdir(exist_ok=True)


def size_candidates(families):
    rows, training = [], []
    for skin, initial in itertools.product((0., .25), ("left", "right")):
        values = {}
        for directory in (old.spatial.OUT / "static", old.base.OUT / "static", old.prior.OUT / "static", STATIC_PRIOR):
            for path in directory.glob("*.json"):
                if ".progress." in path.name:
                    continue
                value = read_json(path)
                if value.get("gamma") != 2. or value.get("skin") != skin or value["length"] < 64:
                    continue
                with mp.workdps(360):
                    delta = mp.mpf(value["states"][initial]["cross_block_min"])
                    values[value["length"]] = dict(length=value["length"], skin=skin, initial=initial,
                        log_delta=float(mp.log(delta)), delta_mp=str(delta), source=path.relative_to(ROOT).as_posix(), sha256=sha256(path))
        ordered = [values[k] for k in sorted(values)]
        assert len(ordered) >= 4
        training.extend(ordered)
        sizes, logs = np.array([r["length"] for r in ordered]), np.array([r["log_delta"] for r in ordered])
        coefficients = np.linalg.lstsq(np.column_stack((sizes, np.log(sizes), np.ones(len(sizes)))), logs, rcond=None)[0]
        linear = np.polyfit(sizes[-3:], logs[-3:], 1)
        for family in families:
            if float(old.skin_mp(family)) != skin:
                continue
            length = family["length"]
            rows.append(dict(family_key=family["key"], length=length, initial=initial, skin=skin,
                predicted_log_delta=float(np.array([length, np.log(length), 1.]) @ coefficients),
                alternative_log_delta=float(np.polyval(linear, length)), coefficients=coefficients.tolist(),
                training_sizes=sizes.tolist(), abs_log_error_gate=.1, status="empirical_extrapolation_not_a_theorem"))
    fronts = []
    for family in families:
        oldkeys = [old.family_key(length, family["skin_mode"], family["skin_value"]) for length in (128, 144)]
        data = [read_json(PRIOR / "roots" / (key + ".json"))["rows"] for key in oldkeys]
        for role in range(4):
            first, second = data[0][role], data[1][role]
            slope = (second["rank2_candidate_time"] - first["rank2_candidate_time"]) / 16
            predicted = second["rank2_candidate_time"] + slope * (family["length"] - 144)
            name = second["initial"]
            if name.startswith("block_cell"):
                name = f"block_cell{(1 if role == 2 else 3)*family['length']//16}"
            fronts.append(dict(family_key=family["key"], length=family["length"], initial=name,
                predicted_time=predicted, slope_per_site=slope, abs_time_error_gate=.1,
                training_sizes=[128, 144], status="two_size_affine_extrapolation_not_a_bound"))
    return dict(training=training, spatial=rows, fronts=fronts)


def initialize():
    setup()
    if CONFIG.exists():
        check_source()
        return dict(status="reused")
    families = [dict(key=old.family_key(length, mode, value), length=length, gamma=2.,
        skin_mode=mode, skin_value=value) for length in (160, 192)
        for mode, value in (("gL", "0"), ("gL", "2"), ("gL", "4"), ("g", "0.25"))]
    value = dict(version=1, families=families, cases=[], pilots=[], ranks=list(RANKS),
        digits=list(DIGITS), batch_size=4, workers=4, model_projector_target=1e-5,
        selection_tolerances=dict(projector=.05, absolute_entropy=.05),
        memory_tolerances=dict(projector=.1, absolute_entropy=.1),
        time_offsets=["-0.065", "0.005", "0.075"], persistence_fraction="0.70",
        local_time_gate=.1, spatial_log_gate=.1,
        scope="new finite sizes and sampled crossings; no continuous cutoff or uniform asymptotic proof",
        observed_pilots=["L160/g0/t84", "L192/g.25/t105.6"],
        pilot_policy="previous precision pilots are development; exclude their time values from this grid")
    write_json(CONFIG, value)
    write_json(OUT / "bootstrap_config.json", value)
    write_json(OUT / "size_hypothesis_lock.json", dict(locked_utc=old.now(), **size_candidates(families)))
    kernels = sorted(set([SOURCE, ninth.SOURCE, *ninth.previous.KERNELS]))
    write_json(OUT / "kernel_manifest.json", [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p)) for p in kernels])
    write_json(OUT / "bootstrap_lock.json", dict(locked_utc=old.now(), source_sha256=sha256(SOURCE),
        prior_artifact_manifest_sha256=sha256(PRIOR / "artifact_manifest.csv"),
        prior_prediction_lock_sha256=sha256(PRIOR / "prediction_lock.json"),
        bootstrap_config_sha256=sha256(OUT / "bootstrap_config.json"),
        size_hypothesis_sha256=sha256(OUT / "size_hypothesis_lock.json"),
        policy="all rounds through nine and previous size pilots are development; no retrospective holdouts"))
    return dict(status="initialized", families=len(families))


def check_source():
    lock = read_json(OUT / "bootstrap_lock.json")
    assert sha256(SOURCE) == lock["source_sha256"]
    assert sha256(OUT / "bootstrap_config.json") == lock["bootstrap_config_sha256"]
    assert sha256(OUT / "size_hypothesis_lock.json") == lock["size_hypothesis_sha256"]
    for row in read_json(OUT / "kernel_manifest.json"):
        assert sha256(ROOT / row["path"]) == row["sha256"], row["path"]


def spatial_job(length):
    target = OUT / "spatial" / f"L{length}.json"
    if target.exists():
        return dict(status="reused", length=length)
    rows, raw = [], []
    for initial in ("left", "right"):
        inverses = []
        for nodes, digits in ((128, 280), (160, 320)):
            _, matrix = old.base.spatial_gram(length, 0., initial, nodes, digits, True)
            with mp.workdps(digits):
                inverses.append(mp.inverse(matrix))
        for family in [f for f in config()["families"] if f["length"] == length]:
            results = []
            for inverse, digits in zip(inverses, (280, 320)):
                with mp.workdps(digits):
                    sign = 1 if initial == "left" else -1
                    skin = old.skin_mp(family)
                    j = mp.matrix([[mp.exp(-sign*skin*(i+k+1))*inverse[i,k] for k in range(inverse.cols)] for i in range(inverse.rows)])
                    tr, tr2 = mp.fsum(j[i,i] for i in range(j.rows)), mp.fsum(x*x for x in j)
                    distribution = [j[i,i]/tr for i in range(j.rows)]
                    mean = mp.fsum(i*x for i,x in enumerate(distribution))
                    variance = mp.fsum((i-mean)**2*x for i,x in enumerate(distribution))
                    results.append(dict(lower=1/tr, upper=tr/tr2, inverse_trace=tr, inverse_square_trace=tr2,
                        mean_cell=mean, variance_cell=variance, log_delta_slope=sign*(1+2*mean),
                        concentration_width=tr*tr/tr2-1))
            with mp.workdps(360):
                change = max(old.spatial.relative(results[0][k], results[1][k]) for k in ("lower", "upper", "inverse_trace", "inverse_square_trace"))
                assert change <= mp.mpf("1e-8"), (length, initial, change)
                row = dict(family_key=family["key"], length=length, skin=float(old.skin_mp(family)), initial=initial,
                    **{k:float(v) for k,v in results[-1].items()}, precision_change=float(change))
                rows.append(row)
                raw.append(dict(**row, values={k:str(v) for k,v in results[-1].items()}, inverse_diagonal_distribution=[str(x) for x in distribution]))
    write_json(target, dict(rows=rows, raw=raw, completed_utc=old.now(), source_sha256=sha256(SOURCE)))
    return dict(status="spatial_complete", length=length, rows=len(rows))


def spatial_lock():
    if (OUT / "spatial_lock.json").exists():
        return dict(status="reused")
    assert not list((OUT / "static").glob("*.json"))
    paths = [OUT / "spatial" / f"L{length}.json" for length in (160, 192)]
    rows = [r for p in paths for r in read_json(p)["rows"]]
    write_csv(OUT / "locked_spatial_predictions.csv", rows)
    write_json(OUT / "spatial_lock.json", dict(locked_utc=old.now(), rows=len(rows),
        inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p)) for p in paths], source_sha256=sha256(SOURCE)))
    return dict(status="spatial_locked", rows=len(rows))


def plan():
    cfg = config()
    if cfg["cases"]:
        return dict(status="reused", cases=len(cfg["cases"]))
    assert not list((OUT / "cases").glob("*.json"))
    cases, fronts = [], []
    for family in cfg["families"]:
        length = family["length"]
        static = read_gzip(OUT / "static" / (family["key"] + ".raw.json.gz"))[-1]
        with mp.workdps(static["dps"]):
            contexts = old.context(static)
            for index, name in enumerate(old.initials(length)[1:]):
                estimate = old.rank1_front(contexts[name])
                assert estimate["lower"] is not None and estimate["upper"] is not None, (family, name, estimate)
                center = mp.mpf(str((estimate["lower"] + estimate["upper"])/2))
                root = ninth.local_root(contexts[name], center)
                root = root[0] if isinstance(root, tuple) else root
                center_decimal = (Decimal(str(float(root)))/Decimal("0.01")).to_integral_value(rounding=ROUND_FLOOR)*Decimal("0.01")
                names = ["charge_density_wave", "left", "right"] if index < 2 else ["charge_density_wave", *old.initials(length)[3:]]
                fronts.append(dict(family_key=family["key"], length=length, initial=name, candidate_time=float(root),
                    group="edge" if index < 2 else "translated", offsets=cfg["time_offsets"]))
                for offset in cfg["time_offsets"]:
                    duration = str(center_decimal + Decimal(offset))
                    assert not (length == 160 and float(old.skin_mp(family)) == 0 and Decimal(duration) == 84)
                    assert not (length == 192 and float(old.skin_mp(family)) == .25 and Decimal(duration) == Decimal("105.6"))
                    cases.append(dict(**family, family_key=family["key"], case_id=family["key"]+"_i"+str(index)+"_t"+duration.replace(".","p"),
                        physical_time=duration, time_fraction=str(Decimal(duration)/length), t1=.5, t2=1., boundary="open",
                        skin=float(old.skin_mp(family)), initials=names, pairs=fine.pair_list(names), digits=list(DIGITS),
                        target_initials=[name], group=fronts[-1]["group"], role="new_size_mechanism", dt=.05, steps=round(float(duration)/.05)))
        duration = str(Decimal(cfg["persistence_fraction"])*length)
        names = old.initials(length)
        cases.append(dict(**family, family_key=family["key"], case_id=family["key"]+"_late", physical_time=duration,
            time_fraction=cfg["persistence_fraction"], t1=.5, t2=1., boundary="open", skin=float(old.skin_mp(family)),
            initials=names, pairs=old.memory_pairs(length), digits=list(DIGITS), target_initials=names[1:],
            group="persistence", role="new_size_mechanism", dt=.05, steps=round(float(duration)/.05)))
    cfg["cases"] = cases
    write_json(CONFIG, cfg)
    write_json(OUT / "front_hypothesis_lock.json", dict(locked_utc=old.now(), rows=fronts, source_sha256=sha256(SOURCE),
        config_sha256=sha256(CONFIG), policy="local rank2 roots from new static geometry, frozen before exponential truth; not first passage"))
    return dict(status="planned", cases=len(cases), states=sum(len(c["initials"]) for c in cases))


def lock():
    if (OUT / "prediction_lock.json").exists():
        check_lock()
        return dict(status="reused")
    assert not list((OUT / "cases").glob("*.json"))
    predictions = [read_json(OUT / "predictions" / (f["key"]+".json")) for f in config()["families"]]
    rows, pairs = [r for v in predictions for r in v["rows"]], [r for v in predictions for r in v["pairs"]]
    assert len(rows) == sum(len(c["initials"]) for c in config()["cases"])
    assert len(pairs) == sum(len(c["pairs"]) for c in config()["cases"])
    write_csv(OUT / "locked_predictions.csv", rows)
    write_csv(OUT / "locked_memory_predictions.csv", pairs)
    paths = [CONFIG, OUT / "front_hypothesis_lock.json", OUT / "spatial_lock.json", OUT / "size_hypothesis_lock.json"]
    paths.extend(p for folder in ("static", "predictions") for p in (OUT/folder).glob("*") if p.is_file() and ".progress." not in p.name)
    write_json(OUT / "prediction_lock.json", dict(locked_utc=old.now(), states=len(rows), pairs=len(pairs),
        config_sha256=sha256(CONFIG), source_sha256=sha256(SOURCE),
        inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p)) for p in sorted(paths)]))
    write_json(OUT / "prediction_lock_hash.json", {name:sha256(OUT/name) for name in ("prediction_lock.json", "locked_predictions.csv", "locked_memory_predictions.csv")})
    return dict(status="predictions_locked", states=len(rows), pairs=len(pairs))


def check_lock():
    check_source()
    assert sha256(CONFIG) == read_json(OUT / "prediction_lock.json")["config_sha256"]
    for name, digest in read_json(OUT / "prediction_lock_hash.json").items():
        assert sha256(OUT/name) == digest, name


def execute(stage, index):
    setup()
    check_source()
    info, tick = old.environment(stage), time.perf_counter()
    key = str(index)
    try:
        if stage == "spatial":
            result = spatial_job((160,192)[index])
        elif stage in ("static", "prediction"):
            family = config()["families"][index]
            key = family["key"]
            result = old.static_job(family) if stage == "static" else fine.prediction_job(family)
        elif stage in ("spatial_lock", "plan", "lock"):
            result = {"spatial_lock":spatial_lock, "plan":plan, "lock":lock}[stage]()
        else:
            case = config()["cases"][index]
            key = case["case_id"]
            result = fine.reference_job(case) if stage == "reference" else fine.audit_case(case)
        emit(dict(stage=stage, **result))
    except Exception as error:
        write_json(OUT / "failures" / (stage+"_"+key+"_"+old.now().replace(":","-")+".json"),
            dict(stage=stage, key=key, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), environment=info))
        raise
    finally:
        info.update(seconds=time.perf_counter()-tick, completed_utc=old.now())
        write_json(OUT / "hpc" / ("environment_"+stage+"_"+key+".json"), info)


def batch_worker(stage, index, cpu):
    os.sched_setaffinity(0, {cpu})
    execute(stage, index)


def batch(stage, index):
    setup()
    indices = list(range(index*4, min(index*4+4, len(config()["cases"]))))
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= len(indices)
    receipt = dict(stage=stage, batch_index=index, case_indices=indices, started_utc=old.now(), cpus=cpus)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(indices), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(batch_worker, stage, case, cpu) for case,cpu in zip(indices,cpus)]
            for future in futures:
                future.result()
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        receipt["completed_utc"] = old.now()
        write_json(OUT / "hpc" / f"batch_{stage}_{index}.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("initialize","spatial","spatial_lock","static","plan","prediction","lock","reference","audit","batch_reference","batch_audit"))
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
