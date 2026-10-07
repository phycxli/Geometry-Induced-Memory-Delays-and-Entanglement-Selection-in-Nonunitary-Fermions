"""Aggregate independent numerical evidence without rewriting earlier results."""

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm, svd

import run_figure_strengthening as study
import analyze_theory_upgrade as theory

ROOT, OUT = study.ROOT, study.OUT


def read_csv(path):
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def kind(row):
    name = row["initial"]
    return ("block1" if int(name[10:])==int(row["length"])//16 else "block3") if name.startswith("block_cell") else name


def complete_files(folder):
    return sorted(p for p in (OUT/folder).glob("*.json") if ".progress." not in p.name)


def early_output():
    target = OUT/"early_output.csv"
    if target.exists():
        return
    raw_path = ROOT/"data/prl_p3_new_sizes/static/L192_g0p25.raw.json.gz"
    raw = study.read_gzip(raw_path)[-1]
    gain = np.array(raw["matrices"]["G"],float)
    perpendicular = np.array(raw["matrices"]["Gperp"],float)
    generator = np.array(study.old.previous.generator(dict(length=192,gamma=2.,skin=".25",t1=.5,t2=1.)).tolist(),float)
    a = theory.constants(.25)["a"]
    rows = []
    with study.old.threadpool_limits(limits=1):
        for duration in np.r_[.001,np.linspace(.05,4,60),5.,6.,8.]:
            flow = expm(generator*duration)
            u,s,_ = svd(flow,check_finite=True)
            distance = np.linalg.norm(perpendicular.T@u[:,:96],2)
            envelope = math.exp(-2*a*duration)
            assert distance <= envelope+1e-9
            rows.append(dict(length=192,skin=.25,time=float(duration),distance_to_G=float(distance),
                             singular_ratio=float(s[96]/s[95]),analytic_output_upper=envelope,
                             method="float64 direct expm and full SVD, early well-conditioned propagation",
                             static_sha256=study.theory.sha(raw_path)))
    study.old.write_csv(target,rows)


def aggregate(require_complete=True):
    cfg = study.config()
    count = {folder:len(complete_files(folder)) for folder in ("spectral","static","curves","traces","response","reference_contexts","references")}
    if require_complete:
        expected = dict(spectral=5,static=9,curves=9,traces=9,response=12,reference_contexts=6,references=len(study.reference_cases()))
        assert count==expected, (count,expected)
        assert study.read_json(OUT/"checks/kernel_checks.json")["status"]=="passed"
        assert study.read_json(OUT/"checks/trace_rank_checks.json")["status"]=="passed"
        assert study.read_json(OUT/"checks/early_chart_checks.json")["status"]=="passed"
        assert study.read_json(OUT/"checks/factorized_residual_checks.json")["status"]=="passed"
        assert study.read_json(OUT/"checks/large_residual_benchmark.json")["status"]=="passed"
    old_roots = read_csv(ROOT/"data/prl_theory_upgrade/roots.csv")
    new_roots = [r for path in complete_files("curves") for r in study.read_json(path)["rows"]]
    roots = [{**r,"cohort":"existing"} for r in old_roots]+[{**r,"cohort":"new"} for r in new_roots]
    widths = [{**r,"cohort":"existing","kind":kind(r)} for r in read_csv(ROOT/"data/prl_theory_upgrade/widths.csv")]
    for family in cfg["families"]:
        for name in study.old.initials(family["length"])[1:]:
            local = {r["level"]:float(r["root_time"]) for r in new_roots if r["family_key"]==family["key"] and r["initial"]==name}
            if set(local)!=set(study.LEVELS):
                continue
            width = local["0.01"]-local["0.3"]
            bound = theory.width_bound(family["skin"])
            assert width < bound
            widths.append(dict(family_key=family["key"],length=family["length"],skin=family["skin"],initial=name,
                               kind=kind(dict(initial=name,length=family["length"])),t05=local["0.05"],width_30_to_01=width,
                               relative_width_30_to_01=width/local["0.05"],uniform_width_upper=bound,cohort="new"))
    traces = [r for path in complete_files("traces") for r in study.read_json(path)["rows"]]
    # Keep signed and reflection cohorts explicit: their controls differ at fixed g versus fixed gL.
    signed = [r for path in complete_files("response") if path.stem.endswith("signed") for r in study.read_json(path)["rows"]]
    reflections = [r for path in complete_files("response") if path.stem.endswith("reflection") for r in study.read_json(path)["rows"]]
    reflection_checks = []
    for row in reflections:
        if not row["folded"]:
            continue
        bulk = next(r for r in reflections if not r["folded"] and r["length"]==row["length"]
                    and r["initial"]==row["initial"] and r["skin"]==row["skin"])
        log_ratio = float(bulk["log_delta"])-float(row["log_delta"])
        reflection_checks.append(dict(length=row["length"],initial=row["initial"],skin=row["skin"],
                                      reflected_log_delta=row["log_delta"],bulk_log_delta=bulk["log_delta"],
                                      log_bulk_to_reflected=log_ratio,bulk_to_reflected=math.exp(log_ratio)))
    references = []
    for path in complete_files("references"):
        value = study.read_json(path)
        references.append(dict(family_key=value["family"]["key"],length=value["family"]["length"],skin=value["family"]["skin"],
                               initial=value["initial"],offset=value["offset"],root=value["root"],
                               **value["values"],precision_change_mp=value["precision_change_mp"]))
    predictions = []
    for row in study.read_json(OUT/"prediction_lock.json")["predictions"]:
        measured = next((r for r in new_roots if r["family_key"]==row["family_key"] and kind(r)==row["kind"] and r["level"]==row["level"]), None)
        if measured:
            predictions.append(dict(**row,actual_time=float(measured["root_time"]),
                                    signed_error=float(measured["root_time"])-row["predicted_time"]))
    for filename,rows in (("roots.csv",roots),("widths.csv",widths),("traces.csv",traces),("signed_response.csv",signed),
                          ("reflection_checks.csv",reflection_checks),("independent_checks.csv",references),("prediction_checks.csv",predictions)):
        if rows:
            study.old.write_csv(OUT/filename,rows)
    early_output()
    validation = numerical_checks(new_roots,traces,signed,references,predictions,require_complete)
    max_mp = lambda rows,key: max((float(r[key]) for r in rows if key in r),default=0.)
    width_values = [float(r["width_30_to_01"]) for r in widths]
    summary = dict(status="complete" if require_complete else "partial",computed_utc=study.old.now(),counts=count,
                   old_threshold_roots=len(old_roots),new_low_threshold_roots=len(new_roots),widths=len(widths),
                   trace_points=len(traces),signed_response_points=len(signed),reflection_comparisons=len(reflection_checks),
                   independent_propagation_points=len(references),frozen_prediction_tests=len(predictions),
                   max_frozen_prediction_error=max((abs(r["signed_error"]) for r in predictions),default=0.),
                   max_relative_frozen_prediction_error=max((abs(r["signed_error"])/r["actual_time"] for r in predictions),default=0.),
                   predictions_above_earlier_015_tolerance=sum(abs(r["signed_error"])>.15 for r in predictions),
                   worst_frozen_prediction=max(predictions,key=lambda r:abs(r["signed_error"])) if predictions else None,
                   max_root_projector_remainder=max_mp(new_roots,"projector_remainder_mp"),
                   max_trace_precision_change=max_mp(traces,"precision_change_mp"),
                   max_independent_precision_change=max_mp(references,"precision_change_mp"),
                   max_independent_reduced_error=max_mp(references,"reduced_error_mp"),
                   numerical_checks=validation,
                   low_window_range=[min(width_values),max(width_values)],
                   reflection_ratio_range=[min((r["bulk_to_reflected"] for r in reflection_checks),default=0.),
                                           max((r["bulk_to_reflected"] for r in reflection_checks),default=0.)],
                   science_boundaries=["No sharp time-coefficient theorem", "Static reflected delta is not exactly whitened alpha",
                      "Low-threshold theorem, no all-threshold cutoff or universal profile", "High-precision convergence, not interval certification"])
    study.write_json(OUT/"summary.json",summary)
    return summary


def numerical_checks(roots,traces,signed,references,predictions,complete):
    keys=[(r["family_key"],r["initial"],r["level"]) for r in roots]
    assert len(keys)==len(set(keys))
    assert all(float(r["projector_remainder_mp"])<1e-5 for r in roots)
    assert all(float(r["precision_change_mp"])<4e-6 for r in roots)
    assert all(0<float(r["distance_to_G"])<=1+1e-12 for r in traces)
    assert all(float(r["precision_change_mp"])<1e-8 for r in traces)
    relative_errors=[float(r["projector_remainder"])/float(r["distance_to_G"]) for r in traces]
    tail_relative_errors=[float(r["projector_remainder"])/float(r["distance_to_G"])
                          for r in traces if int(r["length"])==384]
    assert all(float(r["projector_remainder"])<=.001*float(r["distance_to_G"]) for r in traces if int(r["length"])==384)
    pairs=Counter((r["family_key"],r["initial"]) for r in references)
    assert all(n<=2 for n in pairs.values())
    assert all((float(r["distance_to_G"])>.05)==(float(r["offset"])<0) for r in references)
    assert all(float(r["precision_change_mp"])<1e-10 and float(r["reduced_error_mp"])<1e-5 for r in references)
    assert all(float(r["precision_change_mp"])<1e-8 for r in signed)
    envelope_ratios=[]
    groups={(r["family_key"],r["initial"]) for r in traces}
    for key in groups:
        local=sorted([r for r in traces if (r["family_key"],r["initial"])==key],key=lambda r:float(r["time"]))
        if local[0]["mode"]!="window":continue
        anchor=min(local,key=lambda r:abs(float(r["aligned_time"])))
        assert abs(float(anchor["aligned_time"]))<1e-7
        d0=float(anchor["distance_to_G"])
        c=theory.constants(anchor["skin"])
        k0=d0/math.sqrt(1-d0*d0)
        assert k0<2*c["a"]/c["b"]
        for row in local:
            dt=float(row["time"])-float(anchor["time"])
            if dt<0:continue
            e=math.exp(-2*c["a"]*dt)
            k=k0*e/(1-c["b"]*k0/(2*c["a"])*(1-e))
            upper=k/math.sqrt(1+k*k)
            error=float(row["projector_remainder"])+float(anchor["projector_remainder"])
            assert float(row["distance_to_G"])<=upper*(1+1e-4)+2*error
            envelope_ratios.append(float(row["distance_to_G"])/upper)
    if complete:
        assert len(roots)==144 and len(traces)==541 and len(signed)==42
        assert len(predictions)==64 and len(references)==28 and len(pairs)==14
        assert all(n==2 for n in pairs.values())
        for initial in ("left","right"):
            sign=1 if initial=="left" else -1
            for chi in (-4.,-2.,-1.,1.,2.,4.):
                errors=[abs(float(next(r for r in signed if int(r["length"])==length and r["initial"]==initial
                              and float(r["chi"])==chi)["log_ratio_to_zero"])/chi-sign/2) for length in (64,512)]
                assert errors[1]<errors[0]
    return dict(status="passed",unique_roots=len(keys),trajectory_methods=dict(Counter(r["method"] for r in traces)),
                max_relative_trajectory_remainder=max(relative_errors,default=0.),
                max_L384_relative_trajectory_remainder=max(tail_relative_errors,default=0.),
                relative_remainder_scope="The L384 tail has an additional relative budget; an absolute remainder divided by an extremely small CDW distance can be large",
                independent_bracketed_roots=sum(n==2 for n in pairs.values()),
                persistence_samples=len(envelope_ratios),max_persistence_ratio=max(envelope_ratios,default=0.),
                scope="Converged numerical comparisons within existing exact-arithmetic bounds; no interval certificate or new fitted forecast gate")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial",action="store_true")
    args=parser.parse_args()
    print(json.dumps(aggregate(not args.partial),ensure_ascii=False))
