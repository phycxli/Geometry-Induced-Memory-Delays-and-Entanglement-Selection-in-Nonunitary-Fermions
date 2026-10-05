"""Audit seventh-round frozen predictions, sampled fronts and Slurm artifacts."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp
import numpy as np

import run_selection_front_hpc as run

ROOT,OUT=run.ROOT,run.OUT
FIG=ROOT / "figures/prl_p3_selection_front"
read_json,write_json,write_csv=run.read_json,run.write_json,run.write_csv


def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def grouped(rows,keys):
    result=collections.defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def front(rows,boolean):
    ordered=sorted(rows,key=lambda r:float(r["physical_time"]))
    outcomes=[bool(r[boolean]) for r in ordered]
    if not outcomes[-1]:
        return dict(status="right_censored",last_fail=float(ordered[-1]["physical_time"]),first_persistent_sample_pass=None,persistence_samples=0)
    index=len(outcomes)-1
    while index>0 and outcomes[index-1]:
        index-=1
    return dict(status="left_censored" if index==0 else "sampled_bracket",last_fail=None if index==0 else float(ordered[index-1]["physical_time"]),
                first_persistent_sample_pass=float(ordered[index]["physical_time"]),persistence_samples=len(outcomes)-index)


def summarize(plots=True):
    run.check_lock()
    config=run.config()
    predictions={}
    for family in config["families"]:
        for row in read_json(OUT / "predictions" / (family["key"]+".json"))["rows"]:
            predictions[row["case_id"],row["initial"]]=row
    memory_predictions={}
    for family in config["families"]:
        for row in read_json(OUT / "predictions" / (family["key"]+".json"))["pairs"]:
            memory_predictions[row["case_id"],row["initial_a"],row["initial_b"]]=row
    states,pairs,checks=[],[],[]
    worst=dict(projector_reduced_error=0.,entropy_reduced_error=0.,scalar_precision_change=0.,projector_precision_change=0.)
    locks=read_json(OUT / "prediction_lock.json")
    for case in config["cases"]:
        directory="pilot" if case["role"]=="precision_cost_pilot" else "cases"
        path=OUT / directory / (case["case_id"]+".json")
        value=read_json(path)
        if directory=="cases":
            assert value["started_utc"]>locks["locked_utc"]
            assert value["prediction_lock_sha256"]==run.audit.sha256(OUT / "prediction_lock.json")
        assert run.previous.accepted_precision(value["reference_convergence"])
        worst["scalar_precision_change"]=max(worst["scalar_precision_change"],value["reference_convergence"]["scalar_relative"])
        worst["projector_precision_change"]=max(worst["projector_precision_change"],value["reference_convergence"]["Q"])
        audit_path=OUT / "audits" / (case["case_id"]+".json")
        certificate=read_json(audit_path)
        assert certificate["reference_sha256"]==run.audit.sha256(path)
        assert certificate["reference_raw_sha256"]==run.audit.sha256(path.with_suffix(".raw.json.gz"))
        assert certificate["prediction_raw_sha256"]==run.audit.sha256(OUT / "predictions" / (case["family_key"]+".raw.json.gz"))
        localchecks={r["initial"]:r for r in certificate["rows"]}
        for actual in value["rows"]:
            initial=actual["initial"]
            predicted=predictions[case["case_id"],initial]
            check=localchecks[initial]
            assert all(check[k] for k in ("remainder_covered","entropy_remainder_covered","distance_interval_covered","entropy_interval_covered","bases_orthogonal"))
            distance,entropy_error=float(check["projector_error_mp"]),float(check["entropy_error_mp"])
            row={**predicted,**actual,"projector_selected_actual":actual["distance_to_W"]<=.05,"entropy_selected_actual":actual["entropy_error"]<=.05,
                 "projector_reduced_error":distance,"entropy_reduced_error":entropy_error,
                 "wrong_selected_guarantee":predicted["guaranteed_selected"] and not actual["controlled"],
                 "wrong_unselected_guarantee":predicted["guaranteed_unselected"] and actual["controlled"],
                 "rank1_projector_prediction":predicted["rank1_distance_candidate"]<=.05,
                 "rank1_absolute_distance_error":abs(predicted["rank1_distance_candidate"]-actual["distance_to_W"])}
            states.append(row)
            checks.append(check)
            worst["projector_reduced_error"]=max(worst["projector_reduced_error"],distance)
            worst["entropy_reduced_error"]=max(worst["entropy_reduced_error"],entropy_error)
        for actual in value["pairs"]:
            predicted=memory_predictions[case["case_id"],actual["initial_a"],actual["initial_b"]]
            assert predicted["memory_lower"]-1e-10<=actual["pair_distance"]<=predicted["memory_upper"]+1e-10
            assert predicted["entropy_memory_lower"]-1e-10<=actual["absolute_entropy_difference"]<=predicted["entropy_memory_upper"]+1e-10
            assert not predicted["guaranteed_memory"] or actual["actual_memory"]
            assert not predicted["guaranteed_entropy_memory"] or actual["actual_entropy_memory"]
            pairs.append({**predicted,**actual})
    write_csv(OUT / "actual_entanglement.csv",states)
    write_csv(OUT / "initial_memory.csv",pairs)
    write_csv(OUT / "high_precision_remainder_checks.csv",checks)
    fronts=[]
    for (family,initial),local in grouped(states,("family_key","initial")).items():
        for quantity,key,pkey in (("projector","projector_selected_actual","projector_selected"),("entropy","entropy_selected_actual","entropy_selected"),("joint","controlled","guaranteed_selected")):
            actual,predicted=front(local,key),front(local,pkey)
            fronts.append(dict(family_key=family,initial=initial,quantity=quantity,length=local[0]["length"],skin=local[0]["skin"],
                               **actual,prediction_status=predicted["status"],prediction_last_fail=predicted["last_fail"],
                               prediction_first_pass=predicted["first_persistent_sample_pass"],
                               scope="sampled persistence only; pilot/development points identified in state table"))
    write_csv(OUT / "sampled_selection_fronts.csv",fronts)
    rank1fronts=[]
    for family in config["families"]:
        value=read_json(OUT / "predictions" / (family["key"]+".json"))
        for candidate in value["rank1_front_candidates"]:
            actual=next(r for r in fronts if r["family_key"]==family["key"] and r["initial"]==candidate["initial"] and r["quantity"]=="projector")
            center=(candidate["lower"]+candidate["upper"])/2 if candidate["lower"] is not None and candidate["upper"] is not None else None
            compatible=None if center is None or actual["last_fail"] is None or actual["first_persistent_sample_pass"] is None else actual["last_fail"]<center<=actual["first_persistent_sample_pass"]
            rank1fronts.append(dict(family_key=family["key"],length=family["length"],**candidate,root_time_candidate=center,
                                   root_fraction_candidate=None if center is None else center/family["length"],
                                   actual_status=actual["status"],actual_last_fail=actual["last_fail"],actual_first_persistent_pass=actual["first_persistent_sample_pass"],
                                   compatible_with_sampled_projector_bracket=compatible))
    write_csv(OUT / "rank1_front_candidate_checks.csv",rank1fronts)
    memory_fronts=[]
    for (family,na,nb),local in grouped(pairs,("family_key","initial_a","initial_b")).items():
        for quantity,akey,pkey,upper in (("projector","actual_memory","guaranteed_memory","memory_upper"),("entropy","actual_entropy_memory","guaranteed_entropy_memory","entropy_memory_upper")):
            ordered=sorted(local,key=lambda r:float(r["physical_time"]))
            memory_fronts.append(dict(family_key=family,initial_a=na,initial_b=nb,quantity=quantity,
                                      actual_memory_samples=sum(r[akey] for r in ordered),guaranteed_memory_samples=sum(r[pkey] for r in ordered),
                                      last_sample_time=float(ordered[-1]["physical_time"]),last_actual_memory=ordered[-1][akey],
                                      last_memory_upper=ordered[-1][upper],last_guaranteed_no_memory=ordered[-1][upper]<=.1))
    write_csv(OUT / "sampled_memory_fronts.csv",memory_fronts)
    spatial_checks=[]
    for prediction in csv_rows(OUT / "locked_spatial_predictions.csv"):
        static=read_json(OUT / "static" / (prediction["family_key"]+".json"))
        with mp.workdps(360):
            delta=mp.mpf(static["states"][prediction["initial"]]["cross_block_min"])
            source=read_json(OUT / "spatial" / f"L{static['length']}.json")
            values=next(r["values"] for r in source["raw"] if r["family_key"]==prediction["family_key"] and r["initial"]==prediction["initial"])
            lower,upper=mp.mpf(values["lower"]),mp.mpf(values["upper"])
            covered=lower<=delta<=upper
            assert covered,(prediction,delta)
            spatial_checks.append({**prediction,"covered":covered,"delta_mp":str(delta),"delta":float(delta),"delta_relative_position":float((delta-lower)/(upper-lower))})
    write_csv(OUT / "spatial_inverse_trace_checks.csv",spatial_checks)
    size_checks=[]
    for row in read_json(OUT / "spatial/size_candidate_lock.json")["predictions"]:
        key=run.family_key(row["length"],"gL" if row["skin"]==0 else "g","0" if row["skin"]==0 else "0.25")
        static=read_json(OUT / "static" / (key+".json"))
        with mp.workdps(360):
            actual_log=float(mp.log(mp.mpf(static["states"][row["initial"]]["cross_block_min"])))
        error=row["predicted_log_delta"]-actual_log
        size_checks.append({**row,"actual_log_delta":actual_log,"signed_log_error":error,"predicted_over_actual":float(np.exp(error)),
                            "candidate_passed":abs(error)<=row["falsification_abs_log_threshold"],
                            "pilot_family":key in ("L128_gL0","L144_g0p25")})
    write_csv(OUT / "size_candidate_checks.csv",size_checks)
    subsets={}
    for role,local in grouped(states,("role",)).items():
        localpairs=[r for r in pairs if r["role"]==role[0]]
        subsets[role[0]]=dict(states=len(local),conditions=len({r["case_id"] for r in local}),
                             actual_selected=sum(r["controlled"] for r in local),selected_guarantees=sum(r["guaranteed_selected"] for r in local),
                             unselected_guarantees=sum(r["guaranteed_unselected"] for r in local),
                             undecided=sum(not (r["guaranteed_selected"] or r["guaranteed_unselected"]) for r in local),
                             wrong_guarantees=sum(r["wrong_selected_guarantee"] or r["wrong_unselected_guarantee"] for r in local),
                             rank_target_passed=sum(r["rank_target_passed"] for r in local),rank_max=max(r["rank"] for r in local),
                             pairs=len(localpairs),actual_memory=sum(r["actual_memory"] for r in localpairs),memory_guarantees=sum(r["guaranteed_memory"] for r in localpairs),
                             radial_memory_guarantees=sum(r["radial_lower"]>.1 for r in localpairs),
                             actual_entropy_memory=sum(r["actual_entropy_memory"] for r in localpairs),entropy_memory_guarantees=sum(r["guaranteed_entropy_memory"] for r in localpairs),
                             rank1_projector_misclassified=sum(r["rank1_projector_prediction"]!=r["projector_selected_actual"] for r in local),
                             rank1_max_distance_error=max(r["rank1_absolute_distance_error"] for r in local))
    benchmark=read_json(OUT / "development/benchmark_gmpy.json")
    pybenchmark=read_json(OUT / "development/benchmark_python.json")
    backend_change=max(abs(a[k]-b[k]) for a,b in zip(benchmark["rows"],pybenchmark["rows"]) for k in ("absolute_scalar_change","projector_change","directional_transverse_remainder"))
    production=[r for r in states if r["role"]=="new_size"]
    productionpairs=[r for r in pairs if r["role"]=="new_size"]
    failures=[read_json(p) for p in (OUT / "failures").glob("*.json")]
    value=dict(status="completed_and_audited",date="2026-10-04",families=len(config["families"]),conditions=len(config["cases"]),states=len(states),pairs=len(pairs),
               subsets=subsets,worst=worst,new_size_rank_distribution=dict(collections.Counter(r["rank"] for r in production)),
               spatial_intervals=len(spatial_checks),spatial_intervals_covered=sum(r["covered"] for r in spatial_checks),
               spatial_max_relative_width=max(float(r["relative_width"]) for r in spatial_checks),
               size_candidate_passed=sum(r["candidate_passed"] for r in size_checks),size_candidate_count=len(size_checks),
               size_candidate_max_abs_log_error=max(abs(r["signed_log_error"]) for r in size_checks),
               backend_numeric_change=backend_change,optimized_fixed_rank_speedups=[r["speedup"] for r in benchmark["rows"]],lu_speedup=benchmark["lu"]["speedup"],
               preserved_computational_failures=len(failures),
               new_size_joint_fronts=[r for r in fronts if r["length"]>=128 and r["quantity"]=="joint" and r["initial"] in ("left","right")],
               entropy_selection_without_projector=sum(r["entropy_selected_actual"] and not r["projector_selected_actual"] for r in production),
               new_size_direct_vs_radial_gain=sum(r["guaranteed_memory"] and r["radial_lower"]<=.1 for r in productionpairs),
               scope="finite sampled conditional dynamics with static spectrum preparation; no asymptotic exponent or continuous cutoff or second model")
    write_json(OUT / "summary.json",value)
    write_json(OUT / "next_round_plan.json",dict(status="candidate_not_executed",source="seventh round",
        objective="derive a parameter law with controlled finite-size errors and verify a second local no-click model",
        observed_gates=dict(new_size_states=270,projector_and_entropy_memory_separate=True,HPC_execution_done=True,validated_through_L=144,
                            rank1_misclassified=subsets["new_size"]["rank1_projector_misclassified"],asymptotic_exponent_proved=False,second_model_validated=False),
        priorities=[
            dict(name="weak_skin_front_resolution",tasks=["use all seventh-round values as development", "freeze finer time brackets around the gL0/2/4 crossings before new truth", "evaluate an independently frozen rank2/signed-factor candidate if rank1 failed"]),
            dict(name="analytic_size_rate",tasks=["analyze reflected inverse-Gram extremal vectors instead of replacing finite boundary terms", "derive the leading exponent and finite prefactor or falsify the present empirical size fit", "freeze L160/192 precision pilots and test the squared-conditioning diagnostic"]),
            dict(name="second_local_model",tasks=["define a passive no-click quadratic chain with additional same-sublattice hopping and explicit scalar loss shift", "verify a growth/decay separation and generalize real transposes to Hermitian adjoints", "compare fixed natural initial states at independent precision, with frozen selection and memory tolerances"]),
            dict(name="manuscript_and_novelty",tasks=["compare specific contributions with Mori/Shirai, Vernier and Le Gal original texts", "draft the main/supplement after the parameter law and second-model gate", "quantify measurable proxies and no-click postselection costs before claiming an experiment"])
        ],fixed_tolerances=config["selection_tolerances"],memory_tolerances=config["memory_tolerances"],
        gates=["preserve all seven rounds and failures", "no retrospective holdout", "no continuous cutoff from sampled persistence", "no universal law from eight size checks", "no Born trajectory or experiment implementation claim"]))
    if plots:
        plot(states,pairs,fronts,size_checks)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def plot(states,pairs,fronts,size_checks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.18,"axes.spines.top":False,"axes.spines.right":False})
    colors={"gL0":"#455a64","gL2":"#00897b","gL4":"#c62828","g0p25":"#7551a8"}
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True,layout="constrained")
    for row,length in enumerate((128,144)):
        for col,initial in enumerate(("left","right")):
            ax=axes[row,col]
            for protocol,color in colors.items():
                local=sorted([r for r in states if r["family_key"]==f"L{length}_{protocol}" and r["initial"]==initial],key=lambda r:float(r["physical_time"]))
                x=np.array([float(r["time_fraction"]) for r in local])
                y=np.array([r["distance_to_W"] for r in local])
                label="g = 0.25" if protocol=="g0p25" else protocol.replace("gL","gL = ")
                ax.plot(x,np.maximum(y,1e-14),"o-",color=color,label=label,markersize=4)
                ax.vlines(x,[max(r["distance_lower"],1e-14) for r in local],[max(r["distance_upper"],1e-14) for r in local],color=color,lw=2)
            ax.axhline(.05,color="#303030",ls="--",lw=1)
            ax.set(yscale="log",ylim=(1e-14,1.5),title=f"L={length}, {initial} block")
            if row==1:ax.set_xlabel("t / L")
            if col==0:ax.set_ylabel("Projector distance to output subspace")
    axes[0,0].legend(ncol=2,fontsize=9)
    fig.savefig(FIG / "weak_skin_selection_fronts.png",dpi=190)
    fig.savefig(FIG / "weak_skin_selection_fronts.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    production=[r for r in pairs if r["role"]=="new_size"]
    for label,condition,color in (("left-right",lambda r:r["initial_a"]=="left","#c62828"),("translated",lambda r:r["initial_a"].startswith("block_cell"),"#00897b"),("CDW-block",lambda r:r["initial_a"]=="charge_density_wave","#455a64")):
        local=[r for r in production if condition(r)]
        axes[0].scatter([r["pair_distance"] for r in local],[r["memory_lower"] for r in local],s=16,c=color,label=label,alpha=.65)
        axes[1].scatter([r["radial_lower"] for r in local],[r["memory_lower"] for r in local],s=16,c=color,alpha=.65)
        axes[2].scatter([r["absolute_entropy_difference"] for r in local],[r["entropy_memory_lower"] for r in local],s=16,c=color,alpha=.65)
    for ax in axes:
        ax.axhline(.1,color="#303030",ls="--",lw=.8)
    axes[0].plot([0,1],[0,1],color="#999999",lw=.7)
    axes[1].plot([0,1],[0,1],color="#999999",lw=.7)
    axes[0].set(xlabel="Actual pair projector distance",ylabel="Direct projector lower bound",xlim=(-.02,1.05),ylim=(-.02,1.05))
    axes[1].set(xlabel="Radial lower bound",ylabel="Direct lower bound",xlim=(-.02,1.05),ylim=(-.02,1.05))
    axes[2].set(xlabel="Actual absolute entropy difference (nat)",ylabel="Entropy lower bound (nat)")
    axes[0].legend(fontsize=8)
    fig.savefig(FIG / "projector_and_entropy_memory.png",dpi=190)
    fig.savefig(FIG / "projector_and_entropy_memory.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for initial,color in (("left","#00897b"),("right","#c62828")):
        for skin,marker in ((0.,"o"),(.25,"s")):
            local=[r for r in size_checks if r["initial"]==initial and r["skin"]==skin]
            axes[0].plot([r["length"] for r in local],[r["signed_log_error"] for r in local],marker+"-",c=color,label=f"{initial}, g={skin}")
    axes[0].axhline(0,c="#333333",ls="--",lw=.7)
    axes[0].set(xlabel="New size L",ylabel="Frozen size candidate log error",ylim=(-.003,.003))
    axes[0].legend(fontsize=8)
    new=[r for r in states if r["role"]=="new_size" and r["initial"] in ("left","right")]
    for initial,color in (("left","#00897b"),("right","#c62828")):
        local=[r for r in new if r["initial"]==initial]
        axes[1].scatter([r["distance_to_W"] for r in local],[r["rank1_distance_candidate"] for r in local],c=color,s=17,alpha=.65,label=initial)
    axes[1].plot([0,1],[0,1],c="#999999",lw=.7)
    axes[1].set(xlabel="Actual projector distance",ylabel="Signed rank1 scalar candidate",xlim=(-.02,1.05),ylim=(-.02,1.05))
    axes[1].legend(fontsize=9)
    fig.savefig(FIG / "parameter_candidates.png",dpi=190)
    fig.savefig(FIG / "parameter_candidates.pdf")
    plt.close(fig)


def verify():
    summary=read_json(OUT / "summary.json")
    assert summary["states"]==400 and summary["pairs"]==480
    assert summary["subsets"]["new_size"]["states"]==270
    assert all(v["wrong_guarantees"]==0 for v in summary["subsets"].values())
    run.check_lock()
    for row in read_json(OUT / "prediction_lock.json")["inputs"]:
        assert run.audit.sha256(ROOT / row["path"])==row["sha256"],row["path"]
    history=csv_rows(OUT / "input_manifest.csv")
    for row in history:
        path=ROOT / row["path"]
        assert path.stat().st_size==int(row["bytes"]) and run.audit.sha256(path)==row["sha256"],row["path"]
    spatial=read_json(OUT / "spatial_lock.json")
    for row in spatial["inputs"]:
        assert run.audit.sha256(ROOT / row["path"])==row["sha256"]
    for family in run.config()["families"]:
        value=read_json(OUT / "static" / (family["key"]+".json"))
        assert value["started_utc"]>spatial["locked_utc"]
        assert max(value["reference_convergence"]["relative_scalar"],value["reference_convergence"]["additional_relative"])<=1e-8
        assert max(value["reference_convergence"]["matrix_absolute"],value["reference_convergence"]["gain_entropy"])<=1e-10
    links=0
    generated_targets={(OUT / name).resolve() for name in ("delivery_verification.json","artifact_manifest.csv")}
    deferred_links=[]
    for path in [ROOT / "README.md",*(ROOT / "research_doc").rglob("*.md")]:
        content=path.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)",content):
            target=match.group(1).strip("<>").split("#",1)[0]
            if not target or re.match(r"[a-z]+://",target) or target.startswith("mailto:"):
                continue
            resolved=(path.parent / target).resolve()
            if resolved in generated_targets:
                deferred_links.append((path,resolved))
            else:
                assert resolved.exists(),(path,target)
            links+=1
    inputs=read_json(OUT / "prediction_lock.json")["inputs"]
    value=dict(status="passed",verified_utc=datetime.now(timezone.utc).isoformat(),historical_inputs=len(history),locked_inputs=len(inputs),
               spatial_inputs=len(spatial["inputs"]),states=summary["states"],pairs=summary["pairs"],new_size_states=270,
               all_precision_gates=True,all_pointwise_intervals=True,all_remainder_gates=True,wrong_guarantees=0,local_document_links=links,
               source_sha256=run.audit.sha256(run.SOURCE))
    write_json(OUT / "delivery_verification.json",value)
    paths=[p for directory in (OUT,FIG) for p in directory.rglob("*") if p.is_file() and p.name not in
           ("artifact_manifest.csv","delivery_verification.json","deployment.tar.gz","final_sync.tar.gz","results.tar.gz","return_manifest.csv","final_sync_manifest.csv","final_sync_receipt.json","local_inventory.json","operations.jsonl")]
    paths.extend(p for p in (ROOT / "scripts").glob("*selection_front*") if p.is_file())
    paths.extend([ROOT / "README.md",*(ROOT / "research_doc").rglob("*.md")])
    rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=run.audit.sha256(p)) for p in sorted(set(paths))]
    write_csv(OUT / "artifact_manifest.csv",rows)
    for row in rows:
        path=ROOT / row["path"]
        assert path.stat().st_size==row["bytes"] and run.audit.sha256(path)==row["sha256"]
    for source,target in deferred_links:
        assert target.is_file(),(source,target)
    print(json.dumps(dict(**value,artifact_files=len(rows)),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("summarize","plots","verify"))
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    if args.stage=="summarize":summarize(not args.no_plots)
    elif args.stage=="plots":
        tables=[]
        for name in ("actual_entanglement.csv","initial_memory.csv","sampled_selection_fronts.csv","size_candidate_checks.csv"):
            rows=csv_rows(OUT / name)
            for row in rows:
                for key,value in row.items():
                    if value in ("True","False"):
                        row[key]=value=="True"
                    elif value=="":
                        row[key]=None
                    else:
                        try:row[key]=float(value)
                        except ValueError:pass
            tables.append(rows)
        plot(*tables)
    else:verify()


if __name__=="__main__":
    main()
