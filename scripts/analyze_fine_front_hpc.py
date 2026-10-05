"""Audit fine sampled crossings without assigning a continuous first passage."""

import argparse
import collections
import functools
import json
import re
from pathlib import Path

import run_fine_front_hpc as run
from analyze_selection_front_hpc import front

ROOT,OUT=run.ROOT,run.OUT
FIG=ROOT / "figures/prl_p3_fine_front"
read_json,write_json,write_csv=run.read_json,run.write_json,run.write_csv


@functools.lru_cache(maxsize=None)
def digest(path):
    return run.sha256(path)


def summarize(plots=True):
    run.check_lock()
    cfg=run.config()
    predicted={}
    memory={}
    for family in cfg["families"]:
        value=read_json(OUT / "predictions" / (family["key"]+".json"))
        predicted.update({(r["case_id"],r["initial"]):r for r in value["rows"]})
        memory.update({(r["case_id"],r["initial_a"],r["initial_b"]):r for r in value["pairs"]})
    lock=read_json(OUT / "prediction_lock.json")
    states,pairs,checks=[],[],[]
    for case in cfg["cases"]:
        path=OUT / "cases" / (case["case_id"]+".json")
        value=read_json(path)
        assert value["started_utc"]>lock["locked_utc"]
        assert value["prediction_lock_sha256"]==digest(OUT / "prediction_lock.json")
        assert run.old.previous.accepted_precision(value["reference_convergence"])
        certificate=read_json(OUT / "audits" / path.name)
        assert certificate["reference_sha256"]==digest(path)
        assert certificate["reference_raw_sha256"]==digest(path.with_suffix(".raw.json.gz"))
        assert certificate["prediction_raw_sha256"]==digest(OUT / "predictions" / (case["family_key"]+".raw.json.gz"))
        for check in certificate["rows"]:
            assert all(check[k] for k in ("remainder_covered","entropy_remainder_covered","distance_interval_covered","entropy_interval_covered","bases_orthogonal"))
        checks.extend(certificate["rows"])
        for actual in value["rows"]:
            prediction=predicted[case["case_id"],actual["initial"]]
            states.append({**prediction,**actual,
                           "projector_selected_actual":actual["distance_to_W"]<=.05,
                           "entropy_selected_actual":actual["entropy_error"]<=.05,
                           "wrong_guarantee":(prediction["guaranteed_selected"] and not actual["controlled"]) or
                                             (prediction["guaranteed_unselected"] and actual["controlled"]),
                           "rank1_misclassified":(prediction["rank1_distance_candidate"]<=.05)!=(actual["distance_to_W"]<=.05),
                           "rank1_distance_abs_error":abs(prediction["rank1_distance_candidate"]-actual["distance_to_W"])})
        for actual in value["pairs"]:
            prediction=memory[case["case_id"],actual["initial_a"],actual["initial_b"]]
            assert prediction["memory_lower"]-1e-10<=actual["pair_distance"]<=prediction["memory_upper"]+1e-10
            assert prediction["entropy_memory_lower"]-1e-10<=actual["absolute_entropy_difference"]<=prediction["entropy_memory_upper"]+1e-10
            assert not prediction["guaranteed_memory"] or actual["actual_memory"]
            assert not prediction["guaranteed_entropy_memory"] or actual["actual_entropy_memory"]
            pairs.append({**prediction,**actual})
    write_csv(OUT / "actual_entanglement.csv",states)
    write_csv(OUT / "initial_memory.csv",pairs)
    write_csv(OUT / "high_precision_remainder_checks.csv",checks)
    fronts=[]
    for hypothesis in read_json(OUT / "front_hypothesis_lock.json")["candidates"]:
        local=[r for r in states if r["family_key"]==hypothesis["family_key"] and r["initial"]==hypothesis["initial"]
               and r["physical_time"] in hypothesis["times"]]
        assert len(local)==len(hypothesis["times"])
        for quantity,key in (("projector","projector_selected_actual"),("joint","controlled")):
            result=front(local,key)
            lo,hi=result["last_fail"],result["first_persistent_sample_pass"]
            candidate=hypothesis["candidate_time"]
            bracketed=lo is not None and hi is not None
            error_to_bracket=None if not bracketed else max(lo-candidate,candidate-hi,0.)
            fronts.append({k:v for k,v in hypothesis.items() if k not in ("source","times")}|dict(
                quantity=quantity,**result,bracket_width=None if not bracketed else hi-lo,
                rank1_candidate_compatible=None if not bracketed else lo<candidate<=hi,
                distance_to_sampled_bracket=error_to_bracket,
                root_gate_not_falsified=bracketed and error_to_bracket<=hypothesis["root_abs_error_gate"],
                interpretation="sampled persistent bracket; no continuous first-crossing or monotonicity proof"))
    write_csv(OUT / "fine_front_intervals.csv",fronts)
    shifts=[]
    for length in (128,144):
        for initial in ("left","right"):
            baseline=next(r for r in fronts if r["family_key"]==f"L{length}_gL0" and r["initial"]==initial and r["quantity"]=="projector")
            for chi in (2,4):
                target=next(r for r in fronts if r["family_key"]==f"L{length}_gL{chi}" and r["initial"]==initial and r["quantity"]=="projector")
                values=[baseline["last_fail"],baseline["first_persistent_sample_pass"],target["last_fail"],target["first_persistent_sample_pass"]]
                valid=all(x is not None for x in values)
                lower=values[2]-values[1] if valid else None
                upper=values[3]-values[0] if valid else None
                candidate=target["candidate_time"]-baseline["candidate_time"]
                shifts.append(dict(length=length,initial=initial,gL=chi,candidate_shift=candidate,
                                   sampled_shift_lower=lower,sampled_shift_upper=upper,
                                   resolved_direction=valid and (upper<0 if initial=="left" else lower>0),
                                   candidate_compatible=valid and lower<=candidate<=upper,
                                   scope="difference of local sampled brackets; conditional on tracking the same crossing"))
    write_csv(OUT / "weak_skin_time_shifts.csv",shifts)
    pilots=[]
    for case in cfg["pilots"]:
        value=read_json(OUT / "pilots" / (case["case_id"]+".json"))
        assert run.old.previous.accepted_precision(value["reference_convergence"])
        for row in value["rows"]:
            pilots.append({**row,"case_seconds":value["seconds"],"precision_digits":value["precision_digits"],
                           "convergence":value["reference_convergence"],"role":"precision_cost_development"})
    write_csv(OUT / "size_precision_pilots.csv",pilots)
    responses=[]
    for family in cfg["families"]:
        value=read_json(OUT / "response" / (family["key"]+".json"))
        assert value["status"]=="passed"
        responses.extend(value["rows"])
    assert len(responses)==len(cfg["families"])*4
    write_csv(OUT / "geometry_response.csv",responses)
    groups={}
    for group in ("edge","translated"):
        local=[r for r in states if r["group"]==group]
        localpairs=[r for r in pairs if r["group"]==group]
        groups[group]=dict(states=len(local),pairs=len(localpairs),actual_selected=sum(r["controlled"] for r in local),
                           selected_guarantees=sum(r["guaranteed_selected"] for r in local),
                           unselected_guarantees=sum(r["guaranteed_unselected"] for r in local),
                           undecided=sum(not (r["guaranteed_selected"] or r["guaranteed_unselected"]) for r in local),
                           actual_memory=sum(r["actual_memory"] for r in localpairs),memory_guarantees=sum(r["guaranteed_memory"] for r in localpairs),
                           actual_entropy_memory=sum(r["actual_entropy_memory"] for r in localpairs),entropy_memory_guarantees=sum(r["guaranteed_entropy_memory"] for r in localpairs))
    translated_pairs=[r for r in pairs if r["initial_a"].startswith("block_cell")]
    projector_fronts=[r for r in fronts if r["quantity"]=="projector"]
    value=dict(date="2026-10-04",status="completed_and_audited",families=len(cfg["families"]),conditions=len(cfg["cases"]),
               states=len(states),pairs=len(pairs),size_pilot_conditions=len(cfg["pilots"]),size_pilot_states=len(pilots),groups=groups,
               wrong_guarantees=sum(r["wrong_guarantee"] for r in states),rank_target_passed=sum(r["rank_target_passed"] for r in states),
               rank_max=max(r["rank"] for r in states),rank1_misclassified=sum(r["rank1_misclassified"] for r in states),
               rank1_max_distance_error=max(r["rank1_distance_abs_error"] for r in states),
               projector_front_count=len(projector_fronts),projector_fronts_bracketed=sum(r["status"]=="sampled_bracket" for r in projector_fronts),
               projector_root_gate_not_falsified=sum(r["root_gate_not_falsified"] for r in projector_fronts),
               strict_rank1_front_compatibility=sum(r["rank1_candidate_compatible"] is True for r in projector_fronts),
               weak_shift_checks=len(shifts),weak_shifts_direction_resolved=sum(r["resolved_direction"] for r in shifts),
               weak_shift_candidates_compatible=sum(r["candidate_compatible"] for r in shifts),
               translated_pair_memory=sum(r["actual_memory"] for r in translated_pairs),
               translated_pair_memory_guarantees=sum(r["guaranteed_memory"] for r in translated_pairs),
               translated_pair_entropy_memory=sum(r["actual_entropy_memory"] for r in translated_pairs),
               projector_error_max=max(float(r["projector_error_mp"]) for r in checks),
               entropy_error_max=max(float(r["entropy_error_mp"]) for r in checks),
               entropy_selected_projector_unselected=sum(r["entropy_selected_actual"] and not r["projector_selected_actual"] for r in states),
               response_checks=len(responses),response_chi_fd_error_max=max(r["chi_fd_error"] for r in responses),
               response_time_fd_error_max=max(r["time_fd_error"] for r in responses),
               reciprocal_edge_response=[r for r in responses if r["gL"]==0 and r["group"]=="edge"],
               scope=cfg["scope"])
    assert value["wrong_guarantees"]==0
    write_json(OUT / "summary.json",value)
    write_json(OUT / "next_round_plan.json",dict(status="candidate_not_executed",source="eighth round",
               priorities=["explain any fine-front rank1 failures with a separately frozen signed rank2 candidate",
                           "derive reflected inverse-Gram leading exponent and finite prefactor",
                           "validate a second passive local no-click model with Hermitian adjoints",
                           "extend size pilots only after measured precision/cost gates"],
               fixed_tolerances=cfg["selection_tolerances"],memory_tolerances=cfg["memory_tolerances"],
               gates=["all eighth-round observations become development","no retrospective holdout","no continuous first passage from samples","no asymptotic law from two new sizes"]))
    if plots:plot(states,pairs,fronts,shifts)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def response_job(index):
    mp=run.mp
    family=run.config()["families"][index]
    hypotheses=[r for r in read_json(OUT / "front_hypothesis_lock.json")["candidates"] if r["family_key"]==family["key"]]
    static=read_json(run.old.OUT / "static" / (family["key"]+".json"))
    rows=[]
    snapshot=OUT / "code" / ("analysis_"+run.sha256(Path(__file__)).lower()+".py")
    if not snapshot.exists():snapshot.write_bytes(Path(__file__).read_bytes())
    with mp.workdps(160):
        gain=run.old.spatial.decode(static["matrices"]["G"])
        perpendicular=run.old.spatial.decode(static["matrices"]["Gperp"])
        length=family["length"]
        centered=[(mp.mpf(i//2)-mp.mpf(length//2-1)/2)/length for i in range(length)]
        step=mp.mpf("0.0001")
        for hypothesis in hypotheses:
            receipts=[]
            for case in run.config()["cases"]:
                if case["family_key"]==family["key"] and case["physical_time"] in hypothesis["times"] and hypothesis["initial"] in case["initials"]:
                    path=OUT / "cases" / (case["case_id"]+".json")
                    receipt=read_json(path)
                    actual=next(r for r in receipt["rows"] if r["initial"]==hypothesis["initial"])
                    receipts.append((abs(actual["distance_to_W"]-.05),case,path))
            _,case,path=min(receipts,key=lambda x:x[0])
            raw=run.read_gzip(path.with_suffix(".raw.json.gz"))[-1]
            q=run.old.spatial.decode(raw["states"][hypothesis["initial"]]["Q"])
            left,singular,right=mp.svd(perpendicular.T*q)
            s=singular[0]
            gap=s-singular[1]
            assert gap>mp.mpf("1e-12") and 0<s<1
            hdir=perpendicular*left[:,0]
            gdir=(q*right.T[:,0]-s*hdir)/mp.sqrt(1-s*s)
            vector=mp.sqrt((1-s)/2)*gdir+mp.sqrt((1+s)/2)*hdir
            pq=q*(q.T*vector)
            pg=gain*(gain.T*vector)
            ypq=mp.matrix([centered[i]*pq[i] for i in range(length)])
            ypg=mp.matrix([centered[i]*pg[i] for i in range(length)])
            chi_derivative=2*((vector-pq).T*ypq-(vector-pg).T*ypg)[0]
            generator=run.old.generator(case)
            time_derivative=2*((vector-pq).T*(generator*pq))[0]
            sensitivity=-chi_derivative/time_derivative
            perturbed=[]
            for sign in (-1,1):
                weights=[mp.exp(sign*step*x) for x in centered]
                qchi=mp.qr(mp.matrix([[weights[i]*q[i,j] for j in range(q.cols)] for i in range(length)]),mode="skinny")[0]
                gchi=mp.qr(mp.matrix([[weights[i]*gain[i,j] for j in range(gain.cols)] for i in range(length)]),mode="skinny")[0]
                qt=mp.qr(mp.expm(sign*step*generator)*q,mode="skinny")[0]
                perturbed.append((run.old.pair_distance(qchi,gchi),run.old.pair_distance(qt,gain)))
            fdchi=(perturbed[1][0]-perturbed[0][0])/(2*step)
            fdtime=(perturbed[1][1]-perturbed[0][1])/(2*step)
            errors=(abs(fdchi-chi_derivative),abs(fdtime-time_derivative))
            assert max(errors)<mp.mpf("1e-7"),(family["key"],hypothesis["initial"],errors)
            rows.append(dict(family_key=family["key"],length=length,initial=hypothesis["initial"],group=hypothesis["group"],
                             skin=float(run.old.skin_mp(family)),gL=float(run.old.skin_mp(family)*length),
                             case_id=case["case_id"],physical_time=case["physical_time"],gain_target_distance=float(s),
                             distance_to_W=float(mp.mpf(raw["states"][hypothesis["initial"]]["distance_to_W"])),
                             principal_gap=float(gap),chi_derivative=float(chi_derivative),time_derivative=float(time_derivative),
                             implicit_time_sensitivity=float(sensitivity),chi_fd_error=float(errors[0]),time_fd_error=float(errors[1]),
                             reference_sha256=digest(path),reference_raw_sha256=digest(path.with_suffix(".raw.json.gz"))))
    write_json(OUT / "response" / (family["key"]+".json"),dict(rows=rows,precision_digits=160,finite_difference_step="0.0001",
               absolute_derivative_gate="1e-7",status="passed",source_sha256=digest(Path(__file__)),verified_utc=run.old.now(),
               policy="post-truth mechanism diagnosis; gain-subspace target; not a new frozen predictor or a first-passage theorem"))
    print(json.dumps(dict(status="response_checked",family=family["key"],states=len(rows))),flush=True)


def plot(states,pairs,fronts,shifts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.18,"axes.spines.top":False,"axes.spines.right":False})
    colors={0:"#455a64",2:"#00897b",4:"#c62828"}
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for row,length in enumerate((128,144)):
        for col,initial in enumerate(("left","right")):
            ax=axes[row,col]
            baseline=next(r["candidate_time"] for r in fronts if r["family_key"]==f"L{length}_gL0" and r["initial"]==initial)
            for chi,color in colors.items():
                local=sorted([r for r in states if r["family_key"]==f"L{length}_gL{chi}" and r["initial"]==initial],key=lambda r:float(r["physical_time"]))
                ax.plot([float(r["physical_time"])-baseline for r in local],[r["distance_to_W"] for r in local],"o-",c=color,label=f"gL = {chi}",markersize=4)
                ax.plot([float(r["physical_time"])-baseline for r in local],[r["rank1_distance_candidate"] for r in local],"--",c=color,lw=1)
            ax.axhline(.05,c="#333333",ls=":")
            ax.set(title=f"L={length}, {initial}",yscale="log",ylim=(.002,1.2),xlabel="t - reciprocal candidate time",ylabel="Projector distance")
    axes[0,0].legend(fontsize=9)
    fig.savefig(FIG / "fine_front_distances.png",dpi=190)
    fig.savefig(FIG / "fine_front_distances.pdf")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6),layout="constrained")
    for initial,offset,color in (("left",-.07,"#00897b"),("right",.07,"#c62828")):
        for chi,marker in ((2,"o"),(4,"s")):
            local=[r for r in shifts if r["initial"]==initial and r["gL"]==chi and r["sampled_shift_lower"] is not None]
            centers=[(r["sampled_shift_lower"]+r["sampled_shift_upper"])/2 for r in local]
            ax.errorbar([r["length"]+offset for r in local],centers,yerr=[(r["sampled_shift_upper"]-r["sampled_shift_lower"])/2 for r in local],fmt=marker,c=color,capsize=5,label=f"{initial}, gL={chi}")
            ax.scatter([r["length"]+offset for r in local],[r["candidate_shift"] for r in local],marker="x",c=color,s=40)
    ax.axhline(0,c="#333333",ls="--",lw=.8)
    ax.set(xlabel="System size L",ylabel="Shift from reciprocal sampled crossing",xticks=(128,144),xlim=(125,147))
    ax.legend(ncol=2,fontsize=9)
    fig.savefig(FIG / "weak_skin_shift_intervals.png",dpi=190)
    fig.savefig(FIG / "weak_skin_shift_intervals.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    for length,marker in ((128,"o"),(144,"s")):
        for protocol,color in (("gL0","#455a64"),("g0p25","#7551a8")):
            local=sorted([r for r in pairs if r["family_key"]==f"L{length}_{protocol}" and r["initial_a"].startswith("block_cell")],key=lambda r:float(r["physical_time"]))
            for ax,key in zip(axes,("pair_distance","absolute_entropy_difference")):
                ax.plot([float(r["physical_time"])/length for r in local],[r[key] for r in local],marker+"-",c=color,label=f"L={length}, {protocol}",markersize=4)
                ax.axhline(.1,c="#333333",ls="--",lw=.7)
    axes[0].set(xlabel="t / L",ylabel="Translated pair projector distance")
    axes[1].set(xlabel="t / L",ylabel="Translated pair entropy difference (nat)")
    axes[0].legend(fontsize=8)
    fig.savefig(FIG / "translated_memory_windows.png",dpi=190)
    fig.savefig(FIG / "translated_memory_windows.pdf")
    plt.close(fig)


def verify():
    run.check_lock()
    summary=read_json(OUT / "summary.json")
    assert summary["states"]==len(run.config()["cases"])*3 and summary["wrong_guarantees"]==0
    history=run.old.audit.read_csv(OUT / "input_manifest.csv")
    for row in history:
        path=ROOT / row["path"]
        assert path.stat().st_size==int(row["bytes"]) and digest(path)==row["sha256"],row["path"]
    for row in read_json(OUT / "prediction_lock.json")["inputs"]:
        assert digest(ROOT / row["path"])==row["sha256"],row["path"]
    generated={(OUT / name).resolve() for name in ("artifact_manifest.csv","delivery_verification.json")}
    deferred=[]
    links=0
    documents=[ROOT / "README.md",*(ROOT / "research_doc").rglob("*.md")]
    for path in documents:
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
            target=match.group(1).strip("<>").split("#",1)[0]
            if not target or re.match(r"[a-z]+://",target) or target.startswith("mailto:"):continue
            resolved=(path.parent / target).resolve()
            if resolved in generated:deferred.append(resolved)
            else:assert resolved.exists(),(path,target)
            links+=1
    excluded={"artifact_manifest.csv","delivery_verification.json","deployment.tar.gz","final_sync.tar.gz","results.tar.gz",
              "return_manifest.csv","final_sync_manifest.csv","final_sync_receipt.json","local_inventory.json","operations.jsonl"}
    paths=[p for folder in (OUT,FIG) for p in folder.rglob("*") if p.is_file() and p.name not in excluded]
    paths.extend(p for p in (ROOT / "scripts").glob("*fine_front*") if p.is_file())
    paths.extend(documents)
    rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(paths))]
    write_csv(OUT / "artifact_manifest.csv",rows)
    value=dict(status="passed",verified_utc=run.old.now(),historical_inputs=len(history),artifact_files=len(rows),states=summary["states"],
               pairs=summary["pairs"],pilot_states=summary["size_pilot_states"],local_document_links=links,source_sha256=digest(run.SOURCE),wrong_guarantees=0)
    write_json(OUT / "delivery_verification.json",value)
    assert all(p.is_file() for p in deferred)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("summarize","verify","response"))
    parser.add_argument("--no-plots",action="store_true")
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    if args.stage=="summarize":summarize(not args.no_plots)
    elif args.stage=="response":response_job(args.index)
    else:verify()


if __name__=="__main__":main()
