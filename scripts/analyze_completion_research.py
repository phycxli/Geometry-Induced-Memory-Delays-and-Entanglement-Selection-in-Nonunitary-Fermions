"""Final research analysis: distinct guarantees, reflected geometry and scope."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import mpmath as mp
import numpy as np

import run_rank2_front_hpc as ninth

io=ninth.old
ROOT=io.ROOT
SIZES=ROOT/"data/prl_p3_new_sizes"
COMPLEX=ROOT/"data/prl_p4_passive_complex"
FINAL=ROOT/"data/prl_w_completion"
FIG=ROOT/"figures/prl_w_completion"
EXPERIMENT=ROOT/"data/prl_e_feasibility"
read_json,write_json,write_csv=io.read_json,io.write_json,io.write_csv


def summarize_study(study):
    cfg=read_json(study/"config.json")
    predicted=[read_json(study/"predictions"/(f["key"]+".json")) for f in cfg["families"]]
    pr={(r["case_id"],r["initial"]):r for v in predicted for r in v["rows"]}
    pp={(r["case_id"],r["initial_a"],r["initial_b"]):r for v in predicted for r in v["pairs"]}
    rows,pairs=[],[]
    for case in cfg["cases"]:
        value=read_json(study/"cases"/(case["case_id"]+".json"))
        audit=read_json(study/"audits"/(case["case_id"]+".json"))
        assert all(all(r.get(k,False) for k in ("remainder_covered","entropy_remainder_covered","distance_interval_covered","entropy_interval_covered","bases_orthogonal")) for r in audit["rows"])
        for actual in value["rows"]:
            prediction=pr[actual["case_id"],actual["initial"]]
            selected=actual["distance_to_W"]<=.05 and actual["entropy_error"]<=.05
            row={**case,**actual,**{"pred_"+k:v for k,v in prediction.items() if k not in case and k not in ("initial",)}}
            row.update(actual_selected=selected,actual_projection_selected=actual["distance_to_W"]<=.05,
                actual_entropy_selected=actual["entropy_error"]<=.05,critical_initial=actual["initial"] in case["target_initials"],
                undecided=not(prediction["guaranteed_selected"] or prediction["guaranteed_unselected"]),
                wrong_guarantee=(prediction["guaranteed_selected"] and not selected) or (prediction["guaranteed_unselected"] and selected),
                reference_digits=value["precision_digits"][-1])
            assert not row["wrong_guarantee"],row
            rows.append(row)
        for actual in value["pairs"]:
            prediction=pp[actual["case_id"],actual["initial_a"],actual["initial_b"]]
            row={**case,**actual,**{"pred_"+k:v for k,v in prediction.items() if k not in case and k not in ("initial_a","initial_b")}}
            assert not prediction["guaranteed_memory"] or actual["actual_memory"]
            assert not prediction["guaranteed_entropy_memory"] or actual["actual_entropy_memory"]
            assert prediction["memory_lower"]-1e-12<=actual["pair_distance"]<=prediction["memory_upper"]+1e-12
            assert prediction["entropy_memory_lower"]-1e-12<=actual["absolute_entropy_difference"]<=prediction["entropy_memory_upper"]+1e-12
            pairs.append(row)
    assert len(rows)==len(pr) and len(pairs)==len(pp)
    write_csv(study/"states.csv",rows)
    write_csv(study/"memory.csv",pairs)
    breakdown=[]
    for initial in sorted({r["initial"] for r in rows}):
        values=[r for r in rows if r["initial"]==initial]
        breakdown.append(dict(initial=initial,states=len(values),actual_selected=sum(r["actual_selected"] for r in values),
            guaranteed_selected=sum(r["pred_guaranteed_selected"] for r in values),guaranteed_unselected=sum(r["pred_guaranteed_unselected"] for r in values),
            undecided=sum(r["undecided"] for r in values),maximum_rank=max(r["pred_rank"] for r in values)))
    write_csv(study/"initial_breakdown.csv",breakdown)
    errors=[float(mp.mpf(c["projector_error_mp"])) for case in cfg["cases"] for c in read_json(study/"audits"/(case["case_id"]+".json"))["rows"]]
    summary=dict(conditions=len(cfg["cases"]),states=len(rows),pairs=len(pairs),families=len(cfg["families"]),
        actual_selected=sum(r["actual_selected"] for r in rows),guaranteed_selected=sum(r["pred_guaranteed_selected"] for r in rows),
        guaranteed_unselected=sum(r["pred_guaranteed_unselected"] for r in rows),undecided=sum(r["undecided"] for r in rows),wrong_guarantees=sum(r["wrong_guarantee"] for r in rows),
        remainder_target_passed=sum(r["pred_rank_target_passed"] for r in rows),max_rank=max(r["pred_rank"] for r in rows),
        max_actual_projector_error=max(errors),actual_projector_memory=sum(r["actual_memory"] for r in pairs),guaranteed_projector_memory=sum(r["pred_guaranteed_memory"] for r in pairs),
        actual_entropy_memory=sum(r["actual_entropy_memory"] for r in pairs),guaranteed_entropy_memory=sum(r["pred_guaranteed_entropy_memory"] for r in pairs),
        entropy_selected_projection_unselected=sum(r["actual_entropy_selected"] and not r["actual_projection_selected"] for r in rows),
        reference_digits=sorted({r["reference_digits"] for r in rows}),
        max_prediction_precision_change=max(r["pred_prediction_precision_change"] for r in rows),completed_utc=io.now(),scope=cfg["scope"])
    summary.update(pair_distance_intervals_covered=len(pairs),pair_entropy_intervals_covered=len(pairs),
        audit_digits=sorted({read_json(study/"audits"/(c["case_id"]+".json"))["precision_digits"] for c in cfg["cases"]}))
    write_json(study/"summary.json",summary)
    return summary


def gram_diagnostic(value,initial):
    length=value["length"]
    m=length//4
    graph=io.spatial.decode(value["input_graph"])
    matrix=mp.matrix([[graph[m+i,m-1-j] if initial=="left" else graph[m-1-i,m+j] for j in range(m)] for i in range(m)])
    assert io.p1.mp_norm(matrix-matrix.T)<mp.mpf("1e-60")
    inverse=mp.inverse(matrix)
    tr=mp.fsum(inverse[i,i] for i in range(m))
    tr2=mp.fsum(x*x for x in inverse)
    normalized=inverse/tr
    eig,u=mp.eigsy((normalized+normalized.T)/2)
    p=[inverse[i,i]/tr for i in range(m)]
    mean=mp.fsum(i*x for i,x in enumerate(p))
    variance=mp.fsum((i-mean)**2*x for i,x in enumerate(p))
    vector_mean=mp.fsum(i*abs(u[i,m-1])**2 for i in range(m))
    purity=tr2/(tr*tr)
    tv=mp.fsum(abs(p[i]-abs(u[i,m-1])**2) for i in range(m))/2
    assert tv<=1-purity+mp.mpf("1e-40")
    delta=mp.mpf(value["states"][initial]["cross_block_min"])
    concentration_width=tr*tr/tr2-1
    assert 1/tr<=delta*(1+mp.mpf("1e-40")) and delta<=tr/tr2*(1+mp.mpf("1e-40"))
    sign=1 if initial=="left" else -1
    skin=mp.mpf(str(value["skin"]))
    rho_bound=(mp.mpf(".5")+mp.exp(abs(skin)))/(2+mp.sqrt(4-(mp.mpf("1.25")+mp.cosh(skin))))
    return dict(length=length,skin=float(skin),initial=initial,log_delta=float(mp.log(delta)),effective_exponential_rate=float(-mp.log(delta)/length),
        log_inverse_trace=float(mp.log(tr)),finite_prefactor_log_correction=float(mp.log(delta*tr)),
        inverse_purity=float(purity),concentration_width=float(concentration_width),inverse_mean_cell=float(mean),
        inverse_mean_over_L=float(mean/length),inverse_variance_over_L2=float(variance/(length*length)),
        extremal_mean_over_L=float(vector_mean/length),distribution_total_variation=float(tv),
        trace_log_delta_skin_slope=float(sign*(1+2*mean)),extremal_log_delta_skin_slope=float(sign*(1+2*vector_mean)),
        slope_error_upper=float(2*(m-1)*(1-purity)),uniform_graph_norm_upper=float(rho_bound),
        inverse_diagonal_distribution=[str(x) for x in p],delta_mp=str(delta),trace_mp=str(tr),square_trace_mp=str(tr2))


def geometry():
    targets={}
    for row in read_json(SIZES/"size_hypothesis_lock.json")["training"]:
        targets[(row["length"],row["skin"])]=ROOT/row["source"]
    for family in read_json(SIZES/"config.json")["families"]:
        targets[(family["length"],float(io.skin_mp(family)))]=SIZES/"static"/(family["key"]+".json")
    rows=[]
    for (length,skin),path in sorted(targets.items()):
        value=read_json(path)
        with mp.workdps(max(320,value["dps"])):
            for initial in ("left","right"):
                row=gram_diagnostic(value,initial)
                rows.append(dict(**row,source=path.relative_to(ROOT).as_posix(),source_sha256=io.audit.sha256(path)))
        io.emit(dict(stage="geometry",length=length,skin=skin))
    write_json(SIZES/"reflected_geometry.json",dict(rows=rows,scope="post-static mechanistic analysis; no new holdout claim",completed_utc=io.now()))
    write_csv(SIZES/"reflected_geometry.csv",[{k:v for k,v in r.items() if k!="inverse_diagonal_distribution"} for r in rows])
    checks=[]
    for candidate in read_json(SIZES/"size_hypothesis_lock.json")["spatial"]:
        row=next(r for r in rows if r["length"]==candidate["length"] and r["skin"]==candidate["skin"] and r["initial"]==candidate["initial"])
        error=abs(candidate["predicted_log_delta"]-row["log_delta"])
        checks.append(dict(**candidate,actual_log_delta=row["log_delta"],absolute_log_error=error,gate_passed=error<=candidate["abs_log_error_gate"]))
    write_csv(SIZES/"size_extrapolation_checks.csv",checks)
    write_json(SIZES/"geometry_summary.json",dict(spatial_extrapolations=len(checks),spatial_gate_passed=sum(r["gate_passed"] for r in checks),
        max_log_error=max(r["absolute_log_error"] for r in checks),max_inverse_width=max(r["concentration_width"] for r in rows),
        max_distribution_tv=max(r["distribution_total_variation"] for r in rows),
        boundary_retained=True,asymptotic_exponent_status="finite-size extrapolation; no sharp asymptotic theorem proved",completed_utc=io.now()))
    finite_intervals()
    return dict(status="geometry_analyzed",rows=len(rows),extrapolation_passed=sum(r["gate_passed"] for r in checks),extrapolations=len(checks))


def finite_intervals():
    rows=read_json(SIZES/"reflected_geometry.json")["rows"]
    predictions=[r for length in (160,192) for r in read_json(SIZES/"spatial"/f"L{length}.json")["raw"]]
    intervals=[]
    with mp.workdps(360):
        for prediction in predictions:
            row=next(r for r in rows if r["length"]==prediction["length"] and
                abs(r["skin"]-prediction["skin"])<1e-14 and r["initial"]==prediction["initial"])
            lower,upper=(mp.mpf(prediction["values"][k]) for k in ("lower","upper"))
            actual=mp.mpf(row["delta_mp"])
            intervals.append(dict(family_key=prediction["family_key"],initial=prediction["initial"],lower_mp=str(lower),upper_mp=str(upper),actual_mp=str(actual),
                covered=lower<=actual<=upper,relative_width=float(upper/lower-1)))
    assert all(r["covered"] for r in intervals)
    write_csv(SIZES/"finite_spatial_interval_checks.csv",intervals)
    summary=read_json(SIZES/"geometry_summary.json")
    summary.update(finite_intervals=len(intervals),finite_intervals_covered=sum(r["covered"] for r in intervals),
        maximum_finite_interval_relative_width=max(r["relative_width"] for r in intervals))
    write_json(SIZES/"geometry_summary.json",summary)
    return dict(status="finite_intervals_checked",intervals=len(intervals))


def rank_failure():
    paths=[("L128_gL2_edge_t67p425","right"),("L144_g0p25_edge_t69p325","right")]
    rows=[]
    for caseid,name in paths:
        case=read_json(ninth.OUT/"cases"/(caseid+".json"))["case"]
        truth=io.read_gzip(ninth.OUT/"cases"/(caseid+".raw.json.gz"))[-1]
        static=read_json(ROOT/"data/prl_p3_selection_front/static"/(case["family_key"]+".json"))
        with mp.workdps(280):
            ctx=io.context(static,[name])[name]
            duration=mp.mpf(case["physical_time"])
            tc=io.time_context(ctx,duration)
            actual=io.spatial.decode(truth["states"][name]["Q"])
            actual_d=mp.mpf(truth["states"][name]["distance_to_W"])
            actual_ent=mp.mpf(truth["states"][name]["entropy"])
            for rank in (2,4,6,8,12):
                values,q,_=ninth.previous.at_rank(tc,rank)
                ent=io.previous.correlation_entropy(q)[0]
                error=io.pair_distance(q,actual)
                assert error<=values["projector_remainder"]+mp.mpf("1e-35")
                rows.append(dict(case_id=caseid,initial=name,rank=rank,radial_candidate=float(values["distance_to_G"]),actual_distance=float(actual_d),
                    radial_error=float(abs(values["distance_to_G"]-actual_d)),state_projector_error=float(error),projector_remainder=float(values["projector_remainder"]),
                    entropy_error=float(abs(ent-actual_ent)),actual_entropy=float(actual_ent),reduced_entropy=float(ent),
                    distance_mp=str(values["distance_to_G"]),actual_distance_mp=str(actual_d),state_error_mp=str(error)))
    write_csv(SIZES/"development/rank_failure_resolution.csv",rows)
    write_json(SIZES/"development/rank_failure_resolution.json",dict(rows=rows,scope="two previously observed counterexamples, development only",completed_utc=io.now()))
    return dict(status="rank_failure_analyzed",rows=len(rows))


def slope_job(index):
    cfg=read_json(SIZES/"config.json")
    family=cfg["families"][index]
    static=read_json(SIZES/"static"/(family["key"]+".json"))
    fronts=[r for r in read_json(SIZES/"front_hypothesis_lock.json")["rows"] if r["family_key"]==family["key"]]
    rows=[]
    with mp.workdps(320):
        contexts=io.context(static)
        for front in fronts:
            ctx=contexts[front["initial"]]
            t=mp.mpf(str(front["candidate_time"]))
            f=lambda z:ninth.rank2_geometry(ctx,z)[0]["distance_candidate"]
            derivative=mp.diff(f,t)
            finite=(f(t+mp.mpf("1e-4"))-f(t-mp.mpf("1e-4")))/mp.mpf(".0002")
            assert abs(derivative-finite)<mp.mpf("1e-7")
            rows.append(dict(**front,rank2_local_slope=float(derivative),central_difference_slope=float(finite),
                derivative_check_error=float(abs(derivative-finite)),scope="local reduced branch derivative, not a size-uniform lower bound"))
    write_json(SIZES/"response"/(family["key"]+".json"),dict(rows=rows,completed_utc=io.now()))
    return dict(status="slopes_completed",family=family["key"],rows=len(rows))


def fronts():
    cfg=read_json(SIZES/"config.json")
    observed={(r["case_id"],r["initial"]):r for case in cfg["cases"] for r in read_json(SIZES/"cases"/(case["case_id"]+".json"))["rows"]}
    candidates=read_json(SIZES/"size_hypothesis_lock.json")["fronts"]
    rows=[]
    for front in read_json(SIZES/"front_hypothesis_lock.json")["rows"]:
        cases=[c for c in cfg["cases"] if c["family_key"]==front["family_key"] and c["group"]==front["group"] and c["target_initials"]==[front["initial"]]]
        points=sorted((float(c["physical_time"]),observed[c["case_id"],front["initial"]]["distance_to_W"]) for c in cases)
        intervals=[(a[0],b[0]) for a,b in zip(points,points[1:]) if a[1]>.05 and b[1]<=.05]
        candidate=next(r for r in candidates if r["family_key"]==front["family_key"] and r["initial"]==front["initial"])
        lo,hi=intervals[0] if intervals else (None,None)
        affine=candidate["predicted_time"]
        min_error=max(lo-affine,affine-hi,0) if lo is not None else None
        family=next(f for f in cfg["families"] if f["key"]==front["family_key"])
        rows.append(dict(**{**family,**front},affine_time_candidate=affine,sampled_lower=lo,sampled_upper=hi,
            sampled_bracketed=bool(intervals),rank2_strict_compatible=lo<front["candidate_time"]<=hi if lo is not None else False,
            affine_strict_compatible=lo<affine<=hi if lo is not None else False,
            affine_minimum_abs_error=min_error,affine_gate_not_falsified=min_error<=.1 if lo is not None else False))
    write_csv(SIZES/"front_checks.csv",rows)
    slopes=[r for f in cfg["families"] for r in read_json(SIZES/"response"/(f["key"]+".json"))["rows"]]
    write_csv(SIZES/"slope_checks.csv",slopes)
    summary=dict(fronts=len(rows),sampled_bracketed=sum(r["sampled_bracketed"] for r in rows),rank2_strict_compatible=sum(r["rank2_strict_compatible"] for r in rows),
        affine_strict_compatible=sum(r["affine_strict_compatible"] for r in rows),affine_gate_not_falsified=sum(r["affine_gate_not_falsified"] for r in rows),
        local_slope_range=[min(r["rank2_local_slope"] for r in slopes),max(r["rank2_local_slope"] for r in slopes)],
        continuous_first_crossing_established=False,uniform_slope_proved=False,completed_utc=io.now())
    write_json(SIZES/"front_summary.json",summary)
    return summary


def common_loss_experiment():
    import run_experimental_feasibility as experiment

    cfg=read_json(EXPERIMENT/"config.json")
    values=[read_json(EXPERIMENT/"families"/(f["key"]+".json")) for f in cfg["families"]]
    rows=[]
    with mp.workdps(140):
        original_shifts={v["family"]["key"]:experiment.complex_model.passive_embedding(experiment.hamiltonian(v["family"]))[0] for v in values}
        shifts={(v["family"]["model"],v["family"]["length"]):max(
            original_shifts[w["family"]["key"]] for w in values if (w["family"]["model"],w["family"]["length"])==
            (v["family"]["model"],v["family"]["length"])) for v in values}
        for value in values:
            family=value["family"]
            raw=io.read_gzip(EXPERIMENT/"families"/(family["key"]+".raw.json.gz"))
            common=shifts[family["model"],family["length"]]
            for sample in raw:
                duration=mp.mpf(sample["physical_time"])
                for initial,state in sample["references"][-1]["states"].items():
                    row=next(r for r in value["rows"] if r["initial"]==initial and r["physical_time"]==float(duration))
                    # Recompute c at high precision; rounded display values do not enter the rescaling.
                    c=original_shifts[family["key"]]
                    scale=mp.exp(-2*(common-c)*duration)
                    eig=[mp.mpf(x)*scale for x in state["physical_survival_eigenvalues"]]
                    assert all(0<a<=1+mp.mpf("1e-40") for a in eig)
                    logp=mp.fsum(mp.log(x) for x in eig)
                    required=experiment.required_inefficiency(eig)
                    result=dict(**row,common_loss_shift=float(common),
                        log_no_click_success_common=float(logp),
                        log10_attempts_for_1000_common=float((mp.log(1000)-logp)/mp.log(10)),
                        log10_max_inefficiency_common=float(required),
                        log10_tomography_attempts_common=float((mp.log(value["tomography_budget"]["accepted_shots_total"])-logp)/mp.log(10)))
                    for efficiency in cfg["efficiencies"]:
                        eta=mp.mpf(efficiency)
                        factors=[1-eta+eta*a for a in eig]
                        logobserved=mp.fsum(mp.log(x) for x in factors)
                        result["true_fraction_eta"+efficiency]=float(mp.exp(logp-logobserved))
                        result["log_observed_no_click_eta"+efficiency]=float(logobserved)
                    occupied=[a/(mp.mpf(".01")+mp.mpf(".99")*a) for a in eig]
                    result["particle_number_eta99_common"]=float(mp.fsum(occupied))
                    result["purity_defect_eta99_common"]=float(mp.sqrt(mp.fsum((a*a-a)**2 for a in occupied)))
                    rows.append(result)
    write_csv(EXPERIMENT/"common_loss_states.csv",rows)
    selected=[r for r in rows if r["actual_selected"]]
    summary=dict(states=len(rows),selected=len(selected),common_shift_by_model_length=[dict(model=k[0],length=k[1],shift=float(v)) for k,v in sorted(shifts.items())],
        best_selected=max(selected,key=lambda r:r["log_no_click_success_common"]),
        worst_selected=min(selected,key=lambda r:r["log_no_click_success_common"]),
        selected_log_success_range=[min(r["log_no_click_success_common"] for r in selected),max(r["log_no_click_success_common"] for r in selected)],
        scope="post-truth feasibility development; common scalar shift preserves physical isospectral comparison; dimensionless, no platform implementation",
        completed_utc=io.now())
    write_json(EXPERIMENT/"common_loss_summary.json",summary)
    return dict(status="common_loss_analyzed",states=len(rows),selected=len(selected))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":220})
    colors={"charge_density_wave":"#137b7b","left":"#bc3f42","right":"#3a68a5"}
    complex_rows=io.audit.read_csv(COMPLEX/"states.csv")
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.55),constrained_layout=True)
    for name,color in colors.items():
        selected=[r for r in complex_rows if r["family_key"]=="L64_g0p2" and r["initial"]==name]
        axes[0].plot([float(r["physical_time"]) for r in selected],[float(r["distance_to_W"]) for r in selected],"o-",label=name.replace("charge_density_wave","CDW"),color=color)
        axes[1].plot([float(r["physical_time"]) for r in selected],[float(r["entropy"]) for r in selected],"o-",color=color,label=name.replace("charge_density_wave","CDW"))
    target=sorted({(float(r["physical_time"]),float(r["W_entropy"])) for r in complex_rows if r["family_key"]=="L64_g0p2"})
    axes[1].plot([r[0] for r in target],[r[1] for r in target],"k--",label="output space")
    axes[0].axhline(.05,color="0.4",linestyle=":")
    axes[0].set(xlabel="time",ylabel="projector distance",title="(a) Complex model")
    axes[1].set(xlabel="time",ylabel="entropy (nat)",title="(b) Occupied states")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    for name,color in colors.items():
        selected=[r for r in complex_rows if r["initial"]==name]
        axes[2].scatter([float(r["pred_distance_to_G"]) for r in selected],[float(r["distance_to_W"]) for r in selected],s=9,alpha=.65,color=color)
    axes[2].plot([0,1],[0,1],"k--",linewidth=.8)
    axes[2].set(xlabel="reduced prediction",ylabel="reference",title="(c) Validation")
    for ext in ("png","pdf"):
        fig.savefig(FIG/("complex_model_validation."+ext))
    plt.close(fig)
    if (SIZES/"states.csv").exists():
        rows=io.audit.read_csv(SIZES/"states.csv")
        fig,axes=plt.subplots(1,3,figsize=(7.0,2.55),constrained_layout=True)
        for name,color in colors.items():
            selected=[r for r in rows if r["initial"]==name]
            axes[0].scatter([float(r["pred_distance_to_G"]) for r in selected],[float(r["distance_to_W"]) for r in selected],s=9,alpha=.65,color=color,label=name.replace("charge_density_wave","CDW"))
        axes[0].plot([0,1],[0,1],"k--",linewidth=.8)
        axes[0].set(xlabel="reduced prediction",ylabel="reference",title="(a) L160/192")
        axes[0].legend(fontsize=7)
        spatial=io.audit.read_csv(SIZES/"size_extrapolation_checks.csv")
        for name,color in (("left",colors["left"]),("right",colors["right"])):
            selected=[r for r in spatial if r["initial"]==name]
            axes[1].scatter([float(r["predicted_log_delta"]) for r in selected],[float(r["actual_log_delta"]) for r in selected],s=20,color=color,label=name)
        endpoints=[min(float(r[k]) for r in spatial for k in ("predicted_log_delta","actual_log_delta")),max(float(r[k]) for r in spatial for k in ("predicted_log_delta","actual_log_delta"))]
        axes[1].plot(endpoints,endpoints,"k--",linewidth=.8)
        axes[1].set(xlabel=r"frozen $\log\delta$",ylabel=r"reference $\log\delta$",title="(b) Size extrapolation")
        geometry=io.audit.read_csv(SIZES/"reflected_geometry.csv")
        for name,color in (("left",colors["left"]),("right",colors["right"])):
            selected=sorted([r for r in geometry if r["initial"]==name and float(r["skin"])==0.],key=lambda r:int(r["length"]))
            axes[2].plot([int(r["length"]) for r in selected],[float(r["inverse_mean_over_L"]) for r in selected],"o-",color=color,label=name)
        axes[2].set(xlabel="L",ylabel=r"inverse-Gram $\mu/L$",title="(c) Spatial weight")
        axes[2].legend(fontsize=7)
        for ext in ("png","pdf"):
            fig.savefig(FIG/("new_sizes_geometry."+ext))
        plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(7.0,4.35),constrained_layout=True)
        family="L192_g0p25"
        for name,color in colors.items():
            selected=sorted([r for r in rows if r["family_key"]==family and r["initial"]==name],key=lambda r:float(r["physical_time"]))
            t=[float(r["physical_time"])/192 for r in selected]
            axes[0,0].plot(t,[float(r["distance_to_W"]) for r in selected],"o-",ms=3,color=color,label=name.replace("charge_density_wave","CDW"))
            axes[0,1].plot(t,[float(r["entropy"]) for r in selected],"o-",ms=3,color=color)
        target=sorted({(float(r["physical_time"])/192,float(r["W_entropy"])) for r in rows if r["family_key"]==family})
        axes[0,1].plot([x[0] for x in target],[x[1] for x in target],"k--",label="output space")
        axes[0,0].axhline(.05,color=".4",ls=":")
        axes[0,0].legend(fontsize=7)
        axes[0,1].legend(fontsize=7)
        axes[0,0].set(xlabel=r"$t/L$",ylabel=r"$d(P,W)$",title="(a) L192, g=0.25")
        axes[0,1].set(xlabel=r"$t/L$",ylabel="entropy (nat)",title="(b) Same protocol, different states")
        frows=io.audit.read_csv(SIZES/"front_checks.csv")
        for name,color in (("left",colors["left"]),("right",colors["right"])):
            selected=sorted([r for r in frows if r["initial"]==name and r["skin_mode"]=="gL"],key=lambda r:(int(r["length"]),float(r["skin_value"])))
            for length,marker in ((160,"o"),(192,"s")):
                subset=[r for r in selected if int(r["length"])==length]
                reciprocal=next(r for r in subset if float(r["skin_value"])==0.)
                base=(float(reciprocal["sampled_lower"])+float(reciprocal["sampled_upper"]))/2
                y=[(float(r["sampled_lower"])+float(r["sampled_upper"]))/2-base for r in subset]
                # Differences of two sampling brackets, not statistical confidence intervals.
                errors=[(float(r["sampled_upper"])-float(r["sampled_lower"])+float(reciprocal["sampled_upper"])-float(reciprocal["sampled_lower"]))/2 for r in subset]
                axes[1,0].errorbar([float(r["skin_value"]) for r in subset],y,yerr=errors,fmt=marker+"-",color=color,ms=4,label=f"{name}, L{length}",alpha=.8)
        axes[1,0].set(xlabel=r"$\chi=gL$",ylabel="sampled time shift",title="(c) Weak-skin response")
        axes[1,0].legend(fontsize=6,ncol=2)
        memory=io.audit.read_csv(SIZES/"memory.csv")
        axes[1,1].scatter([float(r["pair_distance"]) for r in memory],[float(r["pred_memory_lower"]) for r in memory],s=10,color="#3a68a5",alpha=.7)
        axes[1,1].plot([0,1],[0,1],"k--",lw=.8)
        axes[1,1].axhline(.1,color=".4",ls=":")
        axes[1,1].set(xlabel="reference two-state distance",ylabel="memory lower bound",title="(d) Direct initial-state memory")
        for ext in ("png","pdf"):
            fig.savefig(FIG/("selection_memory."+ext))
        plt.close(fig)
    experiment=io.audit.read_csv(EXPERIMENT/"common_loss_states.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.0,4.3),constrained_layout=True)
    for name,color in colors.items():
        selected=sorted([r for r in experiment if r["family_key"]=="ssh_L8_g0p15" and r["initial"]==name],key=lambda r:float(r["physical_time"]))
        t=[float(r["physical_time"]) for r in selected]
        axes[0,0].plot(t,[float(r["distance_to_W"]) for r in selected],"o-",color=color,label=name.replace("charge_density_wave","CDW"),ms=4)
        axes[0,1].plot(t,[-float(r["log_no_click_success_common"])/math.log(10) for r in selected],"o-",color=color,ms=4)
        axes[1,0].plot(t,[float(r["true_fraction_eta.99"]) for r in selected],"o-",color=color,ms=4)
        axes[1,1].plot(t,[float(r["log10_max_inefficiency_common"]) for r in selected],"o-",color=color,ms=4)
    axes[0,0].axhline(.05,color=".4",ls=":")
    axes[0,0].legend(fontsize=8)
    axes[0,0].set(xlabel="time",ylabel="projector distance",title="(a) L8, g=0.15")
    axes[0,1].set(xlabel="time",ylabel=r"$-\log_{10}P_0$",title="(b) Physical heralding cost")
    axes[1,0].set(xlabel="time",ylabel=r"$P_0/P_{0.99}$",title="(c) True no-loss fraction")
    axes[1,1].set(xlabel="time",ylabel=r"$\log_{10}(1-\eta)_{\max}$",title="(d) 99% no-loss posterior")
    for ext in ("png","pdf"):
        fig.savefig(FIG/("experimental_resources."+ext))
    plt.close(fig)
    failures=io.audit.read_csv(SIZES/"development/rank_failure_resolution.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.7),constrained_layout=True)
    for axis,case in zip(axes,sorted({r["case_id"] for r in failures})):
        selected=[r for r in failures if r["case_id"]==case]
        for key,label,style in (("state_projector_error","occupied-state error","o-"),("projector_remainder","remainder bound","s--"),("radial_error","radial error","^:")):
            axis.semilogy([int(r["rank"]) for r in selected],[max(float(r[key]),1e-55) for r in selected],style,label=label,ms=4)
        axis.axhline(1e-5,color=".4",ls=":")
        axis.set(xlabel="retained rank",ylabel="error",title="L128, gL=2" if "L128" in case else "L144, g=0.25")
        axis.legend(fontsize=7)
    for ext in ("png","pdf"):
        fig.savefig(FIG/("rank_failure."+ext))
    plt.close(fig)
    return dict(status="figures_generated",directory=FIG.relative_to(ROOT).as_posix())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("complex","sizes","geometry","finite_intervals","rank_failure","slopes","fronts","experiment","figures"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    if args.stage=="complex":value=summarize_study(COMPLEX)
    elif args.stage=="sizes":value=summarize_study(SIZES)
    else:value={"geometry":geometry,"finite_intervals":finite_intervals,"rank_failure":rank_failure,"slopes":lambda:slope_job(args.index),"fronts":fronts,"experiment":common_loss_experiment,"figures":figures}[args.stage]()
    io.emit(value)


if __name__=="__main__":
    main()
