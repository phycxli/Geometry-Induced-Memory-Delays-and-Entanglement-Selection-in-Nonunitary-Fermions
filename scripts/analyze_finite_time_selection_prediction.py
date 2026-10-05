"""Summarize and verify the frozen P2 forecasts and reference calculations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp
import numpy as np

import run_finite_time_selection_prediction as study

ROOT, OUT, FIG = study.ROOT, study.OUT, study.FIG
audit = study.audit


def read_cases(general=False):
    directory = OUT / ("general" if general else "cases")
    values = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]
    assert len(values) == (8 if general else 112)
    return values


def array_path(result):
    if "reused_from" in result:
        return (ROOT / result["reused_from"]).with_suffix(".npz")
    return OUT / ("general" if result.get("general") else "cases") / (result["case"]["case_id"] + ".npz")


def evaluation_role(row):
    if row["role"] == "time" and row["case_id"] in study.configuration()["previously_seen_late_conditions"]:
        return "previously_seen_late"
    return {"early":"training", "time":"new_time", "size":"unseen_size", "weak_skin":"unseen_skin"}[row["role"]]


def selection_interval(trajectory):
    event = next((i for i in range(len(trajectory)) if all(r["controlled"] for r in trajectory[i:])), None)
    return dict(last_failing_sample=trajectory[event-1]["physical_time"] if event is not None and event else None,
                first_sustained_sample=trajectory[event]["physical_time"] if event is not None else None,
                left_censored=event==0, right_censored=event is None,
                observation_end=trajectory[-1]["physical_time"], subsequent_passed_samples=len(trajectory)-event-1 if event is not None else 0)


def interval_error(forecast, interval):
    lower, upper = interval["last_failing_sample"], interval["first_sustained_sample"]
    if interval["right_censored"]:
        return max(0.0, interval["observation_end"]-forecast)
    if interval["left_censored"]:
        return max(0.0, forecast-upper)
    return max(0.0, lower-forecast, forecast-upper)


def summarize():
    study.check_lock()
    products, general = read_cases(), read_cases(True)
    rows, general_rows, memories, density_rows, spectra = [], [], [], [], []
    precision, singular = [], []
    for result in products + general:
        is_general = result.get("general", False)
        with np.load(array_path(result)) as arrays:
            local = []
            for original in result["rows"]:
                row = original.copy()
                q = arrays["Q_" + row["initial"]]
                density = np.sum(abs(q)**2, axis=1)
                n = row["length"]//2
                row.update(right_half_particles=float(density[n:].sum()),
                           evaluation_role="general_initial" if is_general else evaluation_role(row))
                assert abs(density.sum()-n)<1e-10 and density.min()>-1e-12 and density.max()<1+1e-12
                (general_rows if is_general else rows).append(row)
                local.append(row)
                density_rows.extend(dict(case_id=row["case_id"],cohort="general" if is_general else "product",length=row["length"],
                                         skin=row["skin"],physical_time=row["physical_time"],initial=row["initial"],site=i,
                                         density=float(value)) for i,value in enumerate(density))
                eigenvalues=np.linalg.eigvalsh(q[:n,:] @ q[:n,:].conj().T)
                spectra.extend(dict(case_id=row["case_id"],cohort="general" if is_general else "product",initial=row["initial"],
                                    mode=i,occupation=float(value)) for i,value in enumerate(eigenvalues))
            if not is_general:
                first,second=local
                distance=audit.projector_error(arrays["Q_"+first["initial"]],arrays["Q_"+second["initial"]])
                entropy_difference=abs(first["entropy"]-second["entropy"])
                memories.append({**result["case"],"pair_projector_distance":distance,"pair_entropy_difference":entropy_difference,
                                 "memory_retained":distance>0.1 or entropy_difference>0.1,
                                 "controlled_initial_count":sum(r["controlled"] for r in local)})
                reference,double=arrays["s"],arrays["s_double"]
                ratio=reference[n]/reference[n-1]
                dr=double[n]/double[n-1]
                singular.append({**result["case"],"ratio_reference":float(ratio),"ratio_double":float(dr),
                                 "ratio_relative_error_to_reference":float(abs(dr/ratio-1)),
                                 "sigma_N_over_sigma_1":float(reference[n-1]/reference[0]),
                                 "sigma_next_over_sigma_1":float(reference[n]/reference[0])})
        precision.append({**result["case"],"cohort":"general" if is_general else "product",
                          "reused_prior": "reused_from" in result,"reference_dps":result["rows"][0]["reference_dps"],
                          "seconds":result["seconds"],**result["reference_convergence"]})
    study.write_csv("actual_entanglement.csv",rows)
    study.write_csv("general_initial_entanglement.csv",general_rows)
    study.write_csv("initial_memory.csv",memories)
    study.write_csv("local_density.csv",density_rows)
    study.write_csv("correlation_spectra.csv",spectra)
    study.write_csv("reference_precision_checks.csv",precision)
    study.write_csv("singular_precision_diagnostics.csv",singular)
    lookup={(r["case_id"],r["initial"]):r for r in rows}
    predictions=[]
    for original in audit.read_csv(OUT/"locked_predictions.csv"):
        truth=lookup[(original["case_id"],original["initial"])]
        row={**original,"evaluation_role":truth["evaluation_role"],"actual_distance":truth["distance_to_W"],
             "actual_entropy_error":truth["entropy_error"],"actual_controlled":truth["controlled"]}
        physical=row["model"]!="early_log_graph_forecast"
        prediction=float(row["distance_bound"] if physical else row["distance_prediction"])
        entropy=float(row["entropy_bound"] if physical else row["entropy_budget_proxy"])
        row.update(absolute_distance_error=abs(prediction-truth["distance_to_W"]),
                   distance_bound_violation=truth["distance_to_W"]-prediction if physical else None,
                   entropy_bound_violation=truth["entropy_error"]-entropy if physical else None,
                   false_joint_positive=row["predicted_joint_budget_pass"]=="True" and not truth["controlled"],
                   false_joint_negative=row["predicted_joint_budget_pass"]=="False" and truth["controlled"],
                   false_projector_positive=row["predicted_projector_pass"]=="True" and truth["distance_to_W"]>0.05)
        if not physical:
            row["log_graph_error"]=abs(float(row["log_intercept"])-float(row["rate"])*float(row["physical_time"])-math.log(truth["graph_norm"]))
        predictions.append(row)
    study.write_csv("prediction_validation.csv",predictions)
    study.write_csv("prediction_false_positive_states.csv",[
        r for r in predictions if r["model"]=="early_log_graph_forecast"
        and (r["false_projector_positive"] or r["false_joint_positive"])])
    metrics=[]
    for model in ("global_gap_envelope","mode_weighted_envelope","early_log_graph_forecast"):
        for role in ("training","previously_seen_late","new_time","unseen_size","unseen_skin"):
            for initial in (*study.INITIALS,"all"):
                subset=[r for r in predictions if r["model"]==model and r["evaluation_role"]==role and (initial=="all" or r["initial"]==initial)]
                metrics.append(dict(model=model,evaluation_role=role,initial=initial,n=len(subset),
                                    distance_mae=float(np.mean([r["absolute_distance_error"] for r in subset])),
                                    distance_max_error=max(r["absolute_distance_error"] for r in subset),
                                    false_joint_positive=sum(r["false_joint_positive"] for r in subset),
                                    false_joint_negative=sum(r["false_joint_negative"] for r in subset),
                                    false_projector_positive=sum(r["false_projector_positive"] for r in subset),
                                    selected_reference=sum(r["actual_controlled"] for r in subset),
                                    predicted_joint_count=sum(r["predicted_joint_budget_pass"]=="True" for r in subset),
                                    max_distance_bound_violation=max((r["distance_bound_violation"] for r in subset if r["distance_bound_violation"] is not None),default=None),
                                    max_entropy_bound_violation=max((r["entropy_bound_violation"] for r in subset if r["entropy_bound_violation"] is not None),default=None),
                                    mean_log_graph_error=float(np.mean([r["log_graph_error"] for r in subset])) if model=="early_log_graph_forecast" else None,
                                    max_log_graph_error=max(r["log_graph_error"] for r in subset) if model=="early_log_graph_forecast" else None))
    study.write_csv("prediction_metrics.csv",metrics)
    static_rows=[]
    rates=[]
    locked_model=json.loads((OUT/"prediction_lock.json").read_text())["empirical_model"]
    for path in sorted((OUT/"static").glob("*.json")):
        static=json.loads(path.read_text(encoding="utf-8"))
        for initial in study.INITIALS:
            state=static["states"][initial]
            static_rows.append(dict(length=static["length"],skin=static["skin"],initial=initial,
                                    alpha_infinite=float(state["alpha_infinite"]),
                                    log_alpha_infinite=math.log(float(state["alpha_infinite"])),
                                    F_norm=float(state["F_norm"]),
                                    **{k:float(v) for k,v in static["geometry"].items()},
                                    reference_relative_change=static["reference_relative_change"]))
            trajectory={r["physical_time"]:r for r in rows if (r["length"],r["skin"],r["initial"])==(static["length"],static["skin"],initial)}
            def secant_rate(start,end):
                if start not in trajectory or end not in trajectory:
                    return None
                return (math.log(trajectory[start]["graph_norm"])-math.log(trajectory[end]["graph_norm"]))/ (end-start)
            forecast=study.empirical_value(locked_model[initial],static["length"],static["skin"],0)
            rates.append(dict(length=static["length"],skin=static["skin"],initial=initial,
                              frozen_forecast_rate=forecast["rate"],measured_rate_5_to_10=secant_rate(5,10),
                              measured_rate_20_to_25=secant_rate(20,25),twice_min_gain_rate=2*float(static["geometry"]["min_gain_rate"]),
                              status="retrospective diagnostic; never used to revise frozen predictions"))
    study.write_csv("static_overlap_mechanism.csv",static_rows)
    study.write_csv("retrospective_rate_diagnostics.csv",rates)
    intervals=[]
    for length,skin,initial in sorted({(r["length"],r["skin"],r["initial"]) for r in rows}):
        trajectory=sorted([r for r in rows if (r["length"],r["skin"],r["initial"])==(length,skin,initial)],key=lambda r:r["physical_time"])
        intervals.append(dict(length=length,skin=skin,initial=initial,**selection_interval(trajectory)))
    study.write_csv("selection_time_intervals.csv",intervals)
    times=[]
    for forecast in audit.read_csv(OUT/"locked_selection_time_forecasts.csv"):
        interval=next(x for x in intervals if x["length"]==int(forecast["length"]) and x["skin"]==float(forecast["skin"]) and x["initial"]==forecast["initial"])
        times.append({**forecast,**interval,"projector_forecast_interval_error":interval_error(float(forecast["projector_time_forecast"]),interval),
                      "joint_forecast_interval_error":interval_error(float(forecast["joint_budget_time_forecast"]),interval),
                      "projector_comparison_meaning":"projector forecast compared with joint event; entropy can delay the joint event",
                      "interval_meaning":"sample bracket only; no continuity or persistence between samples guaranteed"})
    study.write_csv("selection_time_validation.csv",times)
    initial_summary=[]
    for length in (16,32):
        for skin in (0.0,0.25):
            for duration in (5.0,15.0):
                subset=[r for r in general_rows if (r["length"],r["skin"],r["physical_time"])==(length,skin,duration)]
                random=[r for r in subset if r["initial"].startswith("random_")]
                ground=next(r for r in subset if r["initial"]=="hermitian_ground")
                initial_summary.append(dict(length=length,skin=skin,physical_time=duration,n_random=20,
                                            random_selected=sum(r["controlled"] for r in random),
                                            random_distance_max=max(r["distance_to_W"] for r in random),
                                            random_entropy_error_max=max(r["entropy_error"] for r in random),
                                            ground_distance=ground["distance_to_W"],ground_entropy_error=ground["entropy_error"],
                                            ground_selected=ground["controlled"]))
    study.write_csv("general_initial_summary.csv",initial_summary)
    failures=[r for r in rows+general_rows if r["actual_double_status"]!="reference_validated" or r["W_double_projector_error"]>1e-6 or r["W_double_entropy_error"]>1e-6]
    study.write_csv("double_precision_failed_states.csv",failures)
    physical=[r for r in predictions if r["model"]!="early_log_graph_forecast"]
    heldout=[r for r in predictions if r["model"]=="early_log_graph_forecast" and r["evaluation_role"] in ("new_time","unseen_size","unseen_skin")]
    fresh_roles=("new_time","unseen_size","unseen_skin")
    coverage={model:dict(states=len([r for r in physical if r["model"]==model and r["evaluation_role"] in fresh_roles]),
                         guaranteed_joint_count=sum(r["predicted_joint_budget_pass"]=="True" for r in physical if r["model"]==model and r["evaluation_role"] in fresh_roles),
                         selected_reference=sum(r["actual_controlled"] for r in physical if r["model"]==model and r["evaluation_role"] in fresh_roles))
              for model in ("global_gap_envelope","mode_weighted_envelope")}
    summary=dict(product_conditions=112,new_product_conditions=96,reused_product_conditions=16,general_protocols=8,
                 product_states=len(rows),general_states=len(general_rows),total_states=len(rows+general_rows),
                 selected_product_states=sum(r["controlled"] for r in rows),selected_general_states=sum(r["controlled"] for r in general_rows),
                 memory_retained_product_conditions=sum(r["memory_retained"] for r in memories),
                 prediction_rows=len(predictions),new_heldout_states=len(heldout),
                 empirical_heldout_distance_mae=float(np.mean([r["absolute_distance_error"] for r in heldout])),
                 empirical_heldout_distance_max_error=max(r["absolute_distance_error"] for r in heldout),
                 empirical_heldout_false_joint_positive=sum(r["false_joint_positive"] for r in heldout),
                 empirical_heldout_false_projector_positive=sum(r["false_projector_positive"] for r in heldout),
                 empirical_heldout_false_joint_negative=sum(r["false_joint_negative"] for r in heldout),
                 empirical_heldout_left_distance_mae=float(np.mean([r["absolute_distance_error"] for r in heldout if r["initial"]=="left"])),
                 empirical_heldout_max_log_graph_error=max(r["log_graph_error"] for r in heldout),
                 physical_fresh_validation_coverage=coverage,
                 max_general_distance=max(r["distance_to_W"] for r in general_rows),
                 max_general_entropy_error=max(r["entropy_error"] for r in general_rows),
                 entropy_only_product_states=sum(r["entropy_error"]<=0.05 and r["distance_to_W"]>0.05 for r in rows),
                 max_physical_distance_bound_violation=max(r["distance_bound_violation"] for r in physical),
                 max_physical_entropy_bound_violation=max(r["entropy_bound_violation"] for r in physical),
                 max_double_projector_error=max(r["double_step_projector_error"] for r in rows+general_rows),
                 max_double_entropy_error=max(r["double_step_entropy_error"] for r in rows+general_rows),
                 max_W_projector_error=max(r["W_double_projector_error"] for r in rows+general_rows),
                 max_W_entropy_error=max(r["W_double_entropy_error"] for r in rows+general_rows),
                 double_failed_states=len(failures),
                 double_ratio_error_above_one_percent=sum(r["ratio_relative_error_to_reference"]>0.01 for r in singular),
                 max_reference_relative_scalar_change=max(r["scalar_relative"] for r in precision),
                 max_reference_entropy_change=max(r["S"] for r in precision),
                 max_case_seconds=max(r["seconds"] for r in precision),
                 scope="gamma=2, L16-48, t5-25 deterministic OBC study; no thermodynamic or stochastic Born claim")
    study.write_json(OUT/"summary.json",summary)
    plot(rows,general_rows,predictions,times,memories)
    study.emit(dict(stage="summarize",**summary))


def plot(rows,general_rows,predictions,times,memories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"savefig.dpi":185})
    colors={16:"#188c81",24:"#db9e32",32:"#b43d4c",40:"#436daf",48:"#695d83"}
    def save(fig,name):
        fig.savefig(FIG/(name+".png")); fig.savefig(FIG/(name+".pdf")); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for skin,ax in zip((0.0,0.25),axes):
        for length in (16,24,32,40,48):
            data=sorted([r for r in rows if r["initial"]=="left" and r["skin"]==skin and r["length"]==length],key=lambda r:r["physical_time"])
            ax.semilogy([r["physical_time"] for r in data],[r["distance_to_W"] for r in data],"o-",color=colors[length],ms=4,label=f"L={length}")
        ax.axhline(0.05,color="black",ls="--",lw=1,label="projector tolerance")
        ax.set(title=f"Left-occupied selection, g={skin}",xlabel="Physical time",ylabel="Reference distance to output subspace",ylim=(1e-25,2))
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
    save(fig,"selection_size_time_extension")
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for skin,ax in zip((0.0,0.25),axes[:2]):
        for model,marker,color,label in (("early_log_graph_forecast","o","#188c81","Early forecast"),
                                          ("mode_weighted_envelope","s","#b43d4c","Mode bound"),
                                          ("global_gap_envelope","^","#436daf","Gap bound")):
            subset=sorted([r for r in times if r["initial"]=="left" and float(r["skin"])==skin and r["model"]==model],key=lambda r:r["length"])
            ax.plot([r["length"] for r in subset],[float(r["joint_budget_time_forecast"]) for r in subset],marker+"--",color=color,label=label)
        for item in [r for r in times if r["initial"]=="left" and float(r["skin"])==skin and r["model"]=="early_log_graph_forecast"]:
            lo,hi=item["last_failing_sample"],item["first_sustained_sample"]
            ax.vlines(item["length"],lo,hi,color="black",lw=3)
            ax.plot(item["length"],hi,"_",color="black",ms=10)
        ax.set(title=f"Joint-budget forecasts, g={skin}",xlabel="Sites L",ylabel="Time; black lines: sampled joint bracket")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
    for role,color,label in (("new_time","#188c81","New times"),("unseen_size","#b43d4c","Unseen sizes"),("unseen_skin","#436daf","Unseen g")):
        subset=[r for r in predictions if r["model"]=="early_log_graph_forecast" and r["evaluation_role"]==role and r["initial"]=="left"]
        axes[2].scatter([float(r["distance_prediction"]) for r in subset],[r["actual_distance"] for r in subset],s=20,color=color,label=label,alpha=0.8)
    axes[2].plot([0,1],[0,1],"k--",lw=1)
    axes[2].set(title="Fresh left-state validation",xlabel="Predicted projector distance",ylabel="Reference projector distance",xlim=(-0.03,1.03),ylim=(-0.03,1.03))
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.2)
    save(fig,"locked_forecast_validation")
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True,sharey=True)
    for skin,ax in zip((0.0,0.25),axes):
        for duration,color in ((5.0,"#436daf"),(15.0,"#db9e32"),(20.0,"#b43d4c"),(25.0,"#188c81")):
            r=next(r for r in rows if r["length"]==48 and r["skin"]==skin and r["physical_time"]==duration and r["initial"]=="left")
            with np.load(OUT/"cases"/(r["case_id"]+".npz")) as archive:
                density=np.sum(abs(archive["Q_left"])**2,axis=1)
            ax.plot(np.arange(24),density[::2]+density[1::2],"o-",ms=3,color=color,label=f"t={duration:g}")
        ax.axvline(11.5,color="black",ls="--",lw=1)
        ax.set(title=f"L=48 left-state density, g={skin}",xlabel="Unit cell j",ylabel="Particles per cell",ylim=(-0.05,2.05))
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
    save(fig,"local_density_redistribution")
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for skin,ax in zip((0.0,0.25),axes):
        for initial,color,label in (("left","#b43d4c","Left occupied"),("charge_density_wave","#188c81","CDW")):
            data=sorted([r for r in rows if r["length"]==32 and r["skin"]==skin and r["initial"]==initial],key=lambda r:r["physical_time"])
            ax.semilogy([r["physical_time"] for r in data],[r["distance_to_W"] for r in data],"o-",color=color,label=label)
        for duration in (5.0,15.0):
            data=[r for r in general_rows if r["length"]==32 and r["skin"]==skin and r["physical_time"]==duration and r["initial"].startswith("random_")]
            offsets=np.linspace(-0.18,0.18,len(data))
            ax.scatter(duration+offsets,[r["distance_to_W"] for r in data],marker="x",color="#436daf",s=22,label="Random Slater (20)" if duration==5 else None)
            ground=next(r for r in general_rows if r["length"]==32 and r["skin"]==skin and r["physical_time"]==duration and r["initial"]=="hermitian_ground")
            ax.scatter([duration],[ground["distance_to_W"]],marker="D",color="#db9e32",s=30,label="Hermitian ground" if duration==5 else None)
        ax.axhline(0.05,color="black",ls="--",lw=1)
        ax.set(title=f"Initial-state dependence, L=32, g={skin}",xlabel="Physical time",ylabel="Reference distance to output subspace",ylim=(1e-32,2))
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
    save(fig,"general_initial_state_controls")


def verify():
    study.check_lock()
    lock=json.loads((OUT/"prediction_lock.json").read_text())
    for item in lock["inputs"]:
        assert audit.sha256(ROOT/item["path"])==item["sha256"]
    results=read_cases()+read_cases(True)
    for row in audit.read_csv(OUT/"input_manifest.csv"):
        assert audit.sha256(ROOT/row["path"])==row["sha256"]
    for result in results:
        assert result["config_sha256"]==audit.sha256(OUT/"config.json")
        assert study.accepted_precision(result["reference_convergence"])
        if "reused_from" in result:
            assert audit.sha256(ROOT/result["reused_from"])==result["source_receipt_sha256"]
            continue
        assert audit.sha256(OUT/"code"/(result["script_sha256"].lower()+".py"))==result["script_sha256"]
        if result["case"]["role"]!="early":
            assert result["started_utc"]>lock["locked_utc"]
            assert result["prediction_lock_sha256"]==audit.sha256(OUT/"prediction_lock.json")
        if result.get("general"):
            assert result["initial_array_sha256"]==audit.sha256(OUT/"initial_states"/f"L{result['case']['length']}.npz")
        with np.load(array_path(result)) as arrays:
            assert all(np.all(np.isfinite(arrays[k])) for k in arrays.files)
        raw=array_path(result).with_suffix(".raw.json.gz")
        with gzip.open(raw,"rt",encoding="utf-8") as stream:
            values=json.load(stream)
            assert [r["dps"] for r in values]==[c["dps"] for c in result["precision_checks"]]
    static_checks=[]
    with mp.workdps(160):
        for path in sorted((OUT/"static").glob("*.raw.json.gz")):
            with gzip.open(path,"rt",encoding="utf-8") as stream:
                first,second=json.load(stream)
            for initial in study.INITIALS:
                a=mp.matrix([[mp.mpf(x) for x in row] for row in first["states"][initial]["F"]])
                b=mp.matrix([[mp.mpf(x) for x in row] for row in second["states"][initial]["F"]])
                error=study.p1.mp_norm(mp.matrix([[abs(a[i,j])-abs(b[i,j]) for j in range(a.cols)] for i in range(a.rows)]))/study.p1.mp_norm(b)
                assert error<=mp.mpf("1e-8")
                static_checks.append(dict(length=first["length"],skin=first["skin"],initial=initial,F_abs_relative_fro_change=float(error)))
            for k in second["geometry"]:
                a,b=mp.mpf(first["geometry"][k]),mp.mpf(second["geometry"][k])
                assert abs(a-b)/max(abs(a),abs(b))<=mp.mpf("1e-8")
    study.write_csv("static_precision_checks.csv",static_checks)
    summary=json.loads((OUT/"summary.json").read_text())
    assert summary["total_states"]==392 and summary["product_states"]==224
    assert summary["max_physical_distance_bound_violation"]<=1e-10
    assert summary["max_physical_entropy_bound_violation"]<=1e-10
    csv_rows=audit.read_csv(OUT/"actual_entanglement.csv")+audit.read_csv(OUT/"general_initial_entanglement.csv")
    lookup={(r["case"]["case_id"],bool(r.get("general")),state["initial"]):state for r in results for state in r["rows"]}
    for row in csv_rows:
        truth=lookup[(row["case_id"],row["role"]=="general",row["initial"])]
        assert all(float(row[k])==truth[k] for k in ("entropy","distance_to_W","alpha","ratio","graph_norm"))
    links=0
    for path in [ROOT/"README.md",*sorted((ROOT/"research_doc").rglob("*.md"))]:
        content=path.read_text(encoding="utf-8-sig")
        assert "\ufffd" not in content
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",content):
            target=target.split("#",1)[0]
            if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:",target):
                continue
            assert (path.parent/target.strip("<>")).exists(),f"broken link: {path} {target}"
            links+=1
    study.write_json(OUT/"delivery_verification.json",dict(status="passed",immutable_inputs=len(audit.read_csv(OUT/"input_manifest.csv")),
                                                         protocols=120,states=392,locked_predictions_unchanged=True,
                                                         new_truth_started_after_lock=True,local_links_checked=links,
                                                         static_F_comparisons=len(static_checks),frozen_predictor_inputs=len(lock["inputs"]),arrays_finite=True))
    study.emit(dict(stage="verify",status="passed",states=392,links=links))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("summarize","verify"))
    args=parser.parse_args()
    study.prepare()
    own=Path(__file__)
    (OUT/"code"/(audit.sha256(own).lower()+".py")).write_bytes(own.read_bytes())
    if args.stage=="summarize":
        summarize()
    else:
        verify()
    study.write_json(OUT/f"environment_{args.stage}.json",dict(utc=datetime.now(timezone.utc).isoformat(),command=__import__("sys").argv,
                                                             analysis_source_sha256=audit.sha256(own),compute_source_sha256=study.SOURCE_SHA,
                                                             config_sha256=audit.sha256(OUT/"config.json")))
    if args.stage=="verify":
        files=[p for directory in (OUT,FIG) for p in directory.rglob("*") if p.is_file() and p.name!="artifact_manifest.csv"]
        files.extend([own,ROOT/"scripts/run_finite_time_selection_prediction.py",ROOT/"README.md",*sorted((ROOT/"research_doc").rglob("*.md"))])
        study.write_csv("artifact_manifest.csv",[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=audit.sha256(p)) for p in sorted(files)])


if __name__=="__main__":
    main()
