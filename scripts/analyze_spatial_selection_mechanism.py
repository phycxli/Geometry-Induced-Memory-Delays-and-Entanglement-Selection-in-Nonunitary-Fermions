"""Audit frozen spatial predictions and derive retrospective rate diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits

import run_spatial_selection_mechanism as study

ROOT, OUT, FIG = study.ROOT, study.OUT, study.FIG
audit, previous, p1 = study.audit, study.previous, study.p1
TIMES = (5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 28, 30, 35, 40, 80, 160, 320)
LABELS = dict(charge_density_wave="CDW", left="Left", right="Right", center_block="Center")
COLORS = dict(charge_density_wave="#277da8", left="#c14c44", right="#79589c", center_block="#25846c")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def array_path(result):
    return ROOT / result["array_source"] if "array_source" in result else OUT / "cases" / (result["case"]["case_id"] + ".npz")


def context(static, initial):
    matrices = {key: study.decode(value) for key, value in static["matrices"].items()}
    f = study.decode(static["states"][initial]["F"])
    mu = [mp.mpf(x) for x in static["mu"]]
    rinv = study.right_solve(mp.eye(f.rows), matrices["R"])
    bnorm = [previous.mp_opnorm(matrices["Bperp"][:, i]) for i in range(f.rows)]
    rnorm = [previous.mp_opnorm(rinv[i, :]) for i in range(f.rows)]
    weights = [[abs(f[i, j]) * bnorm[i] * rnorm[j] for j in range(f.cols)] for i in range(f.rows)]
    assert weights[0][0] > 0
    return matrices, f, mu, rinv, weights


def rate_metrics(ctx, duration, derivative=True):
    matrices, f, mu, rinv, weights = ctx
    ft = mp.matrix([[f[i, j] * mp.exp(-(mu[i] + mu[j]) * duration) for j in range(f.cols)] for i in range(f.rows)])
    denominator = matrices["R"] + matrices["Ccross"] * ft
    kg = study.right_solve(matrices["Bperp"] * ft, denominator)
    u, singular, vh = mp.svd(kg)
    kappa = singular[0]
    if not derivative:
        return kappa
    ftprime = mp.matrix([[-(mu[i] + mu[j]) * ft[i, j] for j in range(f.cols)] for i in range(f.rows)])
    kgprime = study.right_solve((matrices["Bperp"] - kg * matrices["Ccross"]) * ftprime, denominator)
    rate = -(u[:, 0].T * kgprime * vh[0, :].T)[0] / kappa
    unorm = previous.mp_opnorm(matrices["Ccross"] * ft * rinv)
    linear = matrices["Bperp"] * ft * rinv
    knorm = previous.mp_opnorm(linear)
    leading = weights[0][0] * mp.exp(-2 * mu[0] * duration)
    remainder_ratio = mp.fsum(weights[i][j] / weights[0][0] * mp.exp(-(mu[i] + mu[j] - 2 * mu[0]) * duration)
                              for i in range(f.rows) for j in range(f.cols) if (i, j) != (0, 0))
    gap = (singular[0] - singular[1]) / singular[0]
    for step in ("0.0001", "0.000001", "0.00000001"):
        h = mp.mpf(step)
        finite_difference = (mp.log(rate_metrics(ctx, duration - h, False)) - mp.log(rate_metrics(ctx, duration + h, False))) / (2 * h)
        if abs(rate-finite_difference) < mp.mpf("1e-5"):
            break
    lower = knorm / (1 + unorm)
    upper = knorm / (1 - unorm) if unorm < 1 else None
    if upper is not None:
        assert lower <= kappa * (1 + mp.mpf("1e-40")) and kappa <= upper * (1 + mp.mpf("1e-40"))
    if gap > mp.mpf("1e-8"):
        assert abs(rate - finite_difference) < mp.mpf("1e-5"), (duration, rate, finite_difference, gap)
    return dict(kappa_G=float(kappa), kappa_G_mp=str(kappa), log_kappa_G=float(mp.log(kappa)), distance_to_G=float(kappa / mp.sqrt(1 + kappa**2)),
                effective_rate=float(rate), rate_finite_difference=float(finite_difference), rate_finite_difference_step=float(h), rate_derivative_error=float(abs(rate-finite_difference)),
                largest_singular_gap=float(gap), rate_simple_top_singular=gap > mp.mpf("1e-8"),
                late_rate=float(2 * mu[0]), next_rate_spacing=float(mu[1]-mu[0]),
                linearized_kappa=float(knorm), linear_relative_error=float(abs(knorm/kappa-1)),
                nonlinear_norm=float(unorm), linear_norm_lower=float(lower), linear_norm_upper=float(upper) if upper is not None else None,
                linear_bound_valid=unorm < 1, leading_mode_kappa=float(leading), leading_relative_error=float(abs(leading/kappa-1)),
                leading_remainder_ratio_upper=float(remainder_ratio), leading_dominance_certified=remainder_ratio < 1,
                leading_mode_coefficient=float(weights[0][0]))


def diagnostics():
    curves, checks, comparisons = [], [], []
    for skin in (0.0, 0.25):
        key = study.identifier(48, 2.0, skin)
        with gzip.open(OUT / "static" / (key + ".raw.json.gz"), "rt", encoding="utf-8") as stream:
            static_values = json.load(stream)[-2:]
        low_curves, high_curves = [], []
        for static, values in zip(static_values, (low_curves, high_curves)):
            with mp.workdps(static["dps"]):
                ctx = context(static, "left")
                for duration in TIMES:
                    row = rate_metrics(ctx, mp.mpf(str(duration)))
                    values.append(dict(length=48, gamma=2.0, skin=skin, initial="left", physical_time=duration,
                                       dps=static["dps"], interpretation="retrospective; frozen predictions unchanged", **row))
        for low, high in zip(low_curves, high_curves):
            with mp.workdps(180):
                change = float(abs(mp.mpf(low["kappa_G_mp"])/mp.mpf(high["kappa_G_mp"])-1))
            assert change <= 1e-8 and abs(low["effective_rate"]-high["effective_rate"]) <= 1e-8
            checks.append(dict(length=48, skin=skin, physical_time=high["physical_time"], kappa_relative_change=change,
                               rate_absolute_change=abs(low["effective_rate"]-high["effective_rate"])))
        curves.extend(high_curves)
        measured = {}
        static = static_values[-1]
        with mp.workdps(180):
            gain = study.decode(static["matrices"]["G"])
            perpendicular = study.decode(static["matrices"]["Gperp"])
            for duration in (5, 10, 20, 25):
                path = study.PRIOR / "cases" / (p1.case_key(48, 2.0, skin, round(duration / 0.05)) + ".raw.json.gz")
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    raw = json.load(stream)[-1]
                q = study.decode(raw["states"]["left"]["Q"])
                kg = study.right_solve(perpendicular.T*q, gain.T*q)
                measured[duration] = previous.mp_opnorm(kg)
                expected = next(r for r in high_curves if r["physical_time"] == duration)["kappa_G"]
                assert abs(float(measured[duration])/expected-1) < 1e-8
            for start, end in ((5, 10), (20, 25)):
                first = next(r for r in high_curves if r["physical_time"] == start)
                second = next(r for r in high_curves if r["physical_time"] == end)
                prior_rows = [r for r in audit.read_csv(study.PRIOR / "actual_entanglement.csv")
                              if int(r["length"]) == 48 and float(r["skin"]) == skin and r["initial"] == "left"]
                old = {float(r["physical_time"]):float(r["graph_norm"]) for r in prior_rows}
                comparisons.append(dict(length=48, skin=skin, start=start, end=end,
                                        rate_static_G=(first["log_kappa_G"]-second["log_kappa_G"])/(end-start),
                                        rate_independent_G=float(mp.log(measured[start]/measured[end])/(end-start)),
                                        rate_prior_moving_W=math.log(old[start]/old[end])/(end-start),
                                        late_rate=first["late_rate"], interpretation="retrospective independent prior truth"))
        study.emit(dict(stage="diagnostics",family=key,points=len(high_curves)))
    study.write_csv("rate_diagnostics.csv", curves)
    study.write_csv("rate_precision_checks.csv", checks)
    study.write_csv("rate_comparison.csv", comparisons)


def literature():
    import requests
    works = (
        ("benzi_razouk2007", "10.1007/s10543-007-0169-x", "稀疏矩阵函数的衰减；与空间交叉块联系"),
        ("benzi_simoncini2015", "10.1137/151006159", "带状矩阵函数的衰减；已核验书目信息的替代入口"),
        ("demko1984", "10.1090/S0025-5718-1984-0758197-9", "带状矩阵逆的衰减；不是本轮初态重叠的直接定理"),
        ("beckermann_townsend2017", "10.1137/16M1096426", "位移结构矩阵的奇异值界；尖锐重叠渐近的候选工具"),
        ("haga2021", "10.1103/PhysRevLett.127.070402", "趋肤导致无闭隙弛豫减慢已有先例"),
        ("mori_shirai2020", "10.1103/PhysRevLett.125.230604", "Liouvillian 间隙与弛豫时间偏离已有先例"),
    )
    rows = []
    for key, doi, relevance in works:
        path = OUT / "literature" / (key + ".json")
        if path.exists():
            rows.append(read_json(path))
            continue
        value = dict(key=key, doi=doi, relevance=relevance, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                     scope="Crossref bibliographic verification; no new full-text novelty audit")
        url = "https://api.crossref.org/works/" + doi
        try:
            response = requests.get(url, timeout=25)
            response.raise_for_status()
            metadata = response.json()["message"]
            assert metadata["DOI"].lower() == doi.lower()
            value.update(status="verified_metadata", title=metadata.get("title", []), authors=metadata.get("author", []),
                         published=metadata.get("published"), source=url, metadata=metadata)
        except Exception as error:
            value.update(status="access_failed", source=url, error=str(error))
        study.write_json(path, value)
        rows.append(value)
    source = ROOT / "data/prl_l0_literature/kawabata2023.txt"
    if source.exists():
        lines = source.read_text(encoding="utf-8").splitlines()
        selected = sorted({i for j, line in enumerate(lines) if "Haga" in line or "slowing down" in line.lower()
                           for i in range(max(0, j-2), min(len(lines), j+5))})
        study.write_json(OUT / "literature/prior_source_context.json",
                         dict(source=source.relative_to(ROOT).as_posix(),sha256=audit.sha256(source),
                              lines=[dict(line=i+1, text=lines[i]) for i in selected],
                              interpretation="Existing review cites prior skin-related relaxation slowing; does not settle Gaussian entanglement novelty."))
    study.write_json(OUT / "literature/index.json",dict(works=[{k:r[k] for k in ("key","doi","status","relevance")} for r in rows]))


def trajectory_interval(rows):
    rows = sorted(rows, key=lambda r:r["physical_time"])
    event = next((i for i in range(len(rows)) if all(r["controlled"] for r in rows[i:])), None)
    return dict(last_fail=rows[event-1]["physical_time"] if event is not None and event else None,
                first_sustained_sample=rows[event]["physical_time"] if event is not None else None,
                observation_end=rows[-1]["physical_time"], right_censored=event is None, left_censored=event == 0,
                subsequent_passed_samples=len(rows)-event-1 if event is not None else 0)


def summarize():
    study.check_lock()
    receipts = [read_json(path) for path in sorted((OUT / "cases").glob("*.json"))]
    assert len(receipts) == len(study.cases()) == 96
    forecasts = {(r["case_id"],r["initial"]):r for r in audit.read_csv(OUT / "locked_predictions.csv")}
    memory_forecasts = {(r["case_id"],r["initial_b"]):r for r in audit.read_csv(OUT / "locked_memory_predictions.csv")}
    rows, memories, density_rows, spectra, precision, singular = [], [], [], [], [], []
    for receipt in receipts:
        case = receipt["case"]
        prediction_path = OUT / "predictions" / (study.identifier(case["length"],case["gamma"],case["skin"]) + ".npz")
        with np.load(array_path(receipt)) as actual, np.load(prediction_path) as predicted:
            for state in receipt["rows"]:
                forecast = forecasts[case["case_id"],state["initial"]]
                phase = np.tile([1.0, 1.0j],case["length"]//2)
                qpredict = phase[:,None]*predicted[case["case_id"]+"_"+state["initial"]]
                q = actual["Q_"+state["initial"]]
                error = audit.projector_error(q, qpredict)
                n = state["length"]//2
                density = np.sum(abs(q)**2,axis=1)
                assert abs(density.sum()-n)<1e-10 and density.min()>-1e-12 and density.max()<1+1e-12
                row = {**state, "reused_prior": "reused_from" in receipt, "prediction_projector_error":error,
                       "prediction_entropy_error":abs(state["entropy"]-float(forecast["entropy"])),
                       **{k:float(forecast[k]) for k in ("distance_lower","distance_upper","entropy_error_lower","entropy_error_upper","distance_to_G","effective_rate")},
                       "guaranteed_selected":forecast["guaranteed_selected"]=="True", "guaranteed_unselected":forecast["guaranteed_unselected"]=="True"}
                row.update(distance_interval_violation=max(row["distance_lower"]-state["distance_to_W"],state["distance_to_W"]-row["distance_upper"],0),
                           entropy_interval_violation=max(row["entropy_error_lower"]-state["entropy_error"],state["entropy_error"]-row["entropy_error_upper"],0),
                           wrong_selected=row["guaranteed_selected"] and not state["controlled"],
                           wrong_unselected=row["guaranteed_unselected"] and state["controlled"])
                rows.append(row)
                density_rows.extend(dict(case_id=case["case_id"],initial=state["initial"],site=i,density=float(x)) for i,x in enumerate(density))
                eigenvalues = np.linalg.eigvalsh(q[:n,:]@q[:n,:].conj().T)
                spectra.extend(dict(case_id=case["case_id"],initial=state["initial"],mode=i,occupation=float(x)) for i,x in enumerate(eigenvalues))
            cdw = next(r for r in receipt["rows"] if r["initial"]=="charge_density_wave")
            for state in receipt["rows"][1:]:
                forecast = memory_forecasts[case["case_id"],state["initial"]]
                lower = float(forecast["distance_lower"])
                measured = audit.projector_error(actual["Q_charge_density_wave"],actual["Q_"+state["initial"]])
                difference = abs(cdw["entropy"]-state["entropy"])
                retained = measured>0.1 or difference>0.1
                memories.append({**case,"initial_a":"charge_density_wave","initial_b":state["initial"],
                                 "lower":lower,"pair_projector_distance":measured,"pair_entropy_difference":difference,
                                 "guaranteed_memory":forecast["guaranteed_memory"]=="True","memory_retained":retained,
                                 "bound_violation":max(0,lower-measured),"controlled_count":cdw["controlled"]+state["controlled"]})
            n = case["length"]//2
            ratio, double = actual["s"][n]/actual["s"][n-1],actual["s_double"][n]/actual["s_double"][n-1]
            singular.append({**case,"ratio_reference":float(ratio),"ratio_double":float(double),
                             "ratio_relative_error":float(abs(double/ratio-1)),"sigma_N_over_sigma_1":float(actual["s"][n-1]/actual["s"][0])})
        precision.append({**case,"reused_prior":"reused_from" in receipt,"reference_dps":receipt["rows"][0]["reference_dps"],
                          "seconds":receipt["seconds"],**receipt["reference_convergence"]})
    assert len(rows)==228 and len(memories)==132
    study.write_csv("actual_entanglement.csv",rows)
    study.write_csv("initial_memory.csv",memories)
    study.write_csv("local_density.csv",density_rows)
    study.write_csv("correlation_spectra.csv",spectra)
    study.write_csv("reference_precision_checks.csv",precision)
    study.write_csv("singular_precision_diagnostics.csv",singular)
    study.write_csv("double_precision_failures.csv",[r for r in rows if r["actual_double_status"]!="reference_validated"])
    overlap = []
    for path in sorted((OUT / "static").glob("*.json")):
        static = read_json(path)
        for initial, state in static["states"].items():
            values = {k:float(v) for k,v in state.items() if k != "F"}
            overlap.append(dict(length=static["length"],gamma=static["gamma"],skin=static["skin"],initial=initial,
                                dps=static["dps"],**values,
                                upper_over_alpha=values.get("overlap_upper",values["alpha"])/values["alpha"]))
    study.write_csv("spatial_overlap_bounds.csv",overlap)
    old_rows = [{**r,"length":int(r["length"]),"gamma":float(r["gamma"]),"skin":float(r["skin"]),
                 "physical_time":float(r["physical_time"]),"controlled":r["controlled"]=="True"}
                for r in audit.read_csv(study.PRIOR / "actual_entanglement.csv")]
    combined = {(r["case_id"],r["initial"]):r for r in old_rows}
    combined.update({(r["case_id"],r["initial"]):r for r in rows})
    intervals = []
    for length in (32,40,48,56,64):
        for skin in (0.0,0.25):
            subset = [r for r in combined.values() if (r["length"],r["gamma"],r["skin"],r["initial"])==(length,2.0,skin,"left")]
            intervals.append(dict(length=length,skin=skin,initial="left",**trajectory_interval(subset)))
    study.write_csv("selection_time_intervals.csv",intervals)
    metrics = []
    for role in ("all","new_only","dense_boundary","window_extension","orientation","unseen_size","gain_strength"):
        subset = [r for r in rows if role=="all" or (role=="new_only" and not r["reused_prior"]) or role in r["roles"]]
        pairs = [r for r in memories if role=="all" or (role=="new_only" and not next(s for s in rows if s["case_id"]==r["case_id"])["reused_prior"]) or role in r["roles"]]
        metrics.append(dict(role=role,states=len(subset),pairs=len(pairs),actual_selected=sum(r["controlled"] for r in subset),
                            guaranteed_selected=sum(r["guaranteed_selected"] for r in subset),guaranteed_unselected=sum(r["guaranteed_unselected"] for r in subset),
                            wrong_selected=sum(r["wrong_selected"] for r in subset),wrong_unselected=sum(r["wrong_unselected"] for r in subset),
                            actual_memory=sum(r["memory_retained"] for r in pairs),guaranteed_memory=sum(r["guaranteed_memory"] for r in pairs),
                            max_projector_prediction_error=max(r["prediction_projector_error"] for r in subset),
                            max_entropy_prediction_error=max(r["prediction_entropy_error"] for r in subset),
                            double_failed=sum(r["actual_double_status"]!="reference_validated" for r in subset)))
    study.write_csv("prediction_metrics.csv",metrics)
    summary = dict(date="2026-10-04",conditions=len(receipts),states=len(rows),pairs=len(memories),static_families=23,
                   new_conditions=sum("reused_from" not in r for r in receipts),new_states=sum(not r["reused_prior"] for r in rows),
                   reused_conditions=sum("reused_from" in r for r in receipts),reused_states=sum(r["reused_prior"] for r in rows),
                   metrics=metrics,selection_intervals=intervals,
                   max_distance_interval_violation=max(r["distance_interval_violation"] for r in rows),
                   max_entropy_interval_violation=max(r["entropy_interval_violation"] for r in rows),
                   max_memory_lower_violation=max(r["bound_violation"] for r in memories),
                   max_double_projector_error=max(r["double_step_projector_error"] for r in rows),
                   singular_ratio_over_1percent=sum(r["ratio_relative_error"]>0.01 for r in singular),
                   reference_digits_new=sorted({r["reference_dps"] for r in rows if not r["reused_prior"]}),
                   orientation_initial_counts={name:dict(selected=sum(r["controlled"] for r in rows if "orientation" in r["roles"] and r["initial"]==name),total=18) for name in study.INITIALS},
                   interpretation="Exact signed static spectral propagation; no sharp overlap exponent or universal scaling established.")
    study.write_json(OUT / "summary.json",summary)
    plot(rows,memories,overlap,intervals)
    next_plan = dict(version=1,source="fourth round, 2026-10-04",status="candidate_not_executed",
                     objective="derive a reduced spatial mechanism and a reliable high-precision size algorithm before HPC production",
                     batches=[dict(name="sharp_overlap",tasks=["derive singular values of Sg[J,I] via off-diagonal matrix-function or displacement structure",
                                                              "separate boundary placement, SSH sublattice and skin metric", "hold out sizes and translated cell offsets"]),
                              dict(name="signed_mode_reduction",tasks=["preserve signs, quantify dominant modal submatrix and remainder", "predict tolerance crossing without full exact propagation", "freeze all parameters before new truth"]),
                              dict(name="precision_then_HPC",tasks=["benchmark high-precision static propagation or arbitrary-precision step QR", "validate L72/80 first with independent references", "choose sparse HPC grid only after precision and reduction gates"]),
                              dict(name="novelty",tasks=["read Haga2021 and Mori2020 original sources", "compare normalized Slater subspace selection and two-initial memory with existing relaxation-delay mechanisms"])],
                     gates=dict(retain_all_double_failures=True,do_not_fit_last_failure_as_transition=True,
                                no_Born_or_thermodynamic_claim=True,no_experiment_until_predictive_observable_and_cost_ready=True))
    study.write_json(OUT / "next_round_plan.json",next_plan)
    study.emit(dict(stage="summarize",states=len(rows),pairs=len(memories),new_states=summary["new_states"]))


def plot(rows, memories, overlap, intervals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
    FIG.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(FIG / (name+".png"),bbox_inches="tight")
        fig.savefig(FIG / (name+".pdf"),bbox_inches="tight")
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.3),layout="constrained")
    for initial in study.INITIALS:
        subset=sorted([r for r in overlap if r["gamma"]==2 and r["skin"]==0 and r["initial"]==initial],key=lambda r:r["length"])
        axs[0].plot([r["length"] for r in subset],[math.log10(r["alpha"]) for r in subset],"o-",label=LABELS[initial],color=COLORS[initial])
        if initial!="charge_density_wave":
            axs[0].fill_between([r["length"] for r in subset],[math.log10(r["overlap_lower"]) for r in subset],
                                [math.log10(r["overlap_upper"]) for r in subset],alpha=.15,color=COLORS[initial])
            subset=[r for r in overlap if r["gamma"]==2 and r["length"]==40 and r["initial"]==initial]
            subset.sort(key=lambda r:r["skin"])
            axs[1].plot([r["skin"] for r in subset],[math.log10(r["alpha"]) for r in subset],"o-",label=LABELS[initial],color=COLORS[initial])
    axs[0].set(xlabel="Sites L",ylabel=r"$\log_{10}\alpha_\infty$",title="(a) Spatial overlap, g = 0")
    axs[1].set(xlabel="Skin parameter g",ylabel=r"$\log_{10}\alpha_\infty$",title="(b) Orientation, L = 40")
    for ax in axs: ax.legend(fontsize=9); ax.grid(alpha=.18)
    save(fig,"spatial_overlap")
    curves=[{k:float(r[k]) for k in ("skin","physical_time","log_kappa_G","effective_rate","late_rate","nonlinear_norm","leading_remainder_ratio_upper")}
            for r in audit.read_csv(OUT / "rate_diagnostics.csv")]
    fig,axs=plt.subplots(2,2,figsize=(11.5,8),layout="constrained")
    for skin,color in ((0.,"#277da8"),(.25,"#c14c44")):
        all_times=[r for r in curves if r["skin"]==skin]
        subset=[r for r in all_times if r["physical_time"]<=40]
        ts=[r["physical_time"] for r in subset]
        label=f"g = {skin:g}"
        axs[0,0].plot(ts,[r["log_kappa_G"]/math.log(10) for r in subset],"o-",label=label,color=color)
        axs[0,1].plot(ts,[r["effective_rate"] for r in subset],"o-",label=label,color=color)
        axs[1,0].semilogy(ts,[r["nonlinear_norm"] for r in subset],"o-",label=label,color=color)
        axs[1,1].semilogy([r["physical_time"] for r in all_times],[r["leading_remainder_ratio_upper"] for r in all_times],"o-",label=label,color=color)
    axs[0,1].axhline(curves[0]["late_rate"],ls="--",color="black",lw=1,label=r"$2\mu_0$")
    axs[0,1].set_yscale("symlog",linthresh=5)
    inset=axs[0,1].inset_axes([.48,.46,.47,.45])
    for skin,color in ((0.,"#277da8"),(.25,"#c14c44")):
        subset=[r for r in curves if r["skin"]==skin and 12.5<=r["physical_time"]<=40]
        inset.plot([r["physical_time"] for r in subset],[r["effective_rate"] for r in subset],"o-",ms=3,color=color)
    inset.axhline(curves[0]["late_rate"],ls="--",color="black",lw=.8)
    inset.set(xlim=(12,41),ylim=(2.58,3.65),title="Late-time detail")
    inset.tick_params(labelsize=8)
    axs[1,0].axhline(1,ls="--",color="black",lw=1)
    axs[1,1].axhline(1,ls="--",color="black",lw=1)
    titles=("(a) Fixed-space graph norm, L = 48, left","(b) Local logarithmic rate","(c) Nonlinear denominator bound","(d) Single slowest mode remainder bound")
    ylabels=(r"$\log_{10}\kappa_G$",r"$-d\log\kappa_G/dt$",r"$\|C F_t R^{-1}\|$",r"$\mathcal{R}/\|H_{00}e^{-2\mu_0t}\|$")
    for ax,title,ylabel in zip(axs.flat,titles,ylabels):
        ax.set(xlabel="Time t",ylabel=ylabel,title=title); ax.legend(fontsize=9); ax.grid(alpha=.18)
    axs[0,1].legend(fontsize=9,loc="lower right")
    save(fig,"rate_crossover")
    fig,axs=plt.subplots(2,3,figsize=(13,7.3),layout="constrained",sharey=True)
    for i,length in enumerate((24,40)):
        for j,skin in enumerate((-.25,0.,.25)):
            ax=axs[i,j]
            for initial in study.INITIALS:
                subset=sorted([r for r in rows if "orientation" in r["roles"] and (r["length"],r["skin"],r["initial"])==(length,skin,initial)],key=lambda r:r["physical_time"])
                ax.plot([r["physical_time"] for r in subset],[max(1e-12,r["distance_to_W"]) for r in subset],"o-",label=LABELS[initial],color=COLORS[initial])
            ax.axhline(.05,color="black",ls="--",lw=1)
            ax.set(yscale="log",ylim=(5e-13,2),xlabel="Time t",ylabel=r"$\|P_Q-P_W\|$",title=f"L = {length}, g = {skin:g}")
            ax.grid(alpha=.18)
    axs[0,0].legend(fontsize=9)
    save(fig,"orientation_reversal")
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.5),layout="constrained")
    for length,marker in ((56,"o"),(64,"s")):
        for skin,color in ((0.,"#277da8"),(.25,"#c14c44")):
            subset=sorted([r for r in rows if "unseen_size" in r["roles"] and (r["length"],r["skin"],r["initial"])==(length,skin,"left")],key=lambda r:r["physical_time"])
            ts=[r["physical_time"] for r in subset]; label=f"L = {length}, g = {skin:g}"
            axs[0,0].semilogy(ts,[max(1e-10,r["distance_to_W"]) for r in subset],marker+"-",color=color,label=label)
            axs[0,1].semilogy(ts,[max(1e-10,r["entropy_error"]) for r in subset],marker+"-",color=color,label=label)
            pair=sorted([r for r in memories if "unseen_size" in r["roles"] and (r["length"],r["skin"])==(length,skin)],key=lambda r:r["physical_time"])
            axs[1,0].semilogy(ts,[max(1e-10,r["pair_projector_distance"]) for r in pair],marker+"-",color=color,label=label)
            axs[1,0].semilogy(ts,[max(1e-10,r["lower"]) for r in pair],"--",color=color,lw=1)
            failed=[r for r in subset if r["actual_double_status"]!="reference_validated"]
            axs[1,1].semilogy(ts,[max(1e-15,r["double_step_projector_error"]) for r in subset],marker+"-",color=color,label=label)
            axs[1,1].scatter([r["physical_time"] for r in failed],[r["double_step_projector_error"] for r in failed],marker="x",s=70,color="black",zorder=4)
    thresholds=(.05,.05,.1,1e-6)
    titles=("(a) Left-state output selection","(b) Absolute entropy error (nat)","(c) CDW-left memory; dashed: lower bound","(d) Double QR error; x: failed budget")
    for ax,threshold,title in zip(axs.flat,thresholds,titles):
        ax.axhline(threshold,ls="--",color="black",lw=1); ax.set(xlabel="Time t",title=title); ax.grid(alpha=.18)
    axs[0,0].legend(fontsize=9)
    save(fig,"unseen_sizes_memory_precision")
    fig,ax=plt.subplots(figsize=(7.5,4.5),layout="constrained")
    for skin,color in ((0.,"#277da8"),(.25,"#c14c44")):
        subset=[r for r in intervals if r["skin"]==skin]
        offset=-.3 if skin==0 else .3
        for r in subset:
            y=r["first_sustained_sample"]; low=r["last_fail"]
            ax.plot([r["length"]+offset]*2,[low,y],color=color,lw=4)
        ax.plot([r["length"]+offset for r in subset],[r["first_sustained_sample"] for r in subset],"o",color=color,label=f"g = {skin:g}")
    ax.set_xticks((32,40,48,56,64))
    ax.set(xlabel="Sites L",ylabel="Joint selection time bracket",title="Left state: last fail to sampled sustained pass")
    ax.legend();ax.grid(alpha=.18)
    save(fig,"sampled_selection_brackets")


def local_links(allow_pending_delivery=False):
    pending={OUT / "delivery_verification.json",OUT / "artifact_manifest.csv"} if allow_pending_delivery else set()
    links=0
    for path in [ROOT / "README.md",*sorted((ROOT / "research_doc").rglob("*.md"))]:
        content=path.read_text(encoding="utf-8-sig")
        assert "\ufffd" not in content
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",content):
            target=target.split("#",1)[0]
            if not target or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:",target):continue
            resolved=(path.parent / target.strip("<>")).resolve()
            assert resolved.exists() or resolved in pending,f"broken link: {path} {target}"
            links+=1
    return links


def verify():
    study.check_lock()
    lock=read_json(OUT / "prediction_lock.json")
    frozen_inputs=audit.read_csv(OUT / "input_manifest.csv")
    for item in frozen_inputs+lock["prediction_inputs"]:
        assert audit.sha256(ROOT / item["path"])==item["sha256"],item["path"]
    assert lock["config_sha256"]==audit.sha256(OUT / "config.json")
    receipts=[read_json(path) for path in sorted((OUT / "cases").glob("*.json"))]
    assert len(receipts)==96
    predicted_raw={}
    for path in (OUT / "predictions").glob("*.raw.json.gz"):
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            for row in json.load(stream):
                predicted_raw[row["case_id"],row["initial"]]=row["values"]
    entropy_checks=[]
    for result in receipts:
        assert result["config_sha256"]==lock["config_sha256"]
        assert previous.accepted_precision(result["reference_convergence"])
        assert audit.sha256(OUT / "code" / (result["script_sha256"].lower()+".py"))==result["script_sha256"]
        if "reused_from" in result:
            assert audit.sha256(ROOT / result["reused_from"])==result["source_receipt_sha256"]
        else:
            assert result["started_utc"]>lock["locked_utc"]
            assert result["prediction_lock_sha256"]==audit.sha256(OUT / "prediction_lock.json")
        with gzip.open(array_path(result).with_suffix(".raw.json.gz"),"rt",encoding="utf-8") as stream:
            values=json.load(stream)
        if "reused_from" not in result:
            assert [v["dps"] for v in values]==result["precision_digits"]
        with mp.workdps(180):
            for state in result["rows"]:
                error=abs(mp.mpf(values[-1]["states"][state["initial"]]["entropy"])-
                          mp.mpf(predicted_raw[state["case_id"],state["initial"]]["entropy"]))
                assert error<mp.mpf("1e-10")
                entropy_checks.append(dict(case_id=state["case_id"],initial=state["initial"],
                                           high_precision_entropy_prediction_error=float(error),reference_dps=values[-1]["dps"]))
        with np.load(array_path(result)) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            for state in result["rows"]:
                q=arrays["Q_"+state["initial"]]
                assert audit.opnorm(q.conj().T@q-np.eye(q.shape[1]))<1e-10
    summary=read_json(OUT / "summary.json")
    for key in ("max_distance_interval_violation","max_entropy_interval_violation","max_memory_lower_violation"):
        assert summary[key]<=1e-9
    assert summary["states"]==228 and summary["pairs"]==132 and summary["new_states"]==204
    assert all(r["wrong_selected"]==0 and r["wrong_unselected"]==0 for r in summary["metrics"])
    assert summary["metrics"][0]["max_projector_prediction_error"]<1e-10
    assert summary["metrics"][0]["max_entropy_prediction_error"]<1e-10
    csv_rows=audit.read_csv(OUT / "actual_entanglement.csv")
    lookup={(r["case_id"],r["initial"]):r for result in receipts for r in result["rows"]}
    for row in csv_rows:
        state=lookup[row["case_id"],row["initial"]]
        assert all(float(row[k])==state[k] for k in ("entropy","distance_to_W","alpha","ratio"))
    static_files=sorted((OUT / "static").glob("*.json"))
    assert len(static_files)==23
    for path in static_files:
        static=read_json(path)
        assert static["reference_convergence"]["relative_scalar"]<=1e-8
        assert max(static["reference_convergence"]["matrix_absolute"],static["reference_convergence"]["gain_entropy"])<=1e-10
        with mp.workdps(180):
            for initial,state in static["states"].items():
                alpha=mp.mpf(state["alpha"])
                if initial=="charge_density_wave":
                    assert mp.mpf(state["cdw_formula_relative_error"])<mp.mpf("1e-50")
                else:
                    assert mp.mpf(state["overlap_lower"])<=alpha<=mp.mpf(state["overlap_upper"])
                    assert alpha<=mp.mpf(state["analytic_tail_upper"])
    rates=audit.read_csv(OUT / "rate_diagnostics.csv")
    assert len(rates)==32 and max(float(r["rate_derivative_error"]) for r in rates)<1e-5
    study.write_csv("high_precision_entropy_prediction_checks.csv",entropy_checks)
    links=local_links(allow_pending_delivery=True)
    study.write_json(OUT / "delivery_verification.json",dict(status="passed",utc=datetime.now(timezone.utc).isoformat(),
                     immutable_inputs=len(frozen_inputs),frozen_prediction_inputs=len(lock["prediction_inputs"]),
                     protocols=96,states=228,pairs=132,new_truth_started_after_lock=True,locked_predictions_unchanged=True,
                     static_families=23,static_state_overlap_checks=92,rate_points=32,local_links_checked=links,arrays_finite=True,
                     reference_convergence_passed=True,double_failures_retained=summary["metrics"][0]["double_failed"],
                     max_raw_entropy_prediction_error=max(r["high_precision_entropy_prediction_error"] for r in entropy_checks),
                     qualification="converged high precision, not interval-arithmetic certificates"))
    study.emit(dict(stage="verify",status="passed",states=228,pairs=132,links=links))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("diagnostics","literature","summarize","verify"))
    args=parser.parse_args()
    study.prepare()
    own=Path(__file__)
    (OUT / "code" / (audit.sha256(own).lower()+".py")).write_bytes(own.read_bytes())
    with threadpool_limits(limits=1):
        globals()[args.stage]()
    study.write_json(OUT / f"environment_analysis_{args.stage}.json",dict(utc=datetime.now(timezone.utc).isoformat(),
                     command=__import__("sys").argv,analysis_source_sha256=audit.sha256(own),compute_source_sha256=study.SOURCE_SHA,
                     config_sha256=audit.sha256(OUT / "config.json")))
    if args.stage=="verify":
        files=[p for directory in (OUT,FIG) for p in directory.rglob("*") if p.is_file() and p.name!="artifact_manifest.csv"]
        files.extend([own,study.SOURCE,ROOT / "README.md",*sorted((ROOT / "research_doc").rglob("*.md"))])
        study.write_csv("artifact_manifest.csv",[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=audit.sha256(p)) for p in sorted(files)])
        assert local_links()==read_json(OUT / "delivery_verification.json")["local_links_checked"]


if __name__=="__main__":
    main()
