"""Local analysis, figures and provenance for the SSH theory extension."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import run_theory_upgrade_hpc as study

ROOT, OUT = study.ROOT, study.OUT
FIG = ROOT / "figures/prl_theory_upgrade"


def files(folder):
    return sorted(p for p in (OUT/folder).glob("*.json") if ".progress." not in p.name)


def constants(g):
    with study.mp.workdps(50):
        return {k:float(v) for k,v in study.constants(g).items()}


def width_bound(g, high=.3, low=.01):
    c = constants(g)
    kh, kl = high/math.sqrt(1-high*high), low/math.sqrt(1-low*low)
    return math.log(kh*(2*c["a"]-c["b"]*kl)/(kl*(2*c["a"]-c["b"]*kh)))/(2*c["a"])


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def backup():
    target = OUT/"manuscript_baseline.zip"
    if target.exists():
        return dict(status="baseline_preserved",sha256=study.sha(target))
    paths = sorted(p for folder in ("main","supplement") for p in (ROOT/folder).rglob("*") if p.is_file())
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p,p.relative_to(ROOT).as_posix())
    rows = [dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=study.sha(p)) for p in paths]
    study.write_json(OUT/"manuscript_baseline_manifest.json",dict(archived_utc=study.now(),zip_sha256=study.sha(target),files=rows))
    return dict(status="baseline_archived",files=len(paths),sha256=study.sha(target))


def response_summary(require_complete=False):
    folder = OUT/"response_extension"
    extension = [study.read_json(p) for p in sorted(folder.glob("L*.json"))
                 if ".progress." not in p.name and "_development" not in p.name]
    development = [study.read_json(p) for p in sorted(folder.glob("*_development.json"))]
    source = [study.read_json(p) for p in files("gram")]+extension
    rows, weak = [], []
    for value in source:
        zero = next(float(r["log_delta"]) for r in value["rows"] if float(r["skin"])==0)
        for raw in value["rows"]:
            length, initial, g = int(raw["length"]),raw["initial"],float(raw["skin"])
            ratio = float(raw["log_delta"])-zero
            row = dict(length=length,initial=initial,skin=g,log_delta=float(raw["log_delta"]),log_ratio_to_zero=ratio)
            if initial=="right":
                central = -(length/2-1)*g
                correction = math.log((1-math.exp(-2*g)/4)/.75)
                lo, hi = central+min(0.,correction),central+max(0.,correction)
                row.update(log_ratio_lower=lo,log_ratio_upper=hi,finite_response_passed=lo-1e-10<=ratio<=hi+1e-10)
            rows.append(row)
            chi = g*length
            if any(abs(chi-c)<1e-10 for c in (2.,4.)):
                expected = chi/2*(1 if initial=="left" else -1)
                weak.append({**row,"chi":chi,"asymptotic_log_ratio":expected,"finite_size_deviation":ratio-expected})
    finite = [r for r in rows if r["initial"]=="right"]
    if require_complete:
        assert len(extension)==4 and len(development)==6 and len(rows)==50 and len(weak)==20
        assert all(v["status"]=="passed" for v in development)
        assert all(r["finite_response_passed"] for r in finite)
    result = dict(gram_families=len(source),response_new_families=len(extension),structure_development_checks=len(development),
                  gram_rows=len(rows),finite_right_intervals=len(finite),finite_right_passed=sum(r["finite_response_passed"] for r in finite),
                  weak_ratios=len(weak),maximum_log_delta_precision_change=max((float(v["precision_change_mp"]) for v in source),default=None),
                  weak_deviation_by_size={str(length):{side:max(abs(r["finite_size_deviation"]) for r in weak if r["length"]==length and r["initial"]==side)
                                                     for side in ("left","right")} for length in sorted({r["length"] for r in weak})},
                  scope="finite right inequalities and weak-skin convergence; no absolute reciprocal exponent or prefactor theorem")
    if rows:
        write_csv(OUT/"spatial_response_checks.csv",rows)
        write_csv(OUT/"weak_skin_response.csv",weak)
    study.write_json(folder/"summary.json",result)
    return result


def root_radius(row):
    cert, level = row["certificate"],float(row["level"])
    c = constants(row["skin"])
    remainder = float(cert["projector_remainder"])
    residual = abs(float(cert["distance_to_G"])-level)
    budget = remainder+residual
    lo, hi = level-budget,level+budget
    if lo<=0 or hi>=c["kappa_cone"]/math.sqrt(1+c["kappa_cone"]**2):
        return dict(root_radius_status="outside_continuous_cone")
    # This lower slope covers the entire pointwise distance uncertainty interval.
    slope = (2*c["a"]-c["b"]*hi/math.sqrt(1-hi*hi))*lo*(1-hi*hi)
    return dict(root_radius_status="analytic_pointwise_formula",point_distance_budget=budget,minimum_decay_slope=slope,
                actual_G_root_radius=budget/slope,reduced_bisection_half_width_upper=1e-6,
                numerical_endpoint_policy="precision convergence only; no interval arithmetic")


def summarize(require_complete=False):
    curves = [study.read_json(p) for p in files("curves")]
    roots = [r for v in curves for r in v["rows"]]
    for row in roots:
        row.update(root_radius(row))
    widths = []
    for value in curves:
        family = value["family"]
        for name in study.old.initials(family["length"])[1:]:
            local = {r["level"]:r for r in value["rows"] if r["initial"]==name}
            time05 = local["0.05"]["root_time"]
            wide = local["0.1"]["root_time"]-local["0.9"]["root_time"]
            narrow = local["0.01"]["root_time"]-local["0.3"]["root_time"]
            bound = width_bound(family["skin"])
            assert narrow <= bound+1e-4
            widths.append(dict(family_key=family["key"],length=family["length"],skin=family["skin"],role=family["role"],initial=name,
                               t05=time05,width_90_to_10=wide,width_30_to_01=narrow,uniform_width_upper=bound,
                               relative_width_90_to_10=wide/time05,relative_width_30_to_01=narrow/time05))
    lock = study.read_json(OUT/"extrapolation_lock.json")
    time_checks = []
    for p in lock["time_predictions"]:
        match = next((r for r in roots if all(r[k]==p[k] for k in ("family_key","initial","level"))),None)
        if match is not None:
            error = abs(match["root_time"]-p["predicted_time"])
            time_checks.append({**p,"root_time":match["root_time"],"absolute_error":error,"passed":error<=p["abs_time_gate"]})
    grams = [r for p in files("gram") for r in study.read_json(p)["rows"]]
    gram_checks = []
    for p in lock["gram_predictions"]:
        match = next((r for r in grams if r["length"]==p["length"] and r["initial"]==p["initial"] and float(r["skin"])==p["skin"]),None)
        if match is not None:
            error = abs(float(match["log_delta"])-p["predicted_log_delta"])
            gram_checks.append({**p,"log_delta":float(match["log_delta"]),"absolute_error":error,"passed":error<=p["abs_log_gate"]})
    references = [study.read_json(p) for p in files("references")]
    reference_checks = []
    for value in references:
        case, v = value["case"], value["values"]
        d, level = float(v["distance_to_G"]), float(case["level"])
        offset = float(case["duration"])-case["prediction_root"]
        bracket_ok = d>level if offset<0 else d<level
        slope_ok = float(v["actual_slope"])<=float(v["theoretical_slope_upper"])+1e-12
        reference_checks.append(dict(case_id=case["case_id"],family_key=case["family_key"],length=case["length"],skin=case["skin"],initial=case["initial"],
                                     level=level,duration=case["duration"],distance_to_G=d,cdw_distance_mp=v["cdw_distance"],
                                     direct_memory=float(v["direct_memory"]),entropy=float(v["entropy"]),gain_entropy=float(v["gain_entropy"]),
                                     absolute_entropy_error=abs(float(v["entropy"])-float(v["gain_entropy"])),
                                     actual_slope=float(v["actual_slope"]),theoretical_slope_upper=float(v["theoretical_slope_upper"]),
                                     root_side_correct=bracket_ok,slope_bound_passed=slope_ok,precision_change_mp=value["precision_change_mp"]))
    if require_complete:
        assert len(curves)==10 and len(grams)==30 and len(reference_checks)==96
        assert all(r["root_side_correct"] and r["slope_bound_passed"] for r in reference_checks)
        assert all(r["rank"]==12 for r in roots),"a rank-24 candidate requires an additional low-precision rank-24 audit"
    new = [r for r in roots if r["role"]=="new_size"]
    response = response_summary(require_complete)
    bounded = [r for r in roots if r["root_radius_status"]=="analytic_pointwise_formula"]
    result = dict(updated_utc=study.now(),families=len(curves),root_levels=len(roots),new_root_levels=len(new),
                  root_statuses=dict(Counter(r["status"] for r in roots)),root_ranks=dict(Counter(r["rank"] for r in roots)),
                  maximum_remainder=max(float(r["certificate"]["projector_remainder"]) for r in roots),
                  maximum_root_precision_change=max(r["precision_change"] for r in roots),
                  continuously_bounded_roots=len(bounded),maximum_actual_G_root_radius=max((r["actual_G_root_radius"] for r in bounded),default=None),
                  width_30_to_01_range=[min(r["width_30_to_01"] for r in widths),max(r["width_30_to_01"] for r in widths)],
                  width_90_to_10_range=[min(r["width_90_to_10"] for r in widths),max(r["width_90_to_10"] for r in widths)],
                  width_bounds={g:width_bound(g) for g in ("0","0.25")},
                  time_extrapolations=dict(completed=len(time_checks),passed=sum(r["passed"] for r in time_checks),maximum_error=max((r["absolute_error"] for r in time_checks),default=None)),
                  gram_extrapolations=dict(completed=len(gram_checks),passed=sum(r["passed"] for r in gram_checks),maximum_error=max((r["absolute_error"] for r in gram_checks),default=None)),
                  independent_references=len(reference_checks),correct_root_sides=sum(r["root_side_correct"] for r in reference_checks),
                  slope_bounds_passed=sum(r["slope_bound_passed"] for r in reference_checks),
                  maximum_reference_precision_change=max((float(r["precision_change_mp"]) for r in reference_checks),default=None),
                  independently_bracketed_G_roots=len(reference_checks)//2,
                  spatial_response=response,
                  scope="low-threshold first crossings to G and persistent W envelopes; broad-window empirical scan; no sharp reflected exponent or full cutoff theorem")
    for name, rows in (("roots",roots),("widths",widths),("time_extrapolation_checks",time_checks),("gram_extrapolation_checks",gram_checks),("reference_checks",reference_checks)):
        if rows:
            write_csv(OUT/(name+".csv"),[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows])
    study.write_json(OUT/"summary.json",result)
    return result


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summarize(True)
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":8,"legend.fontsize":6.5,"axes.titlesize":8,
                         "xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42,"savefig.dpi":180})
    with (OUT/"widths.csv").open(encoding="utf-8",newline="") as stream:
        widths = list(csv.DictReader(stream))
    roots = [r for p in files("curves") for r in study.read_json(p)["rows"]]
    references = [study.read_json(p) for p in files("references")]
    colors = {"left":"#177c57","right":"#be4343","block1":"#277db1","block3":"#cf8a2b"}

    def kind(r):
        return "block1" if r["initial"].startswith("block_cell") and int(r["initial"][10:])==int(r["length"])//16 else "block3" if r["initial"].startswith("block_cell") else r["initial"]

    fig, axes = plt.subplots(2,2,figsize=(7.1,4.9),layout="constrained")
    ax = axes[0,0]
    for g in ("0","0.25"):
        for k in colors:
            selected = sorted([r for r in widths if r["skin"]==g and kind(r)==k],key=lambda r:int(r["length"]))
            ax.plot([int(r["length"]) for r in selected],[float(r["t05"]) for r in selected],"o-" if g=="0" else "s--",ms=3,lw=1,color=colors[k],label=k if g=="0" else None)
    ax.set(xlabel="Sites L",ylabel=r"Selection time $t_{0.05}$",title="(a) Geometry shifts the selection center")
    ax.legend(ncol=2,loc="upper left")
    ax.text(.98,.04,"solid: g=0\ndashed: g=0.25",transform=ax.transAxes,ha="right",va="bottom",fontsize=7)
    ax = axes[0,1]
    for g,k in (("0","left"),("0.25","left"),("0.25","right")):
        for length in (128,160,192,224,256):
            selected = [r for r in roots if r["skin"]==g and r["initial"]==k and r["length"]==length]
            selected.sort(key=lambda r:r["root_time"])
            center = next(r["root_time"] for r in selected if r["level"]=="0.05")
            ax.plot([r["root_time"]-center for r in selected],[float(r["level"]) for r in selected],".-",lw=.7,ms=3,alpha=.5,
                    color=colors[k] if g=="0.25" else "#505050")
    for value in references:
        case = value["case"]
        if case["initial"] in ("left","right"):
            center = next(r["root_time"] for r in roots if r["family_key"]==case["family_key"] and r["initial"]==case["initial"] and r["level"]=="0.05")
            ax.plot(float(case["duration"])-center,float(value["values"]["distance_to_G"]),"o",ms=2.5,color="black",alpha=.6)
    ax.set(xlabel=r"Time from $t_{0.05}$",ylabel=r"Projector distance $d_G$",title="(b) Narrow transition, independent checks",ylim=(0,.95))
    ax.text(.97,.96,"dots: independent exponentials\nlines: reduced threshold scans",transform=ax.transAxes,ha="right",va="top",fontsize=6.5)
    ax = axes[1,0]
    for g in ("0","0.25"):
        for k in colors:
            selected = sorted([r for r in widths if r["skin"]==g and kind(r)==k],key=lambda r:int(r["length"]))
            ax.plot([int(r["length"]) for r in selected],[float(r["width_30_to_01"]) for r in selected],"o-" if g=="0" else "s--",color=colors[k],ms=3,lw=.8)
    for g,style in (("0","-"),("0.25","--")):
        bound = width_bound(g)
        ax.axhline(bound,color="#444444",ls=style,lw=.8,label=f"upper bound g={g}: {bound:.3f}")
    ax.set(xlabel="Sites L",ylabel=r"Width $t_{0.01}-t_{0.3}$",title="(c) A size-uniform continuous bound",ylim=(1.1,2.7))
    ax.legend(loc="upper right")
    ax = axes[1,1]
    with (OUT/"weak_skin_response.csv").open(encoding="utf-8",newline="") as stream:
        checks=list(csv.DictReader(stream))
    for initial in ("left","right"):
        for chi in (2.,4.):
            selected=sorted([r for r in checks if r["initial"]==initial and float(r["chi"])==chi],key=lambda r:int(r["length"]))
            ax.plot([int(r["length"]) for r in selected],[float(r["log_ratio_to_zero"])/chi for r in selected],"o-" if chi==2 else "s--",color=colors[initial],ms=3,lw=1,label=f"{initial}, chi={chi:g}")
    ax.axhline(.5,color="#666666",ls=":",lw=.8)
    ax.axhline(-.5,color="#666666",ls=":",lw=.8)
    ax.set(xlabel="Sites L",ylabel=r"$\log[\delta(\chi/L)/\delta(0)]/\chi$",title="(d) Reflected weak-skin response",ylim=(-.55,.6))
    ax.legend(ncol=2,loc="center right")
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
    for ext in ("png","pdf"):
        fig.savefig(FIG/("continuous_selection."+ext))
    plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(7.1,4.7),layout="constrained")
    hs=np.linspace(0,.25,101)
    cc=[constants(g) for g in hs]
    axes[0,0].plot(hs,[r["rho"] for r in cc],label=r"Graph bound $\rho$")
    axes[0,0].plot(hs,[r["kappa_cone"] for r in cc],label=r"Contraction radius $2a/b$")
    axes[0,0].set(xlabel=r"Skin magnitude $|g|$",ylabel="Tangent radius",title="(a) CDW starts inside the contraction cone")
    axes[0,0].legend()
    ts=np.linspace(0,6,240)
    for g,color in ((0.,"#277db1"),(.25,"#be4343")):
        c=constants(g)
        decay=np.exp(-2*c["a"]*ts)
        k=c["rho"]*decay/(1-c["b"]*c["rho"]/(2*c["a"])*(1-decay))
        axes[0,1].semilogy(ts,k/np.sqrt(1+k*k),color=color,label=f"CDW bound, g={g:g}")
        axes[0,1].semilogy(ts,decay,color=color,ls="--",lw=.8)
    axes[0,1].set(xlabel="Time t",ylabel="Distance upper bound",title="(b) Fast CDW and output-space selection",ylim=(1e-7,1))
    axes[0,1].legend(loc="lower left")
    c=constants(.25)
    eta=(1-c["rho"]**2)/(1+c["rho"]**2)
    rate=(1.25+math.cosh(.25))/4
    const=(.5+math.exp(.25))/(4*(1-rate))
    decay=np.exp(-2*c["a"]*ts)
    kc=c["rho"]*decay/(1-c["b"]*c["rho"]/(2*c["a"])*(1-decay))
    cdw=kc/np.sqrt(1+kc*kc)
    for length,color in ((128,"#277db1"),(256,"#be4343"),(512,"#177c57")):
        zeta=const*rate**(length//4-1)/eta
        lower=eta/(1+zeta/(1-zeta)*np.exp(2*(2.5+math.exp(.25))*ts))
        axes[1,0].plot(ts,np.maximum(0,lower-cdw),color=color,label=f"L={length}")
    axes[1,0].axhline(.1,color="#666666",lw=.8,ls=":")
    axes[1,0].set(xlabel="Time t",ylabel="Block-CDW memory lower bound",title="(c) Analytic retained-memory intervals")
    axes[1,0].legend()
    ref=[v["values"] for v in references]
    axes[1,1].scatter([-float(v["theoretical_slope_upper"]) for v in ref],[-float(v["actual_slope"]) for v in ref],s=9,alpha=.6,color="#277db1")
    hi=max(-float(v["actual_slope"]) for v in ref)*1.08
    axes[1,1].plot([0,hi],[0,hi],color="#666666",ls=":",lw=.8)
    axes[1,1].set(xlabel="Guaranteed decay slope",ylabel="Independent actual decay slope",title="(d) Continuous slope bound audit",xlim=(0,hi),ylim=(0,hi))
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
    for ext in ("png","pdf"):
        fig.savefig(FIG/("analytic_envelopes."+ext))
    plt.close(fig)
    return dict(status="figures_created",files=[p.relative_to(ROOT).as_posix() for p in sorted(FIG.glob("*"))])


def manifest():
    excluded={"deployment.tar.gz","results.tar.gz","final_sync.tar.gz","return_manifest.csv","local_inventory.json","artifact_manifest.csv",
              "deployment_manifest.csv","final_sync_manifest.csv","operations.jsonl","delivery_verification.json","sync_receipt.json"}
    paths=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in excluded)
    paths.extend(p for p in FIG.glob("*") if p.is_file())
    rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=study.sha(p)) for p in paths]
    write_csv(OUT/"artifact_manifest.csv",rows)
    return dict(status="manifest_created",files=len(rows),bytes=sum(r["bytes"] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("backup","summary","figures","manifest"))
    p.add_argument("--complete",action="store_true")
    a=p.parse_args()
    result=backup() if a.stage=="backup" else summarize(a.complete) if a.stage=="summary" else figures() if a.stage=="figures" else manifest()
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
