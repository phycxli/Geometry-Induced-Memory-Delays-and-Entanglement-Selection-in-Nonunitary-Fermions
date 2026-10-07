"""Publication plots from the independently checked figure-strengthening batch."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np

import plot_prl_manuscript_figures as base
import analyze_theory_upgrade as theory

ROOT = base.ROOT
OUT, FIG = ROOT/"data/prl_figure_strengthening", base.FIG
COLORS, NAMES = base.COLORS, base.NAMES
MARKERS = {"left":"s","right":"D","block1":"^","block3":"v"}
STYLES = {0.:"-",.125:(0,(1.2,1.8)),.25:(0,(4,2.3))}


def rows(name):
    with (OUT/name).open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def bottom_label(fig,axis,letter,y):
    position = axis.get_position()
    fig.text((position.x0+position.x1)/2,y,f"({letter})",ha="center",fontsize=9)


def line_handle(color,label,marker=None,ls="-",face=None):
    return Line2D([],[],color=color,label=label,marker=marker,ls=ls,lw=1,ms=3.7,
                  markeredgewidth=.6,markerfacecolor=color if face is None else face)


def prediction_skin(row):
    return float(row["family_key"].split("_g",1)[1].replace("p","."))


def finish(fig,stem):
    for axis in fig.axes:
        base.style(axis)
        axis.tick_params(labelsize=7.2)
        if axis.get_yscale()=="linear":
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    base.save(fig,stem)


def flagship():
    trace = [r for r in rows("traces.csv") if r["mode"]=="full"]
    early = rows("early_output.csv")
    widths = rows("widths.csv")
    predictions = rows("prediction_checks.csv")
    snapshots = base.read_csv("data/prl_manuscript_revision/fig2_same_protocol.csv")
    fig = plt.figure(figsize=(7.1,2.5))
    axes = [fig.add_axes(box) for box in ((.065,.25,.292,.635),(.43,.25,.19,.635),(.705,.25,.279,.635))]
    distance,entropy,times = axes
    for name in ("charge_density_wave","left","right"):
        local = sorted([r for r in trace if r["initial"]==name],key=lambda r:float(r["time"]))
        distance.plot([float(r["time"]) for r in local],[float(r["distance_to_G"]) for r in local],
                      color=COLORS[name],lw=1.25,label=NAMES[name])
    output_times=np.linspace(0,128,180)
    distance.plot(output_times,np.exp(-2*theory.constants(.25)["a"]*output_times),
                  color="#3f4449",ls=(0,(1.2,2)),lw=1,label=r"$P_W$ bound")
    distance.axhline(.05,color="#8b8f93",ls=(0,(1.2,2)),lw=.7)
    distance.axvline(93.125,color="#8b8f93",ls=(0,(1.2,2)),lw=.7)
    distance.set(xlabel=r"$t$",ylabel=r"$d(P,P_G)$",xlim=(0,128),ylim=(-.025,1.04),xticks=(0,40,80,120),yticks=(0,.5,1))
    distance.legend(loc="upper left",bbox_to_anchor=(.035,.955),frameon=False,fontsize=6.6,
                    ncol=2,columnspacing=.7,handlelength=1.5)
    zoom = distance.inset_axes((.12,.24,.44,.42))
    cdw = [r for r in trace if r["initial"]=="charge_density_wave" and float(r["time"])<=4]
    zoom.plot([float(r["time"]) for r in cdw],[float(r["distance_to_G"]) for r in cdw],color=COLORS["charge_density_wave"],lw=.9)
    zoom.plot([float(r["time"]) for r in early if float(r["time"])<=4],
              [float(r["distance_to_G"]) for r in early if float(r["time"])<=4],color="#3f4449",ls=(0,(4,2)),lw=.8)
    zoom.set(xlim=(0,4),ylim=(-.01,.42),xticks=(0,2,4),yticks=(0,.2,.4))
    zoom.tick_params(labelsize=5.8,pad=2)
    zoom.set_xlabel(r"$t$",fontsize=6,labelpad=0)
    for i,name in enumerate(("charge_density_wave","left","right")):
        value = float(next(r["entropy"] for r in snapshots if r["initial"]==name))
        entropy.vlines(i,0,value,colors=COLORS[name],lw=.8)
        entropy.plot(i,value,marker="o" if i==0 else MARKERS[name],color=COLORS[name],ms=5)
        entropy.annotate(f"{value:.4f}",(i,value),xytext=(0,6),textcoords="offset points",ha="center",fontsize=7)
    selected_entropy = float(snapshots[0]["W_entropy"])
    entropy.axhline(selected_entropy,color="#3f4449",ls=(0,(4,2)),lw=.8)
    entropy.set(ylabel=r"$S_A$ (nat)",xticks=(0,1,2),xticklabels=("CDW","Left","Right"),
                xlim=(-.5,2.5),ylim=(0,.36),yticks=(0,.1,.2,.3))
    entropy.legend(handles=[line_handle("#3f4449",r"$S_A(P_W)$",ls=(0,(4,2)))],loc="upper left",frameon=False,fontsize=6.7)
    for name in ("left","right"):
        for g in (0.,.125,.25):
            local = sorted([r for r in widths if r["initial"]==name and float(r["skin"])==g],key=lambda r:int(r["length"]))
            if g!=.125:
                frozen = next(r for r in predictions if r["kind"]==name and r["level"]=="0.05"
                              and prediction_skin(r)==g)
                coef = json.loads(frozen["coefficients"])
                x = np.linspace(128,384,100)
                times.plot(x,np.polyval(coef,x),color=COLORS[name],ls=STYLES[g],lw=.9)
            else:
                times.plot([int(r["length"]) for r in local],[float(r["t05"]) for r in local],color=COLORS[name],ls=STYLES[g],lw=.85)
            for row in local:
                times.plot(int(row["length"]),float(row["t05"]),marker=MARKERS[name],color=COLORS[name],ls="none",ms=3.7,
                           markeredgewidth=.7,markerfacecolor="white" if int(row["length"])>=320 else COLORS[name])
    bound = max(base.brentq(lambda t:base.cdw_projection_bound(t,g)-.05,0,20) for g in (0.,.125,.25))
    times.axhline(bound,color=COLORS["charge_density_wave"],lw=.85)
    times.text(130,10,"CDW bound",color=COLORS["charge_density_wave"],fontsize=6.2)
    times.set(xlabel=r"$L$",ylabel=r"$t_{0.05}$",xlim=(117,395),ylim=(-2,230),xticks=(128,256,384),yticks=(0,75,150,225))
    first = times.legend(handles=[line_handle(COLORS[n],NAMES[n],MARKERS[n]) for n in ("left","right")],
                         loc="upper left",frameon=False,fontsize=6.6,handlelength=1.1)
    times.add_artist(first)
    times.legend(handles=[line_handle("#51565b",label,ls=STYLES[g]) for g,label in
                         ((0.,r"$g=0$"),(.125,r"$g=1/8$"),(.25,r"$g=1/4$"))],loc="lower right",frameon=False,fontsize=6.4,handlelength=2.2)
    for axis,letter in zip(axes,"abc"):
        bottom_label(fig,axis,letter,.035)
    fig.text(.211,.95,r"$L=192,\ g=1/4$",ha="center",fontsize=8)
    fig.text(.525,.95,r"$t=93.125$",ha="center",fontsize=8)
    finish(fig,"fig2_geometry_memory")
    return dict(panels=3,time_trace_points=len(trace),early_output_points=len(early),
                lengths=sorted({int(r["length"]) for r in widths}),skins=[0,.125,.25],
                scope="a: actual distances to G with early output-to-G SVD; b: retained W-referenced entropy snapshot; c: frozen affine predictions at g=0,1/4 and numerical g=1/8 guides")


def response_residuals():
    data = []
    for r in base.read_csv("data/prl_manuscript_revision/fig3_weak_skin.csv"):
        sign = 1 if r["initial"]=="left" else -1
        chi = float(r["chi"])
        data.append(dict(length=int(r["length"]),initial=r["initial"],chi=chi,
                         error=abs(float(r["log_ratio_to_zero"])/chi-sign/2)))
    for r in rows("signed_response.csv"):
        chi = float(r["chi"])
        if chi==0:
            continue
        sign = 1 if r["initial"]=="left" else -1
        row = dict(length=int(r["length"]),initial=r["initial"],chi=chi,
                   error=abs(float(r["log_ratio_to_zero"])/chi-sign/2))
        existing = next((x for x in data if all(x[k]==row[k] for k in ("length","initial","chi"))),None)
        if existing:
            assert abs(existing["error"]-row["error"])<1e-8
        else:
            data.append(row)
    return data


def geometry():
    signed = rows("signed_response.csv")
    residuals = response_residuals()
    fig = plt.figure(figsize=(3.4,3.3))
    axes = [fig.add_axes((.19,.67,.775,.305)),fig.add_axes((.19,.18,.775,.305))]
    for name in ("left","right"):
        sign = 1 if name=="left" else -1
        x = np.linspace(-4,4,50)
        axes[0].plot(x,sign*x/2,color="#8b8f93",ls=(0,(1.2,2)),lw=.8)
        for length,ls in ((64,"-"),(512,(0,(4,2.3)))):
            local = sorted([r for r in signed if r["initial"]==name and int(r["length"])==length],key=lambda r:float(r["chi"]))
            axes[0].plot([float(r["chi"]) for r in local],[float(r["log_ratio_to_zero"]) for r in local],
                         color=COLORS[name],marker=MARKERS[name],ms=3.4,lw=1,ls=ls,
                         markeredgewidth=.7,markerfacecolor=COLORS[name] if length==64 else "white")
        for chi,ls in ((4.,"-"),(-4.,(0,(4,2.3)))):
            local = sorted([r for r in residuals if r["initial"]==name and r["chi"]==chi],key=lambda r:r["length"])
            axes[1].semilogy([r["length"] for r in local],[r["error"] for r in local],
                             color=COLORS[name],marker=MARKERS[name],ms=3.3,lw=1,ls=ls,
                             markeredgewidth=.7,markerfacecolor=COLORS[name] if chi>0 else "white")
    axes[0].set(xlabel=r"$\chi=gL$",ylabel=r"$R_s(\chi,L)$",
                xlim=(-4.3,4.3),ylim=(-2.3,2.3),xticks=(-4,-2,0,2,4),yticks=(-2,0,2))
    family = axes[0].legend(handles=[line_handle(COLORS[n],NAMES[n],MARKERS[n]) for n in ("left","right")],
                           loc="upper center",frameon=False,fontsize=6.3,handlelength=1.5,ncol=2,columnspacing=1)
    axes[0].add_artist(family)
    axes[0].legend(handles=[line_handle("#51565b",r"$L=64$",ls="-"),line_handle("#51565b",r"$L=512$",ls=(0,(4,2.3))),
                           line_handle("#8b8f93",r"$s\chi/2$",ls=(0,(1.2,2)))],
                   loc="lower center",frameon=False,fontsize=6.1,handlelength=1.6,borderaxespad=.3,ncol=3,columnspacing=.8)
    axes[1].set(xlabel=r"$L$",ylabel=r"$|R_s/\chi-s/2|$",xlim=(48,528),xticks=(64,256,512))
    axes[1].legend(handles=[line_handle("#51565b",r"$\chi=+4$",ls="-"),line_handle("#51565b",r"$\chi=-4$",ls=(0,(4,2.3)))],
                   loc="upper right",frameon=False,fontsize=6.4,handlelength=2)
    for axis,letter,y in zip(axes,"ab",(.535,.04)):
        bottom_label(fig,axis,letter,y)
    finish(fig,"fig3_reflection_response")
    return dict(panels=2,main_lengths=[64,512],signed_chi=[-4,-2,-1,0,1,2,4],
                finite_size_errors="absolute error to proved slope for chi=+/-4; connectors are numerical guides",
                scope="reflected static cross-block scale, no actual-time-shift law")


def contraction_envelope(duration,skin,initial_distance=.05):
    c = theory.constants(skin)
    k0 = initial_distance/np.sqrt(1-initial_distance**2)
    e = np.exp(-2*c["a"]*np.asarray(duration))
    k = k0*e/(1-c["b"]*k0/(2*c["a"])*(1-e))
    return k/np.sqrt(1+k*k)


def window():
    traces, widths, references = rows("traces.csv"),rows("widths.csv"),rows("independent_checks.csv")
    fig = plt.figure(figsize=(3.4,3.3))
    axes = [fig.add_axes((.20,.67,.765,.305)),fig.add_axes((.20,.18,.765,.305))]
    centers = {}
    tail_values=[]
    for index,name in enumerate(("left","right","block3")):
        local = sorted([r for r in traces if r["mode"]=="window" and int(r["length"])==384
                        and base.kind(r)==name and float(r["distance_to_G"])<=.3],key=lambda r:float(r["time"]))
        centers[local[0]["initial"]] = float(local[0]["center"])
        tail_values.extend(float(r["distance_to_G"]) for r in local)
        axes[0].semilogy([float(r["aligned_time"]) for r in local],[float(r["distance_to_G"]) for r in local],
                         color=COLORS[name],lw=1.1,label=NAMES[name],
                         ls=("-",(0,(1.2,1.8)),(0,(5,2,1,2)))[index],
                         marker=MARKERS[name],ms=2.6,markevery=(index,6),markeredgewidth=.5)
    shown = [r for r in references if int(r["length"])==384 and float(r["skin"])==.25 and r["initial"] in centers]
    axes[0].scatter([float(r["duration"])-centers[r["initial"]] for r in shown],[float(r["distance_to_G"]) for r in shown],
                    s=10,facecolors="white",edgecolors="#202124",lw=.6,zorder=4)
    x = np.linspace(0,12,100)
    axes[0].semilogy(x,contraction_envelope(x,.25),color="#51565b",ls=(0,(4,2.3)),lw=.9,label="Upper envelope")
    lower=10.**(np.floor(np.log10(min(tail_values)))-1)
    ticks=[10.**exponent for exponent in range(-1,int(np.floor(np.log10(lower)))-1,-5)]
    axes[0].set(xlabel=r"$t-t_{0.05}$",ylabel=r"$d(P,P_G)$",xlim=(-.8,12.2),ylim=(lower,.4),
                xticks=(0,4,8,12),yticks=ticks)
    axes[0].legend(loc="upper right",frameon=False,fontsize=6.,ncol=2,columnspacing=.6,
                   handlelength=1.7,labelspacing=.15)
    for name in ("left","right","block1","block3"):
        for g in (0.,.25):
            local = sorted([r for r in widths if r["kind"]==name and float(r["skin"])==g],key=lambda r:int(r["length"]))
            axes[1].plot([int(r["length"]) for r in local],[100*float(r["relative_width_30_to_01"]) for r in local],
                         color=COLORS[name],ls=STYLES[g],marker=MARKERS[name],lw=.9,ms=3.1,
                         markeredgewidth=.6,markerfacecolor=COLORS[name] if g==0 else "white")
    axes[1].set(xlabel=r"$L$",ylabel=r"$\Delta t/t_{0.05}$ (%)",xlim=(117,395),xticks=(128,256,384))
    family = axes[1].legend(handles=[line_handle(COLORS[n],NAMES[n],MARKERS[n]) for n in MARKERS],
                           loc="upper right",frameon=False,fontsize=5.9,ncol=2,columnspacing=.6,handlelength=1.3,labelspacing=.2)
    axes[1].add_artist(family)
    axes[1].legend(handles=[line_handle("#51565b",r"$g=0$",ls="-"),line_handle("#51565b",r"$g=1/4$",ls=STYLES[.25])],
                   loc="lower left",frameon=False,fontsize=6.2,handlelength=2)
    for axis,letter,y in zip(axes,"ab",(.535,.04)):
        bottom_label(fig,axis,letter,y)
    finish(fig,"fig4_continuous_window")
    return dict(panels=2,independent_displayed_points=len(shown),window_length=384,window_skin=.25,
                post_selection_duration=12,relative_width_definition="(t_.01-t_.3)/t_.05",
                scope="two-precision actual low-threshold samples and independent propagation, analytic G persistence envelope; no universal-profile assertion")


def supplemental():
    widths = rows("widths.csv")
    signed = rows("signed_response.csv")
    reflection = rows("reflection_checks.csv")
    predictions = rows("prediction_checks.csv")
    fig, axes = plt.subplots(3,2,figsize=(6.8,7.8))
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.98,wspace=.34,hspace=.73)
    axes = list(axes.flat)
    original = base.read_csv("data/prl_manuscript_revision/fig3_reflected_geometry.csv")
    for name in ("left","right"):
        for g in (0.,.25):
            local = sorted([r for r in original if r["initial"]==name and float(r["skin"])==g],key=lambda r:int(r["length"]))
            axes[0].plot([int(r["length"]) for r in local],[float(r["log_delta"])/np.log(10) for r in local],
                         color=COLORS[name],ls=STYLES[g],marker=MARKERS[name],ms=3,lw=1,
                         markerfacecolor=COLORS[name] if g==0 else "white",label=NAMES[name]+fr", $g={g:g}$")
        for g in (-.25,0.,.25):
            local = sorted([r for r in reflection if r["initial"]==name and float(r["skin"])==g],key=lambda r:int(r["length"]))
            axes[1].plot([int(r["length"]) for r in local],[float(r["log_bulk_to_reflected"])/np.log(10) for r in local],
                         color=COLORS[name],ls="-" if g==0 else (0,(4,2)) if g>0 else (0,(1.2,1.8)),
                         marker=MARKERS[name],ms=3,lw=1,markerfacecolor=COLORS[name] if g==0 else "white")
        for length,ls in ((64,"-"),(128,(0,(1.2,1.8))),(512,(0,(4,2.3)))):
            local = sorted([r for r in signed if r["initial"]==name and int(r["length"])==length],key=lambda r:float(r["chi"]))
            axes[2].plot([float(r["chi"]) for r in local],[float(r["log_ratio_to_zero"]) for r in local],
                         color=COLORS[name],ls=ls,marker=MARKERS[name],ms=3,lw=1,markerfacecolor=COLORS[name] if length==64 else "white")
    for name in MARKERS:
        for g in (0.,.125,.25):
            local = sorted([r for r in widths if r["kind"]==name and float(r["skin"])==g],key=lambda r:int(r["length"]))
            axes[3].plot([int(r["length"]) for r in local],[float(r["width_30_to_01"]) for r in local],
                         color=COLORS[name],ls=STYLES[g],marker=MARKERS[name],ms=3,lw=.9,
                         markerfacecolor=COLORS[name] if g==0 else "white")
    old_roots = base.read_csv("data/prl_manuscript_revision/fig4_threshold_scans.csv")
    centers = {(r["family_key"],r["initial"]):float(r["root_time"]) for r in old_roots if r["level"]=="0.05"}
    for name,g in (("left",0.),("left",.25),("right",.25)):
        for length in (128,160,192,224,256):
            local = sorted([r for r in old_roots if r["initial"]==name and float(r["skin"])==g and int(r["length"])==length],key=lambda r:float(r["root_time"]))
            center = centers[local[0]["family_key"],name]
            axes[4].plot([float(r["root_time"])-center for r in local],[float(r["level"]) for r in local],
                         color=COLORS[name],ls=STYLES[g],marker=MARKERS[name],ms=2.5,lw=.8,alpha=.7)
    axes[4].axhline(.484,color="#51565b",ls=(0,(1.2,2)),lw=.8)
    for name in MARKERS:
        for g in (0.,.25):
            local = [r for r in predictions if r["kind"]==name and prediction_skin(r)==g]
            axes[5].scatter([int(r["family_key"].split("_")[0][1:]) for r in local],[float(r["signed_error"]) for r in local],
                            marker=MARKERS[name],s=14,edgecolors=COLORS[name],facecolors=COLORS[name] if g==0 else "none",lw=.7)
    axes[5].axhline(0,color="#8b8f93",ls=(0,(1.2,2)),lw=.7)
    definitions = ((r"$L$",r"$\log_{10}\delta$"),(r"$L$",r"$\log_{10}(\delta_{\rm bulk}/\delta_{\rm refl})$"),
                   (r"$\chi=gL$",r"$R_s$"),(r"$L$",r"$\Delta t=t_{0.01}-t_{0.3}$"),
                   (r"$t-t_{0.05}$",r"$d(P,P_G)$"),(r"$L$",r"$t_\epsilon-t_\epsilon^{\rm pred}$"))
    for axis,(xlabel,ylabel),letter in zip(axes,definitions,"abcdef"):
        axis.set(xlabel=xlabel,ylabel=ylabel)
        base.style(axis)
        position=axis.get_position()
        bottom_label(fig,axis,letter,position.y0-.072)
    axes[0].legend(frameon=False,fontsize=6.1,loc="lower left",handlelength=1.8)
    axes[1].legend(handles=[line_handle("#51565b",label,ls=ls) for label,ls in
                           ((r"$g=-1/4$",(0,(1.2,1.8))),(r"$g=0$","-"),(r"$g=1/4$",(0,(4,2))))],frameon=False,fontsize=6.3)
    axes[2].legend(handles=[line_handle("#51565b",fr"$L={length}$",ls=ls) for length,ls in
                           ((64,"-"),(128,(0,(1.2,1.8))),(512,(0,(4,2.3))))],frameon=False,fontsize=6.3,loc="upper center")
    axes[3].legend(handles=[line_handle("#51565b",label,ls=STYLES[g]) for g,label in
                           ((0.,r"$g=0$"),(.125,r"$g=1/8$"),(.25,r"$g=1/4$"))],frameon=False,fontsize=6.3,loc="lower right")
    axes[4].set(xlim=(-1.6,.75),ylim=(-.025,1.03))
    axes[5].set(xticks=(320,384),xlim=(304,400))
    axes[5].legend(handles=[line_handle(COLORS[n],NAMES[n],MARKERS[n]) for n in MARKERS],
                   frameon=False,fontsize=6.1,ncol=2,loc="best",handlelength=1,columnspacing=.7)
    finish(fig,"figS_strengthened_validation")
    return dict(panels=6,absolute_widths=len(widths),reflection_comparisons=len(reflection),frozen_predictions=len(predictions),
                scope="complete parameter controls and retained large-angle diagnostics; no interval certification")


def render(targets=None):
    assert json.loads((OUT/"summary.json").read_text(encoding="utf-8"))["status"]=="complete"
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"Arial","mathtext.fontset":"stix","font.size":8,"axes.labelsize":8.5,
                         "xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42,"axes.linewidth":.65})
    functions = {"fig2_geometry_memory":flagship,"fig3_reflection_response":geometry,"fig4_continuous_window":window,
                 "figS_strengthened_validation":supplemental}
    selected = list(functions) if targets is None else targets
    details = {name:functions[name]() for name in selected}
    inputs = [OUT/name for name in ("roots.csv","widths.csv","traces.csv","early_output.csv","signed_response.csv",
              "reflection_checks.csv","independent_checks.csv","prediction_checks.csv","prediction_lock.json","summary.json")]
    inputs += [base.OUT/name for name in ("fig2_same_protocol.csv","fig3_weak_skin.csv","fig3_reflected_geometry.csv","fig4_threshold_scans.csv")]
    inputs.append(ROOT/"scripts/analyze_theory_upgrade.py")
    record = dict(status="generated",generated_utc=base.datetime.now(base.timezone.utc).isoformat(),source_sha256=base.digest(Path(__file__)),
                  source_inputs={p.relative_to(ROOT).as_posix():base.digest(p) for p in inputs},figures=details,
                  outputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=base.digest(p),bytes=p.stat().st_size)
                           for name in selected for p in (FIG/(name+".pdf"),FIG/(name+".png"))],
                  background="white",panel_titles=False,panel_label_position="below axes",new_scientific_computations=True)
    (OUT/"figure_provenance.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    previous_path = base.OUT/"figure_provenance.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous["figures"].update({k:v for k,v in details.items() if k.startswith("fig2") or k.startswith("fig3") or k.startswith("fig4")})
    previous["source_inputs"].update(record["source_inputs"])
    previous.update(generated_utc=record["generated_utc"],source_sha256=record["source_sha256"],new_scientific_computations=True,
                    strengthening_evidence="data/prl_figure_strengthening/figure_provenance.json",redrawn_figures=selected,
                    scope="Expanded numerical evidence with original historical CSVs preserved")
    previous["outputs"] = [dict(path=(FIG/(name+"."+ext)).relative_to(ROOT).as_posix(),sha256=base.digest(FIG/(name+"."+ext)),
                                bytes=(FIG/(name+"."+ext)).stat().st_size)
                           for name in previous["figures"] for ext in ("pdf","png")]
    previous_path.write_text(json.dumps(previous,indent=2),encoding="utf-8")
    return details


if __name__=="__main__":
    print(json.dumps(render(),ensure_ascii=False))
