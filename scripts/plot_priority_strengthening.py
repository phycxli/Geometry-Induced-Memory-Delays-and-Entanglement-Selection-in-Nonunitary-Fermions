"""Plot static input response alongside frozen actual-time predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np

import plot_prl_manuscript_figures as base
import run_priority_strengthening as study

ROOT, OUT, FIG = base.ROOT, study.OUT, base.FIG
MARKERS = {"left":"s","right":"D"}


def rows(path):
    with (ROOT/path).open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def label(fig,axis,letter,y):
    box=axis.get_position()
    fig.text((box.x0+box.x1)/2,y,f"({letter})",ha="center",fontsize=9)


def finish(fig,target):
    for axis in fig.axes:
        base.style(axis)
        assert not axis.get_title()
        if axis.get_yscale()=="linear":
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    target.parent.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","png"):
        metadata={"CreationDate":None,"ModDate":None} if ext=="pdf" else None
        fig.savefig(target.with_suffix("."+ext),dpi=600,facecolor="white",metadata=metadata)
    plt.close(fig)


def geometry(data,copy_projects=True):
    static=rows("data/prl_figure_strengthening/signed_response.csv")
    fig=plt.figure(figsize=(3.4,3.3))
    axes=[fig.add_axes((.19,.67,.775,.305)),fig.add_axes((.19,.18,.775,.305))]
    for name in ("left","right"):
        color=base.COLORS[name]
        sign=1 if name=="left" else -1
        chi=np.linspace(-4,4,100)
        axes[0].plot(chi,sign*chi/2,color="#8b8f93",ls=(0,(1.2,2)),lw=.8)
        for length,ls in ((64,"-"),(512,(0,(4,2.3)))):
            local=sorted([r for r in static if r["initial"]==name and int(r["length"])==length],key=lambda r:float(r["chi"]))
            axes[0].plot([float(r["chi"]) for r in local],[float(r["log_ratio_to_zero"]) for r in local],
                color=color,marker=MARKERS[name],ms=3.4,lw=1,ls=ls,markeredgewidth=.7,
                markerfacecolor=color if length==64 else "white")
        for length,ls in ((128,"-"),(512,(0,(4,2.3)))):
            local=sorted([r for r in data if r["initial"]==name and int(r["length"])==length and float(r["level"])==.05],key=lambda r:float(r["chi"]))
            slope=(float(local[0]["predicted_time"])-float(local[0]["zero_time"]))/float(local[0]["chi"])
            axes[1].plot(chi,slope*chi,color=color,lw=.95,ls=ls)
            axes[1].plot([float(r["chi"]) for r in local],[float(r["root_time"])-float(r["zero_time"]) for r in local],
                color=color,marker=MARKERS[name],ms=3.5,ls="none",markeredgewidth=.7,
                markerfacecolor=color if length==128 else "white")
    axes[0].set(xlabel=r"$\chi=gL$",ylabel=r"$R_s(\chi,L)$",xlim=(-4.3,4.3),ylim=(-2.3,2.3),xticks=(-4,-2,0,2,4),yticks=(-2,0,2))
    axes[1].set(xlabel=r"$\chi=gL$",ylabel=r"$t_{0.05}(\chi)-t_{0.05}(0)$",xlim=(-4.3,4.3),ylim=(-.82,.82),xticks=(-4,-2,0,2,4),yticks=(-.8,0,.8))
    handles=[Line2D([],[],color=base.COLORS[n],marker=MARKERS[n],lw=1,ms=3.3,label=base.NAMES[n]) for n in ("left","right")]
    legend=axes[0].legend(handles=handles,loc="upper center",ncol=2,frameon=False,fontsize=6.3,handlelength=1.5)
    axes[0].add_artist(legend)
    axes[0].legend(handles=[Line2D([],[],color="#51565b",ls=ls,label=f"$L={length}$") for length,ls in ((64,"-"),(512,(0,(4,2.3))))],
                   loc="lower center",frameon=False,fontsize=6.1,ncol=2,handlelength=1.8)
    axes[1].legend(handles=[Line2D([],[],color="#51565b",ls=ls,marker="o",markerfacecolor=face,ms=3,label=f"$L={length}$")
                    for length,ls,face in ((128,"-","#51565b"),(512,(0,(4,2.3)),"white"))],
                   loc="upper center",frameon=False,fontsize=6.2,ncol=2,handlelength=1.8)
    for axis,letter,y in zip(axes,"ab",(.535,.04)):
        label(fig,axis,letter,y)
    finish(fig,FIG/"fig3_reflection_response")
    for name in (("main","main_zh","supplement","supplment_zh") if copy_projects else ()):
        for ext in ("pdf","png"):
            target=ROOT/name/"figures/prl_manuscript_revision"/f"fig3_reflection_response.{ext}"
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(FIG/f"fig3_reflection_response.{ext}",target)
    return dict(panels=2,static_lengths=[64,512],dynamical_lengths=[128,512],threshold=.05,
                prediction="Lines locked from the zero-bias principal-pair response before nonzero-bias propagation",
                finite_size_scope="Measured sign and prediction accuracy; no asymptotic actual-time coefficient inferred")


def supplemental(data,zero,target=None,copy_projects=True):
    fig=plt.figure(figsize=(6.8,4.9))
    axes=[fig.add_axes(box) for box in ((.10,.64,.36,.33),(.61,.64,.36,.33),(.10,.15,.36,.33),(.61,.15,.36,.33))]
    theory=1/(4*np.sqrt(4-1.5**2))
    for name in ("left","right"):
        local=sorted([r for r in zero if r["initial"]==name and float(r["level"])==.05 and not r["family"].startswith("control")],key=lambda r:int(r["length"]))
        length=[int(r["length"]) for r in local]
        for index,field in ((0,"susceptibility"),(1,"geometry_factor"),(2,"decay_rate")):
            y=[abs(float(r[field])) for r in local]
            if index==2:
                y=[100*(v/(2*np.sqrt(4-1.5**2))-1) for v in y]
            axes[index].plot(length,y,color=base.COLORS[name],marker=MARKERS[name],ms=3.8,lw=1)
    axes[0].axhline(theory,color="#51565b",lw=.9,ls="--")
    for sign in (-1,1):
        axes[0].axhline(theory+sign*.005,color="#9b9b9b",lw=.65,ls=":")
    axes[0].set(ylabel=r"$|\partial_\chi t_{0.05}(0)|$",ylim=(.15,.199),yticks=(.15,.17,.19))
    axes[0].legend(handles=[Line2D([],[],color=base.COLORS[n],marker=MARKERS[n],label=base.NAMES[n],lw=1,ms=3.8) for n in ("left","right")],loc="lower right",frameon=False,fontsize=7)
    axes[1].axhline(.5,color="#51565b",lw=.9,ls="--")
    axes[1].set(ylabel=r"$|\Gamma|$",ylim=(.418,.507),yticks=(.42,.46,.50))
    axes[2].set(ylabel=r"$100[\Lambda/(2\mu_0)-1]$",ylim=(0,2.4),yticks=(0,1,2))
    for axis in axes[:3]:
        axis.set(xlabel=r"$L$",xlim=(108,532),xticks=(128,256,384,512))
    for index,chi in enumerate((.5,1,2,4)):
        local=[r for r in data if abs(float(r["chi"]))==chi]
        for offset,field,color,marker in ((-.025,"linear_error","#21866e","o"),(.025,"static_gap_error","#72548b","x")):
            axes[3].semilogy([chi+offset]*len(local),[max(abs(float(r[field])),1e-9) for r in local],ls="none",marker=marker,
                             color=color,ms=2.7,markeredgewidth=.5,alpha=.6)
    axes[3].axhline(.01,color="#51565b",ls="--",lw=.9)
    axes[3].set(xlabel=r"$|\chi|$",ylabel="Absolute time error",xlim=(.25,4.25),ylim=(2e-6,.5),xticks=(.5,1,2,4))
    axes[3].legend(handles=[Line2D([],[],color=color,marker=marker,ls="none",label=name,ms=3.3) for color,marker,name in
                            (("#21866e","o","Principal pair"),("#72548b","x","Static / gap"))],
                   loc="upper left",frameon=False,fontsize=7,handletextpad=.3)
    for axis,letter,y in zip(axes,"abcd",(.53,.53,.035,.035)):
        label(fig,axis,letter,y)
    destination=OUT/"figures/time_response_validation" if target is None else target
    finish(fig,destination)
    for name in (("supplement","supplment_zh") if copy_projects else ()):
        directory=ROOT/name/"figures/prl_priority_strengthening"
        directory.mkdir(parents=True,exist_ok=True)
        for ext in ("pdf","png"):
            shutil.copy2(destination.with_suffix("."+ext),directory/f"time_response_validation.{ext}")


def render():
    data=rows("data/prl_priority_strengthening/time_response.csv")
    zero=rows("data/prl_priority_strengthening/zero_bias_response.csv")
    assert len(data)==528 and len(zero)==66
    inputs=[ROOT/p for p in ("data/prl_figure_strengthening/signed_response.csv",
                             "data/prl_priority_strengthening/time_response.csv",
                             "data/prl_priority_strengthening/zero_bias_response.csv", "scripts/plot_priority_strengthening.py")]
    before={p.relative_to(ROOT).as_posix():study.digest(p) for p in inputs}
    plt.rcParams.update({"font.family":"Arial","mathtext.fontset":"stix","font.size":8,"axes.labelsize":8.5,
                         "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"pdf.fonttype":42,"axes.linewidth":.65})
    details=geometry(data)
    supplemental(data,zero)
    assert before=={p.relative_to(ROOT).as_posix():study.digest(p) for p in inputs}
    outputs=[FIG/f"fig3_reflection_response.{ext}" for ext in ("pdf","png")]
    outputs += [OUT/f"figures/time_response_validation.{ext}" for ext in ("pdf","png")]
    record=dict(status="generated",generated_utc=study.old.now(),source_inputs=before,
                figures={"fig3_reflection_response":details},outputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=study.digest(p),bytes=p.stat().st_size) for p in outputs],
                background="white",panel_titles=False,panel_label_position="below axes")
    study.save(OUT/"figure_provenance.json",record)
    path=base.OUT/"figure_provenance.json"
    previous=study.load(path)
    if not (OUT/"figure_provenance_before.json").exists():
        study.save(OUT/"figure_provenance_before.json",previous)
    previous["figures"].update(record["figures"])
    previous["source_inputs"].update(before)
    previous.update(generated_utc=record["generated_utc"],source_sha256=study.digest(Path(__file__)),
                    strengthening_evidence="data/prl_priority_strengthening/figure_provenance.json",redrawn_figures=["fig3_reflection_response"])
    previous["outputs"]=[dict(path=(FIG/f"{name}.{ext}").relative_to(ROOT).as_posix(),sha256=study.digest(FIG/f"{name}.{ext}"),bytes=(FIG/f"{name}.{ext}").stat().st_size)
                          for name in previous["figures"] for ext in ("pdf","png")]
    study.save(path,previous)
    return record


if __name__=="__main__":
    print(json.dumps(render(),ensure_ascii=False),flush=True)
