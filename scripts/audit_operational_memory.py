"""Separate algebra/measurement audit and review-packet assembly."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import mpmath as mp
import run_operational_memory as study

ROOT, OUT = study.ROOT, study.OUT


def transpose_solve(a, b):
    return (mp.inverse(a.T) * b.T).T


def proof_algebra():
    rows = []
    with mp.workdps(90):
        for length in (8, 16, 24):
            n, m = length // 2, length // 4
            c = mp.matrix(n)
            for j in range(n):
                c[j,j] = mp.mpf(".5")
                if j:
                    c[j,j-1] = 1
            u, zeta, vt = mp.svd(c)
            v = vt.T
            mu = [mp.sqrt(4-z*z) for z in zeta]
            qplus, nminus = mp.matrix(length,n), mp.matrix(length,n)
            j0, omega, y = mp.matrix(length), mp.matrix(length), mp.matrix(length)
            for j in range(n):
                j0[j,j], j0[n+j,n+j] = 2, -2
                omega[j,2*n-1-j], omega[n+j,n-1-j] = 1, -1
                y[j,j] = y[n+j,n+j] = mp.mpf(j-(n-1)/2)/length
                for k in range(n):
                    j0[j,n+k], j0[n+k,j] = c[j,k], -c[j,k]
                    r = zeta[k]/(2+mu[k])
                    a = 1/mp.sqrt(1+r*r)
                    qplus[j,k], qplus[n+j,k] = a*u[j,k], -a*r*v[j,k]
                    nminus[j,k], nminus[n+j,k] = a*r*u[j,k], -a*v[j,k]
            modes = mp.matrix(length)
            modes[:,:n], modes[:,n:] = qplus, nminus
            inverse = mp.inverse(modes)

            def evolved(duration, initial):
                coefficients = inverse * initial
                for k in range(n):
                    for col in range(n):
                        coefficients[k,col] *= mp.exp(mu[k]*duration)
                        coefficients[n+k,col] *= mp.exp(-mu[k]*duration)
                return modes * coefficients

            def gauge(chi):
                return mp.diag([mp.exp(chi*y[j,j]) for j in range(length)])

            def graph(duration, initial, chi):
                d = gauge(chi)
                growing = mp.qr(d*qplus,mode="skinny")[0]
                orbitals = mp.qr(d*evolved(duration,initial),mode="skinny")[0]
                k = transpose_solve(growing.T*orbitals,(omega*growing).T*orbitals)
                vals, vecs = mp.eigh((k+k.T)/2)
                active = max(range(n),key=lambda index:abs(vals[index]))
                return k, vals[active], growing*vecs[:,active]

            for side, sign in (("left",1),("right",-1)):
                cells = range(m) if side=="left" else range(m,n)
                sites = list(cells) + [n+j for j in cells]
                initial = mp.matrix(length,n)
                for column,site in enumerate(sites):
                    initial[site,column] = 1
                lower, upper = mp.mpf(0), mp.mpf(length)
                epsilon = mp.mpf(".05")
                target = epsilon/mp.sqrt(1-epsilon*epsilon)
                for _ in range(44):
                    mid = (lower+upper)/2
                    _,ell,_ = graph(mid,initial,mp.mpf(0))
                    if abs(ell)>target:
                        lower = mid
                    else:
                        upper = mid
                duration = (lower+upper)/2
                direct = mp.expm(j0*duration)*initial
                direct_q = mp.qr(direct,mode="skinny")[0]
                modal_q = mp.qr(evolved(duration,initial),mode="skinny")[0]
                base = transpose_solve(qplus.T*modal_q,(omega*qplus).T*modal_q)
                direct_error = mp.norm(direct_q*direct_q.T-modal_q*modal_q.T)
                for chi in (mp.mpf("-.5"),mp.mpf(0),mp.mpf(".5")):
                    d = gauge(chi)
                    k,ell,p = graph(duration,initial,chi)
                    jchi = d*j0*mp.inverse(d)
                    gamma = -2*(p.T*y*p)[0]-2*ell*(p.T*y*omega*p)[0]
                    rate = 2*(p.T*jchi*p)[0]+ell*(p.T*jchi*omega*p)[0]
                    h = mp.mpf(".00001")
                    spatial = (mp.log(abs(graph(duration,initial,chi+h)[1]))-mp.log(abs(graph(duration,initial,chi-h)[1])))/(2*h)
                    temporal = (mp.log(abs(graph(duration+h,initial,chi)[1]))-mp.log(abs(graph(duration-h,initial,chi)[1])))/(2*h)
                    spatial_error, temporal_error = abs(spatial-gamma), abs(temporal+rate)
                    # This identity uses the positive square-root frame, not QR's gauge.
                    a_chi = qplus.T*d*d*qplus
                    eig, eigenvectors = mp.eigh(a_chi)
                    root = eigenvectors*mp.diag([mp.sqrt(x) for x in eig])*eigenvectors.T
                    growing = d*qplus*mp.inverse(root)
                    b_chi = growing.T*d*omega*qplus
                    transformed = mp.inverse(root)*base*mp.inverse(root+b_chi*base)
                    orbital = mp.qr(d*evolved(duration,initial),mode="skinny")[0]
                    actual = transpose_solve(growing.T*orbital,(omega*growing).T*orbital)
                    gauge_error = mp.norm(actual-transformed)
                    hamiltonian_error = mp.norm(jchi.T*omega+omega*jchi)
                    symmetry_error = mp.norm(k-k.T)
                    assert max(direct_error,gauge_error,hamiltonian_error,symmetry_error)<mp.mpf("1e-50"), (length,side,str(chi),str(direct_error),str(gauge_error),str(hamiltonian_error),str(symmetry_error))
                    assert max(spatial_error,temporal_error)<mp.mpf("1e-7"), (length,side,str(chi),str(spatial_error),str(temporal_error))
                    rows.append(dict(length=length,side=side,chi=float(chi),zero_bias_crossing=float(duration),
                        direct_exponential_error=float(direct_error),hamiltonian_form_error=float(hamiltonian_error),
                        symmetric_graph_error=float(symmetry_error),common_gauge_error=float(gauge_error),
                        gamma_finite_difference_error=float(spatial_error),lambda_finite_difference_error=float(temporal_error),
                        instantaneous_implicit_response=float(gamma/rate),position=float((p.T*y*p)[0])))
    value = dict(status="passed",completed_utc=study.now(),digits=90,rows=rows,
                 source_sha256=study.sha(Path(__file__)),proof_sha256=study.sha(ROOT/"supplement/asymptotic_time_proof.tex"),
                 scope="Internal independent implementation of finite-chain identities and local response derivatives. These checks do not establish uniform asymptotic estimates or replace external expert review.",
                 external_review_status="pending")
    study.write_json(OUT/"proof_algebra_audit.json",value)
    return {k:v for k,v in value.items() if k!="rows"} | dict(checks=len(rows))


def inference():
    lock = study.check_lock()
    summary = study.read_json(OUT/"summary.json")
    b = summary["systematic_budget"]
    r = summary["conditional_95pct_difference_error"]
    signal = summary["minimum_control_signal"]
    dc = summary["maximum_cdw_distance_to_W"]
    observed_threshold = .75
    value = dict(completed_utc=study.now(),protocol_lock_sha256=study.sha(OUT/"protocol_lock.json"),
       sampling_statement="Given n accepted Bernoulli samples from each preparation, with probability >=.95 each sample mean differs from its measured expectation by at most r/2, and their difference by at most r.",
       actual_data_inference="distance(block,W) >= max(0, abs(observed_contrast)-systematic_budget-statistical_radius-cdw_distance_bound).",
       planned_observed_contrast_lower=signal-b-r,
       proposed_future_positive_memory_reporting_threshold=observed_threshold,
       distance_lower_at_reporting_threshold=observed_threshold-b-r-dc,
       distance_lower_at_nominal_predicted_contrast=summary["nominal"]["signal"]-b-r-dc,
       before_data_95pct_lower_on_resulting_distance_bound=max(0,signal-2*(b+r)-dc),
       statistical_radius=r,systematic_budget=b,
       interval_scope="The planning margin .8116 is a lower bound on the observed contrast on the sampling event, not a .8116 pre-data confidence bound on the physical projector distance. Actual reporting uses the observed contrast.",
       grid_scope="Control validation samples a continuous box; no uniform interval certificate is claimed.",
       systematic_derivation=dict(false_accept="Each bounded occupation mean changes by at most f for contamination fraction f.",
                                 binary_readout="Each calibrated mean bias has magnitude at most b_read.",
                                 orbital_error="For rank-one projectors at distance rho, any 0<=P<=I expectation changes by at most rho."),
       false_acceptance_derivation="For true acceptance p and unit true-accept readout, rejected-run false acceptance beta must satisfy beta <= f*p/((1-f)*(1-p)). With true-accept readout efficiency a, replace p by a*p in the numerator.",
       accepted_sample_scope="Sample counts and attempted-run budgets refer to true accepted runs with perfect classification. Additional classification inefficiency and loading defects increase attempts; contamination bounds must be calibrated.")
    study.write_json(OUT/"measurement_inference.json",value)
    return value


def packet():
    files = ["main/main.tex","main/refs.bib","main/main.pdf",
             "supplement/supplement.tex","supplement/asymptotic_time_proof.tex","supplement/refs.bib","supplement/supplement.pdf",
             "scripts/audit_operational_memory.py","scripts/run_operational_memory.py",
             "data/prl_operational_memory/proof_algebra_audit.json","data/prl_operational_memory/measurement_inference.json",
             "data/prl_operational_memory/protocol_lock.json","data/prl_operational_memory/summary.json",
             "data/prl_operational_memory/independent_checks.json",
             "data/prl_asymptotic_time_exploration/summary.json",
             "research_doc/01_当前研究/实际时间响应证明审读说明.md"]
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/"supplement/figures").rglob("*.pdf")]
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/"main/figures").rglob("*.pdf")]
    files = sorted(set(files))
    manifest = [dict(path=relative,sha256=study.sha(ROOT/relative),bytes=(ROOT/relative).stat().st_size) for relative in files]
    study.write_json(OUT/"review_packet_manifest.json",dict(created_utc=study.now(),files=manifest,
                     review_status="External expert review pending; this file only certifies packet contents."))
    with zipfile.ZipFile(OUT/"proof_review_packet.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(ROOT/relative,relative)
        archive.write(OUT/"review_packet_manifest.json","review_packet_manifest.json")
    with zipfile.ZipFile(OUT/"proof_review_packet.zip") as archive:
        for row in manifest:
            assert hashlib.sha256(archive.read(row["path"])).hexdigest()==row["sha256"]
    receipt = dict(status="packet_verified",files=len(files),zip_sha256=study.sha(OUT/"proof_review_packet.zip"),
                   external_review_status="pending",created_utc=study.now())
    study.write_json(OUT/"review_packet_receipt.json",receipt)
    return receipt


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    study.check_lock()
    summary=study.read_json(OUT/"summary.json")
    with (OUT/"time_curves.csv").open(encoding="utf-8",newline="") as stream:
        rows=list(csv.DictReader(stream))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(1,3,figsize=(10.3,2.85),layout="constrained")
    for length,style in ((4,"-"),(8,"--")):
        values=[row for row in rows if int(row["length"])==length]
        times=[float(row["time"]) for row in values]
        for name,color in (("cdw","#007A87"),("right","#C44C43")):
            axes[0].plot(times,[float(row[name+"_distance_to_W"]) for row in values],style,color=color,
                         label=f"{'CDW' if name=='cdw' else 'Right'}, L={length}")
        axes[1].plot(times,[float(row["signal"]) for row in values],style,color="#424242",label=f"L={length}")
        axes[2].semilogy(times,[float(row["right_success"]) for row in values],style,color="#C44C43",label=f"Right, L={length}")
    signal=summary["nominal"]["signal"]
    budget=summary["systematic_budget"]+summary["conditional_95pct_difference_error"]
    axes[1].errorbar(.5,signal,yerr=[[budget],[min(1-signal,budget)]],fmt="o",color="#007A87",ms=4,capsize=3,label="L=4 bounded budget")
    axes[0].set(xlabel=r"Time $t$",ylabel=r"Distance to $P_W$",xlim=(0,4),ylim=(-.02,1.04))
    axes[1].set(xlabel=r"Time $t$",ylabel=r"Fixed-site contrast $-w_{A_1}$",xlim=(0,4),ylim=(-.02,1.04))
    axes[2].set(xlabel=r"Time $t$",ylabel=r"No-loss probability $P_0$",xlim=(0,1.25),ylim=(1e-7,1))
    for index,ax in enumerate(axes):
        ax.axvline(.5,color="#777777",ls=":",lw=.9)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=7.5,loc="best")
        ax.text(.5,-.28,f"({chr(97+index)})",transform=ax.transAxes,ha="center",va="center")
    target=ROOT/"figures/prl_operational_memory"
    target.mkdir(parents=True,exist_ok=True)
    for suffix in ("pdf","png"):
        fig.savefig(target/("fixed_site_memory."+suffix),dpi=220,facecolor="white")
    plt.close(fig)
    for project in ("supplement","supplment_zh"):
        destination=ROOT/project/"figures/prl_operational_memory"
        destination.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target/"fixed_site_memory.pdf",destination/"fixed_site_memory.pdf")
    study.write_json(OUT/"figure_provenance.json",dict(renderer="scripts/audit_operational_memory.py",renderer_sha256=study.sha(Path(__file__)),
                    computation_source_sha256=study.sha(study.SOURCE),curves_sha256=study.sha(OUT/"time_curves.csv"),
                    pdf_sha256=study.sha(target/"fixed_site_memory.pdf"),budget_interval="Intersected with the physical contrast range [-1,1]."))
    return dict(status="figure_written_with_bounded_budget")


def literature():
    cache=Path("C:/Users/lcx/AppData/Local/Temp/prl_editor_assessment_20261010")
    rows=[]
    for identifier,key in (("2504.08557","Soares2025"),("2601.16002","ZhangSunLi2026Mpemba")):
        receipt=study.read_json(cache/(identifier+".receipt.json"))
        assert study.sha(cache/(identifier+".html"))==receipt["source_sha256"]
        text=(cache/(identifier+".fulltext.txt")).read_text(encoding="utf-8")
        terms=("open boundaries","translation-invariant","Hilbert","spatial")
        excerpts=[]
        for line in text.splitlines():
            if any(term.lower() in line.lower() for term in terms):
                excerpts.append(line[:2400])
        rows.append(dict(citation_key=key,receipt=receipt,selected_excerpts=excerpts[:8],
                         citation_status="SciPost Phys. 19, 094 (2025), DOI 10.21468/SciPostPhys.19.4.094" if key=="Soares2025" else "arXiv preprint; no journal publication claimed",
                         scope="Cached full texts from targeted comparison; not a complete priority search."))
    value=dict(completed_utc=study.now(),references=rows)
    study.write_json(OUT/"literature_comparison.json",value)
    return dict(status="cached_sources_verified",references=len(rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("proof_algebra","inference","packet","figures","literature"))
    args=parser.parse_args()
    print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
