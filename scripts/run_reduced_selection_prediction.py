"""Freeze spatial hypotheses and signed low-rank forecasts before new truth."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import mpmath as mp
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

import run_spatial_selection_mechanism as spatial

previous, audit, p1 = spatial.previous, spatial.audit, spatial.p1
ROOT = spatial.ROOT
PRIOR = spatial.OUT
OUT = ROOT / "data/prl_p2_rank_reduction"
FIG = ROOT / "figures/prl_p2_rank_reduction"
SOURCE = Path(__file__)
SOURCE_SHA = audit.sha256(SOURCE)
DIGITS = (160, 200, 240, 280)
RANKS = (0, 1, 2, 4, 6, 8)
write_json, emit = spatial.write_json, spatial.emit


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def occupied(length, initial):
    if initial.startswith("block_cell"):
        start = 2 * int(initial.removeprefix("block_cell"))
        assert 0 <= start <= length // 2
        return list(range(start, start + length // 2))
    return spatial.occupied(length, initial)


def q0_matrix(length, initial):
    q = mp.matrix(length, length // 2)
    for column, site in enumerate(occupied(length, initial)):
        q[site, column] = 1
    return q


def cases():
    values = []
    def add(length, skin, times, initials, role):
        for duration in times:
            steps = round(duration / .05)
            values.append(dict(case_id=p1.case_key(length, 2., skin, steps), length=length, gamma=2., skin=skin,
                               physical_time=duration, steps=steps, dt=.05, t1=.5, t2=1., boundary="open",
                               initials=list(initials), role=role))
    for length in (72, 80):
        for skin in (0., .25):
            add(length, skin, (30., 35., 40., 45.), ("charge_density_wave", "left", "right"), "unseen_size")
    for skin in (0., .25):
        add(64, skin, (40., 45.), ("charge_density_wave", "left"), "persistence_extension")
    for skin in (-.25, 0., .25):
        add(40, skin, (8., 12., 16.), ("charge_density_wave", "block_cell2", "block_cell7"), "unseen_offset")
    return sorted(values, key=lambda r:(r["length"],r["skin"],r["physical_time"]))


def families():
    return sorted({(r["length"],r["gamma"],r["skin"]) for r in cases()} |
                  {(length,2.,skin) for length in (72,80) for skin in (-.25,0.,.25)})


def configuration():
    return dict(version=1,date="2026-10-04",cases=cases(),families=[list(f) for f in families()],
                ranks=list(RANKS),model_projector_target=1e-5,precision_digits=list(DIGITS),
                reference_goals=dict(projector=1e-10,entropy=1e-10,relative_scalar=1e-8),
                numerical_goals=dict(projector=1e-6,entropy=1e-6,orthogonality=1e-10),
                selection_tolerances=dict(projector=.05,absolute_entropy=.05),memory_tolerances=dict(projector=.1,absolute_entropy=.1),
                spatial_candidate="bulk Hankel moment finite matrix and asymptotic rate .5*log((sqrt(1+qg)+1)/(sqrt(1+qg)-1)); finite OBC reflections tested separately",
                reduced_candidate="rank-r SVD of static signed F; exp(-Lambda*t) on both factors; Woodbury rank-r graph; no full graph in forecast",
                remainder="min(sigma[r]*exp(-2*mu_min*t), weighted Frobenius residual); projector from column perturbation",
                rank_policy="first frozen rank with projector remainder <=1e-5, otherwise rank8 with honest uncertainty",
                interpretation="static full-spectrum preparation, low-rank finite-time evolution; not a universal scaling law or interval certificate",
                prior_plan_sha256=audit.sha256(PRIOR / "next_round_plan.json"))


def prepare():
    for name in ("code","static","predictions","cases","development","spatial","literature","failures"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    path = OUT / "config.json"
    if path.exists():
        assert read_json(path) == configuration()
    else:
        write_json(path,configuration())
    manifest = OUT / "input_manifest.csv"
    if not manifest.exists():
        inputs = {ROOT / r["path"] for r in audit.read_csv(PRIOR / "input_manifest.csv")}
        inputs.update(p for directory in (PRIOR,spatial.FIG) for p in directory.rglob("*") if p.is_file())
        inputs.update((spatial.SOURCE, ROOT / "scripts/analyze_spatial_selection_mechanism.py",
                       ROOT / "research_doc/04_历史研究/07_空间重叠与记忆预测.md"))
        write_csv("input_manifest.csv",[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=audit.sha256(p)) for p in sorted(inputs)])
    for row in audit.read_csv(manifest):
        assert audit.sha256(ROOT / row["path"]) == row["sha256"],row["path"]
    (OUT / "code" / (SOURCE_SHA.lower()+".py")).write_bytes(SOURCE.read_bytes())


def branch_root(gamma=mp.mpf(2)):
    x = (gamma**2 - mp.mpf("1.25")) / mp.mpf(".5")
    return (x-mp.sqrt(x*x-4))/2


def spatial_gram(length, skin, initial, nodes, dps, folded=True):
    """Positive Gram formula from the exact finite tridiagonal resolvent."""
    with mp.workdps(dps):
        m = length // 4
        q = branch_root()
        g = mp.mpf(str(skin))
        points, weights = mp.gauss_quadrature(nodes,"legendre")
        matrix = mp.matrix(m)
        sign = 1 if initial == "left" else -1
        for point, weight in zip(points,weights):
            angle = (point+1)*mp.pi/4
            sine, cosine = mp.sin(angle),mp.cos(angle)
            r = q*sine*sine
            s = mp.mpf("1.25")+mp.mpf(".5")*(r+1/r)
            delta = 1+2*r-r**(4*m+2)-2*r**(4*m+1) if folded else 1+2*r
            jacobian = q*2*sine*cosine*mp.pi/4
            if initial == "left":
                density = mp.sqrt(s-4)/(mp.pi*mp.mpf(".5")*delta)
                vector = [mp.exp(g*i)*(r**i-r**(2*m-i) if folded else r**i) for i in range(m)]
            else:
                density = mp.mpf(".5")*mp.sqrt(s-4)/(mp.pi*s*delta)
                vector = [mp.exp(-g*i)*(r**i*(1+2*r)-r**(2*m-i)-2*r**(2*m-i-1) if folded else r**i*(1+2*r)) for i in range(m)]
            scale = weight*jacobian*density*mp.exp(sign*g)
            for i in range(m):
                for j in range(i,m):
                    matrix[i,j] += scale*vector[i]*vector[j]
        for i in range(m):
            for j in range(i):
                matrix[i,j] = matrix[j,i]
        eigenvalues = mp.eigsy(matrix,eigvals_only=True)
        assert eigenvalues[0]>0
        return eigenvalues[0],matrix


def spatial_lock():
    path = OUT / "spatial_hypothesis_lock.json"
    if path.exists():
        hashes=read_json(OUT / "spatial_lock_hash.json")
        for name,sha in hashes.items():assert audit.sha256(OUT / name)==sha
        return
    assert not list((OUT / "static").glob("L72*.json")) and not list((OUT / "static").glob("L80*.json"))
    rows,raw = [],[]
    for length in (72,80):
        for skin in (-.25,0.,.25):
            for initial in ("left","right"):
                results=[]
                for nodes,dps in ((64,160),(96,200)):
                    results.append(spatial_gram(length,skin,initial,nodes,dps,False)[0])
                with mp.workdps(220):
                    change=spatial.relative(*results)
                    assert change<mp.mpf("1e-8"),(length,skin,initial,change)
                    qg=branch_root()*mp.exp((1 if initial=="left" else -1)*mp.mpf(str(skin)))
                    rate=mp.log((mp.sqrt(1+qg)+1)/(mp.sqrt(1+qg)-1))/2
                    rows.append(dict(length=length,skin=skin,initial=initial,bulk_cross_min=float(results[-1]),
                                     bulk_exponential_rate=float(rate),quadrature_relative_change=float(change),
                                     interpretation="bulk Hankel candidate, not yet an OBC asymptotic theorem"))
                    raw.append({**rows[-1],"bulk_cross_min_mp":str(results[-1])})
                emit(dict(stage="spatial_lock",length=length,skin=skin,initial=initial))
    write_csv("locked_spatial_predictions.csv",rows)
    write_json(OUT / "spatial/bulk_predictions_raw.json",raw)
    value=dict(locked_utc=datetime.now(timezone.utc).isoformat(),rows=len(rows),script_sha256=SOURCE_SHA,
               config_sha256=audit.sha256(OUT / "config.json"),method=configuration()["spatial_candidate"],
               policy="No retuning after new static overlap truth; failed bulk hypothesis stays in results.")
    write_json(path,value)
    write_json(OUT / "spatial_lock_hash.json",{name:audit.sha256(OUT / name) for name in
                                             ("spatial_hypothesis_lock.json","locked_spatial_predictions.csv","spatial/bulk_predictions_raw.json")})


def augment_static(static):
    length,gamma,skin=static["length"],static["gamma"],static["skin"]
    initials=sorted({name for row in cases() if (row["length"],row["gamma"],row["skin"])==(length,gamma,skin) for name in row["initials"]})
    additions=[name for name in initials if name not in static["states"]]
    if not additions:return static
    with mp.workdps(static["dps"]):
        plus,minus=spatial.decode(static["matrices"]["plus"]),spatial.decode(static["matrices"]["minus"])
        n=length//2
        x=mp.matrix([[plus[i,j] if j<n else minus[i,j-n] for j in range(length)] for i in range(length)])
        inverse=mp.inverse(x)
        yplus,yminus=inverse[:n,:],inverse[n:,:]
        input_gain=mp.qr(yplus.T,mode="skinny")[0]
        graph=spatial.decode(static["input_graph"])
        rho=previous.mp_opnorm(graph)
        for name in additions:
            q0=q0_matrix(length,name)
            f=spatial.right_solve(yminus*q0,yplus*q0)
            alpha=previous.mp_smin(input_gain.T*q0)
            cells=sorted({site//2 for site in occupied(length,name)})
            empty=[j for j in range(n) if j not in cells]
            block=mp.matrix([[graph[i,j] for j in cells] for i in empty])
            delta=previous.mp_smin(block)
            lower=min(1,delta)/((1+rho)*mp.sqrt(1+rho**2))
            upper=min(1,delta)
            assert lower<=alpha<=upper
            static["states"][name]=dict(alpha=str(alpha),F=p1.encode_matrix(f),F_norm=str(previous.mp_opnorm(f)),
                                         cross_block_min=str(delta),overlap_lower=str(lower),overlap_upper=str(upper),rho=str(rho),occupied_cells=cells)
    return static


def decompose(static, initials):
    with mp.workdps(static["dps"]):
        for name in initials:
            state=static["states"][name]
            f=spatial.decode(state["F"])
            u,s,vh=mp.svd(f)
            state["compression"]=dict(U=p1.encode_matrix(u),VH=p1.encode_matrix(vh),singular_values=[str(x) for x in s])
    return static


def static_job(family):
    key=spatial.identifier(*family)
    path=OUT / "static" / (key+".json")
    if path.exists():return dict(family=key,status="reused_static")
    assert (OUT / "spatial_hypothesis_lock.json").exists()
    started,tick=datetime.now(timezone.utc).isoformat(),time.perf_counter()
    initials=sorted({name for case in cases() if (case["length"],case["gamma"],case["skin"])==tuple(family) for name in case["initials"]})
    values=[]
    for dps in DIGITS:
        value=decompose(augment_static(spatial.static_reference(family,dps)),initials)
        values.append(value)
        if len(values)>1:
            change=spatial.static_change(values[-2],values[-1])
            with mp.workdps(300):
                additional=[]
                for name in initials:
                    a,b=values[-2]["states"][name],values[-1]["states"][name]
                    additional.append(float(spatial.relative(mp.mpf(a["alpha"]),mp.mpf(b["alpha"]))))
                    if "compression" in b:
                        additional.extend(float(spatial.relative(mp.mpf(sa),mp.mpf(sb))) for sa,sb in zip(a["compression"]["singular_values"][:9],b["compression"]["singular_values"][:9]))
            change["compression_relative"]=max(additional,default=0)
            if max(change["relative_scalar"],change["compression_relative"])<=1e-8 and max(change["matrix_absolute"],change["gain_entropy"])<=1e-10:break
    else:raise ArithmeticError("static compression did not converge")
    with gzip.open(path.with_suffix(".raw.json.gz"),"wt",encoding="utf-8") as stream:json.dump(values,stream)
    write_json(path,{**values[-1],"reference_convergence":change,"precision_digits":[v["dps"] for v in values],
                     "started_utc":started,"seconds":time.perf_counter()-tick,"script_sha256":SOURCE_SHA,
                     "config_sha256":audit.sha256(OUT / "config.json"),"spatial_lock_sha256":audit.sha256(OUT / "spatial_hypothesis_lock.json")})
    return dict(family=key,status="static_converged",dps=dps,seconds=time.perf_counter()-tick)


def compression_context(static,initial):
    matrices={k:spatial.decode(v) for k,v in static["matrices"].items()}
    f=spatial.decode(static["states"][initial]["F"])
    decomposition=static["states"][initial]["compression"]
    u,vh=spatial.decode(decomposition["U"]),spatial.decode(decomposition["VH"])
    sig=[mp.mpf(x) for x in decomposition["singular_values"]]
    rinv=mp.inverse(matrices["R"])
    mu=[mp.mpf(x) for x in static["mu"]]
    residuals={}
    for rank in RANKS:
        residuals[rank]=f-u[:,:rank]*mp.diag(sig[:rank])*vh[:rank,:] if rank else f
    return matrices,u,vh,sig,rinv,mu,residuals


def reduced_at_rank(ctx,duration,rank):
    matrices,u,vh,sig,rinv,mu,residuals=ctx
    n=len(mu)
    decay=[mp.exp(-x*duration) for x in mu]
    weighted=mp.matrix([[residuals[rank][i,j]*decay[i]*decay[j] for j in range(n)] for i in range(n)])
    residual=min(p1.mp_norm(weighted),sig[rank]*mp.exp(-2*mu[0]*duration))
    if rank:
        aa=mp.matrix([[decay[i]*u[i,j]*mp.sqrt(sig[j]) for j in range(rank)] for i in range(n)])
        bt=mp.matrix([[mp.sqrt(sig[i])*vh[i,j]*decay[j] for j in range(n)] for i in range(rank)])
        v=bt*rinv
        c=matrices["Ccross"]*aa
        h=mp.eye(rank)+v*c
        hinv=mp.inverse(h)
        left=matrices["Bperp"]*aa*hinv
        lu,lr=mp.qr(left,mode="skinny")
        rv,rr=mp.qr(v.T,mode="skinny")
        ku,ks,kvh=mp.svd(lr*rr.T)
        uu,vv=lu*ku,rv*kvh.T
        cosine=mp.diag([1/mp.sqrt(1+x*x) for x in ks])
        sine=mp.diag([x/mp.sqrt(1+x*x) for x in ks])
        q=matrices["G"]+matrices["G"]*vv*(cosine-mp.eye(rank))*vv.T+matrices["Gperp"]*uu*sine*vv.T
        kappa=ks[0]
        invtop_bound=1+previous.mp_opnorm(c)*previous.mp_opnorm(hinv)*previous.mp_opnorm(v)
        small_condition=previous.mp_opnorm(h)*previous.mp_opnorm(hinv)
    else:
        q=matrices["G"]
        kappa=mp.mpf(0)
        invtop_bound,small_condition=mp.mpf(1),mp.mpf(1)
    eps=min(mp.mpf(1),residual*previous.mp_opnorm(matrices["minus"])*previous.mp_opnorm(rinv)*invtop_bound)
    return dict(rank=rank,kappa=kappa,distance_to_G=kappa/mp.sqrt(1+kappa*kappa),
                projector_remainder=eps,Ft_remainder=residual,small_system_condition=small_condition),q


def entropy_budget(distance,n):
    x=min(mp.mpf(".5"),distance)
    return n*(-x*mp.log(x)-(1-x)*mp.log(1-x)) if x else mp.mpf(0)


def reduced_prediction(static,initial,duration,ctx=None):
    ctx=ctx or compression_context(static,initial)
    attempts=[]
    for rank in RANKS:
        result,q=reduced_at_rank(ctx,duration,rank)
        attempts.append({k:float(v) for k,v in result.items()})
        if result["projector_remainder"]<=mp.mpf(".00001"):break
    entropy,_=previous.correlation_entropy(q)
    z=mp.mpf(static["q"])*mp.exp(-2*min(ctx[5])*duration)
    p=mp.mpf(static["p"])
    ew=min(mp.mpf(1),z/(p-z)) if z<p else mp.mpf(1)
    eps=result["projector_remainder"]
    dg=result["distance_to_G"]
    error=abs(entropy-mp.mpf(static["gain_entropy"]))
    budget=entropy_budget(eps,static["length"]//2)+entropy_budget(ew,static["length"]//2)
    values={**result,"entropy":entropy,"output_to_G_bound":ew,
            "distance_lower":max(0,dg-eps-ew),"distance_upper":min(1,dg+eps+ew),
            "entropy_error_lower":max(0,error-budget),"entropy_error_upper":error+budget,"entropy_budget":budget}
    return values,q,attempts


def development():
    jobs=((40,0.,"right",15),(40,.25,"right",25),(48,.25,"left",22.5),
          (48,.25,"left",25),(64,0.,"left",30),(64,0.,"left",35),(40,0.,"charge_density_wave",5))
    rows=[]
    for length,skin,initial,duration in jobs:
        source=PRIOR / "static" / (spatial.identifier(length,2.,skin)+".raw.json.gz")
        with gzip.open(source,"rt",encoding="utf-8") as stream:values=json.load(stream)
        static=decompose(values[-1],[initial])
        with mp.workdps(static["dps"]):
            ctx=compression_context(static,initial)
            exact,qe=spatial.graph_prediction(static,initial,mp.mpf(str(duration)),False)
            for rank in RANKS:
                reduced,qr=reduced_at_rank(ctx,mp.mpf(str(duration)),rank)
                distance=p1.mp_distance(qe,qr)
                assert distance<=reduced["projector_remainder"]+mp.mpf("1e-50")
                rows.append(dict(length=length,skin=skin,initial=initial,physical_time=duration,
                                 **{k:float(v) for k,v in reduced.items()},measured_projector_error=float(distance),
                                 exact_distance_to_G=float(exact["distance_to_G"]),stage="development known previous inputs"))
        emit(dict(stage="development",length=length,skin=skin,initial=initial,time=duration))
    write_csv("development/rank_calibration.csv",rows)
    with mp.workdps(160):
        checks=[]
        for length,skin,initial in ((16,0.,"left"),(24,.25,"right"),(40,0.,"left")):
            static=read_json(PRIOR / "static" / (spatial.identifier(length,2.,skin)+".json"))
            exact=mp.mpf(static["states"][initial]["cross_block_min"])
            result,_=spatial_gram(length,skin,initial,64,160,True)
            error=spatial.relative(exact,result)
            assert error<mp.mpf("1e-8"),(length,initial,error)
            checks.append(dict(length=length,skin=skin,initial=initial,folded_gram_min=float(result),
                               prior_cross_min=float(exact),relative_error=float(error)))
        write_csv("development/folded_gram_calibration.csv",checks)
    write_json(OUT / "development/calibration.json",dict(status="passed",ranks=list(RANKS),model_projector_target=1e-5,
                                                        prior_only=True,script_sha256=SOURCE_SHA,rank_tests=len(rows)))


def prediction_job(family):
    key=spatial.identifier(*family)
    path=OUT / "predictions" / (key+".json")
    if path.exists():return dict(family=key,status="reused_predictions")
    with gzip.open(OUT / "static" / (key+".raw.json.gz"),"rt",encoding="utf-8") as stream:static_values=json.load(stream)[-2:]
    planned=[row for row in cases() if (row["length"],row["gamma"],row["skin"])==tuple(family)]
    rows,arrays,raw,memories=[],{},[],[]
    contexts=[]
    for static in static_values:
        with mp.workdps(static["dps"]):contexts.append({name:compression_context(static,name) for name in {name for case in planned for name in case["initials"]}})
    for case in planned:
        local={}
        for initial in case["initials"]:
            results=[]
            for static,contexts_at_precision in zip(static_values,contexts):
                with mp.workdps(static["dps"]):results.append(reduced_prediction(static,initial,mp.mpf(str(case["physical_time"])),contexts_at_precision[initial]))
            low,high=results[0][0],results[1][0]
            with mp.workdps(300):
                change=max(abs(low[k]-high[k]) for k in ("distance_to_G","entropy","projector_remainder","distance_lower","distance_upper","entropy_error_upper"))
                assert change<mp.mpf("1e-10") and low["rank"]==high["rank"]
            values,q,attempts=results[-1]
            numeric={k:float(v) for k,v in values.items()}
            row={**case,"initial":initial,**numeric,"prediction_precision_change":float(change),"prediction_dps":static_values[-1]["dps"],
                 "rank_target_passed":numeric["projector_remainder"]<=1e-5,
                 "guaranteed_selected":numeric["distance_upper"]<=.05 and numeric["entropy_error_upper"]<=.05,
                 "guaranteed_unselected":numeric["distance_lower"]>.05 or numeric["entropy_error_lower"]>.05}
            rows.append(row);local[initial]=row
            arrays[case["case_id"]+"_"+initial]=audit.mp_to_numpy(q)
            with mp.workdps(static_values[-1]["dps"]):raw.append(dict(case_id=case["case_id"],initial=initial,Q=p1.encode_matrix(q),values={k:str(v) for k,v in values.items()},attempts=attempts))
        first=local["charge_density_wave"]
        for initial in case["initials"][1:]:
            second=local[initial]
            lower=max(0,abs(first["distance_to_G"]-second["distance_to_G"])-first["projector_remainder"]-second["projector_remainder"])
            memories.append({**case,"initial_a":"charge_density_wave","initial_b":initial,"memory_lower":lower,"guaranteed_memory":lower>.1})
    write_json(path,dict(family=family,rows=rows,memory_predictions=memories,script_sha256=SOURCE_SHA,
                        static_sha256=audit.sha256(OUT / "static" / (key+".json")),config_sha256=audit.sha256(OUT / "config.json")))
    np.savez_compressed(path.with_suffix(".npz"),**arrays)
    with gzip.open(path.with_suffix(".raw.json.gz"),"wt",encoding="utf-8") as stream:json.dump(raw,stream)
    return dict(family=key,status="reduced_predictions",states=len(rows))


def check_lock():
    for filename,sha in read_json(OUT / "prediction_lock_hash.json").items():assert audit.sha256(OUT / filename)==sha


def lock():
    if (OUT / "prediction_lock.json").exists():check_lock();return
    assert not list((OUT / "cases").glob("*.json"))
    values=[read_json(path) for path in sorted((OUT / "predictions").glob("*.json"))]
    assert len(values)==len(families())
    rows=[row for value in values for row in value["rows"]]
    memories=[row for value in values for row in value["memory_predictions"]]
    assert len(rows)==83 and len(memories)==54
    write_csv("locked_predictions.csv",rows)
    write_csv("locked_memory_predictions.csv",memories)
    inputs=[p for name in ("static","predictions","development") for p in (OUT / name).rglob("*") if p.is_file()]
    value=dict(locked_utc=datetime.now(timezone.utc).isoformat(),states=len(rows),pairs=len(memories),
               config_sha256=audit.sha256(OUT / "config.json"),script_sha256=SOURCE_SHA,
               inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=audit.sha256(p)) for p in sorted(inputs)],
               ranks=list(RANKS),model_projector_target=1e-5,policy="No rank or tolerance adjustment after new truth")
    write_json(OUT / "prediction_lock.json",value)
    write_json(OUT / "prediction_lock_hash.json",{name:audit.sha256(OUT / name) for name in ("prediction_lock.json","locked_predictions.csv","locked_memory_predictions.csv")})
    emit(dict(stage="lock",states=len(rows),pairs=len(memories),utc=value["locked_utc"]))


def double_reference(case):
    from mietf_skin.ssh import ssh_no_click_hamiltonian
    n=case["length"]//2
    h=ssh_no_click_hamiltonian(n,.5,1.,2.,case["skin"],"open")
    matrix=audit.normalize(linalg.expm(-1j*h*case["physical_time"]))
    w,s,_=linalg.svd(matrix,lapack_driver="gesvd")
    arrays=dict(W_double=w[:,:n],s_double=s)
    propagator=linalg.expm(-1j*h*.05)
    for initial in case["initials"]:
        q=np.eye(case["length"])[:,occupied(case["length"],initial)].astype(complex)
        for _ in range(case["steps"]):q=linalg.qr(propagator@q,mode="economic")[0]
        arrays["Q_"+initial+"_double"]=q
    return arrays


def reference(case,dps):
    with mp.workdps(dps):
        matrix=mp.expm(previous.generator(case)*mp.mpf(str(case["physical_time"])))
        scale=p1.mp_norm(matrix);matrix/=scale
        w,s,vh=mp.svd(matrix)
        n=case["length"]//2
        top,bottom=w[:,:n],w[:,n:]
        wentropy,_=previous.correlation_entropy(top)
        phase=np.tile([1.,1.j],n)
        arrays=dict(M=phase[:,None]*audit.mp_to_numpy(matrix)*phase.conj()[None,:],W=phase[:,None]*audit.mp_to_numpy(top),s=np.array([float(x) for x in s]))
        raw=dict(dps=dps,M=p1.encode_matrix(matrix),W=p1.encode_matrix(top),singular_values=[str(x) for x in s],
                 log_fro_scale=str(mp.log(scale)),W_entropy=str(wentropy),states={})
        rows=[]
        for initial in case["initials"]:
            q0=q0_matrix(case["length"],initial)
            q=mp.qr(matrix*q0,mode="skinny")[0]
            entropy,_=previous.correlation_entropy(q)
            distance=previous.mp_opnorm(bottom.T*q)
            alpha=previous.mp_smin(vh[:n,:]*q0)
            ratio=s[n]/s[n-1]
            values=dict(entropy=entropy,distance_to_W=distance,alpha=alpha,ratio=ratio)
            rows.append({**case,"initial":initial,"reference_dps":dps,**{k:float(v) for k,v in values.items()},
                         "W_entropy":float(wentropy),"entropy_error":float(abs(entropy-wentropy)),
                         "controlled":distance<=mp.mpf(".05") and abs(entropy-wentropy)<=mp.mpf(".05")})
            arrays["Q_"+initial]=phase[:,None]*audit.mp_to_numpy(q)
            raw["states"][initial]={**{k:str(v) for k,v in values.items()},"Q":p1.encode_matrix(q)}
        return rows,arrays,raw


def stable_static_q(static,initial,duration,dps):
    tick=time.perf_counter()
    with mp.workdps(dps):
        plus,minus=spatial.decode(static["matrices"]["plus"]),spatial.decode(static["matrices"]["minus"])
        f=spatial.decode(static["states"][initial]["F"])
        mu=[mp.mpf(x) for x in static["mu"]]
        ft=mp.matrix([[f[i,j]*mp.exp(-(mu[i]+mu[j])*duration) for j in range(f.cols)] for i in range(f.rows)])
        q=mp.qr(plus+minus*ft,mode="skinny")[0]
        entropy,_=previous.correlation_entropy(q)
        return q,entropy,time.perf_counter()-tick


def case_job(case):
    check_lock()
    path=OUT / "cases" / (case["case_id"]+".json")
    if path.exists():return dict(case_id=case["case_id"],status="reused_case")
    started,tick=datetime.now(timezone.utc).isoformat(),time.perf_counter()
    double=double_reference(case)
    references=[]
    for dps in DIGITS:
        references.append(reference(case,dps))
        change=spatial.reference_change(references[-2],references[-1]) if len(references)>1 else None
        with path.with_suffix(".events.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(dict(dps=dps,seconds=time.perf_counter()-tick,convergence=change))+"\n")
        if change is not None and previous.accepted_precision(change):break
    else:raise ArithmeticError("independent exponential reference failed to converge")
    rows,arrays,raw=references[-1]
    static=read_json(OUT / "static" / (spatial.identifier(case["length"],2.,case["skin"])+".json"))
    phase=np.tile([1.,1.j],case["length"]//2)
    stable_raw={}
    for row in rows:
        initial=row["initial"]
        qdouble=double["Q_"+initial+"_double"]
        row.update(double_projector_error=audit.projector_error(arrays["Q_"+initial],qdouble),
                   double_entropy_error=abs(row["entropy"]-audit.entropy(qdouble)),double_orthogonality=audit.opnorm(qdouble.conj().T@qdouble-np.eye(qdouble.shape[1])))
        row["double_status"]="reference_validated" if max(row["double_projector_error"],row["double_entropy_error"])<=1e-6 else "reference_failed"
        stable_raw[initial]=[]
        for digits in (80,120):
            qstable,entropy,seconds=stable_static_q(static,initial,mp.mpf(str(case["physical_time"])),digits)
            qarray=phase[:,None]*audit.mp_to_numpy(qstable)
            row[f"static{digits}_projector_error"]=audit.projector_error(arrays["Q_"+initial],qarray)
            with mp.workdps(300):row[f"static{digits}_entropy_error"]=float(abs(entropy-mp.mpf(raw["states"][initial]["entropy"])))
            row[f"static{digits}_seconds"]=seconds
            arrays[f"Q_{initial}_static{digits}"]=qarray
            with mp.workdps(digits):stable_raw[initial].append(dict(dps=digits,Q=p1.encode_matrix(qstable),entropy=str(entropy),seconds=seconds))
        row["stable_status"]="reference_validated" if max(row["static80_projector_error"],row["static80_entropy_error"],row["static120_projector_error"],row["static120_entropy_error"])<=1e-6 else "reference_failed"
    np.savez_compressed(path.with_suffix(".npz"),**double,**arrays)
    with gzip.open(path.with_suffix(".raw.json.gz"),"wt",encoding="utf-8") as stream:json.dump([v[2] for v in references],stream)
    with gzip.open(path.with_suffix(".stable.raw.json.gz"),"wt",encoding="utf-8") as stream:json.dump(stable_raw,stream)
    value=dict(case=case,rows=rows,precision_digits=[v[2]["dps"] for v in references],reference_convergence=change,
               started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-tick,
               script_sha256=SOURCE_SHA,config_sha256=audit.sha256(OUT / "config.json"),prediction_lock_sha256=audit.sha256(OUT / "prediction_lock.json"))
    write_json(path,value)
    return dict(case_id=case["case_id"],status="reference_converged",dps=dps,seconds=value["seconds"],
                double_failed=sum(r["double_status"]!="reference_validated" for r in rows),stable_failed=sum(r["stable_status"]!="reference_validated" for r in rows))


def worker_init(source_directory):
    sys.path.insert(0,source_directory)
    threadpool_limits(limits=1)


def parallel(function,jobs,workers,source_directory):
    failures=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(source_directory,)) as pool:
        pending={pool.submit(function,job):job for job in jobs}
        for future in as_completed(pending):
            job=pending[future]
            try:emit(future.result())
            except Exception as error:
                key=job["case_id"] if isinstance(job,dict) else spatial.identifier(*job)
                result=dict(function=function.__name__,job=key,error_type=type(error).__name__,error=str(error),script_sha256=SOURCE_SHA,utc=datetime.now(timezone.utc).isoformat())
                write_json(OUT / "failures" / (function.__name__+"_"+key+".json"),result)
                failures.append(result);emit(result)
    assert not failures,f"{len(failures)} preserved failed jobs"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("development","spatial_lock","static","lock","validate"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    assert 1<=args.workers<=8
    prepare()
    tick=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="prl_rank_baseline_") as directory:
        with zipfile.ZipFile(ROOT / "data/prl_p0_precision/baseline_snapshot.zip") as archive:
            for name in archive.namelist():
                if name.startswith("src/"):archive.extract(name,directory)
        source_directory=str(Path(directory) / "src")
        sys.path.insert(0,source_directory)
        with threadpool_limits(limits=1):
            if args.stage=="development":development()
            elif args.stage=="spatial_lock":spatial_lock()
            elif args.stage=="static":
                assert (OUT / "development/calibration.json").exists()
                parallel(static_job,families(),args.workers,source_directory)
            elif args.stage=="lock":
                parallel(prediction_job,families(),args.workers,source_directory);lock()
            else:
                check_lock();parallel(case_job,cases(),args.workers,source_directory)
    info=audit.environment(args.stage)
    info.update(stage=args.stage,script_sha256=SOURCE_SHA,config_sha256=audit.sha256(OUT / "config.json"),
                workers=args.workers,elapsed_seconds=time.perf_counter()-tick,scope="local Windows, no HPC execution")
    write_json(OUT / ("environment_"+args.stage+".json"),info)


if __name__=="__main__":main()
