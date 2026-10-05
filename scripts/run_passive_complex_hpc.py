"""Independent complex local passive model, frozen predictions and references."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mpmath as mp
import numpy as np
from scipy import linalg

import run_selection_front_hpc as io

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prl_p4_passive_complex"
CONFIG, SOURCE = OUT / "config.json", Path(__file__)
read_json, write_json, write_csv = io.read_json, io.write_json, io.write_csv
read_gzip, write_gzip, sha256, emit = io.read_gzip, io.write_gzip, io.audit.sha256, io.emit
DIGITS = (120, 160, 200, 240)
RANKS = (0, 1, 2, 4, 6, 8, 12, 16, 24)


def config():
    return read_json(CONFIG)


def encode(matrix):
    return [[[str(mp.re(matrix[i,j])), str(mp.im(matrix[i,j]))] for j in range(matrix.cols)] for i in range(matrix.rows)]


def decode(value):
    return mp.matrix([[mp.mpc(*item) for item in row] for row in value])


def norm(matrix):
    return mp.sqrt(mp.fsum(abs(x)**2 for x in matrix))


def opnorm(matrix):
    return mp.svd(matrix, compute_uv=False)[0]


def smin(matrix):
    values = mp.svd(matrix, compute_uv=False)
    return values[min(matrix.rows,matrix.cols)-1]


def distance(qa,qb):
    return opnorm(qb-qa*(qa.H*qb))


def entropy(q):
    c = q[:q.rows//2,:]*q[:q.rows//2,:].H
    values = mp.eighe((c+c.H)/2,eigvals_only=True)
    allowance = mp.mpf(10)**(-mp.mp.dps//2)
    assert all(-allowance <= x <= 1+allowance for x in values)
    result = mp.mpf(0)
    for x in values:
        x = min(mp.mpf(1),max(mp.mpf(0),x))
        if x:
            result -= x*mp.log(x)
        if x < 1:
            result -= (1-x)*mp.log(1-x)
    return result


def entropy_budget(eps,m):
    x=min(mp.mpf(".5"),eps)
    return m*(-x*mp.log(x)-(1-x)*mp.log(1-x)) if x else mp.mpf(0)


def q0(length,name):
    occupied = io.base.occupied(length,name)
    value = mp.matrix(length,length//2)
    for column,site in enumerate(occupied):
        value[site,column] = 1
    return value


def hamiltonian(family):
    length, cells = family["length"], family["length"]//2
    g, gamma = mp.mpf(str(family["skin"])), mp.mpf(str(family["gamma"]))
    t1, t2, kappa = mp.mpf(".5"), mp.mpf(1), mp.mpf(str(family["same_sublattice"]))
    phi = mp.pi/3
    h=mp.matrix(length)
    for j in range(cells):
        onsite = mp.mpf(".15")*mp.cos(2*mp.pi*j/3)
        h[2*j,2*j], h[2*j+1,2*j+1] = onsite+mp.j*gamma, onsite-mp.j*gamma
        h[2*j,2*j+1] = h[2*j+1,2*j] = t1
        if j+1 < cells:
            h[2*j+2,2*j+1], h[2*j+1,2*j+2] = t2*mp.exp(g), t2*mp.exp(-g)
            for sublattice,phase in ((0,phi),(1,-phi/2)):
                a,b=2*j+sublattice,2*(j+1)+sublattice
                h[b,a], h[a,b] = kappa*mp.exp(g+mp.j*phase), kappa*mp.exp(-g-mp.j*phase)
    return h


def passive_embedding(h):
    imag=(h-h.H)/(2*mp.j)
    c=max(mp.re(imag[i,i])+mp.fsum(abs(imag[i,j]) for j in range(h.cols) if j!=i) for i in range(h.rows))
    loss=c*mp.eye(h.rows)-imag
    jumps=[]
    used=[mp.mpf(0)]*h.rows
    for i in range(h.rows):
        for j in range(i+1,h.cols):
            z=loss[i,j]
            if not z:
                continue
            magnitude=abs(z)
            row=mp.matrix(1,h.rows)
            row[0,i], row[0,j]=mp.sqrt(2*magnitude),mp.sqrt(2*magnitude)*z/magnitude
            jumps.append(row)
            used[i]+=magnitude
            used[j]+=magnitude
    for i in range(h.rows):
        residual=mp.re(loss[i,i])-used[i]
        assert residual >= -mp.mpf(10)**(-mp.mp.dps//2)
        if residual > 0:
            row=mp.matrix(1,h.rows)
            row[0,i]=mp.sqrt(2*residual)
            jumps.append(row)
    reconstructed=mp.matrix(h.rows)
    for row in jumps:
        reconstructed+=row.H*row/2
    return c,loss,jumps,norm(loss-reconstructed)


def initialize():
    for directory in ("static","predictions","cases","audits","failures","development","hpc","code"):
        (OUT/directory).mkdir(parents=True,exist_ok=True)
    if CONFIG.exists():
        check_source()
        return dict(status="reused")
    families=[dict(key=f"L{length}_g{str(g).replace('-','m').replace('.','p')}",length=length,
        skin=g,gamma=2.,same_sublattice=.2) for length in (32,48,64) for g in (0.,.2,-.2)]
    cases=[]
    for family in families:
        names=io.initials(family["length"])
        for fraction in (".25",".45",".60",".80"):
            duration=mp.mpf(fraction)*family["length"]
            cases.append(dict(**family, family_key=family["key"], case_id=family["key"]+"_tau"+fraction.replace('.','p'),
                physical_time=str(duration), time_fraction=fraction, initials=names,pairs=io.memory_pairs(family["length"]),
                target_initials=names, role="independent_complex_model", digits=list(DIGITS)))
    value=dict(version=1,families=families,cases=cases,ranks=list(RANKS),batch_size=4,workers=4,
        selection_tolerances=dict(projector=.05,absolute_entropy=.05),memory_tolerances=dict(projector=.1,absolute_entropy=.1),
        model_projector_target=1e-5,precision_goals=dict(projector=1e-10,entropy=1e-10,relative_scalar=1e-8),
        model="OBC SSH plus phase-carrying same-sublattice hopping and period-three real onsite potential",
        physical_embedding="H_phys=H-i*c*I; c chosen by local diagonal dominance; explicit one/two-site linear loss operators",
        conjugation_policy="Hermitian adjoints in projections, Gram matrices, SVD factors and entropy; transpose only in algebraic right solves",
        scope="conditional fixed-particle no-click states; no typical Born ensemble or thermodynamic transition")
    write_json(CONFIG,value)
    write_json(OUT/"kernel_manifest.json",[dict(path=SOURCE.relative_to(ROOT).as_posix(),sha256=sha256(SOURCE)),
        dict(path=io.SOURCE.relative_to(ROOT).as_posix(),sha256=sha256(io.SOURCE))])
    write_json(OUT/"bootstrap_lock.json",dict(locked_utc=io.now(),source_sha256=sha256(SOURCE),config_sha256=sha256(CONFIG),
        previous_round_lock_sha256=sha256(ROOT/"data/prl_p3_rank2_front/prediction_lock.json")))
    return dict(status="initialized",families=len(families),conditions=len(cases),states=sum(len(c["initials"]) for c in cases))


def check_source():
    value=read_json(OUT/"bootstrap_lock.json")
    assert sha256(SOURCE)==value["source_sha256"] and sha256(CONFIG)==value["config_sha256"]
    for row in read_json(OUT/"kernel_manifest.json"):
        assert sha256(ROOT/row["path"])==row["sha256"]


def static_reference(family,digits):
    with mp.workdps(digits):
        h=hamiltonian(family)
        c,loss,jumps,residual=passive_embedding(h)
        assert residual < mp.mpf("1e-70")
        eigenvalues,x=mp.eig(-mp.j*h)
        order=sorted(range(h.rows),key=lambda j:(float(mp.re(eigenvalues[j])),float(mp.im(eigenvalues[j]))),reverse=True)
        x=mp.matrix([[x[i,j] for j in order] for i in range(x.rows)])
        eigenvalues=[eigenvalues[j] for j in order]
        for column in range(x.cols):
            largest=max(range(x.rows),key=lambda i:abs(x[i,column]))
            phase=x[largest,column]/abs(x[largest,column])
            scale=norm(x[:,column])
            for row in range(x.rows):
                x[row,column]/=phase*scale
        y=mp.inverse(x)
        n=family["length"]//2
        plus,minus=x[:,:n],x[:,n:]
        yp,ym=y[:n,:],y[n:,:]
        gap=min(mp.re(z) for z in eigenvalues[:n])-max(mp.re(z) for z in eigenvalues[n:])
        assert min(mp.re(z) for z in eigenvalues[:n]) > 0 and max(mp.re(z) for z in eigenvalues[n:]) < 0
        complete,_=mp.qr(plus)
        g,gp=complete[:,:n],complete[:,n:]
        r,cross,transverse=g.H*plus,g.H*minus,gp.H*minus
        input_gain=mp.qr(yp.H,mode="skinny")[0]
        states={}
        for name in io.initials(family["length"]):
            aa,bb=yp*q0(family["length"],name),ym*q0(family["length"],name)
            f=io.cached_right_solve(bb,aa)
            u,s,vh=mp.svd(f)
            states[name]=dict(F=encode(f),U=encode(u),VH=encode(vh),singular_values=[str(z) for z in s],
                alpha=str(smin(input_gain.H*q0(family["length"],name))))
        eigen_residual=norm(-mp.j*h*x-x*mp.diag(eigenvalues))/norm(x)
        inverse_residual=norm(x*y-mp.eye(h.rows))
        assert max(eigen_residual,inverse_residual) < mp.mpf("1e-65")
        return dict(family=family,dps=digits,lambda_plus=[[str(mp.re(z)),str(mp.im(z))] for z in eigenvalues[:n]],
            lambda_minus=[[str(mp.re(z)),str(mp.im(z))] for z in eigenvalues[n:]],states=states,gap=str(gap),
            p=str(smin(plus)*smin(yp)),q=str(opnorm(minus)*opnorm(ym)),gain_entropy=str(entropy(g)),loss_shift=str(c),
            passive_reconstruction_residual=str(residual),minimum_loss_eigenvalue=str(mp.eighe(loss,eigvals_only=True)[0]),
            jump_supports=[[j for j in range(row.cols) if row[0,j]!=0] for row in jumps],
            eigen_residual=str(eigen_residual),inverse_residual=str(inverse_residual),
            matrices={name:encode(value) for name,value in dict(plus=plus,minus=minus,G=g,Gperp=gp,R=r,Ccross=cross,Bperp=transverse).items()})


def static_job(family):
    target=OUT/"static"/(family["key"]+".json")
    if target.exists():
        return dict(status="reused",family=family["key"])
    tick,started,values,attempts=time.perf_counter(),io.now(),[],[]
    for digits in DIGITS:
        local=time.perf_counter()
        values.append(static_reference(family,digits))
        change=None
        if len(values)>1:
            with mp.workdps(280):
                a,b=values[-2:]
                relative=max(abs(mp.mpf(a[k])-mp.mpf(b[k]))/abs(mp.mpf(b[k])) for k in ("p","q","gap"))
                for name in b["states"]:
                    fa,fb=decode(a["states"][name]["F"]),decode(b["states"][name]["F"])
                    relative=max(relative,norm(fa-fb)/norm(fb),
                        abs(mp.mpf(a["states"][name]["alpha"])-mp.mpf(b["states"][name]["alpha"]))/mp.mpf(b["states"][name]["alpha"]))
                projector=distance(decode(a["matrices"]["G"]),decode(b["matrices"]["G"]))
                change=dict(relative_scalar=float(relative),projector=float(projector),entropy=float(abs(mp.mpf(a["gain_entropy"])-mp.mpf(b["gain_entropy"]))))
        attempts.append(dict(dps=digits,seconds=time.perf_counter()-local,convergence=change))
        write_json(target.with_suffix(".progress.json"),dict(started_utc=started,attempts=attempts))
        if change and change["relative_scalar"]<=1e-8 and max(change["projector"],change["entropy"])<=1e-10:
            break
    else:
        write_gzip(target.with_suffix(".failed.raw.json.gz"),values)
        raise ArithmeticError("complex static geometry failed cross-precision convergence")
    write_gzip(target.with_suffix(".raw.json.gz"),values)
    write_json(target,dict(**values[-1],attempts=attempts,reference_convergence=change,seconds=time.perf_counter()-tick,
        started_utc=started,completed_utc=io.now(),source_sha256=sha256(SOURCE)))
    return dict(status="static_converged",family=family["key"],dps=digits,seconds=time.perf_counter()-tick)


def contexts(static):
    matrices={k:decode(v) for k,v in static["matrices"].items()}
    shared=dict(matrices=matrices,rinv=mp.inverse(matrices["R"]),static=static,
        lp=[mp.mpc(*z) for z in static["lambda_plus"]],lm=[mp.mpc(*z) for z in static["lambda_minus"]])
    return {name:dict(**shared,f=decode(v["F"]),u=decode(v["U"]),vh=decode(v["VH"]),sig=[mp.mpf(x) for x in v["singular_values"]]) for name,v in static["states"].items()}


def core_smin(g,y,v):
    rank,n=v.rows,v.cols
    b=g.H*y
    span=mp.matrix(n,2*rank)
    span[:,:rank],span[:,rank:]=b,v.H
    j,s,vh=mp.svd(span)
    triangular=mp.diag(s)*vh
    kernel=mp.matrix(2*rank)
    kernel[:rank,rank:]=mp.eye(rank)
    kernel[rank:,:rank]=mp.eye(rank)
    kernel[rank:,rank:]=y.H*y
    small=mp.eye(len(s))+triangular*kernel*triangular.H
    smallest=mp.eighe((small+small.H)/2,eigvals_only=True)[0]
    smallest=min(mp.mpf(1),smallest) if len(s)<n else smallest
    assert smallest > 0
    return mp.sqrt(smallest)


def time_context(ctx,duration):
    n=len(ctx["lp"])
    maximum=min(RANKS[-1],n)
    dm=[mp.exp(z*duration) for z in ctx["lm"]]
    dp=[mp.exp(-z*duration) for z in ctx["lp"]]
    a=mp.matrix([[dm[i]*ctx["u"][i,j]*mp.sqrt(ctx["sig"][j]) for j in range(maximum)] for i in range(n)])
    bt=mp.matrix([[mp.sqrt(ctx["sig"][i])*ctx["vh"][i,j]*dp[j] for j in range(n)] for i in range(maximum)])
    weighted=mp.matrix([[ctx["f"][i,j]*dm[i]*dp[j] for j in range(n)] for i in range(n)])
    return dict(ctx=ctx,a=a,v=bt*ctx["rinv"],y=ctx["matrices"]["minus"]*a,
        delta0=ctx["matrices"]["minus"]*weighted*ctx["rinv"])


def at_rank(tc,rank):
    ctx,m=tc["ctx"],tc["ctx"]["matrices"]
    if rank:
        a,v,y=tc["a"][:,:rank],tc["v"][:rank,:],tc["y"][:,:rank]
        h=mp.eye(rank)+v*m["Ccross"]*a
        left=m["Bperp"]*a*mp.inverse(h)
        lu,lr=mp.qr(left,mode="skinny")
        rv,rr=mp.qr(v.H,mode="skinny")
        ku,ks,kvh=mp.svd(lr*rr.H)
        uu,vv=lu*ku,rv*kvh.H
        cosine=[1/mp.sqrt(1+x*x) for x in ks]
        sine=[x/mp.sqrt(1+x*x) for x in ks]
        ga,ha=m["G"]*vv,m["Gperp"]*uu
        q=m["G"]+(ga*(mp.diag(cosine)-mp.eye(rank))+ha*mp.diag(sine))*vv.H
        kappa,smallest=ks[0],core_smin(m["G"],y,v)
        delta=tc["delta0"]-y*v
        factors=dict(g=ga,h=ha,cosine=cosine,sine=sine)
    else:
        q,kappa,smallest,delta,factors=m["G"],mp.mpf(0),mp.mpf(1),tc["delta0"],None
    numerator=norm(delta)
    directional=min(mp.mpf(1),numerator/smallest)
    transverse=norm(delta-q*(q.H*delta))/(smallest-numerator) if numerator<smallest else mp.mpf(1)
    return dict(rank=rank,distance_to_G=kappa/mp.sqrt(1+kappa*kappa),projector_remainder=min(directional,transverse),
        directional_remainder=directional,transverse_remainder=min(mp.mpf(1),transverse),small_core_smin=smallest),q,factors


def predict(ctx,duration):
    tick=time.perf_counter()
    tc=time_context(ctx,duration)
    attempts=[]
    for rank in RANKS:
        if rank>len(ctx["lp"]):
            continue
        values,q,factors=at_rank(tc,rank)
        attempts.append({k:float(v) for k,v in values.items()})
        if values["projector_remainder"]<=mp.mpf("1e-5"):
            break
    static=ctx["static"]
    z=mp.mpf(static["q"])*mp.exp(-mp.mpf(static["gap"])*duration)
    p=mp.mpf(static["p"])
    ew=min(mp.mpf(1),z/(p-z)) if z<p else mp.mpf(1)
    ent=entropy(q)
    eps,d=values["projector_remainder"],values["distance_to_G"]
    budget=entropy_budget(eps,q.cols)+entropy_budget(ew,q.cols)
    error=abs(ent-mp.mpf(static["gain_entropy"]))
    values.update(entropy=ent,output_to_G_bound=ew,distance_lower=max(0,d-eps-ew),distance_upper=min(1,d+eps+ew),
        entropy_error_lower=max(0,error-budget),entropy_error_upper=error+budget)
    return values,q,factors,attempts,time.perf_counter()-tick


def compact_pair(a,b):
    active=[(f,sign) for f,sign in ((a,1),(b,-1)) if f is not None]
    if not active:
        return mp.mpf(0)
    order=sum(2*f["g"].cols for f,_ in active)
    span=mp.matrix(active[0][0]["g"].rows,order)
    kernel,offset=mp.matrix(order),0
    for f,sign in active:
        rank=f["g"].cols
        span[:,offset:offset+rank],span[:,offset+rank:offset+2*rank]=f["g"],f["h"]
        for j,(c,s) in enumerate(zip(f["cosine"],f["sine"])):
            i,k=offset+j,offset+rank+j
            kernel[i,i],kernel[k,k]=-sign*s*s,sign*s*s
            kernel[i,k]=kernel[k,i]=sign*c*s
        offset+=2*rank
    if span.cols<=span.rows:
        _,triangular=mp.qr(span,mode="skinny")
    else:
        _,s,vh=mp.svd(span)
        triangular=mp.diag(s)*vh
    small=triangular*kernel*triangular.H
    return max(abs(x) for x in mp.eighe((small+small.H)/2,eigvals_only=True))


def pair_bound(a,b):
    va,vb=a[0],b[0]
    direct=compact_pair(a[2],b[2])
    eps=va["projector_remainder"]+vb["projector_remainder"]
    eb=entropy_budget(va["projector_remainder"],a[1].cols)+entropy_budget(vb["projector_remainder"],b[1].cols)
    difference=abs(va["entropy"]-vb["entropy"])
    return dict(reduced_pair_distance=direct,memory_lower=max(0,direct-eps),memory_upper=min(1,direct+eps),
        reduced_entropy_difference=difference,entropy_memory_lower=max(0,difference-eb),entropy_memory_upper=difference+eb)


def kernel_checks():
    rows=[]
    with mp.workdps(100):
        for seed in (3,17,41):
            rng=np.random.default_rng(seed)
            a=rng.normal(size=(12,6))+1j*rng.normal(size=(12,6))
            full,_=mp.qr(mp.matrix(a.tolist()))
            g,gp=full[:,:6],full[:,6:]
            cross=mp.matrix((.04*(rng.normal(size=(6,6))+1j*rng.normal(size=(6,6)))).tolist())
            minus=gp+g*cross
            rinv=mp.eye(6)
            f=mp.matrix((rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))).tolist())
            u,s,vh=mp.svd(f)
            ctx=dict(matrices=dict(G=g,Gperp=gp,minus=minus,Ccross=cross,Bperp=mp.eye(6)),rinv=rinv,f=f,u=u,vh=vh,sig=list(s),
                lp=[mp.mpc(1+.1*i,.2*i) for i in range(6)],lm=[mp.mpc(-1-.1*i,.3*i) for i in range(6)])
            tc=time_context(ctx,mp.mpf(".4"))
            exact=mp.qr(g+tc["delta0"],mode="skinny")[0]
            first=None
            for rank in (0,1,2,4,6):
                values,q,factors=at_rank(tc,rank)
                z=g+tc["y"][:,:rank]*tc["v"][:rank,:] if rank else g
                qr=mp.qr(z,mode="skinny")[0]
                error=distance(q,qr)
                measured=distance(q,exact)
                smallest=smin(z)
                assert error<mp.mpf("1e-70") and measured<=values["projector_remainder"]+mp.mpf("1e-70")
                assert abs(smallest-values["small_core_smin"])<mp.mpf("1e-70")
                if first:
                    assert abs(distance(first[0],q)-compact_pair(first[1],factors))<mp.mpf("1e-70")
                first=(q,factors)
                rows.append(dict(seed=seed,rank=rank,graph_qr_error_mp=str(error),remainder_error_mp=str(measured),
                    remainder_mp=str(values["projector_remainder"]),smin_error_mp=str(abs(smallest-values["small_core_smin"]))))
        family=dict(length=8,skin=.2,gamma=2.,same_sublattice=.2)
        h=hamiltonian(family)
        h0=hamiltonian({**family,"skin":0.})
        d=mp.diag([mp.exp(mp.mpf(".2")*(i//2)) for i in range(8)])
        similarity=norm(h-d*h0*mp.inverse(d))
        c,loss,jumps,residual=passive_embedding(h)
        assert similarity<mp.mpf("1e-70") and residual<mp.mpf("1e-70")
        assert mp.eighe(loss,eigvals_only=True)[0]>=-mp.mpf("1e-70")
    result=dict(status="passed",rows=rows,complex_adjoints_checked=True,similarity_residual_mp=str(similarity),
        passive_reconstruction_residual_mp=str(residual),loss_shift_mp=str(c),jump_max_support=max(sum(z!=0 for z in row) for row in jumps))
    (OUT/"development").mkdir(parents=True,exist_ok=True)
    write_json(OUT/"development/kernel_checks.json",result)
    return dict(status="kernel_checks_passed",checks=len(rows))


def prediction_job(family):
    target=OUT/"predictions"/(family["key"]+".json")
    if target.exists():
        return dict(status="reused",family=family["key"])
    tick,started=time.perf_counter(),io.now()
    statics=read_gzip(OUT/"static"/(family["key"]+".raw.json.gz"))[-2:]
    ctx=[]
    for static in statics:
        with mp.workdps(static["dps"]):
            ctx.append(contexts(static))
    rows,pairs,raw=[],[],[]
    for case in [c for c in config()["cases"] if c["family_key"]==family["key"]]:
        local=[]
        for static,contexts_at_precision in zip(statics,ctx):
            with mp.workdps(static["dps"]):
                local.append({name:predict(contexts_at_precision[name],mp.mpf(case["physical_time"])) for name in case["initials"]})
        with mp.workdps(280):
            for name in case["initials"]:
                low,high=local[0][name][0],local[1][name][0]
                change=max(abs(low[k]-high[k]) for k in high)
                assert change<=mp.mpf("1e-10")
                values={k:float(v) for k,v in high.items()}
                rows.append(dict(**case,initial=name,**values,prediction_precision_change=float(change),seconds=local[1][name][4],
                    rank_target_passed=high["projector_remainder"]<=mp.mpf("1e-5"),
                    guaranteed_selected=high["distance_upper"]<=mp.mpf(".05") and high["entropy_error_upper"]<=mp.mpf(".05"),
                    guaranteed_unselected=high["distance_lower"]>mp.mpf(".05") or high["entropy_error_lower"]>mp.mpf(".05")))
                raw.append(dict(case_id=case["case_id"],initial=name,Q=encode(local[1][name][1]),values={k:str(v) for k,v in high.items()},attempts=local[1][name][3]))
        for na,nb in case["pairs"]:
            checks=[]
            for static,local_at_precision in zip(statics,local):
                with mp.workdps(static["dps"]):
                    checks.append(pair_bound(local_at_precision[na],local_at_precision[nb]))
            with mp.workdps(280):
                change=max(abs(checks[0][k]-checks[1][k]) for k in checks[0])
                assert change<=mp.mpf("1e-10")
                high=checks[-1]
                pairs.append(dict(**case,initial_a=na,initial_b=nb,**{k:float(v) for k,v in high.items()},prediction_precision_change=float(change),
                    guaranteed_memory=high["memory_lower"]>mp.mpf(".1"),guaranteed_entropy_memory=high["entropy_memory_lower"]>mp.mpf(".1")))
    write_gzip(target.with_suffix(".raw.json.gz"),raw)
    write_json(target,dict(rows=rows,pairs=pairs,family=family,source_sha256=sha256(SOURCE),started_utc=started,
        completed_utc=io.now(),seconds=time.perf_counter()-tick,static_sha256=sha256(OUT/"static"/(family["key"]+".raw.json.gz"))))
    return dict(status="prediction_converged",family=family["key"],states=len(rows),pairs=len(pairs))


def lock():
    if (OUT/"prediction_lock.json").exists():
        check_lock()
        return dict(status="reused")
    assert not list((OUT/"cases").glob("*.json"))
    assert read_json(OUT/"development/kernel_checks.json")["status"]=="passed"
    predictions=[read_json(OUT/"predictions"/(f["key"]+".json")) for f in config()["families"]]
    rows,pairs=[r for v in predictions for r in v["rows"]],[r for v in predictions for r in v["pairs"]]
    assert len(rows)==sum(len(c["initials"]) for c in config()["cases"])
    assert len(pairs)==sum(len(c["pairs"]) for c in config()["cases"])
    write_csv(OUT/"locked_predictions.csv",rows)
    write_csv(OUT/"locked_memory_predictions.csv",pairs)
    paths=[CONFIG,OUT/"development/kernel_checks.json",OUT/"kernel_manifest.json"]
    paths.extend(p for folder in ("static","predictions") for p in (OUT/folder).glob('*') if p.is_file() and '.progress.' not in p.name)
    write_json(OUT/"prediction_lock.json",dict(locked_utc=io.now(),source_sha256=sha256(SOURCE),config_sha256=sha256(CONFIG),
        states=len(rows),pairs=len(pairs),inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in sorted(paths)]))
    write_json(OUT/"prediction_lock_hash.json",{name:sha256(OUT/name) for name in ("prediction_lock.json","locked_predictions.csv","locked_memory_predictions.csv")})
    return dict(status="predictions_locked",states=len(rows),pairs=len(pairs))


def check_lock():
    check_source()
    for name,digest in read_json(OUT/"prediction_lock_hash.json").items():
        assert sha256(OUT/name)==digest


def reference(case,digits):
    with mp.workdps(digits):
        h=hamiltonian(case)
        c,_,_,_=passive_embedding(h)
        matrix=mp.expm(-mp.j*h*mp.mpf(case["physical_time"]))
        scale=norm(matrix)
        matrix/=scale
        w,s,vh=mp.svd(matrix)
        n=case["length"]//2
        wt=w[:,:n]
        raw=dict(dps=digits,M=encode(matrix),W=encode(wt),VH=encode(vh),singular_values=[str(x) for x in s],
            W_entropy=str(entropy(wt)),loss_shift=str(c),log_fro_scale=str(mp.log(scale)),states={})
        rows=[]
        arrays=dict(M=io.audit.mp_to_numpy(matrix),W=io.audit.mp_to_numpy(wt),singular_values=np.array([float(x) for x in s]))
        for name in case["initials"]:
            initial=q0(case["length"],name)
            evolved,triangular=mp.qr(matrix*initial,mode="skinny")
            d,ent=distance(evolved,wt),entropy(evolved)
            alpha=smin(vh[:n,:]*initial)
            ratio=s[n]/s[n-1]
            logp=2*mp.fsum(mp.log(abs(triangular[i,i])) for i in range(n))+2*n*(mp.log(scale)-c*mp.mpf(case["physical_time"]))
            assert logp <= mp.mpf("1e-40")
            row=dict(case_id=case["case_id"],initial=name,length=case["length"],skin=case["skin"],physical_time=case["physical_time"],
                entropy=float(ent),W_entropy=float(mp.mpf(raw["W_entropy"])),distance_to_W=float(d),entropy_error=float(abs(ent-mp.mpf(raw["W_entropy"]))),
                alpha=float(alpha),ratio=float(ratio),log_no_click_success=float(logp),
                actual_selected=d<=mp.mpf(".05") and abs(ent-mp.mpf(raw["W_entropy"]))<=mp.mpf(".05"))
            rows.append(row)
            raw["states"][name]=dict(Q=encode(evolved),entropy=str(ent),distance_to_W=str(d),alpha=str(alpha),ratio=str(ratio),log_no_click_success=str(logp))
            arrays['Q_'+name]=io.audit.mp_to_numpy(evolved)
        pairs=[]
        for na,nb in case["pairs"]:
            a,b=raw["states"][na],raw["states"][nb]
            d=distance(decode(a["Q"]),decode(b["Q"]))
            e=abs(mp.mpf(a["entropy"])-mp.mpf(b["entropy"]))
            pairs.append(dict(case_id=case["case_id"],initial_a=na,initial_b=nb,pair_distance=float(d),absolute_entropy_difference=float(e),
                actual_memory=d>mp.mpf(".1"),actual_entropy_memory=e>mp.mpf(".1")))
        return rows,pairs,raw,arrays


def reference_change(a,b):
    with mp.workdps(280):
        ra,rb=a[2],b[2]
        projector=distance(decode(ra["W"]),decode(rb["W"]))
        ent=abs(mp.mpf(ra["W_entropy"])-mp.mpf(rb["W_entropy"]))
        relative=mp.mpf(0)
        for name in rb["states"]:
            sa,sb=ra["states"][name],rb["states"][name]
            projector=max(projector,distance(decode(sa["Q"]),decode(sb["Q"])))
            ent=max(ent,abs(mp.mpf(sa["entropy"])-mp.mpf(sb["entropy"])))
            for key in ("alpha","ratio"):
                relative=max(relative,abs(mp.mpf(sa[key])-mp.mpf(sb[key]))/mp.mpf(sb[key]))
        for index in (len(rb["singular_values"])//2-1,len(rb["singular_values"])//2):
            relative=max(relative,abs(mp.mpf(ra["singular_values"][index])-mp.mpf(rb["singular_values"][index]))/mp.mpf(rb["singular_values"][index]))
    return dict(projector=float(projector),entropy=float(ent),relative_scalar=float(relative))


def reference_job(case):
    check_lock()
    target=OUT/"cases"/(case["case_id"]+".json")
    if target.exists():
        return dict(status="reused",case_id=case["case_id"])
    tick,started,values,attempts=time.perf_counter(),io.now(),[],[]
    for digits in case["digits"]:
        local=time.perf_counter()
        values.append(reference(case,digits))
        change=reference_change(values[-2],values[-1]) if len(values)>1 else None
        attempts.append(dict(dps=digits,seconds=time.perf_counter()-local,convergence=change))
        write_json(target.with_suffix(".progress.json"),dict(started_utc=started,attempts=attempts))
        if change and change["relative_scalar"]<=1e-8 and max(change["projector"],change["entropy"])<=1e-10:
            break
    else:
        write_gzip(target.with_suffix(".failed.raw.json.gz"),[v[2] for v in values])
        raise ArithmeticError("complex exponential reference did not converge")
    rows,pairs,raw,arrays=values[-1]
    write_gzip(target.with_suffix(".raw.json.gz"),[v[2] for v in values])
    np.savez_compressed(target.with_suffix(".npz"),**arrays)
    write_json(target,dict(case=case,rows=rows,pairs=pairs,attempts=attempts,reference_convergence=change,
        precision_digits=[v[2]["dps"] for v in values],source_sha256=sha256(SOURCE),prediction_lock_sha256=sha256(OUT/"prediction_lock.json"),
        started_utc=started,completed_utc=io.now(),seconds=time.perf_counter()-tick))
    return dict(status="reference_converged",case_id=case["case_id"],dps=digits)


def audit_case(case):
    check_lock()
    target=OUT/"audits"/(case["case_id"]+".json")
    if target.exists():
        return dict(status="reused",case_id=case["case_id"])
    truth_path=OUT/"cases"/(case["case_id"]+".json")
    truth=read_gzip(truth_path.with_suffix(".raw.json.gz"))[-1]
    prediction_path=OUT/"predictions"/(case["family_key"]+".raw.json.gz")
    predicted={r["initial"]:r for r in read_gzip(prediction_path) if r["case_id"]==case["case_id"]}
    checks=[]
    with mp.workdps(max(200,truth["dps"])):
        allowance=mp.mpf("1e-35")
        for name in case["initials"]:
            actual,pr=truth["states"][name],predicted[name]
            values=pr["values"]
            qa,qp=decode(actual["Q"]),decode(pr["Q"])
            error=distance(qa,qp)
            eps=mp.mpf(values["projector_remainder"])
            de=abs(mp.mpf(actual["entropy"])-mp.mpf(values["entropy"]))
            d=mp.mpf(actual["distance_to_W"])
            e=abs(mp.mpf(actual["entropy"])-mp.mpf(truth["W_entropy"]))
            gates=dict(remainder_covered=error<=eps+allowance,entropy_remainder_covered=de<=entropy_budget(eps,qa.cols)+allowance,
                distance_interval_covered=mp.mpf(values["distance_lower"])-allowance<=d<=mp.mpf(values["distance_upper"])+allowance,
                entropy_interval_covered=mp.mpf(values["entropy_error_lower"])-allowance<=e<=mp.mpf(values["entropy_error_upper"])+allowance,
                bases_orthogonal=max(norm(q.H*q-mp.eye(q.cols)) for q in (qa,qp))<=allowance)
            assert all(gates.values()),(case["case_id"],name,gates)
            checks.append(dict(case_id=case["case_id"],initial=name,**gates,projector_error_mp=str(error),remainder_mp=str(eps),entropy_error_mp=str(de)))
        pair_checks=[]
        family=read_json(OUT/"predictions"/(case["family_key"]+".json"))
        for actual in read_json(truth_path)["pairs"]:
            pr=next(r for r in family["pairs"] if r["case_id"]==case["case_id"] and r["initial_a"]==actual["initial_a"] and r["initial_b"]==actual["initial_b"])
            a,b=truth["states"][actual["initial_a"]],truth["states"][actual["initial_b"]]
            measured=distance(decode(a["Q"]),decode(b["Q"]))
            diff=abs(mp.mpf(a["entropy"])-mp.mpf(b["entropy"]))
            # The saved pair CSV is rounded; allow only its documented binary64 export error.
            export_allowance=mp.mpf("1e-14")
            assert mp.mpf(str(pr["memory_lower"]))-export_allowance<=measured<=mp.mpf(str(pr["memory_upper"]))+export_allowance
            assert mp.mpf(str(pr["entropy_memory_lower"]))-export_allowance<=diff<=mp.mpf(str(pr["entropy_memory_upper"]))+export_allowance
            pair_checks.append(dict(initial_a=actual["initial_a"],initial_b=actual["initial_b"],projector_covered=True,entropy_covered=True))
    write_json(target,dict(rows=checks,pairs=pair_checks,verified_utc=io.now(),precision_digits=max(200,truth["dps"]),
        reference_sha256=sha256(truth_path),reference_raw_sha256=sha256(truth_path.with_suffix(".raw.json.gz")),prediction_raw_sha256=sha256(prediction_path)))
    return dict(status="audit_passed",case_id=case["case_id"],states=len(checks))


def execute(stage,index):
    check_source()
    info,tick=io.environment(stage),time.perf_counter()
    key=str(index)
    try:
        if stage in ("static","prediction"):
            family=config()["families"][index]
            key=family["key"]
            value=static_job(family) if stage=="static" else prediction_job(family)
        elif stage=="kernel":value=kernel_checks()
        elif stage=="lock":value=lock()
        else:
            case=config()["cases"][index]
            key=case["case_id"]
            value=reference_job(case) if stage=="reference" else audit_case(case)
        emit(dict(stage=stage,**value))
    except Exception as error:
        write_json(OUT/"failures"/(stage+"_"+key+"_"+io.now().replace(":","-")+".json"),
            dict(stage=stage,key=key,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),environment=info))
        raise
    finally:
        info.update(seconds=time.perf_counter()-tick,completed_utc=io.now())
        write_json(OUT/"hpc"/("environment_"+stage+"_"+key+".json"),info)


def batch_worker(stage,index,cpu):
    os.sched_setaffinity(0,{cpu})
    execute(stage,index)


def batch(stage,index):
    indices=list(range(index*4,min(index*4+4,len(config()["cases"]))))
    cpus=sorted(os.sched_getaffinity(0))
    assert len(cpus)>=len(indices)
    receipt=dict(stage=stage,batch_index=index,case_indices=indices,started_utc=io.now(),cpus=cpus)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(indices),mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in [pool.submit(batch_worker,stage,case,cpu) for case,cpu in zip(indices,cpus)]:
                future.result()
        receipt["status"]="completed"
    except Exception as error:
        receipt.update(status="failed",error=str(error))
        raise
    finally:
        receipt["completed_utc"]=io.now()
        write_json(OUT/"hpc"/f"batch_{stage}_{index}.json",receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("initialize","kernel","static","prediction","lock","reference","audit","batch_reference","batch_audit"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    if args.stage=="initialize":emit(initialize())
    elif args.stage.startswith("batch_"):batch(args.stage[6:],args.index)
    else:execute(args.stage,args.index)


if __name__=="__main__":
    main()
