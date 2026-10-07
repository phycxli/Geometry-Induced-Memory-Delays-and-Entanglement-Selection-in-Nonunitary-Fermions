"""Locked geometric-susceptibility study and genuinely new-size controls."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import mpmath as mp
import numpy as np
from scipy.linalg import expm, qr, svd
from scipy.optimize import brentq
from mietf_skin.response import GaugedGraph, contraction_constants, ssh_real_generator
import run_theory_upgrade_hpc as theory

old = theory.old
OUT = ROOT/"data/prl_priority_strengthening"
LEVELS = (0.3, 0.05, 0.01)
CHIS = (-4., -2., -1., -.5, .5, 1., 2., 4.)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config():
    return load(OUT/"config.json")


def initialize():
    families = [dict(key=f"L{n}", length=n, gamma=2., t1=.5, t2=1.,
                     role="retested_size", source=f"data/prl_figure_strengthening/spectral/L{n}.raw.json.gz")
                for n in (128, 192, 256, 320, 384)]
    families += [dict(key=f"L{n}", length=n, gamma=2., t1=.5, t2=1., role="new_dynamical_size", source=None)
                 for n in (448, 512)]
    families += [dict(key=f"control{k}", length=96, gamma=g, t1=t, t2=w,
                      role="new_parameter_control", source=None)
                 for k, (g, t, w) in enumerate(((2.2,.55,.9),(2.,.6,1.05),(2.4,.4,.8),(2.1,.45,1.))) ]
    value = dict(date="2026-10-07", families=families, levels=list(LEVELS), chis=list(CHIS),
                 names=["left","right"], root_tolerance=2e-9, linear_time_gate=.01,
                 asymptotic_susceptibility_gate=.005, direct_reference_offset=.005,
                 interpretation="ret tested sizes are diagnostics; new nonzero-bias predictions lock after zero-bias roots and before scans",
                 conjecture="edge susceptibility approaches -s/[4 sqrt(gamma^2-(t1+t2)^2)], s=+1 left,-1 right",
                 caveat="The conjecture is not implied by the static delta response. Finite-size failures remain reported.")
    if (OUT/"config.json").exists():
        assert config()==value
    else:
        save(OUT/"config.json", value)
        paths = [ROOT/"main/main.tex", ROOT/"supplement/supplement.tex", ROOT/"main_zh/main.tex",
                 ROOT/"supplment_zh/supplement.tex", ROOT/"data/prl_theory_upgrade/roots.csv",
                 ROOT/"data/prl_p3_fine_front/geometry_response.csv", Path(__file__), ROOT/"src/mietf_skin/response.py"]
        save(OUT/"hypothesis_lock.json", dict(locked_utc=old.now(), config_sha256=digest(OUT/"config.json"),
             inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=digest(p)) for p in paths],
             conjecture=value["conjecture"], gates={k:value[k] for k in ("linear_time_gate","asymptotic_susceptibility_gate")},
             scope="finite-difference identities are implementation checks; only new-bias scans are predictor holdouts"))
        import zipfile
        with zipfile.ZipFile(OUT/"manuscript_baseline.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for p in paths[:4]: archive.write(p, p.relative_to(ROOT).as_posix())
    return dict(status="initialized", families=len(families), conjecture=value["conjecture"])


def fast_coupling(length, t1, t2):
    n, t, w = length//2, mp.mpf(str(t1)), mp.mpf(str(t2))
    ratio = w/t
    if ratio <= 1+mp.mpf(1)/n:
        raise ValueError("This analytic kernel assumes one isolated SSH edge singular mode.")
    u, v, singular = mp.matrix(n), mp.matrix(n), []
    for j in range(1, n):
        lo, hi = j*math.pi/(n+1), j*math.pi/n
        seed = brentq(lambda k: math.sin((n+1)*k)+float(ratio)*math.sin(n*k), lo, hi, xtol=1e-14)
        f = lambda k: mp.sin((n+1)*k)+ratio*mp.sin(n*k)
        df = lambda k: (n+1)*mp.cos((n+1)*k)+ratio*n*mp.cos(n*k)
        k = mp.findroot(f, mp.mpf(str(seed)), df=df, solver="newton", maxsteps=30)
        vector = mp.matrix([mp.sin((i+1)*k) for i in range(n)])
        vector /= mp.norm(vector)
        column = mp.matrix([t*vector[i]+(w*vector[i-1] if i else 0) for i in range(n)])
        s = mp.norm(column)
        column /= s
        if column[0]<0: vector, column = -vector, -column
        v[:,j-1], u[:,j-1] = vector, column
        singular.append(s)
    f = lambda q: 1-ratio*q+ratio*q**(2*n+1)-q**(2*n+2)
    q = mp.findroot(f, 1/ratio, solver="newton", maxsteps=30)
    vector = mp.matrix([(-1)**i*q**(n-i-1)*(1-q**(2*i+2)) for i in range(n)])
    vector /= mp.norm(vector)
    column = mp.matrix([t*vector[i]+(w*vector[i-1] if i else 0) for i in range(n)])
    s = mp.norm(column)
    column /= s
    if column[0]<0: vector, column = -vector, -column
    v[:,n-1], u[:,n-1] = vector, column
    singular.append(s)
    return u, singular, v


def spectral_value(family, precision):
    mp.mp.dps = precision
    length, gamma = family["length"], mp.mpf(str(family["gamma"]))
    n = length//2
    u, singular, v = fast_coupling(length, family["t1"], family["t2"])
    mu = [mp.sqrt(gamma*gamma-s*s) for s in singular]
    ratios = [s/(gamma+m) for s,m in zip(singular,mu)]
    plus, minus, yp, ym = mp.matrix(length,n), mp.matrix(length,n), mp.matrix(n,length), mp.matrix(n,length)
    for k in range(n):
        aa, bb = 1/mp.sqrt(1+ratios[k]**2), ratios[k]/mp.sqrt(1+ratios[k]**2)
        for j in range(n):
            plus[2*j,k], plus[2*j+1,k] = aa*u[j,k], -bb*v[j,k]
            minus[2*j,k], minus[2*j+1,k] = bb*u[j,k], -aa*v[j,k]
            yp[k,2*j], yp[k,2*j+1] = gamma/mu[k]*aa*u[j,k], gamma/mu[k]*bb*v[j,k]
            ym[k,2*j], ym[k,2*j+1] = -gamma/mu[k]*bb*u[j,k], -gamma/mu[k]*aa*v[j,k]
    states = {}
    for name in ("left","right"):
        cols = old.base.occupied(length,name)
        numerator = mp.matrix([[ym[i,j] for j in cols] for i in range(n)])
        denominator = mp.matrix([[yp[i,j] for j in cols] for i in range(n)])
        f = old.cached_right_solve(numerator,denominator)
        relative_residual = old.p1.mp_norm(f*denominator-numerator)/old.p1.mp_norm(numerator)
        assert relative_residual < mp.mpf("1e-40"), relative_residual
        states[name] = dict(F=old.p1.encode_matrix(f), solve_relative_residual=str(relative_residual))
    return dict(**family, dps=precision, mu=[str(x) for x in mu], plus=old.p1.encode_matrix(plus),
                minus=old.p1.encode_matrix(minus), states=states,
                kernel="analytic finite-end recurrence plus arbitrary-precision linear solve")


def convert_spectral(raw):
    mp.mp.dps = raw["dps"]
    arrays = {k:np.array([[float(x) for x in row] for row in raw[k]],float) for k in ("plus","minus")}
    arrays["mu"] = np.array(raw["mu"],float)
    for name in ("left","right"):
        matrix = old.spatial.decode(raw["states"][name]["F"])
        arrays[name+"_sign"] = np.array([[float(mp.sign(matrix[i,j])) for j in range(matrix.cols)] for i in range(matrix.rows)])
        arrays[name+"_log"] = np.array([[float(mp.log(abs(matrix[i,j]))) if matrix[i,j] else -np.inf
                                       for j in range(matrix.cols)] for i in range(matrix.rows)])
    return arrays


def direct_norm_bounds(matrix):
    """Exact-arithmetic norm bounds from a full matrix and one trial direction.

    Right-orthogonal splitting gives ||R v|| <= ||R|| <=
    sqrt(||R v||^2 + ||R - (R v)v^H||_F^2). The trial direction comes
    from float64 SVD, while normalization and the complete remainder use mp.
    """
    array=np.array([[float(matrix[i,j]) for j in range(matrix.cols)] for i in range(matrix.rows)])
    _,_,vh=svd(array,full_matrices=False,check_finite=False)
    v=mp.matrix([mp.mpf(float(x)) for x in vh[0]])
    v/=mp.norm(v)
    z=matrix*v
    leading_squared=mp.fsum(abs(x)**2 for x in z)
    tail_squared=mp.fsum(abs(matrix[i,j]-z[i]*mp.conj(v[j]))**2
                         for i in range(matrix.rows) for j in range(matrix.cols))
    return mp.sqrt(leading_squared),mp.sqrt(leading_squared+tail_squared)


def prepare(index):
    family = config()["families"][index]
    target = OUT/"cache"/(family["key"]+".json")
    if target.exists(): return load(target)
    tick = time.perf_counter()
    if family["source"]:
        source = ROOT/family["source"]
        raw = old.read_gzip(source)
    else:
        source = OUT/"spectral"/(family["key"]+".raw.json.gz")
        if source.exists(): raw = old.read_gzip(source)
        else:
            lower = max(200,20*math.ceil((1.3*family["length"]+30)/20))
            raw = []
            for precision in (lower,lower+40):
                raw.append(spectral_value(family,precision))
                save(OUT/"spectral"/(family["key"]+".progress.json"), dict(dps=precision, seconds=time.perf_counter()-tick))
            old.write_gzip(source, raw)
    values = [convert_spectral(value) for value in raw[-2:]]
    change = 0.0
    for k in values[0]:
        finite = np.isfinite(values[0][k]) & np.isfinite(values[1][k])
        assert np.array_equal(np.isfinite(values[0][k]),np.isfinite(values[1][k]))
        change = max(change,float(np.max(np.abs(values[0][k][finite]-values[1][k][finite]))))
    assert change<2e-10, change
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target.with_suffix(".npz"), **values[-1])
    value = dict(family=family, source=source.relative_to(ROOT).as_posix(), source_sha256=digest(source),
                 precision_digits=[r["dps"] for r in raw[-2:]], float_cache_absolute_change=change,
                 seconds=time.perf_counter()-tick, completed_utc=old.now(), cache_sha256=digest(target.with_suffix(".npz")))
    save(target,value)
    return value


def graph_for(family, name, chi):
    arrays = np.load(OUT/"cache"/(family["key"]+".npz"))
    return GaugedGraph(arrays["plus"],arrays["minus"],arrays["mu"],arrays[name+"_log"],arrays[name+"_sign"],chi,
                      family["gamma"],family["t1"],family["t2"])


def root(graph, level, seed):
    width = 1.0
    for _ in range(8):
        lo, hi = max(.1,seed-width), seed+width
        if graph.distance(lo)>level and graph.distance(hi)<level:
            return brentq(lambda t:graph.distance(t)-level,lo,hi,xtol=config()["root_tolerance"])
        width *= 2
    raise ArithmeticError("Failed to find a stable low-threshold bracket.")


def baseline(index):
    family = config()["families"][index]
    target = OUT/"predictions"/(family["key"]+".json")
    if target.exists(): return load(target)
    rows = []
    for name in ("left","right"):
        graph = graph_for(family,name,0.)
        if family["gamma"]==2 and family["t1"]==.5 and family["t2"]==1:
            seed = .54*family["length"]-2
        else:
            seed = max(3.,.54*family["length"]*2/family["gamma"])
        for level in LEVELS:
            seed = root(graph,level,seed)
            response = asdict(graph.response(seed))
            assert response["principal_gap"]>1e-8 and response["decay_rate"]>0
            static_guess = (-1 if name=="left" else 1)/(4*math.sqrt(family["gamma"]**2-(family["t1"]+family["t2"])**2))
            for chi in CHIS:
                rows.append(dict(initial=name, level=level, chi=chi, zero_time=seed,
                                 response=response, predicted_time=seed+chi*response["susceptibility"],
                                 static_gap_guess=seed+chi*static_guess, static_guess_susceptibility=static_guess))
    value = dict(family=family, rows=rows, locked_utc=old.now(), source_sha256=digest(Path(__file__)),
                 cache_sha256=digest(OUT/"cache"/(family["key"]+".npz")),
                 gate=config()["linear_time_gate"], policy="zero-bias first-order prediction; no nonzero-bias root used")
    save(target,value)
    return dict(status="bias_predictions_locked", family=family["key"], predictions=len(rows))


def scan(index):
    family = config()["families"][index]
    target = OUT/"scans"/(family["key"]+".json")
    if target.exists(): return load(target)
    prediction = OUT/"predictions"/(family["key"]+".json")
    locked = load(prediction)
    rows = []
    for name in ("left","right"):
        for chi in CHIS:
            graph = graph_for(family,name,chi)
            for candidate in [r for r in locked["rows"] if r["initial"]==name and r["chi"]==chi]:
                level = candidate["level"]
                duration = root(graph,level,candidate["predicted_time"])
                response = asdict(graph.response(duration))
                step = 1e-4
                fd_chi = (graph_for(family,name,chi+step).distance(duration)
                          -graph_for(family,name,chi-step).distance(duration))/(2*step)
                fd_time = (graph.distance(duration+step)-graph.distance(duration-step))/(2*step)
                assert abs(fd_chi-response["distance_chi_derivative"])<2e-7
                assert abs(fd_time-response["distance_time_derivative"])<2e-7
                constants = contraction_constants(family["gamma"],family["t1"],family["t2"],chi/family["length"])
                k = response["tangent"]
                uniform_bound = (.5-1/family["length"])*(1+k)/(2*constants["a"]-constants["b"]*k)
                assert abs(response["susceptibility"]) <= uniform_bound+1e-8
                rows.append(dict(**{k:v for k,v in candidate.items() if k!="response"},
                                 root_time=duration, response=response, linear_error=duration-candidate["predicted_time"],
                                 static_gap_error=duration-candidate["static_gap_guess"],
                                 linear_pass=abs(duration-candidate["predicted_time"])<=locked["gate"],
                                 uniform_susceptibility_bound=uniform_bound,
                                 chi_finite_difference_error=abs(fd_chi-response["distance_chi_derivative"]),
                                 time_finite_difference_error=abs(fd_time-response["distance_time_derivative"])))
    value = dict(family=family,rows=rows,prediction_sha256=digest(prediction),completed_utc=old.now(),
                 scientific_status="finite-size tests; directional sign is observed, not an asymptotic theorem")
    save(target,value)
    return dict(status="scan_complete", family=family["key"], roots=len(rows), passes=sum(r["linear_pass"] for r in rows))


def kernel():
    rows = []
    for gamma,t1,t2 in ((2.,.5,1.),(2.2,.55,.9),(2.,.6,1.05)):
        for length in (16,24):
            with mp.workdps(100):
                u,s,v = fast_coupling(length,t1,t2)
                c = mp.matrix(length//2)
                for j in range(c.rows):
                    c[j,j]=mp.mpf(str(t1))
                    if j:c[j,j-1]=mp.mpf(str(t2))
                direct_s = mp.svd(c,compute_uv=False)
                spectral_error = max(abs(a-b) for a,b in zip(s,direct_s))
                assert spectral_error<mp.mpf("1e-70"), spectral_error
                family=dict(key="kernel",length=length,gamma=gamma,t1=t1,t2=t2)
                raw=spectral_value(family,100)
                arrays=convert_spectral(raw)
            for name in ("left","right"):
                for chi in (-2.,0.,2.):
                    graph=GaugedGraph(arrays["plus"],arrays["minus"],arrays["mu"],arrays[name+"_log"],arrays[name+"_sign"],chi,gamma,t1,t2)
                    duration=root(graph,.05,.5*length)
                    response=graph.response(duration)
                    with mp.workdps(100):
                        generator=mp.matrix(length)
                        for j in range(length//2):
                            generator[2*j,2*j],generator[2*j+1,2*j+1]=mp.mpf(str(gamma)),-mp.mpf(str(gamma))
                            generator[2*j,2*j+1],generator[2*j+1,2*j]=mp.mpf(str(t1)),-mp.mpf(str(t1))
                            if j:
                                generator[2*j,2*j-1]=mp.mpf(str(t2))*mp.exp(mp.mpf(str(chi))/length)
                                generator[2*j-1,2*j]=-mp.mpf(str(t2))*mp.exp(-mp.mpf(str(chi))/length)
                        flow=mp.expm(mp.mpf(str(duration))*generator)
                        cols=old.base.occupied(length,name)
                        q=mp.qr(mp.matrix([[flow[i,j] for j in cols] for i in range(length)]),mode="skinny")[0]
                        plus=old.spatial.decode(raw["plus"])
                        gain=mp.qr(mp.matrix([[mp.exp(mp.mpf(str(chi))*(mp.mpf(i//2)-mp.mpf(length//2-1)/2)/length)*plus[i,j]
                                              for j in range(length//2)] for i in range(length)]),mode="skinny")[0]
                        residual=q-gain*(gain.T*q)
                        exact=old.previous.mp_opnorm(residual)
                        norm_lower,norm_upper=direct_norm_bounds(residual)
                        assert norm_lower-mp.mpf("1e-70")<=exact<=norm_upper+mp.mpf("1e-70")
                        direct=float(exact)
                    assert abs(direct-response.distance)<2e-9, (length,name,direct,response)
                    step=1e-4
                    times=[]
                    for sign in (-1,1):
                        changed=GaugedGraph(arrays["plus"],arrays["minus"],arrays["mu"],arrays[name+"_log"],arrays[name+"_sign"],chi+sign*step,gamma,t1,t2)
                        times.append(root(changed,.05,duration))
                    fd=(times[1]-times[0])/(2*step)
                    assert abs(fd-response.susceptibility)<1e-6
                    rows.append(dict(length=length,gamma=gamma,t1=t1,t2=t2,initial=name,chi=chi,
                                     spectral_error=str(spectral_error),direct_projector_error=abs(direct-response.distance),
                                     time_response_error=abs(fd-response.susceptibility),
                                     direct_norm_bound_width=float(norm_upper-norm_lower)))
    save(OUT/"kernel_checks.json",dict(rows=rows,status="passed",tests=len(rows),source_sha256=digest(Path(__file__))))
    return dict(status="kernel_passed",tests=len(rows))


def reference(index):
    # Full occupied-column propagation avoids the F solve and graph-distance evaluator.
    cases = [(0,"left",-4.),(4,"right",4.),(5,"left",-4.),(6,"right",4.),(8,"left",4.)]
    family_index,name,chi=cases[index]
    family=config()["families"][family_index]
    target=OUT/"references"/f'{family["key"]}_{name}_{chi:g}.json'
    if target.exists():return load(target)
    progress=target.with_suffix(".progress.json")
    if progress.exists():
        import shutil
        previous=OUT/"reference_attempts"/progress.name
        previous.parent.mkdir(parents=True,exist_ok=True)
        if not previous.exists():shutil.copy2(progress,previous)
    cache=load(OUT/"cache"/(family["key"]+".json"))
    statics=old.read_gzip(ROOT/cache["source"])[-2:]
    zero=load(OUT/"scans"/(family["key"]+".json"))
    chosen=next(r for r in zero["rows"] if r["initial"]==name and r["chi"]==chi and r["level"]==.05)
    results=[]
    for static in statics:
        mp.mp.dps=static["dps"]
        plus,minus=[old.spatial.decode(static[k]) for k in ("plus","minus")]
        mu=[mp.mpf(x) for x in static["mu"]]
        length,n=family["length"],family["length"]//2
        gamma=mp.mpf(str(family["gamma"]))
        cols=old.base.occupied(length,name)
        yp=mp.matrix([[gamma/mu[k]*plus[j,k]*(1 if j%2==0 else -1) for j in cols] for k in range(n)])
        ym=mp.matrix([[gamma/mu[k]*minus[j,k]*(-1 if j%2==0 else 1) for j in cols] for k in range(n)])
        weights=[mp.exp(mp.mpf(str(chi))*(mp.mpf(i//2)-mp.mpf(n-1)/2)/length) for i in range(length)]
        gain=mp.qr(mp.matrix([[weights[i]*plus[i,j] for j in range(n)] for i in range(length)]),mode="skinny")[0]
        for offset in (-config()["direct_reference_offset"],config()["direct_reference_offset"]):
            duration=mp.mpf(str(chosen["root_time"]+offset))
            growing=mp.diag([mp.exp((x-max(mu))*duration) for x in mu])*yp
            decaying=mp.diag([mp.exp((-x-max(mu))*duration) for x in mu])*ym
            frame=plus*growing+minus*decaying
            for i in range(length):
                for j in range(n):frame[i,j]*=weights[i]
            q=mp.qr(frame,mode="skinny")[0]
            residual=q-gain*(gain.T*q)
            lower,upper=direct_norm_bounds(residual)
            distance=(lower+upper)/2
            predicted=graph_for(family,name,chi).distance(float(duration))
            maximum_error=max(abs(lower-mp.mpf(predicted)),abs(upper-mp.mpf(predicted)))
            assert maximum_error<mp.mpf("2e-7")
            assert (lower>mp.mpf(".05")) if offset<0 else (upper<mp.mpf(".05"))
            results.append(dict(dps=static["dps"],offset=offset,time=str(duration),distance=str(distance),
                                float_graph_error=float(maximum_error),norm_lower=str(lower),norm_upper=str(upper),
                                norm_bound_width=str(upper-lower)))
            save(target.with_suffix(".progress.json"),dict(completed=len(results),rows=results))
    value=dict(family=family,initial=name,chi=chi,rows=results,status="passed",
               source_sha256=cache["source_sha256"],method="two-precision complete occupied-column propagation and explicit norm remainder; no F solve or dynamical truncation")
    save(target,value)
    return dict(status="reference_passed",case=target.stem,points=len(results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("initialize","kernel","prepare","baseline","scan","reference"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    try:
        result=globals()[args.stage]() if args.stage in ("initialize","kernel") else globals()[args.stage](args.index)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as error:
        save(OUT/"failures"/f'{args.stage}_{args.index}_{time.time_ns()}.json',dict(stage=args.stage,index=args.index,
             error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":main()
