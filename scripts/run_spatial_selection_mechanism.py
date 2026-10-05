"""Freeze signed static-graph predictions, then validate spatial controls."""

from __future__ import annotations

import argparse
import gzip
import itertools
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

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mpmath as mp
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

import run_finite_time_selection_prediction as previous

audit, p1 = previous.audit, previous.p1
ROOT = previous.ROOT
PRIOR = previous.OUT
OUT = ROOT / "data/prl_p2_spatial_mechanism"
FIG = ROOT / "figures/prl_p2_spatial_mechanism"
SOURCE = Path(__file__)
SOURCE_SHA = audit.sha256(SOURCE)
INITIALS = ("charge_density_wave", "left", "right", "center_block")
DIGITS = (120, 160, 200, 240)
write_json = previous.write_json
emit = previous.emit


def write_csv(name, rows):
    import csv
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def identifier(length, gamma, skin):
    return f"L{length}_gamma{gamma}_g{skin}".replace(".", "p")


def cases():
    plan = json.loads((PRIOR / "next_round_plan.json").read_text(encoding="utf-8"))
    values = {}
    def add(length, skin, duration, role, initials, gamma=2.0):
        steps = round(duration / 0.05)
        key = p1.case_key(length, gamma, skin, steps)
        if key not in values:
            values[key] = dict(case_id=key, length=length, gamma=gamma, skin=skin,
                               physical_time=duration, steps=steps, dt=0.05, t1=0.5, t2=1.0,
                               boundary="open", roles=[], initials=[])
        row = values[key]
        if role not in row["roles"]:
            row["roles"].append(role)
        row["initials"] = [name for name in INITIALS if name in set(row["initials"]) | set(initials)]
    for bracket in plan["batches"][0]["observed_joint_brackets"]:
        times = np.arange(bracket["last_fail"], bracket["first_sample_pass"] + 0.25, 0.5)
        for duration in times:
            add(bracket["length"], bracket["skin"], float(duration), "dense_boundary", INITIALS[:2])
    for skin in (0.0, 0.25):
        for duration in (28.0, 30.0):
            add(48, skin, duration, "window_extension", INITIALS[:2])
    for length, skin, duration in itertools.product((24, 40), (-0.25, 0.0, 0.25), (5.0, 15.0, 25.0)):
        add(length, skin, duration, "orientation", INITIALS)
    for length, skin, duration in itertools.product((56, 64), (0.0, 0.25), (20.0, 25.0, 30.0, 35.0)):
        add(length, skin, duration, "unseen_size", INITIALS[:2])
    for length, skin, duration in itertools.product((24, 40), (0.0, 0.25), (10.0, 25.0, 40.0)):
        add(length, skin, duration, "gain_strength", INITIALS[:2], 1.6)
    return sorted(values.values(), key=lambda r: (r["gamma"], r["length"], r["skin"], r["physical_time"]))


def families():
    values = {(r["length"], r["gamma"], r["skin"]) for r in cases()}
    values.update((length, 2.0, skin) for length in (16, 24, 32, 40, 48) for skin in (-0.25, 0.0, 0.25))
    return sorted(values)


def configuration():
    return dict(study_date="2026-10-04", version=1, cases=cases(), families=[list(family) for family in families()],
                planned_input_sha256=audit.sha256(PRIOR / "next_round_plan.json"),
                precision_digits=list(DIGITS), reference_goals=dict(projector=1e-10, entropy=1e-10, relative_scalar=1e-8),
                numerical_goals=dict(projector=1e-6, entropy=1e-6, orthogonality=1e-10),
                selection_tolerances=dict(projector=0.05, absolute_entropy=0.05),
                memory_tolerances=dict(pair_projector=0.1, pair_absolute_entropy=0.1),
                initials={name: occupied(24, name) for name in INITIALS},
                initial_policy="coordinate products fixed across g,t; center starts at L/4; occupied complete unit cells",
                signed_graph="F(t)=exp(-Lambda*t) F exp(-Lambda*t); Z(t)=Xplus+Xminus F(t); orth(Z(t))",
                distance_interval="max(0,d(Qstatic,G)-eW) <= d(Qactual,W) <= min(1,d(Qstatic,G)+eW)",
                entropy_interval="max(0,abs(Sstatic-SG)-m*h2(min(eW,.5))) to abs(Sstatic-SG)+m*h2(min(eW,.5))",
                direct_memory_lower="abs(d(Qstatic_a,G)-d(Qstatic_b,G))",
                interpretation="exact static spectral propagation plus finite-output perturbation bounds; no reduced universal law or interval arithmetic",
                conditional_extension="gamma1.6 is frozen now; execute after main batches pass prediction/precision checks",
                development="all previous data are development; new-size truth and orientation/dense new truth start after lock")


def prepare():
    for folder in ("code", "static", "predictions", "cases", "failures", "literature"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    config = configuration()
    config_path = OUT / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
    else:
        write_json(config_path, config)
    manifest = OUT / "input_manifest.csv"
    if not manifest.exists():
        inputs = {ROOT / r["path"] for r in audit.read_csv(PRIOR / "input_manifest.csv")}
        inputs.update(p for p in PRIOR.rglob("*") if p.is_file())
        inputs.update(p for p in previous.FIG.rglob("*") if p.is_file())
        inputs.update((ROOT / "scripts/run_finite_time_selection_prediction.py",
                       ROOT / "scripts/analyze_finite_time_selection_prediction.py",
                       ROOT / "research_doc/04_历史研究/06_预测检验与尺寸扩展.md"))
        write_csv("input_manifest.csv", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                                              sha256=audit.sha256(p)) for p in sorted(inputs)])
    for row in audit.read_csv(manifest):
        assert audit.sha256(ROOT / row["path"]) == row["sha256"], f"immutable input changed: {row['path']}"
    (OUT / "code" / (SOURCE_SHA.lower() + ".py")).write_bytes(SOURCE.read_bytes())


def occupied(length, initial):
    n = length // 2
    if initial == "charge_density_wave":
        return list(range(0, length, 2))
    start = {"left": 0, "right": n, "center_block": n // 2}[initial]
    return list(range(start, start + n))


def initial_matrix(length, initial):
    q = mp.matrix(length, length // 2)
    for column, site in enumerate(occupied(length, initial)):
        q[site, column] = 1
    return q


def decode(value):
    return mp.matrix([[mp.mpf(x) for x in row] for row in value])


def right_solve(numerator, denominator):
    result = mp.matrix(numerator.rows, denominator.rows)
    transposed = denominator.T
    for i in range(numerator.rows):
        solution = mp.lu_solve(transposed, mp.matrix([numerator[i, j] for j in range(numerator.cols)]))
        for j in range(result.cols):
            result[i, j] = solution[j]
    return result


def relative(first, second):
    return abs(first - second) / max(abs(first), abs(second)) if first or second else mp.mpf(0)


def static_reference(family, dps):
    length, gamma_value, skin = family
    with mp.workdps(dps):
        n = length // 2
        gamma = mp.mpf(str(gamma_value))
        coupling = mp.matrix(n)
        for j in range(n):
            coupling[j, j] = mp.mpf("0.5")
            if j:
                coupling[j, j - 1] = 1
        u, s, vt = mp.svd(coupling)
        v = vt.T
        for column in range(n):
            if u[0, column] < 0:
                for row in range(n):
                    u[row, column] *= -1
                    v[row, column] *= -1
        mu = [mp.sqrt(gamma**2 - x**2) for x in s]
        xmatrix, ymatrix = mp.matrix(length), mp.matrix(length)
        diagonal = [mp.exp(mp.mpf(str(skin)) * j) for j in range(n)]
        ratios = [s[k] / (gamma + mu[k]) for k in range(n)]
        input_graph = u * mp.diag(ratios) * v.T
        for i in range(n):
            for j in range(n):
                input_graph[i, j] *= diagonal[i] / diagonal[j]
        for mode in range(n):
            a, b = 1 / mp.sqrt(1 + ratios[mode]**2), ratios[mode] / mp.sqrt(1 + ratios[mode]**2)
            for j in range(n):
                xmatrix[2*j, mode], xmatrix[2*j+1, mode] = u[j, mode]*a, -v[j, mode]*b
                xmatrix[2*j, n+mode], xmatrix[2*j+1, n+mode] = u[j, mode]*b, -v[j, mode]*a
                ymatrix[mode, 2*j], ymatrix[mode, 2*j+1] = gamma/mu[mode]*a*u[j, mode], gamma/mu[mode]*b*v[j, mode]
                ymatrix[n+mode, 2*j], ymatrix[n+mode, 2*j+1] = -gamma/mu[mode]*b*u[j, mode], -gamma/mu[mode]*a*v[j, mode]
        for column in range(length):
            norm = mp.sqrt(mp.fsum(abs(diagonal[row//2]*xmatrix[row, column])**2 for row in range(length)))
            for row in range(length):
                xmatrix[row, column] *= diagonal[row//2] / norm
                ymatrix[column, row] *= norm / diagonal[row//2]
        plus, minus = xmatrix[:, :n], xmatrix[:, n:]
        yplus, yminus = ymatrix[:n, :], ymatrix[n:, :]
        full_gain = mp.qr(plus)[0]
        gain, perpendicular = full_gain[:, :n], full_gain[:, n:]
        input_gain = mp.qr(yplus.T, mode="skinny")[0]
        r, cross, transverse = gain.T*plus, gain.T*minus, perpendicular.T*minus
        p = previous.mp_smin(plus) * previous.mp_smin(yplus)
        q = previous.mp_opnorm(minus) * previous.mp_opnorm(yminus)
        rho = previous.mp_opnorm(input_graph)
        states = {}
        for initial in INITIALS:
            q0 = initial_matrix(length, initial)
            aa, bb = yplus*q0, yminus*q0
            f = right_solve(bb, aa)
            alpha = previous.mp_smin(input_gain.T*q0)
            state = dict(alpha=str(alpha), F=p1.encode_matrix(f), F_norm=str(previous.mp_opnorm(f)))
            if initial == "charge_density_wave":
                formula = 1/mp.sqrt(1+rho**2)
                state.update(cdw_formula=str(formula), cdw_formula_relative_error=str(relative(alpha, formula)))
                assert relative(alpha, formula) < mp.mpf("1e-50")
            else:
                cells = sorted({site//2 for site in occupied(length, initial)})
                empty = [j for j in range(n) if j not in cells]
                block = mp.matrix([[input_graph[i, j] for j in cells] for i in empty])
                delta = previous.mp_smin(block)
                lower = min(1, delta)/((1+rho)*mp.sqrt(1+rho**2))
                upper = min(1, delta)
                assert lower <= alpha*(1+mp.mpf("1e-50")) and alpha <= upper*(1+mp.mpf("1e-50"))
                m = len(cells)
                degree = {"left": m-2, "right": m-1, "center_block": (m-2)//2}[initial]
                xmax = s[0]**2
                polynomial = mp.fsum(mp.binomial(2*k, k)/(k+1) * (xmax/(4*gamma**2))**k/(2*gamma) for k in range(degree+1))
                scalar_tail = 1/(gamma+mp.sqrt(gamma**2-xmax))-polynomial
                tail_bound = s[0]*scalar_tail*mp.exp(abs(mp.mpf(str(skin)))*(n-1))
                state.update(cross_block_min=str(delta), overlap_lower=str(lower), overlap_upper=str(upper),
                             rho=str(rho), polynomial_degree=degree, analytic_tail_upper=str(min(1, tail_bound)))
                assert alpha <= tail_bound*(1+mp.mpf("1e-50"))
            states[initial] = state
        check_case = dict(length=length, gamma=gamma_value, skin=skin, t1=0.5, t2=1.0)
        eigen_error = p1.mp_norm(previous.generator(check_case)*xmatrix-xmatrix*mp.diag([*mu,*[-x for x in mu]]))/p1.mp_norm(xmatrix)
        inverse_error = p1.mp_norm(xmatrix*ymatrix-mp.eye(length))
        assert max(eigen_error, inverse_error) < mp.mpf("1e-50")
        return dict(length=length, gamma=gamma_value, skin=skin, dps=dps, mu=[str(x) for x in mu],
                    matrices={key:p1.encode_matrix(value) for key,value in dict(plus=plus,minus=minus,G=gain,Gperp=perpendicular,
                                                                                R=r,Ccross=cross,Bperp=transverse).items()},
                    input_graph=p1.encode_matrix(input_graph), p=str(p),q=str(q),gain_entropy=str(previous.correlation_entropy(gain)[0]),
                    states=states,eigen_residual=str(eigen_error),inverse_residual=str(inverse_error))


def static_change(first, second):
    with mp.workdps(260):
        changes = [relative(mp.mpf(first[k]),mp.mpf(second[k])) for k in ("p","q")]
        absolute = []
        for initial in INITIALS:
            a,b = first["states"][initial],second["states"][initial]
            changes.extend(relative(mp.mpf(a[k]),mp.mpf(b[k])) for k in ("alpha","F_norm"))
            f1,f2 = decode(a["F"]),decode(b["F"])
            changes.append(p1.mp_norm(f1-f2)/p1.mp_norm(f2))
        for key in first["matrices"]:
            a,b = decode(first["matrices"][key]),decode(second["matrices"][key])
            absolute.append(float(p1.mp_norm(a-b)))
        return dict(relative_scalar=float(max(changes)),matrix_absolute=max(absolute),
                    gain_entropy=float(abs(mp.mpf(first["gain_entropy"])-mp.mpf(second["gain_entropy"]))))


def run_static(family):
    path = OUT/"static"/(identifier(*family)+".json")
    if path.exists():
        assert json.loads(path.read_text())["config_sha256"]==audit.sha256(OUT/"config.json")
        return dict(family=identifier(*family),status="reused_static")
    started,tick=datetime.now(timezone.utc).isoformat(),time.perf_counter()
    values=[]
    for dps in DIGITS:
        values.append(static_reference(family,dps))
        if len(values)>1:
            change=static_change(values[-2],values[-1])
            if change["relative_scalar"]<=1e-8 and max(change["matrix_absolute"],change["gain_entropy"])<=1e-10:
                break
    else:
        raise ArithmeticError("static geometry did not converge")
    with gzip.open(path.with_suffix(".raw.json.gz"),"wt",encoding="utf-8") as stream:
        json.dump(values,stream)
    value={**values[-1],"reference_convergence":change,"precision_digits":[v["dps"] for v in values],
           "started_utc":started,"seconds":time.perf_counter()-tick,"script_sha256":SOURCE_SHA,
           "config_sha256":audit.sha256(OUT/"config.json")}
    write_json(path,value)
    return dict(family=identifier(*family),status="converged_static",dps=value["dps"],seconds=value["seconds"])


def graph_prediction(static, initial, duration, derivative=True):
    matrices={k:decode(v) for k,v in static["matrices"].items()}
    mu=[mp.mpf(x) for x in static["mu"]]
    f=decode(static["states"][initial]["F"])
    ft=mp.matrix(f.rows)
    for i in range(f.rows):
        for j in range(f.cols):
            ft[i,j]=f[i,j]*mp.exp(-(mu[i]+mu[j])*duration)
    d=matrices["R"]+matrices["Ccross"]*ft
    kg=right_solve(matrices["Bperp"]*ft,d)
    u,s,vh=mp.svd(kg)
    kappa=s[0]
    dg=kappa/mp.sqrt(1+kappa**2)
    basis=matrices["plus"]+matrices["minus"]*ft
    qstatic=mp.qr(basis,mode="skinny")[0]
    ent,_=previous.correlation_entropy(qstatic)
    z=mp.mpf(static["q"])*mp.exp(-2*min(mu)*duration)
    p=mp.mpf(static["p"])
    ew=min(mp.mpf(1),z/(p-z)) if z<p else mp.mpf(1)
    n=static["length"]//2
    x=min(ew,mp.mpf("0.5"))
    entropy_budget=n*(-x*mp.log(x)-(1-x)*mp.log(1-x)) if x else mp.mpf(0)
    center_error=abs(ent-mp.mpf(static["gain_entropy"]))
    rate=None
    if derivative:
        ftprime=mp.matrix([[-(mu[i]+mu[j])*ft[i,j] for j in range(ft.cols)] for i in range(ft.rows)])
        kgprime=right_solve((matrices["Bperp"]-kg*matrices["Ccross"])*ftprime,d)
        directional=(u[:,0].T*kgprime*vh[0,:].T)[0]
        rate=-directional/kappa
    mode_linear=right_solve(matrices["Bperp"]*ft,matrices["R"])
    linear_norm=previous.mp_opnorm(mode_linear)
    leading=matrices["Bperp"][:,0]*f[0,0]*right_solve(mp.eye(f.rows)[0,:],matrices["R"])
    leading_norm=previous.mp_opnorm(leading)
    values=dict(distance_to_G=dg,kappa_G=kappa,entropy=ent,output_to_G_bound=ew,
                distance_lower=max(0,dg-ew),distance_upper=min(1,dg+ew),
                entropy_error_lower=max(0,center_error-entropy_budget),entropy_error_upper=center_error+entropy_budget,
                entropy_to_G=center_error,output_entropy_budget=entropy_budget,
                linearized_kappa=linear_norm,leading_mode_kappa=leading_norm*mp.exp(-2*mu[0]*duration),
                leading_mode_coefficient=leading_norm,late_rate=2*mu[0],effective_rate=rate,
                largest_singular_gap=(s[0]-s[1])/s[0] if len(s)>1 else mp.mpf(1),
                graph_chart_min=previous.mp_smin(d))
    return values,qstatic


def prediction_change(first,second):
    with mp.workdps(260):
        relative_keys=("kappa_G","linearized_kappa","leading_mode_coefficient")
        absolute_keys=("distance_to_G","entropy","distance_lower","distance_upper","entropy_error_upper")
        return dict(relative_scalar=float(max(relative(first[k],second[k]) for k in relative_keys)),
                    absolute=float(max(abs(first[k]-second[k]) for k in absolute_keys)),
                    rate=float(abs(first["effective_rate"]-second["effective_rate"])))


def run_predictions(family):
    key=identifier(*family)
    path=OUT/"predictions"/(key+".json")
    if path.exists():
        return dict(family=key,status="reused_predictions")
    started,tick=datetime.now(timezone.utc).isoformat(),time.perf_counter()
    with gzip.open(OUT/"static"/(key+".raw.json.gz"),"rt",encoding="utf-8") as stream:
        static_values=json.load(stream)
    low,high=static_values[-2:]
    planned=[r for r in cases() if (r["length"],r["gamma"],r["skin"])==tuple(family)]
    rows,memories,arrays,raw=[],[],{},[]
    for case in planned:
        case_values={}
        for initial in case["initials"]:
            predictions=[]
            for static in (low,high):
                with mp.workdps(static["dps"]):
                    predictions.append(graph_prediction(static,initial,mp.mpf(str(case["physical_time"]))))
            change=prediction_change(predictions[0][0],predictions[1][0])
            assert change["relative_scalar"]<=1e-8 and change["absolute"]<=1e-10 and change["rate"]<=1e-8
            values,qstatic=predictions[-1]
            numeric={k:float(v) if v is not None else None for k,v in values.items()}
            row={**case,"initial":initial,**numeric,"prediction_dps":high["dps"],
                 "prediction_precision_relative":change["relative_scalar"],"prediction_precision_absolute":change["absolute"],
                 "guaranteed_selected":numeric["distance_upper"]<=0.05 and numeric["entropy_error_upper"]<=0.05,
                 "guaranteed_unselected":numeric["distance_lower"]>0.05 or numeric["entropy_error_lower"]>0.05}
            rows.append(row)
            arrays[case["case_id"]+"_"+initial]=audit.mp_to_numpy(qstatic)
            case_values[initial]=row
            with mp.workdps(high["dps"]):
                raw.append(dict(case_id=case["case_id"],initial=initial,values={k:str(v) for k,v in values.items()},
                                Q=p1.encode_matrix(qstatic)))
        for initial in case["initials"][1:]:
            first,second=case_values["charge_density_wave"],case_values[initial]
            lower=abs(first["distance_to_G"]-second["distance_to_G"])
            memories.append({**case,"initial_a":"charge_density_wave","initial_b":initial,
                             "distance_lower":lower,"guaranteed_memory":lower>0.1})
    np.savez_compressed(path.with_suffix(".npz"),**arrays)
    with gzip.open(path.with_suffix(".raw.json.gz"),"wt",encoding="utf-8") as stream:
        json.dump(raw,stream)
    result=dict(family=family,rows=rows,memory_predictions=memories,static_sha256=audit.sha256(OUT/"static"/(key+".json")),
                config_sha256=audit.sha256(OUT/"config.json"),script_sha256=SOURCE_SHA,started_utc=started,
                seconds=time.perf_counter()-tick)
    write_json(path,result)
    return dict(family=key,status="predictions_converged",states=len(rows),seconds=result["seconds"])


def check_lock():
    hashes=json.loads((OUT/"prediction_lock_hash.json").read_text())
    for name,sha in hashes.items():
        assert audit.sha256(OUT/name)==sha


def lock():
    if (OUT/"prediction_lock.json").exists():
        check_lock()
        return
    assert not list((OUT/"cases").glob("*.json"))
    results=[json.loads(p.read_text()) for p in sorted((OUT/"predictions").glob("*.json"))]
    assert len(results)==len(families())
    rows=[row for result in results for row in result["rows"]]
    memories=[row for result in results for row in result["memory_predictions"]]
    write_csv("locked_predictions.csv",rows)
    write_csv("locked_memory_predictions.csv",memories)
    inputs=[p for directory in (OUT/"static",OUT/"predictions") for p in directory.rglob("*") if p.is_file()]
    value=dict(locked_utc=datetime.now(timezone.utc).isoformat(),config_sha256=audit.sha256(OUT/"config.json"),
               script_sha256=SOURCE_SHA,states=len(rows),pairs=len(memories),models=["signed_static_graph_with_output_interval"],
               prediction_inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=audit.sha256(p)) for p in sorted(inputs)],
               formula=configuration(),policy="no predictor or tolerance changes after new truth; fix implementation bugs with preserved evidence only")
    write_json(OUT/"prediction_lock.json",value)
    write_json(OUT/"prediction_lock_hash.json",{name:audit.sha256(OUT/name) for name in
                                               ("prediction_lock.json","locked_predictions.csv","locked_memory_predictions.csv")})
    emit(dict(stage="lock",states=len(rows),pairs=len(memories),locked_utc=value["locked_utc"]))


def double_reference(case):
    from mietf_skin.ssh import ssh_no_click_hamiltonian
    length=case["length"]
    h=ssh_no_click_hamiltonian(length//2,0.5,1.0,case["gamma"],case["skin"],"open")
    matrix=audit.normalize(linalg.expm(-1j*h*case["physical_time"]))
    w,s,_=linalg.svd(matrix,lapack_driver="gesvd")
    arrays=dict(M_double=matrix,W_double=w[:,:length//2],s_double=s)
    for method,dt,count in (("step",0.05,case["steps"]),("half",0.025,case["steps"]*2)):
        propagator=linalg.expm(-1j*h*dt)
        for initial in case["initials"]:
            q=np.eye(length)[:,occupied(length,initial)].astype(complex)
            for _ in range(count):
                q=linalg.qr(propagator@q,mode="economic")[0]
            arrays[f"Q_{initial}_{method}"]=q
    return arrays


def reference(case,dps):
    with mp.workdps(dps):
        matrix=mp.expm(previous.generator(case)*mp.mpf(str(case["physical_time"])))
        scale=p1.mp_norm(matrix)
        matrix/=scale
        w,s,vh=mp.svd(matrix)
        n=case["length"]//2
        top,bottom=w[:,:n],w[:,n:]
        wentropy,_=previous.correlation_entropy(top)
        phase=np.tile([1.0,1.0j],n)
        arrays=dict(M=phase[:,None]*audit.mp_to_numpy(matrix)*phase.conj()[None,:],
                    W=phase[:,None]*audit.mp_to_numpy(top),s=np.array([float(x) for x in s]))
        raw=dict(dps=dps,M=p1.encode_matrix(matrix),W=p1.encode_matrix(top),singular_values=[str(x) for x in s],
                 log_fro_scale=str(mp.log(scale)),W_entropy=str(wentropy),states={})
        rows=[]
        for initial in case["initials"]:
            q0=initial_matrix(case["length"],initial)
            q=mp.qr(matrix*q0,mode="skinny")[0]
            ent,_=previous.correlation_entropy(q)
            distance=previous.mp_opnorm(bottom.T*q)
            alpha=previous.mp_smin(vh[:n,:]*q0)
            ratio=s[n]/s[n-1]
            values=dict(entropy=ent,distance_to_W=distance,alpha=alpha,ratio=ratio)
            density=[float(mp.fsum(abs(q[i,j])**2 for j in range(n))) for i in range(case["length"])]
            rows.append({**case,"initial":initial,"reference_dps":dps,**{k:float(v) for k,v in values.items()},
                         "W_entropy":float(wentropy),"entropy_error":float(abs(ent-wentropy)),
                         "controlled":distance<=mp.mpf("0.05") and abs(ent-wentropy)<=mp.mpf("0.05"),
                         "right_half_particles":sum(density[n:])})
            arrays["Q_"+initial]=phase[:,None]*audit.mp_to_numpy(q)
            raw["states"][initial]={**{k:str(v) for k,v in values.items()},"Q":p1.encode_matrix(q),"density":density}
        return rows,arrays,raw


def reference_change(first,second):
    _,a,ra=first
    rows,b,rb=second
    with mp.workdps(260):
        scalar=max(relative(mp.mpf(ra["states"][initial][k]),mp.mpf(rb["states"][initial][k]))
                   for initial in rb["states"] for k in ("distance_to_W","alpha","ratio"))
        ent=max(abs(mp.mpf(ra["states"][initial]["entropy"])-mp.mpf(rb["states"][initial]["entropy"])) for initial in rb["states"])
    return dict(M=float(linalg.norm(a["M"]-b["M"])),W=audit.projector_error(a["W"],b["W"]),
                Q=max(audit.projector_error(a["Q_"+r["initial"]],b["Q_"+r["initial"]]) for r in rows),
                S=float(ent),scalar_relative=float(scalar))


def run_case(case):
    check_lock()
    path=OUT/"cases"/(case["case_id"]+".json")
    if path.exists():
        assert json.loads(path.read_text())["config_sha256"]==audit.sha256(OUT/"config.json")
        return dict(case_id=case["case_id"],status="reused")
    started,tick=datetime.now(timezone.utc).isoformat(),time.perf_counter()
    prior_path=PRIOR/"cases"/path.name
    if prior_path.exists() and set(case["initials"])<=set(INITIALS[:2]):
        old=json.loads(prior_path.read_text())
        array_source=(ROOT/old["reused_from"]).with_suffix(".npz") if "reused_from" in old else prior_path.with_suffix(".npz")
        value=dict(case=case,rows=[{**r,**case} for r in old["rows"] if r["initial"] in case["initials"]],
                   reused_from=prior_path.relative_to(ROOT).as_posix(),array_source=array_source.relative_to(ROOT).as_posix(),
                   source_receipt_sha256=audit.sha256(prior_path),reference_convergence=old["reference_convergence"],
                   seconds=0,config_sha256=audit.sha256(OUT/"config.json"),script_sha256=SOURCE_SHA)
        write_json(path,value)
        return dict(case_id=case["case_id"],status="prior_reference_reused")
    double=double_reference(case)
    values=[]
    for dps in DIGITS:
        values.append(reference(case,dps))
        change=reference_change(values[-2],values[-1]) if len(values)>1 else None
        with path.with_suffix(".events.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),dps=dps,seconds=time.perf_counter()-tick,convergence=change))+"\n")
        if change is not None and previous.accepted_precision(change):
            break
    else:
        raise ArithmeticError("independent matrix-exponential reference did not converge")
    rows,arrays,raw=values[-1]
    for row in rows:
        initial=row["initial"]
        row.update(reference_status="converged",W_double_projector_error=audit.projector_error(arrays["W"],double["W_double"]),
                   W_double_entropy_error=abs(row["W_entropy"]-audit.entropy(double["W_double"])))
        for method in ("step","half"):
            q=double[f"Q_{initial}_{method}"]
            row[f"double_{method}_projector_error"]=audit.projector_error(arrays["Q_"+initial],q)
            row[f"double_{method}_entropy_error"]=abs(row["entropy"]-audit.entropy(q))
            row[f"double_{method}_orthogonality"]=audit.opnorm(q.conj().T@q-np.eye(q.shape[1]))
        row["actual_double_status"]="reference_validated" if max(row["double_step_projector_error"],row["double_step_entropy_error"])<=1e-6 else "reference_failed"
    np.savez_compressed(path.with_suffix(".npz"),**double,**arrays)
    with gzip.open(path.with_suffix(".raw.json.gz"),"wt",encoding="utf-8") as stream:
        json.dump([v[2] for v in values],stream)
    value=dict(case=case,rows=rows,precision_digits=[v[2]["dps"] for v in values],reference_convergence=change,
               started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-tick,
               config_sha256=audit.sha256(OUT/"config.json"),script_sha256=SOURCE_SHA,
               prediction_lock_sha256=audit.sha256(OUT/"prediction_lock.json"))
    write_json(path,value)
    return dict(case_id=case["case_id"],status="reference_converged",dps=dps,seconds=value["seconds"],
                double_failed=sum(r["actual_double_status"]!="reference_validated" for r in rows))


def worker_init(source_directory):
    sys.path.insert(0,source_directory)
    threadpool_limits(limits=1)


def parallel(function,jobs,workers,source_directory):
    failures=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(source_directory,)) as executor:
        pending={executor.submit(function,job):job for job in jobs}
        for future in as_completed(pending):
            job=pending[future]
            try:
                emit(future.result())
            except Exception as error:
                key=job["case_id"] if isinstance(job,dict) else identifier(*job)
                value=dict(job=key,function=function.__name__,error_type=type(error).__name__,error=str(error),
                           utc=datetime.now(timezone.utc).isoformat(),script_sha256=SOURCE_SHA)
                write_json(OUT/"failures"/(function.__name__+"_"+key+".json"),value)
                failures.append(value)
                emit(value)
    assert not failures, f"{len(failures)} jobs failed; inspect preserved failure receipts"


def calibrate():
    family=(16,2.0,0.0)
    static=static_reference(family,120)
    case=dict(length=16,gamma=2.0,skin=0.0,physical_time=5.0,dt=0.05,steps=100,t1=0.5,t2=1.0,
              initials=list(INITIALS[:2]),case_id=p1.case_key(16,2.0,0.0,100),roles=["development_calibration"],boundary="open")
    independent=reference(case,120)
    errors=[]
    with mp.workdps(120):
        for initial in INITIALS[:2]:
            prediction,qs=graph_prediction(static,initial,mp.mpf(5))
            qr=decode(independent[2]["states"][initial]["Q"])
            error=p1.mp_distance(qs,qr)
            assert error<mp.mpf("1e-50")
            errors.append(float(error))
    write_json(OUT/"calibration.json",dict(status="passed",case=case,static_graph_projector_errors=errors,
                                          script_sha256=SOURCE_SHA,scope="known L16 t5 development point, not a holdout"))
    emit(dict(stage="calibrate",status="passed",max_error=max(errors)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("calibrate","static","lock","validate","extend"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    assert 1<=args.workers<=8
    prepare()
    tick=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="prl_spatial_baseline_") as directory:
        with zipfile.ZipFile(ROOT/"data/prl_p0_precision/baseline_snapshot.zip") as archive:
            for member in archive.namelist():
                if member.startswith("src/"):
                    archive.extract(member,directory)
        source_directory=str(Path(directory)/"src")
        sys.path.insert(0,source_directory)
        with threadpool_limits(limits=1):
            if args.stage=="calibrate":
                calibrate()
            elif args.stage=="static":
                assert (OUT/"calibration.json").exists()
                parallel(run_static,families(),args.workers,source_directory)
            elif args.stage=="lock":
                parallel(run_predictions,families(),args.workers,source_directory)
                lock()
            elif args.stage=="validate":
                check_lock()
                parallel(run_case,[r for r in cases() if r["gamma"]==2.0],args.workers,source_directory)
            else:
                check_lock()
                assert all((OUT/"cases"/(r["case_id"]+".json")).exists() for r in cases() if r["gamma"]==2.0)
                parallel(run_case,[r for r in cases() if r["gamma"]==1.6],args.workers,source_directory)
    info=audit.environment(args.stage)
    info.update(stage=args.stage,script_sha256=SOURCE_SHA,config_sha256=audit.sha256(OUT/"config.json"),
                workers=args.workers,elapsed_seconds=time.perf_counter()-tick,scope="local Windows, no HPC job")
    write_json(OUT/f"environment_{args.stage}.json",info)


if __name__=="__main__":
    main()
