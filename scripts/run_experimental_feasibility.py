"""Small-mode passive embedding, heralding cost and missed-click contamination."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import expm_multiply

import run_passive_complex_hpc as complex_model

io=complex_model.io
ROOT=complex_model.ROOT
OUT=ROOT/"data/prl_e_feasibility"
CONFIG,SOURCE=OUT/"config.json",Path(__file__)
read_json,write_json,write_csv=io.read_json,io.write_json,io.write_csv
emit=io.emit
encode,decode=complex_model.encode,complex_model.decode
DIGITS=(64,96)


def initialize():
    for directory in ("families","development","hpc","failures"):
        (OUT/directory).mkdir(parents=True,exist_ok=True)
    if CONFIG.exists():
        return dict(status="reused")
    families=[dict(key=f"{model}_L{length}_g{str(g).replace('.','p')}",model=model,length=length,skin=g,gamma=2.,
        same_sublattice=.2,t1=.5,t2=1.,boundary="open") for model in ("ssh","complex") for length in (8,12,16) for g in (0.,.15)]
    write_json(CONFIG,dict(families=families,times=[".5","1","2","3","4","6","8"],initials=["charge_density_wave","left","right"],
        efficiencies=[".9",".99",".999",".9999"],precision_digits=list(DIGITS),
        scope="feasibility development, not additional blind mechanism validation or experimental data",
        tomography=dict(absolute_entropy_goal=.05,total_failure_probability=.05,observable_range=[-1,1],settings="L^2 conservative independent correlation settings")))
    write_json(OUT/"configuration_lock.json",dict(locked_utc=io.now(),source_sha256=io.audit.sha256(SOURCE),config_sha256=io.audit.sha256(CONFIG),
        complex_model_source_sha256=io.audit.sha256(complex_model.SOURCE)))
    return dict(status="initialized",families=len(families))


def check_source():
    lock=read_json(OUT/"configuration_lock.json")
    assert io.audit.sha256(SOURCE)==lock["source_sha256"] and io.audit.sha256(CONFIG)==lock["config_sha256"]
    assert io.audit.sha256(complex_model.SOURCE)==lock["complex_model_source_sha256"]


def hamiltonian(family):
    return io.audit.mp_hamiltonian(family) if family["model"]=="ssh" else complex_model.hamiltonian(family)


def required_inefficiency(eigenvalues,target=".99"):
    target=mp.mpf(target)
    lo,hi=mp.mpf(-100),mp.mpf(0)
    for _ in range(110):
        middle=(lo+hi)/2
        missed=mp.power(10,middle)
        fraction=mp.fprod(a/(missed+(1-missed)*a) for a in eigenvalues)
        if fraction>=target:
            lo=middle
        else:
            hi=middle
    return (lo+hi)/2


def tomography_budget(length):
    m=length//2
    lo,hi=mp.mpf(0),mp.mpf(".5")
    for _ in range(100):
        eps=(lo+hi)/2
        if complex_model.entropy_budget(eps,m)<=mp.mpf(".05"):
            lo=eps
        else:
            hi=eps
    entry=(lo+hi)/(2*length)
    settings=length*length
    per_setting=mp.ceil(2*mp.log(2*settings/mp.mpf(".05"))/(entry*entry))
    return dict(correlation_operator_tolerance=float((lo+hi)/2),entry_tolerance=float(entry),settings=settings,
        accepted_shots_per_setting=float(per_setting),accepted_shots_total=float(per_setting*settings),
        interpretation="conservative sufficient Hoeffding and entropy-continuity budget; no platform-specific optimality claim")


def at_precision(family,duration,digits):
    with mp.workdps(digits):
        h=hamiltonian(family)
        c,loss,jumps,residual=complex_model.passive_embedding(h)
        assert residual<mp.mpf("1e-40")
        matrix=mp.expm(-mp.j*h*duration)
        physical=mp.exp(-c*duration)*matrix
        w,s,_=mp.svd(matrix)
        n=family["length"]//2
        wt=w[:,:n]
        went=complex_model.entropy(wt)
        states={}
        rows=[]
        for name in read_json(CONFIG)["initials"]:
            initial=complex_model.q0(family["length"],name)
            evolved=mp.qr(matrix*initial,mode="skinny")[0]
            ent=complex_model.entropy(evolved)
            d=complex_model.distance(evolved,wt)
            amplitudes=physical*initial
            gram=amplitudes.H*amplitudes
            eigenvalues=mp.eighe((gram+gram.H)/2,eigvals_only=True)
            assert all(x>0 and x<=1+mp.mpf("1e-40") for x in eigenvalues)
            logp=mp.fsum(mp.log(x) for x in eigenvalues)
            # This determinant includes the global decay discarded in the normalized state.
            _,r=mp.qr(amplitudes,mode="skinny")
            logp_qr=2*mp.fsum(mp.log(abs(r[j,j])) for j in range(n))
            assert abs(logp-logp_qr)<mp.mpf("1e-25")
            missed_rows=[]
            for efficiency in read_json(CONFIG)["efficiencies"]:
                eta=mp.mpf(efficiency)
                factors=[1-eta+eta*a for a in eigenvalues]
                log_observed=mp.fsum(mp.log(x) for x in factors)
                true_fraction=mp.exp(logp-log_observed)
                missed_rows.append(dict(efficiency=efficiency,log_observed_no_click=str(log_observed),true_no_loss_fraction=str(true_fraction)))
            eta=mp.mpf(".99")
            denominator=(1-eta)*mp.eye(n)+eta*gram
            conditional=amplitudes*mp.inverse(denominator)*amplitudes.H
            number=mp.re(mp.fsum(conditional[j,j] for j in range(conditional.rows)))
            purity_defect=complex_model.norm(conditional*conditional-conditional)
            required=required_inefficiency(eigenvalues)
            row=dict(family_key=family["key"],model=family["model"],length=family["length"],skin=family["skin"],initial=name,
                physical_time=float(duration),entropy=float(ent),W_entropy=float(went),entropy_error=float(abs(ent-went)),distance_to_W=float(d),
                log_no_click_success=float(logp),log10_attempts_for_1000=float((mp.log(1000)-logp)/mp.log(10)),
                log10_max_inefficiency_for_99pct_true=float(required),loss_shift=float(c),minimum_loss_eigenvalue=float(mp.eighe(loss,eigvals_only=True)[0]),
                maximum_jump_support=max(sum(x!=0 for x in jump) for jump in jumps),
                particle_number_at_efficiency99=float(number),purity_defect_at_efficiency99=float(purity_defect),
                fraction_true_at_efficiency99=float(mp.mpf(missed_rows[1]["true_no_loss_fraction"])),
                actual_selected=d<=mp.mpf(".05") and abs(ent-went)<=mp.mpf(".05"),
                log10_tomography_attempts=float((mp.log(tomography_budget(family["length"])["accepted_shots_total"])-logp)/mp.log(10)))
            rows.append(row)
            states[name]=dict(Q=encode(evolved),entropy=str(ent),distance_to_W=str(d),log_no_click_success=str(logp),
                physical_survival_eigenvalues=[str(x) for x in eigenvalues],missed_detection=missed_rows,
                conditional_correlation_at_efficiency99=encode(conditional),log10_required_inefficiency=str(required))
        pairs=[]
        for a,b in (("charge_density_wave","left"),("charge_density_wave","right"),("left","right")):
            pair_distance=complex_model.distance(decode(states[a]["Q"]),decode(states[b]["Q"]))
            diff=abs(mp.mpf(states[a]["entropy"])-mp.mpf(states[b]["entropy"]))
            pairs.append(dict(family_key=family["key"],physical_time=float(duration),initial_a=a,initial_b=b,pair_distance=float(pair_distance),absolute_entropy_difference=float(diff)))
        return rows,pairs,dict(dps=digits,W=encode(wt),W_entropy=str(went),states=states,embedding_residual=str(residual))


def experiment_job(index):
    check_source()
    family=read_json(CONFIG)["families"][index]
    target=OUT/"families"/(family["key"]+".json")
    if target.exists():
        return dict(status="reused",family=family["key"])
    rows,pairs,raw=[],[],[]
    tick,started=time.perf_counter(),io.now()
    for t in read_json(CONFIG)["times"]:
        values=[at_precision(family,mp.mpf(t),digits) for digits in DIGITS]
        with mp.workdps(140):
            projector=complex_model.distance(decode(values[0][2]["W"]),decode(values[1][2]["W"]))
            change=mp.mpf(0)
            for name in read_json(CONFIG)["initials"]:
                a,b=values[0][2]["states"][name],values[1][2]["states"][name]
                projector=max(projector,complex_model.distance(decode(a["Q"]),decode(b["Q"])))
                change=max(change,abs(mp.mpf(a["entropy"])-mp.mpf(b["entropy"])),abs(mp.mpf(a["log_no_click_success"])-mp.mpf(b["log_no_click_success"])))
            assert max(projector,change)<mp.mpf("1e-10"),(family,t,projector,change)
        rows.extend(dict(**r,precision_projector_change=float(projector),precision_scalar_change=float(change)) for r in values[-1][0])
        pairs.extend(values[-1][1])
        raw.append(dict(physical_time=t,references=[v[2] for v in values]))
    io.write_gzip(target.with_suffix(".raw.json.gz"),raw)
    write_json(target,dict(family=family,rows=rows,pairs=pairs,tomography_budget=tomography_budget(family["length"]),
        started_utc=started,completed_utc=io.now(),seconds=time.perf_counter()-tick,source_sha256=io.audit.sha256(SOURCE)))
    return dict(status="family_completed",family=family["key"],states=len(rows),seconds=time.perf_counter()-tick)


def annihilation(length,site):
    size=1<<length
    value=np.zeros((size,size),complex)
    for state in range(size):
        if state&(1<<site):
            value[state^(1<<site),state]=(-1)**((state&((1<<site)-1)).bit_count())
    return value


def fock_check():
    length=4
    operators=[annihilation(length,j) for j in range(length)]
    checks=[]
    with mp.workdps(80):
        for model in ("ssh","complex"):
            family=dict(model=model,length=length,skin=.15,gamma=2.,same_sublattice=.2,t1=.5,t2=1.,boundary="open")
            h=hamiltonian(family)
            c,loss,jumps,_=complex_model.passive_embedding(h)
            hphys=np.array(io.audit.mp_to_numpy(h-mp.j*c*mp.eye(length)))
            many=sum(hphys[i,j]*(operators[i].conj().T@operators[j]) for i in range(length) for j in range(length))
            jump_operators=[sum(complex(row[0,j])*operators[j] for j in range(length)) for row in jumps]
            size=1<<length
            identity=sparse.eye(size,format="csr")
            drift=-1j*(sparse.kron(identity,sparse.csr_matrix(many))-sparse.kron(sparse.csr_matrix(many.conj()),identity))
            recycle=sum(sparse.kron(sparse.csr_matrix(j.conj()),sparse.csr_matrix(j)) for j in jump_operators)
            initial=complex_model.q0(length,"left")
            initial_fock=sum(1<<j for j in io.base.occupied(length,"left"))
            rho=np.zeros((size,size),complex)
            rho[initial_fock,initial_fock]=1
            duration=.7
            amplitudes=io.audit.mp_to_numpy(mp.expm(-mp.j*(h-mp.j*c*mp.eye(length))*duration)*initial)
            gram=amplitudes.conj().T@amplitudes
            for eta in (0.,.97,1.):
                evolved=expm_multiply((drift+(1-eta)*recycle)*duration,rho.reshape(-1,order="F")).reshape((size,size),order="F")
                probability=np.trace(evolved).real
                correlation=np.array([[np.trace(evolved@(operators[j].conj().T@operators[i]))/probability for j in range(length)] for i in range(length)])
                denominator=(1-eta)*np.eye(initial.cols)+eta*gram
                predicted=amplitudes@linalg.solve(denominator,amplitudes.conj().T,assume_a="her")
                predicted_probability=np.linalg.det(denominator).real
                error=float(np.linalg.norm(correlation-predicted,2))
                probability_error=abs(probability-predicted_probability)
                assert max(error,probability_error)<1e-10
                checks.append(dict(model=model,efficiency=eta,correlation_error=error,probability_error=probability_error,
                    exact_fock_dimension=size,liouville_dimension=size*size))
    value=dict(status="passed",rows=checks,scope="independent exact Fock CPTP/no-observed-click check of missed-detection formula, not typical-trajectory sampling",verified_utc=io.now())
    write_json(OUT/"development/fock_missed_detection_check.json",value)
    return dict(status="fock_check_passed",checks=len(checks))


def summarize():
    cfg=read_json(CONFIG)
    values=[read_json(OUT/"families"/(f["key"]+".json")) for f in cfg["families"]]
    rows,pairs=[r for v in values for r in v["rows"]],[r for v in values for r in v["pairs"]]
    write_csv(OUT/"states.csv",rows)
    write_csv(OUT/"pairs.csv",pairs)
    summary=dict(families=len(values),states=len(rows),pairs=len(pairs),selected=sum(r["actual_selected"] for r in rows),
        minimum_log_success=min(r["log_no_click_success"] for r in rows),maximum_log_success=max(r["log_no_click_success"] for r in rows),
        max_reference_change=max(max(r["precision_projector_change"],r["precision_scalar_change"]) for r in rows),
        fock_check=read_json(OUT/"development/fock_missed_detection_check.json")["status"],
        scope=cfg["scope"],completed_utc=io.now())
    write_json(OUT/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("initialize","experiment","fock","summarize"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    if args.stage=="initialize":value=initialize()
    elif args.stage=="experiment":value=experiment_job(args.index)
    elif args.stage=="fock":value=fock_check()
    else:value=summarize()
    emit(value)


if __name__=="__main__":
    main()
