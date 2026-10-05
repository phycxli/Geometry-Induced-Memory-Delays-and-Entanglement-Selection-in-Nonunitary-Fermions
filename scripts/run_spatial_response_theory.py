"""Reflected polynomial response theorem and independently locked new Gram sizes."""

import argparse
import math
import time

import run_theory_upgrade_hpc as study

mp, old, OUT = study.mp, study.old, study.OUT
HERE = study.Path(__file__)
RESPONSE = OUT / "response_extension"


def initialize():
    RESPONSE.mkdir(parents=True,exist_ok=True)
    target = RESPONSE / "prediction_lock.json"
    if target.exists():
        return dict(status="reused")
    assert not list(RESPONSE.glob("L*.json"))
    study.write_json(target,dict(locked_utc=study.now(),source_sha256=study.sha(HERE),lengths=[384,512],initials=["left","right"],
                                gamma=2,skin_values=["-.25","0",".25"],weak_chi=["2","4"],
                                right_ratio_formula="exp(-(L/2-1)*g) times an interval between 1 and (1-exp(-2g)/4)/(3/4)",
                                weak_limit="log(delta(chi/L)/delta(0)) -> s*chi/2; s=+1 left,-1 right",
                                proof_policy="right finite inequality; left weak limit proved by reflected-polynomial coefficient concentration; no sharp absolute exponent"))
    return dict(status="response_predictions_locked")


def checks(index):
    p = sorted(q for q in (OUT/"gram").glob("L*.json") if ".progress." not in q.name)[index]
    raw = study.read_gzip(p.with_suffix(".raw.json.gz"))[-1]
    base = study.read_json(p)
    initial, length = base["rows"][0]["initial"],base["rows"][0]["length"]
    rows=[]
    with mp.workdps(raw["digits"]):
        inverse=old.spatial.decode(raw["inverse"])
        m=inverse.rows
        # Orthogonal polynomials in increasing Chebyshev degree yield checkerboard inverse signs.
        signed=mp.matrix([[(-1)**(i+j)*inverse[i,j] for j in range(m)] for i in range(m)])
        assert all(v>0 for v in signed)
        ratio_min=min(signed[i+1,j]/signed[i,j] for i in range(m-1) for j in range(m))
        if initial=="right":
            assert ratio_min >= 2-mp.mpf("1e-60")
        for row in base["rows"]:
            g=mp.mpf(row["skin"])
            sign=1 if initial=="left" else -1
            j=mp.matrix([[mp.exp(-sign*g*(i+k+1))*inverse[i,k] for k in range(m)] for i in range(m)])
            ev,u=mp.eigsy(j)
            v=u[:,m-1]
            mean=mp.fsum(i*v[i]**2 for i in range(m))
            back=m-1-mean
            derivative=sign*(1+2*mean)
            q=mp.exp(-2*g)/4
            bound=q/(1-q)
            if initial=="right":
                assert back <= bound+mp.mpf("1e-60")
            rows.append(dict(length=length,initial=initial,skin=str(g),extremal_mean_over_L=str(mean/length),
                             back_mean=str(back),right_back_mean_upper=str(bound) if initial=="right" else None,
                             skin_slope=str(derivative),normalized_skin_slope=str(derivative/length)))
    target=RESPONSE/(p.stem+"_development.json")
    study.write_json(target,dict(status="passed",rows=rows,inverse_entry_ratio_min_mp=str(ratio_min),input_sha256=study.sha(p),
                                source_sha256=study.sha(HERE),completed_utc=study.now()))
    return dict(status="response_structure_passed",length=length,initial=initial)


def new_gram(index):
    lock=study.read_json(RESPONSE/"prediction_lock.json")
    assert lock["source_sha256"]==study.sha(HERE)
    length, initial=lock["lengths"][index//2],lock["initials"][index%2]
    target=RESPONSE/f"L{length}_{initial}.json"
    if target.exists():
        return dict(status="reused")
    tick, raws, values=time.perf_counter(),[],[]
    for nodes,digits in ((length//2+32,2*length),(length//2+64,2*length+40)):
        _,matrix=old.base.spatial_gram(length,0,initial,nodes,digits,True)
        with mp.workdps(digits):
            inverse=mp.inverse(matrix)
            m=matrix.rows
            sign=1 if initial=="left" else -1
            result=[]
            for g in (mp.mpf("-.25"),mp.mpf(0),mp.mpf(2)/length,mp.mpf(4)/length,mp.mpf(".25")):
                j=mp.matrix([[inverse[i,k]*mp.exp(-sign*g*(i+k+1)) for k in range(m)] for i in range(m)])
                ev,u=mp.eigsy(j)
                delta=1/ev[m-1]
                v=u[:,m-1]
                mean=mp.fsum(i*v[i]**2 for i in range(m))
                tr=mp.fsum(j[i,i] for i in range(m))
                variance=mp.fsum((i-mean)**2*v[i]**2 for i in range(m))
                result.append(dict(length=length,initial=initial,skin=str(g),log_delta=str(mp.log(delta)),
                                   extremal_mean_over_L=str(mean/length),extremal_variance_over_L2=str(variance/length**2),
                                   normalized_skin_slope=str(sign*(1+2*mean)/length),theta=str(mp.log(delta*tr))))
            zero=next(mp.mpf(r["log_delta"]) for r in result if mp.mpf(r["skin"])==0)
            for row in result:
                g=mp.mpf(row["skin"])
                ratio=mp.mpf(row["log_delta"])-zero
                row["log_ratio_to_zero"]=str(ratio)
                row["weak_asymptotic_log_ratio"]=str(sign*g*length/2)
                if initial=="right":
                    central=-(2*m-1)*g
                    correction=mp.log((1-mp.exp(-2*g)/4)/(mp.mpf(3)/4))
                    lo,hi=central+min(mp.mpf(0),correction),central+max(mp.mpf(0),correction)
                    assert lo-mp.mpf("1e-50")<=ratio<=hi+mp.mpf("1e-50"),(length,g,ratio,lo,hi)
                    row.update(rigorous_log_ratio_lower=str(lo),rigorous_log_ratio_upper=str(hi))
            values.append(result)
            raws.append(dict(nodes=nodes,digits=digits,gram=old.p1.encode_matrix(matrix),inverse=old.p1.encode_matrix(inverse),rows=result))
        study.write_json(target.with_suffix(".progress.json"),dict(nodes=nodes,digits=digits,seconds=time.perf_counter()-tick))
    with mp.workdps(2*length+60):
        change=max(abs(mp.mpf(a["log_delta"])-mp.mpf(b["log_delta"])) for a,b in zip(*values))
        assert change<mp.mpf("1e-8")
    study.write_gzip(target.with_suffix(".raw.json.gz"),raws)
    study.write_json(target,dict(rows=values[-1],precision_change_mp=str(change),source_sha256=study.sha(HERE),
                                prediction_lock_sha256=study.sha(RESPONSE/"prediction_lock.json"),completed_utc=study.now(),seconds=time.perf_counter()-tick))
    return dict(status="new_response_passed",length=length,initial=initial,seconds=time.perf_counter()-tick)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("initialize","checks","gram"))
    p.add_argument("--index",type=int,default=0)
    a=p.parse_args()
    result=initialize() if a.stage=="initialize" else checks(a.index) if a.stage=="checks" else new_gram(a.index)
    study.emit(dict(stage=a.stage,**result))


if __name__=="__main__":
    main()
