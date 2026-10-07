"""Trace entry point with an explicit rank capacity and independent branch checks."""

import argparse
import time
from pathlib import Path

import run_figure_strengthening as study

SOURCE = Path(__file__)
BASE_SOURCE_SHA = study.theory.sha(study.SOURCE)
study.SOURCE = SOURCE
study.old.RANKS = (0,1,2,4,6,8,12,16,24,32)


def checks():
    mp,old = study.mp,study.old
    rows=[]
    with mp.workdps(140):
        spectral=study.spectral_at(64,140)
        static=study.biased_static(spectral,".25")
        contexts=old.context(static,["charge_density_wave","left"])
        direct=study.theory.direct_context(64,mp.mpf(".25"))
        for name in contexts:
            for duration in (mp.mpf(".5"),mp.mpf(4),mp.mpf(35)):
                flow=study.theory.structured_exponential(direct,duration)
                q=mp.qr(flow*old.base.q0_matrix(64,name),mode="skinny")[0]
                expected=old.previous.mp_opnorm(direct["perpendicular"].T*q)
                actual,remainder,method=study.distance_at(contexts[name],duration)
                error=abs(actual-expected)
                assert error <= remainder+mp.mpf("1e-40"), (name,duration,error,remainder)
                rows.append(dict(length=64,initial=name,time=str(duration),error_mp=str(error),remainder_mp=str(remainder),method=method))
    assert any(r["method"]=="dense_graph" for r in rows)
    study.write_json(study.OUT/"checks/trace_rank_checks.json",dict(status="passed",rows=rows,
                     source_sha256=study.theory.sha(SOURCE),base_source_sha256=BASE_SOURCE_SHA,verified_utc=old.now()))
    return dict(status="trace_rank_checks_passed",checks=len(rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("trace","checks"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    tick=time.perf_counter()
    with study.old.threadpool_limits(limits=1):
        if args.stage=="checks":
            value=checks()
            study.old.emit(value)
        else:
            assert study.read_json(study.OUT/"checks/trace_rank_checks.json")["status"]=="passed"
            value=study.trace_job(args.index)
            study.old.emit(dict(stage="trace",**value))
            path=study.OUT/"hpc"/f"trace_wrapper_{args.index}.json"
            study.write_json(path,dict(seconds=time.perf_counter()-tick,source_sha256=study.theory.sha(SOURCE),
                                      base_source_sha256=BASE_SOURCE_SHA,completed_utc=study.old.now()))
