"""Use full columns when early-time compact Gram coordinates are ill conditioned."""

import argparse
from pathlib import Path

import run_figure_traces as trace

study,old,mp=trace.study,trace.study.old,trace.study.mp
SOURCE=Path(__file__)
study.SOURCE=SOURCE
compact_or_dense=study.distance_at


def full_distance(ctx,duration):
    tc=old.time_context(ctx,mp.mpf(str(duration)))
    frame=ctx["matrices"]["plus"]+tc["delta0"]*ctx["matrices"]["R"]
    q=mp.qr(frame,mode="skinny")[0]
    return old.previous.mp_opnorm(ctx["matrices"]["Gperp"].T*q)


def distance_at(ctx,duration):
    try:
        return compact_or_dense(ctx,duration)
    except (ArithmeticError,ZeroDivisionError):
        return full_distance(ctx,duration),mp.mpf(0),"dense_graph_chart_fallback"


study.distance_at=distance_at


def checks():
    rows=[]
    with mp.workdps(260):
        static=study.biased_static(study.spectral_at(128,260),".25")
        ctx=old.context(static,["right"])["right"]
        direct=study.theory.direct_context(128,mp.mpf(".25"))
        for duration in (mp.mpf(0),mp.mpf(".5"),mp.mpf(4)):
            flow=study.theory.structured_exponential(direct,duration)
            q=mp.qr(flow*old.base.q0_matrix(128,"right"),mode="skinny")[0]
            expected=old.previous.mp_opnorm(direct["perpendicular"].T*q)
            actual=full_distance(ctx,duration)
            error=abs(actual-expected)
            assert error<mp.mpf("1e-30"),(duration,error)
            rows.append(dict(length=128,initial="right",time=str(duration),full_column_error_mp=str(error)))
    study.write_json(study.OUT/"checks/early_chart_checks.json",dict(status="passed",rows=rows,
                     source_sha256=study.theory.sha(SOURCE),base_source_sha256=trace.BASE_SOURCE_SHA,verified_utc=old.now()))
    return dict(status="early_chart_checks_passed",checks=len(rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("trace","checks"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    with old.threadpool_limits(limits=1):
        if args.stage=="checks": value=checks()
        else:
            assert study.read_json(study.OUT/"checks/early_chart_checks.json")["status"]=="passed"
            value=study.trace_job(args.index)
    old.emit(value)
