"""Bound the full signed residual before forming costly complete state columns."""

import argparse
from pathlib import Path
import time

import run_figure_traces_safe as safe

study,old,mp=safe.study,safe.old,safe.mp
SOURCE=Path(__file__)
study.SOURCE=SOURCE


def compact_distance(ctx,duration,rank):
    duration=mp.mpf(str(duration))
    n=len(ctx["mu"])
    distance,a,v,_=study.theory.compact_graph(ctx,duration,rank,factors=True)
    decay=[mp.exp(-x*duration) for x in ctx["mu"]]
    bt=mp.matrix([[mp.sqrt(ctx["sig"][i])*ctx["vh"][i,j]*decay[j] for j in range(n)] for i in range(rank)])
    weighted=mp.matrix([[ctx["f"][i,j]*decay[i]*decay[j] for j in range(n)] for i in range(n)])
    residual=weighted-a*bt
    if "full_residual_norm_factor" not in ctx:
        ctx["full_residual_norm_factor"]=old.p1.mp_norm(ctx["matrices"]["minus"])*old.p1.mp_norm(ctx["rinv"])
    # Frobenius submultiplicativity bounds all signed residual entries and weights.
    numerator=ctx["full_residual_norm_factor"]*old.p1.mp_norm(residual)
    y=ctx["matrices"]["minus"]*a
    smin,_,_=old.prior.small_core_smin(ctx["matrices"]["G"],y,v)
    return distance,min(mp.mpf(1),numerator/smin)


def distance_at(ctx,duration):
    try:
        for rank in (12,24):
            distance,remainder=compact_distance(ctx,duration,min(rank,len(ctx["mu"])))
            if remainder<mp.mpf("1e-7"):
                return distance,remainder,f"full_signed_norm_bound_rank_{rank}"
    except (ArithmeticError,ZeroDivisionError):
        pass
    return safe.full_distance(ctx,duration),mp.mpf(0),"dense_graph_chart_fallback"


def checks():
    rows=[]
    with mp.workdps(180):
        for skin in ("-.25","0.25"):
            static=study.biased_static(study.spectral_at(32,180),skin)
            contexts=old.context(static,["left","right"])
            direct=study.theory.direct_context(32,mp.mpf(skin))
            for name,ctx in contexts.items():
                for duration in (mp.mpf(8),mp.mpf(20)):
                    actual=compact_distance(ctx,duration,12)
                    tc=old.time_context(ctx,duration)
                    original=study.theory.fine.at_rank(tc,12)[0]
                    full=study.theory.structured_exponential(direct,duration)*old.base.q0_matrix(32,name)
                    q=mp.qr(full,mode="skinny")[0]
                    independent=old.previous.mp_opnorm(direct["perpendicular"].T*q)
                    error=abs(independent-actual[0])
                    assert abs(actual[0]-original["distance_to_G"])<mp.mpf("1e-60")
                    assert error<=actual[1]+mp.mpf("1e-60")
                    assert original["column_residual_fro"]<=actual[1]*original["small_core_smin"]+mp.mpf("1e-60")
                    rows.append(dict(length=32,skin=skin,initial=name,time=str(duration),
                                     distance_error_mp=str(error),full_signed_bound_mp=str(actual[1]),
                                     original_remainder_mp=str(original["projector_remainder"])))
    study.write_json(study.OUT/"checks/factorized_residual_checks.json",dict(status="passed",rows=rows,
                     source_sha256=study.theory.sha(SOURCE),verified_utc=old.now()))
    return dict(status="factorized_residual_checks_passed",checks=len(rows))


def benchmark():
    family=study.config()["families"][8]
    ctxs=study.contexts(family)
    rows=[]
    for name in ("left","right","block_cell72"):
        ctx=ctxs[name]
        root=study.theory.bracket_root(ctx,".05",12)[0]
        tick=time.perf_counter()
        distance,bound=compact_distance(ctx,root,12)
        seconds=time.perf_counter()-tick
        assert bound<mp.mpf("1e-7")
        rows.append(dict(length=384,initial=name,time=str(root),distance=float(distance),
                         full_signed_bound_mp=str(bound),seconds=seconds))
    study.write_json(study.OUT/"checks/large_residual_benchmark.json",dict(status="passed",rows=rows,
                     source_sha256=study.theory.sha(SOURCE),verified_utc=old.now()))
    return dict(status="large_residual_benchmark_passed",rows=rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("trace","checks","benchmark"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    with old.threadpool_limits(limits=1):
        if args.stage=="checks":value=checks()
        elif args.stage=="benchmark":value=benchmark()
        else:
            assert study.read_json(study.OUT/"checks/factorized_residual_checks.json")["status"]=="passed"
            assert study.read_json(study.OUT/"checks/large_residual_benchmark.json")["status"]=="passed"
            study.distance_at=distance_at
            value=study.trace_job(args.index)
    old.emit(value)
