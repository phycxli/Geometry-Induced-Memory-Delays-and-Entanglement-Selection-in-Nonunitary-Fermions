"""Increase rank only when the complete signed-residual bound requires it."""

import argparse
from pathlib import Path
import time

import run_figure_traces_fast as fast

study,old,mp=fast.study,fast.old,fast.mp
SOURCE=Path(__file__)
study.SOURCE=SOURCE
RANKS=(12,24,32)


def distance_at(ctx,duration):
    try:
        for rank in RANKS:
            distance,remainder=fast.compact_distance(ctx,duration,min(rank,len(ctx["mu"])))
            if remainder<mp.mpf("1e-7"):
                return distance,remainder,f"full_signed_norm_bound_rank_{rank}"
    except (ArithmeticError,ZeroDivisionError):
        pass
    return fast.safe.full_distance(ctx,duration),mp.mpf(0),"dense_graph_chart_fallback"


def benchmark():
    family=study.config()["families"][8]
    ctxs=study.contexts(family)
    rows=[]
    for name in ("left","right","block_cell72"):
        ctx=ctxs[name]
        root=study.theory.bracket_root(ctx,".05",12)[0]
        for rank in RANKS:
            tick=time.perf_counter()
            distance,bound=fast.compact_distance(ctx,root,rank)
            seconds=time.perf_counter()-tick
            row=dict(length=384,initial=name,time=str(root),rank=rank,distance=float(distance),
                     full_signed_bound_mp=str(bound),seconds=seconds)
            rows.append(row)
            old.emit(dict(status="rank_benchmark",**row))
            if bound<mp.mpf("1e-7"):break
    passed=all(any(r["initial"]==name and mp.mpf(r["full_signed_bound_mp"])<mp.mpf("1e-7") for r in rows)
               for name in ("left","right","block_cell72"))
    study.write_json(study.OUT/"checks/large_residual_benchmark.json",dict(status="passed" if passed else "bound_too_conservative",rows=rows,
                     source_sha256=study.theory.sha(SOURCE),residual_source_sha256=study.theory.sha(fast.SOURCE),verified_utc=old.now()))
    assert passed,"Retain the dense signed-residual kernel when the norm product is too conservative"
    return dict(status="large_residual_benchmark_passed",rows=rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("trace","benchmark"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    with old.threadpool_limits(limits=1):
        if args.stage=="benchmark":value=benchmark()
        else:
            assert study.read_json(study.OUT/"checks/factorized_residual_checks.json")["status"]=="passed"
            assert study.read_json(study.OUT/"checks/large_residual_benchmark.json")["status"]=="passed"
            study.distance_at=distance_at
            value=study.trace_job(args.index)
    old.emit(value)
