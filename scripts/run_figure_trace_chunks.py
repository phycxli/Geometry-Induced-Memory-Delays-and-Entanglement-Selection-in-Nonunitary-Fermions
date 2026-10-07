"""Parallel time sampling with immutable chunks and checked trajectory assembly."""

import argparse
from pathlib import Path
import time

import numpy as np
import run_figure_traces_adaptive as adaptive

study,old,mp=adaptive.study,adaptive.old,adaptive.mp
SOURCE=Path(__file__)


def distance_at(ctx,duration):
    try:
        for rank in adaptive.RANKS:
            distance,remainder=adaptive.fast.compact_distance(ctx,duration,rank)
            if remainder<min(mp.mpf("1e-7"),mp.mpf(".001")*distance):
                return distance,remainder,f"relative_signed_norm_bound_rank_{rank}"
    except (ArithmeticError,ZeroDivisionError):
        pass
    return adaptive.fast.safe.full_distance(ctx,duration),mp.mpf(0),"dense_graph_chart_fallback"


def case_and_times(index):
    case=study.trace_cases()[index]
    family=study.config()["families"][8]
    assert case["length"]==384 and case["skin"]==family["skin"]
    roots=study.read_json(study.OUT/"curves"/(family["key"]+".json"))["rows"]
    center=next(r["root_time"] for r in roots if r["initial"]==case["initial"] and r["level"]=="0.05")
    times=sorted({round(float(center+x),10) for x in list(np.linspace(-1.4,1.1,51))+[1.5,2,3,4,6,8,12]})
    return case,family,center,times


def chunk_job(index):
    case_index,chunk=6+index//4,index%4
    case,family,center,times=case_and_times(case_index)
    folder=study.OUT/"trace_chunks"
    folder.mkdir(exist_ok=True)
    path=folder/(case["initial"]+"_"+str(chunk)+".json")
    if path.exists():return dict(status="reused")
    assert study.read_json(study.OUT/"checks/large_residual_benchmark.json")["status"]=="passed"
    tick=time.perf_counter()
    ctxs=[]
    for static in study.read_gzip(study.OUT/"static"/(family["key"]+".raw.json.gz")):
        with mp.workdps(static["dps"]):
            ctxs.append(old.context(static,[case["initial"]])[case["initial"]])
    rows=[]
    for duration in times[chunk::4]:
        values=[]
        for ctx in ctxs:
            with mp.workdps(ctx["static"]["dps"]):
                values.append(distance_at(ctx,duration))
        with mp.workdps(family["digits"][-1]):
            change=abs(values[0][0]-values[1][0])
            assert change<mp.mpf("1e-8")
            rows.append(dict(**case,family_key=family["key"],time=duration,aligned_time=duration-center,center=center,
                         distance_to_G=float(values[-1][0]),distance_to_G_mp=str(values[-1][0]),
                         projector_remainder=float(values[-1][1]),precision_change_mp=str(change),method=values[-1][2]))
        study.write_json(path.with_suffix(".progress.json"),dict(points=len(rows),total=len(times[chunk::4]),seconds=time.perf_counter()-tick))
    study.write_json(path,dict(case=case,chunk=chunk,rows=rows,source_sha256=study.theory.sha(SOURCE),
                     distance_source_sha256=study.theory.sha(adaptive.SOURCE),residual_source_sha256=study.theory.sha(adaptive.fast.SOURCE),
                     completed_utc=old.now(),seconds=time.perf_counter()-tick))
    return dict(status="trace_chunk_complete",initial=case["initial"],chunk=chunk,points=len(rows),seconds=time.perf_counter()-tick)


def merge_job(index):
    case,family,center,times=case_and_times(index)
    path=study.OUT/"traces"/(family["key"]+"_"+case["initial"]+"_window.json")
    assert not path.exists(),"Existing trajectory must be preserved"
    rows=[]
    sources=[]
    for chunk in range(4):
        source=study.OUT/"trace_chunks"/(case["initial"]+"_"+str(chunk)+".json")
        part=study.read_json(source)
        assert part["case"]==case and part["chunk"]==chunk
        assert part["source_sha256"]==study.theory.sha(SOURCE)
        assert part["distance_source_sha256"]==study.theory.sha(adaptive.SOURCE)
        assert part["residual_source_sha256"]==study.theory.sha(adaptive.fast.SOURCE)
        rows.extend(part["rows"])
        sources.append(dict(path=source.relative_to(study.ROOT).as_posix(),sha256=study.theory.sha(source)))
    rows.sort(key=lambda r:r["time"])
    assert [r["time"] for r in rows]==times and len(rows)==58
    assert all(abs(r["center"]-center)<1e-12 for r in rows)
    study.write_json(path,dict(case=case,rows=rows,digits=family["digits"],source_sha256=study.theory.sha(SOURCE),
                    chunks=sources,completed_utc=old.now()))
    return dict(status="trace_chunks_merged",initial=case["initial"],points=len(rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("chunk","merge"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    with old.threadpool_limits(limits=1):
        value=chunk_job(args.index) if args.stage=="chunk" else merge_job(args.index)
    old.emit(value)
