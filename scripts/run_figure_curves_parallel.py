"""Distribute the extended L384 roots over independent occupation geometries."""

import argparse
from pathlib import Path
import time

import run_figure_traces_fast as fast

study,old,mp=fast.study,fast.old,fast.mp
SOURCE=Path(__file__)


def geometry_job(index):
    family=study.config()["families"][6+index//4]
    name=old.initials(family["length"])[1:][index%4]
    folder=study.OUT/"curve_geometries"
    folder.mkdir(exist_ok=True)
    path=folder/(family["key"]+"_"+name+".json")
    if path.exists():return dict(status="reused")
    assert study.read_json(study.OUT/"checks/factorized_residual_checks.json")["status"]=="passed"
    assert study.read_json(study.OUT/"checks/large_residual_benchmark.json")["status"]=="passed"
    tick=time.perf_counter()
    values=[]
    for static in study.read_gzip(study.OUT/"static"/(family["key"]+".raw.json.gz")):
        mp.mp.dps=static["dps"]
        ctx=old.context(static,[name])[name]
        local={}
        for level in study.LEVELS:
            for rank in (12,24,32):
                root=study.theory.bracket_root(ctx,level,rank)[0]
                distance,bound=fast.compact_distance(ctx,root,rank)
                if bound<mp.mpf(".00001"):break
            assert bound<mp.mpf(".00001")
            local[level]=(root,rank,bound)
        values.append(local)
        study.write_json(path.with_suffix(".progress.json"),dict(precisions=len(values),elapsed=time.perf_counter()-tick))
    rows=[]
    for level,(root,rank,bound) in values[-1].items():
        change=abs(root-values[0][level][0])
        assert change<mp.mpf(".000004")
        rows.append(dict(family_key=family["key"],length=family["length"],skin=family["skin"],initial=name,level=level,
                         root_time=float(root),root_time_mp=str(root),rank=rank,precision_change_mp=str(change),
                         projector_remainder_mp=str(bound),remainder_method="full_signed_frobenius_product_bound"))
    study.write_json(path,dict(family=family,initial=name,rows=rows,source_sha256=study.theory.sha(SOURCE),
                    residual_source_sha256=study.theory.sha(fast.SOURCE),prediction_lock_sha256=study.theory.sha(study.OUT/"prediction_lock.json"),
                    completed_utc=old.now(),seconds=time.perf_counter()-tick))
    return dict(status="geometry_roots_complete",family=family["key"],initial=name,seconds=time.perf_counter()-tick)


def merge_job(index):
    family=study.config()["families"][index]
    path=study.OUT/"curves"/(family["key"]+".json")
    assert not path.exists(),"Existing family result must be preserved"
    parts=[]
    sources=[]
    for name in old.initials(family["length"])[1:]:
        source=study.OUT/"curve_geometries"/(family["key"]+"_"+name+".json")
        value=study.read_json(source)
        assert value["family"]==family and value["initial"]==name
        assert value["source_sha256"]==study.theory.sha(SOURCE)
        assert value["residual_source_sha256"]==study.theory.sha(fast.SOURCE)
        assert value["prediction_lock_sha256"]==study.theory.sha(study.OUT/"prediction_lock.json")
        assert {r["level"] for r in value["rows"]}==set(study.LEVELS)
        parts.extend(value["rows"])
        sources.append(dict(path=source.relative_to(study.ROOT).as_posix(),sha256=study.theory.sha(source)))
    assert len(parts)==16
    study.write_json(path,dict(family=family,rows=parts,source_sha256=study.theory.sha(SOURCE),geometry_sources=sources,
                    prediction_lock_sha256=study.theory.sha(study.OUT/"prediction_lock.json"),completed_utc=old.now()))
    return dict(status="geometry_roots_merged",family=family["key"],roots=len(parts))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("geometry","merge"))
    parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args()
    with old.threadpool_limits(limits=1):
        value=geometry_job(args.index) if args.stage=="geometry" else merge_job(args.index)
    old.emit(value)
