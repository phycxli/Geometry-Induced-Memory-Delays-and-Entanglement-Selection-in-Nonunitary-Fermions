"""Recompute the released frozen memory protocol in a separate output directory."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import run_operational_memory as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=study.ROOT/"reproduction_output/operational_memory")
    args=parser.parse_args()
    original=study.OUT
    destination=args.output.resolve()
    if destination==original.resolve() or destination.is_relative_to(original.resolve()):
        raise ValueError("Reproduction output must be separate from the released inputs.")
    source_lock=study.read_json(original/"protocol_lock.json")
    expected=study.read_json(original/"summary.json")
    assert study.sha(study.SOURCE)==source_lock["source_sha256"]
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(original/"protocol_lock.json",destination/"protocol_lock.json")
    study.OUT=destination
    study.validate()
    actual=study.read_json(destination/"summary.json")
    assert actual["status"]==expected["status"]=="passed"
    assert actual["validations"]==expected["validations"]==768
    differences=[]
    for filename in ("control_validation.csv","heldout_times.csv","finite_auxiliary.csv","time_curves.csv"):
        with (original/filename).open(encoding="utf-8",newline="") as stream:
            reference=list(csv.DictReader(stream))
        with (destination/filename).open(encoding="utf-8",newline="") as stream:
            recomputed=list(csv.DictReader(stream))
        assert len(reference)==len(recomputed)
        maximum=0.
        for a,b in zip(reference,recomputed):
            assert a.keys()==b.keys()
            for key in a:
                try:
                    delta=abs(float(a[key])-float(b[key]))
                    maximum=max(maximum,delta)
                    assert delta<1e-10,(filename,key,a[key],b[key])
                except ValueError:
                    assert a[key]==b[key],(filename,key)
        differences.append(dict(file=filename,rows=len(reference),maximum_absolute_error=maximum))
    radius=actual["conditional_95pct_difference_error"]
    systematic=actual["systematic_budget"]
    report=dict(status="passed",completed_utc=study.now(),comparisons=differences,
                protocol_lock_sha256=study.sha(original/"protocol_lock.json"),source_sha256=study.sha(study.SOURCE),
                nominal=actual["nominal"],statistical_error=radius,systematic_budget=systematic,
                observed_data_inference="max(0,abs(observed_contrast)-systematic-statistical_error-cdw_distance_bound)",
                external_review_status="not performed by this numerical reproduction")
    import audit_operational_memory as render
    render.ROOT=destination
    render.OUT=destination
    render.figures()
    report["figure_output"]=str(destination/"figures/prl_operational_memory/fixed_site_memory.png")
    study.write_json(destination/"verification.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
