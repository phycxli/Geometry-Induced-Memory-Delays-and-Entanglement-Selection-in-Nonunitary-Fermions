"""Frozen before-truth protocol for fixed-bias all-mode holdout checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import finite_bias_closure_mp as native
import run_finite_bias_closure as study

ROOT, OUT = study.ROOT, study.OUT / "fixed_bias"


def freeze():
    target = OUT / "prediction_lock.json"
    if target.exists():
        return study.load(target)
    sources = [Path(native.__file__), Path(native.exterior.__file__), Path(__file__),
               ROOT / "scripts/slurm_finite_bias_closure_fixed.sh"]
    value = dict(locked_utc=datetime.now(timezone.utc).isoformat(), lengths=[896, 1024, 1152, 1280],
                 preparation_precisions=[[1260, 1300], [1430, 1470], [1580, 1620], [1750, 1790]],
                 static_export_digits=240, working_precisions=[120, 160],
                 quadrature_nodes=[128, 192], kernel_ranks=[40, 48],
                 skins=["-.1", "-.05", ".05", ".1"], levels=[".05", ".01"], sides=["left", "right"],
                 sources={path.relative_to(ROOT).as_posix(): study.digest(path) for path in sources},
                 gates=dict(time_precision_pair=1e-8, time_kernel_against_exact_model=1e-7,
                            graph_selfadjoint_and_solve_residual=1e-30, distance_root_residual=1e-24,
                            complete_norm_relative_gap=1e-28, gauged_linear_kernel_series_tail=1e-14),
                 extrapolation=dict(lengths=[2048, 4096], working_precisions=[240, 280],
                                    quadrature_nodes=[128, 192], kernel_ranks=[128, 144],
                                    skins=["-.1", "-.05", ".05", ".1"], levels=[".05", ".01"],
                                    scope="Analytic finite-kernel predictions and convergence diagnostics; no full-model truth at these lengths unless separately prepared."),
                 no_fit=True, previous_failures_unchanged=True,
                 scientific_scope="Finite endpoints and fixed low thresholds; fixed-g theorem proof scope and numerical validation are separate records.",
                 method="Prediction without inverse Gram; independent complete-model reflected Gram inversion with FLINT native LU; two independent static and dynamic precisions.",
                 interval_certified=False)
    study.save(target, value)
    return value


def check_sources(lock):
    for relative, expected in lock["sources"].items():
        if study.digest(ROOT / relative) != expected:
            raise ValueError("Frozen computation source changed: " + relative)


def execute(stage, index):
    lock = study.load(OUT / "prediction_lock.json")
    check_sources(lock)
    skins, levels = lock["skins"], lock["levels"]
    if stage in ("predict", "prepare"):
        size_index, precision_index = divmod(index, 2)
        length = lock["lengths"][size_index]
        if stage == "predict":
            return native.predict(length, lock["working_precisions"][precision_index],
                                  lock["quadrature_nodes"][precision_index], lock["kernel_ranks"][precision_index],
                                  skins, levels, OUT / "prediction" / f"L{length}_pair{precision_index}.json")
        return native.prepare_exact(length, lock["preparation_precisions"][size_index][precision_index],
                                    lock["static_export_digits"], OUT / "raw" / f"L{length}_pair{precision_index}.raw.json.gz")
    if stage == "extrapolate":
        config = lock["extrapolation"]
        size_index, precision_index = divmod(index, 2)
        length = config["lengths"][size_index]
        return native.predict(length, config["working_precisions"][precision_index],
                              config["quadrature_nodes"][precision_index], config["kernel_ranks"][precision_index],
                              config["skins"], config["levels"], OUT / "extrapolation" / f"L{length}_pair{precision_index}.json")
    size_index, rem = divmod(index, 2 * len(skins) * 2)
    side_index, rem = divmod(rem, len(skins) * 2)
    skin_index, precision_index = divmod(rem, 2)
    length, side, skin = lock["lengths"][size_index], lock["sides"][side_index], skins[skin_index]
    prediction = OUT / "prediction" / f"L{length}_pair1.json"
    return native.validate(OUT / "raw" / f"L{length}_pair{precision_index}.raw.json.gz", prediction,
                           lock["working_precisions"][precision_index],
                           OUT / "validation" / f"L{length}_{side}_g{skin}_pair{precision_index}.json", side, skin)


def summarize():
    lock = study.load(OUT / "prediction_lock.json")
    comparisons, extrapolations, failures = [], [], []
    for length in lock["lengths"]:
        prediction = [study.load(OUT / "prediction" / f"L{length}_pair{index}.json") for index in range(2)]
        for low, high in zip(prediction[0]["rows"], prediction[1]["rows"], strict=True):
            identity = {key: high[key] for key in ("length", "initial", "skin", "level")}
            pair = abs(float(low["time"]) - float(high["time"]))
            paths = [OUT / "validation" / f"L{length}_{high['initial']}_g{skin}_pair{index}.json"
                     for index in range(2) for skin in lock["skins"] if float(skin) == float(high["skin"])]
            exact = []
            for path in paths:
                record = study.load(path)
                exact.append(next(row for row in record["rows"] if float(row["level"]) == float(high["level"])))
            actual_pair = abs(float(exact[0]["time"]) - float(exact[1]["time"]))
            error = abs(float(exact[1]["actual_minus_prediction"]))
            max_selfadjoint = max(float(sample["graph_metric_selfadjoint_residual"]) for row in exact for sample in row["samples"])
            max_solve = max(float(sample["graph_solve_residual"]) for row in exact for sample in row["samples"])
            accepted = pair < lock["gates"]["time_precision_pair"] and actual_pair < lock["gates"]["time_precision_pair"] and error < lock["gates"]["time_kernel_against_exact_model"]
            row = dict(**identity, prediction_time=float(high["time"]), actual_time=float(exact[1]["time"]),
                       prediction_pair_change=pair, actual_pair_change=actual_pair, absolute_time_error=error,
                       maximum_selfadjoint_residual=max_selfadjoint, maximum_native_solve_residual=max_solve,
                       temporal_rate=float(high["temporal_rate"]), effective_asymptotic_rate=float(high["effective_asymptotic_rate"]),
                       center_residual=float(high["center_residual"]), gauged_linear_tail=float(high["gauged_linear_tail"]), accepted=accepted)
            comparisons.append(row)
            if not accepted:
                failures.append(row)
    for length in lock["extrapolation"]["lengths"]:
        records = [study.load(OUT / "extrapolation" / f"L{length}_pair{index}.json") for index in range(2)]
        for low, high in zip(records[0]["rows"], records[1]["rows"], strict=True):
            extrapolations.append(dict(length=length, initial=high["initial"], skin=high["skin"], level=high["level"],
                time=float(high["time"]), center_residual=float(high["center_residual"]),
                temporal_rate=float(high["temporal_rate"]), effective_asymptotic_rate=float(high["effective_asymptotic_rate"]),
                prediction_pair_change=abs(float(low["time"]) - float(high["time"])),
                gauged_linear_tail=float(high["gauged_linear_tail"]), full_model_truth=False))
    value = dict(status="passed_frozen_gates" if not failures else "failed_frozen_gates", roots=len(comparisons),
                 comparisons=comparisons, failures=failures, extrapolations=extrapolations,
                 maximum_time_error=max(row["absolute_time_error"] for row in comparisons),
                 prediction_lock_sha256=study.digest(OUT / "prediction_lock.json"), no_fit=True,
                 interval_certified=False, finished_utc=datetime.now(timezone.utc).isoformat())
    study.save(OUT / "summary.json", value)
    return {key: value[key] for key in ("status", "roots", "maximum_time_error")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "predict", "prepare", "validate", "extrapolate", "summarize"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    value = freeze() if args.stage == "freeze" else summarize() if args.stage == "summarize" else execute(args.stage, args.index)
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
