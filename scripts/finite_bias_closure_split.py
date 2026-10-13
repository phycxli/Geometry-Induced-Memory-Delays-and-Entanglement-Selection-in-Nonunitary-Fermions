"""Split the unchanged cached predictor into 32 immutable Slurm tasks.

The extrapolation protocol remains in the original prediction lock. Each task
uses the same finite modes, cached metric, and root method for one side/skin.
Assembly checks every block before writing the original 16-row file format.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import mpmath as mp

import finite_bias_closure_accelerated as cached
import finite_bias_closure_mp as native
import run_finite_bias_closure_fixed as frozen
import run_finite_bias_closure as study


SOURCE = Path(__file__).resolve()
SLURM_SOURCE = SOURCE.with_name("slurm_finite_bias_closure_split.sh")
LOCK = frozen.OUT / "split_prediction_lock.json"
BLOCKS = frozen.OUT / "extrapolation_split"
ASSEMBLED = frozen.OUT / "extrapolation_accelerated"
NATIVE_SHA = "b04d5d4e37b9259f7aef28b52f6b8a737428ffb9abc1b469994be0225fbd6694"
CACHE_SHA = "a81f177c40559ed01ec36548b14b60de8598f32a74665790f9fafbe5eeca7433"
METHOD = "all finite modes, arbitrary-precision exterior kernel, exact low-rank graph denominator"
ACCELERATION = "Native cached inverse; per-action residual gates; unchanged analytic kernel and modes"
ROW_FIELDS = {
    "length", "initial", "skin", "level", "time", "tau", "log_coefficient",
    "center_residual", "temporal_rate", "effective_asymptotic_rate", "root_residual",
    "distance", "tangent", "second_tangent", "eigen_imaginary_maximum",
    "metric_solve_residual", "eig_solve_residual", "linear_trace",
    "raw_geometric_tail", "gauged_linear_tail", "denominator_symmetry",
    "omitted_diagonal_raw", "cached_inverse_residual", "samples",
}
BASE_FIELDS = {
    "length", "working_dps", "nodes", "kernel_rank", "method", "source_sha256",
    "functions_sha256", "no_fit", "inverse_gram_used", "interval_certified",
    "acceleration_source_sha256", "acceleration_lock_sha256", "acceleration",
}


def save_new(path, value):
    """Publish a complete file atomically, without replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix="." + path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_original_sources():
    base_path = frozen.OUT / "prediction_lock.json"
    acceleration_path = frozen.OUT / "acceleration_lock.json"
    base, acceleration = study.load(base_path), study.load(acceleration_path)
    frozen.check_sources(base)
    actual = {
        "base_prediction_lock_sha256": study.digest(base_path),
        "base_model_source_sha256": study.digest(native.__file__),
        "acceleration_source_sha256": study.digest(cached.__file__),
    }
    for key, digest in actual.items():
        if acceleration[key] != digest:
            raise ValueError("Original acceleration source or lock changed: " + key)
    if actual["base_model_source_sha256"] != NATIVE_SHA or actual["acceleration_source_sha256"] != CACHE_SHA:
        raise ValueError("Split driver requires the reviewed native and cache sources")
    if not all(acceleration[key] is True for key in ("sources_unchanged", "formulas_unchanged", "gates_unchanged")):
        raise ValueError("Original acceleration scope is incompatible")
    return base, acceleration


def task_specs(base):
    config = base["extrapolation"]
    if (len(config["lengths"]), len(config["working_precisions"]), len(base["sides"]),
            len(config["skins"]), len(config["levels"])) != (2, 2, 2, 4, 2):
        raise ValueError("The frozen extrapolation must contain exactly 32 tasks and two levels")
    if base["sides"] != ["left", "right"]:
        raise ValueError("The native predictor uses left, right row order")
    if len(config["quadrature_nodes"]) != 2 or len(config["kernel_ranks"]) != 2:
        raise ValueError("Frozen precision configurations are incomplete")
    if len(set(config["lengths"])) != 2 or len(set(map(float, config["skins"]))) != 4:
        raise ValueError("Duplicate frozen size or skin")
    tasks = []
    for length in config["lengths"]:
        for pair in range(2):
            for side in base["sides"]:
                for skin in config["skins"]:
                    tasks.append(dict(index=len(tasks), length=length, precision_index=pair,
                                      working_dps=config["working_precisions"][pair],
                                      nodes=config["quadrature_nodes"][pair],
                                      kernel_rank=config["kernel_ranks"][pair], initial=side,
                                      skin=skin, levels=config["levels"],
                                      filename=f"L{length}_{side}_g{skin}_pair{pair}.json"))
    return tasks


def provenance():
    return dict(base_prediction_lock_sha256=study.digest(frozen.OUT / "prediction_lock.json"),
                acceleration_lock_sha256=study.digest(frozen.OUT / "acceleration_lock.json"),
                base_model_source_sha256=study.digest(native.__file__),
                functions_source_sha256=study.digest(native.exterior.__file__),
                acceleration_source_sha256=study.digest(cached.__file__),
                split_source_sha256=study.digest(SOURCE), split_slurm_source_sha256=study.digest(SLURM_SOURCE))


def freeze():
    base, _ = check_original_sources()
    expected = dict(**provenance(), extrapolation=base["extrapolation"], sides=base["sides"],
                    gates=base["gates"], tasks=task_specs(base), task_count=32, roots_per_task=2,
                    formulas_unchanged=True, gates_unchanged=True, original_sources_unchanged=True,
                    root_initialization="Original CachedKernelPredictor.root(level), without a seed override",
                    method="Independent Slurm task per length, precision, side, skin; two original threshold roots",
                    assembly_rows_per_file=16, assembly_files=4, maximum_parallel_tasks_recommended=16,
                    no_fit=True, interval_certified=False)
    if LOCK.exists():
        value = study.load(LOCK)
        if {key: value[key] for key in expected} != expected:
            raise ValueError("Existing split lock disagrees with its frozen sources or configuration")
        return dict(status="split_lock_verified", path=str(LOCK), sha256=study.digest(LOCK), tasks=32)
    value = dict(locked_utc=datetime.now(timezone.utc).isoformat(), **expected)
    save_new(LOCK, value)
    return dict(status="split_lock_frozen", path=str(LOCK), sha256=study.digest(LOCK), tasks=32)


def verified_lock():
    base, _ = check_original_sources()
    lock = study.load(LOCK)
    expected = dict(**provenance(), extrapolation=base["extrapolation"], sides=base["sides"],
                    gates=base["gates"], tasks=task_specs(base), task_count=32, roots_per_task=2)
    if any(lock.get(key) != value for key, value in expected.items()):
        raise ValueError("Split sources, task identities, or frozen parameters changed")
    if not all(lock.get(key) is True for key in
               ("formulas_unchanged", "gates_unchanged", "original_sources_unchanged", "no_fit")):
        raise ValueError("Split lock no longer declares the reviewed unchanged protocol")
    return base, lock


def base_metadata(spec):
    return dict(length=spec["length"], working_dps=spec["working_dps"], nodes=spec["nodes"],
                kernel_rank=spec["kernel_rank"], method=METHOD, source_sha256=study.digest(native.__file__),
                functions_sha256=study.digest(native.exterior.__file__), no_fit=True,
                inverse_gram_used=False, interval_certified=False,
                acceleration_source_sha256=study.digest(cached.__file__),
                acceleration_lock_sha256=study.digest(frozen.OUT / "acceleration_lock.json"),
                acceleration=ACCELERATION)


def check_rows(rows, spec, gates):
    if len(rows) != 2:
        raise ValueError("A split block must contain exactly its two frozen threshold roots")
    with mp.workdps(spec["working_dps"] + 12):
        for row, level in zip(rows, spec["levels"], strict=True):
            if set(row) != ROW_FIELDS:
                raise ValueError("Incomplete or unexpected root fields")
            if row["length"] != spec["length"] or row["initial"] != spec["initial"]:
                raise ValueError("A split root has the wrong length or initial state")
            if mp.mpf(row["skin"]) != mp.mpf(spec["skin"]) or mp.mpf(row["level"]) != mp.mpf(level):
                raise ValueError("A split root has the wrong bias or level")
            for field in ROW_FIELDS - {"initial", "samples"}:
                if not mp.isfinite(mp.mpf(row[field])):
                    raise ArithmeticError("Nonfinite root field: " + field)
            for field in ("metric_solve_residual", "eig_solve_residual", "cached_inverse_residual", "denominator_symmetry"):
                if not 0 <= mp.mpf(row[field]) <= mp.mpf(str(gates["graph_selfadjoint_and_solve_residual"])):
                    raise ArithmeticError("Frozen algebra gate failed: " + field)
            if not 0 <= mp.mpf(row["eigen_imaginary_maximum"]) <= mp.mpf("1e-25"):
                raise ArithmeticError("Original generalized-eigenvalue gate failed")
            if not 0 <= mp.mpf(row["root_residual"]) < mp.mpf(str(gates["distance_root_residual"])):
                raise ArithmeticError("Frozen root gate failed")
            if abs(mp.mpf(row["distance"]) - mp.mpf(level)) >= mp.mpf(str(gates["distance_root_residual"])):
                raise ArithmeticError("Saved distance does not pass the frozen threshold gate")
            if not 0 <= mp.mpf(row["gauged_linear_tail"]) < mp.mpf(str(gates["gauged_linear_kernel_series_tail"])):
                raise ArithmeticError("Frozen gauged linear kernel tail gate failed")
            if not 0 < mp.mpf(row["distance"]) < 1 or mp.mpf(row["tangent"]) <= 0:
                raise ArithmeticError("Invalid positive distance or leading tangent")
            if any(mp.mpf(row[field]) < 0 for field in ("linear_trace", "raw_geometric_tail", "omitted_diagonal_raw")):
                raise ArithmeticError("Negative kernel trace or tail")
            if not row["samples"] or any(set(sample) != {"time", "distance"} for sample in row["samples"]):
                raise ValueError("Root iteration samples are incomplete")
            if any(not mp.isfinite(mp.mpf(sample[field])) for sample in row["samples"] for field in sample):
                raise ArithmeticError("Nonfinite root iteration sample")


def check_block(value, spec, lock):
    expected = dict(**base_metadata(spec), split_source_sha256=lock["split_source_sha256"],
                    split_slurm_source_sha256=lock["split_slurm_source_sha256"],
                    split_prediction_lock_sha256=study.digest(LOCK),
                    base_prediction_lock_sha256=lock["base_prediction_lock_sha256"],
                    split_task=spec, original_frozen_gates=lock["gates"])
    if any(value.get(key) != field for key, field in expected.items()):
        raise ValueError("Split block source, identity, or precision metadata is invalid")
    if set(value) != set(expected) | {"rows", "seconds", "finished_utc", "started_utc"}:
        raise ValueError("Incomplete or unexpected split block metadata")
    if not isinstance(value["seconds"], (int, float)) or value["seconds"] < 0:
        raise ValueError("Invalid split duration")
    if datetime.fromisoformat(value["finished_utc"]) < datetime.fromisoformat(value["started_utc"]):
        raise ValueError("Invalid split timestamps")
    check_rows(value["rows"], spec, lock["gates"])


def compute_rows(spec):
    modes = native.finite_modes(spec["length"], spec["working_dps"])
    graph = cached.CachedKernelPredictor(modes, spec["initial"], spec["skin"], spec["nodes"], spec["kernel_rank"])
    rows = []
    for level in spec["levels"]:
        rows.append(graph.root(level))
        native.event("kernel_root", length=spec["length"], side=spec["initial"],
                     skin=spec["skin"], level=level, time=rows[-1]["time"])
    return rows


def predict(index):
    _, lock = verified_lock()
    if not 0 <= index < len(lock["tasks"]):
        raise ValueError("Task index must lie in 0..31")
    spec = lock["tasks"][index]
    path = BLOCKS / spec["filename"]
    if path.exists():
        check_block(study.load(path), spec, lock)
        return dict(status="existing_split_block_verified_unchanged", path=str(path), sha256=study.digest(path), roots=2)
    started, tick = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    rows = compute_rows(spec)
    value = dict(**base_metadata(spec), rows=rows, seconds=time.perf_counter() - tick,
                 started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                 split_source_sha256=lock["split_source_sha256"],
                 split_slurm_source_sha256=lock["split_slurm_source_sha256"],
                 split_prediction_lock_sha256=study.digest(LOCK),
                 base_prediction_lock_sha256=lock["base_prediction_lock_sha256"],
                 split_task=spec, original_frozen_gates=lock["gates"])
    check_block(value, spec, lock)
    verified_lock()
    save_new(path, value)
    return dict(status="split_kernel_predictions_saved", path=str(path), sha256=study.digest(path),
                task=index, roots=2, seconds=value["seconds"])


def assemble():
    base, lock = verified_lock()
    expected_names = {spec["filename"] for spec in lock["tasks"]}
    actual_names = {path.name for path in BLOCKS.glob("*.json")}
    if actual_names != expected_names:
        raise ValueError(dict(missing=sorted(expected_names - actual_names), unexpected=sorted(actual_names - expected_names)))
    records = {}
    for spec in lock["tasks"]:
        path = BLOCKS / spec["filename"]
        value = study.load(path)
        check_block(value, spec, lock)
        records[spec["index"]] = value
    outputs = []
    for length in base["extrapolation"]["lengths"]:
        pair_values = []
        for pair in range(2):
            specs = [spec for spec in lock["tasks"] if spec["length"] == length and spec["precision_index"] == pair]
            chunks = [records[spec["index"]] for spec in specs]
            rows = [row for chunk in chunks for row in chunk["rows"]]
            if len(rows) != 16 or len({(row["initial"], row["skin"], row["level"]) for row in rows}) != 16:
                raise ValueError("Assembly lost or duplicated a frozen root")
            value = dict(**base_metadata(specs[0]), rows=rows,
                         seconds=sum(chunk["seconds"] for chunk in chunks),
                         finished_utc=max(chunk["finished_utc"] for chunk in chunks),
                         split_source_sha256=lock["split_source_sha256"],
                         split_slurm_source_sha256=lock["split_slurm_source_sha256"],
                         split_prediction_lock_sha256=study.digest(LOCK),
                         base_prediction_lock_sha256=lock["base_prediction_lock_sha256"],
                         original_frozen_gates=lock["gates"],
                         split_blocks=[dict(filename=spec["filename"], task_index=spec["index"],
                                            sha256=study.digest(BLOCKS / spec["filename"])) for spec in specs],
                         parallel_split_wall_span_seconds=(max(datetime.fromisoformat(chunk["finished_utc"]) for chunk in chunks)
                             - min(datetime.fromisoformat(chunk["started_utc"]) for chunk in chunks)).total_seconds(),
                         seconds_scope="Sum of independent block compute durations; includes repeated finite-mode initialization",
                         assembled_utc=datetime.now(timezone.utc).isoformat())
            pair_values.append(value)
            outputs.append((ASSEMBLED / f"L{length}_pair{pair}.json", value))
        with mp.workdps(max(base["extrapolation"]["working_precisions"]) + 12):
            for low, high in zip(pair_values[0]["rows"], pair_values[1]["rows"], strict=True):
                if (low["initial"], low["skin"], low["level"]) != (high["initial"], high["skin"], high["level"]):
                    raise ValueError("Precision pair root identities differ")
                if abs(mp.mpf(low["time"]) - mp.mpf(high["time"])) >= mp.mpf(str(lock["gates"]["time_precision_pair"])):
                    raise ArithmeticError("Original precision pair gate failed")
    for path, value in outputs:
        if path.exists():
            old = study.load(path)
            if any(old.get(key) != value[key] for key in BASE_FIELDS | {"rows"}):
                raise ValueError("Existing original-schema output disagrees; preserving file: " + str(path))
    verified_lock()
    receipt = []
    for path, value in outputs:
        existed = path.exists()
        if not existed:
            save_new(path, value)
        receipt.append(dict(path=str(path), sha256=study.digest(path), roots=16, existing_unchanged=existed))
    return dict(status="split_predictions_assembled", tasks=32, roots=64, files=receipt,
                frozen_predictive_gates_passed=True, complete_model_truth=False)


def regression(output):
    """Compare two representative L512 side/skin points with original calls."""
    base, _ = check_original_sources()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Regression report already exists: " + str(output))
    if frozen.OUT.resolve() in output.resolve().parents:
        raise ValueError("Development regression must write outside frozen scientific outputs")
    tick = time.perf_counter()
    common = dict(length=512, working_dps=base["working_precisions"][0],
                  nodes=base["quadrature_nodes"][0], kernel_rank=base["kernel_ranks"][0], levels=base["levels"])
    skin = base["skins"][0]
    captured = []
    original_class, original_save = native.KernelPredictor, native.save
    native.KernelPredictor = cached.CachedKernelPredictor
    native.save = lambda path, value: captured.append(value)
    before = time.perf_counter()
    try:
        native.predict(common["length"], common["working_dps"], common["nodes"], common["kernel_rank"],
                       [skin], base["levels"], output.with_suffix(".monolithic-unused.json"))
    finally:
        native.KernelPredictor, native.save = original_class, original_save
    monolithic_seconds = time.perf_counter() - before
    if len(captured) != 1:
        raise ValueError("Existing native prediction did not supply one complete result")
    reference = captured[0]
    before = time.perf_counter()
    split_rows = []
    for side in base["sides"]:
        spec = dict(**common, initial=side, skin=skin)
        rows = compute_rows(spec)
        check_rows(rows, spec, base["gates"])
        split_rows.extend(rows)
    split_seconds = time.perf_counter() - before
    if reference["rows"] != split_rows:
        raise ArithmeticError("Splitting changed one or more original L512 root fields")
    comparison = dict(precision_index=0, **common, skins=[skin], roots=len(split_rows),
                      all_root_fields_identical=True, monolithic_seconds=monolithic_seconds,
                      split_serial_seconds=split_seconds,
                      metadata_parameters_identical=all(reference[key] == common[key] for key in
                          ("length", "working_dps", "nodes", "kernel_rank")),
                      predictive_root_algebra_and_tail_gates_passed=True,
                      reference_rows=reference["rows"], split_rows=split_rows)
    value = dict(status="L512_split_regression_passed", comparisons=[comparison],
                 length=512, roots_compared=4, identical_original_root_fields=True,
                 precision_pair_gate="Checked during 32-block assembly, not this single-precision regression",
                 original_frozen_gates=base["gates"],
                 **provenance(), no_fit=True, interval_certified=False, complete_model_truth=False,
                 configuration_scope="L512 development check with original fixed prediction dps120, nodes128, rank40, g=-.1, both sides and both levels",
                 benchmark_scope="Identical cached finite-kernel model and unchanged root calls; both paths run serially locally; no measured Slurm parallel speedup claim",
                 total_seconds=time.perf_counter() - tick, finished_utc=datetime.now(timezone.utc).isoformat())
    save_new(output, value)
    return dict(status=value["status"], path=str(output), sha256=study.digest(output), roots=4,
                identical_original_root_fields=True, total_seconds=value["total_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "predict", "assemble", "regression"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        value = freeze()
    elif args.stage == "predict":
        value = predict(args.index)
    elif args.stage == "assemble":
        value = assemble()
    else:
        if args.output is None:
            parser.error("regression requires --output outside the frozen data folder")
        value = regression(args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
