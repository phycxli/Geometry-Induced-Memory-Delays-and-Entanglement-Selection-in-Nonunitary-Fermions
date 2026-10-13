"""Validate unchanged weak-bias predictions with independent FLINT preparation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

import run_finite_bias_closure as study
import run_finite_bias_closure_fixed as fixed
import finite_bias_closure_mp as native

OUT = study.OUT / "weak_native"


def freeze():
    target = OUT / "preparation_lock.json"
    if target.exists():
        return study.load(target)
    value = dict(locked_utc=datetime.now(timezone.utc).isoformat(),
                 original_prediction_lock_sha256=study.digest(study.OUT / "prediction_lock.json"),
                 fixed_preparation_lock_sha256=study.digest(fixed.OUT / "prediction_lock.json"),
                 native_preparation_source_sha256=study.digest(native.__file__),
                 conversion_and_validation_driver_sha256=study.digest(__file__),
                 formulas_and_prediction_gates_unchanged=True,
                 finite_modes_retained=True, independent_of_kernel_prediction=True,
                 provenance="New FLINT full finite reflected-Gram preparation at the same two static precisions; original frozen float64 validator and signed-log conversion.")
    study.save(target, value)
    study.save(OUT / "prediction_lock.json", study.load(study.OUT / "prediction_lock.json"))
    return value


def validate(index):
    lock = study.load(study.OUT / "prediction_lock.json")
    protocol = study.load(OUT / "preparation_lock.json")
    if protocol["conversion_and_validation_driver_sha256"] != study.digest(__file__):
        raise ValueError("The alternate preparation driver changed")
    length = lock["lengths"][index]
    size_index = study.load(fixed.OUT / "prediction_lock.json")["lengths"].index(length)
    inputs = []
    for pair in range(2):
        path = fixed.OUT / "raw" / f"L{length}_pair{pair}.raw.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        if raw["dps"] != lock["precisions"][index][pair]:
            raise ValueError("The native static precision differs from the original weak-bias protocol")
        arrays = study.spectral.prior.convert_spectral(raw)
        target = OUT / "cache" / f"L{length}_dps{raw['dps']}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with np.load(target) as present:
                if not all(np.array_equal(present[key], value) for key, value in arrays.items()):
                    raise ValueError("Existing native cache differs")
        else:
            np.savez_compressed(target, **arrays)
        study.save(target.with_suffix(".json"), dict(length=length, dps=raw["dps"],
                   raw_sha256=study.digest(path), raw_export_digits=raw["export_digits"],
                   sha256=study.digest(target), native_protocol_sha256=study.digest(OUT / "preparation_lock.json")))
        inputs.append(study.digest(path))
    original = study.OUT
    study.OUT = OUT
    try:
        result = study.validate(index)
    finally:
        study.OUT = original
    study.save(OUT / "validation" / f"L{length}.provenance.json",
               dict(raw_inputs=inputs, original_validator_source_sha256=study.digest(study.__file__),
                    native_driver_source_sha256=study.digest(__file__), alternate_prepare_only=True,
                    original_predictions_unchanged=True, original_gates_unchanged=True, result=result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "validate"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(freeze() if args.stage == "freeze" else validate(args.index), indent=2), flush=True)


if __name__ == "__main__":
    main()
