"""Reviewed arithmetic acceleration of the unchanged finite-kernel model.

This additive driver preserves the original frozen source and records a second
source hash. Caching the high-precision metric inverse removes repeated cubic
factorizations. Its inverse and each action are checked by their residuals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from flint import arb_mat
import mpmath as mp

import finite_bias_closure_mp as native
import run_finite_bias_closure_fixed as frozen
import run_finite_bias_closure as study


class CachedKernelPredictor(native.KernelPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        identity = arb_mat(self.n, self.n)
        for i in range(self.n):
            identity[i, i] = 1
        self.inverse_metric, self.inverse_residual = native.native_solve(self.metric, identity)
        if self.inverse_residual > mp.mpf("1e-30"):
            raise ArithmeticError("The cached inverse did not pass the original algebra gate")
        native.event("inverse_cached", length=self.length, side=self.side, skin=str(self.skin),
                     residual=native.decimal(self.inverse_residual))

    def evaluate(self, duration):
        original = native.native_solve

        def cached(matrix, rhs):
            if matrix is self.metric:
                answer = self.inverse_metric * rhs
                residual = native.max_entry(matrix * answer - rhs) / max(mp.mpf(1), native.max_entry(rhs))
                if residual > mp.mpf("1e-30"):
                    raise ArithmeticError("The cached metric action failed the algebra gate")
                return answer, residual
            return original(matrix, rhs)

        native.native_solve = cached
        try:
            result = super().evaluate(duration)
        finally:
            native.native_solve = original
        return dict(**result, cached_inverse_residual=self.inverse_residual)


def freeze():
    target = frozen.OUT / "acceleration_lock.json"
    if target.exists():
        return study.load(target)
    value = dict(locked_utc=datetime.now(timezone.utc).isoformat(),
                 base_prediction_lock_sha256=study.digest(frozen.OUT / "prediction_lock.json"),
                 base_model_source_sha256=study.digest(native.__file__),
                 acceleration_source_sha256=study.digest(__file__),
                 sources_unchanged=True, formulas_unchanged=True, gates_unchanged=True,
                 inverse_residual_gate=1e-30, maximum_time_difference_gate=1e-8,
                 scope="Additive cached-inverse implementation; original source, output and Slurm records are retained.")
    study.save(target, value)
    return value


def predict(index):
    lock = study.load(frozen.OUT / "prediction_lock.json")
    frozen.check_sources(lock)
    second_lock = study.load(frozen.OUT / "acceleration_lock.json")
    if study.digest(__file__) != second_lock["acceleration_source_sha256"]:
        raise ValueError("The acceleration driver changed after freezing")
    config = lock["extrapolation"]
    size_index, precision_index = divmod(index, 2)
    length = config["lengths"][size_index]
    original_class, original_save = native.KernelPredictor, native.save

    def stamped_save(path, value):
        value.update(acceleration_source_sha256=study.digest(__file__),
                     acceleration_lock_sha256=study.digest(frozen.OUT / "acceleration_lock.json"),
                     acceleration="Native cached inverse; per-action residual gates; unchanged analytic kernel and modes")
        original_save(path, value)

    native.KernelPredictor, native.save = CachedKernelPredictor, stamped_save
    try:
        return native.predict(length, config["working_precisions"][precision_index],
                              config["quadrature_nodes"][precision_index], config["kernel_ranks"][precision_index],
                              config["skins"], config["levels"],
                              frozen.OUT / "extrapolation_accelerated" / f"L{length}_pair{precision_index}.json")
    finally:
        native.KernelPredictor, native.save = original_class, original_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "predict"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    value = freeze() if args.stage == "freeze" else predict(args.index)
    print(json.dumps(value, indent=2), flush=True)


if __name__ == "__main__":
    main()
