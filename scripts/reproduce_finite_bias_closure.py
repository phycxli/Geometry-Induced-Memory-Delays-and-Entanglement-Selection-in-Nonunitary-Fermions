"""Reevaluate the closure theory, validate compact records, and redraw Figs. 3/4.

The default check never recalculates the large matrices. The optional HPC
commands run the original computational modules in a separate scratch folder.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DATA_REL = Path("data/prl_finite_bias_closure_20261013")
PUBLIC_REL = DATA_REL / "public"
THEORY_REL = Path("research_doc/01_当前研究/有限尺寸与固定偏置闭合_20261013/推导与复核/fixed_bias/fixed_bias_predictions_reviewed.json")
STAGE_REL = Path("data/prl_closure_editorial_upgrade_20261013/public_release_stage")
TABLES = ("figure_weak_records.csv", "figure_fixed_records.csv", "figure_fixed_windows.csv")
METRICS = {
    ("left", ".05"): "2630.059994811180",
    ("right", ".05"): "279.6243425247743",
    ("left", ".1"): "358.9010285417291",
    ("right", ".1"): "36.51340035653081",
}
CODE_SEEDS = (
    "finite_bias_closure_mp.py", "finite_bias_closure_functions_mp.py",
    "finite_bias_closure_kernel.py", "finite_bias_closure_weak_native.py",
    "finite_bias_closure_accelerated.py", "finite_bias_closure_split.py",
    "run_finite_bias_closure.py", "run_finite_bias_closure_fixed.py",
    "research_increment_center.py", "plot_prl_closure_upgrade.py",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_table(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_release(project: Path) -> int:
    manifest_path = project / "public_release_manifest.json"
    if not manifest_path.is_file():
        return 0
    rows = load(manifest_path)["files"]
    for row in rows:
        path = (project / row["path"]).resolve()
        if not path.is_relative_to(project.resolve()):
            raise ValueError("Release entry outside the checkout")
        if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Release hash/size mismatch: " + row["path"])
    return len(rows)


def theory_path(project: Path) -> Path:
    public = project / PUBLIC_REL / "theory_reference.json"
    return public if public.is_file() else project / THEORY_REL


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Theory:
    def __init__(self, project: Path, dps: int):
        import mpmath as mp

        self.mp = mp
        mp.mp.dps = dps
        self.center = import_file(project / "scripts/research_increment_center.py", "closure_scalar_center")
        self.mu0 = mp.sqrt(7) / 2
        self.gamma, self.b = mp.mpf(2), mp.mpf(".5")
        self.G0 = self.center.external_potential(mp.mpf(2))
        self.Gprime0 = self.center.external_derivative(mp.mpf(2))
        self.tau = self.G0 / (4 * self.mu0)
        self.beta0 = self.tau * self.b / (2 * self.mu0) + self.Gprime0 / 4
        self.A = 2 * mp.sqrt(2 * self.beta0 / mp.pi) / self.mu0

    @lru_cache(maxsize=None)
    def branch(self, side: str, g_string: str) -> dict:
        mp = self.mp
        g = mp.mpf(g_string)
        s = 1 if side == "left" else -1
        h = s * g
        if not abs(g) <= mp.mpf(".1"):
            raise ValueError("Public scalar evaluator covers |g| <= 0.1 only")
        if h > 0:
            z = 2 * mp.cosh(h)
            mu = mp.sqrt(mp.mpf("2.75") - mp.cosh(h))
            potential = self.center.external_potential(z)
            T = (potential - h) / (4 * mu)
            q = -1 / (4 * mu)
            prime = self.center.external_derivative(z)
            second = mp.diff(self.center.external_derivative, z)
            beta = (mp.sinh(h)**2 * second / 2 + mp.cosh(h) * prime / 4
                    + T * (self.b * mp.cosh(h) / mu
                           + self.b**2 * mp.sinh(h)**2 / mu**3) / 2)
            metric = 4 * self.gamma**2 / ((self.gamma + mu)**2
                                         + (mp.mpf(".5") + mp.exp(-g))**2)
            boundary = 1 - mp.exp(-2 * h)
            if side == "right":
                boundary /= 1 + 2 * mp.exp(-h)
            amplitude = (mp.exp(h - 2 * potential) * self.center.kernel_prefactor(z, side)
                         * metric * (mu / self.gamma)**2 * boundary**2
                         / (2 * mp.sqrt(mp.pi) * mp.sqrt(2 * beta)))
            metric_source = "analytic bulk metric"
        elif h < 0:
            mu = self.mu0
            T = (self.G0 - h) / (4 * mu)
            q = -3 / (2 * mu)
            beta = T * self.b / (2 * mu) + self.Gprime0 / 4
            key = ".05" if abs(g) == mp.mpf(".05") else ".1" if abs(g) == mp.mpf(".1") else None
            if key is None:
                raise ValueError("Decreasing constants use published unfitted V at |g|=0.05 or 0.1")
            metric = mp.mpf(METRICS[side, key])
            amplitude = (mp.exp(h - 2 * self.G0) * self.center.kernel_prefactor(mp.mpf(2), side)
                         * (mu / self.gamma)**2 * metric / (4 * mp.pi * beta**3))
            metric_source = "published 16-significant-digit independent half-axis V reference"
        else:
            mu, T, q, beta = self.mu0, self.tau, -3 / (4 * self.mu0), self.beta0
            boundary = mp.mpf(1) if side == "left" else mp.mpf(1) / 9
            amplitude = (mp.exp(-2 * self.G0) * self.center.kernel_prefactor(mp.mpf(2), side)
                         * 2 * mu**2 / (self.gamma * (self.gamma + mu))
                         * boundary / (2 * mp.sqrt(mp.pi) * (2 * beta)**mp.mpf("1.5")))
            metric_source = "analytic zero-bias metric"
        return dict(T=T, q=q, mu=mu, amplitude=amplitude, metric_source=metric_source)

    def coefficients(self, side: str, g: str, epsilon: str) -> dict:
        mp = self.mp
        branch = self.branch(side, str(float(g)))
        eps = mp.mpf(epsilon)
        if not 0 < eps <= mp.mpf(".3") or (mp.mpf(g) != 0 and eps > mp.mpf(".05")):
            raise ValueError("Outside the declared fixed low-threshold range")
        k = eps / mp.sqrt(1 - eps**2)
        return dict(**branch, c=mp.log(branch["amplitude"] / k) / (2 * branch["mu"]))

    def window(self, side: str, g: str, low: str, high: str):
        mp = self.mp
        mu = self.branch(side, str(float(g)))["mu"]
        lo, hi = mp.mpf(low), mp.mpf(high)
        return mp.log((hi / mp.sqrt(1 - hi**2)) / (lo / mp.sqrt(1 - lo**2))) / (2 * mu)


def check(project: Path, data: Path, dps: int, tolerance: float) -> dict:
    release_files = verify_release(project)
    tables = {name: read_table(data / "comparisons" / name) for name in TABLES}
    weak, fixed, windows = (tables[name] for name in TABLES)
    theory = Theory(project, dps)
    maxima: dict[str, float] = {}
    checks = 0

    def compare(name: str, computed, reference) -> None:
        nonlocal checks
        delta = abs(float(computed) - float(reference))
        maxima[name] = max(maxima.get(name, 0.0), delta)
        checks += 1
        if not math.isfinite(delta) or delta > tolerance:
            raise ValueError(f"{name} mismatch: {delta:.8g} exceeds {tolerance:g}")

    for rows, expected, fields in ((weak, 72, ("length", "initial", "chi", "level")),
                                   (fixed, 96, ("length", "initial", "skin", "level")),
                                   (windows, 48, ("length", "initial", "skin"))):
        if len(rows) != expected or len({tuple(row[key] for key in fields) for row in rows}) != expected:
            raise ValueError("Compact table coverage or uniqueness mismatch")
        if any((row.get("direct_numerical_target") or row.get("finite_kernel_target")
                or row.get("numerical_window_target")) != "G_invar" for row in rows):
            raise ValueError("Numerical target must remain G_invar")
    new_weak = sum(int(row["length"]) in (1152, 1280) for row in weak)
    original_fixed = sum(row["evidence_type"] == "original_model" for row in fixed)
    predictions = sum(row["evidence_type"] == "finite_kernel_prediction_without_original_model_truth" for row in fixed)
    if (new_weak, original_fixed, predictions) != (36, 64, 32):
        raise ValueError("Holdout/original-model/prediction evidence counts disagree")
    refs = load(theory_path(project))["records"]
    compared_refs = 0
    for row in refs:
        if abs(float(row["g"])) not in (0.0, 0.05, 0.1):
            continue
        cc = theory.coefficients(row["initial"], row["g"], row["epsilon"])
        for field, key in (("linear_time_coefficient", "T"), ("log_time_coefficient", "q"),
                           ("absolute_center", "c"), ("effective_decay_mu", "mu")):
            compare(field, cc[key], row[field])
        compared_refs += 1
    for row in fixed:
        cc = theory.coefficients(row["initial"], row["skin"], row["level"])
        center = cc["T"] * int(row["length"]) + cc["q"] * theory.mp.log(int(row["length"])) + cc["c"]
        compare("fixed_asymptotic_time", center, row["asymptotic_time"])
        compare("fixed_rate_limit", 2 * cc["mu"], row["asymptotic_temporal_rate"])
        if row["temporal_rate_evidence_type"] != "finite_kernel_derivative":
            raise ValueError("Saved numerical temporal rates must retain finite-kernel evidence labels")
        compare("fixed_kernel_center_residual", float(row["kernel_time"]) - float(center), row["kernel_asymptotic_residual"])
        compare("fixed_rate_residual", float(row["finite_kernel_temporal_rate"]) - 2 * float(cc["mu"]), row["temporal_rate_asymptotic_error"])
        if row["native_time"]:
            compare("fixed_native_center_residual", float(row["native_time"]) - float(center), row["native_asymptotic_residual"])
    for row in windows:
        width = theory.window(row["initial"], row["skin"], row["lower_threshold"], row["upper_threshold"])
        compare("fixed_window_limit", width, row["asymptotic_window"])
        compare("fixed_kernel_window_residual", float(row["kernel_window"]) - float(width), row["kernel_window_residual"])
        if row["native_window"]:
            compare("fixed_native_window_residual", float(row["native_window"]) - float(width), row["native_window_residual"])
    for row in weak:
        s, length, chi = (1 if row["initial"] == "left" else -1), int(row["length"]), float(row["chi"])
        cc = theory.coefficients(row["initial"], "0", row["level"])
        base = cc["T"] * length + cc["q"] * theory.mp.log(length) + cc["c"]
        original = float(base) - s * chi / (4 * float(theory.mu0))
        corrected = original + s * float(theory.A) * chi / math.sqrt(length)
        response = -s / (4 * float(theory.mu0)) + s * float(theory.A) / math.sqrt(length)
        compare("weak_uncorrected_time_error", float(row["actual_time"]) - original, row["uncorrected_time_error"])
        compare("weak_corrected_time_error", float(row["actual_time"]) - corrected, row["corrected_time_error"])
        compare("weak_corrected_response_error", float(row["actual_response"]) - response, row["corrected_response_error"])
        compare("weak_scaled_response", math.sqrt(length) * (s * float(row["actual_response"]) + 1 / (4 * float(theory.mu0))), row["scaled_response"])
    compare("weak_A", theory.A, ".319673435477154418")
    return dict(status="pass", release_files_verified=release_files,
                scalar_and_saved_record_checks=checks, tolerance=tolerance,
                maxima=maxima, theory_reference_records=compared_refs,
                unique_new_original_model_conditions=new_weak + original_fixed,
                retrospective_weak_conditions=72 - new_weak, large_size_prediction_conditions=predictions,
                weak_A=str(theory.A), beta=str(theory.beta0),
                input_sha256={str((data / "comparisons" / name).relative_to(project)): digest(data / "comparisons" / name) for name in TABLES},
                full_matrix_calculations_rerun=False, finite_kernel_roots_rerun=False,
                numerical_original_model_derivatives_rerun=False,
                decreasing_c_uses_published_unfitted_V_reference=True, interval_certified=False)


def plot(project: Path, data: Path, output: Path) -> None:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, str(project / "scripts/plot_prl_closure_upgrade.py"),
                    "--project", str(project), "--data-dir", str(data),
                    "--theory-file", str(theory_path(project)), "--output-dir", str(output),
                    "--no-install"], check=True, env=environment)


def code_dependencies(project: Path) -> list[Path]:
    pending = [project / "scripts" / name for name in CODE_SEEDS]
    found: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            modules = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for module in modules:
                candidate = project / "scripts" / (module.split(".")[0] + ".py")
                if candidate.is_file():
                    pending.append(candidate)
    found.update((project / "src/mietf_skin").glob("*.py"))
    return sorted(found)


def bundle(project: Path, stage: Path) -> dict:
    if stage.resolve() != (project / STAGE_REL).resolve():
        raise ValueError("Bundle writes only the designated public_release_stage folder")
    stage.mkdir(parents=True, exist_ok=True)
    sources: dict[str, str] = {}

    def copy(source: Path, relative: Path) -> None:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources[source.relative_to(project).as_posix()] = digest(source)

    for path in code_dependencies(project):
        copy(path, path.relative_to(project))
    copy(Path(__file__).resolve(), Path("scripts/reproduce_finite_bias_closure.py"))
    for name in TABLES:
        copy(project / DATA_REL / "comparisons" / name, DATA_REL / "comparisons" / name)
    metric_source = project / "research_doc/01_当前研究/有限尺寸与固定偏置闭合_20261013/推导与复核/finite_size/metric_check/check_endpoint_metric_mp.py"
    copy(metric_source, Path("scripts/check_endpoint_metric_mp.py"))
    original_theory = load(project / THEORY_REL)
    fields = ("initial", "g", "signed_half_axis_bias", "epsilon", "branch",
              "linear_time_coefficient", "log_time_coefficient", "effective_decay_mu",
              "absolute_center", "metrics")
    references = [{key: row[key] for key in fields} for row in original_theory["records"]
                  if abs(float(row["g"])) in (0.0, 0.05, 0.1)]
    write(stage / PUBLIC_REL / "theory_reference.json",
          dict(parameters={"gamma": 2, "t1": .5, "t2": 1}, records=references,
               source_sha256=digest(project / THEORY_REL), no_fit=True,
               independent_half_axis_V=[dict(initial=s, magnitude=h, value=v) for (s, h), v in METRICS.items()],
               metric_scope="V values are unfitted independently computed numerical references, not rigorous 16-digit interval certificates"))
    sources[THEORY_REL.as_posix()] = digest(project / THEORY_REL)
    weak = load(project / DATA_REL / "prediction_lock.json")
    fixed = load(project / DATA_REL / "fixed_bias/prediction_lock.json")
    weak_fields = ("lengths", "precisions", "chi_values", "levels", "theory", "correction_coefficient", "prediction", "gates", "no_fit")
    fixed_fields = ("lengths", "preparation_precisions", "static_export_digits", "working_precisions", "quadrature_nodes", "kernel_ranks", "skins", "levels", "sides", "gates", "extrapolation", "no_fit", "interval_certified")
    write(stage / PUBLIC_REL / "computation_parameters.json", dict(
        weak={key: weak[key] for key in weak_fields}, fixed={key: fixed[key] for key in fixed_fields},
        parameter_source_sha256={"weak": digest(project / DATA_REL / "prediction_lock.json"),
                                 "fixed": digest(project / DATA_REL / "fixed_bias/prediction_lock.json")},
        scope="Public reruns create new provenance locks and output files; original scientific parameters and gates are unchanged"))
    write(stage / PUBLIC_REL / "release_scope.json", dict(
        target="G_invar", direct_Pout_numerical_roots=0,
        weak_table_conditions=72, weak_retrospective_conditions=36, weak_new_holdout_conditions=36,
        fixed_original_model_conditions=64, fixed_precision_records=128,
        large_size_prediction_conditions=32, large_size_precision_records=64,
        large_sizes=[2048, 4096], large_size_original_model_truth=False,
        numerical_temporal_rates="centered finite-kernel differences; no independent original-model derivative",
        fixed_g_scope="fixed 0<|g|<=0.1, same-sided compact sets separated from zero; g=0 separate",
        fixed_threshold_scope="[epsilon_min,0.05], epsilon_min>0",
        weak_scope="chi=gL in a fixed compact set, thresholds [epsilon_min,0.3]",
        sqrt_scope="g=xi/sqrt(L), xi in a fixed compact set, thresholds [epsilon_min,0.05]",
        fidelity_scope="Analytic corollary for pure conditional Slater states at fixed nonzero bias; no new determinant numerical test",
        old_failure_records_overwritten=False, complete_interval_certification=False,
        hpc_private_logs_cache_credentials_and_internal_editorial_records_included=False,
        scalar_check_is_not_full_HPC_recomputation=True))
    requirements = "numpy\nscipy\nmpmath\ngmpy2\npython-flint\nmatplotlib\n"
    (stage / "requirements-finite-bias-closure.txt").write_text(requirements, encoding="utf-8")
    (stage / "FINITE_BIAS_CLOSURE_REPRODUCIBILITY.md").write_text(REPRODUCTION_GUIDE, encoding="utf-8")
    for name in ("slurm_finite_bias_closure_split.sh", "slurm_finite_bias_closure_fixed.sh"):
        target = stage / "scripts" / name
        target.write_text(SLURM_WRAPPER, encoding="utf-8", newline="\n")
    files = [dict(path=path.relative_to(stage).as_posix(), bytes=path.stat().st_size, sha256=digest(path))
             for path in sorted(stage.rglob("*")) if path.is_file()
             and path.name != "public_release_manifest.json" and "qa" not in path.relative_to(stage).parts]
    manifest = dict(status="public_release_staged", files=files, original_source_sha256=sources,
                    total_bytes=sum(row["bytes"] for row in files),
                    original_computation_code_byte_identical=True,
                    portable_slurm_wrappers_are_new=True,
                    historical_failure_data_not_modified=True, git_operations_performed=False)
    write(stage / "public_release_manifest.json", manifest)
    return dict(status=manifest["status"], files=len(files), bytes=manifest["total_bytes"], stage=str(stage))


def hpc_modules(project: Path, output: Path):
    if output.resolve().is_relative_to(project.resolve()):
        raise ValueError("Full HPC rerun output must be outside the published checkout")
    sys.path.insert(0, str(project / "scripts"))
    study = importlib.import_module("run_finite_bias_closure")
    fixed = importlib.import_module("run_finite_bias_closure_fixed")
    native = importlib.import_module("finite_bias_closure_mp")
    weak = importlib.import_module("finite_bias_closure_weak_native")
    cached = importlib.import_module("finite_bias_closure_accelerated")
    split = importlib.import_module("finite_bias_closure_split")
    study.OUT, fixed.OUT = output, output / "fixed_bias"
    native.OUT, weak.OUT = fixed.OUT, output / "weak_native"
    split.LOCK = fixed.OUT / "split_prediction_lock.json"
    split.BLOCKS = fixed.OUT / "extrapolation_split"
    split.ASSEMBLED = fixed.OUT / "extrapolation_accelerated"
    return study, fixed, native, weak, cached, split


def hpc_init(project: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty scratch output directory")
    verify_release(project)
    study, fixed, native, weak, cached, split = hpc_modules(project, output)
    parameters = load(project / PUBLIC_REL / "computation_parameters.json")
    stamp = datetime.now(timezone.utc).isoformat()
    wlock = dict(parameters["weak"], locked_utc=stamp,
                 source_sha256=digest(project / "scripts/run_finite_bias_closure.py"),
                 public_rerun=True)
    source_paths = [Path(native.__file__), Path(native.exterior.__file__), Path(fixed.__file__),
                    project / "scripts/slurm_finite_bias_closure_fixed.sh"]
    flock = dict(parameters["fixed"], locked_utc=stamp, public_rerun=True,
                 sources={path.relative_to(project).as_posix(): digest(path) for path in source_paths})
    study.save(output / "prediction_lock.json", wlock)
    study.save(fixed.OUT / "prediction_lock.json", flock)
    weak.freeze()
    cached.freeze()
    split.freeze()
    return dict(status="public_rerun_initialized", output=str(output), scientific_parameters_unchanged=True,
                large_matrix_tasks_run=0, original_delivery_modified=False)


def hpc_run(project: Path, output: Path, stage: str, index: int) -> dict:
    study, fixed, native, weak, cached, split = hpc_modules(project, output)
    if stage == "fixed-predict":
        return fixed.execute("predict", index)
    if stage == "prepare":
        return fixed.execute("prepare", index)
    if stage == "fixed-validate":
        return fixed.execute("validate", index)
    if stage == "weak-validate":
        return weak.validate(index)
    if stage == "split-predict":
        return split.predict(index)
    if stage == "assemble":
        return split.assemble()
    raise ValueError(stage)


SLURM_WRAPPER = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
: "${REPRO_PROJECT:?absolute checkout path required}"
: "${REPRO_OUTPUT_DIR:?new scratch output path outside checkout required}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPRO_PROJECT"
.venv/bin/python scripts/reproduce_finite_bias_closure.py hpc-run \\
  --output-dir "$REPRO_OUTPUT_DIR" --hpc-stage "${1:?stage required}" \\
  --index "${SLURM_ARRAY_TASK_ID:?array index required}"
"""

REPRODUCTION_GUIDE = """# Finite-size and fixed-bias closure: public reproduction

## 1. What the small reproduction does

The compact CSV files preserve 72 weak-bias conditions (36 retrospective,
36 new holdout), 64 original-model fixed-bias conditions, and 32 finite-kernel
prediction conditions at L=2048/4096. The two precision records per fixed or
large-size condition do not count as two independent physical conditions.
All numerical roots target the growing invariant subspace G_invar. Transfer
to the output singular subspace is analytical; these are not direct Pout SVD
roots. Numerical temporal rates are centered finite-kernel differences.

The default check reevaluates the integrals defining G, the kernel prefactor,
T, q, c, the fixed-threshold window, and the weak response coefficient A. It
compares every compact table row, rather than checking a saved pass flag.
The decreasing-branch c uses four unfitted, independently calculated half-axis
V references at the recommended 16 significant digits. The working precision
of scalar quadrature is not the accuracy of those reference values.
The script does not recompute V, finite-kernel roots, large inverse Grams,
original-model derivatives, or a many-body determinant by default.

## 2. Install and verify

Python 3.11 or later is required. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-finite-bias-closure.txt
python scripts/reproduce_finite_bias_closure.py check
python scripts/reproduce_finite_bias_closure.py plot --output-dir reproduced_figures
```

When public_release_manifest.json is present, the check first verifies every
listed file's byte size and SHA256. It returns status=pass with 100 unique new original-model conditions,
32 large-size prediction conditions, and scalar/record comparison maxima.
It does not apply a new scientific acceptance gate or certify total intervals.
The plot writes Fig. 3 and Fig. 4 as PDF/SVG/PNG plus a figure manifest to the
named output directory. It uses --no-install and leaves manuscript figures
and all source tables unchanged. Filled points are original-model roots;
open points at L2048/4096 are finite-kernel predictions. Packet curves are
schematic, not sampled particle densities. The weak-response ranges span
deterministic parameter conditions, not statistical confidence intervals.

## 3. Interpretation boundaries

Weak response: chi=gL in a fixed bounded compact set, fixed thresholds in
[epsilon_min,0.3]. Fixed-bias closure: fixed 0<|g|<=0.1 on same-sided compact
sets separated from zero, fixed thresholds in [epsilon_min,0.05]. Zero bias
is a separate branch. The g=xi/sqrt(L) crossover uses fixed compact xi and
thresholds in [epsilon_min,0.05]; it cannot be obtained by substituting a
growing chi into the weak theorem. In all cases epsilon_min>0.

The sharp expansion has a nonzero finite-L remainder. The maximum center
residual is about 0.386870 at L4096; that number is not the finite-kernel
computation error. The original-model precision pairing and small arithmetic
residuals do not certify all matrix input and quadrature errors. No full
distance cutoff, arbitrary initial state, Lindblad extension, mixed-state
fidelity, or postselection success probability is inferred. The fixed-bias
many-body fidelity result is an analytical corollary with no new determinant
numerical acceptance. Earlier rejected/failed records remain historical
records and are not overwritten by this release.

## 4. Full HPC recomputation

The original computational Python modules are included with their local import
dependencies. Portable Slurm wrappers are new; the mathematical kernels and
original precision/node/rank/gate parameters are unchanged. Fresh runs create
new provenance locks and output files. Do not run historical initialize or
cleanup commands against the published delivery.

Use a new empty scratch folder outside the checkout. On a Slurm machine with
the environment from section 2 installed:

```bash
export REPRO_PROJECT="$PWD"
export REPRO_OUTPUT_DIR="$(mktemp -d "$HOME/finite_bias_rerun.XXXXXXXX")"
.venv/bin/python scripts/reproduce_finite_bias_closure.py hpc-init --output-dir "$REPRO_OUTPUT_DIR"

# Eight tasks: four sizes x two precision/node/rank configurations.
pred_job=$(sbatch --parsable --array=0-7%8 scripts/slurm_finite_bias_closure_fixed.sh fixed-predict)
# Eight complete reflected-Gram preparations, up to 1790 static digits.
prep_job=$(sbatch --parsable --array=0-7%8 scripts/slurm_finite_bias_closure_fixed.sh prepare)
# 64 tasks, each containing the two fixed thresholds: 128 precision records.
sbatch --array=0-63%16 --dependency="afterok:${pred_job}:${prep_job}" scripts/slurm_finite_bias_closure_fixed.sh fixed-validate
# 36 new weak-bias conditions, using the same independent raw preparations.
sbatch --array=0-1%2 --dependency="afterok:${prep_job}" scripts/slurm_finite_bias_closure_fixed.sh weak-validate
# 32 independent size/precision/side/g tasks; 64 large-size prediction records.
large_job=$(sbatch --parsable --array=0-31%16 scripts/slurm_finite_bias_closure_split.sh split-predict)
sbatch --array=0 --dependency="afterok:${large_job}" scripts/slurm_finite_bias_closure_split.sh assemble
```

The scratch folder contains weak_native/validation, fixed_bias/validation,
fixed_bias/extrapolation_split, and four extrapolation_accelerated files.
HPC reruns can require hours and substantial RAM. The portable default is
4 CPUs, 32 GB and four hours per task; increase limits according to the local
scheduler and hardware if needed. FLINT is configured with the allocated
thread count; this is not a guarantee that every phase uses all cores.
BLAS stays single-threaded. No SSH credentials or original private HPC logs
are part of this release.

To recompute the independent half-axis reference instead of just using it:

```bash
.venv/bin/python scripts/check_endpoint_metric_mp.py --cells 256,512,1024 --dps 80 --nodes 128 --output "$REPRO_OUTPUT_DIR/metric80.json"
.venv/bin/python scripts/check_endpoint_metric_mp.py --cells 512,1024 --dps 100 --nodes 192 --output "$REPRO_OUTPUT_DIR/metric100.json"
```

These recompute numerical integral matrices and precision/truncation pairing;
they do not by themselves supply a total quadrature interval certificate.

## 5. Release integrity

public_release_manifest.json lists the prepared files, byte sizes and SHA256
values. The compact records retain decimal root strings when available.
The release does not contain large caches, private execution logs, credentials,
or internal author/editor deliberations. The existing repository retains its
earlier scientific results and failure records. No old failure is relabeled.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "plot", "bundle", "hpc-init", "hpc-run"))
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path, help="Optional new JSON receipt for the check action")
    parser.add_argument("--dps", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--hpc-stage", choices=("fixed-predict", "prepare", "fixed-validate", "weak-validate", "split-predict", "assemble"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    data = args.data_dir.resolve() if args.data_dir else project / DATA_REL
    if args.action == "check":
        result = check(project, data, args.dps, args.tolerance)
        if args.report:
            if args.report.exists():
                raise FileExistsError("Choose a new check receipt path")
            write(args.report.resolve(), result)
    elif args.action == "plot":
        output = args.output_dir.resolve() if args.output_dir else project / "reproduced_figures"
        plot(project, data, output)
        result = dict(status="figures_reproduced", output=str(output), original_figures_installed=False)
    elif args.action == "bundle":
        result = bundle(project, project / STAGE_REL)
    elif args.action == "hpc-init":
        if args.output_dir is None:
            parser.error("hpc-init requires --output-dir outside the checkout")
        result = hpc_init(project, args.output_dir.resolve())
    else:
        if args.output_dir is None or args.hpc_stage is None:
            parser.error("hpc-run requires --output-dir and --hpc-stage")
        result = hpc_run(project, args.output_dir.resolve(), args.hpc_stage, args.index)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
