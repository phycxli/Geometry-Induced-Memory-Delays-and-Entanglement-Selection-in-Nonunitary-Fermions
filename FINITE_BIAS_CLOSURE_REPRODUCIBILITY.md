# Finite-size and fixed-bias closure: public reproduction

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
