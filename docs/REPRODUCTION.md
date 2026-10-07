# Reproduction And Numerical Conventions

## Physical Quantities

The site basis is `(A_0,B_0,A_1,B_1,...)`. The main-text model may label cells from
one; translated block starts and the complex onsite modulation use zero-based
cell indices. `L` is the number of sites and `N=L/2` the number of particles.
CDW occupies every A site; left/right occupy `N` consecutive sites at the ends;
translated blocks contain `L/4` complete cells.

The evolved orbitals are orthonormalized before forming the Hermitian projector
`P=QQ^dagger`. Distances are operator norms of differences of equal-rank
projectors, equivalently the sine of their largest principal angle. Entropy is
computed from the spatial left-half correlation eigenvalues, with natural logs.
The growing invariant-space target `G` and finite-time left-singular target `W`
are different objects. Figure-data columns state the target being used.

Conditional passive evolution includes a common loss shift. This shift cancels
in the normalized projector but remains in the no-loss probability. In the
pilot, `c=2+sinh(0.15)`. Accepting runs with the same final and initial particle
numbers isolates no-loss runs if loss is the only process that changes particle number.
Conditioning on no observed clicks with inefficient detection gives a different,
generally mixed ensemble.

## Figure Inputs

| Figure | Numerical input or construction |
|---|---|
| Main 1 | Deterministic apparatus/occupation schematic; no simulation data |
| Main 2 | Extended `traces.csv`, `early_output.csv`, `widths.csv`, `prediction_checks.csv`; retained entropy snapshot; analytic output/CDW bounds |
| Main 3 | Static panel: extended `signed_response.csv`; actual-time panel: `data/prl_priority_strengthening/time_response.csv` and zero-bias prediction locks |
| Main 4 | Extended `traces.csv`, `independent_checks.csv`, `widths.csv`; analytic contraction envelope |
| Supplemental 1: rank failure | `data/prl_p3_new_sizes/development/rank_failure_resolution.csv` |
| Supplemental 2: complex model | `data/prl_p4_passive_complex/states.csv` |
| Supplemental 3: resources | `data/prl_e_feasibility/common_loss_states.csv` |
| Supplemental 4: selection/memory | `data/prl_p3_new_sizes/states.csv`, `front_checks.csv`, `memory.csv` |
| Supplemental 5: envelopes | Analytic constants/envelopes and `data/prl_theory_upgrade/reference_checks.csv` |
| Supplemental 6: extended controls | Extended signed response, reflection ratios, absolute widths, frozen forecast errors; retained absolute scales and six-threshold diagnostics |
| Supplemental 7: actual-time response | `data/prl_priority_strengthening/zero_bias_response.csv` and `time_response.csv`: susceptibility, geometric factor, decay rate, and all 528 errors |

The supplemental labels above describe output types. In the current manuscript
their order is envelopes (S1), actual-time response (S2), rank failure (S3),
extended controls (S4), selection/memory (S5), complex model (S6), and resources (S7).

Here "extended" refers to `data/prl_figure_strengthening/`. That directory combines
new results with labeled earlier cohorts without replacing the historical tables.
The 64 larger-size forecasts were frozen using the five preceding sizes; their
errors are reported without fitting a new acceptance tolerance. The static
reflected scale is not the exactly whitened input overlap or an actual-time shift.
Sixteen of 64 new forecasts exceed the preceding 0.15 tolerance; the maximum
absolute error is 0.280337 and maximum relative error is 0.126745%.
The nine actual-state trajectories contain 541 samples. The L384 tail's
maximum reconstruction remainder relative to distance is 1.58301e-7;
an absolute remainder divided by an extremely small CDW distance can be large.

`data/prl_manuscript_revision/fig*.csv` contains the compact per-panel exports.
`figure_provenance.json` records the original assets and input hashes. Reproduction
uses the same tables and analytic formulas; PDF metadata, fonts available on the
host, and layout library versions can change image bytes. The original scientific
table hashes should match exactly. `.gitattributes` disables automatic line-ending
conversion so frozen source and table hashes survive cloning on different hosts.
Supplemental envelopes are drawn by a portable
function that reads the saved slope table instead of large raw reference matrices.

## Production Calculations

The original sources are released with their local import dependencies. They
retain source-hash and historical-input checks. `--help` lists the stages of each
entry point. These entry points are not automatically dispatched by `all`.

| Entry Point | Calculation | Dependencies / Precision |
|---|---|---|
| `run_selection_front_hpc.py` | Static input preparation and direct occupied-column propagation | Historical prediction inputs; arbitrary precision; single-thread BLAS |
| `run_new_sizes_hpc.py` | Larger-size validation, local sampled crossings, reflected Gram checks | Earlier rank-two and directional stages, static matrices, fixed prediction records; 240/280 digits for references |
| `run_passive_complex_hpc.py` | Complex local model, passive embedding, direct references | Model configuration and stage outputs; 120/160-digit references with 200-digit independent checks |
| `run_experimental_feasibility.py` | Small-size success, missed-loss probabilities, tomography budgets | Configuration plus family outputs; 64/96 digits |
| `run_theory_upgrade_hpc.py` | Continuous SSH selection, structured exponential, low-rank roots | Historical statics at L128/160/192, fresh statics at L224/256, extrapolation/reference records; 320/360 digits for new comparisons |
| `run_spatial_response_theory.py` | Reflected weak-skin response | Prediction record and Gram quadrature; `2L` and `2L+40` digits at new sizes |
| `audit_theory_envelopes.py` | Small-matrix checks of analytic inequalities | Uses the released numerical kernels |
| `run_figure_strengthening.py` | Extended L320/384 dynamics, g=1/8 controls, signed response and reflection comparisons | Cached reciprocal spectra at 240/280--540/580 digits; stages `spectral`, `static`, `curves`, `response`, `reflection`, `reference_context`, `reference` |
| `run_figure_traces.py`, `run_figure_traces_safe.py` | Full trajectories and 12-unit post-selection tails | Uses spectral/curve outputs; complete-column QR when compact early coordinates are ill conditioned; stages `checks`, `trace` |
| `run_figure_curves_parallel.py`, `run_figure_trace_chunks.py` | L384 geometry roots and parallel time samples | Four independent geometry tasks per family; four interleaved time chunks per trajectory; hash-checked assembly |
| `run_figure_traces_fast.py`, `run_figure_traces_adaptive.py` | Full signed-residual norm-product bound and rank adaptation | Small direct-state checks precede large-size benchmarking; larger ranks are required when the product bound is too conservative |

The extended dynamic entry points can regenerate reciprocal spectra and static
contexts from their released configuration. They also read earlier curve files
for the L128/192 display traces and L512 raw reflected Gram inverses for some
static scans. Those raw matrices and historic curve JSON files are retained in
the full research project. Quick figure reproduction reads the released compact
tables and does not require those matrices. The production `initialize` command
also records a manuscript backup in the full project; these files are outside
this public code release.

For the extended residual bound, every entry of
`E(t)-A_r(t)B_r(t)` retains its sign and exponential weight. Frobenius
submultiplicativity bounds the discarded complete columns. The rank-12
product bound is too conservative for the L384 left block; the retained
benchmark documents rank adaptation rather than accepting that bound.
Displayed tail samples require a reconstruction remainder below both `1e-7`
and `0.001 * distance`. Complete-column QR remains the fallback. Two
working precisions are compared at each point. These checks establish
numerical convergence, not interval certification.

For the original full pipeline, static preparation precedes low-rank prediction;
predictions are fixed before independent direct propagation; analysis is run after
both results are available. The source guards deliberately reject changed historic
inputs. The original input-manifest files may refer to intermediate matrices that
are not distributed here. Regenerate these in a separate work directory with a
new provenance record; do not remove guards and describe a rerun of already seen
results as new validation. The compact saved prediction records and parameters
are released for inspection. The original baseline ZIP and large raw matrix
archives remain outside this Git repository.

## Actual-Time Response Study

The released `prl_priority_strengthening` configuration contains eleven families,
66 zero-bias crossings, 528 frozen nonzero-bias predictions, two-precision cache
metadata, complete-rank `.npz` caches, and compact direct-reference records.
`python scripts/reproduce_paper.py response` actively recomputes all 528 distances
and susceptibilities from these caches, checks saved thresholds and responses at
`2e-8`, and checks the original `0.01` prediction gate. It does not regenerate
arbitrary-precision static matrices or count already seen data as new validation.

The production entry point is `scripts/run_priority_strengthening.py`. Stages
are `prepare`, `baseline`, `scan`, and `reference`, in that order. New-size and
parameter-control `prepare` tasks (indices 5--10) regenerate their raw matrices
from the released physical parameters. Retested indices 0--4 use historical raw
spectral inputs kept in the full project. Preserve the existing locks and outputs
when investigating a new hypothesis; a rerun needs a distinct provenance record.

The open-end coupling recurrence retains the SSH edge singular mode. Complete
signed logarithms of the initial graph avoid propagating exponentially large
coefficients directly in float64. Large direct references independently propagate
all occupied columns and use arbitrary-precision QR. For their norm measurement,
a float64 SVD supplies a trial direction; its normalization, propagated image,
and complete Frobenius remainder are evaluated at high precision. Exact-arithmetic
lower and upper norm inequalities then check the original reference tolerance.
This accelerates the observable calculation without discarding evolved columns.
The spectral input is shared between the two propagation paths.

## HPC Performance

Use a GMP-backed mpmath runtime for large calculations:

```bash
python -m pip install -e '.[hpc]'
python -c "from mpmath.libmp import BACKEND; print(BACKEND)"
```

The backend should report `gmpy`. Set `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` for independent family processes.
Parallelize parameter families using Slurm arrays; bind processes to the requested
CPUs. The original kernels cache solves with multiple right-hand sides, use thin
QR/SVD for reduced occupied-space cores, and evaluate the SSH propagator through
the coupling SVD and hyperbolic spectral blocks. They preserve signed products
and full exponent weights. A rank cutoff is accepted only with the corresponding
full occupied-subspace remainder bound.

Select the cluster's partition, account, time limit, and memory locally. The
public package does not embed the authors' SSH host configuration or submit jobs
automatically. The calculation configurations record the physical parameters and
precision schedule used for the published comparisons.

## Dependency Versions

The numerical reference environment used NumPy 2.2.6, SciPy 1.16.2,
mpmath 1.3.0, Matplotlib 3.10.7, threadpoolctl 3.6.0, and psutil 7.1.0.
`pyproject.toml` gives compatible ranges; the independent pilot report records
the actual runtime versions. Arial is used for the original main figures; on
systems without Arial, Matplotlib may use its fallback font. This affects
typography, not the underlying numerical tables.
