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
| Main 2 | `data/prl_p3_new_sizes/states.csv`, `data/prl_theory_upgrade/widths.csv`; analytic CDW envelope |
| Main 3 | `data/prl_theory_upgrade/spatial_response_checks.csv`, `weak_skin_response.csv` |
| Main 4 | `data/prl_theory_upgrade/roots.csv`, `reference_checks.csv`, `widths.csv` |
| Supplemental 1: rank failure | `data/prl_p3_new_sizes/development/rank_failure_resolution.csv` |
| Supplemental 2: complex model | `data/prl_p4_passive_complex/states.csv` |
| Supplemental 3: resources | `data/prl_e_feasibility/common_loss_states.csv` |
| Supplemental 4: selection/memory | `data/prl_p3_new_sizes/states.csv`, `front_checks.csv`, `memory.csv` |
| Supplemental 5: envelopes | Analytic constants/envelopes and `data/prl_theory_upgrade/reference_checks.csv` |

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

For the original full pipeline, static preparation precedes low-rank prediction;
predictions are fixed before independent direct propagation; analysis is run after
both results are available. The source guards deliberately reject changed historic
inputs. The original input-manifest files may refer to intermediate matrices that
are not distributed here. Regenerate these in a separate work directory with a
new provenance record; do not remove guards and describe a rerun of already seen
results as new validation. The compact saved prediction records and parameters
are released for inspection. The original baseline ZIP and large raw matrix
archives remain outside this Git repository.

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
