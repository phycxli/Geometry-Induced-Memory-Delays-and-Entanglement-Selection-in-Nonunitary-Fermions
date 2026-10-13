# Geometry-Induced Memory Delays and Entanglement Selection in Nonunitary Fermions

Numerical code and figure data for the study by Chengxi Li, Hao Zhu, Tie-Fu Zhang,
Wanzi Sun, and Wuming Liu.
The calculations concern normalized, fixed-particle, no-loss Slater states in an
open non-Hermitian SSH chain and a complex local extension. Projector distance
measures occupied-subspace memory; it is distinct from a many-body trace distance.

## Fixed-Bias Selection-Time Upgrade

The current manuscript is titled *Isospectral Control of Many-Body Selection
Times*. The new compact release contains the 100 new original-model conditions,
32 larger-size finite-kernel predictions, analytical coefficient evaluators,
coefficient plotting code, and portable Slurm commands. See
[FINITE_BIAS_CLOSURE_REPRODUCIBILITY.md](FINITE_BIAS_CLOSURE_REPRODUCIBILITY.md)
for the complete parameter ranges and commands.

```bash
python -m pip install -r requirements-finite-bias-closure.txt
python scripts/reproduce_finite_bias_closure.py check
python scripts/reproduce_finite_bias_closure.py plot --output-dir reproduction_output/closure_figures
```

The small check reevaluates analytical coefficients and saved record relations;
it does not rerun large matrices or finite-kernel roots. The full calculations
use the separate scratch-directory HPC workflow in the linked guide. Existing
sections below describe earlier releases, including their finite-size failures.

## Current Physical-Narrative Figures

The current main figures emphasize the physical timing control. The same
released fixed-bias and weak-bias records now produce the following panels:

| Current figure | Content | Public output |
|---|---|---|
| Main 3 | Reflected overlap-weight schematic and left/right selection time per site versus four discrete biases, at L1280 and L4096 | `figures/prl_physics_narrative/fig3_reflection_response.*` |
| Main 4 | Left/right waiting time and low-threshold window width versus size at fixed g=0.1 | `figures/prl_physics_narrative/fig4_continuous_window.*` |
| Supplemental coefficient checks | Unfitted logarithmic time terms and the scaled weak-bias response correction | `figures/prl_physics_narrative/fig_s_narrative_checks.*` |

```bash
python scripts/plot_physics_narrative_figures.py --no-install --output-dir figures/prl_physics_narrative
```

The command writes three PDF/PNG/SVG sets and
`figure_narrative_manifest.json`, including the plotted records and input
SHA256 values. It uses the public `theory_reference.json` and does not compute
new times or modify source tables. Filled markers are original-model roots;
open markers at L2048/4096 are finite-kernel predictions. Lines connecting bias
samples are guides, not fits or a continuous numerical derivative scan. The
window's horizontal dashed lines are analytic limits. Roots target the growing
invariant space G; transfer to the output-singular target W is analytic. These
plots do not supply an independent numerical full-Slater-fidelity test.

`reproduce_finite_bias_closure.py plot` and its original assets in
`figures/prl_manuscript_revision/` retain the earlier Fig. 3/4 coefficient-panel
layout. The new output directory preserves this history. Main figures 1 and 2
continue to use the apparatus and preparation-memory assets in
`figures/prl_manuscript_revision/`. The complete current and historical mapping
is in [Reproduction](docs/REPRODUCTION.md).

## Quick Reproduction

Use Python 3.11 or later. From a fresh checkout:

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -e .
python scripts/reproduce_paper.py all
```

The command verifies every released file against `release_manifest.json`,
independently recomputes the three short-time preparations using SciPy and
80-digit mpmath, recomputes 528 principal-pair response conditions from full-rank
caches, and redraws the four earlier main figures and seven cited supplemental
figures. One additional historical geometry plot is also generated.
Results go to `reproduction_output/`; frozen input tables are preserved.
`reproduction_output/verification.json` records numerical differences and output
hashes. Each stage reports `status: passed` on success and fails with a nonzero
exit status when a check fails.

```bash
python scripts/reproduce_paper.py verify
python scripts/reproduce_paper.py pilot
python scripts/reproduce_paper.py response
python scripts/reproduce_paper.py figures
```

The earlier pilot uses `L=8`, `N=4`, `g=0.15`, and `t=0.5`. Expected no-loss probabilities
are approximately `0.359`, `6.70e-4`, and `1.65e-4` for CDW, left, and right.
This is a numerical prediction for a proposed experiment, not experimental data.

## Fixed-Site Memory Protocol

The strengthened short-time proposal uses `L=4`, `N=2`, `g=0.15`, and `t=0.5`.
The readout is the occupation of site `A1` for CDW and right-block preparations.
Its predicted contrast is `-0.986849`, with no-loss probabilities `0.632715`
and `0.0121865`. The readout and control gates were frozen before 768 tolerance
checks, including an explicit auxiliary mode. Run:

```bash
python scripts/reproduce_operational_memory.py
```

This recomputes the control samples, held-out times, finite-auxiliary
comparisons, and time curves into `reproduction_output/operational_memory/`,
checks them against released tables, and repeats arbitrary-precision and
fixed-number Fock-space references. It preserves all released inputs.
Full parameters and sample budgets are in `data/prl_operational_memory/`.
The 1000-accepted-runs budget gives a statistical contrast error of `0.093617`
at 95% confidence. Calibration and false atom-number acceptance remain hardware
requirements. The samples do not certify the entire continuous tolerance box;
there are no experimental observations or external proof-review results here.

## Repository Layout

| Directory | Contents |
|---|---|
| `src/mietf_skin/` | Reusable SSH, Gaussian-state, projector, and diagnostic modules |
| `scripts/` | Original calculation and analysis scripts, their local imports, and the portable reproduction entry point |
| `configs/` | Short-time pilot parameters; original study configurations also remain with their tables under `data/` |
| `data/` | Figure tables, calculation configurations, prediction records, and compact summaries |
| `figures/` | Reference figure assets used in the manuscript |
| `docs/` | Numerical conventions, figure-data mapping, and large-calculation dependencies |

## Scope And Precision

The actual-time study in `data/prl_priority_strengthening/` extends weak-bias
dynamics to L448/512 and adds four hopping/loss controls. Its 528 nonzero-bias
time predictions were fixed from same-family zero-bias responses before the
nonzero-bias calculations. All meet the 0.01 absolute time gate; the maximum
error is 0.00429280. This is a bias prediction, with zero-bias centers supplied
as calculated inputs. All 66 finite-size susceptibilities remain outside the
0.005 tolerance of the unproved static-response/band-edge candidate. Observed
directional signs and increasing agreement do not prove an asymptotic coefficient.
In that earlier release, Main Fig. 3 compared static input response and
actual-time prediction. Its supplemental figure reports geometric factors,
rates, and both errors. The current figure mapping is given above.
The parameter-box interval certificate concerns analytic constants; the
numerical crossing times have precision checks, not interval certificates.

The earlier main figures 2--4 include the extended-size, signed-response, and time-trajectory
checks saved under `data/prl_figure_strengthening/`. Redrawing them
verifies and visualizes those tables; it does not rerun the large calculations.
The extension adds 144 low-threshold candidates, 541 trajectory samples, and
28 independent propagation points bracketing 14 new crossings. Of 64 frozen
time forecasts at L320/384, 16 exceed the preceding 0.15 tolerance; the maximum
absolute error is 0.280337 and maximum relative error is 0.126745%.
The combined 76 low-threshold widths range from 1.2809893 to 1.3125999.
These observations are separate from the continuous low-threshold theorem.
The independent short-time reproduction compares projector distance, entropy,
and log no-loss probability with the saved tables and compares float64 with an
independent arbitrary-precision calculation. The tolerance is `2e-10`.

Original numerical sources are preserved so that source hashes in the study
records can still be checked. Large intermediate matrices, full scheduler logs,
internal research notes, and manuscript files are retained outside this code
release. Some original production stages require these historical matrices or
their regeneration. Their dependencies and stage order are documented in
[Reproduction](docs/REPRODUCTION.md); a fresh checkout is sufficient for the quick
reproduction above, not for every historical production command.

Precision convergence supports numerical accuracy; it is not an interval
arithmetic certificate. Low-threshold continuous contraction is established for
the specified SSH family. The complex model tests reconstruction and should not
be assigned the SSH contraction constants.

## Citation And License

Citation metadata is in `CITATION.cff`. The repository retains its MIT license.
No paper DOI or journal acceptance is asserted by this software release.
