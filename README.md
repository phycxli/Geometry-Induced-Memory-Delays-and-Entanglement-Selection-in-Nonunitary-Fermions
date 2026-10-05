# Geometry-Induced Memory Delays and Entanglement Selection in Nonunitary Fermions

Numerical code and figure data for the study by Chengxi Li, Haozhu, and Wanzi Sun.
The calculations concern normalized, fixed-particle, no-loss Slater states in an
open non-Hermitian SSH chain and a complex local extension. Projector distance
measures occupied-subspace memory; it is distinct from a many-body trace distance.

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
80-digit mpmath, and redraws the four main figures and five cited supplemental
figures. One additional historical geometry plot is also generated.
Results go to `reproduction_output/`; frozen input tables are preserved.
`reproduction_output/verification.json` records numerical differences and output
hashes. Each stage reports `status: passed` on success and fails with a nonzero
exit status when a check fails.

```bash
python scripts/reproduce_paper.py verify
python scripts/reproduce_paper.py pilot
python scripts/reproduce_paper.py figures
```

The pilot uses `L=8`, `N=4`, `g=0.15`, and `t=0.5`. Expected no-loss probabilities
are approximately `0.359`, `6.70e-4`, and `1.65e-4` for CDW, left, and right.
This is a numerical prediction for a proposed experiment, not experimental data.

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

Main figures 2--4 use saved high-precision large-size results. Redrawing them
verifies and visualizes those tables; it does not rerun the large calculations.
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
