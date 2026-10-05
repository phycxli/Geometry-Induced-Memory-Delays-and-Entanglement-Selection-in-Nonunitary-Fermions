"""Subspace-distance diagnostics and entropy-continuity bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagnostics import normalized_product, ssh_step_matrix
from .gaussian import half_chain_entropy, initial_slater_state, orthonormalize


@dataclass
class SubspaceBoundResult:
    cells: int
    length: int
    t1: float
    t2: float
    gain_loss: float
    skin: float
    boundary: str
    dt: float
    steps: int
    initial_pattern: str
    initial_seed: int
    actual_entropy: float
    left_entropy: float
    right_entropy: float
    entropy_diff_left: float
    entropy_diff_right: float
    projector_trace_left_per_particle: float
    projector_trace_right_per_particle: float
    projector_fro_left: float
    projector_fro_right: float
    restricted_trace_left: float
    restricted_trace_right: float
    entropy_lipschitz_bound_left: float
    entropy_lipschitz_bound_right: float
    entropy_audenaert_bound_left: float
    entropy_audenaert_bound_right: float
    singular_log_gap: float
    singular_ratio: float
    top_overlap_min_sv: float
    graph_norm_bound: float


def evolved_subspace(
    step_matrix: np.ndarray,
    steps: int,
    initial_pattern: str,
    initial_seed: int | None = None,
) -> np.ndarray:
    """Return orthonormal occupied orbitals after repeated no-click evolution."""
    length = step_matrix.shape[0]
    orbitals = initial_slater_state(length, filling=0.5, pattern=initial_pattern, rng_seed=initial_seed)
    for _ in range(steps):
        orbitals = orthonormalize(step_matrix @ orbitals)
    return orbitals


def projector(orbitals: np.ndarray) -> np.ndarray:
    return orbitals @ orbitals.conj().T


def restricted_correlation(orbitals: np.ndarray) -> np.ndarray:
    half = orbitals.shape[0] // 2
    return orbitals[:half, :] @ orbitals[:half, :].conj().T


def projection_trace_distance_per_particle(a: np.ndarray, b: np.ndarray) -> float:
    """Return 0.5 ||P_a - P_b||_1 divided by particle number."""
    overlap = a.conj().T @ b
    singular = np.linalg.svd(overlap, compute_uv=False)
    singular = np.clip(singular.real, 0.0, 1.0)
    sin_angles = np.sqrt(np.maximum(0.0, 1.0 - singular**2))
    return float(np.sum(sin_angles) / a.shape[1])


def projection_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    overlap = a.conj().T @ b
    singular = np.linalg.svd(overlap, compute_uv=False)
    singular = np.clip(singular.real, 0.0, 1.0)
    return float(np.sqrt(2.0 * np.sum(1.0 - singular**2)))


def trace_norm(matrix: np.ndarray) -> float:
    return float(np.sum(np.linalg.svd(matrix, compute_uv=False)))


def entropy_lipschitz_constant(ca: np.ndarray, cb: np.ndarray, floor: float = 1e-8) -> float:
    """A conservative local Lipschitz constant for h(x)."""
    vals = np.concatenate([np.linalg.eigvalsh(hermitian_part(ca)), np.linalg.eigvalsh(hermitian_part(cb))])
    vals = np.clip(vals.real, floor, 1.0 - floor)
    margin = float(min(np.min(vals), np.min(1.0 - vals)))
    margin = max(margin, floor)
    return float(np.log((1.0 - margin) / margin))


def binary_entropy_scalar(x: float) -> float:
    """Return h_2(x) with natural logarithms."""
    x = float(np.clip(x, 0.0, 1.0))
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return float(-x * np.log(x) - (1.0 - x) * np.log(1.0 - x))


def fermionic_audenaert_bound(restricted_trace_distance: float, subsystem_size: int) -> float:
    """Global continuity bound for Gaussian correlation entropies.

    If A and B are m x m correlation matrices with eigenvalues in [0,1],
    delta = ||A-B||_1, and tau = min(delta/m, 1 - 1/(2m)), then

        |Tr h(A) - Tr h(B)|
        <= m [tau log(2m-1) + h_2(tau)].

    The proof maps the spectra {n_i, 1-n_i} to probability vectors of length
    2m and applies Audenaert's sharp Fannes inequality up to its saturation
    point.  The bound is global: it has no divergence when occupations approach
    0 or 1.
    """
    if subsystem_size <= 0:
        raise ValueError("subsystem_size must be positive")
    delta = float(np.clip(restricted_trace_distance, 0.0, float(subsystem_size)))
    if delta <= 0.0:
        return 0.0
    dimension = 2 * subsystem_size
    tau = min(delta / float(subsystem_size), 1.0 - 1.0 / float(dimension))
    return float(subsystem_size * (tau * np.log(max(dimension - 1, 1)) + binary_entropy_scalar(tau)))


def hermitian_part(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0


def subspace_bound_diagnostics(
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
    steps: int,
    initial_pattern: str = "charge_density_wave",
    initial_seed: int | None = None,
) -> SubspaceBoundResult:
    """Compare the evolved Slater subspace with finite-time SVD subspaces."""
    length = 2 * cells
    particles = length // 2
    step = ssh_step_matrix(
        cells=cells,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
        dt=dt,
    )
    actual = evolved_subspace(step, steps=steps, initial_pattern=initial_pattern, initial_seed=initial_seed)
    product = normalized_product(step, steps)
    left, singular_values, vh = np.linalg.svd(product, full_matrices=True)
    singular_values = np.maximum(singular_values, 1e-300)
    right = vh.conj().T
    left_top = left[:, :particles]
    right_top = right[:, :particles]

    actual_entropy = half_chain_entropy(actual)
    left_entropy = half_chain_entropy(left_top)
    right_entropy = half_chain_entropy(right_top)

    ca = restricted_correlation(actual)
    cl = restricted_correlation(left_top)
    cr = restricted_correlation(right_top)
    restricted_left = trace_norm(hermitian_part(ca - cl))
    restricted_right = trace_norm(hermitian_part(ca - cr))
    lip_left = entropy_lipschitz_constant(ca, cl)
    lip_right = entropy_lipschitz_constant(ca, cr)
    subsystem_size = length // 2

    q0 = initial_slater_state(length, filling=0.5, pattern=initial_pattern, rng_seed=initial_seed)
    overlap_top = right_top.conj().T @ q0
    top_singular = np.linalg.svd(overlap_top, compute_uv=False)
    min_overlap = float(np.min(top_singular))
    ratio = float(singular_values[particles] / singular_values[particles - 1]) if particles < len(singular_values) else 0.0
    graph_bound = float(ratio / max(min_overlap, 1e-14))

    return SubspaceBoundResult(
        cells=cells,
        length=length,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
        dt=dt,
        steps=steps,
        initial_pattern=initial_pattern,
        initial_seed=-1 if initial_seed is None else int(initial_seed),
        actual_entropy=actual_entropy,
        left_entropy=left_entropy,
        right_entropy=right_entropy,
        entropy_diff_left=abs(actual_entropy - left_entropy),
        entropy_diff_right=abs(actual_entropy - right_entropy),
        projector_trace_left_per_particle=projection_trace_distance_per_particle(actual, left_top),
        projector_trace_right_per_particle=projection_trace_distance_per_particle(actual, right_top),
        projector_fro_left=projection_frobenius(actual, left_top),
        projector_fro_right=projection_frobenius(actual, right_top),
        restricted_trace_left=restricted_left,
        restricted_trace_right=restricted_right,
        entropy_lipschitz_bound_left=lip_left * restricted_left,
        entropy_lipschitz_bound_right=lip_right * restricted_right,
        entropy_audenaert_bound_left=fermionic_audenaert_bound(restricted_left, subsystem_size),
        entropy_audenaert_bound_right=fermionic_audenaert_bound(restricted_right, subsystem_size),
        singular_log_gap=float(np.log(singular_values[particles - 1]) - np.log(singular_values[particles])),
        singular_ratio=ratio,
        top_overlap_min_sv=min_overlap,
        graph_norm_bound=graph_bound,
    )
