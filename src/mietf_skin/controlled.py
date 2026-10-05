"""Controlled-overlap tests for the singular-subspace graph bound."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounds import (
    entropy_lipschitz_constant,
    fermionic_audenaert_bound,
    hermitian_part,
    projection_frobenius,
    projection_trace_distance_per_particle,
    restricted_correlation,
    trace_norm,
)
from .diagnostics import normalized_product, ssh_step_matrix
from .gaussian import half_chain_entropy, orthonormalize
from .hatano import hatano_step_matrix


@dataclass
class ControlledOverlapResult:
    model: str
    cells: int
    length: int
    t1: float
    t2: float
    hopping: float
    gain_loss: float
    skin: float
    boundary: str
    dt: float
    steps: int
    target_overlap: float
    theta: float
    mixing_seed: int
    actual_entropy: float
    left_entropy: float
    right_entropy: float
    entropy_diff_left: float
    entropy_diff_right: float
    actual_entropy_density: float
    left_entropy_density: float
    right_entropy_density: float
    projector_trace_left_per_particle: float
    projector_trace_right_per_particle: float
    projector_spectral_left: float
    projector_spectral_right: float
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
    top_overlap_max_sv: float
    complement_overlap_norm: float
    graph_norm_actual: float
    graph_norm_bound: float
    graph_norm_bound_loose: float
    sin_bound_actual_graph: float
    sin_bound_from_bound: float


def random_unitary(size: int, seed: int) -> np.ndarray:
    """Return a deterministic Haar-like unitary from a complex Gaussian QR."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r).copy()
    phases[np.abs(phases) < 1e-14] = 1.0
    return q * (phases / np.abs(phases)).conj()


def projection_spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return ||P_a-P_b||_2 for two equal-rank subspaces."""
    overlap = a.conj().T @ b
    singular = np.linalg.svd(overlap, compute_uv=False)
    singular = np.clip(singular.real, 0.0, 1.0)
    return float(np.sqrt(np.max(np.maximum(0.0, 1.0 - singular**2))))


def controlled_initial_state(
    right_top: np.ndarray,
    right_bottom: np.ndarray,
    target_overlap: float,
    mixing_seed: int,
) -> np.ndarray:
    """Construct Q0 with controlled s_min(V_+^dagger Q0).

    For half filling, right_top and right_bottom both have N columns.  With a
    unitary R,

        Q0 = s V_+ + sqrt(1-s^2) V_- R

    has orthonormal columns and V_+^dagger Q0 = s I.
    """
    if not (0.0 < target_overlap <= 1.0):
        raise ValueError("target_overlap must lie in (0, 1]")
    particles = right_top.shape[1]
    complement = np.sqrt(max(0.0, 1.0 - target_overlap**2))
    unitary = random_unitary(particles, mixing_seed)
    q0 = target_overlap * right_top + complement * (right_bottom @ unitary)
    return orthonormalize(q0)


def controlled_overlap_family(
    step_matrix: np.ndarray,
    model: str,
    cells: int,
    t1: float,
    t2: float,
    hopping: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
    steps: int,
    target_overlaps: list[float],
    mixing_seeds: list[int],
) -> list[ControlledOverlapResult]:
    """Run a controlled-overlap family for a fixed finite-time matrix."""
    length = step_matrix.shape[0]
    particles = length // 2
    product = normalized_product(step_matrix, steps)
    left, singular_values, vh = np.linalg.svd(product, full_matrices=True)
    singular_values = np.maximum(singular_values, 1e-300)
    right = vh.conj().T

    left_top = left[:, :particles]
    right_top = right[:, :particles]
    right_bottom = right[:, particles:]
    left_entropy = half_chain_entropy(left_top)
    right_entropy = half_chain_entropy(right_top)

    sigma_plus_inv = np.diag(1.0 / singular_values[:particles])
    sigma_minus = np.diag(singular_values[particles:])
    ratio = float(singular_values[particles] / singular_values[particles - 1]) if particles < len(singular_values) else 0.0
    singular_log_gap = float(np.log(singular_values[particles - 1]) - np.log(singular_values[particles]))

    out: list[ControlledOverlapResult] = []
    for target_overlap in target_overlaps:
        theta = float(np.arccos(np.clip(target_overlap, 0.0, 1.0)))
        complement = float(np.sqrt(max(0.0, 1.0 - target_overlap**2)))
        for mixing_seed in mixing_seeds:
            q0 = controlled_initial_state(
                right_top=right_top,
                right_bottom=right_bottom,
                target_overlap=target_overlap,
                mixing_seed=mixing_seed,
            )
            actual = orthonormalize(product @ q0)
            actual_entropy = half_chain_entropy(actual)

            ca = restricted_correlation(actual)
            cl = restricted_correlation(left_top)
            cr = restricted_correlation(right_top)
            restricted_left = trace_norm(hermitian_part(ca - cl))
            restricted_right = trace_norm(hermitian_part(ca - cr))
            lip_left = entropy_lipschitz_constant(ca, cl)
            lip_right = entropy_lipschitz_constant(ca, cr)
            subsystem_size = length // 2

            overlap_top = right_top.conj().T @ q0
            overlap_bottom = right_bottom.conj().T @ q0
            top_singular = np.linalg.svd(overlap_top, compute_uv=False)
            min_overlap = float(np.min(top_singular))
            max_overlap = float(np.max(top_singular))
            complement_norm = float(np.linalg.svd(overlap_bottom, compute_uv=False)[0])
            graph_bound = float(ratio * complement_norm / max(min_overlap, 1e-14))
            graph_bound_loose = float(ratio / max(min_overlap, 1e-14))

            graph_matrix = sigma_minus @ overlap_bottom @ np.linalg.inv(overlap_top) @ sigma_plus_inv
            graph_norm = float(np.linalg.svd(graph_matrix, compute_uv=False)[0])
            sin_graph = float(graph_norm / np.sqrt(1.0 + graph_norm**2))
            sin_bound = float(graph_bound / np.sqrt(1.0 + graph_bound**2))

            out.append(
                ControlledOverlapResult(
                    model=model,
                    cells=cells,
                    length=length,
                    t1=t1,
                    t2=t2,
                    hopping=hopping,
                    gain_loss=gain_loss,
                    skin=skin,
                    boundary=boundary,
                    dt=dt,
                    steps=steps,
                    target_overlap=float(target_overlap),
                    theta=theta,
                    mixing_seed=int(mixing_seed),
                    actual_entropy=actual_entropy,
                    left_entropy=left_entropy,
                    right_entropy=right_entropy,
                    entropy_diff_left=abs(actual_entropy - left_entropy),
                    entropy_diff_right=abs(actual_entropy - right_entropy),
                    actual_entropy_density=actual_entropy / length,
                    left_entropy_density=left_entropy / length,
                    right_entropy_density=right_entropy / length,
                    projector_trace_left_per_particle=projection_trace_distance_per_particle(actual, left_top),
                    projector_trace_right_per_particle=projection_trace_distance_per_particle(actual, right_top),
                    projector_spectral_left=projection_spectral_distance(actual, left_top),
                    projector_spectral_right=projection_spectral_distance(actual, right_top),
                    projector_fro_left=projection_frobenius(actual, left_top),
                    projector_fro_right=projection_frobenius(actual, right_top),
                    restricted_trace_left=restricted_left,
                    restricted_trace_right=restricted_right,
                    entropy_lipschitz_bound_left=lip_left * restricted_left,
                    entropy_lipschitz_bound_right=lip_right * restricted_right,
                    entropy_audenaert_bound_left=fermionic_audenaert_bound(restricted_left, subsystem_size),
                    entropy_audenaert_bound_right=fermionic_audenaert_bound(restricted_right, subsystem_size),
                    singular_log_gap=singular_log_gap,
                    singular_ratio=ratio,
                    top_overlap_min_sv=min_overlap,
                    top_overlap_max_sv=max_overlap,
                    complement_overlap_norm=complement_norm,
                    graph_norm_actual=graph_norm,
                    graph_norm_bound=graph_bound,
                    graph_norm_bound_loose=graph_bound_loose,
                    sin_bound_actual_graph=sin_graph,
                    sin_bound_from_bound=sin_bound,
                )
            )
    return out


def ssh_controlled_overlap_family(
    length: int,
    t1: float,
    t2: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
    steps: int,
    target_overlaps: list[float],
    mixing_seeds: list[int],
) -> list[ControlledOverlapResult]:
    if length % 2 != 0:
        raise ValueError("SSH length must be even")
    cells = length // 2
    step = ssh_step_matrix(cells=cells, t1=t1, t2=t2, gain_loss=gain_loss, skin=skin, boundary=boundary, dt=dt)
    return controlled_overlap_family(
        step_matrix=step,
        model="ssh",
        cells=cells,
        t1=t1,
        t2=t2,
        hopping=float("nan"),
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
        dt=dt,
        steps=steps,
        target_overlaps=target_overlaps,
        mixing_seeds=mixing_seeds,
    )


def hatano_controlled_overlap_family(
    length: int,
    hopping: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
    steps: int,
    target_overlaps: list[float],
    mixing_seeds: list[int],
) -> list[ControlledOverlapResult]:
    step = hatano_step_matrix(length=length, hopping=hopping, gain_loss=gain_loss, skin=skin, boundary=boundary, dt=dt)
    return controlled_overlap_family(
        step_matrix=step,
        model="staggered_hatano_nelson",
        cells=length // 2,
        t1=float("nan"),
        t2=float("nan"),
        hopping=hopping,
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
        dt=dt,
        steps=steps,
        target_overlaps=target_overlaps,
        mixing_seeds=mixing_seeds,
    )
