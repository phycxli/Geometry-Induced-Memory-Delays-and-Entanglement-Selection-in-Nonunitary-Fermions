"""Staggered-loss Hatano-Nelson no-click diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounds import (
    entropy_lipschitz_constant,
    evolved_subspace,
    hermitian_part,
    projection_frobenius,
    projection_trace_distance_per_particle,
    restricted_correlation,
    trace_norm,
)
from .diagnostics import normalized_product
from .gaussian import half_chain_entropy, initial_slater_state
from .models import single_particle_propagator


@dataclass
class HatanoSubspaceResult:
    model: str
    cells: int
    length: int
    hopping: float
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
    singular_log_gap: float
    singular_ratio: float
    top_overlap_min_sv: float
    graph_norm_bound: float


def staggered_hatano_nelson_hamiltonian(
    length: int,
    hopping: float,
    gain_loss: float,
    skin: float,
    boundary: str,
) -> np.ndarray:
    """Return a one-band Hatano-Nelson chain with staggered gain/loss.

    The model is deliberately distinct from the SSH baseline: there is no
    dimerized hopping, only uniform nonreciprocal nearest-neighbor hopping and a
    two-site gain/loss pattern.
    """
    if length < 4 or length % 2 != 0:
        raise ValueError("length must be an even integer >= 4")
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")

    t_right = hopping * np.exp(+skin)
    t_left = hopping * np.exp(-skin)
    h = np.zeros((length, length), dtype=np.complex128)
    for site in range(length):
        h[site, site] = 1j * gain_loss * (1.0 if site % 2 == 0 else -1.0)
    for site in range(length - 1):
        h[site + 1, site] = t_right
        h[site, site + 1] = t_left
    if boundary == "periodic":
        h[0, length - 1] = t_right
        h[length - 1, 0] = t_left
    return h


def hatano_step_matrix(
    length: int,
    hopping: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
) -> np.ndarray:
    h = staggered_hatano_nelson_hamiltonian(
        length=length,
        hopping=hopping,
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
    )
    return single_particle_propagator(h, dt)


def hatano_subspace_diagnostics(
    length: int,
    hopping: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
    steps: int,
    initial_pattern: str = "charge_density_wave",
    initial_seed: int | None = None,
) -> HatanoSubspaceResult:
    """Compare evolved staggered Hatano-Nelson subspace with SVD subspaces."""
    particles = length // 2
    step = hatano_step_matrix(
        length=length,
        hopping=hopping,
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
        dt=dt,
    )
    actual = evolved_subspace(
        step_matrix=step,
        steps=steps,
        initial_pattern=initial_pattern,
        initial_seed=initial_seed,
    )
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

    q0 = initial_slater_state(length, filling=0.5, pattern=initial_pattern, rng_seed=initial_seed)
    overlap_top = right_top.conj().T @ q0
    top_singular = np.linalg.svd(overlap_top, compute_uv=False)
    min_overlap = float(np.min(top_singular))
    ratio = float(singular_values[particles] / singular_values[particles - 1]) if particles < len(singular_values) else 0.0
    graph_bound = float(ratio / max(min_overlap, 1e-14))

    return HatanoSubspaceResult(
        model="staggered_hatano_nelson",
        cells=length // 2,
        length=length,
        hopping=hopping,
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
        singular_log_gap=float(np.log(singular_values[particles - 1]) - np.log(singular_values[particles])),
        singular_ratio=ratio,
        top_overlap_min_sv=min_overlap,
        graph_norm_bound=graph_bound,
    )
