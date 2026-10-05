"""Gaussian Slater-determinant utilities.

The many-body state is represented by an L x N matrix Q whose columns are
orthonormal occupied single-particle orbitals.
"""

from __future__ import annotations

import numpy as np


def orthonormalize(orbitals: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the occupied subspace."""
    q, r = np.linalg.qr(orbitals)
    phases = np.diag(r).copy()
    phases[np.abs(phases) < 1e-14] = 1.0
    q = q * (phases / np.abs(phases)).conj()
    return q


def initial_product_state(length: int, filling: float = 0.5, pattern: str = "charge_density_wave") -> np.ndarray:
    """Create a simple product Slater state.

    Parameters
    ----------
    length:
        Number of lattice sites.
    filling:
        Particle density. The particle number is rounded to the nearest int.
    pattern:
        `charge_density_wave` fills every other site first. `left` fills from
        the left boundary.
    """
    particles = int(round(length * filling))
    if particles <= 0 or particles > length:
        raise ValueError("particle number must be between 1 and length")

    if pattern == "charge_density_wave":
        sites = list(range(0, length, 2)) + list(range(1, length, 2))
        occupied = sites[:particles]
    elif pattern == "left":
        occupied = list(range(particles))
    else:
        raise ValueError(f"unknown initial pattern: {pattern}")

    q = np.zeros((length, particles), dtype=np.complex128)
    for col, site in enumerate(occupied):
        q[site, col] = 1.0
    return q


def random_slater_state(length: int, filling: float = 0.5, rng_seed: int | None = None) -> np.ndarray:
    """Create a Haar-random Slater state by QR factorizing a complex Gaussian matrix."""
    particles = int(round(length * filling))
    if particles <= 0 or particles > length:
        raise ValueError("particle number must be between 1 and length")

    rng = np.random.default_rng(rng_seed)
    orbitals = rng.normal(size=(length, particles)) + 1j * rng.normal(size=(length, particles))
    return orthonormalize(orbitals)


def initial_slater_state(
    length: int,
    filling: float = 0.5,
    pattern: str = "charge_density_wave",
    rng_seed: int | None = None,
) -> np.ndarray:
    """Create a deterministic product state or a random Slater state."""
    if pattern == "random":
        return random_slater_state(length=length, filling=filling, rng_seed=rng_seed)
    return initial_product_state(length=length, filling=filling, pattern=pattern)


def correlation_matrix(orbitals: np.ndarray) -> np.ndarray:
    """Return the one-body correlation matrix C_ij = <c_i^dagger c_j>."""
    return orbitals @ orbitals.conj().T


def half_chain_entropy(orbitals: np.ndarray) -> float:
    """Von Neumann entanglement entropy of the left half chain."""
    length = orbitals.shape[0]
    subsystem = slice(0, length // 2)
    ca = orbitals[subsystem, :] @ orbitals[subsystem, :].conj().T
    vals = np.linalg.eigvalsh((ca + ca.conj().T) / 2.0)
    vals = np.clip(vals.real, 1e-12, 1.0 - 1e-12)
    return float(-np.sum(vals * np.log(vals) + (1.0 - vals) * np.log(1.0 - vals)))


def measure_site(orbitals: np.ndarray, site: int, rng: np.random.Generator) -> tuple[np.ndarray, int, float]:
    """Projectively measure local occupation n_site.

    Returns
    -------
    orbitals:
        Updated normalized Slater determinant.
    outcome:
        1 for occupied, 0 for empty.
    prob_one:
        Born probability for the occupied outcome before sampling.
    """
    length, particles = orbitals.shape
    row = orbitals[site, :].copy()
    prob_one = float(np.clip(np.vdot(row, row).real, 0.0, 1.0))
    outcome = int(rng.random() < prob_one)

    if outcome == 0:
        projected = orbitals.copy()
        projected[site, :] = 0.0
        return orthonormalize(projected), outcome, prob_one

    if prob_one < 1e-14:
        raise FloatingPointError("sampled occupied outcome with near-zero probability")

    _, _, vh = np.linalg.svd(row.reshape(1, particles), full_matrices=True)
    rotation = vh.conj().T
    rotated = orbitals @ rotation

    updated = np.empty_like(orbitals)
    updated[:, 0] = 0.0
    updated[site, 0] = 1.0
    updated[:, 1:] = rotated[:, 1:]
    updated[site, 1:] = 0.0
    return orthonormalize(updated), outcome, prob_one
