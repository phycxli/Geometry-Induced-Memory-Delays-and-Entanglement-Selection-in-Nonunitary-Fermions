"""Single-particle models and non-Bloch diagnostic helpers."""

from __future__ import annotations

import numpy as np


def hatano_nelson_hamiltonian(length: int, hopping: float = 1.0, skin: float = 0.0, boundary: str = "open") -> np.ndarray:
    """Return the Hatano-Nelson single-particle Hamiltonian.

    H = sum_j t_R |j+1><j| + t_L |j><j+1|, with
    t_R = t exp(g), t_L = t exp(-g).

    For open boundaries this is similar to a Hermitian chain, but the similarity
    transformation produces the skin profile. For periodic boundaries the point
    gap winding is visible in the complex spectrum.
    """
    if length < 2:
        raise ValueError("length must be at least 2")
    t_right = hopping * np.exp(skin)
    t_left = hopping * np.exp(-skin)
    h = np.zeros((length, length), dtype=np.complex128)
    for j in range(length - 1):
        h[j + 1, j] = t_right
        h[j, j + 1] = t_left
    if boundary == "periodic":
        h[0, length - 1] = t_right
        h[length - 1, 0] = t_left
    elif boundary != "open":
        raise ValueError("boundary must be 'open' or 'periodic'")
    return h


def single_particle_propagator(h: np.ndarray, dt: float) -> np.ndarray:
    """Compute exp(-i H dt) using eigendecomposition.

    This avoids a scipy dependency. It is adequate for the moderate matrix sizes
    used in the initial scans.
    """
    vals, vecs = np.linalg.eig(h)
    inv_vecs = np.linalg.inv(vecs)
    return vecs @ np.diag(np.exp(-1j * vals * dt)) @ inv_vecs


def skin_length(skin: float) -> float:
    """Hatano-Nelson skin length estimate xi = 1 / |g|."""
    if abs(skin) < 1e-14:
        return float("inf")
    return 1.0 / abs(skin)

