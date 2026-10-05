"""Finite-time Lyapunov diagnostics for non-unitary Gaussian circuits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import hatano_nelson_hamiltonian, single_particle_propagator


@dataclass
class LyapunovResult:
    length: int
    skin: float
    measurement_prob: float
    loss_strength: float
    boundary: str
    exponents: np.ndarray

    @property
    def half_filling_gap(self) -> float:
        n = self.length // 2
        return float(self.exponents[n - 1] - self.exponents[n])


def random_loss_lyapunov(
    length: int,
    skin: float,
    measurement_prob: float,
    loss_strength: float,
    steps: int,
    dt: float,
    seed: int,
    boundary: str = "open",
    hopping: float = 1.0,
) -> LyapunovResult:
    """Compute QR Lyapunov exponents for random no-click loss layers.

    Each step applies A_t = D_t exp(-i H dt), where D_t has entries
    exp(-gamma dt / 2) on monitored sites and 1 otherwise.
    """
    rng = np.random.default_rng(seed)
    h = hatano_nelson_hamiltonian(length, hopping=hopping, skin=skin, boundary=boundary)
    u = single_particle_propagator(h, dt)
    q = np.eye(length, dtype=np.complex128)
    log_diag = np.zeros(length, dtype=float)

    for _ in range(steps):
        monitored = rng.random(length) < measurement_prob
        damping = np.ones(length, dtype=float)
        damping[monitored] = np.exp(-0.5 * loss_strength * dt)
        evolved = damping[:, None] * (u @ q)
        q, r = np.linalg.qr(evolved)
        diag = np.abs(np.diag(r))
        log_diag += np.log(np.maximum(diag, 1e-300))

    exponents = log_diag / (steps * dt)
    return LyapunovResult(
        length=length,
        skin=skin,
        measurement_prob=measurement_prob,
        loss_strength=loss_strength,
        boundary=boundary,
        exponents=exponents,
    )

