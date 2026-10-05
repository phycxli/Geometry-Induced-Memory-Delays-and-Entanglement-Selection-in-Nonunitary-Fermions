"""Trajectory simulation for monitored non-unitary free fermions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussian import half_chain_entropy, initial_product_state, measure_site, orthonormalize
from .models import hatano_nelson_hamiltonian, single_particle_propagator


@dataclass
class TrajectoryResult:
    length: int
    skin: float
    measurement_prob: float
    mean_entropy: float
    std_entropy: float
    final_entropy: float
    measurement_count: int


def run_trajectory(
    length: int,
    skin: float,
    measurement_prob: float,
    steps: int,
    dt: float,
    burn_in: int,
    seed: int,
    hopping: float = 1.0,
    boundary: str = "open",
    initial_pattern: str = "charge_density_wave",
) -> TrajectoryResult:
    """Run one stochastic Gaussian trajectory."""
    rng = np.random.default_rng(seed)
    orbitals = initial_product_state(length, filling=0.5, pattern=initial_pattern)
    h = hatano_nelson_hamiltonian(length, hopping=hopping, skin=skin, boundary=boundary)
    u = single_particle_propagator(h, dt)

    entropies: list[float] = []
    measurement_count = 0
    for step in range(steps):
        orbitals = orthonormalize(u @ orbitals)
        measured_sites = rng.random(length) < measurement_prob
        for site in np.flatnonzero(measured_sites):
            orbitals, _, _ = measure_site(orbitals, int(site), rng)
            measurement_count += 1
        if step >= burn_in:
            entropies.append(half_chain_entropy(orbitals))

    arr = np.asarray(entropies, dtype=float)
    final_entropy = half_chain_entropy(orbitals)
    return TrajectoryResult(
        length=length,
        skin=skin,
        measurement_prob=measurement_prob,
        mean_entropy=float(np.mean(arr)),
        std_entropy=float(np.std(arr)),
        final_entropy=float(final_entropy),
        measurement_count=measurement_count,
    )
