"""Two-sublattice non-Hermitian SSH/no-click baseline models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussian import half_chain_entropy, initial_slater_state, orthonormalize
from .models import single_particle_propagator


@dataclass
class NoClickSSHResult:
    cells: int
    length: int
    t1: float
    t2: float
    gain_loss: float
    skin: float
    boundary: str
    mean_entropy: float
    final_entropy: float
    spectral_imag_width: float
    spectral_real_width: float
    gbz_imag_width: float
    gbz_real_width: float


def ssh_no_click_hamiltonian(
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    skin: float = 0.0,
    boundary: str = "open",
) -> np.ndarray:
    """Return a non-Hermitian SSH single-particle Hamiltonian.

    Basis order is (A_0, B_0, A_1, B_1, ...).

    The no-click monitoring term is represented, after dropping the global
    decay, as +i*gain_loss on A sites and -i*gain_loss on B sites. In a
    monitored-particle/hole convention this gain_loss differs from the raw
    measurement rate by a harmless factor of two.

    `skin` adds nonreciprocity to the intercell link:
    B_j -> A_{j+1}: t2 exp(+skin), A_{j+1} -> B_j: t2 exp(-skin).
    """
    if cells < 2:
        raise ValueError("cells must be at least 2")
    if boundary not in {"open", "periodic"}:
        raise ValueError("boundary must be 'open' or 'periodic'")

    length = 2 * cells
    h = np.zeros((length, length), dtype=np.complex128)

    for cell in range(cells):
        a = 2 * cell
        b = a + 1
        h[a, a] = 1j * gain_loss
        h[b, b] = -1j * gain_loss

        h[b, a] = t1
        h[a, b] = t1

        if cell < cells - 1:
            a_next = 2 * (cell + 1)
            h[a_next, b] = t2 * np.exp(+skin)
            h[b, a_next] = t2 * np.exp(-skin)
        elif boundary == "periodic":
            a_next = 0
            h[a_next, b] = t2 * np.exp(+skin)
            h[b, a_next] = t2 * np.exp(-skin)

    return h


def ssh_bloch_hamiltonian(k: float, t1: float, t2: float, gain_loss: float, skin: float = 0.0) -> np.ndarray:
    """Bloch Hamiltonian for the periodic SSH model."""
    off_ba = t1 + t2 * np.exp(+skin) * np.exp(-1j * k)
    off_ab = t1 + t2 * np.exp(-skin) * np.exp(+1j * k)
    return np.array([[1j * gain_loss, off_ab], [off_ba, -1j * gain_loss]], dtype=np.complex128)


def reciprocal_pt_threshold(t1: float, t2: float) -> float:
    """Reciprocal SSH threshold where Bloch eigenvalues first become complex."""
    return abs(abs(t2) - abs(t1))


def bloch_spectral_widths(
    t1: float,
    t2: float,
    gain_loss: float,
    skin: float = 0.0,
    samples: int = 512,
) -> tuple[float, float]:
    """Return imaginary and real eigenvalue widths sampled over the Bloch BZ."""
    values = []
    for k in np.linspace(-np.pi, np.pi, samples, endpoint=False):
        values.extend(np.linalg.eigvals(ssh_bloch_hamiltonian(k, t1, t2, gain_loss, skin)))
    eig = np.asarray(values)
    return float(np.max(eig.imag) - np.min(eig.imag)), float(np.max(eig.real) - np.min(eig.real))


def gbz_spectral_widths(
    t1: float,
    t2: float,
    gain_loss: float,
    skin: float = 0.0,
    samples: int = 512,
) -> tuple[float, float]:
    """Return the GBZ spectral widths for the current one-parameter skin model.

    For the intercell-only nonreciprocity used here, the OBC GBZ substitution
    removes the explicit skin factor from the reciprocal SSH dispersion. This is
    the simplest test case where Bloch and non-Bloch criteria differ sharply.
    """
    _ = skin
    return bloch_spectral_widths(t1=t1, t2=t2, gain_loss=gain_loss, skin=0.0, samples=samples)


def run_no_click_ssh(
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    steps: int,
    dt: float,
    burn_in: int,
    skin: float = 0.0,
    boundary: str = "periodic",
    initial_pattern: str = "charge_density_wave",
    initial_seed: int | None = None,
) -> NoClickSSHResult:
    """Run deterministic no-click SSH dynamics for one half-filled Slater state."""
    length = 2 * cells
    orbitals = initial_slater_state(length, filling=0.5, pattern=initial_pattern, rng_seed=initial_seed)
    h = ssh_no_click_hamiltonian(cells, t1=t1, t2=t2, gain_loss=gain_loss, skin=skin, boundary=boundary)
    u = single_particle_propagator(h, dt)
    entropies: list[float] = []
    for step in range(steps):
        orbitals = orthonormalize(u @ orbitals)
        if step >= burn_in:
            entropies.append(half_chain_entropy(orbitals))
    imag_width, real_width = bloch_spectral_widths(t1=t1, t2=t2, gain_loss=gain_loss, skin=skin)
    gbz_imag_width, gbz_real_width = gbz_spectral_widths(t1=t1, t2=t2, gain_loss=gain_loss, skin=skin)
    arr = np.asarray(entropies, dtype=float)
    return NoClickSSHResult(
        cells=cells,
        length=length,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
        mean_entropy=float(np.mean(arr)),
        final_entropy=half_chain_entropy(orbitals),
        spectral_imag_width=imag_width,
        spectral_real_width=real_width,
        gbz_imag_width=gbz_imag_width,
        gbz_real_width=gbz_real_width,
    )
