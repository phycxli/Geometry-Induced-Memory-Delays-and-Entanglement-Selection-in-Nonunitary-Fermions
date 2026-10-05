"""Diagnostics for non-unitary one-step free-fermion circuits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussian import half_chain_entropy
from .models import single_particle_propagator
from .ssh import bloch_spectral_widths, gbz_spectral_widths, ssh_no_click_hamiltonian


@dataclass
class CircuitDiagnosticResult:
    cells: int
    length: int
    t1: float
    t2: float
    gain_loss: float
    skin: float
    boundary: str
    dt: float
    diagnostic_steps: int
    qr_lyapunov_gap: float
    qr_lyapunov_spread: float
    sv_lyapunov_gap: float
    right_pr_cut: float
    right_pr_occ_mean: float
    right_pr_occ_median: float
    right_pr_occ_over_l: float
    left_pr_cut: float
    left_pr_occ_mean: float
    left_pr_occ_over_l: float
    right_subspace_entropy: float
    right_subspace_entropy_density: float
    left_subspace_entropy: float
    left_subspace_entropy_density: float
    bloch_imag_width: float
    gbz_imag_width: float


def ssh_step_matrix(
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
) -> np.ndarray:
    """Return one no-click SSH circuit step U = exp(-i H_eff dt)."""
    h = ssh_no_click_hamiltonian(
        cells=cells,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
    )
    return single_particle_propagator(h, dt)


def qr_lyapunov_exponents(step_matrix: np.ndarray, steps: int, dt: float) -> np.ndarray:
    """Compute finite-time QR Lyapunov exponents for repeated circuit steps."""
    length = step_matrix.shape[0]
    q = np.eye(length, dtype=np.complex128)
    log_diag = np.zeros(length, dtype=float)
    for _ in range(steps):
        q, r = np.linalg.qr(step_matrix @ q)
        diag = np.maximum(np.abs(np.diag(r)), 1e-300)
        log_diag += np.log(diag)
    exponents = log_diag / (steps * dt)
    return np.sort(exponents)[::-1]


def normalized_product(step_matrix: np.ndarray, steps: int) -> np.ndarray:
    """Return a scalar-normalized finite-time product.

    The scalar normalization prevents overflow and does not affect singular
    vectors or singular-value gaps.
    """
    length = step_matrix.shape[0]
    product = np.eye(length, dtype=np.complex128)
    for _ in range(steps):
        product = step_matrix @ product
        norm = np.linalg.norm(product)
        if norm == 0.0:
            raise FloatingPointError("finite-time product underflowed to zero")
        product = product / norm
    return product


def participation_lengths(vectors: np.ndarray) -> np.ndarray:
    """Return inverse-participation lengths for column vectors."""
    weights = np.abs(vectors) ** 2
    weights_sum = np.sum(weights, axis=0, keepdims=True)
    weights = weights / np.maximum(weights_sum, 1e-300)
    return 1.0 / np.maximum(np.sum(weights**2, axis=0), 1e-300)


def ssh_circuit_diagnostics(
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
    diagnostic_steps: int,
) -> CircuitDiagnosticResult:
    """Compute Lyapunov and singular-vector diagnostics for one SSH circuit."""
    length = 2 * cells
    half = length // 2
    step = ssh_step_matrix(
        cells=cells,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
        dt=dt,
    )

    qre = qr_lyapunov_exponents(step, steps=diagnostic_steps, dt=dt)
    qr_gap = float(qre[half - 1] - qre[half])
    qr_spread = float(qre[0] - qre[-1])

    product = normalized_product(step, diagnostic_steps)
    left, singular_values, vh = np.linalg.svd(product, full_matrices=True)
    singular_values = np.maximum(singular_values, 1e-300)
    sv_gap = float((np.log(singular_values[half - 1]) - np.log(singular_values[half])) / (diagnostic_steps * dt))

    right = vh.conj().T
    right_pr = participation_lengths(right)
    left_pr = participation_lengths(left)
    cut_slice = slice(max(0, half - 1), min(length, half + 1))
    occ_slice = slice(0, half)
    bloch_imag, _ = bloch_spectral_widths(t1=t1, t2=t2, gain_loss=gain_loss, skin=skin)
    gbz_imag, _ = gbz_spectral_widths(t1=t1, t2=t2, gain_loss=gain_loss, skin=skin)

    return CircuitDiagnosticResult(
        cells=cells,
        length=length,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin=skin,
        boundary=boundary,
        dt=dt,
        diagnostic_steps=diagnostic_steps,
        qr_lyapunov_gap=qr_gap,
        qr_lyapunov_spread=qr_spread,
        sv_lyapunov_gap=sv_gap,
        right_pr_cut=float(np.mean(right_pr[cut_slice])),
        right_pr_occ_mean=float(np.mean(right_pr[occ_slice])),
        right_pr_occ_median=float(np.median(right_pr[occ_slice])),
        right_pr_occ_over_l=float(np.mean(right_pr[occ_slice]) / length),
        left_pr_cut=float(np.mean(left_pr[cut_slice])),
        left_pr_occ_mean=float(np.mean(left_pr[occ_slice])),
        left_pr_occ_over_l=float(np.mean(left_pr[occ_slice]) / length),
        right_subspace_entropy=half_chain_entropy(right[:, occ_slice]),
        right_subspace_entropy_density=half_chain_entropy(right[:, occ_slice]) / length,
        left_subspace_entropy=half_chain_entropy(left[:, occ_slice]),
        left_subspace_entropy_density=half_chain_entropy(left[:, occ_slice]) / length,
        bloch_imag_width=bloch_imag,
        gbz_imag_width=gbz_imag,
    )
