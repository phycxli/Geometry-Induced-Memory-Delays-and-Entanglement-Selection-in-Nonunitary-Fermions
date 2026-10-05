"""Isospectral counterexamples for singular-subspace entanglement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagnostics import normalized_product, participation_lengths, ssh_step_matrix
from .gaussian import half_chain_entropy
from .models import single_particle_propagator
from .ssh import bloch_spectral_widths, gbz_spectral_widths, ssh_no_click_hamiltonian


@dataclass
class IsospectralResult:
    cells: int
    length: int
    t1: float
    t2: float
    gain_loss: float
    skin_base: float
    boundary: str
    dt: float
    steps: int
    alpha: float
    gauge_condition: float
    eig_max_abs_diff: float
    eig_hausdorff_abs_diff: float
    eig_rel_l2_diff: float
    trace_moment_max_abs_diff: float
    eig_imag_width: float
    eig_real_width: float
    bloch_imag_width: float
    gbz_imag_width: float
    singular_log_gap: float
    singular_ratio: float
    left_entropy: float
    left_entropy_density: float
    left_entropy_over_log_l: float
    right_entropy: float
    right_entropy_density: float
    left_pr_occ_mean: float
    left_pr_occ_over_l: float
    right_pr_occ_mean: float
    right_pr_occ_over_l: float


@dataclass
class PhysicalIsospectralResult:
    cells: int
    length: int
    t1: float
    t2: float
    gain_loss: float
    skin: float
    boundary: str
    dt: float
    steps: int
    local_similarity_condition: float
    h_eig_hausdorff_abs_diff: float
    h_trace_moment_max_abs_diff: float
    product_eig_hausdorff_abs_diff: float
    product_trace_moment_max_abs_diff: float
    gbz_imag_width: float
    bloch_imag_width: float
    singular_log_gap: float
    singular_ratio: float
    left_entropy: float
    left_entropy_density: float
    left_entropy_over_log_l: float
    right_entropy: float
    right_entropy_density: float
    left_pr_occ_mean: float
    left_pr_occ_over_l: float
    right_pr_occ_mean: float
    right_pr_occ_over_l: float


def diagonal_similarity(matrix: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Return D(alpha) M D(alpha)^-1 and cond(D).

    The diagonal profile is centered so that alpha changes the left/right
    singular geometry without adding a global scale to D.
    """
    length = matrix.shape[0]
    sites = np.arange(length, dtype=float) - 0.5 * (length - 1)
    log_d = alpha * sites
    diagonal = np.exp(log_d)
    inverse = np.exp(-log_d)
    transformed = (diagonal[:, None] * matrix) * inverse[None, :]
    condition = float(np.exp(np.max(log_d) - np.min(log_d)))
    return transformed, condition


def isospectral_family(
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    skin_base: float,
    boundary: str,
    dt: float,
    steps: int,
    alphas: list[float],
) -> list[IsospectralResult]:
    """Scan diagonal similarities of one finite-time SSH product."""
    step = ssh_step_matrix(
        cells=cells,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin=skin_base,
        boundary=boundary,
        dt=dt,
    )
    product = normalized_product(step, steps)
    base_eig = np.linalg.eigvals(product)
    base_trace_moments = trace_moments(product, count=4)
    rows: list[IsospectralResult] = []
    bloch_imag, _ = bloch_spectral_widths(t1=t1, t2=t2, gain_loss=gain_loss, skin=skin_base)
    gbz_imag, _ = gbz_spectral_widths(t1=t1, t2=t2, gain_loss=gain_loss, skin=skin_base)
    for alpha in alphas:
        transformed, condition = diagonal_similarity(product, alpha)
        rows.append(
            isospectral_diagnostics(
                matrix=transformed,
                base_eig=base_eig,
                base_trace_moments=base_trace_moments,
                cells=cells,
                t1=t1,
                t2=t2,
                gain_loss=gain_loss,
                skin_base=skin_base,
                boundary=boundary,
                dt=dt,
                steps=steps,
                alpha=alpha,
                gauge_condition=condition,
                bloch_imag_width=bloch_imag,
                gbz_imag_width=gbz_imag,
            )
        )
    return rows


def physical_skin_isospectral_family(
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    boundary: str,
    dt: float,
    steps: int,
    skins: list[float],
) -> list[PhysicalIsospectralResult]:
    """Scan the local OBC SSH skin family H(g)=D_g H(0) D_g^-1.

    For open boundaries and intercell-only nonreciprocity, changing ``skin`` is
    a local imaginary-gauge similarity transform.  The OBC Hamiltonian spectrum
    and GBZ spectrum are fixed, while singular vectors of the finite-time
    product need not be fixed.
    """
    if boundary != "open":
        raise ValueError("the physical isospectral skin family is only exact for open boundaries")
    base_h = ssh_no_click_hamiltonian(
        cells=cells,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin=0.0,
        boundary=boundary,
    )
    base_step = single_particle_propagator(base_h, dt)
    base_product = normalized_product(base_step, steps)
    base_h_eig = np.linalg.eigvals(base_h)
    base_product_eig = np.linalg.eigvals(base_product)
    base_h_trace = trace_moments(base_h, count=4)
    base_product_trace = trace_moments(base_product, count=4)

    rows: list[PhysicalIsospectralResult] = []
    for skin in skins:
        h = ssh_no_click_hamiltonian(
            cells=cells,
            t1=t1,
            t2=t2,
            gain_loss=gain_loss,
            skin=skin,
            boundary=boundary,
        )
        step = single_particle_propagator(h, dt)
        product = normalized_product(step, steps)
        rows.append(
            physical_isospectral_diagnostics(
                h=h,
                product=product,
                base_h_eig=base_h_eig,
                base_product_eig=base_product_eig,
                base_h_trace=base_h_trace,
                base_product_trace=base_product_trace,
                cells=cells,
                t1=t1,
                t2=t2,
                gain_loss=gain_loss,
                skin=skin,
                boundary=boundary,
                dt=dt,
                steps=steps,
            )
        )
    return rows


def physical_isospectral_diagnostics(
    h: np.ndarray,
    product: np.ndarray,
    base_h_eig: np.ndarray,
    base_product_eig: np.ndarray,
    base_h_trace: np.ndarray,
    base_product_trace: np.ndarray,
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
    steps: int,
) -> PhysicalIsospectralResult:
    length = h.shape[0]
    particles = length // 2
    h_eig = np.linalg.eigvals(h)
    product_eig = np.linalg.eigvals(product)
    h_trace_diff = trace_moments(h, count=len(base_h_trace)) - base_h_trace
    product_trace_diff = trace_moments(product, count=len(base_product_trace)) - base_product_trace
    left, singular_values, vh = np.linalg.svd(product, full_matrices=True)
    singular_values = np.maximum(singular_values, 1e-300)
    right = vh.conj().T
    left_top = left[:, :particles]
    right_top = right[:, :particles]
    left_pr = participation_lengths(left)
    right_pr = participation_lengths(right)
    occ = slice(0, particles)
    bloch_imag, _ = bloch_spectral_widths(t1=t1, t2=t2, gain_loss=gain_loss, skin=skin)
    gbz_imag, _ = gbz_spectral_widths(t1=t1, t2=t2, gain_loss=gain_loss, skin=skin)
    left_entropy = half_chain_entropy(left_top)
    right_entropy = half_chain_entropy(right_top)
    return PhysicalIsospectralResult(
        cells=cells,
        length=length,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin=float(skin),
        boundary=boundary,
        dt=dt,
        steps=steps,
        local_similarity_condition=float(np.exp(abs(skin) * max(cells - 1, 0))),
        h_eig_hausdorff_abs_diff=spectral_hausdorff_distance(base_h_eig, h_eig),
        h_trace_moment_max_abs_diff=float(np.max(np.abs(h_trace_diff))),
        product_eig_hausdorff_abs_diff=spectral_hausdorff_distance(base_product_eig, product_eig),
        product_trace_moment_max_abs_diff=float(np.max(np.abs(product_trace_diff))),
        gbz_imag_width=gbz_imag,
        bloch_imag_width=bloch_imag,
        singular_log_gap=float(np.log(singular_values[particles - 1]) - np.log(singular_values[particles])),
        singular_ratio=float(singular_values[particles] / singular_values[particles - 1]),
        left_entropy=left_entropy,
        left_entropy_density=left_entropy / length,
        left_entropy_over_log_l=left_entropy / np.log(length),
        right_entropy=right_entropy,
        right_entropy_density=right_entropy / length,
        left_pr_occ_mean=float(np.mean(left_pr[occ])),
        left_pr_occ_over_l=float(np.mean(left_pr[occ]) / length),
        right_pr_occ_mean=float(np.mean(right_pr[occ])),
        right_pr_occ_over_l=float(np.mean(right_pr[occ]) / length),
    )


def isospectral_diagnostics(
    matrix: np.ndarray,
    base_eig: np.ndarray,
    base_trace_moments: np.ndarray,
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    skin_base: float,
    boundary: str,
    dt: float,
    steps: int,
    alpha: float,
    gauge_condition: float,
    bloch_imag_width: float,
    gbz_imag_width: float,
) -> IsospectralResult:
    """Compute spectrum-preserving and singular-geometry diagnostics."""
    length = matrix.shape[0]
    particles = length // 2
    eig = np.linalg.eigvals(matrix)
    eig_distance = spectral_hausdorff_distance(base_eig, eig)
    eig_sorted_diff = np.sort_complex(eig) - np.sort_complex(base_eig)
    eig_norm = np.linalg.norm(base_eig)
    moment_diff = trace_moments(matrix, count=len(base_trace_moments)) - base_trace_moments
    scaled = matrix / max(np.linalg.norm(matrix), 1e-300)
    left, singular_values, vh = np.linalg.svd(scaled, full_matrices=True)
    singular_values = np.maximum(singular_values, 1e-300)
    right = vh.conj().T
    left_top = left[:, :particles]
    right_top = right[:, :particles]
    left_pr = participation_lengths(left)
    right_pr = participation_lengths(right)
    occ = slice(0, particles)
    singular_ratio = float(singular_values[particles] / singular_values[particles - 1])
    singular_log_gap = float(np.log(singular_values[particles - 1]) - np.log(singular_values[particles]))
    left_entropy = half_chain_entropy(left_top)
    right_entropy = half_chain_entropy(right_top)
    return IsospectralResult(
        cells=cells,
        length=length,
        t1=t1,
        t2=t2,
        gain_loss=gain_loss,
        skin_base=skin_base,
        boundary=boundary,
        dt=dt,
        steps=steps,
        alpha=float(alpha),
        gauge_condition=float(gauge_condition),
        eig_max_abs_diff=eig_distance,
        eig_hausdorff_abs_diff=eig_distance,
        eig_rel_l2_diff=float(np.linalg.norm(eig_sorted_diff) / max(eig_norm, 1e-300)),
        trace_moment_max_abs_diff=float(np.max(np.abs(moment_diff))),
        eig_imag_width=float(np.max(eig.imag) - np.min(eig.imag)),
        eig_real_width=float(np.max(eig.real) - np.min(eig.real)),
        bloch_imag_width=bloch_imag_width,
        gbz_imag_width=gbz_imag_width,
        singular_log_gap=singular_log_gap,
        singular_ratio=singular_ratio,
        left_entropy=left_entropy,
        left_entropy_density=left_entropy / length,
        left_entropy_over_log_l=left_entropy / np.log(length),
        right_entropy=right_entropy,
        right_entropy_density=right_entropy / length,
        left_pr_occ_mean=float(np.mean(left_pr[occ])),
        left_pr_occ_over_l=float(np.mean(left_pr[occ]) / length),
        right_pr_occ_mean=float(np.mean(right_pr[occ])),
        right_pr_occ_over_l=float(np.mean(right_pr[occ]) / length),
    )


def spectral_hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the symmetric nearest-neighbor spectral distance."""
    directed_ab = max(float(np.min(np.abs(value - b))) for value in a)
    directed_ba = max(float(np.min(np.abs(value - a))) for value in b)
    return max(directed_ab, directed_ba)


def trace_moments(matrix: np.ndarray, count: int) -> np.ndarray:
    """Return Tr M^k, k=1,...,count, as stable similarity invariants."""
    out = []
    power = np.eye(matrix.shape[0], dtype=np.complex128)
    for _ in range(count):
        power = power @ matrix
        out.append(np.trace(power))
    return np.asarray(out, dtype=np.complex128)
