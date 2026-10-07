"""Finite-time geometric response of a simple occupied-space principal angle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr, solve, svd


@dataclass(frozen=True)
class AngleResponse:
    distance: float
    tangent: float
    principal_gap: float
    selected_position: float
    transverse_position: float
    position_cross: float
    geometry_factor: float
    decay_rate: float
    susceptibility: float
    distance_chi_derivative: float
    distance_time_derivative: float
    principal_residual: float


def ssh_real_generator(length: int, skin: float = 0.0, gamma: float = 2.0,
                       t1: float = 0.5, t2: float = 1.0) -> np.ndarray:
    """Generator after a fixed sublattice phase rotation; units are J/hbar."""
    result = np.diag(np.tile([gamma, -gamma], length // 2))
    for j in range(length // 2):
        result[2*j, 2*j+1], result[2*j+1, 2*j] = t1, -t1
        if j:
            result[2*j, 2*j-1] = t2*np.exp(skin)
            result[2*j-1, 2*j] = -t2*np.exp(-skin)
    return result


def contraction_constants(gamma: float, t1: float, t2: float,
                          skin: float) -> dict[str, float]:
    h = abs(skin)
    beta = t1*t1+t2*t2+2*t1*t2*np.cosh(h)
    if beta >= gamma*gamma:
        raise ValueError("The separated-graph series is not controlled.")
    rho = (t1+t2*np.exp(h))/(gamma+np.sqrt(gamma*gamma-beta))
    eta = (1-rho*rho)/(1+rho*rho)
    a, b = gamma*eta-t2*np.sinh(h), 2*(t1+t2*np.cosh(h))
    if rho >= 1 or a <= 0:
        raise ValueError("The uniform numerical-range bound is not positive.")
    k = 2*a/b
    return dict(rho=float(rho), eta=float(eta), a=float(a), b=float(b),
                cone=float(k/np.sqrt(1+k*k)), tail_ratio=float(beta/gamma**2),
                cdw_spectral_graph_bound=float(np.sqrt(1+rho*rho)/eta))


class GaugedGraph:
    """Full-rank propagation using signed logarithms of a precision-checked F.

    No modal truncation is applied. This float64 evaluation is intended for
    the well-conditioned low-threshold region and requires precision checks.
    """

    def __init__(self, plus: np.ndarray, minus: np.ndarray, mu: np.ndarray,
                 logabs: np.ndarray, signs: np.ndarray, chi: float,
                 gamma: float = 2.0, t1: float = 0.5, t2: float = 1.0):
        self.length, self.n = plus.shape
        self.positions = (np.arange(self.length)//2-(self.n-1)/2)/self.length
        weights = np.exp(chi*self.positions)
        self.plus, self.minus = weights[:, None]*plus, weights[:, None]*minus
        full, r = qr(self.plus, mode="full", check_finite=False)
        self.gain, self.perp, self.r = full[:, :self.n], full[:, self.n:], r[:self.n]
        self.cross = self.gain.conj().T@self.minus
        self.bottom = self.perp.conj().T@self.minus
        self.mu, self.logabs, self.signs = mu, logabs, signs
        self.generator = ssh_real_generator(self.length, chi/self.length, gamma, t1, t2)

    def graph(self, time: float) -> np.ndarray:
        exponent = self.logabs-time*(self.mu[:, None]+self.mu[None, :])
        if np.max(exponent) > 650:
            raise ArithmeticError("The trial time precedes the stable graph region.")
        ft = self.signs*np.exp(exponent)
        upper, lower = self.r+self.cross@ft, self.bottom@ft
        return solve(upper.T, lower.T, check_finite=False).T

    def distance(self, time: float) -> float:
        k = float(svd(self.graph(time), compute_uv=False, check_finite=False)[0])
        return k/np.hypot(1, k)

    def response(self, time: float) -> AngleResponse:
        u, singular, vh = svd(self.graph(time), check_finite=False)
        k = float(singular[0])
        d, c = k/np.hypot(1, k), 1/np.hypot(1, k)
        p, r = self.gain@vh.conj().T[:, 0], self.perp@u[:, 0]
        xp = float(np.real(np.vdot(p, self.positions*p)))
        xr = float(np.real(np.vdot(r, self.positions*r)))
        cross = float(np.real(np.vdot(p, self.positions*r)))
        geometric = xr-xp-2*k*cross
        rate = float(np.real(np.vdot(p, self.generator@p)
                            -np.vdot(r, self.generator@r)
                            +k*np.vdot(p, self.generator@r)))
        second = float(singular[1]/np.hypot(1, singular[1])) if len(singular)>1 else 0.0
        residual = float(np.linalg.norm(self.perp.conj().T@(self.generator@p)))
        return AngleResponse(d, k, d-second, xp, xr, cross, geometric, rate,
                             geometric/rate, d*(1-d*d)*geometric,
                             -d*(1-d*d)*rate, residual)

    def frame(self, time: float) -> np.ndarray:
        exponent = self.logabs-time*(self.mu[:, None]+self.mu[None, :])
        return self.plus+self.minus@(self.signs*np.exp(exponent))
