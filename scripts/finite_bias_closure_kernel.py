"""No-fit finite-size predictor from the reviewed exterior strong kernel.

All finite-chain modes are retained. Only the convergent Szego kernel geometric
series is truncated, with its explicit positive trace-tail bound recorded.
This module never reads fitted times, reflected inverse Grams, or truth caches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import brentq
from scipy.special import roots_legendre


@dataclass(frozen=True)
class Parameters:
    gamma: float = 2.0
    t1: float = 0.5
    t2: float = 1.0

    @property
    def a0(self):
        return self.t1**2 + self.t2**2

    @property
    def b(self):
        return self.t1*self.t2

    @property
    def a(self):
        return (self.gamma**2-self.a0)/self.b

    @property
    def mu0(self):
        return math.sqrt(self.gamma**2-(self.t1+self.t2)**2)


def quadrature(nodes):
    points, weights = roots_legendre(nodes)
    return (points+1)*math.pi/2, weights/2


def kernel_functions(z, side, parameters, nodes=256):
    """G(z), C_s(z), and rho(z) in the exact strong-kernel formula."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    p = parameters
    angles, weights = quadrature(nodes)
    root = np.sqrt(p.a-z)
    band_root = np.sqrt(p.a-2*np.cos(angles))
    potential = 2*np.log(root[:, None]+band_root)@weights
    rho = z/(2*p.a-z+2*np.sqrt(p.a*(p.a-z)))
    u = (1+np.cos(angles))/(2*p.a)
    aa = 2/(1+np.sqrt(1-4*u*u))
    rr = u*aa
    bottom = (math.sqrt(p.a*p.t2/p.t1)*aa*aa*(1-rr*rr)
              /(math.pi*(1+p.t2/p.t1*rr)))
    if side == "right":
        bottom *= p.t1*p.t1/(p.a0*u+p.b)
    elif side != "left":
        raise ValueError("side must be left or right")
    poisson = ((1-rho[:, None]**2)
               /(1-2*rho[:, None]*np.cos(angles)+rho[:, None]**2))
    outer_square = np.exp(poisson@(weights*np.log(bottom)))
    capacity = 1/(4*p.a)
    if side == "left":
        prefactor = 1/(2*math.pi*capacity*outer_square*(1-rho)**3*(1+rho))
    else:
        prefactor = 1/(2*math.pi*capacity**2*outer_square*(1-rho*rho)**3)
    return potential, prefactor, rho


def coefficients(parameters=Parameters(), nodes=256):
    p = parameters
    angles, weights = quadrature(nodes)
    root = math.sqrt(p.a-2)
    g2 = float(kernel_functions([2], "left", p, nodes)[0][0])
    derivative = -float((1/(root*(root+np.sqrt(p.a-2*np.cos(angles)))))@weights)
    tau = g2/(4*p.mu0)
    beta = tau*p.b/(2*p.mu0)+derivative/4
    if beta <= 0 or g2 <= 2:
        raise ValueError("Require the strict filtering regime G_a(2)>2 and beta>0")
    return dict(mu0=p.mu0, G2=g2, Gprime2=derivative, tau=tau, beta=beta,
                log_coefficient=-3/(4*p.mu0),
                mean_endpoint_coefficient=2*math.sqrt(2*beta/math.pi),
                first_response_coefficient=2*math.sqrt(2*beta/math.pi)/p.mu0)


def asymptotic_time(length, side, epsilon, chi, parameters=Parameters(), nodes=256):
    p = parameters
    cc = coefficients(p, nodes)
    omega = 1.0 if side == "left" else (p.t1/(p.t1+p.t2))**2
    cs = float(kernel_functions([2], side, p, nodes)[1][0])
    ds = ((2*p.mu0**2/(p.gamma*(p.gamma+p.mu0)))
          *omega/(2*math.sqrt(math.pi)*(2*cc["beta"])**1.5))
    target = epsilon/math.sqrt(1-epsilon**2)
    constant = (math.log(cs*ds/target)-2*cc["G2"])/(2*p.mu0)
    s = 1 if side == "left" else -1
    base = cc["tau"]*length+cc["log_coefficient"]*math.log(length)+constant
    original = base-s*chi/(4*p.mu0)
    correction = s*cc["first_response_coefficient"]*chi/math.sqrt(length)
    return dict(zero_center=base, original_time=original, first_corrected_time=original+correction,
                first_shift_correction=correction, threshold_constant=constant)


def finite_modes(length, parameters=Parameters()):
    """Band sine recurrence plus the isolated edge mode; no division by edge zeta."""
    if length < 8 or length % 4:
        raise ValueError("length must be a multiple of four, at least eight")
    p, n = parameters, length//2
    ratio = p.t2/p.t1
    if ratio <= 1+1/n:
        raise ValueError("Require one isolated SSH edge singular mode")
    if p.gamma <= p.t1+p.t2:
        raise ValueError("Require gamma>t1+t2")
    v, u = np.zeros((n, n)), np.zeros((n, n))
    singular, signs, waves = np.zeros(n), np.zeros(n), np.zeros(n)
    cells = np.arange(1, n+1, dtype=float)
    for j in range(1, n):
        lo, hi = j*math.pi/(n+1), j*math.pi/n
        k = brentq(lambda x: math.sin((n+1)*x)+ratio*math.sin(n*x),
                   lo, hi, xtol=5e-15, rtol=9e-16)
        vv = np.sin(cells*k)
        vv /= np.linalg.norm(vv)
        zeta = math.sqrt(p.a0+2*p.b*math.cos(k))
        uu = p.t1*vv.copy()
        uu[1:] += p.t2*vv[:-1]
        uu /= zeta
        uu /= np.linalg.norm(uu)
        if uu[0] < 0:
            uu, vv = -uu, -vv
        sigma = 1.0 if float(uu@vv[::-1]) > 0 else -1.0
        u[:, j-1], v[:, j-1] = uu, vv
        singular[j-1], signs[j-1], waves[j-1] = zeta, sigma, k
    q0 = 1/ratio
    edge_function = lambda q: 1-ratio*q+ratio*q**(2*n+1)-q**(2*n+2)
    upper = (q0+1)/2
    q = q0 if edge_function(q0) < 1e-15 else brentq(edge_function, q0, upper, xtol=5e-15)
    distances = np.arange(n, dtype=float)
    vv = (-1.0)**distances*np.exp((n-1-distances)*math.log(q))*(1-np.exp((2*distances+2)*math.log(q)))
    vv /= np.linalg.norm(vv)
    sigma = (-1.0)**(n-1)
    uu = sigma*vv[::-1]
    # The edge scale may underflow; its vectors remain normalized and well defined.
    log_edge_scale = math.log(p.t1*(1-q*q))+(n-1)*math.log(q)-math.log1p(-q**(2*n))
    zeta = math.exp(log_edge_scale) if log_edge_scale > -745 else 0.0
    u[:, -1], v[:, -1], signs[-1], singular[-1], waves[-1] = uu, vv, sigma, zeta, math.nan
    mu = np.sqrt(p.gamma*p.gamma-singular*singular)
    ratios = singular/(p.gamma+mu)
    aa = 1/np.sqrt(1+ratios*ratios)
    plus = np.empty((length, n))
    plus[0::2], plus[1::2] = u*aa, -v*(aa*ratios)
    omega_plus = np.empty_like(plus)
    omega_plus[0::2], omega_plus[1::2] = plus[1::2][::-1], -plus[0::2][::-1]
    return dict(u=u, v=v, singular=singular, signs=signs, mu=mu, ratios=ratios,
                aa=aa, plus=plus, omega_plus=omega_plus, waves=waves,
                edge_q=q, edge_log_singular=log_edge_scale)


class FiniteStrongKernel:
    def __init__(self, length, side, chi, parameters=Parameters(), nodes=256, kernel_rank=10):
        self.length, self.side, self.chi, self.p = length, side, float(chi), parameters
        self.nodes, self.rank = nodes, kernel_rank
        if abs(chi) > 12:
            raise ValueError("This float64 weak-bias implementation is restricted to |chi|<=12")
        self.c = coefficients(parameters, nodes)
        self.modes = finite_modes(length, parameters)
        m = self.modes
        self.mu = m["mu"]
        self.z = (m["singular"]**2-parameters.a0)/parameters.b
        self.potential, prefactor, self.rho = kernel_functions(self.z, side, parameters, nodes)
        self.n = length//4-1
        boundary = m["v"][0]*m["signs"] if side == "left" else m["u"][0]
        self.amplitude = (self.mu/(parameters.gamma*m["aa"])*boundary*np.sqrt(prefactor))
        self.powers = self.rho[:, None]**np.arange(kernel_rank)
        self.geometric_factor = np.sqrt(1-self.rho*self.rho)[:, None]*self.powers
        self.positions = (np.arange(length)//2-(length//2-1)/2)/length
        self.weights = np.exp(chi*self.positions)
        self.q = m["plus"]
        self.oq = m["omega_plus"]
        self.metric = self.q.T@(self.weights[:, None]**2*self.q)
        self.metric = (self.metric+self.metric.T)/2
        self.metric_factor = cho_factor(self.metric, lower=True, check_finite=False)
        self.metric_cross = self.q.T@(self.weights[:, None]**2*self.oq)
        self.cchi = cho_solve(self.metric_factor, self.metric_cross, check_finite=False)
        self.cchi = (self.cchi+self.cchi.T)/2
        self.zz = m["signs"]*m["singular"]/self.mu
        self.frame_orthogonality = float(np.linalg.norm(self.q.T@self.q-np.eye(length//2), "fro"))
        self.lagrangian_residual = float(np.linalg.norm(self.q.T@self.oq, "fro"))

    def solve(self, time, detail=False):
        exponent = self.n*self.potential-self.mu*time
        if np.max(exponent) > 650:
            raise ArithmeticError("Time is outside the stable low-threshold region")
        vector = self.amplitude*np.exp(exponent)
        ff = vector[:, None]*self.geometric_factor
        inverse_metric_ff = cho_solve(self.metric_factor, ff, check_finite=False)
        ss = ff.T@inverse_metric_ff
        ss = (ss+ss.T)/2
        true_denominator = np.eye(self.rank)-ff.T@(self.zz[:, None]*ff)
        gauge_denominator = true_denominator-ff.T@self.cchi@ff
        gauge_denominator = (gauge_denominator+gauge_denominator.T)/2
        minimum = float(eigh(gauge_denominator, eigvals_only=True, check_finite=False)[0])
        if minimum <= 0:
            raise ArithmeticError("The strong-kernel true denominator is not positive")
        values, vectors = eigh(ss, gauge_denominator, check_finite=False)
        tangent = float(values[-1])
        if tangent < 0:
            raise ArithmeticError("Negative leading tangent")
        distance = tangent/math.hypot(1, tangent)
        if not detail:
            return distance
        active = vectors[:, -1]
        physical = self.weights*(self.q@(inverse_metric_ff@active))
        physical /= np.linalg.norm(physical)
        transverse = np.empty_like(physical)
        transverse[0::2], transverse[1::2] = physical[1::2][::-1], -physical[0::2][::-1]
        ymean = float(physical@(self.positions*physical))
        ycross = float(physical@(self.positions*transverse))
        generator_physical = self.apply_generator(physical)
        generator_transverse = self.apply_generator(transverse)
        geometry = -2*ymean+2*tangent*ycross
        rate = float(2*physical@generator_physical-tangent*physical@generator_transverse)
        dt_ff = -self.mu[:, None]*ff
        dt_ss = dt_ff.T@inverse_metric_ff+ff.T@cho_solve(self.metric_factor, dt_ff, check_finite=False)
        zzff = self.zz[:, None]*ff+self.cchi@ff
        dt_denominator = -(dt_ff.T@zzff+ff.T@(self.zz[:, None]*dt_ff+self.cchi@dt_ff))
        log_tangent_derivative = float(active@(dt_ss-tangent*dt_denominator)@active/tangent)
        s = 1 if self.side == "left" else -1
        mean_distance = self.length/4-0.5-s*self.length*ymean
        linear_trace = float(vector@vector)
        tail_bound = float(np.max(np.abs(self.rho))**(2*self.rank)*linear_trace)
        second = max(0.0, float(values[-2])) if self.rank>1 else 0.0
        return dict(time=float(time), distance=distance, tangent=tangent,
                    susceptibility=geometry/rate, geometry_factor=geometry,
                    decay_rate=rate, temporal_rate_from_differentiated_predictor=-log_tangent_derivative,
                    temporal_rate_consistency=abs(rate+log_tangent_derivative),
                    mean_endpoint_distance_cells=float(mean_distance), selected_position=ymean,
                    position_cross=ycross, principal_gap=distance-second/math.hypot(1, second),
                    true_denominator_minimum=float(eigh(true_denominator, eigvals_only=True, check_finite=False)[0]),
                    gauged_denominator_minimum=minimum,
                    gauged_denominator_condition=float(np.linalg.cond(gauge_denominator)),
                    metric_condition=float(np.linalg.cond(self.metric)),
                    positive_kernel_trace=linear_trace, geometric_series_operator_tail_bound=tail_bound,
                    omitted_diagonal_remainder_bound=float(math.exp(-2*self.p.mu0*time)),
                    all_finite_chain_modes=self.length//2, kernel_series_rank=self.rank,
                    frame_orthogonality_frobenius=self.frame_orthogonality,
                    frame_lagrangian_frobenius=self.lagrangian_residual)

    def apply_generator(self, orbital):
        p = self.p
        result = np.empty_like(orbital)
        result[0::2] = p.gamma*orbital[0::2]+p.t1*orbital[1::2]
        result[1::2] = -p.gamma*orbital[1::2]-p.t1*orbital[0::2]
        result[2::2] += p.t2*math.exp(self.chi/self.length)*orbital[1:-1:2]
        result[1:-1:2] -= p.t2*math.exp(-self.chi/self.length)*orbital[2::2]
        return result

    def crossing(self, epsilon):
        if not 0 < epsilon <= 0.3:
            raise ValueError("Require a fixed low threshold 0<epsilon<=.3")
        asymptotic = asymptotic_time(self.length, self.side, epsilon, self.chi, self.p, self.nodes)
        seed = asymptotic["first_corrected_time"]
        width = 2.0
        for _ in range(7):
            lo, hi = max(0.1, seed-width), seed+width
            if self.solve(lo) > epsilon and self.solve(hi) < epsilon:
                root = brentq(lambda duration: self.solve(duration)-epsilon, lo, hi, xtol=2e-10)
                return dict(length=self.length, initial=self.side, epsilon=float(epsilon), chi=self.chi,
                            **asymptotic, **self.solve(root, detail=True))
            width *= 2
        raise ArithmeticError("Failed to bracket the low-threshold crossing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="896,1024")
    parser.add_argument("--levels", default=".3,.05,.01")
    parser.add_argument("--chis", default="-4,0,4")
    parser.add_argument("--sides", default="left,right")
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--kernel-rank", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--t1", type=float, default=.5)
    parser.add_argument("--t2", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parameters = Parameters(args.gamma, args.t1, args.t2)
    records = []
    for length in map(int, args.lengths.split(",")):
        for side in args.sides.split(","):
            for chi in map(float, args.chis.split(",")):
                predictor = FiniteStrongKernel(length, side, chi, parameters, args.nodes, args.kernel_rank)
                for level in map(float, args.levels.split(",")):
                    records.append(predictor.crossing(level))
                print(f"predicted L={length} {side} chi={chi:g}", flush=True)
    source = Path(__file__)
    value = dict(status="analytic_no_fit_finite_kernel_prediction", created_utc=datetime.now(timezone.utc).isoformat(),
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 coefficients=coefficients(parameters, args.nodes), parameters=vars(parameters),
                 quadrature_nodes=args.nodes, records=records,
                 scope="Bounded chi, complete endpoint half-cell occupations and fixed low thresholds; reviewed strong-kernel sufficient conditions are required.",
                 scientific_error="The exact reflection-kernel replacement has asymptotic O(exp(-sigma*n)) error, n=L/4-1. No numerical constant in that bound is asserted.",
                 numerical_error="Float64 quadrature and all-mode recurrence require a separate nodes/rank check. This is not interval certification.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
