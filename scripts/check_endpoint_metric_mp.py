"""MP/FLINT half-axis metric from a scalar resolvent integral.

Every retained matrix entry is generated without a distance cutoff. Arb
controls arithmetic on the supplied decimal quadrature inputs; the input
integrals are numerical, so this is not a complete interval certificate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mpmath as mp
from flint import arb, arb_mat, ctx


def scalar_coefficients(maximum, dps, nodes):
    mp.mp.dps = dps
    gamma, t1, t2 = mp.mpf(2), mp.mpf(".5"), mp.mpf(1)
    a0, b, kappa = t1*t1+t2*t2, t1*t2, t2/t1
    aa = (gamma*gamma-a0)/b
    q = 2/(aa+mp.sqrt(aa*aa-4))
    points, weights = mp.gauss_quadrature(nodes, "legendre")
    terms, reflected = [mp.mpf(0)]*(maximum+1), [mp.mpf(0)]*(maximum+1)
    for point, weight in zip(points, weights):
        theta = (point+1)*mp.pi/4
        sine, cosine = mp.sin(theta), mp.cos(theta)
        r = q*sine*sine
        scale = (weight*q*cosine*cosine*mp.sqrt(1/q-r)
                 /(2*mp.sqrt(b)*(r+kappa)*(r+1/kappa)))
        reflection = (r+kappa)/(1+kappa*r)
        power = mp.mpf(1)
        for degree in range(maximum+1):
            contribution = scale*power
            terms[degree] += contribution
            reflected[degree] += contribution*reflection
            power *= r
    return terms, reflected


def rigorous_spatial_bound(h, cells):
    """Galerkin-tail bound; proof is supplied separately from this script."""
    h = mp.mpf(str(h))
    if not mp.mpf(".05") <= h <= mp.mpf(".1"):
        raise ValueError("The stated analytic constants use .05 <= |g| <= .1.")
    delta = mp.mpf(".5")
    rho_bound = mp.mpf(".72")
    inverse_bound = 1/(1-rho_bound*rho_bound)
    rr = mp.exp(-(delta-h))
    envelope_sum = 1+2*(rr/(1-rr)+rr/(1-rr)**2)
    forcing_bound = mp.mpf("1.21")
    solution_bound = forcing_bound*inverse_bound*envelope_sum
    operator_envelope = rho_bound*envelope_sum
    r = mp.exp(-2*h)
    n = mp.mpf(cells)
    tail = r**cells*((n+1)**2/(1-r)+2*(n+1)*r/(1-r)**2+r*(1+r)/(1-r)**3)
    coefficient = (operator_envelope**2*(2*solution_bound**2+forcing_bound*solution_bound)
                   +forcing_bound**2)
    return coefficient*tail


def metric_record(side, h, cells, dps, nodes, table):
    start = time.perf_counter()
    terms, reflected = table
    mp.mp.dps = dps
    ctx.dps = dps+15
    t1, t2 = mp.mpf(".5"), mp.mpf(1)
    magnitude = mp.mpf(str(h))
    exponent = [mp.exp(magnitude*j) for j in range(cells)]
    aa = 4/(2+mp.sqrt(mp.mpf("1.75")))

    def h_element(i, j):
        if side == "left":
            return terms[abs(i-j)]-terms[i+j+2]
        return terms[abs(i-j)]-reflected[i+j+1]

    matrix_entries = []
    for i in range(cells):
        row = []
        for j in range(cells):
            value = t1*h_element(i, j)
            if side == "left" and i:
                value += t2*h_element(i-1, j)
            elif side == "right":
                value += t2*h_element(i+1, j)
            row.append(arb(mp.nstr(value*exponent[i]/exponent[j], dps)))
        matrix_entries.append(row)
    tilted = arb_mat(matrix_entries)
    metric = tilted.transpose()*tilted
    for j in range(cells):
        metric[j, j] += 1
    forcing = []
    for j in range(cells):
        qj = mp.mpf(j+1) if side == "left" else mp.mpf(3*j+1)/9
        forcing.append([arb(mp.nstr(aa*qj/exponent[j], dps))])
    vector = arb_mat(forcing)
    solution = metric.solve(vector)
    result = (vector.transpose()*solution)[0, 0]
    residual = metric*solution-vector
    residual_upper = max(float(abs(residual[j, 0]).upper()) for j in range(cells))
    return dict(initial=side, magnitude=str(magnitude), cells=cells, dps=dps, quadrature_nodes=nodes,
                metric_midpoint=result.mid().str(dps-8, radius=False),
                arithmetic_ball=result.str(dps-8), arithmetic_relative_radius=str(result.rad()/abs(result.mid())),
                solve_residual_upper_float=residual_upper,
                analytical_spatial_truncation_bound=str(rigorous_spatial_bound(h, cells)),
                seconds=time.perf_counter()-start,
                method="Every retained half-axis entry from Stieltjes resolvent integral; native Arb solve; exact M0 q eigenfunction.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default="256,512")
    parser.add_argument("--magnitudes", default=".05,.1")
    parser.add_argument("--sides", default="left,right")
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = list(map(int, args.cells.split(",")))
    start = time.perf_counter()
    table = scalar_coefficients(2*max(lengths)+3, args.dps, args.nodes)
    table_seconds = time.perf_counter()-start
    records = []
    for cells in lengths:
        for h in args.magnitudes.split(","):
            for side in args.sides.split(","):
                record = metric_record(side, h, cells, args.dps, args.nodes, table)
                records.append(record)
                print(f"MP metric {side} h={h} cells={cells}: {record['metric_midpoint']} ({record['seconds']:.2f}s)", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(dict(status="partial", records=records), indent=2)+"\n", encoding="utf-8")
    source = Path(__file__)
    result = dict(status="independent_uncropped_half_axis_mp_check", created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), table_seconds=table_seconds,
                  records=records,
                  finite_space="Crop P_n S_infinity P_n after computing every entry, not f of a terminated Jacobi matrix.",
                  no_hard_cutoff="All off-diagonal distances inside the retained block are computed.",
                  mathematical_error="Analytic Galerkin bound covers spatial cropping. Quadrature and precision changes are numerical evidence. Arb balls cover arithmetic for supplied decimal entries only.",
                  scope="Main parameters; decreasing branch at fixed magnitudes .05/.1; numerical V_s constants; no truth dataset read.")
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
