"""Arbitrary-precision G, C and rho for the no-fit exterior kernel.

The returned Gauss quadrature uses the current mpmath context throughout.
Working precision agreement remains numerical evidence, not interval proof.
"""

from __future__ import annotations

import mpmath as mp


def parameters(gamma="2", t1=".5", t2="1"):
    gamma, t1, t2 = mp.mpf(str(gamma)), mp.mpf(str(t1)), mp.mpf(str(t2))
    a0, b = t1*t1+t2*t2, t1*t2
    return dict(gamma=gamma, t1=t1, t2=t2, a0=a0, b=b,
                a=(gamma*gamma-a0)/b, kappa=t2/t1,
                mu0=mp.sqrt(gamma*gamma-(t1+t2)**2))


def tables(p=None, nodes=128):
    p = p or parameters()
    points, legendre_weights = mp.gauss_quadrature(nodes, "legendre")
    result = dict(p=p, nodes=nodes, dps=mp.mp.dps, weights=[], band_root=[], cosine=[],
                  log_h_left=[], log_h_right=[])
    for point, weight in zip(points, legendre_weights):
        cosine = mp.cos((point+1)*mp.pi/2)
        u = (1+cosine)/(2*p["a"])
        factor = 2/(1+mp.sqrt(1-4*u*u))
        r = u*factor
        bottom = (mp.sqrt(p["a"]*p["kappa"])*factor*factor*(1-r*r)
                  /(mp.pi*(1+p["kappa"]*r)))
        result["weights"].append(weight/2)
        result["band_root"].append(mp.sqrt(p["a"]-2*cosine))
        result["cosine"].append(cosine)
        result["log_h_left"].append(mp.log(bottom))
        result["log_h_right"].append(mp.log(bottom*p["t1"]**2/(p["a0"]*u+p["b"])))
    return result


def functions(z, side, table=None):
    table = table or tables()
    p = table["p"]
    z = mp.mpf(z)
    root = mp.sqrt(p["a"]-z)
    rho = z/(2*p["a"]-z+2*mp.sqrt(p["a"]*(p["a"]-z)))
    potential = 2*mp.fsum(weight*mp.log(root+band_root)
                          for weight, band_root in zip(table["weights"], table["band_root"]))
    outer_log_square = mp.fsum(weight*(1-rho*rho)*log_h/(1-2*rho*cosine+rho*rho)
                               for weight, cosine, log_h in zip(table["weights"], table["cosine"],
                                                                table["log_h_"+side]))
    capacity = 1/(4*p["a"])
    if side == "left":
        prefactor = mp.exp(-outer_log_square)/(2*mp.pi*capacity*(1-rho)**3*(1+rho))
    elif side == "right":
        prefactor = mp.exp(-outer_log_square)/(2*mp.pi*capacity**2*(1-rho*rho)**3)
    else:
        raise ValueError("side must be left or right")
    return potential, prefactor, rho


def log_kernel(z, w, degree, side, table=None):
    table = table or tables()
    gz, cz, rz = functions(z, side, table)
    gw, cw, rw = functions(w, side, table)
    return (degree*(gz+gw)+(mp.log(cz)+mp.log(cw))/2
            +(mp.log1p(-rz*rz)+mp.log1p(-rw*rw))/2-mp.log1p(-rz*rw))


def core_parameters(table=None):
    table = table or tables()
    p = table["p"]
    g2, _, _ = functions(mp.mpf(2), "left", table)
    root = mp.sqrt(p["a"]-2)
    derivative = -mp.fsum(weight/(root*(root+band_root))
                          for weight, band_root in zip(table["weights"], table["band_root"]))
    tau = g2/(4*p["mu0"])
    beta = tau*p["b"]/(2*p["mu0"])+derivative/4
    return dict(G2=g2, Gprime2=derivative, tau=tau, beta=beta,
                first_response_coefficient=2*mp.sqrt(2*beta/mp.pi)/p["mu0"])
