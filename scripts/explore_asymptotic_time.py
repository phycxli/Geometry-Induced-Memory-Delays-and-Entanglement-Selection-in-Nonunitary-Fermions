"""Independent exploration; never edits the manuscript or earlier evidence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import gzip
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import mpmath as mp
from scipy.linalg import svd
import run_priority_strengthening as prior

OUT = ROOT/"data/prl_asymptotic_time_exploration"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def initialize():
    target = OUT/"protected_baseline.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    files = sorted(p for name in ("main", "supplement", "main_zh", "supplment_zh", "figures")
                   for p in (ROOT/name).rglob("*") if p.is_file())
    value = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                 scope="Exploration only. No manuscript, figure, earlier source or dataset mutations.",
                 files=[dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                             sha256=digest(p)) for p in files])
    save(target, value)
    return dict(status="protected", files=len(files))


def verify():
    baseline = json.loads((OUT/"protected_baseline.json").read_text(encoding="utf-8"))
    failures = [item["path"] for item in baseline["files"]
                if not (ROOT/item["path"]).exists() or digest(ROOT/item["path"]) != item["sha256"]]
    value = dict(status="pass" if not failures else "fail", checked=len(baseline["files"]),
                 failures=failures, checked_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/"protected_verification.json", value)
    assert not failures, failures
    return value


def omega(length):
    result = np.zeros((length, length))
    n = length//2
    for j in range(n):
        result[2*j, 2*(n-1-j)+1] = 1
        result[2*j+1, 2*(n-1-j)] = -1
    return result


def diagnostic(family, initial, chi, level, seed):
    graph = prior.graph_for(family, initial, chi)
    duration = prior.root(graph, level, seed) if chi else seed
    k_matrix = graph.graph(duration)
    u, singular, vh = svd(k_matrix, check_finite=False)
    p = graph.gain@vh.conj().T[:, 0]
    r = graph.perp@u[:, 0]
    response = asdict(graph.response(duration))
    om = omega(family["length"])
    alternate = om@graph.gain
    overlap = np.real(np.vdot(r, om@p))
    symplectic_graph = alternate.conj().T@graph.perp@k_matrix
    symp_sign = 1. if overlap >= 0 else -1.
    ypos = graph.positions
    backward_p = np.exp(-chi*ypos)*p
    backward_p /= np.linalg.norm(backward_p)
    data = np.load(prior.OUT/"cache"/(family["key"]+".npz"))
    spectral = np.abs(data["plus"].conj().T@backward_p)**2
    mu0 = np.sqrt(family["gamma"]**2-(family["t1"]+family["t2"])**2)
    n = family["length"]//2
    s = 1 if initial == "left" else -1
    p_margin = (n-1)/2-s*(np.arange(family["length"])//2-(n-1)/2)
    r_margin = (n-1)/2+s*(np.arange(family["length"])//2-(n-1)/2)
    p_prob, r_prob = np.abs(p)**2, np.abs(r)**2
    row = dict(family=family["key"], length=family["length"], initial=initial, chi=chi,
               level=level, time=duration, time_per_length=duration/family["length"], **response,
               omega_pair_error=float(np.linalg.norm(r-symp_sign*om@p)),
               gain_lagrangian_error=float(np.linalg.norm(graph.gain.conj().T@om@graph.gain, 2)),
               occupied_lagrangian_error=float(np.linalg.norm(graph.frame(duration).conj().T@om@graph.frame(duration), 2)),
               graph_symmetry_error=float(np.linalg.norm(symplectic_graph-symplectic_graph.T, 2)),
               hamiltonian_error=float(np.linalg.norm(graph.generator.T@om+om@graph.generator, 2)),
               position_pair_error=float(abs(response["selected_position"]+response["transverse_position"])),
               p_margin_mean=float(p_prob@p_margin), r_margin_mean=float(r_prob@r_margin),
               p_margin_second=float(p_prob@(p_margin*p_margin)),
               r_margin_second=float(r_prob@(r_margin*r_margin)),
               spectral_mean=float(spectral@data["mu"]),
               spectral_excess=float(spectral@(data["mu"]-mu0)),
               spectral_second=float(spectral@((data["mu"]-mu0)**2)),
               spectral_norm=float(np.sum(spectral)),
               rate_cross=float(singular[0]*np.real(np.vdot(p, graph.generator@r))),
               mu0=float(mu0), conjectured=float(-s/(4*mu0)),
               scaled_susceptibility_error=float(np.sqrt(family["length"])*(response["susceptibility"]+s/(4*mu0))))
    key = f"{family['key']}_{initial}_chi{chi:g}_eps{level:g}"
    (OUT/"vectors").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/"vectors"/(key+".npz"), p=p, r=r, mu=data["mu"],
                        spectral=spectral, positions=ypos, singular=singular)
    return row


def diagnose():
    initialize()
    with (prior.OUT/"zero_bias_response.csv").open(encoding="utf-8", newline="") as stream:
        baseline = list(csv.DictReader(stream))
    rows = []
    tick = time.perf_counter()
    for family in prior.config()["families"]:
        for initial in ("left", "right"):
            for level in (.3, .05, .01):
                root = next(row for row in baseline if row["family"] == family["key"]
                            and row["initial"] == initial and float(row["level"]) == level)
                for chi in (0., -4., 4.):
                    rows.append(diagnostic(family, initial, chi, level, float(root["zero_time"])))
        print(f"diagnosed {family['key']}: {time.perf_counter()-tick:.2f}s", flush=True)
    with (OUT/"diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    maxima = {key:max(abs(row[key]) for row in rows) for key in
              ("omega_pair_error", "gain_lagrangian_error", "graph_symmetry_error",
               "hamiltonian_error", "position_pair_error")}
    value = dict(status="diagnostic_only", count=len(rows), maxima=maxima, seconds=time.perf_counter()-tick,
                 old_data_policy="Every tested family and bias has already been studied; none is an unseen validation.",
                 source_sha256=digest(Path(__file__)))
    save(OUT/"diagnostic_summary.json", value)
    return value


def reflected_spectral(family, precision):
    """Half-size reflected solves, retaining every mode of the finite chain."""
    mp.mp.dps = precision
    length = family["length"]
    n, m = length//2, length//4
    gamma = mp.mpf(str(family["gamma"]))
    u, singular, v = prior.fast_coupling(length, family["t1"], family["t2"])
    mu = [mp.sqrt(gamma*gamma-z*z) for z in singular]
    ratios = [z/(gamma+w) for z, w in zip(singular, mu)]
    aa = [1/mp.sqrt(1+r*r) for r in ratios]
    phases = [mp.sign(mp.fsum(u[j,k]*v[n-1-j,k] for j in range(n))) for k in range(n)]
    plus, minus = mp.matrix(length, n), mp.matrix(length, n)
    for k in range(n):
        for j in range(n):
            plus[2*j,k], plus[2*j+1,k] = aa[k]*u[j,k], -aa[k]*ratios[k]*v[j,k]
            minus[2*j,k], minus[2*j+1,k] = aa[k]*ratios[k]*u[j,k], -aa[k]*v[j,k]
    states, grams = {}, {}
    for name in ("left", "right"):
        occupied = list(range(m)) if name == "left" else list(range(m,n))
        empty = list(range(m,n)) if name == "left" else list(range(m))
        ucross = mp.matrix([[u[j,k]*ratios[k] for k in range(n)] for j in empty])
        vcross = mp.matrix([[v[j,k] for k in range(n)] for j in occupied])
        cross = ucross*vcross.T
        gram = mp.matrix([[cross[i,m-1-j] if name == "left" else cross[m-1-i,j]
                           for j in range(m)] for i in range(m)])
        weights = mp.matrix([[v[m-1-i,k] if name == "left" else v[m+i,k]
                              for k in range(n)] for i in range(m)])
        inverse = mp.inverse(gram)
        kernel = weights.T*inverse*weights
        f = mp.matrix([[-kernel[i,j]*aa[j]/aa[i]*(1-ratios[j]**2)*phases[j]
                        -(ratios[j] if i == j else 0) for j in range(n)] for i in range(n)])
        residual = prior.old.p1.mp_norm(inverse*gram-mp.eye(m))
        symmetry = prior.old.p1.mp_norm(gram-gram.T)/prior.old.p1.mp_norm(gram)
        assert residual < mp.mpf("1e-40"), (name, residual)
        assert symmetry < mp.mpf("1e-50"), (name, symmetry)
        states[name] = dict(F=prior.old.p1.encode_matrix(f), inverse_residual=str(residual),
                            gram_symmetry_relative=str(symmetry))
        grams[name] = gram
    value = dict(**family, dps=precision, mu=[str(x) for x in mu],
                 plus=prior.old.p1.encode_matrix(plus), minus=prior.old.p1.encode_matrix(minus),
                 states=states, kernel="finite coupling recurrence and two reflected m-by-m inverses; all L/2 modes")
    return value, dict(u=u, v=v, mu=mu, ratios=ratios, phases=phases, aa=aa, grams=grams)


def structure():
    initialize()
    rows = []
    tick = time.perf_counter()
    for length in (24, 32, 48, 64):
        family = dict(key=f"structure_L{length}", length=length, gamma=2., t1=.5, t2=1.)
        raw, details = reflected_spectral(family, 130)
        independent = prior.spectral_value(family, 130)
        n, m = length//2, length//4
        q = (mp.mpf("5.5")-mp.sqrt(mp.mpf("5.5")**2-4))/2
        h = (1+mp.sqrt(1-q))**4/(4*q)
        degree = m-1
        roots = [q*(1+mp.cos((2*j-1)*mp.pi/(2*degree)))/2 for j in range(1,degree+1)]
        polynomial = [mp.mpf(1)]
        for root in roots:
            xroot = root+1/root
            product = [mp.mpf(0)]*(len(polynomial)+2)
            for i, coefficient in enumerate(polynomial):
                product[i] += coefficient
                product[i+1] -= xroot*coefficient
                product[i+2] += coefficient
            polynomial = product
        coeff = [polynomial[i]-(polynomial[i-2] if i >= 2 else 0) for i in range(m)]
        p_two = mp.fprod(2-r-1/r for r in roots)
        transform = mp.eye(m)
        for i in range(m-1):
            transform[i,i+1] = 2
        for name in ("left", "right"):
            f = prior.old.spatial.decode(raw["states"][name]["F"])
            direct = prior.old.spatial.decode(independent["states"][name]["F"])
            error = prior.old.p1.mp_norm(f-direct)/prior.old.p1.mp_norm(direct)
            gram = details["grams"][name]
            basis = mp.matrix(coeff) if name == "left" else mp.inverse(transform.T)*mp.matrix(coeff)
            gram_norm = (basis.T*gram*basis)[0]
            ratio = p_two*p_two/gram_norm
            asserted_lower = h**(2*degree)/4
            assert error < mp.mpf("1e-60"), error
            assert ratio >= asserted_lower, (name, ratio, asserted_lower)
            modal_errors = []
            for k in range(n):
                coupling_squared = 4-details["mu"][k]**2
                x = (coupling_squared-mp.mpf("1.25"))/mp.mpf(".5")
                expected = mp.fprod(x-r-1/r for r in roots)
                if name == "left":
                    actual = mp.fsum(details["v"][m-1-i,k]*basis[i] for i in range(m))
                    expected *= details["v"][0,k]
                else:
                    actual = mp.fsum(details["v"][m+i,k]*basis[i] for i in range(m))
                    expected *= details["phases"][k]*details["u"][0,k]
                modal_errors.append(abs(actual-expected)/(1+abs(expected)))
            assert max(modal_errors) < mp.mpf("1e-60"), max(modal_errors)
            rows.append(dict(length=length, initial=name, reconstruction_relative=str(error),
                             modal_polynomial_relative=str(max(modal_errors)),
                             trial_log_ratio=str(mp.log(ratio)),
                             trial_log_lower=str(mp.log(asserted_lower)),
                             inverse_residual=raw["states"][name]["inverse_residual"]))
        print(f"structure L{length}: {time.perf_counter()-tick:.2f}s", flush=True)
    value = dict(status="pass", rows=rows, seconds=time.perf_counter()-tick,
                 policy="Independent full occupied-coordinate solve versus reflected half-size inverse; 130 decimal digits.")
    save(OUT/"structure_checks.json", value)
    return value


def freeze():
    initialize()
    target = OUT/"unseen_validation_lock.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    with (OUT/"diagnostics.csv").open(encoding="utf-8", newline="") as stream:
        diagnostics = list(csv.DictReader(stream))
    selected = [row for row in diagnostics if row["family"].startswith("L")
                and float(row["chi"]) == 0. and float(row["level"]) == .05]
    design, observations = [], []
    for row in selected:
        s = 1 if row["initial"] == "left" else -1
        length = float(row["length"])
        design.append([1., 1/np.sqrt(length) if s == 1 else 0., 1/np.sqrt(length) if s == -1 else 0.])
        observations.append(s*float(row["scaled_susceptibility_error"]))
    coefficients = np.linalg.lstsq(design, observations, rcond=None)[0]
    mu0 = np.sqrt(1.75)
    predictions = []
    for length in (640,768):
        for initial, s, index in (("left",1,1),("right",-1,2)):
            correction = (coefficients[0]+coefficients[index]/np.sqrt(length))/np.sqrt(length)
            predictions.append(dict(length=length, initial=initial, level=.05,
                                    susceptibility=float(-s/(4*mu0)+s*correction)))
    value = dict(locked_utc=datetime.now(timezone.utc).isoformat(), lengths=[640,768],
                 precision_rule="ceil((1.3*L+30)/20)*20, and 40 additional decimal digits",
                 levels=[.3,.05,.01], chis=[-4.,0.,4.],
                 candidate="dt_epsilon/dchi -> -s/(4*mu0); fixed epsilon and bounded chi, L=4m",
                 policy="The limiting coefficient is derived in the separate proof draft; the finite correction fit is empirical.",
                 empirical_coefficients=dict(common=float(coefficients[0]), left=float(coefficients[1]), right=float(coefficients[2])),
                 empirical_predictions=predictions, empirical_susceptibility_gate=.0003,
                 cache_change_gate=2e-10, small_structure_gate="1e-60",
                 known_input_sha256=digest(OUT/"diagnostics.csv"), source_sha256=digest(Path(__file__)))
    save(target, value)
    return value


def prepare(index):
    lock = json.loads((OUT/"unseen_validation_lock.json").read_text(encoding="utf-8"))
    length = lock["lengths"][index//2]
    precision = 20*int(np.ceil((1.3*length+30)/20))+40*(index%2)
    target = OUT/"spectral"/f"L{length}_dps{precision}.raw.json.gz"
    if target.exists():
        return dict(status="reused", length=length, precision=precision, sha256=digest(target))
    tick = time.perf_counter()
    family = dict(key=f"L{length}", length=length, gamma=2., t1=.5, t2=1.)
    raw, _ = reflected_spectral(family, precision)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        json.dump(raw, stream, ensure_ascii=False)
    value = dict(status="prepared", length=length, precision=precision,
                 seconds=time.perf_counter()-tick, sha256=digest(target), source_sha256=digest(Path(__file__)))
    save(target.with_name(target.name.replace(".raw.json.gz", ".json")), value)
    return value


def unseen(index):
    lock = json.loads((OUT/"unseen_validation_lock.json").read_text(encoding="utf-8"))
    length = lock["lengths"][index]
    precision = 20*int(np.ceil((1.3*length+30)/20))
    sources = [OUT/"spectral"/f"L{length}_dps{p}.raw.json.gz" for p in (precision,precision+40)]
    arrays = []
    for source in sources:
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        arrays.append(prior.convert_spectral(raw))
    maximum = 0.
    for key in arrays[0]:
        finite = np.isfinite(arrays[0][key]) & np.isfinite(arrays[1][key])
        assert np.array_equal(np.isfinite(arrays[0][key]), np.isfinite(arrays[1][key]))
        maximum = max(maximum, float(np.max(np.abs(arrays[0][key][finite]-arrays[1][key][finite]))))
    assert maximum < lock["cache_change_gate"], maximum
    data = arrays[-1]
    (OUT/"cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/"cache"/f"L{length}.npz", **data)
    from mietf_skin.response import GaugedGraph
    rows = []
    for initial in ("left", "right"):
        for chi in lock["chis"]:
            graph = GaugedGraph(data["plus"], data["minus"], data["mu"], data[initial+"_log"],
                                data[initial+"_sign"], chi)
            seed = .54*length-2
            for level in lock["levels"]:
                seed = prior.root(graph, level, seed)
                response = asdict(graph.response(seed))
                s = 1 if initial == "left" else -1
                u, singular, vh = svd(graph.graph(seed), check_finite=False)
                p, r = graph.gain@vh.conj().T[:,0], graph.perp@u[:,0]
                margin = (length//2-1)/2-s*(np.arange(length)//2-(length//2-1)/2)
                backward = np.exp(-chi*graph.positions)*p
                backward /= np.linalg.norm(backward)
                spectral = np.abs(data["plus"].conj().T@backward)**2
                row = dict(length=length, initial=initial, chi=chi, level=level, time=seed,
                           **response, p_margin_mean=float((np.abs(p)**2)@margin),
                           spectral_excess=float(spectral@(data["mu"]-np.sqrt(1.75))))
                if chi == 0 and level == .05:
                    predicted = next(x for x in lock["empirical_predictions"]
                                     if x["length"] == length and x["initial"] == initial)
                    row["empirical_prediction"] = predicted["susceptibility"]
                    row["empirical_error"] = abs(response["susceptibility"]-predicted["susceptibility"])
                    row["empirical_pass"] = row["empirical_error"] <= lock["empirical_susceptibility_gate"]
                rows.append(row)
    value = dict(status="computed", length=length, rows=rows, maximum_cache_change=maximum,
                 precision_digits=[precision, precision+40], input_sha256=[digest(source) for source in sources],
                 lock_sha256=digest(OUT/"unseen_validation_lock.json"), source_sha256=digest(Path(__file__)))
    save(OUT/"unseen"/f"L{length}.json", value)
    return dict(status=value["status"], length=length, count=len(rows), maximum_cache_change=maximum,
                empirical_errors=[r["empirical_error"] for r in rows if "empirical_error" in r])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("initialize", "diagnose", "verify", "structure", "freeze", "prepare", "unseen"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    value = globals()[args.stage](args.index) if args.stage in ("prepare", "unseen") else globals()[args.stage]()
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
