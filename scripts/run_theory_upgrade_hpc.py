"""Theory extension: continuous contraction, transition widths and reflected Gram."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mpmath as mp
import numpy as np

import run_selection_front_hpc as old
import run_fine_front_hpc as fine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prl_theory_upgrade"
SOURCE = Path(__file__)
write_json, read_json = old.write_json, old.read_json
write_gzip, read_gzip = old.write_gzip, old.read_gzip
now, emit = old.now, old.emit
LEVELS = ("0.9", "0.5", "0.3", "0.1", "0.05", "0.01")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def constants(g):
    h = abs(mp.mpf(str(g)))
    rho = (mp.mpf(".5") + mp.exp(h)) / (2 + mp.sqrt(4 - mp.mpf("1.25") - mp.cosh(h)))
    a = 2 * (1 - rho**2) / (1 + rho**2) - mp.sinh(h)
    b = 2 * (mp.mpf(".5") + mp.cosh(h))
    return dict(rho=rho, a=a, b=b, kappa_cone=2*a/b, cdw_initial_rate=2*a-b*rho)


def candidate_config():
    families = []
    for length in (128, 160, 192, 224, 256):
        for g in ("0", "0.25"):
            key = f"L{length}_g{g}".replace(".", "p")
            historic = ROOT / ("data/prl_p3_selection_front" if length == 128 else "data/prl_p3_new_sizes") / "static"
            source = historic / (f"L{length}_gL0.raw.json.gz" if g == "0" else f"L{length}_g0p25.raw.json.gz")
            families.append(dict(key=key, length=length, skin=g, role="development" if length <= 192 else "new_size",
                                 historic=source.relative_to(ROOT).as_posix() if length <= 192 else None,
                                 digits=[200, 240] if length == 128 else [240, 280] if length <= 192 else [320, 360]))
    return dict(version=1, date="2026-10-05", families=families,
                gram_sizes=[224, 256, 320], gram_initials=["left", "right"],
                levels=list(LEVELS), maximum_root_rank=24, root_time_tolerance="0.000002",
                projector_error_target="0.00001", new_time_offsets=["-0.08", "0.08"],
                candidate_policy="old sizes are development; affine time and log-overlap extrapolations freeze before new-size statics",
                reference_policy="independent structured exponential from a fresh coupling SVD; two precisions; small direct expm checks",
                scope="SSH gamma2 OBC, |g|<=.25; unique continuous crossings only inside the proven cone; no asymptotic cutoff claim")


def config():
    return read_json(OUT / "config.json")


def initialize():
    for folder in ("hpc", "static", "curves", "gram", "references", "direct_contexts", "failures", "code", "development"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    cfg = candidate_config()
    if (OUT / "config.json").exists():
        assert config() == cfg
    else:
        write_json(OUT / "config.json", cfg)
    inputs = [ROOT / f["historic"] for f in cfg["families"] if f["historic"]]
    inputs.extend([ROOT / "data/prl_p3_new_sizes/reflected_geometry.csv", ROOT / "main/main.tex", ROOT / "supplement/supplement.tex"])
    if not (OUT / "baseline_manifest.json").exists():
        write_json(OUT / "baseline_manifest.json", [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in inputs])
    with mp.workdps(60):
        rows = [{"skin": g, **{k: str(v) for k, v in constants(g).items()}} for g in ("0", "0.25")]
        assert all(mp.mpf(r["cdw_initial_rate"]) > 0 for r in rows)
        write_json(OUT / "theorem_constants.json", dict(rows=rows, formula="D+||T|| <= -2a||T||+b||T||^2", scope=cfg["scope"]))
    return dict(status="initialized", families=len(cfg["families"]))


def load_static(family):
    path = ROOT / family["historic"] if family["historic"] else OUT / "static" / (family["key"] + ".raw.json.gz")
    return read_gzip(path)[-2:]


def static_job(family):
    if family["historic"]:
        return dict(status="historic_input", family=family["key"])
    assert (OUT / "extrapolation_lock.json").exists()
    path = OUT / "static" / (family["key"] + ".json")
    if path.exists():
        return dict(status="reused", family=family["key"])
    tick, values = time.perf_counter(), []
    names = old.initials(family["length"])
    old.spatial.right_solve = old.cached_right_solve
    for digits in family["digits"]:
        with mp.workdps(digits):
            value = old.spatial.static_reference((family["length"], 2., family["skin"]), digits)
            value = old.augment(value, names)
        values.append(value)
        write_json(path.with_suffix(".progress.json"), dict(dps=digits, seconds=time.perf_counter()-tick))
    change = old.static_change(*values, names)
    assert max(change["relative_scalar"], change["additional_relative"]) < 1e-8
    assert max(change["matrix_absolute"], change["gain_entropy"]) < 1e-10
    write_gzip(path.with_suffix(".raw.json.gz"), values)
    write_json(path, dict(family=family, convergence=change, source_sha256=sha(SOURCE), completed_utc=now(), seconds=time.perf_counter()-tick))
    return dict(status="new_static_converged", family=family["key"], seconds=time.perf_counter()-tick)


def compact_graph(ctx, duration, rank, factors=False):
    n, matrices = len(ctx["mu"]), ctx["matrices"]
    rank = min(rank, n)
    decay = [mp.exp(-x*duration) for x in ctx["mu"]]
    a = mp.matrix([[decay[i]*ctx["u"][i,j]*mp.sqrt(ctx["sig"][j]) for j in range(rank)] for i in range(n)])
    v = mp.matrix([[mp.sqrt(ctx["sig"][i])*ctx["vh"][i,j]*decay[j] for j in range(n)] for i in range(rank)]) * ctx["rinv"]
    b = matrices["Bperp"]*a
    h = mp.eye(rank) + v*(matrices["Ccross"]*a)
    left = b*mp.inverse(h)
    _, rleft = mp.qr(left, mode="skinny")
    _, rright = mp.qr(v.T, mode="skinny")
    singular = mp.svd(rleft*rright.T, compute_uv=False)
    kappa = singular[0]
    distance = kappa/mp.sqrt(1+kappa*kappa)
    return (distance, a, v, left) if factors else distance


def bracket_root(ctx, level, rank):
    length = ctx["static"]["length"]
    lo, hi = mp.mpf(".15")*length, mp.mpf(".8")*length
    target = mp.mpf(level)
    assert compact_graph(ctx, lo, rank) > target and compact_graph(ctx, hi, rank) < target
    while hi-lo > mp.mpf("0.000002"):
        middle = (lo+hi)/2
        if compact_graph(ctx, middle, rank) > target:
            lo = middle
        else:
            hi = middle
    return (lo+hi)/2, lo, hi


def certificate(ctx, duration, rank):
    # The compact candidate needs a full signed residual before becoming a guarantee.
    tc = old.time_context(ctx, duration)
    values, _, _ = fine.at_rank(tc, min(rank, len(ctx["mu"])))
    c = constants(ctx["static"]["skin"])
    upper = min(mp.mpf(1), values["distance_to_G"]+values["projector_remainder"])
    lower = max(mp.mpf(0), values["distance_to_G"]-values["projector_remainder"])
    k = upper/mp.sqrt(1-upper**2) if upper < 1 else mp.inf
    rate = 2*c["a"]-c["b"]*k
    static = ctx["static"]
    z = mp.mpf(static["q"])*mp.exp(-2*min(ctx["mu"])*duration)
    ew = z/(mp.mpf(static["p"])-z) if z < mp.mpf(static["p"]) else mp.mpf(1)
    return {**values, "lower_to_G": lower, "upper_to_G": upper,
            "output_to_G_bound": min(mp.mpf(1), ew), "persistent_cone_certified": bool(rate > 0),
            "certified_log_kappa_rate": max(mp.mpf(0), rate)}


def curves_job(family):
    path = OUT / "curves" / (family["key"] + ".json")
    if path.exists():
        return dict(status="reused", family=family["key"])
    if family["role"] == "new_size":
        assert (OUT / "extrapolation_lock.json").exists()
    tick, rows, precision = time.perf_counter(), [], []
    statics = load_static(family)
    names = old.initials(family["length"])[1:]
    results = []
    for static in statics:
        local = {}
        with mp.workdps(static["dps"]):
            old.RANKS = tuple(x for x in (0, 1, 2, 4, 6, 8, 12, 16, 24) if x <= static["length"]//2)
            contexts = old.context(static, names)
            for name in names:
                for level in LEVELS:
                    local[name, level] = bracket_root(contexts[name], level, 12)[0]
        results.append(local)
        precision.append(static["dps"])
        write_json(path.with_suffix(".progress.json"), dict(family=family["key"], root_digits=precision, elapsed=time.perf_counter()-tick))
    static = statics[-1]
    with mp.workdps(static["dps"]):
        for (name, level), root in results[-1].items():
            change = abs(root-results[0][name, level])
            assert change < mp.mpf(".000004"), (name, level, change)
            ctx = contexts[name]
            cert = certificate(ctx, root, 12)
            rank = 12
            if cert["projector_remainder"] > mp.mpf(".00001"):
                rank = 24
                root = bracket_root(ctx, level, rank)[0]
                cert = certificate(ctx, root, rank)
            if cert["projector_remainder"] > mp.mpf(".00001"):
                status = "candidate_only_remainder_failed"
            else:
                status = "certified_radius" if cert["persistent_cone_certified"] else "sampled_large_angle_branch"
            rows.append(dict(family_key=family["key"], length=family["length"], skin=family["skin"], role=family["role"], initial=name,
                             level=level, root_time_mp=str(root), root_time=float(root), rank=rank, status=status,
                             precision_change=float(change), certificate={k:bool(v) if isinstance(v,bool) else str(v) for k,v in cert.items()}))
            write_json(path.with_suffix(".progress.json"), dict(completed=len(rows), elapsed=time.perf_counter()-tick))
    write_json(path, dict(family=family, rows=rows, precision_digits=precision, source_sha256=sha(SOURCE), completed_utc=now(), seconds=time.perf_counter()-tick))
    return dict(status="curves_completed", family=family["key"], roots=len(rows), seconds=time.perf_counter()-tick)


def extrapolation_lock():
    path = OUT / "extrapolation_lock.json"
    if path.exists():
        return dict(status="reused")
    assert not list((OUT / "static").glob("*.json"))
    values = [read_json(OUT / "curves" / (f["key"]+".json")) for f in config()["families"] if f["role"] == "development"]
    rows = [r for v in values for r in v["rows"]]
    time_predictions = []
    for family in [f for f in config()["families"] if f["role"] == "new_size"]:
        for name in old.initials(family["length"])[1:]:
            kind = "block1" if name.startswith("block_cell") and int(name[10:]) == family["length"]//16 else "block3" if name.startswith("block_cell") else name
            for level in LEVELS:
                selected = []
                for r in rows:
                    initial = r["initial"]
                    row_kind = "block1" if initial.startswith("block_cell") and int(initial[10:]) == r["length"]//16 else "block3" if initial.startswith("block_cell") else initial
                    if row_kind == kind and r["skin"] == family["skin"] and r["level"] == level:
                        selected.append(r)
                x, y = np.array([r["length"] for r in selected]), np.array([r["root_time"] for r in selected])
                coef = np.polyfit(x, y, 1)
                value = float(np.polyval(coef, family["length"]))
                time_predictions.append(dict(family_key=family["key"], initial=name, level=level, predicted_time=value,
                                             coefficients=coef.tolist(), training_sizes=x.tolist(), abs_time_gate=.15))
    import csv
    with (ROOT / "data/prl_p3_new_sizes/reflected_geometry.csv").open(encoding="utf-8", newline="") as stream:
        old_geometry = list(csv.DictReader(stream))
    gram_predictions = []
    for initial in ("left", "right"):
        for g in (0., .25):
            selected = [r for r in old_geometry if r["initial"] == initial and float(r["skin"]) == g and int(r["length"]) >= 96]
            x = np.array([int(r["length"]) for r in selected], float)
            y = np.array([float(r["log_delta"]) for r in selected])
            coefficients = np.linalg.lstsq(np.column_stack((x, np.log(x), np.ones(len(x)))), y, rcond=None)[0]
            for length in config()["gram_sizes"]:
                gram_predictions.append(dict(length=length, initial=initial, skin=g,
                                             predicted_log_delta=float(np.array([length, math.log(length), 1])@coefficients),
                                             coefficients=coefficients.tolist(), abs_log_gate=.1, training_sizes=x.tolist()))
    write_json(path, dict(locked_utc=now(), time_predictions=time_predictions, gram_predictions=gram_predictions,
                         source_sha256=sha(SOURCE), policy="empirical extrapolations; failure is retained; no inferred asymptotic theorem"))
    return dict(status="extrapolations_locked", time_predictions=len(time_predictions), gram_predictions=len(gram_predictions))


def gram_job(index):
    cfg = config()
    length, initial = cfg["gram_sizes"][index//2], cfg["gram_initials"][index%2]
    assert (OUT / "extrapolation_lock.json").exists()
    path = OUT / "gram" / f"L{length}_{initial}.json"
    if path.exists():
        return dict(status="reused")
    tick, raw, evaluations = time.perf_counter(), [], []
    dps0 = max(320, 2*length)
    for nodes, digits in ((length//2+32, dps0), (length//2+64, dps0+40)):
        _, matrix = old.base.spatial_gram(length, 0., initial, nodes, digits, True)
        local = []
        with mp.workdps(digits):
            inverse = mp.inverse(matrix)
            sign, m = (1 if initial == "left" else -1), matrix.rows
            for g in (mp.mpf("-.25"), mp.mpf(0), mp.mpf(2)/length, mp.mpf(4)/length, mp.mpf(".25")):
                j = mp.matrix([[inverse[i,k]*mp.exp(-sign*g*(i+k+1)) for k in range(m)] for i in range(m)])
                a1 = mp.fsum(j[i,i] for i in range(m))
                a2 = mp.fsum(v*v for v in j)
                p = [j[i,i]/a1 for i in range(m)]
                mean = mp.fsum(i*p[i] for i in range(m))
                variance = mp.fsum((i-mean)**2*p[i] for i in range(m))
                eigen = mp.eigsy(j, eigvals_only=True)
                delta = 1/eigen[m-1]
                local.append(dict(length=length, initial=initial, skin=str(g), log_delta=str(mp.log(delta)),
                                  lower_mp=str(1/a1), upper_mp=str(a1/a2), inverse_purity=str(a2/a1**2),
                                  theta=str(mp.log(delta*a1)), mean_over_L=str(mean/length), variance_over_L2=str(variance/length**2),
                                  trace_skin_slope=str(sign*(1+2*mean)), diagonal_distribution=[str(v) for v in p]))
            raw.append(dict(nodes=nodes, digits=digits, gram=old.p1.encode_matrix(matrix), inverse=old.p1.encode_matrix(inverse), rows=local))
        evaluations.append(local)
        write_json(path.with_suffix(".progress.json"), dict(nodes=nodes, digits=digits, seconds=time.perf_counter()-tick))
    with mp.workdps(dps0+60):
        change = max(abs(mp.mpf(a["log_delta"])-mp.mpf(b["log_delta"])) for a,b in zip(*evaluations))
        assert change < mp.mpf("1e-8"), change
    write_gzip(path.with_suffix(".raw.json.gz"), raw)
    write_json(path, dict(rows=evaluations[-1], precision_change_mp=str(change), source_sha256=sha(SOURCE), completed_utc=now(), seconds=time.perf_counter()-tick))
    return dict(status="gram_converged", length=length, initial=initial, seconds=time.perf_counter()-tick)


def direct_context(length, g):
    n = length//2
    c = mp.matrix(n)
    for j in range(n):
        c[j,j] = mp.mpf(".5")
        if j:
            c[j,j-1] = 1
    u, singular, vh = mp.svd(c)
    mu = [mp.sqrt(4-x*x) for x in singular]
    a = [mp.sqrt((2+x)/4) for x in mu]
    b = [s/(4*aa) for s,aa in zip(singular,a)]
    gain = mp.matrix(length, n)
    for i in range(n):
        for k in range(n):
            gain[2*i,k] = mp.exp(g*i)*u[i,k]*a[k]
            gain[2*i+1,k] = -mp.exp(g*i)*vh[k,i]*b[k]
    qfull = mp.qr(gain)[0]
    return dict(u=u, v=vh.T, mu=mu, singular=list(singular), gain=qfull[:,:n], perpendicular=qfull[:,n:], skin=g)


def structured_exponential(ctx, duration):
    u, v, mu, g = ctx["u"], ctx["v"], ctx["mu"], ctx["skin"]
    n, maximum = u.rows, max(mu)
    plus = [mp.exp((x-maximum)*duration) for x in mu]
    minus = [mp.exp((-x-maximum)*duration) for x in mu]
    co = [(x+y)/2 for x,y in zip(plus,minus)]
    si = [(x-y)/(2*m) for x,y,m in zip(plus,minus,mu)]
    aa = u*mp.diag([x+2*y for x,y in zip(co,si)])*u.T
    bb = v*mp.diag([x-2*y for x,y in zip(co,si)])*v.T
    ab = u*mp.diag([s*y for s,y in zip(ctx["singular"],si)])*v.T
    ba = -ab.T
    result = mp.matrix(2*n)
    for i in range(n):
        for j in range(n):
            scale = mp.exp(g*(i-j))
            result[2*i,2*j], result[2*i+1,2*j+1] = scale*aa[i,j], scale*bb[i,j]
            result[2*i,2*j+1], result[2*i+1,2*j] = scale*ab[i,j], scale*ba[i,j]
    return result


def direct_context_job(family):
    assert (OUT / "reference_lock.json").exists()
    path = OUT / "direct_contexts" / (family["key"]+".json")
    if path.exists():
        return dict(status="reused", family=family["key"])
    tick, raw = time.perf_counter(), []
    for digits in family["digits"]:
        with mp.workdps(digits):
            ctx = direct_context(family["length"], mp.mpf(family["skin"]))
            generator = old.previous.generator(dict(length=family["length"],gamma=2.,skin=family["skin"],t1=.5,t2=1.))
            a11 = ctx["gain"].T*generator*ctx["gain"]
            a22 = ctx["perpendicular"].T*generator*ctx["perpendicular"]
            sym_plus = mp.eigsy((a11+a11.T)/2,eigvals_only=True)[0]
            sym_minus = mp.eigsy((a22+a22.T)/2,eigvals_only=True)[a22.rows-1]
            c = constants(family["skin"])
            assert sym_plus >= c["a"]-mp.mpf("1e-50") and sym_minus <= -c["a"]+mp.mpf("1e-50")
            raw.append(dict(dps=digits, matrices={k:old.p1.encode_matrix(ctx[k]) for k in ("u","v","gain","perpendicular")},
                            mu=[str(x) for x in ctx["mu"]], singular=[str(x) for x in ctx["singular"]], skin=str(ctx["skin"]),
                            gain_entropy=str(old.previous.correlation_entropy(ctx["gain"])[0]),
                            hermitian_gain_min=str(sym_plus), hermitian_perpendicular_max=str(sym_minus)))
    write_gzip(path.with_suffix(".raw.json.gz"), raw)
    write_json(path, dict(family=family, source_sha256=sha(SOURCE), prediction_lock_sha256=sha(OUT/"reference_lock.json"),
                         completed_utc=now(), seconds=time.perf_counter()-tick))
    return dict(status="independent_context_completed", family=family["key"], seconds=time.perf_counter()-tick)


def kernel_checks():
    rows = []
    with mp.workdps(80):
        for length in (8, 12, 16):
            for g in (mp.mpf("-.25"), mp.mpf(0), mp.mpf(".25")):
                ctx = direct_context(length, g)
                case = dict(length=length, gamma=2., skin=str(g), t1=.5, t2=1.)
                generator = old.previous.generator(case)
                for duration in (mp.mpf(".7"), mp.mpf(3)):
                    expected = mp.expm(generator*duration)*mp.exp(-max(ctx["mu"])*duration)
                    actual = structured_exponential(ctx, duration)
                    error = old.p1.mp_norm(expected-actual)
                    assert error < mp.mpf("1e-65"), (length,g,error)
                    rows.append(dict(length=length, skin=str(g), time=str(duration), error_mp=str(error)))
    write_json(OUT / "development/kernel_checks.json", dict(rows=rows, status="passed", source_sha256=sha(SOURCE), verified_utc=now()))
    return dict(status="kernel_passed", checks=len(rows))


def lock_references():
    path = OUT / "reference_lock.json"
    if path.exists():
        return dict(status="reused")
    assert not list((OUT / "references").glob("*.json"))
    assert read_json(OUT / "development/kernel_checks.json")["status"] == "passed"
    cases = []
    for family in config()["families"]:
        if family["role"] != "new_size":
            continue
        curves = read_json(OUT / "curves" / (family["key"]+".json"))
        for row in curves["rows"]:
            if row["level"] not in ("0.3", "0.1", "0.05"):
                continue
            for offset in config()["new_time_offsets"]:
                duration = str(float(row["root_time"])+float(offset))
                cases.append(dict(case_id=f'{family["key"]}_{row["initial"]}_d{row["level"]}_o{offset}'.replace(".","p").replace("-","m"),
                                  family_key=family["key"], length=family["length"], skin=family["skin"], initial=row["initial"],
                                  duration=duration, level=row["level"], prediction_root=row["root_time"], digits=family["digits"],
                                  initial_error= row["certificate"]["projector_remainder"]))
    kernels = [p for p in (ROOT/"scripts").glob("*.py") if p.name.startswith(("run_", "audit_"))]
    write_json(path, dict(locked_utc=now(), cases=cases, source_sha256=sha(SOURCE),
                         config_sha256=sha(OUT/"config.json"), kernels=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in kernels],
                         curve_hashes={f["key"]:sha(OUT/"curves"/(f["key"]+".json")) for f in config()["families"]},
                         extrapolation_sha256=sha(OUT/"extrapolation_lock.json")))
    return dict(status="references_locked", cases=len(cases))


def reference_job(index):
    lock = read_json(OUT / "reference_lock.json")
    assert lock["source_sha256"] == sha(SOURCE)
    case = lock["cases"][index]
    path = OUT / "references" / (case["case_id"]+".json")
    if path.exists():
        return dict(status="reused", case_id=case["case_id"])
    tick, raw = time.perf_counter(), []
    direct_path = OUT / "direct_contexts" / (case["family_key"]+".raw.json.gz")
    direct_values = read_gzip(direct_path)
    for digits, direct_value in zip(case["digits"], direct_values):
        with mp.workdps(digits):
            assert digits == direct_value["dps"]
            ctx = {k:old.spatial.decode(v) for k,v in direct_value["matrices"].items()}
            ctx.update(mu=[mp.mpf(x) for x in direct_value["mu"]], singular=[mp.mpf(x) for x in direct_value["singular"]],skin=mp.mpf(case["skin"]))
            exponential = structured_exponential(ctx, mp.mpf(case["duration"]))
            columns = old.base.occupied(case["length"], case["initial"])
            q0 = mp.matrix([[exponential[i,j] for j in columns] for i in range(exponential.rows)])
            q = mp.qr(q0, mode="skinny")[0]
            cdw = mp.qr(mp.matrix([[exponential[i,j] for j in range(0,case["length"],2)] for i in range(case["length"])]), mode="skinny")[0]
            su, singular, svh = mp.svd(ctx["perpendicular"].T*q)
            d = singular[0]
            d_cdw = old.previous.mp_opnorm(ctx["perpendicular"].T*cdw)
            memory = old.pair_distance(q, cdw)
            entropy = old.previous.correlation_entropy(q)[0]
            sg = mp.mpf(direct_value["gain_entropy"])
            generators = old.previous.generator(dict(length=case["length"],gamma=2.,skin=case["skin"],t1=.5,t2=1.))
            qv = q*svh.T[:,0]
            velocity = generators*qv
            velocity -= q*(q.T*velocity)
            principal_perpendicular = ctx["perpendicular"]*su[:,0]
            exact_slope = (principal_perpendicular.T*velocity)[0]
            c = constants(case["skin"])
            kappa = d/mp.sqrt(1-d*d)
            slope_bound = -(2*c["a"]-c["b"]*kappa)*d*(1-d*d)
            assert exact_slope <= slope_bound+mp.mpf("1e-35"), (case["case_id"],exact_slope,slope_bound)
            raw.append(dict(dps=digits, distance_to_G=str(d), cdw_distance=str(d_cdw), direct_memory=str(memory), entropy=str(entropy),
                            gain_entropy=str(sg), actual_slope=str(exact_slope), theoretical_slope_upper=str(slope_bound),
                            hermitian_gain_min=direct_value["hermitian_gain_min"], hermitian_perpendicular_max=direct_value["hermitian_perpendicular_max"],
                            orthogonality=str(old.p1.mp_norm(q.T*q-mp.eye(q.cols))), Q=old.p1.encode_matrix(q), G=old.p1.encode_matrix(ctx["gain"])))
        write_json(path.with_suffix(".progress.json"), dict(dps=digits, seconds=time.perf_counter()-tick))
    with mp.workdps(max(case["digits"])+20):
        keys = ("distance_to_G", "cdw_distance", "direct_memory", "entropy", "gain_entropy", "actual_slope")
        change = max(abs(mp.mpf(raw[0][k])-mp.mpf(raw[1][k])) for k in keys)
        assert change < mp.mpf("1e-10"), (case["case_id"], change)
    write_gzip(path.with_suffix(".raw.json.gz"), raw)
    write_json(path, dict(case=case, values={k:v for k,v in raw[-1].items() if k not in ("Q","G")},
                         precision_change_mp=str(change), source_sha256=sha(SOURCE), prediction_lock_sha256=sha(OUT/"reference_lock.json"), direct_context_sha256=sha(direct_path),
                         completed_utc=now(), seconds=time.perf_counter()-tick))
    return dict(status="reference_completed", case_id=case["case_id"], seconds=time.perf_counter()-tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("initialize", "kernel", "static", "curves", "extrapolation_lock", "gram", "lock", "direct_context", "reference"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    tick = time.perf_counter()
    try:
        with old.threadpool_limits(limits=1):
            if args.stage == "initialize": result = initialize()
            elif args.stage == "kernel": result = kernel_checks()
            elif args.stage == "extrapolation_lock": result = extrapolation_lock()
            elif args.stage == "gram": result = gram_job(args.index)
            elif args.stage == "lock": result = lock_references()
            elif args.stage == "reference": result = reference_job(args.index)
            else:
                family = config()["families"][args.index]
                result = static_job(family) if args.stage == "static" else direct_context_job(family) if args.stage == "direct_context" else curves_job(family)
        emit(dict(stage=args.stage, **result))
    except Exception as error:
        write_json(OUT / "failures" / f'{args.stage}_{args.index}_{now().replace(":","-")}.json',
                   dict(stage=args.stage,index=args.index,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),source_sha256=sha(SOURCE)))
        raise
    finally:
        if (OUT/"hpc").exists():
            write_json(OUT/"hpc"/f"environment_{args.stage}_{args.index}.json",
                       dict(source_sha256=sha(SOURCE), stage=args.stage,index=args.index,job_id=os.environ.get("SLURM_JOB_ID"),
                            node=os.environ.get("SLURM_JOB_NODELIST"),mp_backend=mp.libmp.BACKEND,seconds=time.perf_counter()-tick,utc=now()))


if __name__ == "__main__":
    main()
