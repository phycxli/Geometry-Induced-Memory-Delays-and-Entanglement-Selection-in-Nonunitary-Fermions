"""High-precision evidence for the strengthened PRL figures (2026-10-07)."""

from __future__ import annotations

import argparse
import math
import os
import time
import traceback
import zipfile
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mpmath as mp
import numpy as np
import run_theory_upgrade_hpc as theory

old = theory.old
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prl_figure_strengthening"
SOURCE = Path(__file__)
read_json, write_json = old.read_json, old.write_json
read_gzip, write_gzip = old.read_gzip, old.write_gzip
LEVELS = ("0.3", "0.1", "0.05", "0.01")


def digits(length):
    lower = max(240, 20 * math.ceil((1.3 * length + 30) / 20))
    return [lower, lower + 40]


def config():
    return read_json(OUT / "config.json")


def initialize():
    import csv
    for folder in ("hpc", "spectral", "families", "static", "curves", "traces", "response", "reference_contexts", "references", "checks", "failures"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    families = []
    for length in (128, 192, 256, 320, 384):
        for skin in ("0", "0.125", "0.25"):
            if length < 320 and skin != "0.125":
                continue
            families.append(dict(key=f"L{length}_g{skin}".replace(".", "p"), length=length, skin=skin,
                                 digits=digits(length), role="intermediate_skin" if length < 320 else "new_size"))
    cfg = dict(date="2026-10-07", families=families, lengths=[128, 192, 256, 320, 384],
               response_lengths=[64, 128, 512], response_chi=["-4", "-2", "-1", "0", "1", "2", "4"],
               reflection_lengths=[128, 256, 512], levels=list(LEVELS), root_tolerance="0.000002",
               projector_budget="0.00001", reference_offsets=["-.08", ".08"],
               scope="SSH OBC gamma=2 t1=.5; |g|<=.25; low thresholds to G; empirical extrapolation separate from theorem")
    if (OUT / "config.json").exists():
        assert config() == cfg
        return dict(status="reused")
    write_json(OUT / "config.json", cfg)
    paths = [ROOT / "data/prl_theory_upgrade/roots.csv", ROOT / "data/prl_theory_upgrade/widths.csv",
             ROOT / "data/prl_manuscript_revision/fig3_weak_skin.csv", ROOT / "scripts/run_theory_upgrade_hpc.py",
             ROOT / "scripts/run_selection_front_hpc.py", ROOT / "scripts/run_fine_front_hpc.py"]
    manifest = [dict(path=p.relative_to(ROOT).as_posix(), sha256=theory.sha(p)) for p in paths]
    write_json(OUT / "input_manifest.json", manifest)
    with (paths[0]).open(encoding="utf-8", newline="") as stream:
        roots = list(csv.DictReader(stream))
    predictions = []
    for family in families:
        if family["role"] != "new_size" or family["skin"] == "0.125":
            continue
        for kind in ("left", "right", "block1", "block3"):
            for level in LEVELS:
                local = [r for r in roots if r["skin"] == family["skin"] and r["level"] == level
                         and (r["initial"] if not r["initial"].startswith("block_cell") else
                              "block1" if int(r["initial"][10:]) == int(r["length"]) // 16 else "block3") == kind]
                x = np.array([int(r["length"]) for r in local], float)
                y = np.array([float(r["root_time"]) for r in local])
                coef = np.polyfit(x, y, 1)
                predictions.append(dict(family_key=family["key"], kind=kind, level=level, coefficients=coef.tolist(),
                                        predicted_time=float(np.polyval(coef, family["length"])), training_sizes=x.tolist()))
    write_json(OUT / "prediction_lock.json", dict(locked_utc=old.now(), predictions=predictions,
               source_sha256=theory.sha(SOURCE), inputs=manifest,
               policy="Affine predictions from all five existing sizes; no refit after new results; record errors without inventing a success gate"))
    manuscript_files = [ROOT / name / filename for name, stem in (("main", "main"), ("main_zh", "main"),
                         ("supplement", "supplement"), ("supplment_zh", "supplement"))
                         for filename in (stem + ".tex", stem + ".pdf", "refs.bib")]
    manuscript_files += list((ROOT / "figures/prl_manuscript_revision").glob("fig[234]*"))
    manuscript_files += list((ROOT / "data/prl_manuscript_revision").glob("fig[234]*.csv"))
    with zipfile.ZipFile(OUT / "before_revision.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in manuscript_files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    write_json(OUT / "before_revision.json", dict(files=[dict(path=p.relative_to(ROOT).as_posix(), sha256=theory.sha(p))
               for p in manuscript_files], archive_sha256=theory.sha(OUT / "before_revision.zip")))
    return dict(status="initialized", families=len(families), frozen_predictions=len(predictions))


def spectral_at(length, precision):
    n = length // 2
    coupling = mp.matrix(n)
    for j in range(n):
        coupling[j, j] = mp.mpf(".5")
        if j:
            coupling[j, j-1] = 1
    u, singular, vh = mp.svd(coupling)
    v = vh.T
    for column in range(n):
        if u[0, column] < 0:
            u[:, column], v[:, column] = -u[:, column], -v[:, column]
    mu = [mp.sqrt(4-x*x) for x in singular]
    ratios = [s/(2+x) for s, x in zip(singular, mu)]
    plus, minus, yp, ym = mp.matrix(length, n), mp.matrix(length, n), mp.matrix(n, length), mp.matrix(n, length)
    for k in range(n):
        aa = 1/mp.sqrt(1+ratios[k]**2)
        bb = ratios[k]*aa
        for j in range(n):
            plus[2*j,k], plus[2*j+1,k] = u[j,k]*aa, -v[j,k]*bb
            minus[2*j,k], minus[2*j+1,k] = u[j,k]*bb, -v[j,k]*aa
            yp[k,2*j], yp[k,2*j+1] = 2/mu[k]*aa*u[j,k], 2/mu[k]*bb*v[j,k]
            ym[k,2*j], ym[k,2*j+1] = -2/mu[k]*bb*u[j,k], -2/mu[k]*aa*v[j,k]
    states = {}
    for name in old.initials(length):
        q0 = old.base.q0_matrix(length, name)
        f = old.cached_right_solve(ym*q0, yp*q0)
        fu, fs, fvh = mp.svd(f)
        states[name] = dict(F=old.p1.encode_matrix(f), F_norm=str(fs[0]), growing_input_full_rank=True,
                            compression=dict(U=old.p1.encode_matrix(fu), VH=old.p1.encode_matrix(fvh), singular_values=[str(x) for x in fs]))
    full = mp.matrix(length)
    full[:,:n], full[:,n:] = plus, minus
    inverse = mp.matrix(length)
    inverse[:n,:], inverse[n:,:] = yp, ym
    inverse_error = old.p1.mp_norm(full*inverse-mp.eye(length))
    assert inverse_error < mp.mpf("1e-50")
    return dict(length=length, gamma=2., skin=0., dps=precision, mu=[str(x) for x in mu], states=states,
                plus=old.p1.encode_matrix(plus), minus=old.p1.encode_matrix(minus), inverse_residual=str(inverse_error))


def spectral_job(index):
    length = config()["lengths"][index]
    path = OUT / "spectral" / f"L{length}.json"
    if path.exists():
        return dict(status="reused", length=length)
    tick, values = time.perf_counter(), []
    for precision in digits(length):
        with mp.workdps(precision):
            values.append(spectral_at(length, precision))
        write_json(path.with_suffix(".progress.json"), dict(dps=precision, seconds=time.perf_counter()-tick))
    with mp.workdps(digits(length)[-1]):
        differences = []
        for name in old.initials(length):
            a, b = [old.spatial.decode(value["states"][name]["F"]) for value in values]
            differences.append(old.p1.mp_norm(a-b)/old.p1.mp_norm(b))
            sa, sb = [value["states"][name]["compression"]["singular_values"] for value in values]
            differences.extend(old.spatial.relative(mp.mpf(a), mp.mpf(b)) for a,b in zip(sa[:24], sb[:24]))
        change = max(differences)
        assert change < mp.mpf("1e-12"), change
    write_gzip(path.with_suffix(".raw.json.gz"), values)
    write_json(path, dict(length=length, digits=digits(length), relative_change_mp=str(change),
                         source_sha256=theory.sha(SOURCE), completed_utc=old.now(), seconds=time.perf_counter()-tick))
    return dict(status="spectral_cache_complete", length=length, seconds=time.perf_counter()-tick)


def biased_static(spectral, skin):
    length, n = spectral["length"], spectral["length"]//2
    g = mp.mpf(skin)
    plus, minus = [old.spatial.decode(spectral[k]) for k in ("plus", "minus")]
    for i in range(length):
        weight = mp.exp(g*(i//2-(n-1)/2))
        for k in range(n):
            plus[i,k] *= weight
            minus[i,k] *= weight
    full = mp.qr(plus)[0]
    gain, perpendicular = full[:,:n], full[:,n:]
    c = theory.constants(skin)
    return dict(length=length, gamma=2., skin=float(g), dps=spectral["dps"], mu=spectral["mu"], states=spectral["states"],
                matrices={k:old.p1.encode_matrix(v) for k,v in dict(plus=plus,minus=minus,G=gain,Gperp=perpendicular,
                         R=gain.T*plus,Ccross=gain.T*minus,Bperp=perpendicular.T*minus).items()},
                gain_entropy=str(old.previous.correlation_entropy(gain)[0]),
                output_bound_policy="analytic exp(-2 a t); unnormalized spectral basis with centered diagonal similarity",
                theorem_a=str(c["a"]))


def static_job(index):
    family = config()["families"][index]
    path = OUT / "static" / (family["key"]+".json")
    if path.exists():
        return dict(status="reused", family=family["key"])
    tick, values = time.perf_counter(), []
    for spectral in read_gzip(OUT / "spectral" / f'L{family["length"]}.raw.json.gz'):
        with mp.workdps(spectral["dps"]):
            values.append(biased_static(spectral, family["skin"]))
    with mp.workdps(family["digits"][-1]):
        change = old.p1.mp_norm(old.spatial.decode(values[0]["matrices"]["G"])*old.spatial.decode(values[0]["matrices"]["G"]).T
                             -old.spatial.decode(values[1]["matrices"]["G"])*old.spatial.decode(values[1]["matrices"]["G"]).T)
        assert change < mp.mpf("1e-12")
    write_gzip(path.with_suffix(".raw.json.gz"), values)
    write_json(path, dict(family=family, gain_projector_change_mp=str(change), source_sha256=theory.sha(SOURCE),
                         completed_utc=old.now(), seconds=time.perf_counter()-tick))
    return dict(status="biased_static_complete", family=family["key"], seconds=time.perf_counter()-tick)


def contexts(family, precision_index=-1):
    static = read_gzip(OUT / "static" / (family["key"]+".raw.json.gz"))[precision_index]
    mp.mp.dps = static["dps"]
    old.RANKS = (0, 1, 2, 4, 6, 8, 12, 16, 24, 32)
    return old.context(static)


def point_certificate(ctx, duration, rank):
    values, _, _ = theory.fine.at_rank(old.time_context(ctx, duration), rank)
    return values


def curves_job(index):
    family = config()["families"][index]
    path = OUT / "curves" / (family["key"]+".json")
    if path.exists():
        return dict(status="reused", family=family["key"])
    tick, results, rows = time.perf_counter(), [], []
    for precision_index in (0, 1):
        ctxs = contexts(family, precision_index)
        local = {}
        for name in old.initials(family["length"])[1:]:
            for level in LEVELS:
                local[name, level] = theory.bracket_root(ctxs[name], level, 12)[0]
        results.append(local)
        write_json(path.with_suffix(".progress.json"), dict(root_precisions=precision_index+1, elapsed=time.perf_counter()-tick))
    for (name, level), candidate in results[-1].items():
        change = abs(candidate-results[0][name,level])
        assert change < mp.mpf(".000004")
        ctx, root, rank = ctxs[name], candidate, 12
        cert = point_certificate(ctx, root, rank)
        if cert["projector_remainder"] > mp.mpf(".00001"):
            rank = 24
            root = theory.bracket_root(ctx, level, rank)[0]
            cert = point_certificate(ctx, root, rank)
        assert cert["projector_remainder"] < mp.mpf(".00001"), (family["key"],name,level,cert)
        rows.append(dict(family_key=family["key"], length=family["length"], skin=family["skin"], initial=name,
                         level=level, root_time=float(root), root_time_mp=str(root), rank=rank,
                         precision_change_mp=str(change), projector_remainder_mp=str(cert["projector_remainder"])))
    write_json(path, dict(family=family, rows=rows, source_sha256=theory.sha(SOURCE),
                         prediction_lock_sha256=theory.sha(OUT/"prediction_lock.json"), completed_utc=old.now(), seconds=time.perf_counter()-tick))
    return dict(status="roots_complete", family=family["key"], roots=len(rows), seconds=time.perf_counter()-tick)


def distance_at(ctx, duration):
    duration = mp.mpf(str(duration))
    # Dense graph propagation supplies early/plateau samples when a compact state has a large remainder.
    tc = old.time_context(ctx, duration)
    for rank in (12, 24):
        values, _, _ = theory.fine.at_rank(tc, min(rank, len(ctx["mu"])))
        if values["projector_remainder"] < mp.mpf("1e-7"):
            return values["distance_to_G"], values["projector_remainder"], f"low_rank_{rank}"
    frame = ctx["matrices"]["plus"] + tc["delta0"]*ctx["matrices"]["R"]
    q = mp.qr(frame, mode="skinny")[0]
    distance = old.previous.mp_opnorm(ctx["matrices"]["Gperp"].T*q)
    return distance, mp.mpf(0), "dense_graph"


def trace_cases():
    return [dict(length=192,skin="0.25",initial=name,mode="full") for name in ("charge_density_wave","left","right")] + [
        dict(length=length,skin=skin,initial=name,mode="window") for length,skin,name in
        ((128,"0","left"),(192,"0.125","left"),(192,"0.125","right"),
         (384,"0.25","left"),(384,"0.25","right"),(384,"0.25","block_cell72"))]


def trace_job(index):
    case = trace_cases()[index]
    key = f'L{case["length"]}_g{case["skin"]}'.replace(".","p")
    path = OUT / "traces" / f'{key}_{case["initial"]}_{case["mode"]}.json'
    if path.exists():
        return dict(status="reused")
    tick = time.perf_counter()
    spectral = read_gzip(OUT / "spectral" / f'L{case["length"]}.raw.json.gz')
    ctxs = []
    for value in spectral:
        with mp.workdps(value["dps"]):
            ctxs.append(old.context(biased_static(value,case["skin"]),[case["initial"]])[case["initial"]])
    family = next((f for f in config()["families"] if f["key"]==key), None)
    if case["initial"] == "charge_density_wave":
        center = 0.
        times = list(np.linspace(0, 8, 33)) + [12.,20.,40.,60.,80.,100.,120.,128.]
    else:
        roots = read_json((OUT if family else ROOT/"data/prl_theory_upgrade")/"curves"/(key+".json"))["rows"]
        center = next(r["root_time"] for r in roots if r["initial"]==case["initial"] and r["level"]=="0.05")
        times = [center+x for x in np.linspace(-1.4, 1.1, 51)] + [center+x for x in (1.5,2,3,4,6,8,12)]
        if case["mode"] == "full":
            times += [0.,.5,1.,2.,4.,8.,12.,20.,40.,60.,70.,80.,85.,90.,100.,110.,120.,128.]
    times = sorted({round(float(x),10) for x in times if x >= 0})
    rows = []
    # Every plotted point is evaluated at both precisions; interpolation is only a visual connector.
    for i, duration in enumerate(times):
        evaluated = []
        for ctx in ctxs:
            with mp.workdps(ctx["static"]["dps"]):
                evaluated.append(distance_at(ctx,duration))
        with mp.workdps(spectral[-1]["dps"]):
            change = abs(evaluated[0][0]-evaluated[1][0])
            assert change < mp.mpf("1e-8"), (case,duration,change)
        rows.append(dict(**case,family_key=key,time=duration,aligned_time=duration-center,center=center,
                         distance_to_G=float(evaluated[-1][0]), distance_to_G_mp=str(evaluated[-1][0]),
                         projector_remainder=float(evaluated[-1][1]), precision_change_mp=str(change),method=evaluated[-1][2]))
        write_json(path.with_suffix(".progress.json"),dict(points=i+1,total=len(times),seconds=time.perf_counter()-tick))
    write_json(path,dict(case=case,rows=rows,digits=digits(case["length"]),source_sha256=theory.sha(SOURCE),
                         completed_utc=old.now(),seconds=time.perf_counter()-tick))
    return dict(status="trace_complete",case=case,points=len(rows),seconds=time.perf_counter()-tick)


def response_job(index, reflection=False):
    lengths = config()["reflection_lengths" if reflection else "response_lengths"]
    length, initial = lengths[index//2], ("left","right")[index%2]
    path = OUT / "response" / f'L{length}_{initial}_{"reflection" if reflection else "signed"}.json'
    if path.exists():
        return dict(status="reused")
    tick, results = time.perf_counter(), []
    folded_options = (False,True) if reflection else (True,)
    for folded in folded_options:
        if folded and length==512:
            source = ROOT / f'data/prl_theory_upgrade/response_extension/L{length}_{initial}.raw.json.gz'
            raw_values = read_gzip(source)
            matrices = [(r["digits"],r["inverse"]) for r in raw_values]
        else:
            matrices = []
            for nodes,precision in ((length//2+32,max(240,2*length)),(length//2+64,max(240,2*length)+40)):
                with mp.workdps(precision):
                    _,matrix = old.base.spatial_gram(length,0.,initial,nodes,precision,folded)
                    matrices.append((precision,mp.inverse(matrix)))
        evaluations = []
        for precision,inverse in matrices:
            with mp.workdps(precision):
                if isinstance(inverse, list):
                    inverse = old.spatial.decode(inverse)
                sign, m = (1 if initial=="left" else -1),inverse.rows
                local = []
                gs = [mp.mpf(x) for x in ("-.25","0",".25")] if reflection else [mp.mpf(x)/length for x in config()["response_chi"]]
                for g in gs:
                    j = mp.matrix([[inverse[i,k]*mp.exp(-sign*g*(i+k+1)) for k in range(m)] for i in range(m)])
                    largest = mp.eigsy(j,eigvals_only=True)[m-1]
                    local.append(dict(length=length,initial=initial,skin=str(g),chi=float(g*length),folded=folded,
                                      log_delta_mp=str(-mp.log(largest)),log_delta=float(-mp.log(largest))))
                zero = next(mp.mpf(r["log_delta_mp"]) for r in local if mp.mpf(r["skin"])==0)
                for row in local:
                    response = mp.mpf(row["log_delta_mp"])-zero
                    row.update(log_ratio_to_zero=float(response),weak_prediction=sign*row["chi"]/2,
                               response_residual=float(response)-sign*row["chi"]/2)
                evaluations.append(local)
        with mp.workdps(max(r[0] for r in matrices)):
            change = max(abs(mp.mpf(a["log_delta_mp"])-mp.mpf(b["log_delta_mp"])) for a,b in zip(*evaluations))
            assert change < mp.mpf("1e-8"), (length,initial,folded,change)
        results.extend({**row,"precision_change_mp":str(change)} for row in evaluations[-1])
    write_json(path,dict(rows=results,source_sha256=theory.sha(SOURCE),completed_utc=old.now(),seconds=time.perf_counter()-tick))
    return dict(status="response_complete",length=length,initial=initial,reflection=reflection,seconds=time.perf_counter()-tick)


def reference_context_job(index):
    family = config()["families"][index]
    path = OUT / "reference_contexts" / (family["key"]+".json")
    if path.exists():
        return dict(status="reused")
    values, tick = [], time.perf_counter()
    for precision in family["digits"]:
        with mp.workdps(precision):
            ctx = theory.direct_context(family["length"],mp.mpf(family["skin"]))
            values.append(dict(dps=precision,matrices={k:old.p1.encode_matrix(ctx[k]) for k in ("u","v","gain","perpendicular")},
                               mu=[str(x) for x in ctx["mu"]],singular=[str(x) for x in ctx["singular"]],skin=family["skin"]))
    write_gzip(path.with_suffix(".raw.json.gz"),values)
    write_json(path,dict(family=family,source_sha256=theory.sha(SOURCE),completed_utc=old.now(),seconds=time.perf_counter()-tick))
    return dict(status="reference_context_complete",family=family["key"],seconds=time.perf_counter()-tick)


def reference_cases():
    return [(i,name,offset) for i,f in enumerate(config()["families"]) if f["length"] >= 320
            for name in (("left","right","block_cell"+str(3*f["length"]//16)) if f["skin"]=="0.25" else ("left","right"))
            for offset in config()["reference_offsets"]]


def reference_job(index):
    family_index,name,offset = reference_cases()[index]
    family = config()["families"][family_index]
    path = OUT / "references" / f'{family["key"]}_{name}_{"before" if mp.mpf(offset)<0 else "after"}.json'
    if path.exists():
        return dict(status="reused")
    root = next(r["root_time_mp"] for r in read_json(OUT/"curves"/(family["key"]+".json"))["rows"]
                if r["initial"]==name and r["level"]=="0.05")
    results, tick = [], time.perf_counter()
    for raw in read_gzip(OUT/"reference_contexts"/(family["key"]+".raw.json.gz")):
        with mp.workdps(raw["dps"]):
            duration = mp.mpf(root)+mp.mpf(offset)
            ctx = {k:old.spatial.decode(v) for k,v in raw["matrices"].items()}
            ctx.update(mu=[mp.mpf(x) for x in raw["mu"]],singular=[mp.mpf(x) for x in raw["singular"]],skin=mp.mpf(family["skin"]))
            flow = theory.structured_exponential(ctx,duration)
            columns = old.base.occupied(family["length"],name)
            q = mp.qr(mp.matrix([[flow[i,j] for j in columns] for i in range(flow.rows)]),mode="skinny")[0]
            distance = old.previous.mp_opnorm(ctx["perpendicular"].T*q)
            reduced = contexts(family,0 if raw["dps"]==family["digits"][0] else 1)[name]
            expected = theory.compact_graph(reduced,duration,12)
            assert abs(distance-expected) < mp.mpf(".00001")
            assert (distance > mp.mpf(".05")) == (mp.mpf(offset)<0)
            results.append(dict(dps=raw["dps"],distance_to_G_mp=str(distance),duration_mp=str(duration),
                                distance_to_G=float(distance),duration=float(duration),reduced_error_mp=str(abs(distance-expected)),
                                orthogonality_mp=str(old.p1.mp_norm(q.T*q-mp.eye(q.cols)))))
    with mp.workdps(family["digits"][-1]):
        change = abs(mp.mpf(results[0]["distance_to_G_mp"])-mp.mpf(results[1]["distance_to_G_mp"]))
        assert change < mp.mpf("1e-10")
    write_json(path,dict(family=family,initial=name,offset=offset,root=float(root),values=results[-1],
                         precision_change_mp=str(change),source_sha256=theory.sha(SOURCE),completed_utc=old.now(),seconds=time.perf_counter()-tick))
    return dict(status="independent_reference_complete",family=family["key"],initial=name,seconds=time.perf_counter()-tick)


def checks():
    rows = []
    with mp.workdps(100):
        spectral = spectral_at(16,100)
        for skin in ("-.25","0",".125",".25"):
            static = biased_static(spectral,skin)
            direct = theory.direct_context(16,mp.mpf(skin))
            gain = old.spatial.decode(static["matrices"]["G"])
            projector_error = old.p1.mp_norm(gain*gain.T-direct["gain"]*direct["gain"].T)
            assert projector_error < mp.mpf("1e-80")
            ctxs = old.context(static)
            for name in old.initials(16):
                for duration in (mp.mpf(".5"),mp.mpf(4),mp.mpf(9)):
                    flow = mp.expm(old.previous.generator(dict(length=16,gamma=2.,skin=skin,t1=.5,t2=1.))*duration)
                    q = mp.qr(flow*old.base.q0_matrix(16,name),mode="skinny")[0]
                    expected = old.previous.mp_opnorm(direct["perpendicular"].T*q)
                    actual,remainder,method = distance_at(ctxs[name],duration)
                    error = abs(actual-expected)
                    assert error <= remainder+mp.mpf("1e-70"), (skin,name,duration,error,remainder)
                    rows.append(dict(skin=skin,initial=name,time=str(duration),distance_error_mp=str(error),
                                     projector_error_mp=str(projector_error),method=method))
    write_json(OUT/"checks/kernel_checks.json",dict(status="passed",rows=rows,source_sha256=theory.sha(SOURCE),verified_utc=old.now()))
    return dict(status="kernel_checks_passed",checks=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("initialize","checks","spectral","static","curves","trace","response","reflection","reference_context","reference"))
    parser.add_argument("--index",type=int,default=0)
    args = parser.parse_args()
    tick = time.perf_counter()
    try:
        with old.threadpool_limits(limits=1):
            if args.stage=="initialize": value=initialize()
            elif args.stage=="checks": value=checks()
            elif args.stage=="spectral": value=spectral_job(args.index)
            elif args.stage=="static": value=static_job(args.index)
            elif args.stage=="curves": value=curves_job(args.index)
            elif args.stage=="trace": value=trace_job(args.index)
            elif args.stage in ("response","reflection"): value=response_job(args.index,args.stage=="reflection")
            elif args.stage=="reference_context": value=reference_context_job(args.index)
            else: value=reference_job(args.index)
        old.emit(dict(stage=args.stage,**value))
    except Exception as error:
        write_json(OUT/"failures"/f'{args.stage}_{args.index}_{old.now().replace(":","-")}.json',
                   dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),source_sha256=theory.sha(SOURCE)))
        raise
    finally:
        if (OUT/"hpc").exists():
            write_json(OUT/"hpc"/f'environment_{args.stage}_{args.index}.json',dict(stage=args.stage,index=args.index,
                       job_id=os.environ.get("SLURM_JOB_ID"),node=os.environ.get("SLURM_JOB_NODELIST"),
                       seconds=time.perf_counter()-tick,mp_backend=mp.libmp.BACKEND,source_sha256=theory.sha(SOURCE),utc=old.now()))


if __name__=="__main__":
    main()
