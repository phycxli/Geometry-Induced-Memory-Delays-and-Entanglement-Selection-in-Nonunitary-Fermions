"""Summarize frozen predictions and certify the parameter-box constants."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import zipfile

import mpmath as mp

import run_priority_strengthening as study

ROOT, OUT = study.ROOT, study.OUT


def write_csv(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT/name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def interval_record(value):
    return dict(enclosure=str(value), lower=math.nextafter(float(value.a),-math.inf),
                upper=math.nextafter(float(value.b),math.inf))


def bounds():
    # Analytic monotonicity puts all worst constants at this one endpoint.
    iv = mp.iv
    iv.dps = 80
    gamma, t1, t2, h = [iv.mpf(x) for x in ("2", ".6", "1.05", ".2")]
    beta = t1*t1+t2*t2+2*t1*t2*(iv.exp(h)+iv.exp(-h))/2
    rho = (t1+t2*iv.exp(h))/(gamma+iv.sqrt(gamma*gamma-beta))
    eta = (1-rho*rho)/(1+rho*rho)
    a = gamma*eta-t2*(iv.exp(h)-iv.exp(-h))/2
    b = 2*(t1+t2*(iv.exp(h)+iv.exp(-h))/2)
    cone = (2*a/b)/iv.sqrt(1+(2*a/b)**2)
    fstar = iv.sqrt(1+rho*rho)/eta
    assert beta.b < gamma*gamma and rho.b < 1 and a.a > 0 and cone.a > iv.mpf(".3")
    levels = []
    for eps in (".3", ".05", ".01"):
        e = iv.mpf(eps)
        k = e/iv.sqrt(1-e*e)
        time = iv.ln(fstar*(1+k)/k)/(2*a)
        susceptibility = iv.mpf(".5")*(1+k)/(2*a-b*k)
        levels.append(dict(threshold=float(eps), cdw_sufficient_time=interval_record(time),
                           absolute_susceptibility_bound=interval_record(susceptibility)))
    value = dict(status="passed", checked_utc=study.old.now(), precision_digits=iv.dps,
        method="mpmath interval arithmetic at the analytically monotone worst endpoint",
        scope="Outward enclosures for analytic constants only, not numerical crossing-time certificates",
        parameter_box=dict(gamma=[2,2.4],t1=[.4,.6],t2=[.8,1.05],abs_g=[0,.2]),
        endpoint=dict(gamma=2,t1=.6,t2=1.05,abs_g=.2),
        constants={key:interval_record(value) for key,value in
                   dict(beta=beta,rho=rho,eta=eta,a=a,b=b,cone=cone,
                        tail_ratio=beta/gamma**2,cdw_spectral_graph_bound=fstar).items()},
        levels=levels,
        monotonicity="beta and rho increase with t1,t2,h and rho decreases with gamma; eta increases with gamma and decreases with t1,t2,h; a is minimal and b maximal at the recorded endpoint")
    study.save(OUT/"parameter_box_certificate.json",value)
    return value


def analyze(require_references=True):
    config = study.config()
    lock = study.load(OUT/"hypothesis_lock.json")
    assert study.digest(OUT/"config.json")==lock["config_sha256"]
    assert all(config[key]==value for key,value in lock["gates"].items())
    for row in lock["inputs"]:
        if row["path"].startswith("data/"):
            assert study.digest(ROOT/row["path"])==row["sha256"],row["path"]
    scans, zeros, widths = [], [], []
    for family in config["families"]:
        key = family["key"]
        cache = study.load(OUT/"cache"/(key+".json"))
        assert study.digest(ROOT/cache["source"]) == cache["source_sha256"]
        assert study.digest(OUT/"cache"/(key+".npz")) == cache["cache_sha256"]
        prediction = study.load(OUT/"predictions"/(key+".json"))
        result = study.load(OUT/"scans"/(key+".json"))
        assert result["prediction_sha256"] == study.digest(OUT/"predictions"/(key+".json"))
        assert prediction["locked_utc"] < result["completed_utc"]
        assert len(prediction["rows"]) == len(result["rows"]) == 48
        for row in prediction["rows"]:
            if row["chi"] != config["chis"][0]:
                continue
            response = row["response"]
            zeros.append(dict(family=key,**{k:family[k] for k in ("length","gamma","t1","t2","role")},
                              initial=row["initial"],level=row["level"],zero_time=row["zero_time"],
                              static_guess_susceptibility=row["static_guess_susceptibility"],
                              conjecture_difference=abs(response["susceptibility"]-row["static_guess_susceptibility"]),
                              conjecture_finite_size_pass=abs(response["susceptibility"]-row["static_guess_susceptibility"])<=config["asymptotic_susceptibility_gate"],
                              **response))
        for row in result["rows"]:
            flat = dict(family=key,**{k:family[k] for k in ("length","gamma","t1","t2","role")},
                        **{k:v for k,v in row.items() if k!="response"},**row["response"])
            flat["directional_sign_observed"] = flat["susceptibility"]*(-1 if row["initial"]=="left" else 1)>0
            scans.append(flat)
        for name in ("left","right"):
            for chi in config["chis"]:
                relevant = [r for r in result["rows"] if r["initial"]==name and r["chi"]==chi]
                roots = {r["level"]:r["root_time"] for r in relevant}
                constants = study.contraction_constants(family["gamma"],family["t1"],family["t2"],chi/family["length"])
                a,b = constants["a"],constants["b"]
                hi,lo = .3/math.sqrt(1-.3**2),.01/math.sqrt(1-.01**2)
                bound = math.log(hi*(2*a-b*lo)/(lo*(2*a-b*hi)))/(2*a)
                width = roots[.01]-roots[.3]
                assert 0<width<=bound+1e-7
                widths.append(dict(family=key,length=family["length"],initial=name,chi=chi,
                                   width=width,analytic_bound=bound,relative_width=width/roots[.05]))
    references = []
    for path in sorted((OUT/"references").glob("*.json")):
        if path.name.endswith(".progress.json"):
            continue
        record = study.load(path)
        assert record["status"] == "passed" and len(record["rows"]) == 4
        with mp.workdps(max(r["dps"] for r in record["rows"])+30):
            changes = [abs(mp.mpf(record["rows"][i]["distance"])-mp.mpf(record["rows"][i+2]["distance"])) for i in (0,1)]
            precision_change = str(max(changes))
        references.append(dict(family=record["family"]["key"],initial=record["initial"],chi=record["chi"],
                               points=len(record["rows"]),max_float_error=max(r["float_graph_error"] for r in record["rows"]),
                               max_norm_bound_width=max(float(r.get("norm_bound_width",0)) for r in record["rows"]),
                               two_precision_distance_change=precision_change))
    if require_references:assert len(references)==5
    assert len(scans)==528 and len(zeros)==66 and len(widths)==176
    write_csv("time_response.csv",scans)
    write_csv("zero_bias_response.csv",zeros)
    write_csv("window_checks.csv",widths)
    write_csv("reference_checks.csv",references)
    per_family = []
    for family in config["families"]:
        rows = [r for r in scans if r["family"]==family["key"]]
        z = [r for r in zeros if r["family"]==family["key"]]
        per_family.append(dict(family=family["key"],role=family["role"],roots=len(rows),
          linear_passes=sum(r["linear_pass"] for r in rows),max_linear_error=max(abs(r["linear_error"]) for r in rows),
          max_static_gap_error=max(abs(r["static_gap_error"]) for r in rows),
          conjecture_finite_size_passes=sum(r["conjecture_finite_size_pass"] for r in z),
          zero_susceptibility_left=[r["susceptibility"] for r in z if r["initial"]=="left"],
          zero_susceptibility_right=[r["susceptibility"] for r in z if r["initial"]=="right"]))
    write_csv("family_summary.csv",per_family)
    value = dict(status="passed" if len(references)==5 else "awaiting_direct_references", analyzed_utc=study.old.now(),families=per_family,
      roots=len(scans),new_bias_predictions=len(scans),zero_bias_roots=len(zeros),
      linear_gate=config["linear_time_gate"],linear_passes=sum(r["linear_pass"] for r in scans),
      linear_failures=sum(not r["linear_pass"] for r in scans),
      max_linear_error=max(abs(r["linear_error"]) for r in scans),
      max_static_gap_error=max(abs(r["static_gap_error"]) for r in scans),
      finite_size_conjecture_gate=config["asymptotic_susceptibility_gate"],
      finite_size_conjecture_passes=sum(r["conjecture_finite_size_pass"] for r in zeros),
      finite_size_conjecture_tests=len(zeros),
      observed_directional_sign_passes=sum(r["directional_sign_observed"] for r in scans),
      smallest_principal_gap=min(r["principal_gap"] for r in scans),
      max_invariance_residual=max(r["principal_residual"] for r in scans),
      max_chi_derivative_error=max(r["chi_finite_difference_error"] for r in scans),
      max_time_derivative_error=max(r["time_finite_difference_error"] for r in scans),
      max_susceptibility_bound_fraction=max(abs(r["susceptibility"])/r["uniform_susceptibility_bound"] for r in scans),
      width_checks=len(widths),width_min=min(r["width"] for r in widths),width_max=max(r["width"] for r in widths),
      references=references,
      prediction_scope="Same-family nonzero-bias holdout after zero-bias measurement; not blind prediction of a new-size zero-bias center",
      scientific_status="Exact finite-size response and uniform bounds; observed signs and finite-size trends do not prove an asymptotic time coefficient",
      preserved_failures=[p.relative_to(ROOT).as_posix() for p in sorted((OUT/"failures").glob("*.json"))])
    study.save(OUT/"summary.json",value)
    snapshot = OUT/"source_snapshot.zip"
    if len(references)==5 and not snapshot.exists():
        paths = [ROOT/p for p in ("src/mietf_skin/response.py","scripts/run_priority_strengthening.py",
                                  "scripts/manage_priority_strengthening.py","scripts/analyze_priority_strengthening.py",
                                  "scripts/slurm_priority_strengthening.sh","scripts/plot_priority_strengthening.py",
                                  "scripts/verify_priority_strengthening.py")]
        with zipfile.ZipFile(snapshot,"x",zipfile.ZIP_DEFLATED) as archive:
            for p in paths:
                archive.write(p,p.relative_to(ROOT).as_posix())
        original = {r["path"]:r["sha256"] for r in lock["inputs"]}
        study.save(OUT/"source_provenance.json",dict(snapshot_sha256=study.digest(snapshot),
          files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=study.digest(p),
                      original_hypothesis_hash=original.get(p.relative_to(ROOT).as_posix())) for p in paths],
          original_hypothesis_lock_preserved=True,config_and_gates_verified=True,
          implementation_repairs="Replaced unstable float64 small-size reference by a 100-digit direct exponential; removed a dead expression; included raw high-precision matrices in incremental retrieval; replaced expensive large-reference mp SVD by complete-column norm bounds, rechecked against mp SVD in 36 small cases. Scientific hypotheses and gates unchanged."))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("bounds","analyze"))
    parser.add_argument("--allow-pending-references",action="store_true")
    args = parser.parse_args()
    result=bounds() if args.stage=="bounds" else analyze(not args.allow_pending_references)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
