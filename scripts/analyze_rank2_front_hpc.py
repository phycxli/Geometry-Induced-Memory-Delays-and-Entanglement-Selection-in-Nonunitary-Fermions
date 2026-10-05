"""Compare frozen rank1, signed rank2, and certified adaptive predictions."""

import argparse
import collections
import functools
import json
import re
from pathlib import Path

import run_rank2_front_hpc as run
from analyze_selection_front_hpc import front

ROOT, OUT = run.ROOT, run.OUT
FIG = ROOT / "figures/prl_p3_rank2_front"
read_json, write_json, write_csv = run.read_json, run.write_json, run.write_csv


@functools.lru_cache(maxsize=None)
def digest(path):
    return run.sha256(path)


def summarize(plots=True):
    run.setup()
    run.check_lock()
    cfg, lock = run.config(), read_json(OUT / "prediction_lock.json")
    predicted, memories, development, kernel_checks = {}, {}, [], []
    for family in cfg["families"]:
        value = read_json(OUT / "predictions" / (family["key"] + ".json"))
        assert value["fixed_rank2_complete"]
        predicted.update({(r["case_id"], r["initial"]): r for r in value["rows"]})
        memories.update({(r["case_id"], r["initial_a"], r["initial_b"]): r for r in value["pairs"]})
        value = read_json(OUT / "development" / (family["key"] + ".json"))
        development.extend(value["rows"])
        kernel_checks.extend(value["checks"])
    states, pairs, checks, pair_checks = [], [], [], []
    precision = collections.Counter()
    for case in cfg["cases"]:
        path = OUT / "cases" / (case["case_id"] + ".json")
        value, certificate = read_json(path), read_json(OUT / "audits" / path.name)
        assert value["started_utc"] > lock["locked_utc"]
        assert value["prediction_lock_sha256"] == digest(OUT / "prediction_lock.json")
        assert run.old.previous.accepted_precision(value["reference_convergence"])
        assert certificate["fixed_rank2_checks_complete"]
        assert certificate["reference_sha256"] == digest(path)
        assert certificate["reference_raw_sha256"] == digest(path.with_suffix(".raw.json.gz"))
        assert certificate["prediction_raw_sha256"] == digest(OUT / "predictions" / (case["family_key"] + ".raw.json.gz"))
        assert certificate["rank2_prediction_raw_sha256"] == digest(OUT / "predictions" / (case["family_key"] + ".rank2.raw.json.gz"))
        precision[str(value["precision_digits"])] += 1
        for model, key in (("adaptive", "rows"), ("fixed_rank2", "rank2_rows")):
            for row in certificate[key]:
                assert all(row[k] for k in ("remainder_covered", "entropy_remainder_covered", "distance_interval_covered", "entropy_interval_covered", "bases_orthogonal"))
                checks.append(dict(model=model, **row))
        for row in certificate["rank2_pairs"]:
            assert row["projector_interval_covered"] and row["entropy_interval_covered"]
            pair_checks.append(row)
        for actual in value["rows"]:
            prediction = predicted[case["case_id"], actual["initial"]]
            projector_passed = actual["distance_to_W"] <= .05
            states.append({**prediction, **actual,
                           "projector_selected_actual": projector_passed,
                           "entropy_selected_actual": actual["entropy_error"] <= .05,
                           "wrong_guarantee": (prediction["guaranteed_selected"] and not actual["controlled"]) or
                                               (prediction["guaranteed_unselected"] and actual["controlled"]),
                           "rank2_wrong_guarantee": (prediction["rank2_guaranteed_selected"] and not actual["controlled"]) or
                                                     (prediction["rank2_guaranteed_unselected"] and actual["controlled"]),
                           "rank1_misclassified": (prediction["rank1_distance_candidate"] <= .05) != projector_passed,
                           "rank2_misclassified": (prediction["rank2_distance_candidate"] <= .05) != projector_passed,
                           "rank1_distance_abs_error": abs(prediction["rank1_distance_candidate"] - actual["distance_to_W"]),
                           "rank2_distance_abs_error": abs(prediction["rank2_distance_candidate"] - actual["distance_to_W"])})
        for actual in value["pairs"]:
            prediction = memories[case["case_id"], actual["initial_a"], actual["initial_b"]]
            for prefix in ("", "rank2_"):
                assert prediction[prefix + "memory_lower"] - 1e-10 <= actual["pair_distance"] <= prediction[prefix + "memory_upper"] + 1e-10
                assert prediction[prefix + "entropy_memory_lower"] - 1e-10 <= actual["absolute_entropy_difference"] <= prediction[prefix + "entropy_memory_upper"] + 1e-10
                assert not prediction[prefix + "guaranteed_memory"] or actual["actual_memory"]
                assert not prediction[prefix + "guaranteed_entropy_memory"] or actual["actual_entropy_memory"]
            pairs.append({**prediction, **actual})
    fronts = []
    for hypothesis in read_json(OUT / "front_hypothesis_lock.json")["candidates"]:
        local = [r for r in states if r["family_key"] == hypothesis["family_key"] and r["initial"] == hypothesis["initial"]
                 and r["physical_time"] in hypothesis["times"]]
        assert len(local) == len(hypothesis["times"])
        for quantity, key in (("projector", "projector_selected_actual"), ("joint", "controlled")):
            result = front(local, key)
            lo, hi = result["last_fail"], result["first_persistent_sample_pass"]
            bracketed = lo is not None and hi is not None
            distance = lambda value: None if not bracketed else max(lo - value, value - hi, 0.)
            r1, r2 = hypothesis["rank1_candidate_time"], hypothesis["rank2_candidate_time"]
            fronts.append({k: v for k, v in hypothesis.items() if k != "times"} | dict(quantity=quantity, **result,
                bracket_width=None if not bracketed else hi - lo,
                rank1_candidate_compatible=None if not bracketed else lo < r1 <= hi,
                rank2_candidate_compatible=None if not bracketed else lo < r2 <= hi,
                rank1_distance_to_bracket=distance(r1), rank2_distance_to_bracket=distance(r2),
                rank2_gate_not_falsified=bracketed and distance(r2) <= hypothesis["root_abs_error_gate"],
                interpretation="local sampled bracket; not a continuous first-crossing guarantee"))
    shifts = []
    for length in (128, 144):
        for initial in ("left", "right"):
            baseline = next(r for r in fronts if r["family_key"] == f"L{length}_gL0" and r["initial"] == initial and r["quantity"] == "projector")
            for chi in (2, 4):
                target = next(r for r in fronts if r["family_key"] == f"L{length}_gL{chi}" and r["initial"] == initial and r["quantity"] == "projector")
                values = [baseline["last_fail"], baseline["first_persistent_sample_pass"], target["last_fail"], target["first_persistent_sample_pass"]]
                valid = all(x is not None for x in values)
                lower, upper = (values[2] - values[1], values[3] - values[0]) if valid else (None, None)
                candidate = target["rank2_candidate_time"] - baseline["rank2_candidate_time"]
                shifts.append(dict(length=length, initial=initial, gL=chi, rank2_candidate_shift=candidate,
                                   sampled_shift_lower=lower, sampled_shift_upper=upper,
                                   direction_resolved=valid and (upper < 0 if initial == "left" else lower > 0),
                                   candidate_compatible=valid and lower <= candidate <= upper,
                                   scope="difference of two local sampled intervals, conditional on tracking the same branch"))
    write_csv(OUT / "actual_entanglement.csv", states)
    write_csv(OUT / "initial_memory.csv", pairs)
    write_csv(OUT / "high_precision_remainder_checks.csv", checks)
    write_csv(OUT / "rank2_pair_checks.csv", pair_checks)
    write_csv(OUT / "fine_front_intervals.csv", fronts)
    write_csv(OUT / "weak_skin_time_shifts.csv", shifts)
    write_csv(OUT / "development_comparison.csv", development)
    write_csv(OUT / "kernel_benchmarks.csv", kernel_checks)
    projector_fronts = [r for r in fronts if r["quantity"] == "projector"]
    old_failures = [r for r in development if r["rank1_misclassified"]]
    tiers = {}
    for label, prefix in (("adaptive", ""), ("fixed_rank2", "rank2_")):
        tiers[label] = dict(selected=sum(r[prefix + "guaranteed_selected"] for r in states),
                            unselected=sum(r[prefix + "guaranteed_unselected"] for r in states),
                            undecided=sum(not (r[prefix + "guaranteed_selected"] or r[prefix + "guaranteed_unselected"]) for r in states),
                            wrong_guarantees=sum(r[prefix + "wrong_guarantee"] for r in states),
                            target_passed=sum(r["rank_target_passed" if not prefix else "rank2_target_passed"] for r in states),
                            projector_memory_guarantees=sum(r[prefix + "guaranteed_memory"] for r in pairs),
                            entropy_memory_guarantees=sum(r[prefix + "guaranteed_entropy_memory"] for r in pairs))
    targeted_states = [r for r in states if r["initial"] in r["target_initials"]]
    rank2_checks = {(r["case_id"], r["initial"]): r for r in checks if r["model"] == "fixed_rank2"}
    groups = {
        "charge_density_wave": [r for r in states if r["initial"] == "charge_density_wave"],
        "targeted_block": targeted_states,
        "companion_block": [r for r in states if r["initial"] != "charge_density_wave" and r["initial"] not in r["target_initials"]],
    }
    scope_diagnostics = {
        name: dict(states=len(rows),
                   projector_error_max=max(float(rank2_checks[r["case_id"], r["initial"]]["projector_error_mp"]) for r in rows),
                   distance_error_max=max(r["rank2_distance_abs_error"] for r in rows),
                   target_passed=sum(r["rank2_target_passed"] for r in rows),
                   undecided=sum(not (r["rank2_guaranteed_selected"] or r["rank2_guaranteed_unselected"]) for r in rows))
        for name, rows in groups.items()
    }
    value = dict(date="2026-10-04", status="completed_and_audited", families=len(cfg["families"]), conditions=len(cfg["cases"]),
                 states=len(states), pairs=len(pairs), development_states=len(development), kernel_checks=len(kernel_checks),
                 targeted_states=len(targeted_states),
                 targeted_rank1_misclassified=sum(r["rank1_misclassified"] for r in targeted_states),
                 targeted_rank2_misclassified=sum(r["rank2_misclassified"] for r in targeted_states),
                 fixed_rank2_scope_diagnostics=scope_diagnostics,
                 old_rank1_failures=len(old_failures), old_failures_corrected_by_rank2=sum(not r["rank2_misclassified"] for r in old_failures),
                 development_rank2_misclassified=sum(r["rank2_misclassified"] for r in development),
                 new_rank1_misclassified=sum(r["rank1_misclassified"] for r in states),
                 new_rank2_misclassified=sum(r["rank2_misclassified"] for r in states),
                 rank1_distance_error_max=max(r["rank1_distance_abs_error"] for r in states),
                 rank2_distance_error_max=max(r["rank2_distance_abs_error"] for r in states),
                 actual_selected=sum(r["controlled"] for r in states), actual_projector_memory=sum(r["actual_memory"] for r in pairs),
                 actual_entropy_memory=sum(r["actual_entropy_memory"] for r in pairs), tiers=tiers,
                 rank_max=max(r["rank"] for r in states), reference_precision=dict(precision),
                 fronts_bracketed=sum(r["status"] == "sampled_bracket" for r in projector_fronts),
                 strict_rank1_front_compatibility=sum(r["rank1_candidate_compatible"] is True for r in projector_fronts),
                 strict_rank2_front_compatibility=sum(r["rank2_candidate_compatible"] is True for r in projector_fronts),
                 rank2_root_gate_not_falsified=sum(r["rank2_gate_not_falsified"] for r in projector_fronts),
                 weak_shift_directions_resolved=sum(r["direction_resolved"] for r in shifts),
                 weak_shift_candidates_compatible=sum(r["candidate_compatible"] for r in shifts),
                 entropy_selected_projector_unselected=sum(r["entropy_selected_actual"] and not r["projector_selected_actual"] for r in states),
                 adaptive_projector_error_max=max(float(r["projector_error_mp"]) for r in checks if r["model"] == "adaptive"),
                 fixed_rank2_projector_error_max=max(float(r["projector_error_mp"]) for r in checks if r["model"] == "fixed_rank2"),
                 kernel_error_max=max(float(r["error_mp"]) for r in kernel_checks),
                 kernel_speedup_min=min(r["speedup"] for r in kernel_checks), kernel_speedup_max=max(r["speedup"] for r in kernel_checks),
                 scope=cfg["scope"])
    assert all(t["wrong_guarantees"] == 0 for t in tiers.values())
    write_json(OUT / "summary.json", value)
    write_json(OUT / "next_round_plan.json", dict(status="candidate_not_executed", source="ninth round", priorities=[
        "explain the remaining rank2 threshold error and distinguish radial distance prediction from companion-state reconstruction",
        "derive reflected inverse-Gram exponent, finite prefactor, and size-uniform crossing slope",
        "validate a second passive local no-click model with Hermitian adjoints and frozen predictions",
        "use measured L160/192 precision/cost gates to design new-size mechanism tests",
        "review closest literature and consolidate the manuscript"],
        fixed_tolerances=cfg["selection_tolerances"], memory_tolerances=cfg["memory_tolerances"],
        gates=["all ninth-round observations are development", "retain every undecided fixed-rank2 point",
               "no retrospective holdout", "no continuous first crossing from samples", "no universal rank2 claim"]))
    if plots:
        plot(states, checks, fronts, tiers)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def plot(states, checks, fronts, tiers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .18,
                         "axes.spines.top": False, "axes.spines.right": False})
    targeted = [r for r in states if r["initial"] in r["target_initials"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for ax, key, title, color in ((axes[0], "rank1_distance_candidate", "Rank 1", "#b13e48"),
                                 (axes[1], "rank2_distance_candidate", "Signed rank 2", "#007f86")):
        ax.scatter([r["distance_to_W"] for r in targeted], [r[key] for r in targeted], s=20, alpha=.65, color=color)
        lo = min(min(r["distance_to_W"], r[key]) for r in targeted) * .97
        hi = max(max(r["distance_to_W"], r[key]) for r in targeted) * 1.03
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=1)
        ax.axhline(.05, color="#333333", ls=":")
        ax.axvline(.05, color="#333333", ls=":")
        bad = [r for r in targeted if r["rank1_misclassified" if key.startswith("rank1") else "rank2_misclassified"]]
        if bad:
            ax.scatter([r["distance_to_W"] for r in bad], [r[key] for r in bad], marker="x", s=52,
                       color="#202020", linewidths=1.3, label=f"Threshold errors: {len(bad)}", zorder=4)
            ax.legend(loc="upper left", fontsize=9)
        ax.set(xlabel="Independent projector distance", ylabel="Frozen candidate distance", title=title,
               xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    fig.savefig(FIG / "rank2_candidate_validation.png", dpi=190)
    fig.savefig(FIG / "rank2_candidate_validation.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout="constrained")
    for ax, initial in zip(axes.flat, ("left", "right", "translated_first", "translated_second")):
        for length, offset, color in ((128, -.08, "#007f86"), (144, .08, "#b13e48")):
            name = initial if initial in ("left", "right") else f"block_cell{length//16 if initial == 'translated_first' else 3*length//16}"
            for position, protocol in enumerate(("gL0", "gL2", "gL4", "g0p25")):
                row = next(r for r in fronts if r["quantity"] == "projector" and r["family_key"] == f"L{length}_{protocol}" and r["initial"] == name)
                if row["last_fail"] is None or row["first_persistent_sample_pass"] is None:
                    continue
                lo, hi = row["last_fail"] - row["rank2_candidate_time"], row["first_persistent_sample_pass"] - row["rank2_candidate_time"]
                ax.errorbar(position + offset, (lo + hi)/2, yerr=(hi - lo)/2, fmt="o", color=color, capsize=4,
                            label=f"L={length}" if position == 0 else None)
                ax.scatter(position + offset, row["rank1_candidate_time"] - row["rank2_candidate_time"], marker="x", color=color)
        ax.axhline(0, ls="--", color="#333333", lw=.8)
        ax.set(title=initial.replace("_", " "), xticks=range(4), xticklabels=("gL=0", "gL=2", "gL=4", "g=0.25"),
               ylabel="Sampled t - rank2 candidate t")
    axes[0, 0].legend(fontsize=9)
    fig.savefig(FIG / "rank2_front_intervals.png", dpi=190)
    fig.savefig(FIG / "rank2_front_intervals.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    local = [r for r in checks if r["model"] == "fixed_rank2"]
    x = [max(float(r["remainder_mp"]), 1e-280) for r in local]
    y = [max(float(r["projector_error_mp"]), 1e-280) for r in local]
    axes[0].scatter(x, y, s=13, alpha=.5, color="#007f86")
    lo, hi = min(x + y) * .5, max(x + y) * 2
    axes[0].plot([lo, hi], [lo, hi], "--", color="#444444", lw=.8)
    axes[0].axvline(1e-5, ls=":", color="#b13e48")
    axes[0].set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
                xlabel="Frozen rank2 remainder budget", ylabel="Independent rank2 projector error")
    positions = np.arange(2)
    bottom = np.zeros(2)
    for key, label, color in (("selected", "Selected", "#007f86"), ("unselected", "Unselected", "#b13e48"),
                              ("undecided", "Undecided", "#879092")):
        values = [tiers[model][key] for model in ("fixed_rank2", "adaptive")]
        axes[1].bar(positions, values, bottom=bottom, label=label, color=color, width=.55)
        for i, value in enumerate(values):
            if value:
                axes[1].text(i, bottom[i] + value/2, str(value), ha="center", va="center", color="white", fontsize=10)
        bottom += values
    axes[1].set(xticks=positions, xticklabels=("Fixed rank 2", "Adaptive"), ylabel="New states", ylim=(0, len(states)*1.12))
    axes[1].legend(ncol=3, fontsize=8, loc="upper center")
    fig.savefig(FIG / "rank2_remainder_and_guarantees.png", dpi=190)
    fig.savefig(FIG / "rank2_remainder_and_guarantees.pdf")
    plt.close(fig)


def scope_check():
    run.setup()
    run.check_lock()
    checks = run.old.audit.read_csv(OUT / "high_precision_remainder_checks.csv")
    worst = max((r for r in checks if r["model"] == "fixed_rank2"), key=lambda r: float(r["projector_error_mp"]))
    case = next(r for r in run.config()["cases"] if r["case_id"] == worst["case_id"])
    statics = run.read_gzip(run.old.OUT / "static" / (case["family_key"] + ".raw.json.gz"))[-2:]
    raw = run.read_gzip(OUT / "predictions" / (case["family_key"] + ".rank2.raw.json.gz"))
    saved = next(r for r in raw["rows"] if r["case_id"] == case["case_id"] and r["initial"] == worst["initial"])
    reference = run.read_gzip(OUT / "cases" / (case["case_id"] + ".raw.json.gz"))[-1]
    rows, bases = [], []
    for static, digits in ((statics[0], statics[0]["dps"]), (statics[1], statics[1]["dps"]), (statics[1], 320)):
        with run.mp.workdps(digits):
            ctx = run.old.context(static, [worst["initial"]])[worst["initial"]]
            tc = run.old.time_context(ctx, run.mp.mpf(case["physical_time"]))
            values, q, _ = run.previous.at_rank(tc, 2)
            direct, _ = run.mp.qr(ctx["matrices"]["G"] + tc["y"][:, :2] * tc["v"][:2, :], mode="skinny")
            discrepancy = run.old.pair_distance(q, direct)
            orthogonality = run.old.p1.mp_norm(direct.T * direct - run.mp.eye(direct.cols))
            actual = run.old.spatial.decode(reference["states"][worst["initial"]]["Q"])
            error = run.old.pair_distance(direct, actual)
            assert discrepancy < run.mp.mpf("1e-35") and orthogonality < run.mp.mpf("1e-35")
            assert abs(error - run.mp.mpf(worst["projector_error_mp"])) < run.mp.mpf("1e-35")
            rows.append(dict(static_digits=static["dps"], working_digits=digits,
                             graph_vs_full_qr_mp=str(discrepancy), full_qr_orthogonality_mp=str(orthogonality),
                             full_qr_actual_error_mp=str(error), remainder_mp=str(values["projector_remainder"])))
            bases.append(direct)
    with run.mp.workdps(320):
        changes = [run.old.pair_distance(q, bases[-1]) for q in bases[:-1]]
        saved_error = run.old.pair_distance(run.old.spatial.decode(saved["Q"]), bases[-1])
        assert max([saved_error, *changes]) < run.mp.mpf("1e-35")
    value = dict(status="passed", case_id=case["case_id"], initial=worst["initial"], rows=rows,
                 precision_projector_changes_mp=[str(v) for v in changes], saved_basis_change_mp=str(saved_error),
                 reference_sha256=digest(OUT / "cases" / (case["case_id"] + ".raw.json.gz")),
                 policy="post-truth extremal representation audit; not a new-time prediction test", verified_utc=run.old.now())
    write_json(OUT / "fixed_rank2_scope_check.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def verify():
    run.setup()
    run.check_lock()
    summary = read_json(OUT / "summary.json")
    assert read_json(OUT / "fixed_rank2_scope_check.json")["status"] == "passed"
    assert summary["states"] == len(run.config()["cases"]) * 3
    assert all(t["wrong_guarantees"] == 0 for t in summary["tiers"].values())
    history = run.old.audit.read_csv(OUT / "input_manifest.csv")
    for row in history:
        path = ROOT / row["path"]
        assert path.stat().st_size == int(row["bytes"]) and digest(path) == row["sha256"], row["path"]
    for row in read_json(OUT / "prediction_lock.json")["inputs"]:
        assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
    generated = {(OUT / name).resolve() for name in ("artifact_manifest.csv", "delivery_verification.json")}
    deferred, links = [], 0
    documents = [ROOT / "README.md", *(ROOT / "research_doc").rglob("*.md")]
    for path in documents:
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = match.group(1).strip("<>").split("#", 1)[0]
            if not target or re.match(r"[a-z]+://", target) or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if resolved in generated:
                deferred.append(resolved)
            else:
                assert resolved.exists(), (path, target)
            links += 1
    excluded = {"artifact_manifest.csv", "delivery_verification.json", "deployment.tar.gz", "final_sync.tar.gz", "results.tar.gz",
                "return_manifest.csv", "final_sync_manifest.csv", "final_sync_receipt.json", "local_inventory.json", "operations.jsonl"}
    paths = [p for folder in (OUT, FIG) for p in folder.rglob("*") if p.is_file() and p.name not in excluded]
    paths.extend(p for p in (ROOT / "scripts").glob("*rank2_front*") if p.is_file())
    paths.extend(documents)
    rows = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(set(paths))]
    write_csv(OUT / "artifact_manifest.csv", rows)
    value = dict(status="passed", verified_utc=run.old.now(), historical_inputs=len(history), artifact_files=len(rows),
                 states=summary["states"], pairs=summary["pairs"], source_sha256=digest(run.SOURCE), local_document_links=links,
                 adaptive_wrong_guarantees=0, fixed_rank2_wrong_guarantees=0,
                 candidate_rank2_errors=summary["new_rank2_misclassified"])
    write_json(OUT / "delivery_verification.json", value)
    assert all(p.is_file() for p in deferred)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("summarize", "scope", "verify"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.stage == "summarize":
        summarize(not args.no_plots)
    elif args.stage == "scope":
        scope_check()
    else:
        verify()


if __name__ == "__main__":
    main()
