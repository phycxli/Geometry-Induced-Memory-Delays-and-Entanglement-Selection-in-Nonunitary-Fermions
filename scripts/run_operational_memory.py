"""Frozen single-site memory protocol, control checks, and independent references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import mpmath as mp
import numpy as np
import scipy
from scipy import linalg, stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prl_operational_memory"
SOURCE = Path(__file__)
PARAMETERS = ("time_offset", "skin_offset", "intra_relative", "inter_relative",
              "loss_a_relative", "loss_b_relative", "phase_error")
BOUNDS = (.02, .01, .02, .02, .02, .02, .03)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def occupied(length, name):
    if name == "cdw":
        return list(range(0, length, 2))
    if name == "left":
        return list(range(length // 2))
    return list(range(length // 2, length))


def physical_hamiltonian(length, skin=.15, errors=None, auxiliary=None):
    errors = errors or {}
    g = skin + errors.get("skin_offset", 0.)
    assert g > 0
    cells = length // 2
    q = np.sinh(g)
    q_nominal = np.sinh(skin)
    size = length if auxiliary is None else length + cells - 1
    coherent = np.zeros((size, size), complex)
    damping = np.zeros((size, size), complex)
    for j in range(cells):
        a, b = 2 * j, 2 * j + 1
        coherent[a, b] = coherent[b, a] = .5 * (1 + errors.get("intra_relative", 0.))
        residual_a = q_nominal if j == 0 else 0.
        residual_b = 4. + (q_nominal if j == cells - 1 else 0.)
        damping[a, a] += residual_a * (1 + errors.get("loss_a_relative", 0.))
        damping[b, b] += residual_b * (1 + errors.get("loss_b_relative", 0.))
        if j + 1 < cells:
            other = 2 * j + 2
            coherent[other, b] = coherent[b, other] = np.cosh(g) * (1 + errors.get("inter_relative", 0.))
            phase = 1j * np.exp(1j * errors.get("phase_error", 0.))
            if auxiliary is None:
                row = np.zeros(size, complex)
                row[other], row[b] = np.sqrt(q), np.sqrt(q) * phase
                damping += np.outer(row.conj(), row)
            else:
                d = length + j
                v = np.sqrt(auxiliary * q / 2)
                coherent[d, other], coherent[d, b] = v, v * phase
                coherent[other, d], coherent[b, d] = v, v * phase.conjugate()
                damping[d, d] = auxiliary / 2
    assert np.min(linalg.eigvalsh(damping)) >= -1e-12
    return coherent - 1j * damping


def at_point(length, duration, skin=.15, site=0, errors=None, auxiliary=None):
    h = physical_hamiltonian(length, skin, errors, auxiliary)
    duration += (errors or {}).get("time_offset", 0.)
    full = linalg.expm(-1j * h * duration)
    matrix = full[:length, :length]
    w, singular, _ = linalg.svd(matrix)
    pw = w[:, :length // 2] @ w[:, :length // 2].conj().T
    rates, modes = linalg.eig(-1j * physical_hamiltonian(length, skin) + (2 + np.sinh(skin)) * np.eye(length))
    growing = linalg.qr(modes[:, rates.real > 0], mode="economic")[0]
    pg = growing @ growing.conj().T
    states = {}
    for name in ("cdw", "left", "right"):
        x = matrix[:, occupied(length, name)]
        basis, r = linalg.qr(x, mode="economic")
        p = basis @ basis.conj().T
        p0 = float(np.prod(abs(np.diag(r)) ** 2))
        states[name] = dict(occupation=float(p[site, site].real), success=p0,
                            distance_to_W=float(linalg.norm(p - pw, 2)),
                            distance_to_nominal_G=float(linalg.norm(p - pg, 2)),
                            projector=p)
    signal = states["cdw"]["occupation"] - states["right"]["occupation"]
    return dict(length=length, time=duration, skin=skin, site=site,
                auxiliary=auxiliary or 0, signal=signal, witness=-signal,
                singular_ratio=float(singular[length // 2] / singular[length // 2 - 1]),
                direct_memory=float(linalg.norm(states["cdw"]["projector"] - states["right"]["projector"], 2)),
                states=states)


def flat(value):
    row = {k: v for k, v in value.items() if k != "states"}
    for name, state in value["states"].items():
        row.update({name + "_" + k: v for k, v in state.items() if k != "projector"})
    return row


def develop():
    if (OUT / "protocol_lock.json").exists():
        raise RuntimeError("Protocol already frozen; development must not overwrite it.")
    files = ["main/main.tex", "supplement/supplement.tex", "supplement/asymptotic_time_proof.tex",
             "main_zh/main.tex", "supplment_zh/supplement.tex", "README.md", "research_doc/研究文档导航.md"]
    files += [project + "/refs.bib" for project in ("main", "supplement", "main_zh", "supplment_zh")]
    baseline = []
    for relative in files:
        target = OUT / "baseline" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        baseline.append(dict(path=relative, sha256=sha(target)))
    write_json(OUT / "baseline_manifest.json", baseline)
    rows = []
    for length, duration, site in itertools.product((4, 8), (.3, .4, .5, .6, .7, .8), range(4)):
        rows.append(flat(at_point(length, duration, site=site)))
    write_csv(OUT / "development.csv", rows)
    value = dict(completed_utc=now(), source_sha256=sha(SOURCE), rows=len(rows),
                 selected_candidate=flat(at_point(4, .5)),
                 scope="Development only: chosen after looking at small-system signal/cost tradeoffs; not blind validation.")
    write_json(OUT / "development.json", value)
    return value


def freeze():
    if (OUT / "protocol_lock.json").exists():
        raise RuntimeError("A protocol lock already exists.")
    assert read_json(OUT / "development.json")["source_sha256"] == sha(SOURCE)
    value = dict(locked_utc=now(), source_sha256=sha(SOURCE), development_sha256=sha(OUT / "development.csv"),
                 nominal=dict(length=4, particles=2, skin=.15, time=.5, site_zero_based=0,
                              readout="A1 site occupation; same fixed readout for both preparations"),
                 controls=dict(zip(PARAMETERS, BOUNDS)),
                 gate=dict(signal_min=.90, cdw_to_W_max=.06, right_to_W_min=.90, singular_ratio_max=.30, success_min=.007),
                 validation=dict(corners=128, random=256, random_seed=20261010, auxiliary_rate=100.,
                                 heldout_times=[.45, .475, .525, .55, .6]),
                 statistics=dict(accepted_per_preparation=1000, conditional_failure_probability=.05,
                                 false_accepted_fraction_max=.01, binary_readout_bias_max=.02,
                                 orbital_projector_distance_max=.01),
                 scope="One local readout, two separate preparations; effective-model development followed by frozen control validation; no experimental data.")
    write_json(OUT / "protocol_lock.json", value)
    return value


def check_lock():
    lock = read_json(OUT / "protocol_lock.json")
    assert lock["source_sha256"] == sha(SOURCE), "Source changed after protocol freeze."
    return lock


def fock_reference(length, duration):
    sites = list(itertools.combinations(range(length), length // 2))
    states = [sum(1 << j for j in occupied_sites) for occupied_sites in sites]
    index = {bits: i for i, bits in enumerate(states)}
    single = physical_hamiltonian(length)
    many = np.zeros((len(states), len(states)), complex)
    bilinears = {}
    for i, j in itertools.product(range(length), repeat=2):
        op = np.zeros_like(many)
        for column, bits in enumerate(states):
            if not bits & (1 << j):
                continue
            sign = (-1) ** ((bits & ((1 << j) - 1)).bit_count())
            after = bits ^ (1 << j)
            if after & (1 << i):
                continue
            sign *= (-1) ** ((after & ((1 << i) - 1)).bit_count())
            op[index[after | (1 << i)], column] = sign
        bilinears[i, j] = op
        many += single[i, j] * op
    propagation = linalg.expm(-1j * many * duration)
    reference = at_point(length, duration)
    rows = []
    for name in ("cdw", "left", "right"):
        initial = sum(1 << j for j in occupied(length, name))
        psi = propagation[:, index[initial]]
        probability = float(np.vdot(psi, psi).real)
        psi = psi / np.sqrt(probability)
        correlation = np.array([[np.vdot(psi, bilinears[j, i] @ psi) for j in range(length)] for i in range(length)])
        target = reference["states"][name]
        error = float(linalg.norm(correlation - target["projector"], 2))
        p_error = abs(probability - target["success"])
        assert max(error, p_error) < 1e-10
        rows.append(dict(length=length, initial=name, number_sector_dimension=len(states),
                         projector_error=error, success_error=p_error))
    return rows


def precision_reference(length, digits):
    with mp.workdps(digits):
        g, duration, c = mp.mpf(".15"), mp.mpf(".5"), 2 + mp.sinh(mp.mpf(".15"))
        h = mp.matrix(length)
        for j in range(length // 2):
            h[2*j, 2*j], h[2*j+1, 2*j+1] = 2*mp.j-mp.j*c, -2*mp.j-mp.j*c
            h[2*j, 2*j+1] = h[2*j+1, 2*j] = mp.mpf(".5")
            if j+1 < length//2:
                h[2*j+2, 2*j+1], h[2*j+1, 2*j+2] = mp.exp(g), mp.exp(-g)
        matrix = mp.expm(-mp.j * h * duration)
        out = {}
        for name in ("cdw", "left", "right"):
            x = mp.matrix([[matrix[i, j] for j in occupied(length, name)] for i in range(length)])
            q, r = mp.qr(x, mode="skinny")
            p = q * q.H
            out[name] = dict(projector=[[[str(mp.re(p[i,j])), str(mp.im(p[i,j]))] for j in range(length)] for i in range(length)],
                             occupation=str(mp.re(p[0,0])), success=str(mp.fprod(abs(r[i,i])**2 for i in range(length//2))))
        return out


def validate():
    lock = check_lock()
    started = now()
    nominal = flat(at_point(4, .5))
    variables = list(lock["controls"])
    bounds = np.array([lock["controls"][v] for v in variables])
    samples = [np.array(signs) * bounds for signs in itertools.product((-1, 1), repeat=len(variables))]
    rng = np.random.default_rng(lock["validation"]["random_seed"])
    samples += list(rng.uniform(-1, 1, size=(256, len(variables))) * bounds)
    rows = []
    for auxiliary in (None, 100.):
        for number, vector in enumerate(samples):
            errors = dict(zip(variables, vector))
            row = flat(at_point(4, .5, errors=errors, auxiliary=auxiliary))
            row.update(errors)
            row.update(case=number, role="corner" if number < 128 else "seeded_random")
            gate = lock["gate"]
            row["passed"] = (row["signal"] >= gate["signal_min"] and row["cdw_distance_to_W"] <= gate["cdw_to_W_max"]
                             and row["right_distance_to_W"] >= gate["right_to_W_min"]
                             and row["singular_ratio"] <= gate["singular_ratio_max"]
                             and row["right_success"] >= gate["success_min"])
            rows.append(row)
    write_csv(OUT / "control_validation.csv", rows)
    heldout = [flat(at_point(4, t, auxiliary=a)) for a in (None, 100.) for t in lock["validation"]["heldout_times"]]
    write_csv(OUT / "heldout_times.csv", heldout)
    auxiliary_rows = [flat(at_point(length, .5, auxiliary=a)) for length in (4,8) for a in (50.,100.,200.,500.,1000.)]
    write_csv(OUT / "finite_auxiliary.csv", auxiliary_rows)
    precision, changes = {}, []
    for length in (4, 8):
        references = {str(d): precision_reference(length, d) for d in (60, 90)}
        precision[str(length)] = references
        floating = at_point(length, .5)
        with mp.workdps(100):
            for name in ("cdw", "left", "right"):
                low, high = references["60"][name], references["90"][name]
                error = max(abs(mp.mpf(low[k]) - mp.mpf(high[k])) for k in ("occupation", "success"))
                flo_error = max(abs(float(high[k]) - floating["states"][name][k]) for k in ("occupation", "success"))
                changes.append(dict(length=length, initial=name, scalar_precision_change=float(error), float_scalar_error=flo_error))
                assert error < mp.mpf("1e-50") and flo_error < 1e-12
    write_json(OUT / "precision_references.json", precision)
    independent = fock_reference(4, .5) + fock_reference(8, .5)
    write_json(OUT / "independent_checks.json", dict(precision=changes, fock=independent,
              scope="Separate arbitrary-precision one-particle calculation and exact fixed-number Fock-space evolution; internal verification, not external proof review."))
    stats_lock = lock["statistics"]
    n = stats_lock["accepted_per_preparation"]
    radius = 2 * math.sqrt(math.log(4 / stats_lock["conditional_failure_probability"]) / (2*n))
    systematic = 2 * sum(stats_lock[k] for k in ("false_accepted_fraction_max", "binary_readout_bias_max", "orbital_projector_distance_max"))
    minimum_signal = min(row["signal"] for row in rows)
    minimum_success = min(row["right_success"] for row in rows)
    budget = {}
    for role, p_cdw, p_right in (("nominal", nominal["cdw_success"], nominal["right_success"]),
                                 ("control_grid_worst_case", min(row["cdw_success"] for row in rows), minimum_success)):
        p0s = {"cdw": p_cdw, "right": p_right}
        budget[role] = dict(expected_attempts={key:n/p for key,p in p0s.items()},
                           trials_sufficient_for_both_counts_with_95pct_probability={key:int(stats.nbinom.ppf(.975,n,p))+n for key,p in p0s.items()},
                           false_accept_probability_per_rejected_run_max={key:stats_lock["false_accepted_fraction_max"]*p/((1-stats_lock["false_accepted_fraction_max"])*(1-p)) for key,p in p0s.items()})
    dynamics = [flat(at_point(length, float(t))) for length in (4,8) for t in np.linspace(.05,4,80)]
    write_csv(OUT / "time_curves.csv", dynamics)
    summary = dict(started_utc=started, completed_utc=now(), protocol_lock_sha256=sha(OUT / "protocol_lock.json"),
                   source_sha256=sha(SOURCE), status="passed" if all(r["passed"] for r in rows) else "gate_failed",
                   nominal=nominal, four_particle_comparison=flat(at_point(8,.5)),
                   validations=len(rows), failed_cases=sum(not r["passed"] for r in rows),
                   minimum_control_signal=minimum_signal, maximum_cdw_distance_to_W=max(r["cdw_distance_to_W"] for r in rows),
                   minimum_right_distance_to_W=min(r["right_distance_to_W"] for r in rows),
                   maximum_singular_ratio=max(r["singular_ratio"] for r in rows), minimum_right_success=minimum_success,
                   accepted_per_preparation=n, conditional_95pct_difference_error=radius, systematic_budget=systematic,
                   worst_grid_signal_after_systematics_and_statistics=minimum_signal-systematic-radius,
                   minimum_block_distance_from_W_inferred_from_witness=minimum_signal-systematic-radius-max(r["cdw_distance_to_W"] for r in rows),
                   attempts=budget, exact_fock_checks=len(independent), high_precision_checks=len(changes),
                   runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__),
                   limitations=["Finite parameter grid and seeded random cases do not certify the entire continuous control box.",
                     "Statistical confidence is conditional on collecting the prescribed accepted samples and calibrated systematic bounds.",
                     "Successful atom-number postselection needs low false acceptance; this is a calibration requirement, not demonstrated performance.",
                     "Small-system short-time memory does not establish large-size experimental feasibility or entanglement tomography.",
                     "No hardware data and no external independent proof review."])
    write_json(OUT / "summary.json", summary)
    return {k:v for k,v in summary.items() if k not in ("attempts", "four_particle_comparison", "limitations", "runtime")}


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lock = check_lock()
    summary = read_json(OUT / "summary.json")
    with (OUT / "time_curves.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.linewidth":.8,
                         "pdf.fonttype":42, "ps.fonttype":42, "savefig.facecolor":"white"})
    fig, axes = plt.subplots(1,3,figsize=(10.3,2.85), layout="constrained")
    colors = {"cdw":"#007A87", "right":"#C44C43"}
    for length, style in ((4,"-"),(8,"--")):
        values = [row for row in rows if int(row["length"])==length]
        times = np.array([float(row["time"]) for row in values])
        for name in ("cdw", "right"):
            axes[0].plot(times,[float(row[name+"_distance_to_W"]) for row in values],style,
                         color=colors[name],label=f"{name.upper() if name=='cdw' else 'Right'}, L={length}")
        axes[1].plot(times,[float(row["signal"]) for row in values],style,color="#424242",label=f"L={length}")
        axes[2].semilogy(times,[float(row["right_success"]) for row in values],style,color=colors["right"],label=f"Right, L={length}")
    axes[0].set(xlabel=r"Time $t$",ylabel=r"Distance to $P_W$",ylim=(-.02,1.04),xlim=(0,4))
    axes[1].set(xlabel=r"Time $t$",ylabel=r"Fixed-site contrast $-w_{A_1}$",ylim=(-.02,1.04),xlim=(0,4))
    axes[2].set(xlabel=r"Time $t$",ylabel=r"No-loss probability $P_0$",xlim=(0,1.25),ylim=(1e-7,1))
    axes[1].errorbar(.5,summary["nominal"]["signal"],yerr=summary["systematic_budget"]+summary["conditional_95pct_difference_error"],
                     fmt="o",color="#007A87",ms=4,capsize=3,label="L=4 error budget")
    for index, ax in enumerate(axes):
        ax.axvline(lock["nominal"]["time"],color="#777777",ls=":",lw=.9)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=7.5,loc="best")
        ax.text(.5,-.28,f"({chr(97+index)})",transform=ax.transAxes,ha="center",va="center")
    target = ROOT / "figures/prl_operational_memory"
    target.mkdir(parents=True,exist_ok=True)
    for suffix in ("pdf","png"):
        fig.savefig(target/("fixed_site_memory."+suffix),dpi=220)
    plt.close(fig)
    for project in ("supplement","supplment_zh"):
        destination=ROOT/project/"figures/prl_operational_memory"
        destination.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target/"fixed_site_memory.pdf",destination/"fixed_site_memory.pdf")
    write_json(OUT/"figure_provenance.json",dict(source_sha256=sha(SOURCE),curves_sha256=sha(OUT/"time_curves.csv"),
                protocol_lock_sha256=sha(OUT/"protocol_lock.json"),pdf_sha256=sha(target/"fixed_site_memory.pdf"),
                scope="Scientific plot from fixed-site calculations; uncertainty bar is the declared combined budget, not measured scatter."))
    return dict(status="figure_written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("develop","freeze","validate","figures"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
