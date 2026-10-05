"""Reproduce released figure data and the independent short-time SSH pilot."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(relative: str) -> list[dict]:
    with (ROOT / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify() -> dict:
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Manifest path outside the checkout")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Released file differs: {item['path']}")
    return dict(status="passed", files=len(manifest["files"]))


def pilot() -> dict:
    import mpmath as mp
    import numpy as np
    import scipy
    from scipy import linalg
    from mietf_skin.ssh import ssh_no_click_hamiltonian

    parameters = json.loads((ROOT / "configs/short_time_pilot.json").read_text(encoding="utf-8"))
    length, skin, time = parameters["length"], parameters["skin"], parameters["time"]
    n = length // 2
    h = ssh_no_click_hamiltonian(n, .5, 1., 2., skin, "open")
    m = linalg.expm(-1j * h * time)
    w = linalg.svd(m)[0][:, :n]
    c = 2 + np.sinh(skin)
    states = {"charge_density_wave": list(range(0, length, 2)),
              "left": list(range(n)), "right": list(range(n, length))}
    rows = [r for r in read_csv("data/prl_e_feasibility/common_loss_states.csv")
            if r["model"] == "ssh" and int(r["length"]) == length
            and float(r["skin"]) == skin and float(r["physical_time"]) == time]
    if len(rows) != 3:
        raise ValueError("Expected three frozen pilot rows")
    expected = {r["initial"]: r for r in rows}
    report = []
    with mp.workdps(parameters["precision_digits"]):
        # Build the high-precision Hamiltonian independently of the float64 module.
        hm = mp.matrix(length)
        gm, tm = mp.mpf(str(skin)), mp.mpf(str(time))
        for j in range(n):
            hm[2*j, 2*j], hm[2*j+1, 2*j+1] = 2j, -2j
            hm[2*j, 2*j+1] = hm[2*j+1, 2*j] = mp.mpf(".5")
            if j + 1 < n:
                hm[2*j+2, 2*j+1] = mp.exp(gm)
                hm[2*j+1, 2*j+2] = mp.exp(-gm)
        mm = mp.expm(-mp.j * hm * tm)
        wm = mp.svd(mm)[0][:, :n]
        cm = 2 + mp.sinh(gm)
        for name, occupied in states.items():
            q0 = np.eye(length)[:, occupied]
            x = m @ q0
            q = linalg.qr(x, mode="economic")[0]
            distance = float(linalg.norm(q - w @ (w.conj().T @ q), 2))
            values = linalg.eigvalsh(q[:n] @ q[:n].conj().T)
            if np.min(values) < -1e-12 or np.max(values) > 1 + 1e-12:
                raise ArithmeticError("Correlation eigenvalues outside the physical interval")
            values = np.clip(values, 0., 1.)
            entropy = float(np.sum(scipy.special.entr(values) + scipy.special.entr(1-values)))
            sign, logdet = np.linalg.slogdet(x.conj().T @ x)
            if abs(sign - 1) > 1e-10:
                raise ArithmeticError("Nonpositive occupied Gram determinant")
            log_success = float(logdet - 2 * n * c * time)
            q0m = mp.matrix(length, n)
            for k, site in enumerate(occupied):
                q0m[site, k] = 1
            xm = mm * q0m
            qm = mp.qr(xm, mode="skinny")[0]
            dm = mp.svd(qm - wm * (wm.H * qm), compute_uv=False)[0]
            ev = mp.eighe(qm[:n, :] * qm[:n, :].H, eigvals_only=True)
            em = mp.fsum(-v*mp.log(v) - (1-v)*mp.log(1-v) for v in ev if 0 < v < 1)
            lm = mp.re(mp.log(mp.det(xm.H * xm))) - 2*n*cm*tm
            actual = dict(distance_to_W=distance, entropy=entropy,
                          log_no_click_success_common=log_success)
            high = dict(distance_to_W=float(dm), entropy=float(em),
                        log_no_click_success_common=float(lm))
            errors = {key: abs(value-float(expected[name][key])) for key, value in actual.items()}
            precision_errors = {key: abs(value-high[key]) for key, value in actual.items()}
            if max(*errors.values(), *precision_errors.values()) > parameters["comparison_tolerance"]:
                raise ArithmeticError(f"Pilot comparison failed for {name}")
            report.append(dict(initial=name, **actual, no_loss_probability=float(np.exp(log_success)),
                               table_errors=errors, float64_high_precision_errors=precision_errors))
    return dict(status="passed", parameters=parameters, states=report,
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, mpmath=mp.__version__))


def analytic_supplement(directory: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from analyze_theory_upgrade import constants

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7), layout="constrained")
    hs = np.linspace(0, .25, 101)
    cc = [constants(g) for g in hs]
    axes[0, 0].plot(hs, [r["rho"] for r in cc], label=r"Graph bound $\rho$")
    axes[0, 0].plot(hs, [r["kappa_cone"] for r in cc], label=r"Contraction radius $2a/b$")
    axes[0, 0].set(xlabel=r"Skin magnitude $|g|$", ylabel="Tangent radius")
    axes[0, 0].legend()
    ts = np.linspace(0, 6, 240)
    for g, color in ((0., "#277db1"), (.25, "#be4343")):
        c = constants(g)
        decay = np.exp(-2*c["a"]*ts)
        k = c["rho"]*decay/(1-c["b"]*c["rho"]/(2*c["a"])*(1-decay))
        axes[0, 1].semilogy(ts, k/np.sqrt(1+k*k), color=color, label=f"CDW bound, g={g:g}")
        axes[0, 1].semilogy(ts, decay, color=color, ls="--", lw=.8)
    axes[0, 1].set(xlabel="Time t", ylabel="Distance upper bound", ylim=(1e-7, 1))
    axes[0, 1].legend(loc="lower left")
    c = constants(.25)
    eta = (1-c["rho"]**2)/(1+c["rho"]**2)
    rate = (1.25+np.cosh(.25))/4
    const = (.5+np.exp(.25))/(4*(1-rate))
    decay = np.exp(-2*c["a"]*ts)
    kc = c["rho"]*decay/(1-c["b"]*c["rho"]/(2*c["a"])*(1-decay))
    cdw = kc/np.sqrt(1+kc*kc)
    for length, color in ((128, "#277db1"), (256, "#be4343"), (512, "#177c57")):
        zeta = const*rate**(length//4-1)/eta
        lower = eta/(1+zeta/(1-zeta)*np.exp(2*(2.5+np.exp(.25))*ts))
        axes[1, 0].plot(ts, np.maximum(0, lower-cdw), color=color, label=f"L={length}")
    axes[1, 0].axhline(.1, color=".4", lw=.8, ls=":")
    axes[1, 0].set(xlabel="Time t", ylabel="Block-CDW memory lower bound")
    axes[1, 0].legend()
    references = read_csv("data/prl_theory_upgrade/reference_checks.csv")
    sx = [-float(r["theoretical_slope_upper"]) for r in references]
    sy = [-float(r["actual_slope"]) for r in references]
    axes[1, 1].scatter(sx, sy, s=9, alpha=.6, color="#277db1")
    hi = max(sy)*1.08
    axes[1, 1].plot([0, hi], [0, hi], color=".4", ls=":", lw=.8)
    axes[1, 1].set(xlabel="Guaranteed decay slope", ylabel="Independent actual decay slope",
                   xlim=(0, hi), ylim=(0, hi))
    for letter, ax in zip("abcd", axes.flat):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.5, -.24, f"({letter})", transform=ax.transAxes, ha="center")
    for ext in ("pdf", "png"):
        fig.savefig(directory / ("analytic_envelopes." + ext), dpi=220)
    plt.close(fig)


def figures(output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_prl_manuscript_figures as drawing
    import analyze_completion_research as supplement

    drawing.FIG = output / "main_figures"
    drawing.FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "axes.linewidth": .7, "legend.fontsize": 7})
    info = {
        "fig1_occupations": drawing.occupations(),
        "fig2_geometry_memory": drawing.flagship(read_csv(drawing.INPUTS[0]), read_csv(drawing.INPUTS[1]), False),
        "fig3_reflection_response": drawing.geometry(read_csv(drawing.INPUTS[5]), read_csv(drawing.INPUTS[4]), False),
        "fig4_continuous_window": drawing.window(read_csv(drawing.INPUTS[2]), read_csv(drawing.INPUTS[3]), read_csv(drawing.INPUTS[1]), False),
    }
    supplement.FIG = output / "supplement_figures"
    supplement.figures()
    analytic_supplement(supplement.FIG)
    files = sorted(p for p in output.rglob("*") if p.suffix in (".pdf", ".png"))
    return dict(status="passed", main_figures=info, outputs=[dict(path=p.relative_to(output).as_posix(),
                      bytes=p.stat().st_size, sha256=digest(p)) for p in files],
                scope="Redrawing from released tables; no recomputation of large-size dynamics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "pilot", "figures", "all"))
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction_output")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or output.is_relative_to(ROOT / "data") or output.is_relative_to(ROOT / "figures"):
        parser.error("Use a separate output directory, not the frozen data or figure directories")
    output.mkdir(parents=True, exist_ok=True)
    stages = ("verify", "pilot", "figures") if args.stage == "all" else (args.stage,)
    report = {"created_utc": datetime.now(timezone.utc).isoformat()}
    for stage in stages:
        report[stage] = verify() if stage == "verify" else pilot() if stage == "pilot" else figures(output)
        print(json.dumps({"stage": stage, "status": report[stage]["status"]}), flush=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
