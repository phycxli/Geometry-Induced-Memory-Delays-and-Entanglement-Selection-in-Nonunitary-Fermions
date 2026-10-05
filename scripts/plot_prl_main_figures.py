from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    iso = read_rows(ROOT / "data" / "physical_isospectral_skin_v1.csv")
    controlled = read_rows(ROOT / "data" / "controlled_overlap_v1_controlled_overlap_scan.csv")
    joint = read_rows(ROOT / "data" / "quicklarge_ssh_joint_scan.csv")
    lt = read_rows(ROOT / "data" / "singular_lt_scaling_v1.csv")
    lt_summary = read_rows(ROOT / "data" / "singular_lt_scaling_v1_summary.csv")

    summary = build_summary(iso, controlled, joint, lt_summary)
    write_rows(ROOT / "data" / "prl_main_figures_summary.csv", summary)
    plot_theorem_counterexample(iso, controlled, summary, ROOT / "figures" / "prl_main_fig1_theorem_counterexample.png")
    plot_collapse_scaling(joint, lt, lt_summary, summary, ROOT / "figures" / "prl_main_fig2_collapse_scaling.png")
    for row in summary:
        print(f"{row['metric']},{row['value']}")


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            converted = {}
            for key, value in row.items():
                if key in {"boundary", "model", "preferred_scaling", "metric"}:
                    converted[key] = value
                else:
                    try:
                        converted[key] = float(value)
                    except ValueError:
                        converted[key] = value
            rows.append(converted)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def build_summary(iso: list[dict], controlled: list[dict], joint: list[dict], lt_summary: list[dict]) -> list[dict]:
    max_l = max(row["length"] for row in iso)
    iso_l = [row for row in iso if row["length"] == max_l]
    g0 = min(iso_l, key=lambda row: row["skin"])
    gmax = max(iso_l, key=lambda row: row["skin"])
    h_drift = max(row["h_trace_moment_max_abs_diff"] for row in iso)
    eig_drift = max(row["h_eig_hausdorff_abs_diff"] for row in iso)
    gbz_width = max(abs(row["gbz_imag_width"]) for row in iso_l)

    controlled_unique = collapse_duplicate_controlled_rows(controlled)
    overlap_corr = pearson(
        np.asarray([row["top_overlap_min_sv"] for row in controlled_unique]),
        np.asarray([row["target_overlap"] for row in controlled_unique]),
    )
    projector_corr = pearson(
        np.asarray([row["projector_spectral_left"] for row in controlled_unique]),
        np.asarray([row["sin_bound_actual_graph"] for row in controlled_unique]),
    )
    entropy_trace_corr = pearson(
        np.asarray([row["entropy_diff_left"] for row in controlled_unique]),
        np.asarray([row["restricted_trace_left"] for row in controlled_unique]),
    )
    audenaert_violation = max(row["entropy_diff_left"] - row["entropy_audenaert_bound_left"] for row in controlled_unique)

    r_left = pearson(
        np.asarray([row["left_subspace_entropy_density"] for row in joint]),
        np.asarray([row["S_density"] for row in joint]),
    )
    r_gbz = pearson(
        np.asarray([row["gbz_imag_width"] for row in joint]),
        np.asarray([row["S_density"] for row in joint]),
    )
    r_bloch = pearson(
        np.asarray([row["bloch_imag_width"] for row in joint]),
        np.asarray([row["S_density"] for row in joint]),
    )
    r_sv_gap = pearson(
        np.asarray([row["sv_lyapunov_gap"] for row in joint]),
        np.asarray([row["S_density"] for row in joint]),
    )

    lt_skin05 = next(row for row in lt_summary if row["boundary"] == "open" and row["skin"] == 0.25 and row["gamma"] == 0.5)
    lt_skin_all = [row for row in lt_summary if row["boundary"] == "open" and row["skin"] == 0.25]
    lt_recip05 = next(row for row in lt_summary if row["boundary"] == "open" and row["skin"] == 0.0 and row["gamma"] == 0.5)

    metrics = {
        "isospectral_largest_length": max_l,
        "isospectral_entropy_g0": g0["left_entropy"],
        "isospectral_entropy_gmax": gmax["left_entropy"],
        "isospectral_entropy_drop": g0["left_entropy"] - gmax["left_entropy"],
        "isospectral_entropy_ratio_gmax_over_g0": gmax["left_entropy"] / g0["left_entropy"],
        "isospectral_h_trace_moment_max_abs_diff": h_drift,
        "isospectral_h_eig_hausdorff_max_abs_diff": eig_drift,
        "isospectral_gbz_width_largest_family_max": gbz_width,
        "controlled_unique_points": len(controlled_unique),
        "controlled_overlap_corr": overlap_corr,
        "controlled_projector_exact_corr": projector_corr,
        "controlled_entropy_trace_corr": entropy_trace_corr,
        "controlled_audenaert_violation_max": audenaert_violation,
        "joint_points": len(joint),
        "collapse_corr_left_density": r_left,
        "collapse_corr_gbz_width": r_gbz,
        "collapse_corr_bloch_width": r_bloch,
        "collapse_corr_sv_gap": r_sv_gap,
        "lt_skin_gamma05_log_slope_min": lt_skin05["log_slope_min"],
        "lt_skin_gamma05_log_slope_max": lt_skin05["log_slope_max"],
        "lt_skin_gamma05_area_limit_min": lt_skin05["area_limit_min"],
        "lt_skin_gamma05_area_limit_max": lt_skin05["area_limit_max"],
        "lt_skin_all_abs_log_slope_max": max(max(abs(row["log_slope_min"]), abs(row["log_slope_max"])) for row in lt_skin_all),
        "lt_recip_gamma05_log_slope_min": lt_recip05["log_slope_min"],
        "lt_recip_gamma05_log_slope_max": lt_recip05["log_slope_max"],
    }
    return [{"metric": key, "value": value} for key, value in metrics.items()]


def collapse_duplicate_controlled_rows(rows: list[dict]) -> list[dict]:
    by_key = {}
    for row in rows:
        key = (
            row["model"],
            row["length"],
            row["gain_loss"],
            row["skin"],
            row["boundary"],
            row["steps"],
            row["target_overlap"],
            row["theta"],
            row["mixing_seed"],
        )
        by_key[key] = row
    return list(by_key.values())


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def metric(summary: list[dict], key: str) -> float:
    return float(next(row["value"] for row in summary if row["metric"] == key))


def plot_theorem_counterexample(rows: list[dict], controlled: list[dict], summary: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    lengths = sorted({row["length"] for row in rows})
    colors = dict(zip(lengths, plt.cm.viridis(np.linspace(0.15, 0.9, len(lengths)))))

    ax = axes[0, 0]
    for length in lengths:
        sub = sorted([row for row in rows if row["length"] == length], key=lambda row: row["skin"])
        ax.plot(
            [row["skin"] for row in sub],
            [row["left_entropy"] for row in sub],
            marker="o",
            linewidth=1.15,
            color=colors[length],
            label=rf"$L={int(length)}$",
        )
    ax.set_xlabel(r"physical skin $g$")
    ax.set_ylabel(r"$S_A(W_+)$")
    ax.set_title(r"(a) isospectral but different entropy")
    ax.text(
        0.03,
        0.05,
        rf"$\max|\Delta \mathrm{{Tr}}H^k|={metric(summary, 'isospectral_h_trace_moment_max_abs_diff'):.1e}$"
        "\n"
        rf"$S: {metric(summary, 'isospectral_entropy_g0'):.2f}\to {metric(summary, 'isospectral_entropy_gmax'):.2f}$ at $L=128$",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    max_l = max(lengths)
    sub = sorted([row for row in rows if row["length"] == max_l], key=lambda row: row["skin"])
    ax.plot([row["skin"] for row in sub], [row["h_trace_moment_max_abs_diff"] for row in sub], marker="o", label=r"$\max_k|\Delta{\rm Tr}H^k|$")
    ax.plot([row["skin"] for row in sub], [abs(row["gbz_imag_width"]) for row in sub], marker="s", label=r"GBZ width")
    ax.set_yscale("log")
    ax.set_xlabel(r"physical skin $g$")
    ax.set_ylabel(r"spectral invariant drift")
    ax.set_title(r"(b) eigenvalue data fixed")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    controlled_unique = collapse_duplicate_controlled_rows(controlled)
    ax = axes[1, 0]
    x = np.asarray([row["target_overlap"] for row in controlled_unique])
    y = np.asarray([row["top_overlap_min_sv"] for row in controlled_unique])
    ax.scatter(x, y, s=16, alpha=0.75)
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8)
    ax.set_xlabel(r"designed overlap")
    ax.set_ylabel(r"$s_{\min}(V_+^\dagger Q_0)$")
    ax.set_title(r"(c) overlap control")
    ax.text(0.05, 0.88, rf"$r={metric(summary, 'controlled_overlap_corr'):.3f}$", transform=ax.transAxes, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    x = np.asarray([row["restricted_trace_left"] for row in controlled_unique])
    y = np.asarray([row["entropy_diff_left"] for row in controlled_unique])
    c = np.asarray([row["target_overlap"] for row in controlled_unique])
    sc = ax.scatter(x, y, c=c, s=17, alpha=0.78, cmap="plasma")
    ax.set_xlabel(r"$\|C_A(T)-C_A(W_+)\|_1$")
    ax.set_ylabel(r"$|S_A(Q_T)-S_A(W_+)|$")
    ax.set_title(r"(d) entropy-continuity step")
    ax.text(
        0.05,
        0.84,
        rf"$r={metric(summary, 'controlled_entropy_trace_corr'):.3f}$"
        "\n"
        rf"Aud. max viol. ${metric(summary, 'controlled_audenaert_violation_max'):.1e}$",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, fraction=0.047, pad=0.03, label=r"overlap")

    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_collapse_scaling(joint: list[dict], lt: list[dict], lt_summary: list[dict], summary: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    groups = sorted({(row["boundary"], row["skin"]) for row in joint})
    colors = dict(zip(groups, plt.cm.tab10(np.linspace(0, 1, len(groups)))))
    markers = {"periodic": "o", "open": "s"}

    ax = axes[0, 0]
    for group in groups:
        sub = [row for row in joint if (row["boundary"], row["skin"]) == group]
        ax.scatter(
            [row["left_subspace_entropy_density"] for row in sub],
            [row["S_density"] for row in sub],
            s=21,
            marker=markers[group[0]],
            color=colors[group],
            alpha=0.78,
            label=rf"{group[0]}, $g={group[1]:g}$",
        )
    ax.set_xlabel(r"$S_A(W_+)/L$")
    ax.set_ylabel(r"$S_A(Q_T)/L$")
    ax.set_title(r"(a) singular-subspace collapse")
    ax.text(0.05, 0.88, rf"$r={metric(summary, 'collapse_corr_left_density'):.3f}$, $n={int(metric(summary, 'joint_points'))}$", transform=ax.transAxes, fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6, ncol=2)

    ax = axes[0, 1]
    for group in groups:
        sub = [row for row in joint if (row["boundary"], row["skin"]) == group]
        ax.scatter(
            [row["gbz_imag_width"] for row in sub],
            [row["S_density"] for row in sub],
            s=21,
            marker=markers[group[0]],
            color=colors[group],
            alpha=0.78,
        )
    ax.set_xlabel(r"GBZ imaginary width")
    ax.set_ylabel(r"$S_A(Q_T)/L$")
    ax.set_title(r"(b) eigenvalue diagnostic fails")
    ax.text(0.05, 0.88, rf"$r={metric(summary, 'collapse_corr_gbz_width'):.3f}$", transform=ax.transAxes, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    times = sorted({row["steps"] for row in lt})
    time_colors = dict(zip(times, plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))))
    focus_times = [time for time in times if time in {100.0, 300.0, 600.0}]
    for skin, ls in [(0.0, "-"), (0.25, "--")]:
        for steps in focus_times:
            sub = sorted(
                [
                    row
                    for row in lt
                    if row["boundary"] == "open"
                    and abs(row["skin"] - skin) < 1e-12
                    and abs(row["gamma"] - 0.5) < 1e-12
                    and row["steps"] == steps
                ],
                key=lambda row: row["length"],
            )
            ax.plot(
                [row["length"] for row in sub],
                [row["left_entropy"] for row in sub],
                marker="o" if skin == 0.0 else "s",
                markersize=3,
                linewidth=1.0,
                linestyle=ls,
                color=time_colors[steps],
                label=rf"$g={skin:g},T={int(steps)}$" if steps in {100.0, 600.0} else None,
            )
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$S_A(W_+)$")
    ax.set_title(r"(c) $L,T$ scaling, $\gamma=0.5$")
    ax.text(
        0.05,
        0.07,
        rf"skin: $a(T)\in[{metric(summary, 'lt_skin_gamma05_log_slope_min'):.2f},{metric(summary, 'lt_skin_gamma05_log_slope_max'):.2f}]$"
        "\n"
        rf"recip.: $a(T)\in[{metric(summary, 'lt_recip_gamma05_log_slope_min'):.2f},{metric(summary, 'lt_recip_gamma05_log_slope_max'):.2f}]$",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")

    ax = axes[1, 1]
    for skin, marker in [(0.0, "o"), (0.25, "s")]:
        sub = sorted(
            [row for row in lt_summary if row["boundary"] == "open" and abs(row["skin"] - skin) < 1e-12],
            key=lambda row: row["gamma"],
        )
        ax.errorbar(
            [row["gamma"] for row in sub],
            [row["log_slope_mean"] for row in sub],
            yerr=[row["log_slope_std"] for row in sub],
            marker=marker,
            linewidth=1.2,
            capsize=2.5,
            label=rf"$g={skin:g}$",
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$a(T)$ in $S_A(W_+)\sim a(T)\log L+b$")
    ax.set_title(r"(d) skin keeps $a(T)\simeq0$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
