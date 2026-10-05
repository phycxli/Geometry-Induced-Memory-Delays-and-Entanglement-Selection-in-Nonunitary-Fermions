from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mietf_skin.diagnostics import participation_lengths, ssh_step_matrix
from mietf_skin.gaussian import half_chain_entropy
from run_subspace_bound_scan import parse_float_list, parse_int_list, parse_str_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L,T scaling scan of the top left singular-subspace entropy.")
    parser.add_argument("--tag", default="singular_lt_scaling_v1")
    parser.add_argument("--boundaries", default="open")
    parser.add_argument("--skins", default="0,0.25")
    parser.add_argument("--gammas", default="0.3,0.5,0.7")
    parser.add_argument("--cells", default="32,48,64,80,96")
    parser.add_argument("--time-steps", default="100,200,300,450,600")
    parser.add_argument("--t1", type=float, default=0.5)
    parser.add_argument("--t2", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    boundaries = parse_str_list(args.boundaries)
    skins = parse_float_list(args.skins)
    gammas = parse_float_list(args.gammas)
    cells_list = parse_int_list(args.cells)
    steps_list = parse_int_list(args.time_steps)

    rows = []
    total = len(boundaries) * len(skins) * len(gammas) * len(cells_list) * len(steps_list)
    job = 0
    for boundary in boundaries:
        for skin in skins:
            for gamma in gammas:
                for cells in cells_list:
                    step = ssh_step_matrix(
                        cells=cells,
                        t1=args.t1,
                        t2=args.t2,
                        gain_loss=gamma,
                        skin=skin,
                        boundary=boundary,
                        dt=args.dt,
                    )
                    for steps in steps_list:
                        row = singular_subspace_row(
                            step=step,
                            cells=cells,
                            t1=args.t1,
                            t2=args.t2,
                            gain_loss=gamma,
                            skin=skin,
                            boundary=boundary,
                            dt=args.dt,
                            steps=steps,
                        )
                        rows.append(row)
                        job += 1
                        print(
                            f"[{job:3d}/{total}] {boundary:8s} skin={skin:.2f} "
                            f"gamma={gamma:.2f} L={row['length']:3d} T={steps:3d} "
                            f"S_W/L={row['left_entropy_density']:.4f} "
                            f"S_W/logL={row['left_entropy_over_log_l']:.3f}"
                        )

    out = ROOT / "data" / f"{args.tag}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


def normalized_matrix_power(matrix: np.ndarray, power: int) -> np.ndarray:
    """Return matrix**power up to an irrelevant scalar normalization."""
    if power < 1:
        raise ValueError("power must be positive")
    result = np.eye(matrix.shape[0], dtype=np.complex128)
    base = np.array(matrix, dtype=np.complex128, copy=True)
    n = power
    while n:
        if n & 1:
            result = normalize(result @ base)
        n >>= 1
        if n:
            base = normalize(base @ base)
    return result


def normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix)
    if norm == 0.0:
        raise FloatingPointError("matrix power underflowed to zero")
    return matrix / norm


def singular_subspace_row(
    step: np.ndarray,
    cells: int,
    t1: float,
    t2: float,
    gain_loss: float,
    skin: float,
    boundary: str,
    dt: float,
    steps: int,
) -> dict:
    length = 2 * cells
    particles = length // 2
    product = normalized_matrix_power(step, steps)
    left, singular_values, vh = np.linalg.svd(product, full_matrices=True)
    singular_values = np.maximum(singular_values, 1e-300)
    right = vh.conj().T
    left_top = left[:, :particles]
    right_top = right[:, :particles]
    left_entropy = half_chain_entropy(left_top)
    right_entropy = half_chain_entropy(right_top)
    left_pr = participation_lengths(left)
    right_pr = participation_lengths(right)
    occ = slice(0, particles)
    return {
        "boundary": boundary,
        "skin": skin,
        "gamma": gain_loss,
        "cells": cells,
        "length": length,
        "t1": t1,
        "t2": t2,
        "dt": dt,
        "steps": steps,
        "physical_time": dt * steps,
        "left_entropy": left_entropy,
        "left_entropy_density": left_entropy / length,
        "left_entropy_over_log_l": left_entropy / np.log(length),
        "right_entropy": right_entropy,
        "right_entropy_density": right_entropy / length,
        "right_entropy_over_log_l": right_entropy / np.log(length),
        "singular_log_gap": float(np.log(singular_values[particles - 1]) - np.log(singular_values[particles])),
        "singular_ratio": float(singular_values[particles] / singular_values[particles - 1]),
        "left_pr_occ_mean": float(np.mean(left_pr[occ])),
        "left_pr_occ_over_l": float(np.mean(left_pr[occ]) / length),
        "right_pr_occ_mean": float(np.mean(right_pr[occ])),
        "right_pr_occ_over_l": float(np.mean(right_pr[occ]) / length),
    }


if __name__ == "__main__":
    main()
