from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mietf_skin.bounds import subspace_bound_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evolved SSH subspace to top singular subspaces.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--boundaries", default="periodic,open")
    parser.add_argument("--skins", default="0,0.25")
    parser.add_argument("--gammas", default="0.3,0.5,0.7,1.0")
    parser.add_argument("--cells", default="32,64")
    parser.add_argument("--steps-list", default="100,300")
    parser.add_argument("--initial-patterns", default="charge_density_wave,left")
    parser.add_argument("--random-seeds", default="0,1,2,3,4")
    parser.add_argument("--t1", type=float, default=0.5)
    parser.add_argument("--t2", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        boundaries = ["periodic", "open"]
        skins = [0.0, 0.25]
        gammas = [0.3, 0.5, 0.7, 1.0]
        cells_list = [32, 64]
        steps_list = [100, 300]
        initial_patterns = ["charge_density_wave", "left"]
        random_seeds = [0, 1, 2, 3, 4]
        tag = args.tag or "quick"
    else:
        boundaries = parse_str_list(args.boundaries)
        skins = parse_float_list(args.skins)
        gammas = parse_float_list(args.gammas)
        cells_list = parse_int_list(args.cells)
        steps_list = parse_int_list(args.steps_list)
        initial_patterns = parse_str_list(args.initial_patterns)
        random_seeds = parse_int_list(args.random_seeds)
        tag = args.tag or "custom"

    rows = []
    pattern_seed_pairs = expand_pattern_seed_pairs(initial_patterns, random_seeds)
    total = len(boundaries) * len(skins) * len(gammas) * len(cells_list) * len(steps_list) * len(pattern_seed_pairs)
    job = 0
    for boundary in boundaries:
        for skin in skins:
            for gamma in gammas:
                for cells in cells_list:
                    for steps in steps_list:
                        for initial_pattern, initial_seed in pattern_seed_pairs:
                            result = subspace_bound_diagnostics(
                                cells=cells,
                                t1=args.t1,
                                t2=args.t2,
                                gain_loss=gamma,
                                skin=skin,
                                boundary=boundary,
                                dt=args.dt,
                                steps=steps,
                                initial_pattern=initial_pattern,
                                initial_seed=initial_seed,
                            )
                            row = result.__dict__.copy()
                            row["actual_entropy_density"] = result.actual_entropy / result.length
                            row["left_entropy_density"] = result.left_entropy / result.length
                            row["right_entropy_density"] = result.right_entropy / result.length
                            rows.append(row)
                            job += 1
                            print(
                                f"[{job:4d}/{total}] {boundary:8s} skin={skin:.2f} "
                                f"gamma={gamma:.2f} L={result.length:3d} T={steps:3d} "
                                f"init={initial_pattern:20s} seed={result.initial_seed:3d} "
                                f"dS_left={result.entropy_diff_left:.3f} dS_right={result.entropy_diff_right:.3f} "
                                f"D_left={result.projector_trace_left_per_particle:.3f}"
                            )

    out = ROOT / "data" / f"{tag}_subspace_bound_scan.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def expand_pattern_seed_pairs(initial_patterns: list[str], random_seeds: list[int]) -> list[tuple[str, int | None]]:
    pairs: list[tuple[str, int | None]] = []
    for pattern in initial_patterns:
        if pattern == "random":
            pairs.extend(("random", seed) for seed in random_seeds)
        else:
            pairs.append((pattern, None))
    return pairs


if __name__ == "__main__":
    main()
