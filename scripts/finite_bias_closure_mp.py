"""Independent arbitrary-precision prediction and complete-model validation.

The predictor evaluates the exterior kernel, without an inverse Gram. The
validation instead inverts both reflected finite Grams and retains every mode.
FLINT accelerates dense linear algebra. Decimal input accuracy and asymptotic
kernel replacement error are not certified by the FLINT arithmetic intervals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from flint import arb, arb_mat, ctx
import mpmath as mp
from scipy.optimize import brentq

import finite_bias_closure_functions_mp as exterior

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prl_finite_bias_closure_20261013/fixed_bias"


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decimal(value, digits=None):
    if isinstance(value, arb):
        value = mp.mpf(value.mid().str(mp.mp.dps + 8, radius=False))
    return mp.nstr(value, digits or max(40, mp.mp.dps - 12))


def midpoint(value):
    return mp.mpf(value.mid().str(mp.mp.dps + 8, radius=False))


def ball(value):
    return arb(mp.nstr(value, mp.mp.dps + 8))


def setup(precision):
    mp.mp.dps = precision
    ctx.dps = precision + 12
    ctx.threads = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError("Refusing to overwrite existing scientific output: " + str(path))
    path.write_text(text, encoding="utf-8")


def event(name, **data):
    print(json.dumps(dict(event=name, **data)), flush=True)


def symmetrize(matrix):
    return (matrix + matrix.transpose()) * arb(".5")


def max_entry(matrix):
    return max((abs(midpoint(x)) for x in matrix.entries()), default=mp.mpf(0))


def native_solve(matrix, rhs):
    # Approximate LU is much faster at large precision; its residual is audited.
    answer = matrix.solve(rhs, algorithm="approx")
    residual = max_entry(matrix * answer - rhs) / max(mp.mpf(1), max_entry(rhs))
    return answer, residual


def finite_modes(length, precision):
    """Finite roots with stable sine recurrence and reflected edge vectors."""
    setup(precision)
    if length < 8 or length % 4:
        raise ValueError("Require L divisible by four, L>=8")
    n, p = length // 2, exterior.parameters()
    u, v = arb_mat(n, n), arb_mat(n, n)
    singular, phases = [], []
    for j in range(1, n):
        seed = brentq(lambda x: math.sin((n + 1) * x) + 2 * math.sin(n * x),
                      j * math.pi / (n + 1), j * math.pi / n, xtol=5e-15)
        k = mp.findroot(lambda x: mp.sin((n + 1) * x) + 2 * mp.sin(n * x),
                        mp.mpf(str(seed)),
                        df=lambda x: (n + 1) * mp.cos((n + 1) * x) + 2 * n * mp.cos(n * x),
                        solver="newton", maxsteps=30)
        cosine, sine = mp.cos(k), mp.sin(k)
        delta_prime = (1 + 2 * cosine) / (5 + 4 * cosine)
        scale = mp.sqrt(2 / (n + delta_prime))
        before, current = mp.mpf(0), sine
        vector = []
        for _ in range(n):
            vector.append(scale * current)
            before, current = current, 2 * cosine * current - before
        phase = mp.mpf((-1) ** (j - 1))
        for cell in range(n):
            v[cell, j - 1] = ball(vector[cell])
            u[cell, j - 1] = ball(phase * vector[n - 1 - cell])
        singular.append(mp.sqrt(p["a0"] + 2 * p["b"] * cosine))
        phases.append(phase)
    q = mp.findroot(lambda x: 1 - 2 * x + 2 * x ** (2 * n + 1) - x ** (2 * n + 2),
                    mp.mpf(".5"), solver="newton", maxsteps=30)
    vector = [(-1) ** cell * q ** (n - 1 - cell) * (1 - q ** (2 * cell + 2)) for cell in range(n)]
    scale = mp.sqrt(mp.fsum(x * x for x in vector))
    phase = mp.mpf((-1) ** (n - 1))
    for cell in range(n):
        v[cell, n - 1] = ball(vector[cell] / scale)
        u[cell, n - 1] = ball(phase * vector[n - 1 - cell] / scale)
    singular.append(p["t1"] * (1 - q * q) * q ** (n - 1) / (1 - q ** (2 * n)))
    phases.append(phase)
    mu = [mp.sqrt(p["gamma"] ** 2 - z * z) for z in singular]
    ratios = [z / (p["gamma"] + m) for z, m in zip(singular, mu)]
    aa = [1 / mp.sqrt(1 + r * r) for r in ratios]
    plus, minus = arb_mat(length, n), arb_mat(length, n)
    for cell in range(n):
        for k in range(n):
            a, r = ball(aa[k]), ball(ratios[k])
            plus[2 * cell, k], plus[2 * cell + 1, k] = a * u[cell, k], -a * r * v[cell, k]
            minus[2 * cell, k], minus[2 * cell + 1, k] = a * r * u[cell, k], -a * v[cell, k]
    event("finite_modes", length=length, dps=precision)
    return dict(length=length, n=n, p=p, u=u, v=v, singular=singular,
                phases=phases, mu=mu, ratios=ratios, aa=aa, plus=plus, minus=minus)


def omega(matrix):
    length, columns = matrix.nrows(), matrix.ncols()
    result = arb_mat(length, columns)
    for cell in range(length // 2):
        reflected = length // 2 - 1 - cell
        for k in range(columns):
            result[2 * cell, k] = matrix[2 * reflected + 1, k]
            result[2 * cell + 1, k] = -matrix[2 * reflected, k]
    return result


def row_weight(matrix, weights):
    return arb_mat(matrix.nrows(), matrix.ncols(),
                   [matrix[i, j] * ball(weights[i]) for i in range(matrix.nrows()) for j in range(matrix.ncols())])


def encoded(matrix, digits):
    return [[decimal(matrix[i, j], digits) for j in range(matrix.ncols())] for i in range(matrix.nrows())]


def decoded(rows):
    return arb_mat([[arb(x) for x in row] for row in rows])


def prepare_exact(length, precision, export_digits, output):
    tick = time.perf_counter()
    modes = finite_modes(length, precision)
    n, m = length // 2, length // 4
    u, v, ratio = modes["u"], modes["v"], modes["ratios"]
    states = {}
    for side in ("left", "right"):
        occupied = list(range(m)) if side == "left" else list(range(m, n))
        empty = list(range(m, n)) if side == "left" else list(range(m))
        ucross = arb_mat(m, n, [u[j, k] * ball(ratio[k]) for j in empty for k in range(n)])
        vcross = arb_mat(m, n, [v[j, k] for j in occupied for k in range(n)])
        cross = ucross * vcross.transpose()
        gram = arb_mat(m, m, [cross[i, m - 1 - j] if side == "left" else cross[m - 1 - i, j]
                              for i in range(m) for j in range(m)])
        weights = arb_mat(m, n, [v[m - 1 - i, k] if side == "left" else v[m + i, k]
                                 for i in range(m) for k in range(n)])
        solve, residual = native_solve(gram, weights)
        if residual > mp.mpf("1e-50"):
            raise ArithmeticError("Reflected Gram solve residual exceeds the frozen preparation gate")
        kernel = weights.transpose() * solve
        f0 = arb_mat(n, n)
        aa, phases = modes["aa"], modes["phases"]
        for i in range(n):
            for j in range(n):
                f0[i, j] = (-kernel[i, j] * ball(aa[j] / aa[i] * (1 - ratio[j] ** 2) * phases[j])
                            - (ball(ratio[j]) if i == j else 0))
        states[side] = dict(F=encoded(f0, export_digits), inverse_action_relative_residual=decimal(residual),
                            gram_symmetry_relative=decimal(max_entry(gram - gram.transpose()) / max_entry(gram)))
        event("exact_reflected", length=length, side=side, seconds=time.perf_counter() - tick)
    value = dict(length=length, dps=precision, export_digits=export_digits,
                 mu=[decimal(x, export_digits) for x in modes["mu"]],
                 plus=encoded(modes["plus"], export_digits), minus=encoded(modes["minus"], export_digits),
                 states=states, source_sha256=checksum(__file__), all_finite_modes_retained=True,
                 method="stable high-precision finite recurrence; two exact finite reflected Gram solves with FLINT native LU",
                 interval_certified=False, seconds=time.perf_counter() - tick,
                 finished_utc=datetime.now(timezone.utc).isoformat())
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
    save(path.with_suffix(".receipt.json"), dict(length=length, dps=precision, export_digits=export_digits,
         sha256=checksum(path), bytes=path.stat().st_size, seconds=value["seconds"], source_sha256=checksum(__file__)))
    return dict(status="prepared_exact_complete", path=str(path), seconds=value["seconds"])


class KernelPredictor:
    def __init__(self, modes, side, skin, nodes, rank):
        self.modes, self.side = modes, side
        self.length, self.n = modes["length"], modes["n"]
        self.skin, self.rank, self.p = mp.mpf(skin), rank, modes["p"]
        self.degree, self.mu = self.length // 4 - 1, modes["mu"]
        self.positions = [mp.mpf(i // 2) - mp.mpf(self.n - 1) / 2 for i in range(self.length)]
        self.weights = [mp.exp(self.skin * x) for x in self.positions]
        q = modes["plus"]
        self.weighted_q = row_weight(q, self.weights)
        self.metric = symmetrize(self.weighted_q.transpose() * self.weighted_q)
        cross = q.transpose() * row_weight(omega(q), [x * x for x in self.weights])
        self.cchi, residual = native_solve(self.metric, cross)
        self.cchi = symmetrize(self.cchi)
        self.preparation_residual = residual
        if residual > mp.mpf("1e-30"):
            raise ArithmeticError("High-precision weighted metric solve did not pass")
        if "kernel_tables" not in modes:
            modes["kernel_tables"] = {}
        if nodes not in modes["kernel_tables"]:
            table = exterior.tables(self.p, nodes)
            modes["kernel_tables"][nodes] = table, exterior.core_parameters(table)
        table, self.core = modes["kernel_tables"][nodes]
        key = (side, nodes)
        if "kernel_values" not in modes:
            modes["kernel_values"] = {}
        if key not in modes["kernel_values"]:
            potential_values, rho_values, amplitudes = [], [], []
            for k in range(self.n):
                z = (modes["singular"][k] ** 2 - self.p["a0"]) / self.p["b"]
                potential, prefactor, rho = exterior.functions(z, side, table)
                boundary = (midpoint(modes["v"][0, k]) * modes["phases"][k]
                            if side == "left" else midpoint(modes["u"][0, k]))
                potential_values.append(potential)
                rho_values.append(rho)
                amplitudes.append(self.mu[k] * boundary * mp.sqrt(prefactor) / (self.p["gamma"] * modes["aa"][k]))
            modes["kernel_values"][key] = potential_values, rho_values, amplitudes
        self.potential, self.rho, self.amplitude = modes["kernel_values"][key]
        self.zz = [modes["phases"][k] * modes["singular"][k] / self.mu[k] for k in range(self.n)]
        event("kernel_prepared", length=self.length, side=side, skin=str(skin), nodes=nodes, rank=rank)

    def evaluate(self, duration):
        vector = [self.amplitude[k] * mp.exp(self.degree * self.potential[k] - self.mu[k] * duration) for k in range(self.n)]
        ff = arb_mat(self.n, self.rank, [ball(vector[k] * mp.sqrt(1 - self.rho[k] ** 2) * self.rho[k] ** j)
                    for k in range(self.n) for j in range(self.rank)])
        inverse, residual = native_solve(self.metric, ff)
        ss = symmetrize(ff.transpose() * inverse)
        zzff = arb_mat(self.n, self.rank, [ff[k, j] * ball(self.zz[k]) for k in range(self.n) for j in range(self.rank)])
        true = arb_mat(self.rank, self.rank)
        for k in range(self.rank):
            true[k, k] = 1
        true = symmetrize(true - ff.transpose() * zzff)
        denominator = symmetrize(true - ff.transpose() * self.cchi * ff)
        eigenmatrix, eig_residual = native_solve(denominator, ss)
        values = eigenmatrix.eig(algorithm="approx")
        real = sorted(midpoint(x.real) for x in values)
        imaginary = max(abs(midpoint(x.imag)) for x in values)
        tangent = real[-1]
        if tangent <= 0 or imaginary > mp.mpf("1e-25"):
            raise ArithmeticError("Invalid leading generalized eigenvalue")
        distance = tangent / mp.sqrt(1 + tangent * tangent)
        linear_trace = mp.fsum(x * x for x in vector)
        raw_tail = max(abs(x) for x in self.rho) ** (2 * self.rank) * linear_trace
        # This is an explicit bound for the linear kernel only, before feedback.
        gauge_tail = mp.exp(abs(self.skin) * (self.n - 1)) * raw_tail
        return dict(distance=distance, tangent=tangent, second_tangent=real[-2],
                    eigen_imaginary_maximum=imaginary, metric_solve_residual=residual,
                    eig_solve_residual=eig_residual, linear_trace=linear_trace,
                    raw_geometric_tail=raw_tail, gauged_linear_tail=gauge_tail,
                    denominator_symmetry=max_entry(denominator - denominator.transpose()),
                    omitted_diagonal_raw=mp.exp(-2 * self.p["mu0"] * duration))

    def root(self, level):
        epsilon = mp.mpf(level)
        s = 1 if self.side == "left" else -1
        h = s * self.skin
        if h > 0:
            z = 2 * mp.cosh(h)
            mu = mp.sqrt(self.p["gamma"] ** 2 - self.p["a0"] - self.p["b"] * z)
            potential = exterior.functions(z, self.side, self.modes["kernel_tables"][max(self.modes["kernel_tables"])][0])[0]
            tau, logcoef = (potential - h) / (4 * mu), -1 / (4 * mu)
        else:
            mu = self.p["mu0"]
            tau = (self.core["G2"] - h) / (4 * mu)
            logcoef = -3 / (2 * mu) if h < 0 else -3 / (4 * mu)
        duration = tau * self.length + logcoef * mp.log(self.length)
        samples, previous = [], None
        for _ in range(24):
            row = self.evaluate(duration)
            residual = mp.log(row["tangent"] / (epsilon / mp.sqrt(1 - epsilon * epsilon)))
            samples.append(dict(time=decimal(duration), distance=decimal(row["distance"])))
            if abs(residual) < mp.mpf("1e-28"):
                slope_step = mp.mpf(".0002")
                past, future = self.evaluate(duration - slope_step), self.evaluate(duration + slope_step)
                rate = mp.log(past["tangent"] / future["tangent"]) / (2 * slope_step)
                return dict(length=self.length, initial=self.side, skin=decimal(self.skin), level=decimal(epsilon),
                            time=decimal(duration), tau=decimal(tau), log_coefficient=decimal(logcoef),
                            center_residual=decimal(duration - tau * self.length - logcoef * mp.log(self.length)),
                            temporal_rate=decimal(rate), effective_asymptotic_rate=decimal(2 * mu),
                            root_residual=decimal(abs(row["distance"] - epsilon)),
                            **{key: decimal(value) for key, value in row.items()}, samples=samples)
            rate = 2 * mu
            if previous is not None and duration != previous[0]:
                secant_rate = -(residual - previous[1]) / (duration - previous[0])
                if secant_rate > 0:
                    rate = secant_rate
            previous = duration, residual
            duration += max(mp.mpf(-3), min(mp.mpf(3), residual / rate))
        raise ArithmeticError("Kernel root did not converge")


class ExactDynamics:
    def __init__(self, raw, side, skin):
        self.length, self.n = raw["length"], raw["length"] // 2
        self.mu, self.f0 = [mp.mpf(x) for x in raw["mu"]], decoded(raw["states"][side]["F"])
        weights = [mp.exp(mp.mpf(skin) * (mp.mpf(i // 2) - mp.mpf(self.n - 1) / 2)) for i in range(self.length)]
        plus, minus = decoded(raw["plus"]), decoded(raw["minus"])
        self.gain, self.loss = row_weight(plus, weights), row_weight(minus, weights)
        self.metric = symmetrize(self.gain.transpose() * self.gain)
        transverse = omega(self.gain)
        self.cross = self.gain.transpose() * self.loss
        self.bottom = transverse.transpose() * self.loss
        self.cross_coordinates, self.cross_solve_residual = native_solve(self.metric, self.cross)
        self.bottom_coordinates, self.bottom_solve_residual = native_solve(self.metric, self.bottom)
        self.identity = arb_mat(self.n, self.n)
        for k in range(self.n):
            self.identity[k, k] = 1
        self.symplectic_error = max_entry(self.gain.transpose() * transverse)
        if max(self.symplectic_error, self.cross_solve_residual, self.bottom_solve_residual) > mp.mpf("1e-30"):
            raise ArithmeticError("The growing frame or metric solve failed the frozen precision gate")

    def evaluate(self, duration, norm_gate=mp.mpf("1e-28")):
        ft = arb_mat(self.n, self.n, [self.f0[i, j] * ball(mp.exp(-duration * (self.mu[i] + self.mu[j])))
                      for i in range(self.n) for j in range(self.n)])
        upper_coordinates = self.identity + self.cross_coordinates * ft
        lower_coordinates = self.bottom_coordinates * ft
        graph_t, solve_residual = native_solve(upper_coordinates.transpose(), lower_coordinates.transpose())
        graph = graph_t.transpose()
        symmetric = self.metric * graph
        selfadjoint_error = max_entry(symmetric - symmetric.transpose()) / max(mp.mpf(1), max_entry(symmetric))
        if max(solve_residual, selfadjoint_error) > mp.mpf("1e-30"):
            raise ArithmeticError("The exact graph solve or physical self-adjointness failed the frozen gate")
        action = graph * graph
        trace = mp.fsum(midpoint(action[i, i]) for i in range(self.n))
        vector = arb_mat(self.n, 1, [1 + (i % 7) / 13 for i in range(self.n)])
        evidence = None
        for iteration in range(16):
            vector = action * vector
            norm = midpoint((vector.transpose() * self.metric * vector)[0, 0])
            if norm <= 0:
                raise ArithmeticError("The physical metric norm is not positive")
            vector = vector * ball(1 / mp.sqrt(norm))
            image = action * vector
            rayleigh = midpoint((vector.transpose() * self.metric * image)[0, 0])
            error_vector = image - vector * ball(rayleigh)
            residual_squared = max(mp.mpf(0), midpoint((error_vector.transpose() * self.metric * error_vector)[0, 0]))
            remainder = max(mp.mpf(0), trace - rayleigh)
            bound = (rayleigh + remainder + mp.sqrt((rayleigh - remainder) ** 2 + 4 * residual_squared)) / 2
            low, high = mp.sqrt(max(mp.mpf(0), rayleigh)), mp.sqrt(max(rayleigh, bound))
            evidence = dict(tangent_lower=low, tangent_upper=high, relative_norm_gap=(high - low) / low,
                            ritz_residual_squared=residual_squared, full_psd_remainder_trace=remainder,
                            metric_solve_residual=max(self.cross_solve_residual, self.bottom_solve_residual), graph_solve_residual=solve_residual,
                            graph_metric_selfadjoint_residual=selfadjoint_error, symplectic_error=self.symplectic_error,
                            ritz_iterations=iteration + 1)
            if (high - low) / low < norm_gate:
                tangent = (low + high) / 2
                return tangent / mp.sqrt(1 + tangent * tangent), evidence
        raise ArithmeticError("Complete-model norm bound unresolved: " + decimal(evidence["relative_norm_gap"]))

    def root(self, seed, level):
        epsilon, duration = mp.mpf(level), mp.mpf(seed)
        previous, samples = None, []
        for _ in range(9):
            distance, evidence = self.evaluate(duration)
            samples.append(dict(time=decimal(duration), distance=decimal(distance),
                                **{key: decimal(value) if not isinstance(value, int) else value for key, value in evidence.items()}))
            if abs(distance - epsilon) < mp.mpf("1e-25"):
                return dict(time=decimal(duration), distance=decimal(distance), root_residual=decimal(abs(distance - epsilon)),
                            complete_norm_relative_gap=decimal(evidence["relative_norm_gap"]), samples=samples)
            slope = -mp.sqrt(7) * distance * (1 - distance * distance)
            if previous is not None and duration != previous[0]:
                secant = (distance - previous[1]) / (duration - previous[0])
                if secant < 0:
                    slope = secant
            previous = duration, distance
            duration -= max(mp.mpf(-2), min(mp.mpf(2), (distance - epsilon) / slope))
        raise ArithmeticError("Complete-model root did not converge")


def predict(length, precision, nodes, rank, skins, levels, output):
    tick = time.perf_counter()
    modes = finite_modes(length, precision)
    rows = []
    for side in ("left", "right"):
        for skin in skins:
            graph = KernelPredictor(modes, side, skin, nodes, rank)
            for level in levels:
                rows.append(graph.root(level))
                event("kernel_root", length=length, side=side, skin=skin, level=level, time=rows[-1]["time"])
    value = dict(length=length, working_dps=precision, nodes=nodes, kernel_rank=rank, rows=rows,
                 method="all finite modes, arbitrary-precision exterior kernel, exact low-rank graph denominator",
                 source_sha256=checksum(__file__), functions_sha256=checksum(exterior.__file__),
                 no_fit=True, inverse_gram_used=False, interval_certified=False,
                 seconds=time.perf_counter() - tick, finished_utc=datetime.now(timezone.utc).isoformat())
    save(output, value)
    return dict(status="kernel_predictions_saved", roots=len(rows), seconds=value["seconds"])


def validate(raw_path, prediction_path, precision, output, requested_side=None, requested_skin=None):
    tick = time.perf_counter()
    setup(precision)
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    predictions = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    rows = []
    for side in ("left", "right"):
        if requested_side and requested_side != side:
            continue
        skins = sorted(set(row["skin"] for row in predictions["rows"] if row["initial"] == side))
        for skin in skins:
            if requested_skin is not None and mp.mpf(requested_skin) != mp.mpf(skin):
                continue
            graph = ExactDynamics(raw, side, skin)
            for predicted in predictions["rows"]:
                if predicted["initial"] != side or predicted["skin"] != skin:
                    continue
                result = graph.root(predicted["time"], predicted["level"])
                difference = mp.mpf(result["time"]) - mp.mpf(predicted["time"])
                rows.append(dict(length=raw["length"], initial=side, skin=skin, level=predicted["level"],
                                 predicted_time=predicted["time"], actual_minus_prediction=decimal(difference), **result))
                event("exact_root", length=raw["length"], side=side, skin=skin,
                      level=predicted["level"], error=decimal(difference))
    value = dict(rows=rows, working_dps=precision, static_dps=raw["dps"], export_digits=raw["export_digits"],
                 input_sha256=checksum(raw_path), prediction_sha256=checksum(prediction_path), source_sha256=checksum(__file__),
                 all_finite_modes_retained=True, inverse_gram_used=True, interval_certified=False,
                 seconds=time.perf_counter() - tick, finished_utc=datetime.now(timezone.utc).isoformat())
    save(output, value)
    return dict(status="exact_complete_model_validated", roots=len(rows),
                maximum_time_error=max(abs(float(row["actual_minus_prediction"])) for row in rows), seconds=value["seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "prepare-exact", "validate"))
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--precision", type=int, default=140)
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--rank", type=int, default=40)
    parser.add_argument("--export-digits", type=int, default=210)
    parser.add_argument("--skins", default="-.05,-.025,.025,.05")
    parser.add_argument("--levels", default=".05,.01")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--side", choices=("left", "right"))
    parser.add_argument("--skin")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "predict":
        value = predict(args.length, args.precision, args.nodes, args.rank,
                        args.skins.split(","), args.levels.split(","), args.output)
    elif args.action == "prepare-exact":
        value = prepare_exact(args.length, args.precision, args.export_digits, args.output)
    else:
        value = validate(args.raw, args.prediction, args.precision, args.output, args.side, args.skin)
    print(json.dumps(value, indent=2), flush=True)


if __name__ == "__main__":
    main()
