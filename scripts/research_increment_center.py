"""Reflected-Gram absolute exponents and external-evaluation diagnostics.

The rate command evaluates the analytic quantities without fitting finite data.
The finite command retains the exact reflection terms at zero bias. It is an
arbitrary-precision numerical audit, not an interval certificate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mpmath as mp


def constants(parameters: dict | None = None) -> dict[str, mp.mpf]:
    parameters = parameters or {"gamma": "2", "t1": ".5", "t2": "1"}
    gamma, t1, t2 = (mp.mpf(str(parameters[key])) for key in ("gamma", "t1", "t2"))
    if t1 <= 0 or t2 <= t1 or gamma <= t1 + t2:
        raise ValueError("Require gamma > t1+t2 > 0 and t2 > t1 > 0.")
    a0, b = t1 * t1 + t2 * t2, t1 * t2
    a = (gamma * gamma - a0) / b
    return {"gamma": gamma, "t1": t1, "t2": t2, "a0": a0, "a": a, "b": b,
            "kappa": t2 / t1, "q": 2 / (a + mp.sqrt(a * a - 4)),
            "mu0": mp.sqrt(gamma * gamma - (t1 + t2) ** 2)}


def external_potential(z: mp.mpf, parameters: dict | None = None) -> mp.mpf:
    """Logarithmic potential of the explicit balayage measure, z < a."""
    a = constants(parameters)["a"]
    if z >= a:
        raise ValueError("The real exterior formula requires z < a.")
    root = mp.sqrt(a - z)
    integrand = lambda theta: mp.log(root + mp.sqrt(a - 2 * mp.cos(theta)))
    return 2 * mp.quad(integrand, [0, mp.pi / 2, mp.pi]) / mp.pi


def compact_equilibrium_density(u: mp.mpf, parameters: dict | None = None) -> mp.mpf:
    """Density on (0, 1/a); endpoint values are intentionally excluded."""
    a = constants(parameters)["a"]
    if not 0 < u < 1 / a:
        raise ValueError("The equilibrium density is defined on (0, 1/a).")
    integrand = lambda theta: mp.sqrt(a - 2 * mp.cos(theta)) / (1 - 2 * u * mp.cos(theta))
    inner = mp.quad(integrand, [0, mp.pi / 2, mp.pi]) / mp.pi
    return inner / (mp.pi * mp.sqrt(u) * mp.sqrt(1 - a * u))


def external_derivative(z: mp.mpf, parameters: dict | None = None) -> mp.mpf:
    a = constants(parameters)["a"]
    root = mp.sqrt(a - z)
    integrand = lambda theta: 1 / (root * (root + mp.sqrt(a - 2 * mp.cos(theta))))
    return -mp.quad(integrand, [0, mp.pi / 2, mp.pi]) / mp.pi


def strong_kernel_condition(parameters: dict | None = None) -> dict:
    c = constants(parameters)
    u_max = mp.mpf(25) / (16 * c["a"])
    analytic = u_max < min(mp.mpf(".5"), c["kappa"] / (1 + c["kappa"] ** 2))
    if u_max >= mp.mpf(".5"):
        return {"pass": False, "ellipse_u_max": str(u_max),
                "reason": "The radius-4 ellipse reaches a square-root branch point."}
    m0 = -mp.log((1 + mp.sqrt(1 - 4 * u_max ** 2)) / 2)
    minimum_q = mp.log((1 + mp.sqrt(1 - 4 / c["a"] ** 2)) / 2)
    truncation_exponent = 2 * m0 - minimum_q
    margin = mp.log(2) - truncation_exponent
    return {"pass": bool(analytic and margin > 0), "analytic_bottom_weight_pass": bool(analytic),
            "ellipse_u_max": str(u_max), "complex_Q_bound": str(m0),
            "weighted_truncation_exponent_bound": str(truncation_exponent),
            "exponential_error_margin": str(margin),
            "meaning": "Explicit sufficient condition for the elementary strong-kernel proof; failure does not disprove asymptotics."}


def kernel_prefactor(z: mp.mpf, side: str, parameters: dict | None = None) -> mp.mpf:
    if not z:
        raise ValueError("This prefactor interface excludes z=0.")
    c = constants(parameters)
    endpoint, capacity = 1 / c["a"], 1 / (4 * c["a"])
    exterior_u = 1 / z
    root = mp.sqrt(exterior_u * (exterior_u - endpoint))
    if exterior_u < 0:
        root = -root
    conformal = (exterior_u - endpoint / 2 + root) / (2 * capacity)

    def log_analytic_weight(theta: mp.mpf) -> mp.mpf:
        u = endpoint * (1 + mp.cos(theta)) / 2
        factor = 2 / (1 + mp.sqrt(1 - 4 * u * u))
        r = u * factor
        bottom = (mp.sqrt(c["a"] * c["kappa"]) * factor ** 2 * (1 - r * r)
                  / (mp.pi * (1 + c["kappa"] * r)))
        if side == "right":
            bottom *= c["t1"] ** 2 / (c["a0"] * u + c["b"])
        return mp.log(bottom)

    integrand = lambda theta: (log_analytic_weight(theta)
                               * (conformal ** 2 - 1)
                               / (conformal ** 2 - 2 * conformal * mp.cos(theta) + 1))
    log_outer = mp.quad(integrand, [0, mp.pi / 2, mp.pi]) / (2 * mp.pi)
    outer_square = mp.exp(2 * log_outer)
    inverse_conformal = 1 / conformal
    if side == "left":
        return 1 / (2 * mp.pi * capacity * outer_square
                    * (1 - inverse_conformal) ** 3 * (1 + inverse_conformal))
    if side == "right":
        return 1 / (2 * mp.pi * capacity ** 2 * outer_square
                    * (1 - inverse_conformal ** 2) ** 3)
    raise ValueError("side must be left or right.")


def prefactor_record(dps: int, parameters: dict | None = None, levels: list[str] | None = None) -> dict:
    with mp.workdps(dps):
        c = constants(parameters)
        levels = levels or [".3", ".05", ".01"]
        right_point = -(c["kappa"] + 1 / c["kappa"])
        g2 = external_potential(mp.mpf(2), parameters)
        gl = external_potential(mp.mpf(-2), parameters)
        gr = external_potential(right_point, parameters)
        tau = g2 / (4 * c["mu0"])
        derivative_2 = external_derivative(mp.mpf(2), parameters)
        derivative_left = external_derivative(mp.mpf(-2), parameters)
        gaussian_beta = tau * c["b"] / (2 * c["mu0"]) + derivative_2 / 4
        if gaussian_beta <= 0:
            raise ValueError("The boundary Gaussian coefficient is nonpositive; this interface requires the filtering regime.")
        coefficients = {side: {name: kernel_prefactor(z, side, parameters) for name, z in
                               (("2", mp.mpf(2)), ("minus_2", mp.mpf(-2)), ("right_point", right_point))}
                        for side in ("left", "right")}
        left_prefactor = (mp.sqrt(mp.pi) * (-2 * derivative_left) ** mp.mpf("1.5")
                          * mp.exp(2 * gl) / (4 * coefficients["left"]["minus_2"]))
        right_prefactor = (mp.exp(2 * gr) / ((1 - c["kappa"] ** -2)
                           * coefficients["right"]["right_point"]))
        mu_pi = mp.sqrt(c["gamma"] ** 2 - (c["t2"] - c["t1"]) ** 2)
        overlap_ratio_left = (c["gamma"] + mu_pi) / (2 * c["gamma"])
        band_boundary_factor = (2 * c["mu0"] ** 2 / (c["gamma"] * (c["gamma"] + c["mu0"]))
                                / (2 * mp.sqrt(mp.pi) * (2 * gaussian_beta) ** mp.mpf("1.5")))
        center_constants = []
        for level in levels:
            epsilon = mp.mpf(level)
            if not 0 < epsilon < c["mu0"] / c["gamma"]:
                raise ValueError("Each threshold must be between zero and mu0/gamma.")
            target = epsilon / mp.sqrt(1 - epsilon * epsilon)
            for side in ("left", "right"):
                boundary_multiplier = mp.mpf(1) if side == "left" else (c["t1"] / (c["t1"] + c["t2"])) ** 2
                amplitude = coefficients[side]["2"] * band_boundary_factor * boundary_multiplier
                center_constants.append({"initial": side, "epsilon": str(epsilon),
                                         "constant_time_center": str((mp.log(amplitude / target) - 2 * g2)
                                                                     / (2 * c["mu0"]))})
        return {"status": "analytic_prefactors_from_strong_kernel_proof",
                "decimal_digits": dps, "parameters": {key: str(c[key]) for key in ("gamma", "t1", "t2", "a", "kappa", "mu0")},
                "strong_kernel_sufficient_condition": strong_kernel_condition(parameters),
                "time_filter_condition_pass": bool(g2 > 2),
                "kernel_prefactors": {side: {key: str(value) for key, value in entries.items()}
                                      for side, entries in coefficients.items()},
                "gram_left_prefactor_L_power_3_over_2": str(left_prefactor),
                "gram_right_prefactor_L_power_0": str(right_prefactor),
                "input_overlap_left_to_gram_ratio_limit": str(overlap_ratio_left),
                "input_overlap_right_to_gram_ratio_limit": "1",
                "input_overlap_left_prefactor_L_power_3_over_2": str(left_prefactor * overlap_ratio_left),
                "input_overlap_right_prefactor_L_power_0": str(right_prefactor),
                "time_linear_coefficient": str(tau),
                "time_log_coefficient": str(-3 / (4 * c["mu0"])),
                "G_derivative_2": str(derivative_2), "G_derivative_minus_2": str(derivative_left),
                "gaussian_beta": str(gaussian_beta), "left_boundary_gaussian_factor": str(band_boundary_factor),
                "center_constants": center_constants,
                "scope": "Prefactor theorem requires the explicit strong-kernel sufficient condition. Time constants also require G_a(2)>2 and fixed low thresholds.",
                "evidence": "Analytic derivation and reproducible quadrature; no finite-size fit and no interval certification."}


def rate_record(dps: int, parameters: dict | None = None) -> dict:
    with mp.workdps(dps):
        c = constants(parameters)
        right_point = -(c["kappa"] + 1 / c["kappa"])
        values = {name: external_potential(z, parameters) for name, z in
                  (("G_0", mp.mpf(0)), ("G_2", mp.mpf(2)),
                   ("G_minus_2", mp.mpf(-2)), ("G_right_point", right_point))}
        record = {"status": "analytic_rates_not_fit", "decimal_digits": dps,
                  **{key: str(c[key]) for key in ("gamma", "t1", "t2", "kappa", "a", "q", "mu0")},
                  "right_point": str(right_point),
                  **{key: str(value) for key, value in values.items()},
                  "time_linear_coefficient": str(values["G_2"] / (4 * c["mu0"])),
                  "gram_left_exponential_rate": str(values["G_minus_2"] / 2),
                  "gram_right_exponential_rate": str(values["G_right_point"] / 2),
                  "right_delta_times_kernel_limit": str(1 / (1 - c["kappa"] ** -2)),
                  "time_filter_condition_pass": bool(values["G_2"] > 2),
                  "time_filter_margin": str(values["G_2"] - 2),
                  "threshold_upper_reconstruction": str(c["mu0"] / c["gamma"]),
                  "time_log_coefficient_candidate_unproved": str(-3 / (4 * c["mu0"])),
                  "scope": "Fixed gamma>t1+t2, t2>t1>0, L=4m, zero bias. Time result additionally requires G_a(2)>2 and a fixed low threshold.",
                  "proved_scope": "Matched leading exponents; exact right kernel factor. With the additional explicit strong-kernel condition, the prefactors command provides the full absolute constants.",
                  "unproved_scope": "Parameter regimes outside the sufficient strong-kernel condition, fixed nonzero bias, full-distance cutoff."}
        return record


def chebyshev_vector(m: int, z: mp.mpf) -> mp.matrix:
    values = [mp.mpf(1)]
    if m > 1:
        values.append(z)
    for _ in range(2, m):
        values.append(z * values[-1] - values[-2])
    return mp.matrix(list(reversed(values)))


def reflected_phi_grams(length: int, nodes: int, parameters: dict | None = None) -> dict[str, mp.matrix]:
    if length < 8 or length % 4:
        raise ValueError("length must be a multiple of 4, at least 8.")
    m = length // 4
    c = constants(parameters)
    matrices = {side: mp.matrix(m) for side in ("left", "right")}
    moments = {side: [mp.mpf(0) for _ in range(4 * m + 1)] for side in matrices}
    points, weights = mp.gauss_quadrature(nodes, "legendre")
    for point, weight in zip(points, weights):
        angle = (point + 1) * mp.pi / 4
        sine, cosine = mp.sin(angle), mp.cos(angle)
        r = c["q"] * sine * sine
        spectral_s = c["a0"] + c["b"] * (r + 1 / r)
        reflection = 1 + c["kappa"] * r - r ** (4 * m + 2) - c["kappa"] * r ** (4 * m + 1)
        jacobian = c["q"] * 2 * sine * cosine * mp.pi / 4
        left_scale = weight * jacobian * mp.sqrt(spectral_s - c["gamma"] ** 2) / (mp.pi * c["t1"] * reflection)
        scales = {"left": left_scale, "right": left_scale * c["t1"] ** 2 / spectral_s}
        power = mp.mpf(1)
        for degree in range(4 * m + 1):
            for side, scale in scales.items():
                moments[side][degree] += scale * power
            power *= r
    for side, matrix in matrices.items():
        values = moments[side]
        for i in range(m):
            for j in range(i, m):
                matrix[i, j] = (values[i + j] - values[2 * m + i - j]
                                - values[2 * m - i + j] + values[4 * m - i - j])
        for i in range(m):
            for j in range(i):
                matrix[i, j] = matrix[j, i]
    return matrices


def evaluation_kernel(inverse_phi: mp.matrix, z: mp.mpf) -> mp.mpf:
    vector = chebyshev_vector(inverse_phi.rows, z)
    return (vector.T * inverse_phi * vector)[0]


def finite_record(length: int, nodes: int, dps: int, parameters: dict | None = None) -> list[dict]:
    with mp.workdps(dps):
        m, n = length // 4, length // 4 - 1
        c = constants(parameters)
        grams = reflected_phi_grams(length, nodes, parameters)
        transform = mp.eye(m)
        for i in range(m - 1):
            transform[i, i + 1] = c["kappa"]
        right_point = -(c["kappa"] + 1 / c["kappa"])
        gamma_values = {"2": external_potential(mp.mpf(2), parameters),
                        "minus_2": external_potential(mp.mpf(-2), parameters),
                        "right_point": external_potential(right_point, parameters)}
        rows = []
        for side, phi_gram in grams.items():
            inverse_phi = mp.inverse(phi_gram)
            gram = phi_gram if side == "left" else transform * phi_gram * transform.T
            inverse = inverse_phi if side == "left" else mp.inverse(gram)
            eigenvalues = mp.eigsy(gram, eigvals_only=True)
            delta = eigenvalues[0]
            if delta <= 0:
                raise ArithmeticError("Minimum Gram eigenvalue is nonpositive; increase precision and nodes.")
            trace = mp.fsum(inverse[i, i] for i in range(m))
            square_trace = mp.fsum(inverse[i, j] * inverse[j, i] for i in range(m) for j in range(m))
            kernels = {name: evaluation_kernel(inverse_phi, z) for name, z in
                       (("2", mp.mpf(2)), ("minus_2", mp.mpf(-2)),
                        ("right_point", right_point))}
            if min(kernels.values()) <= 0:
                raise ArithmeticError("Evaluation kernel is nonpositive; increase precision and nodes.")
            gram_key = "minus_2" if side == "left" else "right_point"
            gram_rate = gamma_values[gram_key] / 2
            product = delta * kernels[gram_key]
            h_n = mp.mpf(m * (m + 1) * (2 * m + 1)) / 6
            geometric = 1 - c["kappa"] ** -2
            lower, upper = (mp.mpf(1), h_n) if side == "left" else (geometric, geometric ** -2)
            rows.append({
                "length": length, "m": m, "n": n, "initial": side,
                **{key: str(c[key]) for key in ("gamma", "t1", "t2", "kappa", "a", "mu0")},
                "right_point": str(right_point),
                "nodes": nodes, "dps": dps, "minimum_gram_eigenvalue": str(delta),
                "minus_log_delta_over_L": str(-mp.log(delta) / length),
                "gram_rate_target": str(gram_rate),
                "log_delta_plus_rate_L": str(mp.log(delta) + gram_rate * length),
                "inverse_trace": str(trace), "inverse_purity": str(square_trace / trace ** 2),
                "delta_times_relevant_kernel": str(product),
                "delta_kernel_finite_lower": str(lower), "delta_kernel_finite_upper": str(upper),
                "finite_kernel_bound_pass": bool(lower <= product <= upper),
                "right_kernel_prefactor_limit": str(1 / geometric) if side == "right" else None,
                "center_from_kernel": str(mp.log(kernels["2"]) / (2 * c["mu0"])),
                "time_linear_coefficient_target": str(gamma_values["2"] / (4 * c["mu0"])),
                "time_filter_condition_pass": bool(gamma_values["2"] > 2),
                "delta_left_prefactor_diagnostic": str(delta * mp.exp(gram_rate * length) / mp.mpf(length) ** mp.mpf("1.5")) if side == "left" else None,
                "delta_right_prefactor_diagnostic": str(delta * mp.exp(gram_rate * length)) if side == "right" else None,
                **{f"kernel_{key}": str(value) for key, value in kernels.items()},
                **{f"log_kernel_{key}_over_2n": str(mp.log(value) / (2 * n)) for key, value in kernels.items()},
                **{f"kernel_{key}_potential_target": str(gamma_values[key]) for key in kernels},
                **{f"log_kernel_{key}_minus_2nG": str(mp.log(value) - 2 * n * gamma_values[key])
                   for key, value in kernels.items()},
                "right_rank_one_error_scale": str(mp.sqrt(kernels["minus_2"] / kernels["right_point"]))
                    if side == "right" else None,
                "status": "high_precision_numerical_not_interval_certificate",
            })
        return rows


def compare_precision(rows: list[dict], refined: list[dict], tolerance: str) -> dict:
    by_case = {(row["length"], row["initial"]): row for row in refined}
    comparisons = []
    keys = ("minimum_gram_eigenvalue", "inverse_trace", "kernel_2", "kernel_minus_2", "kernel_right_point")
    with mp.workdps(max(row["dps"] for row in refined)):
        for row in rows:
            other = by_case[(row["length"], row["initial"])]
            relative = {key: abs(mp.mpf(row[key]) / mp.mpf(other[key]) - 1) for key in keys}
            maximum = max(relative.values())
            comparisons.append({"length": row["length"], "initial": row["initial"],
                                "relative_changes": {key: str(value) for key, value in relative.items()},
                                "maximum_relative_change": str(maximum),
                                "pass": bool(maximum <= mp.mpf(tolerance))})
    return {"tolerance": tolerance, "all_pass": all(item["pass"] for item in comparisons),
            "cases": comparisons, "meaning": "Precision and quadrature agreement, not an interval certificate."}


def center_audit(input_path: Path, dps: int, parameters: dict | None = None) -> dict:
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        inputs = list(csv.DictReader(stream))
    required = {"length", "initial", "level", "time"}
    if not inputs or not required.issubset(inputs[0]):
        raise ValueError("Center CSV must contain length, initial, level, and time columns.")
    with mp.workdps(dps):
        levels = sorted({row["level"] for row in inputs}, key=mp.mpf)
        theory = prefactor_record(dps, parameters, levels)
        constant_map = {(row["initial"], mp.mpf(row["epsilon"])): mp.mpf(row["constant_time_center"])
                        for row in theory["center_constants"]}
        tau = mp.mpf(theory["time_linear_coefficient"])
        logarithm = mp.mpf(theory["time_log_coefficient"])
        rows = []
        for row in inputs:
            length = int(row["length"])
            level = mp.mpf(row["level"])
            constant = constant_map[(row["initial"], level)]
            prediction = tau * length + logarithm * mp.log(length) + constant
            actual = mp.mpf(row["time"])
            rows.append({"length": length, "initial": row["initial"], "level": str(level),
                         "actual_time_as_stored": str(actual), "analytic_predicted_time": str(prediction),
                         "actual_minus_prediction": str(actual - prediction),
                         "absolute_residual": str(abs(actual - prediction)),
                         "analytic_constant_center": str(constant),
                         "status": "prediction_audit_no_fit"})
        rows.sort(key=lambda row: (row["initial"], mp.mpf(row["level"]), row["length"]))
        summaries = []
        for initial, level in sorted({(row["initial"], row["level"]) for row in rows}):
            group = [row for row in rows if row["initial"] == initial and row["level"] == level]
            summaries.append({"initial": initial, "level": level, "count": len(group),
                              "minimum_length": group[0]["length"], "maximum_length": group[-1]["length"],
                              "residual_at_minimum_length": group[0]["actual_minus_prediction"],
                              "residual_at_maximum_length": group[-1]["actual_minus_prediction"],
                              "maximum_absolute_residual": str(max(mp.mpf(row["absolute_residual"]) for row in group))})
        return {"input": str(input_path.resolve()), "theory": theory, "rows": rows, "summaries": summaries,
                "policy": "Coefficients are analytic and fixed before reading finite center values; no fit is performed."}


def input_overlap_record(length: int, dps: int, parameters: dict | None = None) -> list[dict]:
    if length < 8 or length % 4:
        raise ValueError("length must be a multiple of 4, at least 8.")
    with mp.workdps(dps):
        c = constants(parameters)
        m, size = length // 4, length // 2
        coupling = mp.eye(size) * c["t1"]
        for i in range(1, size):
            coupling[i, i - 1] = c["t2"]
        spectrum, right_vectors = mp.eigsy(coupling.T * coupling)
        if spectrum[0] <= 0:
            raise ArithmeticError("The edge singular value was lost; increase precision.")
        singular = [mp.sqrt(value) for value in spectrum]
        graph_values = [value / (c["gamma"] + mp.sqrt(c["gamma"] ** 2 - value ** 2)) for value in singular]
        left_vectors = coupling * right_vectors * mp.diag([1 / value for value in singular])
        graph = left_vectors * mp.diag(graph_values) * right_vectors.T
        mu_pi = mp.sqrt(c["gamma"] ** 2 - (c["t2"] - c["t1"]) ** 2)
        target_left = (c["gamma"] + mu_pi) / (2 * c["gamma"])
        expectation_target = ((c["t2"] - c["t1"]) / (c["gamma"] + mu_pi)) ** 2
        rows = []
        for side in ("left", "right"):
            occupied = list(range(m)) if side == "left" else list(range(m, size))
            empty = list(range(m, size)) if side == "left" else list(range(m))
            block = mp.matrix([[graph[i, j] for j in occupied] for i in empty])
            within = mp.matrix([[graph[i, j] for j in occupied] for i in occupied])
            gram = (mp.matrix([[block[i, m - 1 - j] for j in range(m)] for i in range(m)])
                    if side == "left" else
                    mp.matrix([[block[m - 1 - i, j] for j in range(m)] for i in range(m)]))
            eigenvalues, eigenvectors = mp.eigsy(gram)
            delta = eigenvalues[0]
            if delta <= 0:
                raise ArithmeticError("Reflected Gram lost positivity; increase precision.")
            inverse_block = mp.inverse(block)
            scaled_triangular_inverse = mp.matrix(size)
            scaled_shear = -delta * within * inverse_block
            scaled_block = delta * inverse_block
            for i in range(m):
                scaled_triangular_inverse[i, i] = delta
                for j in range(m):
                    scaled_triangular_inverse[i, m + j] = scaled_shear[i, j]
                    scaled_triangular_inverse[m + i, m + j] = scaled_block[i, j]
            order = occupied + empty
            reordered_left_vectors = mp.matrix([[left_vectors[i, j] for j in range(size)] for i in order])
            scaled_inverse = (scaled_triangular_inverse * reordered_left_vectors
                              * mp.diag([mp.sqrt(1 + value ** 2) for value in graph_values]))
            squares = mp.eigsy(scaled_inverse * scaled_inverse.T, eigvals_only=True)
            overlap_ratio = 1 / mp.sqrt(squares[size - 1])
            alpha = delta * overlap_ratio
            minimal_vector = eigenvectors[:, 0]
            input_vector = mp.matrix(size, 1)
            output_vector = mp.matrix(size, 1)
            for i in range(m):
                input_vector[occupied[i]] = minimal_vector[m - 1 - i] if side == "left" else minimal_vector[i]
                output_vector[empty[i]] = minimal_vector[i] if side == "left" else minimal_vector[m - 1 - i]
            norm_input = mp.norm(graph * input_vector) ** 2
            norm_output = mp.norm(graph.T * output_vector) ** 2
            inverse_gram = mp.inverse(gram)
            trace = mp.fsum(inverse_gram[i, i] for i in range(m))
            square_trace = mp.fsum(inverse_gram[i, j] * inverse_gram[j, i] for i in range(m) for j in range(m))
            rows.append({"length": length, "initial": side, "dps": dps,
                         **{key: str(c[key]) for key in ("gamma", "t1", "t2", "kappa", "a", "mu0")},
                         "alpha_infinite": str(alpha), "minimum_gram_eigenvalue": str(delta),
                         "alpha_to_delta": str(overlap_ratio),
                         "alpha_to_delta_limit": str(target_left if side == "left" else mp.mpf(1)),
                         "input_graph_expectation": str(norm_input),
                         "output_graph_expectation": str(norm_output),
                         "graph_expectation_limit": str(expectation_target if side == "left" else mp.mpf(0)),
                         "inverse_purity": str(square_trace / trace ** 2),
                         "rank_one_ratio_estimate": str(1 / mp.sqrt((1 + norm_input) * (1 + norm_output))),
                         "reflected_symmetry_relative_error": str(mp.norm(gram - gram.T, p=mp.inf) / mp.norm(gram, p=mp.inf)),
                         "status": "independent_matrix_function_whitened_inverse_not_interval_certificate"})
        return rows


def compare_input_precision(rows: list[dict], refined: list[dict], tolerance: str) -> dict:
    by_case = {(row["length"], row["initial"]): row for row in refined}
    cases = []
    with mp.workdps(max(row["dps"] for row in refined)):
        for row in rows:
            other = by_case[(row["length"], row["initial"])]
            changes = {key: abs(mp.mpf(row[key]) / mp.mpf(other[key]) - 1)
                       for key in ("alpha_infinite", "minimum_gram_eigenvalue", "alpha_to_delta")}
            maximum = max(changes.values())
            cases.append({"length": row["length"], "initial": row["initial"],
                          "relative_changes": {key: str(value) for key, value in changes.items()},
                          "maximum_relative_change": str(maximum), "pass": bool(maximum <= mp.mpf(tolerance))})
    return {"all_pass": all(row["pass"] for row in cases), "tolerance": tolerance, "cases": cases,
            "meaning": "Precision agreement for independent whitened overlap, not interval certification."}


def save_record(record: dict, output: Path | None, csv_output: Path | None) -> None:
    serialized = json.dumps(record, ensure_ascii=True, indent=2)
    if output is None:
        print(serialized)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print(json.dumps({"written": str(output.resolve()), "rows": len(record.get("rows", []))}))
    if csv_output is not None:
        rows = record.get("rows", [])
        if not rows:
            raise ValueError("CSV output requires the finite command.")
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    rates_parser = subparsers.add_parser("rates")
    prefactors_parser = subparsers.add_parser("prefactors")
    finite_parser = subparsers.add_parser("finite")
    centers_parser = subparsers.add_parser("center-audit")
    input_parser = subparsers.add_parser("input-audit")
    for subparser in (rates_parser, prefactors_parser, finite_parser, centers_parser, input_parser):
        subparser.add_argument("--dps", type=int, default=100)
        subparser.add_argument("--output", type=Path)
        subparser.add_argument("--gamma", default="2")
        subparser.add_argument("--t1", default=".5")
        subparser.add_argument("--t2", default="1")
    prefactors_parser.add_argument("--levels", default=".3,.05,.01")
    centers_parser.add_argument("--input", type=Path, required=True)
    centers_parser.add_argument("--csv", type=Path)
    finite_parser.add_argument("--lengths", default="24,32,48")
    finite_parser.add_argument("--nodes", type=int, default=96)
    finite_parser.add_argument("--csv", type=Path)
    finite_parser.add_argument("--precision-pair", action="store_true")
    finite_parser.add_argument("--refined-dps", type=int)
    finite_parser.add_argument("--refined-nodes", type=int)
    finite_parser.add_argument("--agreement-tolerance", default="1e-20")
    input_parser.add_argument("--lengths", default="24,32")
    input_parser.add_argument("--csv", type=Path)
    input_parser.add_argument("--precision-pair", action="store_true")
    input_parser.add_argument("--refined-dps", type=int)
    input_parser.add_argument("--agreement-tolerance", default="1e-20")
    arguments = parser.parse_args()
    if arguments.dps < 30:
        parser.error("--dps must be at least 30.")
    parameters = {key: getattr(arguments, key) for key in ("gamma", "t1", "t2")}
    if arguments.command == "rates":
        save_record(rate_record(arguments.dps, parameters), arguments.output, None)
    elif arguments.command == "prefactors":
        save_record(prefactor_record(arguments.dps, parameters, arguments.levels.split(",")), arguments.output, None)
    elif arguments.command == "center-audit":
        save_record(center_audit(arguments.input, arguments.dps, parameters), arguments.output, arguments.csv)
    elif arguments.command == "input-audit":
        lengths = [int(value) for value in arguments.lengths.split(",")]
        rows = []
        for length in lengths:
            rows.extend(input_overlap_record(length, arguments.dps, parameters))
        record = {"rows": rows, "parameters": parameters,
                  "method": "Independent complete coupling matrix function and scaled whitened inverse; no reflected quadrature used."}
        if arguments.precision_pair:
            refined_dps = arguments.refined_dps or arguments.dps + 40
            if refined_dps <= arguments.dps:
                parser.error("The refined precision must exceed the first run.")
            refined = []
            for length in lengths:
                refined.extend(input_overlap_record(length, refined_dps, parameters))
            record["refined_rows"] = refined
            record["precision_comparison"] = compare_input_precision(rows, refined, arguments.agreement_tolerance)
        save_record(record, arguments.output, arguments.csv)
    else:
        lengths = [int(value) for value in arguments.lengths.split(",")]
        if arguments.nodes < max(lengths) // 4:
            parser.error("--nodes must be at least the largest Gram matrix order.")
        rows = []
        for length in lengths:
            rows.extend(finite_record(length, arguments.nodes, arguments.dps, parameters))
        record = {"rates": rate_record(arguments.dps, parameters), "rows": rows,
                  "precision_policy": "Repeat with higher dps and quadrature nodes before accepting finite values."}
        if arguments.precision_pair:
            refined_dps = arguments.refined_dps or arguments.dps + 40
            refined_nodes = arguments.refined_nodes or max(arguments.nodes + 32, arguments.nodes * 3 // 2)
            if refined_dps <= arguments.dps or refined_nodes <= arguments.nodes:
                parser.error("The refined precision and nodes must both exceed the first run.")
            refined = []
            for length in lengths:
                refined.extend(finite_record(length, refined_nodes, refined_dps, parameters))
            record["refined_rows"] = refined
            record["precision_comparison"] = compare_precision(rows, refined, arguments.agreement_tolerance)
        save_record(record, arguments.output, arguments.csv)


if __name__ == "__main__":
    main()
