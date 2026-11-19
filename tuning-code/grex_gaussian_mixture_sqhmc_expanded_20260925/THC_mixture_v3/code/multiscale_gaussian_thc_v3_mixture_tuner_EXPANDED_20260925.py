"""Tune refreshed-mixture THC-v3 on the five-dimensional multiscale Gaussian."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import numba
from numba import njit


PROTOCOL_VERSION = "MSG5-expanded-mixture-SQHMC-20260925-v1.0.0"
IMPLEMENTATION_VERSION = "exactB-W-subcycled-fullMH-cachefree-20260830-v2"
RANDOM_STREAM_VERSION = "seedsequence-pcg64-five-substreams-v1.0.0"
DIAGNOSTICS_VERSION = "multichain-geyer-ims-rank-rhat-v1.0.0"
CANONICAL_OUTPUT_FILENAME = 'THC_v3_mixture_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt'

CANONICAL_INNER_R = 8
CANONICAL_MIXTURE_WEIGHTS = np.asarray(
    [0.5641470, 0.1084317, 0.3274213], dtype=np.float64
)
CANONICAL_MIXTURE_CENTERS = np.asarray(
    [0.7161277448, 0.1776353476, 0.0], dtype=np.float64
)


@dataclass(frozen=True, order=True)
class Configuration:
    delta: float
    Q: float
    epsilon: float
    L: int


@dataclass(frozen=True)
class TuningInputs:
    beta: float
    omegas: Sequence[float]
    mass_matrix: Sequence[Sequence[float]]
    standardized_starts: Sequence[Sequence[float]]
    observable_policy: str
    mixture_weights: Sequence[float]
    mixture_centers: Sequence[float]
    inner_r: int
    delta_values: Sequence[float]
    Q_values: Sequence[float]
    epsilon_values: Sequence[float]
    L_values: Sequence[int]
    seed_blocks: Sequence[Sequence[int]]
    n_warmup: int
    n_retained: int
    n_workers: int
    run_mode: str
    run_label: str
    output_directory: str | os.PathLike[str] | None
    output_filename: str
    allow_overwrite: bool
    print_progress: bool
    print_report: bool
    run_self_tests: bool


@dataclass(frozen=True)
class PreparedInputs:
    beta: float
    omegas: np.ndarray
    omega_squared: np.ndarray
    target_variances: np.ndarray
    mass_matrix: np.ndarray
    inverse_mass_matrix: np.ndarray
    mass_cholesky: np.ndarray
    mass_is_identity: bool
    standardized_starts: np.ndarray
    physical_starts: np.ndarray
    mixture_weights: np.ndarray
    mixture_centers: np.ndarray
    phase_radii: np.ndarray
    inner_r: int
    dimension: int
    n_chains: int
    observable_names: tuple[str, ...]
    exact_observable_means: np.ndarray
    exact_observable_sds: np.ndarray
    delta_values: tuple[float, ...]
    Q_values: tuple[float, ...]
    epsilon_values: tuple[float, ...]
    L_values: tuple[int, ...]
    seed_blocks: tuple[tuple[int, ...], ...]
    n_warmup: int
    n_retained: int
    n_workers: int
    run_mode: str
    run_label: str
    output_directory: str | os.PathLike[str] | None
    output_filename: str
    allow_overwrite: bool
    print_progress: bool
    print_report: bool
    run_self_tests: bool
    input_fingerprint: str


@dataclass(frozen=True)
class IATResult:
    tau: float
    status: str
    pair_count: int
    last_included_lag: int
    first_excluded_lag: int
    cutoff_reason: str

    @property
    def is_defined(self) -> bool:
        return (
            self.status.startswith("ok_")
            and math.isfinite(self.tau)
            and self.tau > 0.0
        )


@dataclass(frozen=True)
class RhatResult:
    rank: float
    folded: float
    combined: float
    status: str


@dataclass(frozen=True)
class ObservableDiagnostic:
    name: str
    iat: IATResult
    ess: float
    efficiency: float
    rhat: RhatResult
    empirical_mean: float
    exact_mean: float
    raw_moment_error: float
    standardized_moment_error: float


@dataclass(frozen=True)
class BlockResult:
    epsilon: float
    L: int
    block_index: int
    seeds: tuple[int, ...]
    status: str
    acceptance: float
    retained_gradient_calls: int
    gradients_per_retained_draw: float
    warmup_nonfinite_trajectories: int
    retained_nonfinite_trajectories: int
    observable_diagnostics: tuple[ObservableDiagnostic, ...]
    worst_observable: str
    worst_iat: float
    block_efficiency: float
    maximum_rhat_observable: str
    maximum_rhat: float
    maximum_moment_error_observable: str
    maximum_standardized_moment_error: float


@dataclass(frozen=True)
class BlockAggregate:
    status: str
    geometric_mean_efficiency: float
    minimum_block_efficiency: float
    median_block_efficiency: float
    maximum_block_efficiency: float
    mean_acceptance: float
    maximum_rhat: float
    maximum_standardized_moment_error: float
    total_nonfinite_trajectories: int


@dataclass(frozen=True)
class MixtureMomentDiagnostic:
    empirical_mean: float
    exact_mean: float
    mean_error: float
    empirical_second_moment: float
    exact_second_moment: float
    second_moment_error: float


@dataclass(frozen=True)
class MixtureBlockResult:
    physical: BlockResult
    post_mh_thermostat: MixtureMomentDiagnostic
    refreshed_thermostat: MixtureMomentDiagnostic


@dataclass(frozen=True)
class ConfigurationResult:
    configuration: Configuration
    origin: str
    status: str
    block_results: tuple[MixtureBlockResult, ...]
    geometric_mean_efficiency: float
    minimum_block_efficiency: float
    median_block_efficiency: float
    maximum_block_efficiency: float
    mean_acceptance: float
    maximum_rhat: float
    maximum_standardized_moment_error: float
    total_nonfinite_trajectories: int








@njit(cache=False, inline="always")
def _potential_energy(q: np.ndarray, omega_squared: np.ndarray) -> float:
    total = 0.0
    for index in range(q.size):
        total += omega_squared[index] * q[index] * q[index]
    return 0.5 * total


@njit(cache=False, inline="always")
def _target_gradient(q: np.ndarray, omega_squared: np.ndarray) -> np.ndarray:
    result = np.empty_like(q)
    for index in range(q.size):
        result[index] = omega_squared[index] * q[index]
    return result


@njit(cache=False, inline="always")
def _quadratic_form(
    p: np.ndarray,
    inverse_mass_matrix: np.ndarray,
) -> float:
    total = 0.0
    for row in range(p.size):
        inverse_mass_p = 0.0
        for column in range(p.size):
            inverse_mass_p += inverse_mass_matrix[row, column] * p[column]
        total += p[row] * inverse_mass_p
    return total


@njit(cache=False, inline="always")
def _log_rho_and_score(
    s: float,
    delta: float,
    weights: np.ndarray,
    centers: np.ndarray,
) -> tuple[float, float]:
    """Stable log mixture density and d log(rho)/ds."""

    if not math.isfinite(s) or abs(s) > 1.0e100:
        return -math.inf, (-math.inf if s > 0.0 else math.inf)
    maximum = -math.inf
    for component in range(weights.size):
        scaled = (s - centers[component]) / delta
        value = math.log(weights[component]) - 0.5 * scaled * scaled
        if value > maximum:
            maximum = value
    if not math.isfinite(maximum):
        return -math.inf, (-math.inf if s > 0.0 else math.inf)

    denominator = 0.0
    score_numerator = 0.0
    inverse_delta_squared = 1.0 / (delta * delta)
    for component in range(weights.size):
        difference = s - centers[component]
        scaled = difference / delta
        log_weight = math.log(weights[component]) - 0.5 * scaled * scaled
        responsibility = math.exp(log_weight - maximum)
        denominator += responsibility
        score_numerator += responsibility * (-difference * inverse_delta_squared)
    log_normalizer = math.log(delta) + 0.5 * math.log(2.0 * math.pi)
    return (
        maximum + math.log(denominator) - log_normalizer,
        score_numerator / denominator,
    )


@njit(cache=False, inline="always")
def _safe_exp(exponent: float) -> float:
    if exponent > 700.0:
        return math.inf
    if exponent < -745.0:
        return 0.0
    return math.exp(exponent)


@njit(cache=False, inline="always")
def _state_is_finite(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
) -> bool:
    if not math.isfinite(s) or not math.isfinite(p_s):
        return False
    for index in range(q.size):
        if not math.isfinite(q[index]) or not math.isfinite(p[index]):
            return False
    return True


@njit(cache=False)
def _hamiltonian(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    beta: float,
    delta: float,
    Q: float,
    weights: np.ndarray,
    centers: np.ndarray,
    omega_squared: np.ndarray,
    inverse_mass_matrix: np.ndarray,
) -> float:
    inverse_h_squared = _safe_exp(-2.0 * s)
    if not math.isfinite(inverse_h_squared) or inverse_h_squared <= 0.0:
        return math.inf
    log_rho, _score = _log_rho_and_score(s, delta, weights, centers)
    if not math.isfinite(log_rho):
        return math.inf
    return (
        _potential_energy(q, omega_squared)
        + 0.5 * inverse_h_squared * _quadratic_form(p, inverse_mass_matrix)
        + (q.size / beta) * s
        + 0.5 * p_s * p_s / Q
        - log_rho / beta
    )


_LOG_MAX_FLOAT64 = 709.782712893384
_LOG_MIN_SUBNORMAL_FLOAT64 = -744.4400719213812
_LOG_TWO = 0.6931471805599453
_SMALL_TAU = 1.0e-4
_SQRT_FLOAT64_EPS = 1.4901161193847656e-8


@njit(cache=False, inline="always")
def _quadratic_and_inverse_mass_p_stable(
    p: np.ndarray,
    inverse_mass_matrix: np.ndarray,
) -> tuple[float, np.ndarray]:
    inverse_mass_p = np.empty_like(p)
    quadratic = 0.0
    for row in range(p.size):
        value = 0.0
        for column in range(p.size):
            value += inverse_mass_matrix[row, column] * p[column]
        inverse_mass_p[row] = value
        quadratic += p[row] * value
    return quadratic, inverse_mass_p


@njit(cache=False, inline="always")
def _all_zero(vector: np.ndarray) -> bool:
    for index in range(vector.size):
        if vector[index] != 0.0:
            return False
    return True


@njit(cache=False, inline="always")
def _finite_state_stable(
    q: np.ndarray, p: np.ndarray, s: float, p_s: float
) -> bool:
    if not math.isfinite(s) or not math.isfinite(p_s):
        return False
    for index in range(q.size):
        if not math.isfinite(q[index]) or not math.isfinite(p[index]):
            return False
    return True


@njit(cache=False, inline="always")
def _logaddexp_pair(first: float, second: float) -> float:
    if first == -math.inf:
        return second
    if second == -math.inf:
        return first
    maximum = first if first >= second else second
    minimum = second if first >= second else first
    return maximum + math.log1p(math.exp(minimum - maximum))


@njit(cache=False, inline="always")
def _sinhc_small(value: float) -> float:
    square = value * value
    return (
        1.0 + square / 6.0 + square * square / 120.0
        + square * square * square / 5040.0
    )


@njit(cache=False, inline="always")
def _small_tau_log_denominator(tau: float, b: float) -> float:
    square = tau * tau
    delta = (
        b * tau + 0.5 * square + b * tau * square / 6.0
        + square * square / 24.0 + b * tau * square * square / 120.0
        + square * square * square / 720.0
    )
    return math.log1p(delta)


@njit(cache=False, inline="always")
def _log_abs_sinh(value: float) -> float:
    absolute = abs(value)
    if absolute == 0.0:
        return -math.inf
    if absolute < _SMALL_TAU:
        return math.log(absolute) + math.log(_sinhc_small(value))
    return absolute + math.log(-math.expm1(-2.0 * absolute)) - _LOG_TWO


@njit(cache=False, inline="always")
def _signed_exp(log_absolute: float, sign: float) -> tuple[float, bool]:
    if log_absolute > _LOG_MAX_FLOAT64:
        return 0.0, False
    if log_absolute < _LOG_MIN_SUBNORMAL_FLOAT64:
        return 0.0, True
    value = math.exp(log_absolute)
    if sign < 0.0:
        value = -value
    return value, math.isfinite(value)


@njit(cache=False, inline="always")
def _compute_signed_tau(
    amplitude: float, duration: float, Q: float
) -> tuple[float, bool]:
    rate = amplitude / Q
    if math.isfinite(rate):
        tau = duration * rate
        if math.isfinite(tau) and (tau != 0.0 or duration == 0.0):
            return tau, True
    log_absolute_tau = math.log(abs(duration)) + math.log(amplitude) - math.log(Q)
    if log_absolute_tau > _LOG_MAX_FLOAT64:
        return 0.0, False
    if log_absolute_tau < _LOG_MIN_SUBNORMAL_FLOAT64:
        return math.copysign(0.0, duration), True
    return math.copysign(math.exp(log_absolute_tau), duration), True


@njit(cache=False)
def exact_loglinear_B_flow_stable(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    Q: float,
    duration: float,
    inverse_mass_matrix: np.ndarray,
) -> tuple[np.ndarray, float, float, bool]:
    """Exact log-linear B flow in stable float64 arithmetic."""

    flowed_q = q.copy()
    if (
        not math.isfinite(s) or not math.isfinite(p_s)
        or not math.isfinite(Q) or Q <= 0.0
        or not math.isfinite(duration)
    ):
        return flowed_q, s, p_s, False
    if duration == 0.0:
        return flowed_q, s, p_s, _finite_state_stable(flowed_q, p, s, p_s)
    momentum_quadratic, inverse_mass_p = (
        _quadratic_and_inverse_mass_p_stable(p, inverse_mass_matrix)
    )
    if not math.isfinite(momentum_quadratic) or momentum_quadratic < 0.0:
        return flowed_q, s, p_s, False
    if momentum_quadratic == 0.0:
        if not _all_zero(inverse_mass_p):
            return flowed_q, s, p_s, False
        flowed_s = s + duration * (p_s / Q)
        return flowed_q, flowed_s, p_s, _finite_state_stable(flowed_q, p, flowed_s, p_s)

    sqrt_quadratic = math.sqrt(momentum_quadratic)
    log_kinetic_amplitude = 0.5 * math.log(Q) - s + math.log(sqrt_quadratic)
    if log_kinetic_amplitude > _LOG_MAX_FLOAT64:
        return flowed_q, s, p_s, False
    kinetic_amplitude = (
        0.0 if log_kinetic_amplitude < _LOG_MIN_SUBNORMAL_FLOAT64
        else math.exp(log_kinetic_amplitude)
    )
    amplitude = math.hypot(p_s, kinetic_amplitude)
    if not math.isfinite(amplitude):
        return flowed_q, s, p_s, False
    if amplitude == 0.0:
        return flowed_q, s, p_s, _finite_state_stable(flowed_q, p, s, p_s)
    tau, tau_is_finite = _compute_signed_tau(amplitude, duration, Q)
    if not tau_is_finite:
        return flowed_q, s, p_s, False

    absolute_b = min(abs(p_s) / amplitude, 1.0)
    sine_component = kinetic_amplitude / amplitude
    small_weight = 0.5 * sine_component * sine_component / (1.0 + absolute_b)
    small_weight = min(max(small_weight, 0.0), 0.5)
    if p_s >= 0.0:
        minus_weight = small_weight
        plus_weight = 1.0 - small_weight
    else:
        plus_weight = small_weight
        minus_weight = 1.0 - small_weight
    log_plus_weight = math.log(plus_weight) if plus_weight > 0.0 else -math.inf
    log_minus_weight = math.log(minus_weight) if minus_weight > 0.0 else -math.inf
    log_denominator = _logaddexp_pair(
        log_plus_weight + tau, log_minus_weight - tau
    )
    if not math.isfinite(log_denominator):
        return flowed_q, s, p_s, False
    if abs(tau) < _SMALL_TAU:
        b = min(max(p_s / amplitude, -1.0), 1.0)
        log_denominator = _small_tau_log_denominator(tau, b)
    flowed_s = s + log_denominator

    if plus_weight == 0.0:
        momentum_ratio = -1.0
    elif minus_weight == 0.0:
        momentum_ratio = 1.0
    else:
        rapidity = tau + 0.5 * (log_plus_weight - log_minus_weight)
        momentum_ratio = math.tanh(rapidity)
    flowed_p_s = amplitude * momentum_ratio
    momentum_change = flowed_p_s - p_s
    momentum_scale = max(amplitude, abs(p_s), np.finfo(np.float64).tiny)
    if abs(momentum_change) > _SQRT_FLOAT64_EPS * momentum_scale:
        integrated_inverse_h_squared = momentum_change / momentum_quadratic
        if math.isfinite(integrated_inverse_h_squared):
            direct_q = flowed_q.copy()
            direct_is_finite = True
            for coordinate in range(direct_q.size):
                increment = integrated_inverse_h_squared * inverse_mass_p[coordinate]
                direct_q[coordinate] += increment
                if not math.isfinite(increment) or not math.isfinite(direct_q[coordinate]):
                    direct_is_finite = False
                    break
            if direct_is_finite:
                return direct_q, flowed_s, flowed_p_s, _finite_state_stable(
                    direct_q, p, flowed_s, flowed_p_s
                )

    if tau == 0.0:
        log_abs_integral = -2.0 * s + math.log(abs(duration))
    elif abs(tau) < _SMALL_TAU:
        log_abs_integral = (
            -2.0 * s + math.log(abs(duration))
            + math.log(_sinhc_small(tau)) - log_denominator
        )
    else:
        log_abs_integral = (
            math.log(Q) - math.log(amplitude) - 2.0 * s
            + _log_abs_sinh(tau) - log_denominator
        )
    integral_sign = -1.0 if duration < 0.0 else 1.0
    for coordinate in range(flowed_q.size):
        velocity = inverse_mass_p[coordinate]
        if velocity == 0.0:
            continue
        increment, increment_is_finite = _signed_exp(
            log_abs_integral + math.log(abs(velocity)),
            integral_sign if velocity > 0.0 else -integral_sign,
        )
        if not increment_is_finite:
            return q.copy(), s, p_s, False
        flowed_q[coordinate] += increment
    return flowed_q, flowed_s, flowed_p_s, _finite_state_stable(
        flowed_q, p, flowed_s, flowed_p_s
    )


@njit(cache=False, nogil=True)
def _exactB_inner_proposal(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    current_gradient: np.ndarray,
    beta: float,
    delta: float,
    Q: float,
    inner_r: int,
    epsilon: float,
    L: int,
    weights: np.ndarray,
    centers: np.ndarray,
    omega_squared: np.ndarray,
    inverse_mass_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, int, bool]:
    proposed_q = q.copy()
    proposed_p = p.copy()
    proposed_s = s
    proposed_p_s = p_s
    proposed_gradient = current_gradient.copy()
    gradient_calls = 0
    dimension = proposed_q.size

    for coordinate in range(dimension):
        proposed_p[coordinate] -= 0.5 * epsilon * proposed_gradient[coordinate]

    inner_duration = epsilon / inner_r
    half_inner_duration = 0.5 * inner_duration
    for outer_step in range(L):
        for _inner in range(inner_r):
            _log_rho, score = _log_rho_and_score(
                proposed_s, delta, weights, centers
            )
            proposed_p_s += (
                half_inner_duration * (score - dimension) / beta
            )
            proposed_q, proposed_s, proposed_p_s, finite = (
                exact_loglinear_B_flow_stable(
                proposed_q,
                proposed_p,
                proposed_s,
                proposed_p_s,
                Q,
                inner_duration,
                inverse_mass_matrix,
                )
            )
            if not finite:
                return (
                    proposed_q,
                    proposed_p,
                    proposed_s,
                    proposed_p_s,
                    proposed_gradient,
                    gradient_calls,
                    False,
                )
            _log_rho, score = _log_rho_and_score(
                proposed_s, delta, weights, centers
            )
            proposed_p_s += (
                half_inner_duration * (score - dimension) / beta
            )

        proposed_gradient = _target_gradient(proposed_q, omega_squared)
        gradient_calls += 1
        kick_duration = 0.5 * epsilon if outer_step == L - 1 else epsilon
        for coordinate in range(dimension):
            proposed_p[coordinate] -= kick_duration * proposed_gradient[coordinate]
        if not _state_is_finite(
            proposed_q, proposed_p, proposed_s, proposed_p_s
        ):
            return (
                proposed_q,
                proposed_p,
                proposed_s,
                proposed_p_s,
                proposed_gradient,
                gradient_calls,
                False,
            )

    return (
        proposed_q,
        proposed_p,
        proposed_s,
        proposed_p_s,
        proposed_gradient,
        gradient_calls,
        True,
    )


@njit(cache=False, nogil=True)
def _simulate_chain(
    standard_physical_momenta: np.ndarray,
    standard_thermostat_momenta: np.ndarray,
    mixture_uniforms: np.ndarray,
    standard_thermostat_positions: np.ndarray,
    acceptance_uniforms: np.ndarray,
    q_initial: np.ndarray,
    beta: float,
    delta: float,
    Q: float,
    inner_r: int,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    weights: np.ndarray,
    centers: np.ndarray,
    omega_squared: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int, int]:
    total_iterations = n_warmup + n_retained
    dimension = q_initial.size
    samples = np.empty((n_retained, dimension), dtype=np.float64)
    post_mh_s_samples = np.empty(n_retained, dtype=np.float64)
    refreshed_s_samples = np.empty(n_retained, dtype=np.float64)
    q = q_initial.copy()
    current_gradient = _target_gradient(q, omega_squared)
    warmup_gradient_calls = 1 if n_warmup > 0 else 0
    retained_gradient_calls = 0 if n_warmup > 0 else 1
    accepted_retained = 0
    warmup_nonfinite = 0
    retained_nonfinite = 0
    inverse_sqrt_beta = 1.0 / math.sqrt(beta)
    p_s_scale = math.sqrt(Q / beta)

    for iteration in range(total_iterations):
        component = weights.size - 1
        cumulative = 0.0
        for candidate in range(weights.size):
            cumulative += weights[candidate]
            if mixture_uniforms[iteration] < cumulative:
                component = candidate
                break
        s = centers[component] + delta * standard_thermostat_positions[iteration]

        p_scale = _safe_exp(s) * inverse_sqrt_beta
        p = np.empty(dimension, dtype=np.float64)
        for row in range(dimension):
            transformed = 0.0
            for column in range(dimension):
                transformed += (
                    mass_cholesky[row, column]
                    * standard_physical_momenta[iteration, column]
                )
            p[row] = p_scale * transformed
        p_s = p_s_scale * standard_thermostat_momenta[iteration]

        initial_energy = _hamiltonian(
            q,
            p,
            s,
            p_s,
            beta,
            delta,
            Q,
            weights,
            centers,
            omega_squared,
            inverse_mass_matrix,
        )
        proposal = _exactB_inner_proposal(
            q,
            p,
            s,
            p_s,
            current_gradient,
            beta,
            delta,
            Q,
            inner_r,
            epsilon,
            L,
            weights,
            centers,
            omega_squared,
            inverse_mass_matrix,
        )
        proposed_q, proposed_p, proposed_s, proposed_p_s = proposal[:4]
        if iteration < n_warmup:
            warmup_gradient_calls += proposal[5]
        else:
            retained_gradient_calls += proposal[5]

        proposal_is_finite = proposal[6]
        if proposal_is_finite:
            for coordinate in range(dimension):
                proposed_p[coordinate] = -proposed_p[coordinate]
            proposed_p_s = -proposed_p_s

        accepted = False
        if proposal_is_finite and math.isfinite(initial_energy):
            proposed_energy = _hamiltonian(
                proposed_q,
                proposed_p,
                proposed_s,
                proposed_p_s,
                beta,
                delta,
                Q,
                weights,
                centers,
                omega_squared,
                inverse_mass_matrix,
            )
            energy_change = proposed_energy - initial_energy
            if math.isfinite(energy_change):
                uniform = acceptance_uniforms[iteration]
                log_uniform = -math.inf if uniform <= 0.0 else math.log(uniform)
                accepted = log_uniform < -beta * energy_change
            else:
                proposal_is_finite = False
        else:
            proposal_is_finite = False

        if not proposal_is_finite:
            if iteration < n_warmup:
                warmup_nonfinite += 1
            else:
                retained_nonfinite += 1
        elif accepted:
            q = proposed_q
            current_gradient = proposal[4]

        if iteration >= n_warmup:
            retained_index = iteration - n_warmup
            samples[retained_index] = q
            refreshed_s_samples[retained_index] = s
            post_mh_s_samples[retained_index] = proposed_s if accepted else s
            if accepted:
                accepted_retained += 1

    return (
        samples,
        post_mh_s_samples,
        refreshed_s_samples,
        accepted_retained,
        warmup_nonfinite,
        retained_nonfinite,
        warmup_gradient_calls,
        retained_gradient_calls,
    )


def _autocovariance_fft(chains: np.ndarray) -> np.ndarray:
    """Return per-chain biased autocovariances with denominator N at every lag."""

    chains = np.asarray(chains, dtype=np.float64)
    centered = chains - np.mean(chains, axis=1, keepdims=True)
    n_draws = centered.shape[1]
    fft_size = 1 << (2 * n_draws - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size, axis=1)
    autocovariance = np.fft.irfft(
        transformed * np.conjugate(transformed),
        n=fft_size,
        axis=1,
    )[:, :n_draws]
    return autocovariance / n_draws


def _multi_chain_geyer_iat(chains: np.ndarray) -> IATResult:
    """Multi-chain Geyer initial-positive and initial-monotone IAT."""

    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        return IATResult(math.nan, "undefined_shape", 0, -1, -1, "shape")
    n_chains, n_draws = chains.shape
    if n_chains < 2:
        return IATResult(
            math.nan, "undefined_insufficient_chains", 0, -1, -1, "chains"
        )
    if n_draws < 4:
        return IATResult(
            math.nan, "undefined_insufficient_draws", 0, -1, -1, "draws"
        )
    if not np.all(np.isfinite(chains)):
        return IATResult(
            math.nan, "undefined_nonfinite_input", 0, -1, -1, "nonfinite_input"
        )

    chain_variances = np.var(chains, axis=1, ddof=1)
    within_variance = float(np.mean(chain_variances))
    chain_means = np.mean(chains, axis=1)
    variance_plus = (
        ((n_draws - 1.0) / n_draws) * within_variance
        + float(np.var(chain_means, ddof=1))
    )
    if (
        not math.isfinite(within_variance)
        or not math.isfinite(variance_plus)
        or within_variance <= np.finfo(np.float64).tiny
        or variance_plus <= np.finfo(np.float64).tiny
    ):
        return IATResult(
            math.nan, "undefined_zero_variance", 0, -1, -1, "variance"
        )

    mean_autocovariance = np.mean(_autocovariance_fft(chains), axis=0)
    rho = np.empty(n_draws, dtype=np.float64)
    rho[0] = 1.0
    rho[1:] = 1.0 - (
        within_variance - mean_autocovariance[1:]
    ) / variance_plus

    usable_lags = n_draws if n_draws % 2 == 0 else n_draws - 1
    raw_pair_sums = rho[:usable_lags].reshape(-1, 2).sum(axis=1)
    included_count = raw_pair_sums.size
    first_excluded_lag = -1
    cutoff_reason = "end_of_available_lags"

    for pair_index, pair_sum in enumerate(raw_pair_sums):
        if not math.isfinite(float(pair_sum)):
            return IATResult(
                math.nan,
                "undefined_nonfinite_pair",
                pair_index,
                2 * pair_index - 1,
                2 * pair_index,
                "nonfinite_pair",
            )
        if pair_sum <= 0.0:
            included_count = pair_index
            first_excluded_lag = 2 * pair_index
            cutoff_reason = "first_nonpositive_pair"
            break

    if included_count == 0:
        return IATResult(
            math.nan,
            "undefined_no_positive_pair",
            0,
            -1,
            first_excluded_lag,
            cutoff_reason,
        )

    monotone_pairs = np.minimum.accumulate(raw_pair_sums[:included_count])
    tau = -1.0 + 2.0 * float(np.sum(monotone_pairs))
    last_included_lag = 2 * (included_count - 1) + 1
    if not math.isfinite(tau) or tau <= 0.0:
        return IATResult(
            math.nan,
            "undefined_nonpositive_tau",
            included_count,
            last_included_lag,
            first_excluded_lag,
            cutoff_reason,
        )

    status = (
        "ok_truncated_nonpositive"
        if cutoff_reason == "first_nonpositive_pair"
        else "ok_full_window"
    )
    return IATResult(
        tau,
        status,
        included_count,
        last_included_lag,
        first_excluded_lag,
        cutoff_reason,
    )


def _basic_rhat(chains: np.ndarray) -> float:
    """Return the unfloored between/within-chain R-hat."""

    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        return math.nan
    n_chains, n_draws = chains.shape
    if n_chains < 2 or n_draws < 2 or not np.all(np.isfinite(chains)):
        return math.nan
    within_variance = float(np.mean(np.var(chains, axis=1, ddof=1)))
    between_variance = n_draws * float(
        np.var(np.mean(chains, axis=1), ddof=1)
    )
    if not math.isfinite(within_variance) or not math.isfinite(between_variance):
        return math.nan
    if within_variance <= np.finfo(float).tiny:
        return math.inf if between_variance > np.finfo(float).tiny else math.nan
    variance_estimate = (
        ((n_draws - 1.0) / n_draws) * within_variance
        + between_variance / n_draws
    )
    if variance_estimate < 0.0 or not math.isfinite(variance_estimate):
        return math.nan
    return math.sqrt(variance_estimate / within_variance)


def _split_chains(chains: np.ndarray) -> np.ndarray:
    """Split every chain into equal first and last halves."""

    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        return np.empty((0, 0), dtype=np.float64)
    half = chains.shape[1] // 2
    if half < 2:
        return np.empty((0, 0), dtype=np.float64)
    return np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based stable average ranks, preserving rejection ties."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    ranks = np.empty(flat.size, dtype=np.float64)
    starts = np.concatenate(
        ([0], np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1)
    )
    ends = np.concatenate((starts[1:], [flat.size]))
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
    return ranks


def _standard_normal_ppf(probabilities: np.ndarray) -> np.ndarray:
    """Vectorized inverse standard-normal CDF using Acklam's approximation."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError("normal inverse-CDF probabilities must lie in (0, 1)")

    coefficient_a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    coefficient_b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    coefficient_c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    coefficient_d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    lower_cutoff = 0.02425
    upper_cutoff = 1.0 - lower_cutoff
    result = np.empty_like(probabilities)
    lower = probabilities < lower_cutoff
    central = (probabilities >= lower_cutoff) & (
        probabilities <= upper_cutoff
    )
    upper = probabilities > upper_cutoff

    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(probabilities[lower]))
        numerator = (
            ((((coefficient_c[0] * q + coefficient_c[1]) * q
               + coefficient_c[2]) * q + coefficient_c[3]) * q
             + coefficient_c[4]) * q + coefficient_c[5]
        )
        denominator = (
            (((coefficient_d[0] * q + coefficient_d[1]) * q
              + coefficient_d[2]) * q + coefficient_d[3]) * q + 1.0
        )
        result[lower] = numerator / denominator

    if np.any(central):
        q = probabilities[central] - 0.5
        r = q * q
        numerator = (
            (((((coefficient_a[0] * r + coefficient_a[1]) * r
                + coefficient_a[2]) * r + coefficient_a[3]) * r
              + coefficient_a[4]) * r + coefficient_a[5]) * q
        )
        denominator = (
            ((((coefficient_b[0] * r + coefficient_b[1]) * r
               + coefficient_b[2]) * r + coefficient_b[3]) * r
             + coefficient_b[4]) * r + 1.0
        )
        result[central] = numerator / denominator

    if np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - probabilities[upper]))
        numerator = (
            ((((coefficient_c[0] * q + coefficient_c[1]) * q
               + coefficient_c[2]) * q + coefficient_c[3]) * q
             + coefficient_c[4]) * q + coefficient_c[5]
        )
        denominator = (
            (((coefficient_d[0] * q + coefficient_d[1]) * q
              + coefficient_d[2]) * q + coefficient_d[3]) * q + 1.0
        )
        result[upper] = -(numerator / denominator)

    return result


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Apply Blom rank normalization with average ranks for ties."""

    values = np.asarray(values, dtype=np.float64)
    ranks = _average_ranks(values)
    probabilities = (ranks - 3.0 / 8.0) / (ranks.size + 1.0 / 4.0)
    return _standard_normal_ppf(probabilities).reshape(values.shape)


def _rank_normalized_split_folded_rhat(chains: np.ndarray) -> RhatResult:
    """Return rank-normalized split and folded rank-normalized split R-hat."""

    split = _split_chains(chains)
    if split.size == 0:
        return RhatResult(math.nan, math.nan, math.nan, "undefined_split")
    if not np.all(np.isfinite(split)):
        return RhatResult(math.nan, math.nan, math.nan, "undefined_nonfinite")

    rank_rhat = _basic_rhat(_rank_normalize(split))
    folded = np.abs(split - np.median(split))
    folded_rhat = _basic_rhat(_rank_normalize(folded))
    if math.isnan(rank_rhat) or math.isnan(folded_rhat):
        return RhatResult(
            rank_rhat,
            folded_rhat,
            math.nan,
            "undefined_zero_or_invalid_variance",
        )
    status = "ok_infinite" if math.isinf(rank_rhat) or math.isinf(folded_rhat) else "ok"
    return RhatResult(rank_rhat, folded_rhat, max(rank_rhat, folded_rhat), status)


def _observable_specification(
    target_variances: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    target_variances = np.asarray(target_variances, dtype=np.float64)
    if target_variances.ndim != 1 or target_variances.size == 0:
        raise ValueError("target_variances must be a nonempty vector")
    names: list[str] = []
    exact_means: list[float] = []
    exact_sds: list[float] = []
    for coordinate in range(target_variances.size):
        names.append(f"q{coordinate + 1}")
        exact_means.append(0.0)
        exact_sds.append(math.sqrt(float(target_variances[coordinate])))
    for first in range(target_variances.size):
        for second in range(first, target_variances.size):
            if first == second:
                names.append(f"q{first + 1}^2")
                exact_means.append(float(target_variances[first]))
                exact_sds.append(math.sqrt(2.0) * float(target_variances[first]))
            else:
                names.append(f"q{first + 1}*q{second + 1}")
                exact_means.append(0.0)
                exact_sds.append(
                    math.sqrt(float(target_variances[first]))
                    * math.sqrt(float(target_variances[second]))
                )
    return (
        tuple(names),
        np.asarray(exact_means, dtype=np.float64),
        np.asarray(exact_sds, dtype=np.float64),
    )


def _physical_observables(samples: np.ndarray, dimension: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != dimension:
        raise ValueError(
            f"samples must have shape (chains,draws,{dimension}); got {samples.shape}"
        )
    values: list[np.ndarray] = [
        samples[:, :, index] for index in range(dimension)
    ]
    for first in range(dimension):
        for second in range(first, dimension):
            values.append(samples[:, :, first] * samples[:, :, second])
    return np.stack(values, axis=2)


def _analyze_block(
    *,
    samples: np.ndarray,
    epsilon: float,
    L: int,
    block_index: int,
    seeds: Sequence[int],
    accepted_retained: int,
    retained_gradient_calls: int,
    warmup_nonfinite: int,
    retained_nonfinite: int,
    observable_names: Sequence[str],
    exact_observable_means: np.ndarray,
    exact_observable_sds: np.ndarray,
) -> BlockResult:
    samples = np.asarray(samples, dtype=np.float64)
    total_retained = samples.shape[0] * samples.shape[1]
    acceptance = accepted_retained / total_retained
    gradients_per_draw = retained_gradient_calls / total_retained
    observable_values = _physical_observables(samples, samples.shape[2])
    diagnostics: list[ObservableDiagnostic] = []
    simulation_is_nonfinite = warmup_nonfinite > 0 or retained_nonfinite > 0
    if observable_values.shape[2] != len(observable_names):
        raise RuntimeError("observable values/specification length mismatch")
    for index, name in enumerate(observable_names):
        chains = observable_values[:, :, index]
        iat = _multi_chain_geyer_iat(chains)
        rhat = _rank_normalized_split_folded_rhat(chains)
        empirical_mean = float(np.mean(chains))
        exact_mean = float(exact_observable_means[index])
        raw_error = empirical_mean - exact_mean
        standardized_error = abs(raw_error) / float(exact_observable_sds[index])
        if (
            not simulation_is_nonfinite
            and iat.is_defined
            and gradients_per_draw > 0.0
        ):
            ess = total_retained / iat.tau
            efficiency = 1.0 / (gradients_per_draw * iat.tau)
        else:
            ess = math.nan
            efficiency = math.nan
        diagnostics.append(
            ObservableDiagnostic(
                name=name,
                iat=iat,
                ess=ess,
                efficiency=efficiency,
                rhat=rhat,
                empirical_mean=empirical_mean,
                exact_mean=exact_mean,
                raw_moment_error=raw_error,
                standardized_moment_error=standardized_error,
            )
        )

    if simulation_is_nonfinite:
        status = "undefined_nonfinite_simulation"
    elif not all(item.iat.is_defined for item in diagnostics):
        status = "undefined_required_iat"
    elif not math.isfinite(gradients_per_draw) or gradients_per_draw <= 0.0:
        status = "undefined_gradient_denominator"
    else:
        status = "ok"

    if all(item.iat.is_defined for item in diagnostics):
        worst_index = max(
            range(len(diagnostics)), key=lambda item: diagnostics[item].iat.tau
        )
        worst_observable = diagnostics[worst_index].name
        worst_iat = diagnostics[worst_index].iat.tau
        block_efficiency = (
            diagnostics[worst_index].efficiency if status == "ok" else math.nan
        )
    else:
        undefined = [
            index for index, item in enumerate(diagnostics)
            if not item.iat.is_defined
        ]
        worst_observable = diagnostics[undefined[0]].name if undefined else "undefined"
        worst_iat = math.nan
        block_efficiency = math.nan

    if all(not math.isnan(item.rhat.combined) for item in diagnostics):
        maximum_rhat_index = max(
            range(len(diagnostics)),
            key=lambda item: diagnostics[item].rhat.combined,
        )
        maximum_rhat_observable = diagnostics[maximum_rhat_index].name
        maximum_rhat = diagnostics[maximum_rhat_index].rhat.combined
    else:
        maximum_rhat_observable = "undefined"
        maximum_rhat = math.nan
    maximum_moment_index = max(
        range(len(diagnostics)),
        key=lambda item: diagnostics[item].standardized_moment_error,
    )
    return BlockResult(
        epsilon=float(epsilon),
        L=int(L),
        block_index=int(block_index),
        seeds=tuple(int(seed) for seed in seeds),
        status=status,
        acceptance=acceptance,
        retained_gradient_calls=int(retained_gradient_calls),
        gradients_per_retained_draw=gradients_per_draw,
        warmup_nonfinite_trajectories=int(warmup_nonfinite),
        retained_nonfinite_trajectories=int(retained_nonfinite),
        observable_diagnostics=tuple(diagnostics),
        worst_observable=worst_observable,
        worst_iat=worst_iat,
        block_efficiency=block_efficiency,
        maximum_rhat_observable=maximum_rhat_observable,
        maximum_rhat=maximum_rhat,
        maximum_moment_error_observable=diagnostics[maximum_moment_index].name,
        maximum_standardized_moment_error=(
            diagnostics[maximum_moment_index].standardized_moment_error
        ),
    )


def _aggregate_blocks(block_results: Sequence[BlockResult]) -> BlockAggregate:
    blocks = tuple(block_results)
    if not blocks:
        raise ValueError("at least one block is required")
    efficiencies = np.asarray(
        [block.block_efficiency for block in blocks], dtype=np.float64
    )
    valid = all(block.status == "ok" for block in blocks) and np.all(
        np.isfinite(efficiencies) & (efficiencies > 0.0)
    )
    if valid:
        geometric_mean = float(np.exp(np.mean(np.log(efficiencies))))
        minimum = float(np.min(efficiencies))
        median = float(np.median(efficiencies))
        maximum = float(np.max(efficiencies))
        status = "ok"
    else:
        geometric_mean = minimum = median = maximum = math.nan
        status = "undefined_block_metric"
    rhat_values = np.asarray([block.maximum_rhat for block in blocks])
    maximum_rhat = (
        float(np.max(rhat_values)) if not np.any(np.isnan(rhat_values)) else math.nan
    )
    moment_values = np.asarray(
        [block.maximum_standardized_moment_error for block in blocks]
    )
    maximum_moment_error = (
        float(np.max(moment_values))
        if np.all(np.isfinite(moment_values))
        else math.nan
    )
    total_nonfinite = sum(
        block.warmup_nonfinite_trajectories
        + block.retained_nonfinite_trajectories
        for block in blocks
    )
    return BlockAggregate(
        status=status,
        geometric_mean_efficiency=geometric_mean,
        minimum_block_efficiency=minimum,
        median_block_efficiency=median,
        maximum_block_efficiency=maximum,
        mean_acceptance=float(np.mean([block.acceptance for block in blocks])),
        maximum_rhat=maximum_rhat,
        maximum_standardized_moment_error=maximum_moment_error,
        total_nonfinite_trajectories=int(total_nonfinite),
    )


def _exact_mixture_moments(
    delta: float,
    weights: np.ndarray,
    centers: np.ndarray,
) -> tuple[float, float]:
    mean = float(np.dot(weights, centers))
    second = float(
        delta * delta
        + np.dot(weights, centers * centers)
    )
    return mean, second


def _thermostat_moment_diagnostic(
    samples: np.ndarray,
    delta: float,
    weights: np.ndarray,
    centers: np.ndarray,
) -> MixtureMomentDiagnostic:
    exact_mean, exact_second = _exact_mixture_moments(delta, weights, centers)
    values = np.asarray(samples, dtype=np.float64)
    empirical_mean = float(np.mean(values))
    empirical_second = float(np.mean(values * values))
    return MixtureMomentDiagnostic(
        empirical_mean=empirical_mean,
        exact_mean=exact_mean,
        mean_error=empirical_mean - exact_mean,
        empirical_second_moment=empirical_second,
        exact_second_moment=exact_second,
        second_moment_error=empirical_second - exact_second,
    )


def _aggregate_configuration(
    configuration: Configuration,
    origin: str,
    blocks: Sequence[MixtureBlockResult],
) -> ConfigurationResult:
    wrapped = tuple(blocks)
    aggregate = _aggregate_blocks(tuple(item.physical for item in wrapped))
    return ConfigurationResult(
        configuration=configuration,
        origin=origin,
        status=aggregate.status,
        block_results=wrapped,
        geometric_mean_efficiency=aggregate.geometric_mean_efficiency,
        minimum_block_efficiency=aggregate.minimum_block_efficiency,
        median_block_efficiency=aggregate.median_block_efficiency,
        maximum_block_efficiency=aggregate.maximum_block_efficiency,
        mean_acceptance=aggregate.mean_acceptance,
        maximum_rhat=aggregate.maximum_rhat,
        maximum_standardized_moment_error=(
            aggregate.maximum_standardized_moment_error
        ),
        total_nonfinite_trajectories=aggregate.total_nonfinite_trajectories,
    )



CANONICAL_TUNING_SEED_BLOCKS = ((567753504, 3554327166, 2159929848, 2183746224), (3182901450, 2500054966, 2494512194, 1125703007), (3372379365, 3829013153, 2779258000, 3673553045), (1042366973, 555022780, 2234701741, 38076647), (4147003693, 637552847, 1213094654, 3282446157), (1983886449, 1778341926, 2298613957, 753872443), (1913436284, 580417278, 2885056474, 2371270245), (2021755070, 2190757929, 2004101126, 1738982583), (2993064774, 535236755, 4082694471, 2365720593), (158278125, 3022481575, 1887590644, 2677325223))
CANONICAL_DELTAS = (0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
CANONICAL_Q_VALUES = (0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0, 250.0)
CANONICAL_EPSILONS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.062, 0.064, 0.065, 0.066)
CANONICAL_L_VALUES = (1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 45, 60, 90, 150, 225, 300)
CANONICAL_STANDARDIZED_STARTS = np.asarray(
    [
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, -2.0, 2.0, -2.0, 2.0],
        [-2.0, 2.0, -2.0, 2.0, -2.0],
        [-2.0, -2.0, -2.0, -2.0, -2.0],
    ],
    dtype=np.float64,
)
CANONICAL_OMEGAS = np.asarray(
    [
        1.0,
        2.340347319320716,
        5.477225575051661,
        12.818610191887021,
        30.0,
    ],
    dtype=np.float64,
)

EXPECTED_FIRST_STREAM_SHA256 = '39d70cf94657cce45d02760f6ce2b9e5438382649a2af8beb281dea897127dca'
EXPECTED_GREX_FIRST_STREAM_SHA256 = 'b369498ff054229c57a98f0bddc05b77f312d34e2e672d50afcaf7e692d2a90b'


def _normalize_real_grid(name: str, values: Sequence[float]) -> tuple[float, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable") from error
    normalized: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name}[{index}] must be real")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name}[{index}] must be finite and positive")
        normalized.append(number)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be nonempty and contain no duplicates")
    if tuple(sorted(normalized)) != tuple(normalized):
        raise ValueError(f"{name} must be strictly increasing")
    return tuple(normalized)


def _normalize_integer_grid(name: str, values: Sequence[int]) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable") from error
    normalized: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{name}[{index}] must be an integer")
        number = int(value)
        if number <= 0:
            raise ValueError(f"{name}[{index}] must be positive")
        normalized.append(number)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be nonempty and contain no duplicates")
    if tuple(sorted(normalized)) != tuple(normalized):
        raise ValueError(f"{name} must be strictly increasing")
    return tuple(normalized)


def _require_integer(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _prepare_inputs(inputs: TuningInputs) -> PreparedInputs:
    beta = float(inputs.beta)
    if beta != 1.0:
        raise ValueError("beta is locked to exactly 1.0")
    omegas = np.asarray(inputs.omegas, dtype=np.float64)
    if omegas.shape != (5,) or not np.all(np.isfinite(omegas)) or np.any(omegas <= 0.0):
        raise ValueError("omegas must contain five finite positive values")
    mass_matrix = np.asarray(inputs.mass_matrix, dtype=np.float64)
    if mass_matrix.shape != (5, 5) or not np.all(np.isfinite(mass_matrix)):
        raise ValueError("mass_matrix must be finite 5x5")
    if not np.array_equal(mass_matrix, mass_matrix.T):
        raise ValueError("mass_matrix must be exactly symmetric")
    try:
        mass_cholesky = np.linalg.cholesky(mass_matrix)
        inverse_mass_matrix = np.linalg.inv(mass_matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass_matrix must be positive definite") from error
    standardized_starts = np.asarray(inputs.standardized_starts, dtype=np.float64)
    if standardized_starts.shape != (4, 5) or not np.all(np.isfinite(standardized_starts)):
        raise ValueError("standardized_starts must be finite 4x5")
    # Use exactly the same operation order as the expanded HMC/S-QHMC
    # preparation so the physical starts match bit for bit, not just in theory.
    target_variances = np.ascontiguousarray(1.0 / (beta * (omegas * omegas)))
    physical_starts = np.ascontiguousarray(
        standardized_starts * np.sqrt(target_variances)[None, :]
    )
    names, exact_means, exact_sds = _observable_specification(target_variances)
    if inputs.observable_policy != "ALL_LINEAR_AND_UNIQUE_QUADRATIC":
        raise ValueError("observable policy must be ALL_LINEAR_AND_UNIQUE_QUADRATIC")
    if len(names) != 20:
        raise RuntimeError("the fixed 5D observable set must contain 20 terms")

    mixture_weights = np.asarray(inputs.mixture_weights, dtype=np.float64)
    mixture_centers = np.asarray(inputs.mixture_centers, dtype=np.float64)
    if (
        mixture_weights.ndim != 1
        or mixture_weights.size == 0
        or mixture_centers.shape != mixture_weights.shape
        or not np.all(np.isfinite(mixture_weights))
        or not np.all(np.isfinite(mixture_centers))
        or np.any(mixture_weights <= 0.0)
    ):
        raise ValueError(
            "mixture_weights and mixture_centers must be same-length finite vectors with positive weights"
        )
    mixture_weight_sum = float(np.sum(mixture_weights))
    if not np.isclose(
        mixture_weight_sum, 1.0, rtol=0.0, atol=2e-7
    ):
        raise ValueError("mixture_weights must sum to one")
    # Normalize exactly once.  Close the final categorical interval with the
    # same left-to-right floating-point sum used by the native refresh so that
    # its prepared probabilities end at exactly 1.0.  Every later refresh,
    # Hamiltonian, fingerprint, diagnostic, and report uses this same array.
    mixture_weights = mixture_weights / mixture_weight_sum
    preceding_weight_sum = 0.0
    for component in range(mixture_weights.size - 1):
        preceding_weight_sum += float(mixture_weights[component])
    final_weight = 1.0 - preceding_weight_sum
    if not math.isfinite(final_weight) or final_weight <= 0.0:
        raise ValueError(
            "normalized mixture weights leave no positive final component"
        )
    mixture_weights[-1] = final_weight
    categorical_sum = 0.0
    for component in range(mixture_weights.size):
        categorical_sum += float(mixture_weights[component])
    if categorical_sum != 1.0:
        raise ValueError(
            "normalized mixture weights do not close exactly in categorical order"
        )
    phase_radii = 1.8 / np.exp(mixture_centers)
    if not np.all(np.isfinite(phase_radii)) or np.any(phase_radii <= 0.0):
        raise ValueError("mixture_centers imply invalid phase radii")
    inner_r = _require_integer("inner_r", inputs.inner_r, 1)

    deltas = _normalize_real_grid("delta_values", inputs.delta_values)
    Q_values = _normalize_real_grid("Q_values", inputs.Q_values)
    epsilons = _normalize_real_grid("epsilon_values", inputs.epsilon_values)
    lengths = _normalize_integer_grid("L_values", inputs.L_values)
    try:
        raw_blocks = tuple(tuple(block) for block in inputs.seed_blocks)
    except TypeError as error:
        raise ValueError("seed_blocks must be a nested iterable") from error
    if not raw_blocks or any(len(block) != 4 for block in raw_blocks):
        raise ValueError("at least one four-chain tuning block is required")
    seed_blocks: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for block_index, block in enumerate(raw_blocks):
        normalized_block: list[int] = []
        for chain_index, seed in enumerate(block):
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
                raise ValueError(f"seed {block_index},{chain_index} must be an integer")
            number = int(seed)
            if not 0 <= number <= 2**32 - 1:
                raise ValueError("every seed must fit uint32")
            if number in seen:
                raise ValueError("tuning seeds must be globally unique")
            seen.add(number)
            normalized_block.append(number)
        seed_blocks.append(tuple(normalized_block))

    n_warmup = _require_integer("n_warmup", inputs.n_warmup, 0)
    n_retained = _require_integer("n_retained", inputs.n_retained, 4)
    n_workers = _require_integer("n_workers", inputs.n_workers, 1)

    run_mode = str(inputs.run_mode).strip().upper()
    if run_mode not in {"CANONICAL", "CUSTOM"}:
        raise ValueError("run_mode must be CANONICAL or CUSTOM")
    output_filename = str(inputs.output_filename)
    if not output_filename.lower().endswith(".txt"):
        raise ValueError("output_filename must end in .txt")
    if Path(output_filename).name != output_filename:
        raise ValueError("output_filename must not contain a directory")
    if run_mode == "CUSTOM":
        if not str(inputs.run_label).startswith("CUSTOM:"):
            raise ValueError("CUSTOM run_label must begin with 'CUSTOM:'")
        if output_filename == CANONICAL_OUTPUT_FILENAME:
            raise ValueError("CUSTOM mode cannot use the canonical output filename")

    if run_mode == "CANONICAL":
        canonical_checks = (
            np.array_equal(omegas, CANONICAL_OMEGAS),
            np.array_equal(mass_matrix, np.eye(5)),
            np.array_equal(standardized_starts, CANONICAL_STANDARDIZED_STARTS),
            np.array_equal(mixture_weights, CANONICAL_MIXTURE_WEIGHTS),
            np.array_equal(mixture_centers, CANONICAL_MIXTURE_CENTERS),
            inner_r == CANONICAL_INNER_R,
            deltas == CANONICAL_DELTAS,
            Q_values == CANONICAL_Q_VALUES,
            epsilons == CANONICAL_EPSILONS,
            lengths == CANONICAL_L_VALUES,
            tuple(seed_blocks) == CANONICAL_TUNING_SEED_BLOCKS,
            n_warmup == 1000,
            n_retained == 10000,
            inputs.run_label == "PRODUCTION TUNING",
            bool(inputs.run_self_tests),
            output_filename == CANONICAL_OUTPUT_FILENAME,
        )
        if not all(canonical_checks):
            raise RuntimeError("CANONICAL inputs differ from the frozen production design")

    fingerprint_payload = {
        "protocol": PROTOCOL_VERSION,
        "implementation": IMPLEMENTATION_VERSION,
        "beta": beta,
        "omegas": omegas.tolist(),
        "mass_matrix": mass_matrix.tolist(),
        "standardized_starts": standardized_starts.tolist(),
        "observable_policy": inputs.observable_policy,
        "delta_values": deltas,
        "Q_values": Q_values,
        "epsilon_values": epsilons,
        "L_values": lengths,
        "seed_blocks": seed_blocks,
        "lengths": [n_warmup, n_retained],
        "search": "full Cartesian; no screening; no automatic extension",
        "inner_r": inner_r,
        "weights": mixture_weights.tolist(),
        "centers": mixture_centers.tolist(),
        "derived_phase_radii": phase_radii.tolist(),
    }
    encoded = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    input_fingerprint = hashlib.sha256(encoded).hexdigest()
    return PreparedInputs(
        beta=beta,
        omegas=omegas,
        omega_squared=omegas * omegas,
        target_variances=target_variances,
        mass_matrix=mass_matrix,
        inverse_mass_matrix=inverse_mass_matrix,
        mass_cholesky=mass_cholesky,
        mass_is_identity=bool(np.array_equal(mass_matrix, np.eye(5))),
        standardized_starts=standardized_starts,
        physical_starts=physical_starts,
        mixture_weights=mixture_weights,
        mixture_centers=mixture_centers,
        phase_radii=phase_radii,
        inner_r=inner_r,
        dimension=5,
        n_chains=4,
        observable_names=names,
        exact_observable_means=exact_means,
        exact_observable_sds=exact_sds,
        delta_values=deltas,
        Q_values=Q_values,
        epsilon_values=epsilons,
        L_values=lengths,
        seed_blocks=tuple(seed_blocks),
        n_warmup=n_warmup,
        n_retained=n_retained,
        n_workers=n_workers,
        run_mode=run_mode,
        run_label=str(inputs.run_label),
        output_directory=inputs.output_directory,
        output_filename=output_filename,
        allow_overwrite=bool(inputs.allow_overwrite),
        print_progress=bool(inputs.print_progress),
        print_report=bool(inputs.print_report),
        run_self_tests=bool(inputs.run_self_tests),
        input_fingerprint=input_fingerprint,
    )



RandomStreams = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _make_random_streams(
    seed: int,
    total_iterations: int,
    dimension: int,
) -> RandomStreams:
    """Create the five fixed PCG64 substreams under the parity contract."""

    root = np.random.SeedSequence(int(seed))
    children = root.spawn(5)
    physical_rng = np.random.Generator(np.random.PCG64(children[0]))
    thermostat_position_rng = np.random.Generator(np.random.PCG64(children[1]))
    acceptance_rng = np.random.Generator(np.random.PCG64(children[2]))
    mixture_rng = np.random.Generator(np.random.PCG64(children[3]))
    thermostat_momentum_rng = np.random.Generator(np.random.PCG64(children[4]))
    return (
        physical_rng.standard_normal((total_iterations, dimension)).astype(
            np.float64, copy=False
        ),
        thermostat_momentum_rng.standard_normal(total_iterations).astype(
            np.float64, copy=False
        ),
        mixture_rng.random(total_iterations).astype(np.float64, copy=False),
        thermostat_position_rng.standard_normal(total_iterations).astype(
            np.float64, copy=False
        ),
        acceptance_rng.random(total_iterations).astype(np.float64, copy=False),
    )


def _stream_sha256(streams: RandomStreams) -> str:
    digest = hashlib.sha256()
    for array in streams:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _stream_cache(prepared: PreparedInputs) -> tuple[tuple[RandomStreams, ...], ...]:
    total = prepared.n_warmup + prepared.n_retained
    return tuple(
        tuple(
            _make_random_streams(seed, total, prepared.dimension)
            for seed in block
        )
        for block in prepared.seed_blocks
    )


def _slice_streams(streams: RandomStreams, total: int) -> RandomStreams:
    return tuple(array[:total] for array in streams)  # type: ignore[return-value]


def _run_chain_job(
    streams: RandomStreams,
    q_initial: np.ndarray,
    prepared: PreparedInputs,
    configuration: Configuration,
    n_warmup: int,
    n_retained: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int, int]:
    return _simulate_chain(
        *streams,
        q_initial,
        prepared.beta,
        configuration.delta,
        configuration.Q,
        prepared.inner_r,
        configuration.epsilon,
        configuration.L,
        n_warmup,
        n_retained,
        prepared.mixture_weights,
        prepared.mixture_centers,
        prepared.omega_squared,
        prepared.mass_cholesky,
        prepared.inverse_mass_matrix,
    )


def _run_configuration_block(
    *,
    configuration: Configuration,
    origin: str,
    block_index: int,
    prepared: PreparedInputs,
    cache: tuple[tuple[RandomStreams, ...], ...],
    n_warmup: int,
    n_retained: int,
) -> MixtureBlockResult:
    total = n_warmup + n_retained
    jobs = [
        (
            _slice_streams(cache[block_index][chain_index], total),
            prepared.physical_starts[chain_index],
            prepared,
            configuration,
            n_warmup,
            n_retained,
        )
        for chain_index in range(prepared.n_chains)
    ]
    if prepared.n_workers > 1:
        with ThreadPoolExecutor(
            max_workers=min(prepared.n_workers, prepared.n_chains)
        ) as executor:
            outputs = list(executor.map(lambda arguments: _run_chain_job(*arguments), jobs))
    else:
        outputs = [_run_chain_job(*arguments) for arguments in jobs]
    samples = np.stack([output[0] for output in outputs], axis=0)
    post_mh_s = np.stack([output[1] for output in outputs], axis=0)
    refreshed_s = np.stack([output[2] for output in outputs], axis=0)
    physical = _analyze_block(
        samples=samples,
        epsilon=configuration.epsilon,
        L=configuration.L,
        block_index=block_index,
        seeds=prepared.seed_blocks[block_index],
        accepted_retained=sum(output[3] for output in outputs),
        retained_gradient_calls=sum(output[7] for output in outputs),
        warmup_nonfinite=sum(output[4] for output in outputs),
        retained_nonfinite=sum(output[5] for output in outputs),
        observable_names=prepared.observable_names,
        exact_observable_means=prepared.exact_observable_means,
        exact_observable_sds=prepared.exact_observable_sds,
    )
    return MixtureBlockResult(
        physical=physical,
        post_mh_thermostat=_thermostat_moment_diagnostic(
            post_mh_s,
            configuration.delta,
            prepared.mixture_weights,
            prepared.mixture_centers,
        ),
        refreshed_thermostat=_thermostat_moment_diagnostic(
            refreshed_s,
            configuration.delta,
            prepared.mixture_weights,
            prepared.mixture_centers,
        ),
    )


def _evaluate_configuration(
    configuration: Configuration,
    origin: str,
    prepared: PreparedInputs,
    cache: tuple[tuple[RandomStreams, ...], ...],
    n_warmup: int,
    n_retained: int,
) -> ConfigurationResult:
    blocks = tuple(
        _run_configuration_block(
            configuration=configuration,
            origin=origin,
            block_index=block_index,
            prepared=prepared,
            cache=cache,
            n_warmup=n_warmup,
            n_retained=n_retained,
        )
        for block_index in range(len(prepared.seed_blocks))
    )
    return _aggregate_configuration(configuration, origin, blocks)


def _defined_rank_key(result: ConfigurationResult) -> tuple[float, ...]:
    """Largest GM, then manuscript minimum-score and gradient-cost tie rules."""

    if result.status != "ok":
        raise ValueError("undefined-score configuration is not rankable")
    item = result.configuration
    return (
        result.geometric_mean_efficiency,
        result.minimum_block_efficiency,
        -result.block_results[0].physical.gradients_per_retained_draw,
        -item.epsilon,
        -float(item.L),
        -item.delta,
        -item.Q,
    )


def _select_best(results: Sequence[ConfigurationResult]) -> ConfigurationResult:
    defined = [result for result in results if result.status == "ok"]
    if not defined:
        raise RuntimeError("no configuration has a mathematically defined score")
    return max(defined, key=_defined_rank_key)








def _base_configurations(prepared: PreparedInputs) -> tuple[Configuration, ...]:
    return tuple(
        Configuration(delta, Q, epsilon, L)
        for epsilon in prepared.epsilon_values
        for L in prepared.L_values
        for delta in prepared.delta_values
        for Q in prepared.Q_values
    )









def _B_invariant(
    p: np.ndarray,
    s: float,
    p_s: float,
    Q: float,
    inverse_mass_matrix: np.ndarray,
) -> float:
    return p_s * p_s + Q * math.exp(-2.0 * s) * float(
        p @ inverse_mass_matrix @ p
    )


def _production_self_tests(prepared: PreparedInputs) -> list[str]:
    messages: list[str] = []
    base_count = len(_base_configurations(prepared))
    if prepared.run_mode == "CANONICAL" and base_count != 156492:
        raise RuntimeError("canonical full-grid count seal failed")
    messages.append(f"PASS: full Cartesian grid with {base_count} settings; no screening")
    for delta in (0.02, 0.06, 0.10):
        for s in (-0.05, 0.15, 0.65, 0.80):
            log_rho, score = _log_rho_and_score(
                s, delta, prepared.mixture_weights, prepared.mixture_centers
            )
            step = 1.0e-6
            numerical = (
                _log_rho_and_score(
                    s + step,
                    delta,
                    prepared.mixture_weights,
                    prepared.mixture_centers,
                )[0]
                - _log_rho_and_score(
                    s - step,
                    delta,
                    prepared.mixture_weights,
                    prepared.mixture_centers,
                )[0]
            ) / (2.0 * step)
            if not math.isfinite(log_rho) or not math.isclose(
                score, numerical, rel_tol=2e-7, abs_tol=2e-7
            ):
                raise RuntimeError("stable mixture score finite-difference test failed")
    messages.append("PASS: fixed mixture log density and analytic score")

    identity = np.eye(5, dtype=np.float64)
    q = np.asarray([0.2, -0.1, 0.05, -0.02, 0.01], dtype=np.float64)
    p = np.asarray([0.3, -0.4, 0.2, 0.1, -0.25], dtype=np.float64)
    zero_p = np.zeros(5, dtype=np.float64)
    zero_duration = exact_loglinear_B_flow_stable(
        q, p, 0.31, -0.22, 1.0, 0.0, identity
    )
    if not zero_duration[3] or not np.array_equal(zero_duration[0], q):
        raise RuntimeError("exact-B zero-duration edge test failed")
    zero_momentum = exact_loglinear_B_flow_stable(
        q, zero_p, 0.31, -0.22, 2.0, 7.5, identity
    )
    expected_s = 0.31 + 7.5 * (-0.22 / 2.0)
    if (
        not zero_momentum[3]
        or not np.array_equal(zero_momentum[0], q)
        or zero_momentum[2] != -0.22
        or not math.isclose(zero_momentum[1], expected_s, rel_tol=0.0, abs_tol=2e-15)
    ):
        raise RuntimeError("exact-B zero-physical-momentum branch failed")
    huge_translation = exact_loglinear_B_flow_stable(
        q, zero_p, 0.0, 1e-100, 1.0, 1e100, identity
    )
    if not huge_translation[3] or not math.isclose(
        huge_translation[1], 1.0, rel_tol=0.0, abs_tol=2e-15
    ):
        raise RuntimeError("exact-B large signed-time translation failed")

    # These native edge cases specifically rule out the old |tau|>350
    # rejection and loss of the exponentially small turnaround branch.
    q1 = np.asarray([0.0], dtype=np.float64)
    identity1 = np.eye(1, dtype=np.float64)
    p_large_tau = np.asarray([math.sqrt(0.75)], dtype=np.float64)
    positive_500 = exact_loglinear_B_flow_stable(
        q1, p_large_tau, 0.0, 0.5, 1.0, 500.0, identity1
    )
    negative_500 = exact_loglinear_B_flow_stable(
        q1, p_large_tau, 0.0, 0.5, 1.0, -500.0, identity1
    )
    if (
        not positive_500[3]
        or not negative_500[3]
        or not np.all(np.isfinite(positive_500[0]))
        or not np.all(np.isfinite(negative_500[0]))
        or not math.isfinite(positive_500[1])
        or not math.isfinite(negative_500[1])
        or not math.isclose(abs(positive_500[2]), 1.0, rel_tol=2e-14, abs_tol=2e-14)
        or not math.isclose(abs(negative_500[2]), 1.0, rel_tol=2e-14, abs_tol=2e-14)
    ):
        raise RuntimeError("exact-B signed |tau|=500 edge test failed")

    tiny_p = np.asarray([1.0e-20], dtype=np.float64)
    tiny_forward = exact_loglinear_B_flow_stable(
        q1, tiny_p, 0.0, -1.0, 1.0, 100.0, identity1
    )
    tiny_reverse = exact_loglinear_B_flow_stable(
        q1, tiny_p, 0.0, 1.0, 1.0, -100.0, identity1
    )
    if (
        not tiny_forward[3]
        or not tiny_reverse[3]
        or tiny_forward[2] <= 0.0
        or tiny_reverse[2] >= 0.0
        or not np.all(np.isfinite(tiny_forward[0]))
        or not np.all(np.isfinite(tiny_reverse[0]))
        or not math.isfinite(tiny_forward[1])
        or not math.isfinite(tiny_reverse[1])
    ):
        raise RuntimeError("exact-B tiny-branch turnaround edge test failed")

    forward_B = exact_loglinear_B_flow_stable(
        q, p, 0.31, -0.22, 1.0, 0.037, identity
    )
    reverse_B = exact_loglinear_B_flow_stable(
        forward_B[0], p, forward_B[1], forward_B[2], 1.0, -0.037, identity
    )
    reverse_B_error = max(
        float(np.max(np.abs(reverse_B[0] - q))),
        abs(reverse_B[1] - 0.31),
        abs(reverse_B[2] + 0.22),
    )
    invariant_before = _B_invariant(p, 0.31, -0.22, 1.0, identity)
    invariant_after = _B_invariant(
        p, forward_B[1], forward_B[2], 1.0, identity
    )
    if (
        not forward_B[3]
        or not reverse_B[3]
        or reverse_B_error > 2e-12
        or not math.isclose(
            invariant_before, invariant_after, rel_tol=2e-12, abs_tol=2e-14
        )
    ):
        raise RuntimeError("stable exact-B reversal/invariant test failed")
    messages.append(
        "PASS: stable exact-B zero, signed-|tau|=500, tiny-turnaround, "
        f"invariant, and reversal tests ({reverse_B_error:.3e})"
    )

    gradient = _target_gradient(q, prepared.omega_squared)
    forward = _exactB_inner_proposal(
        q, p, 0.31, -0.22, gradient, 1.0, 0.06, 1.0,
        prepared.inner_r, 0.001, 7,
        prepared.mixture_weights, prepared.mixture_centers,
        prepared.omega_squared, prepared.inverse_mass_matrix,
    )
    reverse = _exactB_inner_proposal(
        forward[0], forward[1], forward[2], forward[3], forward[4],
        1.0, 0.06, 1.0, prepared.inner_r, -0.001, 7,
        prepared.mixture_weights, prepared.mixture_centers, prepared.omega_squared,
        prepared.inverse_mass_matrix,
    )
    reversal_error = max(
        float(np.max(np.abs(reverse[0] - q))),
        float(np.max(np.abs(reverse[1] - p))),
        abs(reverse[2] - 0.31),
        abs(reverse[3] + 0.22),
    )
    if (
        not forward[6] or not reverse[6]
        or forward[5] != 7 or reverse[5] != 7
        or reversal_error > 3e-10
    ):
        raise RuntimeError(f"full symmetric proposal reversal failed: {reversal_error:.3e}")
    messages.append(
        f"PASS: r={prepared.inner_r} W/B-subcycled proposal reversal={reversal_error:.3e}"
    )

    streams = _make_random_streams(prepared.seed_blocks[0][0], 12, 5)
    repeated = _make_random_streams(prepared.seed_blocks[0][0], 12, 5)
    if any(not np.array_equal(first, second) for first, second in zip(streams, repeated)):
        raise RuntimeError("random substream repeatability failed")
    smoke = _simulate_chain(
        *streams,
        prepared.physical_starts[0],
        1.0,
        0.06,
        1.0,
        prepared.inner_r,
        0.001,
        3,
        4,
        8,
        prepared.mixture_weights,
        prepared.mixture_centers,
        prepared.omega_squared,
        prepared.mass_cholesky,
        prepared.inverse_mass_matrix,
    )
    if (
        smoke[4] != 0 or smoke[5] != 0
        or smoke[6] != 13 or smoke[7] != 24
        or not np.all(np.isfinite(smoke[0]))
    ):
        raise RuntimeError("finite native smoke chain or L-gradient accounting failed")
    messages.append("PASS: exact auxiliary refresh, finite native smoke chain, and L-cost")

    digest_streams = _make_random_streams(
        prepared.seed_blocks[0][0],
        prepared.n_warmup + prepared.n_retained,
        prepared.dimension,
    )
    stream_digest = _stream_sha256(digest_streams)
    if (
        prepared.run_mode == "CANONICAL"
        and EXPECTED_FIRST_STREAM_SHA256
        and stream_digest not in (
            EXPECTED_FIRST_STREAM_SHA256, EXPECTED_GREX_FIRST_STREAM_SHA256
        )
    ):
        raise RuntimeError(
            "first tuning stream SHA-256 differs from the sealed portable value"
        )
    messages.append(f"PASS: common five-substream SHA-256={stream_digest}")
    return messages



def _format_number(value: float, specification: str = ".8g") -> str:
    return format(value, specification) if math.isfinite(value) else str(value)


def _format_result(result: ConfigurationResult) -> str:
    item = result.configuration
    block_efficiencies = [
        block.physical.block_efficiency for block in result.block_results
    ]
    block_iats = [block.physical.worst_iat for block in result.block_results]
    block_observables = [
        block.physical.worst_observable for block in result.block_results
    ]
    worst_iat_diagnostics = []
    for wrapped in result.block_results:
        physical = wrapped.physical
        diagnostic = next(
            item
            for item in physical.observable_diagnostics
            if item.name == physical.worst_observable
        )
        worst_iat_diagnostics.append(diagnostic.iat)
    block_pair_counts = [item.pair_count for item in worst_iat_diagnostics]
    block_last_lags = [item.last_included_lag for item in worst_iat_diagnostics]
    block_first_excluded = [
        item.first_excluded_lag for item in worst_iat_diagnostics
    ]
    block_cutoff_reasons = [item.cutoff_reason for item in worst_iat_diagnostics]
    block_gradients = [
        wrapped.physical.gradients_per_retained_draw
        for wrapped in result.block_results
    ]
    return (
        f"origin={result.origin}, delta={item.delta:.6g}, Q={item.Q:.6g}, "
        f"epsilon={item.epsilon:.6g}, L={item.L}, status={result.status}, "
        f"GM_eff={_format_number(result.geometric_mean_efficiency, '.12g')}, "
        f"min_eff={_format_number(result.minimum_block_efficiency, '.12g')}, "
        f"acceptance={_format_number(result.mean_acceptance)}, "
        f"max_Rhat={_format_number(result.maximum_rhat)}, "
        f"max_moment_error={_format_number(result.maximum_standardized_moment_error)}, "
        f"nonfinite={result.total_nonfinite_trajectories}, "
        f"block_eff={block_efficiencies}, block_worst_IAT={block_iats}, "
        f"block_worst_observable={block_observables}, "
        f"block_actual_gradients_per_draw={block_gradients}, "
        f"block_Geyer_pair_count={block_pair_counts}, "
        f"block_Geyer_last_lag={block_last_lags}, "
        f"block_Geyer_first_excluded={block_first_excluded}, "
        f"block_Geyer_cutoff_reason={block_cutoff_reasons}"
    )


def _evaluate_many(
    configurations: Sequence[Configuration],
    *,
    origin: str,
    stage_label: str,
    prepared: PreparedInputs,
    cache: tuple[tuple[RandomStreams, ...], ...],
    n_warmup: int,
    n_retained: int,
) -> tuple[ConfigurationResult, ...]:
    configs = tuple(configurations)
    results: list[ConfigurationResult] = []
    for index, configuration in enumerate(configs, start=1):
        result = _evaluate_configuration(
            configuration,
            origin,
            prepared,
            cache,
            n_warmup,
            n_retained,
        )
        results.append(result)
        if prepared.print_progress:
            print(
                f"[{stage_label} {index}/{len(configs)}] {_format_result(result)}",
                flush=True,
            )
    return tuple(results)


def _runtime_summary() -> str:
    return (
        f"platform={platform.platform()}; Python={platform.python_version()}; "
        f"NumPy={np.__version__}; Numba={numba.__version__}"
    )


def _resolve_output_path(prepared: PreparedInputs) -> Path:
    if prepared.output_directory is not None:
        directory = Path(prepared.output_directory).expanduser().resolve()
    else:
        script_name = globals().get("__file__")
        directory = (
            Path(script_name).resolve().parent
            if isinstance(script_name, str) and script_name
            else Path.cwd().resolve()
        )
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {directory}")
    return directory / prepared.output_filename

















def _build_full_report(configuration_results, representative, prepared, self_test_messages):
    """Full-grid TXT report. No short-run rows are mixed into the score."""
    from dataclasses import asdict
    import platform
    lines = [
        'REFRESHED-MIXTURE THC-v3: EXPANDED 5D GAUSSIAN TUNING',
        '=' * 88,
        f'Protocol: {PROTOCOL_VERSION}',
        f'Run mode: {prepared.run_mode}; run label: {prepared.run_label}',
        f'Input fingerprint: {prepared.input_fingerprint}',
        f'Python: {platform.python_version()}; NumPy: {np.__version__}',
        'h(s)=exp(s); refresh s,p,p_s at every trajectory; full-state Metropolis correction.',
        'H=U+exp(-2*s)*p.T*inv(M)*p/2+d*s/beta+p_s**2/(2*Q)-log(rho_delta(s))/beta.',
        f'Fixed mixture weights: {prepared.mixture_weights.tolist()}',
        f'Fixed mixture centers: {prepared.mixture_centers.tolist()}',
        f'Inner exact B/W substeps: {prepared.inner_r}; no target gradient in these substeps.',
        f'beta={prepared.beta}; omega={prepared.omegas.tolist()}',
        f'Base mass: {prepared.mass_matrix.tolist()}',
        f'Standardized starts: {prepared.standardized_starts.tolist()}',
        f'Physical starts: {prepared.physical_starts.tolist()}',
        f'Observable set: {prepared.observable_names}',
        f'Seed blocks: {prepared.seed_blocks}',
        f'Blocks={len(prepared.seed_blocks)}; chains/block={prepared.n_chains}; warmup={prepared.n_warmup}; retained={prepared.n_retained}',
        f'delta grid: {prepared.delta_values}', f'Q grid: {prepared.Q_values}',
        f'epsilon grid: {prepared.epsilon_values}', f'L grid: {prepared.L_values}',
        f'Tested configurations: {len(configuration_results)}',
        f'Defined configurations: {sum(r.status == "ok" for r in configuration_results)}',
        'Every configuration uses every block at the full lengths; no screening or automatic grid expansion.',
        'These are tuning data, not a held-out final comparison.',
        'IAT: all four chains together; biased autocovariances (denominator N at every lag);',
        'Geyer initial-positive pairs, cumulative minima; no IAT floor or manual lag cutoff.',
        'R-hat: maximum of rank-normalized split and folded split values; descriptive, not a gate.',
        'Block efficiency: 1/(actual retained target gradients per draw * worst physical-observable IAT).',
        'Score: geometric mean of all block efficiencies. Undefined IAT or a nonfinite trajectory makes the score undefined.',
        'Retained cost excludes warmup; cached gradients are not counted twice. Valid full trajectories cost L new gradients.',
        'Selection: largest score; exact ties use larger minimum block efficiency, lower gradient cost, then smaller epsilon, L, delta, Q.',
        'The number of grid settings differs between methods; this is not equal total tuning compute.',
        '', 'SELF TESTS', *self_test_messages, '', 'ALL CONFIGURATION SUMMARIES', '-'*88,
    ]
    lines.extend(_format_result(r) for r in configuration_results)
    lines.extend(['', 'SELECTED FIXED REPRESENTATIVE', '-'*88])
    if representative is None:
        lines.append('None: no configuration has a defined full-block score. No result was silently removed.')
    else:
        lines.append(_format_result(representative))
        for key, grid in [('delta',prepared.delta_values),('Q',prepared.Q_values),('epsilon',prepared.epsilon_values),('L',prepared.L_values)]:
            value=getattr(representative.configuration,key)
            if value == grid[0] or value == grid[-1]:
                lines.append(f'Grid boundary: {key}={value}; best tested setting, not an established optimum.')
        lines.extend(['', 'SELECTED REPRESENTATIVE: ALL BLOCKS, OBSERVABLES, MOMENTS AND CUTOFFS'])
        for wrapped in representative.block_results:
            # Compact labelled records retain every reported number and status.
            lines.append(json.dumps(asdict(wrapped), sort_keys=True, allow_nan=True))
    lines.extend(['', 'END OF REPORT'])
    return '\n'.join(lines)


def run_mixture_thc_tuning(inputs: TuningInputs, *, preflight_only: bool = False):
    prepared = _prepare_inputs(inputs)
    self_tests = (_production_self_tests(prepared)
                  if preflight_only or prepared.run_self_tests or prepared.run_mode == 'CANONICAL'
                  else ['SKIPPED: CUSTOM self-tests disabled'])
    configurations = _base_configurations(prepared)
    if prepared.print_progress or preflight_only:
        print(*self_tests, sep='\n', flush=True)
        print(f'FULL GRID: {len(configurations)} configurations; {len(prepared.seed_blocks)} blocks; '
              f'{prepared.n_chains} chains/block; {prepared.n_warmup} warmup + {prepared.n_retained} retained.', flush=True)
    if preflight_only:
        print('PREFLIGHT COMPLETE. Full tuning has NOT run. Set GREX_PREFLIGHT_ONLY=False to run directly, or use the checkpoint launcher.', flush=True)
        return {'preflight_only':True, 'prepared_inputs':prepared,
                'self_test_messages':self_tests, 'output_path':None,
                'input_fingerprint':prepared.input_fingerprint}
    destination = _resolve_output_path(prepared)
    cache = _stream_cache(prepared)
    results = _evaluate_many(configurations, origin='EXPANDED_FULL_GRID', stage_label='FULL',
                             prepared=prepared, cache=cache,
                             n_warmup=prepared.n_warmup, n_retained=prepared.n_retained)
    representative = _select_best(results) if any(r.status == 'ok' for r in results) else None
    report = _build_full_report(results, representative, prepared, self_tests)
    destination.write_text(report+'\n', encoding='utf-8')
    if prepared.print_report:
        print(report, flush=True)
        print(f'Report saved to: {destination}', flush=True)
    return {'configuration_results':results, 'representative':representative,
            'report':report, 'output_path':destination, 'prepared_inputs':prepared}



# Tuning inputs.

GREX_PREFLIGHT_ONLY = True

USER_INPUTS = TuningInputs(
    beta=1.0,
    omegas=np.asarray(
        [1.0, 2.340347319320716, 5.477225575051661, 12.818610191887021, 30.0],
        dtype=np.float64,
    ),
    mass_matrix=np.eye(5, dtype=np.float64),
    standardized_starts=np.asarray(
        [
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [2.0, -2.0, 2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0, 2.0, -2.0],
            [-2.0, -2.0, -2.0, -2.0, -2.0],
        ],
        dtype=np.float64,
    ),
    observable_policy="ALL_LINEAR_AND_UNIQUE_QUADRATIC",
    epsilon_values=(
        0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020,
        0.025, 0.030, 0.035, 0.040, 0.045, 0.050,
        0.055, 0.060, 0.062, 0.064, 0.065, 0.066,
    ),
    L_values=(
        1, 2, 3, 5, 8, 10, 12, 15, 20,
        25, 30, 40, 45, 60, 90, 150, 225, 300,
    ),
    seed_blocks=(
        (567753504, 3554327166, 2159929848, 2183746224),
        (3182901450, 2500054966, 2494512194, 1125703007),
        (3372379365, 3829013153, 2779258000, 3673553045),
        (1042366973, 555022780, 2234701741, 38076647),
        (4147003693, 637552847, 1213094654, 3282446157),
        (1983886449, 1778341926, 2298613957, 753872443),
        (1913436284, 580417278, 2885056474, 2371270245),
        (2021755070, 2190757929, 2004101126, 1738982583),
        (2993064774, 535236755, 4082694471, 2365720593),
        (158278125, 3022481575, 1887590644, 2677325223),
    ),
    n_warmup=1000,
    n_retained=10000,
    n_workers=4,
    run_self_tests=True,
    print_progress=True,
    print_report=True,
    run_mode="CANONICAL",
    run_label="PRODUCTION TUNING",
    output_directory=None,
    mixture_weights=(0.5641470, 0.1084317, 0.3274213),
    mixture_centers=(0.7161277448, 0.1776353476, 0.0),
    inner_r=8,
    # Mixture width: dense near the old delta=0.02, plus much wider values.
    delta_values=(
        0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175,
        0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1,
        0.15, 0.2, 0.3, 0.5, 0.75, 1.0,
    ),
    # Thermostat mass: dense near the old Q=2, spanning 0.025 to 250.
    Q_values=(
        0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.4, 0.5, 0.75,
        1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 35.0,
        50.0, 75.0, 100.0, 150.0, 250.0,
    ),
    output_filename='THC_v3_mixture_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt',
    allow_overwrite=False,
)


if __name__ == "__main__":
    run_mixture_thc_tuning(
        USER_INPUTS,
        preflight_only=GREX_PREFLIGHT_ONLY,
    )
