"""Tune slice-NUTS on the six-dimensional Neal's funnel."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import linecache
import math
import numbers
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

NUMBA_DIRECT_RUN = True
NUMBA_CACHE_MODE = "persistent cache disabled; in-memory JIT remains enabled"
NUMBA_CACHE_DIRECTORY = "not used"

try:
    import numba

    if NUMBA_DIRECT_RUN:
        def njit(*args, **kwargs):
            direct_options = dict(kwargs)
            direct_options["cache"] = False
            return numba.njit(*args, **direct_options)
    else:
        numba.config.CACHE_DIR = NUMBA_CACHE_DIRECTORY
        njit = numba.njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - only exercised without Numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """Fallback decorator that leaves functions unchanged."""

        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator



SHARED_PROTOCOL_VERSION = "FUNNEL6-EXPANDED-20260925-strict-proposal-v1"
COMMON_INPUT_SCHEMA_VERSION = "funnel6-fair-playground-NUTS-inputs-v1"
INPUT_PREPARATION_VERSION = "funnel6-validated-derived-NUTS-inputs-v1.0.0"
DIAGNOSTICS_VERSION = "multichain-geyer-ims-rank-rhat-v1.0.0"
COMMON_PIPELINE_VERSION = "funnel6-NUTS-observables-score-v1.0.0"
NUTS_IMPLEMENTATION_VERSION = "hoffman-gelman-2014-slice-NUTS-algorithm6-observational-counters-v1"
RANDOM_STREAM_VERSION = "numba-chain-local-numpy-rng-v1.0.0"
SUPPORTED_OBSERVABLE_POLICY = "ALL_LINEAR_AND_UNIQUE_QUADRATIC"
CANONICAL_OUTPUT_FILENAME = "NUTS_Neals_funnel_tuning_results_GREX_PORTABLE_V1.txt"



@dataclass(frozen=True)
class RunInputs:
    """Every independent scientific and execution input for one run."""

    beta: float
    sigma_y: float
    dimension_x: int
    mass_matrix: Sequence[Sequence[float]]
    funnel_starts: Sequence[Sequence[float]]
    observable_policy: str
    target_acceptance_values: Sequence[float]
    maximum_tree_depth: int
    delta_max: float
    dual_averaging_gamma: float
    dual_averaging_t0: float
    dual_averaging_kappa: float
    reasonable_epsilon_max_search_steps: int
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
    """Validated inputs plus quantities derived from them exactly once."""

    beta: float
    sigma_y: float
    transverse_dimension: int
    target_parameters: np.ndarray
    target_variances: np.ndarray
    mass_matrix: np.ndarray
    inverse_mass_matrix: np.ndarray
    mass_cholesky: np.ndarray
    mass_is_identity: bool
    funnel_starts: np.ndarray
    physical_starts: np.ndarray
    dimension: int
    n_chains: int
    target_acceptance_values: tuple[float, ...]
    maximum_tree_depth: int
    delta_max: float
    dual_averaging_gamma: float
    dual_averaging_t0: float
    dual_averaging_kappa: float
    reasonable_epsilon_max_search_steps: int
    seed_blocks: tuple[tuple[int, ...], ...]
    n_warmup: int
    n_retained: int
    n_workers: int
    observable_policy: str
    observable_names: tuple[str, ...]
    exact_observable_means: np.ndarray
    exact_observable_sds: np.ndarray
    run_mode: str
    run_label: str
    output_directory: str | os.PathLike[str] | None
    output_filename: str
    allow_overwrite: bool
    print_progress: bool
    print_report: bool
    run_self_tests: bool
    common_input_components: dict[str, object]
    common_input_fingerprint: str
    canonical_mismatches: tuple[str, ...]


@dataclass(frozen=True)
class IATResult:
    """One Geyer initial-positive/initial-monotone IAT result."""

    tau: float
    status: str
    pair_count: int
    last_included_lag: int
    first_excluded_lag: int
    cutoff_reason: str

    @property
    def is_defined(self) -> bool:
        return self.status.startswith("ok_") and math.isfinite(self.tau) and self.tau > 0.0


@dataclass(frozen=True)
class RhatResult:
    """Rank-normalized split and folded rank-normalized split R-hat."""

    rank: float
    folded: float
    combined: float
    status: str


@dataclass(frozen=True)
class ObservableDiagnostic:
    """Diagnostics for one required physical observable in one seed block."""

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
    """All diagnostics for one configuration on one multi-chain seed block."""

    target_acceptance: float
    block_index: int
    seeds: tuple[int, ...]
    status: str
    mean_acceptance_statistic: float
    adapted_step_sizes: tuple[float, ...]
    retained_gradient_calls: int
    gradients_per_retained_draw: float
    epsilon_search_gradient_calls: int
    warmup_gradient_calls: int
    retained_divergences: int
    retained_maximum_depth_hits: int
    mean_retained_tree_depth: float
    mean_retained_leapfrog_steps: float
    nonfinite_samples: int
    observable_diagnostics: tuple[ObservableDiagnostic, ...]
    worst_observable: str
    worst_iat: float
    block_efficiency: float
    maximum_rhat_observable: str
    maximum_rhat: float
    maximum_moment_error_observable: str
    maximum_standardized_moment_error: float
    warmup_nonfinite_leaves: int = 0
    retained_nonfinite_leaves: int = 0


@dataclass(frozen=True)
class BlockAggregate:
    """Method-neutral aggregation of fixed seed-block results."""

    status: str
    geometric_mean_efficiency: float
    minimum_block_efficiency: float
    median_block_efficiency: float
    maximum_block_efficiency: float
    mean_acceptance_statistic: float
    mean_retained_gradients_per_draw: float
    total_retained_divergences: int
    total_retained_maximum_depth_hits: int
    maximum_rhat: float
    maximum_standardized_moment_error: float
    total_nonfinite_samples: int


@dataclass(frozen=True)
class ConfigurationResult:
    """Block aggregate used to choose one fixed NUTS representative."""

    target_acceptance: float
    status: str
    block_results: tuple[BlockResult, ...]
    geometric_mean_efficiency: float
    minimum_block_efficiency: float
    median_block_efficiency: float
    maximum_block_efficiency: float
    mean_acceptance_statistic: float
    mean_retained_gradients_per_draw: float
    total_retained_divergences: int
    total_retained_maximum_depth_hits: int
    maximum_rhat: float
    maximum_standardized_moment_error: float
    total_nonfinite_samples: int


# Target, leapfrog, and chain simulation.


@njit(cache=True, inline="always")
def _safe_exp_target(exponent: float) -> float:
    """Return exp(exponent) without platform-dependent overflow errors."""

    if exponent > 709.782712893384:
        return math.inf
    if exponent < -745.1332191019411:
        return 0.0
    return math.exp(exponent)


@njit(cache=True, inline="always")
def _potential_energy(q: np.ndarray, target_parameters: np.ndarray) -> float:
    """Centered Neal-funnel potential for q=(y,x_1,...,x_d)."""

    sigma_y_squared = target_parameters[0]
    transverse_dimension = int(target_parameters[1])
    beta = target_parameters[2]
    y = q[0]
    squared_radius = 0.0
    for coordinate in range(1, q.size):
        squared_radius += q[coordinate] * q[coordinate]
    inverse_scale = _safe_exp_target(-y)
    conditional_energy = (
        0.0 if squared_radius == 0.0 else 0.5 * inverse_scale * squared_radius
    )
    return (
        0.5 * y * y / sigma_y_squared
        + 0.5 * transverse_dimension * y / beta
        + conditional_energy
    )


@njit(cache=True, inline="always")
def _target_gradient(q: np.ndarray, target_parameters: np.ndarray) -> np.ndarray:
    """Exact gradient of the centered Neal-funnel potential."""

    sigma_y_squared = target_parameters[0]
    transverse_dimension = int(target_parameters[1])
    beta = target_parameters[2]
    y = q[0]
    squared_radius = 0.0
    for coordinate in range(1, q.size):
        squared_radius += q[coordinate] * q[coordinate]
    inverse_scale = _safe_exp_target(-y)
    conditional_gradient_y = (
        0.0 if squared_radius == 0.0 else -0.5 * inverse_scale * squared_radius
    )
    gradient = np.empty(q.size, dtype=np.float64)
    gradient[0] = (
        y / sigma_y_squared
        + 0.5 * transverse_dimension / beta
        + conditional_gradient_y
    )
    for coordinate in range(1, q.size):
        value = q[coordinate]
        gradient[coordinate] = 0.0 if value == 0.0 else inverse_scale * value
    return gradient


@njit(cache=True, inline="always")
def _kinetic_energy(
    p: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> float:
    if mass_is_identity:
        value = 0.0
        for coordinate in range(p.size):
            value += 0.5 * p[coordinate] * p[coordinate]
        return value

    value = 0.0
    for row in range(p.size):
        inverse_mass_p = 0.0
        for column in range(p.size):
            inverse_mass_p += inverse_mass_matrix[row, column] * p[column]
        value += 0.5 * p[row] * inverse_mass_p
    return value


@njit(cache=True, inline="always")
def _vector_is_finite(values: np.ndarray) -> bool:
    for value in values:
        if not math.isfinite(value):
            return False
    return True


@njit(cache=True, inline="always")
def _joint_log_density(
    potential: float,
    p: np.ndarray,
    beta: float,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> float:
    return -beta * (
        potential + _kinetic_energy(p, inverse_mass_matrix, mass_is_identity)
    )


@njit(cache=True, nogil=True)
def _leapfrog_one(
    q: np.ndarray,
    p: np.ndarray,
    gradient: np.ndarray,
    signed_epsilon: float,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """One reversible leapfrog step using one new target-gradient call."""

    new_q = q.copy()
    new_p = p.copy()
    for coordinate in range(new_p.size):
        new_p[coordinate] -= 0.5 * signed_epsilon * gradient[coordinate]

    if mass_is_identity:
        for coordinate in range(new_q.size):
            new_q[coordinate] += signed_epsilon * new_p[coordinate]
    else:
        for row in range(new_q.size):
            velocity = 0.0
            for column in range(new_p.size):
                velocity += inverse_mass_matrix[row, column] * new_p[column]
            new_q[row] += signed_epsilon * velocity

    new_gradient = _target_gradient(new_q, target_parameters)
    for coordinate in range(new_p.size):
        new_p[coordinate] -= 0.5 * signed_epsilon * new_gradient[coordinate]
    new_potential = _potential_energy(new_q, target_parameters)
    finite = bool(
        _vector_is_finite(new_q)
        and _vector_is_finite(new_p)
        and _vector_is_finite(new_gradient)
        and math.isfinite(new_potential)
    )
    return new_q, new_p, new_gradient, new_potential, finite


@njit(cache=True, inline="always")
def _has_u_turn(
    q_minus: np.ndarray,
    p_minus: np.ndarray,
    q_plus: np.ndarray,
    p_plus: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> bool:
    """General-mass U-turn rule using endpoint velocities M^{-1}p."""

    left_product = 0.0
    right_product = 0.0
    if mass_is_identity:
        for coordinate in range(q_minus.size):
            displacement = q_plus[coordinate] - q_minus[coordinate]
            left_product += displacement * p_minus[coordinate]
            right_product += displacement * p_plus[coordinate]
    else:
        for row in range(q_minus.size):
            displacement = q_plus[row] - q_minus[row]
            left_velocity = 0.0
            right_velocity = 0.0
            for column in range(q_minus.size):
                left_velocity += inverse_mass_matrix[row, column] * p_minus[column]
                right_velocity += inverse_mass_matrix[row, column] * p_plus[column]
            left_product += displacement * left_velocity
            right_product += displacement * right_velocity
    return bool(left_product < 0.0 or right_product < 0.0)


@njit(cache=True, inline="always")
def _acceptance_statistic(joint_new: float, joint_initial: float) -> float:
    if not math.isfinite(joint_new):
        return 0.0
    log_ratio = joint_new - joint_initial
    if log_ratio >= 0.0:
        return 1.0
    if log_ratio < -745.0:
        return 0.0
    return math.exp(log_ratio)


@njit(cache=True, nogil=True)
def _build_tree(
    q: np.ndarray,
    p: np.ndarray,
    gradient: np.ndarray,
    potential: float,
    log_slice: float,
    direction: int,
    depth: int,
    epsilon: float,
    joint_initial: float,
    beta: float,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
    delta_max: float,
):
    """Recursive memory-efficient BuildTree from original Algorithm 3/6."""

    if depth == 0:
        new_q, new_p, new_gradient, new_potential, finite = _leapfrog_one(
            q,
            p,
            gradient,
            direction * epsilon,
            target_parameters,
            inverse_mass_matrix,
            mass_is_identity,
        )
        joint_new = _joint_log_density(
            new_potential,
            new_p,
            beta,
            inverse_mass_matrix,
            mass_is_identity,
        )
        finite = bool(finite and math.isfinite(joint_new))
        n_valid = int(finite and log_slice <= joint_new)
        continue_tree = bool(finite and log_slice < delta_max + joint_new)
        alpha = _acceptance_statistic(joint_new, joint_initial) if finite else 0.0
        return (
            new_q, new_p, new_gradient, new_potential,
            new_q, new_p, new_gradient, new_potential,
            new_q, new_gradient, new_potential,
            n_valid, continue_tree, alpha, 1, 1, int(not continue_tree), int(not finite),
        )

    first = _build_tree(
        q,
        p,
        gradient,
        potential,
        log_slice,
        direction,
        depth - 1,
        epsilon,
        joint_initial,
        beta,
        target_parameters,
        inverse_mass_matrix,
        mass_is_identity,
        delta_max,
    )
    if not first[12]:
        return first

    if direction == -1:
        second = _build_tree(
            first[0], first[1], first[2], first[3], log_slice,
            direction, depth - 1, epsilon, joint_initial, beta,
            target_parameters, inverse_mass_matrix, mass_is_identity, delta_max,
        )
        combined_q_minus = second[0]
        combined_p_minus = second[1]
        combined_gradient_minus = second[2]
        combined_potential_minus = second[3]
        combined_q_plus = first[4]
        combined_p_plus = first[5]
        combined_gradient_plus = first[6]
        combined_potential_plus = first[7]
    else:
        second = _build_tree(
            first[4], first[5], first[6], first[7], log_slice,
            direction, depth - 1, epsilon, joint_initial, beta,
            target_parameters, inverse_mass_matrix, mass_is_identity, delta_max,
        )
        combined_q_minus = first[0]
        combined_p_minus = first[1]
        combined_gradient_minus = first[2]
        combined_potential_minus = first[3]
        combined_q_plus = second[4]
        combined_p_plus = second[5]
        combined_gradient_plus = second[6]
        combined_potential_plus = second[7]

    total_valid = first[11] + second[11]
    candidate_q = first[8]
    candidate_gradient = first[9]
    candidate_potential = first[10]
    if total_valid > 0 and np.random.random() < second[11] / total_valid:
        candidate_q = second[8]
        candidate_gradient = second[9]
        candidate_potential = second[10]

    continue_tree = bool(
        second[12]
        and not _has_u_turn(
            combined_q_minus,
            combined_p_minus,
            combined_q_plus,
            combined_p_plus,
            inverse_mass_matrix,
            mass_is_identity,
        )
    )
    return (
        combined_q_minus,
        combined_p_minus,
        combined_gradient_minus,
        combined_potential_minus,
        combined_q_plus,
        combined_p_plus,
        combined_gradient_plus,
        combined_potential_plus,
        candidate_q,
        candidate_gradient,
        candidate_potential,
        total_valid,
        continue_tree,
        first[13] + second[13],
        first[14] + second[14],
        first[15] + second[15],
        first[16] + second[16],
        first[17] + second[17],
    )


@njit(cache=True, inline="always")
def _draw_momentum(
    dimension: int,
    beta: float,
    mass_cholesky: np.ndarray,
    mass_is_identity: bool,
) -> np.ndarray:
    scale = 1.0 / math.sqrt(beta)
    standard = np.random.standard_normal(dimension)
    momentum = np.empty(dimension, dtype=np.float64)
    if mass_is_identity:
        for coordinate in range(dimension):
            momentum[coordinate] = scale * standard[coordinate]
    else:
        for row in range(dimension):
            value = 0.0
            for column in range(dimension):
                value += mass_cholesky[row, column] * standard[column]
            momentum[row] = scale * value
    return momentum


@njit(cache=True, nogil=True)
def _find_reasonable_epsilon(
    q: np.ndarray,
    gradient: np.ndarray,
    potential: float,
    beta: float,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
    maximum_search_steps: int,
) -> tuple[float, int, bool]:
    """Algorithm 4; the position is not changed by this search."""

    momentum = _draw_momentum(q.size, beta, mass_cholesky, mass_is_identity)
    joint_initial = _joint_log_density(
        potential, momentum, beta, inverse_mass_matrix, mass_is_identity
    )
    epsilon = 1.0
    gradient_calls = 0
    new_q, new_p, new_gradient, new_potential, finite = _leapfrog_one(
        q,
        momentum,
        gradient,
        epsilon,
        target_parameters,
        inverse_mass_matrix,
        mass_is_identity,
    )
    gradient_calls += 1
    joint_new = _joint_log_density(
        new_potential, new_p, beta, inverse_mass_matrix, mass_is_identity
    )
    log_acceptance = joint_new - joint_initial if finite else -math.inf
    threshold = -math.log(2.0)
    direction = 1 if log_acceptance > threshold else -1

    for _attempt in range(maximum_search_steps):
        should_continue = bool(
            (direction == 1 and log_acceptance > threshold)
            or (direction == -1 and log_acceptance < threshold)
        )
        if not should_continue:
            return epsilon, gradient_calls, True
        epsilon = epsilon * (2.0 if direction == 1 else 0.5)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            return epsilon, gradient_calls, False
        new_q, new_p, new_gradient, new_potential, finite = _leapfrog_one(
            q,
            momentum,
            gradient,
            epsilon,
            target_parameters,
            inverse_mass_matrix,
            mass_is_identity,
        )
        gradient_calls += 1
        joint_new = _joint_log_density(
            new_potential, new_p, beta, inverse_mass_matrix, mass_is_identity
        )
        log_acceptance = joint_new - joint_initial if finite else -math.inf
    # The last allowed rescaling is itself a valid Algorithm 4 trial.  Accept
    # it when it crossed the 0.5 threshold instead of reporting a false limit
    # failure solely because the loop has just ended.
    should_continue = bool(
        (direction == 1 and log_acceptance > threshold)
        or (direction == -1 and log_acceptance < threshold)
    )
    return epsilon, gradient_calls, not should_continue


@njit(cache=True, inline="always")
def _safe_exp_step_size(log_epsilon: float) -> float:
    # This prevents a transient dual-averaging proposal from overflowing.  It
    # does not hide failure: such a huge step produces acceptance statistic 0
    # and is immediately corrected by the next update.
    if log_epsilon > 700.0:
        return math.exp(700.0)
    if log_epsilon < -700.0:
        return math.exp(-700.0)
    return math.exp(log_epsilon)


@njit(cache=True, nogil=True)
def _nuts_iteration(
    q: np.ndarray,
    gradient: np.ndarray,
    potential: float,
    epsilon: float,
    beta: float,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
    maximum_tree_depth: int,
    delta_max: float,
):
    """One original slice-NUTS transition with a predeclared depth cap."""

    initial_momentum = _draw_momentum(
        q.size, beta, mass_cholesky, mass_is_identity
    )
    joint_initial = _joint_log_density(
        potential,
        initial_momentum,
        beta,
        inverse_mass_matrix,
        mass_is_identity,
    )
    log_slice = joint_initial - np.random.exponential(1.0)

    q_minus = q.copy()
    p_minus = initial_momentum.copy()
    gradient_minus = gradient.copy()
    potential_minus = potential
    q_plus = q.copy()
    p_plus = initial_momentum.copy()
    gradient_plus = gradient.copy()
    potential_plus = potential
    candidate_q = q.copy()
    candidate_gradient = gradient.copy()
    candidate_potential = potential
    n_valid = 1
    continue_tree = True
    depth = 0
    total_steps = 0
    total_divergences = 0
    total_nonfinite_leaves = 0
    final_alpha_sum = 0.0
    final_n_alpha = 1

    while continue_tree and depth < maximum_tree_depth:
        direction = -1 if np.random.random() < 0.5 else 1
        if direction == -1:
            subtree = _build_tree(
                q_minus,
                p_minus,
                gradient_minus,
                potential_minus,
                log_slice,
                direction,
                depth,
                epsilon,
                joint_initial,
                beta,
                target_parameters,
                inverse_mass_matrix,
                mass_is_identity,
                delta_max,
            )
            q_minus, p_minus = subtree[0], subtree[1]
            gradient_minus, potential_minus = subtree[2], subtree[3]
        else:
            subtree = _build_tree(
                q_plus,
                p_plus,
                gradient_plus,
                potential_plus,
                log_slice,
                direction,
                depth,
                epsilon,
                joint_initial,
                beta,
                target_parameters,
                inverse_mass_matrix,
                mass_is_identity,
                delta_max,
            )
            q_plus, p_plus = subtree[4], subtree[5]
            gradient_plus, potential_plus = subtree[6], subtree[7]

        # Printed Algorithm 3 uses min(1,n_new/n_old), before incrementing n.
        if subtree[12]:
            replacement_probability = min(1.0, subtree[11] / n_valid)
            if np.random.random() < replacement_probability:
                candidate_q = subtree[8]
                candidate_gradient = subtree[9]
                candidate_potential = subtree[10]

        n_valid += subtree[11]
        # Algorithm 6 adapts with the final doubling only, not all doublings.
        final_alpha_sum = subtree[13]
        final_n_alpha = subtree[14]
        total_steps += subtree[15]
        total_divergences += subtree[16]
        total_nonfinite_leaves += subtree[17]
        continue_tree = bool(
            subtree[12]
            and not _has_u_turn(
                q_minus,
                p_minus,
                q_plus,
                p_plus,
                inverse_mass_matrix,
                mass_is_identity,
            )
        )
        depth += 1

    maximum_depth_hit = int(continue_tree and depth >= maximum_tree_depth)
    acceptance_statistic = final_alpha_sum / final_n_alpha
    return (
        candidate_q.copy(),
        candidate_gradient.copy(),
        candidate_potential,
        acceptance_statistic,
        depth,
        total_steps,
        total_divergences,
        maximum_depth_hit,
        total_nonfinite_leaves,
    )


@njit(cache=True, nogil=True)
def _dual_averaging_update(
    warmup_index: int,
    acceptance_statistic: float,
    target_acceptance: float,
    mu: float,
    h_bar: float,
    log_epsilon_bar: float,
    gamma: float,
    t0: float,
    kappa: float,
) -> tuple[float, float, float]:
    """One Algorithm 6 dual-averaging update."""

    weight = 1.0 / (warmup_index + t0)
    new_h_bar = (1.0 - weight) * h_bar + weight * (
        target_acceptance - acceptance_statistic
    )
    new_log_epsilon = mu - (
        math.sqrt(warmup_index) / gamma
    ) * new_h_bar
    averaging_weight = warmup_index ** (-kappa)
    new_log_epsilon_bar = (
        averaging_weight * new_log_epsilon
        + (1.0 - averaging_weight) * log_epsilon_bar
    )
    return new_h_bar, new_log_epsilon, new_log_epsilon_bar


@njit(cache=True, nogil=True)
def _simulate_nuts_chain(
    seed: int,
    q_initial: np.ndarray,
    beta: float,
    target_acceptance: float,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
    maximum_tree_depth: int,
    delta_max: float,
    dual_averaging_gamma: float,
    dual_averaging_t0: float,
    dual_averaging_kappa: float,
    reasonable_epsilon_max_search_steps: int,
):
    """Run one complete Algorithm 6 chain and return retained diagnostics."""

    np.random.seed(seed)
    samples = np.empty((n_retained, q_initial.size), dtype=np.float64)
    q = q_initial.copy()
    potential = _potential_energy(q, target_parameters)
    gradient = _target_gradient(q, target_parameters)

    initial_epsilon, search_leaf_gradients, search_success = (
        _find_reasonable_epsilon(
            q,
            gradient,
            potential,
            beta,
            target_parameters,
            mass_cholesky,
            inverse_mass_matrix,
            mass_is_identity,
            reasonable_epsilon_max_search_steps,
        )
    )
    # Include the one initial-position gradient in the non-scoring search cost.
    epsilon_search_gradient_calls = 1 + search_leaf_gradients
    if not search_success or not math.isfinite(initial_epsilon) or initial_epsilon <= 0.0:
        for draw in range(n_retained):
            for coordinate in range(q.size):
                samples[draw, coordinate] = math.nan
        return (
            samples, math.nan, math.nan, epsilon_search_gradient_calls, 0, 0,
            0, 0, math.nan, math.nan, n_retained, 0, 0, 0,
        )

    mu = math.log(10.0 * initial_epsilon)
    log_epsilon = math.log(initial_epsilon)
    # Printed Algorithm 6 initializes epsilon_bar_0=1.
    log_epsilon_bar = 0.0
    h_bar = 0.0
    warmup_gradient_calls = 0
    warmup_nonfinite_leaves = 0

    for warmup_index in range(1, n_warmup + 1):
        epsilon = _safe_exp_step_size(log_epsilon)
        result = _nuts_iteration(
            q,
            gradient,
            potential,
            epsilon,
            beta,
            target_parameters,
            mass_cholesky,
            inverse_mass_matrix,
            mass_is_identity,
            maximum_tree_depth,
            delta_max,
        )
        q, gradient, potential = result[0], result[1], result[2]
        warmup_gradient_calls += result[5]
        warmup_nonfinite_leaves += result[8]
        h_bar, log_epsilon, log_epsilon_bar = _dual_averaging_update(
            warmup_index,
            result[3],
            target_acceptance,
            mu,
            h_bar,
            log_epsilon_bar,
            dual_averaging_gamma,
            dual_averaging_t0,
            dual_averaging_kappa,
        )

    adapted_epsilon = _safe_exp_step_size(log_epsilon_bar)
    retained_gradient_calls = 0
    retained_divergences = 0
    retained_maximum_depth_hits = 0
    sum_acceptance_statistic = 0.0
    sum_tree_depth = 0.0
    sum_leapfrog_steps = 0.0
    nonfinite_samples = 0
    retained_nonfinite_leaves = 0

    for retained_index in range(n_retained):
        result = _nuts_iteration(
            q,
            gradient,
            potential,
            adapted_epsilon,
            beta,
            target_parameters,
            mass_cholesky,
            inverse_mass_matrix,
            mass_is_identity,
            maximum_tree_depth,
            delta_max,
        )
        q, gradient, potential = result[0], result[1], result[2]
        retained_gradient_calls += result[5]
        retained_divergences += result[6]
        retained_maximum_depth_hits += result[7]
        retained_nonfinite_leaves += result[8]
        sum_acceptance_statistic += result[3]
        sum_tree_depth += result[4]
        sum_leapfrog_steps += result[5]
        finite = bool(
            _vector_is_finite(q)
            and _vector_is_finite(gradient)
            and math.isfinite(potential)
        )
        if not finite:
            nonfinite_samples += 1
        for coordinate in range(q.size):
            samples[retained_index, coordinate] = q[coordinate]

    inverse_retained = 1.0 / n_retained
    return (
        samples,
        adapted_epsilon,
        sum_acceptance_statistic * inverse_retained,
        epsilon_search_gradient_calls,
        warmup_gradient_calls,
        retained_gradient_calls,
        retained_divergences,
        retained_maximum_depth_hits,
        sum_tree_depth * inverse_retained,
        sum_leapfrog_steps * inverse_retained,
        nonfinite_samples,
        1,
        warmup_nonfinite_leaves,
        retained_nonfinite_leaves,
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


# Observables, block analysis, and representative selection.


def _observable_specification(
    beta: float,
    sigma_y: float,
    transverse_dimension: int,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Exact moments for every physical linear and unique quadratic."""

    dimension = 1 + int(transverse_dimension)
    variance_y = sigma_y * sigma_y / beta
    exp_half_variance = _safe_exp_target(0.5 * variance_y)
    exp_variance = _safe_exp_target(variance_y)
    exp_twice_variance = _safe_exp_target(2.0 * variance_y)
    variance_x = exp_half_variance / beta
    coordinate_names = ["y"] + [
        f"x{index}" for index in range(1, transverse_dimension + 1)
    ]
    names: list[str] = list(coordinate_names)
    means: list[float] = [0.0] * dimension
    variances: list[float] = [variance_y] + [variance_x] * transverse_dimension

    for first in range(dimension):
        for second in range(first, dimension):
            names.append(
                f"{coordinate_names[first]}^2"
                if first == second
                else f"{coordinate_names[first]}*{coordinate_names[second]}"
            )
            if first == 0 and second == 0:
                means.append(variance_y)
                variances.append(2.0 * variance_y * variance_y)
            elif first == 0:
                means.append(0.0)
                variances.append(
                    (variance_y + variance_y * variance_y)
                    * exp_half_variance
                    / beta
                )
            elif first == second:
                means.append(variance_x)
                variances.append(
                    (3.0 * exp_twice_variance - exp_variance)
                    / (beta * beta)
                )
            else:
                means.append(0.0)
                variances.append(exp_twice_variance / (beta * beta))

    exact_means = np.asarray(means, dtype=np.float64)
    exact_variances = np.asarray(variances, dtype=np.float64)
    if not np.all(np.isfinite(exact_variances)) or np.any(exact_variances <= 0.0):
        raise ValueError(
            "Derived Neal-funnel observable variances must be finite and positive."
        )
    return tuple(names), exact_means, np.sqrt(exact_variances)


def _latent_to_physical(
    funnel_starts: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Map entered (y,z_1,...,z_d) rows to physical (y,x_1,...,x_d)."""

    latent = np.asarray(funnel_starts, dtype=np.float64)
    physical = np.empty_like(latent)
    physical[:, 0] = latent[:, 0]
    conditional_scales = np.exp(0.5 * latent[:, 0]) / math.sqrt(beta)
    physical[:, 1:] = latent[:, 1:] * conditional_scales[:, None]
    return np.ascontiguousarray(physical)


def _physical_to_latent(
    physical_states: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Map physical (y,x_1,...,x_d) rows back to (y,z_1,...,z_d)."""

    states = np.asarray(physical_states, dtype=np.float64)
    latent = np.empty_like(states)
    latent[:, 0] = states[:, 0]
    inverse_scales = math.sqrt(beta) * np.exp(-0.5 * states[:, 0])
    latent[:, 1:] = states[:, 1:] * inverse_scales[:, None]
    return np.ascontiguousarray(latent)


def _physical_observables(samples: np.ndarray, dimension: int) -> np.ndarray:
    """Return q_i and all unique q_i*q_j for every chain and draw."""

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != dimension:
        raise ValueError(
            f"samples must have shape (chains, draws, {dimension}); received {samples.shape}."
        )
    values: list[np.ndarray] = [samples[:, :, index] for index in range(dimension)]
    for first in range(dimension):
        for second in range(first, dimension):
            values.append(samples[:, :, first] * samples[:, :, second])
    return np.stack(values, axis=2)


def _analyze_block(
    *,
    samples: np.ndarray,
    target_acceptance: float,
    block_index: int,
    seeds: Sequence[int],
    mean_acceptance_statistic: float,
    adapted_step_sizes: Sequence[float],
    retained_gradient_calls: int,
    epsilon_search_gradient_calls: int,
    warmup_gradient_calls: int,
    retained_divergences: int,
    retained_maximum_depth_hits: int,
    mean_retained_tree_depth: float,
    mean_retained_leapfrog_steps: float,
    nonfinite_samples: int,
    observable_names: Sequence[str],
    exact_observable_means: np.ndarray,
    exact_observable_sds: np.ndarray,
    warmup_nonfinite_leaves: int = 0,
    retained_nonfinite_leaves: int = 0,
) -> BlockResult:
    """Calculate the complete agreed diagnostics for one tuning block."""

    samples = np.asarray(samples, dtype=np.float64)
    total_retained = samples.shape[0] * samples.shape[1]
    gradients_per_draw = retained_gradient_calls / total_retained
    dimension = int(samples.shape[2])
    observable_values = _physical_observables(samples, dimension)
    diagnostics: list[ObservableDiagnostic] = []
    # Match the already-frozen expanded funnel THC rule. A failed proposal
    # invalidates efficiency, not the descriptive IAT/R-hat of retained states.
    simulation_is_nonfinite = (
        nonfinite_samples > 0 or not np.all(np.isfinite(samples))
        or warmup_nonfinite_leaves > 0 or retained_nonfinite_leaves > 0
    )

    if observable_values.shape[2] != len(observable_names):
        raise RuntimeError("Observable values and specification have different lengths.")

    for observable_index, name in enumerate(observable_names):
        chains = observable_values[:, :, observable_index]
        iat = _multi_chain_geyer_iat(chains)
        rhat = _rank_normalized_split_folded_rhat(chains)
        empirical_mean = float(np.mean(chains))
        exact_mean = float(exact_observable_means[observable_index])
        raw_error = empirical_mean - exact_mean
        standardized_error = abs(raw_error) / float(
            exact_observable_sds[observable_index]
        )
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
            range(len(diagnostics)), key=lambda index: diagnostics[index].iat.tau
        )
        worst_observable = diagnostics[worst_index].name
        worst_iat = diagnostics[worst_index].iat.tau
        block_efficiency = (
            diagnostics[worst_index].efficiency if status == "ok" else math.nan
        )
    else:
        undefined_iat_indices = [
            index for index, item in enumerate(diagnostics) if not item.iat.is_defined
        ]
        worst_observable = (
            diagnostics[undefined_iat_indices[0]].name
            if undefined_iat_indices
            else "undefined"
        )
        worst_iat = math.nan
        block_efficiency = math.nan

    if all(not math.isnan(item.rhat.combined) for item in diagnostics):
        maximum_rhat_index = max(
            range(len(diagnostics)),
            key=lambda index: diagnostics[index].rhat.combined,
        )
        maximum_rhat_observable = diagnostics[maximum_rhat_index].name
        maximum_rhat = diagnostics[maximum_rhat_index].rhat.combined
    else:
        maximum_rhat_observable = "undefined"
        maximum_rhat = math.nan

    maximum_moment_index = max(
        range(len(diagnostics)),
        key=lambda index: diagnostics[index].standardized_moment_error,
    )

    return BlockResult(
        target_acceptance=float(target_acceptance),
        block_index=int(block_index),
        seeds=tuple(int(seed) for seed in seeds),
        status=status,
        mean_acceptance_statistic=float(mean_acceptance_statistic),
        adapted_step_sizes=tuple(float(value) for value in adapted_step_sizes),
        retained_gradient_calls=int(retained_gradient_calls),
        gradients_per_retained_draw=gradients_per_draw,
        epsilon_search_gradient_calls=int(epsilon_search_gradient_calls),
        warmup_gradient_calls=int(warmup_gradient_calls),
        retained_divergences=int(retained_divergences),
        retained_maximum_depth_hits=int(retained_maximum_depth_hits),
        mean_retained_tree_depth=float(mean_retained_tree_depth),
        mean_retained_leapfrog_steps=float(mean_retained_leapfrog_steps),
        nonfinite_samples=int(nonfinite_samples),
        warmup_nonfinite_leaves=int(warmup_nonfinite_leaves),
        retained_nonfinite_leaves=int(retained_nonfinite_leaves),
        observable_diagnostics=tuple(diagnostics),
        worst_observable=worst_observable,
        worst_iat=worst_iat,
        block_efficiency=block_efficiency,
        maximum_rhat_observable=maximum_rhat_observable,
        maximum_rhat=maximum_rhat,
        maximum_moment_error_observable=diagnostics[maximum_moment_index].name,
        maximum_standardized_moment_error=diagnostics[
            maximum_moment_index
        ].standardized_moment_error,
    )


def _aggregate_blocks(block_results: Sequence[BlockResult]) -> BlockAggregate:
    """Apply the common across-seed GM score and descriptive summaries."""

    blocks = tuple(block_results)
    if not blocks:
        raise ValueError("At least one seed-block result is required.")
    block_efficiencies = np.asarray(
        [block.block_efficiency for block in blocks], dtype=np.float64
    )
    valid = all(block.status == "ok" for block in blocks) and np.all(
        np.isfinite(block_efficiencies) & (block_efficiencies > 0.0)
    )

    if valid:
        geometric_mean = float(np.exp(np.mean(np.log(block_efficiencies))))
        minimum = float(np.min(block_efficiencies))
        median = float(np.median(block_efficiencies))
        maximum = float(np.max(block_efficiencies))
        status = "ok"
    else:
        geometric_mean = math.nan
        minimum = math.nan
        median = math.nan
        maximum = math.nan
        status = "undefined_block_metric"

    rhat_values = np.asarray([block.maximum_rhat for block in blocks])
    maximum_rhat = float(np.max(rhat_values)) if not np.any(np.isnan(rhat_values)) else math.nan
    moment_values = np.asarray(
        [block.maximum_standardized_moment_error for block in blocks]
    )
    maximum_moment_error = (
        float(np.max(moment_values))
        if np.all(np.isfinite(moment_values))
        else math.nan
    )
    total_nonfinite = sum(block.nonfinite_samples for block in blocks)

    return BlockAggregate(
        status=status,
        geometric_mean_efficiency=geometric_mean,
        minimum_block_efficiency=minimum,
        median_block_efficiency=median,
        maximum_block_efficiency=maximum,
        mean_acceptance_statistic=float(
            np.mean([block.mean_acceptance_statistic for block in blocks])
        ),
        mean_retained_gradients_per_draw=float(
            np.mean([block.gradients_per_retained_draw for block in blocks])
        ),
        total_retained_divergences=int(
            sum(block.retained_divergences for block in blocks)
        ),
        total_retained_maximum_depth_hits=int(
            sum(block.retained_maximum_depth_hits for block in blocks)
        ),
        maximum_rhat=maximum_rhat,
        maximum_standardized_moment_error=maximum_moment_error,
        total_nonfinite_samples=int(total_nonfinite),
    )


def _aggregate_configuration(
    target_acceptance: float,
    block_results: Sequence[BlockResult],
) -> ConfigurationResult:
    """Wrap the block aggregate with one fixed NUTS target acceptance."""

    blocks = tuple(block_results)
    aggregate = _aggregate_blocks(blocks)
    return ConfigurationResult(
        target_acceptance=float(target_acceptance),
        status=aggregate.status,
        block_results=blocks,
        geometric_mean_efficiency=aggregate.geometric_mean_efficiency,
        minimum_block_efficiency=aggregate.minimum_block_efficiency,
        median_block_efficiency=aggregate.median_block_efficiency,
        maximum_block_efficiency=aggregate.maximum_block_efficiency,
        mean_acceptance_statistic=aggregate.mean_acceptance_statistic,
        mean_retained_gradients_per_draw=aggregate.mean_retained_gradients_per_draw,
        total_retained_divergences=aggregate.total_retained_divergences,
        total_retained_maximum_depth_hits=(
            aggregate.total_retained_maximum_depth_hits
        ),
        maximum_rhat=aggregate.maximum_rhat,
        maximum_standardized_moment_error=aggregate.maximum_standardized_moment_error,
        total_nonfinite_samples=aggregate.total_nonfinite_samples,
    )


def _select_representative(
    configuration_results: Sequence[ConfigurationResult],
) -> ConfigurationResult:
    """Select by GM efficiency with a deterministic exact-tie rule."""

    eligible = [result for result in configuration_results if result.status == "ok"]
    if not eligible:
        raise RuntimeError("No NUTS configuration has a defined full-block tuning score.")

    # Exact ties: larger minimum block score, fewer gradients/draw, then the
    # larger target acceptance.  No near-tie tolerance is introduced.
    return max(
        eligible,
        key=lambda result: (
            result.geometric_mean_efficiency,
            result.minimum_block_efficiency,
            -result.mean_retained_gradients_per_draw,
            result.target_acceptance,
        ),
    )


# Validation, execution, and reporting.


def _normalize_grid(
    target_acceptance_values: Sequence[float],
) -> tuple[float, ...]:
    try:
        raw_values = tuple(target_acceptance_values)
    except TypeError as error:
        raise ValueError(
            "target_acceptance_values must be an iterable of real numbers."
        ) from error
    normalized_list: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(
                f"target_acceptance_values[{index}] must be a real number."
            )
        normalized_list.append(float(value))
    normalized = tuple(normalized_list)
    if not normalized or any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in normalized
    ):
        raise ValueError("Every target acceptance must lie strictly in (0,1).")
    if len(set(normalized)) != len(normalized):
        raise ValueError("target acceptance values must not contain duplicates.")
    return normalized


def _require_integer(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer; received {value!r}.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}; received {normalized}.")
    return normalized


def _validate_seed_blocks(
    seed_blocks: Sequence[Sequence[int]],
    n_chains: int,
) -> tuple[tuple[int, ...], ...]:
    try:
        raw_blocks = tuple(seed_blocks)
    except TypeError as error:
        raise ValueError("seed_blocks must be an iterable of seed blocks.") from error
    normalized_blocks: list[tuple[int, ...]] = []
    for block_index, block in enumerate(raw_blocks):
        try:
            raw_seeds = tuple(block)
        except TypeError as error:
            raise ValueError(
                f"seed_blocks[{block_index}] must be an iterable of integer seeds."
            ) from error
        normalized_block: list[int] = []
        for seed_index, seed in enumerate(raw_seeds):
            if isinstance(seed, (bool, np.bool_)) or not isinstance(
                seed, numbers.Integral
            ):
                raise ValueError(
                    f"seed_blocks[{block_index}][{seed_index}] must be an integer."
                )
            normalized_block.append(int(seed))
        normalized_blocks.append(tuple(normalized_block))
    normalized = tuple(normalized_blocks)
    if not normalized:
        raise ValueError("At least one tuning seed block is required.")
    for block_index, block in enumerate(normalized):
        if len(block) != n_chains:
            raise ValueError(
                f"seed_blocks[{block_index}] must contain exactly {n_chains} seeds; "
                f"received {len(block)}."
            )
        if len(set(block)) != len(block):
            raise ValueError(f"Seeds in seed_blocks[{block_index}] must be distinct.")
        if any(seed < 0 or seed > np.iinfo(np.uint32).max for seed in block):
            raise ValueError(
                f"Every seed in seed_blocks[{block_index}] must lie in [0, 2**32 - 1]."
            )
    flat = [seed for block in normalized for seed in block]
    if len(set(flat)) != len(flat):
        raise ValueError("Every tuning seed must be unique across all blocks.")
    return normalized


def _json_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _common_input_components(
    *,
    beta: float,
    sigma_y: float,
    transverse_dimension: int,
    mass_matrix: np.ndarray,
    funnel_starts: np.ndarray,
    target_acceptance_values: Sequence[float],
    maximum_tree_depth: int,
    delta_max: float,
    dual_averaging_gamma: float,
    dual_averaging_t0: float,
    dual_averaging_kappa: float,
    reasonable_epsilon_max_search_steps: int,
    seed_blocks: Sequence[Sequence[int]],
    n_warmup: int,
    n_retained: int,
    observable_policy: str,
) -> dict[str, object]:
    coordinate_names = ["y"] + [
        f"x{index}" for index in range(1, transverse_dimension + 1)
    ]
    ordered_observables = list(coordinate_names)
    for first in range(len(coordinate_names)):
        for second in range(first, len(coordinate_names)):
            ordered_observables.append(
                f"{coordinate_names[first]}^2"
                if first == second
                else f"{coordinate_names[first]}*{coordinate_names[second]}"
            )
    return {
        "target": {
            "family": "CENTERED_NEALS_FUNNEL",
            "dimension": 1 + int(transverse_dimension),
            "transverse_dimension": int(transverse_dimension),
            "beta": float(beta),
            "sigma_y": float(sigma_y),
            "potential": (
                "y^2/(2*sigma_y^2)+d_x*y/(2*beta)"
                "+exp(-y)*sum_i(x_i^2)/2"
            ),
        },
        "geometry": {"mass_matrix": mass_matrix.tolist()},
        "starts": {
            "coordinate_system": (
                "(y,z_1,...,z_d), x_i=exp(y/2)*z_i/sqrt(beta)"
            ),
            "funnel_starts": funnel_starts.tolist(),
        },
        "nuts_method": {
            "algorithm": NUTS_IMPLEMENTATION_VERSION,
            "target_acceptance_values": list(target_acceptance_values),
            "maximum_tree_depth": int(maximum_tree_depth),
            "delta_max": float(delta_max),
            "dual_averaging_gamma": float(dual_averaging_gamma),
            "dual_averaging_t0": float(dual_averaging_t0),
            "dual_averaging_kappa": float(dual_averaging_kappa),
            "reasonable_epsilon_max_search_steps": int(
                reasonable_epsilon_max_search_steps
            ),
        },
        "seed_blocks": {"seed_blocks": [list(block) for block in seed_blocks]},
        "run_lengths": {
            "n_warmup": int(n_warmup),
            "n_retained": int(n_retained),
        },
        "observable_policy": {
            "policy": observable_policy,
            "ordered_observables": ordered_observables,
            "thermostat_is_scored": False,
            "target_specific_nonlinear_observables_are_scored": False,
        },
    }


def _prepare_inputs(inputs: RunInputs) -> PreparedInputs:
    """Validate every input and derive all dependent target quantities."""

    if not isinstance(inputs, RunInputs):
        raise TypeError("Input preparation expects one RunInputs object.")

    if isinstance(inputs.beta, (bool, np.bool_)) or not isinstance(
        inputs.beta, numbers.Real
    ):
        raise ValueError("beta must be a real number.")
    beta = float(inputs.beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive.")

    if isinstance(inputs.sigma_y, (bool, np.bool_)) or not isinstance(
        inputs.sigma_y, numbers.Real
    ):
        raise ValueError("sigma_y must be a real number.")
    sigma_y = float(inputs.sigma_y)
    if not math.isfinite(sigma_y) or sigma_y <= 0.0:
        raise ValueError("sigma_y must be finite and strictly positive.")
    transverse_dimension = _require_integer("dimension_x", inputs.dimension_x, 1)
    dimension = 1 + transverse_dimension

    try:
        mass_matrix = np.asarray(inputs.mass_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("mass_matrix must be a numeric square matrix.") from error
    expected_mass_shape = (dimension, dimension)
    if mass_matrix.shape != expected_mass_shape:
        raise ValueError(
            f"mass_matrix must have shape {expected_mass_shape}; received {mass_matrix.shape}."
        )
    if not np.all(np.isfinite(mass_matrix)):
        raise ValueError("mass_matrix must contain only finite values.")
    if not np.array_equal(mass_matrix, mass_matrix.T):
        raise ValueError("mass_matrix must be exactly symmetric.")
    mass_matrix = np.ascontiguousarray(mass_matrix.copy())
    try:
        mass_cholesky = np.linalg.cholesky(mass_matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass_matrix must be positive definite.") from error
    inverse_mass_matrix = np.linalg.inv(mass_matrix)
    if not np.all(np.isfinite(inverse_mass_matrix)):
        raise ValueError("The inverse of mass_matrix is nonfinite.")
    mass_cholesky = np.ascontiguousarray(mass_cholesky)
    inverse_mass_matrix = np.ascontiguousarray(inverse_mass_matrix)
    mass_is_identity = bool(np.array_equal(mass_matrix, np.eye(dimension)))

    try:
        funnel_starts = np.asarray(inputs.funnel_starts, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("funnel_starts must be a numeric matrix.") from error
    if funnel_starts.ndim != 2 or funnel_starts.shape[1] != dimension:
        raise ValueError(
            "funnel_starts must have shape (n_chains, 1 + dimension_x); "
            f"received {funnel_starts.shape} for dimension {dimension}."
        )
    if funnel_starts.shape[0] < 2:
        raise ValueError("At least two chain starts are required.")
    if not np.all(np.isfinite(funnel_starts)):
        raise ValueError("funnel_starts must contain only finite values.")
    funnel_starts = np.ascontiguousarray(funnel_starts.copy())
    n_chains = int(funnel_starts.shape[0])

    target_acceptance_values = _normalize_grid(inputs.target_acceptance_values)
    maximum_tree_depth = _require_integer(
        "maximum_tree_depth", inputs.maximum_tree_depth, 1
    )
    reasonable_epsilon_max_search_steps = _require_integer(
        "reasonable_epsilon_max_search_steps",
        inputs.reasonable_epsilon_max_search_steps,
        1,
    )
    scalar_method_inputs: dict[str, float] = {}
    for name in (
        "delta_max",
        "dual_averaging_gamma",
        "dual_averaging_t0",
        "dual_averaging_kappa",
    ):
        raw_value = getattr(inputs, name)
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value, numbers.Real
        ):
            raise ValueError(f"{name} must be a real number.")
        scalar_method_inputs[name] = float(raw_value)
    delta_max = scalar_method_inputs["delta_max"]
    dual_averaging_gamma = scalar_method_inputs["dual_averaging_gamma"]
    dual_averaging_t0 = scalar_method_inputs["dual_averaging_t0"]
    dual_averaging_kappa = scalar_method_inputs["dual_averaging_kappa"]
    if not math.isfinite(delta_max) or delta_max <= 0.0:
        raise ValueError("delta_max must be finite and positive.")
    if not math.isfinite(dual_averaging_gamma) or dual_averaging_gamma <= 0.0:
        raise ValueError("dual_averaging_gamma must be finite and positive.")
    if not math.isfinite(dual_averaging_t0) or dual_averaging_t0 <= 0.0:
        raise ValueError("dual_averaging_t0 must be finite and positive.")
    if not math.isfinite(dual_averaging_kappa) or not (
        0.5 < dual_averaging_kappa <= 1.0
    ):
        raise ValueError("dual_averaging_kappa must lie in (0.5,1].")
    seed_blocks = _validate_seed_blocks(inputs.seed_blocks, n_chains)
    n_warmup = _require_integer("n_warmup", inputs.n_warmup, 1)
    n_retained = _require_integer("n_retained", inputs.n_retained, 4)
    n_workers = _require_integer("n_workers", inputs.n_workers, 1)

    if not isinstance(inputs.observable_policy, str):
        raise ValueError("observable_policy must be a string.")
    observable_policy = inputs.observable_policy.strip().upper()
    if observable_policy != SUPPORTED_OBSERVABLE_POLICY:
        raise ValueError(
            "This fair-playground code supports only "
            f"observable_policy='{SUPPORTED_OBSERVABLE_POLICY}'."
        )

    for field_name in (
        "allow_overwrite",
        "print_progress",
        "print_report",
        "run_self_tests",
    ):
        if not isinstance(getattr(inputs, field_name), (bool, np.bool_)):
            raise ValueError(f"{field_name} must be True or False.")

    if not isinstance(inputs.run_mode, str):
        raise ValueError("run_mode must be 'CANONICAL' or 'CUSTOM'.")
    run_mode = inputs.run_mode.strip().upper()
    if run_mode not in {"CANONICAL", "CUSTOM"}:
        raise ValueError("run_mode must be exactly 'CANONICAL' or 'CUSTOM'.")
    if not isinstance(inputs.run_label, str) or not inputs.run_label.strip():
        raise ValueError("run_label must be a nonempty string.")
    run_label = inputs.run_label.strip()

    if inputs.output_directory is not None and not isinstance(
        inputs.output_directory, (str, os.PathLike)
    ):
        raise ValueError("output_directory must be None or a filesystem path.")
    if not isinstance(inputs.output_filename, str) or not inputs.output_filename.strip():
        raise ValueError("output_filename must be a nonempty string.")
    output_filename = inputs.output_filename.strip()
    if Path(output_filename).name != output_filename or any(
        separator in output_filename for separator in ("/", "\\")
    ):
        raise ValueError(
            "output_filename must be a filename only; use output_directory for its folder."
        )
    if Path(output_filename).suffix.lower() != ".txt":
        raise ValueError("output_filename must end in .txt.")

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        variance_y = sigma_y * sigma_y / beta
        variance_x = _safe_exp_target(0.5 * variance_y) / beta
        target_parameters = np.ascontiguousarray(
            np.asarray(
                [sigma_y * sigma_y, float(transverse_dimension), beta],
                dtype=np.float64,
            )
        )
        target_variances = np.ascontiguousarray(
            np.asarray(
                [variance_y] + [variance_x] * transverse_dimension,
                dtype=np.float64,
            )
        )
        physical_starts = _latent_to_physical(funnel_starts, beta)
    if not np.all(np.isfinite(target_parameters)) or np.any(target_parameters <= 0.0):
        raise ValueError("Derived target_parameters values must be finite and positive.")
    if not np.all(np.isfinite(target_variances)) or np.any(
        target_variances <= 0.0
    ):
        raise ValueError(
            "Derived target variances must be finite and positive; check beta and sigma_y."
        )
    if not np.all(np.isfinite(physical_starts)):
        raise ValueError(
            "Derived physical starts are nonfinite; check beta, sigma_y, and funnel_starts."
        )
    observable_names, exact_observable_means, exact_observable_sds = (
        _observable_specification(beta, sigma_y, transverse_dimension)
    )
    if not np.all(np.isfinite(exact_observable_means)):
        raise ValueError("Derived exact observable means must be finite.")
    if not np.all(np.isfinite(exact_observable_sds)) or np.any(
        exact_observable_sds <= 0.0
    ):
        raise ValueError(
            "Derived exact observable standard deviations must be finite and positive."
        )

    components = _common_input_components(
        beta=beta,
        sigma_y=sigma_y,
        transverse_dimension=transverse_dimension,
        mass_matrix=mass_matrix,
        funnel_starts=funnel_starts,
        target_acceptance_values=target_acceptance_values,
        maximum_tree_depth=maximum_tree_depth,
        delta_max=delta_max,
        dual_averaging_gamma=dual_averaging_gamma,
        dual_averaging_t0=dual_averaging_t0,
        dual_averaging_kappa=dual_averaging_kappa,
        reasonable_epsilon_max_search_steps=(
            reasonable_epsilon_max_search_steps
        ),
        seed_blocks=seed_blocks,
        n_warmup=n_warmup,
        n_retained=n_retained,
        observable_policy=observable_policy,
    )
    canonical_starts = np.asarray(
        [
            [-5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [-2.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            [2.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [5.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ]
    )
    canonical_seed_blocks = (
        (3857660299, 1344569885, 2010517369, 2974962254),
        (2914251580, 2216611040, 1467312042, 1966646752),
        (26638442, 2162183804, 2288357861, 3874567655),
        (917698748, 796766395, 2769691025, 1469822251),
        (983213432, 4224231832, 2042092071, 4006032910),
    )
    canonical_checks = {
        "target": beta == 1.0 and sigma_y == 3.0 and transverse_dimension == 5,
        "geometry": np.array_equal(mass_matrix, np.eye(6)),
        "starts": np.array_equal(funnel_starts, canonical_starts),
        "nuts_method": (
            target_acceptance_values == (0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
            and maximum_tree_depth == 12
            and delta_max == 1000.0
            and dual_averaging_gamma == 0.05
            and dual_averaging_t0 == 10.0
            and dual_averaging_kappa == 0.75
            and reasonable_epsilon_max_search_steps == 100
        ),
        "seed_blocks": seed_blocks == canonical_seed_blocks,
        "run_lengths": n_warmup == 1000 and n_retained == 4000,
        "observable_policy": observable_policy == SUPPORTED_OBSERVABLE_POLICY,
    }
    canonical_mismatches = tuple(
        name for name, matches in canonical_checks.items() if not matches
    )
    common_input_fingerprint = _json_fingerprint(
        {"schema": COMMON_INPUT_SCHEMA_VERSION, "components": components}
    )

    for array in (
        target_parameters,
        target_variances,
        mass_matrix,
        inverse_mass_matrix,
        mass_cholesky,
        funnel_starts,
        physical_starts,
        exact_observable_means,
        exact_observable_sds,
    ):
        array.setflags(write=False)

    return PreparedInputs(
        beta=beta,
        sigma_y=sigma_y,
        transverse_dimension=transverse_dimension,
        target_parameters=target_parameters,
        target_variances=target_variances,
        mass_matrix=mass_matrix,
        inverse_mass_matrix=inverse_mass_matrix,
        mass_cholesky=mass_cholesky,
        mass_is_identity=mass_is_identity,
        funnel_starts=funnel_starts,
        physical_starts=physical_starts,
        dimension=dimension,
        n_chains=n_chains,
        target_acceptance_values=target_acceptance_values,
        maximum_tree_depth=maximum_tree_depth,
        delta_max=delta_max,
        dual_averaging_gamma=dual_averaging_gamma,
        dual_averaging_t0=dual_averaging_t0,
        dual_averaging_kappa=dual_averaging_kappa,
        reasonable_epsilon_max_search_steps=(
            reasonable_epsilon_max_search_steps
        ),
        seed_blocks=seed_blocks,
        n_warmup=n_warmup,
        n_retained=n_retained,
        n_workers=n_workers,
        observable_policy=observable_policy,
        observable_names=observable_names,
        exact_observable_means=exact_observable_means,
        exact_observable_sds=exact_observable_sds,
        run_mode=run_mode,
        run_label=run_label,
        output_directory=inputs.output_directory,
        output_filename=output_filename,
        allow_overwrite=bool(inputs.allow_overwrite),
        print_progress=bool(inputs.print_progress),
        print_report=bool(inputs.print_report),
        run_self_tests=bool(inputs.run_self_tests),
        common_input_components=components,
        common_input_fingerprint=common_input_fingerprint,
        canonical_mismatches=canonical_mismatches,
    )


def _validate_run_mode(prepared: PreparedInputs) -> None:
    """Separate canonical competition runs from explicitly custom runs."""

    if prepared.run_mode == "CANONICAL":
        if prepared.canonical_mismatches:
            mismatch_text = ", ".join(prepared.canonical_mismatches)
            raise ValueError(
                "CANONICAL input mismatch in: "
                + mismatch_text
                + ". Restore the agreed bottom-block values, or use CUSTOM mode "
                "with a clearly labelled, non-comparable output."
            )
        if prepared.run_label.upper() != "PRODUCTION TUNING":
            raise ValueError(
                "CANONICAL mode requires run_label='PRODUCTION TUNING'."
            )
        if not prepared.run_self_tests:
            raise ValueError("CANONICAL mode requires run_self_tests=True.")
    else:
        if not prepared.run_label.upper().startswith("CUSTOM:"):
            raise ValueError("CUSTOM mode requires run_label beginning with 'CUSTOM:'.")
        if prepared.output_filename.casefold() == CANONICAL_OUTPUT_FILENAME.casefold():
            raise ValueError(
                "CUSTOM mode must use an output_filename different from the canonical report."
            )


def _resolve_output_destination(prepared: PreparedInputs) -> Path:
    if prepared.output_directory is None:
        try:
            directory = Path(__file__).resolve().parent
        except NameError:  # Jupyter execution after pasting the script
            directory = Path.cwd().resolve()
    else:
        directory = Path(prepared.output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / prepared.output_filename
    if destination.exists() and destination.is_dir():
        raise ValueError(f"Output destination is a directory: {destination}")
    if destination.exists() and not prepared.allow_overwrite:
        raise FileExistsError(
            f"Output already exists and allow_overwrite=False: {destination}"
        )
    return destination


def _diagnostics_fingerprint() -> str:
    return _source_fingerprint(
        (
            IATResult,
            _autocovariance_fft,
            _multi_chain_geyer_iat,
            _basic_rhat,
            _split_chains,
            _average_ranks,
            _standard_normal_ppf,
            _rank_normalize,
            _rank_normalized_split_folded_rhat,
        )
    )


def _source_fingerprint(functions: Sequence[object]) -> str:
    """Hash exact definitions from a script or one pasted IPython cell."""

    try:
        source = "\n".join(inspect.getsource(function) for function in functions)
    except (OSError, TypeError):
        filenames: set[str] = set()
        for function in functions:
            source_object = getattr(function, "py_func", function)
            code = getattr(source_object, "__code__", None)
            if code is not None:
                filenames.add(code.co_filename)
        if len(filenames) != 1:
            return "unavailable"
        filename = filenames.pop()
        source_lines = linecache.getlines(filename)
        if not source_lines:
            return "unavailable"
        try:
            tree = ast.parse("".join(source_lines), filename=filename)
        except SyntaxError:
            return "unavailable"
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
        }
        extracted: list[str] = []
        for function in functions:
            name = getattr(function, "__name__", None)
            node = definitions.get(name)
            if node is None or node.end_lineno is None:
                return "unavailable"
            start_line = min(
                [node.lineno]
                + [
                    decorator.lineno
                    for decorator in node.decorator_list
                ]
            )
            extracted.append(
                "".join(
                    source_lines[
                        start_line - 1 : node.end_lineno
                    ]
                )
            )
        source = "\n".join(extracted)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _source_verification_mode() -> str:
    """Report which source-attestation path the environment used."""

    try:
        inspect.getsource(RunInputs)
        inspect.getsource(IATResult)
        inspect.getsource(BlockAggregate)
    except (OSError, TypeError):
        probe = _source_fingerprint((RunInputs, _prepare_inputs))
        if probe != "unavailable":
            return "exact IPython-cell source attestation"
        return "source attestation unavailable"
    return "exact inspect.getsource attestation"


def _common_pipeline_fingerprint() -> str:
    """Hash physical observables, diagnostics, and scoring code."""

    return _source_fingerprint(
        (
            _observable_specification,
            _physical_observables,
            _analyze_block,
            BlockAggregate,
            _aggregate_blocks,
        )
    )


def _input_preparation_fingerprint() -> str:
    """Hash common input validation and all scientifically derived values."""

    return _source_fingerprint(
        (
            RunInputs,
            PreparedInputs,
            _normalize_grid,
            _require_integer,
            _validate_seed_blocks,
            _json_fingerprint,
            _latent_to_physical,
            _physical_to_latent,
            _common_input_components,
            _prepare_inputs,
        )
    )


def _nuts_implementation_fingerprint() -> str:
    """Hash NUTS dynamics, adaptation, orchestration, and tie-breaking code."""

    return _source_fingerprint(
        (
            _safe_exp_target,
            _potential_energy,
            _target_gradient,
            _kinetic_energy,
            _vector_is_finite,
            _joint_log_density,
            _leapfrog_one,
            _has_u_turn,
            _acceptance_statistic,
            _build_tree,
            _draw_momentum,
            _find_reasonable_epsilon,
            _safe_exp_step_size,
            _nuts_iteration,
            _dual_averaging_update,
            _simulate_nuts_chain,
            _aggregate_configuration,
            _select_representative,
            _run_chain_job,
            _run_configuration_block,
        )
    )


def _shared_protocol_fingerprint(prepared: PreparedInputs) -> str:
    payload = {
        "version": SHARED_PROTOCOL_VERSION,
        "common_input_fingerprint": prepared.common_input_fingerprint,
        "common_input_components": prepared.common_input_components,
        "target_variances": prepared.target_variances.tolist(),
        "physical_starts": prepared.physical_starts.tolist(),
        "inverse_mass_matrix": prepared.inverse_mass_matrix.tolist(),
        "observables": list(prepared.observable_names),
        "exact_observable_means": prepared.exact_observable_means.tolist(),
        "exact_observable_sds": prepared.exact_observable_sds.tolist(),
        "iat": DIAGNOSTICS_VERSION,
        "diagnostics_source_fingerprint": _diagnostics_fingerprint(),
        "common_pipeline_version": COMMON_PIPELINE_VERSION,
        "common_pipeline_source_fingerprint": _common_pipeline_fingerprint(),
        "input_preparation_version": INPUT_PREPARATION_VERSION,
        "input_preparation_source_fingerprint": _input_preparation_fingerprint(),
        "random_streams": RANDOM_STREAM_VERSION,
        "gradient_accounting": "every executed retained NUTS leapfrog leaf; cached current-state gradient; no artificial +1",
        "score": "GM_blocks(min_observable(1/(actual_postwarmup_gradients_per_draw*IAT)))",
    }
    return _json_fingerprint(payload)


def _format_number(value: float, specification: str = ".6g") -> str:
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, specification)


def _format_efficiency_grid(
    results: Sequence[ConfigurationResult],
    target_acceptance_values: Sequence[float],
) -> str:
    block_count = len(results[0].block_results) if results else 0
    lines = [
        f"{block_count}-block geometric-mean worst-observable efficiency by delta",
        "delta          GM_efficiency",
    ]
    lookup = {result.target_acceptance: result for result in results}
    for target_acceptance in target_acceptance_values:
        value = lookup[float(target_acceptance)].geometric_mean_efficiency
        lines.append(
            f"{target_acceptance:7.4f}          {_format_number(value, '.12e')}"
        )
    return "\n".join(lines)


def _build_report(
    *,
    configuration_results: Sequence[ConfigurationResult],
    representative: ConfigurationResult | None,
    prepared: PreparedInputs,
    self_test_messages: Sequence[str],
) -> str:
    mode_note = (
        "CANONICAL / COMPARABLE"
        if prepared.run_mode == "CANONICAL"
        else "CUSTOM / EXPANDED; comparable only under the shared expanded funnel protocol"
    )
    lines: list[str] = [
        f"FAIR-PLAYGROUND ORIGINAL SLICE-NUTS TUNING REPORT: {prepared.dimension}D CENTERED NEAL'S FUNNEL",
        "=" * 88,
        f"Run label: {prepared.run_label}",
        f"Run mode: {mode_note}",
        f"Canonical scientific inputs match: {not prepared.canonical_mismatches}",
        "Canonical mismatches: "
        + (", ".join(prepared.canonical_mismatches) or "none"),
        f"Normalized input fingerprint: {prepared.common_input_fingerprint}",
        f"Shared protocol version: {SHARED_PROTOCOL_VERSION}",
        f"Shared protocol fingerprint: {_shared_protocol_fingerprint(prepared)}",
        f"NUTS implementation version: {NUTS_IMPLEMENTATION_VERSION}",
        f"NUTS implementation source fingerprint: {_nuts_implementation_fingerprint()}",
        f"Diagnostics version: {DIAGNOSTICS_VERSION}",
        f"Diagnostics source fingerprint: {_diagnostics_fingerprint()}",
        f"Source verification mode: {_source_verification_mode()}",
        f"Random stream version: {RANDOM_STREAM_VERSION}",
        f"Numba available: {NUMBA_AVAILABLE}",
        f"Numba cache mode: {NUMBA_CACHE_MODE}",
        f"Requested worker threads: {prepared.n_workers}",
        f"Actual chain parallelism: {min(prepared.n_workers, prepared.n_chains) if NUMBA_AVAILABLE else 1}",
        f"Output filename: {prepared.output_filename}",
        "",
        "TARGET, GEOMETRY, AND INITIAL CONDITIONS",
        "-" * 88,
        f"d={prepared.dimension}, d_x={prepared.transverse_dimension}, beta={prepared.beta:g}, sigma_y={prepared.sigma_y:g}",
        "mass matrix M="
        + np.array2string(prepared.mass_matrix, precision=12, separator=", "),
        "marginal target variances="
        + np.array2string(prepared.target_variances, precision=12, separator=", "),
        "q=(y,x_1,...,x_d); y~N(0,sigma_y^2/beta); x_i|y~N(0,exp(y)/beta).",
        "U(q)=y^2/(2*sigma_y^2)+d_x*y/(2*beta)+exp(-y)*sum_i(x_i^2)/2.",
        "grad_y U=y/sigma_y^2+d_x/(2*beta)-exp(-y)*sum_i(x_i^2)/2; grad_x U=exp(-y)*x.",
        "p~N(0,M/beta); K(p)=0.5*p^T*M^-1*p; mass adaptation is disabled.",
    ]
    for chain_index in range(prepared.n_chains):
        lines.append(
            f"chain {chain_index + 1}: entered (y,z_1,...,z_d)="
            + np.array2string(
                prepared.funnel_starts[chain_index], precision=6, separator=", "
            )
            + "; physical q=(y,x_1,...,x_d)="
            + np.array2string(
                prepared.physical_starts[chain_index], precision=12, separator=", "
            )
        )

    lines.extend(
        [
            "",
            "NUTS AND TUNING DESIGN",
            "-" * 88,
            "Sampler: original Hoffman--Gelman slice-based Algorithm 6; not multinomial NUTS.",
            "There is no final Metropolis accept/reject decision.",
            "U-turn products use endpoint velocities M^-1*p.",
            f"target acceptance values: {list(prepared.target_acceptance_values)}",
            f"dual averaging: gamma={prepared.dual_averaging_gamma:g}, t0={prepared.dual_averaging_t0:g}, kappa={prepared.dual_averaging_kappa:g}",
            f"Delta_max={prepared.delta_max:g}; maximum tree depth={prepared.maximum_tree_depth}",
            f"reasonable-epsilon search limit={prepared.reasonable_epsilon_max_search_steps}",
            f"blocks={len(prepared.seed_blocks)}; chains/block={prepared.n_chains}",
            f"warmup/chain={prepared.n_warmup}; retained/chain={prepared.n_retained}",
            "Every delta restarts every chain from the same stated seed and physical start.",
            "Each chain adapts epsilon during warmup; averaged epsilon is frozen before retained sampling.",
            "Final-comparison seeds are absent from this tuner.",
            "",
            "TUNING SEED BLOCKS",
            "-" * 88,
        ]
    )
    for block_index, block in enumerate(prepared.seed_blocks, start=1):
        lines.append(f"block {block_index}: {list(block)}")

    lines.extend(
        [
            "",
            "DIAGNOSTICS AND SCORE",
            "-" * 88,
            f"Required physical observables ({len(prepared.observable_names)}): "
            + ", ".join(prepared.observable_names),
            "IAT: multi-chain Geyer initial-positive then initial-monotone sequence; no IAT floor.",
            "Every selected-observable line reports pair count, last included lag, first excluded lag, and cutoff reason.",
            "R-hat: max(rank-normalized split, folded rank-normalized split); descriptive, no threshold.",
            "Moment error: |empirical-exact| / exact observable SD; descriptive, no threshold.",
            "Gradient cost: every executed retained leapfrog leaf, including divergent and discarded leaves.",
            "The selected-state gradient is cached; no artificial +1 is charged per transition.",
            "Block score=min_f 1/(actual retained gradients/draw * IAT(f)).",
            f"Configuration score=geometric mean of {len(prepared.seed_blocks)} block scores.",
            "Efficiencies define the primary score; GM worst IAT is descriptive only. Finite-energy divergences and depth hits are not gates.",
            "Undefined required IAT, nonfinite selected states, or nonfinite leaves during warmup/retention make the score undefined.",
            "The initial reasonable-epsilon search is initialization, not a transition; its probes are excluded from that rule and its gradient cost is reported separately.",
            "No failed block is dropped. Descriptive worst IAT, R-hat, and moments remain visible.",
            "Exact-tie rule: larger minimum block score, fewer gradients/draw, then larger delta.",
            "",
            "INTERNAL SELF-TESTS",
            "-" * 88,
        ]
    )
    lines.extend(self_test_messages)

    ordered_results = sorted(
        configuration_results, key=lambda item: item.target_acceptance
    )
    lines.extend(
        [
            "",
            "CONFIGURATION SUMMARY",
            "-" * 88,
            "delta status                     GM_eff       min_eff    median_eff       max_eff  accept_stat grad/draw  div depth_hit max_Rhat max_moment_SD nonfinite",
        ]
    )
    for result in ordered_results:
        lines.append(
            f"{result.target_acceptance:7.4f} {result.status:<26} "
            f"{_format_number(result.geometric_mean_efficiency, '.6e'):>12} "
            f"{_format_number(result.minimum_block_efficiency, '.6e'):>12} "
            f"{_format_number(result.median_block_efficiency, '.6e'):>12} "
            f"{_format_number(result.maximum_block_efficiency, '.6e'):>12} "
            f"{result.mean_acceptance_statistic:11.6f} "
            f"{result.mean_retained_gradients_per_draw:9.4f} "
            f"{result.total_retained_divergences:4d} "
            f"{result.total_retained_maximum_depth_hits:9d} "
            f"{_format_number(result.maximum_rhat, '.6f'):>8} "
            f"{_format_number(result.maximum_standardized_moment_error, '.6e'):>13} "
            f"{result.total_nonfinite_samples:9d}"
        )

    lines.extend(
        [
            "",
            _format_efficiency_grid(ordered_results, prepared.target_acceptance_values),
            "",
            "ALL CONFIGURATION-BLOCK RESULTS",
            "-" * 88,
            "delta block status                     acc_stat grad/draw mean_steps mean_depth div depth_hit worst_obs worst_IAT block_eff max_Rhat max_moment_SD nonfinite",
        ]
    )
    for result in ordered_results:
        for block in result.block_results:
            lines.append(
                f"{block.target_acceptance:7.4f} {block.block_index + 1:5d} "
                f"{block.status:<26} {block.mean_acceptance_statistic:8.5f} "
                f"{block.gradients_per_retained_draw:9.4f} "
                f"{block.mean_retained_leapfrog_steps:10.4f} "
                f"{block.mean_retained_tree_depth:10.4f} "
                f"{block.retained_divergences:4d} "
                f"{block.retained_maximum_depth_hits:9d} "
                f"{block.worst_observable:<9} "
                f"{_format_number(block.worst_iat, '.7g'):>9} "
                f"{_format_number(block.block_efficiency, '.6e'):>12} "
                f"{_format_number(block.maximum_rhat, '.6f'):>8} "
                f"{_format_number(block.maximum_standardized_moment_error, '.6e'):>13} "
                f"{block.nonfinite_samples:9d} warmup_nonfinite_leaves={block.warmup_nonfinite_leaves} retained_nonfinite_leaves={block.retained_nonfinite_leaves}"
            )
            lines.append(
                "      adapted_eps="
                + np.array2string(
                    np.asarray(block.adapted_step_sizes),
                    precision=12,
                    separator=", ",
                )
                + f"; epsilon_search_gradients={block.epsilon_search_gradient_calls}"
                + f"; warmup_gradients={block.warmup_gradient_calls}"
                + f"; retained_gradients={block.retained_gradient_calls}"
            )

    if representative is None:
        lines.extend(["", "NO DEFINED REPRESENTATIVE", "Every configuration has an undefined required block score.", "END OF REPORT"])
        return "\n".join(lines)

    block_efficiencies = np.asarray(
        [block.block_efficiency for block in representative.block_results]
    )
    lines.extend(
        [
            "",
            "SELECTED FIXED NUTS REPRESENTATIVE",
            "-" * 88,
            f"target acceptance delta={representative.target_acceptance:.12g}",
            "The representative includes the fixed Algorithm 6 adaptation rule; epsilon remains chain-specific.",
            "Block efficiencies="
            + np.array2string(block_efficiencies, precision=12, separator=", "),
            f"Geometric mean={representative.geometric_mean_efficiency:.12e}",
            f"Minimum={representative.minimum_block_efficiency:.12e}",
            f"Median={representative.median_block_efficiency:.12e}",
            f"Maximum={representative.maximum_block_efficiency:.12e}",
            f"Mean retained acceptance statistic={representative.mean_acceptance_statistic:.12f}",
            f"Mean actual retained gradients/draw={representative.mean_retained_gradients_per_draw:.12f}",
            f"Retained divergences across tuning blocks={representative.total_retained_divergences}",
            f"Maximum-depth hits across tuning blocks={representative.total_retained_maximum_depth_hits}",
            f"Maximum R-hat across tuning blocks={representative.maximum_rhat:.12f}",
            f"Maximum standardized moment error across tuning blocks={representative.maximum_standardized_moment_error:.12e}",
            "This representative has not seen final-comparison seeds."
            if prepared.run_mode == "CANONICAL"
            else "This expanded representative has not seen new final-comparison seeds; compare only under this shared protocol.",
            "",
            "SELECTED REPRESENTATIVE: ALL OBSERVABLES IN ALL BLOCKS",
            "-" * 88,
            "block observable IAT status pairs last_lag first_excl ESS efficiency rank_Rhat fold_Rhat max_Rhat empirical_mean exact_mean raw_error moment_SD cutoff_reason",
        ]
    )
    for block in representative.block_results:
        for item in block.observable_diagnostics:
            lines.append(
                f"{block.block_index + 1:5d} {item.name:<9} "
                f"{_format_number(item.iat.tau, '.8g'):>10} "
                f"{item.iat.status:<30} {item.iat.pair_count:5d} "
                f"{item.iat.last_included_lag:8d} {item.iat.first_excluded_lag:10d} "
                f"{_format_number(item.ess, '.7g'):>10} "
                f"{_format_number(item.efficiency, '.7e'):>12} "
                f"{_format_number(item.rhat.rank, '.7f'):>9} "
                f"{_format_number(item.rhat.folded, '.7f'):>9} "
                f"{_format_number(item.rhat.combined, '.7f'):>9} "
                f"{item.empirical_mean:14.7e} {item.exact_mean:14.7e} "
                f"{item.raw_moment_error:13.6e} "
                f"{item.standardized_moment_error:9.6e} "
                f"{item.iat.cutoff_reason}"
            )
    lines.extend(["", "END OF REPORT"])
    return "\n".join(lines)


def _run_chain_job(
    seed: int,
    q_initial: np.ndarray,
    beta: float,
    target_acceptance: float,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
    maximum_tree_depth: int,
    delta_max: float,
    dual_averaging_gamma: float,
    dual_averaging_t0: float,
    dual_averaging_kappa: float,
    reasonable_epsilon_max_search_steps: int,
):
    return _simulate_nuts_chain(
        seed,
        q_initial,
        beta,
        target_acceptance,
        n_warmup,
        n_retained,
        target_parameters,
        mass_cholesky,
        inverse_mass_matrix,
        mass_is_identity,
        maximum_tree_depth,
        delta_max,
        dual_averaging_gamma,
        dual_averaging_t0,
        dual_averaging_kappa,
        reasonable_epsilon_max_search_steps,
    )


def _run_configuration_block(
    *,
    target_acceptance: float,
    block_index: int,
    seeds: Sequence[int],
    prepared: PreparedInputs,
) -> BlockResult:
    jobs = [
        (
            int(seeds[chain_index]),
            prepared.physical_starts[chain_index],
            prepared.beta,
            target_acceptance,
            prepared.n_warmup,
            prepared.n_retained,
            prepared.target_parameters,
            prepared.mass_cholesky,
            prepared.inverse_mass_matrix,
            prepared.mass_is_identity,
            prepared.maximum_tree_depth,
            prepared.delta_max,
            prepared.dual_averaging_gamma,
            prepared.dual_averaging_t0,
            prepared.dual_averaging_kappa,
            prepared.reasonable_epsilon_max_search_steps,
        )
        for chain_index in range(prepared.n_chains)
    ]

    # Numba owns a thread-local RNG state after np.random.seed(seed).  The
    # pure-Python fallback uses NumPy's process-global legacy RNG, so it must
    # remain serial to preserve chain-local reproducibility.
    if NUMBA_AVAILABLE and prepared.n_workers > 1:
        with ThreadPoolExecutor(
            max_workers=min(prepared.n_workers, prepared.n_chains)
        ) as executor:
            chain_outputs = list(executor.map(lambda arguments: _run_chain_job(*arguments), jobs))
    else:
        chain_outputs = [_run_chain_job(*arguments) for arguments in jobs]

    samples = np.stack([output[0] for output in chain_outputs], axis=0)
    adapted_step_sizes = tuple(output[1] for output in chain_outputs)
    mean_acceptance_statistic = float(
        np.mean([output[2] for output in chain_outputs])
    )
    epsilon_search_gradient_calls = sum(output[3] for output in chain_outputs)
    warmup_gradient_calls = sum(output[4] for output in chain_outputs)
    retained_gradient_calls = sum(output[5] for output in chain_outputs)
    retained_divergences = sum(output[6] for output in chain_outputs)
    retained_maximum_depth_hits = sum(output[7] for output in chain_outputs)
    mean_retained_tree_depth = float(
        np.mean([output[8] for output in chain_outputs])
    )
    mean_retained_leapfrog_steps = float(
        np.mean([output[9] for output in chain_outputs])
    )
    nonfinite_samples = sum(output[10] for output in chain_outputs)
    gradients_per_retained_draw = retained_gradient_calls / (
        prepared.n_chains * prepared.n_retained
    )
    if not math.isclose(
        gradients_per_retained_draw,
        mean_retained_leapfrog_steps,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "NUTS retained gradient count differs from executed leaf count."
        )

    return _analyze_block(
        samples=samples,
        target_acceptance=target_acceptance,
        block_index=block_index,
        seeds=seeds,
        mean_acceptance_statistic=mean_acceptance_statistic,
        adapted_step_sizes=adapted_step_sizes,
        retained_gradient_calls=retained_gradient_calls,
        epsilon_search_gradient_calls=epsilon_search_gradient_calls,
        warmup_gradient_calls=warmup_gradient_calls,
        retained_divergences=retained_divergences,
        retained_maximum_depth_hits=retained_maximum_depth_hits,
        mean_retained_tree_depth=mean_retained_tree_depth,
        mean_retained_leapfrog_steps=mean_retained_leapfrog_steps,
        nonfinite_samples=nonfinite_samples,
        warmup_nonfinite_leaves=sum(output[12] for output in chain_outputs),
        retained_nonfinite_leaves=sum(output[13] for output in chain_outputs),
        observable_names=prepared.observable_names,
        exact_observable_means=prepared.exact_observable_means,
        exact_observable_sds=prepared.exact_observable_sds,
    )


def run_self_tests(prepared: PreparedInputs) -> tuple[str, ...]:
    """Run deterministic implementation checks before a tuning campaign."""

    messages: list[str] = []
    dimension = prepared.dimension

    q_gradient_test = np.empty(dimension, dtype=np.float64)
    q_gradient_test[0] = -0.23
    q_gradient_test[1:] = np.linspace(
        0.21, -0.14, prepared.transverse_dimension
    )
    analytic_gradient = _target_gradient(q_gradient_test, prepared.target_parameters)
    finite_difference_gradient = np.empty(dimension, dtype=np.float64)
    for coordinate in range(dimension):
        finite_difference_step = 1.0e-6 * max(
            1.0, abs(q_gradient_test[coordinate])
        )
        q_plus = q_gradient_test.copy()
        q_minus = q_gradient_test.copy()
        q_plus[coordinate] += finite_difference_step
        q_minus[coordinate] -= finite_difference_step
        finite_difference_gradient[coordinate] = (
            _potential_energy(q_plus, prepared.target_parameters)
            - _potential_energy(q_minus, prepared.target_parameters)
        ) / (2.0 * finite_difference_step)
    if not np.allclose(
        analytic_gradient,
        finite_difference_gradient,
        rtol=2.0e-8,
        atol=2.0e-8,
    ):
        raise RuntimeError("Self-test failed: analytic target gradient.")
    reconstructed_starts = _physical_to_latent(
        prepared.physical_starts, prepared.beta
    )
    if not np.allclose(
        reconstructed_starts,
        prepared.funnel_starts,
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise RuntimeError("Self-test failed: funnel latent-start conversion.")
    if not np.allclose(
        prepared.mass_cholesky @ prepared.mass_cholesky.T,
        prepared.mass_matrix,
        rtol=1.0e-12,
        atol=1.0e-12,
    ) or not np.allclose(
        prepared.inverse_mass_matrix @ prepared.mass_matrix,
        np.eye(dimension),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("Self-test failed: mass factorization or inverse.")
    messages.append(
        "PASS: analytic funnel gradient, latent starts, and mass algebra"
    )

    if (
        prepared.beta == 1.0
        and prepared.sigma_y == 3.0
        and prepared.transverse_dimension == 5
    ):
        oracle_q = np.concatenate(
            (np.asarray([0.37]), np.linspace(-0.5, 0.75, 5))
        )
        oracle_potential = 1.2968599877275981
        oracle_gradient = np.asarray(
            [
                2.176856678939069,
                -0.3453671653186773,
                -0.1295126869945040,
                0.08634179132966934,
                0.3021962696538427,
                0.5180507479780160,
            ]
        )
        if not math.isclose(
            _potential_energy(oracle_q, prepared.target_parameters),
            oracle_potential,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        ) or not np.allclose(
            _target_gradient(oracle_q, prepared.target_parameters),
            oracle_gradient,
            rtol=0.0,
            atol=2.0e-13,
        ):
            raise RuntimeError("Self-test failed: literal centered-funnel oracle.")
        messages.append("PASS: literal canonical centered-funnel energy/gradient oracle")

    expected_variance_y = prepared.sigma_y * prepared.sigma_y / prepared.beta
    expected_variance_x = _safe_exp_target(
        0.5 * expected_variance_y
    ) / prepared.beta
    if not np.allclose(
        prepared.target_variances,
        np.asarray(
            [expected_variance_y]
            + [expected_variance_x] * prepared.transverse_dimension
        ),
        rtol=2.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("Self-test failed: exact funnel marginal variances.")
    observable_index = {
        name: index for index, name in enumerate(prepared.observable_names)
    }
    expected_checks = {
        "y^2": (expected_variance_y, math.sqrt(2.0) * expected_variance_y),
        "y*x1": (
            0.0,
            math.sqrt(
                (expected_variance_y + expected_variance_y**2)
                * _safe_exp_target(0.5 * expected_variance_y)
                / prepared.beta
            ),
        ),
        "x1^2": (
            expected_variance_x,
            math.sqrt(
                (
                    3.0 * _safe_exp_target(2.0 * expected_variance_y)
                    - _safe_exp_target(expected_variance_y)
                )
                / (prepared.beta * prepared.beta)
            ),
        ),
    }
    if prepared.transverse_dimension >= 2:
        expected_checks["x1*x2"] = (
            0.0,
            _safe_exp_target(expected_variance_y) / prepared.beta,
        )
    for name, (expected_mean, expected_sd) in expected_checks.items():
        index = observable_index[name]
        if not math.isclose(
            prepared.exact_observable_means[index],
            expected_mean,
            rel_tol=2.0e-15,
            abs_tol=2.0e-15,
        ) or not math.isclose(
            prepared.exact_observable_sds[index],
            expected_sd,
            rel_tol=2.0e-15,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                f"Self-test failed: exact funnel moment for {name}."
            )
    messages.append("PASS: exact centered-funnel linear and quadratic moments")

    if dimension == 6:
        mapping_state = np.asarray([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
        mapping_values = _physical_observables(
            mapping_state.reshape(1, 1, 6), 6
        )[0, 0]
        expected_mapping = np.asarray(
            [
                2.0, 3.0, 5.0, 7.0, 11.0, 13.0,
                4.0, 6.0, 10.0, 14.0, 22.0, 26.0,
                9.0, 15.0, 21.0, 33.0, 39.0,
                25.0, 35.0, 55.0, 65.0,
                49.0, 77.0, 91.0,
                121.0, 143.0,
                169.0,
            ]
        )
        if not np.array_equal(mapping_values, expected_mapping):
            raise RuntimeError("Self-test failed: observable value ordering.")
        expected_names = (
            "y", "x1", "x2", "x3", "x4", "x5",
            "y^2", "y*x1", "y*x2", "y*x3", "y*x4", "y*x5",
            "x1^2", "x1*x2", "x1*x3", "x1*x4", "x1*x5",
            "x2^2", "x2*x3", "x2*x4", "x2*x5",
            "x3^2", "x3*x4", "x3*x5",
            "x4^2", "x4*x5", "x5^2",
        )
        if prepared.observable_names != expected_names:
            raise RuntimeError("Self-test failed: observable name ordering.")
        messages.append("PASS: 27 observable names and values share one exact order")

    autocovariance_input = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]
    )
    expected_autocovariance = np.asarray(
        [
            [1.25, 0.3125, -0.375, -0.5625],
            [5.0, 1.25, -1.5, -2.25],
        ]
    )
    if not np.allclose(
        _autocovariance_fft(autocovariance_input),
        expected_autocovariance,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise RuntimeError("Self-test failed: biased FFT autocovariance.")
    odd_split_input = np.asarray(
        [[0.0, 1.0, 99.0, 2.0, 3.0], [10.0, 11.0, 99.0, 12.0, 13.0]]
    )
    expected_split = np.asarray(
        [[0.0, 1.0], [10.0, 11.0], [2.0, 3.0], [12.0, 13.0]]
    )
    if not np.array_equal(_split_chains(odd_split_input), expected_split):
        raise RuntimeError("Self-test failed: odd-length split convention.")
    expected_ranks = np.asarray([1.5, 1.5, 3.0, 4.0])
    if not np.array_equal(
        _average_ranks(np.asarray([10.0, 10.0, 20.0, 30.0])), expected_ranks
    ):
        raise RuntimeError("Self-test failed: average ranks for ties.")
    messages.append("PASS: biased autocovariance, odd split, and rejection-tie ranks")

    ppf_values = _standard_normal_ppf(np.asarray([0.025, 0.5, 0.975]))
    if not np.allclose(
        ppf_values,
        np.asarray([-1.959963984540054, 0.0, 1.959963984540054]),
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise RuntimeError("Self-test failed: inverse-normal approximation.")
    messages.append("PASS: inverse-normal approximation and symmetry")

    rng = np.random.default_rng(24681357)
    iid = rng.standard_normal((4, 6000))
    iid_rhat = _rank_normalized_split_folded_rhat(iid)
    if iid_rhat.status != "ok" or not (0.98 < iid_rhat.combined < 1.03):
        raise RuntimeError("Self-test failed: IID rank-normalized R-hat.")
    shifted = iid.copy()
    shifted[0] += 2.0
    shifted_rhat = _rank_normalized_split_folded_rhat(shifted)
    if shifted_rhat.status != "ok" or shifted_rhat.combined <= 1.1:
        raise RuntimeError("Self-test failed: shifted-chain R-hat sensitivity.")
    unequal_scale = iid * np.asarray([1.0, 1.0, 4.0, 4.0])[:, np.newaxis]
    scale_rhat = _rank_normalized_split_folded_rhat(unequal_scale)
    if scale_rhat.status != "ok" or scale_rhat.folded <= scale_rhat.rank + 0.05:
        raise RuntimeError("Self-test failed: folded R-hat scale sensitivity.")
    messages.append(
        "PASS: rank-normalized split/folded R-hat IID, location, and scale checks"
    )

    phi = 0.8
    ar = np.empty((4, 20000), dtype=np.float64)
    ar[:, 0] = rng.normal(scale=1.0 / math.sqrt(1.0 - phi * phi), size=4)
    innovations = rng.standard_normal((4, ar.shape[1] - 1))
    for draw in range(1, ar.shape[1]):
        ar[:, draw] = phi * ar[:, draw - 1] + innovations[:, draw - 1]
    ar_iat = _multi_chain_geyer_iat(ar)
    if not ar_iat.is_defined or not (6.0 < ar_iat.tau < 12.5):
        raise RuntimeError("Self-test failed: AR(1) Geyer IAT.")
    constant_iat = _multi_chain_geyer_iat(np.ones((4, 100), dtype=np.float64))
    if constant_iat.status != "undefined_zero_variance":
        raise RuntimeError("Self-test failed: constant-series IAT status.")
    nonfinite_iat = _multi_chain_geyer_iat(
        np.asarray([[0.0, 1.0, math.nan, 2.0]] * 4)
    )
    if nonfinite_iat.status != "undefined_nonfinite_input":
        raise RuntimeError("Self-test failed: nonfinite-series IAT status.")

    negative_phi = -0.5
    antithetic = np.empty((4, 20000), dtype=np.float64)
    antithetic[:, 0] = rng.normal(
        scale=1.0 / math.sqrt(1.0 - negative_phi * negative_phi), size=4
    )
    antithetic_innovations = rng.standard_normal((4, antithetic.shape[1] - 1))
    for draw in range(1, antithetic.shape[1]):
        antithetic[:, draw] = (
            negative_phi * antithetic[:, draw - 1]
            + antithetic_innovations[:, draw - 1]
        )
    antithetic_iat = _multi_chain_geyer_iat(antithetic)
    if not antithetic_iat.is_defined or not (0.15 < antithetic_iat.tau < 0.65):
        raise RuntimeError("Self-test failed: unfloored antithetic Geyer IAT.")
    messages.append(
        f"PASS: Geyer IAT positive AR(1)={ar_iat.tau:.6f}, antithetic AR(1)={antithetic_iat.tau:.6f}, and undefined statuses"
    )

    q0 = np.linspace(0.3, -0.2, dimension) * np.sqrt(prepared.target_variances)
    p0 = prepared.mass_cholesky @ np.linspace(0.7, -0.3, dimension)
    test_epsilon = 1.0e-4
    grad0 = _target_gradient(q0, prepared.target_parameters)
    q1, p1, grad1, _u1, finite1 = _leapfrog_one(
        q0, p0, grad0, test_epsilon, prepared.target_parameters,
        prepared.inverse_mass_matrix, prepared.mass_is_identity,
    )
    q2, p2, _grad2, _u2, finite2 = _leapfrog_one(
        q1, p1, grad1, -test_epsilon, prepared.target_parameters,
        prepared.inverse_mass_matrix, prepared.mass_is_identity,
    )
    if (
        not finite1
        or not finite2
        or not np.allclose(q2, q0, rtol=0.0, atol=2.0e-12)
        or not np.allclose(p2, p0, rtol=0.0, atol=2.0e-12)
    ):
        raise RuntimeError("Self-test failed: one-step leapfrog reversibility.")

    adversarial_mass = np.diag(np.asarray([100.0, 1.0]))
    adversarial_inverse = np.linalg.inv(adversarial_mass)
    uturn_q_minus = np.asarray([0.0, 0.0])
    uturn_q_plus = np.asarray([1.0, 1.0])
    uturn_p = np.asarray([1.0, -0.5])
    if float((uturn_q_plus - uturn_q_minus) @ uturn_p) <= 0.0 or not _has_u_turn(
        uturn_q_minus,
        uturn_p,
        uturn_q_plus,
        uturn_p,
        adversarial_inverse,
        False,
    ):
        raise RuntimeError("Self-test failed: velocity-based general-mass U-turn.")
    if _has_u_turn(
        np.asarray([0.0]), np.asarray([1.0]), np.asarray([1.0]),
        np.asarray([0.0]), np.eye(1), True,
    ):
        raise RuntimeError("Self-test failed: zero U-turn product must not stop.")
    messages.append(
        "PASS: leapfrog reversibility and identity/general-mass velocity U-turn rule"
    )

    tree_q = np.asarray([-0.5, 0.0], dtype=np.float64)
    tree_p = np.asarray([0.2, -0.1])
    tree_target_parameters = np.asarray([1.0, 1.0, 1.0])
    tree_gradient = _target_gradient(tree_q, tree_target_parameters)
    tree_potential = _potential_energy(tree_q, tree_target_parameters)
    tree_joint = _joint_log_density(tree_potential, tree_p, 1.0, np.eye(2), True)
    tree = _build_tree(
        tree_q, tree_p, tree_gradient, tree_potential, tree_joint - 100.0,
        1, 3, 1.0e-3, tree_joint, 1.0, tree_target_parameters,
        np.eye(2), True, 1000.0,
    )
    if tree[15] != 8 or tree[14] != 8 or tree[11] != 8 or not tree[12]:
        raise RuntimeError("Self-test failed: balanced tree leaf accounting.")
    messages.append(
        "PASS: depth-3 balanced tree has 8 executed/valid leaves and 8 adaptation states"
    )

    acceptance_sequence = (0.90, 0.20, 1.00, 0.65, 0.80)
    mu = math.log(0.5)
    h_bar = 0.0
    log_epsilon_bar = 0.0
    manual_h = h_bar
    manual_bar = log_epsilon_bar
    for warmup_index, acceptance_value in enumerate(acceptance_sequence, start=1):
        h_bar, log_epsilon, log_epsilon_bar = _dual_averaging_update(
            warmup_index, acceptance_value, 0.60, mu, h_bar,
            log_epsilon_bar, 0.05, 10.0, 0.75,
        )
        weight = 1.0 / (warmup_index + 10.0)
        manual_h = (1.0 - weight) * manual_h + weight * (0.60 - acceptance_value)
        manual_log_epsilon = mu - math.sqrt(warmup_index) * manual_h / 0.05
        average_weight = warmup_index ** (-0.75)
        manual_bar = average_weight * manual_log_epsilon + (
            1.0 - average_weight
        ) * manual_bar
        if not (
            math.isclose(h_bar, manual_h, rel_tol=0.0, abs_tol=1.0e-15)
            and math.isclose(log_epsilon, manual_log_epsilon, rel_tol=0.0, abs_tol=1.0e-14)
            and math.isclose(log_epsilon_bar, manual_bar, rel_tol=0.0, abs_tol=1.0e-14)
        ):
            raise RuntimeError("Self-test failed: dual-averaging recurrence.")
    messages.append("PASS: Algorithm 6 dual-averaging recurrence")

    smoke_arguments = (
        998877,
        prepared.physical_starts[0],
        prepared.beta,
        0.80,
        16,
        32,
        prepared.target_parameters,
        prepared.mass_cholesky,
        prepared.inverse_mass_matrix,
        prepared.mass_is_identity,
        5,
        prepared.delta_max,
        prepared.dual_averaging_gamma,
        prepared.dual_averaging_t0,
        prepared.dual_averaging_kappa,
        prepared.reasonable_epsilon_max_search_steps,
    )
    output = _simulate_nuts_chain(*smoke_arguments)
    repeated_output = _simulate_nuts_chain(*smoke_arguments)
    different_seed_output = _simulate_nuts_chain(998878, *smoke_arguments[1:])
    if (
        output[0].shape != (32, dimension)
        or output[11] != 1
        or not np.array_equal(output[0], repeated_output[0])
        or output[1:] != repeated_output[1:]
        or np.array_equal(output[0], different_seed_output[0])
        or not math.isfinite(output[1])
        or output[1] <= 0.0
        or output[5] != int(round(output[9] * 32))
        or output[10] != 0
    ):
        raise RuntimeError(
            "Self-test failed: deterministic NUTS chain or gradient accounting."
        )
    messages.append(
        "PASS: deterministic chain-local RNG, frozen adapted epsilon, retained leaf-gradient accounting, and finite smoke chain"
    )

    expected_observable_count = dimension + dimension * (dimension + 1) // 2
    if len(prepared.observable_names) != expected_observable_count or len(
        set(prepared.observable_names)
    ) != expected_observable_count:
        raise RuntimeError("Self-test failed: required observable set.")
    synthetic_samples = rng.standard_normal(
        (prepared.n_chains, 120, dimension)
    ) * np.sqrt(prepared.target_variances)[np.newaxis, np.newaxis, :]
    synthetic_samples[0, 0, 0] = math.nan
    invalid_block = _analyze_block(
        samples=synthetic_samples,
        target_acceptance=0.8,
        block_index=0,
        seeds=prepared.seed_blocks[0],
        mean_acceptance_statistic=0.8,
        adapted_step_sizes=(0.01,) * prepared.n_chains,
        retained_gradient_calls=prepared.n_chains * 120 * 15,
        epsilon_search_gradient_calls=20,
        warmup_gradient_calls=100,
        retained_divergences=1,
        retained_maximum_depth_hits=0,
        mean_retained_tree_depth=4.0,
        mean_retained_leapfrog_steps=15.0,
        nonfinite_samples=1,
        observable_names=prepared.observable_names,
        exact_observable_means=prepared.exact_observable_means,
        exact_observable_sds=prepared.exact_observable_sds,
    )
    if invalid_block.status != "undefined_nonfinite_simulation" or any(
        math.isfinite(item.efficiency)
        for item in invalid_block.observable_diagnostics
    ):
        raise RuntimeError("Self-test failed: nonfinite block efficiency propagation.")
    valid_block_a = replace(
        invalid_block,
        status="ok",
        block_index=0,
        block_efficiency=0.25,
        nonfinite_samples=0,
    )
    valid_block_b = replace(valid_block_a, block_index=1, block_efficiency=1.0)
    valid_aggregate = _aggregate_blocks((valid_block_a, valid_block_b))
    invalid_aggregate = _aggregate_blocks((invalid_block,))
    if (
        valid_aggregate.status != "ok"
        or not math.isclose(
            valid_aggregate.geometric_mean_efficiency,
            0.5,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or invalid_aggregate.status != "undefined_block_metric"
        or not math.isnan(invalid_aggregate.geometric_mean_efficiency)
    ):
        raise RuntimeError("Self-test failed: common across-block GM aggregation.")
    messages.append(
        f"PASS: {expected_observable_count}-observable exact moments, nonfinite propagation, and common block GM"
    )
    # Observational counters must not conflate finite-energy divergence with
    # a nonfinite step. Repeat these checks on the GREX native runtime.
    bad_tree = _build_tree(
        tree_q, tree_p, tree_gradient, tree_potential, tree_joint - 1.0,
        1, 0, 1.0e200, tree_joint, 1.0, tree_target_parameters,
        np.eye(2), True, 1000.0,
    )
    finite_divergent_tree = _build_tree(
        tree_q, tree_p, tree_gradient, tree_potential, tree_joint + 100.0,
        1, 0, 1.0e-3, tree_joint, 1.0, tree_target_parameters,
        np.eye(2), True, 1.0,
    )
    if bad_tree[17] != 1 or finite_divergent_tree[17] != 0 or tree[17] != 0:
        raise RuntimeError("Self-test failed: nonfinite leaf counters.")
    if finite_divergent_tree[16] != 1 or bad_tree[16] != 1:
        raise RuntimeError("Self-test failed: divergence accounting.")
    messages.append("PASS: separate nonfinite-leaf and finite-energy-divergence counters")
    return tuple(messages)


def run_nuts_tuning(
    inputs: RunInputs,
    *,
    preflight_only: bool = False,
) -> dict[str, object]:
    """Run preflight or one completely specified NUTS tuning campaign."""

    if not isinstance(preflight_only, (bool, np.bool_)):
        raise ValueError("preflight_only must be True or False.")
    prepared = _prepare_inputs(inputs)
    _validate_run_mode(prepared)
    self_test_messages = (
        run_self_tests(prepared)
        if prepared.run_self_tests
        else ("SKIPPED: run_self_tests=False (CUSTOM mode only).",)
    )

    if preflight_only:
        message = (
            "PASS: portable cache-free original slice-NUTS preflight; target, "
            "gradient, mass mechanics, recursive tree, U-turn rule, dual "
            "averaging, RNG reproducibility, gradient accounting, diagnostics, "
            "and finite smoke chain passed. No tuning configuration was run "
            "and no report was written. Set GREX_PREFLIGHT_ONLY=False and "
            "execute the entire cell again for production tuning."
        )
        print("\n".join(self_test_messages))
        print(message)
        return {
            "preflight_only": True,
            "message": message,
            "prepared_inputs": prepared,
            "self_test_messages": self_test_messages,
        }

    destination = _resolve_output_destination(prepared)
    configuration_results: list[ConfigurationResult] = []
    total_configurations = len(prepared.target_acceptance_values)
    for configuration_number, target_acceptance in enumerate(
        prepared.target_acceptance_values, start=1
    ):
        if prepared.print_progress:
            print(
                f"[{configuration_number:>2}/{total_configurations}] "
                f"NUTS delta={target_acceptance:.6g}: running "
                f"{len(prepared.seed_blocks)} tuning blocks...",
                flush=True,
            )
        block_results: list[BlockResult] = []
        for block_index, seeds in enumerate(prepared.seed_blocks):
            block_results.append(
                _run_configuration_block(
                    target_acceptance=target_acceptance,
                    block_index=block_index,
                    seeds=seeds,
                    prepared=prepared,
                )
            )
        aggregate = _aggregate_configuration(
            target_acceptance, block_results
        )
        configuration_results.append(aggregate)
        if prepared.print_progress:
            print(
                "    status="
                + aggregate.status
                + ", GM efficiency="
                + _format_number(aggregate.geometric_mean_efficiency, ".8e"),
                flush=True,
            )

    representative = (_select_representative(configuration_results)
                      if any(r.status == "ok" for r in configuration_results) else None)
    report = _build_report(
        configuration_results=configuration_results,
        representative=representative,
        prepared=prepared,
        self_test_messages=self_test_messages,
    )
    destination.write_text(report + "\n", encoding="utf-8")

    if prepared.print_report:
        print("\n" + report)
        print(f"\nThe identical report was saved to:\n{destination}")

    return {
        "preflight_only": False,
        "configuration_results": tuple(configuration_results),
        "representative": representative,
        "report": report,
        "output_path": destination,
        "prepared_inputs": prepared,
        "input_fingerprint": prepared.common_input_fingerprint,
        "input_preparation_fingerprint": _input_preparation_fingerprint(),
        "shared_protocol_fingerprint": _shared_protocol_fingerprint(prepared),
        "diagnostics_fingerprint": _diagnostics_fingerprint(),
        "common_pipeline_fingerprint": _common_pipeline_fingerprint(),
        "nuts_implementation_fingerprint": _nuts_implementation_fingerprint(),
        "source_verification_mode": _source_verification_mode(),
    }


# Tuning inputs.

GREX_PREFLIGHT_ONLY = True

USER_INPUTS = RunInputs(
    # ----- Target and shared physical geometry -----
    beta=1.0,
    sigma_y=3.0,
    dimension_x=5,
    mass_matrix=np.eye(6, dtype=np.float64),

    # Enter (y,z_1,...,z_5).  The code derives x_i=exp(y/2)*z_i/sqrt(beta).
    funnel_starts=np.asarray(
        [
            [-5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [-2.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            [2.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [5.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    ),
    observable_policy="ALL_LINEAR_AND_UNIQUE_QUADRATIC",

    # ----- Original Algorithm 6 NUTS tuning design -----
    target_acceptance_values=(0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
    maximum_tree_depth=12,
    delta_max=1000.0,
    dual_averaging_gamma=0.05,
    dual_averaging_t0=10.0,
    dual_averaging_kappa=0.75,
    reasonable_epsilon_max_search_steps=100,

    # ----- Tuning seed blocks; one seed per chain -----
    seed_blocks=(
        (3857660299, 1344569885, 2010517369, 2974962254),
        (2914251580, 2216611040, 1467312042, 1966646752),
        (26638442, 2162183804, 2288357861, 3874567655),
        (917698748, 796766395, 2769691025, 1469822251),
        (983213432, 4224231832, 2042092071, 4006032910),
    ),

    # ----- Per-chain run lengths and execution controls -----
    n_warmup=1000,
    n_retained=4000,
    n_workers=4,
    run_self_tests=True,
    print_progress=True,
    print_report=True,

    # ----- Run identity and TXT output -----
    run_mode="CANONICAL",
    run_label="PRODUCTION TUNING",
    # None -> beside this file for scripts/%run/import; pasted Jupyter -> kernel cwd.
    output_directory=None,
    output_filename="NUTS_Neals_funnel_tuning_results_GREX_PORTABLE_V1.txt",
    allow_overwrite=True,
)


EXPANDED_DELTA_VALUES = (
    .10, .15, .20, .25, .30, .35, .40, .425, .45, .475, .50, .525,
    .55, .575, .60, .625, .65, .675, .70, .725, .75, .775, .80, .825,
    .85, .875, .90, .925, .95, .96, .97, .98, .99, .995, .999,
)
REJECTION_POLICY = "STRICT_ANY_NONFINITE_WARMUP_OR_RETAINED_PROPOSAL_UNDEFINED"
EXPANDED_SEED_DOMAIN = "FUNNEL-HMC-SCALAR-CONDITIONAL-EXPANDED-20260925-TUNING"
EXPANDED_ADDED_BLOCKS = tuple(tuple(int.from_bytes(hashlib.sha256(
    f"{EXPANDED_SEED_DOMAIN}:block={b}:chain={c}".encode("ascii")).digest()[:4], "big")
    for c in range(1,5)) for b in range(6,11))
USER_INPUTS = replace(
    USER_INPUTS, target_acceptance_values=EXPANDED_DELTA_VALUES,
    seed_blocks=tuple(USER_INPUTS.seed_blocks)+EXPANDED_ADDED_BLOCKS,
    n_warmup=1000, n_retained=10000, n_workers=4,
    run_mode="CUSTOM",
    run_label="CUSTOM: FUNNEL6-EXPANDED-20260925 NUTS TUNING ONLY",
    output_filename="NUTS_Neals_funnel_expanded_tuning.txt",
    allow_overwrite=False,
)
assert len(USER_INPUTS.seed_blocks) == 10
assert len(set(seed for b in USER_INPUTS.seed_blocks for seed in b)) == 40


if __name__ == "__main__":
    run_nuts_tuning(USER_INPUTS, preflight_only=GREX_PREFLIGHT_ONLY)
