"""Independent frozen NUTS held-out run on the 5D Gaussian.
Default: preflight only. Production: python NUTS.py --run
No tuning or scientific overrides. NumPy and Numba required.
Reports and checkpoints are TXT files beside this standalone script.
"""

# BEGIN FROZEN ENGINE
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

NUMBA_DIRECT_RUN = __name__ == "__main__"

if NUMBA_DIRECT_RUN:
    # Numba's IPython cache locator ignores NUMBA_CACHE_DIR.  Disable only
    # persistent disk caching in a direct script/Jupyter run; JIT compilation
    # still occurs once in memory for the production campaign.
    NUMBA_CACHE_MODE = "persistent cache disabled for direct script/Jupyter run"
    NUMBA_CACHE_DIRECTORY = "not used"
elif os.environ.get("NUMBA_CACHE_DIR"):
    NUMBA_CACHE_MODE = "caller-provided isolated cache"
    NUMBA_CACHE_DIRECTORY = os.environ["NUMBA_CACHE_DIR"]
else:
    NUMBA_CACHE_MODE = "fresh private import cache"
    NUMBA_CACHE_DIRECTORY = tempfile.mkdtemp(prefix="msg5_nuts_tuner_")
    os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIRECTORY

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

SHARED_PROTOCOL_VERSION = "MSG5-expanded-10blocks-20260925"

COMMON_INPUT_SCHEMA_VERSION = "expanded-NUTS-inputs-20260925"

INPUT_PREPARATION_VERSION = "validated-derived-NUTS-inputs-v1.0.0"

DIAGNOSTICS_VERSION = "multichain-geyer-ims-rank-rhat-v1.0.0"

COMMON_PIPELINE_VERSION = "msg-NUTS-observables-score-v1.0.0"

NUTS_IMPLEMENTATION_VERSION = "hoffman-gelman-2014-slice-NUTS-algorithm6-v1.0.0"

RANDOM_STREAM_VERSION = "numba-chain-local-numpy-rng-v1.0.0"

SUPPORTED_OBSERVABLE_POLICY = "ALL_LINEAR_AND_UNIQUE_QUADRATIC"

CANONICAL_OUTPUT_FILENAME = "NUTS_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt"

@dataclass(frozen=True)
class RunInputs:
    """Every independent scientific and execution input for one run."""

    beta: float
    omegas: Sequence[float]
    mass_matrix: Sequence[Sequence[float]]
    standardized_starts: Sequence[Sequence[float]]
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
    omegas: np.ndarray
    omega_squared: np.ndarray
    target_variances: np.ndarray
    mass_matrix: np.ndarray
    inverse_mass_matrix: np.ndarray
    mass_cholesky: np.ndarray
    mass_is_identity: bool
    standardized_starts: np.ndarray
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

@njit(cache=True, inline="always")
def _potential_energy(q: np.ndarray, omega_squared: np.ndarray) -> float:
    value = 0.0
    for coordinate in range(q.size):
        value += 0.5 * omega_squared[coordinate] * q[coordinate] * q[coordinate]
    return value

@njit(cache=True, inline="always")
def _target_gradient(q: np.ndarray, omega_squared: np.ndarray) -> np.ndarray:
    gradient = np.empty(q.size, dtype=np.float64)
    for coordinate in range(q.size):
        gradient[coordinate] = omega_squared[coordinate] * q[coordinate]
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
    omega_squared: np.ndarray,
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

    new_gradient = _target_gradient(new_q, omega_squared)
    for coordinate in range(new_p.size):
        new_p[coordinate] -= 0.5 * signed_epsilon * new_gradient[coordinate]
    new_potential = _potential_energy(new_q, omega_squared)
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
    omega_squared: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
    delta_max: float,
):
    """Recursive memory-efficient BuildTree from original Algorithm 3/6.

    Return layout:
      q-,p-,grad-,U-, q+,p+,grad+,U+, q*,grad*,U*,
      n_valid, continue, alpha_sum, n_alpha, leapfrog_steps, divergences.
    """

    if depth == 0:
        new_q, new_p, new_gradient, new_potential, finite = _leapfrog_one(
            q,
            p,
            gradient,
            direction * epsilon,
            omega_squared,
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
            n_valid, continue_tree, alpha, 1, 1, int(not continue_tree),
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
        omega_squared,
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
            omega_squared, inverse_mass_matrix, mass_is_identity, delta_max,
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
            omega_squared, inverse_mass_matrix, mass_is_identity, delta_max,
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
    omega_squared: np.ndarray,
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
        omega_squared,
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
            omega_squared,
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
    omega_squared: np.ndarray,
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
                omega_squared,
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
                omega_squared,
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
    omega_squared: np.ndarray,
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
    potential = _potential_energy(q, omega_squared)
    gradient = _target_gradient(q, omega_squared)

    initial_epsilon, search_leaf_gradients, search_success = (
        _find_reasonable_epsilon(
            q,
            gradient,
            potential,
            beta,
            omega_squared,
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
            0, 0, math.nan, math.nan, n_retained, 0,
        )

    mu = math.log(10.0 * initial_epsilon)
    log_epsilon = math.log(initial_epsilon)
    # Printed Algorithm 6 initializes epsilon_bar_0=1.
    log_epsilon_bar = 0.0
    h_bar = 0.0
    warmup_gradient_calls = 0

    for warmup_index in range(1, n_warmup + 1):
        epsilon = _safe_exp_step_size(log_epsilon)
        result = _nuts_iteration(
            q,
            gradient,
            potential,
            epsilon,
            beta,
            omega_squared,
            mass_cholesky,
            inverse_mass_matrix,
            mass_is_identity,
            maximum_tree_depth,
            delta_max,
        )
        q, gradient, potential = result[0], result[1], result[2]
        warmup_gradient_calls += result[5]
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

    for retained_index in range(n_retained):
        result = _nuts_iteration(
            q,
            gradient,
            potential,
            adapted_epsilon,
            beta,
            omega_squared,
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
    """Multi-chain Geyer initial-positive and initial-monotone IAT.

    The autocorrelation estimate uses the between-chain variance correction
    employed by modern multi-chain ESS estimators.  Adjacent sums
    P_k = rho_(2k) + rho_(2k+1) are retained up to, but not including, the
    first nonpositive pair.  Retained pair sums are monotonized by cumulative
    minima.  No lower floor is imposed on the resulting IAT.
    """

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
    """Return names, exact means, and exact target standard deviations."""

    target_variances = np.asarray(target_variances, dtype=np.float64)
    if target_variances.ndim != 1 or target_variances.size == 0:
        raise ValueError("target_variances must be a nonempty one-dimensional array.")
    dimension = int(target_variances.size)
    names: list[str] = []
    exact_means: list[float] = []
    exact_standard_deviations: list[float] = []

    for coordinate in range(dimension):
        names.append(f"q{coordinate + 1}")
        exact_means.append(0.0)
        exact_standard_deviations.append(math.sqrt(target_variances[coordinate]))

    for first in range(dimension):
        for second in range(first, dimension):
            if first == second:
                names.append(f"q{first + 1}^2")
                exact_means.append(float(target_variances[first]))
                exact_standard_deviations.append(
                    math.sqrt(2.0) * float(target_variances[first])
                )
            else:
                names.append(f"q{first + 1}*q{second + 1}")
                exact_means.append(0.0)
                exact_standard_deviations.append(
                    math.sqrt(float(target_variances[first]))
                    * math.sqrt(float(target_variances[second]))
                )

    return (
        tuple(names),
        np.asarray(exact_means, dtype=np.float64),
        np.asarray(exact_standard_deviations, dtype=np.float64),
    )

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
) -> BlockResult:
    """Calculate the complete agreed diagnostics for one tuning block."""

    samples = np.asarray(samples, dtype=np.float64)
    total_retained = samples.shape[0] * samples.shape[1]
    gradients_per_draw = retained_gradient_calls / total_retained
    dimension = int(samples.shape[2])
    observable_values = _physical_observables(samples, dimension)
    diagnostics: list[ObservableDiagnostic] = []
    simulation_is_nonfinite = nonfinite_samples > 0 or not np.all(np.isfinite(samples))

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
    omegas: np.ndarray,
    mass_matrix: np.ndarray,
    standardized_starts: np.ndarray,
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
    return {
        "target": {
            "dimension": int(omegas.size),
            "beta": float(beta),
            "omegas": omegas.tolist(),
        },
        "geometry": {"mass_matrix": mass_matrix.tolist()},
        "starts": {"standardized_starts": standardized_starts.tolist()},
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
        "observable_policy": {"policy": observable_policy},
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

    try:
        omegas = np.asarray(inputs.omegas, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("omegas must be a numeric one-dimensional vector.") from error
    if omegas.ndim != 1 or omegas.size == 0:
        raise ValueError(
            f"omegas must be a nonempty one-dimensional vector; received shape {omegas.shape}."
        )
    if not np.all(np.isfinite(omegas)) or np.any(omegas <= 0.0):
        raise ValueError("Every omega must be finite and strictly positive.")
    omegas = np.ascontiguousarray(omegas.copy())
    dimension = int(omegas.size)

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
        standardized_starts = np.asarray(
            inputs.standardized_starts, dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError("standardized_starts must be a numeric matrix.") from error
    if standardized_starts.ndim != 2 or standardized_starts.shape[1] != dimension:
        raise ValueError(
            "standardized_starts must have shape (n_chains, dimension); "
            f"received {standardized_starts.shape} for dimension {dimension}."
        )
    if standardized_starts.shape[0] < 2:
        raise ValueError("At least two chain starts are required.")
    if not np.all(np.isfinite(standardized_starts)):
        raise ValueError("standardized_starts must contain only finite values.")
    standardized_starts = np.ascontiguousarray(standardized_starts.copy())
    n_chains = int(standardized_starts.shape[0])

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
        omega_squared = np.ascontiguousarray(omegas * omegas)
        target_variances = np.ascontiguousarray(1.0 / (beta * omega_squared))
        physical_starts = np.ascontiguousarray(
            standardized_starts * np.sqrt(target_variances)[np.newaxis, :]
        )
    if not np.all(np.isfinite(omega_squared)) or np.any(omega_squared <= 0.0):
        raise ValueError("Derived omega_squared values must be finite and positive.")
    if not np.all(np.isfinite(target_variances)) or np.any(
        target_variances <= 0.0
    ):
        raise ValueError(
            "Derived target variances must be finite and positive; check beta and omegas."
        )
    if not np.all(np.isfinite(physical_starts)):
        raise ValueError(
            "Derived physical starts are nonfinite; check beta, omegas, and standardized_starts."
        )
    observable_names, exact_observable_means, exact_observable_sds = (
        _observable_specification(target_variances)
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
        omegas=omegas,
        mass_matrix=mass_matrix,
        standardized_starts=standardized_starts,
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
    canonical_omegas = np.asarray(
        [1.0, 2.340347319320716, 5.477225575051661, 12.818610191887021, 30.0]
    )
    canonical_starts = np.asarray(
        [
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [2.0, -2.0, 2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0, 2.0, -2.0],
            [-2.0, -2.0, -2.0, -2.0, -2.0],
        ]
    )
    canonical_seed_blocks = (
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
    )
    canonical_checks = {
        "target": beta == 1.0 and np.array_equal(omegas, canonical_omegas),
        "geometry": np.array_equal(mass_matrix, np.eye(5)),
        "starts": np.array_equal(standardized_starts, canonical_starts),
        "nuts_method": (
            target_acceptance_values == (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999)
            and maximum_tree_depth == 12
            and delta_max == 1000.0
            and dual_averaging_gamma == 0.05
            and dual_averaging_t0 == 10.0
            and dual_averaging_kappa == 0.75
            and reasonable_epsilon_max_search_steps == 100
        ),
        "seed_blocks": seed_blocks == canonical_seed_blocks,
        "run_lengths": n_warmup == 1000 and n_retained == 10000,
        "observable_policy": observable_policy == SUPPORTED_OBSERVABLE_POLICY,
    }
    canonical_mismatches = tuple(
        name for name, matches in canonical_checks.items() if not matches
    )
    common_input_fingerprint = _json_fingerprint(
        {"schema": COMMON_INPUT_SCHEMA_VERSION, "components": components}
    )

    for array in (
        omegas,
        omega_squared,
        target_variances,
        mass_matrix,
        inverse_mass_matrix,
        mass_cholesky,
        standardized_starts,
        physical_starts,
        exact_observable_means,
        exact_observable_sds,
    ):
        array.setflags(write=False)

    return PreparedInputs(
        beta=beta,
        omegas=omegas,
        omega_squared=omega_squared,
        target_variances=target_variances,
        mass_matrix=mass_matrix,
        inverse_mass_matrix=inverse_mass_matrix,
        mass_cholesky=mass_cholesky,
        mass_is_identity=mass_is_identity,
        standardized_starts=standardized_starts,
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
            _common_input_components,
            _prepare_inputs,
        )
    )

def _nuts_implementation_fingerprint() -> str:
    """Hash NUTS dynamics, adaptation, orchestration, and tie-breaking code."""

    return _source_fingerprint(
        (
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

def _run_chain_job(
    seed: int,
    q_initial: np.ndarray,
    beta: float,
    target_acceptance: float,
    n_warmup: int,
    n_retained: int,
    omega_squared: np.ndarray,
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
        omega_squared,
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
            prepared.omega_squared,
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
        observable_names=prepared.observable_names,
        exact_observable_means=prepared.exact_observable_means,
        exact_observable_sds=prepared.exact_observable_sds,
    )

def run_self_tests(prepared: PreparedInputs) -> tuple[str, ...]:
    """Run deterministic implementation checks before a tuning campaign."""

    messages: list[str] = []
    dimension = prepared.dimension

    standardized_gradient_point = np.linspace(0.25, -0.15, dimension)
    q_gradient_test = standardized_gradient_point * np.sqrt(
        prepared.target_variances
    )
    analytic_gradient = _target_gradient(q_gradient_test, prepared.omega_squared)
    finite_difference_gradient = np.empty(dimension, dtype=np.float64)
    for coordinate in range(dimension):
        finite_difference_step = 1.0e-6 * math.sqrt(
            prepared.target_variances[coordinate]
        )
        q_plus = q_gradient_test.copy()
        q_minus = q_gradient_test.copy()
        q_plus[coordinate] += finite_difference_step
        q_minus[coordinate] -= finite_difference_step
        finite_difference_gradient[coordinate] = (
            _potential_energy(q_plus, prepared.omega_squared)
            - _potential_energy(q_minus, prepared.omega_squared)
        ) / (2.0 * finite_difference_step)
    if not np.allclose(
        analytic_gradient,
        finite_difference_gradient,
        rtol=2.0e-8,
        atol=2.0e-8,
    ):
        raise RuntimeError("Self-test failed: analytic target gradient.")
    reconstructed_starts = prepared.physical_starts / np.sqrt(
        prepared.target_variances
    )[np.newaxis, :]
    if not np.allclose(reconstructed_starts, prepared.standardized_starts):
        raise RuntimeError("Self-test failed: standardized-start conversion.")
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
        "PASS: analytic target gradient, standardized starts, and mass algebra"
    )

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
    grad0 = _target_gradient(q0, prepared.omega_squared)
    q1, p1, grad1, _u1, finite1 = _leapfrog_one(
        q0, p0, grad0, test_epsilon, prepared.omega_squared,
        prepared.inverse_mass_matrix, prepared.mass_is_identity,
    )
    q2, p2, _grad2, _u2, finite2 = _leapfrog_one(
        q1, p1, grad1, -test_epsilon, prepared.omega_squared,
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

    tree_q = np.zeros(2, dtype=np.float64)
    tree_p = np.asarray([0.2, -0.1])
    tree_omega_squared = np.asarray([1.0, 1.0])
    tree_gradient = _target_gradient(tree_q, tree_omega_squared)
    tree_potential = _potential_energy(tree_q, tree_omega_squared)
    tree_joint = _joint_log_density(tree_potential, tree_p, 1.0, np.eye(2), True)
    tree = _build_tree(
        tree_q, tree_p, tree_gradient, tree_potential, tree_joint - 100.0,
        1, 3, 1.0e-3, tree_joint, 1.0, tree_omega_squared,
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
        prepared.omega_squared,
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
    return tuple(messages)
# END FROZEN ENGINE

HELDOUT_METHOD = 'NUTS'
HELDOUT_PROTOCOL = json.loads('{"IAT":"Unchanged multi-chain Geyer initial-positive then initial-monotone sequence; biased autocovariance denominator N at every lag; no IAT floor.","NUTS_adaptation":"Original slice-NUTS. Find a reasonable initial step size separately for each chain, then dual averaging during the 1000 warmup transitions. Freeze step size for retention. No mass adaptation. Maximum depth 12.","Rhat":"Maximum of rank-normalized split and folded rank-normalized split R-hat; descriptive, no eligibility threshold.","THC_refresh":"h(s)=exp(s). At EVERY trajectory redraw s from the fixed three-component Gaussian mixture (component SD delta), then p|s~N(0,exp(2s)*M0/beta) and p_s~N(0,Q/beta). Eight exact-B inner substeps. Same scalar coupling and integrator as the completed tuner.","acceptance_reporting":"NUTS mean adaptation statistic is not a Metropolis acceptance probability; report it separately from THC acceptance.","beta":1.0,"comparison_id":"MSG5_NUTS_THC_V3_HELDOUT_20261012_v1","date_frozen":"2026-10-12","failure_rules":{"NUTS":"Preserve completed NUTS tuner: nonfinite retained samples, undefined required IAT or invalid gradient denominator make the block/full score undefined. Divergences and maximum-depth hits are reported, not eligibility gates. No separate nonfinite-trajectory counter is available from this engine.","THC_v3":"Preserve completed THC tuner: any nonfinite warmup OR retained trajectory, undefined required IAT or invalid gradient denominator makes the block/full score undefined. Rejected numerical proposals retain the previous valid state."},"gradient_cost":"Actual newly executed target-gradient calls in retained transitions only. For NUTS count all executed leapfrog leaves, even if later discarded; cached endpoint gradients are not counted twice. Initial epsilon search and warmup are excluded. THC g=30 when paths finish; inner thermostat substeps do not call the target gradient.","mass_matrix":[[1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,1.0]],"methods":{"NUTS":{"delta_max":1000.0,"dual_averaging_gamma":0.05,"dual_averaging_kappa":0.75,"dual_averaging_t0":10.0,"maximum_tree_depth":12,"reasonable_epsilon_max_search_steps":100,"target_acceptance":0.425},"THC_v3":{"L":30,"Q":150.0,"delta":0.03,"epsilon":0.06,"inner_r":8,"mixture_centers":[0.7161277448,0.1776353476,0.0],"mixture_weights":[0.564147,0.1084317,0.3274213]}},"n_blocks":10,"n_chains":4,"n_retained":10000,"n_warmup":1000,"no_retuning":true,"observable_policy":"ALL_LINEAR_AND_UNIQUE_QUADRATIC","omegas":[1.0,2.340347319320716,5.477225575051661,12.818610191887021,30.0],"paired_comparison":"S_THC_v3/S_NUTS and paired block wins. Approximate 95% Student-t interval on the mean of ten log(E_THC_v3/E_NUTS) values, exponentiated. Only if all required block efficiencies are defined.","random_stream_pairing":"Same seed identifiers and physical starts in corresponding blocks. Original RNGs are preserved: NUTS uses Numba thread-local legacy NumPy RNG; THC uses SeedSequence child streams. Individual random draws are NOT claimed to match. Pairing hashes seal starts, seeds and run lengths, not common random draws.","score":"Block E=1/(g*max_observable_IAT); method S=geometric mean of ALL ten block efficiencies. Never omit or replace an undefined block.","secondary_outcomes":"Maximum and descriptive geometric mean of blockwise worst IAT; maximum R-hat; exact-moment errors; method-specific acceptance and numerical events. No final-data reselection.","seed_audit":{"counters":{},"derivation":"First 4 SHA256 bytes, big-endian, of DOMAIN|PHASE|block=BB|chain=CC|counter=K. Increment only for a recorded occurrence or duplicate, never based on a simulation.","domain":"paper-on-thc|MSG5|NUTS-vs-mixture-THC-v3|held-out|20261012|v1","overlaps_after_audit":0,"path_size_inventory_sha256":"9e8dad262a7314a8b1a1f326c0bfb20a82d9b7347456d60b3efebb54a3e10948","rejected_candidate_values":[],"scope":"Recorded integer seed occurrences in local project text/source files, including all downloaded tuning reports; unrecorded remote-only seeds cannot be audited.","text_bytes_scanned":3052832529,"text_files_scanned":2631},"seed_blocks":[[391192756,3143993186,3060751096,1174930784],[4023656444,445786559,696028177,759370020],[3796412522,4273519042,968431406,2658791864],[2278452339,3833703034,2609885098,1064954484],[2998990180,2196080407,3542549881,2816729059],[188393197,2492866977,3247650100,1331415199],[1179011026,2234181262,3238678466,1066414069],[2082130863,3392641845,2267232776,2528753683],[2553944459,3962867567,313019393,2258564451],[3340866976,2919099271,1516619022,176952639]],"selection":"Best-tested geometric-mean gradient-normalized-efficiency setting per method, selected only from completed expanded tuning.","source_seals":{"NUTS":{"embedded_engine_sha256":"7374dbcc1d03ca901f1253b1b8e5052a4555f3696a4c6917bc47a8166c7afb64","source_sha256":"55e2b1c8b2b71f134f133aed14aabe9f3911fcf9a5d68005058ffd5db299cabf"},"THC_v3":{"embedded_engine_sha256":"9291f5affb3bae3868d592b6392772d74ea63dd01e777a68094bd0f2797ed73e","source_sha256":"f21b6cf546d7828205b407e4f5b964443255aaae33ee0e249f4bfa98e62627a4"}},"standardized_starts":[[2.0,2.0,2.0,2.0,2.0],[2.0,-2.0,2.0,-2.0,2.0],[-2.0,2.0,-2.0,2.0,-2.0],[-2.0,-2.0,-2.0,-2.0,-2.0]],"target":"5D multiscale Gaussian","tuning_report_sha256":{"NUTS":"bb1779f096d927448c59c5f856741a6cf11062de66a1f35f3934a294a8d0425c","THC_v3":"dcb01b11c076309cbe9ecc7c7f13fc46bb94716f7cd4e483c45b41aba7df6415"},"tuning_selected_scores":{"NUTS":0.008799874131004292,"THC_v3":0.012349636336375558}}')
HELDOUT_PROTOCOL_SHA256 = 'a92a607a09b67e627ebf5c48a0b1cfb491818d7e8c000bbcc137c19c5b5bdb0d'
SOFTWARE_TEST_SEEDS = [[3932309987, 2612053419, 4167575049, 2684694298], [4096899401, 1076928998, 2015620295, 2777591411], [1502738036, 1645572173, 2246234972, 1000410892], [1120992845, 3376072990, 2429650415, 3848975734], [3964436366, 836712093, 4209862168, 1704558273], [636421021, 3909719451, 2171452629, 1079090453], [3893440932, 4085960855, 2599138576, 2402157372], [3413376067, 4072818304, 2360668251, 1516690195], [1391704805, 2890467096, 2201870020, 3106712180], [222831043, 1570969352, 1861917706, 551754833]]
# ---------------------------------------------------------------------------
# Fixed held-out driver. No parameter search or selection is performed here.
# ---------------------------------------------------------------------------
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import platform
import sys
import tempfile
import time


def _heldout_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _heldout_guard():
    if _heldout_hash(HELDOUT_PROTOCOL) != HELDOUT_PROTOCOL_SHA256:
        raise RuntimeError('Frozen scientific protocol changed. Do not retune final runs.')
    source = Path(__file__).read_text(encoding='utf-8')
    # Join marker pieces so this guard does not contain another whole marker.
    engine = source.split('# BEGIN ' + 'FROZEN ENGINE\n', 1)[1].split('# END ' + 'FROZEN ENGINE\n', 1)[0]
    expected = HELDOUT_PROTOCOL['source_seals'][HELDOUT_METHOD]['embedded_engine_sha256']
    if hashlib.sha256(engine.encode()).hexdigest() != expected:
        raise RuntimeError('The embedded tuning engine has been modified.')
    import numba
    final = [s for block in HELDOUT_PROTOCOL['seed_blocks'] for s in block]
    test = [s for block in SOFTWARE_TEST_SEEDS for s in block]
    if len(final) != 40 or len(set(final)) != 40 or len(test) != 40 or len(set(test)) != 40:
        raise RuntimeError('Expected forty unique seeds in each schedule.')
    if set(final) & set(test):
        raise RuntimeError('Test seeds overlap reserved final seeds.')
    return expected


def _heldout_protocol(testing=False):
    p = json.loads(json.dumps(HELDOUT_PROTOCOL))
    if testing:
        p.update(comparison_id=p['comparison_id'] + '_SOFTWARE_TEST_ONLY',
                 n_warmup=32, n_retained=512, seed_blocks=SOFTWARE_TEST_SEEDS)
    return p


def _heldout_prepare(workers=4, testing=False):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('workers must be an integer from one through four')
    p = _heldout_protocol(testing)
    setting = p['methods'][HELDOUT_METHOD]
    arguments = dict(
        beta=p['beta'], omegas=p['omegas'], mass_matrix=p['mass_matrix'],
        standardized_starts=p['standardized_starts'], observable_policy=p['observable_policy'],
        seed_blocks=p['seed_blocks'], n_warmup=p['n_warmup'], n_retained=p['n_retained'],
        n_workers=workers, run_mode='CUSTOM',
        run_label='CUSTOM: SOFTWARE TEST ONLY' if testing else 'CUSTOM: FROZEN HELD-OUT',
        output_directory=None, output_filename=HELDOUT_METHOD + '_results.txt',
        allow_overwrite=False, print_progress=False, print_report=False, run_self_tests=True)
    # CUSTOM routes the original validator, not an invitation to retune.
    # This separate driver seals every scientific input.
    if HELDOUT_METHOD == 'NUTS':
        options = {k: v for k, v in setting.items() if k != 'target_acceptance'}
        prepared = _prepare_inputs(RunInputs(**arguments, **options,
            target_acceptance_values=(setting['target_acceptance'],)))
        _validate_run_mode(prepared)
    else:
        prepared = _prepare_inputs(TuningInputs(**arguments,
            epsilon_values=(setting['epsilon'],), L_values=(setting['L'],),
            mixture_weights=setting['mixture_weights'], mixture_centers=setting['mixture_centers'],
            inner_r=setting['inner_r'], delta_values=(setting['delta'],), Q_values=(setting['Q'],)))
    return prepared


def _heldout_common(prepared):
    return prepared


def _heldout_clean(value):
    if isinstance(value, dict):
        return {str(k): _heldout_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_heldout_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return float(value)
    return value


def _heldout_restore_numbers(value):
    if isinstance(value, dict):
        return {k: _heldout_restore_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_heldout_restore_numbers(v) for v in value]
    if value is None:
        return math.nan
    if value == 'Infinity':
        return math.inf
    if value == '-Infinity':
        return -math.inf
    return value


def _heldout_decode_block(record):
    data = _heldout_restore_numbers(record['physical'])
    data['seeds'] = tuple(data['seeds'])
    if 'adapted_step_sizes' in data:
        data['adapted_step_sizes'] = tuple(data['adapted_step_sizes'])
    observables = []
    for values in data['observable_diagnostics']:
        values = dict(values)
        values['iat'] = IATResult(**values['iat'])
        values['rhat'] = RhatResult(**values['rhat'])
        observables.append(ObservableDiagnostic(**values))
    data['observable_diagnostics'] = tuple(observables)
    return BlockResult(**data)


def _heldout_one_block(prepared, index):
    seeds = prepared.seed_blocks[index]
    setting = HELDOUT_PROTOCOL['methods'][HELDOUT_METHOD]
    pairing = dict(seeds=list(seeds), physical_starts=prepared.physical_starts.tolist(),
                   n_warmup=prepared.n_warmup, n_retained=prepared.n_retained)
    if HELDOUT_METHOD == 'NUTS':
        physical = _run_configuration_block(target_acceptance=setting['target_acceptance'],
            block_index=index, seeds=seeds, prepared=prepared)
        auxiliary = {}
        rng_record = dict(kind='Numba thread-local legacy NumPy RNG; reseeded for each chain',
                          seeds=list(seeds), full_streams_hash=None)
    else:
        streams = tuple(_make_random_streams(s, prepared.n_warmup + prepared.n_retained,
                                            prepared.dimension) for s in seeds)
        cache = [()] * len(prepared.seed_blocks)
        cache[index] = streams
        output = _run_configuration_block(
            configuration=Configuration(delta=setting['delta'], Q=setting['Q'],
                                        epsilon=setting['epsilon'], L=setting['L']),
            origin='FROZEN_HELDOUT', block_index=index, prepared=prepared,
            cache=tuple(cache), n_warmup=prepared.n_warmup, n_retained=prepared.n_retained)
        physical = output.physical
        auxiliary = {'post_MH_s': asdict(output.post_mh_thermostat),
                     'refreshed_s': asdict(output.refreshed_thermostat)}
        stream_hash = hashlib.sha256()
        for stream in streams:
            for array in stream:
                stream_hash.update(np.ascontiguousarray(array, dtype='<f8').tobytes(order='C'))
        rng_record = dict(kind='Unchanged SeedSequence child streams', seeds=list(seeds),
                          full_streams_hash=stream_hash.hexdigest())
    record = _heldout_clean(dict(physical=asdict(physical), auxiliary=auxiliary,
                                pairing_sha256=_heldout_hash(pairing), rng=rng_record))
    record['record_sha256'] = _heldout_hash(record)
    return record


def _heldout_preflight(workers=4):
    _heldout_guard()
    prepared = _heldout_prepare(workers, testing=True)
    messages = list(run_self_tests(prepared) if HELDOUT_METHOD == 'NUTS'
                    else _production_self_tests(prepared))
    sample = _heldout_one_block(prepared, 0)
    block = _heldout_decode_block(sample)
    if len(block.observable_diagnostics) != 20 or block.retained_gradient_calls <= 0:
        raise RuntimeError('Software-check diagnostics are incomplete.')
    if HELDOUT_METHOD == 'NUTS':
        if block.nonfinite_samples or len(block.adapted_step_sizes) != 4:
            raise RuntimeError('NUTS software check produced invalid retained samples/step sizes.')
        expected = prepared.n_chains * prepared.n_retained * block.mean_retained_leapfrog_steps
    else:
        if block.warmup_nonfinite_trajectories or block.retained_nonfinite_trajectories:
            raise RuntimeError('Unexpected nonfinite THC software-check trajectory.')
        expected = prepared.n_chains * prepared.n_retained * block.L
    if not math.isclose(block.retained_gradient_calls, expected, rel_tol=0, abs_tol=1e-7):
        raise RuntimeError('Retained target-gradient accounting failed.')
    restored = _heldout_clean(asdict(_heldout_decode_block(json.loads(json.dumps(sample)))))
    if restored != sample['physical']:
        raise RuntimeError('Checkpoint serialization changed diagnostics.')
    print(f'{HELDOUT_METHOD}: preflight passed. Final seeds were NOT simulated.', flush=True)
    return messages


def _heldout_runtime():
    import llvmlite
    return dict(python=platform.python_version(), numpy=np.__version__,
                numba=numba.__version__, llvmlite=llvmlite.__version__)


@contextmanager
def _heldout_lock(path):
    # OS locks are released even if a process is killed. The harmless lock file
    # remains; its presence alone does NOT mean a run is still active.
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another process is already using this method/output folder.') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _heldout_write(path, text, replace_existing=False):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if replace_existing:
            # Only this method's validated checkpoint is replaced, under its lock.
            os.replace(temporary, path)
        else:
            # Atomic create without clobbering an existing result, on GREX and Windows.
            os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _heldout_validate_checkpoint(saved, metadata):
    payload = {k: v for k, v in saved.items() if k != 'checkpoint_sha256'}
    if saved.get('checkpoint_sha256') != _heldout_hash(payload):
        raise RuntimeError('Checkpoint checksum failed; do not use this checkpoint.')
    for key in ('comparison_id', 'method', 'protocol_sha256', 'protocol', 'script_sha256', 'runtime', 'testing'):
        if saved['metadata'][key] != metadata[key]:
            raise RuntimeError('Checkpoint does not match this frozen run: ' + key)
    records = saved['records']
    p = metadata['protocol']
    if len(records) > p['n_blocks']:
        raise RuntimeError('Checkpoint has too many blocks.')
    prepared = _heldout_prepare(1, metadata['testing'])
    for index, record in enumerate(records):
        body = {k: v for k, v in record.items() if k != 'record_sha256'}
        if record.get('record_sha256') != _heldout_hash(body):
            raise RuntimeError('Block checksum mismatch.')
        block = _heldout_decode_block(record)
        setting = p['methods'][HELDOUT_METHOD]
        correct_setting = (block.target_acceptance == setting['target_acceptance'] and len(block.adapted_step_sizes) == 4
                           if HELDOUT_METHOD == 'NUTS' else block.epsilon == setting['epsilon'] and block.L == setting['L'])
        pairing = dict(seeds=p['seed_blocks'][index], physical_starts=prepared.physical_starts.tolist(),
                       n_warmup=p['n_warmup'], n_retained=p['n_retained'])
        if (block.block_index != index or list(block.seeds) != p['seed_blocks'][index]
                or not correct_setting or len(block.observable_diagnostics) != 20
                or record['pairing_sha256'] != _heldout_hash(pairing)):
            raise RuntimeError('Checkpoint identifiers do not match the frozen schedule.')
    return records


def _heldout_checkpoint(path, metadata, records, replace_existing):
    payload = dict(format='MSG5-heldout-checkpoint-v1', metadata=metadata, records=records)
    payload['checkpoint_sha256'] = _heldout_hash(payload)
    _heldout_write(path, json.dumps(payload, indent=2, allow_nan=False) + '\n', replace_existing)


def _heldout_number(value):
    if math.isnan(float(value)):
        return 'undefined'
    return format(float(value), '.12g')


def _heldout_build_report(metadata, records, completed_utc):
    p = metadata['protocol']
    if len(records) != p['n_blocks']:
        raise RuntimeError('A final report requires all ten blocks, even if a score is undefined.')
    blocks = tuple(_heldout_decode_block(record) for record in records)
    aggregate = _aggregate_blocks(blocks)
    taus = np.array([b.worst_iat for b in blocks])
    valid_taus = bool(np.all(np.isfinite(taus) & (taus > 0)))
    summary = dict(asdict(aggregate),
        maximum_blockwise_worst_IAT=float(np.max(taus)) if valid_taus else math.nan,
        geometric_mean_worst_IAT=float(np.exp(np.mean(np.log(taus)))) if valid_taus else math.nan,
        blocks_completed=len(blocks), blocks_score_defined=sum(b.status == 'ok' for b in blocks),
        retained_draws_per_method=p['n_blocks'] * p['n_chains'] * p['n_retained'],
        total_retained_gradient_calls=sum(b.retained_gradient_calls for b in blocks),
        mean_gradient_calls_per_draw=float(np.mean([b.gradients_per_retained_draw for b in blocks])))
    if HELDOUT_METHOD == 'NUTS':
        summary.update(total_epsilon_search_gradient_calls=sum(b.epsilon_search_gradient_calls for b in blocks),
                       total_warmup_gradient_calls=sum(b.warmup_gradient_calls for b in blocks))
    else:
        summary.update(warmup_nonfinite_trajectories=sum(b.warmup_nonfinite_trajectories for b in blocks),
                       retained_nonfinite_trajectories=sum(b.retained_nonfinite_trajectories for b in blocks))
    label = 'SOFTWARE TEST ONLY -- NOT HELD-OUT RESULTS' if metadata['testing'] else 'FRESH HELD-OUT RESULTS'
    text = [f'{HELDOUT_METHOD}: {label}', '=' * 88, 'Comparison: ' + p['comparison_id'],
        'Protocol SHA256: ' + metadata['protocol_sha256'], 'Script SHA256: ' + metadata['script_sha256'],
        'Completed UTC: ' + completed_utc, 'Runtime: ' + json.dumps(metadata['runtime'], sort_keys=True),
        'Fixed settings: ' + json.dumps(p['methods'][HELDOUT_METHOD], sort_keys=True),
        f"{p['n_blocks']} paired blocks; {p['n_chains']} chains/block; {p['n_warmup']} warmup + {p['n_retained']} retained draws/chain.",
        'Target: U(q)=0.5*sum(omega_j^2*q_j^2); beta=1; base mass=I_5.',
        'Observables: all 5 coordinates and all 15 unique quadratic products.',
        p['IAT'], p['Rhat'], p['gradient_cost'], p['score'],
        'Numerical-event rules (unchanged from each completed tuner; not identical counters):',
        'NUTS: ' + p['failure_rules']['NUTS'], 'THC-v3: ' + p['failure_rules']['THC_v3'],
        p['acceptance_reporting'], p['random_stream_pairing'],
        'No final-data parameter selection. NUTS adapts only within the discarded warmup of each chain.',
        p['NUTS_adaptation'] if HELDOUT_METHOD == 'NUTS' else p['THC_refresh'], '', 'METHOD SUMMARY']
    for key, value in summary.items():
        display = value if isinstance(value, (str, int)) else _heldout_number(value)
        text.append(f'{key}: {display}')
    text += ['', 'BLOCK RESULTS (one-based block numbers)',
             'block status g worst_IAT worst_observable efficiency max_Rhat max_moment_SD']
    for b in blocks:
        text.append(' '.join(map(str, [b.block_index + 1, b.status,
            _heldout_number(b.gradients_per_retained_draw), _heldout_number(b.worst_iat), b.worst_observable,
            _heldout_number(b.block_efficiency), _heldout_number(b.maximum_rhat),
            _heldout_number(b.maximum_standardized_moment_error)])))
    if HELDOUT_METHOD == 'NUTS':
        text += ['', 'NUTS DIAGNOSTICS (adaptation statistic is NOT MH acceptance)',
                 'block mean_adaptation_statistic retained_divergences maximum_depth_hits nonfinite_retained_samples mean_tree_depth mean_leapfrog_steps epsilon_search_gradients warmup_gradients retained_gradients adapted_step_sizes_by_chain']
        for b in blocks:
            values = [b.block_index + 1, _heldout_number(b.mean_acceptance_statistic),
                b.retained_divergences, b.retained_maximum_depth_hits, b.nonfinite_samples,
                _heldout_number(b.mean_retained_tree_depth), _heldout_number(b.mean_retained_leapfrog_steps),
                b.epsilon_search_gradient_calls, b.warmup_gradient_calls, b.retained_gradient_calls,
                ','.join(_heldout_number(x) for x in b.adapted_step_sizes)]
            text.append(' '.join(map(str, values)))
    else:
        text += ['', 'THC DIAGNOSTICS', 'block MH_acceptance nonfinite_warmup_trajectories nonfinite_retained_trajectories retained_gradients']
        for b in blocks:
            text.append(' '.join(map(str, [b.block_index+1, _heldout_number(b.acceptance),
                b.warmup_nonfinite_trajectories, b.retained_nonfinite_trajectories, b.retained_gradient_calls])))
    text += ['', 'EXACT AND POOLED ESTIMATED PHYSICAL MOMENTS',
             'Equal chain/block lengths: estimates average all retained draws, including repeated positions.',
             'observable exact_mean estimated_mean raw_error']
    for index, item in enumerate(blocks[0].observable_diagnostics):
        empirical = float(np.mean([b.observable_diagnostics[index].empirical_mean for b in blocks]))
        text.append(f'{item.name:<8} {_heldout_number(item.exact_mean):>18} '
                    f'{_heldout_number(empirical):>18} {_heldout_number(empirical-item.exact_mean):>18}')
    text += ['', 'ALL PHYSICAL OBSERVABLES IN ALL BLOCKS',
             'block observable IAT IAT_status pairs last_lag first_excluded_lag ESS efficiency rank_Rhat folded_Rhat max_Rhat empirical_mean exact_mean raw_error moment_SD']
    for b in blocks:
        for item in b.observable_diagnostics:
            values = [b.block_index+1, item.name, _heldout_number(item.iat.tau), item.iat.status,
                item.iat.pair_count, item.iat.last_included_lag, item.iat.first_excluded_lag,
                *[_heldout_number(v) for v in (item.ess, item.efficiency, item.rhat.rank, item.rhat.folded,
                    item.rhat.combined, item.empirical_mean, item.exact_mean, item.raw_moment_error,
                    item.standardized_moment_error)]]
            text.append(' '.join(map(str, values)))
    if HELDOUT_METHOD == 'THC_v3':
        text += ['', 'AUXILIARY MOMENTS (descriptive only)',
                 'block variable exact_mean estimated_mean exact_second_moment estimated_second_moment']
        for i, record in enumerate(records):
            for variable, values in record['auxiliary'].items():
                t = _heldout_restore_numbers(values)
                text.append(' '.join([str(i+1), variable] + [_heldout_number(t[k]) for k in
                    ('exact_mean', 'empirical_mean', 'exact_second_moment', 'empirical_second_moment')]))
    compact_blocks = []
    for record in records:
        entry = {k: v for k, v in record['physical'].items() if k != 'observable_diagnostics'}
        entry.update(pairing_sha256=record['pairing_sha256'])
        compact_blocks.append(entry)
    comparison_record = _heldout_clean(dict(format='MSG5-heldout-comparison-report-v2',
        metadata=metadata, completed_utc=completed_utc, summary=summary, blocks=compact_blocks))
    text += ['', 'BEGIN COMPARISON_JSON', json.dumps(comparison_record, indent=2, allow_nan=False),
             'END COMPARISON_JSON', '',
             'Compare ONLY against the partner with this protocol SHA256 and matching block seeds/pairing hashes.',
             'This report does not declare a winner. Keep both full reports, including undefined scores.']
    return '\n'.join(text) + '\n'


def _heldout_execute(output_directory, workers=4, resume=False, *, testing=False, _stop_after=None):
    _heldout_guard()
    if _stop_after is not None and not testing:
        raise ValueError('Partial-run testing is forbidden for the final comparison.')
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = HELDOUT_METHOD + ('_SOFTWARE_TEST' if testing else '')
    report_path = output / (prefix + '_results.txt')
    checkpoint_path = output / (prefix + '_checkpoint.txt')
    with _heldout_lock(output / ('.' + prefix + '.lock')):
        if report_path.exists():
            raise FileExistsError('Report already exists; no new chains were run: ' + str(report_path))
        if checkpoint_path.exists() and not resume:
            raise FileExistsError('Checkpoint exists. Use --run --resume to continue the SAME frozen experiment.')
        if resume and not checkpoint_path.is_file():
            raise FileNotFoundError('There is no checkpoint to resume.')
        protocol = _heldout_protocol(testing)
        metadata = dict(comparison_id=protocol['comparison_id'], method=HELDOUT_METHOD,
                        protocol=protocol, protocol_sha256=_heldout_hash(protocol),
                        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        runtime=_heldout_runtime(), platform=platform.platform(),
                        started_utc=datetime.now(timezone.utc).isoformat(),
                        testing=testing, workers=workers)
        records = []
        if resume:
            saved = json.loads(checkpoint_path.read_text(encoding='utf-8'))
            records = _heldout_validate_checkpoint(saved, metadata)
            metadata = saved['metadata']
        _heldout_preflight(workers)
        prepared = _heldout_prepare(workers, testing)
        if not resume:
            _heldout_checkpoint(checkpoint_path, metadata, records, False)
        for index in range(len(records), protocol['n_blocks']):
            record = _heldout_one_block(prepared, index)
            records.append(record)
            _heldout_checkpoint(checkpoint_path, metadata, records, True)
            # Progress never chooses whether to continue: all ten blocks always run.
            print(f'{prefix}: block {index+1}/{protocol["n_blocks"]} saved.', flush=True)
            if testing and _stop_after is not None and len(records) >= _stop_after:
                return checkpoint_path
        report = _heldout_build_report(metadata, records, datetime.now(timezone.utc).isoformat())
        _heldout_write(report_path, report, False)
        print(report, end='', flush=True)
        print('\nThe identical report was saved to: ' + str(report_path), flush=True)
        return report_path


def _heldout_main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--run', action='store_true', help='Run the frozen final experiment, not a tuning grid.')
    choice.add_argument('--preflight', action='store_true', help='Software checks only (also the default).')
    parser.add_argument('--resume', action='store_true', help='Continue this method\'s validated checkpoint; requires --run.')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4,
                        help='Independent chain workers, 1-4. Request matching CPUs from Slurm.')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Report/checkpoint folder; defaults to beside this script.')
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error('--resume requires --run')
    if args.run:
        _heldout_execute(args.output_dir, args.workers, args.resume)
    else:
        _heldout_preflight(args.workers)
        print('Ready. To run the frozen held-out experiment, add --run. No final report was written.')


if __name__ == '__main__':
    _heldout_main()
