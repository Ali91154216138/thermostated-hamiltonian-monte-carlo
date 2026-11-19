"""Frozen independent HMC held-out run on the 2D Banana.
Default: separate-seed preflight only. Add --run for the final experiment.
No parameter overrides, tuning, retries with new seeds, or companion imports.
Outputs: identical printed/saved TXT plus resumable TXT checkpoint.
"""
# HELDOUT_ENGINE_BEGIN
from __future__ import annotations

import ast

import hashlib

import inspect

import json

import linecache

import math

import numbers

import os

import platform

import sys

from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass, replace

from pathlib import Path

from typing import Sequence

import numpy as np

NUMBA_RUNTIME_EPOCH = "BANANA_HMC_GREX_PORTABLE_20260829D"

NUMBA_DIRECT_RUN = __name__ == "__main__"

NUMBA_CACHE_MODE = (
    "persistent cache disabled in all modes; native epoch " + NUMBA_RUNTIME_EPOCH
)

NUMBA_CACHE_DIRECTORY = "not used"

try:
    import llvmlite
    import numba

    def njit(*args, **kwargs):
        epoch_options = dict(kwargs)
        epoch_options["cache"] = False
        return numba.njit(*args, **epoch_options)

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - only exercised without Numba
    NUMBA_AVAILABLE = False
    llvmlite = None

    def njit(*args, **kwargs):
        """Fallback decorator that leaves functions unchanged."""

        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator

SHARED_PROTOCOL_VERSION = "BANANA2-fair-playground-v1.0.0"

COMMON_INPUT_SCHEMA_VERSION = "banana-fair-playground-common-inputs-v1"

INPUT_PREPARATION_VERSION = "banana-validated-derived-inputs-v1.0.0"

DIAGNOSTICS_VERSION = "multichain-geyer-ims-rank-rhat-v1.0.0"

COMMON_PIPELINE_VERSION = "banana-common-rng-observables-score-v1.0.0"

HMC_IMPLEMENTATION_VERSION = (
    "banana-hmc-general-mass-cached-leapfrog-grex-portable-epoch-v1.0.0"
)

RANDOM_STREAM_VERSION = "seedsequence-pcg64-three-substreams-v1.0.0"

HMC_RUNTIME_PREFLIGHT_VERSION = (
    "banana2-hmc-grex-portable-native-preflight-20260829d-v1"
)

SUPPORTED_OBSERVABLE_POLICY = "ALL_LINEAR_AND_UNIQUE_QUADRATIC"

CANONICAL_OUTPUT_FILENAME = "HMC_banana_tuning_results_GREX_PORTABLE_V1.txt"

EXPECTED_INPUT_PREPARATION_FINGERPRINT = (
    "8e6767f332cf26810005489774bc928995cea9f20b3ebf9b36aadbde48c438cf"
)

EXPECTED_DIAGNOSTICS_FINGERPRINT = (
    "dc2148c37f893c7fa5c22431760b2c46ac3aed96cce35909b333eb2abce91274"
)

EXPECTED_COMMON_PIPELINE_FINGERPRINT = (
    "cf9335a2258570451506d874de40f366c0708ace98ee1c66622d3404fd540d37"
)

EXPECTED_SHARED_PROTOCOL_FINGERPRINT = (
    "a7679d36fdd46df9c30b96bb24634f4dccf746a221021587e6702eded60ab351"
)

CANONICAL_COMMON_INPUT_FINGERPRINT = (
    "0fb7398fba5b8e8c874f498a68471df0a140a341cb289f87e8e698ea4ac26136"
)

CANONICAL_COMPONENT_FINGERPRINTS = {
    "target": "679c5ea9f588d1f56b9ccd4e24c47b41fc4252deb66fac36c81bc8fe55cb05f4",
    "geometry": "98d8f7dfdce05a4dafb5c26a60e0bf859ad9b7d3e208219f6f8f127e9818b8e8",
    "starts": "f642876a93ceef9a971e9dd61d58012ed61335688461e82f29c11c5eae59b204",
    "grid": "f073b3f13a1f3982af5b86d8c44601d30266815bb24a592a3c51e08295884d21",
    "seed_blocks": "948d0b505684247137db516ad687448ad531f704063f3a6d18f3e16e26b09591",
    "run_lengths": "c422ddc0f1a384edfb22455b94fb23abe5aff53663e98d685037840463996db2",
    "observable_policy": "32cfff3f4d30970724331c5b750837b2a5f24260263ec5fa91d42f28de12fef8",
}

@dataclass(frozen=True)
class RunInputs:
    """Every independent common scientific and execution input for one run."""

    beta: float
    sigma_x: float
    sigma_y: float
    banana_b: float
    mass_matrix: Sequence[Sequence[float]]
    standardized_starts: Sequence[Sequence[float]]
    observable_policy: str
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
    """Validated inputs plus quantities derived from them exactly once."""

    beta: float
    sigma_x: float
    sigma_y: float
    banana_b: float
    target_parameters: np.ndarray
    target_variances: np.ndarray
    mass_matrix: np.ndarray
    inverse_mass_matrix: np.ndarray
    mass_cholesky: np.ndarray
    mass_is_identity: bool
    standardized_starts: np.ndarray
    physical_starts: np.ndarray
    dimension: int
    n_chains: int
    epsilon_values: tuple[float, ...]
    L_values: tuple[int, ...]
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
    """Method-neutral aggregation of fixed seed-block results."""

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
class ConfigurationResult:
    """Block aggregate used to choose one fixed HMC representative."""

    epsilon: float
    L: int
    status: str
    block_results: tuple[BlockResult, ...]
    geometric_mean_efficiency: float
    minimum_block_efficiency: float
    median_block_efficiency: float
    maximum_block_efficiency: float
    mean_acceptance: float
    maximum_rhat: float
    maximum_standardized_moment_error: float
    total_nonfinite_trajectories: int

@njit(cache=False, inline="always")
def _potential_energy_grex_20260829d(
    q: np.ndarray,
    target_parameters: np.ndarray,
) -> float:
    """Centered Banana potential for q=(x,y)."""

    sigma_x_squared = target_parameters[0]
    sigma_y_squared = target_parameters[1]
    bend = target_parameters[2]
    x = q[0]
    y = q[1]
    residual = y - bend * (x * x - sigma_x_squared)
    return (
        0.5 * x * x / sigma_x_squared
        + 0.5 * residual * residual / sigma_y_squared
    )

@njit(cache=False, inline="always")
def _target_gradient_grex_20260829d(
    q: np.ndarray,
    target_parameters: np.ndarray,
) -> np.ndarray:
    """Gradient of the centered Banana potential with respect to (x,y)."""

    sigma_x_squared = target_parameters[0]
    sigma_y_squared = target_parameters[1]
    bend = target_parameters[2]
    x = q[0]
    y = q[1]
    residual = y - bend * (x * x - sigma_x_squared)
    gradient = np.empty(2, dtype=np.float64)
    gradient[0] = (
        x / sigma_x_squared
        - 2.0 * bend * x * residual / sigma_y_squared
    )
    gradient[1] = residual / sigma_y_squared
    return gradient

@njit(cache=False, inline="always")
def _kinetic_energy_grex_20260829d(
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

@njit(cache=False, inline="always")
def _vector_is_finite_grex_20260829d(values: np.ndarray) -> bool:
    for value in values:
        if not math.isfinite(value):
            return False
    return True

@njit(cache=False, nogil=True)
def _leapfrog_proposal_grex_20260829d(
    q: np.ndarray,
    p: np.ndarray,
    current_gradient: np.ndarray,
    epsilon: float,
    L: int,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Integrate one HMC proposal using a cached starting gradient.

    Each of the L drift steps evaluates the target gradient once.  The final
    gradient is returned and can be cached for the next trajectory if the
    proposal is accepted.  A rejected proposal retains the old cached gradient.
    """

    proposed_q = q.copy()
    proposed_p = p.copy()
    proposed_gradient = current_gradient.copy()
    gradient_calls = 0

    for coordinate in range(proposed_p.size):
        proposed_p[coordinate] -= 0.5 * epsilon * proposed_gradient[coordinate]
    if not _vector_is_finite_grex_20260829d(proposed_p):
        return proposed_q, proposed_p, proposed_gradient, math.nan, gradient_calls, False

    for step in range(L):
        if mass_is_identity:
            for coordinate in range(proposed_q.size):
                proposed_q[coordinate] += epsilon * proposed_p[coordinate]
        else:
            for row in range(proposed_q.size):
                velocity = 0.0
                for column in range(proposed_p.size):
                    velocity += inverse_mass_matrix[row, column] * proposed_p[column]
                proposed_q[row] += epsilon * velocity
        if not _vector_is_finite_grex_20260829d(proposed_q):
            return proposed_q, proposed_p, proposed_gradient, math.nan, gradient_calls, False

        proposed_gradient = _target_gradient_grex_20260829d(
            proposed_q, target_parameters
        )
        gradient_calls += 1
        if not _vector_is_finite_grex_20260829d(proposed_gradient):
            return proposed_q, proposed_p, proposed_gradient, math.nan, gradient_calls, False

        kick = 0.5 * epsilon if step == L - 1 else epsilon
        for coordinate in range(proposed_p.size):
            proposed_p[coordinate] -= kick * proposed_gradient[coordinate]
        if not _vector_is_finite_grex_20260829d(proposed_p):
            return proposed_q, proposed_p, proposed_gradient, math.nan, gradient_calls, False

    proposed_potential = _potential_energy_grex_20260829d(
        proposed_q, target_parameters
    )
    if not math.isfinite(proposed_potential):
        return proposed_q, proposed_p, proposed_gradient, proposed_potential, gradient_calls, False

    return (
        proposed_q,
        proposed_p,
        proposed_gradient,
        proposed_potential,
        gradient_calls,
        True,
    )

@njit(cache=False, nogil=True)
def _simulate_hmc_chain_grex_20260829d(
    standard_momenta: np.ndarray,
    acceptance_uniforms: np.ndarray,
    q_initial: np.ndarray,
    beta: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, int, int, int, int, int]:
    """Run one chain using pre-generated, configuration-paired random streams."""

    total_iterations = n_warmup + n_retained
    samples = np.empty((n_retained, q_initial.size), dtype=np.float64)
    q = q_initial.copy()
    potential = _potential_energy_grex_20260829d(q, target_parameters)
    current_gradient = _target_gradient_grex_20260829d(q, target_parameters)

    # The initial gradient is charged to warmup when warmup exists.  The cache
    # then carries into retained sampling, so every retained trajectory uses L
    # new gradients.  With no warmup, the initial call is charged to sampling.
    warmup_gradient_calls = 1 if n_warmup > 0 else 0
    retained_gradient_calls = 0 if n_warmup > 0 else 1
    accepted_retained = 0
    warmup_nonfinite = 0
    retained_nonfinite = 0
    momentum_scale = 1.0 / math.sqrt(beta)

    for iteration in range(total_iterations):
        p = np.empty(q.size, dtype=np.float64)
        if mass_is_identity:
            for coordinate in range(q.size):
                p[coordinate] = momentum_scale * standard_momenta[iteration, coordinate]
        else:
            for row in range(q.size):
                transformed_momentum = 0.0
                for column in range(q.size):
                    transformed_momentum += (
                        mass_cholesky[row, column]
                        * standard_momenta[iteration, column]
                    )
                p[row] = momentum_scale * transformed_momentum
        kinetic_old = _kinetic_energy_grex_20260829d(
            p, inverse_mass_matrix, mass_is_identity
        )

        (
            proposed_q,
            proposed_p,
            proposed_gradient,
            proposed_potential,
            proposal_gradient_calls,
            proposal_is_finite,
        ) = _leapfrog_proposal_grex_20260829d(
            q,
            p,
            current_gradient,
            epsilon,
            L,
            target_parameters,
            inverse_mass_matrix,
            mass_is_identity,
        )

        if iteration < n_warmup:
            warmup_gradient_calls += proposal_gradient_calls
        else:
            retained_gradient_calls += proposal_gradient_calls

        accepted = False
        if proposal_is_finite:
            # Complete the standard involutive HMC proposal.  The flip leaves
            # kinetic energy unchanged and the momentum is discarded after
            # the decision, but making it explicit seals the detailed-balance
            # convention used by the Metropolis step.
            for coordinate in range(proposed_p.size):
                proposed_p[coordinate] = -proposed_p[coordinate]
            kinetic_new = _kinetic_energy_grex_20260829d(
                proposed_p, inverse_mass_matrix, mass_is_identity
            )
            delta_hamiltonian = (proposed_potential + kinetic_new) - (
                potential + kinetic_old
            )
            if math.isfinite(delta_hamiltonian):
                uniform = acceptance_uniforms[iteration]
                log_uniform = -math.inf if uniform <= 0.0 else math.log(uniform)
                accepted = log_uniform < -beta * delta_hamiltonian
            else:
                proposal_is_finite = False

        if not proposal_is_finite:
            if iteration < n_warmup:
                warmup_nonfinite += 1
            else:
                retained_nonfinite += 1
        elif accepted:
            q = proposed_q
            potential = proposed_potential
            current_gradient = proposed_gradient

        if iteration >= n_warmup:
            retained_index = iteration - n_warmup
            for coordinate in range(q.size):
                samples[retained_index, coordinate] = q[coordinate]
            if accepted:
                accepted_retained += 1

    return (
        samples,
        accepted_retained,
        warmup_nonfinite,
        retained_nonfinite,
        warmup_gradient_calls,
        retained_gradient_calls,
    )

_potential_energy = _potential_energy_grex_20260829d

_target_gradient = _target_gradient_grex_20260829d

_kinetic_energy = _kinetic_energy_grex_20260829d

_vector_is_finite = _vector_is_finite_grex_20260829d

_leapfrog_proposal = _leapfrog_proposal_grex_20260829d

_simulate_hmc_chain = _simulate_hmc_chain_grex_20260829d

def _make_random_streams(
    seed: int,
    total_iterations: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create physical-momentum and acceptance streams from fixed substreams.

    Three child SeedSequences are always spawned in this order:
      1. physical standard-normal momenta,
      2. thermostat standard-normal momenta (used here; reserved by HMC),
      3. Metropolis uniforms.

    HMC materializes child stream 2 for future parity but does not use it in an
    HMC transition.  The separate streams keep physical normals and uniforms
    invariant when the THC coordinate is added.
    """

    root = np.random.SeedSequence(int(seed))
    physical_sequence, thermostat_sequence, acceptance_sequence = root.spawn(3)
    physical_rng = np.random.Generator(np.random.PCG64(physical_sequence))
    thermostat_rng = np.random.Generator(np.random.PCG64(thermostat_sequence))
    acceptance_rng = np.random.Generator(np.random.PCG64(acceptance_sequence))
    standard_momenta = physical_rng.standard_normal(
        size=(total_iterations, dimension)
    ).astype(np.float64, copy=False)
    thermostat_standard_momenta = thermostat_rng.standard_normal(total_iterations).astype(
        np.float64, copy=False
    )
    uniforms = acceptance_rng.random(total_iterations).astype(np.float64, copy=False)
    return standard_momenta, thermostat_standard_momenta, uniforms

def _retained_standardized_momentum_second_moment(
    stream_cache: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_warmup: int,
) -> float:
    """Return pooled E[beta*p^T M^-1 p/d] from retained momentum refreshes.

    For p=M^(1/2)z/sqrt(beta), beta*p^T*M^-1*p/d equals z^T*z/d, so the
    stored physical standard-normal stream supplies this diagnostic for every
    positive-definite mass matrix.  It is never an observable, IAT input,
    gate, or selection score.
    """

    sum_of_squares = 0.0
    number_of_components = 0
    for physical_momenta, _thermostat_momenta, _uniforms in stream_cache:
        retained = physical_momenta[n_warmup:]
        sum_of_squares += float(np.sum(retained * retained))
        number_of_components += retained.size
    if number_of_components == 0:
        return math.nan
    return sum_of_squares / number_of_components

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
    beta: float,
    sigma_x: float,
    sigma_y: float,
    banana_b: float,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Exact means and target SDs for (x,y,x^2,x*y,y^2)."""

    variance_x = sigma_x * sigma_x / beta
    variance_r = sigma_y * sigma_y / beta
    centering = sigma_x * sigma_x
    bend = banana_b

    shifted_square_second = (
        3.0 * variance_x**2
        - 2.0 * centering * variance_x
        + centering**2
    )
    shifted_square_fourth = (
        105.0 * variance_x**4
        - 60.0 * centering * variance_x**3
        + 18.0 * centering**2 * variance_x**2
        - 4.0 * centering**3 * variance_x
        + centering**4
    )

    mean_y = bend * (variance_x - centering)
    mean_y_squared = (
        variance_r + bend * bend * shifted_square_second
    )
    variance_y = (
        variance_r + 2.0 * bend * bend * variance_x**2
    )
    variance_x_squared = 2.0 * variance_x**2
    variance_xy = (
        variance_x * variance_r
        + bend
        * bend
        * variance_x
        * (
            15.0 * variance_x**2
            - 6.0 * centering * variance_x
            + centering**2
        )
    )
    fourth_y = (
        3.0 * variance_r**2
        + 6.0
        * bend
        * bend
        * variance_r
        * shifted_square_second
        + bend**4 * shifted_square_fourth
    )
    variance_y_squared = fourth_y - mean_y_squared**2

    names = ("x", "y", "x^2", "x*y", "y^2")
    exact_means = np.asarray(
        [0.0, mean_y, variance_x, 0.0, mean_y_squared],
        dtype=np.float64,
    )
    exact_variances = np.asarray(
        [
            variance_x,
            variance_y,
            variance_x_squared,
            variance_xy,
            variance_y_squared,
        ],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(exact_variances))
        or np.any(exact_variances <= 0.0)
    ):
        raise ValueError(
            "Derived Banana observable variances must be finite and positive."
        )
    return names, exact_means, np.sqrt(exact_variances)

def _latent_to_physical(
    standardized_starts: np.ndarray,
    beta: float,
    sigma_x: float,
    sigma_y: float,
    banana_b: float,
) -> np.ndarray:
    """Map standardized latent (z_x,z_r) rows to physical (x,y) rows."""

    latent = np.asarray(standardized_starts, dtype=np.float64)
    x = sigma_x * latent[:, 0] / math.sqrt(beta)
    residual = sigma_y * latent[:, 1] / math.sqrt(beta)
    y = banana_b * (x * x - sigma_x * sigma_x) + residual
    return np.ascontiguousarray(np.column_stack((x, y)))

def _physical_to_latent(
    physical_states: np.ndarray,
    beta: float,
    sigma_x: float,
    sigma_y: float,
    banana_b: float,
) -> np.ndarray:
    """Map physical (x,y) rows back to standardized latent (z_x,z_r)."""

    states = np.asarray(physical_states, dtype=np.float64)
    x = states[:, 0]
    y = states[:, 1]
    residual = y - banana_b * (x * x - sigma_x * sigma_x)
    scale = math.sqrt(beta)
    return np.ascontiguousarray(
        np.column_stack(
            (
                scale * x / sigma_x,
                scale * residual / sigma_y,
            )
        )
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
    """Calculate the complete agreed diagnostics for one tuning block."""

    samples = np.asarray(samples, dtype=np.float64)
    total_retained = samples.shape[0] * samples.shape[1]
    acceptance = accepted_retained / total_retained
    gradients_per_draw = retained_gradient_calls / total_retained
    dimension = int(samples.shape[2])
    observable_values = _physical_observables(samples, dimension)
    diagnostics: list[ObservableDiagnostic] = []
    simulation_is_nonfinite = warmup_nonfinite > 0 or retained_nonfinite > 0

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

def _aggregate_configuration(
    epsilon: float,
    L: int,
    block_results: Sequence[BlockResult],
) -> ConfigurationResult:
    """Wrap the common block aggregate with HMC configuration parameters."""

    blocks = tuple(block_results)
    aggregate = _aggregate_blocks(blocks)
    return ConfigurationResult(
        epsilon=float(epsilon),
        L=int(L),
        status=aggregate.status,
        block_results=blocks,
        geometric_mean_efficiency=aggregate.geometric_mean_efficiency,
        minimum_block_efficiency=aggregate.minimum_block_efficiency,
        median_block_efficiency=aggregate.median_block_efficiency,
        maximum_block_efficiency=aggregate.maximum_block_efficiency,
        mean_acceptance=aggregate.mean_acceptance,
        maximum_rhat=aggregate.maximum_rhat,
        maximum_standardized_moment_error=aggregate.maximum_standardized_moment_error,
        total_nonfinite_trajectories=aggregate.total_nonfinite_trajectories,
    )

def _select_representative(
    configuration_results: Sequence[ConfigurationResult],
) -> ConfigurationResult:
    """Select by GM efficiency with a deterministic exact-tie rule."""

    eligible = [result for result in configuration_results if result.status == "ok"]
    if not eligible:
        raise RuntimeError("No HMC configuration has a defined five-block tuning score.")

    # Exact ties: larger minimum block score, then fewer gradients/draw, then
    # smaller epsilon, then smaller L.  No near-tie tolerance is introduced.
    return max(
        eligible,
        key=lambda result: (
            result.geometric_mean_efficiency,
            result.minimum_block_efficiency,
            -result.block_results[0].gradients_per_retained_draw,
            -result.epsilon,
            -result.L,
        ),
    )

def _normalize_grid(
    epsilon_values: Sequence[float],
    L_values: Sequence[int],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    try:
        raw_epsilons = tuple(epsilon_values)
    except TypeError as error:
        raise ValueError("epsilon_values must be an iterable of real numbers.") from error
    epsilons_list: list[float] = []
    for index, value in enumerate(raw_epsilons):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"epsilon_values[{index}] must be a real number.")
        epsilons_list.append(float(value))
    epsilons = tuple(epsilons_list)

    try:
        raw_lengths = tuple(L_values)
    except TypeError as error:
        raise ValueError("L_values must be an iterable of integers.") from error
    lengths_list: list[int] = []
    for index, value in enumerate(raw_lengths):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError(f"L_values[{index}] must be an integer.")
        lengths_list.append(int(value))
    lengths = tuple(lengths_list)
    if not epsilons or any(not math.isfinite(value) or value <= 0 for value in epsilons):
        raise ValueError("Every epsilon must be finite and positive.")
    if len(set(epsilons)) != len(epsilons):
        raise ValueError("epsilon values must not contain duplicates.")
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("Every L must be a positive integer.")
    if len(set(lengths)) != len(lengths):
        raise ValueError("L values must not contain duplicates.")
    return epsilons, lengths

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
    sigma_x: float,
    sigma_y: float,
    banana_b: float,
    mass_matrix: np.ndarray,
    standardized_starts: np.ndarray,
    epsilon_values: Sequence[float],
    L_values: Sequence[int],
    seed_blocks: Sequence[Sequence[int]],
    n_warmup: int,
    n_retained: int,
    observable_policy: str,
) -> dict[str, object]:
    return {
        "target": {
            "family": "CENTERED_BANANA",
            "dimension": 2,
            "beta": float(beta),
            "sigma_x": float(sigma_x),
            "sigma_y": float(sigma_y),
            "banana_b": float(banana_b),
            "residual_definition": "r=y-b*(x^2-sigma_x^2)",
        },
        "geometry": {"mass_matrix": mass_matrix.tolist()},
        "starts": {
            "coordinate_system": "(sqrt(beta)*x/sigma_x, sqrt(beta)*r/sigma_y)",
            "standardized_starts": standardized_starts.tolist(),
        },
        "grid": {
            "epsilon_values": list(epsilon_values),
            "L_values": list(L_values),
        },
        "seed_blocks": {"seed_blocks": [list(block) for block in seed_blocks]},
        "run_lengths": {
            "n_warmup": int(n_warmup),
            "n_retained": int(n_retained),
        },
        "observable_policy": {
            "policy": observable_policy,
            "ordered_observables": ["x", "y", "x^2", "x*y", "y^2"],
            "residual_is_scored": False,
        },
    }

def _prepare_inputs(inputs: RunInputs) -> PreparedInputs:
    """Validate inputs and derive the complete centered-Banana specification."""

    if not isinstance(inputs, RunInputs):
        raise TypeError("Input preparation expects one RunInputs object.")

    def positive_real(name: str, value: object) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Real
        ):
            raise ValueError(f"{name} must be a real number.")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive.")
        return result

    beta = positive_real("beta", inputs.beta)
    sigma_x = positive_real("sigma_x", inputs.sigma_x)
    sigma_y = positive_real("sigma_y", inputs.sigma_y)
    if isinstance(inputs.banana_b, (bool, np.bool_)) or not isinstance(
        inputs.banana_b, numbers.Real
    ):
        raise ValueError("banana_b must be a real number.")
    banana_b = float(inputs.banana_b)
    if not math.isfinite(banana_b):
        raise ValueError("banana_b must be finite.")

    dimension = 2
    sigma_x_squared = sigma_x * sigma_x
    sigma_y_squared = sigma_y * sigma_y
    if (
        not math.isfinite(sigma_x_squared)
        or sigma_x_squared <= 0.0
        or not math.isfinite(sigma_y_squared)
        or sigma_y_squared <= 0.0
    ):
        raise ValueError("Squared Banana scales must be finite and positive.")
    target_parameters = np.ascontiguousarray(
        np.asarray(
            [sigma_x_squared, sigma_y_squared, banana_b],
            dtype=np.float64,
        )
    )

    try:
        mass_matrix = np.asarray(inputs.mass_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("mass_matrix must be a numeric 2-by-2 matrix.") from error
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
            "standardized_starts must have shape (n_chains, 2); "
            f"received {standardized_starts.shape}."
        )
    if standardized_starts.shape[0] < 2:
        raise ValueError("At least two chain starts are required.")
    if not np.all(np.isfinite(standardized_starts)):
        raise ValueError("standardized_starts must contain only finite values.")
    standardized_starts = np.ascontiguousarray(standardized_starts.copy())
    n_chains = int(standardized_starts.shape[0])

    epsilon_values, L_values = _normalize_grid(
        inputs.epsilon_values, inputs.L_values
    )
    seed_blocks = _validate_seed_blocks(inputs.seed_blocks, n_chains)
    n_warmup = _require_integer("n_warmup", inputs.n_warmup, 0)
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
        variance_x = sigma_x_squared / beta
        variance_r = sigma_y_squared / beta
        variance_y = (
            variance_r + 2.0 * banana_b * banana_b * variance_x * variance_x
        )
        target_variances = np.ascontiguousarray(
            np.asarray([variance_x, variance_y], dtype=np.float64)
        )
        physical_starts = _latent_to_physical(
            standardized_starts,
            beta,
            sigma_x,
            sigma_y,
            banana_b,
        )
    if not np.all(np.isfinite(target_parameters)):
        raise ValueError("Derived Banana target parameters are nonfinite.")
    if not np.all(np.isfinite(target_variances)) or np.any(
        target_variances <= 0.0
    ):
        raise ValueError(
            "Derived physical marginal variances must be finite and positive."
        )
    if not np.all(np.isfinite(physical_starts)):
        raise ValueError(
            "Derived physical starts are nonfinite; check target and standardized starts."
        )

    observable_names, exact_observable_means, exact_observable_sds = (
        _observable_specification(beta, sigma_x, sigma_y, banana_b)
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
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        banana_b=banana_b,
        mass_matrix=mass_matrix,
        standardized_starts=standardized_starts,
        epsilon_values=epsilon_values,
        L_values=L_values,
        seed_blocks=seed_blocks,
        n_warmup=n_warmup,
        n_retained=n_retained,
        observable_policy=observable_policy,
    )
    component_fingerprints = {
        name: _json_fingerprint(value) for name, value in components.items()
    }
    canonical_mismatches = tuple(
        name
        for name in components
        if component_fingerprints[name]
        != CANONICAL_COMPONENT_FINGERPRINTS.get(name, "")
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
        standardized_starts,
        physical_starts,
        exact_observable_means,
        exact_observable_sds,
    ):
        array.setflags(write=False)

    return PreparedInputs(
        beta=beta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        banana_b=banana_b,
        target_parameters=target_parameters,
        target_variances=target_variances,
        mass_matrix=mass_matrix,
        inverse_mass_matrix=inverse_mass_matrix,
        mass_cholesky=mass_cholesky,
        mass_is_identity=mass_is_identity,
        standardized_starts=standardized_starts,
        physical_starts=physical_starts,
        dimension=dimension,
        n_chains=n_chains,
        epsilon_values=epsilon_values,
        L_values=L_values,
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
        if prepared.canonical_mismatches or (
            prepared.common_input_fingerprint != CANONICAL_COMMON_INPUT_FINGERPRINT
        ):
            mismatch_text = ", ".join(prepared.canonical_mismatches) or "overall hash"
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
            source_path = Path(__file__).resolve()
            directory = (
                source_path.parent
                if source_path.name == "banana_hmc_tuner.py"
                else Path.cwd().resolve()
            )
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
    """Report which parity path the current execution environment used."""

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
    """Hash code that must remain byte-identical in the later HMC tuner."""

    return _source_fingerprint(
        (
            _make_random_streams,
            _retained_standardized_momentum_second_moment,
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

def _hmc_implementation_fingerprint() -> str:
    """Hash HMC-specific dynamics, aggregation wrapper, and tie-breaking code."""

    return _source_fingerprint(
        (
            _potential_energy,
            _target_gradient,
            _kinetic_energy,
            _vector_is_finite,
            _leapfrog_proposal,
            _simulate_hmc_chain,
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
        "target_parameters": prepared.target_parameters.tolist(),
        "target_variances": prepared.target_variances.tolist(),
        "physical_starts": prepared.physical_starts.tolist(),
        "inverse_mass_matrix": prepared.inverse_mass_matrix.tolist(),
        "observables": list(prepared.observable_names),
        "exact_observable_means": prepared.exact_observable_means.tolist(),
        "exact_observable_sds": prepared.exact_observable_sds.tolist(),
        "auxiliary_momentum_moment": "pooled retained beta*p^T*M^-1*p/d with exact value 1; descriptive only",
        "iat": DIAGNOSTICS_VERSION,
        "diagnostics_source_fingerprint": _diagnostics_fingerprint(),
        "common_pipeline_version": COMMON_PIPELINE_VERSION,
        "common_pipeline_source_fingerprint": _common_pipeline_fingerprint(),
        "input_preparation_version": INPUT_PREPARATION_VERSION,
        "input_preparation_source_fingerprint": _input_preparation_fingerprint(),
        "random_streams": RANDOM_STREAM_VERSION,
        "gradient_accounting": "actual postwarmup full-gradient calls with current endpoint gradient cached across trajectories",
        "score": "GM_blocks(min_observable(1/(actual_postwarmup_gradients_per_draw*IAT)))",
    }
    return _json_fingerprint(payload)

def _validate_common_parity_seals(prepared: PreparedInputs) -> None:
    """Require byte-identical shared HMC/THC machinery before simulation."""

    actual_source_seals = {
        "input preparation": _input_preparation_fingerprint(),
        "diagnostics": _diagnostics_fingerprint(),
        "common pipeline": _common_pipeline_fingerprint(),
    }
    expected_source_seals = {
        "input preparation": EXPECTED_INPUT_PREPARATION_FINGERPRINT,
        "diagnostics": EXPECTED_DIAGNOSTICS_FINGERPRINT,
        "common pipeline": EXPECTED_COMMON_PIPELINE_FINGERPRINT,
    }
    mismatches = [
        name
        for name, actual in actual_source_seals.items()
        if actual != expected_source_seals[name]
    ]
    if mismatches:
        raise RuntimeError(
            "Shared HMC/THC source parity failed for: "
            + ", ".join(mismatches)
        )
    if prepared.run_mode == "CANONICAL":
        shared_fingerprint = _shared_protocol_fingerprint(prepared)
        if shared_fingerprint != EXPECTED_SHARED_PROTOCOL_FINGERPRINT:
            raise RuntimeError(
                "Canonical shared-protocol parity failed: expected "
                + EXPECTED_SHARED_PROTOCOL_FINGERPRINT
                + ", received "
                + shared_fingerprint
            )

def _format_number(value: float, specification: str = ".6g") -> str:
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, specification)

def _format_efficiency_grid(
    results: Sequence[ConfigurationResult],
    epsilon_values: Sequence[float],
    L_values: Sequence[int],
) -> str:
    lookup = {(result.epsilon, result.L): result for result in results}
    column_width = 14
    block_count = len(results[0].block_results) if results else 0
    header_label = "epsilon \\ L"
    lines = [
        f"{block_count}-block geometric-mean worst-observable efficiency grid",
        f"{header_label:>12}" + "".join(f"{L:>{column_width}d}" for L in L_values),
    ]
    for epsilon in epsilon_values:
        cells = []
        for L in L_values:
            value = lookup[(float(epsilon), int(L))].geometric_mean_efficiency
            cells.append(f"{_format_number(value, '.6e'):>{column_width}}")
        lines.append(f"{epsilon:>12.6g}" + "".join(cells))
    return "\n".join(lines)

def _build_report(
    *,
    configuration_results: Sequence[ConfigurationResult],
    representative: ConfigurationResult,
    auxiliary_momentum_moments: Sequence[float],
    prepared: PreparedInputs,
    self_test_messages: Sequence[str],
) -> str:
    lines: list[str] = []
    mode_note = (
        "CANONICAL / COMPARABLE"
        if prepared.run_mode == "CANONICAL"
        else "CUSTOM / NON-COMPARABLE"
    )
    lines.extend(
        [
            "FAIR-PLAYGROUND HMC TUNING REPORT: 2D CENTERED BANANA",
            "=" * 72,
            f"Run label: {prepared.run_label}",
            f"Run mode: {mode_note}",
            f"Common-input schema version: {COMMON_INPUT_SCHEMA_VERSION}",
            f"Canonical common inputs match: {not prepared.canonical_mismatches and prepared.common_input_fingerprint == CANONICAL_COMMON_INPUT_FINGERPRINT}",
            f"Canonical expected common-input fingerprint: {CANONICAL_COMMON_INPUT_FINGERPRINT}",
            f"Actual common-input fingerprint: {prepared.common_input_fingerprint}",
            "Canonical component mismatches: "
            + (", ".join(prepared.canonical_mismatches) or "none"),
            f"Input-preparation version: {INPUT_PREPARATION_VERSION}",
            f"Input-preparation source fingerprint: {_input_preparation_fingerprint()}",
            f"Expected input-preparation source fingerprint: {EXPECTED_INPUT_PREPARATION_FINGERPRINT}",
            f"Shared protocol version: {SHARED_PROTOCOL_VERSION}",
            f"Shared protocol fingerprint: {_shared_protocol_fingerprint(prepared)}",
            f"Expected shared protocol fingerprint: {EXPECTED_SHARED_PROTOCOL_FINGERPRINT}",
            f"Diagnostics version: {DIAGNOSTICS_VERSION}",
            f"Diagnostics source fingerprint: {_diagnostics_fingerprint()}",
            f"Expected diagnostics source fingerprint: {EXPECTED_DIAGNOSTICS_FINGERPRINT}",
            f"Common RNG/observable/score pipeline version: {COMMON_PIPELINE_VERSION}",
            f"Common pipeline source fingerprint: {_common_pipeline_fingerprint()}",
            f"Expected common pipeline source fingerprint: {EXPECTED_COMMON_PIPELINE_FINGERPRINT}",
            f"HMC implementation version: {HMC_IMPLEMENTATION_VERSION}",
            f"HMC implementation source fingerprint: {_hmc_implementation_fingerprint()}",
            f"Source-parity verification mode: {_source_verification_mode()}",
            f"Random-stream version: {RANDOM_STREAM_VERSION}",
            f"HMC native-preflight version: {HMC_RUNTIME_PREFLIGHT_VERSION}",
            f"Native runtime epoch: {NUMBA_RUNTIME_EPOCH}",
            f"Runtime environment: {_runtime_environment_summary()}",
            f"Tuning-block-1 random-stream SHA-256: {_first_tuning_block_stream_digest(prepared)}",
            f"Numba available: {NUMBA_AVAILABLE}",
            f"Numba cache mode: {NUMBA_CACHE_MODE}",
            f"Numba cache directory: {NUMBA_CACHE_DIRECTORY}",
            f"Worker threads: {prepared.n_workers}",
            f"Run self-tests: {prepared.run_self_tests}",
            f"Print progress: {prepared.print_progress}",
            f"Print full report: {prepared.print_report}",
            f"Requested output directory: {prepared.output_directory!r}",
            f"Output filename: {prepared.output_filename}",
            f"Allow overwrite: {prepared.allow_overwrite}",
            "",
            "TARGET AND GEOMETRY",
            "-" * 72,
            f"d={prepared.dimension}, beta={prepared.beta:g}",
            f"sigma_x={prepared.sigma_x:.12g}, sigma_y={prepared.sigma_y:.12g}, b={prepared.banana_b:.12g}",
            "r(x,y)=y-b*(x^2-sigma_x^2)",
            "U(x,y)=x^2/(2*sigma_x^2)+r(x,y)^2/(2*sigma_y^2)",
            "grad U=(x/sigma_x^2-2*b*x*r/sigma_y^2, r/sigma_y^2)",
            "mass matrix M="
            + np.array2string(prepared.mass_matrix, precision=10, separator=", "),
            "inverse mass M^-1="
            + np.array2string(prepared.inverse_mass_matrix, precision=10, separator=", "),
            f"M is identity: {prepared.mass_is_identity}",
            "physical marginal variances Var(x),Var(y)="
            + np.array2string(
                prepared.target_variances, precision=10, separator=", "
            ),
            "Hamiltonian H(q,p)=U(q)+0.5*p^T*M^-1*p; p~N(0,M/beta)",
            "Metropolis log ratio=-beta*Delta H; beta is not folded into grad U",
            "Integrator: cached-gradient symmetric kick-drift-kick leapfrog, final momentum negation, and Metropolis correction",
            "",
            "ENTERED TUNING DESIGN",
            "-" * 72,
            f"epsilon values ({len(prepared.epsilon_values)}): {list(prepared.epsilon_values)}",
            f"L values ({len(prepared.L_values)}): {list(prepared.L_values)}",
            f"Configurations: {len(prepared.epsilon_values) * len(prepared.L_values)}",
            f"Blocks: {len(prepared.seed_blocks)}; chains/block: {prepared.n_chains}",
            f"Warmup/chain: {prepared.n_warmup}; retained/chain: {prepared.n_retained}",
            "The same random base momenta and uniforms are reused across configurations within each seed block.",
            "Physical and acceptance streams are separate PCG64 child streams; the middle child stream is reserved for THC.",
            "There is no fixed momentum start: p~N(0,M/beta) is freshly drawn at every HMC transition.",
            "Championship seeds are absent from this tuner.",
            "",
            "TUNING SEED BLOCKS",
        ]
    )
    for block_index, block in enumerate(prepared.seed_blocks, start=1):
        lines.append(f"block {block_index}: {list(block)}")

    lines.extend(["", "INITIAL CONDITIONS (printed exactly as used)"])
    for chain_index in range(prepared.n_chains):
        lines.append(
            f"chain {chain_index + 1}: (z_x,z_r)="
            + np.array2string(
                prepared.standardized_starts[chain_index],
                precision=6,
                separator=", ",
            )
            + "; q="
            + np.array2string(
                prepared.physical_starts[chain_index],
                precision=12,
                separator=", ",
            )
            + f"; r={prepared.physical_starts[chain_index, 1] - prepared.banana_b * (prepared.physical_starts[chain_index, 0] ** 2 - prepared.sigma_x ** 2):.12g}"
            + f"; U={_potential_energy(prepared.physical_starts[chain_index], prepared.target_parameters):.12g}"
        )

    lines.extend(
        [
            "",
            "EXACT PHYSICAL OBSERVABLE MOMENTS",
            "-" * 72,
            "The scored set is exactly (x,y,x^2,x*y,y^2); residual r is excluded.",
        ]
    )
    for name, mean, sd in zip(
        prepared.observable_names,
        prepared.exact_observable_means,
        prepared.exact_observable_sds,
    ):
        lines.append(
            f"{name:<5}: exact mean={mean:.12e}; exact SD={sd:.12e}"
        )

    lines.extend(
        [
            "",
            "MINIMAL AUXILIARY MOMENTUM DIAGNOSTIC (DESCRIPTIVE ONLY)",
            "-" * 72,
            "Quantity: pooled retained beta*p^T*M^-1*p/d; exact value=1.",
            "This is not a score observable, IAT input, threshold, or selection gate.",
        ]
    )
    for block_index, value in enumerate(auxiliary_momentum_moments, start=1):
        lines.append(
            f"block {block_index}: empirical={value:.12e}, exact=1.000000000000e+00, error={value - 1.0:+.12e}"
        )

    lines.extend(
        [
            "",
            "DIAGNOSTICS AND SCORE DEFINITIONS",
            "-" * 72,
            f"Observable policy input: {prepared.observable_policy}",
            f"Required observables ({len(prepared.observable_names)}): "
            + ", ".join(prepared.observable_names),
            "IAT: multi-chain Geyer initial-positive sequence followed by initial-monotone sequence.",
            "Autocovariance convention: biased denominator N at every lag.",
            "No IAT floor is imposed. Each observable reports status, retained pair count, and last included lag.",
            "R-hat: maximum of rank-normalized split and folded rank-normalized split R-hat; raw value, no threshold.",
            "Moment diagnostic: |empirical-exact| divided by the exact target SD of that observable; descriptive only.",
            "Actual postwarmup target-gradient calls are counted. The current gradient is cached across trajectories.",
            "With warmup present, every retained HMC transition costs exactly L new target-gradient calls.",
            "Block score=min_f 1/(gradients_per_retained_draw*IAT(f)).",
            f"Configuration score=geometric mean of the {len(prepared.seed_blocks)} block scores.",
            "No IAT/ESS averaging over observables or blocks is performed.",
            "Any required undefined IAT or any nonfinite trajectory makes efficiency undefined.",
            "No R-hat, acceptance, moment, or other eligibility/screening gate is applied.",
            "Selection tie-break (exact ties only): larger minimum block score, fewer gradients/draw, smaller epsilon, smaller L.",
            "IAT status glossary:",
            "  ok_truncated_nonpositive = stopped immediately before the first nonpositive adjacent-lag pair.",
            "  ok_full_window = every available adjacent-lag pair was positive; the sample window was exhausted.",
            "  undefined_shape / undefined_insufficient_chains / undefined_insufficient_draws = required array structure or length is absent.",
            "  undefined_nonfinite_input / undefined_nonfinite_pair = nonfinite data or correlation-pair estimate occurred.",
            "  undefined_zero_variance = within-chain or total variance needed by the estimator is zero/invalid.",
            "  undefined_no_positive_pair = the initial adjacent-lag pair was not positive.",
            "  undefined_nonpositive_tau = unfloored IPS/IMS produced a nonpositive final IAT.",
            "",
            "INTERNAL SELF-TESTS",
            "-" * 72,
        ]
    )
    lines.extend(self_test_messages)

    lines.extend(
        [
            "",
            "CONFIGURATION SUMMARY",
            "-" * 72,
            "epsilon      L  status                         GM_eff       min_eff    median_eff       max_eff   mean_acc    max_Rhat  max_moment_SD  nonfinite",
        ]
    )
    ordered_results = sorted(
        configuration_results, key=lambda item: (item.epsilon, item.L)
    )
    for result in ordered_results:
        lines.append(
            f"{result.epsilon:7.4f} {result.L:6d}  {result.status:<28} "
            f"{_format_number(result.geometric_mean_efficiency, '.6e'):>12} "
            f"{_format_number(result.minimum_block_efficiency, '.6e'):>12} "
            f"{_format_number(result.median_block_efficiency, '.6e'):>12} "
            f"{_format_number(result.maximum_block_efficiency, '.6e'):>12} "
            f"{result.mean_acceptance:10.6f} "
            f"{_format_number(result.maximum_rhat, '.6f'):>11} "
            f"{_format_number(result.maximum_standardized_moment_error, '.6e'):>14} "
            f"{result.total_nonfinite_trajectories:10d}"
        )

    lines.extend(
        [
            "",
            _format_efficiency_grid(
                ordered_results, prepared.epsilon_values, prepared.L_values
            ),
        ]
    )

    lines.extend(
        [
            "",
            "ALL CONFIGURATION-BLOCK RESULTS",
            "-" * 72,
            "epsilon      L block status                         accept  grad/draw  worst_obs     worst_iat_status                 worst_IAT last_lag pairs    block_eff   max_Rhat max_moment_SD fail_W fail_R",
        ]
    )
    for result in ordered_results:
        for block in result.block_results:
            worst_diagnostic = next(
                (
                    item
                    for item in block.observable_diagnostics
                    if item.name == block.worst_observable
                ),
                None,
            )
            last_lag = worst_diagnostic.iat.last_included_lag if worst_diagnostic else -1
            pairs = worst_diagnostic.iat.pair_count if worst_diagnostic else 0
            worst_iat_status = worst_diagnostic.iat.status if worst_diagnostic else "undefined"
            lines.append(
                f"{block.epsilon:7.4f} {block.L:6d} {block.block_index + 1:5d} "
                f"{block.status:<28} {block.acceptance:7.4f} "
                f"{block.gradients_per_retained_draw:10.4f} "
                f"{block.worst_observable:<13} "
                f"{worst_iat_status:<32} "
                f"{_format_number(block.worst_iat, '.6g'):>10} "
                f"{last_lag:8d} {pairs:5d} "
                f"{_format_number(block.block_efficiency, '.6e'):>12} "
                f"{_format_number(block.maximum_rhat, '.6f'):>10} "
                f"{_format_number(block.maximum_standardized_moment_error, '.6e'):>13} "
                f"{block.warmup_nonfinite_trajectories:6d} "
                f"{block.retained_nonfinite_trajectories:6d}"
            )

    block_efficiencies = np.asarray(
        [block.block_efficiency for block in representative.block_results]
    )
    lines.extend(
        [
            "",
            "SELECTED FIXED HMC REPRESENTATIVE",
            "-" * 72,
            f"epsilon={representative.epsilon:.12g}, L={representative.L}, trajectory time T={representative.epsilon * representative.L:.12g}",
            "Block efficiencies="
            + np.array2string(block_efficiencies, precision=12, separator=", "),
            f"Geometric mean={representative.geometric_mean_efficiency:.12e}",
            f"Minimum={representative.minimum_block_efficiency:.12e}",
            f"Median={representative.median_block_efficiency:.12e}",
            f"Maximum={representative.maximum_block_efficiency:.12e}",
            f"Mean retained acceptance={representative.mean_acceptance:.12f}",
            f"Maximum raw rank/split/folded R-hat across blocks={representative.maximum_rhat:.12f}",
            f"Maximum standardized moment error across blocks={representative.maximum_standardized_moment_error:.12e}",
            (
                "This representative is selected only from canonical tuning blocks. It has not seen championship seeds."
                if prepared.run_mode == "CANONICAL"
                else "This is a CUSTOM / NON-COMPARABLE representative; no championship-seed claim is made."
            ),
            "",
            "SELECTED REPRESENTATIVE: ALL OBSERVABLES IN ALL BLOCKS",
            "-" * 72,
            "block observable       IAT iat_status                     pairs last_lag first_excl        ESS   efficiency rank_Rhat fold_Rhat  max_Rhat empirical_mean    exact_mean     raw_error moment_SD",
        ]
    )
    for block in representative.block_results:
        for item in block.observable_diagnostics:
            lines.append(
                f"{block.block_index + 1:5d} {item.name:<12} "
                f"{_format_number(item.iat.tau, '.8g'):>10} "
                f"{item.iat.status:<30} "
                f"{item.iat.pair_count:5d} "
                f"{item.iat.last_included_lag:8d} "
                f"{item.iat.first_excluded_lag:10d} "
                f"{_format_number(item.ess, '.7g'):>10} "
                f"{_format_number(item.efficiency, '.7e'):>12} "
                f"{_format_number(item.rhat.rank, '.7f'):>9} "
                f"{_format_number(item.rhat.folded, '.7f'):>9} "
                f"{_format_number(item.rhat.combined, '.7f'):>9} "
                f"{item.empirical_mean:14.7e} "
                f"{item.exact_mean:14.7e} "
                f"{item.raw_moment_error:13.6e} "
                f"{item.standardized_moment_error:9.6e}"
            )

    lines.extend(
        [
            "",
            "END OF REPORT",
        ]
    )
    return "\n".join(lines)

def _run_chain_job(
    standard_momenta: np.ndarray,
    uniforms: np.ndarray,
    q_initial: np.ndarray,
    beta: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, int, int, int, int, int]:
    return _simulate_hmc_chain(
        standard_momenta,
        uniforms,
        q_initial,
        beta,
        epsilon,
        L,
        n_warmup,
        n_retained,
        target_parameters,
        mass_cholesky,
        inverse_mass_matrix,
        mass_is_identity,
    )

def _run_configuration_block(
    *,
    epsilon: float,
    L: int,
    block_index: int,
    seeds: Sequence[int],
    stream_cache: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    prepared: PreparedInputs,
) -> BlockResult:
    jobs = [
        (
            stream_cache[chain_index][0],
            stream_cache[chain_index][2],
            prepared.physical_starts[chain_index],
            prepared.beta,
            epsilon,
            L,
            prepared.n_warmup,
            prepared.n_retained,
            prepared.target_parameters,
            prepared.mass_cholesky,
            prepared.inverse_mass_matrix,
            prepared.mass_is_identity,
        )
        for chain_index in range(prepared.n_chains)
    ]

    if prepared.n_workers > 1:
        with ThreadPoolExecutor(
            max_workers=min(prepared.n_workers, prepared.n_chains)
        ) as executor:
            chain_outputs = list(executor.map(lambda arguments: _run_chain_job(*arguments), jobs))
    else:
        chain_outputs = [_run_chain_job(*arguments) for arguments in jobs]

    samples = np.stack([output[0] for output in chain_outputs], axis=0)
    accepted = sum(output[1] for output in chain_outputs)
    warmup_nonfinite = sum(output[2] for output in chain_outputs)
    retained_nonfinite = sum(output[3] for output in chain_outputs)
    retained_gradient_calls = sum(output[5] for output in chain_outputs)

    return _analyze_block(
        samples=samples,
        epsilon=epsilon,
        L=L,
        block_index=block_index,
        seeds=seeds,
        accepted_retained=accepted,
        retained_gradient_calls=retained_gradient_calls,
        warmup_nonfinite=warmup_nonfinite,
        retained_nonfinite=retained_nonfinite,
        observable_names=prepared.observable_names,
        exact_observable_means=prepared.exact_observable_means,
        exact_observable_sds=prepared.exact_observable_sds,
    )

def _reference_hmc_proposal(
    q: np.ndarray,
    p: np.ndarray,
    current_gradient: np.ndarray,
    epsilon: float,
    L: int,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Independent pure-NumPy oracle for the cached-gradient leapfrog."""

    def potential_and_gradient(state: np.ndarray) -> tuple[float, np.ndarray]:
        sx2, sy2, bend = (float(value) for value in target_parameters)
        x, y = (float(value) for value in state)
        residual = y - bend * (x * x - sx2)
        potential = 0.5 * x * x / sx2 + 0.5 * residual * residual / sy2
        gradient = np.asarray(
            [x / sx2 - 2.0 * bend * x * residual / sy2, residual / sy2],
            dtype=np.float64,
        )
        return potential, gradient

    proposed_q = np.asarray(q, dtype=np.float64).copy()
    proposed_p = np.asarray(p, dtype=np.float64).copy()
    proposed_gradient = np.asarray(current_gradient, dtype=np.float64).copy()
    proposed_p -= 0.5 * epsilon * proposed_gradient
    calls = 0
    for step in range(int(L)):
        velocity = proposed_p if mass_is_identity else inverse_mass_matrix @ proposed_p
        proposed_q += epsilon * velocity
        _potential, proposed_gradient = potential_and_gradient(proposed_q)
        calls += 1
        proposed_p -= (0.5 * epsilon if step == L - 1 else epsilon) * proposed_gradient
    proposed_potential, check_gradient = potential_and_gradient(proposed_q)
    finite = bool(
        np.all(np.isfinite(proposed_q))
        and np.all(np.isfinite(proposed_p))
        and np.all(np.isfinite(proposed_gradient))
        and math.isfinite(proposed_potential)
    )
    if not np.array_equal(check_gradient, proposed_gradient):
        raise RuntimeError("Reference HMC oracle produced inconsistent endpoint gradients.")
    return proposed_q, proposed_p, proposed_gradient, proposed_potential, calls, finite

def _validate_compiled_reference_microcheck() -> str:
    """Compare current native dispatchers with an independent Python oracle."""

    cases = (
        (
            np.asarray([0.8, -0.7]),
            np.asarray([0.2, -0.15]),
            np.asarray([4.0, 0.04, 0.25]),
            np.eye(2),
            True,
            0.001,
            7,
        ),
        (
            np.asarray([0.2, -0.12]),
            np.asarray([0.35, -0.28]),
            np.asarray([2.25, 0.09, 0.15]),
            np.linalg.inv(np.asarray([[2.0, 0.3], [0.3, 0.8]])),
            False,
            0.004,
            3,
        ),
    )
    for q, p, parameters, inverse_mass, identity, epsilon, length in cases:
        gradient = _target_gradient(q, parameters)
        native = _leapfrog_proposal(
            q, p, gradient, epsilon, length, parameters, inverse_mass, identity
        )
        reference = _reference_hmc_proposal(
            q, p, gradient, epsilon, length, parameters, inverse_mass, identity
        )
        for native_array, reference_array in zip(native[:3], reference[:3]):
            if not np.allclose(native_array, reference_array, rtol=0.0, atol=2.0e-13):
                raise RuntimeError("Compiled/reference HMC micro-oracle mismatch.")
        if (
            not math.isclose(native[3], reference[3], rel_tol=0.0, abs_tol=2.0e-13)
            or native[4:] != reference[4:]
        ):
            raise RuntimeError("Compiled/reference HMC scalar micro-oracle mismatch.")
    return "PASS: compiled/reference HMC micro-oracle for identity and general mass"

def _runtime_environment_summary() -> str:
    """Return the numerical runtime that accompanies stochastic results."""

    numba_version = numba.__version__ if NUMBA_AVAILABLE else "unavailable"
    llvmlite_version = llvmlite.__version__ if NUMBA_AVAILABLE else "unavailable"
    return (
        f"platform={platform.platform()}; Python={platform.python_version()}; "
        f"NumPy={np.__version__}; Numba={numba_version}; llvmlite={llvmlite_version}"
    )

def _first_tuning_block_stream_digest(prepared: PreparedInputs) -> str:
    """Hash canonical little-endian random arrays without simulating a chain."""

    total_iterations = prepared.n_warmup + prepared.n_retained
    digest = hashlib.sha256()
    for seed in prepared.seed_blocks[0]:
        for array in _make_random_streams(seed, total_iterations, prepared.dimension):
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()

def _validate_portable_native_runtime(prepared: PreparedInputs) -> str:
    """Verify current cache-free dispatchers without a stochastic path canary."""

    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "Canonical Grex production requires Numba. Select the configured "
            "Python 3.12 sampling kernel before starting the tuning grid."
        )
    _validate_compiled_reference_microcheck()
    if not getattr(_simulate_hmc_chain, "signatures", ()):
        momenta, _reserved, uniforms = _make_random_streams(
            998877, 2, prepared.dimension
        )
        _simulate_hmc_chain(
            momenta,
            uniforms,
            prepared.physical_starts[0],
            prepared.beta,
            0.001,
            1,
            1,
            1,
            prepared.target_parameters,
            prepared.mass_cholesky,
            prepared.inverse_mass_matrix,
            prepared.mass_is_identity,
        )
    expected_dispatchers = (
        ("_potential_energy", _potential_energy, _potential_energy_grex_20260829d),
        ("_target_gradient", _target_gradient, _target_gradient_grex_20260829d),
        ("_kinetic_energy", _kinetic_energy, _kinetic_energy_grex_20260829d),
        ("_vector_is_finite", _vector_is_finite, _vector_is_finite_grex_20260829d),
        ("_leapfrog_proposal", _leapfrog_proposal, _leapfrog_proposal_grex_20260829d),
        ("_simulate_hmc_chain", _simulate_hmc_chain, _simulate_hmc_chain_grex_20260829d),
    )
    problems: list[str] = []
    for public_name, active, expected in expected_dispatchers:
        if active is not expected:
            problems.append(f"{public_name} does not point to the current epoch")
            continue
        python_function = getattr(active, "py_func", None)
        if python_function is None or not python_function.__name__.endswith("_grex_20260829d"):
            problems.append(f"{public_name} has the wrong Python function identity")
        cache_object = getattr(active, "_cache", None)
        if cache_object is not None and cache_object.__class__.__name__ != "NullCache":
            problems.append(f"{public_name} has persistent native caching enabled")
    if not getattr(_leapfrog_proposal, "signatures", ()):
        problems.append("the HMC proposal was not compiled by self-tests")
    if not getattr(_simulate_hmc_chain, "signatures", ()):
        problems.append("the complete HMC chain was not compiled by self-tests")
    if problems:
        raise RuntimeError("Portable native-runtime preflight failed: " + "; ".join(problems))
    stream_digest = _first_tuning_block_stream_digest(prepared)
    expected_digest = "4b75bb878357604485c75d467561c269a2330d8c2121e91fcd6d910c3a10783f"
    if prepared.run_mode == "CANONICAL" and stream_digest != expected_digest:
        raise RuntimeError(
            "Canonical random-stream parity failed: expected "
            + expected_digest
            + ", received "
            + stream_digest
        )
    return (
        "PASS: portable cache-free HMC native preflight "
        f"({HMC_RUNTIME_PREFLIGHT_VERSION}); {_runtime_environment_summary()}; "
        "compiled/reference micro-oracle=PASS; tuning-block-1 stream SHA-256="
        + stream_digest
    )

def run_self_tests(prepared: PreparedInputs) -> tuple[str, ...]:
    """Run deterministic common and HMC-specific checks before tuning."""

    messages: list[str] = []
    dimension = prepared.dimension
    canonical_parameters = np.asarray([4.0, 0.04, 0.25], dtype=np.float64)
    hand_point = np.asarray([1.0, -0.45], dtype=np.float64)
    if (
        not math.isclose(_potential_energy(hand_point, canonical_parameters), 1.25, rel_tol=0.0, abs_tol=2.0e-14)
        or not np.allclose(
            _target_gradient(hand_point, canonical_parameters),
            np.asarray([-3.5, 7.5]),
            rtol=0.0,
            atol=2.0e-13,
        )
    ):
        raise RuntimeError("Self-test failed: literal Banana target oracle.")

    q_test = np.asarray([1.46, -0.31 * math.sqrt(prepared.target_variances[1])])
    analytic = _target_gradient(q_test, prepared.target_parameters)
    finite_difference = np.empty(2)
    for coordinate in range(2):
        step = 1.0e-6 * max(1.0, abs(float(q_test[coordinate])))
        plus, minus = q_test.copy(), q_test.copy()
        plus[coordinate] += step
        minus[coordinate] -= step
        finite_difference[coordinate] = (
            _potential_energy(plus, prepared.target_parameters)
            - _potential_energy(minus, prepared.target_parameters)
        ) / (2.0 * step)
    if not np.allclose(analytic, finite_difference, rtol=2.0e-8, atol=2.0e-8):
        raise RuntimeError("Self-test failed: analytic Banana gradient.")
    reconstructed = _physical_to_latent(
        prepared.physical_starts,
        prepared.beta,
        prepared.sigma_x,
        prepared.sigma_y,
        prepared.banana_b,
    )
    if not np.allclose(reconstructed, prepared.standardized_starts, rtol=0.0, atol=2.0e-14):
        raise RuntimeError("Self-test failed: latent/physical start round trip.")
    if not np.allclose(prepared.mass_cholesky @ prepared.mass_cholesky.T, prepared.mass_matrix, atol=1.0e-12) or not np.allclose(prepared.inverse_mass_matrix @ prepared.mass_matrix, np.eye(2), atol=1.0e-12):
        raise RuntimeError("Self-test failed: mass algebra.")

    names, means, sds = _observable_specification(1.0, 2.0, 0.2, 0.25)
    expected_means = np.asarray([0.0, 0.0, 4.0, 0.0, 2.04])
    expected_sds = np.sqrt(np.asarray([4.0, 2.04, 32.0, 40.16, 56.3232]))
    if names != ("x", "y", "x^2", "x*y", "y^2") or not np.allclose(means, expected_means, atol=2.0e-14) or not np.allclose(sds, expected_sds, atol=2.0e-13):
        raise RuntimeError("Self-test failed: exact Banana moments.")
    nodes, weights = np.polynomial.hermite.hermgauss(8)
    nodes = math.sqrt(2.0) * nodes
    weights = weights / math.sqrt(math.pi)
    q_means, q_seconds = np.zeros(5), np.zeros(5)
    for i, zx in enumerate(nodes):
        for j, zr in enumerate(nodes):
            x = 2.0 * zx
            y = 0.25 * (x * x - 4.0) + 0.2 * zr
            values = np.asarray([x, y, x * x, x * y, y * y])
            weight = weights[i] * weights[j]
            q_means += weight * values
            q_seconds += weight * values * values
    q_sds = np.sqrt(np.maximum(0.0, q_seconds - q_means * q_means))
    if not np.allclose(q_means, expected_means, atol=2.0e-12) or not np.allclose(q_sds, expected_sds, atol=3.0e-12):
        raise RuntimeError("Self-test failed: independent Gauss-Hermite moments.")
    messages.append("PASS: Banana target, gradient, starts, mass, and exact physical moments")

    autocovariance_input = np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    expected_autocovariance = np.asarray([[1.25, 0.3125, -0.375, -0.5625], [5.0, 1.25, -1.5, -2.25]])
    if not np.allclose(_autocovariance_fft(autocovariance_input), expected_autocovariance, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("Self-test failed: biased FFT autocovariance.")
    odd = np.asarray([[0.0, 1.0, 99.0, 2.0, 3.0], [10.0, 11.0, 99.0, 12.0, 13.0]])
    expected_split = np.asarray([[0.0, 1.0], [10.0, 11.0], [2.0, 3.0], [12.0, 13.0]])
    if not np.array_equal(_split_chains(odd), expected_split) or not np.array_equal(_average_ranks(np.asarray([10.0, 10.0, 20.0, 30.0])), np.asarray([1.5, 1.5, 3.0, 4.0])):
        raise RuntimeError("Self-test failed: split/rank conventions.")
    ppf = _standard_normal_ppf(np.asarray([0.025, 0.5, 0.975]))
    if not np.allclose(ppf, np.asarray([-1.959963984540054, 0.0, 1.959963984540054]), atol=2.0e-7):
        raise RuntimeError("Self-test failed: inverse-normal approximation.")
    rng = np.random.default_rng(24681357)
    iid = rng.standard_normal((4, 5000))
    iid_rhat = _rank_normalized_split_folded_rhat(iid)
    shifted = iid.copy(); shifted[0] += 2.0
    shifted_rhat = _rank_normalized_split_folded_rhat(shifted)
    scaled = iid * np.asarray([1.0, 1.0, 4.0, 4.0])[:, None]
    scale_rhat = _rank_normalized_split_folded_rhat(scaled)
    if iid_rhat.status != "ok" or not (0.98 < iid_rhat.combined < 1.03) or shifted_rhat.combined <= 1.1 or scale_rhat.folded <= scale_rhat.rank + 0.05:
        raise RuntimeError("Self-test failed: rank-normalized R-hat behavior.")
    messages.append("PASS: autocovariance, split/ranks, inverse normal, and rank/folded R-hat")

    def ar_chains(phi: float) -> np.ndarray:
        output = np.empty((4, 16000))
        output[:, 0] = rng.normal(scale=1.0 / math.sqrt(1.0 - phi * phi), size=4)
        innovations = rng.standard_normal((4, output.shape[1] - 1))
        for draw in range(1, output.shape[1]):
            output[:, draw] = phi * output[:, draw - 1] + innovations[:, draw - 1]
        return output
    positive_iat = _multi_chain_geyer_iat(ar_chains(0.8))
    antithetic_iat = _multi_chain_geyer_iat(ar_chains(-0.5))
    constant_iat = _multi_chain_geyer_iat(np.ones((4, 100)))
    nonfinite_iat = _multi_chain_geyer_iat(np.asarray([[0.0, 1.0, math.nan, 2.0]] * 4))
    if not positive_iat.is_defined or not (6.0 < positive_iat.tau < 12.5) or not antithetic_iat.is_defined or not (0.15 < antithetic_iat.tau < 0.65) or constant_iat.status != "undefined_zero_variance" or nonfinite_iat.status != "undefined_nonfinite_input":
        raise RuntimeError("Self-test failed: Geyer IAT behavior.")
    messages.append(f"PASS: Geyer positive IAT={positive_iat.tau:.6f}, antithetic IAT={antithetic_iat.tau:.6f}, and undefined statuses")

    mechanics_cases = (
        (np.asarray([0.8, -0.7]), prepared.mass_cholesky @ np.asarray([0.3, -0.2]), prepared.target_parameters, prepared.inverse_mass_matrix, prepared.mass_is_identity, 0.001, 17),
        (np.asarray([0.2, -0.12]), np.asarray([0.35, -0.28]), np.asarray([2.25, 0.09, 0.15]), np.linalg.inv(np.asarray([[2.0, 0.3], [0.3, 0.8]])), False, 0.004, 3),
    )
    for q0, p0, parameters, inverse_mass, identity, epsilon, length in mechanics_cases:
        grad0 = _target_gradient(q0, parameters)
        q1, p1, grad1, _u1, calls1, finite1 = _leapfrog_proposal(q0, p0, grad0, epsilon, length, parameters, inverse_mass, identity)
        q2, p2, _grad2, _u2, calls2, finite2 = _leapfrog_proposal(q1, -p1, grad1, epsilon, length, parameters, inverse_mass, identity)
        if not finite1 or not finite2 or calls1 != length or calls2 != length or not np.allclose(q2, q0, rtol=0.0, atol=3.0e-11) or not np.allclose(p2, -p0, rtol=0.0, atol=3.0e-11):
            raise RuntimeError("Self-test failed: HMC reversibility/gradient count.")
    general_q, general_p, general_parameters, general_inverse, _, _, _ = mechanics_cases[1]
    expected_kinetic = 0.5 * float(general_p @ general_inverse @ general_p)
    if not math.isclose(_kinetic_energy(general_p, general_inverse, False), expected_kinetic, rel_tol=1.0e-14, abs_tol=1.0e-14):
        raise RuntimeError("Self-test failed: general-mass kinetic energy.")
    messages.append("PASS: canonical and general-mass HMC reversibility with exact gradient counts")

    def proposal_map(state: np.ndarray, epsilon: float, length: int) -> np.ndarray:
        q0, p0 = state[:2], state[2:]
        gradient0 = _target_gradient(q0, general_parameters)
        q1, p1, _g1, _u1, calls, finite = _leapfrog_proposal(q0, p0, gradient0, epsilon, length, general_parameters, general_inverse, False)
        if not finite or calls != length:
            raise RuntimeError("Self-test failed: finite proposal map.")
        return np.concatenate((q1, p1))
    base_state = np.concatenate((general_q, general_p))
    jacobian = np.empty((4, 4))
    jacobian_step = 1.0e-6
    for coordinate in range(4):
        plus, minus = base_state.copy(), base_state.copy()
        plus[coordinate] += jacobian_step
        minus[coordinate] -= jacobian_step
        jacobian[:, coordinate] = (proposal_map(plus, 0.004, 3) - proposal_map(minus, 0.004, 3)) / (2.0 * jacobian_step)
    determinant = float(np.linalg.det(jacobian))
    if not math.isclose(abs(determinant), 1.0, rel_tol=0.0, abs_tol=3.0e-7):
        raise RuntimeError("Self-test failed: HMC volume preservation.")
    initial_energy = _potential_energy(general_q, general_parameters) + _kinetic_energy(general_p, general_inverse, False)
    coarse = proposal_map(base_state, 0.04, 1)
    fine = proposal_map(base_state, 0.02, 2)
    coarse_error = abs(_potential_energy(coarse[:2], general_parameters) + _kinetic_energy(coarse[2:], general_inverse, False) - initial_energy)
    fine_error = abs(_potential_energy(fine[:2], general_parameters) + _kinetic_energy(fine[2:], general_inverse, False) - initial_energy)
    ratio = coarse_error / fine_error
    if not math.isfinite(ratio) or not (3.5 < ratio < 4.5):
        raise RuntimeError("Self-test failed: second-order fixed-time energy error.")
    messages.append(f"PASS: volume determinant={determinant:.12f} and second-order energy ratio={ratio:.6f}")

    messages.append(_validate_compiled_reference_microcheck())
    momenta, thermostat_momenta, uniforms = _make_random_streams(998877, 5, dimension)
    replay = _make_random_streams(998877, 5, dimension)
    if not all(np.array_equal(first, second) for first, second in zip((momenta, thermostat_momenta, uniforms), replay)):
        raise RuntimeError("Self-test failed: deterministic random substreams.")
    digest = hashlib.sha256()
    for array in (momenta, thermostat_momenta, uniforms):
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes(order="C"))
    if digest.hexdigest() != "9ca4f4a3a225c694403ef747f0ad9b52adec1e81de5e8d3c07396ad68be5ca83":
        raise RuntimeError("Self-test failed: tiny random-stream digest.")
    chain = _simulate_hmc_chain(momenta, uniforms, prepared.physical_starts[0], prepared.beta, 0.001, 4, 2, 3, prepared.target_parameters, prepared.mass_cholesky, prepared.inverse_mass_matrix, prepared.mass_is_identity)
    if chain[0].shape != (3, 2) or chain[2] != 0 or chain[3] != 0 or chain[4] != 9 or chain[5] != 12:
        raise RuntimeError("Self-test failed: tiny-chain accounting or finiteness.")
    messages.append("PASS: three-substream RNG replay/digest and tiny-chain 9/12 gradient accounting")

    latent_x = rng.standard_normal((prepared.n_chains, 120)) * prepared.sigma_x / math.sqrt(prepared.beta)
    residual = rng.standard_normal((prepared.n_chains, 120)) * prepared.sigma_y / math.sqrt(prepared.beta)
    synthetic = np.empty((prepared.n_chains, 120, 2))
    synthetic[:, :, 0] = latent_x
    synthetic[:, :, 1] = prepared.banana_b * (latent_x * latent_x - prepared.sigma_x**2) + residual
    invalid_block = _analyze_block(
        samples=synthetic, epsilon=0.01, L=15, block_index=0,
        seeds=prepared.seed_blocks[0], accepted_retained=prepared.n_chains * 100,
        retained_gradient_calls=prepared.n_chains * 120 * 15,
        warmup_nonfinite=1, retained_nonfinite=0,
        observable_names=prepared.observable_names,
        exact_observable_means=prepared.exact_observable_means,
        exact_observable_sds=prepared.exact_observable_sds,
    )
    if invalid_block.status != "undefined_nonfinite_simulation" or any(math.isfinite(item.efficiency) for item in invalid_block.observable_diagnostics):
        raise RuntimeError("Self-test failed: nonfinite metric propagation.")
    block_a = replace(invalid_block, status="ok", block_index=0, block_efficiency=0.25, warmup_nonfinite_trajectories=0)
    block_b = replace(block_a, block_index=1, block_efficiency=1.0)
    valid_aggregate = _aggregate_blocks((block_a, block_b))
    invalid_aggregate = _aggregate_blocks((invalid_block,))
    if valid_aggregate.status != "ok" or not math.isclose(valid_aggregate.geometric_mean_efficiency, 0.5, abs_tol=1.0e-15) or invalid_aggregate.status != "undefined_block_metric":
        raise RuntimeError("Self-test failed: block GM aggregation.")
    messages.append("PASS: five-observable construction, nonfinite propagation, and block GM")
    return tuple(messages)

def run_hmc_tuning(
    inputs: RunInputs,
    *,
    preflight_only: bool = False,
) -> dict[str, object]:
    """Run one completely specified HMC tuning campaign."""

    if not isinstance(preflight_only, (bool, np.bool_)):
        raise ValueError("preflight_only must be True or False.")
    prepared = _prepare_inputs(inputs)
    _validate_run_mode(prepared)
    _validate_common_parity_seals(prepared)

    self_test_messages = (
        run_self_tests(prepared)
        if prepared.run_self_tests
        else ("SKIPPED: run_self_tests=False (CUSTOM mode only).",)
    )
    native_preflight_message = _validate_portable_native_runtime(prepared)
    print(native_preflight_message, flush=True)
    if preflight_only:
        print(
            "GREX PREFLIGHT COMPLETE: the current cache-free native HMC "
            "sampler, Banana equations, reversibility/volume/order checks, "
            "gradient accounting, RNG, diagnostics, and finite smoke chain "
            "all passed. No tuning trajectory or Cartesian-grid configuration "
            "was simulated, and no report was written. Set "
            "GREX_PREFLIGHT_ONLY=False in the bottom input block and execute "
            "this entire cell again to start production tuning.",
            flush=True,
        )
        return {
            "preflight_only": True,
            "preflight_message": native_preflight_message,
            "self_test_messages": self_test_messages,
            "prepared_inputs": prepared,
            "output_path": None,
            "common_input_fingerprint": prepared.common_input_fingerprint,
            "shared_protocol_fingerprint": _shared_protocol_fingerprint(prepared),
        }

    destination = _resolve_output_destination(prepared)
    total_iterations = prepared.n_warmup + prepared.n_retained
    stream_caches: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for block in prepared.seed_blocks:
        stream_caches.append(
            [
                _make_random_streams(seed, total_iterations, prepared.dimension)
                for seed in block
            ]
        )

    auxiliary_momentum_moments = tuple(
        _retained_standardized_momentum_second_moment(cache, prepared.n_warmup)
        for cache in stream_caches
    )

    # Compile the Numba kernels before worker threads are opened.
    compile_momenta, _compile_thermostat_momenta, compile_uniforms = (
        _make_random_streams(13579, 2, prepared.dimension)
    )
    _simulate_hmc_chain(
        compile_momenta,
        compile_uniforms,
        prepared.physical_starts[0],
        prepared.beta,
        prepared.epsilon_values[0],
        1,
        1,
        1,
        prepared.target_parameters,
        prepared.mass_cholesky,
        prepared.inverse_mass_matrix,
        prepared.mass_is_identity,
    )

    configuration_results: list[ConfigurationResult] = []
    total_configurations = len(prepared.epsilon_values) * len(prepared.L_values)
    configuration_number = 0

    for epsilon in prepared.epsilon_values:
        for L in prepared.L_values:
            configuration_number += 1
            if prepared.print_progress:
                print(
                    f"[{configuration_number:>3}/{total_configurations}] "
                    f"HMC epsilon={epsilon:.6g}, L={L}: running "
                    f"{len(prepared.seed_blocks)} tuning blocks...",
                    flush=True,
                )
            block_results: list[BlockResult] = []
            for block_index, seeds in enumerate(prepared.seed_blocks):
                block_results.append(
                    _run_configuration_block(
                        epsilon=epsilon,
                        L=L,
                        block_index=block_index,
                        seeds=seeds,
                        stream_cache=stream_caches[block_index],
                        prepared=prepared,
                    )
                )
            aggregate = _aggregate_configuration(epsilon, L, block_results)
            configuration_results.append(aggregate)
            if prepared.print_progress:
                print(
                    "    status="
                    + aggregate.status
                    + ", GM efficiency="
                    + _format_number(aggregate.geometric_mean_efficiency, ".8e"),
                    flush=True,
                )

    representative = _select_representative(configuration_results)
    report = _build_report(
        configuration_results=configuration_results,
        representative=representative,
        auxiliary_momentum_moments=auxiliary_momentum_moments,
        prepared=prepared,
        self_test_messages=self_test_messages + (native_preflight_message,),
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
        "common_input_fingerprint": prepared.common_input_fingerprint,
        "input_preparation_fingerprint": _input_preparation_fingerprint(),
        "shared_protocol_fingerprint": _shared_protocol_fingerprint(prepared),
        "diagnostics_fingerprint": _diagnostics_fingerprint(),
        "common_pipeline_fingerprint": _common_pipeline_fingerprint(),
        "hmc_implementation_fingerprint": _hmc_implementation_fingerprint(),
        "source_parity_verification_mode": _source_verification_mode(),
    }

GREX_PREFLIGHT_ONLY = True
# HELDOUT_ENGINE_END

HELDOUT_METHOD = 'HMC'
HELDOUT_PROTOCOL = json.loads('{"IAT":"Original multi-chain Geyer initial-positive then initial-monotone sequence; biased autocovariance denominator N, no floor or hand-set lag cutoff.","Rhat":"Maximum of rank-normalized split and folded rank-normalized split R-hat; descriptive, no threshold.","comparison_id":"BANANA2_HMC_BOUNDED_THC_V4_HELDOUT_20261012_v1","date_frozen":"2026-10-12","exact_means":[0.0,0.0,4.0,0.0,2.04],"failure_rule":"Unchanged in all three tuning engines: any warmup or retained nonfinite trajectory, or undefined required IAT, makes block efficiency and full method score undefined; all blocks remain in the report.","gradient_cost":"Actual newly executed physical-target gradients in retained transitions only, excluding warmup and cached reuse. Finite trajectories cost L per draw (HMC 80; bounded THC 60; THC-v4 20). Thermostat/coupling derivatives are not physical-target calls; this is not wall-clock cost.","input_templates":{"HMC":{"L_values":[80],"allow_overwrite":false,"banana_b":0.25,"beta":1.0,"epsilon_values":[0.04],"mass_matrix":[[0.25,0.0],[0.0,0.49019607843137253]],"n_retained":10000,"n_warmup":1000,"n_workers":4,"observable_policy":"ALL_LINEAR_AND_UNIQUE_QUADRATIC","output_directory":null,"output_filename":"HMC_results.txt","print_progress":false,"print_report":false,"run_label":"CUSTOM: FROZEN HELD-OUT","run_mode":"CUSTOM","run_self_tests":true,"seed_blocks":[[2880099833,2489486650,2451017942,2880490925],[129025579,3302271073,1664117088,152623616],[249633759,3154472917,2530652495,148029082],[4038453250,698588159,3957987257,3768817257],[2893569068,137308266,2372641710,3314743546],[4179363669,2029140000,3971043061,1777623370],[3209876497,339851714,1427706838,547900704],[4188427725,3436669780,439669184,1957906058],[2428438358,161299918,3357008242,1421873829],[1887429089,2356290723,3781647023,3446686430]],"sigma_x":2.0,"sigma_y":0.2,"standardized_starts":[[-2.0,-2.0],[-2.0,2.0],[2.0,-2.0],[2.0,2.0]]},"THC_bounded":{"L_values":[60],"Q_values":[0.5],"a_values":[4.0],"allow_overwrite":false,"banana_b":0.25,"beta":1.0,"coupling_family":"EXP_TANH","epsilon_values":[0.05],"lambda_values":[0.5596157879354227],"mass_matrix":[[0.25,0.0],[0.0,0.49019607843137253]],"n_retained":10000,"n_warmup":1000,"n_workers":4,"observable_policy":"ALL_LINEAR_AND_UNIQUE_QUADRATIC","output_directory":null,"output_filename":"THC_bounded_results.txt","print_progress":false,"print_report":false,"run_label":"CUSTOM: FROZEN HELD-OUT","run_mode":"CUSTOM","run_self_tests":true,"seed_blocks":[[2880099833,2489486650,2451017942,2880490925],[129025579,3302271073,1664117088,152623616],[249633759,3154472917,2530652495,148029082],[4038453250,698588159,3957987257,3768817257],[2893569068,137308266,2372641710,3314743546],[4179363669,2029140000,3971043061,1777623370],[3209876497,339851714,1427706838,547900704],[4188427725,3436669780,439669184,1957906058],[2428438358,161299918,3357008242,1421873829],[1887429089,2356290723,3781647023,3446686430]],"sigma_x":2.0,"sigma_y":0.2,"standardized_starts":[[-2.0,-2.0],[-2.0,2.0],[2.0,-2.0],[2.0,2.0]],"thermostat_sigma":1.0,"thermostat_starts":[-1.0,1.0,1.0,-1.0]},"THC_v4":{"L_values":[20],"Q_values":[0.1],"a_values":[8.0],"allow_overwrite":false,"automatic_boundary_expansion":false,"banana_b":0.25,"beta":1.0,"boundary_Q_lower":0.003125,"boundary_Q_upper":2.0,"boundary_a_lower":0.5,"boundary_a_upper":64.0,"boundary_kappa_lower":0.0625,"boundary_kappa_upper":1.5,"boundary_keep_fixed_configurations":1,"boundary_tau_lower":0.0625,"boundary_tau_upper":4.0,"conditional_standardized_starts":[-1.0,1.0,1.0,-1.0],"epsilon_values":[0.1],"forced_full_configurations":[],"kappa_values":[0.5],"mass_matrix":[[0.25,0.0],[0.0,0.49019607843137253]],"n_retained":10000,"n_warmup":1000,"n_workers":4,"observable_policy":"ALL_LINEAR_AND_UNIQUE_QUADRATIC","output_directory":null,"output_filename":"THC_v4_results.txt","print_progress":false,"print_report":false,"run_label":"CUSTOM: FROZEN HELD-OUT","run_mode":"CUSTOM","run_self_tests":true,"seed_blocks":[[2880099833,2489486650,2451017942,2880490925],[129025579,3302271073,1664117088,152623616],[249633759,3154472917,2530652495,148029082],[4038453250,698588159,3957987257,3768817257],[2893569068,137308266,2372641710,3314743546],[4179363669,2029140000,3971043061,1777623370],[3209876497,339851714,1427706838,547900704],[4188427725,3436669780,439669184,1957906058],[2428438358,161299918,3357008242,1421873829],[1887429089,2356290723,3781647023,3446686430]],"sigma_x":2.0,"sigma_y":0.2,"stage1_anchor_pairs":[[0.1,20]],"stage1_block_indices":[1],"stage1_keep_method_tuples":1,"stage1_n_retained":4,"stage1_n_warmup":0,"stage2_block_indices":[2],"stage2_keep_fixed_configurations":1,"stage2_n_retained":4,"stage2_n_warmup":0,"standardized_starts":[[-2.0,-2.0],[-2.0,2.0],[2.0,-2.0],[2.0,2.0]],"tau_values":[0.5]}},"legacy_stage_fields":"THC-v4 stage and boundary fields only satisfy unchanged input validation. The held-out driver never invokes search, ranking or boundary expansion.","methods":{"HMC":{"L":80,"epsilon":0.04},"THC_bounded":{"L":60,"Q":0.5,"a":4.0,"epsilon":0.05,"lambda_value":0.5596157879354227},"THC_v4":{"L":20,"Q":0.1,"a":8.0,"epsilon":0.1,"kappa":0.5,"tau":0.5}},"n_blocks":10,"n_chains":4,"n_retained":10000,"n_warmup":1000,"no_retuning":true,"paired_comparison":"THC_bounded/HMC and THC_v4/HMC using shared fresh starts/seeds and raw PCG64 substreams. A shared HMC run serves both comparisons. Approximate pointwise 95% Student-t intervals on ten paired log-efficiency ratios; no claim of simultaneous family-wise coverage.","physical_observables":["x","y","x^2","x*y","y^2"],"physical_starts":[[-4.0,2.6],[-4.0,3.4],[4.0,2.6],[4.0,3.4]],"provenance":{"HMC":{"engine_sha256":"fa141378ade96b5985cd93a6e476dad0a981b7e56ad83eecce11f86e358e7583","source_path":"grex_banana_hmc_bounded_v4_expanded_20260925\\\\code\\\\banana_hmc_EXPANDED_20260925.py","source_sha256":"e72350e0c0ff79621ca8ced7768d9ff1829558d4bd2a04a5f3c0e4c8d6be32fd","tested_configurations":143,"tuning_report":"results\\\\GREX_tuning_2026-10-07\\\\Banana\\\\HMC\\\\results\\\\HMC_Banana_expanded_tuning_20260925.txt","tuning_report_sha256":"e48dc23d11c066ec68ec7b97797d71bbe1cfce146644ab7217b9b07349f34c7b","tuning_score":0.004123623804605706},"THC_bounded":{"engine_sha256":"217890866b49f4b64d2291b98b037b6b950cd17679ff834346a002ee52808f46","source_path":"grex_banana_hmc_bounded_v4_expanded_20260925\\\\code\\\\banana_bounded_EXPANDED_20260925.py","source_sha256":"1f870e92b4b73938aadf3a0f27436802d5d2a1bf56c59f36fd41c96b489a1b43","tested_configurations":81081,"tuning_report":"results\\\\GREX_tuning_2026-10-07\\\\Banana\\\\Bounded_scale_THC\\\\results\\\\FINAL_TUNING_REPORT.txt","tuning_report_sha256":"83809b680ce61fbfc1a069aacb5f53b1902bed6fb838260706f507c56e58c4e1","tuning_score":0.004989031260354801},"THC_v4":{"engine_sha256":"8337b4a7f0fb269f8debb519c381f7cc95aa346703db33a8a117981c74f2927c","source_path":"grex_banana_hmc_bounded_v4_expanded_20260925\\\\code\\\\banana_v4_EXPANDED_20260925.py","source_sha256":"cd4aeca59e2bfaa460907b75d290bcafd1c38a3cba6b8694a7347811a930e22c","tested_configurations":45278,"tuning_report":"results\\\\GREX_tuning_2026-10-07\\\\Banana\\\\Conditional_shear_THC_v4\\\\results\\\\FINAL_TUNING_REPORT.txt","tuning_report_sha256":"691e2fef7b747d3f90dadacf3a89aef0f0c0610b74d1a0a106c35a9d423381b0","tuning_score":0.041179178669632575}},"score":"Block E=1/(actual retained target-gradient calls per draw * max physical-observable IAT); S=geometric mean of ALL ten block efficiencies.","secondary_outcomes":"Maximum and descriptive geometric mean of blockwise worst IAT, max R-hat, acceptance, exact/estimated moments, nonfinite event counts. No reselection after inspection.","seed_audit":{"derivation":"First four SHA256 bytes (big endian) of DOMAIN|PHASE|block=BB|chain=CC|counter=K; counters change only for recorded collisions, never scores.","domain":"paper-on-thc|BANANA2|HMC-bounded-v4|held-out|20261012|v1","overlaps_after_audit":0,"path_size_inventory_sha256":"25883a26b7aca587a5a8bc3c7d07600a268caaf2147f1903bc93b5f9d2d71c07","rejected_candidate_values":[],"scope":"Recorded integer occurrences in local project sources/reports, including downloaded tuning and previous held-out files; unknown remote-only seeds cannot be audited.","text_bytes_scanned":3053417852,"text_files_scanned":2654},"seed_blocks":[[2880099833,2489486650,2451017942,2880490925],[129025579,3302271073,1664117088,152623616],[249633759,3154472917,2530652495,148029082],[4038453250,698588159,3957987257,3768817257],[2893569068,137308266,2372641710,3314743546],[4179363669,2029140000,3971043061,1777623370],[3209876497,339851714,1427706838,547900704],[4188427725,3436669780,439669184,1957906058],[2428438358,161299918,3357008242,1421873829],[1887429089,2356290723,3781647023,3446686430]],"selection":"One best tested gradient-normalized-efficiency representative per method, chosen only from the completed expanded tuning reports. No secondary-IAT reselection.","target":"2D centered Banana","target_definition":"U=x^2/(2*sigma_x^2)+(y-b*(x^2-sigma_x^2))^2/(2*sigma_y^2); beta=1, sigma_x=2, sigma_y=0.2, b=0.25.","thermostat_refresh":"Both THC coordinates s persist. Only physical and thermostat momenta refresh. Rejection restores both old q and s."}')
HELDOUT_PROTOCOL_SHA256 = '2a58f8520c71b0c9172bd7448a4ee1dcc487f6f4c0d61555ecc7774a3d597ead'
SOFTWARE_TEST_SEEDS = [[3457391801, 2530152304, 3733223790, 2360747083], [2422645997, 2761630847, 1986666772, 732349441], [2180884550, 4228537742, 2126947739, 2339551230], [471704273, 524668083, 756930411, 3844415107], [2836023456, 1195830159, 701982472, 3161265260], [2309126044, 2406109090, 3967555137, 3397635173], [1868015777, 1999588096, 2072149320, 2181710436], [609324951, 2771352225, 160704928, 3765900630], [3789285514, 891159038, 2963934625, 3777897026], [788453779, 462336479, 840193605, 164650783]]
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
        raise RuntimeError('Frozen protocol changed. Do not retune this final experiment.')
    if not NUMBA_AVAILABLE:
        raise RuntimeError('Numba is required for this fixed production experiment.')
    text = Path(__file__).read_text(encoding='utf-8')
    engine = text.split('# HELDOUT_ENGINE_BEGIN\n', 1)[1].split('# HELDOUT_ENGINE_END\n', 1)[0]
    expected = HELDOUT_PROTOCOL['provenance'][HELDOUT_METHOD]['engine_sha256']
    if hashlib.sha256(engine.encode()).hexdigest() != expected:
        raise RuntimeError('Scientific engine differs from the completed tuning source.')
    final = [s for b in HELDOUT_PROTOCOL['seed_blocks'] for s in b]
    test = [s for b in SOFTWARE_TEST_SEEDS for s in b]
    if len(final) != 40 or len(test) != 40 or len(set(final + test)) != 80:
        raise RuntimeError('Final and software-test seeds must be unique and disjoint.')
    return expected


def _heldout_protocol(testing=False):
    p = json.loads(json.dumps(HELDOUT_PROTOCOL))
    if testing:
        p.update(comparison_id=p['comparison_id'] + '_SOFTWARE_TEST_ONLY',
                 n_warmup=32, n_retained=512, seed_blocks=SOFTWARE_TEST_SEEDS)
    return p


def _heldout_prepare(workers=4, testing=False):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('workers must be an integer from 1 through 4')
    p = _heldout_protocol(testing)
    arguments = json.loads(json.dumps(p['input_templates'][HELDOUT_METHOD]))
    arguments.update(seed_blocks=p['seed_blocks'], n_warmup=p['n_warmup'],
                     n_retained=p['n_retained'], n_workers=workers,
                     run_label='CUSTOM: SOFTWARE TEST ONLY' if testing else 'CUSTOM: FROZEN HELD-OUT')
    if HELDOUT_METHOD == 'HMC':
        prepared = _prepare_inputs(RunInputs(**arguments))
        _validate_run_mode(prepared)
        _validate_common_parity_seals(prepared)
    elif HELDOUT_METHOD == 'THC_bounded':
        prepared = _prepare_thc_inputs(THCRunInputs(**arguments))
        _validate_thc_run_mode(prepared)
        _validate_common_parity_seals(prepared.common)
    else:
        prepared = _prepare_inputs(THCRunInputs(**arguments))
    common = _heldout_common(prepared)
    if common.n_chains != 4 or len(common.observable_names) != 5:
        raise RuntimeError('Unexpected number of chains or observables.')
    np.testing.assert_allclose(common.physical_starts, p['physical_starts'], rtol=0, atol=1e-14)
    return prepared


def _heldout_common(prepared):
    return prepared.common if HELDOUT_METHOD == 'THC_bounded' else prepared


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
    observables = []
    for values in data['observable_diagnostics']:
        values = dict(values)
        values['iat'] = IATResult(**values['iat'])
        values['rhat'] = RhatResult(**values['rhat'])
        observables.append(ObservableDiagnostic(**values))
    data['observable_diagnostics'] = tuple(observables)
    return BlockResult(**data)


def _heldout_one_block(prepared, index):
    common = _heldout_common(prepared)
    seeds = common.seed_blocks[index]
    streams = [_make_random_streams(s, common.n_warmup + common.n_retained, common.dimension) for s in seeds]
    physical_hash = hashlib.sha256()
    all_hash = hashlib.sha256()
    for stream in streams:
        for i, array in enumerate(stream):
            data = np.ascontiguousarray(array, dtype='<f8').tobytes(order='C')
            all_hash.update(data)
            if i in (0, 2):
                physical_hash.update(data)
    kwargs = dict(epsilon=common.epsilon_values[0], L=common.L_values[0],
                  block_index=index, seeds=seeds, stream_cache=streams, prepared=prepared)
    if HELDOUT_METHOD == 'HMC':
        physical = _run_configuration_block(**kwargs)
        thermostat = None
    elif HELDOUT_METHOD == 'THC_bounded':
        result = _run_configuration_block(**kwargs, lambda_value=prepared.lambda_values[0],
                                         a=prepared.a_values[0], Q=prepared.Q_values[0])
        physical, thermostat = result.physical, asdict(result.thermostat)
    else:
        setting = HELDOUT_PROTOCOL['methods'][HELDOUT_METHOD]
        # Preserve the original v4 block runner's indexing; only this block is populated.
        caches = [None] * len(common.seed_blocks)
        caches[index] = streams
        result = _run_configuration_block(FixedConfiguration(**setting), index,
            common.n_warmup, common.n_retained, caches, prepared)
        physical, thermostat = result.physical, asdict(result.conditional)
    record = _heldout_clean(dict(physical=asdict(physical), thermostat=thermostat,
        physical_and_acceptance_stream_sha256=physical_hash.hexdigest(),
        all_three_streams_sha256=all_hash.hexdigest()))
    record['record_sha256'] = _heldout_hash(record)
    return record


def _heldout_preflight(workers=4):
    _heldout_guard()
    # Never pass production seeds to self-tests or native compilation checks.
    prepared = _heldout_prepare(workers, testing=True)
    messages = list(run_self_tests(prepared))
    if HELDOUT_METHOD != 'THC_v4':
        messages.append(_validate_portable_native_runtime(prepared))
    sample = _heldout_one_block(prepared, 0)
    block = _heldout_decode_block(sample)
    common = _heldout_common(prepared)
    if len(block.observable_diagnostics) != 5:
        raise RuntimeError('Software-check observable count mismatch.')
    if block.warmup_nonfinite_trajectories or block.retained_nonfinite_trajectories:
        raise RuntimeError('Software-check trajectory became nonfinite; investigate, do not change the frozen setting.')
    if block.retained_gradient_calls != common.n_chains * common.n_retained * block.L:
        raise RuntimeError('Gradient-accounting check failed.')
    restored = _heldout_clean(asdict(_heldout_decode_block(json.loads(json.dumps(sample)))))
    if restored != sample['physical']:
        raise RuntimeError('Checkpoint serialization round trip failed.')
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
    for key in ('comparison_id', 'method', 'protocol_sha256', 'script_sha256', 'runtime', 'testing'):
        if saved['metadata'][key] != metadata[key]:
            raise RuntimeError('Checkpoint does not match this frozen run: ' + key)
    records = saved['records']
    if len(records) > metadata['protocol']['n_blocks']:
        raise RuntimeError('Checkpoint has too many blocks.')
    for index, record in enumerate(records):
        body = {k: v for k, v in record.items() if k != 'record_sha256'}
        if record.get('record_sha256') != _heldout_hash(body):
            raise RuntimeError('Block checksum mismatch.')
        block = _heldout_decode_block(record)
        setting = metadata['protocol']['methods'][HELDOUT_METHOD]
        if (block.block_index != index or list(block.seeds) != metadata['protocol']['seed_blocks'][index]
                or block.epsilon != setting['epsilon'] or block.L != setting['L']
                or len(block.observable_diagnostics) != 5):
            raise RuntimeError('Checkpoint block identifiers do not match the frozen schedule.')
    return records


def _heldout_checkpoint(path, metadata, records, replace_existing):
    payload = dict(format='BANANA2-heldout-checkpoint-v1', metadata=metadata, records=records)
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
                   mean_gradient_calls_per_draw=float(np.mean([b.gradients_per_retained_draw for b in blocks])),
                   warmup_nonfinite_trajectories=sum(b.warmup_nonfinite_trajectories for b in blocks),
                   retained_nonfinite_trajectories=sum(b.retained_nonfinite_trajectories for b in blocks))
    label = 'SOFTWARE TEST ONLY -- NOT HELD-OUT RESULTS' if metadata['testing'] else 'FRESH HELD-OUT RESULTS'
    text = [f'{HELDOUT_METHOD}: {label}', '=' * 88,
            'Comparison: ' + p['comparison_id'],
            'Protocol SHA256: ' + metadata['protocol_sha256'],
            'Script SHA256: ' + metadata['script_sha256'],
            'Completed UTC: ' + completed_utc,
            'Runtime: ' + json.dumps(metadata['runtime'], sort_keys=True),
            'Fixed settings: ' + json.dumps(p['methods'][HELDOUT_METHOD], sort_keys=True),
            f"{p['n_blocks']} paired blocks; {p['n_chains']} chains/block; "
            f"{p['n_warmup']} warmup + {p['n_retained']} retained draws/chain.",
            'Target: ' + p['target_definition'],
            'Observables: x, y, x^2, x*y, y^2. Base mass diag(0.25,1/2.04).',
            'IAT: unchanged multi-chain Geyer IPS/IMS, biased denominator N, no floor.',
            'R-hat: max(rank-normalized split, folded rank-normalized split); descriptive.',
            'Gradient cost: actual retained target-gradient calls; warmup excluded; cached gradients not counted twice.',
            'Nonfinite warmup/retained trajectory OR undefined required IAT => undefined full score.',
            'No blocks omitted. Numerical rejection repeats the previous state.',
            'No adaptation, tuning or selection has been performed using these final blocks.']
    if HELDOUT_METHOD != 'HMC':
        text += [('h(s)=exp(log(1.75)*tanh(s/4)); rho(s)=N(0,1); Q=0.5.' if HELDOUT_METHOD == 'THC_bounded' else 's|x~N(x,0.5^2); c(s)=0.5*8*tanh(s/8); G=A.T*M0*A with A=[[1,0],[-c(s),1]]; Q=0.1.'),
                 's persists. Only p and p_s are refreshed; rejection restores q and s.',
                 'Thermostat moments are descriptive, never physical-score observables.']
    text += ['', 'METHOD SUMMARY']
    for key, value in summary.items():
        display = value if isinstance(value, (str, int)) else _heldout_number(value)
        text.append(f'{key}: {display}')
    text += ['', 'BLOCK RESULTS (one-based block numbers)',
             'block status g worst_IAT worst_observable efficiency acceptance max_Rhat max_moment_SD nonfinite_warmup nonfinite_retained']
    for b in blocks:
        values = [b.block_index + 1, b.status, _heldout_number(b.gradients_per_retained_draw),
                  _heldout_number(b.worst_iat), b.worst_observable, _heldout_number(b.block_efficiency),
                  _heldout_number(b.acceptance), _heldout_number(b.maximum_rhat),
                  _heldout_number(b.maximum_standardized_moment_error),
                  b.warmup_nonfinite_trajectories, b.retained_nonfinite_trajectories]
        text.append(' '.join(map(str, values)))
    text += ['', 'EXACT AND POOLED ESTIMATED PHYSICAL MOMENTS',
             'Equal chain/block lengths: estimates average all retained draws, including rejection repeats.',
             'observable exact_mean estimated_mean raw_error']
    for index, item in enumerate(blocks[0].observable_diagnostics):
        empirical = float(np.mean([b.observable_diagnostics[index].empirical_mean for b in blocks]))
        text.append(f'{item.name:<8} {_heldout_number(item.exact_mean):>18} '
                    f'{_heldout_number(empirical):>18} {_heldout_number(empirical-item.exact_mean):>18}')
    text += ['', 'ALL PHYSICAL OBSERVABLES IN ALL BLOCKS',
             'block observable IAT IAT_status pairs last_lag first_excluded_lag ESS efficiency rank_Rhat folded_Rhat max_Rhat empirical_mean exact_mean raw_error moment_SD']
    for b in blocks:
        for item in b.observable_diagnostics:
            values = [b.block_index + 1, item.name, _heldout_number(item.iat.tau), item.iat.status,
                      item.iat.pair_count, item.iat.last_included_lag, item.iat.first_excluded_lag,
                      *[_heldout_number(v) for v in (item.ess, item.efficiency, item.rhat.rank,
                         item.rhat.folded, item.rhat.combined, item.empirical_mean, item.exact_mean,
                         item.raw_moment_error, item.standardized_moment_error)]]
            text.append(' '.join(map(str, values)))
    if HELDOUT_METHOD != 'HMC':
        text += ['', ('THERMOSTAT MOMENTS (descriptive only; exact E[s]=0, E[s^2]=1)' if HELDOUT_METHOD == 'THC_bounded' else 'CONDITIONAL MOMENTS (descriptive only; eta=s-x, exact E[eta]=0, E[eta^2]=0.25)'),
                 'block mean_auxiliary second_moment_auxiliary']
        for i, record in enumerate(records):
            t = _heldout_restore_numbers(record['thermostat'])
            text.append(f"{i+1} {_heldout_number(t['empirical_mean'])} {_heldout_number(t['empirical_second_moment'])}")
    compact_blocks = []
    for record in records:
        entry = {k: v for k, v in record['physical'].items() if k != 'observable_diagnostics'}
        entry.update(physical_and_acceptance_stream_sha256=record['physical_and_acceptance_stream_sha256'],
                     all_three_streams_sha256=record['all_three_streams_sha256'])
        compact_blocks.append(entry)
    comparison_record = _heldout_clean(dict(
        format='BANANA2-heldout-comparison-report-v1', metadata=metadata,
        completed_utc=completed_utc, summary=summary, blocks=compact_blocks))
    text += ['', 'BEGIN COMPARISON_JSON', json.dumps(comparison_record, indent=2, allow_nan=False),
             'END COMPARISON_JSON', '',
             'Compare ONLY against the partner with the same protocol SHA256 and all matching block seeds/stream hashes.',
             'This standalone report does not declare a winner. Keep both full reports, including undefined scores.']
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
