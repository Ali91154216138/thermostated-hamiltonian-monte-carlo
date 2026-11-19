"""Run held-out S-QHMC on the Banana target."""
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

NUMBA_RUNTIME_EPOCH = "BANANA2_SQHMC_GREX_PORTABLE_20260830_V1"

NUMBA_CACHE_MODE = "persistent cache disabled in all modes"

NUMBA_CACHE_DIRECTORY = "not used"

try:
    import llvmlite
    import numba

    def njit(*args, **kwargs):
        portable_options = dict(kwargs)
        portable_options["cache"] = False
        return numba.njit(*args, **portable_options)

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

SQHMC_IMPLEMENTATION_VERSION = (
    "banana2-liu-zhang-scalar-log10normal-standardized-cached-leapfrog-grex-v1"
)

SQHMC_METHOD_INPUT_SCHEMA_VERSION = "banana-s-qhmc-method-inputs-v1"

SQHMC_RUNTIME_PREFLIGHT_VERSION = (
    "banana2-s-qhmc-portable-native-preflight-20260830-v1"
)

RANDOM_STREAM_VERSION = "seedsequence-pcg64-three-substreams-v1.0.0"

SUPPORTED_OBSERVABLE_POLICY = "ALL_LINEAR_AND_UNIQUE_QUADRATIC"

CANONICAL_OUTPUT_FILENAME = "S_QHMC_banana_tuning_results_GREX_PORTABLE_V1.txt"

CANONICAL_MU_M_VALUES = (-1.0, 0.0, 1.0)

CANONICAL_SIGMA_M_VALUES = (
    0.005,
    0.01,
    0.03,
    0.10,
    0.25,
    0.50,
    1.0,
    2.0,
    3.0,
)

EXPECTED_INPUT_PREPARATION_FINGERPRINT = (
    "8e6767f332cf26810005489774bc928995cea9f20b3ebf9b36aadbde48c438cf"
)

EXPECTED_DIAGNOSTICS_FINGERPRINT = (
    "dc2148c37f893c7fa5c22431760b2c46ac3aed96cce35909b333eb2abce91274"
)

EXPECTED_COMMON_PIPELINE_FINGERPRINT = (
    "6f5b462436209cddf4236cee615ac0e0bdb0988fd01d247225caa4533997221c"
)

EXPECTED_SHARED_PROTOCOL_FINGERPRINT = (
    "a7679d36fdd46df9c30b96bb24634f4dccf746a221021587e6702eded60ab351"
)

EXPECTED_SQHMC_IMPLEMENTATION_FINGERPRINT = (
    "27360a9782eae60f00d05508acdb617b231303812b6eee8831d97df601647bd9"
)

CANONICAL_SQHMC_METHOD_INPUT_FINGERPRINT = (
    "d34f801d9e2941a71cf636908f9cb50df00d90acaebf3fe0f58d6607cca98adc"
)

CANONICAL_FIRST_TUNING_BLOCK_STREAM_DIGEST = (
    "4b75bb878357604485c75d467561c269a2330d8c2121e91fcd6d910c3a10783f"
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
class SQHMCRunInputs(RunInputs):
    """Common inputs plus Liu--Zhang scalar S-QHMC inputs."""

    mu_m_values: Sequence[float]
    sigma_m_values: Sequence[float]

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
class PreparedSQHMCInputs:
    """Validated method grid wrapped around unchanged common inputs."""

    common: PreparedInputs
    mu_m_values: tuple[float, ...]
    sigma_m_values: tuple[float, ...]
    method_input_components: dict[str, object]
    method_input_fingerprint: str
    canonical_method_mismatches: tuple[str, ...]

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
class MassDiagnostic:
    """Descriptive retained moments of the sampled standard mass normal z."""

    empirical_mean: float
    exact_mean: float
    mean_error: float
    empirical_second_moment: float
    exact_second_moment: float
    second_moment_error: float

@dataclass(frozen=True)
class SQHMCBlockResult:
    """Physical block score plus a non-scoring mass-stream check."""

    physical: BlockResult
    mass: MassDiagnostic

    def __getattr__(self, name: str) -> object:
        """Forward method-neutral report fields to the physical block."""

        # During checkpoint unpickling, physical may not yet be restored.
        return getattr(object.__getattribute__(self, "physical"), name)

@dataclass(frozen=True)
class ConfigurationResult:
    """Block aggregate used to choose one fixed S-QHMC representative."""

    mu_m: float
    sigma_m: float
    epsilon: float
    L: int
    status: str
    block_results: tuple[SQHMCBlockResult, ...]
    geometric_mean_efficiency: float
    minimum_block_efficiency: float
    median_block_efficiency: float
    maximum_block_efficiency: float
    mean_acceptance: float
    maximum_rhat: float
    maximum_standardized_moment_error: float
    total_nonfinite_trajectories: int

@njit(cache=False, inline="always")
def _potential_energy_sqhmc_banana_grex_v1(
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
def _target_gradient_sqhmc_banana_grex_v1(
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
def _base_kinetic_energy_sqhmc_banana_grex_v1(
    standardized_p: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> float:
    """Kinetic energy of v=p/sqrt(m) under the fixed base mass M0."""

    if mass_is_identity:
        value = 0.0
        for coordinate in range(standardized_p.size):
            value += 0.5 * standardized_p[coordinate] * standardized_p[coordinate]
        return value

    value = 0.0
    for row in range(standardized_p.size):
        inverse_mass_p = 0.0
        for column in range(standardized_p.size):
            inverse_mass_p += (
                inverse_mass_matrix[row, column] * standardized_p[column]
            )
        value += 0.5 * standardized_p[row] * inverse_mass_p
    return value

@njit(cache=False, inline="always")
def _vector_is_finite_sqhmc_banana_grex_v1(values: np.ndarray) -> bool:
    for value in values:
        if not math.isfinite(value):
            return False
    return True

@njit(cache=False, nogil=True)
def _standardized_leapfrog_proposal_sqhmc_banana_grex_v1(
    q: np.ndarray,
    standardized_p: np.ndarray,
    current_gradient: np.ndarray,
    effective_epsilon: float,
    L: int,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Integrate one fixed-random-mass proposal using v=p/sqrt(m)."""

    proposed_q = q.copy()
    proposed_p = standardized_p.copy()
    proposed_gradient = current_gradient.copy()
    gradient_calls = 0

    for coordinate in range(proposed_p.size):
        proposed_p[coordinate] -= 0.5 * effective_epsilon * proposed_gradient[coordinate]
    if not _vector_is_finite_sqhmc_banana_grex_v1(proposed_p):
        return proposed_q, proposed_p, proposed_gradient, math.nan, gradient_calls, False

    for step in range(L):
        if mass_is_identity:
            for coordinate in range(proposed_q.size):
                proposed_q[coordinate] += effective_epsilon * proposed_p[coordinate]
        else:
            for row in range(proposed_q.size):
                velocity = 0.0
                for column in range(proposed_p.size):
                    velocity += inverse_mass_matrix[row, column] * proposed_p[column]
                proposed_q[row] += effective_epsilon * velocity
        if not _vector_is_finite_sqhmc_banana_grex_v1(proposed_q):
            return proposed_q, proposed_p, proposed_gradient, math.nan, gradient_calls, False

        proposed_gradient = _target_gradient_sqhmc_banana_grex_v1(
            proposed_q, target_parameters
        )
        gradient_calls += 1
        if not _vector_is_finite_sqhmc_banana_grex_v1(proposed_gradient):
            return proposed_q, proposed_p, proposed_gradient, math.nan, gradient_calls, False

        kick = 0.5 * effective_epsilon if step == L - 1 else effective_epsilon
        for coordinate in range(proposed_p.size):
            proposed_p[coordinate] -= kick * proposed_gradient[coordinate]
        if not _vector_is_finite_sqhmc_banana_grex_v1(proposed_p):
            return proposed_q, proposed_p, proposed_gradient, math.nan, gradient_calls, False

    proposed_potential = _potential_energy_sqhmc_banana_grex_v1(
        proposed_q, target_parameters
    )
    if not math.isfinite(proposed_potential):
        return proposed_q, proposed_p, proposed_gradient, proposed_potential, gradient_calls, False
    return proposed_q, proposed_p, proposed_gradient, proposed_potential, gradient_calls, True

@njit(cache=False, nogil=True)
def _simulate_s_qhmc_chain_sqhmc_banana_grex_v1(
    standard_momenta: np.ndarray,
    mass_standard_normals: np.ndarray,
    acceptance_uniforms: np.ndarray,
    q_initial: np.ndarray,
    beta: float,
    mu_m: float,
    sigma_m: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    """Run scalar S-QHMC with one independent log10 mass per trajectory."""

    total_iterations = n_warmup + n_retained
    samples = np.empty((n_retained, q_initial.size), dtype=np.float64)
    retained_mass_standard_normals = np.empty(n_retained, dtype=np.float64)
    q = q_initial.copy()
    potential = _potential_energy_sqhmc_banana_grex_v1(q, target_parameters)
    current_gradient = _target_gradient_sqhmc_banana_grex_v1(q, target_parameters)
    warmup_gradient_calls = 1 if n_warmup > 0 else 0
    retained_gradient_calls = 0 if n_warmup > 0 else 1
    accepted_retained = 0
    warmup_nonfinite = 0
    retained_nonfinite = 0
    momentum_scale = 1.0 / math.sqrt(beta)
    log_ten = math.log(10.0)

    for iteration in range(total_iterations):
        mass_z = mass_standard_normals[iteration]
        log10_mass = mu_m + sigma_m * mass_z
        inverse_sqrt_mass_exponent = -0.5 * log_ten * log10_mass
        mass_is_valid = (
            math.isfinite(log10_mass)
            and math.isfinite(inverse_sqrt_mass_exponent)
            and inverse_sqrt_mass_exponent < 709.0
            and inverse_sqrt_mass_exponent > -744.0
        )
        inverse_sqrt_mass = math.nan
        effective_epsilon = math.nan
        if mass_is_valid:
            inverse_sqrt_mass = math.exp(inverse_sqrt_mass_exponent)
            effective_epsilon = epsilon * inverse_sqrt_mass
            mass_is_valid = (
                math.isfinite(inverse_sqrt_mass)
                and inverse_sqrt_mass > 0.0
                and math.isfinite(effective_epsilon)
                and effective_epsilon > 0.0
            )

        standardized_p = np.empty(q.size, dtype=np.float64)
        if mass_is_identity:
            for coordinate in range(q.size):
                standardized_p[coordinate] = momentum_scale * standard_momenta[iteration, coordinate]
        else:
            for row in range(q.size):
                transformed_momentum = 0.0
                for column in range(q.size):
                    transformed_momentum += mass_cholesky[row, column] * standard_momenta[iteration, column]
                standardized_p[row] = momentum_scale * transformed_momentum
        kinetic_old = _base_kinetic_energy_sqhmc_banana_grex_v1(
            standardized_p, inverse_mass_matrix, mass_is_identity
        )

        proposal_gradient_calls = 0
        proposal_is_finite = mass_is_valid
        proposed_q = q.copy()
        proposed_p = standardized_p.copy()
        proposed_gradient = current_gradient.copy()
        proposed_potential = math.nan
        if proposal_is_finite:
            (
                proposed_q,
                proposed_p,
                proposed_gradient,
                proposed_potential,
                proposal_gradient_calls,
                proposal_is_finite,
            ) = _standardized_leapfrog_proposal_sqhmc_banana_grex_v1(
                q,
                standardized_p,
                current_gradient,
                effective_epsilon,
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
            for coordinate in range(proposed_p.size):
                proposed_p[coordinate] = -proposed_p[coordinate]
            kinetic_new = _base_kinetic_energy_sqhmc_banana_grex_v1(
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
            retained_mass_standard_normals[retained_index] = mass_z
            if accepted:
                accepted_retained += 1

    return (
        samples,
        retained_mass_standard_normals,
        accepted_retained,
        warmup_nonfinite,
        retained_nonfinite,
        warmup_gradient_calls,
        retained_gradient_calls,
    )

_potential_energy = _potential_energy_sqhmc_banana_grex_v1

_target_gradient = _target_gradient_sqhmc_banana_grex_v1

_kinetic_energy = _base_kinetic_energy_sqhmc_banana_grex_v1

_vector_is_finite = _vector_is_finite_sqhmc_banana_grex_v1

_leapfrog_proposal = _standardized_leapfrog_proposal_sqhmc_banana_grex_v1

_simulate_s_qhmc_chain = _simulate_s_qhmc_chain_sqhmc_banana_grex_v1

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
    # A rejected nonfinite proposal is not a nonfinite retained state.
    # Count every event below, but score all finite retained draws (including repeats).
    simulation_is_nonfinite = not np.all(np.isfinite(samples))

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

def _mass_diagnostic(values: np.ndarray) -> MassDiagnostic:
    """Check E[z]=0 and E[z^2]=1 for the retained mass-normal stream."""

    array = np.asarray(values, dtype=np.float64)
    empirical_mean = float(np.mean(array))
    empirical_second = float(np.mean(array * array))
    return MassDiagnostic(
        empirical_mean=empirical_mean,
        exact_mean=0.0,
        mean_error=empirical_mean,
        empirical_second_moment=empirical_second,
        exact_second_moment=1.0,
        second_moment_error=empirical_second - 1.0,
    )

def _aggregate_configuration(
    mu_m: float,
    sigma_m: float,
    epsilon: float,
    L: int,
    block_results: Sequence[SQHMCBlockResult],
) -> ConfigurationResult:
    """Wrap the common block aggregate with S-QHMC configuration parameters."""

    blocks = tuple(block_results)
    aggregate = _aggregate_blocks(tuple(block.physical for block in blocks))
    return ConfigurationResult(
        mu_m=float(mu_m),
        sigma_m=float(sigma_m),
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
        raise RuntimeError("No S-QHMC configuration has a defined five-block tuning score.")

    # Exact ties: larger minimum block score, then fewer gradients/draw, then
    # smaller epsilon, then smaller L.  No near-tie tolerance is introduced.
    return max(
        eligible,
        key=lambda result: (
            result.geometric_mean_efficiency,
            result.minimum_block_efficiency,
            -result.block_results[0].physical.gradients_per_retained_draw,
            -result.epsilon,
            -result.L,
            -result.mu_m,
            -result.sigma_m,
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

def _normalize_method_grid(
    name: str,
    values: Sequence[float],
    *,
    strictly_positive: bool,
) -> tuple[float, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of real numbers.") from error
    if not raw:
        raise ValueError(f"{name} must not be empty.")
    normalized: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name}[{index}] must be a real number.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite.")
        if strictly_positive and number <= 0.0:
            raise ValueError(f"{name}[{index}] must be strictly positive.")
        normalized.append(number)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return tuple(normalized)

def _prepare_s_qhmc_inputs(inputs: SQHMCRunInputs) -> PreparedSQHMCInputs:
    if not isinstance(inputs, SQHMCRunInputs):
        raise TypeError("Expected one SQHMCRunInputs object.")
    common = _prepare_inputs(inputs)
    mu_values = _normalize_method_grid(
        "mu_m_values", inputs.mu_m_values, strictly_positive=False
    )
    sigma_values = _normalize_method_grid(
        "sigma_m_values", inputs.sigma_m_values, strictly_positive=True
    )
    components = {
        "mass_law": "omega~N(mu_m,sigma_m^2); m=10^omega; M=m*M0",
        "mu_m_values": list(mu_values),
        "sigma_m_values": list(sigma_values),
    }
    mismatches: list[str] = []
    if mu_values != CANONICAL_MU_M_VALUES:
        mismatches.append("mu_m_values")
    if sigma_values != CANONICAL_SIGMA_M_VALUES:
        mismatches.append("sigma_m_values")
    return PreparedSQHMCInputs(
        common=common,
        mu_m_values=mu_values,
        sigma_m_values=sigma_values,
        method_input_components=components,
        method_input_fingerprint=_json_fingerprint(
            {"schema": SQHMC_METHOD_INPUT_SCHEMA_VERSION, "components": components}
        ),
        canonical_method_mismatches=tuple(mismatches),
    )

def _validate_s_qhmc_run_mode(prepared: PreparedSQHMCInputs) -> None:
    _validate_run_mode(prepared.common)
    if prepared.common.run_mode == "CANONICAL":
        if prepared.canonical_method_mismatches or (
            prepared.method_input_fingerprint
            != CANONICAL_SQHMC_METHOD_INPUT_FINGERPRINT
        ):
            raise ValueError(
                "CANONICAL S-QHMC method-input mismatch in: "
                + (", ".join(prepared.canonical_method_mismatches) or "overall hash")
            )
        implementation_fingerprint = _sqhmc_implementation_fingerprint()
        # A file-backed script and an ordinary IPython cell expose their exact
        # source, so the implementation seal is enforced there.  Some Jupyter
        # front ends execute a pasted cell without retaining its text in
        # inspect/linecache.  In that environment an exact source hash is
        # mathematically unavailable; the mandatory portable native preflight
        # below remains the guard before any tuning trajectory is simulated.
        if implementation_fingerprint not in (
            EXPECTED_SQHMC_IMPLEMENTATION_FINGERPRINT,
            "unavailable",
        ):
            raise RuntimeError(
                "Canonical S-QHMC implementation source seal failed: received "
                + implementation_fingerprint
                + ", expected "
                + EXPECTED_SQHMC_IMPLEMENTATION_FINGERPRINT
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
                if source_path.name
                == "banana_s_qhmc_tuner_GREX_PORTABLE.py"
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

def _sqhmc_implementation_fingerprint() -> str:
    """Hash S-QHMC-specific dynamics, aggregation wrapper, and tie-breaking code."""

    return _source_fingerprint(
        (
            SQHMCRunInputs,
            PreparedSQHMCInputs,
            MassDiagnostic,
            SQHMCBlockResult,
            ConfigurationResult,
            _normalize_method_grid,
            _prepare_s_qhmc_inputs,
            _potential_energy,
            _target_gradient,
            _kinetic_energy,
            _vector_is_finite,
            _leapfrog_proposal,
            _simulate_s_qhmc_chain,
            _reference_standardized_leapfrog,
            _reference_direct_random_mass_leapfrog,
            _validate_compiled_reference_microcheck,
            _mass_diagnostic,
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
    """Enforce every shared source seal that the host can attest exactly."""

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
        if actual not in (expected_source_seals[name], "unavailable")
    ]
    if mismatches:
        raise RuntimeError(
            "Shared HMC/THC source parity failed for: "
            + ", ".join(mismatches)
        )
    source_unavailable = any(
        actual == "unavailable" for actual in actual_source_seals.values()
    )
    if prepared.run_mode == "CANONICAL" and not source_unavailable:
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

def _boundary_labels(
    result: ConfigurationResult,
    prepared: PreparedSQHMCInputs,
) -> tuple[str, ...]:
    """Return descriptive grid-edge labels; never alter selection or the grid."""

    common = prepared.common
    labels: list[str] = []
    for name, value, grid in (
        ("mu_m", result.mu_m, prepared.mu_m_values),
        ("sigma_m", result.sigma_m, prepared.sigma_m_values),
        ("epsilon", result.epsilon, common.epsilon_values),
        ("L", result.L, common.L_values),
    ):
        if len(grid) <= 1:
            continue
        if value == min(grid):
            labels.append(name + "_min")
        if value == max(grid):
            labels.append(name + "_max")
    return tuple(labels)

def _format_efficiency_grid(
    results: Sequence[ConfigurationResult],
    mu_m_values: Sequence[float],
    sigma_m_values: Sequence[float],
    epsilon_values: Sequence[float],
    L_values: Sequence[int],
) -> str:
    lookup = {
        (result.mu_m, result.sigma_m, result.epsilon, result.L): result
        for result in results
    }
    column_width = 14
    block_count = len(results[0].block_results) if results else 0
    header_label = "epsilon \\ L"
    lines: list[str] = []
    for mu_m in mu_m_values:
        for sigma_m in sigma_m_values:
            lines.extend(
                [
                    f"mu_m={mu_m:.12g}, sigma_m={sigma_m:.12g}: "
                    f"{block_count}-block GM worst-observable efficiency",
                    f"{header_label:>12}"
                    + "".join(f"{L:>{column_width}d}" for L in L_values),
                ]
            )
            for epsilon in epsilon_values:
                cells = []
                for L in L_values:
                    value = lookup[
                        (float(mu_m), float(sigma_m), float(epsilon), int(L))
                    ].geometric_mean_efficiency
                    cells.append(f"{_format_number(value, '.6e'):>{column_width}}")
                lines.append(f"{epsilon:>12.6g}" + "".join(cells))
            lines.append("")
    return "\n".join(lines)

def _build_report(
    *,
    configuration_results: Sequence[ConfigurationResult],
    representative: ConfigurationResult,
    auxiliary_momentum_moments: Sequence[float],
    prepared: PreparedSQHMCInputs,
    self_test_messages: Sequence[str],
) -> str:
    method = prepared
    prepared = method.common
    lines: list[str] = []
    mode_note = (
        "CANONICAL / COMPARABLE"
        if prepared.run_mode == "CANONICAL"
        else "CUSTOM / NON-COMPARABLE"
    )
    lines.extend(
        [
            "FAIR-PLAYGROUND S-QHMC TUNING REPORT: 2D CENTERED BANANA",
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
            f"Shared protocol version: {SHARED_PROTOCOL_VERSION}",
            f"Shared protocol fingerprint: {_shared_protocol_fingerprint(prepared)}",
            f"Diagnostics version: {DIAGNOSTICS_VERSION}",
            f"Diagnostics source fingerprint: {_diagnostics_fingerprint()}",
            f"Common RNG/observable/score pipeline version: {COMMON_PIPELINE_VERSION}",
            f"Common pipeline source fingerprint: {_common_pipeline_fingerprint()}",
            f"S-QHMC implementation version: {SQHMC_IMPLEMENTATION_VERSION}",
            f"Expected S-QHMC implementation source fingerprint: {EXPECTED_SQHMC_IMPLEMENTATION_FINGERPRINT}",
            f"S-QHMC implementation source fingerprint: {_sqhmc_implementation_fingerprint()}",
            f"S-QHMC method-input schema: {SQHMC_METHOD_INPUT_SCHEMA_VERSION}",
            f"Canonical expected S-QHMC method-input fingerprint: {CANONICAL_SQHMC_METHOD_INPUT_FINGERPRINT}",
            f"S-QHMC method-input fingerprint: {method.method_input_fingerprint}",
            "Canonical method-input mismatches: "
            + (", ".join(method.canonical_method_mismatches) or "none"),
            f"Runtime preflight version: {SQHMC_RUNTIME_PREFLIGHT_VERSION}",
            f"Runtime epoch: {NUMBA_RUNTIME_EPOCH}",
            f"Runtime environment: {_runtime_environment_summary()}",
            f"Source-parity verification mode: {_source_verification_mode()}",
            f"Random-stream version: {RANDOM_STREAM_VERSION}",
            f"Canonical expected tuning-block-1 stream SHA-256: {CANONICAL_FIRST_TUNING_BLOCK_STREAM_DIGEST}",
            f"Actual tuning-block-1 stream SHA-256: {_first_tuning_block_stream_digest(method)}",
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
            f"d={prepared.dimension}, beta={prepared.beta:g}, "
            f"sigma_x={prepared.sigma_x:g}, sigma_y={prepared.sigma_y:g}, "
            f"b={prepared.banana_b:g}",
            "shared base mass matrix M0="
            + np.array2string(prepared.mass_matrix, precision=10, separator=", "),
            f"M0 is identity: {prepared.mass_is_identity}",
            "marginal physical variances="
            + np.array2string(
                prepared.target_variances, precision=10, separator=", "
            ),
            "q=(x,y); r=y-b*(x^2-sigma_x^2).",
            "U=x^2/(2*sigma_x^2)+r^2/(2*sigma_y^2).",
            "Under exp(-beta*U), x and r are independent centered Gaussians.",
            "Integrator: symmetric kick-drift-kick leapfrog plus Metropolis correction",
            "Scalar mass law: omega~N(mu_m,sigma_m^2), m=10^omega, M=m*M0.",
            "Mass is sampled once per trajectory and held fixed for all L steps.",
            "Native algebra: v=p/sqrt(m), epsilon_eff=epsilon/sqrt(m); exactly equivalent to p~N(0,m*M0/beta).",
            "",
            "ENTERED TUNING DESIGN",
            "-" * 72,
            f"mu_m values ({len(method.mu_m_values)}): {list(method.mu_m_values)}",
            f"sigma_m values ({len(method.sigma_m_values)}): {list(method.sigma_m_values)}",
            f"epsilon values ({len(prepared.epsilon_values)}): {list(prepared.epsilon_values)}",
            f"L values ({len(prepared.L_values)}): {list(prepared.L_values)}",
            "Configurations: "
            + str(
                len(method.mu_m_values)
                * len(method.sigma_m_values)
                * len(prepared.epsilon_values)
                * len(prepared.L_values)
            ),
            f"Blocks: {len(prepared.seed_blocks)}; chains/block: {prepared.n_chains}",
            f"Warmup/chain: {prepared.n_warmup}; retained/chain: {prepared.n_retained}",
            "The same physical normals, mass normals, and uniforms are reused across configurations within each seed block.",
            "The three arrays come from separate PCG64 child streams.",
            "There is no fixed momentum or mass start: both are freshly drawn at every transition.",
            "Final-comparison seeds are absent from this tuner.",
            "",
            "TUNING SEED BLOCKS",
        ]
    )
    for block_index, block in enumerate(prepared.seed_blocks, start=1):
        lines.append(f"block {block_index}: {list(block)}")

    lines.extend(["", "INITIAL CONDITIONS (printed exactly as used)"])
    for chain_index in range(prepared.n_chains):
        q_start = prepared.physical_starts[chain_index]
        residual = q_start[1] - prepared.banana_b * (
            q_start[0] * q_start[0] - prepared.sigma_x * prepared.sigma_x
        )
        lines.append(
            f"chain {chain_index + 1}: entered (z_x,z_r)="
            + np.array2string(
                prepared.standardized_starts[chain_index],
                precision=6,
                separator=", ",
            )
            + "; physical (x,y)="
            + np.array2string(
                prepared.physical_starts[chain_index],
                precision=12,
                separator=", ",
            )
            + f"; residual r={residual:.12g}"
            + f"; U={_potential_energy(q_start, prepared.target_parameters):.12g}"
        )

    lines.extend(
        [
            "",
            "EXACT PHYSICAL OBSERVABLE MOMENTS",
            "-" * 72,
            "The scored set is x, y, x^2, x*y, and y^2. The latent residual r, momenta, and mass variables are excluded.",
        ]
    )
    for name, mean, sd in zip(
        prepared.observable_names,
        prepared.exact_observable_means,
        prepared.exact_observable_sds,
    ):
        lines.append(f"{name:<8}: exact mean={mean:.12e}; exact SD={sd:.12e}")

    lines.extend(
        [
            "",
            "MINIMAL AUXILIARY MOMENTUM DIAGNOSTIC (DESCRIPTIVE ONLY)",
            "-" * 72,
            "Quantity: pooled retained beta*v^T*M0^-1*v/d; exact value=1, v=p/sqrt(m).",
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
            "With warmup present, every finite retained S-QHMC transition costs exactly L new target-gradient calls.",
            "Block score=min_f 1/(gradients_per_retained_draw*IAT(f)).",
            f"Configuration score=geometric mean of the {len(prepared.seed_blocks)} block scores.",
            "No IAT/ESS averaging over observables or blocks is performed.",
            "Any required undefined IAT or any nonfinite trajectory makes efficiency undefined.",
            "No R-hat, acceptance, moment, or other eligibility/screening gate is applied.",
            "Selection tie-break (exact ties only): larger minimum block score, fewer gradients/draw, smaller epsilon, smaller L, smaller mu_m, smaller sigma_m.",
            "Grid-boundary labels are descriptive only; there is no automatic expansion or retuning.",
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
            "mu_m sigma_m epsilon      L  status                         GM_eff       min_eff    median_eff       max_eff   mean_acc    max_Rhat  max_moment_SD  nonfinite boundary",
        ]
    )
    ordered_results = sorted(
        configuration_results,
        key=lambda item: (item.mu_m, item.sigma_m, item.epsilon, item.L),
    )
    for result in ordered_results:
        lines.append(
            f"{result.mu_m:4.1f} {result.sigma_m:7.3g} "
            f"{result.epsilon:7.4f} {result.L:6d}  {result.status:<28} "
            f"{_format_number(result.geometric_mean_efficiency, '.6e'):>12} "
            f"{_format_number(result.minimum_block_efficiency, '.6e'):>12} "
            f"{_format_number(result.median_block_efficiency, '.6e'):>12} "
            f"{_format_number(result.maximum_block_efficiency, '.6e'):>12} "
            f"{result.mean_acceptance:10.6f} "
            f"{_format_number(result.maximum_rhat, '.6f'):>11} "
            f"{_format_number(result.maximum_standardized_moment_error, '.6e'):>14} "
            f"{result.total_nonfinite_trajectories:10d} "
            + (",".join(_boundary_labels(result, method)) or "interior")
        )

    lines.extend(
        [
            "",
            _format_efficiency_grid(
                ordered_results,
                method.mu_m_values,
                method.sigma_m_values,
                prepared.epsilon_values,
                prepared.L_values,
            ),
        ]
    )

    lines.extend(
        [
            "",
            "ALL CONFIGURATION-BLOCK RESULTS",
            "-" * 72,
            "mu_m sigma_m epsilon      L block status                         accept  grad/draw  worst_obs     worst_iat_status                 worst_IAT last_lag pairs    block_eff   max_Rhat max_moment_SD fail_W fail_R",
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
                f"{result.mu_m:4.1f} {result.sigma_m:7.3g} "
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
            "SELECTED FIXED S-QHMC REPRESENTATIVE",
            "-" * 72,
            f"mu_m={representative.mu_m:.12g}, sigma_m={representative.sigma_m:.12g}",
            f"epsilon={representative.epsilon:.12g}, L={representative.L}, raw path parameter epsilon*L={representative.epsilon * representative.L:.12g}",
            "Median effective trajectory time "
            f"L*epsilon*10^(-mu_m/2)={representative.L * representative.epsilon * 10.0 ** (-0.5 * representative.mu_m):.12g}; "
            "the realized effective time is random when sigma_m>0.",
            "Descriptive grid-boundary labels="
            + (",".join(_boundary_labels(representative, method)) or "interior"),
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
                "This representative is selected only from canonical tuning blocks. It has not seen final-comparison seeds."
                if prepared.run_mode == "CANONICAL"
                else "This is a CUSTOM / NON-COMPARABLE representative; no final-comparison-seed claim is made."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "SELECTED REPRESENTATIVE MASS-NORMAL MOMENTS (DESCRIPTIVE ONLY)",
            "-" * 72,
            "For omega=mu_m+sigma_m*z, exact E[z]=0 and E[z^2]=1.",
            "These values are not observables, IAT inputs, thresholds, or selection scores.",
        ]
    )
    for block in representative.block_results:
        lines.append(
            f"block {block.block_index + 1}: mean(z)={block.mass.empirical_mean:.12e}; "
            f"mean(z^2)={block.mass.empirical_second_moment:.12e}; "
            f"errors=({block.mass.mean_error:+.12e}, {block.mass.second_moment_error:+.12e})"
        )
    lines.extend(
        [
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
    mass_standard_normals: np.ndarray,
    uniforms: np.ndarray,
    q_initial: np.ndarray,
    beta: float,
    mu_m: float,
    sigma_m: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    return _simulate_s_qhmc_chain(
        standard_momenta,
        mass_standard_normals,
        uniforms,
        q_initial,
        beta,
        mu_m,
        sigma_m,
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
    mu_m: float,
    sigma_m: float,
    epsilon: float,
    L: int,
    block_index: int,
    seeds: Sequence[int],
    stream_cache: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    prepared: PreparedSQHMCInputs,
) -> SQHMCBlockResult:
    common = prepared.common
    # Child stream 2 keeps its historical common-helper position for exact
    # HMC/THC source parity; S-QHMC consumes it as the trajectory mass normal.
    jobs = [
        (
            stream_cache[chain_index][0],
            stream_cache[chain_index][1],
            stream_cache[chain_index][2],
            common.physical_starts[chain_index],
            common.beta,
            mu_m,
            sigma_m,
            epsilon,
            L,
            common.n_warmup,
            common.n_retained,
            common.target_parameters,
            common.mass_cholesky,
            common.inverse_mass_matrix,
            common.mass_is_identity,
        )
        for chain_index in range(common.n_chains)
    ]

    if common.n_workers > 1:
        with ThreadPoolExecutor(
            max_workers=min(common.n_workers, common.n_chains)
        ) as executor:
            chain_outputs = list(executor.map(lambda arguments: _run_chain_job(*arguments), jobs))
    else:
        chain_outputs = [_run_chain_job(*arguments) for arguments in jobs]

    samples = np.stack([output[0] for output in chain_outputs], axis=0)
    mass_z = np.stack([output[1] for output in chain_outputs], axis=0)
    accepted = sum(output[2] for output in chain_outputs)
    warmup_nonfinite = sum(output[3] for output in chain_outputs)
    retained_nonfinite = sum(output[4] for output in chain_outputs)
    retained_gradient_calls = sum(output[6] for output in chain_outputs)

    physical = _analyze_block(
        samples=samples,
        epsilon=epsilon,
        L=L,
        block_index=block_index,
        seeds=seeds,
        accepted_retained=accepted,
        retained_gradient_calls=retained_gradient_calls,
        warmup_nonfinite=warmup_nonfinite,
        retained_nonfinite=retained_nonfinite,
        observable_names=common.observable_names,
        exact_observable_means=common.exact_observable_means,
        exact_observable_sds=common.exact_observable_sds,
    )
    return SQHMCBlockResult(physical=physical, mass=_mass_diagnostic(mass_z))

def _reference_standardized_leapfrog(
    q: np.ndarray,
    standardized_p: np.ndarray,
    current_gradient: np.ndarray,
    effective_epsilon: float,
    L: int,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Independent, short-path Python oracle for the native proposal."""

    q_ref = np.asarray(q, dtype=np.float64).copy()
    p_ref = np.asarray(standardized_p, dtype=np.float64).copy()
    gradient_ref = np.asarray(current_gradient, dtype=np.float64).copy()
    p_ref -= 0.5 * effective_epsilon * gradient_ref
    for step in range(int(L)):
        q_ref += effective_epsilon * (inverse_mass_matrix @ p_ref)
        gradient_ref = _target_gradient(q_ref, target_parameters)
        p_ref -= (
            0.5 * effective_epsilon if step == L - 1 else effective_epsilon
        ) * gradient_ref
    potential_ref = float(_potential_energy(q_ref, target_parameters))
    finite = bool(
        np.all(np.isfinite(q_ref))
        and np.all(np.isfinite(p_ref))
        and np.all(np.isfinite(gradient_ref))
        and math.isfinite(potential_ref)
    )
    return q_ref, p_ref, gradient_ref, potential_ref, int(L), finite

def _reference_direct_random_mass_leapfrog(
    q: np.ndarray,
    standardized_p: np.ndarray,
    current_gradient: np.ndarray,
    epsilon: float,
    L: int,
    scalar_mass: float,
    target_parameters: np.ndarray,
    inverse_base_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal p/M implementation used only to audit standardized algebra."""

    q_ref = np.asarray(q, dtype=np.float64).copy()
    sqrt_mass = math.sqrt(float(scalar_mass))
    p_ref = sqrt_mass * np.asarray(standardized_p, dtype=np.float64)
    gradient = np.asarray(current_gradient, dtype=np.float64).copy()
    p_ref -= 0.5 * epsilon * gradient
    for step in range(int(L)):
        q_ref += (epsilon / scalar_mass) * (inverse_base_mass @ p_ref)
        gradient = _target_gradient(q_ref, target_parameters)
        p_ref -= (0.5 * epsilon if step == L - 1 else epsilon) * gradient
    return q_ref, p_ref / sqrt_mass

def _validate_compiled_reference_microcheck() -> str:
    """Compare native and independent Python dynamics on two fixed cases."""

    cases = (
        (
            np.asarray([0.17, -0.11], dtype=np.float64),
            np.asarray([0.31, -0.24], dtype=np.float64),
            0.013,
            7,
            np.asarray([4.0, 0.04, 0.25], dtype=np.float64),
            np.eye(2, dtype=np.float64),
            True,
        ),
        (
            np.asarray([0.2, -0.1], dtype=np.float64),
            np.asarray([0.4, -0.25], dtype=np.float64),
            0.007,
            11,
            np.asarray([2.25, 0.09, 0.15], dtype=np.float64),
            np.linalg.inv(
                np.asarray([[2.0, 0.3], [0.3, 0.8]], dtype=np.float64)
            ),
            False,
        ),
    )
    for case_index, (
        q,
        p,
        epsilon_eff,
        L,
        target_parameters,
        inverse_mass,
        mass_is_identity,
    ) in enumerate(cases, start=1):
        gradient = _target_gradient(q, target_parameters)
        native = _leapfrog_proposal(
            q,
            p,
            gradient,
            epsilon_eff,
            L,
            target_parameters,
            inverse_mass,
            mass_is_identity,
        )
        reference = _reference_standardized_leapfrog(
            q,
            p,
            gradient,
            epsilon_eff,
            L,
            target_parameters,
            inverse_mass,
        )
        if not all(
            np.allclose(native[index], reference[index], rtol=5e-12, atol=5e-13)
            for index in (0, 1, 2)
        ) or not math.isclose(
            float(native[3]), float(reference[3]), rel_tol=5e-12, abs_tol=5e-13
        ) or native[4:] != reference[4:]:
            raise RuntimeError(
                "Portable native micro-oracle mismatch in case "
                + str(case_index)
            )
    return "PASS: compiled/reference scalar S-QHMC micro-oracle (identity and general base mass)"

def run_self_tests(prepared_s_qhmc: PreparedSQHMCInputs) -> tuple[str, ...]:
    """Run deterministic implementation checks before a tuning campaign."""

    messages: list[str] = []
    if not isinstance(prepared_s_qhmc, PreparedSQHMCInputs):
        raise TypeError("Self-tests require PreparedSQHMCInputs.")
    if prepared_s_qhmc.common.run_mode == "CANONICAL":
        expected_count = 3 * 9 * 8 * 6
        actual_count = (
            len(prepared_s_qhmc.mu_m_values)
            * len(prepared_s_qhmc.sigma_m_values)
            * len(prepared_s_qhmc.common.epsilon_values)
            * len(prepared_s_qhmc.common.L_values)
        )
        if actual_count != expected_count:
            raise RuntimeError("Self-test failed: canonical grid must have 1296 cells.")
    prepared = prepared_s_qhmc.common
    messages.append(_validate_compiled_reference_microcheck())
    # Exact scalar-mass algebra: m=10^omega and epsilon_eff=epsilon/sqrt(m).
    test_omega = -0.37
    test_mass = 10.0 ** test_omega
    test_epsilon_raw = 0.04
    algebraic = test_epsilon_raw * math.exp(-0.5 * math.log(10.0) * test_omega)
    direct = test_epsilon_raw / math.sqrt(test_mass)
    if not math.isclose(algebraic, direct, rel_tol=1e-15, abs_tol=0.0):
        raise RuntimeError("Self-test failed: standardized scalar-mass algebra.")
    audit_q = np.asarray([0.2, -0.11], dtype=np.float64)
    audit_v = np.asarray([0.31, 0.17], dtype=np.float64)
    audit_target_parameters = np.asarray([2.25, 0.09, 0.15], dtype=np.float64)
    audit_inverse_base_mass = np.linalg.inv(
        np.asarray([[1.7, 0.2], [0.2, 0.9]], dtype=np.float64)
    )
    audit_gradient = _target_gradient(audit_q, audit_target_parameters)
    standardized_result = _reference_standardized_leapfrog(
        audit_q,
        audit_v,
        audit_gradient,
        direct,
        13,
        audit_target_parameters,
        audit_inverse_base_mass,
    )
    direct_result = _reference_direct_random_mass_leapfrog(
        audit_q,
        audit_v,
        audit_gradient,
        test_epsilon_raw,
        13,
        test_mass,
        audit_target_parameters,
        audit_inverse_base_mass,
    )
    if not np.allclose(
        standardized_result[0], direct_result[0], rtol=1e-13, atol=1e-14
    ) or not np.allclose(
        standardized_result[1], direct_result[1], rtol=1e-13, atol=1e-14
    ):
        raise RuntimeError("Self-test failed: direct random-mass equivalence.")
    messages.append(
        "PASS: m=10^omega standardized and literal p/(m*M0) leapfrog equivalence"
    )
    dimension = prepared.dimension
    canonical_parameters = np.asarray([4.0, 0.04, 0.25], dtype=np.float64)
    hand_point = np.asarray([1.0, -0.45], dtype=np.float64)
    if (
        not math.isclose(
            _potential_energy(hand_point, canonical_parameters),
            1.25,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        )
        or not np.allclose(
            _target_gradient(hand_point, canonical_parameters),
            np.asarray([-3.5, 7.5]),
            rtol=0.0,
            atol=2.0e-13,
        )
    ):
        raise RuntimeError("Self-test failed: literal Banana target oracle.")

    q_gradient_test = np.asarray(
        [1.46, -0.31 * math.sqrt(prepared.target_variances[1])],
        dtype=np.float64,
    )
    analytic_gradient = _target_gradient(q_gradient_test, prepared.target_parameters)
    finite_difference_gradient = np.empty(dimension, dtype=np.float64)
    for coordinate in range(dimension):
        finite_difference_step = 1.0e-6 * max(
            1.0, abs(float(q_gradient_test[coordinate]))
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
        raise RuntimeError("Self-test failed: analytic Banana gradient.")

    reconstructed_starts = _physical_to_latent(
        prepared.physical_starts,
        prepared.beta,
        prepared.sigma_x,
        prepared.sigma_y,
        prepared.banana_b,
    )
    if not np.allclose(
        reconstructed_starts,
        prepared.standardized_starts,
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise RuntimeError("Self-test failed: latent/physical start round trip.")
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

    names, means, sds = _observable_specification(1.0, 2.0, 0.2, 0.25)
    expected_names = ("x", "y", "x^2", "x*y", "y^2")
    expected_means = np.asarray([0.0, 0.0, 4.0, 0.0, 2.04])
    expected_variances = np.asarray([4.0, 2.04, 32.0, 40.16, 56.3232])
    expected_sds = np.sqrt(expected_variances)
    if (
        names != expected_names
        or not np.allclose(means, expected_means, rtol=0.0, atol=2.0e-14)
        or not np.allclose(sds, expected_sds, rtol=0.0, atol=2.0e-13)
    ):
        raise RuntimeError("Self-test failed: exact Banana moments.")

    nodes, weights = np.polynomial.hermite.hermgauss(8)
    nodes = math.sqrt(2.0) * nodes
    weights = weights / math.sqrt(math.pi)
    quadrature_means = np.zeros(5)
    quadrature_seconds = np.zeros(5)
    for x_index, z_x in enumerate(nodes):
        for r_index, z_r in enumerate(nodes):
            x = 2.0 * z_x
            y = 0.25 * (x * x - 4.0) + 0.2 * z_r
            values = np.asarray([x, y, x * x, x * y, y * y])
            weight = weights[x_index] * weights[r_index]
            quadrature_means += weight * values
            quadrature_seconds += weight * values * values
    quadrature_sds = np.sqrt(
        np.maximum(
            0.0,
            quadrature_seconds - quadrature_means * quadrature_means,
        )
    )
    if not np.allclose(
        quadrature_means, expected_means, rtol=0.0, atol=2.0e-12
    ) or not np.allclose(
        quadrature_sds, expected_sds, rtol=0.0, atol=3.0e-12
    ):
        raise RuntimeError("Self-test failed: independent Gauss-Hermite moments.")
    expected_physical_starts = np.asarray(
        [[-4.0, 2.6], [-4.0, 3.4], [4.0, 2.6], [4.0, 3.4]],
        dtype=np.float64,
    )
    if prepared.run_mode == "CANONICAL" and not np.allclose(
        prepared.physical_starts,
        expected_physical_starts,
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise RuntimeError("Self-test failed: canonical physical starts.")
    messages.append(
        "PASS: Banana target, gradient, starts, mass, five exact moments, "
        "and Gauss-Hermite oracle"
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

    q0 = np.asarray([0.8, -0.7], dtype=np.float64)
    standard_p0 = np.linspace(0.7, -0.3, dimension)
    p0 = prepared.mass_cholesky @ standard_p0 / math.sqrt(prepared.beta)
    test_epsilon = 0.001
    grad0 = _target_gradient(q0, prepared.target_parameters)
    q1, p1, grad1, _u1, calls1, finite1 = _leapfrog_proposal(
        q0,
        p0,
        grad0,
        test_epsilon,
        17,
        prepared.target_parameters,
        prepared.inverse_mass_matrix,
        prepared.mass_is_identity,
    )
    q2, p2, _grad2, _u2, calls2, finite2 = _leapfrog_proposal(
        q1,
        -p1,
        grad1,
        test_epsilon,
        17,
        prepared.target_parameters,
        prepared.inverse_mass_matrix,
        prepared.mass_is_identity,
    )
    if (
        not finite1
        or not finite2
        or calls1 != 17
        or calls2 != 17
        or not np.allclose(q2, q0, rtol=0.0, atol=2.0e-11)
        or not np.allclose(p2, -p0, rtol=0.0, atol=2.0e-11)
    ):
        raise RuntimeError("Self-test failed: leapfrog reversibility or gradient count.")

    general_mass = np.asarray([[2.0, 0.3], [0.3, 0.8]], dtype=np.float64)
    general_inverse = np.linalg.inv(general_mass)
    general_q0 = np.asarray([0.2, -0.1])
    general_p0 = np.asarray([0.4, -0.25])
    general_target_parameters = np.asarray([2.25, 0.09, 0.15])
    general_grad0 = _target_gradient(general_q0, general_target_parameters)
    gq1, gp1, gg1, _gu1, gcalls1, gfinite1 = _leapfrog_proposal(
        general_q0,
        general_p0,
        general_grad0,
        0.01,
        23,
        general_target_parameters,
        general_inverse,
        False,
    )
    gq2, gp2, _gg2, _gu2, gcalls2, gfinite2 = _leapfrog_proposal(
        gq1,
        -gp1,
        gg1,
        0.01,
        23,
        general_target_parameters,
        general_inverse,
        False,
    )
    expected_kinetic = 0.5 * float(general_p0 @ general_inverse @ general_p0)
    if (
        not gfinite1
        or not gfinite2
        or gcalls1 != 23
        or gcalls2 != 23
        or not np.allclose(gq2, general_q0, rtol=0.0, atol=2.0e-12)
        or not np.allclose(gp2, -general_p0, rtol=0.0, atol=2.0e-12)
        or not math.isclose(
            _kinetic_energy(general_p0, general_inverse, False),
            expected_kinetic,
            rel_tol=1.0e-14,
            abs_tol=1.0e-14,
        )
    ):
        raise RuntimeError("Self-test failed: general base-mass mechanics.")

    volume_state = np.concatenate((general_q0, general_p0))

    def reference_proposal_map(state: np.ndarray) -> np.ndarray:
        q_state = state[:2]
        p_state = state[2:]
        result = _reference_standardized_leapfrog(
            q_state,
            p_state,
            _target_gradient(q_state, general_target_parameters),
            0.003,
            5,
            general_target_parameters,
            general_inverse,
        )
        if not result[5]:
            raise RuntimeError("Self-test failed: nonfinite reference proposal map.")
        return np.concatenate((result[0], result[1]))

    difference_step = 1.0e-6
    jacobian = np.column_stack(
        [
            (
                reference_proposal_map(
                    volume_state + difference_step * np.eye(4)[coordinate]
                )
                - reference_proposal_map(
                    volume_state - difference_step * np.eye(4)[coordinate]
                )
            )
            / (2.0 * difference_step)
            for coordinate in range(4)
        ]
    )
    jacobian_determinant = float(np.linalg.det(jacobian))
    if not math.isclose(
        jacobian_determinant, 1.0, rel_tol=0.0, abs_tol=5.0e-7
    ):
        raise RuntimeError("Self-test failed: leapfrog volume preservation.")
    messages.append(
        "PASS: identity-base and general-base-mass leapfrog reversibility, "
        "volume preservation, and exact gradient counts"
    )

    momenta, mass_normals, uniforms = _make_random_streams(
        998877, 5, dimension
    )
    repeated_momenta, repeated_mass_normals, repeated_uniforms = (
        _make_random_streams(998877, 5, dimension)
    )
    if not np.array_equal(momenta, repeated_momenta) or not np.array_equal(
        uniforms, repeated_uniforms
    ) or not np.array_equal(
        mass_normals, repeated_mass_normals
    ):
        raise RuntimeError("Self-test failed: deterministic random substreams.")
    tiny_digest = hashlib.sha256()
    for array in (momenta, mass_normals, uniforms):
        tiny_digest.update(
            np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")
        )
    if tiny_digest.hexdigest() != (
        "9ca4f4a3a225c694403ef747f0ad9b52adec1e81de5e8d3c07396ad68be5ca83"
    ):
        raise RuntimeError("Self-test failed: tiny random-stream digest.")
    output = _simulate_s_qhmc_chain(
        momenta,
        mass_normals,
        uniforms,
        prepared.physical_starts[0],
        prepared.beta,
        0.0,
        0.25,
        test_epsilon,
        4,
        2,
        3,
        prepared.target_parameters,
        prepared.mass_cholesky,
        prepared.inverse_mass_matrix,
        prepared.mass_is_identity,
    )
    if (
        output[0].shape != (3, dimension)
        or output[1].shape != (3,)
        or output[5] != 9
        or output[6] != 12
    ):
        raise RuntimeError("Self-test failed: phase-specific gradient accounting.")
    if output[3] != 0 or output[4] != 0:
        raise RuntimeError("Self-test failed: unexpected nonfinite trajectory.")

    invalid_iterations = 5
    invalid_output = _simulate_s_qhmc_chain(
        np.zeros((invalid_iterations, dimension), dtype=np.float64),
        np.full(invalid_iterations, -1.0e6, dtype=np.float64),
        np.full(invalid_iterations, 0.5, dtype=np.float64),
        prepared.physical_starts[0],
        prepared.beta,
        0.0,
        1.0,
        0.04,
        4,
        2,
        3,
        prepared.target_parameters,
        prepared.mass_cholesky,
        prepared.inverse_mass_matrix,
        prepared.mass_is_identity,
    )
    if (
        invalid_output[2] != 0
        or invalid_output[3] != 2
        or invalid_output[4] != 3
        or invalid_output[5] != 1
        or invalid_output[6] != 0
        or not np.array_equal(
            invalid_output[0],
            np.repeat(prepared.physical_starts[0][np.newaxis, :], 3, axis=0),
        )
    ):
        raise RuntimeError("Self-test failed: invalid random-mass propagation.")
    messages.append(
        "PASS: deterministic random substreams, warmup/retained gradient "
        "accounting, finite-chain smoke test, and invalid-mass propagation"
    )

    expected_observable_count = dimension + dimension * (dimension + 1) // 2
    if len(prepared.observable_names) != expected_observable_count or len(
        set(prepared.observable_names)
    ) != expected_observable_count:
        raise RuntimeError("Self-test failed: required observable set.")
    synthetic_samples = rng.standard_normal(
        (prepared.n_chains, 120, dimension)
    ) * np.sqrt(prepared.target_variances)[np.newaxis, np.newaxis, :]
    restored_block = _analyze_block(
        samples=synthetic_samples,
        epsilon=0.01,
        L=15,
        block_index=0,
        seeds=prepared.seed_blocks[0],
        accepted_retained=prepared.n_chains * 100,
        retained_gradient_calls=prepared.n_chains * 120 * 15,
        warmup_nonfinite=1,
        retained_nonfinite=0,
        observable_names=prepared.observable_names,
        exact_observable_means=prepared.exact_observable_means,
        exact_observable_sds=prepared.exact_observable_sds,
    )
    if restored_block.status != "ok" or not math.isfinite(restored_block.block_efficiency):
        raise RuntimeError("Self-test failed: safely rejected proposal must retain a finite-chain score.")
    if restored_block.warmup_nonfinite_trajectories != 1:
        raise RuntimeError("Self-test failed: rejection event was not counted.")
    synthetic_samples[0, 0, 0] = math.nan
    invalid_block = _analyze_block(
        samples=synthetic_samples,
        epsilon=0.01,
        L=15,
        block_index=0,
        seeds=prepared.seed_blocks[0],
        accepted_retained=prepared.n_chains * 100,
        retained_gradient_calls=prepared.n_chains * 120 * 15,
        warmup_nonfinite=1,
        retained_nonfinite=0,
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
        warmup_nonfinite_trajectories=0,
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

def _runtime_environment_summary() -> str:
    numba_version = numba.__version__ if NUMBA_AVAILABLE else "unavailable"
    llvmlite_version = llvmlite.__version__ if NUMBA_AVAILABLE else "unavailable"
    return (
        f"platform={platform.platform()}; Python={platform.python_version()}; "
        f"NumPy={np.__version__}; Numba={numba_version}; "
        f"llvmlite={llvmlite_version}"
    )

def _first_tuning_block_stream_digest(prepared: PreparedSQHMCInputs) -> str:
    common = prepared.common
    total = common.n_warmup + common.n_retained
    digest = hashlib.sha256()
    for seed in common.seed_blocks[0]:
        for array in _make_random_streams(seed, total, common.dimension):
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()

def _validate_portable_native_runtime(prepared: PreparedSQHMCInputs) -> str:
    """Verify current cache-free dispatchers without a stochastic path canary."""

    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "Canonical Grex production requires Numba. Select the configured "
            "Python 3.12 sampling kernel before starting the tuning grid."
        )
    common = prepared.common
    _validate_compiled_reference_microcheck()
    if not getattr(_simulate_s_qhmc_chain, "signatures", ()):
        momenta, mass_normals, uniforms = _make_random_streams(
            998877, 2, common.dimension
        )
        _simulate_s_qhmc_chain(
            momenta,
            mass_normals,
            uniforms,
            common.physical_starts[0],
            common.beta,
            prepared.mu_m_values[0],
            prepared.sigma_m_values[0],
            0.001,
            1,
            1,
            1,
            common.target_parameters,
            common.mass_cholesky,
            common.inverse_mass_matrix,
            common.mass_is_identity,
        )
    expected_dispatchers = (
        (
            "_potential_energy",
            _potential_energy,
            _potential_energy_sqhmc_banana_grex_v1,
        ),
        (
            "_target_gradient",
            _target_gradient,
            _target_gradient_sqhmc_banana_grex_v1,
        ),
        (
            "_kinetic_energy",
            _kinetic_energy,
            _base_kinetic_energy_sqhmc_banana_grex_v1,
        ),
        (
            "_vector_is_finite",
            _vector_is_finite,
            _vector_is_finite_sqhmc_banana_grex_v1,
        ),
        (
            "_leapfrog_proposal",
            _leapfrog_proposal,
            _standardized_leapfrog_proposal_sqhmc_banana_grex_v1,
        ),
        (
            "_simulate_s_qhmc_chain",
            _simulate_s_qhmc_chain,
            _simulate_s_qhmc_chain_sqhmc_banana_grex_v1,
        ),
    )
    problems: list[str] = []
    for public_name, active, expected in expected_dispatchers:
        if active is not expected:
            problems.append(f"{public_name} does not point to the current epoch")
            continue
        if getattr(active, "py_func", None) is None:
            problems.append(f"{public_name} is not a Numba dispatcher")
        cache_object = getattr(active, "_cache", None)
        if cache_object is not None and cache_object.__class__.__name__ != "NullCache":
            problems.append(f"{public_name} has persistent native caching enabled")
    if not getattr(_leapfrog_proposal, "signatures", ()):
        problems.append("the S-QHMC proposal was not compiled by self-tests")
    if not getattr(_simulate_s_qhmc_chain, "signatures", ()):
        problems.append("the complete S-QHMC chain was not compiled by self-tests")
    if problems:
        raise RuntimeError(
            "Portable native-runtime preflight failed: " + "; ".join(problems)
        )
    stream_digest = _first_tuning_block_stream_digest(prepared)
    if (
        common.run_mode == "CANONICAL"
        and stream_digest != CANONICAL_FIRST_TUNING_BLOCK_STREAM_DIGEST
    ):
        raise RuntimeError(
            "Canonical random-stream parity failed: expected "
            + CANONICAL_FIRST_TUNING_BLOCK_STREAM_DIGEST
            + ", received "
            + stream_digest
        )
    return (
        "PASS: current epoch-specific cache-free Numba dispatchers, aliases, "
        "complete chain, and compiled/reference micro-oracle are active; "
        "tuning-block-1 stream SHA-256="
        + stream_digest
    )

def run_s_qhmc_tuning(
    inputs: SQHMCRunInputs,
    *,
    preflight_only: bool = False,
) -> dict[str, object]:
    """Run one completely specified S-QHMC tuning campaign."""
    if not preflight_only:
        raise RuntimeError("Use launcher/checkpoint_runner.py for the expanded full-grid production run.")


    prepared = _prepare_s_qhmc_inputs(inputs)
    common = prepared.common
    _validate_s_qhmc_run_mode(prepared)
    _validate_common_parity_seals(common)

    self_test_messages = (
        run_self_tests(prepared)
        if common.run_self_tests or preflight_only
        else ("SKIPPED: run_self_tests=False (CUSTOM mode only).",)
    )
    native_message = _validate_portable_native_runtime(prepared)
    stream_digest = _first_tuning_block_stream_digest(prepared)
    if preflight_only:
        message = (
            "PASS: portable cache-free scalar S-QHMC native preflight "
            f"({SQHMC_RUNTIME_PREFLIGHT_VERSION}); "
            + _runtime_environment_summary()
            + "; compiled/reference micro-oracle=PASS; "
            + "target/gradient, exact moments, random-mass equivalence, "
            + "reversibility, RNG, diagnostics, and finite smoke chain=PASS; "
            + "tuning-block-1 stream SHA-256="
            + stream_digest
            + "\nGREX PREFLIGHT COMPLETE: no tuning trajectory or Cartesian-grid "
            + "configuration was simulated, and no report was written. Use "
            + "launcher/checkpoint_runner.py for expanded production tuning."
        )
        print(message, flush=True)
        return {
            "preflight_only": True,
            "message": message,
            "prepared_inputs": prepared,
            "self_tests": self_test_messages,
            "native_runtime": native_message,
            "stream_digest": stream_digest,
        }

    destination = _resolve_output_destination(common)
    total_iterations = common.n_warmup + common.n_retained
    stream_caches: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for block in common.seed_blocks:
        stream_caches.append(
            [
                _make_random_streams(seed, total_iterations, common.dimension)
                for seed in block
            ]
        )

    auxiliary_momentum_moments = tuple(
        _retained_standardized_momentum_second_moment(cache, common.n_warmup)
        for cache in stream_caches
    )

    # Compile the Numba kernels before worker threads are opened.
    compile_momenta, compile_mass_normals, compile_uniforms = (
        _make_random_streams(13579, 2, common.dimension)
    )
    _simulate_s_qhmc_chain(
        compile_momenta,
        compile_mass_normals,
        compile_uniforms,
        common.physical_starts[0],
        common.beta,
        prepared.mu_m_values[0],
        prepared.sigma_m_values[0],
        common.epsilon_values[0],
        1,
        1,
        1,
        common.target_parameters,
        common.mass_cholesky,
        common.inverse_mass_matrix,
        common.mass_is_identity,
    )

    configuration_results: list[ConfigurationResult] = []
    total_configurations = (
        len(prepared.mu_m_values)
        * len(prepared.sigma_m_values)
        * len(common.epsilon_values)
        * len(common.L_values)
    )
    configuration_number = 0

    for mu_m in prepared.mu_m_values:
        for sigma_m in prepared.sigma_m_values:
            for epsilon in common.epsilon_values:
                for L in common.L_values:
                    configuration_number += 1
                    if common.print_progress:
                        print(
                            f"[{configuration_number:>4}/{total_configurations}] "
                            f"S-QHMC mu_m={mu_m:.6g}, sigma_m={sigma_m:.6g}, "
                            f"epsilon={epsilon:.6g}, L={L}: running "
                            f"{len(common.seed_blocks)} tuning blocks...",
                            flush=True,
                        )
                    block_results: list[SQHMCBlockResult] = []
                    for block_index, seeds in enumerate(common.seed_blocks):
                        block_results.append(
                            _run_configuration_block(
                                mu_m=mu_m,
                                sigma_m=sigma_m,
                                epsilon=epsilon,
                                L=L,
                                block_index=block_index,
                                seeds=seeds,
                                stream_cache=stream_caches[block_index],
                                prepared=prepared,
                            )
                        )
                    aggregate = _aggregate_configuration(
                        mu_m, sigma_m, epsilon, L, block_results
                    )
                    configuration_results.append(aggregate)
                    if common.print_progress:
                        print(
                            "    status="
                            + aggregate.status
                            + ", GM efficiency="
                            + _format_number(
                                aggregate.geometric_mean_efficiency, ".8e"
                            ),
                            flush=True,
                        )

    representative = _select_representative(configuration_results)
    report = _build_report(
        configuration_results=configuration_results,
        representative=representative,
        auxiliary_momentum_moments=auxiliary_momentum_moments,
        prepared=prepared,
        self_test_messages=self_test_messages + (native_message,),
    )
    destination.write_text(report + "\n", encoding="utf-8")

    if common.print_report:
        print("\n" + report)
        print(f"\nThe identical report was saved to:\n{destination}")

    return {
        "configuration_results": tuple(configuration_results),
        "representative": representative,
        "report": report,
        "output_path": destination,
        "prepared_inputs": prepared,
        "common_input_fingerprint": common.common_input_fingerprint,
        "method_input_fingerprint": prepared.method_input_fingerprint,
        "input_preparation_fingerprint": _input_preparation_fingerprint(),
        "shared_protocol_fingerprint": _shared_protocol_fingerprint(common),
        "diagnostics_fingerprint": _diagnostics_fingerprint(),
        "common_pipeline_fingerprint": _common_pipeline_fingerprint(),
        "s_qhmc_implementation_fingerprint": _sqhmc_implementation_fingerprint(),
        "source_parity_verification_mode": _source_verification_mode(),
    }

GREX_PREFLIGHT_ONLY = True
# HELDOUT_ENGINE_END

HELDOUT_METHOD = 'S_QHMC'
HELDOUT_PROTOCOL = json.loads('{"IAT":"Original multi-chain Geyer initial-positive then initial-monotone sequence; biased autocovariance denominator N, no floor or hand-set lag cutoff.","Rhat":"Maximum of rank-normalized split and folded rank-normalized split R-hat; descriptive, no threshold.","SQHMC_refresh":"At each trajectory log10(m)~N(0.5,0.0025^2); M=m*M0 and p~N(0,M/beta). This mass stays fixed for that leapfrog trajectory and Metropolis decision.","THC_refresh":"h(s)=1+0.025*cos(s/0.5), alpha=0.025, phi=0, a=0.5, Q=4, rho(s)=N(0,1). s persists, starting at (-1,1,1,-1). Only p|s~N(0,h(s)^2*M0/beta) and p_s~N(0,Q/beta) refresh each trajectory. Rejection restores q and s.","comparison_id":"BANANA2_SQHMC_PERIODIC_THC_V2_HELDOUT_20261012_v1","date_frozen":"2026-10-12","exact_means":[0.0,0.0,4.0,0.0,2.04],"failure_rule":"Unchanged expanded tuning rule for this pair: rejected nonfinite proposals are counted, but finite retained states (including rejection repeats) are scored. A nonfinite retained state, undefined required IAT, or invalid actual gradient denominator makes the block and full method score undefined. No omitted or replacement blocks. This differs from the earlier HMC/bounded/v4 zero-event rule.","gradient_cost":"Actual newly executed physical-target gradients during retained transitions only; warmup and cached reuse excluded. Finite trajectories cost L (S-QHMC 30, THC-v2 60); thermostat derivatives are excluded. Not wall-clock cost.","input_templates":{"S_QHMC":{"L_values":[30],"allow_overwrite":false,"banana_b":0.25,"beta":1.0,"epsilon_values":[0.12],"mass_matrix":[[0.25,0.0],[0.0,0.49019607843137253]],"mu_m_values":[0.5],"n_retained":10000,"n_warmup":1000,"n_workers":4,"observable_policy":"ALL_LINEAR_AND_UNIQUE_QUADRATIC","output_directory":null,"output_filename":"S_QHMC_results.txt","print_progress":false,"print_report":false,"run_label":"CUSTOM: FROZEN HELD-OUT","run_mode":"CUSTOM","run_self_tests":true,"seed_blocks":[[1657368863,3792359500,3417096587,1121828610],[3588632714,1198709160,160716073,3848830813],[2237602816,2341327334,140246127,368899257],[423389250,329042214,252825639,4263613775],[632968939,3143972171,1557189394,1289041858],[2720520449,161697784,2659971218,690560112],[1461231770,3322832141,102545162,54283471],[2321049068,1450827581,4212370027,147843710],[1226675181,2834445672,2199338227,288850969],[3797197053,1239296244,1373558472,2298438043]],"sigma_m_values":[0.0025],"sigma_x":2.0,"sigma_y":0.2,"standardized_starts":[[-2.0,-2.0],[-2.0,2.0],[2.0,-2.0],[2.0,2.0]]},"THC_v2":{"L_values":[60],"Q_values":[4.0],"a_values":[0.5],"allow_overwrite":false,"alpha_values":[0.025],"automatic_boundary_expansion":false,"banana_b":0.25,"beta":1.0,"boundary_Q_lower":0.003125,"boundary_Q_upper":16.0,"boundary_a_lower":0.0625,"boundary_a_upper":8.0,"boundary_alpha_lower":0.0125,"boundary_alpha_upper":0.975,"boundary_keep_fixed_configurations":1,"coupling_family":"PHASE_PERIODIC_COS_SHIFT","epsilon_values":[0.05],"mass_matrix":[[0.25,0.0],[0.0,0.49019607843137253]],"n_retained":10000,"n_warmup":1000,"n_workers":4,"observable_policy":"ALL_LINEAR_AND_UNIQUE_QUADRATIC","output_directory":null,"output_filename":"THC_v2_results.txt","phi_values":[0.0],"print_progress":false,"print_report":false,"run_label":"CUSTOM: FROZEN HELD-OUT","run_mode":"CUSTOM","run_self_tests":true,"seed_blocks":[[1657368863,3792359500,3417096587,1121828610],[3588632714,1198709160,160716073,3848830813],[2237602816,2341327334,140246127,368899257],[423389250,329042214,252825639,4263613775],[632968939,3143972171,1557189394,1289041858],[2720520449,161697784,2659971218,690560112],[1461231770,3322832141,102545162,54283471],[2321049068,1450827581,4212370027,147843710],[1226675181,2834445672,2199338227,288850969],[3797197053,1239296244,1373558472,2298438043]],"sigma_x":2.0,"sigma_y":0.2,"stage1_anchor_pairs":[[0.05,60]],"stage1_block_indices":[1],"stage1_keep_method_tuples":1,"stage1_n_retained":4,"stage1_n_warmup":1,"stage2_block_indices":[2],"stage2_keep_fixed_configurations":1,"stage2_n_retained":4,"stage2_n_warmup":1,"standardized_starts":[[-2.0,-2.0],[-2.0,2.0],[2.0,-2.0],[2.0,2.0]],"thermostat_sigma":1.0,"thermostat_starts":[-1.0,1.0,1.0,-1.0]}},"legacy_stage_fields":"Periodic stage/boundary fields only satisfy original input validation. The held-out driver does not call search or boundary expansion.","methods":{"S_QHMC":{"L":30,"epsilon":0.12,"mu_m":0.5,"sigma_m":0.0025},"THC_v2":{"L":60,"Q":4.0,"a":0.5,"coupling_code":0.025,"epsilon":0.05}},"n_blocks":10,"n_chains":4,"n_retained":10000,"n_warmup":1000,"no_retuning":true,"paired_comparison":"THC-v2/S-QHMC score ratio, paired efficiency wins and approximate 95% Student-t interval from the ten paired log-efficiency ratios, only if all required scores are defined.","physical_observables":["x","y","x^2","x*y","y^2"],"physical_starts":[[-4.0,2.6],[-4.0,3.4],[4.0,2.6],[4.0,3.4]],"provenance":{"S_QHMC":{"engine_sha256":"d2918d206c1c98d0489693af325239ff52baeecdc74c865dd26912fd345edf61","source_path":"grex_banana_periodic_sqhmc_expanded_20260925\\\\code\\\\banana_sqhmc_EXPANDED_20260925.py","source_sha256":"faf3832419a10ec16ccea5afd0258222dfa617fce45401ffede1115e7110b2af","tested_configurations":16731,"tuning_report":"results\\\\GREX_tuning_2026-10-07\\\\Banana\\\\S_QHMC\\\\results\\\\S_QHMC_Banana_expanded_tuning_20260925.txt","tuning_report_sha256":"383cb9416cd1c8149788dc4c949d56ebeb8c2722e45564d30ceaacc6a4f8afbb","tuning_score":0.00532775349272218},"THC_v2":{"engine_sha256":"19a50c6511623e9f26239e208871df09337ba79896cc85433a4a9284214b0814","source_path":"grex_banana_periodic_sqhmc_expanded_20260925\\\\code\\\\banana_periodic_EXPANDED_20260925.py","source_sha256":"fcaf9b268fdad16818595c3460de4d09c407d37a87e17a183489a996bd752ed9","tested_configurations":41533,"tuning_report":"results\\\\GREX_tuning_2026-10-07\\\\Banana\\\\Periodic_THC_v2\\\\results\\\\FINAL_TUNING_REPORT.txt","tuning_report_sha256":"225e1b65614bc9b4a952842532a58019bca89c936d339a28bca48ba049e6a9e0","tuning_score":0.0055055523899574625}},"random_stream_pairing":"Original SeedSequence/PCG64 child 0 physical normals, child 1 auxiliary normals (S-QHMC mass or THC p_s), child 2 MH uniforms. All three raw stream hashes must match in each paired block.","score":"E_b=1/(actual retained gradients per draw * worst physical-observable IAT); S=geometric mean of ALL ten block efficiencies.","scoring_policy_evidence":"grex_banana_periodic_sqhmc_expanded_20260925/README.txt, SCORING AND NUMERICAL EVENTS; both embedded _analyze_block implementations and native self-tests. Selected tuning winners include 199 S-QHMC and 5 THC-v2 nonfinite proposal events, counted but not dropped.","secondary_outcomes":"Maximum and descriptive geometric mean of blockwise worst IAT; max R-hat, acceptance, exact/estimated moments, nonfinite counts. No final-data reselection.","seed_audit":{"derivation":"First four SHA256 bytes (big endian) of DOMAIN|PHASE|block=BB|chain=CC|counter=K; counters change only for recorded collisions, never scores.","domain":"paper-on-thc|BANANA2|SQHMC-periodic-THC-v2|held-out|20261012|v1","overlaps_after_audit":0,"path_size_inventory_sha256":"f3f59287321980ce23adc109c71f239c3ec77c5babeaa4c0857803d8915ad6f9","rejected_candidate_values":[],"scope":"Recorded integer occurrences in local project sources/reports, including downloaded tuning and previous held-out files; unknown remote-only seeds cannot be audited.","text_bytes_scanned":3053982661,"text_files_scanned":2668},"seed_blocks":[[1657368863,3792359500,3417096587,1121828610],[3588632714,1198709160,160716073,3848830813],[2237602816,2341327334,140246127,368899257],[423389250,329042214,252825639,4263613775],[632968939,3143972171,1557189394,1289041858],[2720520449,161697784,2659971218,690560112],[1461231770,3322832141,102545162,54283471],[2321049068,1450827581,4212370027,147843710],[1226675181,2834445672,2199338227,288850969],[3797197053,1239296244,1373558472,2298438043]],"selection":"Best tested geometric-mean gradient-normalized-efficiency setting from each completed expanded tuning search. No secondary-IAT reselection.","target":"2D centered Banana","target_definition":"U=x^2/(2*sigma_x^2)+(y-b*(x^2-sigma_x^2))^2/(2*sigma_y^2); beta=1, sigma_x=2, sigma_y=0.2, b=0.25; base mass diag(0.25,1/2.04)."}')
HELDOUT_PROTOCOL_SHA256 = '4e72052d86f08b0a3dca10b858488f39a744c8f53cb7c96fbca4a9b611de3717'
SOFTWARE_TEST_SEEDS = [[3720769319, 1749717704, 3146778781, 696256032], [1938650841, 2172330983, 1833394801, 2766403035], [4084974976, 1212302958, 3012620754, 3350469992], [1391415721, 2964201174, 160802772, 1199972139], [2020871140, 1237681223, 2314057531, 1335789887], [3716553219, 3500193913, 350985146, 1725311191], [1523879963, 3480548969, 2449510745, 4091924], [1251443900, 2682813049, 896097826, 3585124020], [2745076950, 2211089614, 287787612, 254795466], [4268253304, 3235119161, 849059260, 1781929638]]
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
        raise RuntimeError('Frozen protocol changed. Do not retune this experiment.')
    if not NUMBA_AVAILABLE:
        raise RuntimeError('Numba is required for this production experiment.')
    text = Path(__file__).read_text(encoding='utf-8')
    engine = text.split('# HELDOUT_ENGINE_' + 'BEGIN\n', 1)[1].split('# HELDOUT_ENGINE_' + 'END\n', 1)[0]
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
    if HELDOUT_METHOD == 'S_QHMC':
        prepared = _prepare_s_qhmc_inputs(SQHMCRunInputs(**arguments))
        _validate_s_qhmc_run_mode(prepared)
    else:
        prepared = _prepare_thc_inputs(THCRunInputs(**arguments))
        _validate_thc_run_mode(prepared)
    common = prepared.common
    _validate_common_parity_seals(common)
    if common.n_chains != 4 or len(common.observable_names) != 5:
        raise RuntimeError('Unexpected number of chains or observables.')
    np.testing.assert_allclose(common.physical_starts, p['physical_starts'], rtol=0, atol=1e-14)
    return prepared


def _heldout_common(prepared):
    return prepared.common


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
    common = prepared.common
    seeds = common.seed_blocks[index]
    streams = [_make_random_streams(s, common.n_warmup + common.n_retained, common.dimension) for s in seeds]
    physical_hash, all_hash = hashlib.sha256(), hashlib.sha256()
    for stream in streams:
        for i, array in enumerate(stream):
            data = np.ascontiguousarray(array, dtype='<f8').tobytes(order='C')
            all_hash.update(data)
            if i in (0, 2):
                physical_hash.update(data)
    result = _run_configuration_block(**HELDOUT_PROTOCOL['methods'][HELDOUT_METHOD],
        block_index=index, seeds=seeds, stream_cache=streams, prepared=prepared)
    auxiliary = ({'standard_mass_normal': asdict(result.mass)} if HELDOUT_METHOD == 'S_QHMC'
                 else {'post_MH_s': asdict(result.thermostat)})
    record = _heldout_clean(dict(physical=asdict(result.physical), auxiliary=auxiliary,
        physical_and_acceptance_stream_sha256=physical_hash.hexdigest(),
        all_three_streams_sha256=all_hash.hexdigest()))
    record['record_sha256'] = _heldout_hash(record)
    return record


def _heldout_preflight(workers=4):
    _heldout_guard()
    prepared = _heldout_prepare(workers, testing=True)
    messages = list(run_self_tests(prepared))
    messages.append(_validate_portable_native_runtime(prepared))
    sample = _heldout_one_block(prepared, 0)
    block = _heldout_decode_block(sample)
    common = prepared.common
    if len(block.observable_diagnostics) != 5:
        raise RuntimeError('Software-check observable count mismatch.')
    max_calls = common.n_chains * common.n_retained * block.L
    if not 0 < block.retained_gradient_calls <= max_calls:
        raise RuntimeError('Retained gradient accounting failed.')
    if not block.retained_nonfinite_trajectories and block.retained_gradient_calls != max_calls:
        raise RuntimeError('Finite trajectories must cost exactly L gradients per retained draw.')
    # Preserve this pair's tuning rule: a rejected nonfinite proposal is an
    # event count, not an invalid retained state or an automatic score veto.
    if _heldout_clean(asdict(_heldout_decode_block(json.loads(json.dumps(sample))))) != sample['physical']:
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
            'Observables: x, y, x^2, x*y, y^2.',
            'IAT: unchanged multi-chain Geyer IPS/IMS, biased denominator N, no floor.',
            'R-hat: max(rank-normalized split, folded rank-normalized split); descriptive.',
            'Gradient cost: actual retained target-gradient calls; warmup excluded; cached gradients not counted twice.',
            'Scoring rule: ' + p['failure_rule'],
            'No blocks omitted. Numerical rejection repeats the previous state.',
            'No adaptation, tuning or selection has been performed using these final blocks.']
    text += [p['THC_refresh'] if HELDOUT_METHOD == 'THC_v2' else p['SQHMC_refresh'],
             p['random_stream_pairing'], 'Auxiliary moments are descriptive, never score observables.']
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
