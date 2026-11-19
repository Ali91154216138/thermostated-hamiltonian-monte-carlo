"""Tune periodic THC-v2 on the Banana target."""

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

NUMBA_RUNTIME_EPOCH = "GREX_JUPYTER_PHASE_PERIODIC_V2_20260830A"
NUMBA_DIRECT_RUN = __name__ == "__main__"
NUMBA_CACHE_MODE = (
    "persistent cache disabled in all modes; native epoch "
    + NUMBA_RUNTIME_EPOCH
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



SHARED_PROTOCOL_VERSION = "BANANA2-phase-periodic-v2-successive-halving-v1.0.0"
COMMON_INPUT_SCHEMA_VERSION = "banana-fair-playground-common-inputs-v1"
INPUT_PREPARATION_VERSION = "banana-validated-derived-inputs-v1.0.0"
DIAGNOSTICS_VERSION = "multichain-geyer-ims-rank-rhat-v1.0.0"
COMMON_PIPELINE_VERSION = "banana-common-rng-observables-score-v1.0.0"
THC_IMPLEMENTATION_VERSION = (
    "banana-thc-phase-periodic-general-mass-psk-grex-portable-v2.0.0"
)
THC_METHOD_INPUT_SCHEMA_VERSION = "banana-phase-periodic-thc-v2-inputs-v1"
RANDOM_STREAM_VERSION = "seedsequence-pcg64-three-substreams-v1.0.0"
THC_RUNTIME_PREFLIGHT_VERSION = (
    "banana2-phase-periodic-thc-v2-native-preflight-20260830a-v1"
)
SUPPORTED_OBSERVABLE_POLICY = "ALL_LINEAR_AND_UNIQUE_QUADRATIC"
SUPPORTED_COUPLING_FAMILY = "PHASE_PERIODIC_COS_SHIFT"
CANONICAL_OUTPUT_FILENAME = "THC_phase_periodic_v2_banana_tuning_results_GREX_PORTABLE_V1.txt"

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
class THCRunInputs(RunInputs):
    """One bottom-block object containing common plus THC-only inputs."""

    coupling_family: str
    alpha_values: Sequence[float]
    phi_values: Sequence[float]
    a_values: Sequence[float]
    Q_values: Sequence[float]
    thermostat_sigma: float
    thermostat_starts: Sequence[float]
    stage1_anchor_pairs: Sequence[Sequence[float]]
    stage1_block_indices: Sequence[int]
    stage1_n_warmup: int
    stage1_n_retained: int
    stage1_keep_method_tuples: int
    stage2_block_indices: Sequence[int]
    stage2_n_warmup: int
    stage2_n_retained: int
    stage2_keep_fixed_configurations: int
    boundary_keep_fixed_configurations: int
    automatic_boundary_expansion: bool
    boundary_alpha_lower: float
    boundary_alpha_upper: float
    boundary_a_lower: float
    boundary_a_upper: float
    boundary_Q_lower: float
    boundary_Q_upper: float


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
class PreparedTHCInputs:
    """Validated thermostat inputs wrapped around the unchanged common inputs."""

    common: PreparedInputs
    coupling_family: str
    phase_codes: tuple[float, ...]
    alpha_values: tuple[float, ...]
    phi_values: tuple[float, ...]
    a_values: tuple[float, ...]
    Q_values: tuple[float, ...]
    thermostat_sigma: float
    thermostat_starts: np.ndarray
    stage1_anchor_pairs: tuple[tuple[float, int], ...]
    stage1_block_indices: tuple[int, ...]
    stage1_n_warmup: int
    stage1_n_retained: int
    stage1_keep_method_tuples: int
    stage2_block_indices: tuple[int, ...]
    stage2_n_warmup: int
    stage2_n_retained: int
    stage2_keep_fixed_configurations: int
    boundary_keep_fixed_configurations: int
    automatic_boundary_expansion: bool
    boundary_alpha_lower: float
    boundary_alpha_upper: float
    boundary_a_lower: float
    boundary_a_upper: float
    boundary_Q_lower: float
    boundary_Q_upper: float
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
class ThermostatDiagnostic:
    """Compact exact thermostat-coordinate moments; never a score or gate."""

    empirical_mean: float
    exact_mean: float
    mean_error: float
    empirical_second_moment: float
    exact_second_moment: float
    second_moment_error: float


@dataclass(frozen=True)
class THCBlockResult:
    """One common physical block result plus descriptive THC-only moments."""

    physical: BlockResult
    thermostat: ThermostatDiagnostic


@dataclass(frozen=True)
class THCConfigurationResult:
    """Five-block aggregate for one fixed joint THC configuration."""

    coupling_code: float
    a: float
    Q: float
    epsilon: float
    L: int
    status: str
    block_results: tuple[THCBlockResult, ...]
    geometric_mean_efficiency: float
    minimum_block_efficiency: float
    median_block_efficiency: float
    maximum_block_efficiency: float
    mean_acceptance: float
    maximum_rhat: float
    maximum_standardized_moment_error: float
    total_nonfinite_trajectories: int
    origin: str = "BASE_GRID"


@dataclass(frozen=True, order=True)
class MethodTuple:
    """One fixed phase-periodic coupling/thermostat tuple."""

    coupling_code: float
    a: float
    Q: float


@dataclass(frozen=True, order=True)
class FixedConfiguration:
    """One method tuple paired with one fixed common integrator."""

    coupling_code: float
    a: float
    Q: float
    epsilon: float
    L: int


# Target, THC splitting, and chain simulation.


@njit(cache=False, inline="always")
def _potential_energy_grex_phase_v2_20260830a(
    q: np.ndarray, target_parameters: np.ndarray
) -> float:
    """Banana potential; target_parameters=(sigma_x^2,sigma_y^2,b)."""

    x = q[0]
    y = q[1]
    sigma_x_squared = target_parameters[0]
    sigma_y_squared = target_parameters[1]
    bend = target_parameters[2]
    residual = y - bend * (x * x - sigma_x_squared)
    return (
        0.5 * x * x / sigma_x_squared
        + 0.5 * residual * residual / sigma_y_squared
    )


@njit(cache=False, inline="always")
def _target_gradient_grex_phase_v2_20260830a(
    q: np.ndarray, target_parameters: np.ndarray
) -> np.ndarray:
    """Exact gradient of the two-dimensional Banana potential."""

    x = q[0]
    y = q[1]
    sigma_x_squared = target_parameters[0]
    sigma_y_squared = target_parameters[1]
    bend = target_parameters[2]
    residual = y - bend * (x * x - sigma_x_squared)
    gradient = np.empty(2, dtype=np.float64)
    gradient[0] = (
        x / sigma_x_squared
        - 2.0 * bend * x * residual / sigma_y_squared
    )
    gradient[1] = residual / sigma_y_squared
    return gradient


@njit(cache=False, inline="always")
def _kinetic_energy_grex_phase_v2_20260830a(
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
def _vector_is_finite_grex_phase_v2_20260830a(values: np.ndarray) -> bool:
    for value in values:
        if not math.isfinite(value):
            return False
    return True


@njit(cache=False, inline="always")
def _log_h_and_derivative_grex_phase_v2_20260830a(
    s: float,
    coupling_code: float,
    a: float,
) -> tuple[float, float]:
    """Return log(h) and d log(h)/ds for the encoded phase family."""

    if (
        not math.isfinite(s)
        or not math.isfinite(coupling_code)
        or not math.isfinite(a)
        or a <= 0.0
    ):
        return math.nan, math.nan
    if coupling_code < 0.0:
        alpha = -coupling_code
        phi = 0.5 * math.pi
    else:
        alpha = coupling_code
        phi = 0.0
    if not math.isfinite(alpha) or alpha < 0.0 or alpha >= 1.0:
        return math.nan, math.nan
    angle = s / a - phi
    if not math.isfinite(angle):
        return math.nan, math.nan
    cosine = math.cos(angle)
    sine = math.sin(angle)
    h = 1.0 + alpha * cosine
    if not math.isfinite(h) or h <= 0.0:
        return math.nan, math.nan
    log_h = math.log(h)
    log_h_derivative = -(alpha / a) * sine / h
    if not math.isfinite(log_h) or not math.isfinite(log_h_derivative):
        return math.nan, math.nan
    return log_h, log_h_derivative


@njit(cache=False)
def _thc_hamiltonian_grex_phase_v2_20260830a(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    beta: float,
    coupling_code: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> float:
    """Evaluate the extended THC Hamiltonian up to one additive constant."""

    log_h, _log_h_derivative = _log_h_and_derivative_grex_phase_v2_20260830a(
        s, coupling_code, a
    )
    potential = _potential_energy_grex_phase_v2_20260830a(q, target_parameters)
    base_kinetic = _kinetic_energy_grex_phase_v2_20260830a(
        p, inverse_mass_matrix, mass_is_identity
    )
    physical_kinetic = math.exp(-2.0 * log_h) * base_kinetic
    metric_correction = (q.size / beta) * log_h
    thermostat_kinetic = 0.5 * p_s * p_s / Q
    thermostat_potential = (
        0.5 * s * s / (beta * thermostat_sigma * thermostat_sigma)
    )
    return (
        potential
        + physical_kinetic
        + metric_correction
        + thermostat_kinetic
        + thermostat_potential
    )


@njit(cache=False, inline="always")
def _thermostat_rotation_grex_phase_v2_20260830a(
    s: float,
    p_s: float,
    cosine: float,
    sine: float,
    Q: float,
    thermostat_frequency: float,
) -> tuple[float, float]:
    """Apply the exact harmonic S-flow using the old pair (s,p_s)."""

    old_s = s
    old_p_s = p_s
    new_s = (
        old_s * cosine
        + old_p_s * sine / (Q * thermostat_frequency)
    )
    new_p_s = (
        old_p_s * cosine
        - Q * thermostat_frequency * old_s * sine
    )
    return new_s, new_p_s


@njit(cache=False, inline="always")
def _thc_state_is_finite_grex_phase_v2_20260830a(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
) -> bool:
    return (
        math.isfinite(s)
        and math.isfinite(p_s)
        and _vector_is_finite_grex_phase_v2_20260830a(q)
        and _vector_is_finite_grex_phase_v2_20260830a(p)
    )


@njit(cache=False, nogil=True)
def _thc_proposal_grex_phase_v2_20260830a(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    current_gradient: np.ndarray,
    beta: float,
    coupling_code: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    epsilon: float,
    L: int,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    int,
    bool,
]:
    """Apply L cached-gradient P/S/K/S/P steps without the final momentum flip."""

    proposed_q = q.copy()
    proposed_p = p.copy()
    proposed_s = s
    proposed_p_s = p_s
    proposed_gradient = current_gradient.copy()
    gradient_calls = 0
    dimension = proposed_q.size
    thermostat_frequency = 1.0 / (
        thermostat_sigma * math.sqrt(beta * Q)
    )
    half_angle = 0.5 * epsilon * thermostat_frequency
    cosine = math.cos(half_angle)
    sine = math.sin(half_angle)

    # Initial P_(epsilon/2): physical potential plus metric determinant.
    _log_h, log_h_derivative = _log_h_and_derivative_grex_phase_v2_20260830a(
        proposed_s, coupling_code, a
    )
    for coordinate in range(dimension):
        proposed_p[coordinate] -= (
            0.5 * epsilon * proposed_gradient[coordinate]
        )
    proposed_p_s -= (
        0.5
        * epsilon
        * (dimension / beta)
        * log_h_derivative
    )
    if not _thc_state_is_finite_grex_phase_v2_20260830a(
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

    for step in range(L):
        # S_(epsilon/2): exact Gaussian thermostat-coordinate rotation.
        proposed_s, proposed_p_s = _thermostat_rotation_grex_phase_v2_20260830a(
            proposed_s,
            proposed_p_s,
            cosine,
            sine,
            Q,
            thermostat_frequency,
        )
        if not math.isfinite(proposed_s) or not math.isfinite(proposed_p_s):
            return (
                proposed_q,
                proposed_p,
                proposed_s,
                proposed_p_s,
                proposed_gradient,
                gradient_calls,
                False,
            )

        # K_epsilon: scaled physical drift and kinetic feedback into p_s.
        log_h, log_h_derivative = _log_h_and_derivative_grex_phase_v2_20260830a(
            proposed_s, coupling_code, a
        )
        inverse_h_squared = math.exp(-2.0 * log_h)
        momentum_quadratic = 0.0
        if mass_is_identity:
            for coordinate in range(dimension):
                momentum_quadratic += (
                    proposed_p[coordinate] * proposed_p[coordinate]
                )
                proposed_q[coordinate] += (
                    epsilon
                    * inverse_h_squared
                    * proposed_p[coordinate]
                )
        else:
            for row in range(dimension):
                inverse_mass_p = 0.0
                for column in range(dimension):
                    inverse_mass_p += (
                        inverse_mass_matrix[row, column]
                        * proposed_p[column]
                    )
                momentum_quadratic += (
                    proposed_p[row] * inverse_mass_p
                )
                proposed_q[row] += (
                    epsilon * inverse_h_squared * inverse_mass_p
                )
        proposed_p_s += (
            epsilon
            * inverse_h_squared
            * log_h_derivative
            * momentum_quadratic
        )
        if not _thc_state_is_finite_grex_phase_v2_20260830a(
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

        # S_(epsilon/2) again.
        proposed_s, proposed_p_s = _thermostat_rotation_grex_phase_v2_20260830a(
            proposed_s,
            proposed_p_s,
            cosine,
            sine,
            Q,
            thermostat_frequency,
        )
        if not math.isfinite(proposed_s) or not math.isfinite(proposed_p_s):
            return (
                proposed_q,
                proposed_p,
                proposed_s,
                proposed_p_s,
                proposed_gradient,
                gradient_calls,
                False,
            )

        # Adjacent P half-kicks merge.  Evaluate one new target gradient at
        # each endpoint and leave only the final boundary kick as a half-kick.
        proposed_gradient = _target_gradient_grex_phase_v2_20260830a(
            proposed_q, target_parameters
        )
        gradient_calls += 1
        if not _vector_is_finite_grex_phase_v2_20260830a(proposed_gradient):
            return (
                proposed_q,
                proposed_p,
                proposed_s,
                proposed_p_s,
                proposed_gradient,
                gradient_calls,
                False,
            )
        kick_duration = (
            0.5 * epsilon if step == L - 1 else epsilon
        )
        _log_h, log_h_derivative = _log_h_and_derivative_grex_phase_v2_20260830a(
            proposed_s, coupling_code, a
        )
        for coordinate in range(dimension):
            proposed_p[coordinate] -= (
                kick_duration * proposed_gradient[coordinate]
            )
        proposed_p_s -= (
            kick_duration
            * (dimension / beta)
            * log_h_derivative
        )
        if not _thc_state_is_finite_grex_phase_v2_20260830a(
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
def _simulate_thc_chain_grex_phase_v2_20260830a(
    standard_momenta: np.ndarray,
    thermostat_standard_momenta: np.ndarray,
    acceptance_uniforms: np.ndarray,
    q_initial: np.ndarray,
    s_initial: float,
    beta: float,
    coupling_code: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    """Run one refreshed-momentum, persistent-s THC chain."""

    total_iterations = n_warmup + n_retained
    dimension = q_initial.size
    samples = np.empty((n_retained, dimension), dtype=np.float64)
    thermostat_samples = np.empty(n_retained, dtype=np.float64)
    q = q_initial.copy()
    s = s_initial
    current_gradient = _target_gradient_grex_phase_v2_20260830a(q, target_parameters)

    # The same endpoint-gradient cache and phase charging used by HMC.
    warmup_gradient_calls = 1 if n_warmup > 0 else 0
    retained_gradient_calls = 0 if n_warmup > 0 else 1
    accepted_retained = 0
    warmup_nonfinite = 0
    retained_nonfinite = 0
    inverse_sqrt_beta = 1.0 / math.sqrt(beta)
    thermostat_momentum_scale = math.sqrt(Q / beta)

    for iteration in range(total_iterations):
        log_h, _log_h_derivative = _log_h_and_derivative_grex_phase_v2_20260830a(
            s, coupling_code, a
        )
        physical_momentum_scale = (
            math.exp(log_h) * inverse_sqrt_beta
        )
        p = np.empty(dimension, dtype=np.float64)
        if mass_is_identity:
            for coordinate in range(dimension):
                p[coordinate] = (
                    physical_momentum_scale
                    * standard_momenta[iteration, coordinate]
                )
        else:
            for row in range(dimension):
                transformed_momentum = 0.0
                for column in range(dimension):
                    transformed_momentum += (
                        mass_cholesky[row, column]
                        * standard_momenta[iteration, column]
                    )
                p[row] = (
                    physical_momentum_scale * transformed_momentum
                )
        p_s = (
            thermostat_momentum_scale
            * thermostat_standard_momenta[iteration]
        )
        initial_energy = _thc_hamiltonian_grex_phase_v2_20260830a(
            q,
            p,
            s,
            p_s,
            beta,
            coupling_code,
            a,
            Q,
            thermostat_sigma,
            target_parameters,
            inverse_mass_matrix,
            mass_is_identity,
        )

        (
            proposed_q,
            proposed_p,
            proposed_s,
            proposed_p_s,
            proposed_gradient,
            proposal_gradient_calls,
            proposal_is_finite,
        ) = _thc_proposal_grex_phase_v2_20260830a(
            q,
            p,
            s,
            p_s,
            current_gradient,
            beta,
            coupling_code,
            a,
            Q,
            thermostat_sigma,
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

        # R o Psi^L is the involutive deterministic proposal.  H is even in
        # both momenta, but the explicit flip is retained as part of the map.
        if proposal_is_finite:
            for coordinate in range(dimension):
                proposed_p[coordinate] = -proposed_p[coordinate]
            proposed_p_s = -proposed_p_s

        accepted = False
        if proposal_is_finite and math.isfinite(initial_energy):
            proposed_energy = _thc_hamiltonian_grex_phase_v2_20260830a(
                proposed_q,
                proposed_p,
                proposed_s,
                proposed_p_s,
                beta,
                coupling_code,
                a,
                Q,
                thermostat_sigma,
                target_parameters,
                inverse_mass_matrix,
                mass_is_identity,
            )
            delta_hamiltonian = proposed_energy - initial_energy
            if math.isfinite(delta_hamiltonian):
                uniform = acceptance_uniforms[iteration]
                log_uniform = (
                    -math.inf if uniform <= 0.0 else math.log(uniform)
                )
                accepted = (
                    log_uniform < -beta * delta_hamiltonian
                )
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
            s = proposed_s
            current_gradient = proposed_gradient

        if iteration >= n_warmup:
            retained_index = iteration - n_warmup
            for coordinate in range(dimension):
                samples[retained_index, coordinate] = q[coordinate]
            thermostat_samples[retained_index] = s
            if accepted:
                accepted_retained += 1

    return (
        samples,
        thermostat_samples,
        accepted_retained,
        warmup_nonfinite,
        retained_nonfinite,
        warmup_gradient_calls,
        retained_gradient_calls,
    )


_potential_energy = _potential_energy_grex_phase_v2_20260830a
_target_gradient = _target_gradient_grex_phase_v2_20260830a
_kinetic_energy = _kinetic_energy_grex_phase_v2_20260830a
_vector_is_finite = _vector_is_finite_grex_phase_v2_20260830a
_log_h_and_derivative = _log_h_and_derivative_grex_phase_v2_20260830a
_thc_hamiltonian = _thc_hamiltonian_grex_phase_v2_20260830a
_thermostat_rotation = _thermostat_rotation_grex_phase_v2_20260830a
_thc_state_is_finite = _thc_state_is_finite_grex_phase_v2_20260830a
_thc_proposal = _thc_proposal_grex_phase_v2_20260830a
_simulate_thc_chain = _simulate_thc_chain_grex_phase_v2_20260830a


def _decode_coupling_code(coupling_code: float) -> tuple[float, float, str]:
    """Decode the compact internal code into (alpha, phi, phase label)."""

    if coupling_code < 0.0:
        return -float(coupling_code), 0.5 * math.pi, "SINE"
    return float(coupling_code), 0.0, "COSINE"


def _encode_coupling_code(alpha: float, phi: float) -> float:
    """Encode the two declared phases without changing the native signature."""

    if phi == 0.0:
        return float(alpha)
    if phi == 0.5 * math.pi:
        return -float(alpha)
    raise ValueError("phi must be exactly 0 or pi/2")


def _reference_log_h_and_derivative(
    s: float, coupling_code: float, a: float
) -> tuple[float, float]:
    """Independent scalar Python expression for the phase-periodic coupling."""

    if (
        not math.isfinite(s)
        or not math.isfinite(coupling_code)
        or not math.isfinite(a)
        or a <= 0.0
    ):
        return math.nan, math.nan
    alpha, phi, _label = _decode_coupling_code(coupling_code)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha >= 1.0:
        return math.nan, math.nan
    angle = s / a - phi
    if not math.isfinite(angle):
        return math.nan, math.nan
    cosine = math.cos(angle)
    sine = math.sin(angle)
    h = 1.0 + alpha * cosine
    if not math.isfinite(h) or h <= 0.0:
        return math.nan, math.nan
    log_h = math.log(h)
    derivative = -(alpha / a) * sine / h
    if not math.isfinite(log_h) or not math.isfinite(derivative):
        return math.nan, math.nan
    return log_h, derivative


def _reference_thc_proposal(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    current_gradient: np.ndarray,
    beta: float,
    coupling_code: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    epsilon: float,
    L: int,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, int, bool]:
    """Independent short-path Python oracle for the compiled THC proposal."""

    reference_q = np.asarray(q, dtype=np.float64).copy()
    reference_p = np.asarray(p, dtype=np.float64).copy()
    reference_s = float(s)
    reference_p_s = float(p_s)
    reference_gradient = np.asarray(
        current_gradient, dtype=np.float64
    ).copy()
    dimension = reference_q.size
    thermostat_frequency = 1.0 / (
        thermostat_sigma * math.sqrt(beta * Q)
    )
    half_angle = 0.5 * epsilon * thermostat_frequency
    cosine = math.cos(half_angle)
    sine = math.sin(half_angle)

    _log_h, log_h_derivative = _reference_log_h_and_derivative(
        reference_s, coupling_code, a
    )
    reference_p -= 0.5 * epsilon * reference_gradient
    reference_p_s -= (
        0.5 * epsilon * (dimension / beta) * log_h_derivative
    )

    gradient_calls = 0
    for step in range(L):
        old_s = reference_s
        old_p_s = reference_p_s
        reference_s = (
            old_s * cosine
            + old_p_s * sine / (Q * thermostat_frequency)
        )
        reference_p_s = (
            old_p_s * cosine
            - Q * thermostat_frequency * old_s * sine
        )

        log_h, log_h_derivative = _reference_log_h_and_derivative(
            reference_s, coupling_code, a
        )
        inverse_h_squared = math.exp(-2.0 * log_h)
        if mass_is_identity:
            inverse_mass_p = reference_p.copy()
        else:
            inverse_mass_p = inverse_mass_matrix @ reference_p
        momentum_quadratic = float(reference_p @ inverse_mass_p)
        reference_q += epsilon * inverse_h_squared * inverse_mass_p
        reference_p_s += (
            epsilon
            * inverse_h_squared
            * log_h_derivative
            * momentum_quadratic
        )

        old_s = reference_s
        old_p_s = reference_p_s
        reference_s = (
            old_s * cosine
            + old_p_s * sine / (Q * thermostat_frequency)
        )
        reference_p_s = (
            old_p_s * cosine
            - Q * thermostat_frequency * old_s * sine
        )

        x = float(reference_q[0])
        y = float(reference_q[1])
        sigma_x_squared = float(target_parameters[0])
        sigma_y_squared = float(target_parameters[1])
        bend = float(target_parameters[2])
        residual = y - bend * (x * x - sigma_x_squared)
        reference_gradient = np.asarray(
            [
                x / sigma_x_squared
                - 2.0 * bend * x * residual / sigma_y_squared,
                residual / sigma_y_squared,
            ],
            dtype=np.float64,
        )
        gradient_calls += 1
        kick_duration = 0.5 * epsilon if step == L - 1 else epsilon
        _log_h, log_h_derivative = _reference_log_h_and_derivative(
            reference_s, coupling_code, a
        )
        reference_p -= kick_duration * reference_gradient
        reference_p_s -= (
            kick_duration
            * (dimension / beta)
            * log_h_derivative
        )

    finite = bool(
        np.all(np.isfinite(reference_q))
        and np.all(np.isfinite(reference_p))
        and np.all(np.isfinite(reference_gradient))
        and math.isfinite(reference_s)
        and math.isfinite(reference_p_s)
    )
    return (
        reference_q,
        reference_p,
        reference_s,
        reference_p_s,
        reference_gradient,
        gradient_calls,
        finite,
    )


def _validate_compiled_reference_microcheck() -> str:
    """Compare current native Banana dynamics with a same-host Python oracle."""

    banana_parameters = np.asarray([4.0, 0.04, 0.25], dtype=np.float64)
    cases = (
        (
            np.asarray([0.17, -0.11], dtype=np.float64),
            np.asarray([0.31, -0.24], dtype=np.float64),
            0.23,
            -0.16,
            1.0,
            0.4,
            0.75,
            0.5,
            1.0,
            0.004,
            3,
            banana_parameters,
            np.eye(2, dtype=np.float64),
            True,
        ),
        (
            np.asarray([0.2, -0.1], dtype=np.float64),
            np.asarray([0.4, -0.25], dtype=np.float64),
            -0.2,
            0.15,
            1.3,
            -0.6,
            1.1,
            0.7,
            1.3,
            0.002,
            7,
            np.asarray([2.25, 0.09, 0.15], dtype=np.float64),
            np.linalg.inv(
                np.asarray([[2.0, 0.3], [0.3, 0.8]], dtype=np.float64)
            ),
            False,
        ),
    )
    for case_index, case in enumerate(cases, start=1):
        (
            q,
            p,
            s,
            p_s,
            beta,
            coupling_code,
            a,
            Q,
            thermostat_sigma,
            epsilon,
            L,
            target_parameters,
            inverse_mass_matrix,
            mass_is_identity,
        ) = case
        gradient = _target_gradient(q, target_parameters)
        native = _thc_proposal(
            q,
            p,
            s,
            p_s,
            gradient,
            beta,
            coupling_code,
            a,
            Q,
            thermostat_sigma,
            epsilon,
            L,
            target_parameters,
            inverse_mass_matrix,
            mass_is_identity,
        )
        reference = _reference_thc_proposal(
            q,
            p,
            s,
            p_s,
            gradient,
            beta,
            coupling_code,
            a,
            Q,
            thermostat_sigma,
            epsilon,
            L,
            target_parameters,
            inverse_mass_matrix,
            mass_is_identity,
        )
        arrays_match = all(
            np.allclose(
                native[index],
                reference[index],
                rtol=5.0e-12,
                atol=5.0e-13,
            )
            for index in (0, 1, 4)
        )
        scalars_match = all(
            math.isclose(
                float(native[index]),
                float(reference[index]),
                rel_tol=5.0e-12,
                abs_tol=5.0e-13,
            )
            for index in (2, 3)
        )
        if (
            not arrays_match
            or not scalars_match
            or native[5] != reference[5]
            or bool(native[6]) != bool(reference[6])
        ):
            raise RuntimeError(
                "Portable native-runtime preflight failed: compiled/reference "
                f"THC proposal mismatch in Banana microcheck case {case_index}."
            )

    # Exercise the independently parameterized acceptance energy as well as
    # the proposal force path.
    q = np.asarray([0.8, -0.7], dtype=np.float64)
    p = np.asarray([0.2, -0.15], dtype=np.float64)
    s = 0.31
    p_s = -0.27
    beta = 1.2
    coupling_code = -0.6
    a = 1.4
    Q = 0.25
    thermostat_sigma = 0.9
    target_parameters = np.asarray([4.0, 0.04, 0.25], dtype=np.float64)
    inverse_mass = np.asarray([[4.0, 0.0], [0.0, 2.04]], dtype=np.float64)
    residual = q[1] - 0.25 * (q[0] * q[0] - 4.0)
    potential = 0.5 * q[0] * q[0] / 4.0 + 0.5 * residual * residual / 0.04
    log_h, _log_h_derivative = _reference_log_h_and_derivative(
        s, coupling_code, a
    )
    expected_hamiltonian = (
        potential
        + 0.5 * math.exp(-2.0 * log_h) * float(p @ inverse_mass @ p)
        + (2.0 / beta) * log_h
        + 0.5 * p_s * p_s / Q
        + 0.5 * s * s / (beta * thermostat_sigma * thermostat_sigma)
    )
    native_hamiltonian = _thc_hamiltonian(
        q,
        p,
        s,
        p_s,
        beta,
        coupling_code,
        a,
        Q,
        thermostat_sigma,
        target_parameters,
        inverse_mass,
        False,
    )
    if not math.isclose(
        native_hamiltonian,
        expected_hamiltonian,
        rel_tol=5.0e-14,
        abs_tol=5.0e-14,
    ):
        raise RuntimeError(
            "Portable native-runtime preflight failed: Banana Hamiltonian oracle."
        )
    return (
        "PASS: compiled Banana THC proposal and Hamiltonian match independent "
        "same-host Python oracles for identity and general mass"
    )


def _make_random_streams(
    seed: int,
    total_iterations: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create physical-momentum and acceptance streams from fixed substreams."""

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
    """Return pooled E[beta*p^T M^-1 p/d] from retained momentum refreshes."""

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


def _thermostat_diagnostic(
    thermostat_samples: np.ndarray,
    thermostat_sigma: float,
) -> ThermostatDiagnostic:
    """Return the agreed descriptive E[s] and E[s^2] checks."""

    values = np.asarray(thermostat_samples, dtype=np.float64)
    empirical_mean = float(np.mean(values))
    empirical_second_moment = float(np.mean(values * values))
    exact_mean = 0.0
    exact_second_moment = thermostat_sigma * thermostat_sigma
    return ThermostatDiagnostic(
        empirical_mean=empirical_mean,
        exact_mean=exact_mean,
        mean_error=empirical_mean - exact_mean,
        empirical_second_moment=empirical_second_moment,
        exact_second_moment=exact_second_moment,
        second_moment_error=(
            empirical_second_moment - exact_second_moment
        ),
    )


def _aggregate_thc_configuration(
    coupling_code: float,
    a: float,
    Q: float,
    epsilon: float,
    L: int,
    block_results: Sequence[THCBlockResult],
    origin: str = "BASE_GRID",
) -> THCConfigurationResult:
    """Aggregate one fixed joint THC configuration over common seed blocks."""

    blocks = tuple(block_results)
    aggregate = _aggregate_blocks(
        tuple(block.physical for block in blocks)
    )
    return THCConfigurationResult(
        coupling_code=float(coupling_code),
        a=float(a),
        Q=float(Q),
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
        maximum_standardized_moment_error=(
            aggregate.maximum_standardized_moment_error
        ),
        total_nonfinite_trajectories=(
            aggregate.total_nonfinite_trajectories
        ),
        origin=str(origin),
    )


def _select_thc_representative(
    configuration_results: Sequence[THCConfigurationResult],
) -> THCConfigurationResult:
    """Select one fixed THC representative with an exact-tie rule."""

    eligible = [
        result
        for result in configuration_results
        if result.status == "ok"
    ]
    if not eligible:
        raise RuntimeError(
            "No THC configuration has a defined five-block tuning score."
        )

    # Exact ties: larger minimum block score, fewer gradients/draw, then
    # smaller epsilon, L, alpha, phi, a, and Q.  No near-tie tolerance or
    # diagnostic threshold is introduced.
    return max(
        eligible,
        key=_thc_configuration_rank_key,
    )


def _thc_configuration_rank_key(
    result: THCConfigurationResult,
) -> tuple[float, ...]:
    """Deterministic score/tie key used at every halving stage."""

    alpha, phi, _phase_label = _decode_coupling_code(
        result.coupling_code
    )
    return (
        result.geometric_mean_efficiency,
        result.minimum_block_efficiency,
        -result.block_results[0].physical.gradients_per_retained_draw,
        -result.epsilon,
        -float(result.L),
        -alpha,
        -phi,
        -result.a,
        -result.Q,
    )


# Validation, execution, and reporting.


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
    allow_zero: bool,
) -> tuple[float, ...]:
    """Validate one ordered THC-only real-valued tuning grid."""

    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of real numbers.") from error
    normalized: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Real
        ):
            raise ValueError(f"{name}[{index}] must be a real number.")
        numeric = float(value)
        lower_ok = numeric >= 0.0 if allow_zero else numeric > 0.0
        if not math.isfinite(numeric) or not lower_ok:
            qualifier = "nonnegative" if allow_zero else "strictly positive"
            raise ValueError(
                f"Every value in {name} must be finite and {qualifier}."
            )
        normalized.append(numeric)
    result = tuple(normalized)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _prepare_thc_inputs(inputs: THCRunInputs) -> PreparedTHCInputs:
    """Validate THC-only inputs after unchanged common input preparation."""

    if not isinstance(inputs, THCRunInputs):
        raise TypeError("THC input preparation expects one THCRunInputs object.")
    common = _prepare_inputs(inputs)

    if not isinstance(inputs.coupling_family, str):
        raise ValueError("coupling_family must be a string.")
    coupling_family = inputs.coupling_family.strip().upper()
    if coupling_family != SUPPORTED_COUPLING_FAMILY:
        raise ValueError(
            "This phase-periodic THC-v2 tuner supports only "
            f"coupling_family='{SUPPORTED_COUPLING_FAMILY}'."
        )

    alpha_values = _normalize_method_grid(
        "alpha_values", inputs.alpha_values, allow_zero=False
    )
    if any(value >= 1.0 for value in alpha_values):
        raise ValueError("Every alpha must satisfy 0 < alpha < 1.")
    try:
        raw_phi_values = tuple(float(value) for value in inputs.phi_values)
    except (TypeError, ValueError) as error:
        raise ValueError("phi_values must be real numbers.") from error
    if not raw_phi_values or len(set(raw_phi_values)) != len(raw_phi_values):
        raise ValueError("phi_values must be nonempty and unique.")
    allowed_phases = (0.0, 0.5 * math.pi)
    if any(value not in allowed_phases for value in raw_phi_values):
        raise ValueError("Every phi must be exactly 0 or pi/2.")
    phi_values = tuple(raw_phi_values)
    phase_codes = tuple(
        _encode_coupling_code(alpha, phi)
        for phi in phi_values
        for alpha in alpha_values
    )
    a_values = _normalize_method_grid(
        "a_values", inputs.a_values, allow_zero=False
    )
    Q_values = _normalize_method_grid(
        "Q_values", inputs.Q_values, allow_zero=False
    )

    def positive_integer(name: str, value: object, minimum: int) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral
        ):
            raise ValueError(f"{name} must be an integer.")
        result = int(value)
        if result < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        return result

    try:
        raw_anchor_pairs = tuple(tuple(pair) for pair in inputs.stage1_anchor_pairs)
    except TypeError as error:
        raise ValueError("stage1_anchor_pairs must be pairs.") from error
    anchor_pairs: list[tuple[float, int]] = []
    for index, pair in enumerate(raw_anchor_pairs):
        if len(pair) != 2:
            raise ValueError(f"stage1_anchor_pairs[{index}] must have length 2.")
        epsilon = float(pair[0])
        L_raw = pair[1]
        if isinstance(L_raw, (bool, np.bool_)) or not isinstance(
            L_raw, numbers.Integral
        ):
            raise ValueError("Every anchor L must be an integer.")
        L = int(L_raw)
        if epsilon not in common.epsilon_values or L not in common.L_values:
            raise ValueError(
                f"Anchor {(epsilon, L)} is outside the unchanged common grid."
            )
        anchor_pairs.append((epsilon, L))
    stage1_anchor_pairs = tuple(anchor_pairs)
    if not stage1_anchor_pairs or len(set(stage1_anchor_pairs)) != len(stage1_anchor_pairs):
        raise ValueError("stage1_anchor_pairs must be nonempty and unique.")

    def normalize_block_indices(name: str, values: Sequence[int]) -> tuple[int, ...]:
        try:
            raw = tuple(values)
        except TypeError as error:
            raise ValueError(f"{name} must be an integer sequence.") from error
        normalized: list[int] = []
        for value in raw:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numbers.Integral
            ):
                raise ValueError(f"Every {name} entry must be an integer.")
            one_based = int(value)
            if one_based < 1 or one_based > len(common.seed_blocks):
                raise ValueError(f"Every {name} entry must be in 1..{len(common.seed_blocks)}.")
            normalized.append(one_based - 1)
        result = tuple(normalized)
        if not result or len(set(result)) != len(result):
            raise ValueError(f"{name} must be nonempty and unique.")
        return result

    stage1_block_indices = normalize_block_indices(
        "stage1_block_indices", inputs.stage1_block_indices
    )
    stage2_block_indices = normalize_block_indices(
        "stage2_block_indices", inputs.stage2_block_indices
    )
    stage1_n_warmup = positive_integer(
        "stage1_n_warmup", inputs.stage1_n_warmup, 1
    )
    stage1_n_retained = positive_integer(
        "stage1_n_retained", inputs.stage1_n_retained, 4
    )
    stage2_n_warmup = positive_integer(
        "stage2_n_warmup", inputs.stage2_n_warmup, stage1_n_warmup
    )
    stage2_n_retained = positive_integer(
        "stage2_n_retained", inputs.stage2_n_retained, stage1_n_retained
    )
    if stage2_n_warmup > common.n_warmup or stage2_n_retained > common.n_retained:
        raise ValueError("Stage-2 lengths cannot exceed the final lengths.")
    method_tuple_count = len(phase_codes) * len(a_values) * len(Q_values)
    stage1_keep_method_tuples = positive_integer(
        "stage1_keep_method_tuples", inputs.stage1_keep_method_tuples, 1
    )
    if stage1_keep_method_tuples > method_tuple_count:
        raise ValueError("stage1_keep_method_tuples exceeds the method grid.")
    stage2_keep_fixed_configurations = positive_integer(
        "stage2_keep_fixed_configurations",
        inputs.stage2_keep_fixed_configurations,
        1,
    )
    boundary_keep_fixed_configurations = positive_integer(
        "boundary_keep_fixed_configurations",
        inputs.boundary_keep_fixed_configurations,
        1,
    )

    boundary_alpha_lower = float(inputs.boundary_alpha_lower)
    boundary_alpha_upper = float(inputs.boundary_alpha_upper)
    boundary_a_lower = float(inputs.boundary_a_lower)
    boundary_a_upper = float(inputs.boundary_a_upper)
    boundary_Q_lower = float(inputs.boundary_Q_lower)
    boundary_Q_upper = float(inputs.boundary_Q_upper)
    if not (
        0.0 < boundary_alpha_lower < alpha_values[0]
        and alpha_values[-1] < boundary_alpha_upper < 1.0
        and 0.0 < boundary_a_lower < a_values[0]
        and boundary_a_upper > a_values[-1]
        and 0.0 < boundary_Q_lower < Q_values[0]
        and boundary_Q_upper > Q_values[-1]
    ):
        raise ValueError("Declared boundary values must lie outside the base grid.")

    if isinstance(inputs.thermostat_sigma, (bool, np.bool_)) or not isinstance(
        inputs.thermostat_sigma, numbers.Real
    ):
        raise ValueError("thermostat_sigma must be a real number.")
    thermostat_sigma = float(inputs.thermostat_sigma)
    if not math.isfinite(thermostat_sigma) or thermostat_sigma <= 0.0:
        raise ValueError(
            "thermostat_sigma must be finite and strictly positive."
        )
    thermostat_variance = thermostat_sigma * thermostat_sigma
    if (
        not math.isfinite(thermostat_variance)
        or thermostat_variance <= 0.0
    ):
        raise ValueError(
            "thermostat_sigma**2 must be finite and strictly positive."
        )
    for Q in Q_values:
        thermostat_momentum_variance = Q / common.beta
        oscillator_denominator = (
            common.beta * Q * thermostat_variance
        )
        if (
            not math.isfinite(thermostat_momentum_variance)
            or thermostat_momentum_variance <= 0.0
            or not math.isfinite(oscillator_denominator)
            or oscillator_denominator <= 0.0
        ):
            raise ValueError(
                "Each Q must give finite, strictly positive thermostat "
                "momentum variance Q/beta and oscillator denominator "
                "beta*Q*thermostat_sigma**2."
            )

    try:
        thermostat_starts = np.asarray(
            inputs.thermostat_starts, dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "thermostat_starts must be a numeric one-dimensional vector."
        ) from error
    if thermostat_starts.shape != (common.n_chains,):
        raise ValueError(
            "thermostat_starts must contain one scalar per physical chain; "
            f"expected shape {(common.n_chains,)}, received "
            f"{thermostat_starts.shape}."
        )
    if not np.all(np.isfinite(thermostat_starts)):
        raise ValueError(
            "thermostat_starts must contain only finite values."
        )
    thermostat_starts = np.ascontiguousarray(thermostat_starts.copy())

    components: dict[str, object] = {
        "coupling": {
            "family": coupling_family,
            "formula": "h=1+alpha*cos(s/a-phi)",
        },
        "phase_grid": {
            "alpha_values": list(alpha_values),
            "phi_values": list(phi_values),
            "encoded_phase_codes": list(phase_codes),
        },
        "a_grid": {"a_values": list(a_values)},
        "Q_grid": {"Q_values": list(Q_values)},
        "thermostat_density_and_starts": {
            "thermostat_sigma": thermostat_sigma,
            "thermostat_starts": thermostat_starts.tolist(),
        },
        "successive_halving": {
            "stage1_anchor_pairs": [list(pair) for pair in stage1_anchor_pairs],
            "stage1_blocks_one_based": [value + 1 for value in stage1_block_indices],
            "stage1_lengths": [stage1_n_warmup, stage1_n_retained],
            "stage1_keep_method_tuples": stage1_keep_method_tuples,
            "stage2_blocks_one_based": [value + 1 for value in stage2_block_indices],
            "stage2_lengths": [stage2_n_warmup, stage2_n_retained],
            "stage2_keep_fixed_configurations": stage2_keep_fixed_configurations,
            "final_blocks_one_based": list(range(1, len(common.seed_blocks) + 1)),
            "final_lengths": [common.n_warmup, common.n_retained],
            "boundary_keep_fixed_configurations": boundary_keep_fixed_configurations,
        },
        "boundary_layer": {
            "automatic": bool(inputs.automatic_boundary_expansion),
            "alpha": [boundary_alpha_lower, boundary_alpha_upper],
            "a": [boundary_a_lower, boundary_a_upper],
            "Q": [boundary_Q_lower, boundary_Q_upper],
            "epsilon_L_expansion": False,
            "second_expansion": False,
        },
    }
    component_fingerprints = {
        name: _json_fingerprint(value)
        for name, value in components.items()
    }
    canonical_method_mismatches: tuple[str, ...] = ()
    method_input_fingerprint = _json_fingerprint(
        {
            "schema": THC_METHOD_INPUT_SCHEMA_VERSION,
            "components": components,
        }
    )
    thermostat_starts.setflags(write=False)

    return PreparedTHCInputs(
        common=common,
        coupling_family=coupling_family,
        phase_codes=phase_codes,
        alpha_values=alpha_values,
        phi_values=phi_values,
        a_values=a_values,
        Q_values=Q_values,
        thermostat_sigma=thermostat_sigma,
        thermostat_starts=thermostat_starts,
        stage1_anchor_pairs=stage1_anchor_pairs,
        stage1_block_indices=stage1_block_indices,
        stage1_n_warmup=stage1_n_warmup,
        stage1_n_retained=stage1_n_retained,
        stage1_keep_method_tuples=stage1_keep_method_tuples,
        stage2_block_indices=stage2_block_indices,
        stage2_n_warmup=stage2_n_warmup,
        stage2_n_retained=stage2_n_retained,
        stage2_keep_fixed_configurations=stage2_keep_fixed_configurations,
        boundary_keep_fixed_configurations=boundary_keep_fixed_configurations,
        automatic_boundary_expansion=bool(inputs.automatic_boundary_expansion),
        boundary_alpha_lower=boundary_alpha_lower,
        boundary_alpha_upper=boundary_alpha_upper,
        boundary_a_lower=boundary_a_lower,
        boundary_a_upper=boundary_a_upper,
        boundary_Q_lower=boundary_Q_lower,
        boundary_Q_upper=boundary_Q_upper,
        method_input_components=components,
        method_input_fingerprint=method_input_fingerprint,
        canonical_method_mismatches=canonical_method_mismatches,
    )


def _validate_thc_run_mode(prepared: PreparedTHCInputs) -> None:
    """Apply common mode separation; live method inputs are hash-reported."""

    _validate_run_mode(prepared.common)
    if prepared.canonical_method_mismatches:
        raise ValueError(
            "Invalid phase-periodic THC method inputs: "
            + ", ".join(prepared.canonical_method_mismatches)
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
            script_path = Path(__file__).resolve()
        except NameError:  # Jupyter execution after pasting the script
            directory = Path.cwd().resolve()
        else:
            # IPython can retain an unrelated/stale __file__.  Trust it only
            # for this exact standalone filename; otherwise use the kernel cwd.
            directory = (
                script_path.parent
                if script_path.name == "banana_phase_periodic_thc_v2_tuner_GREX_PORTABLE.py"
                else Path.cwd().resolve()
            )
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


def _thc_implementation_fingerprint() -> str:
    """Hash THC-only input handling, dynamics, aggregation, and execution."""

    return _source_fingerprint(
        (
            THCRunInputs,
            PreparedTHCInputs,
            ThermostatDiagnostic,
            THCBlockResult,
            THCConfigurationResult,
            _normalize_method_grid,
            _prepare_thc_inputs,
            _potential_energy,
            _target_gradient,
            _kinetic_energy,
            _vector_is_finite,
            _log_h_and_derivative,
            _thc_hamiltonian,
            _thermostat_rotation,
            _thc_state_is_finite,
            _thc_proposal,
            _simulate_thc_chain,
            _thermostat_diagnostic,
            _aggregate_thc_configuration,
            _select_thc_representative,
            _thc_configuration_rank_key,
            _run_chain_job,
            _run_configuration_block,
            _evaluate_fixed_configuration,
            _rank_defined_configurations,
            _canonical_halving_design_mismatches,
            _boundary_method_layer,
            _build_phase_periodic_report,
            run_phase_periodic_tuning,
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
    """Require live source attestation without a nonportable fixed digest."""

    source_attestations = (
        _input_preparation_fingerprint(),
        _diagnostics_fingerprint(),
        _common_pipeline_fingerprint(),
        _shared_protocol_fingerprint(prepared),
    )
    if any(value == "unavailable" for value in source_attestations):
        raise RuntimeError(
            "Live source attestation is unavailable. Execute this entire "
            "standalone file as one Jupyter cell or run the saved script."
        )


def _format_number(value: float, specification: str = ".6g") -> str:
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, specification)


def _run_chain_job(
    standard_momenta: np.ndarray,
    thermostat_standard_momenta: np.ndarray,
    uniforms: np.ndarray,
    q_initial: np.ndarray,
    s_initial: float,
    beta: float,
    coupling_code: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    return _simulate_thc_chain(
        standard_momenta,
        thermostat_standard_momenta,
        uniforms,
        q_initial,
        s_initial,
        beta,
        coupling_code,
        a,
        Q,
        thermostat_sigma,
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
    coupling_code: float,
    a: float,
    Q: float,
    epsilon: float,
    L: int,
    block_index: int,
    seeds: Sequence[int],
    stream_cache: Sequence[
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    prepared: PreparedTHCInputs,
    n_warmup: int | None = None,
    n_retained: int | None = None,
) -> THCBlockResult:
    common = prepared.common
    actual_warmup = common.n_warmup if n_warmup is None else int(n_warmup)
    actual_retained = common.n_retained if n_retained is None else int(n_retained)
    total_iterations = actual_warmup + actual_retained
    jobs = [
        (
            stream_cache[chain_index][0][:total_iterations],
            stream_cache[chain_index][1][:total_iterations],
            stream_cache[chain_index][2][:total_iterations],
            common.physical_starts[chain_index],
            prepared.thermostat_starts[chain_index],
            common.beta,
            coupling_code,
            a,
            Q,
            prepared.thermostat_sigma,
            epsilon,
            L,
            actual_warmup,
            actual_retained,
            common.target_parameters,
            common.mass_cholesky,
            common.inverse_mass_matrix,
            common.mass_is_identity,
        )
        for chain_index in range(common.n_chains)
    ]

    if common.n_workers > 1:
        with ThreadPoolExecutor(
            max_workers=min(
                common.n_workers, common.n_chains
            )
        ) as executor:
            chain_outputs = list(
                executor.map(
                    lambda arguments: _run_chain_job(*arguments),
                    jobs,
                )
            )
    else:
        chain_outputs = [
            _run_chain_job(*arguments) for arguments in jobs
        ]

    samples = np.stack(
        [output[0] for output in chain_outputs], axis=0
    )
    thermostat_samples = np.stack(
        [output[1] for output in chain_outputs], axis=0
    )
    accepted = sum(output[2] for output in chain_outputs)
    warmup_nonfinite = sum(
        output[3] for output in chain_outputs
    )
    retained_nonfinite = sum(
        output[4] for output in chain_outputs
    )
    retained_gradient_calls = sum(
        output[6] for output in chain_outputs
    )

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
    thermostat = _thermostat_diagnostic(
        thermostat_samples, prepared.thermostat_sigma
    )
    return THCBlockResult(
        physical=physical, thermostat=thermostat
    )


def _fixed_configuration_from_result(
    result: THCConfigurationResult,
) -> FixedConfiguration:
    return FixedConfiguration(
        coupling_code=result.coupling_code,
        a=result.a,
        Q=result.Q,
        epsilon=result.epsilon,
        L=result.L,
    )


def _method_tuple_from_result(
    result: THCConfigurationResult,
) -> MethodTuple:
    return MethodTuple(
        coupling_code=result.coupling_code,
        a=result.a,
        Q=result.Q,
    )


def _evaluate_fixed_configuration(
    configuration: FixedConfiguration,
    *,
    block_indices: Sequence[int],
    n_warmup: int,
    n_retained: int,
    origin: str,
    stream_caches: Sequence[
        Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ],
    prepared: PreparedTHCInputs,
) -> THCConfigurationResult:
    """Evaluate one fixed configuration on the declared common blocks."""

    blocks: list[THCBlockResult] = []
    for block_index in block_indices:
        blocks.append(
            _run_configuration_block(
                coupling_code=configuration.coupling_code,
                a=configuration.a,
                Q=configuration.Q,
                epsilon=configuration.epsilon,
                L=configuration.L,
                block_index=block_index,
                seeds=prepared.common.seed_blocks[block_index],
                stream_cache=stream_caches[block_index],
                prepared=prepared,
                n_warmup=n_warmup,
                n_retained=n_retained,
            )
        )
    return _aggregate_thc_configuration(
        configuration.coupling_code,
        configuration.a,
        configuration.Q,
        configuration.epsilon,
        configuration.L,
        blocks,
        origin=origin,
    )


def _rank_defined_configurations(
    results: Sequence[THCConfigurationResult],
    keep: int,
) -> tuple[THCConfigurationResult, ...]:
    """Rank only mathematically defined scores; impose no diagnostic gate."""

    defined = [result for result in results if result.status == "ok"]
    defined.sort(key=_thc_configuration_rank_key, reverse=True)
    return tuple(defined[: min(int(keep), len(defined))])


def _canonical_halving_design_mismatches(
    prepared: PreparedTHCInputs,
) -> tuple[str, ...]:
    """Return deviations from the frozen phase-periodic Banana campaign."""

    common = prepared.common
    expected: dict[str, object] = {
        "alpha_values": (0.10, 0.30, 0.60, 0.90),
        "phi_values": (0.0, 0.5 * math.pi),
        "a_values": (0.50, 0.75, 1.0, 1.5, 2.0),
        "Q_values": (0.10, 0.25, 0.50, 1.0, 2.0, 4.0),
        "thermostat_sigma": 1.0,
        "thermostat_starts": (-1.0, 1.0, 1.0, -1.0),
        "stage1_anchor_pairs": (
            (0.02, 120),
            (0.04, 80),
            (0.05, 40),
            (0.06, 20),
        ),
        "stage1_block_indices": (0,),
        "stage1_lengths": (250, 1000),
        "stage1_keep": 12,
        "stage2_block_indices": (1, 2),
        "stage2_lengths": (500, 2000),
        "stage2_keep": 12,
        "final_lengths": (1000, 4000),
        "boundary_keep": 4,
        "boundary_values": (0.05, 0.95, 0.375, 3.0, 0.05, 8.0),
        "automatic_boundary_expansion": True,
    }
    actual: dict[str, object] = {
        "alpha_values": prepared.alpha_values,
        "phi_values": prepared.phi_values,
        "a_values": prepared.a_values,
        "Q_values": prepared.Q_values,
        "thermostat_sigma": prepared.thermostat_sigma,
        "thermostat_starts": tuple(
            float(value) for value in prepared.thermostat_starts
        ),
        "stage1_anchor_pairs": prepared.stage1_anchor_pairs,
        "stage1_block_indices": prepared.stage1_block_indices,
        "stage1_lengths": (
            prepared.stage1_n_warmup,
            prepared.stage1_n_retained,
        ),
        "stage1_keep": prepared.stage1_keep_method_tuples,
        "stage2_block_indices": prepared.stage2_block_indices,
        "stage2_lengths": (
            prepared.stage2_n_warmup,
            prepared.stage2_n_retained,
        ),
        "stage2_keep": prepared.stage2_keep_fixed_configurations,
        "final_lengths": (common.n_warmup, common.n_retained),
        "boundary_keep": prepared.boundary_keep_fixed_configurations,
        "boundary_values": (
            prepared.boundary_alpha_lower,
            prepared.boundary_alpha_upper,
            prepared.boundary_a_lower,
            prepared.boundary_a_upper,
            prepared.boundary_Q_lower,
            prepared.boundary_Q_upper,
        ),
        "automatic_boundary_expansion": prepared.automatic_boundary_expansion,
    }
    return tuple(name for name in expected if actual[name] != expected[name])


def _boundary_method_layer(
    base_winner: THCConfigurationResult,
    prepared: PreparedTHCInputs,
) -> tuple[tuple[MethodTuple, ...], tuple[str, ...]]:
    """Construct the one predeclared local outward method-boundary layer."""

    alpha, phi, _phase_label = _decode_coupling_code(
        base_winner.coupling_code
    )
    alpha_min = min(prepared.alpha_values)
    alpha_max = max(prepared.alpha_values)
    a_min = min(prepared.a_values)
    a_max = max(prepared.a_values)
    Q_min = min(prepared.Q_values)
    Q_max = max(prepared.Q_values)
    alpha_choices = [alpha]
    a_choices = [base_winner.a]
    Q_choices = [base_winner.Q]
    labels: list[str] = []

    if alpha == alpha_min:
        alpha_choices.append(prepared.boundary_alpha_lower)
        labels.append("alpha_lower")
    elif alpha == alpha_max:
        alpha_choices.append(prepared.boundary_alpha_upper)
        labels.append("alpha_upper")
    if base_winner.a == a_min:
        a_choices.append(prepared.boundary_a_lower)
        labels.append("a_lower")
    elif base_winner.a == a_max:
        a_choices.append(prepared.boundary_a_upper)
        labels.append("a_upper")
    if base_winner.Q == Q_min:
        Q_choices.append(prepared.boundary_Q_lower)
        labels.append("Q_lower")
    elif base_winner.Q == Q_max:
        Q_choices.append(prepared.boundary_Q_upper)
        labels.append("Q_upper")

    base_method = _method_tuple_from_result(base_winner)
    layer = {
        MethodTuple(
            coupling_code=_encode_coupling_code(candidate_alpha, phi),
            a=candidate_a,
            Q=candidate_Q,
        )
        for candidate_alpha in alpha_choices
        for candidate_a in a_choices
        for candidate_Q in Q_choices
    }
    layer.discard(base_method)
    return tuple(sorted(layer)), tuple(labels)


def _whole_source_sha256() -> str:
    """Hash the saved standalone source when it is available."""

    try:
        source_path = Path(__file__).resolve()
    except NameError:
        return "unavailable_in_pasted_cell"
    if (
        source_path.name
        != "banana_phase_periodic_thc_v2_tuner_GREX_PORTABLE.py"
        or not source_path.is_file()
    ):
        return "unavailable_in_pasted_cell"
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _phase_method_text(coupling_code: float, a: float, Q: float) -> str:
    alpha, phi, phase_label = _decode_coupling_code(coupling_code)
    return (
        f"phase={phase_label}, alpha={alpha:.12g}, phi={phi:.12g}, "
        f"a={a:.12g}, Q={Q:.12g}"
    )


def _configuration_summary_line(
    result: THCConfigurationResult,
) -> str:
    physical_blocks = tuple(block.physical for block in result.block_results)
    worst_diagnostics = tuple(
        next(
            diagnostic
            for diagnostic in block.observable_diagnostics
            if diagnostic.name == block.worst_observable
        )
        for block in physical_blocks
    )
    return (
        _phase_method_text(result.coupling_code, result.a, result.Q)
        + f", epsilon={result.epsilon:.12g}, L={result.L}, "
        + f"origin={result.origin}, status={result.status}, "
        + "GM_efficiency="
        + _format_number(result.geometric_mean_efficiency, ".10e")
        + ", min_block_efficiency="
        + _format_number(result.minimum_block_efficiency, ".10e")
        + ", max_Rhat="
        + _format_number(result.maximum_rhat, ".9f")
        + ", max_std_moment_error="
        + _format_number(result.maximum_standardized_moment_error, ".9e")
        + f", nonfinite={result.total_nonfinite_trajectories}"
        + ", block_efficiencies="
        + repr([block.block_efficiency for block in physical_blocks])
        + ", block_worst_IATs="
        + repr([block.worst_iat for block in physical_blocks])
        + ", block_worst_observables="
        + repr([block.worst_observable for block in physical_blocks])
        + ", gradients_per_draw="
        + repr([block.gradients_per_retained_draw for block in physical_blocks])
        + ", Geyer_last_lags="
        + repr(
            [diagnostic.iat.last_included_lag for diagnostic in worst_diagnostics]
        )
        + ", Geyer_first_excluded_lags="
        + repr(
            [diagnostic.iat.first_excluded_lag for diagnostic in worst_diagnostics]
        )
        + ", Geyer_cutoff_reasons="
        + repr([diagnostic.iat.cutoff_reason for diagnostic in worst_diagnostics])
    )


def _append_result_list(
    lines: list[str],
    title: str,
    results: Sequence[THCConfigurationResult],
) -> None:
    lines.extend(["", title, "-" * len(title)])
    if not results:
        lines.append("NONE")
        return
    for index, result in enumerate(results, start=1):
        lines.append(f"{index:05d}  {_configuration_summary_line(result)}")


def _build_phase_periodic_report(
    *,
    prepared: PreparedTHCInputs,
    self_test_messages: Sequence[str],
    auxiliary_momentum_moments: Sequence[float],
    stage1_results: Sequence[THCConfigurationResult],
    stage1_promoted_methods: Sequence[MethodTuple],
    stage2_results: Sequence[THCConfigurationResult],
    stage2_promoted: Sequence[THCConfigurationResult],
    base_finalists: Sequence[THCConfigurationResult],
    boundary_labels: Sequence[str],
    boundary_stage2_results: Sequence[THCConfigurationResult],
    boundary_stage2_promoted: Sequence[THCConfigurationResult],
    boundary_finalists: Sequence[THCConfigurationResult],
    representative: THCConfigurationResult,
) -> str:
    """Build the complete plain-text tuning audit and final diagnostics."""

    common = prepared.common
    lines: list[str] = [
        "PHASE-PERIODIC THC-V2 TUNING REPORT: 2D BANANA",
        "=" * 56,
        f"run label: {common.run_label}",
        f"runtime: {_runtime_environment_summary()}",
        f"source verification: {_source_verification_mode()}",
        f"whole saved source SHA-256: {_whole_source_sha256()}",
        f"common input SHA-256: {common.common_input_fingerprint}",
        f"THC method input SHA-256: {prepared.method_input_fingerprint}",
        f"input-preparation source SHA-256: {_input_preparation_fingerprint()}",
        f"diagnostics source SHA-256: {_diagnostics_fingerprint()}",
        f"common-pipeline source SHA-256: {_common_pipeline_fingerprint()}",
        f"THC implementation source SHA-256: {_thc_implementation_fingerprint()}",
        f"shared protocol SHA-256: {_shared_protocol_fingerprint(common)}",
        "",
        "TARGET, GEOMETRY, AND DYNAMICS",
        "------------------------------",
        "d=2; beta=1; sigma_x=2; sigma_y=0.2; b=0.25",
        "r=y-b*(x^2-sigma_x^2)",
        "U=x^2/(2*sigma_x^2)+r^2/(2*sigma_y^2)",
        f"mass M={common.mass_matrix.tolist()}",
        f"inverse mass={common.inverse_mass_matrix.tolist()}",
        f"standardized starts={common.standardized_starts.tolist()}",
        f"physical starts={common.physical_starts.tolist()}",
        f"persistent thermostat starts={prepared.thermostat_starts.tolist()}",
        "rho(s)=N(0,1); p is refreshed from N(0,h(s)^2*M/beta); "
        "p_s is refreshed from N(0,Q/beta); s persists",
        "h(s)=1+alpha*cos(s/a-phi), with 0<alpha<1",
        "H=U(q)+p^T M^{-1}p/(2h(s)^2)+(d/beta)log h(s)"
        "+p_s^2/(2Q)+s^2/(2beta)",
        "integrator=symmetric P/S/K/S/P exact-subflow splitting; "
        "full-state Metropolis correction",
        "actual retained target-gradient calls are counted; cached endpoint "
        "gradients are reused",
        "",
        "OBSERVABLES AND SCORE",
        "---------------------",
        f"observables={list(common.observable_names)}",
        f"exact means={common.exact_observable_means.tolist()}",
        f"exact SDs={common.exact_observable_sds.tolist()}",
        "IAT=multi-chain Geyer initial-positive plus initial-monotone sequence",
        "efficiency(f)=1/(actual gradients per retained draw * IAT(f))",
        "block score=min efficiency over the five observables",
        "fixed-configuration score=geometric mean of its declared block scores",
        "exact tie key=larger GM, larger minimum block score, fewer gradients/draw, "
        "then smaller epsilon, L, alpha, phi, a, and Q",
        "No R-hat, moment, acceptance, or nonfinite-count threshold is used.",
        "Any undefined required IAT or nonfinite simulation gives undefined "
        "efficiency and cannot have a finite tuning score.",
        "",
        "FROZEN SUCCESSIVE-HALVING DESIGN",
        "---------------------------------",
        f"base alpha={list(prepared.alpha_values)}",
        f"phases={list(prepared.phi_values)} (0=cosine, pi/2=sine)",
        f"base a={list(prepared.a_values)}",
        f"base Q={list(prepared.Q_values)}",
        f"full common epsilon={list(common.epsilon_values)}",
        f"full common L={list(common.L_values)}",
        f"tuning seed blocks={list(common.seed_blocks)}",
        f"stage 1: anchors={list(prepared.stage1_anchor_pairs)}, "
        f"blocks={[i + 1 for i in prepared.stage1_block_indices]}, "
        f"lengths={prepared.stage1_n_warmup}+{prepared.stage1_n_retained}, "
        f"keep={prepared.stage1_keep_method_tuples} method tuples",
        f"stage 2: full {len(common.epsilon_values)}x{len(common.L_values)} "
        "epsilon-L grid, "
        f"blocks={[i + 1 for i in prepared.stage2_block_indices]}, "
        f"lengths={prepared.stage2_n_warmup}+{prepared.stage2_n_retained}, "
        f"keep={prepared.stage2_keep_fixed_configurations} fixed configurations",
        f"stage 3: all five blocks, lengths={common.n_warmup}+{common.n_retained}",
        "Stages 1 and 2 use deterministic prefixes of the same declared tuning "
        "streams used at Stage 3. This is nested-budget successive halving; all "
        "five blocks are tuning data and no held-out final stream is constructed.",
        "boundary rule: one local outward method layer only; alpha=.05/.95, "
        "a=.375/3, Q=.05/8; full unchanged epsilon-L grid; "
        "stage-2 blocks/lengths; "
        f"keep={prepared.boundary_keep_fixed_configurations}; then all five "
        "full blocks; no epsilon-L expansion and no second expansion",
        "boundary trigger labels="
        + (", ".join(boundary_labels) if boundary_labels else "NONE"),
        "",
        "SELF-TESTS AND RANDOM-STREAM AUDIT",
        "----------------------------------",
    ]
    lines.extend(self_test_messages)
    for index, moment in enumerate(auxiliary_momentum_moments, start=1):
        lines.append(
            f"tuning block {index} standardized refreshed-p second moment="
            f"{moment:.12g} (exact 1; descriptive only)"
        )

    _append_result_list(lines, "STAGE 1 ALL ANCHOR RESULTS", stage1_results)
    lines.extend(["", "STAGE 1 PROMOTED METHOD TUPLES", "-------------------------------"])
    for index, method in enumerate(stage1_promoted_methods, start=1):
        lines.append(
            f"{index:02d}  "
            + _phase_method_text(method.coupling_code, method.a, method.Q)
        )
    _append_result_list(lines, "STAGE 2 ALL FIXED-CONFIGURATION RESULTS", stage2_results)
    _append_result_list(lines, "STAGE 2 PROMOTED FIXED CONFIGURATIONS", stage2_promoted)
    _append_result_list(lines, "BASE STAGE 3 FULL FIVE-BLOCK FINALISTS", base_finalists)
    _append_result_list(lines, "BOUNDARY STAGE 2 ALL RESULTS", boundary_stage2_results)
    _append_result_list(lines, "BOUNDARY STAGE 2 PROMOTED CONFIGURATIONS", boundary_stage2_promoted)
    _append_result_list(lines, "BOUNDARY STAGE 3 FULL FIVE-BLOCK FINALISTS", boundary_finalists)

    lines.extend(
        [
            "",
            "FROZEN REPRESENTATIVE",
            "---------------------",
            _configuration_summary_line(representative),
            "This one fixed representative is selected by the declared score "
            "and exact-tie rule only.",
            "",
            "FROZEN REPRESENTATIVE: EVERY BLOCK AND OBSERVABLE",
            "--------------------------------------------------",
        ]
    )
    for thc_block in representative.block_results:
        block = thc_block.physical
        lines.extend(
            [
                "",
                f"BLOCK {block.block_index + 1}; seeds={list(block.seeds)}; "
                f"status={block.status}",
                f"acceptance={block.acceptance:.12g}; retained_gradient_calls="
                f"{block.retained_gradient_calls}; gradients/draw="
                f"{block.gradients_per_retained_draw:.12g}",
                f"warmup_nonfinite={block.warmup_nonfinite_trajectories}; "
                f"retained_nonfinite={block.retained_nonfinite_trajectories}",
                f"worst observable={block.worst_observable}; worst IAT="
                f"{_format_number(block.worst_iat, '.12g')}; block efficiency="
                f"{_format_number(block.block_efficiency, '.12e')}",
                f"maximum Rhat observable={block.maximum_rhat_observable}; "
                f"maximum Rhat={_format_number(block.maximum_rhat, '.12g')}",
                "maximum standardized moment-error observable="
                f"{block.maximum_moment_error_observable}; value="
                f"{_format_number(block.maximum_standardized_moment_error, '.12g')}",
                f"thermostat mean={thc_block.thermostat.empirical_mean:.12g}; "
                f"exact=0; error={thc_block.thermostat.mean_error:.12g}",
                f"thermostat second moment="
                f"{thc_block.thermostat.empirical_second_moment:.12g}; exact="
                f"{thc_block.thermostat.exact_second_moment:.12g}; error="
                f"{thc_block.thermostat.second_moment_error:.12g}",
                "observable | IAT | status | pair_count | last_lag | "
                "first_excluded_lag | cutoff_reason | ESS | efficiency | "
                "rank_Rhat | folded_Rhat | combined_Rhat | empirical_mean | "
                "exact_mean | raw_error | standardized_error",
            ]
        )
        for diagnostic in block.observable_diagnostics:
            iat = diagnostic.iat
            rhat = diagnostic.rhat
            lines.append(
                f"{diagnostic.name} | {_format_number(iat.tau, '.12g')} | "
                f"{iat.status} | {iat.pair_count} | {iat.last_included_lag} | "
                f"{iat.first_excluded_lag} | {iat.cutoff_reason} | "
                f"{_format_number(diagnostic.ess, '.12g')} | "
                f"{_format_number(diagnostic.efficiency, '.12e')} | "
                f"{_format_number(rhat.rank, '.12g')} | "
                f"{_format_number(rhat.folded, '.12g')} | "
                f"{_format_number(rhat.combined, '.12g')} | "
                f"{diagnostic.empirical_mean:.12g} | "
                f"{diagnostic.exact_mean:.12g} | "
                f"{diagnostic.raw_moment_error:.12g} | "
                f"{diagnostic.standardized_moment_error:.12g}"
            )
    lines.extend(["", "END OF REPORT"])
    return "\n".join(lines)


def _runtime_environment_summary() -> str:
    """Return the numerical runtime that must accompany stochastic results."""

    numba_version = numba.__version__ if NUMBA_AVAILABLE else "unavailable"
    llvmlite_version = (
        llvmlite.__version__ if NUMBA_AVAILABLE else "unavailable"
    )
    python_version = platform.python_version()
    return (
        f"platform={platform.platform()}; Python={python_version}; "
        f"NumPy={np.__version__}; Numba={numba_version}; "
        f"llvmlite={llvmlite_version}"
    )


def _first_tuning_block_stream_digest(prepared: PreparedTHCInputs) -> str:
    """Hash canonical little-endian random arrays without simulating a chain."""

    common = prepared.common
    total_iterations = common.n_warmup + common.n_retained
    digest = hashlib.sha256()
    for seed in common.seed_blocks[0]:
        for array in _make_random_streams(
            seed, total_iterations, common.dimension
        ):
            canonical = np.ascontiguousarray(array, dtype="<f8")
            digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _validate_portable_native_runtime(prepared: PreparedTHCInputs) -> str:
    """Verify the live cache-free epoch without pinning a stochastic path."""

    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "Canonical Grex production requires Numba. Select the configured "
            "Python 3.12 sampling kernel before starting the tuning grid."
        )

    expected_dispatchers = (
        ("_potential_energy", _potential_energy, _potential_energy_grex_phase_v2_20260830a),
        ("_target_gradient", _target_gradient, _target_gradient_grex_phase_v2_20260830a),
        ("_kinetic_energy", _kinetic_energy, _kinetic_energy_grex_phase_v2_20260830a),
        ("_vector_is_finite", _vector_is_finite, _vector_is_finite_grex_phase_v2_20260830a),
        (
            "_log_h_and_derivative",
            _log_h_and_derivative,
            _log_h_and_derivative_grex_phase_v2_20260830a,
        ),
        ("_thc_hamiltonian", _thc_hamiltonian, _thc_hamiltonian_grex_phase_v2_20260830a),
        (
            "_thermostat_rotation",
            _thermostat_rotation,
            _thermostat_rotation_grex_phase_v2_20260830a,
        ),
        (
            "_thc_state_is_finite",
            _thc_state_is_finite,
            _thc_state_is_finite_grex_phase_v2_20260830a,
        ),
        ("_thc_proposal", _thc_proposal, _thc_proposal_grex_phase_v2_20260830a),
        (
            "_simulate_thc_chain",
            _simulate_thc_chain,
            _simulate_thc_chain_grex_phase_v2_20260830a,
        ),
    )
    problems: list[str] = []
    for public_name, active_dispatcher, expected_dispatcher in expected_dispatchers:
        if active_dispatcher is not expected_dispatcher:
            problems.append(f"{public_name} does not point to the current epoch")
            continue
        python_function = getattr(active_dispatcher, "py_func", None)
        if python_function is None or not python_function.__name__.endswith(
            "_grex_phase_v2_20260830a"
        ):
            problems.append(f"{public_name} has the wrong Python function identity")
        cache_object = getattr(active_dispatcher, "_cache", None)
        if (
            cache_object is not None
            and cache_object.__class__.__name__ != "NullCache"
        ):
            problems.append(f"{public_name} has persistent native caching enabled")

    if not getattr(_thc_proposal, "signatures", ()):
        problems.append("the THC proposal was not compiled by the self-tests")
    if not getattr(_simulate_thc_chain, "signatures", ()):
        problems.append("the complete THC chain was not compiled by the self-tests")

    if problems:
        raise RuntimeError(
            "Portable native-runtime preflight failed: " + "; ".join(problems)
        )

    _validate_compiled_reference_microcheck()
    return (
        "PASS: portable cache-free THC native preflight "
        f"({THC_RUNTIME_PREFLIGHT_VERSION}); {_runtime_environment_summary()}; "
        "compiled/reference micro-oracle=PASS; "
        "tuning-block-1 stream SHA-256="
        f"{_first_tuning_block_stream_digest(prepared)}"
    )


def run_self_tests(
    prepared: PreparedTHCInputs,
) -> tuple[str, ...]:
    """Run deterministic common and THC-specific checks before tuning."""

    common = prepared.common
    messages: list[str] = []
    dimension = common.dimension

    # Literal hand oracle prevents a mutually consistent target/gradient typo.
    canonical_parameters = np.asarray([4.0, 0.04, 0.25], dtype=np.float64)
    hand_point = np.asarray([1.0, -0.45], dtype=np.float64)
    hand_potential = _potential_energy(hand_point, canonical_parameters)
    hand_gradient = _target_gradient(hand_point, canonical_parameters)
    if (
        not math.isclose(
            hand_potential, 1.25, rel_tol=0.0, abs_tol=2.0e-14
        )
        or not np.allclose(
            hand_gradient,
            np.asarray([-3.5, 7.5]),
            rtol=0.0,
            atol=2.0e-13,
        )
    ):
        raise RuntimeError("Self-test failed: literal Banana target oracle.")

    q_gradient_test = np.asarray(
        [0.73 * common.sigma_x, -0.31 * math.sqrt(common.target_variances[1])],
        dtype=np.float64,
    )
    analytic_gradient = _target_gradient(
        q_gradient_test, common.target_parameters
    )
    finite_difference_gradient = np.empty(
        dimension, dtype=np.float64
    )
    for coordinate in range(dimension):
        finite_difference_step = (
            1.0e-6
            * max(
                1.0,
                abs(float(q_gradient_test[coordinate])),
                math.sqrt(common.target_variances[coordinate]),
            )
        )
        q_plus = q_gradient_test.copy()
        q_minus = q_gradient_test.copy()
        q_plus[coordinate] += finite_difference_step
        q_minus[coordinate] -= finite_difference_step
        finite_difference_gradient[coordinate] = (
            _potential_energy(
                q_plus, common.target_parameters
            )
            - _potential_energy(
                q_minus, common.target_parameters
            )
        ) / (2.0 * finite_difference_step)
    if not np.allclose(
        analytic_gradient,
        finite_difference_gradient,
        rtol=2.0e-8,
        atol=2.0e-8,
    ):
        raise RuntimeError(
            "Self-test failed: analytic Banana target gradient."
        )

    reconstructed_starts = _physical_to_latent(
        common.physical_starts,
        common.beta,
        common.sigma_x,
        common.sigma_y,
        common.banana_b,
    )
    if not np.allclose(
        reconstructed_starts,
        common.standardized_starts,
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise RuntimeError(
            "Self-test failed: Banana latent/physical start round trip."
        )
    if not np.allclose(
        common.mass_cholesky
        @ common.mass_cholesky.T,
        common.mass_matrix,
        rtol=1.0e-12,
        atol=1.0e-12,
    ) or not np.allclose(
        common.inverse_mass_matrix
        @ common.mass_matrix,
        np.eye(dimension),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Self-test failed: mass factorization or inverse."
        )

    canonical_names, canonical_means, canonical_sds = (
        _observable_specification(1.0, 2.0, 0.2, 0.25)
    )
    expected_means = np.asarray([0.0, 0.0, 4.0, 0.0, 2.04])
    expected_sds = np.sqrt(
        np.asarray([4.0, 2.04, 32.0, 40.16, 56.3232])
    )
    if (
        canonical_names != ("x", "y", "x^2", "x*y", "y^2")
        or not np.allclose(
            canonical_means, expected_means, rtol=0.0, atol=2.0e-14
        )
        or not np.allclose(
            canonical_sds, expected_sds, rtol=0.0, atol=2.0e-13
        )
    ):
        raise RuntimeError("Self-test failed: exact Banana moment constants.")

    # Eight-point Gauss-Hermite product quadrature independently integrates
    # every polynomial needed for means and second moments (maximum degree 8).
    nodes, weights = np.polynomial.hermite.hermgauss(8)
    normal_nodes = math.sqrt(2.0) * nodes
    normalized_weights = weights / math.sqrt(math.pi)
    quadrature_means = np.zeros(5, dtype=np.float64)
    quadrature_seconds = np.zeros(5, dtype=np.float64)
    for first in range(normal_nodes.size):
        x_value = 2.0 * normal_nodes[first]
        for second in range(normal_nodes.size):
            residual_value = 0.2 * normal_nodes[second]
            y_value = 0.25 * (x_value * x_value - 4.0) + residual_value
            values = np.asarray(
                [
                    x_value,
                    y_value,
                    x_value * x_value,
                    x_value * y_value,
                    y_value * y_value,
                ],
                dtype=np.float64,
            )
            weight = normalized_weights[first] * normalized_weights[second]
            quadrature_means += weight * values
            quadrature_seconds += weight * values * values
    quadrature_sds = np.sqrt(
        np.maximum(0.0, quadrature_seconds - quadrature_means**2)
    )
    if (
        not np.allclose(
            quadrature_means, expected_means, rtol=0.0, atol=2.0e-12
        )
        or not np.allclose(
            quadrature_sds, expected_sds, rtol=0.0, atol=3.0e-12
        )
    ):
        raise RuntimeError(
            "Self-test failed: independent Gauss-Hermite Banana moments."
        )
    messages.append(
        "PASS: literal Banana oracle, analytic gradient, latent round trip, "
        "mass algebra, and independent exact physical moments"
    )

    autocovariance_input = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ]
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
        raise RuntimeError(
            "Self-test failed: biased FFT autocovariance."
        )
    odd_split_input = np.asarray(
        [
            [0.0, 1.0, 99.0, 2.0, 3.0],
            [10.0, 11.0, 99.0, 12.0, 13.0],
        ]
    )
    expected_split = np.asarray(
        [
            [0.0, 1.0],
            [10.0, 11.0],
            [2.0, 3.0],
            [12.0, 13.0],
        ]
    )
    if not np.array_equal(
        _split_chains(odd_split_input),
        expected_split,
    ):
        raise RuntimeError(
            "Self-test failed: odd-length split convention."
        )
    expected_ranks = np.asarray(
        [1.5, 1.5, 3.0, 4.0]
    )
    if not np.array_equal(
        _average_ranks(
            np.asarray([10.0, 10.0, 20.0, 30.0])
        ),
        expected_ranks,
    ):
        raise RuntimeError(
            "Self-test failed: average ranks for ties."
        )
    ppf_values = _standard_normal_ppf(
        np.asarray([0.025, 0.5, 0.975])
    )
    if not np.allclose(
        ppf_values,
        np.asarray(
            [
                -1.959963984540054,
                0.0,
                1.959963984540054,
            ]
        ),
        rtol=0.0,
        atol=2.0e-7,
    ):
        raise RuntimeError(
            "Self-test failed: inverse-normal approximation."
        )
    messages.append(
        "PASS: biased autocovariance, chain splitting, ranks, and inverse normal"
    )

    rng = np.random.default_rng(24681357)
    iid = rng.standard_normal((4, 6000))
    iid_rhat = _rank_normalized_split_folded_rhat(
        iid
    )
    if (
        iid_rhat.status != "ok"
        or not (0.98 < iid_rhat.combined < 1.03)
    ):
        raise RuntimeError(
            "Self-test failed: IID rank-normalized R-hat."
        )
    shifted = iid.copy()
    shifted[0] += 2.0
    shifted_rhat = (
        _rank_normalized_split_folded_rhat(shifted)
    )
    if (
        shifted_rhat.status != "ok"
        or shifted_rhat.combined <= 1.1
    ):
        raise RuntimeError(
            "Self-test failed: shifted-chain R-hat sensitivity."
        )
    unequal_scale = iid * np.asarray(
        [1.0, 1.0, 4.0, 4.0]
    )[:, np.newaxis]
    scale_rhat = (
        _rank_normalized_split_folded_rhat(
            unequal_scale
        )
    )
    if (
        scale_rhat.status != "ok"
        or scale_rhat.folded
        <= scale_rhat.rank + 0.05
    ):
        raise RuntimeError(
            "Self-test failed: folded R-hat scale sensitivity."
        )
    messages.append(
        "PASS: rank-normalized split/folded R-hat IID, location, and scale checks"
    )

    phi = 0.8
    ar = np.empty((4, 20000), dtype=np.float64)
    ar[:, 0] = rng.normal(
        scale=1.0 / math.sqrt(1.0 - phi * phi),
        size=4,
    )
    innovations = rng.standard_normal(
        (4, ar.shape[1] - 1)
    )
    for draw in range(1, ar.shape[1]):
        ar[:, draw] = (
            phi * ar[:, draw - 1]
            + innovations[:, draw - 1]
        )
    ar_iat = _multi_chain_geyer_iat(ar)
    if (
        not ar_iat.is_defined
        or not (6.0 < ar_iat.tau < 12.5)
    ):
        raise RuntimeError(
            "Self-test failed: AR(1) Geyer IAT."
        )
    constant_iat = _multi_chain_geyer_iat(
        np.ones((4, 100), dtype=np.float64)
    )
    if constant_iat.status != "undefined_zero_variance":
        raise RuntimeError(
            "Self-test failed: constant-series IAT status."
        )
    nonfinite_iat = _multi_chain_geyer_iat(
        np.asarray(
            [[0.0, 1.0, math.nan, 2.0]] * 4
        )
    )
    if (
        nonfinite_iat.status
        != "undefined_nonfinite_input"
    ):
        raise RuntimeError(
            "Self-test failed: nonfinite-series IAT status."
        )
    negative_phi = -0.5
    antithetic = np.empty(
        (4, 20000), dtype=np.float64
    )
    antithetic[:, 0] = rng.normal(
        scale=1.0
        / math.sqrt(
            1.0 - negative_phi * negative_phi
        ),
        size=4,
    )
    antithetic_innovations = (
        rng.standard_normal(
            (4, antithetic.shape[1] - 1)
        )
    )
    for draw in range(1, antithetic.shape[1]):
        antithetic[:, draw] = (
            negative_phi * antithetic[:, draw - 1]
            + antithetic_innovations[:, draw - 1]
        )
    antithetic_iat = _multi_chain_geyer_iat(
        antithetic
    )
    if (
        not antithetic_iat.is_defined
        or not (0.15 < antithetic_iat.tau < 0.65)
    ):
        raise RuntimeError(
            "Self-test failed: unfloored antithetic Geyer IAT."
        )
    messages.append(
        (
            "PASS: Geyer IAT positive AR(1)="
            f"{ar_iat.tau:.6f}, antithetic AR(1)="
            f"{antithetic_iat.tau:.6f}, and undefined statuses"
        )
    )

    test_a = prepared.a_values[0]
    coupling_s = 0.37
    coupling_step = 1.0e-6
    test_codes = (
        _encode_coupling_code(prepared.alpha_values[0], 0.0),
        _encode_coupling_code(prepared.alpha_values[0], 0.5 * math.pi),
    )
    for test_coupling_code in test_codes:
        alpha, _phi, _phase = _decode_coupling_code(test_coupling_code)
        log_h, analytic_log_h_derivative = _log_h_and_derivative(
            coupling_s, test_coupling_code, test_a
        )
        log_h_plus, _ = _log_h_and_derivative(
            coupling_s + coupling_step,
            test_coupling_code,
            test_a,
        )
        log_h_minus, _ = _log_h_and_derivative(
            coupling_s - coupling_step,
            test_coupling_code,
            test_a,
        )
        finite_difference_log_h_derivative = (
            log_h_plus - log_h_minus
        ) / (2.0 * coupling_step)
        h = math.exp(log_h)
        if (
            not math.isclose(
                analytic_log_h_derivative,
                finite_difference_log_h_derivative,
                rel_tol=2.0e-9,
                abs_tol=2.0e-11,
            )
            or h < 1.0 - alpha
            or h > 1.0 + alpha
        ):
            raise RuntimeError(
                "Self-test failed: phase-periodic coupling bound or derivative."
            )
    messages.append(
        "PASS: sine/cosine phase-periodic coupling bounds and analytic logarithmic derivatives"
    )

    invalid_coupling_inputs = (
        (math.inf, test_codes[0], test_a),
        (0.1, math.inf, test_a),
        (0.1, test_codes[0], 0.0),
        (0.1, 1.0, test_a),
    )
    for invalid_s, invalid_code, invalid_a in invalid_coupling_inputs:
        native_invalid = _log_h_and_derivative(
            invalid_s, invalid_code, invalid_a
        )
        reference_invalid = _reference_log_h_and_derivative(
            invalid_s, invalid_code, invalid_a
        )
        if not (
            math.isnan(native_invalid[0])
            and math.isnan(native_invalid[1])
            and math.isnan(reference_invalid[0])
            and math.isnan(reference_invalid[1])
        ):
            raise RuntimeError(
                "Self-test failed: nonfinite phase coupling guard."
            )
    messages.append(
        "PASS: invalid/nonfinite phase inputs return NaN for safe proposal rejection"
    )

    test_coupling_code = test_codes[0]

    q0 = np.asarray([0.8, -0.7], dtype=np.float64)
    standard_p0 = np.asarray([0.7, -0.3], dtype=np.float64)
    p0 = (
        common.mass_cholesky @ standard_p0
        / math.sqrt(common.beta)
    )
    s0 = 0.31
    p_s0 = -0.27
    # Fixed conservative local step: the Banana Hessian varies with position.
    test_epsilon = 0.001
    grad0 = _target_gradient(
        q0, common.target_parameters
    )
    (
        q1,
        p1,
        s1,
        p_s1,
        grad1,
        calls1,
        finite1,
    ) = _thc_proposal(
        q0,
        p0,
        s0,
        p_s0,
        grad0,
        common.beta,
        test_coupling_code,
        test_a,
        prepared.Q_values[0],
        prepared.thermostat_sigma,
        test_epsilon,
        17,
        common.target_parameters,
        common.inverse_mass_matrix,
        common.mass_is_identity,
    )
    (
        q2,
        p2,
        s2,
        p_s2,
        _grad2,
        calls2,
        finite2,
    ) = _thc_proposal(
        q1,
        -p1,
        s1,
        -p_s1,
        grad1,
        common.beta,
        test_coupling_code,
        test_a,
        prepared.Q_values[0],
        prepared.thermostat_sigma,
        test_epsilon,
        17,
        common.target_parameters,
        common.inverse_mass_matrix,
        common.mass_is_identity,
    )
    if (
        not finite1
        or not finite2
        or calls1 != 17
        or calls2 != 17
        or not np.allclose(
            q2, q0, rtol=0.0, atol=4.0e-11
        )
        or not np.allclose(
            p2, -p0, rtol=0.0, atol=4.0e-11
        )
        or not math.isclose(
            s2, s0, rel_tol=0.0, abs_tol=4.0e-11
        )
        or not math.isclose(
            p_s2,
            -p_s0,
            rel_tol=0.0,
            abs_tol=4.0e-11,
        )
    ):
        raise RuntimeError(
            "Self-test failed: THC reversibility or gradient count."
        )

    general_mass = np.asarray(
        [[2.0, 0.3], [0.3, 0.8]],
        dtype=np.float64,
    )
    general_inverse = np.linalg.inv(general_mass)
    general_q0 = np.asarray([0.2, -0.1])
    general_p0 = np.asarray([0.4, -0.25])
    general_s0 = -0.2
    general_p_s0 = 0.15
    general_target_parameters = np.asarray(
        [2.25, 0.09, 0.15], dtype=np.float64
    )
    general_grad0 = _target_gradient(
        general_q0, general_target_parameters
    )
    (
        gq1,
        gp1,
        gs1,
        gps1,
        gg1,
        gcalls1,
        gfinite1,
    ) = _thc_proposal(
        general_q0,
        general_p0,
        general_s0,
        general_p_s0,
        general_grad0,
        1.3,
        0.2,
        1.1,
        0.7,
        1.3,
        0.002,
        23,
        general_target_parameters,
        general_inverse,
        False,
    )
    (
        gq2,
        gp2,
        gs2,
        gps2,
        _gg2,
        gcalls2,
        gfinite2,
    ) = _thc_proposal(
        gq1,
        -gp1,
        gs1,
        -gps1,
        gg1,
        1.3,
        0.2,
        1.1,
        0.7,
        1.3,
        0.002,
        23,
        general_target_parameters,
        general_inverse,
        False,
    )
    if (
        not gfinite1
        or not gfinite2
        or gcalls1 != 23
        or gcalls2 != 23
        or not np.allclose(
            gq2,
            general_q0,
            rtol=0.0,
            atol=3.0e-12,
        )
        or not np.allclose(
            gp2,
            -general_p0,
            rtol=0.0,
            atol=3.0e-12,
        )
        or not math.isclose(
            gs2,
            general_s0,
            rel_tol=0.0,
            abs_tol=3.0e-12,
        )
        or not math.isclose(
            gps2,
            -general_p_s0,
            rel_tol=0.0,
            abs_tol=3.0e-12,
        )
    ):
        raise RuntimeError(
            "Self-test failed: general-mass THC reversibility."
        )
    messages.append(
        "PASS: identity/current and general-mass THC reversibility with exact target-gradient counts"
    )

    # Independent local checks of volume preservation and second-order error.
    jacobian_state = np.asarray(
        [0.2, -0.12, 0.35, -0.28, 0.17, -0.11],
        dtype=np.float64,
    )

    def general_flow(state: np.ndarray, step_size: float, steps: int) -> np.ndarray:
        flow_q = state[:2].copy()
        flow_p = state[2:4].copy()
        flow_s = float(state[4])
        flow_p_s = float(state[5])
        flow_gradient = _target_gradient(flow_q, general_target_parameters)
        flow_output = _thc_proposal(
            flow_q,
            flow_p,
            flow_s,
            flow_p_s,
            flow_gradient,
            1.2,
            math.log(1.25),
            1.1,
            0.7,
            0.9,
            step_size,
            steps,
            general_target_parameters,
            general_inverse,
            False,
        )
        if not flow_output[-1]:
            raise RuntimeError("Self-test failed: nonfinite local THC flow.")
        return np.asarray(
            [
                flow_output[0][0],
                flow_output[0][1],
                flow_output[1][0],
                flow_output[1][1],
                flow_output[2],
                flow_output[3],
            ]
        )

    jacobian_step = 1.0e-6
    numerical_jacobian = np.empty((6, 6), dtype=np.float64)
    for column in range(6):
        plus = jacobian_state.copy()
        minus = jacobian_state.copy()
        plus[column] += jacobian_step
        minus[column] -= jacobian_step
        numerical_jacobian[:, column] = (
            general_flow(plus, 0.004, 3)
            - general_flow(minus, 0.004, 3)
        ) / (2.0 * jacobian_step)
    jacobian_determinant = float(np.linalg.det(numerical_jacobian))
    if not math.isclose(
        jacobian_determinant,
        1.0,
        rel_tol=0.0,
        abs_tol=3.0e-7,
    ):
        raise RuntimeError("Self-test failed: THC proposal volume preservation.")

    def general_hamiltonian(state: np.ndarray) -> float:
        return _thc_hamiltonian(
            state[:2],
            state[2:4],
            float(state[4]),
            float(state[5]),
            1.2,
            math.log(1.25),
            1.1,
            0.7,
            0.9,
            general_target_parameters,
            general_inverse,
            False,
        )

    initial_local_energy = general_hamiltonian(jacobian_state)
    one_step_error = abs(
        general_hamiltonian(general_flow(jacobian_state, 0.04, 1))
        - initial_local_energy
    )
    two_step_error = abs(
        general_hamiltonian(general_flow(jacobian_state, 0.02, 2))
        - initial_local_energy
    )
    energy_error_ratio = one_step_error / two_step_error
    if not (3.5 < energy_error_ratio < 4.5):
        raise RuntimeError("Self-test failed: second-order THC energy error.")
    messages.append(
        "PASS: finite-difference volume preservation and second-order fixed-time energy error"
    )

    # At alpha=0, the physical P/K dynamics reduce exactly to HMC leapfrog.
    control_q = q0.copy()
    control_p = p0.copy()
    control_gradient = grad0.copy()
    control_p -= 0.5 * test_epsilon * control_gradient
    for step in range(9):
        if common.mass_is_identity:
            control_q += test_epsilon * control_p
        else:
            control_q += (
                test_epsilon
                * (
                    common.inverse_mass_matrix
                    @ control_p
                )
            )
        control_gradient = _target_gradient(
            control_q, common.target_parameters
        )
        control_p -= (
            (0.5 * test_epsilon if step == 8 else test_epsilon)
            * control_gradient
        )
    (
        zero_q,
        zero_p,
        _zero_s,
        _zero_p_s,
        _zero_gradient,
        zero_calls,
        zero_finite,
    ) = _thc_proposal(
        q0,
        p0,
        s0,
        p_s0,
        grad0,
        common.beta,
        0.0,
        test_a,
        prepared.Q_values[0],
        prepared.thermostat_sigma,
        test_epsilon,
        9,
        common.target_parameters,
        common.inverse_mass_matrix,
        common.mass_is_identity,
    )
    if (
        not zero_finite
        or zero_calls != 9
        or not np.allclose(
            zero_q,
            control_q,
            rtol=0.0,
            atol=2.0e-13,
        )
        or not np.allclose(
            zero_p,
            control_p,
            rtol=0.0,
            atol=2.0e-13,
        )
    ):
        raise RuntimeError(
            "Self-test failed: alpha=0 HMC physical-control parity."
        )
    messages.append(
        "PASS: alpha=0 physical proposal matches cached general-mass HMC leapfrog"
    )

    momenta, thermostat_momenta, uniforms = (
        _make_random_streams(
            998877, 5, dimension
        )
    )
    (
        repeated_momenta,
        repeated_thermostat_momenta,
        repeated_uniforms,
    ) = _make_random_streams(
        998877, 5, dimension
    )
    if (
        not np.array_equal(
            momenta, repeated_momenta
        )
        or not np.array_equal(
            thermostat_momenta,
            repeated_thermostat_momenta,
        )
        or not np.array_equal(
            uniforms, repeated_uniforms
        )
    ):
        raise RuntimeError(
            "Self-test failed: deterministic random substreams."
        )
    output = _simulate_thc_chain(
        momenta,
        thermostat_momenta,
        uniforms,
        common.physical_starts[0],
        prepared.thermostat_starts[0],
        common.beta,
        test_coupling_code,
        test_a,
        prepared.Q_values[0],
        prepared.thermostat_sigma,
        test_epsilon,
        4,
        2,
        3,
        common.target_parameters,
        common.mass_cholesky,
        common.inverse_mass_matrix,
        common.mass_is_identity,
    )
    if (
        output[0].shape != (3, dimension)
        or output[1].shape != (3,)
        or output[5] != 9
        or output[6] != 12
        or output[3] != 0
        or output[4] != 0
    ):
        raise RuntimeError(
            "Self-test failed: THC chain shape or gradient accounting."
        )
    messages.append(
        "PASS: paired three-substream RNG, warmup/retained gradient accounting, and finite THC smoke chain"
    )

    expected_observable_count = (
        dimension
        + dimension * (dimension + 1) // 2
    )
    if (
        len(common.observable_names)
        != expected_observable_count
        or len(set(common.observable_names))
        != expected_observable_count
    ):
        raise RuntimeError(
            "Self-test failed: required observable set."
        )
    latent_synthetic = rng.standard_normal(
        (common.n_chains, 120, 2)
    )
    synthetic_x = (
        common.sigma_x
        * latent_synthetic[:, :, 0]
        / math.sqrt(common.beta)
    )
    synthetic_r = (
        common.sigma_y
        * latent_synthetic[:, :, 1]
        / math.sqrt(common.beta)
    )
    synthetic_y = (
        common.banana_b
        * (
            synthetic_x * synthetic_x
            - common.sigma_x * common.sigma_x
        )
        + synthetic_r
    )
    synthetic_samples = np.stack(
        (synthetic_x, synthetic_y), axis=2
    )
    restored_block = _analyze_block(
        samples=synthetic_samples,
        epsilon=0.01,
        L=15,
        block_index=0,
        seeds=common.seed_blocks[0],
        accepted_retained=common.n_chains * 100,
        retained_gradient_calls=(
            common.n_chains * 120 * 15
        ),
        warmup_nonfinite=1,
        retained_nonfinite=0,
        observable_names=common.observable_names,
        exact_observable_means=(
            common.exact_observable_means
        ),
        exact_observable_sds=(
            common.exact_observable_sds
        ),
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
        seeds=common.seed_blocks[0],
        accepted_retained=common.n_chains * 100,
        retained_gradient_calls=(
            common.n_chains * 120 * 15
        ),
        warmup_nonfinite=1,
        retained_nonfinite=0,
        observable_names=common.observable_names,
        exact_observable_means=(
            common.exact_observable_means
        ),
        exact_observable_sds=(
            common.exact_observable_sds
        ),
    )
    if (
        invalid_block.status
        != "undefined_nonfinite_simulation"
        or any(
            math.isfinite(item.efficiency)
            for item in invalid_block.observable_diagnostics
        )
    ):
        raise RuntimeError(
            "Self-test failed: nonfinite block efficiency propagation."
        )
    valid_block_a = replace(
        invalid_block,
        status="ok",
        block_index=0,
        block_efficiency=0.25,
        warmup_nonfinite_trajectories=0,
    )
    valid_block_b = replace(
        valid_block_a,
        block_index=1,
        block_efficiency=1.0,
    )
    valid_aggregate = _aggregate_blocks(
        (valid_block_a, valid_block_b)
    )
    invalid_aggregate = _aggregate_blocks(
        (invalid_block,)
    )
    if (
        valid_aggregate.status != "ok"
        or not math.isclose(
            valid_aggregate.geometric_mean_efficiency,
            0.5,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or invalid_aggregate.status
        != "undefined_block_metric"
        or not math.isnan(
            invalid_aggregate.geometric_mean_efficiency
        )
    ):
        raise RuntimeError(
            "Self-test failed: common block aggregation."
        )
    thermostat_check = _thermostat_diagnostic(
        np.asarray(
            [
                -prepared.thermostat_sigma,
                prepared.thermostat_sigma,
            ]
        ),
        prepared.thermostat_sigma,
    )
    if (
        thermostat_check.empirical_mean != 0.0
        or not math.isclose(
            thermostat_check.empirical_second_moment,
            prepared.thermostat_sigma**2,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise RuntimeError(
            "Self-test failed: thermostat exact-moment diagnostic."
        )
    expected_method_tuples = (
        len(prepared.phase_codes)
        * len(prepared.a_values)
        * len(prepared.Q_values)
    )
    expected_stage1_evaluations = (
        expected_method_tuples * len(prepared.stage1_anchor_pairs)
    )
    expected_stage2_evaluations = (
        prepared.stage1_keep_method_tuples
        * len(common.epsilon_values)
        * len(common.L_values)
    )
    if expected_method_tuples <= 0:
        raise RuntimeError(
            "Self-test failed: phase-periodic method-grid cardinality."
        )
    messages.append(
        (
            f"PASS: {expected_observable_count}-observable exact moments, "
            "nonfinite propagation, common GM, thermostat moments, and "
            f"successive-halving cardinalities (method tuples="
            f"{expected_method_tuples}, stage-1 evaluations="
            f"{expected_stage1_evaluations}, maximum base stage-2 "
            f"evaluations={expected_stage2_evaluations})"
        )
    )
    return tuple(messages)


def run_phase_periodic_tuning(
    inputs: THCRunInputs,
    *,
    preflight_only: bool = False,
) -> dict[str, object]:
    """Run the frozen three-stage phase-periodic Banana tuning campaign."""
    if not preflight_only:
        raise RuntimeError("Use launcher/checkpoint_runner.py for the expanded full-grid production run.")


    if not isinstance(preflight_only, bool):
        raise TypeError("preflight_only must be exactly True or False.")
    if preflight_only and inputs.run_self_tests is not True:
        raise ValueError("preflight_only=True requires run_self_tests=True.")

    prepared = _prepare_thc_inputs(inputs)
    common = prepared.common
    _validate_thc_run_mode(prepared)
    _validate_common_parity_seals(common)
    if common.run_mode == "CANONICAL":
        design_mismatches = _canonical_halving_design_mismatches(prepared)
        if design_mismatches:
            raise ValueError(
                "CANONICAL successive-halving design mismatch in: "
                + ", ".join(design_mismatches)
            )

    self_test_messages = (
        run_self_tests(prepared)
        if common.run_self_tests
        else ("SKIPPED: run_self_tests=False (CUSTOM mode only).",)
    )
    runtime_preflight_message = (
        _validate_portable_native_runtime(prepared)
        if common.run_self_tests
        else "SKIPPED: portable native preflight requires run_self_tests=True."
    )
    self_test_messages = (*self_test_messages, runtime_preflight_message)
    if common.print_progress:
        print(runtime_preflight_message, flush=True)

    if preflight_only:
        print(
            "GREX PREFLIGHT COMPLETE: target and gradient oracles, exact "
            "phase-periodic coupling derivatives, extended Hamiltonian, "
            "P/S/K/S/P reversibility, Metropolis chain, RNG, Geyer IAT, "
            "R-hat, score propagation, and live cache-free native dispatch "
            "all passed. No tuning configuration was simulated and no "
            "report was written. Use launcher/checkpoint_runner.py for the "
            "expanded full-grid production run.",
            flush=True,
        )
        return {
            "preflight_only": True,
            "prepared_inputs": prepared,
            "self_test_messages": self_test_messages,
            "whole_source_sha256": _whole_source_sha256(),
            "common_input_fingerprint": common.common_input_fingerprint,
            "thc_method_input_fingerprint": prepared.method_input_fingerprint,
            "input_preparation_fingerprint": _input_preparation_fingerprint(),
            "diagnostics_fingerprint": _diagnostics_fingerprint(),
            "common_pipeline_fingerprint": _common_pipeline_fingerprint(),
            "thc_implementation_fingerprint": _thc_implementation_fingerprint(),
            "shared_protocol_fingerprint": _shared_protocol_fingerprint(common),
        }

    destination = _resolve_output_destination(common)
    total_iterations = common.n_warmup + common.n_retained
    stream_caches: list[
        list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = [
        [
            _make_random_streams(seed, total_iterations, common.dimension)
            for seed in seed_block
        ]
        for seed_block in common.seed_blocks
    ]
    auxiliary_momentum_moments = tuple(
        _retained_standardized_momentum_second_moment(
            cache, common.n_warmup
        )
        for cache in stream_caches
    )

    # Compile the complete native sampler before the first tuning candidate.
    compile_momenta, compile_ps, compile_uniforms = _make_random_streams(
        13579, 2, common.dimension
    )
    _simulate_thc_chain(
        compile_momenta,
        compile_ps,
        compile_uniforms,
        common.physical_starts[0],
        prepared.thermostat_starts[0],
        common.beta,
        prepared.phase_codes[0],
        prepared.a_values[0],
        prepared.Q_values[0],
        prepared.thermostat_sigma,
        common.epsilon_values[0],
        1,
        1,
        1,
        common.target_parameters,
        common.mass_cholesky,
        common.inverse_mass_matrix,
        common.mass_is_identity,
    )

    method_tuples = tuple(
        MethodTuple(coupling_code, a, Q)
        for coupling_code in prepared.phase_codes
        for a in prepared.a_values
        for Q in prepared.Q_values
    )
    stage1_results: list[THCConfigurationResult] = []
    stage1_best_by_method: list[
        tuple[MethodTuple, THCConfigurationResult]
    ] = []
    for method_index, method in enumerate(method_tuples, start=1):
        method_anchor_results: list[THCConfigurationResult] = []
        if common.print_progress:
            print(
                f"[stage 1 method {method_index:03d}/{len(method_tuples)}] "
                + _phase_method_text(method.coupling_code, method.a, method.Q),
                flush=True,
            )
        for epsilon, L in prepared.stage1_anchor_pairs:
            result = _evaluate_fixed_configuration(
                FixedConfiguration(
                    method.coupling_code,
                    method.a,
                    method.Q,
                    epsilon,
                    L,
                ),
                block_indices=prepared.stage1_block_indices,
                n_warmup=prepared.stage1_n_warmup,
                n_retained=prepared.stage1_n_retained,
                origin="BASE_STAGE1_ANCHOR",
                stream_caches=stream_caches,
                prepared=prepared,
            )
            stage1_results.append(result)
            method_anchor_results.append(result)
        defined = _rank_defined_configurations(
            method_anchor_results, len(method_anchor_results)
        )
        if defined:
            stage1_best_by_method.append((method, defined[0]))

    stage1_best_by_method.sort(
        key=lambda item: _thc_configuration_rank_key(item[1]),
        reverse=True,
    )
    promoted_method_records = stage1_best_by_method[
        : min(prepared.stage1_keep_method_tuples, len(stage1_best_by_method))
    ]
    if not promoted_method_records:
        raise RuntimeError(
            "Stage 1 produced no method tuple with a defined anchor score."
        )
    stage1_promoted_methods = tuple(
        item[0] for item in promoted_method_records
    )

    stage2_results: list[THCConfigurationResult] = []
    stage2_total = (
        len(stage1_promoted_methods)
        * len(common.epsilon_values)
        * len(common.L_values)
    )
    stage2_index = 0
    for method in stage1_promoted_methods:
        for epsilon in common.epsilon_values:
            for L in common.L_values:
                stage2_index += 1
                if common.print_progress and (
                    stage2_index == 1 or stage2_index % 12 == 0
                ):
                    print(
                        f"[stage 2 {stage2_index:03d}/{stage2_total}]",
                        flush=True,
                    )
                stage2_results.append(
                    _evaluate_fixed_configuration(
                        FixedConfiguration(
                            method.coupling_code,
                            method.a,
                            method.Q,
                            epsilon,
                            L,
                        ),
                        block_indices=prepared.stage2_block_indices,
                        n_warmup=prepared.stage2_n_warmup,
                        n_retained=prepared.stage2_n_retained,
                        origin="BASE_STAGE2_FULL_COMMON_GRID",
                        stream_caches=stream_caches,
                        prepared=prepared,
                    )
                )
    stage2_promoted = _rank_defined_configurations(
        stage2_results, prepared.stage2_keep_fixed_configurations
    )
    if not stage2_promoted:
        raise RuntimeError(
            "Stage 2 produced no fixed configuration with a defined score."
        )

    base_finalists: list[THCConfigurationResult] = []
    for index, stage2_result in enumerate(stage2_promoted, start=1):
        if common.print_progress:
            print(
                f"[stage 3 base {index:02d}/{len(stage2_promoted)}]",
                flush=True,
            )
        base_finalists.append(
            _evaluate_fixed_configuration(
                _fixed_configuration_from_result(stage2_result),
                block_indices=tuple(range(len(common.seed_blocks))),
                n_warmup=common.n_warmup,
                n_retained=common.n_retained,
                origin="BASE_STAGE3_FINALIST",
                stream_caches=stream_caches,
                prepared=prepared,
            )
        )
    base_winner = _select_thc_representative(base_finalists)

    boundary_methods: tuple[MethodTuple, ...] = ()
    boundary_labels: tuple[str, ...] = ()
    if prepared.automatic_boundary_expansion:
        boundary_methods, boundary_labels = _boundary_method_layer(
            base_winner, prepared
        )
    boundary_stage2_results: list[THCConfigurationResult] = []
    boundary_origin = (
        "BOUNDARY_LAYER[" + ",".join(boundary_labels) + "]"
        if boundary_labels
        else "BOUNDARY_LAYER_NOT_TRIGGERED"
    )
    boundary_total = (
        len(boundary_methods)
        * len(common.epsilon_values)
        * len(common.L_values)
    )
    boundary_index = 0
    for method in boundary_methods:
        for epsilon in common.epsilon_values:
            for L in common.L_values:
                boundary_index += 1
                if common.print_progress and (
                    boundary_index == 1 or boundary_index % 12 == 0
                ):
                    print(
                        f"[boundary stage 2 {boundary_index:03d}/"
                        f"{boundary_total}]",
                        flush=True,
                    )
                boundary_stage2_results.append(
                    _evaluate_fixed_configuration(
                        FixedConfiguration(
                            method.coupling_code,
                            method.a,
                            method.Q,
                            epsilon,
                            L,
                        ),
                        block_indices=prepared.stage2_block_indices,
                        n_warmup=prepared.stage2_n_warmup,
                        n_retained=prepared.stage2_n_retained,
                        origin=boundary_origin + ":STAGE2",
                        stream_caches=stream_caches,
                        prepared=prepared,
                    )
                )
    boundary_stage2_promoted = _rank_defined_configurations(
        boundary_stage2_results,
        prepared.boundary_keep_fixed_configurations,
    )
    boundary_finalists: list[THCConfigurationResult] = []
    for index, stage2_result in enumerate(
        boundary_stage2_promoted, start=1
    ):
        if common.print_progress:
            print(
                f"[boundary stage 3 {index:02d}/"
                f"{len(boundary_stage2_promoted)}]",
                flush=True,
            )
        boundary_finalists.append(
            _evaluate_fixed_configuration(
                _fixed_configuration_from_result(stage2_result),
                block_indices=tuple(range(len(common.seed_blocks))),
                n_warmup=common.n_warmup,
                n_retained=common.n_retained,
                origin=boundary_origin + ":STAGE3_FINALIST",
                stream_caches=stream_caches,
                prepared=prepared,
            )
        )

    representative = _select_thc_representative(
        (*base_finalists, *boundary_finalists)
    )
    report = _build_phase_periodic_report(
        prepared=prepared,
        self_test_messages=self_test_messages,
        auxiliary_momentum_moments=auxiliary_momentum_moments,
        stage1_results=stage1_results,
        stage1_promoted_methods=stage1_promoted_methods,
        stage2_results=stage2_results,
        stage2_promoted=stage2_promoted,
        base_finalists=base_finalists,
        boundary_labels=boundary_labels,
        boundary_stage2_results=boundary_stage2_results,
        boundary_stage2_promoted=boundary_stage2_promoted,
        boundary_finalists=boundary_finalists,
        representative=representative,
    )
    destination.write_text(report + "\n", encoding="utf-8")
    if common.print_report:
        print("\n" + report)
        print(f"\nThe identical report was saved to:\n{destination}")
    return {
        "preflight_only": False,
        "stage1_results": tuple(stage1_results),
        "stage1_promoted_methods": stage1_promoted_methods,
        "stage2_results": tuple(stage2_results),
        "stage2_promoted": stage2_promoted,
        "base_finalists": tuple(base_finalists),
        "boundary_trigger_labels": boundary_labels,
        "boundary_stage2_results": tuple(boundary_stage2_results),
        "boundary_stage2_promoted": boundary_stage2_promoted,
        "boundary_finalists": tuple(boundary_finalists),
        "representative": representative,
        "report": report,
        "output_path": destination,
        "prepared_inputs": prepared,
        "whole_source_sha256": _whole_source_sha256(),
        "common_input_fingerprint": common.common_input_fingerprint,
        "thc_method_input_fingerprint": prepared.method_input_fingerprint,
        "input_preparation_fingerprint": _input_preparation_fingerprint(),
        "diagnostics_fingerprint": _diagnostics_fingerprint(),
        "common_pipeline_fingerprint": _common_pipeline_fingerprint(),
        "thc_implementation_fingerprint": _thc_implementation_fingerprint(),
        "shared_protocol_fingerprint": _shared_protocol_fingerprint(common),
    }


# Tuning inputs.

GREX_PREFLIGHT_ONLY = True

USER_INPUTS = THCRunInputs(
    # ----- Banana target and shared physical geometry (identical to HMC) -----
    beta=1.0,
    sigma_x=2.0,
    sigma_y=0.2,
    banana_b=0.25,
    # M0=Cov(x,y)^(-1)=diag(1/4,1/2.04) at the canonical target.
    mass_matrix=np.asarray(
        [
            [0.25, 0.0],
            [0.0, 0.49019607843137253],
        ],
        dtype=np.float64,
    ),

    # Coordinates are (z_x,z_r)=(sqrt(beta)*x/sigma_x,
    # sqrt(beta)*r/sigma_y).  Physical (x,y) starts are derived.
    standardized_starts=np.asarray(
        [
            [-2.0, -2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    ),
    observable_policy="ALL_LINEAR_AND_UNIQUE_QUADRATIC",

    # ----- Common HMC/THC epsilon-L tuning grid -----
    epsilon_values=(
        0.005,
        0.010,
        0.020,
        0.030,
        0.040,
        0.050,
        0.060,
        0.080,
    ),
    L_values=(10, 20, 40, 80, 120, 160),

    # ----- Common tuning seed blocks; one seed per chain -----
    seed_blocks=(
        (123, 456, 789, 101112),
        (
            3545294353,
            3961674632,
            3627514536,
            3198582250,
        ),
        (
            788645248,
            3651519711,
            1811141324,
            1814869264,
        ),
        (
            3192285063,
            2251015777,
            396444798,
            765150135,
        ),
        (
            1562880678,
            1785952873,
            2908883741,
            2216662531,
        ),
    ),

    # ----- Phase-periodic THC-v2 method grid -----
    coupling_family="PHASE_PERIODIC_COS_SHIFT",
    # h(s)=1+alpha*cos(s/a-phi); phi=0 is cosine, phi=pi/2 is sine.
    alpha_values=(0.10, 0.30, 0.60, 0.90),
    phi_values=(0.0, 0.5 * math.pi),
    a_values=(0.50, 0.75, 1.0, 1.5, 2.0),
    Q_values=(0.10, 0.25, 0.50, 1.0, 2.0, 4.0),
    thermostat_sigma=1.0,
    # s persists; it is not independently refreshed between trajectories.
    thermostat_starts=(-1.0, 1.0, 1.0, -1.0),

    # ----- Predeclared successive halving -----
    stage1_anchor_pairs=(
        (0.02, 120),
        (0.04, 80),
        (0.05, 40),
        (0.06, 20),
    ),
    # Block numbers entered here are one-based for readability.
    stage1_block_indices=(1,),
    stage1_n_warmup=250,
    stage1_n_retained=1000,
    stage1_keep_method_tuples=12,
    stage2_block_indices=(2, 3),
    stage2_n_warmup=500,
    stage2_n_retained=2000,
    stage2_keep_fixed_configurations=12,

    # One outward method-boundary layer; never expand epsilon/L or twice.
    automatic_boundary_expansion=True,
    boundary_alpha_lower=0.05,
    boundary_alpha_upper=0.95,
    boundary_a_lower=0.375,
    boundary_a_upper=3.0,
    boundary_Q_lower=0.05,
    boundary_Q_upper=8.0,
    boundary_keep_fixed_configurations=4,

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
    # None -> beside this file for scripts/%run/import; pasted Jupyter -> cwd.
    output_directory=None,
    output_filename="THC_phase_periodic_v2_banana_tuning_results_GREX_PORTABLE_V1.txt",
    allow_overwrite=True,
)


# Production tuning settings.
from expanded_protocol import expanded_inputs
USER_INPUTS = expanded_inputs(USER_INPUTS, "periodic")

if __name__ == "__main__":
    run_phase_periodic_tuning(
        USER_INPUTS,
        preflight_only=GREX_PREFLIGHT_ONLY,
    )

