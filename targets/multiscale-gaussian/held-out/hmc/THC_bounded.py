"""Independent frozen held-out THC_bounded run on the 5D multiscale Gaussian.
Default: software preflight only. Production: python THC_bounded.py --run
No tuning or parameter overrides. Reports are saved beside this script.
NumPy and Numba required. This file imports no companion project scripts.
The scientific engine below is copied exactly from the verified GREX tuner.
See HELDOUT_PROTOCOL below for all fixed scientific inputs.
"""

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

NUMBA_RUNTIME_EPOCH = "GREX_JUPYTER_PORTABLE_20260829B"

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

SHARED_PROTOCOL_VERSION = "MSG5-expanded-20260925-v1.0.0"

COMMON_INPUT_SCHEMA_VERSION = "fair-playground-common-inputs-v1"

INPUT_PREPARATION_VERSION = "validated-derived-common-inputs-v1.0.0"

DIAGNOSTICS_VERSION = "multichain-geyer-ims-rank-rhat-v1.0.0"

COMMON_PIPELINE_VERSION = "msg-common-rng-observables-score-v1.1.0"

THC_IMPLEMENTATION_VERSION = (
    "thc-exp-tanh-general-mass-psk-grex-portable-epoch-v1.2.0"
)

THC_METHOD_INPUT_SCHEMA_VERSION = "msg-thc-method-inputs-v1"

RANDOM_STREAM_VERSION = "seedsequence-pcg64-three-substreams-v1.0.0"

THC_RUNTIME_PREFLIGHT_VERSION = (
    "msg5-thc-grex-portable-native-preflight-20260829b-v1"
)

SUPPORTED_OBSERVABLE_POLICY = "ALL_LINEAR_AND_UNIQUE_QUADRATIC"

SUPPORTED_COUPLING_FAMILY = "EXP_TANH"

CANONICAL_OUTPUT_FILENAME = (
    "THC_multiscale_Gaussian_tuning_results_EXPANDED_20260925.txt"
)

EXPECTED_INPUT_PREPARATION_FINGERPRINT = (
    "efb6c8b3003d3523159ce3947d894918997c184964ced1364b83a94c43d2b2ce"
)

EXPECTED_DIAGNOSTICS_FINGERPRINT = (
    "dc2148c37f893c7fa5c22431760b2c46ac3aed96cce35909b333eb2abce91274"
)

EXPECTED_COMMON_PIPELINE_FINGERPRINT = (
    "90a12c763a6ba4c6ac26cdac4565cf8f171a38f4000ef3360ed1e2bcf750f364"
)

EXPECTED_SHARED_PROTOCOL_FINGERPRINT = (
    "29d4ece93a4c6dd3f0b8bab97605c342cbdd71359a80f00a850266f022f2287f"
)

CANONICAL_THC_METHOD_INPUT_FINGERPRINT = (
    "d6201ef92a6da8708455dbf4b9a5f8e058b97fe1d62864adcc973cf5e02f6db9"
)

CANONICAL_THC_COMPONENT_FINGERPRINTS = {
    "coupling": "f3c2f82d67a5f2e579c2f396c1c38de3464db63dd3af53beea68ba65a702e49f",
    "lambda_grid": "01fec9b90505d0ba5a638c458f8168537fc199c342afb0af4d14ffaacca31846",
    "a_grid": "a3f4f14c986d29e502158b7ebb7712a31e5d42926714066b0d3c9afd84606444",
    "Q_grid": "17ba736da3bb55c42a64ededc80b4e2376385589adca17fc964373af59e7aaa6",
    "thermostat_density_and_starts": "0a17290bf500dff228b2ae99b7158e7aec1e671460fa2d0559a878dd21a432ca"
}

CANONICAL_COMMON_INPUT_FINGERPRINT = (
    "7feca50f9ee5abb6a7590c554b74380e276f5fd88be5ad3b662658ad789a5039"
)

CANONICAL_COMPONENT_FINGERPRINTS = {
    "target": "114edfc3a078b771f85dc1cf0980403f55ffc5d58f510820ce2f2ef82b1064ff",
    "geometry": "608facf302e6e43aa7758d7b9b6611e40e7643d6d3b84cf734ea3eb7aad13c6c",
    "starts": "d3ce9ceff450ad7761d9147658ce58f04563ff625c45b9fb513373b86deff3a2",
    "grid": "25ad65a8ff0c070b517141e34e40a2f22aa6056999db46378a62a42938d4514c",
    "seed_blocks": "d666b736e46f4580c77e3f12d0434836e4e7e965316db66d326d56becccced18",
    "run_lengths": "2b24c643053cba6025008aafa20a617246687f73f84aba27ff082d6fba3ca872",
    "observable_policy": "74270ffecff1ae5bd44972494f8ac978f7cc0b046f606ba0f7c2703a4ef730d6"
}

@dataclass(frozen=True)
class RunInputs:
    """Every independent common scientific and execution input for one run."""

    beta: float
    omegas: Sequence[float]
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
    lambda_values: Sequence[float]
    a_values: Sequence[float]
    Q_values: Sequence[float]
    thermostat_sigma: float
    thermostat_starts: Sequence[float]

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
    lambda_values: tuple[float, ...]
    a_values: tuple[float, ...]
    Q_values: tuple[float, ...]
    thermostat_sigma: float
    thermostat_starts: np.ndarray
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
    """All-block aggregate for one fixed joint THC configuration."""

    lambda_value: float
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

@njit(cache=True, inline="always")
def _potential_energy_grex_20260829b(
    q: np.ndarray, omega_squared: np.ndarray
) -> float:
    value = 0.0
    for coordinate in range(q.size):
        value += 0.5 * omega_squared[coordinate] * q[coordinate] * q[coordinate]
    return value

@njit(cache=True, inline="always")
def _target_gradient_grex_20260829b(
    q: np.ndarray, omega_squared: np.ndarray
) -> np.ndarray:
    gradient = np.empty(q.size, dtype=np.float64)
    for coordinate in range(q.size):
        gradient[coordinate] = omega_squared[coordinate] * q[coordinate]
    return gradient

@njit(cache=True, inline="always")
def _kinetic_energy_grex_20260829b(
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
def _vector_is_finite_grex_20260829b(values: np.ndarray) -> bool:
    for value in values:
        if not math.isfinite(value):
            return False
    return True

@njit(cache=True, inline="always")
def _log_h_and_derivative_grex_20260829b(
    s: float,
    lambda_value: float,
    a: float,
) -> tuple[float, float]:
    """Return log h and its derivative for h=exp(lambda*tanh(s/a))."""

    tanh_value = math.tanh(s / a)
    log_h = lambda_value * tanh_value
    log_h_derivative = (lambda_value / a) * (
        1.0 - tanh_value * tanh_value
    )
    return log_h, log_h_derivative

@njit(cache=True)
def _thc_hamiltonian_grex_20260829b(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    beta: float,
    lambda_value: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    omega_squared: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> float:
    """Evaluate the extended THC Hamiltonian up to one additive constant."""

    log_h, _log_h_derivative = _log_h_and_derivative_grex_20260829b(
        s, lambda_value, a
    )
    potential = _potential_energy_grex_20260829b(q, omega_squared)
    base_kinetic = _kinetic_energy_grex_20260829b(
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

@njit(cache=True, inline="always")
def _thermostat_rotation_grex_20260829b(
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

@njit(cache=True, inline="always")
def _thc_state_is_finite_grex_20260829b(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
) -> bool:
    return (
        math.isfinite(s)
        and math.isfinite(p_s)
        and _vector_is_finite_grex_20260829b(q)
        and _vector_is_finite_grex_20260829b(p)
    )

@njit(cache=True, nogil=True)
def _thc_proposal_grex_20260829b(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    current_gradient: np.ndarray,
    beta: float,
    lambda_value: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    epsilon: float,
    L: int,
    omega_squared: np.ndarray,
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
    _log_h, log_h_derivative = _log_h_and_derivative_grex_20260829b(
        proposed_s, lambda_value, a
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
    if not _thc_state_is_finite_grex_20260829b(
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
        proposed_s, proposed_p_s = _thermostat_rotation_grex_20260829b(
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
        log_h, log_h_derivative = _log_h_and_derivative_grex_20260829b(
            proposed_s, lambda_value, a
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
        if not _thc_state_is_finite_grex_20260829b(
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
        proposed_s, proposed_p_s = _thermostat_rotation_grex_20260829b(
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
        proposed_gradient = _target_gradient_grex_20260829b(
            proposed_q, omega_squared
        )
        gradient_calls += 1
        if not _vector_is_finite_grex_20260829b(proposed_gradient):
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
        _log_h, log_h_derivative = _log_h_and_derivative_grex_20260829b(
            proposed_s, lambda_value, a
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
        if not _thc_state_is_finite_grex_20260829b(
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

@njit(cache=True, nogil=True)
def _simulate_thc_chain_grex_20260829b(
    standard_momenta: np.ndarray,
    thermostat_standard_momenta: np.ndarray,
    acceptance_uniforms: np.ndarray,
    q_initial: np.ndarray,
    s_initial: float,
    beta: float,
    lambda_value: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    omega_squared: np.ndarray,
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
    current_gradient = _target_gradient_grex_20260829b(q, omega_squared)

    # The same endpoint-gradient cache and phase charging used by HMC.
    warmup_gradient_calls = 1 if n_warmup > 0 else 0
    retained_gradient_calls = 0 if n_warmup > 0 else 1
    accepted_retained = 0
    warmup_nonfinite = 0
    retained_nonfinite = 0
    inverse_sqrt_beta = 1.0 / math.sqrt(beta)
    thermostat_momentum_scale = math.sqrt(Q / beta)

    for iteration in range(total_iterations):
        log_h, _log_h_derivative = _log_h_and_derivative_grex_20260829b(
            s, lambda_value, a
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
        initial_energy = _thc_hamiltonian_grex_20260829b(
            q,
            p,
            s,
            p_s,
            beta,
            lambda_value,
            a,
            Q,
            thermostat_sigma,
            omega_squared,
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
        ) = _thc_proposal_grex_20260829b(
            q,
            p,
            s,
            p_s,
            current_gradient,
            beta,
            lambda_value,
            a,
            Q,
            thermostat_sigma,
            epsilon,
            L,
            omega_squared,
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
            proposed_energy = _thc_hamiltonian_grex_20260829b(
                proposed_q,
                proposed_p,
                proposed_s,
                proposed_p_s,
                beta,
                lambda_value,
                a,
                Q,
                thermostat_sigma,
                omega_squared,
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

_potential_energy = _potential_energy_grex_20260829b

_target_gradient = _target_gradient_grex_20260829b

_kinetic_energy = _kinetic_energy_grex_20260829b

_vector_is_finite = _vector_is_finite_grex_20260829b

_log_h_and_derivative = _log_h_and_derivative_grex_20260829b

_thc_hamiltonian = _thc_hamiltonian_grex_20260829b

_thermostat_rotation = _thermostat_rotation_grex_20260829b

_thc_state_is_finite = _thc_state_is_finite_grex_20260829b

_thc_proposal = _thc_proposal_grex_20260829b

_simulate_thc_chain = _simulate_thc_chain_grex_20260829b

def _reference_thc_proposal(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    current_gradient: np.ndarray,
    beta: float,
    lambda_value: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    epsilon: float,
    L: int,
    omega_squared: np.ndarray,
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

    tanh_value = math.tanh(reference_s / a)
    log_h_derivative = (lambda_value / a) * (
        1.0 - tanh_value * tanh_value
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

        tanh_value = math.tanh(reference_s / a)
        log_h = lambda_value * tanh_value
        log_h_derivative = (lambda_value / a) * (
            1.0 - tanh_value * tanh_value
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

        reference_gradient = omega_squared * reference_q
        gradient_calls += 1
        kick_duration = 0.5 * epsilon if step == L - 1 else epsilon
        tanh_value = math.tanh(reference_s / a)
        log_h_derivative = (lambda_value / a) * (
            1.0 - tanh_value * tanh_value
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
    """Compare current native dynamics with an independent same-host oracle."""

    cases = (
        (
            np.asarray([0.17, -0.11, 0.06], dtype=np.float64),
            np.asarray([0.31, -0.24, 0.19], dtype=np.float64),
            0.23,
            -0.16,
            1.0,
            0.4054651081081644,
            0.75,
            0.5,
            1.0,
            0.004,
            3,
            np.asarray([1.0, 9.0, 25.0], dtype=np.float64),
            np.eye(3, dtype=np.float64),
            True,
        ),
        (
            np.asarray([0.2, -0.1], dtype=np.float64),
            np.asarray([0.4, -0.25], dtype=np.float64),
            -0.2,
            0.15,
            1.3,
            0.2,
            1.1,
            0.7,
            1.3,
            0.002,
            7,
            np.asarray([1.0, 9.0], dtype=np.float64),
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
            lambda_value,
            a,
            Q,
            thermostat_sigma,
            epsilon,
            L,
            omega_squared,
            inverse_mass_matrix,
            mass_is_identity,
        ) = case
        gradient = omega_squared * q
        native = _thc_proposal(
            q,
            p,
            s,
            p_s,
            gradient,
            beta,
            lambda_value,
            a,
            Q,
            thermostat_sigma,
            epsilon,
            L,
            omega_squared,
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
            lambda_value,
            a,
            Q,
            thermostat_sigma,
            epsilon,
            L,
            omega_squared,
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
                f"THC proposal mismatch in microcheck case {case_index}."
            )
    return (
        "PASS: compiled THC proposal matches an independent same-host Python "
        "oracle for identity and general mass"
    )

def _make_random_streams(
    seed: int,
    total_iterations: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create physical-momentum and acceptance streams from fixed substreams.

    Three child SeedSequences are always spawned in this order:
      1. physical standard-normal momenta,
      2. thermostat standard-normal momenta (reserved for the later THC code),
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
    lambda_value: float,
    a: float,
    Q: float,
    epsilon: float,
    L: int,
    block_results: Sequence[THCBlockResult],
) -> THCConfigurationResult:
    """Aggregate one fixed joint THC configuration over common seed blocks."""

    blocks = tuple(block_results)
    aggregate = _aggregate_blocks(
        tuple(block.physical for block in blocks)
    )
    return THCConfigurationResult(
        lambda_value=float(lambda_value),
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
            "No THC configuration has a defined all-block tuning score."
        )

    # Exact ties: larger minimum block score, fewer gradients/draw, then
    # smaller epsilon, L, lambda, a, and Q.  No near-tie tolerance or
    # diagnostic threshold is introduced.
    return max(
        eligible,
        key=lambda result: (
            result.geometric_mean_efficiency,
            result.minimum_block_efficiency,
            -result.block_results[
                0
            ].physical.gradients_per_retained_draw,
            -result.epsilon,
            -result.L,
            -result.lambda_value,
            -result.a,
            -result.Q,
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
    omegas: np.ndarray,
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
            "dimension": int(omegas.size),
            "beta": float(beta),
            "omegas": omegas.tolist(),
        },
        "geometry": {"mass_matrix": mass_matrix.tolist()},
        "starts": {"standardized_starts": standardized_starts.tolist()},
        "grid": {
            "epsilon_values": list(epsilon_values),
            "L_values": list(L_values),
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
            "This THC-v1 tuner supports only "
            f"coupling_family='{SUPPORTED_COUPLING_FAMILY}'."
        )

    lambda_values = _normalize_method_grid(
        "lambda_values", inputs.lambda_values, allow_zero=True
    )
    maximum_safe_lambda = 0.5 * math.log(np.finfo(np.float64).max)
    if any(value >= maximum_safe_lambda for value in lambda_values):
        raise ValueError(
            "Every lambda must be below "
            f"{maximum_safe_lambda:.12g} so h(s)^(-2) remains representable."
        )
    a_values = _normalize_method_grid(
        "a_values", inputs.a_values, allow_zero=False
    )
    Q_values = _normalize_method_grid(
        "Q_values", inputs.Q_values, allow_zero=False
    )

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
        "coupling": {"family": coupling_family},
        "lambda_grid": {"lambda_values": list(lambda_values)},
        "a_grid": {"a_values": list(a_values)},
        "Q_grid": {"Q_values": list(Q_values)},
        "thermostat_density_and_starts": {
            "thermostat_sigma": thermostat_sigma,
            "thermostat_starts": thermostat_starts.tolist(),
        },
    }
    component_fingerprints = {
        name: _json_fingerprint(value)
        for name, value in components.items()
    }
    canonical_method_mismatches = tuple(
        name
        for name in components
        if component_fingerprints[name]
        != CANONICAL_THC_COMPONENT_FINGERPRINTS.get(name, "")
    )
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
        lambda_values=lambda_values,
        a_values=a_values,
        Q_values=Q_values,
        thermostat_sigma=thermostat_sigma,
        thermostat_starts=thermostat_starts,
        method_input_components=components,
        method_input_fingerprint=method_input_fingerprint,
        canonical_method_mismatches=canonical_method_mismatches,
    )

def _validate_thc_run_mode(prepared: PreparedTHCInputs) -> None:
    """Apply common mode separation and the independent THC-input seal."""

    _validate_run_mode(prepared.common)
    if prepared.common.run_mode == "CANONICAL":
        if prepared.canonical_method_mismatches or (
            prepared.method_input_fingerprint
            != CANONICAL_THC_METHOD_INPUT_FINGERPRINT
        ):
            mismatch_text = (
                ", ".join(prepared.canonical_method_mismatches)
                or "overall thermostat hash"
            )
            raise ValueError(
                "CANONICAL THC-input mismatch in: "
                + mismatch_text
                + ". Restore the agreed bottom-block thermostat values, "
                "or use CUSTOM mode with a non-comparable output."
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
    """Hash code that must remain byte-identical in the later THC tuner."""

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

def _run_chain_job(
    standard_momenta: np.ndarray,
    thermostat_standard_momenta: np.ndarray,
    uniforms: np.ndarray,
    q_initial: np.ndarray,
    s_initial: float,
    beta: float,
    lambda_value: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    omega_squared: np.ndarray,
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
        lambda_value,
        a,
        Q,
        thermostat_sigma,
        epsilon,
        L,
        n_warmup,
        n_retained,
        omega_squared,
        mass_cholesky,
        inverse_mass_matrix,
        mass_is_identity,
    )

def _run_configuration_block(
    *,
    lambda_value: float,
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
) -> THCBlockResult:
    common = prepared.common
    jobs = [
        (
            stream_cache[chain_index][0],
            stream_cache[chain_index][1],
            stream_cache[chain_index][2],
            common.physical_starts[chain_index],
            prepared.thermostat_starts[chain_index],
            common.beta,
            lambda_value,
            a,
            Q,
            prepared.thermostat_sigma,
            epsilon,
            L,
            common.n_warmup,
            common.n_retained,
            common.omega_squared,
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
    """Verify the live cache-free epoch without pinning a stochastic path.

    Long Metropolis trajectories are not bitwise portable across operating
    systems and math-library/compiler builds.  A one-bit difference near an
    accept/reject boundary can legitimately change every later draw.  The
    mathematical, reversibility, gradient-accounting, RNG, and finite-chain
    checks are performed by ``run_self_tests`` immediately before this audit.
    This function checks that those tests exercised the intended newly named
    cache-free native dispatchers rather than an older notebook definition.
    """

    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "Canonical Grex production requires Numba. Select the configured "
            "Python 3.12 sampling kernel before starting the tuning grid."
        )

    expected_dispatchers = (
        ("_potential_energy", _potential_energy, _potential_energy_grex_20260829b),
        ("_target_gradient", _target_gradient, _target_gradient_grex_20260829b),
        ("_kinetic_energy", _kinetic_energy, _kinetic_energy_grex_20260829b),
        ("_vector_is_finite", _vector_is_finite, _vector_is_finite_grex_20260829b),
        (
            "_log_h_and_derivative",
            _log_h_and_derivative,
            _log_h_and_derivative_grex_20260829b,
        ),
        ("_thc_hamiltonian", _thc_hamiltonian, _thc_hamiltonian_grex_20260829b),
        (
            "_thermostat_rotation",
            _thermostat_rotation,
            _thermostat_rotation_grex_20260829b,
        ),
        (
            "_thc_state_is_finite",
            _thc_state_is_finite,
            _thc_state_is_finite_grex_20260829b,
        ),
        ("_thc_proposal", _thc_proposal, _thc_proposal_grex_20260829b),
        (
            "_simulate_thc_chain",
            _simulate_thc_chain,
            _simulate_thc_chain_grex_20260829b,
        ),
    )
    problems: list[str] = []
    for public_name, active_dispatcher, expected_dispatcher in expected_dispatchers:
        if active_dispatcher is not expected_dispatcher:
            problems.append(f"{public_name} does not point to the current epoch")
            continue
        python_function = getattr(active_dispatcher, "py_func", None)
        if python_function is None or not python_function.__name__.endswith(
            "_grex_20260829b"
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

    standardized_gradient_point = np.linspace(
        0.25, -0.15, dimension
    )
    q_gradient_test = (
        standardized_gradient_point
        * np.sqrt(common.target_variances)
    )
    analytic_gradient = _target_gradient(
        q_gradient_test, common.omega_squared
    )
    finite_difference_gradient = np.empty(
        dimension, dtype=np.float64
    )
    for coordinate in range(dimension):
        finite_difference_step = (
            1.0e-6
            * math.sqrt(
                common.target_variances[coordinate]
            )
        )
        q_plus = q_gradient_test.copy()
        q_minus = q_gradient_test.copy()
        q_plus[coordinate] += finite_difference_step
        q_minus[coordinate] -= finite_difference_step
        finite_difference_gradient[coordinate] = (
            _potential_energy(
                q_plus, common.omega_squared
            )
            - _potential_energy(
                q_minus, common.omega_squared
            )
        ) / (2.0 * finite_difference_step)
    if not np.allclose(
        analytic_gradient,
        finite_difference_gradient,
        rtol=2.0e-8,
        atol=2.0e-8,
    ):
        raise RuntimeError(
            "Self-test failed: analytic target gradient."
        )
    reconstructed_starts = (
        common.physical_starts
        / np.sqrt(common.target_variances)[
            np.newaxis, :
        ]
    )
    if not np.allclose(
        reconstructed_starts,
        common.standardized_starts,
    ):
        raise RuntimeError(
            "Self-test failed: standardized-start conversion."
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
    messages.append(
        "PASS: analytic target gradient, standardized starts, and mass algebra"
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

    test_lambda = prepared.lambda_values[0]
    test_a = prepared.a_values[0]
    coupling_s = 0.37
    coupling_step = 1.0e-6
    log_h, analytic_log_h_derivative = (
        _log_h_and_derivative(
            coupling_s, test_lambda, test_a
        )
    )
    log_h_plus, _ = _log_h_and_derivative(
        coupling_s + coupling_step,
        test_lambda,
        test_a,
    )
    log_h_minus, _ = _log_h_and_derivative(
        coupling_s - coupling_step,
        test_lambda,
        test_a,
    )
    finite_difference_log_h_derivative = (
        log_h_plus - log_h_minus
    ) / (2.0 * coupling_step)
    if (
        not math.isclose(
            analytic_log_h_derivative,
            finite_difference_log_h_derivative,
            rel_tol=2.0e-9,
            abs_tol=2.0e-11,
        )
        or abs(log_h) > test_lambda
    ):
        raise RuntimeError(
            "Self-test failed: bounded coupling derivative."
        )
    messages.append(
        "PASS: exp-tanh coupling bound and analytic logarithmic derivative"
    )

    q0 = (
        np.linspace(0.3, -0.2, dimension)
        * np.sqrt(common.target_variances)
    )
    standard_p0 = np.linspace(
        0.7, -0.3, dimension
    )
    p0 = (
        common.mass_cholesky @ standard_p0
        / math.sqrt(common.beta)
    )
    s0 = 0.31
    p_s0 = -0.27
    modal_matrix = (
        common.inverse_mass_matrix
        @ np.diag(common.omega_squared)
    )
    maximum_frequency = math.sqrt(
        max(
            float(
                np.max(
                    np.real(
                        np.linalg.eigvals(
                            modal_matrix
                        )
                    )
                )
            ),
            1.0,
        )
    )
    test_epsilon = (
        0.02
        / (
            maximum_frequency
            * math.exp(test_lambda)
        )
    )
    grad0 = _target_gradient(
        q0, common.omega_squared
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
        test_lambda,
        test_a,
        prepared.Q_values[0],
        prepared.thermostat_sigma,
        test_epsilon,
        17,
        common.omega_squared,
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
        test_lambda,
        test_a,
        prepared.Q_values[0],
        prepared.thermostat_sigma,
        test_epsilon,
        17,
        common.omega_squared,
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
    general_omega_squared = np.asarray(
        [1.0, 9.0]
    )
    general_grad0 = _target_gradient(
        general_q0, general_omega_squared
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
        general_omega_squared,
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
        general_omega_squared,
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
        flow_gradient = _target_gradient(flow_q, general_omega_squared)
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
            general_omega_squared,
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
            general_omega_squared,
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

    # At lambda=0, the physical P/K dynamics reduce exactly to HMC leapfrog.
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
            control_q, common.omega_squared
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
        common.omega_squared,
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
            "Self-test failed: lambda=0 HMC physical-control parity."
        )
    messages.append(
        "PASS: lambda=0 physical proposal matches cached general-mass HMC leapfrog"
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
        test_lambda,
        test_a,
        prepared.Q_values[0],
        prepared.thermostat_sigma,
        test_epsilon,
        4,
        2,
        3,
        common.omega_squared,
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
    synthetic_samples = (
        rng.standard_normal(
            (common.n_chains, 120, dimension)
        )
        * np.sqrt(common.target_variances)[
            np.newaxis, np.newaxis, :
        ]
    )
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
    expected_configurations = (
        len(prepared.lambda_values)
        * len(prepared.a_values)
        * len(prepared.Q_values)
        * len(common.epsilon_values)
        * len(common.L_values)
    )
    if expected_configurations <= 0:
        raise RuntimeError(
            "Self-test failed: joint grid cardinality."
        )
    messages.append(
        (
            f"PASS: {expected_observable_count}-observable exact moments, "
            "nonfinite propagation, common GM, thermostat moments, and "
            f"{expected_configurations}-cell Cartesian design"
        )
    )
    return tuple(messages)

# Fixed held-out inputs. Do not edit them after opening final results.
HELDOUT_METHOD = 'THC_bounded'
HELDOUT_PROTOCOL = json.loads('{"IAT":"Unchanged all-chain Geyer initial-positive then initial-monotone sequence; biased autocovariance denominator N at every lag; no IAT floor.","Rhat":"Maximum of rank-normalized split and folded rank-normalized split R-hat; descriptive, no selection threshold.","THC_refresh":"Only p and p_s are redrawn each transition. s persists; reject restores q AND s.","beta":1.0,"comparison_id":"MSG5_HMC_BOUNDED_THC_HELDOUT_20261012_v1","date_frozen":"2026-10-12","failure_rule":"Unchanged Gaussian rule: any nonfinite trajectory in warmup OR retention, or any undefined required IAT, makes the affected block efficiency and full method score undefined. No dropping or replacement of blocks.","gradient_cost":"Actual newly executed target-gradient calls during retained transitions only; reuse cached endpoint gradient. Warmup excluded. Expected g=15 for HMC, 25 for THC when all trajectories finish.","mass_matrix":[[1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,1.0]],"methods":{"HMC":{"L":15,"epsilon":0.055},"THC_bounded":{"L":25,"Q":0.25,"a":0.5,"coupling_family":"EXP_TANH","epsilon":0.04,"lambda_value":0.22314355131420976,"thermostat_sigma":1.0,"thermostat_starts":[0.0,0.0,0.0,0.0]}},"n_blocks":10,"n_chains":4,"n_retained":10000,"n_warmup":1000,"no_retuning":true,"observable_policy":"ALL_LINEAR_AND_UNIQUE_QUADRATIC","omegas":[1.0,2.340347319320716,5.477225575051661,12.818610191887021,30.0],"paired_comparison":"S_THC/S_HMC and paired block wins; a 95% Student-t interval on the mean of ten log(E_THC/E_HMC) values, exponentiated. Approximate block-level uncertainty; only defined if all required scores are defined.","score":"Block E=1/(g*max_observable_IAT); method S=geometric mean of ALL ten block efficiencies.","secondary_outcomes":"Maximum and descriptive geometric mean of blockwise worst IAT; maximum R-hat; acceptance; exact-moment errors; numerical events. No reselection after held-out inspection.","seed_audit":{"counters":{},"derivation":"First 4 SHA256 bytes, big-endian, of DOMAIN|PHASE|block=BB|chain=CC|counter=K. Increment only for a recorded occurrence or duplicate, never based on a simulation.","domain":"paper-on-thc|MSG5|HMC-vs-bounded-THC|held-out|20261012|v1","overlaps_after_audit":0,"path_size_inventory_sha256":"09afeb9642902cc34be0716c1ea080013f7e46ab37e7029f4569fffefcfcfe1d","rejected_candidate_values":[],"scope":"Recorded integer seed occurrences in local project text/source files, including all downloaded tuning reports; unrecorded remote-only seeds cannot be audited.","text_bytes_scanned":3052539390,"text_files_scanned":2618},"seed_blocks":[[100627600,3507966996,1457724111,3279080072],[636973577,3160402427,1140582546,3315606578],[2526761325,2069866603,1232498498,2667830376],[4132157476,855810834,3577245579,2610819126],[344236651,579907653,203260244,1558913547],[2013339583,3298764728,3800075317,1550030833],[3213789136,2403803,3446466256,3362034346],[740760712,1235417406,593841059,3488148568],[2728399884,2359026335,1035746545,1682714210],[1822352728,2462754932,3438856697,264149603]],"selection":"One best-tested gradient-normalized-efficiency representative per method, selected only using the completed expanded tuning runs.","source_seals":{"HMC":{"common_pipeline":"90a12c763a6ba4c6ac26cdac4565cf8f171a38f4000ef3360ed1e2bcf750f364","diagnostics":"dc2148c37f893c7fa5c22431760b2c46ac3aed96cce35909b333eb2abce91274","implementation":"4deefc8080d0b9a216cd80fc90ef29715d47f967dd08cb10897fc1aa24085d16","input_preparation":"efb6c8b3003d3523159ce3947d894918997c184964ced1364b83a94c43d2b2ce","source_sha256":"bbd02b487b0e3fe5082edb140d6553ae841b5e16964b6df2ad9fd5c87a915d29"},"THC_bounded":{"common_pipeline":"90a12c763a6ba4c6ac26cdac4565cf8f171a38f4000ef3360ed1e2bcf750f364","diagnostics":"dc2148c37f893c7fa5c22431760b2c46ac3aed96cce35909b333eb2abce91274","implementation":"11e71f7ea84a285d2bab640c1f2bc15351004a567fdffe93a7409aeac305f034","input_preparation":"efb6c8b3003d3523159ce3947d894918997c184964ced1364b83a94c43d2b2ce","source_sha256":"9c2e846ccc8c4a939a91937b12de45755c369f3863eea61fe2ba609201e3a708"}},"standardized_starts":[[2.0,2.0,2.0,2.0,2.0],[2.0,-2.0,2.0,-2.0,2.0],[-2.0,2.0,-2.0,2.0,-2.0],[-2.0,-2.0,-2.0,-2.0,-2.0]],"target":"5D multiscale Gaussian","tuning_report_sha256":{"HMC":"d3b99c52ed8c5f343134b699831777c611491c4cf6bea9f70f10c84d774ce099","THC_bounded":"0d92d3b4acccca490ebd5087dbb5342432a5784f612227a0ba96aead3dc02fe0"}}')
HELDOUT_PROTOCOL_SHA256 = '1f3cea89e37149d954f1d73f708ea2b662b891760d4953cfedb15d18b3887dfe'
SOFTWARE_TEST_SEEDS = [[2758641246, 3934563322, 1943090285, 4018007532], [1591663876, 4190706769, 138180850, 1330750951], [2293564723, 1127476765, 1797475913, 588494531], [3891440282, 1804643437, 179666077, 1015273864], [2896551031, 540901230, 1816732640, 44474283], [3383016790, 2433462216, 3705378905, 1646824408], [1200339377, 3448706444, 3749031675, 1856621062], [1980574640, 1535557055, 3066659044, 3653952789], [3847978028, 934311850, 3749472748, 2213487748], [4258696011, 1649635790, 3610337756, 3576668734]]

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
        raise RuntimeError('Frozen protocol changed. Do not retune the final comparison.')
    if not NUMBA_AVAILABLE:
        raise RuntimeError('Numba is required for this fixed production experiment.')
    expected = HELDOUT_PROTOCOL['source_seals'][HELDOUT_METHOD]
    actual = {
        'diagnostics': _diagnostics_fingerprint(),
        'common_pipeline': _common_pipeline_fingerprint(),
        'input_preparation': _input_preparation_fingerprint(),
        'implementation': (_hmc_implementation_fingerprint() if HELDOUT_METHOD == 'HMC'
                           else _thc_implementation_fingerprint()),
    }
    for key, value in actual.items():
        if value != expected[key]:
            raise RuntimeError('Scientific source seal mismatch: ' + key)
    final = [s for block in HELDOUT_PROTOCOL['seed_blocks'] for s in block]
    test = [s for block in SOFTWARE_TEST_SEEDS for s in block]
    if len(final) != 40 or len(set(final)) != 40 or len(test) != 40 or len(set(test)) != 40:
        raise RuntimeError('The fixed seed schedules must each contain 40 unique seeds.')
    if set(final) & set(test):
        raise RuntimeError('Software-test seeds overlap final seeds.')
    return actual


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
    setting = p['methods'][HELDOUT_METHOD]
    arguments = dict(
        beta=p['beta'], omegas=p['omegas'], mass_matrix=p['mass_matrix'],
        standardized_starts=p['standardized_starts'], observable_policy=p['observable_policy'],
        epsilon_values=(setting['epsilon'],), L_values=(setting['L'],),
        seed_blocks=p['seed_blocks'], n_warmup=p['n_warmup'], n_retained=p['n_retained'],
        n_workers=workers, run_mode='CUSTOM',
        run_label='CUSTOM: FROZEN HELD-OUT' if not testing else 'CUSTOM: SOFTWARE TEST ONLY',
        output_directory=None, output_filename=HELDOUT_METHOD + '_results.txt',
        allow_overwrite=False, print_progress=False, print_report=False, run_self_tests=True,
    )
    # CUSTOM is only the legacy tuner's input-validation label: this driver has
    # its OWN immutable held-out protocol seal and accepts no scientific overrides.
    if HELDOUT_METHOD == 'HMC':
        prepared = _prepare_inputs(RunInputs(**arguments))
        _validate_run_mode(prepared)
    else:
        prepared = _prepare_thc_inputs(THCRunInputs(
            **arguments, coupling_family=setting['coupling_family'],
            lambda_values=(setting['lambda_value'],), a_values=(setting['a'],),
            Q_values=(setting['Q'],), thermostat_sigma=setting['thermostat_sigma'],
            thermostat_starts=setting['thermostat_starts']))
        _validate_thc_run_mode(prepared)
        _validate_common_parity_seals(prepared.common)
    return prepared


def _heldout_common(prepared):
    return prepared if HELDOUT_METHOD == 'HMC' else prepared.common


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
    streams = [_make_random_streams(s, common.n_warmup + common.n_retained, common.dimension)
               for s in seeds]
    physical_hash = hashlib.sha256()
    all_hash = hashlib.sha256()
    for stream in streams:
        for i, array in enumerate(stream):
            payload = np.ascontiguousarray(array, dtype='<f8').tobytes(order='C')
            all_hash.update(payload)
            if i in (0, 2):
                physical_hash.update(payload)
    kwargs = dict(epsilon=common.epsilon_values[0], L=common.L_values[0],
                  block_index=index, seeds=seeds, stream_cache=streams, prepared=prepared)
    if HELDOUT_METHOD == 'HMC':
        physical = _run_configuration_block(**kwargs)
        thermostat = None
    else:
        output = _run_configuration_block(**kwargs, lambda_value=prepared.lambda_values[0],
                                          a=prepared.a_values[0], Q=prepared.Q_values[0])
        physical = output.physical
        thermostat = asdict(output.thermostat)
    record = _heldout_clean(dict(
        physical=asdict(physical), thermostat=thermostat,
        physical_and_acceptance_stream_sha256=physical_hash.hexdigest(),
        all_three_streams_sha256=all_hash.hexdigest(),
    ))
    record['record_sha256'] = _heldout_hash(record)
    return record


def _heldout_preflight(workers=4):
    _heldout_guard()
    # NEVER pass the production prepared inputs to tests or preflight.
    prepared = _heldout_prepare(workers, testing=True)
    messages = list(run_self_tests(prepared))
    if HELDOUT_METHOD != 'HMC':
        messages.append(_validate_portable_native_runtime(prepared))
    sample = _heldout_one_block(prepared, 0)
    block = _heldout_decode_block(sample)
    common = _heldout_common(prepared)
    if len(block.observable_diagnostics) != 20 or block.retained_gradient_calls <= 0:
        raise RuntimeError('Software-check block did not return the expected diagnostics.')
    if block.warmup_nonfinite_trajectories or block.retained_nonfinite_trajectories:
        raise RuntimeError('Software-check trajectory unexpectedly became nonfinite.')
    if block.retained_gradient_calls != common.n_chains * common.n_retained * block.L:
        raise RuntimeError('Gradient-accounting check failed on finite software-check trajectories.')
    restored = _heldout_clean(asdict(_heldout_decode_block(json.loads(json.dumps(sample)))))
    if restored != sample['physical']:
        raise RuntimeError('Checkpoint serialization round-trip failed.')
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
                or len(block.observable_diagnostics) != 20):
            raise RuntimeError('Checkpoint block identifiers do not match the frozen schedule.')
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
            'Target: U(q)=0.5*sum(omega_j^2*q_j^2); beta=1; base mass=I_5.',
            'Observables: all 5 coordinates and all 15 unique quadratic products.',
            'IAT: unchanged multi-chain Geyer IPS/IMS, biased denominator N, no floor.',
            'R-hat: max(rank-normalized split, folded rank-normalized split); descriptive.',
            'Gradient cost: actual retained target-gradient calls; warmup excluded; cached gradients not counted twice.',
            'Nonfinite warmup/retained trajectory OR undefined required IAT => undefined full score.',
            'No blocks omitted. Numerical rejection repeats the previous state.',
            'No adaptation, tuning or selection has been performed using these final blocks.']
    if HELDOUT_METHOD != 'HMC':
        text += ['h(s)=exp(log(1.25)*tanh(s/0.5)); rho(s)=N(0,1); Q=0.25.',
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
        text += ['', 'THERMOSTAT MOMENTS (descriptive only; exact E[s]=0, E[s^2]=1)',
                 'block mean_s second_moment_s']
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
        format='MSG5-heldout-comparison-report-v1', metadata=metadata,
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
