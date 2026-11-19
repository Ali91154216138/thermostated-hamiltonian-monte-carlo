"""Fair-playground conditional/dynamic-THC tuner for centered Neal's funnel.

This is a fully standalone Grex/Jupyter program.  It does not import a shared
protocol, diagnostics module, or neighbouring sampler file.  Every independent
 common and thermostat-specific input is collected in one control panel at the
end of this file.  The paired Neal-funnel HMC tuner uses the identical common
 target, base mass, starts, epsilon-L grid, seed blocks, run lengths, observables,
diagnostics, and target-gradient accounting sealed here.

Production target
-----------------
    q=(y,x_1,...,x_5), beta=1, sigma_y=3
    U=y^2/(18)+(5/2)y+exp(-y)||x||^2/2
    physical base mass M0=I_6

Under exp(-beta*U), y is centered Gaussian and x_i|y is centered Gaussian
with variance exp(y) at the canonical beta.  Starts are entered as
(y,z_1,...,z_5), with x_i=exp(y/2)z_i.

Conditional/dynamic THC construction
------------------------------------
This is the target-informed, method-native Neal-funnel geometry used in the
manuscript.  It is deliberately a separate competition track from the bounded
scalar EXP_TANH THC implementation.  With q=(y,x),

    h_x(s)=exp(-s/2),       G(s)=diag(1,exp(-s) I_5),
    C(y,s)=(s-y)^2/(2*tau^2),

and

    H=U + p_y^2/2 + exp(s)||p_x||^2/2 + p_s^2/(2Q)
        +(s-y)^2/(2*tau^2) - d_x*s/(2*beta).

At every transition p_y~N(0,1/beta), p_x|s~N(0,exp(-s)I/beta), and
p_s~N(0,Q/beta).  A symmetric exact-subflow
V/2-A_y/2-A_s/2-A_x-A_s/2-A_y/2-V/2 split is followed by a complete momentum
flip and full-state Metropolis test.  The persistent variables are (q,s).

Tuning protocol
---------------
Every fixed (tau, Q, epsilon, L) configuration is evaluated on the same
five seed blocks.  Each block contains four chains, fixed dispersed starts,
1000 warmup iterations, and 4000 retained draws.  The score observables are
exactly all six physical linears and all 21 unique quadratic monomials.  No
thermostat or target-specific nonlinear quantity is a score observable.

For each block and observable f,

    efficiency(f) = 1 / (actual target gradients per retained draw * IAT(f)).

The block score is the minimum efficiency over observables.  A configuration's
tuning score is the geometric mean of its five block scores.  No R-hat or
moment threshold is used.  If any required IAT is undefined, or any trajectory
becomes nonfinite, the corresponding efficiency is undefined.

The first Grex execution defaults to a short cache-free native preflight.  It
does not simulate a tuning configuration or write a report.  Production prints
the complete report and writes the identical text in the script/Jupyter
working directory unless another output directory is entered.

Required package: numpy
Strongly recommended for the production grid: numba
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

# Grex/Jupyter standalone native epoch.  Persistent Numba caching is disabled
# in every execution mode, and every compiled THC function below has a new
# epoch-specific Python name.  Consequently an older notebook dispatcher or
# cached native helper cannot satisfy any call made by this program.
NUMBA_RUNTIME_EPOCH = "FUNNEL6_CONDITIONAL_THC_GREX_PORTABLE_20260829"
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


# ---------------------------------------------------------------------------
# Internal protocol metadata and integrity seals -- not user inputs.
# ---------------------------------------------------------------------------

SHARED_PROTOCOL_VERSION = "FUNNEL6-fair-playground-v1.0.0"
COMMON_INPUT_SCHEMA_VERSION = "funnel6-fair-playground-common-inputs-v1"
INPUT_PREPARATION_VERSION = "funnel6-validated-derived-inputs-v1.0.0"
DIAGNOSTICS_VERSION = "multichain-geyer-ims-rank-rhat-v1.0.0"
COMMON_PIPELINE_VERSION = "funnel6-common-rng-observables-score-v1.0.0"
THC_IMPLEMENTATION_VERSION = (
    "funnel6-conditional-thc-target-informed-v-ay-as-ax-grex-portable-v1.0.0"
)
THC_METHOD_INPUT_SCHEMA_VERSION = "funnel6-conditional-thc-method-inputs-v1"
RANDOM_STREAM_VERSION = "seedsequence-pcg64-three-substreams-v1.0.0"
THC_RUNTIME_PREFLIGHT_VERSION = (
    "funnel6-conditional-thc-grex-portable-native-preflight-20260829-v1"
)
SUPPORTED_OBSERVABLE_POLICY = "ALL_LINEAR_AND_UNIQUE_QUADRATIC"
SUPPORTED_COUPLING_FAMILY = "FUNNEL_CONDITIONAL_EXP_MINUS_HALF_S"
CANONICAL_OUTPUT_FILENAME = (
    "Conditional_THC_Neals_funnel_tuning_results_GREX_PORTABLE_V1.txt"
)

# Frozen only after the deterministic preflight and tiny non-production
# end-to-end audit pass.  Together they attest to the sealed scientific core
# functions and the normalized bottom-block inputs.
EXPECTED_INPUT_PREPARATION_FINGERPRINT = (
    "5b0bb03dbb52bbecb255c0b0fcf3a4bc44ef165ac698f79dacc4f05f2ea32b88"
)
EXPECTED_DIAGNOSTICS_FINGERPRINT = (
    "dc2148c37f893c7fa5c22431760b2c46ac3aed96cce35909b333eb2abce91274"
)
EXPECTED_COMMON_PIPELINE_FINGERPRINT = (
    "8cfc3cb145c8f40fd5f0c0893effc0e7387cd4a7ffdfdf34c4db9417e17fdddf"
)
EXPECTED_SHARED_PROTOCOL_FINGERPRINT = (
    "fecd1a7e17d649c1dbf76593b14dfa94659a47d85c9c59d29ff93de8b07f84fa"
)
EXPECTED_THC_IMPLEMENTATION_FINGERPRINT = (
    "0f0aa1a72cbb12c4cc2ee22f3f26f491ac1444401f28cd89896302e76bf79712"
)
CANONICAL_THC_METHOD_INPUT_FINGERPRINT = (
    "7eb2e833c92e4dacf476b424e7c8deb6e0ec82f99aff8c8f4d9d4a8e9b87fa71"
)
CANONICAL_THC_COMPONENT_FINGERPRINTS = {
    "coupling": "3ce3af10b57de487ddc2b5777b817c3f712685bf587be00e086796a5455662d5",
    "tau_grid": "8dc0100c7d3f128a563c48c272b4645d4b674abb050b2720f2be866c1d9fa652",
    "Q_grid": "dfd451a375a42c3427542b0b6e3a05040d2b03c54a558e2afe9e24340fa15b1e",
    "thermostat_starts": "a8428d56d1ebb21a7c750c5c42cc300cdb8980b49f06c8be932277614278e6de",
}
CANONICAL_COMMON_INPUT_FINGERPRINT = (
    "8703e78a5a8baf0200b92b8faa517199d8d9264978417d241d8220b8328a5da4"
)
CANONICAL_COMPONENT_FINGERPRINTS = {
    "target": "5ccd0f0803212c9e53262cbc451b79fe58ff059d07d46fb4b581402784525c97",
    "geometry": "691d6e966cc3f65a75bd5cfb6bf8d7bf9fa59d606c8781de358e70053d240391",
    "starts": "3a749da4112587766bfa06a94d1764b8bae59f7fa70e89b73f0ccbf00ad54bac",
    "grid": "9ff8f85f3d53a3b75876d6086b25fb58ada11e715217d53e465f1ed79f9a8a6d",
    "seed_blocks": "5cab10fc914db7b21506aa9c32034b1d91bc7b6a3ed96f6ef9b22c0a430901f5",
    "run_lengths": "c422ddc0f1a384edfb22455b94fb23abe5aff53663e98d685037840463996db2",
    "observable_policy": "074c368c6ebd5bcc93d2647a2bae2da79d790572060e9b32cc0705ef5264a64c",
}


@dataclass(frozen=True)
class RunInputs:
    """Every independent common scientific and execution input for one run."""

    beta: float
    sigma_y: float
    dimension_x: int
    mass_matrix: Sequence[Sequence[float]]
    funnel_starts: Sequence[Sequence[float]]
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
    tau_values: Sequence[float]
    Q_values: Sequence[float]
    thermostat_starts: Sequence[float]


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
    tau_values: tuple[float, ...]
    Q_values: tuple[float, ...]
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
    coupling_difference: ThermostatDiagnostic
    thermostat: ThermostatDiagnostic


@dataclass(frozen=True)
class THCConfigurationResult:
    """Five-block aggregate for one fixed joint THC configuration."""

    tau: float
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


# ---------------------------------------------------------------------------
# Target, THC splitting, and chain simulation.
# ---------------------------------------------------------------------------


@njit(cache=False, inline="always")
def _safe_exp_grex_20260829g(exponent: float) -> float:
    """Return exp(exponent) without platform-dependent overflow exceptions."""

    if exponent > 709.782712893384:
        return math.inf
    if exponent < -745.1332191019411:
        return 0.0
    return math.exp(exponent)


@njit(cache=False, inline="always")
def _potential_energy_grex_20260829g(
    q: np.ndarray, target_parameters: np.ndarray
) -> float:
    """Centered Neal-funnel potential for q=(y,x_1,...,x_d)."""

    sigma_y_squared = target_parameters[0]
    transverse_dimension = int(target_parameters[1])
    beta = target_parameters[2]
    y = q[0]
    squared_radius = 0.0
    for coordinate in range(1, q.size):
        squared_radius += q[coordinate] * q[coordinate]
    inverse_scale = _safe_exp_grex_20260829g(-y)
    conditional_energy = (
        0.0 if squared_radius == 0.0 else 0.5 * inverse_scale * squared_radius
    )
    return (
        0.5 * y * y / sigma_y_squared
        + 0.5 * transverse_dimension * y / beta
        + conditional_energy
    )


@njit(cache=False, inline="always")
def _target_gradient_grex_20260829g(
    q: np.ndarray, target_parameters: np.ndarray
) -> np.ndarray:
    """Exact gradient of the centered Neal-funnel potential."""

    sigma_y_squared = target_parameters[0]
    transverse_dimension = int(target_parameters[1])
    beta = target_parameters[2]
    y = q[0]
    squared_radius = 0.0
    for coordinate in range(1, q.size):
        squared_radius += q[coordinate] * q[coordinate]
    inverse_scale = _safe_exp_grex_20260829g(-y)
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


@njit(cache=False, inline="always")
def _kinetic_energy_grex_20260829g(
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
def _vector_is_finite_grex_20260829g(values: np.ndarray) -> bool:
    for value in values:
        if not math.isfinite(value):
            return False
    return True


# ---------------------------------------------------------------------------
# Active conditional/dynamic-THC epoch.
#
# This conditional implementation is independent of the separate scalar
# EXP_TANH tuner.  Every public alias below points to this native epoch.
# ---------------------------------------------------------------------------


@njit(cache=False, inline="always")
def _conditional_thc_state_is_finite_grex_20260829(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
) -> bool:
    return (
        math.isfinite(s)
        and math.isfinite(p_s)
        and _vector_is_finite_grex_20260829g(q)
        and _vector_is_finite_grex_20260829g(p)
    )


@njit(cache=False)
def _conditional_thc_hamiltonian_grex_20260829(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    beta: float,
    tau: float,
    Q: float,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> float:
    """Target-informed funnel Hamiltonian, up to an additive constant."""

    if not mass_is_identity:
        return math.inf
    potential = _potential_energy_grex_20260829g(q, target_parameters)
    if (
        not math.isfinite(potential)
        or not math.isfinite(s)
        or not math.isfinite(p_s)
        or not _vector_is_finite_grex_20260829g(p)
    ):
        return math.inf
    exp_s = _safe_exp_grex_20260829g(s)
    if not math.isfinite(exp_s):
        return math.inf
    transverse_momentum_squared = 0.0
    for coordinate in range(1, p.size):
        transverse_momentum_squared += p[coordinate] * p[coordinate]
    difference = s - q[0]
    transverse_dimension = q.size - 1
    energy = (
        potential
        + 0.5 * p[0] * p[0]
        + 0.5 * exp_s * transverse_momentum_squared
        + 0.5 * p_s * p_s / Q
        + 0.5 * difference * difference / (tau * tau)
        - 0.5 * transverse_dimension * s / beta
    )
    return energy if math.isfinite(energy) else math.inf


@njit(cache=False, nogil=True)
def _conditional_thc_proposal_grex_20260829(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    current_gradient: np.ndarray,
    beta: float,
    tau: float,
    Q: float,
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
    """Apply L cached-gradient V/Ay/As/Ax/As/Ay/V steps."""

    proposed_q = q.copy()
    proposed_p = p.copy()
    proposed_s = s
    proposed_p_s = p_s
    proposed_gradient = current_gradient.copy()
    gradient_calls = 0
    if not mass_is_identity:
        return (
            proposed_q, proposed_p, proposed_s, proposed_p_s,
            proposed_gradient, gradient_calls, False,
        )

    half_step = 0.5 * epsilon
    inverse_tau_squared = 1.0 / (tau * tau)
    transverse_dimension = proposed_q.size - 1
    metric_correction_force = 0.5 * transverse_dimension / beta

    # Initial V(epsilon/2): target, s-y tracking, and log-determinant term.
    difference_force = (proposed_s - proposed_q[0]) * inverse_tau_squared
    proposed_p[0] -= half_step * (
        proposed_gradient[0] - difference_force
    )
    for coordinate in range(1, proposed_q.size):
        proposed_p[coordinate] -= half_step * proposed_gradient[coordinate]
    proposed_p_s -= half_step * (
        difference_force - metric_correction_force
    )
    if not _conditional_thc_state_is_finite_grex_20260829(
        proposed_q, proposed_p, proposed_s, proposed_p_s
    ):
        return (
            proposed_q, proposed_p, proposed_s, proposed_p_s,
            proposed_gradient, gradient_calls, False,
        )

    for step in range(L):
        # A_y(epsilon/2), then A_s(epsilon/2).
        proposed_q[0] += half_step * proposed_p[0]
        proposed_s += half_step * proposed_p_s / Q

        # A_x(epsilon): transverse drift plus exact kinetic feedback on p_s.
        exp_s = _safe_exp_grex_20260829g(proposed_s)
        if not math.isfinite(exp_s):
            return (
                proposed_q, proposed_p, proposed_s, proposed_p_s,
                proposed_gradient, gradient_calls, False,
            )
        transverse_momentum_squared = 0.0
        for coordinate in range(1, proposed_q.size):
            transverse_momentum_squared += (
                proposed_p[coordinate] * proposed_p[coordinate]
            )
        kinetic_feedback = 0.5 * exp_s * transverse_momentum_squared
        if not math.isfinite(kinetic_feedback):
            return (
                proposed_q, proposed_p, proposed_s, proposed_p_s,
                proposed_gradient, gradient_calls, False,
            )
        for coordinate in range(1, proposed_q.size):
            proposed_q[coordinate] += (
                epsilon * exp_s * proposed_p[coordinate]
            )
        proposed_p_s -= epsilon * kinetic_feedback

        # Second A_s(epsilon/2), then second A_y(epsilon/2).
        proposed_s += half_step * proposed_p_s / Q
        proposed_q[0] += half_step * proposed_p[0]

        # One new target gradient at each trajectory endpoint.
        proposed_gradient = _target_gradient_grex_20260829g(
            proposed_q, target_parameters
        )
        gradient_calls += 1
        if not _vector_is_finite_grex_20260829g(proposed_gradient):
            return (
                proposed_q, proposed_p, proposed_s, proposed_p_s,
                proposed_gradient, gradient_calls, False,
            )

        kick_duration = half_step if step == L - 1 else epsilon
        difference_force = (
            (proposed_s - proposed_q[0]) * inverse_tau_squared
        )
        proposed_p[0] -= kick_duration * (
            proposed_gradient[0] - difference_force
        )
        for coordinate in range(1, proposed_q.size):
            proposed_p[coordinate] -= (
                kick_duration * proposed_gradient[coordinate]
            )
        proposed_p_s -= kick_duration * (
            difference_force - metric_correction_force
        )
        if not _conditional_thc_state_is_finite_grex_20260829(
            proposed_q, proposed_p, proposed_s, proposed_p_s
        ):
            return (
                proposed_q, proposed_p, proposed_s, proposed_p_s,
                proposed_gradient, gradient_calls, False,
            )

    return (
        proposed_q, proposed_p, proposed_s, proposed_p_s,
        proposed_gradient, gradient_calls, True,
    )


@njit(cache=False, nogil=True)
def _simulate_conditional_thc_chain_grex_20260829(
    standard_momenta: np.ndarray,
    thermostat_standard_momenta: np.ndarray,
    acceptance_uniforms: np.ndarray,
    q_initial: np.ndarray,
    s_initial: float,
    beta: float,
    tau: float,
    Q: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    """Run one refreshed-momentum, persistent-s conditional THC chain."""

    total_iterations = n_warmup + n_retained
    dimension = q_initial.size
    samples = np.empty((n_retained, dimension), dtype=np.float64)
    thermostat_samples = np.empty(n_retained, dtype=np.float64)
    q = q_initial.copy()
    s = s_initial
    current_gradient = _target_gradient_grex_20260829g(q, target_parameters)
    warmup_gradient_calls = 1 if n_warmup > 0 else 0
    retained_gradient_calls = 0 if n_warmup > 0 else 1
    accepted_retained = 0
    warmup_nonfinite = 0
    retained_nonfinite = 0
    inverse_sqrt_beta = 1.0 / math.sqrt(beta)
    thermostat_momentum_scale = math.sqrt(Q / beta)

    for iteration in range(total_iterations):
        p = np.empty(dimension, dtype=np.float64)
        p[0] = inverse_sqrt_beta * standard_momenta[iteration, 0]
        transverse_scale = (
            _safe_exp_grex_20260829g(-0.5 * s) * inverse_sqrt_beta
        )
        for coordinate in range(1, dimension):
            p[coordinate] = (
                transverse_scale * standard_momenta[iteration, coordinate]
            )
        p_s = (
            thermostat_momentum_scale
            * thermostat_standard_momenta[iteration]
        )
        initial_energy = _conditional_thc_hamiltonian_grex_20260829(
            q, p, s, p_s, beta, tau, Q, target_parameters,
            inverse_mass_matrix, mass_is_identity,
        )
        (
            proposed_q, proposed_p, proposed_s, proposed_p_s,
            proposed_gradient, proposal_gradient_calls, proposal_is_finite,
        ) = _conditional_thc_proposal_grex_20260829(
            q, p, s, p_s, current_gradient, beta, tau, Q, epsilon, L,
            target_parameters, inverse_mass_matrix, mass_is_identity,
        )

        if iteration < n_warmup:
            warmup_gradient_calls += proposal_gradient_calls
        else:
            retained_gradient_calls += proposal_gradient_calls

        # Full physical and thermostat momentum flip makes R o Phi involutive.
        if proposal_is_finite:
            for coordinate in range(dimension):
                proposed_p[coordinate] = -proposed_p[coordinate]
            proposed_p_s = -proposed_p_s

        accepted = False
        if proposal_is_finite and math.isfinite(initial_energy):
            proposed_energy = _conditional_thc_hamiltonian_grex_20260829(
                proposed_q, proposed_p, proposed_s, proposed_p_s,
                beta, tau, Q, target_parameters,
                inverse_mass_matrix, mass_is_identity,
            )
            delta_hamiltonian = proposed_energy - initial_energy
            if math.isfinite(delta_hamiltonian):
                uniform = acceptance_uniforms[iteration]
                log_uniform = -math.inf if uniform <= 0.0 else math.log(uniform)
                accepted = log_uniform < -beta * delta_hamiltonian
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
        samples, thermostat_samples, accepted_retained,
        warmup_nonfinite, retained_nonfinite,
        warmup_gradient_calls, retained_gradient_calls,
    )


def _reference_conditional_thc_proposal(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    current_gradient: np.ndarray,
    beta: float,
    tau: float,
    Q: float,
    epsilon: float,
    L: int,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, int, bool]:
    """Independent Python oracle for the active conditional split."""

    reference_q = np.asarray(q, dtype=np.float64).copy()
    reference_p = np.asarray(p, dtype=np.float64).copy()
    reference_s = float(s)
    reference_p_s = float(p_s)
    reference_gradient = np.asarray(current_gradient, dtype=np.float64).copy()
    if not mass_is_identity:
        return (
            reference_q, reference_p, reference_s, reference_p_s,
            reference_gradient, 0, False,
        )
    half_step = 0.5 * epsilon
    inverse_tau_squared = 1.0 / (tau * tau)
    d_x = reference_q.size - 1
    determinant_force = 0.5 * d_x / beta

    force_difference = (reference_s - reference_q[0]) * inverse_tau_squared
    reference_p[0] -= half_step * (reference_gradient[0] - force_difference)
    reference_p[1:] -= half_step * reference_gradient[1:]
    reference_p_s -= half_step * (force_difference - determinant_force)

    gradient_calls = 0
    for step in range(L):
        reference_q[0] += half_step * reference_p[0]
        reference_s += half_step * reference_p_s / Q
        exp_s = math.exp(reference_s)
        transverse_momentum_squared = float(reference_p[1:] @ reference_p[1:])
        reference_q[1:] += epsilon * exp_s * reference_p[1:]
        reference_p_s -= 0.5 * epsilon * exp_s * transverse_momentum_squared
        reference_s += half_step * reference_p_s / Q
        reference_q[0] += half_step * reference_p[0]

        sigma_y_squared = float(target_parameters[0])
        target_d = int(target_parameters[1])
        target_beta = float(target_parameters[2])
        y = float(reference_q[0])
        inverse_scale = math.exp(-y)
        squared_radius = float(reference_q[1:] @ reference_q[1:])
        reference_gradient = np.empty_like(reference_q)
        reference_gradient[0] = (
            y / sigma_y_squared + 0.5 * target_d / target_beta
            - 0.5 * inverse_scale * squared_radius
        )
        reference_gradient[1:] = inverse_scale * reference_q[1:]
        gradient_calls += 1
        kick = half_step if step == L - 1 else epsilon
        force_difference = (
            (reference_s - reference_q[0]) * inverse_tau_squared
        )
        reference_p[0] -= kick * (
            reference_gradient[0] - force_difference
        )
        reference_p[1:] -= kick * reference_gradient[1:]
        reference_p_s -= kick * (force_difference - determinant_force)

    finite = bool(
        np.all(np.isfinite(reference_q))
        and np.all(np.isfinite(reference_p))
        and np.all(np.isfinite(reference_gradient))
        and math.isfinite(reference_s)
        and math.isfinite(reference_p_s)
    )
    return (
        reference_q, reference_p, reference_s, reference_p_s,
        reference_gradient, gradient_calls, finite,
    )


def _validate_conditional_compiled_reference_microcheck() -> str:
    """Verify native proposal and Hamiltonian against independent oracles."""

    target_parameters = np.asarray([9.0, 5.0, 1.0], dtype=np.float64)
    identity = np.eye(6, dtype=np.float64)
    cases = (
        (0.23, -0.16, 1.0, 0.75, 0.25, 0.004, 3),
        (-0.2, 0.15, 1.3, 0.5, 0.7, 0.002, 7),
    )
    for case_index, (s, p_s, beta, tau, Q, epsilon, L) in enumerate(
        cases, start=1
    ):
        parameters = np.asarray([9.0, 5.0, beta], dtype=np.float64)
        q = np.asarray([0.17, -0.11, 0.08, 0.04, -0.09, 0.13])
        p = np.asarray([0.31, -0.24, 0.07, 0.12, -0.05, 0.09])
        gradient = _target_gradient(q, parameters)
        native = _conditional_thc_proposal_grex_20260829(
            q, p, s, p_s, gradient, beta, tau, Q, epsilon, L,
            parameters, identity, True,
        )
        reference = _reference_conditional_thc_proposal(
            q, p, s, p_s, gradient, beta, tau, Q, epsilon, L,
            parameters, identity, True,
        )
        arrays_match = all(
            np.allclose(native[index], reference[index], rtol=5e-12, atol=5e-13)
            for index in (0, 1, 4)
        )
        scalars_match = all(
            math.isclose(float(native[index]), float(reference[index]),
                         rel_tol=5e-12, abs_tol=5e-13)
            for index in (2, 3)
        )
        if (
            not arrays_match or not scalars_match
            or native[5] != reference[5]
            or bool(native[6]) != bool(reference[6])
        ):
            raise RuntimeError(
                "Conditional THC compiled/reference proposal mismatch in "
                f"microcheck case {case_index}."
            )

    q = np.asarray([0.8, -0.7, 0.2, -0.1, 0.05, 0.3])
    p = np.asarray([0.2, -0.15, 0.08, 0.04, -0.06, 0.11])
    s = 0.31
    p_s = -0.27
    beta = 1.2
    tau = 0.75
    Q = 0.25
    parameters = np.asarray([9.0, 5.0, beta], dtype=np.float64)
    y = float(q[0])
    potential = (
        0.5 * y * y / 9.0 + 2.5 * y / beta
        + 0.5 * math.exp(-y) * float(q[1:] @ q[1:])
    )
    expected = (
        potential + 0.5 * p[0] * p[0]
        + 0.5 * math.exp(s) * float(p[1:] @ p[1:])
        + 0.5 * p_s * p_s / Q
        + 0.5 * (s - y) ** 2 / (tau * tau)
        - 2.5 * s / beta
    )
    received = _conditional_thc_hamiltonian_grex_20260829(
        q, p, s, p_s, beta, tau, Q, parameters, identity, True
    )
    if not math.isclose(received, expected, rel_tol=5e-14, abs_tol=5e-14):
        raise RuntimeError("Conditional THC Hamiltonian micro-oracle failed.")
    return (
        "PASS: conditional THC compiled proposal and Hamiltonian match "
        "independent same-host Python oracles"
    )


# Active public aliases.  Validation explicitly checks these identities.
_thc_hamiltonian = _conditional_thc_hamiltonian_grex_20260829
_thc_proposal = _conditional_thc_proposal_grex_20260829
_simulate_thc_chain = _simulate_conditional_thc_chain_grex_20260829
_reference_thc_proposal = _reference_conditional_thc_proposal
_validate_compiled_reference_microcheck = (
    _validate_conditional_compiled_reference_microcheck
)
_potential_energy = _potential_energy_grex_20260829g
_target_gradient = _target_gradient_grex_20260829g
_kinetic_energy = _kinetic_energy_grex_20260829g
_vector_is_finite = _vector_is_finite_grex_20260829g
_thc_state_is_finite = _conditional_thc_state_is_finite_grex_20260829


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


# ---------------------------------------------------------------------------
# Canonical diagnostics.  These functions must be copied exactly into HMC.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Observables, block analysis, and representative selection.
# ---------------------------------------------------------------------------


def _observable_specification(
    beta: float,
    sigma_y: float,
    transverse_dimension: int,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Exact moments for all physical linears and unique quadratics.

    Under exp(-beta*U), y~N(0,sigma_y^2/beta) and
    x_i|y~N(0,exp(y)/beta).  The returned ordering is identical to
    _physical_observables: all coordinates first, then q_i*q_j for i<=j.
    """

    dimension = 1 + int(transverse_dimension)
    variance_y = sigma_y * sigma_y / beta
    exp_half_variance = math.exp(0.5 * variance_y)
    exp_twice_variance = math.exp(2.0 * variance_y)
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
                    (3.0 * exp_twice_variance - math.exp(variance_y))
                    / (beta * beta)
                )
            else:
                means.append(0.0)
                variances.append(exp_twice_variance / (beta * beta))

    exact_means = np.asarray(means, dtype=np.float64)
    exact_variances = np.asarray(variances, dtype=np.float64)
    if (
        not np.all(np.isfinite(exact_variances))
        or np.any(exact_variances <= 0.0)
    ):
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
    exact_mean: float,
    exact_second_moment: float,
) -> ThermostatDiagnostic:
    """Return one descriptive exact first/second-moment check."""

    values = np.asarray(thermostat_samples, dtype=np.float64)
    empirical_mean = float(np.mean(values))
    empirical_second_moment = float(np.mean(values * values))
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
    tau: float,
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
        tau=float(tau),
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
            "No THC configuration has a defined five-block tuning score."
        )

    # Exact ties: larger minimum block score, fewer gradients/draw, then
    # smaller epsilon, L, tau, and Q.  No near-tie tolerance or
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
            -result.tau,
            -result.Q,
        ),
    )


# ---------------------------------------------------------------------------
# Validation, execution, and reporting.
# ---------------------------------------------------------------------------


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
    sigma_y: float,
    transverse_dimension: int,
    mass_matrix: np.ndarray,
    funnel_starts: np.ndarray,
    epsilon_values: Sequence[float],
    L_values: Sequence[int],
    seed_blocks: Sequence[Sequence[int]],
    n_warmup: int,
    n_retained: int,
    observable_policy: str,
) -> dict[str, object]:
    coordinate_names = ["y"] + [
        f"x{index}" for index in range(1, transverse_dimension + 1)
    ]
    ordered_observables = list(coordinate_names)
    for first in range(1 + transverse_dimension):
        for second in range(first, 1 + transverse_dimension):
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
            "potential": "y^2/(2*sigma_y^2)+d_x*y/(2*beta)+exp(-y)*sum(x_i^2)/2",
        },
        "geometry": {"mass_matrix": mass_matrix.tolist()},
        "starts": {
            "coordinate_system": "(y,z_1,...,z_d), x_i=exp(y/2)*z_i/sqrt(beta)",
            "funnel_starts": funnel_starts.tolist(),
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
            "ordered_observables": ordered_observables,
            "thermostat_is_scored": False,
            "target_specific_nonlinear_observables_are_scored": False,
        },
    }


def _prepare_inputs(inputs: RunInputs) -> PreparedInputs:
    """Validate inputs and derive the complete centered-funnel specification."""

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
    sigma_y = positive_real("sigma_y", inputs.sigma_y)
    transverse_dimension = _require_integer(
        "dimension_x", inputs.dimension_x, 1
    )
    dimension = 1 + transverse_dimension
    sigma_y_squared = sigma_y * sigma_y
    if not math.isfinite(sigma_y_squared) or sigma_y_squared <= 0.0:
        raise ValueError("Squared funnel scale must be finite and positive.")
    target_parameters = np.ascontiguousarray(
        np.asarray(
            [sigma_y_squared, float(transverse_dimension), beta],
            dtype=np.float64,
        )
    )

    try:
        mass_matrix = np.asarray(inputs.mass_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"mass_matrix must be a numeric {dimension}-by-{dimension} matrix."
        ) from error
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
        funnel_starts = np.asarray(
            inputs.funnel_starts, dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError("funnel_starts must be a numeric matrix.") from error
    if funnel_starts.ndim != 2 or funnel_starts.shape[1] != dimension:
        raise ValueError(
            f"funnel_starts must have shape (n_chains, {dimension}); "
            f"received {funnel_starts.shape}."
        )
    if funnel_starts.shape[0] < 2:
        raise ValueError("At least two chain starts are required.")
    if not np.all(np.isfinite(funnel_starts)):
        raise ValueError("funnel_starts must contain only finite values.")
    funnel_starts = np.ascontiguousarray(funnel_starts.copy())
    n_chains = int(funnel_starts.shape[0])

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
        variance_y = sigma_y_squared / beta
        variance_x = math.exp(0.5 * variance_y) / beta
        target_variances = np.ascontiguousarray(
            np.asarray(
                [variance_y] + [variance_x] * transverse_dimension,
                dtype=np.float64,
            )
        )
        physical_starts = _latent_to_physical(
            funnel_starts,
            beta,
        )
    if not np.all(np.isfinite(target_parameters)):
        raise ValueError("Derived Neal-funnel target parameters are nonfinite.")
    if not np.all(np.isfinite(target_variances)) or np.any(
        target_variances <= 0.0
    ):
        raise ValueError(
            "Derived physical marginal variances must be finite and positive."
        )
    if not np.all(np.isfinite(physical_starts)):
        raise ValueError(
            "Derived physical starts are nonfinite; check beta and funnel_starts."
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
            "This conditional-THC tuner supports only "
            f"coupling_family='{SUPPORTED_COUPLING_FAMILY}'."
        )

    tau_values = _normalize_method_grid(
        "tau_values", inputs.tau_values, allow_zero=False
    )
    Q_values = _normalize_method_grid(
        "Q_values", inputs.Q_values, allow_zero=False
    )
    for Q in Q_values:
        thermostat_momentum_variance = Q / common.beta
        if (
            not math.isfinite(thermostat_momentum_variance)
            or thermostat_momentum_variance <= 0.0
        ):
            raise ValueError(
                "Each Q must give finite, strictly positive thermostat "
                "momentum variance Q/beta."
            )
    for tau in tau_values:
        coupling_variance = tau * tau / common.beta
        if not math.isfinite(coupling_variance) or coupling_variance <= 0.0:
            raise ValueError(
                "Each tau must give finite positive conditional variance "
                "tau**2/beta for s-y."
            )

    if not common.mass_is_identity:
        raise ValueError(
            "The conditional funnel geometry is defined here only for the "
            "agreed identity base mass M0=I_6."
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
        "tau_grid": {"tau_values": list(tau_values)},
        "Q_grid": {"Q_values": list(Q_values)},
        "thermostat_starts": {
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
        tau_values=tau_values,
        Q_values=Q_values,
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
                if script_path.name == "neals_funnel_conditional_thc_tuner.py"
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
            _thc_hamiltonian,
            _thc_state_is_finite,
            _thc_proposal,
            _simulate_thc_chain,
            _reference_thc_proposal,
            _validate_compiled_reference_microcheck,
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


def _format_efficiency_grids(
    results: Sequence[THCConfigurationResult],
    prepared: PreparedTHCInputs,
) -> str:
    """Format one common epsilon-L efficiency table per thermostat tuple."""

    common = prepared.common
    lookup = {
        (
            result.tau,
            result.Q,
            result.epsilon,
            result.L,
        ): result
        for result in results
    }
    column_width = 14
    block_count = (
        len(results[0].block_results) if results else 0
    )
    header_label = "epsilon \\ L"
    sections: list[str] = []
    for tau in prepared.tau_values:
        for Q in prepared.Q_values:
            sections.extend(
                [
                    (
                        f"{block_count}-block GM worst-observable "
                        "efficiency grid: "
                        f"tau={tau:.12g}, Q={Q:.8g}"
                    ),
                    f"{header_label:>12}"
                    + "".join(
                        f"{L:>{column_width}d}"
                        for L in common.L_values
                    ),
                ]
            )
            for epsilon in common.epsilon_values:
                cells: list[str] = []
                for L in common.L_values:
                    value = lookup[
                        (
                            float(tau),
                            float(Q),
                            float(epsilon),
                            int(L),
                        )
                    ].geometric_mean_efficiency
                    cells.append(
                        f"{_format_number(value, '.6e'):>{column_width}}"
                    )
                sections.append(
                    f"{epsilon:>12.6g}" + "".join(cells)
                )
            sections.append("")
    return "\n".join(sections).rstrip()


def _boundary_labels(
    representative: THCConfigurationResult,
    prepared: PreparedTHCInputs,
) -> tuple[str, ...]:
    """Return declared grid boundaries occupied by the selected point."""

    common = prepared.common
    labels: list[str] = []
    fields = (
        ("tau", representative.tau, prepared.tau_values),
        ("Q", representative.Q, prepared.Q_values),
        ("epsilon", representative.epsilon, common.epsilon_values),
        ("L", representative.L, common.L_values),
    )
    for name, value, grid in fields:
        if len(grid) <= 1:
            continue
        if value == min(grid):
            labels.append(f"{name}=lower")
        if value == max(grid):
            labels.append(f"{name}=upper")
    return tuple(labels)


def _build_report(
    *,
    configuration_results: Sequence[THCConfigurationResult],
    representative: THCConfigurationResult,
    auxiliary_momentum_moments: Sequence[float],
    prepared: PreparedTHCInputs,
    self_test_messages: Sequence[str],
) -> str:
    common = prepared.common
    lines: list[str] = []
    mode_note = (
        "CANONICAL / COMPARABLE"
        if common.run_mode == "CANONICAL"
        else "CUSTOM / NON-COMPARABLE"
    )
    total_configurations = (
        len(prepared.tau_values)
        * len(prepared.Q_values)
        * len(common.epsilon_values)
        * len(common.L_values)
    )
    lines.extend(
        [
            (
                "FAIR-PLAYGROUND CONDITIONAL/DYNAMIC THC TUNING REPORT: "
                f"{common.dimension}D CENTERED NEAL'S FUNNEL"
            ),
            "=" * 72,
            f"Run label: {common.run_label}",
            f"Run mode: {mode_note}",
            f"Common-input schema version: {COMMON_INPUT_SCHEMA_VERSION}",
            (
                "Canonical common inputs match: "
                f"{not common.canonical_mismatches and common.common_input_fingerprint == CANONICAL_COMMON_INPUT_FINGERPRINT}"
            ),
            (
                "Canonical expected common-input fingerprint: "
                + CANONICAL_COMMON_INPUT_FINGERPRINT
            ),
            (
                "Actual common-input fingerprint: "
                + common.common_input_fingerprint
            ),
            "Canonical common-component mismatches: "
            + (", ".join(common.canonical_mismatches) or "none"),
            (
                "THC-method input schema version: "
                + THC_METHOD_INPUT_SCHEMA_VERSION
            ),
            (
                "Canonical THC-method inputs match: "
                f"{not prepared.canonical_method_mismatches and prepared.method_input_fingerprint == CANONICAL_THC_METHOD_INPUT_FINGERPRINT}"
            ),
            (
                "Canonical expected THC-method fingerprint: "
                + CANONICAL_THC_METHOD_INPUT_FINGERPRINT
            ),
            (
                "Actual THC-method fingerprint: "
                + prepared.method_input_fingerprint
            ),
            "Canonical THC-component mismatches: "
            + (
                ", ".join(prepared.canonical_method_mismatches)
                or "none"
            ),
            f"Input-preparation version: {INPUT_PREPARATION_VERSION}",
            (
                "Input-preparation source fingerprint: "
                + _input_preparation_fingerprint()
            ),
            f"Shared protocol version: {SHARED_PROTOCOL_VERSION}",
            (
                "Shared protocol fingerprint: "
                + _shared_protocol_fingerprint(common)
            ),
            f"Diagnostics version: {DIAGNOSTICS_VERSION}",
            (
                "Diagnostics source fingerprint: "
                + _diagnostics_fingerprint()
            ),
            (
                "Common RNG/observable/score pipeline version: "
                + COMMON_PIPELINE_VERSION
            ),
            (
                "Common pipeline source fingerprint: "
                + _common_pipeline_fingerprint()
            ),
            f"THC implementation version: {THC_IMPLEMENTATION_VERSION}",
            (
                "THC implementation source fingerprint: "
                + _thc_implementation_fingerprint()
            ),
            (
                "Expected THC implementation source fingerprint: "
                + EXPECTED_THC_IMPLEMENTATION_FINGERPRINT
            ),
            (
                "Source-parity verification mode: "
                + _source_verification_mode()
            ),
            f"Random-stream version: {RANDOM_STREAM_VERSION}",
            f"Numba available: {NUMBA_AVAILABLE}",
            f"Numba cache mode: {NUMBA_CACHE_MODE}",
            f"Numba cache directory: {NUMBA_CACHE_DIRECTORY}",
            f"Portable native preflight version: {THC_RUNTIME_PREFLIGHT_VERSION}",
            f"Numerical runtime: {_runtime_environment_summary()}",
            (
                "Tuning-block-1 random-stream SHA-256: "
                + _first_tuning_block_stream_digest(prepared)
            ),
            f"Worker threads: {common.n_workers}",
            f"Run self-tests: {common.run_self_tests}",
            f"Print progress: {common.print_progress}",
            f"Print full report: {common.print_report}",
            f"Requested output directory: {common.output_directory!r}",
            f"Output filename: {common.output_filename}",
            f"Allow overwrite: {common.allow_overwrite}",
            "",
            "TARGET, SHARED GEOMETRY, AND THC HAMILTONIAN",
            "-" * 72,
            f"physical dimension={common.dimension}, transverse d_x={common.transverse_dimension}, beta={common.beta:g}",
            f"sigma_y={common.sigma_y:.12g}; q=(y,x1,...,x{common.transverse_dimension})",
            "U(y,x)=y^2/(2*sigma_y^2)+d_x*y/(2*beta)+exp(-y)*||x||^2/2",
            "grad_y U=y/sigma_y^2+d_x/(2*beta)-exp(-y)*||x||^2/2",
            "grad_x U=exp(-y)*x",
            "mass matrix M0="
            + np.array2string(
                common.mass_matrix, precision=12, separator=", "
            ),
            f"M0 is identity: {common.mass_is_identity}",
            "inverse mass M0^-1="
            + np.array2string(
                common.inverse_mass_matrix,
                precision=12,
                separator=", ",
            ),
            "diag(Cov_beta(y,x1,...,x_d))="
            + np.array2string(
                common.target_variances,
                precision=12,
                separator=", ",
            ),
            "Base mass convention: M0=I; y is unscaled and only x uses h_x(s).",
            "h_x(s)=exp(-s/2); G(s)=diag(1,exp(-s)I_d_x).",
            "Tracking coupling C(y,s)=(s-y)^2/(2*tau^2).",
            "Invariant conditional law: s|y~N(y,tau^2/beta).",
            f"coupling family={prepared.coupling_family}",
            "Competition track: target-informed method-native conditional geometry; it must be labelled separately from shared-geometry scalar THC.",
            (
                "H=U+p_y^2/2+exp(s)||p_x||^2/2+p_s^2/(2Q)"
                "+(s-y)^2/(2*tau^2)-d_x*s/(2*beta), up to constants"
            ),
            f"The determinant factor uses transverse d_x={common.transverse_dimension}, not total D={common.dimension}.",
            (
                "Integrator: symmetric exact-subflow "
                "V/2-A_y/2-A_s/2-A_x-A_s/2-A_y/2-V/2, "
                "L steps, terminal (p,p_s) flip, full-state Metropolis correction"
            ),
            (
                "Refresh law: p_y~N(0,1/beta), "
                "p_x|s~N(0,exp(-s)I_d_x/beta), "
                "p_s~N(0,Q/beta); q and s persist"
            ),
            "",
            "ENTERED JOINT TUNING DESIGN",
            "-" * 72,
            (
                f"tau values ({len(prepared.tau_values)}): "
                f"{list(prepared.tau_values)}"
            ),
            (
                f"Q values ({len(prepared.Q_values)}): "
                f"{list(prepared.Q_values)}"
            ),
            (
                f"epsilon values ({len(common.epsilon_values)}): "
                f"{list(common.epsilon_values)}"
            ),
            (
                f"L values ({len(common.L_values)}): "
                f"{list(common.L_values)}"
            ),
            f"Full Cartesian fixed configurations: {total_configurations}",
            (
                f"Blocks: {len(common.seed_blocks)}; "
                f"chains/block: {common.n_chains}"
            ),
            (
                f"Warmup/chain: {common.n_warmup}; "
                f"retained/chain: {common.n_retained}"
            ),
            (
                "Every fixed joint configuration is run on every entered "
                "seed block; there is no successive halving or screening."
            ),
            (
                "The same three parameter-independent random substreams are "
                "reused across configurations within every seed block."
            ),
            (
                "The HMC tuning result is not an input to this program and "
                "does not alter the THC grid."
            ),
            "Championship seeds are absent from this tuner.",
            "",
            "TUNING SEED BLOCKS",
        ]
    )
    for block_index, block in enumerate(
        common.seed_blocks, start=1
    ):
        lines.append(f"block {block_index}: {list(block)}")

    lines.extend(
        ["", "INITIAL CONDITIONS (printed exactly as used)"]
    )
    for chain_index in range(common.n_chains):
        physical_start = common.physical_starts[chain_index]
        conditional_radius = (
            common.beta
            * math.exp(-physical_start[0])
            * float(physical_start[1:] @ physical_start[1:])
        )
        start_potential = _potential_energy(
            physical_start, common.target_parameters
        )
        lines.append(
            f"chain {chain_index + 1}: entered (y,z1,...,z_d)="
            + np.array2string(
                common.funnel_starts[chain_index],
                precision=6,
                separator=", ",
            )
            + "; q=(y,x1,...,x_d)="
            + np.array2string(
                physical_start,
                precision=12,
                separator=", ",
            )
            + f"; beta*exp(-y)*||x||^2={conditional_radius:.12g}; U={start_potential:.12g}"
            + f"; s={prepared.thermostat_starts[chain_index]:.12g}"
        )
    lines.extend(
        [
            (
                "Momentum starts: none; p and p_s are freshly drawn from "
                "their conditional laws at every transition."
            ),
            "",
            "MINIMAL AUXILIARY MOMENTUM DIAGNOSTIC (DESCRIPTIVE ONLY)",
            "-" * 72,
            (
                "Quantity: pooled retained "
                "mean of squared standard-normal momentum refresh variates; "
                "exact value=1."
            ),
            (
                "For this conditional sampler the quantity is "
                "beta*(p_y^2+exp(s)*||p_x||^2)/D "
                "= beta*p^T*G(s)^(-1)*p/D."
            ),
            (
                "This is not a score observable, IAT input, threshold, "
                "or selection gate."
            ),
        ]
    )
    for block_index, value in enumerate(
        auxiliary_momentum_moments, start=1
    ):
        lines.append(
            f"block {block_index}: empirical={value:.12e}, "
            "exact=1.000000000000e+00, "
            f"error={value - 1.0:+.12e}"
        )

    lines.extend(
        [
            "",
            "DIAGNOSTICS AND SCORE DEFINITIONS",
            "-" * 72,
            f"Observable policy input: {common.observable_policy}",
            (
                f"Required physical observables "
                f"({len(common.observable_names)}): "
                + ", ".join(common.observable_names)
            ),
            "Exact physical observable means (same order): "
            + np.array2string(
                common.exact_observable_means,
                precision=12,
                separator=", ",
            ),
            "Exact physical observable target SDs (same order): "
            + np.array2string(
                common.exact_observable_sds,
                precision=12,
                separator=", ",
            ),
            (
                "No target-specific nonlinear quantity is an IAT, R-hat, "
                "moment-score, or efficiency observable."
            ),
            (
                "Thermostat diagnostics for each tau: E[s-y]=0, "
                "E[(s-y)^2]=tau^2/beta, E[s]=0, and "
                "E[s^2]=(sigma_y^2+tau^2)/beta; "
                "descriptive only, no IAT, efficiency, threshold, or gate."
            ),
            (
                "IAT: multi-chain Geyer initial-positive sequence followed "
                "by initial-monotone sequence."
            ),
            (
                "Autocovariance convention: biased denominator N at every lag."
            ),
            (
                "No IAT floor is imposed. Each physical observable reports "
                "status, retained pair count, and last included lag."
            ),
            (
                "R-hat: maximum of rank-normalized split and folded "
                "rank-normalized split R-hat; raw value, no threshold."
            ),
            (
                "Moment diagnostic: |empirical-exact| divided by the exact "
                "target SD of that physical observable; descriptive only."
            ),
            (
                "Actual postwarmup target-gradient calls are counted. "
                "The current physical gradient is cached across trajectories."
            ),
            (
                "Block score=min_f "
                "1/(gradients_per_retained_draw*IAT(f))."
            ),
            (
                f"Configuration score=geometric mean of the "
                f"{len(common.seed_blocks)} block scores."
            ),
            (
                "No IAT/ESS averaging over observables or blocks is performed."
            ),
            (
                "Any required undefined physical IAT or any nonfinite "
                "trajectory makes efficiency undefined."
            ),
            (
                "Selection tie-break (exact ties only): larger minimum block "
                "score, fewer gradients/draw, then smaller epsilon, L, "
                "tau and Q."
            ),
            (
                "A selected boundary point is reported but is not rejected "
                "and does not trigger an automatic grid expansion."
            ),
            "IAT status glossary:",
            (
                "  ok_truncated_nonpositive = stopped immediately before "
                "the first nonpositive adjacent-lag pair."
            ),
            (
                "  ok_full_window = every available adjacent-lag pair was "
                "positive; the sample window was exhausted."
            ),
            (
                "  undefined_shape / undefined_insufficient_chains / "
                "undefined_insufficient_draws = required structure or length "
                "is absent."
            ),
            (
                "  undefined_nonfinite_input / undefined_nonfinite_pair = "
                "nonfinite data or correlation-pair estimate occurred."
            ),
            (
                "  undefined_zero_variance = within-chain or total variance "
                "needed by the estimator is zero/invalid."
            ),
            (
                "  undefined_no_positive_pair = the initial adjacent-lag "
                "pair was not positive."
            ),
            (
                "  undefined_nonpositive_tau = unfloored IPS/IMS produced "
                "a nonpositive final IAT."
            ),
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
            (
                "    tau       Q epsilon      L  status"
                "                         GM_eff       min_eff    median_eff"
                "       max_eff   mean_acc    max_Rhat  max_moment_SD"
                " max_abs_r2_err max_abs_s2_err nonfinite"
            ),
        ]
    )
    ordered_results = sorted(
        configuration_results,
        key=lambda item: (
            item.tau,
            item.Q,
            item.epsilon,
            item.L,
        ),
    )
    for result in ordered_results:
        maximum_abs_r2_error = max(
            abs(block.coupling_difference.second_moment_error)
            for block in result.block_results
        )
        maximum_abs_s2_error = max(
            abs(block.thermostat.second_moment_error)
            for block in result.block_results
        )
        lines.append(
            f"{result.tau:7.3f} {result.Q:7.3f} "
            f"{result.epsilon:7.4f} {result.L:6d}  "
            f"{result.status:<28} "
            f"{_format_number(result.geometric_mean_efficiency, '.6e'):>12} "
            f"{_format_number(result.minimum_block_efficiency, '.6e'):>12} "
            f"{_format_number(result.median_block_efficiency, '.6e'):>12} "
            f"{_format_number(result.maximum_block_efficiency, '.6e'):>12} "
            f"{result.mean_acceptance:10.6f} "
            f"{_format_number(result.maximum_rhat, '.6f'):>11} "
            f"{_format_number(result.maximum_standardized_moment_error, '.6e'):>14} "
            f"{maximum_abs_r2_error:14.6e} "
            f"{maximum_abs_s2_error:14.6e} "
            f"{result.total_nonfinite_trajectories:10d}"
        )

    lines.extend(
        [
            "",
            _format_efficiency_grids(
                ordered_results, prepared
            ),
            "",
            "ALL CONFIGURATION-BLOCK RESULTS",
            "-" * 72,
            (
                "    tau       Q epsilon      L block status"
                "                         accept  grad/draw  worst_obs"
                "     worst_iat_status                 worst_IAT last_lag"
                " pairs    block_eff   max_Rhat max_moment_SD"
                "     mean_s-y       E_(s-y)2   err_(s-y)2"
                "       mean_s          E_s2       E_s2_err fail_W fail_R"
            ),
        ]
    )
    for result in ordered_results:
        for wrapped_block in result.block_results:
            physical = wrapped_block.physical
            coupling_difference = wrapped_block.coupling_difference
            thermostat = wrapped_block.thermostat
            worst_diagnostic = next(
                (
                    item
                    for item in physical.observable_diagnostics
                    if item.name == physical.worst_observable
                ),
                None,
            )
            last_lag = (
                worst_diagnostic.iat.last_included_lag
                if worst_diagnostic
                else -1
            )
            pairs = (
                worst_diagnostic.iat.pair_count
                if worst_diagnostic
                else 0
            )
            worst_iat_status = (
                worst_diagnostic.iat.status
                if worst_diagnostic
                else "undefined"
            )
            lines.append(
                f"{result.tau:7.3f} {result.Q:7.3f} "
                f"{physical.epsilon:7.4f} {physical.L:6d} "
                f"{physical.block_index + 1:5d} "
                f"{physical.status:<28} "
                f"{physical.acceptance:7.4f} "
                f"{physical.gradients_per_retained_draw:10.4f} "
                f"{physical.worst_observable:<13} "
                f"{worst_iat_status:<32} "
                f"{_format_number(physical.worst_iat, '.6g'):>10} "
                f"{last_lag:8d} {pairs:5d} "
                f"{_format_number(physical.block_efficiency, '.6e'):>12} "
                f"{_format_number(physical.maximum_rhat, '.6f'):>10} "
                f"{_format_number(physical.maximum_standardized_moment_error, '.6e'):>13} "
                f"{coupling_difference.empirical_mean:12.5e} "
                f"{coupling_difference.empirical_second_moment:12.5e} "
                f"{coupling_difference.second_moment_error:+12.5e} "
                f"{thermostat.empirical_mean:12.5e} "
                f"{thermostat.empirical_second_moment:12.5e} "
                f"{thermostat.second_moment_error:+12.5e} "
                f"{physical.warmup_nonfinite_trajectories:6d} "
                f"{physical.retained_nonfinite_trajectories:6d}"
            )

    physical_blocks = tuple(
        block.physical for block in representative.block_results
    )
    block_efficiencies = np.asarray(
        [block.block_efficiency for block in physical_blocks]
    )
    boundary_labels = _boundary_labels(
        representative, prepared
    )
    lines.extend(
        [
            "",
            "SELECTED FIXED THC REPRESENTATIVE",
            "-" * 72,
            (
                f"tau={representative.tau:.12g}, Q={representative.Q:.12g}, "
                f"epsilon={representative.epsilon:.12g}, "
                f"L={representative.L}, "
                f"trajectory time T={representative.epsilon * representative.L:.12g}"
            ),
            (
                "Selected point on declared grid boundary: "
                + (", ".join(boundary_labels) or "none")
            ),
            (
                "Block efficiencies="
                + np.array2string(
                    block_efficiencies,
                    precision=12,
                    separator=", ",
                )
            ),
            (
                "Geometric mean="
                f"{representative.geometric_mean_efficiency:.12e}"
            ),
            (
                "Minimum="
                f"{representative.minimum_block_efficiency:.12e}"
            ),
            (
                "Median="
                f"{representative.median_block_efficiency:.12e}"
            ),
            (
                "Maximum="
                f"{representative.maximum_block_efficiency:.12e}"
            ),
            (
                "Mean retained acceptance="
                f"{representative.mean_acceptance:.12f}"
            ),
            (
                "Maximum raw rank/split/folded R-hat across blocks="
                f"{representative.maximum_rhat:.12f}"
            ),
            (
                "Maximum standardized physical-moment error across blocks="
                f"{representative.maximum_standardized_moment_error:.12e}"
            ),
            (
                "This representative is selected only from canonical tuning "
                "blocks. It has not seen championship seeds."
                if common.run_mode == "CANONICAL"
                else (
                    "This is a CUSTOM / NON-COMPARABLE representative; "
                    "no championship-seed claim is made."
                )
            ),
            "",
            "SELECTED REPRESENTATIVE: CONDITIONAL THERMOSTAT MOMENTS BY BLOCK",
            "-" * 72,
            (
                "block     mean_s-y exact_mean       error"
                "       E_(s-y)2     exact_r2       error"
                "       mean_s          E_s2       exact_s2       error"
            ),
        ]
    )
    for block_index, wrapped_block in enumerate(
        representative.block_results, start=1
    ):
        difference = wrapped_block.coupling_difference
        item = wrapped_block.thermostat
        lines.append(
            f"{block_index:5d} "
            f"{difference.empirical_mean:12.6e} "
            f"{difference.exact_mean:12.6e} "
            f"{difference.mean_error:+12.6e} "
            f"{difference.empirical_second_moment:12.6e} "
            f"{difference.exact_second_moment:12.6e} "
            f"{difference.second_moment_error:+12.6e} "
            f"{item.empirical_mean:12.6e} "
            f"{item.empirical_second_moment:12.6e} "
            f"{item.exact_second_moment:12.6e} "
            f"{item.second_moment_error:+12.6e}"
        )

    lines.extend(
        [
            "",
            "SELECTED REPRESENTATIVE: ALL PHYSICAL OBSERVABLES IN ALL BLOCKS",
            "-" * 72,
            (
                "block observable       IAT iat_status"
                "                     pairs last_lag first_excl"
                "        ESS   efficiency rank_Rhat fold_Rhat"
                "  max_Rhat empirical_mean    exact_mean"
                "     raw_error  std_error"
            ),
        ]
    )
    for wrapped_block in representative.block_results:
        physical = wrapped_block.physical
        for item in physical.observable_diagnostics:
            lines.append(
                f"{physical.block_index + 1:5d} {item.name:<12} "
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

    lines.extend(["", "END OF REPORT"])
    return "\n".join(lines)


def _run_chain_job(
    standard_momenta: np.ndarray,
    thermostat_standard_momenta: np.ndarray,
    uniforms: np.ndarray,
    q_initial: np.ndarray,
    s_initial: float,
    beta: float,
    tau: float,
    Q: float,
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
        tau,
        Q,
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
    tau: float,
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
            tau,
            Q,
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
    coupling_difference = _thermostat_diagnostic(
        thermostat_samples - samples[:, :, 0],
        0.0,
        tau * tau / common.beta,
    )
    thermostat = _thermostat_diagnostic(
        thermostat_samples,
        0.0,
        (common.sigma_y * common.sigma_y + tau * tau) / common.beta,
    )
    return THCBlockResult(
        physical=physical,
        coupling_difference=coupling_difference,
        thermostat=thermostat,
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
        ("_potential_energy", _potential_energy, _potential_energy_grex_20260829g),
        ("_target_gradient", _target_gradient, _target_gradient_grex_20260829g),
        ("_kinetic_energy", _kinetic_energy, _kinetic_energy_grex_20260829g),
        ("_vector_is_finite", _vector_is_finite, _vector_is_finite_grex_20260829g),
        (
            "_thc_hamiltonian",
            _thc_hamiltonian,
            _conditional_thc_hamiltonian_grex_20260829,
        ),
        (
            "_thc_state_is_finite",
            _thc_state_is_finite,
            _conditional_thc_state_is_finite_grex_20260829,
        ),
        (
            "_thc_proposal",
            _thc_proposal,
            _conditional_thc_proposal_grex_20260829,
        ),
        (
            "_simulate_thc_chain",
            _simulate_thc_chain,
            _simulate_conditional_thc_chain_grex_20260829,
        ),
    )
    problems: list[str] = []
    for public_name, active_dispatcher, expected_dispatcher in expected_dispatchers:
        if active_dispatcher is not expected_dispatcher:
            problems.append(f"{public_name} does not point to the current epoch")
            continue
        python_function = getattr(active_dispatcher, "py_func", None)
        expected_python_function = getattr(expected_dispatcher, "py_func", None)
        if python_function is None or python_function is not expected_python_function:
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
    """Run deterministic common and conditional-THC checks before tuning."""

    common = prepared.common
    messages: list[str] = []
    dimension = common.dimension
    parameters = common.target_parameters

    # A dimension- and target-parameter-aware literal oracle.  This deliberately
    # evaluates the formula independently of the production implementation so
    # CUSTOM runs can alter beta, sigma_y, and d_x without invalidating preflight.
    hand_q = np.empty(dimension, dtype=np.float64)
    hand_q[0] = 0.37
    hand_q[1:] = np.linspace(-0.5, 0.75, common.transverse_dimension)
    hand_squared_radius = float(np.dot(hand_q[1:], hand_q[1:]))
    hand_inverse_scale = math.exp(-float(hand_q[0]))
    expected_potential = (
        0.5 * hand_q[0] * hand_q[0] / (common.sigma_y * common.sigma_y)
        + 0.5 * common.transverse_dimension * hand_q[0] / common.beta
        + 0.5 * hand_inverse_scale * hand_squared_radius
    )
    expected_gradient = np.empty(dimension, dtype=np.float64)
    expected_gradient[0] = (
        hand_q[0] / (common.sigma_y * common.sigma_y)
        + 0.5 * common.transverse_dimension / common.beta
        - 0.5 * hand_inverse_scale * hand_squared_radius
    )
    expected_gradient[1:] = hand_inverse_scale * hand_q[1:]
    hand_gradient = _target_gradient(hand_q, parameters)
    if (
        not math.isclose(_potential_energy(hand_q, parameters), expected_potential,
                         rel_tol=0.0, abs_tol=2e-14)
        or not np.allclose(
            hand_gradient,
            expected_gradient,
            rtol=0.0,
            atol=2e-13,
        )
    ):
        raise RuntimeError("Self-test failed: literal Neal-funnel oracle.")

    q_gradient_test = np.empty(dimension, dtype=np.float64)
    q_gradient_test[0] = -0.23
    q_gradient_test[1:] = np.linspace(
        0.21, -0.14, common.transverse_dimension
    )
    analytic = _target_gradient(q_gradient_test, parameters)
    numerical = np.empty(dimension)
    for coordinate in range(dimension):
        step = 1e-6 * max(1.0, abs(float(q_gradient_test[coordinate])))
        plus = q_gradient_test.copy()
        minus = q_gradient_test.copy()
        plus[coordinate] += step
        minus[coordinate] -= step
        numerical[coordinate] = (
            _potential_energy(plus, parameters)
            - _potential_energy(minus, parameters)
        ) / (2.0 * step)
    if not np.allclose(analytic, numerical, rtol=2e-8, atol=2e-8):
        raise RuntimeError("Self-test failed: analytic target gradient.")
    if not np.allclose(
        _physical_to_latent(common.physical_starts, common.beta),
        common.funnel_starts,
        rtol=0.0,
        atol=2e-14,
    ):
        raise RuntimeError("Self-test failed: latent/physical start round trip.")
    if not common.mass_is_identity:
        raise RuntimeError("Self-test failed: conditional track requires M0=I.")
    messages.append(
        "PASS: literal funnel oracle, target gradient, starts, and identity base mass"
    )

    names, means, sds = _observable_specification(
        common.beta,
        common.sigma_y,
        common.transverse_dimension,
    )
    expected_observable_count = dimension + dimension * (dimension + 1) // 2
    if (
        len(names) != expected_observable_count
        or len(set(names)) != expected_observable_count
        or not np.array_equal(means, common.exact_observable_means)
        or not np.array_equal(sds, common.exact_observable_sds)
    ):
        raise RuntimeError("Self-test failed: exact physical observables.")
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    # Standard-normal quadrature checks the exponential-moment identity without
    # imposing a hidden beta or sigma_y restriction on otherwise valid CUSTOM runs.
    normal_nodes = math.sqrt(2.0) * nodes
    normalized_weights = weights / math.sqrt(math.pi)
    if (
        not math.isclose(float(np.sum(normalized_weights*np.exp(normal_nodes))),
                         math.exp(0.5), rel_tol=3e-14)
        or not math.isclose(
            float(np.sum(normalized_weights*np.exp(2.0*normal_nodes))),
            math.exp(2.0), rel_tol=3e-13,
        )
    ):
        raise RuntimeError("Self-test failed: Gauss-Hermite moment oracle.")
    messages.append(
        f"PASS: all {expected_observable_count} exact physical moments "
        "and quadrature oracle"
    )

    autocovariance_input = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]
    )
    expected_autocovariance = np.asarray(
        [[1.25, 0.3125, -0.375, -0.5625], [5.0, 1.25, -1.5, -2.25]]
    )
    if not np.allclose(
        _autocovariance_fft(autocovariance_input),
        expected_autocovariance,
        rtol=0.0,
        atol=1e-14,
    ):
        raise RuntimeError("Self-test failed: biased FFT autocovariance.")
    rng = np.random.default_rng(24681357)
    iid = rng.standard_normal((4, 6000))
    iid_rhat = _rank_normalized_split_folded_rhat(iid)
    if iid_rhat.status != "ok" or not (0.98 < iid_rhat.combined < 1.03):
        raise RuntimeError("Self-test failed: IID rank/folded R-hat.")
    shifted = iid.copy()
    shifted[0] += 2.0
    if _rank_normalized_split_folded_rhat(shifted).combined <= 1.1:
        raise RuntimeError("Self-test failed: R-hat location sensitivity.")
    phi = 0.8
    ar = np.empty((4, 20000))
    ar[:, 0] = rng.normal(scale=1.0/math.sqrt(1.0-phi*phi), size=4)
    innovations = rng.standard_normal((4, ar.shape[1]-1))
    for draw in range(1, ar.shape[1]):
        ar[:, draw] = phi*ar[:, draw-1] + innovations[:, draw-1]
    ar_iat = _multi_chain_geyer_iat(ar)
    if not ar_iat.is_defined or not (6.0 < ar_iat.tau < 12.5):
        raise RuntimeError("Self-test failed: Geyer AR(1) IAT.")
    messages.append(
        "PASS: biased autocovariance, Geyer cutoff, and rank split/folded R-hat"
    )

    _validate_compiled_reference_microcheck()
    # Fixed, well-resolved verification fixture, not the first production grid
    # point. The expanded tau=.125,Q=.003125 corner is too stiff for the old
    # 0.04/0.02 convergence test to be in its asymptotic regime. Keep that
    # original test's tau=.5,Q=.1 and tolerances; do NOT change production.
    tau = 0.5
    Q = 0.1
    q0 = np.empty(dimension, dtype=np.float64)
    q0[0] = 0.2
    q0[1:] = np.linspace(0.1, -0.15, common.transverse_dimension)
    p0 = np.empty(dimension, dtype=np.float64)
    p0[0] = 0.7
    p0[1:] = np.linspace(-0.3, 0.25, common.transverse_dimension)
    s0 = 0.31
    p_s0 = -0.27
    epsilon = 0.001
    gradient0 = _target_gradient(q0, common.target_parameters)
    forward = _thc_proposal(
        q0, p0, s0, p_s0, gradient0, common.beta, tau, Q,
        epsilon, 17, common.target_parameters,
        common.inverse_mass_matrix, common.mass_is_identity,
    )
    reverse = _thc_proposal(
        forward[0], -forward[1], forward[2], -forward[3], forward[4],
        common.beta, tau, Q, epsilon, 17, common.target_parameters,
        common.inverse_mass_matrix, common.mass_is_identity,
    )
    if (
        not forward[6] or not reverse[6] or forward[5] != 17 or reverse[5] != 17
        or not np.allclose(reverse[0], q0, rtol=0.0, atol=5e-11)
        or not np.allclose(reverse[1], -p0, rtol=0.0, atol=5e-11)
        or not math.isclose(reverse[2], s0, rel_tol=0.0, abs_tol=5e-11)
        or not math.isclose(reverse[3], -p_s0, rel_tol=0.0, abs_tol=5e-11)
    ):
        raise RuntimeError("Self-test failed: reversibility or gradient count.")
    messages.append(
        "PASS: compiled/reference oracle and conditional-THC reversibility"
    )

    def flow(state: np.ndarray, step_size: float, steps: int) -> np.ndarray:
        flow_q = state[:dimension].copy()
        flow_p = state[dimension:2*dimension].copy()
        output = _thc_proposal(
            flow_q, flow_p, float(state[-2]), float(state[-1]),
            _target_gradient(flow_q, common.target_parameters),
            common.beta, tau, Q, step_size, steps,
            common.target_parameters, common.inverse_mass_matrix, True,
        )
        if not output[6]:
            raise RuntimeError("Self-test failed: nonfinite local flow.")
        return np.concatenate((output[0], output[1], [output[2], output[3]]))

    local_state = np.concatenate((q0, p0, [s0, p_s0]))
    phase_dimension = 2 * dimension + 2
    jacobian = np.empty((phase_dimension, phase_dimension))
    jacobian_step = 1e-6
    for column in range(phase_dimension):
        plus = local_state.copy()
        minus = local_state.copy()
        plus[column] += jacobian_step
        minus[column] -= jacobian_step
        jacobian[:, column] = (
            flow(plus, 0.004, 3) - flow(minus, 0.004, 3)
        ) / (2.0*jacobian_step)
    if not math.isclose(float(np.linalg.det(jacobian)), 1.0,
                        rel_tol=0.0, abs_tol=4e-7):
        raise RuntimeError("Self-test failed: volume preservation.")

    def energy(state: np.ndarray) -> float:
        return _thc_hamiltonian(
            state[:dimension], state[dimension:2*dimension],
            float(state[-2]), float(state[-1]), common.beta, tau, Q,
            common.target_parameters, common.inverse_mass_matrix, True,
        )

    initial_energy = energy(local_state)
    coarse_error = abs(energy(flow(local_state, 0.04, 1))-initial_energy)
    fine_error = abs(energy(flow(local_state, 0.02, 2))-initial_energy)
    ratio = coarse_error/fine_error
    if not (3.3 < ratio < 4.8):
        raise RuntimeError("Self-test failed: second-order energy error.")
    messages.append("PASS: volume preservation and second-order fixed-time error")

    momenta, thermostat_momenta, uniforms = _make_random_streams(
        998877, 5, dimension
    )
    repeated = _make_random_streams(998877, 5, dimension)
    if not all(
        np.array_equal(first, second)
        for first, second in zip((momenta, thermostat_momenta, uniforms), repeated)
    ):
        raise RuntimeError("Self-test failed: deterministic random substreams.")
    chain_output = _simulate_thc_chain(
        momenta, thermostat_momenta, uniforms, common.physical_starts[0],
        prepared.thermostat_starts[0], common.beta, tau, Q, epsilon, 4, 2, 3,
        common.target_parameters, common.mass_cholesky,
        common.inverse_mass_matrix, common.mass_is_identity,
    )
    if (
        chain_output[0].shape != (3, dimension)
        or chain_output[1].shape != (3,)
        or chain_output[5] != 9 or chain_output[6] != 12
        or chain_output[3] != 0 or chain_output[4] != 0
    ):
        raise RuntimeError("Self-test failed: chain or gradient accounting.")

    diagnostic = _thermostat_diagnostic(
        np.asarray([-tau/math.sqrt(common.beta), tau/math.sqrt(common.beta)]),
        0.0,
        tau*tau/common.beta,
    )
    if diagnostic.empirical_mean != 0.0 or not math.isclose(
        diagnostic.empirical_second_moment, diagnostic.exact_second_moment,
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise RuntimeError("Self-test failed: conditional thermostat moments.")
    expected_configurations = (
        len(prepared.tau_values) * len(prepared.Q_values)
        * len(common.epsilon_values) * len(common.L_values)
    )
    if expected_configurations <= 0:
        raise RuntimeError("Self-test failed: Cartesian grid cardinality.")
    messages.append(
        "PASS: RNG, finite smoke chain, exact gradient charging, conditional moments, "
        f"and {expected_configurations}-cell design"
    )
    return tuple(messages)


def run_thc_tuning(
    inputs: THCRunInputs,
    *,
    preflight_only: bool = False,
) -> dict[str, object]:
    """Run the Grex preflight or one completely specified THC campaign."""

    if not isinstance(preflight_only, bool):
        raise TypeError("preflight_only must be exactly True or False.")
    if preflight_only and inputs.run_self_tests is not True:
        raise ValueError("preflight_only=True requires run_self_tests=True.")

    prepared = _prepare_thc_inputs(inputs)
    common = prepared.common
    _validate_thc_run_mode(prepared)
    _validate_common_parity_seals(common)
    if (
        common.run_mode == "CANONICAL"
        and _thc_implementation_fingerprint()
        != EXPECTED_THC_IMPLEMENTATION_FINGERPRINT
    ):
        raise RuntimeError(
            "Canonical THC implementation source seal failed: expected "
            + EXPECTED_THC_IMPLEMENTATION_FINGERPRINT
            + ", received "
            + _thc_implementation_fingerprint()
        )

    self_test_messages = (
        run_self_tests(prepared)
        if common.run_self_tests
        else (
            "SKIPPED: run_self_tests=False (CUSTOM mode only).",
        )
    )
    runtime_preflight_message = (
        _validate_portable_native_runtime(prepared)
        if common.run_self_tests
        else "SKIPPED: portable native preflight requires run_self_tests=True."
    )
    if common.print_progress:
        print(runtime_preflight_message, flush=True)
    self_test_messages = (
        *self_test_messages,
        runtime_preflight_message,
    )

    if preflight_only:
        print(
            "GREX PREFLIGHT COMPLETE: the current cache-free native sampler, "
            "equations, reversibility checks, gradient accounting, RNG, "
            "diagnostics, and finite smoke chain all passed. No tuning "
            "trajectory or Cartesian-grid configuration was simulated, and "
            "no report was written. Set GREX_PREFLIGHT_ONLY=False in the bottom input "
            "block and execute this entire cell again to start production "
            "tuning.",
            flush=True,
        )
        return {
            "preflight_only": True,
            "prepared_inputs": prepared,
            "self_test_messages": self_test_messages,
            "runtime_environment": _runtime_environment_summary(),
            "common_input_fingerprint": common.common_input_fingerprint,
            "input_preparation_fingerprint": (
                _input_preparation_fingerprint()
            ),
            "shared_protocol_fingerprint": (
                _shared_protocol_fingerprint(common)
            ),
            "diagnostics_fingerprint": _diagnostics_fingerprint(),
            "common_pipeline_fingerprint": (
                _common_pipeline_fingerprint()
            ),
            "thc_implementation_fingerprint": (
                _thc_implementation_fingerprint()
            ),
            "source_parity_verification_mode": (
                _source_verification_mode()
            ),
        }

    # Preflight-only mode performs no filesystem mutation.  Resolve/create the
    # output directory only when a production or CUSTOM campaign will run.
    destination = _resolve_output_destination(common)

    total_iterations = (
        common.n_warmup + common.n_retained
    )
    stream_caches: list[
        list[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
        ]
    ] = []
    for block in common.seed_blocks:
        stream_caches.append(
            [
                _make_random_streams(
                    seed,
                    total_iterations,
                    common.dimension,
                )
                for seed in block
            ]
        )

    auxiliary_momentum_moments = tuple(
        _retained_standardized_momentum_second_moment(
            cache, common.n_warmup
        )
        for cache in stream_caches
    )

    # Compile Numba kernels before worker threads are opened.
    (
        compile_momenta,
        compile_thermostat_momenta,
        compile_uniforms,
    ) = _make_random_streams(
        13579, 2, common.dimension
    )
    _simulate_thc_chain(
        compile_momenta,
        compile_thermostat_momenta,
        compile_uniforms,
        common.physical_starts[0],
        prepared.thermostat_starts[0],
        common.beta,
        prepared.tau_values[0],
        prepared.Q_values[0],
        common.epsilon_values[0],
        1,
        1,
        1,
        common.target_parameters,
        common.mass_cholesky,
        common.inverse_mass_matrix,
        common.mass_is_identity,
    )

    configuration_results: list[
        THCConfigurationResult
    ] = []
    total_configurations = (
        len(prepared.tau_values)
        * len(prepared.Q_values)
        * len(common.epsilon_values)
        * len(common.L_values)
    )
    configuration_number = 0

    for tau in prepared.tau_values:
        for Q in prepared.Q_values:
            for epsilon in common.epsilon_values:
                for L in common.L_values:
                    configuration_number += 1
                    if common.print_progress:
                        print(
                            f"[{configuration_number:>4}/"
                            f"{total_configurations}] conditional THC "
                            f"tau={tau:.6g}, Q={Q:.6g}, "
                            f"epsilon={epsilon:.6g}, L={L}: "
                            f"running {len(common.seed_blocks)} "
                            "tuning blocks...",
                            flush=True,
                        )
                    block_results: list[THCBlockResult] = []
                    for block_index, seeds in enumerate(common.seed_blocks):
                        block_results.append(
                            _run_configuration_block(
                                tau=tau,
                                Q=Q,
                                epsilon=epsilon,
                                L=L,
                                block_index=block_index,
                                seeds=seeds,
                                stream_cache=stream_caches[block_index],
                                prepared=prepared,
                            )
                        )
                    aggregate = _aggregate_thc_configuration(
                        tau,
                        Q,
                        epsilon,
                        L,
                        block_results,
                    )
                    configuration_results.append(aggregate)
                    if common.print_progress:
                        print(
                            "    status="
                            + aggregate.status
                            + ", GM efficiency="
                            + _format_number(
                                aggregate.geometric_mean_efficiency,
                                ".8e",
                            ),
                            flush=True,
                        )

    representative = _select_thc_representative(
        configuration_results
    )
    report = _build_report(
        configuration_results=configuration_results,
        representative=representative,
        auxiliary_momentum_moments=(
            auxiliary_momentum_moments
        ),
        prepared=prepared,
        self_test_messages=self_test_messages,
    )
    destination.write_text(
        report + "\n", encoding="utf-8"
    )

    if common.print_report:
        print("\n" + report)
        print(
            "\nThe identical report was saved to:\n"
            f"{destination}"
        )

    return {
        "configuration_results": tuple(
            configuration_results
        ),
        "representative": representative,
        "report": report,
        "output_path": destination,
        "prepared_inputs": prepared,
        "common_input_fingerprint": (
            common.common_input_fingerprint
        ),
        "thc_method_input_fingerprint": (
            prepared.method_input_fingerprint
        ),
        "input_preparation_fingerprint": (
            _input_preparation_fingerprint()
        ),
        "shared_protocol_fingerprint": (
            _shared_protocol_fingerprint(common)
        ),
        "diagnostics_fingerprint": (
            _diagnostics_fingerprint()
        ),
        "common_pipeline_fingerprint": (
            _common_pipeline_fingerprint()
        ),
        "thc_implementation_fingerprint": (
            _thc_implementation_fingerprint()
        ),
        "source_parity_verification_mode": (
            _source_verification_mode()
        ),
    }


# ===========================================================================
#                         ALL USER INPUTS ARE HERE
# ===========================================================================
# Enter only independent choices.  The code derives total dimension, squared
# scale, marginal covariance, physical starts, inverse/Cholesky mass matrices,
# exact observable moments, observable count, full Cartesian configuration
# count, and all fingerprints.
#
# CANONICAL mode accepts only the agreed fair-playground common and THC values.
# For changed experiments, use CUSTOM mode, begin the run label with "CUSTOM:",
# and choose a non-canonical output filename.

# First execution on Grex: leave True.  The cell runs only a short native
# preflight and writes no report.  After the printed PASS and GREX PREFLIGHT
# COMPLETE lines, change only this value to False and execute the entire cell
# again to run the full 432-configuration production grid.
GREX_PREFLIGHT_ONLY = True

USER_INPUTS = THCRunInputs(
    # ----- Neal's funnel target and shared base mass (identical to HMC) -----
    beta=1.0,
    sigma_y=3.0,
    dimension_x=5,
    # Latest-manuscript convention: the common base mass is exactly I_6.
    mass_matrix=np.eye(6, dtype=np.float64),

    # Rows are (y,z1,...,z5); physical x_i=exp(y/2)*z_i/sqrt(beta).
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

    # ----- Common HMC/THC epsilon-L tuning grid -----
    epsilon_values=(
        0.040,
        0.060,
        0.080,
        0.100,
        0.120,
        0.150,
    ),
    L_values=(20, 30, 40, 50, 60, 80),

    # ----- Common tuning seed blocks; one seed per chain -----
    seed_blocks=(
        (3857660299, 1344569885, 2010517369, 2974962254),
        (
            2914251580,
            2216611040,
            1467312042,
            1966646752,
        ),
        (
            26638442,
            2162183804,
            2288357861,
            3874567655,
        ),
        (
            917698748,
            796766395,
            2769691025,
            1469822251,
        ),
        (
            983213432,
            4224231832,
            2042092071,
            4006032910,
        ),
    ),

    # ----- Target-informed conditional coupling and joint tau-Q grid -----
    coupling_family="FUNNEL_CONDITIONAL_EXP_MINUS_HALF_S",
    # This compact grid contains the manuscript representative tau=.75,Q=.25.
    tau_values=(0.5, 0.75, 1.0),
    Q_values=(0.1, 0.25, 0.5, 1.0),
    # s persists; it is not independently refreshed between trajectories.
    thermostat_starts=(-6.0, -1.0, 1.0, 6.0),

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
    output_filename=(
        "Conditional_THC_Neals_funnel_tuning_results_GREX_PORTABLE_V1.txt"
    ),
    allow_overwrite=True,
)


# Expanded campaign: the original equations, kernels, diagnostics and source
# seals above are unchanged. Use launcher/checkpoint_runner.py for production.
# These are new CUSTOM tuning inputs, not the archived CANONICAL experiment.
from expanded_protocol import expanded_inputs
USER_INPUTS = expanded_inputs(USER_INPUTS, "conditional")

if __name__ == "__main__":
    run_thc_tuning(
        USER_INPUTS,
        preflight_only=GREX_PREFLIGHT_ONLY,
    )
