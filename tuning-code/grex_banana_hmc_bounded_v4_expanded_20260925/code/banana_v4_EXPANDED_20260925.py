"""Standalone production tuner for conditional-shear THC-v4 on the 2D Banana.

The physical target and fair-playground diagnostics are identical to the
frozen Banana HMC and phase-periodic THC-v2 tuners.  The new extended target is

    pi(q,s) propto exp[-beta U(q)] exp[-(s-x)^2/(2 tau^2)],

and its metric is

    c(s) = kappa*a*tanh(s/a),
    A(s) = [[1,0],[-c(s),1]],          G(s) = A(s)^T M0 A(s).

Because det A(s)=1, det G(s)=det M0 is constant.  Integrating out s and the
momenta therefore leaves exactly the original Banana marginal.  The extended
Hamiltonian used by the code is

    H = U(q) + (s-x)^2/(2 beta tau^2)
        + p^T G(s)^(-1) p / 2 + p_s^2/(2Q).

One trajectory uses the symmetric exact-subflow P/T/K/T/P split.  The complete
state is momentum-flipped and Metropolis corrected.  Only evaluations of the
physical target gradient are charged.  The score is the minimum, over x, y,
x^2, x*y, y^2, of 1/(gradients per retained draw times Geyer IAT).  Fixed
configurations are aggregated by the geometric mean over five tuning blocks.

The bounded shear gives |c(s)|<kappa*a and keeps the metric uniformly
well-conditioned.  The staged search is smaller than the phase-periodic
THC-v2 search.  The derived control
(kappa,tau,Q,a,epsilon,L)=(0.5,0.5,0.1,8,0.05,40) is always evaluated on all
five full-length blocks even if halving removes it.

After base five-block selection, one automatic predeclared boundary layer is
allowed.  If the winner is on a method-grid boundary, each active coordinate
is moved alone to its outward value, the common epsilon-L grid is searched on
the stage-2 blocks, and at most four boundary finalists receive five-block
evaluation.  Joint corners and a second layer are forbidden.

This file is self-contained: it imports no neighbouring sampler.  Every user
input is in the final control panel.  Its first execution is preflight-only,
uses cache-free native compilation, writes nothing, and never constructs or
mentions a final-comparison random seed.
"""

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


try:
    import llvmlite
    import numba

    def njit(*args, **kwargs):
        options = dict(kwargs)
        options["cache"] = False
        return numba.njit(*args, **options)

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    llvmlite = None
    numba = None
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator


PROTOCOL_VERSION = "BANANA2-CONDITIONAL-SHEAR-THC-V4-HALVING-V1.1.0"
IMPLEMENTATION_VERSION = "conditional-shear-bounded-tanh-PTKTP-full-MH-v4.1.0"
DIAGNOSTICS_VERSION = "multichain-geyer-ims-rank-split-folded-rhat-v1.0.0"
RNG_VERSION = "seedsequence-pcg64-three-substreams-v1.0.0"
PREFLIGHT_VERSION = "banana-conditional-shear-v4-native-preflight-20260901b-v2"
SUPPORTED_OBSERVABLE_POLICY = "ALL_LINEAR_AND_UNIQUE_QUADRATIC"
CANONICAL_OUTPUT_FILENAME = (
    "THC_v4_conditional_shear_banana_tuning_results_GREX_PORTABLE_V1.txt"
)
HMC_SHARED_COMMON_INPUT_SHA256 = (
    "0fb7398fba5b8e8c874f498a68471df0a140a341cb289f87e8e698ea4ac26136"
)


@dataclass(frozen=True)
class THCRunInputs:
    """Every independent scientific and execution input for one run."""

    beta: float
    sigma_x: float
    sigma_y: float
    banana_b: float
    mass_matrix: Sequence[Sequence[float]]
    standardized_starts: Sequence[Sequence[float]]
    conditional_standardized_starts: Sequence[float]
    observable_policy: str
    epsilon_values: Sequence[float]
    L_values: Sequence[int]
    seed_blocks: Sequence[Sequence[int]]
    kappa_values: Sequence[float]
    tau_values: Sequence[float]
    Q_values: Sequence[float]
    a_values: Sequence[float]
    stage1_anchor_pairs: Sequence[Sequence[float]]
    stage1_block_indices: Sequence[int]
    stage1_n_warmup: int
    stage1_n_retained: int
    stage1_keep_method_tuples: int
    stage2_block_indices: Sequence[int]
    stage2_n_warmup: int
    stage2_n_retained: int
    stage2_keep_fixed_configurations: int
    automatic_boundary_expansion: bool
    boundary_kappa_lower: float
    boundary_kappa_upper: float
    boundary_tau_lower: float
    boundary_tau_upper: float
    boundary_Q_lower: float
    boundary_Q_upper: float
    boundary_a_lower: float
    boundary_a_upper: float
    boundary_keep_fixed_configurations: int
    forced_full_configurations: Sequence[Sequence[float]]
    n_warmup: int
    n_retained: int
    n_workers: int
    run_self_tests: bool
    print_progress: bool
    print_report: bool
    run_mode: str
    run_label: str
    output_directory: str | os.PathLike[str] | None
    output_filename: str
    allow_overwrite: bool


@dataclass(frozen=True)
class PreparedInputs:
    beta: float
    sigma_x: float
    sigma_y: float
    banana_b: float
    target_parameters: np.ndarray
    mass_matrix: np.ndarray
    inverse_mass_matrix: np.ndarray
    mass_cholesky: np.ndarray
    standardized_starts: np.ndarray
    physical_starts: np.ndarray
    conditional_standardized_starts: np.ndarray
    observable_policy: str
    observable_names: tuple[str, ...]
    exact_observable_means: np.ndarray
    exact_observable_sds: np.ndarray
    epsilon_values: tuple[float, ...]
    L_values: tuple[int, ...]
    seed_blocks: tuple[tuple[int, ...], ...]
    kappa_values: tuple[float, ...]
    tau_values: tuple[float, ...]
    Q_values: tuple[float, ...]
    a_values: tuple[float, ...]
    stage1_anchor_pairs: tuple[tuple[float, int], ...]
    stage1_block_indices: tuple[int, ...]
    stage1_n_warmup: int
    stage1_n_retained: int
    stage1_keep_method_tuples: int
    stage2_block_indices: tuple[int, ...]
    stage2_n_warmup: int
    stage2_n_retained: int
    stage2_keep_fixed_configurations: int
    automatic_boundary_expansion: bool
    boundary_kappa_lower: float
    boundary_kappa_upper: float
    boundary_tau_lower: float
    boundary_tau_upper: float
    boundary_Q_lower: float
    boundary_Q_upper: float
    boundary_a_lower: float
    boundary_a_upper: float
    boundary_keep_fixed_configurations: int
    forced_full_configurations: tuple["FixedConfiguration", ...]
    n_warmup: int
    n_retained: int
    n_workers: int
    run_self_tests: bool
    print_progress: bool
    print_report: bool
    run_mode: str
    run_label: str
    output_directory: str | os.PathLike[str] | None
    output_filename: str
    allow_overwrite: bool
    input_fingerprint: str

    @property
    def dimension(self) -> int:
        return 2

    @property
    def n_chains(self) -> int:
        return int(self.physical_starts.shape[0])


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
class ConditionalDiagnostic:
    empirical_mean: float
    exact_mean: float
    mean_error: float
    empirical_second_moment: float
    exact_second_moment: float
    second_moment_error: float


@dataclass(frozen=True, order=True)
class MethodTuple:
    kappa: float
    tau: float
    Q: float
    a: float


@dataclass(frozen=True, order=True)
class FixedConfiguration:
    kappa: float
    tau: float
    Q: float
    a: float
    epsilon: float
    L: int


@dataclass(frozen=True)
class THCBlockResult:
    physical: BlockResult
    conditional: ConditionalDiagnostic


@dataclass(frozen=True)
class ConfigurationResult:
    configuration: FixedConfiguration
    origin: str
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
# Target, extended Hamiltonian, exact split map, and Metropolis chain.
# ---------------------------------------------------------------------------


@njit(inline="always")
def _potential_energy(q: np.ndarray, parameters: np.ndarray) -> float:
    sigma_x, sigma_y, banana_b = parameters
    x = q[0]
    residual = q[1] - banana_b * (x * x - sigma_x * sigma_x)
    return 0.5 * (x * x / (sigma_x * sigma_x) + residual * residual / (sigma_y * sigma_y))


@njit(inline="always")
def _target_gradient(q: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    sigma_x, sigma_y, banana_b = parameters
    x = q[0]
    residual = q[1] - banana_b * (x * x - sigma_x * sigma_x)
    gradient = np.empty(2, dtype=np.float64)
    gradient[0] = x / (sigma_x * sigma_x) - 2.0 * banana_b * x * residual / (sigma_y * sigma_y)
    gradient[1] = residual / (sigma_y * sigma_y)
    return gradient


@njit(inline="always")
def _state_is_finite(q: np.ndarray, p: np.ndarray, s: float, p_s: float) -> bool:
    return (
        math.isfinite(q[0])
        and math.isfinite(q[1])
        and math.isfinite(p[0])
        and math.isfinite(p[1])
        and math.isfinite(s)
        and math.isfinite(p_s)
    )


@njit(inline="always")
def _shear_and_derivative(
    s: float,
    kappa: float,
    a: float,
) -> tuple[float, float]:
    scaled = s / a
    tangent = math.tanh(scaled)
    return kappa * a * tangent, kappa * (1.0 - tangent * tangent)


@njit(inline="always")
def _kinetic_terms(
    p: np.ndarray,
    s: float,
    kappa: float,
    a: float,
    inverse_mass: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Return c,c',z0,z1,u0,u1 for the bounded shear metric."""

    c, dc = _shear_and_derivative(s, kappa, a)
    z0 = p[0] + c * p[1]
    z1 = p[1]
    u0 = inverse_mass[0, 0] * z0 + inverse_mass[0, 1] * z1
    u1 = inverse_mass[1, 0] * z0 + inverse_mass[1, 1] * z1
    return c, dc, z0, z1, u0, u1


@njit
def _hamiltonian(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    beta: float,
    kappa: float,
    tau: float,
    Q: float,
    a: float,
    target_parameters: np.ndarray,
    inverse_mass: np.ndarray,
) -> float:
    _c, _dc, z0, z1, u0, u1 = _kinetic_terms(
        p, s, kappa, a, inverse_mass
    )
    eta = s - q[0]
    return (
        _potential_energy(q, target_parameters)
        + 0.5 * eta * eta / (beta * tau * tau)
        + 0.5 * (z0 * u0 + z1 * u1)
        + 0.5 * p_s * p_s / Q
    )


@njit(nogil=True)
def _proposal(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    current_gradient: np.ndarray,
    beta: float,
    kappa: float,
    tau: float,
    Q: float,
    a: float,
    epsilon: float,
    L: int,
    target_parameters: np.ndarray,
    inverse_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, int, bool]:
    """Apply P(e/2) [T(e/2) K(e) T(e/2) P(e)]^(L) exactly."""

    proposed_q = q.copy()
    proposed_p = p.copy()
    proposed_s = s
    proposed_p_s = p_s
    proposed_gradient = current_gradient.copy()
    gradient_calls = 0
    inverse_beta_tau_squared = 1.0 / (beta * tau * tau)

    eta = proposed_s - proposed_q[0]
    proposed_p[0] -= 0.5 * epsilon * (
        proposed_gradient[0] - eta * inverse_beta_tau_squared
    )
    proposed_p[1] -= 0.5 * epsilon * proposed_gradient[1]
    proposed_p_s -= 0.5 * epsilon * eta * inverse_beta_tau_squared
    if not _state_is_finite(proposed_q, proposed_p, proposed_s, proposed_p_s):
        return proposed_q, proposed_p, proposed_s, proposed_p_s, proposed_gradient, gradient_calls, False

    for step in range(L):
        # T: exact p_s^2/(2Q) drift.
        proposed_s += 0.5 * epsilon * proposed_p_s / Q

        # K: exact flow of p^T G(s)^(-1)p/2 at fixed (p,s).
        c, dc, _z0, _z1, u0, u1 = _kinetic_terms(
            proposed_p, proposed_s, kappa, a, inverse_mass
        )
        proposed_q[0] += epsilon * u0
        proposed_q[1] += epsilon * (c * u0 + u1)
        proposed_p_s -= epsilon * dc * proposed_p[1] * u0

        # T again.
        proposed_s += 0.5 * epsilon * proposed_p_s / Q
        if not _state_is_finite(proposed_q, proposed_p, proposed_s, proposed_p_s):
            return proposed_q, proposed_p, proposed_s, proposed_p_s, proposed_gradient, gradient_calls, False

        proposed_gradient = _target_gradient(proposed_q, target_parameters)
        gradient_calls += 1
        if not np.all(np.isfinite(proposed_gradient)):
            return proposed_q, proposed_p, proposed_s, proposed_p_s, proposed_gradient, gradient_calls, False

        kick = 0.5 * epsilon if step == L - 1 else epsilon
        eta = proposed_s - proposed_q[0]
        proposed_p[0] -= kick * (
            proposed_gradient[0] - eta * inverse_beta_tau_squared
        )
        proposed_p[1] -= kick * proposed_gradient[1]
        proposed_p_s -= kick * eta * inverse_beta_tau_squared
        if not _state_is_finite(proposed_q, proposed_p, proposed_s, proposed_p_s):
            return proposed_q, proposed_p, proposed_s, proposed_p_s, proposed_gradient, gradient_calls, False

    return proposed_q, proposed_p, proposed_s, proposed_p_s, proposed_gradient, gradient_calls, True


@njit(nogil=True)
def _simulate_chain(
    standard_momenta: np.ndarray,
    thermostat_standard_momenta: np.ndarray,
    acceptance_uniforms: np.ndarray,
    q_initial: np.ndarray,
    eta_initial_standardized: float,
    beta: float,
    kappa: float,
    tau: float,
    Q: float,
    a: float,
    epsilon: float,
    L: int,
    n_warmup: int,
    n_retained: int,
    target_parameters: np.ndarray,
    mass_cholesky: np.ndarray,
    inverse_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    """Run one persistent-(q,s), fully refreshed-(p,p_s) MH chain."""

    total_iterations = n_warmup + n_retained
    samples = np.empty((n_retained, 2), dtype=np.float64)
    eta_samples = np.empty(n_retained, dtype=np.float64)
    q = q_initial.copy()
    s = q[0] + tau * eta_initial_standardized
    current_gradient = _target_gradient(q, target_parameters)
    warmup_gradient_calls = 1 if n_warmup > 0 else 0
    retained_gradient_calls = 0 if n_warmup > 0 else 1
    accepted_retained = 0
    warmup_nonfinite = 0
    retained_nonfinite = 0
    thermostat_scale = math.sqrt(Q / beta)

    for iteration in range(total_iterations):
        # latent ~ N(0,M0/beta), then p=A(s)^T latent ~ N(0,G(s)/beta)
        z0 = standard_momenta[iteration, 0] / math.sqrt(beta)
        z1 = standard_momenta[iteration, 1] / math.sqrt(beta)
        latent0 = mass_cholesky[0, 0] * z0
        latent1 = mass_cholesky[1, 0] * z0 + mass_cholesky[1, 1] * z1
        c, _dc = _shear_and_derivative(s, kappa, a)
        p = np.empty(2, dtype=np.float64)
        p[0] = latent0 - c * latent1
        p[1] = latent1
        p_s = thermostat_scale * thermostat_standard_momenta[iteration]

        initial_energy = _hamiltonian(
            q, p, s, p_s, beta, kappa, tau, Q, a,
            target_parameters, inverse_mass,
        )
        (
            proposed_q,
            proposed_p,
            proposed_s,
            proposed_p_s,
            proposed_gradient,
            proposal_gradient_calls,
            proposal_is_finite,
        ) = _proposal(
            q, p, s, p_s, current_gradient, beta, kappa, tau, Q, a,
            epsilon, L, target_parameters, inverse_mass,
        )
        if iteration < n_warmup:
            warmup_gradient_calls += proposal_gradient_calls
        else:
            retained_gradient_calls += proposal_gradient_calls

        # Full-state momentum flip makes the deterministic proposal involutive.
        if proposal_is_finite:
            proposed_p[0] = -proposed_p[0]
            proposed_p[1] = -proposed_p[1]
            proposed_p_s = -proposed_p_s

        accepted = False
        if proposal_is_finite and math.isfinite(initial_energy):
            proposed_energy = _hamiltonian(
                proposed_q, proposed_p, proposed_s, proposed_p_s,
                beta, kappa, tau, Q, a, target_parameters, inverse_mass,
            )
            delta_energy = proposed_energy - initial_energy
            if math.isfinite(delta_energy):
                uniform = acceptance_uniforms[iteration]
                log_uniform = -math.inf if uniform <= 0.0 else math.log(uniform)
                accepted = log_uniform < -beta * delta_energy
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
            index = iteration - n_warmup
            samples[index, 0] = q[0]
            samples[index, 1] = q[1]
            eta_samples[index] = s - q[0]
            if accepted:
                accepted_retained += 1

    return (
        samples,
        eta_samples,
        accepted_retained,
        warmup_nonfinite,
        retained_nonfinite,
        warmup_gradient_calls,
        retained_gradient_calls,
    )


# ---------------------------------------------------------------------------
# Random streams and method-neutral fair-playground diagnostics.
# ---------------------------------------------------------------------------


def _make_random_streams(
    seed: int,
    total_iterations: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create physical normals, thermostat normals, and MH uniforms."""

    root = np.random.SeedSequence(int(seed))
    physical_sequence, thermostat_sequence, acceptance_sequence = root.spawn(3)
    physical_rng = np.random.Generator(np.random.PCG64(physical_sequence))
    thermostat_rng = np.random.Generator(np.random.PCG64(thermostat_sequence))
    acceptance_rng = np.random.Generator(np.random.PCG64(acceptance_sequence))
    physical = physical_rng.standard_normal((total_iterations, dimension))
    thermostat = thermostat_rng.standard_normal(total_iterations)
    uniforms = acceptance_rng.random(total_iterations)
    return (
        physical.astype(np.float64, copy=False),
        thermostat.astype(np.float64, copy=False),
        uniforms.astype(np.float64, copy=False),
    )


def _autocovariance_fft(chains: np.ndarray) -> np.ndarray:
    """Per-chain biased autocovariances with denominator N at every lag."""

    chains = np.asarray(chains, dtype=np.float64)
    centered = chains - np.mean(chains, axis=1, keepdims=True)
    n_draws = centered.shape[1]
    fft_size = 1 << (2 * n_draws - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size, axis=1)
    autocovariance = np.fft.irfft(
        transformed * np.conjugate(transformed), n=fft_size, axis=1
    )[:, :n_draws]
    return autocovariance / n_draws


def _multi_chain_geyer_iat(chains: np.ndarray) -> IATResult:
    """Multi-chain Geyer initial-positive and initial-monotone IAT."""

    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        return IATResult(math.nan, "undefined_shape", 0, -1, -1, "shape")
    n_chains, n_draws = chains.shape
    if n_chains < 2:
        return IATResult(math.nan, "undefined_insufficient_chains", 0, -1, -1, "chains")
    if n_draws < 4:
        return IATResult(math.nan, "undefined_insufficient_draws", 0, -1, -1, "draws")
    if not np.all(np.isfinite(chains)):
        return IATResult(math.nan, "undefined_nonfinite_input", 0, -1, -1, "nonfinite_input")

    within_variance = float(np.mean(np.var(chains, axis=1, ddof=1)))
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
        return IATResult(math.nan, "undefined_zero_variance", 0, -1, -1, "variance")

    mean_autocovariance = np.mean(_autocovariance_fft(chains), axis=0)
    rho = np.empty(n_draws, dtype=np.float64)
    rho[0] = 1.0
    rho[1:] = 1.0 - (within_variance - mean_autocovariance[1:]) / variance_plus
    usable_lags = n_draws if n_draws % 2 == 0 else n_draws - 1
    raw_pairs = rho[:usable_lags].reshape(-1, 2).sum(axis=1)
    included_count = raw_pairs.size
    first_excluded_lag = -1
    cutoff_reason = "end_of_available_lags"
    for pair_index, pair_sum in enumerate(raw_pairs):
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
    monotone_pairs = np.minimum.accumulate(raw_pairs[:included_count])
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
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        return math.nan
    n_chains, n_draws = chains.shape
    if n_chains < 2 or n_draws < 2 or not np.all(np.isfinite(chains)):
        return math.nan
    within = float(np.mean(np.var(chains, axis=1, ddof=1)))
    between = n_draws * float(np.var(np.mean(chains, axis=1), ddof=1))
    if not math.isfinite(within) or not math.isfinite(between):
        return math.nan
    if within <= np.finfo(float).tiny:
        return math.inf if between > np.finfo(float).tiny else math.nan
    variance_estimate = ((n_draws - 1.0) / n_draws) * within + between / n_draws
    if variance_estimate < 0.0 or not math.isfinite(variance_estimate):
        return math.nan
    return math.sqrt(variance_estimate / within)


def _split_chains(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        return np.empty((0, 0), dtype=np.float64)
    half = chains.shape[1] // 2
    if half < 2:
        return np.empty((0, 0), dtype=np.float64)
    return np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    ranks = np.empty(flat.size, dtype=np.float64)
    starts = np.concatenate(([0], np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1))
    ends = np.concatenate((starts[1:], [flat.size]))
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
    return ranks


def _standard_normal_ppf(probabilities: np.ndarray) -> np.ndarray:
    """Vectorized inverse standard-normal CDF using Acklam's approximation."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError("normal inverse-CDF probabilities must lie in (0,1)")
    a = np.array([
        -3.969683028665376e01, 2.209460984245205e02,
        -2.759285104469687e02, 1.383577518672690e02,
        -3.066479806614716e01, 2.506628277459239e00,
    ])
    b = np.array([
        -5.447609879822406e01, 1.615858368580409e02,
        -1.556989798598866e02, 6.680131188771972e01,
        -1.328068155288572e01,
    ])
    c = np.array([
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e00, -2.549732539343734e00,
        4.374664141464968e00, 2.938163982698783e00,
    ])
    d = np.array([
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00,
    ])
    low = 0.02425
    high = 1.0 - low
    result = np.empty_like(probabilities)
    lower = probabilities < low
    central = (probabilities >= low) & (probabilities <= high)
    upper = probabilities > high
    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(probabilities[lower]))
        num = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
        den = ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        result[lower] = num / den
    if np.any(central):
        q = probabilities[central] - 0.5
        r = q * q
        num = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
        den = (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        result[central] = num / den
    if np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - probabilities[upper]))
        num = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
        den = ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        result[upper] = -(num / den)
    return result


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ranks = _average_ranks(values)
    probabilities = (ranks - 3.0 / 8.0) / (ranks.size + 1.0 / 4.0)
    return _standard_normal_ppf(probabilities).reshape(values.shape)


def _rank_normalized_split_folded_rhat(chains: np.ndarray) -> RhatResult:
    split = _split_chains(chains)
    if split.size == 0:
        return RhatResult(math.nan, math.nan, math.nan, "undefined_split")
    if not np.all(np.isfinite(split)):
        return RhatResult(math.nan, math.nan, math.nan, "undefined_nonfinite")
    rank_rhat = _basic_rhat(_rank_normalize(split))
    folded = np.abs(split - np.median(split))
    folded_rhat = _basic_rhat(_rank_normalize(folded))
    if math.isnan(rank_rhat) or math.isnan(folded_rhat):
        return RhatResult(rank_rhat, folded_rhat, math.nan, "undefined_zero_or_invalid_variance")
    status = "ok_infinite" if math.isinf(rank_rhat) or math.isinf(folded_rhat) else "ok"
    return RhatResult(rank_rhat, folded_rhat, max(rank_rhat, folded_rhat), status)


def _observable_specification(
    beta: float,
    sigma_x: float,
    sigma_y: float,
    banana_b: float,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Exact means and target SDs for x,y,x^2,x*y,y^2."""

    variance_x = sigma_x * sigma_x / beta
    variance_r = sigma_y * sigma_y / beta
    centering = sigma_x * sigma_x
    bend = banana_b
    shifted_square_second = (
        3.0 * variance_x**2 - 2.0 * centering * variance_x + centering**2
    )
    shifted_square_fourth = (
        105.0 * variance_x**4
        - 60.0 * centering * variance_x**3
        + 18.0 * centering**2 * variance_x**2
        - 4.0 * centering**3 * variance_x
        + centering**4
    )
    mean_y = bend * (variance_x - centering)
    mean_y_squared = variance_r + bend * bend * shifted_square_second
    variance_y = variance_r + 2.0 * bend * bend * variance_x**2
    variance_x_squared = 2.0 * variance_x**2
    variance_xy = variance_x * variance_r + bend*bend*variance_x*(
        15.0*variance_x**2 - 6.0*centering*variance_x + centering**2
    )
    fourth_y = (
        3.0 * variance_r**2
        + 6.0 * bend*bend * variance_r * shifted_square_second
        + bend**4 * shifted_square_fourth
    )
    variance_y_squared = fourth_y - mean_y_squared**2
    names = ("x", "y", "x^2", "x*y", "y^2")
    means = np.asarray([0.0, mean_y, variance_x, 0.0, mean_y_squared])
    variances = np.asarray([
        variance_x, variance_y, variance_x_squared, variance_xy,
        variance_y_squared,
    ])
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("Derived observable variances must be finite and positive.")
    return names, means.astype(np.float64), np.sqrt(variances).astype(np.float64)


def _latent_to_physical(
    starts: np.ndarray,
    beta: float,
    sigma_x: float,
    sigma_y: float,
    banana_b: float,
) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.float64)
    x = sigma_x * starts[:, 0] / math.sqrt(beta)
    residual = sigma_y * starts[:, 1] / math.sqrt(beta)
    y = banana_b * (x*x - sigma_x*sigma_x) + residual
    return np.ascontiguousarray(np.column_stack((x, y)))


def _physical_to_latent(
    states: np.ndarray,
    beta: float,
    sigma_x: float,
    sigma_y: float,
    banana_b: float,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    x = states[:, 0]
    residual = states[:, 1] - banana_b * (x*x - sigma_x*sigma_x)
    scale = math.sqrt(beta)
    return np.ascontiguousarray(np.column_stack((
        scale*x/sigma_x, scale*residual/sigma_y,
    )))


def _physical_observables(samples: np.ndarray, dimension: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != dimension:
        raise ValueError(f"samples must have shape (chains,draws,{dimension})")
    values = [samples[:, :, index] for index in range(dimension)]
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
    observables = _physical_observables(samples, int(samples.shape[2]))
    diagnostics: list[ObservableDiagnostic] = []
    simulation_is_nonfinite = warmup_nonfinite > 0 or retained_nonfinite > 0
    if observables.shape[2] != len(observable_names):
        raise RuntimeError("Observable values/specification length mismatch.")
    for index, name in enumerate(observable_names):
        chains = observables[:, :, index]
        iat = _multi_chain_geyer_iat(chains)
        rhat = _rank_normalized_split_folded_rhat(chains)
        empirical_mean = float(np.mean(chains))
        exact_mean = float(exact_observable_means[index])
        raw_error = empirical_mean - exact_mean
        standardized_error = abs(raw_error) / float(exact_observable_sds[index])
        if not simulation_is_nonfinite and iat.is_defined and gradients_per_draw > 0.0:
            ess = total_retained / iat.tau
            efficiency = 1.0 / (gradients_per_draw * iat.tau)
        else:
            ess = math.nan
            efficiency = math.nan
        diagnostics.append(ObservableDiagnostic(
            name, iat, ess, efficiency, rhat, empirical_mean, exact_mean,
            raw_error, standardized_error,
        ))
    if simulation_is_nonfinite:
        status = "undefined_nonfinite_simulation"
    elif not all(item.iat.is_defined for item in diagnostics):
        status = "undefined_required_iat"
    elif not math.isfinite(gradients_per_draw) or gradients_per_draw <= 0.0:
        status = "undefined_gradient_denominator"
    else:
        status = "ok"
    if all(item.iat.is_defined for item in diagnostics):
        worst_index = max(range(len(diagnostics)), key=lambda i: diagnostics[i].iat.tau)
        worst_observable = diagnostics[worst_index].name
        worst_iat = diagnostics[worst_index].iat.tau
        block_efficiency = diagnostics[worst_index].efficiency if status == "ok" else math.nan
    else:
        bad = [i for i, item in enumerate(diagnostics) if not item.iat.is_defined]
        worst_observable = diagnostics[bad[0]].name if bad else "undefined"
        worst_iat = math.nan
        block_efficiency = math.nan
    if all(not math.isnan(item.rhat.combined) for item in diagnostics):
        rhat_index = max(range(len(diagnostics)), key=lambda i: diagnostics[i].rhat.combined)
        maximum_rhat_observable = diagnostics[rhat_index].name
        maximum_rhat = diagnostics[rhat_index].rhat.combined
    else:
        maximum_rhat_observable = "undefined"
        maximum_rhat = math.nan
    moment_index = max(
        range(len(diagnostics)),
        key=lambda i: diagnostics[i].standardized_moment_error,
    )
    return BlockResult(
        float(epsilon), int(L), int(block_index), tuple(int(s) for s in seeds),
        status, acceptance, int(retained_gradient_calls), gradients_per_draw,
        int(warmup_nonfinite), int(retained_nonfinite), tuple(diagnostics),
        worst_observable, worst_iat, block_efficiency,
        maximum_rhat_observable, maximum_rhat,
        diagnostics[moment_index].name,
        diagnostics[moment_index].standardized_moment_error,
    )


def _aggregate_blocks(block_results: Sequence[BlockResult]) -> BlockAggregate:
    blocks = tuple(block_results)
    if not blocks:
        raise ValueError("At least one block is required.")
    efficiencies = np.asarray([b.block_efficiency for b in blocks])
    valid = all(b.status == "ok" for b in blocks) and np.all(
        np.isfinite(efficiencies) & (efficiencies > 0.0)
    )
    if valid:
        gm = float(np.exp(np.mean(np.log(efficiencies))))
        minimum = float(np.min(efficiencies))
        median = float(np.median(efficiencies))
        maximum = float(np.max(efficiencies))
        status = "ok"
    else:
        gm = minimum = median = maximum = math.nan
        status = "undefined_block_metric"
    rhats = np.asarray([b.maximum_rhat for b in blocks])
    max_rhat = float(np.max(rhats)) if not np.any(np.isnan(rhats)) else math.nan
    moments = np.asarray([b.maximum_standardized_moment_error for b in blocks])
    max_moment = float(np.max(moments)) if np.all(np.isfinite(moments)) else math.nan
    nonfinite = sum(
        b.warmup_nonfinite_trajectories + b.retained_nonfinite_trajectories
        for b in blocks
    )
    return BlockAggregate(
        status, gm, minimum, median, maximum,
        float(np.mean([b.acceptance for b in blocks])),
        max_rhat, max_moment, int(nonfinite),
    )


def _conditional_diagnostic(eta_samples: np.ndarray, tau: float) -> ConditionalDiagnostic:
    values = np.asarray(eta_samples, dtype=np.float64)
    mean = float(np.mean(values))
    second = float(np.mean(values * values))
    exact_second = tau * tau
    return ConditionalDiagnostic(
        mean, 0.0, mean, second, exact_second, second - exact_second
    )


# ---------------------------------------------------------------------------
# Input validation and deterministic preparation.
# ---------------------------------------------------------------------------


def _finite_float(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly True or False.")
    return value


def _float_grid(name: str, values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(_finite_float(f"{name}[{i}]", value, positive=True) for i, value in enumerate(values))
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates.")
    return result


def _int_grid(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(_integer(f"{name}[{i}]", value) for i, value in enumerate(values))
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates.")
    return result


def _seed_blocks(values: Sequence[Sequence[int]], n_chains: int) -> tuple[tuple[int, ...], ...]:
    blocks: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for block_index, block in enumerate(values):
        seeds = tuple(_integer(f"seed_blocks[{block_index}]", seed, minimum=0) for seed in block)
        if len(seeds) != n_chains:
            raise ValueError(f"seed block {block_index + 1} must contain {n_chains} seeds.")
        for seed in seeds:
            if seed in seen:
                raise ValueError("Every tuning seed must be unique.")
            if seed > 2**32 - 1:
                raise ValueError("Tuning seeds must fit unsigned 32-bit integers.")
            seen.add(seed)
        blocks.append(seeds)
    if not blocks:
        raise ValueError("At least one tuning seed block is required.")
    return tuple(blocks)


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_mismatches(prepared: PreparedInputs) -> tuple[str, ...]:
    """Seal the declared production experiment while retaining CUSTOM mode."""

    expected = {
        "beta": 1.0,
        "sigma_x": 2.0,
        "sigma_y": 0.2,
        "banana_b": 0.25,
        "mass": [[0.25, 0.0], [0.0, 0.49019607843137253]],
        "starts": [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
        "eta_starts": [-1.0, 1.0, 1.0, -1.0],
        "epsilon": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08],
        "L": [10, 20, 40, 80, 120, 160],
        "seeds": [
            [123, 456, 789, 101112],
            [3545294353, 3961674632, 3627514536, 3198582250],
            [788645248, 3651519711, 1811141324, 1814869264],
            [3192285063, 2251015777, 396444798, 765150135],
            [1562880678, 1785952873, 2908883741, 2216662531],
        ],
        "kappa": [0.25, 0.4, 0.5, 0.65],
        "tau": [0.25, 0.5, 1.0],
        "Q": [0.05, 0.1, 0.25, 0.5],
        "a": [2.0, 4.0, 8.0, 16.0],
        "anchors": [[0.04, 80], [0.05, 40], [0.06, 20]],
        "stage1_blocks": [0],
        "stage1_lengths": [250, 1000],
        "stage1_keep": 12,
        "stage2_blocks": [1, 2],
        "stage2_lengths": [500, 2000],
        "stage2_keep": 8,
        "automatic_boundary_expansion": True,
        "boundary_values": [0.125, 0.8, 0.125, 2.0, 0.025, 1.0, 1.0, 32.0],
        "boundary_keep": 4,
        "forced": [[0.5, 0.5, 0.1, 8.0, 0.05, 40]],
        "production_lengths": [1000, 4000],
        "observable_policy": SUPPORTED_OBSERVABLE_POLICY,
        "run_self_tests": True,
        "run_label": "PRODUCTION TUNING",
        "output": CANONICAL_OUTPUT_FILENAME,
    }
    actual = {
        "beta": prepared.beta,
        "sigma_x": prepared.sigma_x,
        "sigma_y": prepared.sigma_y,
        "banana_b": prepared.banana_b,
        "mass": prepared.mass_matrix.tolist(),
        "starts": prepared.standardized_starts.tolist(),
        "eta_starts": prepared.conditional_standardized_starts.tolist(),
        "epsilon": list(prepared.epsilon_values),
        "L": list(prepared.L_values),
        "seeds": [list(block) for block in prepared.seed_blocks],
        "kappa": list(prepared.kappa_values),
        "tau": list(prepared.tau_values),
        "Q": list(prepared.Q_values),
        "a": list(prepared.a_values),
        "anchors": [[eps, L] for eps, L in prepared.stage1_anchor_pairs],
        "stage1_blocks": list(prepared.stage1_block_indices),
        "stage1_lengths": [prepared.stage1_n_warmup, prepared.stage1_n_retained],
        "stage1_keep": prepared.stage1_keep_method_tuples,
        "stage2_blocks": list(prepared.stage2_block_indices),
        "stage2_lengths": [prepared.stage2_n_warmup, prepared.stage2_n_retained],
        "stage2_keep": prepared.stage2_keep_fixed_configurations,
        "automatic_boundary_expansion": prepared.automatic_boundary_expansion,
        "boundary_values": [
            prepared.boundary_kappa_lower, prepared.boundary_kappa_upper,
            prepared.boundary_tau_lower, prepared.boundary_tau_upper,
            prepared.boundary_Q_lower, prepared.boundary_Q_upper,
            prepared.boundary_a_lower, prepared.boundary_a_upper,
        ],
        "boundary_keep": prepared.boundary_keep_fixed_configurations,
        "forced": [[c.kappa, c.tau, c.Q, c.a, c.epsilon, c.L] for c in prepared.forced_full_configurations],
        "production_lengths": [prepared.n_warmup, prepared.n_retained],
        "observable_policy": prepared.observable_policy,
        "run_self_tests": prepared.run_self_tests,
        "run_label": prepared.run_label,
        "output": prepared.output_filename,
    }
    return tuple(key for key in expected if actual[key] != expected[key])


def _prepare_inputs(inputs: THCRunInputs) -> PreparedInputs:
    if not isinstance(inputs, THCRunInputs):
        raise TypeError("run_conditional_shear_tuning expects THCRunInputs.")
    beta = _finite_float("beta", inputs.beta, positive=True)
    sigma_x = _finite_float("sigma_x", inputs.sigma_x, positive=True)
    sigma_y = _finite_float("sigma_y", inputs.sigma_y, positive=True)
    banana_b = _finite_float("banana_b", inputs.banana_b)
    mass = np.asarray(inputs.mass_matrix, dtype=np.float64)
    if mass.shape != (2, 2) or not np.all(np.isfinite(mass)):
        raise ValueError("mass_matrix must be finite 2 by 2.")
    if not np.allclose(mass, mass.T, rtol=0.0, atol=1e-14):
        raise ValueError("mass_matrix must be symmetric.")
    try:
        mass_cholesky = np.linalg.cholesky(mass)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass_matrix must be positive definite.") from error
    inverse_mass = np.linalg.inv(mass)
    starts = np.asarray(inputs.standardized_starts, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1] != 2 or starts.shape[0] < 2:
        raise ValueError("standardized_starts must have shape (at least 2,2).")
    if not np.all(np.isfinite(starts)):
        raise ValueError("standardized_starts must be finite.")
    eta_starts = np.asarray(inputs.conditional_standardized_starts, dtype=np.float64)
    if eta_starts.shape != (starts.shape[0],) or not np.all(np.isfinite(eta_starts)):
        raise ValueError("conditional_standardized_starts must have one finite value per chain.")
    observable_policy = str(inputs.observable_policy)
    if observable_policy != SUPPORTED_OBSERVABLE_POLICY:
        raise ValueError(f"observable_policy must be {SUPPORTED_OBSERVABLE_POLICY}.")
    observable_names, exact_means, exact_sds = _observable_specification(
        beta, sigma_x, sigma_y, banana_b
    )
    epsilon_values = _float_grid("epsilon_values", inputs.epsilon_values)
    L_values = _int_grid("L_values", inputs.L_values)
    seeds = _seed_blocks(inputs.seed_blocks, starts.shape[0])
    kappa_values = _float_grid("kappa_values", inputs.kappa_values)
    tau_values = _float_grid("tau_values", inputs.tau_values)
    Q_values = _float_grid("Q_values", inputs.Q_values)
    a_values = _float_grid("a_values", inputs.a_values)

    anchors: list[tuple[float, int]] = []
    for i, pair in enumerate(inputs.stage1_anchor_pairs):
        if len(pair) != 2:
            raise ValueError(f"stage1_anchor_pairs[{i}] must be (epsilon,L).")
        epsilon = _finite_float(f"stage1_anchor_pairs[{i}].epsilon", pair[0], positive=True)
        L = _integer(f"stage1_anchor_pairs[{i}].L", pair[1])
        if epsilon not in epsilon_values or L not in L_values:
            raise ValueError("Every stage-1 anchor must belong to the common epsilon-L grid.")
        anchors.append((epsilon, L))
    if not anchors or len(set(anchors)) != len(anchors):
        raise ValueError("stage1_anchor_pairs must be nonempty and unique.")

    def block_indices(name: str, values: Sequence[int]) -> tuple[int, ...]:
        one_based = tuple(_integer(name, value) for value in values)
        zero_based = tuple(value - 1 for value in one_based)
        if not zero_based or len(set(zero_based)) != len(zero_based):
            raise ValueError(f"{name} must be nonempty and unique.")
        if min(zero_based) < 0 or max(zero_based) >= len(seeds):
            raise ValueError(f"{name} is outside the five tuning blocks.")
        return zero_based

    stage1_blocks = block_indices("stage1_block_indices", inputs.stage1_block_indices)
    stage2_blocks = block_indices("stage2_block_indices", inputs.stage2_block_indices)
    if set(stage1_blocks) & set(stage2_blocks):
        raise ValueError("Stage-1 and stage-2 block roles must be disjoint.")

    automatic_boundary_expansion = _boolean(
        "automatic_boundary_expansion", inputs.automatic_boundary_expansion
    )
    boundary_kappa_lower = _finite_float(
        "boundary_kappa_lower", inputs.boundary_kappa_lower, positive=True
    )
    boundary_kappa_upper = _finite_float(
        "boundary_kappa_upper", inputs.boundary_kappa_upper, positive=True
    )
    boundary_tau_lower = _finite_float(
        "boundary_tau_lower", inputs.boundary_tau_lower, positive=True
    )
    boundary_tau_upper = _finite_float(
        "boundary_tau_upper", inputs.boundary_tau_upper, positive=True
    )
    boundary_Q_lower = _finite_float(
        "boundary_Q_lower", inputs.boundary_Q_lower, positive=True
    )
    boundary_Q_upper = _finite_float(
        "boundary_Q_upper", inputs.boundary_Q_upper, positive=True
    )
    boundary_a_lower = _finite_float(
        "boundary_a_lower", inputs.boundary_a_lower, positive=True
    )
    boundary_a_upper = _finite_float(
        "boundary_a_upper", inputs.boundary_a_upper, positive=True
    )
    boundary_keep = _integer(
        "boundary_keep_fixed_configurations",
        inputs.boundary_keep_fixed_configurations,
    )
    outward_checks = (
        ("kappa", boundary_kappa_lower, min(kappa_values), boundary_kappa_upper, max(kappa_values)),
        ("tau", boundary_tau_lower, min(tau_values), boundary_tau_upper, max(tau_values)),
        ("Q", boundary_Q_lower, min(Q_values), boundary_Q_upper, max(Q_values)),
        ("a", boundary_a_lower, min(a_values), boundary_a_upper, max(a_values)),
    )
    for name, lower, grid_minimum, upper, grid_maximum in outward_checks:
        if not lower < grid_minimum:
            raise ValueError(f"boundary_{name}_lower must lie below the base grid.")
        if not upper > grid_maximum:
            raise ValueError(f"boundary_{name}_upper must lie above the base grid.")

    forced: list[FixedConfiguration] = []
    for i, row in enumerate(inputs.forced_full_configurations):
        if len(row) != 6:
            raise ValueError(
                "Each forced configuration must be (kappa,tau,Q,a,epsilon,L)."
            )
        configuration = FixedConfiguration(
            _finite_float(f"forced[{i}].kappa", row[0], positive=True),
            _finite_float(f"forced[{i}].tau", row[1], positive=True),
            _finite_float(f"forced[{i}].Q", row[2], positive=True),
            _finite_float(f"forced[{i}].a", row[3], positive=True),
            _finite_float(f"forced[{i}].epsilon", row[4], positive=True),
            _integer(f"forced[{i}].L", row[5]),
        )
        if configuration.epsilon not in epsilon_values or configuration.L not in L_values:
            raise ValueError("Forced configurations must use the common epsilon-L grid.")
        forced.append(configuration)
    if len(set(forced)) != len(forced):
        raise ValueError("forced_full_configurations contains duplicates.")

    stage1_n_warmup = _integer("stage1_n_warmup", inputs.stage1_n_warmup, minimum=0)
    stage1_n_retained = _integer("stage1_n_retained", inputs.stage1_n_retained, minimum=4)
    stage1_keep = _integer("stage1_keep_method_tuples", inputs.stage1_keep_method_tuples)
    stage2_n_warmup = _integer("stage2_n_warmup", inputs.stage2_n_warmup, minimum=0)
    stage2_n_retained = _integer("stage2_n_retained", inputs.stage2_n_retained, minimum=4)
    stage2_keep = _integer(
        "stage2_keep_fixed_configurations", inputs.stage2_keep_fixed_configurations
    )
    n_warmup = _integer("n_warmup", inputs.n_warmup, minimum=0)
    n_retained = _integer("n_retained", inputs.n_retained, minimum=4)
    n_workers = _integer("n_workers", inputs.n_workers)
    run_self_tests = _boolean("run_self_tests", inputs.run_self_tests)
    print_progress = _boolean("print_progress", inputs.print_progress)
    print_report = _boolean("print_report", inputs.print_report)
    allow_overwrite = _boolean("allow_overwrite", inputs.allow_overwrite)
    if not isinstance(inputs.run_mode, str):
        raise TypeError("run_mode must be a string.")
    if not isinstance(inputs.run_label, str):
        raise TypeError("run_label must be a string.")
    if not isinstance(inputs.output_filename, str):
        raise TypeError("output_filename must be a string.")
    if inputs.output_directory is not None and not isinstance(
        inputs.output_directory, (str, os.PathLike)
    ):
        raise TypeError("output_directory must be None, a string, or os.PathLike.")
    run_mode = inputs.run_mode.strip().upper()
    run_label = inputs.run_label.strip()
    output_filename = inputs.output_filename.strip()
    if run_mode not in {"CANONICAL", "CUSTOM"}:
        raise ValueError("run_mode must be CANONICAL or CUSTOM.")
    if not run_label:
        raise ValueError("run_label cannot be empty.")
    if run_mode == "CUSTOM" and not run_label.startswith("CUSTOM:"):
        raise ValueError("CUSTOM run_label must begin with 'CUSTOM:'.")
    if Path(output_filename).name != output_filename or not output_filename.casefold().endswith(".txt"):
        raise ValueError("output_filename must be one plain .txt filename.")
    if (
        run_mode == "CUSTOM"
        and output_filename.casefold() == CANONICAL_OUTPUT_FILENAME.casefold()
    ):
        raise ValueError("CUSTOM mode must use a noncanonical output filename.")

    payload = {
        "protocol": PROTOCOL_VERSION,
        "target": [beta, sigma_x, sigma_y, banana_b],
        "mass": mass.tolist(),
        "starts": starts.tolist(),
        "eta_starts": eta_starts.tolist(),
        "observables": observable_policy,
        "epsilon": epsilon_values,
        "L": L_values,
        "seeds": seeds,
        "kappa": kappa_values,
        "tau": tau_values,
        "Q": Q_values,
        "a": a_values,
        "anchors": anchors,
        "stage1": [stage1_blocks, stage1_n_warmup, stage1_n_retained, stage1_keep],
        "stage2": [stage2_blocks, stage2_n_warmup, stage2_n_retained, stage2_keep],
        "boundary": {
            "automatic": automatic_boundary_expansion,
            "kappa": [boundary_kappa_lower, boundary_kappa_upper],
            "tau": [boundary_tau_lower, boundary_tau_upper],
            "Q": [boundary_Q_lower, boundary_Q_upper],
            "a": [boundary_a_lower, boundary_a_upper],
            "keep": boundary_keep,
            "rule": "one-coordinate-at-a-time-from-base-five-block-winner; one layer; no corners",
            "maximum_method_count": 4,
            "layer_count": 1,
            "joint_corners": False,
        },
        "forced": [[c.kappa,c.tau,c.Q,c.a,c.epsilon,c.L] for c in forced],
        "production": [n_warmup, n_retained],
    }
    prepared = PreparedInputs(
        beta=beta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        banana_b=banana_b,
        target_parameters=np.asarray([sigma_x, sigma_y, banana_b], dtype=np.float64),
        mass_matrix=np.ascontiguousarray(mass),
        inverse_mass_matrix=np.ascontiguousarray(inverse_mass),
        mass_cholesky=np.ascontiguousarray(mass_cholesky),
        standardized_starts=np.ascontiguousarray(starts),
        physical_starts=_latent_to_physical(starts, beta, sigma_x, sigma_y, banana_b),
        conditional_standardized_starts=np.ascontiguousarray(eta_starts),
        observable_policy=observable_policy,
        observable_names=observable_names,
        exact_observable_means=exact_means,
        exact_observable_sds=exact_sds,
        epsilon_values=epsilon_values,
        L_values=L_values,
        seed_blocks=seeds,
        kappa_values=kappa_values,
        tau_values=tau_values,
        Q_values=Q_values,
        a_values=a_values,
        stage1_anchor_pairs=tuple(anchors),
        stage1_block_indices=stage1_blocks,
        stage1_n_warmup=stage1_n_warmup,
        stage1_n_retained=stage1_n_retained,
        stage1_keep_method_tuples=stage1_keep,
        stage2_block_indices=stage2_blocks,
        stage2_n_warmup=stage2_n_warmup,
        stage2_n_retained=stage2_n_retained,
        stage2_keep_fixed_configurations=stage2_keep,
        automatic_boundary_expansion=automatic_boundary_expansion,
        boundary_kappa_lower=boundary_kappa_lower,
        boundary_kappa_upper=boundary_kappa_upper,
        boundary_tau_lower=boundary_tau_lower,
        boundary_tau_upper=boundary_tau_upper,
        boundary_Q_lower=boundary_Q_lower,
        boundary_Q_upper=boundary_Q_upper,
        boundary_a_lower=boundary_a_lower,
        boundary_a_upper=boundary_a_upper,
        boundary_keep_fixed_configurations=boundary_keep,
        forced_full_configurations=tuple(forced),
        n_warmup=n_warmup,
        n_retained=n_retained,
        n_workers=n_workers,
        run_self_tests=run_self_tests,
        print_progress=print_progress,
        print_report=print_report,
        run_mode=run_mode,
        run_label=run_label,
        output_directory=inputs.output_directory,
        output_filename=output_filename,
        allow_overwrite=allow_overwrite,
        input_fingerprint=_fingerprint(payload),
    )
    max_methods = len(kappa_values)*len(tau_values)*len(Q_values)*len(a_values)
    if prepared.stage1_keep_method_tuples > max_methods:
        raise ValueError("stage1_keep_method_tuples exceeds method-grid size.")
    max_stage2 = prepared.stage1_keep_method_tuples*len(epsilon_values)*len(L_values)
    if prepared.stage2_keep_fixed_configurations > max_stage2:
        raise ValueError("stage2_keep_fixed_configurations exceeds stage-2 size.")
    if prepared.boundary_keep_fixed_configurations > 4:
        raise ValueError(
            "boundary_keep_fixed_configurations cannot exceed the four "
            "one-coordinate boundary methods."
        )
    if prepared.stage1_n_warmup + prepared.stage1_n_retained > prepared.n_warmup + prepared.n_retained:
        raise ValueError("Stage-1 lengths exceed the cached production stream.")
    if prepared.stage2_n_warmup + prepared.stage2_n_retained > prepared.n_warmup + prepared.n_retained:
        raise ValueError("Stage-2 lengths exceed the cached production stream.")
    if run_mode == "CANONICAL":
        mismatches = _canonical_mismatches(prepared)
        if mismatches:
            raise ValueError("CANONICAL input mismatch in: " + ", ".join(mismatches))
    return prepared


def _resolve_output_destination(prepared: PreparedInputs) -> Path:
    if prepared.output_directory is None:
        try:
            script_path = Path(__file__).resolve()
        except NameError:
            directory = Path.cwd().resolve()
        else:
            directory = (
                script_path.parent
                if script_path.name
                == "banana_conditional_shear_thc_v4_tuner_GREX_PORTABLE.py"
                else Path.cwd().resolve()
            )
    else:
        directory = Path(prepared.output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / prepared.output_filename
    if destination.exists() and destination.is_dir():
        raise ValueError(f"Output destination is a directory: {destination}")
    if destination.exists() and not prepared.allow_overwrite:
        raise FileExistsError(f"Output exists and allow_overwrite=False: {destination}")
    return destination


# ---------------------------------------------------------------------------
# Block execution, staged halving, and fixed-representative selection.
# ---------------------------------------------------------------------------


def _run_chain_job(
    prepared: PreparedInputs,
    configuration: FixedConfiguration,
    streams: tuple[np.ndarray, np.ndarray, np.ndarray],
    chain_index: int,
    n_warmup: int,
    n_retained: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int]:
    total = n_warmup + n_retained
    physical, thermostat, uniforms = streams
    return _simulate_chain(
        physical[:total], thermostat[:total], uniforms[:total],
        prepared.physical_starts[chain_index],
        float(prepared.conditional_standardized_starts[chain_index]),
        prepared.beta, configuration.kappa, configuration.tau,
        configuration.Q, configuration.a, configuration.epsilon,
        configuration.L, n_warmup, n_retained,
        prepared.target_parameters, prepared.mass_cholesky,
        prepared.inverse_mass_matrix,
    )


def _run_configuration_block(
    configuration: FixedConfiguration,
    block_index: int,
    n_warmup: int,
    n_retained: int,
    stream_caches: Sequence[Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    prepared: PreparedInputs,
) -> THCBlockResult:
    jobs = [
        (prepared, configuration, stream_caches[block_index][chain], chain, n_warmup, n_retained)
        for chain in range(prepared.n_chains)
    ]
    if prepared.n_workers == 1:
        outputs = [_run_chain_job(*job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=min(prepared.n_workers, prepared.n_chains)) as pool:
            futures = [pool.submit(_run_chain_job, *job) for job in jobs]
            outputs = [future.result() for future in futures]
    samples = np.asarray([output[0] for output in outputs])
    eta_samples = np.asarray([output[1] for output in outputs])
    accepted = sum(output[2] for output in outputs)
    warmup_nonfinite = sum(output[3] for output in outputs)
    retained_nonfinite = sum(output[4] for output in outputs)
    retained_gradients = sum(output[6] for output in outputs)
    physical = _analyze_block(
        samples=samples,
        epsilon=configuration.epsilon,
        L=configuration.L,
        block_index=block_index,
        seeds=prepared.seed_blocks[block_index],
        accepted_retained=accepted,
        retained_gradient_calls=retained_gradients,
        warmup_nonfinite=warmup_nonfinite,
        retained_nonfinite=retained_nonfinite,
        observable_names=prepared.observable_names,
        exact_observable_means=prepared.exact_observable_means,
        exact_observable_sds=prepared.exact_observable_sds,
    )
    return THCBlockResult(physical, _conditional_diagnostic(eta_samples, configuration.tau))


def _evaluate_fixed_configuration(
    configuration: FixedConfiguration,
    *,
    block_indices: Sequence[int],
    n_warmup: int,
    n_retained: int,
    origin: str,
    stream_caches: Sequence[Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    prepared: PreparedInputs,
) -> ConfigurationResult:
    block_results = tuple(
        _run_configuration_block(
            configuration, block_index, n_warmup, n_retained,
            stream_caches, prepared,
        )
        for block_index in block_indices
    )
    aggregate = _aggregate_blocks(tuple(item.physical for item in block_results))
    return ConfigurationResult(
        configuration, origin, aggregate.status, block_results,
        aggregate.geometric_mean_efficiency,
        aggregate.minimum_block_efficiency,
        aggregate.median_block_efficiency,
        aggregate.maximum_block_efficiency,
        aggregate.mean_acceptance,
        aggregate.maximum_rhat,
        aggregate.maximum_standardized_moment_error,
        aggregate.total_nonfinite_trajectories,
    )


def _rank_key(result: ConfigurationResult) -> tuple[float, ...]:
    c = result.configuration
    if result.status != "ok" or not math.isfinite(result.geometric_mean_efficiency):
        return (-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf)
    # The first coordinate is the declared score.  Remaining coordinates only
    # make exact numerical ties deterministic and never use R-hat/moment gates.
    return (
        result.geometric_mean_efficiency,
        result.minimum_block_efficiency,
        result.median_block_efficiency,
        -float(result.total_nonfinite_trajectories),
        -c.kappa, -c.tau, -c.Q, -c.a,
        -(c.epsilon * c.L),
    )


def _rank_defined(
    results: Sequence[ConfigurationResult],
    keep: int,
) -> tuple[ConfigurationResult, ...]:
    defined = [
        result for result in results
        if result.status == "ok"
        and math.isfinite(result.geometric_mean_efficiency)
        and result.geometric_mean_efficiency > 0.0
    ]
    defined.sort(key=_rank_key, reverse=True)
    return tuple(defined[: min(keep, len(defined))])


def _select_representative(results: Sequence[ConfigurationResult]) -> ConfigurationResult:
    ranked = _rank_defined(results, len(results))
    if not ranked:
        raise RuntimeError("No full five-block configuration has a defined score.")
    return ranked[0]


def _boundary_method_layer(
    base_winner: ConfigurationResult,
    prepared: PreparedInputs,
) -> tuple[tuple[MethodTuple, ...], tuple[str, ...]]:
    """Return one-coordinate outward moves from active method boundaries."""

    configuration = base_winner.configuration
    candidates: list[MethodTuple] = []
    labels: list[str] = []

    def add(label: str, *, kappa: float = configuration.kappa,
            tau: float = configuration.tau, Q: float = configuration.Q,
            a: float = configuration.a) -> None:
        candidate = MethodTuple(kappa, tau, Q, a)
        if candidate not in candidates:
            candidates.append(candidate)
            labels.append(label)

    if configuration.kappa == min(prepared.kappa_values):
        add("kappa_lower", kappa=prepared.boundary_kappa_lower)
    elif configuration.kappa == max(prepared.kappa_values):
        add("kappa_upper", kappa=prepared.boundary_kappa_upper)
    if configuration.tau == min(prepared.tau_values):
        add("tau_lower", tau=prepared.boundary_tau_lower)
    elif configuration.tau == max(prepared.tau_values):
        add("tau_upper", tau=prepared.boundary_tau_upper)
    if configuration.Q == min(prepared.Q_values):
        add("Q_lower", Q=prepared.boundary_Q_lower)
    elif configuration.Q == max(prepared.Q_values):
        add("Q_upper", Q=prepared.boundary_Q_upper)
    if configuration.a == min(prepared.a_values):
        add("a_lower", a=prepared.boundary_a_lower)
    elif configuration.a == max(prepared.a_values):
        add("a_upper", a=prepared.boundary_a_upper)
    return tuple(candidates), tuple(labels)


# ---------------------------------------------------------------------------
# Independent mathematical/native preflight oracles.
# ---------------------------------------------------------------------------


def _reference_potential(q: np.ndarray, parameters: np.ndarray) -> float:
    sx, sy, bend = (float(value) for value in parameters)
    x, y = (float(value) for value in q)
    residual = y - bend * (x*x - sx*sx)
    return 0.5 * (x*x/(sx*sx) + residual*residual/(sy*sy))


def _reference_gradient(q: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    sx, sy, bend = (float(value) for value in parameters)
    x, y = (float(value) for value in q)
    residual = y - bend * (x*x - sx*sx)
    return np.asarray([
        x/(sx*sx) - 2.0*bend*x*residual/(sy*sy),
        residual/(sy*sy),
    ])


def _reference_shear(s: float, kappa: float, a: float) -> tuple[float, float]:
    tangent = math.tanh(s/a)
    return kappa*a*tangent, kappa*(1.0-tangent*tangent)


def _reference_hamiltonian(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    beta: float,
    configuration: FixedConfiguration,
    parameters: np.ndarray,
    inverse_mass: np.ndarray,
) -> float:
    c, _dc = _reference_shear(s, configuration.kappa, configuration.a)
    z = np.asarray([p[0] + c*p[1], p[1]])
    eta = s - q[0]
    return (
        _reference_potential(q, parameters)
        + eta*eta/(2.0*beta*configuration.tau**2)
        + 0.5*float(z @ inverse_mass @ z)
        + p_s*p_s/(2.0*configuration.Q)
    )


def _reference_proposal(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    beta: float,
    configuration: FixedConfiguration,
    parameters: np.ndarray,
    inverse_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, int, bool]:
    qn = np.asarray(q, dtype=np.float64).copy()
    pn = np.asarray(p, dtype=np.float64).copy()
    sn = float(s)
    psn = float(p_s)
    gradient = _reference_gradient(qn, parameters)
    inverse_beta_tau2 = 1.0/(beta*configuration.tau**2)
    eta = sn-qn[0]
    pn -= 0.5*configuration.epsilon*np.asarray([
        gradient[0]-eta*inverse_beta_tau2, gradient[1]
    ])
    psn -= 0.5*configuration.epsilon*eta*inverse_beta_tau2
    calls = 0
    for step in range(configuration.L):
        sn += 0.5*configuration.epsilon*psn/configuration.Q
        c, dc = _reference_shear(sn, configuration.kappa, configuration.a)
        z = np.asarray([pn[0]+c*pn[1], pn[1]])
        u = inverse_mass @ z
        qn[0] += configuration.epsilon*u[0]
        qn[1] += configuration.epsilon*(c*u[0]+u[1])
        psn -= configuration.epsilon*dc*pn[1]*u[0]
        sn += 0.5*configuration.epsilon*psn/configuration.Q
        gradient = _reference_gradient(qn, parameters)
        calls += 1
        kick = 0.5*configuration.epsilon if step == configuration.L-1 else configuration.epsilon
        eta = sn-qn[0]
        pn -= kick*np.asarray([gradient[0]-eta*inverse_beta_tau2, gradient[1]])
        psn -= kick*eta*inverse_beta_tau2
    finite = bool(
        np.all(np.isfinite(qn)) and np.all(np.isfinite(pn))
        and math.isfinite(sn) and math.isfinite(psn)
    )
    return qn, pn, sn, psn, gradient, calls, finite


def _flow_state(
    state: np.ndarray,
    epsilon: float,
    L: int,
    configuration: FixedConfiguration,
    prepared: PreparedInputs,
) -> np.ndarray:
    q = np.asarray(state[:2], dtype=np.float64)
    p = np.asarray(state[2:4], dtype=np.float64)
    s = float(state[4])
    p_s = float(state[5])
    output = _proposal(
        q, p, s, p_s, _target_gradient(q, prepared.target_parameters),
        prepared.beta, configuration.kappa, configuration.tau,
        configuration.Q, configuration.a, epsilon, L,
        prepared.target_parameters, prepared.inverse_mass_matrix,
    )
    if not output[6]:
        raise RuntimeError("Preflight flow unexpectedly became nonfinite.")
    return np.asarray([
        output[0][0], output[0][1], output[1][0], output[1][1],
        output[2], output[3],
    ])


def _runtime_environment() -> str:
    numpy_version = np.__version__
    if NUMBA_AVAILABLE:
        native = f"Numba={numba.__version__}; llvmlite={llvmlite.__version__}"
    else:
        native = "Numba=unavailable (Python fallback)"
    return (
        f"platform={platform.platform()}; Python={platform.python_version()}; "
        f"NumPy={numpy_version}; {native}"
    )


def _implementation_fingerprint() -> str:
    code_type = type((lambda: None).__code__)

    def pack(tag: bytes, items: Sequence[bytes]) -> bytes:
        payload = bytearray(tag)
        for item in items:
            payload.extend(len(item).to_bytes(8, "big"))
            payload.extend(item)
        return bytes(payload)

    def constant_payload(constant: object) -> bytes:
        if isinstance(constant, code_type):
            return code_payload(constant)
        if constant is None:
            return b"none"
        if constant is Ellipsis:
            return b"ellipsis"
        if isinstance(constant, bool):
            return b"bool:1" if constant else b"bool:0"
        if isinstance(constant, int):
            return b"int:" + str(constant).encode("ascii")
        if isinstance(constant, float):
            return b"float:" + constant.hex().encode("ascii")
        if isinstance(constant, complex):
            return pack(
                b"complex",
                [constant_payload(constant.real), constant_payload(constant.imag)],
            )
        if isinstance(constant, str):
            return b"str:" + constant.encode("utf-8")
        if isinstance(constant, bytes):
            return b"bytes:" + constant
        if isinstance(constant, tuple):
            return pack(b"tuple", [constant_payload(item) for item in constant])
        if isinstance(constant, (set, frozenset)):
            items = sorted(constant_payload(item) for item in constant)
            tag = b"frozenset" if isinstance(constant, frozenset) else b"set"
            return pack(tag, items)
        if isinstance(constant, dict):
            entries = sorted(
                pack(b"item", [constant_payload(key), constant_payload(value)])
                for key, value in constant.items()
            )
            return pack(b"dict", entries)
        return (
            f"{type(constant).__module__}.{type(constant).__qualname__}:"
            f"{repr(constant)}"
        ).encode("utf-8")

    def code_payload(code: object) -> bytes:
        pieces = [
            getattr(code, "co_code"),
            constant_payload(getattr(code, "co_names")),
            constant_payload(getattr(code, "co_varnames")),
            constant_payload(getattr(code, "co_freevars")),
            constant_payload(getattr(code, "co_cellvars")),
            constant_payload((
                getattr(code, "co_argcount"),
                getattr(code, "co_posonlyargcount"),
                getattr(code, "co_kwonlyargcount"),
            )),
        ]
        for constant in getattr(code, "co_consts"):
            pieces.append(constant_payload(constant))
        return pack(b"code", pieces)

    functions = (
        _potential_energy, _target_gradient, _shear_and_derivative,
        _kinetic_terms, _hamiltonian, _proposal, _simulate_chain,
        _make_random_streams, _multi_chain_geyer_iat,
        _autocovariance_fft, _basic_rhat, _split_chains,
        _average_ranks, _standard_normal_ppf, _rank_normalize,
        _rank_normalized_split_folded_rhat, _observable_specification,
        _latent_to_physical, _physical_to_latent, _physical_observables,
        _analyze_block, _aggregate_blocks, _conditional_diagnostic,
        _finite_float, _integer, _boolean, _float_grid, _int_grid,
        _seed_blocks, _jsonable, _fingerprint, _prepare_inputs,
        _canonical_mismatches, _resolve_output_destination,
        _run_chain_job, _run_configuration_block,
        _evaluate_fixed_configuration, _rank_key, _rank_defined,
        _select_representative, _boundary_method_layer,
        _reference_potential, _reference_gradient, _reference_shear,
        _reference_hamiltonian, _reference_proposal, _flow_state,
        run_self_tests, _number, _configuration_text, _result_summary,
        _append_result_rows, _append_five_block_details, _build_report,
        _stream_second_moment, run_conditional_shear_tuning,
    )
    digest = hashlib.sha256()
    schema = {
        "constants": [
            PROTOCOL_VERSION, IMPLEMENTATION_VERSION, DIAGNOSTICS_VERSION,
            RNG_VERSION, SUPPORTED_OBSERVABLE_POLICY,
            HMC_SHARED_COMMON_INPUT_SHA256,
        ],
        "THCRunInputs": list(THCRunInputs.__annotations__.items()),
        "PreparedInputs": list(PreparedInputs.__annotations__.items()),
        "FixedConfiguration": list(FixedConfiguration.__annotations__.items()),
        "ConfigurationResult": list(ConfigurationResult.__annotations__.items()),
    }
    digest.update(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for function in functions:
        candidate = getattr(function, "py_func", function)
        digest.update(code_payload(candidate.__code__))
    return digest.hexdigest()


def _whole_source_sha256() -> str:
    try:
        source_path = Path(__file__).resolve()
        if (
            source_path.name
            == "banana_conditional_shear_thc_v4_tuner_GREX_PORTABLE.py"
            and source_path.is_file()
        ):
            return hashlib.sha256(source_path.read_bytes()).hexdigest()
    except NameError:
        pass
    return "semantic-source-protocol-sha256:" + _implementation_fingerprint()


def run_self_tests(prepared: PreparedInputs) -> tuple[str, ...]:
    """Run independent math, map, RNG, diagnostic, and native checks."""

    messages: list[str] = []
    parameters = prepared.target_parameters
    control = FixedConfiguration(0.5, 0.5, 0.1, 8.0, 0.05, 40)

    # Target oracle and exact latent/physical start round trip.
    q = np.asarray([0.73, -0.41])
    native_u = float(_potential_energy(q, parameters))
    reference_u = _reference_potential(q, parameters)
    native_gradient = np.asarray(_target_gradient(q, parameters))
    reference_gradient = _reference_gradient(q, parameters)
    finite_difference = np.empty(2)
    h = 1e-6
    for index in range(2):
        shift = np.zeros(2)
        shift[index] = h
        finite_difference[index] = (
            _reference_potential(q+shift, parameters)
            - _reference_potential(q-shift, parameters)
        )/(2.0*h)
    if not math.isclose(native_u, reference_u, rel_tol=0.0, abs_tol=1e-13):
        raise RuntimeError("Self-test failed: target potential oracle.")
    if not np.allclose(native_gradient, reference_gradient, rtol=0.0, atol=2e-13):
        raise RuntimeError("Self-test failed: target gradient oracle.")
    if not np.allclose(native_gradient, finite_difference, rtol=2e-8, atol=2e-8):
        raise RuntimeError("Self-test failed: finite-difference target gradient.")
    recovered = _physical_to_latent(
        prepared.physical_starts, prepared.beta, prepared.sigma_x,
        prepared.sigma_y, prepared.banana_b,
    )
    if not np.allclose(recovered, prepared.standardized_starts, rtol=0.0, atol=2e-14):
        raise RuntimeError("Self-test failed: Banana start-coordinate round trip.")
    messages.append("PASS: Banana target, gradient, moments, and start-coordinate oracles")

    # Bounded shear, derivative, determinant, and Hamiltonian oracle.
    for s_value in (-50.0, -3.1, 0.0, 2.4, 50.0):
        c_native, dc_native = _shear_and_derivative(s_value, control.kappa, control.a)
        c_reference, dc_reference = _reference_shear(s_value, control.kappa, control.a)
        derivative_fd = (
            _reference_shear(s_value+h, control.kappa, control.a)[0]
            - _reference_shear(s_value-h, control.kappa, control.a)[0]
        )/(2.0*h)
        if not math.isclose(c_native, c_reference, rel_tol=0.0, abs_tol=2e-14):
            raise RuntimeError("Self-test failed: bounded shear oracle.")
        if not math.isclose(dc_native, dc_reference, rel_tol=0.0, abs_tol=2e-14):
            raise RuntimeError("Self-test failed: bounded shear derivative oracle.")
        if not math.isclose(dc_native, derivative_fd, rel_tol=2e-8, abs_tol=2e-9):
            raise RuntimeError("Self-test failed: bounded shear finite difference.")
        if abs(c_native) >= control.kappa*control.a + 1e-14:
            raise RuntimeError("Self-test failed: shear boundedness.")
        A = np.asarray([[1.0, 0.0], [-c_native, 1.0]])
        metric = A.T @ prepared.mass_matrix @ A
        if not math.isclose(np.linalg.det(metric), np.linalg.det(prepared.mass_matrix), rel_tol=2e-12, abs_tol=2e-14):
            raise RuntimeError("Self-test failed: constant metric determinant.")
        if float(np.min(np.linalg.eigvalsh(metric))) <= 0.0:
            raise RuntimeError("Self-test failed: metric positive definiteness.")
    p = np.asarray([0.31, -0.47])
    s = 0.62
    p_s = -0.29
    native_h = float(_hamiltonian(
        q, p, s, p_s, prepared.beta, control.kappa, control.tau,
        control.Q, control.a, parameters, prepared.inverse_mass_matrix,
    ))
    reference_h = _reference_hamiltonian(
        q, p, s, p_s, prepared.beta, control, parameters,
        prepared.inverse_mass_matrix,
    )
    if not math.isclose(native_h, reference_h, rel_tol=2e-14, abs_tol=2e-14):
        raise RuntimeError("Self-test failed: extended Hamiltonian oracle.")
    messages.append("PASS: bounded shear, derivative, constant determinant, SPD metric, Hamiltonian")

    # Compiled proposal versus independent reference.
    micro = FixedConfiguration(0.43, 0.61, 0.17, 5.0, 0.013, 3)
    q0 = np.asarray([0.37, -0.22])
    p0 = np.asarray([-0.28, 0.41])
    s0 = 0.18
    ps0 = -0.33
    native = _proposal(
        q0, p0, s0, ps0, _target_gradient(q0, parameters),
        prepared.beta, micro.kappa, micro.tau, micro.Q, micro.a,
        micro.epsilon, micro.L, parameters, prepared.inverse_mass_matrix,
    )
    reference = _reference_proposal(
        q0, p0, s0, ps0, prepared.beta, micro, parameters,
        prepared.inverse_mass_matrix,
    )
    for native_value, reference_value in zip(native[:5], reference[:5]):
        if not np.allclose(native_value, reference_value, rtol=2e-13, atol=2e-13):
            raise RuntimeError("Self-test failed: compiled/reference split micro-oracle.")
    if native[5:] != reference[5:]:
        raise RuntimeError("Self-test failed: split metadata oracle.")
    messages.append("PASS: compiled/reference exact P/T/K/T/P micro-oracle")

    # Reversibility and full-flip involution.
    state0 = np.asarray([0.21, -0.14, 0.32, -0.27, 0.19, 0.23])
    forward = _flow_state(state0, 0.011, 4, micro, prepared)
    backward = _flow_state(forward, -0.011, 4, micro, prepared)
    if not np.allclose(backward, state0, rtol=0.0, atol=3e-12):
        raise RuntimeError("Self-test failed: split reversibility.")
    flipped = forward.copy()
    flipped[2:4] *= -1.0
    flipped[5] *= -1.0
    second = _flow_state(flipped, 0.011, 4, micro, prepared)
    second[2:4] *= -1.0
    second[5] *= -1.0
    if not np.allclose(second, state0, rtol=0.0, atol=3e-12):
        raise RuntimeError("Self-test failed: full-flip proposal involution.")
    messages.append("PASS: split reversibility and full-state-flip involution")

    # Six-dimensional finite-difference volume oracle.
    jacobian = np.empty((6, 6))
    jac_h = 2e-6
    for column in range(6):
        shift = np.zeros(6)
        shift[column] = jac_h
        jacobian[:, column] = (
            _flow_state(state0+shift, 0.009, 2, micro, prepared)
            - _flow_state(state0-shift, 0.009, 2, micro, prepared)
        )/(2.0*jac_h)
    determinant = float(np.linalg.det(jacobian))
    if not math.isclose(determinant, 1.0, rel_tol=3e-7, abs_tol=3e-7):
        raise RuntimeError(f"Self-test failed: volume determinant={determinant:.17g}.")
    messages.append(f"PASS: six-dimensional volume oracle det={determinant:.12g}")

    # Second-order energy behavior at a fixed total integration time.
    energy_configuration = FixedConfiguration(0.5, 0.5, 0.1, 8.0, 0.04, 2)
    initial_energy = _reference_hamiltonian(
        state0[:2], state0[2:4], state0[4], state0[5], prepared.beta,
        energy_configuration, parameters, prepared.inverse_mass_matrix,
    )
    energy_errors = []
    for step_size, steps in ((0.04, 2), (0.02, 4), (0.01, 8)):
        flowed = _flow_state(state0, step_size, steps, energy_configuration, prepared)
        energy_errors.append(abs(
            _reference_hamiltonian(
                flowed[:2], flowed[2:4], flowed[4], flowed[5], prepared.beta,
                energy_configuration, parameters, prepared.inverse_mass_matrix,
            ) - initial_energy
        ))
    if not (energy_errors[1] < 0.45*energy_errors[0] and energy_errors[2] < 0.45*energy_errors[1]):
        raise RuntimeError(f"Self-test failed: second-order energy sequence {energy_errors}.")
    messages.append("PASS: symmetric-split second-order energy behavior")

    # Exact physical marginal: the conditional integral is independent of x.
    eta_grid = np.linspace(-8.0*control.tau, 8.0*control.tau, 40001)
    conditional_integral = float(np.trapezoid(
        np.exp(-0.5*(eta_grid/control.tau)**2), eta_grid
    ))
    exact_integral = math.sqrt(2.0*math.pi)*control.tau
    if not math.isclose(conditional_integral, exact_integral, rel_tol=2e-15, abs_tol=3e-15):
        raise RuntimeError("Self-test failed: conditional marginal normalization.")
    messages.append("PASS: integrating s leaves the exact Banana physical marginal")

    # Conditional momentum covariance and RNG stream separation/reproducibility.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(86420)))
    standard = rng.standard_normal((120000, 2))
    latent = standard @ prepared.mass_cholesky.T / math.sqrt(prepared.beta)
    c_value, _ = _reference_shear(1.3, control.kappa, control.a)
    momentum = np.column_stack((latent[:, 0]-c_value*latent[:, 1], latent[:, 1]))
    empirical_covariance = np.cov(momentum, rowvar=False, ddof=1)
    A = np.asarray([[1.0, 0.0], [-c_value, 1.0]])
    exact_covariance = A.T @ prepared.mass_matrix @ A / prepared.beta
    if not np.allclose(empirical_covariance, exact_covariance, rtol=0.018, atol=0.006):
        raise RuntimeError("Self-test failed: conditional momentum covariance.")
    streams1 = _make_random_streams(97531, 64, 2)
    streams2 = _make_random_streams(97531, 64, 2)
    if not all(np.array_equal(first, second) for first, second in zip(streams1, streams2)):
        raise RuntimeError("Self-test failed: RNG reproducibility.")
    if np.array_equal(streams1[0][:, 0], streams1[1]):
        raise RuntimeError("Self-test failed: RNG child-stream separation.")
    stream_digest = hashlib.sha256(b"".join(array.tobytes() for array in streams1)).hexdigest()
    messages.append(f"PASS: momentum covariance and independent reproducible RNG; digest={stream_digest}")

    # Diagnostics and undefined-status propagation.
    diagnostic_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(24680)))
    innovations = diagnostic_rng.standard_normal((4, 2400))
    ar = np.empty_like(innovations)
    ar[:, 0] = innovations[:, 0]
    for draw in range(1, ar.shape[1]):
        ar[:, draw] = 0.8*ar[:, draw-1] + innovations[:, draw]
    iat = _multi_chain_geyer_iat(ar)
    rhat = _rank_normalized_split_folded_rhat(ar)
    if not iat.is_defined or not (4.0 < iat.tau < 18.0):
        raise RuntimeError("Self-test failed: Geyer IAT synthetic oracle.")
    if not math.isfinite(rhat.combined) or rhat.combined > 1.05:
        raise RuntimeError("Self-test failed: rank/split/folded R-hat oracle.")
    synthetic = np.stack((ar[:, :400], ar[:, :400]), axis=2)
    undefined = _analyze_block(
        samples=synthetic, epsilon=0.05, L=40, block_index=0,
        seeds=(1,2,3,4), accepted_retained=1600,
        retained_gradient_calls=64000, warmup_nonfinite=0,
        retained_nonfinite=1, observable_names=("x","y","x^2","x*y","y^2"),
        exact_observable_means=np.zeros(5), exact_observable_sds=np.ones(5),
    )
    if undefined.status != "undefined_nonfinite_simulation" or math.isfinite(undefined.block_efficiency):
        raise RuntimeError("Self-test failed: undefined metric propagation.")
    messages.append("PASS: Geyer cutoff, rank/split/folded R-hat, undefined propagation")

    # Tiny complete native MH chain; this is not a tuning-grid evaluation.
    physical, thermostat, uniforms = _make_random_streams(112358, 48, 2)
    smoke = _simulate_chain(
        physical, thermostat, uniforms, prepared.physical_starts[0],
        float(prepared.conditional_standardized_starts[0]), prepared.beta,
        control.kappa, control.tau, control.Q, control.a, 0.01, 3,
        8, 40, parameters, prepared.mass_cholesky,
        prepared.inverse_mass_matrix,
    )
    if not np.all(np.isfinite(smoke[0])) or not np.all(np.isfinite(smoke[1])):
        raise RuntimeError("Self-test failed: finite native smoke chain.")
    if smoke[3] != 0 or smoke[4] != 0 or smoke[6] != 3*40:
        raise RuntimeError("Self-test failed: smoke nonfinite/gradient accounting.")
    if NUMBA_AVAILABLE and not getattr(_simulate_chain, "signatures", ()):
        raise RuntimeError("Self-test failed: native sampler was not compiled.")
    messages.append("PASS: tiny full Metropolis chain and exact target-gradient accounting")
    return tuple(messages)


# ---------------------------------------------------------------------------
# Plain-text production report.
# ---------------------------------------------------------------------------


def _number(value: float, specification: str = ".12g") -> str:
    if math.isnan(value):
        return "undefined"
    if math.isinf(value):
        return "+infinity" if value > 0.0 else "-infinity"
    return format(value, specification)


def _configuration_text(configuration: FixedConfiguration) -> str:
    return (
        f"kappa={configuration.kappa:.12g}, tau={configuration.tau:.12g}, "
        f"Q={configuration.Q:.12g}, a={configuration.a:.12g}, "
        f"epsilon={configuration.epsilon:.12g}, L={configuration.L}"
    )


def _result_summary(prefix: str, result: ConfigurationResult) -> str:
    return (
        f"{prefix}: {_configuration_text(result.configuration)}; "
        f"origin={result.origin}; status={result.status}; "
        f"GM_efficiency={_number(result.geometric_mean_efficiency)}; "
        f"min_efficiency={_number(result.minimum_block_efficiency)}; "
        f"median_efficiency={_number(result.median_block_efficiency)}; "
        f"max_efficiency={_number(result.maximum_block_efficiency)}; "
        f"mean_acceptance={_number(result.mean_acceptance)}; "
        f"max_Rhat={_number(result.maximum_rhat)}; "
        f"max_moment_error={_number(result.maximum_standardized_moment_error)}; "
        f"nonfinite={result.total_nonfinite_trajectories}"
    )


def _append_result_rows(
    lines: list[str],
    heading: str,
    results: Sequence[ConfigurationResult],
) -> None:
    lines.extend(("", heading, "-" * len(heading)))
    if not results:
        lines.append("none")
        return
    for index, result in enumerate(results, start=1):
        lines.append(_result_summary(f"row {index}", result))


def _append_five_block_details(
    lines: list[str],
    heading: str,
    results: Sequence[ConfigurationResult],
) -> None:
    """Print enough data to reconstruct every five-block finalist score."""

    lines.extend(("", heading, "-" * len(heading)))
    if not results:
        lines.append("none")
        return
    for result_index, result in enumerate(results, start=1):
        lines.append(_result_summary(f"finalist {result_index}", result))
        for item in result.block_results:
            block = item.physical
            conditional = item.conditional
            lines.append(
                f"  block={block.block_index+1}; seeds={block.seeds}; "
                f"status={block.status}; efficiency={_number(block.block_efficiency)}; "
                f"worst_observable={block.worst_observable}; "
                f"worst_IAT={_number(block.worst_iat)}; "
                f"retained_gradient_calls={block.retained_gradient_calls}; "
                f"gradients_per_retained_draw={_number(block.gradients_per_retained_draw)}; "
                f"acceptance={_number(block.acceptance)}; "
                f"max_Rhat={_number(block.maximum_rhat)} "
                f"({block.maximum_rhat_observable}); "
                f"max_moment_error={_number(block.maximum_standardized_moment_error)} "
                f"({block.maximum_moment_error_observable}); "
                f"nonfinite_warmup={block.warmup_nonfinite_trajectories}; "
                f"nonfinite_retained={block.retained_nonfinite_trajectories}; "
                f"eta_mean={_number(conditional.empirical_mean)}; "
                f"eta_second={_number(conditional.empirical_second_moment)}; "
                f"eta_second_exact={_number(conditional.exact_second_moment)}"
            )
            for diagnostic in block.observable_diagnostics:
                lines.append(
                    f"    observable={diagnostic.name}; IAT={_number(diagnostic.iat.tau)}; "
                    f"IAT_status={diagnostic.iat.status}; pairs={diagnostic.iat.pair_count}; "
                    f"last_included_lag={diagnostic.iat.last_included_lag}; "
                    f"first_excluded_lag={diagnostic.iat.first_excluded_lag}; "
                    f"cutoff={diagnostic.iat.cutoff_reason}; "
                    f"efficiency={_number(diagnostic.efficiency)}; "
                    f"Rhat_max={_number(diagnostic.rhat.combined)}; "
                    f"empirical_mean={_number(diagnostic.empirical_mean)}; "
                    f"exact_mean={_number(diagnostic.exact_mean)}; "
                    f"standardized_moment_error={_number(diagnostic.standardized_moment_error)}"
                )


def _build_report(
    *,
    prepared: PreparedInputs,
    self_test_messages: Sequence[str],
    stream_second_moments: Sequence[float],
    stage1_results: Sequence[ConfigurationResult],
    promoted_methods: Sequence[MethodTuple],
    stage2_results: Sequence[ConfigurationResult],
    stage2_promoted: Sequence[ConfigurationResult],
    base_full_results: Sequence[ConfigurationResult],
    base_winner: ConfigurationResult,
    boundary_labels: Sequence[str],
    boundary_methods: Sequence[MethodTuple],
    boundary_stage2_results: Sequence[ConfigurationResult],
    boundary_stage2_promoted: Sequence[ConfigurationResult],
    boundary_full_results: Sequence[ConfigurationResult],
    full_results: Sequence[ConfigurationResult],
    representative: ConfigurationResult,
) -> str:
    lines: list[str] = [
        "THC-v4 CONDITIONAL-SHEAR BANANA PRODUCTION TUNING REPORT",
        "========================================================",
        f"Protocol: {PROTOCOL_VERSION}",
        f"Implementation: {IMPLEMENTATION_VERSION}",
        f"Diagnostics: {DIAGNOSTICS_VERSION}",
        f"RNG: {RNG_VERSION}",
        f"Run mode: {prepared.run_mode}",
        f"Run label: {prepared.run_label}",
        f"Input fingerprint SHA-256: {prepared.input_fingerprint}",
        f"Stable source/protocol fingerprint SHA-256: {_implementation_fingerprint()}",
        f"Whole-source SHA-256: {_whole_source_sha256()}",
        f"Shared HMC common-input SHA-256: {HMC_SHARED_COMMON_INPUT_SHA256}",
        f"Runtime: {_runtime_environment()}",
        "Final-comparison random streams: not present and not constructed",
        "",
        "MODEL",
        "-----",
        "Target: r=y-b(x^2-sigma_x^2), U=x^2/(2 sigma_x^2)+r^2/(2 sigma_y^2)",
        f"beta={prepared.beta:.12g}; sigma_x={prepared.sigma_x:.12g}; "
        f"sigma_y={prepared.sigma_y:.12g}; b={prepared.banana_b:.12g}",
        "Conditional auxiliary: s|x ~ Normal(x,tau^2)",
        "Bounded shear: c(s)=kappa*a*tanh(s/a), c'(s)=kappa*sech^2(s/a)",
        "A(s)=[[1,0],[-c(s),1]], G(s)=A(s)^T M0 A(s), det G(s)=det M0",
        "Hamiltonian: U(q)+(s-x)^2/(2 beta tau^2)+p^T G(s)^(-1)p/2+p_s^2/(2Q)",
        "Integrator: exact symmetric P/T/K/T/P; full momentum flip; full-state MH",
        "Momentum refresh: p|s~N(0,G(s)/beta), p_s~N(0,Q/beta)",
        f"M0={prepared.mass_matrix.tolist()}",
        f"standardized physical starts={prepared.standardized_starts.tolist()}",
        f"standardized eta=s-x starts={prepared.conditional_standardized_starts.tolist()}",
        "",
        "FAIR SCORE",
        "----------",
        "Physical observables: " + ", ".join(prepared.observable_names),
        "IAT: multi-chain Geyer initial-positive plus initial-monotone sequence",
        "R-hat: maximum of rank-normalized split and folded rank-normalized split R-hat",
        "Cost: actual physical target-gradient calls per retained draw",
        "Observable efficiency: 1/(gradients per retained draw * IAT)",
        "Block score: minimum efficiency over all physical observables",
        "Stage-1 score: GM over its declared block subset only (ranking screen)",
        "Stage-2 score: GM over its declared block subset only (ranking screen)",
        "Stage-3/final tuning score: GM over all five tuning blocks",
        "No R-hat, moment, acceptance, or safety threshold enters selection",
        "Any nonfinite trajectory or undefined required IAT makes that block score undefined",
        "",
        "TUNING DESIGN",
        "-------------",
        f"epsilon grid={list(prepared.epsilon_values)}",
        f"L grid={list(prepared.L_values)}",
        f"kappa grid={list(prepared.kappa_values)}",
        f"tau grid={list(prepared.tau_values)}",
        f"Q grid={list(prepared.Q_values)}",
        f"a grid={list(prepared.a_values)}",
        f"seed blocks={prepared.seed_blocks}",
        f"stage 1 anchors={prepared.stage1_anchor_pairs}; blocks={tuple(i+1 for i in prepared.stage1_block_indices)}; "
        f"length={prepared.stage1_n_warmup}+{prepared.stage1_n_retained}; keep methods={prepared.stage1_keep_method_tuples}",
        f"stage 2 common grid; blocks={tuple(i+1 for i in prepared.stage2_block_indices)}; "
        f"length={prepared.stage2_n_warmup}+{prepared.stage2_n_retained}; keep fixed configurations={prepared.stage2_keep_fixed_configurations}",
        f"stage 3 all five blocks; length={prepared.n_warmup}+{prepared.n_retained}",
        "automatic boundary rule: after base five-block selection, if its method "
        "lies on a base-grid boundary, move that coordinate alone to its one "
        "predeclared outward value; no joint corners and no second layer",
        f"automatic boundary expansion={prepared.automatic_boundary_expansion}",
        "boundary values=" + str({
            "kappa": [prepared.boundary_kappa_lower, prepared.boundary_kappa_upper],
            "tau": [prepared.boundary_tau_lower, prepared.boundary_tau_upper],
            "Q": [prepared.boundary_Q_lower, prepared.boundary_Q_upper],
            "a": [prepared.boundary_a_lower, prepared.boundary_a_upper],
        }),
        f"boundary keep fixed configurations={prepared.boundary_keep_fixed_configurations}",
        "forced full configurations=" + str([
            [c.kappa,c.tau,c.Q,c.a,c.epsilon,c.L]
            for c in prepared.forced_full_configurations
        ]),
        f"base method tuples={len(prepared.kappa_values)*len(prepared.tau_values)*len(prepared.Q_values)*len(prepared.a_values)}",
        f"stage-1 evaluations={len(stage1_results)}; stage-2 evaluations={len(stage2_results)}; "
        f"base full evaluations={len(base_full_results)}; boundary stage-2 evaluations="
        f"{len(boundary_stage2_results)}; boundary full evaluations={len(boundary_full_results)}; "
        f"total full evaluations={len(full_results)}",
        "",
        "PREFLIGHT",
        "---------",
    ]
    lines.extend(self_test_messages)
    lines.extend((
        "",
        "RANDOM-STREAM MOMENT CHECK (DESCRIPTIVE ONLY)",
        "-----------------------------------------------",
    ))
    for block_index, value in enumerate(stream_second_moments, start=1):
        lines.append(f"block {block_index}: pooled physical standard-normal second moment={_number(value)}")

    _append_result_rows(lines, "STAGE 1 ALL ANCHOR RESULTS", stage1_results)
    lines.extend(("", "STAGE 1 PROMOTED METHOD TUPLES", "------------------------------"))
    for index, method in enumerate(promoted_methods, start=1):
        lines.append(
            f"{index}: kappa={method.kappa:.12g}, tau={method.tau:.12g}, "
            f"Q={method.Q:.12g}, a={method.a:.12g}"
        )
    _append_result_rows(lines, "STAGE 2 ALL FIXED-CONFIGURATION RESULTS", stage2_results)
    _append_result_rows(lines, "STAGE 2 PROMOTED FIXED CONFIGURATIONS", stage2_promoted)
    _append_five_block_details(lines, "BASE FULL FIVE-BLOCK RESULTS", base_full_results)
    lines.extend((
        "",
        "BASE FIVE-BLOCK WINNER BEFORE BOUNDARY EXPANSION",
        "------------------------------------------------",
        _result_summary("base winner", base_winner),
        f"boundary triggered={bool(boundary_methods)}; labels={tuple(boundary_labels)}; "
        f"method count={len(boundary_methods)}",
    ))
    for index, method in enumerate(boundary_methods, start=1):
        lines.append(
            f"boundary method {index}: kappa={method.kappa:.12g}, "
            f"tau={method.tau:.12g}, Q={method.Q:.12g}, a={method.a:.12g}"
        )
    _append_result_rows(lines, "BOUNDARY STAGE-2 RESULTS", boundary_stage2_results)
    _append_result_rows(lines, "BOUNDARY STAGE-2 PROMOTED", boundary_stage2_promoted)
    _append_five_block_details(lines, "BOUNDARY FULL FIVE-BLOCK RESULTS", boundary_full_results)

    lines.extend((
        "",
        "SELECTED FIXED REPRESENTATIVE",
        "-----------------------------",
        _result_summary("selected", representative),
        "Selection used the five-block GM only; the following details are descriptive.",
    ))
    for item in representative.block_results:
        block = item.physical
        conditional = item.conditional
        lines.append(
            f"block {block.block_index+1}; seeds={block.seeds}; status={block.status}; "
            f"acceptance={_number(block.acceptance)}; "
            f"retained_gradient_calls={block.retained_gradient_calls}; "
            f"gradients/draw={_number(block.gradients_per_retained_draw)}; "
            f"worst_observable={block.worst_observable}; worst_IAT={_number(block.worst_iat)}; "
            f"block_efficiency={_number(block.block_efficiency)}; "
            f"max_Rhat={_number(block.maximum_rhat)} ({block.maximum_rhat_observable}); "
            f"max_moment_error={_number(block.maximum_standardized_moment_error)} "
            f"({block.maximum_moment_error_observable}); nonfinite_warmup={block.warmup_nonfinite_trajectories}; "
            f"nonfinite_retained={block.retained_nonfinite_trajectories}; "
            f"eta_mean={_number(conditional.empirical_mean)} exact=0; "
            f"eta_second={_number(conditional.empirical_second_moment)} "
            f"exact={_number(conditional.exact_second_moment)}"
        )
        for diagnostic in block.observable_diagnostics:
            lines.append(
                f"  observable={diagnostic.name}; IAT={_number(diagnostic.iat.tau)}; "
                f"IAT_status={diagnostic.iat.status}; pairs={diagnostic.iat.pair_count}; "
                f"last_included_lag={diagnostic.iat.last_included_lag}; "
                f"first_excluded_lag={diagnostic.iat.first_excluded_lag}; "
                f"cutoff={diagnostic.iat.cutoff_reason}; ESS={_number(diagnostic.ess)}; "
                f"efficiency={_number(diagnostic.efficiency)}; "
                f"Rhat_rank={_number(diagnostic.rhat.rank)}; "
                f"Rhat_folded={_number(diagnostic.rhat.folded)}; "
                f"Rhat_max={_number(diagnostic.rhat.combined)}; "
                f"empirical_mean={_number(diagnostic.empirical_mean)}; "
                f"exact_mean={_number(diagnostic.exact_mean)}; "
                f"standardized_moment_error={_number(diagnostic.standardized_moment_error)}"
            )
    lines.extend((
        "",
        "END OF REPORT",
    ))
    return "\n".join(lines)


def _stream_second_moment(
    cache: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_warmup: int,
) -> float:
    total = 0.0
    count = 0
    for physical, _thermostat, _uniforms in cache:
        retained = physical[n_warmup:]
        total += float(np.sum(retained*retained))
        count += retained.size
    return total/count if count else math.nan


# ---------------------------------------------------------------------------
# Public preflight/production entry point.
# ---------------------------------------------------------------------------


def run_conditional_shear_tuning(
    inputs: THCRunInputs,
    *,
    preflight_only: bool = False,
) -> dict[str, object]:
    """Run preflight or the frozen three-stage tuning campaign."""

    if not isinstance(preflight_only, bool):
        raise TypeError("preflight_only must be exactly True or False.")
    if preflight_only and inputs.run_self_tests is not True:
        raise ValueError("preflight_only=True requires run_self_tests=True.")
    prepared = _prepare_inputs(inputs)
    self_test_messages = (
        run_self_tests(prepared)
        if prepared.run_self_tests
        else ("SKIPPED: run_self_tests=False (CUSTOM mode only).",)
    )
    if preflight_only:
        native_status = "cache-free Numba native" if NUMBA_AVAILABLE else "Python fallback"
        print(
            f"PASS: {PREFLIGHT_VERSION}; {native_status}; {_runtime_environment()}; "
            f"input SHA-256={prepared.input_fingerprint}; implementation SHA-256="
            f"{_implementation_fingerprint()}",
            flush=True,
        )
        print(
            "GREX PREFLIGHT COMPLETE: target/gradient, bounded shear, metric "
            "determinant and positivity, Hamiltonian, compiled/reference split, "
            "reversibility, full-flip involution, volume, second-order energy, "
            "exact physical marginal, momentum refresh, RNG, diagnostics, "
            "gradient accounting, and a tiny finite MH chain all passed. No "
            "tuning candidate was evaluated, no report was written, and no "
            "final-comparison random stream exists in this program. Set "
            "GREX_PREFLIGHT_ONLY=False and execute the entire cell again for "
            "production tuning.",
            flush=True,
        )
        return {
            "preflight_only": True,
            "prepared_inputs": prepared,
            "self_test_messages": self_test_messages,
            "input_fingerprint": prepared.input_fingerprint,
            "implementation_fingerprint": _implementation_fingerprint(),
            "whole_source_sha256": _whole_source_sha256(),
        }

    destination = _resolve_output_destination(prepared)
    total_iterations = prepared.n_warmup + prepared.n_retained
    stream_caches: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
        [
            _make_random_streams(seed, total_iterations, prepared.dimension)
            for seed in block
        ]
        for block in prepared.seed_blocks
    ]
    stream_second_moments = tuple(
        _stream_second_moment(cache, prepared.n_warmup)
        for cache in stream_caches
    )

    # Compile the complete sampler with a private preflight seed before stage 1.
    compile_streams = _make_random_streams(314159, 2, prepared.dimension)
    first_method = MethodTuple(
        prepared.kappa_values[0], prepared.tau_values[0],
        prepared.Q_values[0], prepared.a_values[0],
    )
    _simulate_chain(
        compile_streams[0], compile_streams[1], compile_streams[2],
        prepared.physical_starts[0],
        float(prepared.conditional_standardized_starts[0]),
        prepared.beta, first_method.kappa, first_method.tau,
        first_method.Q, first_method.a, prepared.epsilon_values[0], 1,
        1, 1, prepared.target_parameters, prepared.mass_cholesky,
        prepared.inverse_mass_matrix,
    )

    method_tuples = tuple(
        MethodTuple(kappa, tau, Q, a)
        for kappa in prepared.kappa_values
        for tau in prepared.tau_values
        for Q in prepared.Q_values
        for a in prepared.a_values
    )
    stage1_results: list[ConfigurationResult] = []
    best_by_method: list[tuple[MethodTuple, ConfigurationResult]] = []
    for method_index, method in enumerate(method_tuples, start=1):
        if prepared.print_progress:
            print(
                f"[stage 1 method {method_index:03d}/{len(method_tuples)}] "
                f"kappa={method.kappa:g}, tau={method.tau:g}, Q={method.Q:g}, a={method.a:g}",
                flush=True,
            )
        method_results: list[ConfigurationResult] = []
        for epsilon, L in prepared.stage1_anchor_pairs:
            result = _evaluate_fixed_configuration(
                FixedConfiguration(method.kappa, method.tau, method.Q, method.a, epsilon, L),
                block_indices=prepared.stage1_block_indices,
                n_warmup=prepared.stage1_n_warmup,
                n_retained=prepared.stage1_n_retained,
                origin="BASE_STAGE1_ANCHOR",
                stream_caches=stream_caches,
                prepared=prepared,
            )
            stage1_results.append(result)
            method_results.append(result)
        ranked_method = _rank_defined(method_results, len(method_results))
        if ranked_method:
            best_by_method.append((method, ranked_method[0]))
    best_by_method.sort(key=lambda item: _rank_key(item[1]), reverse=True)
    promoted_records = best_by_method[: prepared.stage1_keep_method_tuples]
    if not promoted_records:
        raise RuntimeError("Stage 1 produced no method tuple with a defined score.")
    promoted_methods = tuple(record[0] for record in promoted_records)

    stage2_results: list[ConfigurationResult] = []
    stage2_total = len(promoted_methods)*len(prepared.epsilon_values)*len(prepared.L_values)
    stage2_index = 0
    for method in promoted_methods:
        for epsilon in prepared.epsilon_values:
            for L in prepared.L_values:
                stage2_index += 1
                if prepared.print_progress and (stage2_index == 1 or stage2_index % 12 == 0):
                    print(f"[stage 2 {stage2_index:03d}/{stage2_total}]", flush=True)
                stage2_results.append(_evaluate_fixed_configuration(
                    FixedConfiguration(method.kappa, method.tau, method.Q, method.a, epsilon, L),
                    block_indices=prepared.stage2_block_indices,
                    n_warmup=prepared.stage2_n_warmup,
                    n_retained=prepared.stage2_n_retained,
                    origin="BASE_STAGE2_FULL_COMMON_GRID",
                    stream_caches=stream_caches,
                    prepared=prepared,
                ))
    stage2_promoted = _rank_defined(
        stage2_results, prepared.stage2_keep_fixed_configurations
    )
    if not stage2_promoted:
        raise RuntimeError("Stage 2 produced no fixed configuration with a defined score.")

    base_full_results: list[ConfigurationResult] = []
    full_configurations = [result.configuration for result in stage2_promoted]
    promoted_configurations = tuple(full_configurations)
    for forced in prepared.forced_full_configurations:
        if forced not in full_configurations:
            full_configurations.append(forced)
    for index, configuration in enumerate(full_configurations, start=1):
        forced = configuration in prepared.forced_full_configurations
        origin = (
            "FORCED_GEOMETRIC_CONTROL_FULL_FIVE_BLOCKS"
            if forced and configuration not in promoted_configurations
            else "BASE_STAGE3_FINALIST"
        )
        if prepared.print_progress:
            print(
                f"[stage 3 {index:02d}/{len(full_configurations)}] "
                + _configuration_text(configuration),
                flush=True,
            )
        base_full_results.append(_evaluate_fixed_configuration(
            configuration,
            block_indices=tuple(range(len(prepared.seed_blocks))),
            n_warmup=prepared.n_warmup,
            n_retained=prepared.n_retained,
            origin=origin,
            stream_caches=stream_caches,
            prepared=prepared,
        ))
    base_winner = _select_representative(base_full_results)

    boundary_methods: tuple[MethodTuple, ...] = ()
    boundary_labels: tuple[str, ...] = ()
    if prepared.automatic_boundary_expansion:
        boundary_methods, boundary_labels = _boundary_method_layer(
            base_winner, prepared
        )
    boundary_stage2_results: list[ConfigurationResult] = []
    boundary_total = (
        len(boundary_methods)
        * len(prepared.epsilon_values)
        * len(prepared.L_values)
    )
    boundary_index = 0
    boundary_origin = (
        "BOUNDARY_LAYER_ONE_COORDINATE[" + ",".join(boundary_labels) + "]"
        if boundary_labels
        else "BOUNDARY_LAYER_NOT_TRIGGERED"
    )
    for method in boundary_methods:
        for epsilon in prepared.epsilon_values:
            for L in prepared.L_values:
                boundary_index += 1
                if prepared.print_progress and (
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
                            method.kappa, method.tau, method.Q, method.a,
                            epsilon, L,
                        ),
                        block_indices=prepared.stage2_block_indices,
                        n_warmup=prepared.stage2_n_warmup,
                        n_retained=prepared.stage2_n_retained,
                        origin=boundary_origin + ":STAGE2",
                        stream_caches=stream_caches,
                        prepared=prepared,
                    )
                )
    boundary_stage2_promoted = _rank_defined(
        boundary_stage2_results,
        prepared.boundary_keep_fixed_configurations,
    )
    boundary_full_results: list[ConfigurationResult] = []
    for index, stage2_result in enumerate(boundary_stage2_promoted, start=1):
        if prepared.print_progress:
            print(
                f"[boundary stage 3 {index:02d}/"
                f"{len(boundary_stage2_promoted)}]",
                flush=True,
            )
        boundary_full_results.append(
            _evaluate_fixed_configuration(
                stage2_result.configuration,
                block_indices=tuple(range(len(prepared.seed_blocks))),
                n_warmup=prepared.n_warmup,
                n_retained=prepared.n_retained,
                origin=boundary_origin + ":FULL_FIVE_BLOCKS",
                stream_caches=stream_caches,
                prepared=prepared,
            )
        )

    full_results = [*base_full_results, *boundary_full_results]
    representative = _select_representative(full_results)
    report = _build_report(
        prepared=prepared,
        self_test_messages=self_test_messages,
        stream_second_moments=stream_second_moments,
        stage1_results=stage1_results,
        promoted_methods=promoted_methods,
        stage2_results=stage2_results,
        stage2_promoted=stage2_promoted,
        base_full_results=base_full_results,
        base_winner=base_winner,
        boundary_labels=boundary_labels,
        boundary_methods=boundary_methods,
        boundary_stage2_results=boundary_stage2_results,
        boundary_stage2_promoted=boundary_stage2_promoted,
        boundary_full_results=boundary_full_results,
        full_results=full_results,
        representative=representative,
    )
    destination.write_text(report + "\n", encoding="utf-8")
    if prepared.print_report:
        print("\n" + report)
        print(f"\nThe identical report was saved to:\n{destination}")
    return {
        "preflight_only": False,
        "stage1_results": tuple(stage1_results),
        "stage1_promoted_methods": promoted_methods,
        "stage2_results": tuple(stage2_results),
        "stage2_promoted": stage2_promoted,
        "base_full_results": tuple(base_full_results),
        "base_winner": base_winner,
        "boundary_trigger_labels": boundary_labels,
        "boundary_triggered": bool(boundary_methods),
        "boundary_methods": boundary_methods,
        "boundary_stage2_results": tuple(boundary_stage2_results),
        "boundary_stage2_promoted": boundary_stage2_promoted,
        "boundary_full_results": tuple(boundary_full_results),
        "full_results": tuple(full_results),
        "representative": representative,
        "report": report,
        "output_path": destination,
        "prepared_inputs": prepared,
        "input_fingerprint": prepared.input_fingerprint,
        "implementation_fingerprint": _implementation_fingerprint(),
        "whole_source_sha256": _whole_source_sha256(),
    }


# ===========================================================================
#                         ALL USER INPUTS ARE HERE
# ===========================================================================
# Leave GREX_PREFLIGHT_ONLY=True for the first complete execution.  It performs
# only the cache-free preflight and writes nothing.  After PASS, change only it
# to False and execute the entire cell again for production tuning.

GREX_PREFLIGHT_ONLY = True

USER_INPUTS = THCRunInputs(
    # ----- Banana target and shared HMC physical geometry -----
    beta=1.0,
    sigma_x=2.0,
    sigma_y=0.2,
    banana_b=0.25,
    mass_matrix=np.asarray([
        [0.25, 0.0],
        [0.0, 0.49019607843137253],
    ], dtype=np.float64),

    # Physical starts are standardized (z_x,z_r); q is derived exactly.
    standardized_starts=np.asarray([
        [-2.0, -2.0],
        [-2.0, 2.0],
        [2.0, -2.0],
        [2.0, 2.0],
    ], dtype=np.float64),
    # eta=s-x begins at tau times these standardized values.
    conditional_standardized_starts=(-1.0, 1.0, 1.0, -1.0),
    observable_policy="ALL_LINEAR_AND_UNIQUE_QUADRATIC",

    # ----- Common HMC/THC integrator grid -----
    epsilon_values=(0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080),
    L_values=(10, 20, 40, 80, 120, 160),

    # ----- The same five tuning seed blocks used by HMC and THC-v2 -----
    seed_blocks=(
        (123, 456, 789, 101112),
        (3545294353, 3961674632, 3627514536, 3198582250),
        (788645248, 3651519711, 1811141324, 1814869264),
        (3192285063, 2251015777, 396444798, 765150135),
        (1562880678, 1785952873, 2908883741, 2216662531),
    ),

    # ----- Bounded conditional-shear method grid -----
    # c(s)=kappa*a*tanh(s/a).  a=8 is four sigma_x.
    kappa_values=(0.25, 0.40, 0.50, 0.65),
    tau_values=(0.25, 0.50, 1.0),
    Q_values=(0.05, 0.10, 0.25, 0.50),
    a_values=(2.0, 4.0, 8.0, 16.0),

    # ----- Predeclared staged successive halving -----
    stage1_anchor_pairs=((0.04, 80), (0.05, 40), (0.06, 20)),
    stage1_block_indices=(1,),       # one-based tuning block numbers
    stage1_n_warmup=250,
    stage1_n_retained=1000,
    stage1_keep_method_tuples=12,
    stage2_block_indices=(2, 3),
    stage2_n_warmup=500,
    stage2_n_retained=2000,
    stage2_keep_fixed_configurations=8,

    # One automatic outward layer, only when the base five-block winner lies
    # on a method-grid boundary.  Each triggered coordinate is moved alone;
    # no joint corners and no second boundary layer are allowed.
    automatic_boundary_expansion=True,
    boundary_kappa_lower=0.125,
    boundary_kappa_upper=0.80,
    boundary_tau_lower=0.125,
    boundary_tau_upper=2.0,
    boundary_Q_lower=0.025,
    boundary_Q_upper=1.0,
    boundary_a_lower=1.0,
    boundary_a_upper=32.0,
    boundary_keep_fixed_configurations=4,

    # Always evaluate this derived control at full length on all five blocks.
    # Row order: (kappa,tau,Q,a,epsilon,L).
    forced_full_configurations=((0.50, 0.50, 0.10, 8.0, 0.05, 40),),

    # ----- Full five-block lengths and execution -----
    n_warmup=1000,
    n_retained=4000,
    n_workers=4,
    run_self_tests=True,
    print_progress=True,
    print_report=True,

    # ----- Run identity and TXT output -----
    run_mode="CANONICAL",
    run_label="PRODUCTION TUNING",
    output_directory=None,  # script folder; pasted Jupyter cell -> kernel cwd
    output_filename="THC_v4_conditional_shear_banana_tuning_results_GREX_PORTABLE_V1.txt",
    allow_overwrite=True,
)



# Expanded campaign inputs replace the archived defaults above.
# CUSTOM distinguishes this new sealed experiment from the old five-block one.
from expanded_protocol import expanded_inputs
USER_INPUTS = expanded_inputs(USER_INPUTS, 'v4')

if __name__ == "__main__":
    run_conditional_shear_tuning(USER_INPUTS, preflight_only=True)
    print("Production uses launcher/checkpoint_runner.py; see README.txt.")

