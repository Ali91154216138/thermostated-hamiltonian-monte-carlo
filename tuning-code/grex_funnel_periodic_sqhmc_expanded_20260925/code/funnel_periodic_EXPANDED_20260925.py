"""Periodic THC on the 6D centered Neal funnel, for the enlarged S-QHMC comparison.

h(s)=1+alpha*cos(s/a-phi), rho(s)=N(0,1), s persists.
The d*log(h)/beta correction leaves the beta=1 physical target unchanged.
Kernel source is adapted from the existing scalar funnel P/S/K/S/P implementation:
only coupling evaluation and explicit alpha/phi arguments differ.
Common target, diagnostics, block scoring and random streams are imported unchanged.
Run through launcher/checkpoint_runner.py for production checkpointing.
"""
from __future__ import annotations
from dataclasses import dataclass, fields, replace
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence
import contextlib
import io
import math
import numpy as np
import shared_scalar_base as base
import expanded_protocol as protocol

njit=base.njit
NUMBA_AVAILABLE=base.NUMBA_AVAILABLE
THCBlockResult=base.THCBlockResult
_make_random_streams=base._make_random_streams
_analyze_block=base._analyze_block
_aggregate_blocks=base._aggregate_blocks
_thermostat_diagnostic=base._thermostat_diagnostic
_validate_common_parity_seals=base._validate_common_parity_seals
_diagnostics_fingerprint=base._diagnostics_fingerprint
_common_pipeline_fingerprint=base._common_pipeline_fingerprint
_input_preparation_fingerprint=base._input_preparation_fingerprint
_potential_energy_periodic_funnel_v1=base._potential_energy
_target_gradient_periodic_funnel_v1=base._target_gradient
_kinetic_energy_periodic_funnel_v1=base._kinetic_energy
_vector_is_finite_periodic_funnel_v1=base._vector_is_finite
_target_gradient=base._target_gradient
_potential_energy=base._potential_energy

@dataclass(frozen=True)
class PeriodicInputs(base.RunInputs):
    alpha_values: Sequence[float]
    phi_values: Sequence[float]
    a_values: Sequence[float]
    Q_values: Sequence[float]
    thermostat_sigma: float
    thermostat_starts: Sequence[float]

@dataclass(frozen=True)
class PreparedTHCInputs:
    common: base.PreparedInputs
    alpha_values: tuple[float,...]
    phi_values: tuple[float,...]
    a_values: tuple[float,...]
    Q_values: tuple[float,...]
    thermostat_sigma: float
    thermostat_starts: np.ndarray

def _prepare_thc_inputs(inputs):
    common=base._prepare_inputs(inputs)
    base._validate_run_mode(common)
    _validate_common_parity_seals(common)
    alpha=base._normalize_method_grid('alpha_values',inputs.alpha_values,allow_zero=True)
    if any(x>=1 for x in alpha): raise ValueError('Require 0<=alpha<1')
    phi=base._normalize_method_grid('phi_values',inputs.phi_values,allow_zero=True)
    if any(x>=2*math.pi for x in phi): raise ValueError('Require 0<=phi<2*pi')
    a=base._normalize_method_grid('a_values',inputs.a_values,allow_zero=False)
    Q=base._normalize_method_grid('Q_values',inputs.Q_values,allow_zero=False)
    sigma=float(inputs.thermostat_sigma)
    starts=np.asarray(inputs.thermostat_starts,dtype=np.float64)
    if not math.isfinite(sigma) or sigma<=0: raise ValueError('Positive finite thermostat sigma required')
    if starts.shape!=(common.n_chains,) or not np.all(np.isfinite(starts)):
        raise ValueError('One finite s start per chain required')
    return PreparedTHCInputs(common,alpha,phi,a,Q,sigma,starts.copy())

@dataclass(frozen=True)
class THCConfigurationResult:
    """Full-block aggregate for one fixed joint THC configuration."""

    alpha: float
    phi: float
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



@njit(cache=False, inline="always")
def _log_h_and_derivative_periodic_funnel_v1(
    s: float,
    alpha: float, phi: float,
    a: float,
) -> tuple[float, float]:
    """Positive periodic coupling; beta and the physical target are unchanged."""
    if not math.isfinite(s):
        return math.nan, math.nan
    angle = s/a - phi
    h = 1.0 + alpha*math.cos(angle)
    return math.log(h), -(alpha/a)*math.sin(angle)/h


@njit(cache=False)
def _thc_hamiltonian_periodic_funnel_v1(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    beta: float,
    alpha: float, phi: float,
    a: float,
    Q: float,
    thermostat_sigma: float,
    target_parameters: np.ndarray,
    inverse_mass_matrix: np.ndarray,
    mass_is_identity: bool,
) -> float:
    """Evaluate the extended THC Hamiltonian up to one additive constant."""

    log_h, _log_h_derivative = _log_h_and_derivative_periodic_funnel_v1(
        s, alpha, phi, a
    )
    potential = _potential_energy_periodic_funnel_v1(q, target_parameters)
    base_kinetic = _kinetic_energy_periodic_funnel_v1(
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
def _thermostat_rotation_periodic_funnel_v1(
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
def _thc_state_is_finite_periodic_funnel_v1(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
) -> bool:
    return (
        math.isfinite(s)
        and math.isfinite(p_s)
        and _vector_is_finite_periodic_funnel_v1(q)
        and _vector_is_finite_periodic_funnel_v1(p)
    )


@njit(cache=False, nogil=True)
def _thc_proposal_periodic_funnel_v1(
    q: np.ndarray,
    p: np.ndarray,
    s: float,
    p_s: float,
    current_gradient: np.ndarray,
    beta: float,
    alpha: float, phi: float,
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
    _log_h, log_h_derivative = _log_h_and_derivative_periodic_funnel_v1(
        proposed_s, alpha, phi, a
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
    if not _thc_state_is_finite_periodic_funnel_v1(
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
        proposed_s, proposed_p_s = _thermostat_rotation_periodic_funnel_v1(
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
        log_h, log_h_derivative = _log_h_and_derivative_periodic_funnel_v1(
            proposed_s, alpha, phi, a
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
        if not _thc_state_is_finite_periodic_funnel_v1(
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
        proposed_s, proposed_p_s = _thermostat_rotation_periodic_funnel_v1(
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
        proposed_gradient = _target_gradient_periodic_funnel_v1(
            proposed_q, target_parameters
        )
        gradient_calls += 1
        if not _vector_is_finite_periodic_funnel_v1(proposed_gradient):
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
        _log_h, log_h_derivative = _log_h_and_derivative_periodic_funnel_v1(
            proposed_s, alpha, phi, a
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
        if not _thc_state_is_finite_periodic_funnel_v1(
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
def _simulate_thc_chain_periodic_funnel_v1(
    standard_momenta: np.ndarray,
    thermostat_standard_momenta: np.ndarray,
    acceptance_uniforms: np.ndarray,
    q_initial: np.ndarray,
    s_initial: float,
    beta: float,
    alpha: float, phi: float,
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
    current_gradient = _target_gradient_periodic_funnel_v1(q, target_parameters)

    # The same endpoint-gradient cache and phase charging used by HMC.
    warmup_gradient_calls = 1 if n_warmup > 0 else 0
    retained_gradient_calls = 0 if n_warmup > 0 else 1
    accepted_retained = 0
    warmup_nonfinite = 0
    retained_nonfinite = 0
    inverse_sqrt_beta = 1.0 / math.sqrt(beta)
    thermostat_momentum_scale = math.sqrt(Q / beta)

    for iteration in range(total_iterations):
        log_h, _log_h_derivative = _log_h_and_derivative_periodic_funnel_v1(
            s, alpha, phi, a
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
        initial_energy = _thc_hamiltonian_periodic_funnel_v1(
            q,
            p,
            s,
            p_s,
            beta,
            alpha, phi,
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
        ) = _thc_proposal_periodic_funnel_v1(
            q,
            p,
            s,
            p_s,
            current_gradient,
            beta,
            alpha, phi,
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
            proposed_energy = _thc_hamiltonian_periodic_funnel_v1(
                proposed_q,
                proposed_p,
                proposed_s,
                proposed_p_s,
                beta,
                alpha, phi,
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



_log_h_and_derivative=_log_h_and_derivative_periodic_funnel_v1
_thc_hamiltonian=_thc_hamiltonian_periodic_funnel_v1
_thc_proposal=_thc_proposal_periodic_funnel_v1
_simulate_thc_chain=_simulate_thc_chain_periodic_funnel_v1

def _run_chain_job(
    standard_momenta: np.ndarray,
    thermostat_standard_momenta: np.ndarray,
    uniforms: np.ndarray,
    q_initial: np.ndarray,
    s_initial: float,
    beta: float,
    alpha: float, phi: float,
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
        alpha, phi,
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
    alpha: float, phi: float,
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
            alpha, phi,
            a,
            Q,
            prepared.thermostat_sigma,
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
    thermostat = _thermostat_diagnostic(
        thermostat_samples, prepared.thermostat_sigma
    )
    return THCBlockResult(
        physical=physical, thermostat=thermostat
    )



def _aggregate_thc_configuration(
    alpha: float,
    phi: float,
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
        alpha=float(alpha),
        phi=float(phi),
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
            "No THC configuration has a defined full-block tuning score."
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
            -result.alpha,
            -result.phi,
            -result.a,
            -result.Q,
        ),
    )



def run_thc_tuning(inputs,*,preflight_only=False):
    if not preflight_only:
        raise ValueError('Use launcher/checkpoint_runner.py for resumable production.')
    if inputs.run_self_tests is not True: raise ValueError('Preflight tests are required')
    p=_prepare_thc_inputs(inputs)
    # Exercise the unmodified common diagnostic, target and RNG tests in the
    # archived base module. This is a small fixed fixture, not a tuning search.
    shared={f.name:getattr(inputs,f.name) for f in fields(base.RunInputs)}
    reference_inputs=replace(base.USER_INPUTS,**shared,lambda_values=(math.log(1.35),),
        a_values=(.75,),Q_values=(.1,),thermostat_sigma=inputs.thermostat_sigma,
        thermostat_starts=inputs.thermostat_starts)
    with contextlib.redirect_stdout(io.StringIO()):
        original=base.run_thc_tuning(reference_inputs,preflight_only=True)
    from periodic_validation import validate_periodic
    checks=validate_periodic(__import__('sys').modules[__name__],p)
    messages=tuple('COMMON BASE CHECK: '+x for x in original['self_test_messages'])+checks
    for line in messages: print(line,flush=True)
    return dict(prepared_inputs=p,self_test_messages=messages,preflight_only=True)

_common={f.name:getattr(base.USER_INPUTS,f.name) for f in fields(base.RunInputs)}
_common.update(protocol.common_changes('periodic'))
USER_INPUTS=PeriodicInputs(**_common,alpha_values=protocol.ALPHA,phi_values=protocol.PHI,
    a_values=protocol.A,Q_values=protocol.Q,thermostat_sigma=1.,
    thermostat_starts=(-1.,1.,1.,-1.))

if __name__=='__main__':
    run_thc_tuning(USER_INPUTS,preflight_only=True)
