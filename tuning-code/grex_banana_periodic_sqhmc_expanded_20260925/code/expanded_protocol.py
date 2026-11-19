"""Shared expanded Banana periodic THC-v2 / S-QHMC tuning design.
All settings use full lengths and all ten blocks. No staged search, no held-out run.
The seed domain deliberately matches the other 2026-09-25 Banana tuning jobs.
"""
from dataclasses import replace
import hashlib
import math

EPSILON = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)
L_VALUES = (5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240)
ALPHA = (0.025, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 0.95)
PHI = (0.0, math.pi/2)
A_VALUES = (0.125, 0.25, 0.375, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
Q_VALUES = (0.00625, 0.0125, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0)
MU_M = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
SIGMA_M = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 3.0)
N_WARMUP, N_RETAINED, N_WORKERS = 1000, 10000, 4
REJECTION_POLICY = "COUNT_NONFINITE_PROPOSALS_SCORE_FINITE_RETAINED_STATES"
SEED_DOMAIN = 'BANANA-HMC-BOUNDED-V4-EXPANDED-20260925-TUNING'
OLD_BLOCKS = (
    (123, 456, 789, 101112),
    (3545294353, 3961674632, 3627514536, 3198582250),
    (788645248, 3651519711, 1811141324, 1814869264),
    (3192285063, 2251015777, 396444798, 765150135),
    (1562880678, 1785952873, 2908883741, 2216662531),
)
ADDED_BLOCKS = tuple(tuple(int.from_bytes(hashlib.sha256(
    f'{SEED_DOMAIN}:block={b}:chain={c}'.encode()).digest()[:4], 'big')
    for c in range(1, 5)) for b in range(6, 11))
SEED_BLOCKS = OLD_BLOCKS + ADDED_BLOCKS

LABELS = {'periodic': 'THC_v2_periodic', 'sqhmc': 'S_QHMC'}
EXPECTED_COUNTS = {'periodic': 314600, 'sqhmc': 16731}

def expanded_inputs(original, method):
    """Change the declared search budget, grids and output identity only."""
    common = dict(epsilon_values=EPSILON, L_values=L_VALUES,
        seed_blocks=SEED_BLOCKS, n_warmup=N_WARMUP, n_retained=N_RETAINED,
        n_workers=N_WORKERS, run_self_tests=True, print_progress=True,
        print_report=True, run_mode='CUSTOM',
        run_label='CUSTOM: BANANA_PERIODIC_SQHMC_FULL_GRID_20260925',
        output_filename=LABELS[method]+'_Banana_expanded_tuning_20260925.txt',
        allow_overwrite=False)
    if method == 'periodic':
        # Keep the manuscript law N(0,1) and the original cosine/sine phases.
        # Legacy staged fields only satisfy the archived input validator;
        # the production driver does NOT execute any stage or boundary search.
        common.update(alpha_values=ALPHA, phi_values=PHI, a_values=A_VALUES,
            Q_values=Q_VALUES, thermostat_sigma=1.0,
            automatic_boundary_expansion=False,
            stage1_n_warmup=N_WARMUP, stage2_n_warmup=N_WARMUP,
            stage1_n_retained=N_RETAINED, stage2_n_retained=N_RETAINED,
            boundary_alpha_lower=0.0125, boundary_alpha_upper=0.975,
            boundary_a_lower=0.0625, boundary_a_upper=8.0,
            boundary_Q_lower=0.003125, boundary_Q_upper=16.0)
    elif method == 'sqhmc':
        common.update(mu_m_values=MU_M, sigma_m_values=SIGMA_M)
    else:
        raise ValueError(method)
    return replace(original, **common)

