"""Define the common tuning settings for the Banana target."""
from dataclasses import replace
import hashlib
import math

EPSILON = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)
L_VALUES = (5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240)
AMPLITUDES = (1.025, 1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.0, 3.0)
BOUNDED_A = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
BOUNDED_Q = (0.0125, 0.025, 0.05, 0.10, 0.25, 0.5, 1.0, 2.0, 4.0)
V4_KAPPA = (0.125, 0.25, 0.40, 0.50, 0.65, 0.80, 1.0)
V4_TAU = (0.125, 0.25, 0.50, 1.0, 2.0)
V4_Q = (0.00625, 0.0125, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0)
V4_A = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
N_WARMUP = 1000
N_RETAINED = 10000
N_WORKERS = 4

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
LABELS = {'hmc': 'HMC', 'bounded': 'THC_bounded_scale', 'v4': 'THC_v4_conditional_shear'}

def expanded_inputs(original, method):
    """Change only the declared tuning design and output identity."""
    common = dict(epsilon_values=EPSILON, L_values=L_VALUES,
        seed_blocks=SEED_BLOCKS, n_warmup=N_WARMUP, n_retained=N_RETAINED,
        n_workers=N_WORKERS, run_self_tests=True, print_progress=True,
        print_report=True, run_mode='CUSTOM',
        run_label='CUSTOM: EXPANDED_FULL_GRID_20260925_10_BLOCKS',
        output_filename=LABELS[method]+'_Banana_expanded_tuning_20260925.txt',
        allow_overwrite=False)
    if method == 'bounded':
        common.update(lambda_values=tuple(math.log(x) for x in AMPLITUDES),
                      a_values=BOUNDED_A, Q_values=BOUNDED_Q)
    elif method == 'v4':
        common.update(kappa_values=V4_KAPPA, tau_values=V4_TAU,
            Q_values=V4_Q, a_values=V4_A, automatic_boundary_expansion=False,
            boundary_kappa_lower=0.0625, boundary_kappa_upper=1.5,
            boundary_tau_lower=0.0625, boundary_tau_upper=4.0,
            boundary_Q_lower=0.003125, boundary_Q_upper=2.0,
            boundary_a_lower=0.5, boundary_a_upper=64.0)
    elif method != 'hmc':
        raise ValueError(method)
    return replace(original, **common)
