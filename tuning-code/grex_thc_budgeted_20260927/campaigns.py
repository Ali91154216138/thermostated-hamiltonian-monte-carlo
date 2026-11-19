"""Approved reduced-search design. No four-day termination deadline."""
from pathlib import Path
import math

REMOTE_BASE = Path('/project/6021890/alireza7/THC_Experiments')
REMOTE_NEW = REMOTE_BASE / '2026-09-27_budgeted_tuning_01'
SELECTION_DOMAIN = 'THC-BUDGETED-20260927-v1'


def spec(key, local, remote, label, source, method, driver, kind,
         count, workers, cap, job, anchor):
    return dict(key=key, local=local, remote=remote, label=label, source=source,
                method=method, driver=driver, kind=kind, count=count,
                workers=workers, maximum_additional=cap, original_job=job,
                anchor=anchor)


SPECS = {
    'gaussian_bounded': spec('gaussian_bounded',
        'grex_gaussian_expanded_20260925',
        'multiscale_Gaussian/2026-09-25_expanded_01', 'THC',
        'THC/code/multiscale_gaussian_thc_tuner_EXPANDED_20260925_GREX_RUN.py',
        'thc', 'run_thc_tuning', 'gaussian', 81648, 4, 12000, '7596292',
        dict(lambda_value=math.log(1.25), a=.75, Q=1., epsilon=.06, L=30)),
    'gaussian_mixture': spec('gaussian_mixture',
        'grex_gaussian_mixture_sqhmc_expanded_20260925',
        'multiscale_Gaussian/2026-09-25_mixture_vs_SQHMC_expanded_01',
        'THC_mixture_v3',
        'THC_mixture_v3/code/multiscale_gaussian_thc_v3_mixture_tuner_EXPANDED_20260925.py',
        'mixture', 'run_mixture_thc_tuning', 'mixture', 156492, 4, 6000, '7596302',
        dict(delta=.02, Q=2., epsilon=.06, L=30)),
    'banana_bounded': spec('banana_bounded',
        'grex_banana_hmc_bounded_v4_expanded_20260925',
        'Banana/2026-09-25_hmc_bounded_v4_expanded_01', 'THC_bounded_scale',
        'code/banana_bounded_EXPANDED_20260925.py',
        'bounded', 'run_thc_tuning', 'banana', 81081, 4, None, '7596316',
        dict(lambda_value=math.log(1.35), a=1.5, Q=.5, epsilon=.04, L=80)),
    'banana_v4': spec('banana_v4',
        'grex_banana_hmc_bounded_v4_expanded_20260925',
        'Banana/2026-09-25_hmc_bounded_v4_expanded_01', 'THC_v4_conditional_shear',
        'code/banana_v4_EXPANDED_20260925.py',
        'v4', 'run_conditional_shear_tuning', 'banana', 240240, 4, 20000, '7596317',
        dict(kappa=.5, tau=.5, Q=.025, a=16., epsilon=.04, L=80)),
    'banana_periodic': spec('banana_periodic',
        'grex_banana_periodic_sqhmc_expanded_20260925',
        'Banana/2026-09-25_periodic_vs_SQHMC_expanded_01', 'THC_v2_periodic',
        'code/banana_periodic_EXPANDED_20260925.py',
        'periodic', 'run_phase_periodic_tuning', 'banana_periodic',
        314600, 4, 20000, '7596322',
        dict(coupling_code=.1, a=.75, Q=.1, epsilon=.04, L=80)),
    'funnel_scalar': spec('funnel_scalar',
        'grex_funnel_hmc_scalar_conditional_expanded_20260925',
        'Neals_funnel/2026-09-25_hmc_scalar_conditional_expanded_01', 'Scalar_THC',
        'code/funnel_scalar_EXPANDED_20260925.py',
        'scalar', 'run_thc_tuning', 'funnel', 1285200, 8, 30000, '7596336',
        dict(lambda_value=math.log(2.), a=.75, Q=2., epsilon=.08, L=20)),
    'funnel_periodic': spec('funnel_periodic',
        'grex_funnel_periodic_sqhmc_expanded_20260925',
        'Neals_funnel/2026-09-25_periodic_sqhmc_expanded_01', 'Periodic_THC',
        'code/funnel_periodic_EXPANDED_20260925.py',
        'periodic', 'run_thc_tuning', 'funnel', 1285200, 8, 30000, '7596357',
        dict(alpha=.1, phi=0., a=.75, Q=.1, epsilon=.04, L=80)),
}


def paths(key, local_base=None):
    s = SPECS[key]
    old = Path(local_base) / s['local'] if local_base else REMOTE_BASE / s['remote']
    return old, old / s['source'], old / s['label']


def source_files(key, local_base=None):
    old, source, run = paths(key, local_base)
    return sorted(set([old / 'launcher/checkpoint_runner.py', *source.parent.glob('*.py')]))
