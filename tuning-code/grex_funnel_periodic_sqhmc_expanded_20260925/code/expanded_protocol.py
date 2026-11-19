"""Frozen four-method funnel tuning extension. No held-out draws or screening."""
from dataclasses import replace
import math
from shared_funnel_protocol import EPSILON, LEAPFROG, A, Q, SEED_BLOCKS

PROTOCOL_ID = 'FUNNEL-PERIODIC-SQHMC-EXPANDED-20260925-v1'
ALPHA = (.025,.05,.10,.20,.30,.45,.60,.75,.90,.95)
PHI = (0., math.pi/2)
MU_M = (-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.)
SIGMA_M = (.001,.0025,.005,.0075,.01,.02,.03,.05,.10,.25,.50,1.,2.,3.)
METHOD_LABELS = {'periodic':'Periodic_THC','sqhmc':'S_QHMC'}
SHARDS = {'periodic':16,'sqhmc':4}
COUNTS = {'periodic':1285200,'sqhmc':31752}
REUSE_CAMPAIGN = '/project/6021890/alireza7/THC_Experiments/Neals_funnel/2026-09-25_hmc_scalar_conditional_expanded_01'
REUSE_JOBS = {'Scalar_THC':'7596336','Conditional_THC':'7596337'}

def common_changes(method):
    return dict(epsilon_values=EPSILON,L_values=LEAPFROG,seed_blocks=SEED_BLOCKS,
        n_warmup=1000,n_retained=10000,n_workers=4,run_self_tests=True,
        print_progress=True,print_report=False,run_mode='CUSTOM',
        run_label='CUSTOM: '+PROTOCOL_ID+' '+METHOD_LABELS[method]+' TUNING ONLY',
        output_directory=None,output_filename=METHOD_LABELS[method]+'_Neals_funnel_expanded_tuning.txt',
        allow_overwrite=False)

def expanded_inputs(original,method):
    if method!='sqhmc': raise ValueError(method)
    return replace(original,**common_changes(method),mu_m_values=MU_M,sigma_m_values=SIGMA_M)

assert len(ALPHA)*len(PHI)*len(A)*len(Q)*len(EPSILON)*len(LEAPFROG)==COUNTS['periodic']
assert len(MU_M)*len(SIGMA_M)*len(EPSILON)*len(LEAPFROG)==COUNTS['sqhmc']
