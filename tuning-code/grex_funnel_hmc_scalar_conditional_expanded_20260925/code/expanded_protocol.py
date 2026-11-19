"""Define the common tuning settings for the six-dimensional Neal's funnel."""
from dataclasses import replace
import hashlib
import math

PROTOCOL_ID = "FUNNEL-HMC-SCALAR-CONDITIONAL-EXPANDED-20260925-v1"
EPSILON = (.0025,.005,.0075,.01,.015,.02,.03,.04,.05,.06,.08,.10,.12,.15,.18,.20,.25,.30)
LEAPFROG = (5,10,15,20,30,40,50,60,80,100,120,160,240,320)
SCALE_FACTORS = (1.005,1.01,1.025,1.05,1.10,1.15,1.25,1.35,1.5,1.75,2,2.5,3,4,6,8,12,16,24,32)
LAMBDA = tuple(math.log(v) for v in SCALE_FACTORS)
A = (.0625,.125,.25,.375,.5,.75,1,1.5,2,3,4,6,8,12,16)
Q = (.003125,.00625,.0125,.025,.05,.1,.2,.25,.5,1,2,4,8,16,32,64,128)
TAU = (.125,.2,.25,.375,.5,.625,.75,.875,1,1.25,1.5,2,3,4)
ORIGINAL_TUNING_BLOCKS = (
    (3857660299,1344569885,2010517369,2974962254),
    (2914251580,2216611040,1467312042,1966646752),
    (26638442,2162183804,2288357861,3874567655),
    (917698748,796766395,2769691025,1469822251),
    (983213432,4224231832,2042092071,4006032910),
)
SEED_DOMAIN = "FUNNEL-HMC-SCALAR-CONDITIONAL-EXPANDED-20260925-TUNING"
ADDED_TUNING_BLOCKS = tuple(tuple(int.from_bytes(hashlib.sha256(
    f"{SEED_DOMAIN}:block={b}:chain={c}".encode("ascii")).digest()[:4], "big")
    for c in range(1,5)) for b in range(6,11))
SEED_BLOCKS = ORIGINAL_TUNING_BLOCKS + ADDED_TUNING_BLOCKS
METHOD_LABELS = {"hmc":"HMC", "scalar":"Scalar_THC", "conditional":"Conditional_THC"}
SHARDS = {"hmc":1, "scalar":16, "conditional":4}
COUNTS = {"hmc":252, "scalar":1285200, "conditional":59976}

def expanded_inputs(original, method):
    """Only grid, run size/identity and output controls differ from original inputs."""
    extra = {}
    if method == "scalar":
        extra = dict(lambda_values=LAMBDA, a_values=A, Q_values=Q)
    elif method == "conditional":
        extra = dict(tau_values=TAU, Q_values=Q)
    elif method != "hmc":
        raise ValueError(method)
    return replace(original, epsilon_values=EPSILON, L_values=LEAPFROG,
        seed_blocks=SEED_BLOCKS, n_warmup=1000, n_retained=10000,
        n_workers=4, run_self_tests=True, print_progress=True, print_report=False,
        run_mode="CUSTOM", run_label=f"CUSTOM: {PROTOCOL_ID} {METHOD_LABELS[method]} TUNING ONLY",
        output_directory=None, output_filename=f"{METHOD_LABELS[method]}_Neals_funnel_expanded_tuning.txt",
        allow_overwrite=False, **extra)

assert len(set(seed for block in SEED_BLOCKS for seed in block)) == 40
assert len(EPSILON)*len(LEAPFROG) == COUNTS['hmc']
assert len(LAMBDA)*len(A)*len(Q)*COUNTS['hmc'] == COUNTS['scalar']
assert len(TAU)*len(Q)*COUNTS['hmc'] == COUNTS['conditional']
