"""Checkpoint-preserving, reduced-search wrapper around frozen numerical code.

No sampling equations, seeds, draw counts, score gates or selectors are changed.
The four-day request is an ETA target, NOT a termination condition.
"""
from __future__ import annotations
import argparse
import contextlib
import dataclasses
import hashlib
import heapq
import importlib.metadata
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import zlib

from campaigns import SPECS, SELECTION_DOMAIN, REMOTE_NEW, paths, source_files

STOP = None


def stop_requested(signum, frame):
    global STOP
    STOP = 'allocation' if signum == getattr(signal, 'SIGUSR1', -1) else 'cancel'
    print('Stop request: commit this block; preserve all previous blocks.', flush=True)


def plain(value):
    if dataclasses.is_dataclass(value): return plain(dataclasses.asdict(value))
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if hasattr(value, 'tolist'): return plain(value.tolist())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    return value


def encode(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as out:
            out.write(value); out.flush(); os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


@contextlib.contextmanager
def lock_file(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        else:
            import msvcrt
            if f.tell() == 0: f.write('0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        try: yield
        finally:
            if os.name != 'posix':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=120)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Grid:
    def __init__(self, keys, values):
        self.keys = tuple(keys)
        self.values = tuple(tuple(v) for v in values)
        self.size = math.prod(map(len, self.values))

    def __len__(self): return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size: raise IndexError(index)
        answer = {}
        for k, vals in reversed(tuple(zip(self.keys, self.values))):
            index, j = divmod(index, len(vals)); answer[k] = vals[j]
        return {k: answer[k] for k in self.keys}

    def index(self, settings):
        i = 0
        for k, vals in zip(self.keys, self.values):
            j = min(range(len(vals)), key=lambda n: abs(vals[n] - settings[k]))
            if not math.isclose(vals[j], settings[k], rel_tol=1e-13, abs_tol=1e-15):
                raise ValueError(f'Out-of-grid anchor: {k}={settings[k]}')
            i = i * len(vals) + j
        return i

    def axis(self, name): return self.values[self.keys.index(name)]


class Adapter:
    """One adapter per interpreter avoids collisions in archived module names."""
    def __init__(self, key, local_base=None, preflight=False, inputs=None):
        self.key, self.spec = key, SPECS[key]
        self.old, self.source, self.old_run = paths(key, local_base)
        self.runner = import_file('_budget_original_runner', self.old / 'launcher/checkpoint_runner.py')
        self.module = self.runner.load_tuner(self.source, self.spec['method'])
        self.inputs = inputs or dataclasses.replace(self.module.USER_INPUTS,
            output_directory=str(self.old_run / 'results'))
        if preflight:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = getattr(self.module, self.spec['driver'])(self.inputs, preflight_only=True)
            self.prepared = result['prepared_inputs']
            self.preflight_text = output.getvalue()
            self.tests = result.get('self_test_messages', result.get('self_tests', ()))
            if not self.tests: raise RuntimeError('Original preflight returned no tests')
        else:
            fn = '_prepare_inputs' if self.spec['method'] in ('mixture', 'v4') else '_prepare_thc_inputs'
            self.prepared = getattr(self.module, fn)(self.inputs)
            self.tests, self.preflight_text = (), ''
        p = self.prepared
        self.common = getattr(p, 'common', p)
        method, kind = self.spec['method'], self.spec['kind']
        if method == 'mixture':
            # IMPORTANT: this archived tuner orders epsilon,L BEFORE delta,Q.
            keys = ('epsilon', 'L', 'delta', 'Q')
        elif method == 'v4': keys = ('kappa', 'tau', 'Q', 'a', 'epsilon', 'L')
        elif kind == 'banana_periodic': keys = ('coupling_code', 'a', 'Q', 'epsilon', 'L')
        elif kind == 'funnel' and method == 'periodic': keys = ('alpha', 'phi', 'a', 'Q', 'epsilon', 'L')
        else: keys = ('lambda_value', 'a', 'Q', 'epsilon', 'L')
        values = []
        for k in keys:
            owner = self.common if k in ('epsilon', 'L') else p
            attr = {'lambda_value': 'lambda_values', 'coupling_code': 'phase_codes'}.get(k, k + '_values')
            values.append(getattr(owner, attr))
        self.grid = Grid(keys, values)
        self.cache = None

    def check_production(self):
        c = self.common
        if (len(self.grid), c.n_warmup, c.n_retained, c.n_chains, len(c.seed_blocks)) != (
                self.spec['count'], 1000, 10000, 4, 10):
            raise RuntimeError('Frozen production grid or evaluation budget changed')
        if len(set(s for row in c.seed_blocks for s in row)) != 40:
            raise RuntimeError('Seed schedule is not 40 unique seeds')
        if not getattr(self.module, 'NUMBA_AVAILABLE', hasattr(self.module, 'numba')):
            raise RuntimeError('Numba required')

    def streams(self):
        if self.cache is None:
            c = self.common
            self.cache = tuple(tuple(self.module._make_random_streams(s,
                c.n_warmup + c.n_retained, c.dimension) for s in row) for row in c.seed_blocks)
        return self.cache

    def run_block(self, index, block):
        m, c, p = self.module, self.common, self.prepared
        settings = self.grid[index]; cache = self.streams()
        if self.spec['method'] == 'mixture':
            return m._run_configuration_block(configuration=m.Configuration(**settings),
                origin='EXPANDED_FULL_GRID', block_index=block, prepared=p,
                cache=cache, n_warmup=c.n_warmup, n_retained=c.n_retained)
        if self.spec['method'] == 'v4':
            return m._run_configuration_block(m.FixedConfiguration(**settings), block,
                c.n_warmup, c.n_retained, cache, p)
        return m._run_configuration_block(**settings, block_index=block,
            seeds=c.seed_blocks[block], stream_cache=cache[block], prepared=p)

    def aggregate(self, index, blocks):
        return self.runner.aggregate(self.module, self.spec['method'], self.grid[index], blocks)

    def select(self, best, candidate):
        if candidate.status != 'ok': return best
        if best is None: return candidate
        if self.spec['kind'] == 'gaussian':
            return self.module._select_thc_representative((best, candidate))
        return self.runner.select_pair(self.module, self.spec['method'], best, candidate)

    def sources(self):
        return {str(p.relative_to(self.old)).replace('\\', '/'): sha(p) for p in source_files(self.key,
            self.old.parent if self.old.name == self.spec['local'] else None)}


def physical(block): return getattr(block, 'physical', block)


def block_summary(block):
    b = physical(block)
    names = ('block_index', 'seeds', 'status', 'worst_observable', 'worst_iat',
             'block_efficiency', 'acceptance', 'retained_gradient_calls',
             'gradients_per_retained_draw', 'maximum_rhat',
             'maximum_standardized_moment_error', 'warmup_nonfinite_trajectories',
             'retained_nonfinite_trajectories')
    return {k: getattr(b, k) for k in names}


def result_summary(adapter, index, result):
    iats = [physical(b).worst_iat for b in result.block_results]
    valid = all(math.isfinite(v) and v > 0 for v in iats)
    return dict(global_index=index, settings=adapter.grid[index], status=result.status,
        geometric_mean_efficiency=result.geometric_mean_efficiency,
        max_block_worst_IAT=max(iats) if valid else math.inf,
        geometric_mean_worst_IAT=math.exp(math.fsum(map(math.log, iats)) / len(iats)) if valid else math.inf,
        maximum_Rhat=result.maximum_rhat, mean_acceptance=result.mean_acceptance,
        maximum_moment_error=result.maximum_standardized_moment_error,
        minimum_block_efficiency=result.minimum_block_efficiency,
        total_nonfinite_trajectories=result.total_nonfinite_trajectories,
        blocks=[block_summary(b) for b in result.block_results])


def iat_key(summary):
    return (summary['max_block_worst_IAT'], summary['geometric_mean_worst_IAT'],
            -summary['geometric_mean_efficiency'], summary['global_index'])


def old_databases(adapter):
    run = adapter.old_run
    if adapter.spec['kind'] == 'funnel':
        found = sorted(run.glob('parts/part_*/checkpoints/blocks.sqlite3'))
        if len(found) != 16: raise RuntimeError('Expected all 16 original parts')
        return found
    return [run / 'checkpoints/blocks.sqlite3']


def scan_database(path, grid_size, blocks=10, verify_payloads=False):
    masks = {}; digest = hashlib.sha256()
    with contextlib.closing(readonly(path)) as db:
        check = db.execute('PRAGMA quick_check').fetchone()[0]
        if check != 'ok': raise RuntimeError(f'Checkpoint integrity: {check}')
        manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        fields = 'config,block,digest,payload' if verify_payloads else 'config,block,digest'
        count = 0
        for row in db.execute(f'SELECT {fields} FROM blocks ORDER BY config,block'):
            i, b, expected = row[:3]
            if not (0 <= i < grid_size and 0 <= b < blocks): raise RuntimeError('Bad checkpoint indices')
            if masks.get(i, 0) & (1 << b): raise RuntimeError('Duplicate checkpoint block')
            masks[i] = masks.get(i, 0) | (1 << b)
            if verify_payloads and hashlib.sha256(row[3]).hexdigest() != expected:
                raise RuntimeError(f'Corrupt payload {path}: {i},{b}')
            digest.update(f'{i}:{b}:{expected}\n'.encode()); count += 1
    return dict(manifest=manifest, masks=masks, blocks=count, row_digest=digest.hexdigest())


def merge_masks(scans):
    masks = {}
    for scan in scans:
        for i, mask in scan['masks'].items():
            if masks.get(i, 0) & mask: raise RuntimeError('Overlapping original parts')
            masks[i] = masks.get(i, 0) | mask
    return masks


def select_indices(grid, masks, cap, key, anchor, block_count=10):
    """Balance NEW settings across thermostat tuples; never inspect scores."""
    full = (1 << block_count) - 1
    complete = {i for i, mask in masks.items() if mask == full}
    if cap is None: return sorted(set(range(len(grid))) - complete), sorted(complete)
    selected = {i for i, mask in masks.items() if mask != full}
    # Preserve all partially evaluated configurations and reference settings.
    if anchor is not None:
        anchor_index = grid.index(anchor)
        if anchor_index not in complete: selected.add(anchor_index)
    if len(selected) > cap: raise RuntimeError('Budget smaller than mandatory partial/reference settings')
    if len(selected) == cap: return sorted(selected), sorted(complete)
    thermostat_keys = [k for k in grid.keys if k not in ('epsilon', 'L')]
    tuples = list(itertools.product(*(grid.axis(k) for k in thermostat_keys)))
    pairs = list(itertools.product(grid.axis('epsilon'), grid.axis('L')))
    seed = int.from_bytes(hashlib.sha256((SELECTION_DOMAIN + key).encode()).digest(), 'big')
    rng = random.Random(seed); rng.shuffle(tuples)
    candidates = []
    for values in tuples:
        thermo = dict(zip(thermostat_keys, values))
        order = list(pairs); rng.shuffle(order)
        candidates.append([grid.index(dict(thermo, epsilon=e, L=L)) for e, L in order])
    for level in range(len(pairs)):
        for group in candidates:
            i = group[level]
            if i not in complete: selected.add(i)
            if len(selected) >= cap: return sorted(selected), sorted(complete)
    return sorted(selected), sorted(complete)


def assign_parts(grid, new_indices, old_complete, masks, parts, block_count=10):
    """Greedy cost balance, using L only, not observed performance/quality."""
    owners = [[] for _ in range(parts)]
    heap = [(0, p) for p in range(parts)]
    weighted = [( (block_count - masks.get(i, 0).bit_count()) * (grid[i]['L'] + 20), i)
                for i in new_indices]
    for cost, i in sorted(weighted, reverse=True):
        load, p = heapq.heappop(heap); owners[p].append(i)
        heapq.heappush(heap, (load + cost, p))
    # Completed data need reporting but do not require new simulation.
    for n, i in enumerate(old_complete): owners[n % parts].append(i)
    for p, values in enumerate(owners):
        random.Random(f'{SELECTION_DOMAIN}:{p}').shuffle(values)
    flat = list(itertools.chain.from_iterable(owners))
    if len(flat) != len(set(flat)) or set(flat) != set(new_indices) | set(old_complete):
        raise RuntimeError('Part assignment is not disjoint/exhaustive')
    return owners


def runtime(adapter):
    # The first Gaussian runner recorded numba.__version__; the later runners
    # recorded the installed distribution version (which has a CC build suffix).
    numba_version = (adapter.module.numba.__version__ if adapter.spec['kind'] == 'gaussian'
                     else importlib.metadata.version('numba'))
    return dict(python=platform.python_version(), numpy=adapter.module.np.__version__,
        numba=numba_version, llvmlite=importlib.metadata.version('llvmlite'))


def wrapper_sources():
    return {name: sha(Path(__file__).parent / name) for name in
            ('budgeted_runner.py', 'campaigns.py', 'run_budgeted.sbatch')}


def validate_original(adapter, manifest):
    m = manifest.get('common', manifest)
    if m['method'] != adapter.spec['method']: raise RuntimeError('Wrong original method')
    if m['source_sha256'] != sha(adapter.source): raise RuntimeError('Original source changed')
    launcher = m.get('launcher_sha256', m.get('runner_sha256'))
    if launcher != sha(adapter.old / 'launcher/checkpoint_runner.py'):
        raise RuntimeError('Original launcher changed')
    for k, v in runtime(adapter).items():
        if m[k] != v: raise RuntimeError(f'Numerical runtime changed: {k}: {m[k]} != {v}')
    expected_inputs = (json.loads(json.dumps(dataclasses.asdict(adapter.inputs), default=str))
                       if adapter.spec['kind'] == 'gaussian' else plain(adapter.inputs))
    if m['inputs'] != expected_inputs: raise RuntimeError('Original input manifest differs')
    if 'protocol_sha256' in m:
        if m['protocol_sha256'] != sha(adapter.source.parent / 'expanded_protocol.py'):
            raise RuntimeError('Original shared protocol changed')
    for name, expected in m.get('code_sha256', {}).items():
        if sha(adapter.source.parent / name) != expected: raise RuntimeError(f'Changed code: {name}')


def checked_value(row):
    if row is None or hashlib.sha256(row[1]).hexdigest() != row[0]:
        raise RuntimeError('Missing or corrupt checkpoint payload')
    return pickle.loads(zlib.decompress(row[1]))


def check_grid(adapter):
    r, p, m = adapter.runner, adapter.prepared, adapter.spec['method']
    if adapter.spec['kind'] in ('gaussian', 'funnel'): original = r.configurations(p, m)
    else: original = r.configurations(adapter.module, p, m)
    if len(original) != len(adapter.grid): raise RuntimeError('Original grid size mismatch')
    # Every index is compared: saved blocks retain their original global indices.
    for i in range(len(original)):
        if original[i] != adapter.grid[i]: raise RuntimeError(f'Grid indexing mismatch at {i}')
    print(f'PASS: all {len(original)} original configuration indices match.', flush=True)


def native_preflight(key, destination, local_base=None):
    adapter = Adapter(key, local_base, preflight=True)
    adapter.check_production(); check_grid(adapter)
    expected_file = Path(__file__).parent / 'expected_sources.txt'
    if expected_file.exists():
        expected = json.loads(expected_file.read_text())[key]
        if adapter.sources() != expected: raise RuntimeError('Local/deployed original source mismatch')
    tests = list(adapter.tests)
    reproduction = None
    if local_base is None:
        for path in old_databases(adapter):
            with contextlib.closing(readonly(path)) as db:
                manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
                validate_original(adapter, manifest)
                if reproduction is None:
                    # One exact full-length block parity check, using an already committed record.
                    row = db.execute('SELECT config,block,digest,payload FROM blocks ORDER BY config,block LIMIT 1').fetchone()
                    if row: reproduction = (row[0], row[1], checked_value(row[2:]))
        if reproduction is None: raise RuntimeError('No old checkpoint to verify')
        i, b, prior = reproduction
        current = adapter.run_block(i, b)
        if encode(current) != encode(prior):
            raise RuntimeError(f'Exact full-length checkpoint replay failed: {i},{b}')
        tests.append(f'PASS: exact 1000+10000-draw checkpoint replay at original setting {i}, block {b}.')
    # Benchmark complete production blocks, NOT reduced-length tuning scores.
    # Reference thermostat + highest L; include small step sizes and high coupling.
    base = dict(adapter.spec['anchor'])
    indices = [adapter.grid.index(base)]
    for e in (min(adapter.grid.axis('epsilon')), base['epsilon']):
        indices.append(adapter.grid.index(dict(base, epsilon=e, L=max(adapter.grid.axis('L')))))
    extreme = dict(base, epsilon=min(adapter.grid.axis('epsilon')), L=max(adapter.grid.axis('L')))
    for name in adapter.grid.keys:
        if name not in ('epsilon', 'L'):
            extreme[name] = min(adapter.grid.axis(name)) if name in ('Q', 'a', 'delta') else max(adapter.grid.axis(name))
    indices.append(adapter.grid.index(extreme))
    timings = []
    for i in dict.fromkeys(indices):
        started = time.monotonic(); value = adapter.run_block(i, 0); elapsed = time.monotonic() - started
        result = adapter.aggregate(i, [value] * len(adapter.common.seed_blocks))
        # Also exercise all report fields and the frozen original selector.
        summary = result_summary(adapter, i, result); adapter.select(None, result)
        timings.append(dict(index=i, settings=adapter.grid[i], seconds_per_block=elapsed,
                            block=block_summary(value)))
        print(f'BENCHMARK {key} L={adapter.grid[i]["L"]}: {elapsed:.3f}s/full block', flush=True)
    tests.extend(('PASS: original preflight and frozen numerical input checks.',
                  'PASS: original aggregate/selector and extended IAT report fields.'))
    record = dict(key=key, tests=tests, source_checksums=adapter.sources(), runtime=runtime(adapter),
                  wrapper_checksums=wrapper_sources(), timings=timings,
                  timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  original_grid_size=len(adapter.grid), benchmark_only=True)
    atomic(destination, encode(record) + '\n')
    print('\n'.join(tests), flush=True)
    return record


def snapshot_and_freeze(key, root, cap=None, local_base=None):
    """Run ONLY after the specified original Slurm job has stopped."""
    adapter = Adapter(key, local_base); adapter.check_production()
    root = Path(root); folder = root / key
    if local_base is None:
        jobs = subprocess.check_output(['squeue', '-r', '-u', 'alireza7', '-h', '-o', '%F'], text=True).split()
        if adapter.spec['original_job'] in jobs: raise RuntimeError('Original job still active: no snapshot taken')
    preflight = json.loads((root / 'preflight' / f'{key}.txt').read_text())
    if preflight['source_checksums'] != adapter.sources() or preflight['runtime'] != runtime(adapter):
        raise RuntimeError('Preflight provenance mismatch')
    if preflight['wrapper_checksums'] != wrapper_sources(): raise RuntimeError('Wrapper changed after preflight')
    with lock_file(folder / 'setup.lock'):
        if (folder / 'FROZEN.txt').exists():
            raise RuntimeError('Selection already frozen; do not silently overwrite it')
        scans = []; snapshots = []
        for n, path in enumerate(old_databases(adapter)):
            target = folder / 'preserved' / f'original_part_{n:03d}.sqlite3'
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): raise RuntimeError(f'Snapshot exists; verify before retry: {target}')
            with contextlib.closing(readonly(path)) as src, contextlib.closing(sqlite3.connect(target)) as dst:
                src.backup(dst)
            scan = scan_database(target, len(adapter.grid), verify_payloads=True)
            validate_original(adapter, scan['manifest'])
            scans.append(scan)
            snapshots.append(dict(original=str(path), preserved=str(target.relative_to(folder)),
                file_sha256=sha(target), row_sha256=scan['row_digest'], blocks=scan['blocks']))
            print(f'PRESERVED {key} part {n}: {scan["blocks"]} verified blocks', flush=True)
        masks = merge_masks(scans)
        allowed = adapter.spec['maximum_additional']
        chosen_cap = allowed if cap is None else cap
        if allowed is None and cap is not None: raise RuntimeError('Banana bounded must keep its full grid')
        if allowed is not None and not 1 <= chosen_cap <= allowed: raise RuntimeError('Cap exceeds approval')
        # Conservative measured model: full reference/extreme blocks, 50% margin,
        # plus 30 minutes startup/report time. ETA is NEVER a stopping condition.
        coefficient = max(t['seconds_per_block'] / (t['settings']['L'] + 20) for t in preflight['timings'])
        while True:
            new, complete = select_indices(adapter.grid, masks, chosen_cap, key, adapter.spec['anchor'])
            owners = assign_parts(adapter.grid, new, complete, masks, adapter.spec['workers'])
            max_load = max(sum((10 - masks.get(i, 0).bit_count()) * (adapter.grid[i]['L'] + 20) for i in part) for part in owners)
            estimated_days = (1.5 * coefficient * max_load + 1800) / 86400
            if estimated_days <= 3.5 or chosen_cap is None: break
            reduced = max(1, int(chosen_cap * 3.4 / estimated_days))
            if reduced >= chosen_cap: raise RuntimeError('Cannot meet ETA target with current selection')
            chosen_cap = reduced
            print(f'Before freezing: reduce {key} additional cap to {chosen_cap} based on native timing.', flush=True)
        if estimated_days > 4:
            raise RuntimeError(f'Estimated {estimated_days:.3f} compute days exceeds target; select a smaller cap before freeze')
        for part, indices in enumerate(owners):
            atomic(folder / 'parts' / f'part_{part:03d}' / 'indices.txt', encode(indices) + '\n')
        record = dict(format='THC-budgeted-v1', key=key, workers=len(owners),
            original_grid=len(adapter.grid), preserved_complete_configurations=len(complete),
            additional_configurations=len(new), total_configurations=len(new) + len(complete),
            preserved_blocks=sum(s['blocks'] for s in scans),
            remaining_blocks=sum(10 - masks.get(i, 0).bit_count() for i in new),
            approved_maximum_additional=allowed, chosen_cap=chosen_cap,
            selection_domain=SELECTION_DOMAIN, score_independent_selection=True,
            inputs=plain(adapter.inputs), runtime=runtime(adapter), source_checksums=adapter.sources(),
            wrapper_checksums=wrapper_sources(), snapshots=snapshots,
            parts=[dict(part=p, configurations=len(v), indices_sha256=sha(folder / 'parts' / f'part_{p:03d}' / 'indices.txt')) for p, v in enumerate(owners)],
            estimated_compute_days_with_margin=estimated_days, timing_coefficient=coefficient,
            wallclock_cutoff=None, held_out=False, original_job=adapter.spec['original_job'],
            frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        atomic(folder / 'FROZEN.txt', encode(record) + '\n')
        print(f'FROZEN {key}: {len(complete)} reused complete + {len(new)} additional configurations; {len(owners)} workers; conservative compute ETA {estimated_days:.2f}d.', flush=True)
        return record


class OverlayStore:
    """Read immutable original snapshots; write only this new part's database."""
    def __init__(self, path, manifest, snapshots):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=120)
        self.db.execute('PRAGMA journal_mode=DELETE'); self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS blocks(config INTEGER,block INTEGER,digest TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(config,block))')
        row = self.db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if row and row[0] != encode(manifest):
            self.db.close()
            raise RuntimeError('New checkpoint manifest changed')
        if not row:
            with self.db: self.db.execute('INSERT INTO metadata VALUES (?,?)', ('manifest', encode(manifest)))
        self.originals = [readonly(p) for p in snapshots]
        self.owner = {}
        for n, db in enumerate(self.originals):
            for (i,) in db.execute('SELECT DISTINCT config FROM blocks'):
                if i in self.owner: raise RuntimeError('Original configuration occurs in multiple parts')
                self.owner[i] = n

    def rows(self, i):
        values = {}
        databases = ([self.originals[self.owner[i]]] if i in self.owner else []) + [self.db]
        for db in databases:
            for b, digest, payload in db.execute('SELECT block,digest,payload FROM blocks WHERE config=?', (i,)):
                if b in values: raise RuntimeError('Duplicate reused/new block')
                values[b] = checked_value((digest, payload))
        return values

    def put(self, i, b, value):
        payload = zlib.compress(pickle.dumps(value, protocol=5), 1)
        with self.db:
            self.db.execute('INSERT INTO blocks VALUES (?,?,?,?)', (i, b, hashlib.sha256(payload).hexdigest(), payload))

    def close(self):
        self.db.close()
        for db in self.originals: db.close()


def report_header(adapter, frozen):
    c = adapter.common
    return '\n'.join((f'{adapter.key}: FULL-LENGTH TUNING ONLY',
        'No held-out comparison or claim of a final win. Four days is an ETA target, not a cutoff.',
        'Reduced, deterministic, score-independent coverage of the original grid; all previously completed settings included.',
        'Banana bounded-scale is the exception: its entire original grid is retained.',
        'Every included configuration: 10 common seed blocks, 4 chains, 1000 warmup + 10000 retained per chain.',
        'Original kernels, targets, starts, exact moments, seeds, IAT/R-hat, gradient-cost accounting, nonfinite policy and efficiency tie-breaks unchanged.',
        'IAT and R-hat are descriptive; primary selection uses the original geometric-mean efficiency score.',
        'Secondary IAT selection: smallest maximum blockwise worst IAT among configurations with defined scores; then GM worst IAT, greater efficiency, global index.',
        'Exact observable means: ' + encode(dict(zip(c.observable_names, c.exact_observable_means))),
        'Physical starts: ' + encode(c.physical_starts),
        'FROZEN MANIFEST: ' + encode(frozen),
        'ALL CONFIGURATION AND BLOCK SUMMARIES:')) + '\n'


def write_part_report(adapter, store, indices, destination, frozen):
    best = best_iat = None; best_index = iat_index = None; defined = 0
    fd, temp = tempfile.mkstemp(prefix='report.', dir=destination.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as out:
            out.write(report_header(adapter, frozen))
            for i in sorted(indices):
                if STOP: raise InterruptedError('Report paused; checkpoints preserved')
                rows = store.rows(i)
                if set(rows) != set(range(10)): raise RuntimeError('Incomplete configuration in report')
                result = adapter.aggregate(i, [rows[b] for b in range(10)])
                summary = result_summary(adapter, i, result)
                out.write(encode(summary) + '\n')
                if result.status == 'ok':
                    defined += 1
                    selected = adapter.select(best, result)
                    if selected is result: best, best_index = result, i
                    if best_iat is None or iat_key(summary) < iat_key(result_summary(adapter, iat_index, best_iat)):
                        best_iat, iat_index = result, i
            out.write(f'DEFINED/TESTED: {defined}/{len(indices)}\nEND OF PART REPORT\n')
            out.flush(); os.fsync(out.fileno())
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    return dict(defined=defined, primary_index=best_index, secondary_IAT_index=iat_index)


def validate_frozen(adapter, folder):
    frozen_path = folder / 'FROZEN.txt'
    record = json.loads(frozen_path.read_text())
    if record['source_checksums'] != adapter.sources(): raise RuntimeError('Original source checksum changed')
    if record['wrapper_checksums'] != wrapper_sources(): raise RuntimeError('Wrapper checksum changed')
    if record['runtime'] != runtime(adapter) or record['inputs'] != plain(adapter.inputs):
        raise RuntimeError('Frozen runtime/input mismatch')
    return record


def finalize(adapter, folder, frozen):
    with lock_file(folder / 'finalize.lock', blocking=True):
        markers = [folder / 'parts' / f'part_{p:03d}' / 'COMPLETED.txt' for p in range(frozen['workers'])]
        if not all(p.exists() for p in markers): return
        final = folder / 'results' / 'FINAL_TUNING_REPORT.txt'
        final.parent.mkdir(parents=True, exist_ok=True)
        if (folder / 'COMPLETED.txt').exists():
            marker = json.loads((folder / 'COMPLETED.txt').read_text())
            if sha(final) != marker['report_sha256']: raise RuntimeError('Final report checksum mismatch')
            return
        best = best_iat = None; best_index = iat_index = None; defined = 0
        for marker in markers:
            record = json.loads(marker.read_text())
            report = marker.parent / 'configuration_results.txt'
            if sha(report) != record['report_sha256']: raise RuntimeError('Part report checksum changed')
            defined += record['defined']
            summary_file = marker.parent / 'selected_records.txt'
            if sha(summary_file) != record['selected_sha256']: raise RuntimeError('Selected records changed')
            candidates = pickle.loads(zlib.decompress(bytes.fromhex(summary_file.read_text())))
            for label, pair in candidates.items():
                if pair is None: continue
                i, result = pair
                if label == 'primary':
                    selected = adapter.select(best, result)
                    if selected is result: best, best_index = result, i
                else:
                    if best_iat is None or iat_key(result_summary(adapter, i, result)) < iat_key(result_summary(adapter, iat_index, best_iat)):
                        best_iat, iat_index = result, i
        text = report_header(adapter, frozen)
        text += f'Combined DEFINED/TESTED: {defined}/{frozen["total_configurations"]}\n'
        for label, i, result in (('PRIMARY EFFICIENCY REPRESENTATIVE', best_index, best), ('SECONDARY WORST-IAT REPRESENTATIVE', iat_index, best_iat)):
            text += label + '\n'
            text += 'NONE: no defined score.\n' if result is None else json.dumps(plain(result_summary(adapter, i, result)), indent=2) + '\n'
            if result is not None:
                text += 'ALL OBSERVABLES, EXACT/ESTIMATED MOMENTS, RHAT COMPONENTS, GEYER CUTOFFS\n'
                text += json.dumps(plain(result), indent=2) + '\n'
        text += 'All candidates including failed configurations are in parts/part_*/configuration_results.txt.\nEND OF FINAL TUNING REPORT\n'
        atomic(final, text)
        record = dict(report=str(final), report_sha256=sha(final), configurations=frozen['total_configurations'],
                      defined=defined, primary_index=best_index, secondary_IAT_index=iat_index,
                      finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        atomic(folder / 'COMPLETED.txt', encode(record) + '\n')
        print(text, flush=True)


def run_worker(key, root, part, stop_after_blocks=None, local_base=None):
    adapter = Adapter(key, local_base, preflight=True); adapter.check_production()
    folder = Path(root) / key; frozen = validate_frozen(adapter, folder)
    if not 0 <= part < frozen['workers']: raise RuntimeError('Invalid part')
    part_dir = folder / 'parts' / f'part_{part:03d}'
    indexfile = part_dir / 'indices.txt'
    if sha(indexfile) != frozen['parts'][part]['indices_sha256']: raise RuntimeError('Part selection changed')
    indices = json.loads(indexfile.read_text())
    snapshots = [folder / s['preserved'] for s in frozen['snapshots']]
    # Snapshot payloads were all validated at freeze and are rechecked when read.
    for path, s in zip(snapshots, frozen['snapshots']):
        if sha(path) != s['file_sha256']: raise RuntimeError('Preserved snapshot changed')
    with lock_file(part_dir / 'campaign.lock'):
        store = OverlayStore(part_dir / 'blocks.sqlite3',
                             dict(frozen_sha256=sha(folder / 'FROZEN.txt'), part=part), snapshots)
        try:
            marker = part_dir / 'COMPLETED.txt'
            if marker.exists():
                record = json.loads(marker.read_text())
                if sha(part_dir / 'configuration_results.txt') != record['report_sha256']:
                    raise RuntimeError('Completed part changed')
                finalize(adapter, folder, frozen); return 0
            atomic(part_dir / 'PREFLIGHT.txt', '\n'.join(adapter.tests) + '\nPASS: frozen inputs, sources, preserved snapshots and part selection.\n')
            print(f'PREFLIGHT PASSED: {key} part {part+1}/{frozen["workers"]}; {len(indices)} configurations.', flush=True)
            start = time.monotonic(); new = completed = 0
            total = len(indices) * 10
            for i in indices:
                rows = store.rows(i)
                for b in range(10):
                    if b in rows: completed += 1; continue
                    if STOP or (stop_after_blocks is not None and new >= stop_after_blocks):
                        return 143 if STOP == 'cancel' else 75
                    value = adapter.run_block(i, b)
                    store.put(i, b, value); new += 1; completed += 1
                    progress = dict(key=key, part=part, visited_saved_blocks=completed, assigned_blocks=total,
                        new_blocks_this_session=new, session_seconds=time.monotonic()-start,
                        index=i, settings=adapter.grid[i], last_block=block_summary(value),
                        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                    atomic(part_dir / 'PROGRESS.txt', encode(progress) + '\n')
                    print(f'{key} part {part}: {completed}/{total} blocks visited; {new} newly saved; {progress["session_seconds"]:.1f}s', flush=True)
            if STOP: return 143 if STOP == 'cancel' else 75
            try:
                winners = write_part_report(adapter, store, indices, part_dir / 'configuration_results.txt', frozen)
            except InterruptedError: return 143 if STOP == 'cancel' else 75
            selected = {}
            for label, field in (('primary', 'primary_index'), ('secondary_IAT', 'secondary_IAT_index')):
                i = winners[field]
                selected[label] = None if i is None else (i, adapter.aggregate(i, [store.rows(i)[b] for b in range(10)]))
            selected_path = part_dir / 'selected_records.txt'
            atomic(selected_path, zlib.compress(pickle.dumps(selected, protocol=5), 1).hex())
            record = dict(winners, configurations=len(indices), report_sha256=sha(part_dir / 'configuration_results.txt'),
                          selected_sha256=sha(selected_path), frozen_sha256=sha(folder / 'FROZEN.txt'))
            atomic(marker, encode(record) + '\n')
            finalize(adapter, folder, frozen)
            print(f'PART COMPLETE: {key} {part}', flush=True)
            return 0
        finally: store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'freeze', 'run'))
    parser.add_argument('--key', choices=tuple(SPECS), required=True)
    parser.add_argument('--root', type=Path, default=REMOTE_NEW)
    parser.add_argument('--local-base', type=Path)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--cap', type=int)
    args = parser.parse_args()
    if hasattr(signal, 'SIGUSR1'): signal.signal(signal.SIGUSR1, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    if args.action == 'preflight':
        native_preflight(args.key, args.root / 'preflight' / f'{args.key}.txt', args.local_base)
    elif args.action == 'freeze': snapshot_and_freeze(args.key, args.root, args.cap, args.local_base)
    else: return run_worker(args.key, args.root, args.part, local_base=args.local_base)
    return 0


if __name__ == '__main__': raise SystemExit(main())
