"""Block-resumable full-grid tuner for expanded Gaussian mixture THC/S-QHMC.

Does not change either numerical kernel or diagnostic/selection formulas.
No screening: every configuration is evaluated on every declared block.
Completed block records are trusted local pickle data protected by SHA256.
Results are read and reported one configuration at a time to bound memory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import platform
import signal
import sqlite3
import sys
import tempfile
import time
import zlib

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print('Stop notice received; saving the current block before stopping.', flush=True)


def atomic_text(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def jsonable(value):
    if dataclasses.is_dataclass(value): return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'): return jsonable(value.tolist())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    return value


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def load_tuner(source, method):
    name='_expanded_gaussian_'+method
    spec=importlib.util.spec_from_file_location(name,source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class Checkpoints:
    def __init__(self,path,manifest):
        self.db=sqlite3.connect(path,timeout=60)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS blocks (config INTEGER,block INTEGER,digest TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(config,block))')
        encoded=json.dumps(manifest,sort_keys=True,separators=(',',':'))
        old=self.db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if old and old[0]!=encoded:
            self.db.close()
            raise RuntimeError('Checkpoint inputs, source, launcher, or numerical runtime differ. Use a new run folder.')
        if not old:
            with self.db: self.db.execute('INSERT INTO metadata VALUES (?,?)',('manifest',encoded))

    def contains(self,config,block):
        return self.db.execute('SELECT 1 FROM blocks WHERE config=? AND block=?',(config,block)).fetchone() is not None

    def get(self,config,block):
        row=self.db.execute('SELECT digest,payload FROM blocks WHERE config=? AND block=?',(config,block)).fetchone()
        if row is None: raise RuntimeError(f'Missing completed block {config},{block}')
        if hashlib.sha256(row[1]).hexdigest()!=row[0]: raise RuntimeError('Block checksum mismatch')
        return pickle.loads(zlib.decompress(row[1]))

    def put(self,config,block,value):
        payload=zlib.compress(pickle.dumps(value,protocol=5),level=1)
        with self.db:
            self.db.execute('INSERT INTO blocks VALUES (?,?,?,?)',(config,block,hashlib.sha256(payload).hexdigest(),payload))

    def count(self): return self.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]
    def close(self): self.db.close()


def configurations(module,prepared,method):
    if method=='mixture':
        return [dataclasses.asdict(c) for c in module._base_configurations(prepared)]
    return [dict(mu_m=m,sigma_m=s,epsilon=e,L=L) for m,s,e,L in itertools.product(
        prepared.mu_m_values,prepared.sigma_m_values,prepared.common.epsilon_values,prepared.common.L_values)]


def aggregate(module,method,settings,blocks):
    if method=='mixture':
        return module._aggregate_configuration(module.Configuration(**settings),'EXPANDED_FULL_GRID',blocks)
    return module._aggregate_configuration(settings['mu_m'],settings['sigma_m'],settings['epsilon'],settings['L'],blocks)


def select_pair(module,method,best,candidate):
    if candidate.status!='ok': return best
    if best is None: return candidate
    select=module._select_best if method=='mixture' else module._select_representative
    return select((best,candidate))


def settings_of(result,method):
    if method=='mixture': return dataclasses.asdict(result.configuration)
    return {k:getattr(result,k) for k in ('mu_m','sigma_m','epsilon','L')}


def result_summary(result,method):
    fields=('status','geometric_mean_efficiency','minimum_block_efficiency','maximum_block_efficiency',
            'mean_acceptance','maximum_rhat','maximum_standardized_moment_error','total_nonfinite_trajectories')
    summary={'settings':settings_of(result,method),**{k:getattr(result,k) for k in fields}}
    summary['blocks']=[]
    for wrapped in result.block_results:
        b=wrapped.physical
        worst=next((d for d in b.observable_diagnostics if d.name==b.worst_observable),None)
        summary['blocks'].append({
            'block':b.block_index+1,'seeds':b.seeds,'status':b.status,'worst_observable':b.worst_observable,
            'worst_IAT':b.worst_iat,'efficiency':b.block_efficiency,'acceptance':b.acceptance,
            'retained_gradient_calls':b.retained_gradient_calls,'gradients_per_draw':b.gradients_per_retained_draw,
            'maximum_Rhat':b.maximum_rhat,'maximum_moment_error':b.maximum_standardized_moment_error,
            'nonfinite_warmup':b.warmup_nonfinite_trajectories,'nonfinite_retained':b.retained_nonfinite_trajectories,
            'Geyer_worst':dataclasses.asdict(worst.iat) if worst else None})
    return jsonable(summary)


def write_report(store,module,method,settings_list,prepared,tests,manifest,destination):
    """Atomic, streaming TXT report; retains all block summaries including failures."""
    common=prepared if method=='mixture' else prepared.common
    fd,tempname=tempfile.mkstemp(prefix='full-report.',suffix='.tmp',dir=destination.parent)
    best=None
    defined=0
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as out:
            def line(value=''): out.write(str(value)+'\n')
            line(f'EXPANDED 5D MULTISCALE GAUSSIAN: {method.upper()} TUNING')
            line('='*88)
            line('Full Cartesian search. No short screening, halving, or automatic boundary expansion.')
            line('Tuning only: these are not held-out results or evidence of a final win.')
            line(f'{len(settings_list)} settings; {len(common.seed_blocks)} blocks; {common.n_chains} chains per block.')
            line(f'Each chain: {common.n_warmup} warmup + {common.n_retained} retained draws at every setting.')
            line('Same target, base mass, physical starts, observables and tuning seeds across methods.')
            line('The number of settings differs: equal per-setting evaluation, not equal total tuning compute.')
            line('IAT: all chains in a block; biased autocovariances with denominator N at every lag;')
            line('Geyer initial-positive pairs, cumulative minima; no floor or manual lag cutoff.')
            line('R-hat: rank-normalized split and folded split maximum. Moment errors and R-hat are descriptive, not gates.')
            line('Block efficiency=1/(actual retained target gradients per draw * worst physical-observable IAT).')
            line('Score=geometric mean of every block efficiency. Undefined IAT or any nonfinite trajectory makes the score undefined.')
            line('Cost counts newly executed retained target gradients, excludes warmup, and does not double-count cached gradients.')
            line('Valid full trajectories cost L new target gradients per draw; inner THC substeps call no target gradient.')
            line('Gradient cost is not wall-clock cost.')
            if method=='mixture':
                line('h(s)=exp(s); s~three-component Gaussian mixture, p|s and p_s are refreshed at every trajectory.')
                line('H=U+exp(-2*s)*p.T*inv(M)*p/2+d*s/beta+p_s**2/(2*Q)-log(rho_delta(s))/beta.')
                line(f'Inner exact B/W substeps={prepared.inner_r}; mixture weights={prepared.mixture_weights.tolist()}; centers={prepared.mixture_centers.tolist()}.')
                line('Exact score ties: larger minimum block efficiency, lower gradient cost, then smaller epsilon, L, delta, Q.')
            else:
                line('At each trajectory: a~N(mu_m,sigma_m^2), mass factor=10**a, p~N(0,mass_factor*M0/beta).')
                line('The mass stays fixed along the entire leapfrog path; Metropolis uses that same mass.')
                line('Exact score ties: larger minimum block efficiency, fewer gradients, then smaller epsilon, L, mu_m, sigma_m.')
            line('FULL INPUTS AND SOURCE/RUNTIME CHECKSUMS')
            line(json.dumps(manifest,indent=2,sort_keys=True))
            line('PHYSICAL STARTS'); line(common.physical_starts.tolist())
            line('OBSERVABLES AND EXACT MEANS'); line(common.observable_names); line(common.exact_observable_means.tolist())
            line('SELF TESTS'); line('\n'.join(tests))
            line('ALL CONFIGURATION AND BLOCK SUMMARIES (labelled records; nan/inf explicitly shown)')
            for i,settings in enumerate(settings_list):
                if STOP_REQUESTED: raise InterruptedError('Report generation paused; all completed blocks remain saved.')
                result=aggregate(module,method,settings,[store.get(i,b) for b in range(len(common.seed_blocks))])
                defined+=int(result.status=='ok')
                best=select_pair(module,method,best,result)
                line(json.dumps(result_summary(result,method),sort_keys=True,separators=(',',':')))
            line(f'DEFINED/TESTED: {defined}/{len(settings_list)}')
            line('SELECTED FIXED REPRESENTATIVE')
            if best is None:
                line('NONE: all full-block scores undefined. All failures are retained above.')
            else:
                line(json.dumps(result_summary(best,method),sort_keys=True,indent=2))
                for key,value in settings_of(best,method).items():
                    grid_owner=common if key in ('epsilon','L') else prepared
                    grid=getattr(grid_owner,key+'_values')
                    if value==grid[0] or value==grid[-1]: line(f'Grid boundary: {key}={value}. Best tested setting, not a proven optimum.')
                line('SELECTED REPRESENTATIVE: ALL OBSERVABLES, MOMENTS, RHAT COMPONENTS AND GEYER CUTOFFS')
                for block in best.block_results: line(json.dumps(jsonable(block),sort_keys=True,indent=2))
            line('END OF REPORT')
            out.flush(); os.fsync(out.fileno())
        os.replace(tempname,destination)
    finally:
        if os.path.exists(tempname): os.unlink(tempname)
    return best


def campaign(module,method,source,run_dir,inputs=None,stop_after_blocks=None):
    """Optional short inputs/stop count support isolated tests, not screening."""
    run_dir=Path(run_dir).resolve()
    for folder in ('results','checkpoints','logs'): (run_dir/folder).mkdir(parents=True,exist_ok=True)
    inputs=dataclasses.replace(inputs or module.USER_INPUTS,output_directory=str(run_dir/'results'))
    destination=run_dir/'results'/inputs.output_filename
    completed_path=run_dir/'results/COMPLETED.txt'
    manifest={'format':'expanded-mixture-SQHMC-full-grid-blocks-v1','method':method,
              'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs':jsonable(inputs),'python':platform.python_version(),'numpy':module.np.__version__,
              'numba':importlib.metadata.version('numba'),'llvmlite':importlib.metadata.version('llvmlite')}
    lock=(run_dir/'checkpoints/campaign.lock').open('a+')
    store=None
    try:
        if os.name=='posix':
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            import msvcrt
            if lock.tell()==0: lock.write('0'); lock.flush()
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        store=Checkpoints(run_dir/'checkpoints/blocks.sqlite3',manifest)
        if completed_path.exists():
            complete=json.loads(completed_path.read_text(encoding='utf-8'))
            if not destination.is_file() or file_sha256(destination)!=complete['report_sha256']:
                raise RuntimeError('Completed report checksum mismatch')
            print(f'Already complete: {destination}',flush=True); return 0
        if destination.exists():
            raise RuntimeError('Report exists without a completion marker. Preserve it and verify before restarting.')
        atomic_text(run_dir/'run_manifest.txt',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        driver=module.run_mixture_thc_tuning if method=='mixture' else module.run_s_qhmc_tuning
        preflight=driver(inputs,preflight_only=True)
        prepared=preflight['prepared_inputs']
        common=prepared if method=='mixture' else prepared.common
        tests=preflight.get('self_test_messages',preflight.get('self_tests',()))
        settings_list=configurations(module,prepared,method)
        n_blocks=len(common.seed_blocks); total_blocks=len(settings_list)*n_blocks
        saved=store.count()
        if saved>total_blocks: raise RuntimeError('Checkpoint larger than declared grid')
        print(f'{method}: {len(settings_list)} settings; {saved}/{total_blocks} blocks saved.',flush=True)
        cache=tuple(tuple(module._make_random_streams(seed,common.n_warmup+common.n_retained,common.dimension)
                          for seed in block) for block in common.seed_blocks)
        new=0; start=time.monotonic()
        for i,settings in enumerate(settings_list):
            for b,seeds in enumerate(common.seed_blocks):
                if store.contains(i,b): continue
                if STOP_REQUESTED or (stop_after_blocks is not None and new>=stop_after_blocks):
                    print(f'Saved for resume: {saved}/{total_blocks} blocks.',flush=True); return 75
                if method=='mixture':
                    value=module._run_configuration_block(configuration=module.Configuration(**settings),origin='EXPANDED_FULL_GRID',
                        block_index=b,prepared=prepared,cache=cache,n_warmup=common.n_warmup,n_retained=common.n_retained)
                else:
                    value=module._run_configuration_block(**settings,block_index=b,seeds=seeds,stream_cache=cache[b],prepared=prepared)
                store.put(i,b,value); saved+=1; new+=1
                progress=f'{method}: saved {saved}/{total_blocks} blocks; setting {i+1}/{len(settings_list)}, block {b+1}/{n_blocks}; session {time.monotonic()-start:.1f}s'
                atomic_text(run_dir/'results/PROGRESS.txt',progress+'\n'+json.dumps(settings,sort_keys=True)+'\n')
                print(progress,flush=True)
        if store.count()!=total_blocks: raise RuntimeError('Incomplete checkpoint: no final report written')
        if STOP_REQUESTED:
            print('All blocks saved; resume to generate the full report.',flush=True); return 75
        try:
            best=write_report(store,module,method,settings_list,prepared,tests,manifest,destination)
        except InterruptedError:
            print('All blocks saved; report generation will restart on resume.',flush=True)
            return 75
        complete={'method':method,'settings':len(settings_list),'blocks':total_blocks,
                  'report':str(destination),'report_sha256':file_sha256(destination),
                  'selected_settings':settings_of(best,method) if best else None,
                  'selected_score':best.geometric_mean_efficiency if best else None,
                  'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        atomic_text(completed_path,json.dumps(complete,sort_keys=True,indent=2)+'\n')
        if common.print_report:
            with destination.open(encoding='utf-8') as report:
                for line in report: print(line,end='')
        print(f'COMPLETE: {destination}',flush=True)
        return 0
    finally:
        if store is not None: store.close()
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('mixture','sqhmc'),required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    return campaign(load_tuner(args.source.resolve(),args.method),args.method,args.source.resolve(),args.run_dir)


if __name__=='__main__': raise SystemExit(main())
