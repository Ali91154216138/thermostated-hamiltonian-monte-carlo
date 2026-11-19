"""Run tuning jobs on the five-dimensional multiscale Gaussian with checkpoints."""
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


def write_report(store,module,settings_list,prepared,tests,manifest,destination):
    results=[]
    for i,settings in enumerate(settings_list):
        if STOP_REQUESTED: raise InterruptedError('Reporting paused; saved blocks are intact.')
        blocks=[store.get(i,b) for b in range(len(prepared.seed_blocks))]
        results.append(module._aggregate_configuration(settings['target_acceptance'],blocks))
    best=module._select_representative(results) if any(r.status=='ok' for r in results) else None
    report=module._build_report(configuration_results=results,representative=best,
                                prepared=prepared,self_test_messages=tests)
    report+='\n\nFULL INPUTS AND SOURCE/RUNTIME CHECKSUMS\n'+json.dumps(manifest,sort_keys=True,indent=2)
    report+='\n\nDefined/tested settings: '+str(sum(r.status=='ok' for r in results))+'/'+str(len(results))
    report+='\nEqual per-setting run budgets do not imply equal total tuning compute.\n'
    if best is not None and best.target_acceptance in (prepared.target_acceptance_values[0],prepared.target_acceptance_values[-1]):
        report+='Selected delta is on a grid boundary: best tested point, not a proven optimum.\n'
    atomic_text(destination,report+'\n')
    return best


def campaign(module,method,source,run_dir,inputs=None,stop_after_blocks=None):
    """Optional short inputs/stop count support isolated tests, not screening."""
    if method != 'nuts': raise ValueError('This launcher is for NUTS only.')
    if not module.NUMBA_AVAILABLE:
        raise RuntimeError('Production requires Numba; refusing a silent slow fallback.')
    run_dir=Path(run_dir).resolve()
    for folder in ('results','checkpoints','logs'): (run_dir/folder).mkdir(parents=True,exist_ok=True)
    inputs=dataclasses.replace(inputs or module.USER_INPUTS,output_directory=str(run_dir/'results'))
    destination=run_dir/'results'/inputs.output_filename
    completed_path=run_dir/'results/COMPLETED.txt'
    manifest={'format':'expanded-NUTS-full-grid-blocks-v1','method':method,
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
        driver=module.run_nuts_tuning
        preflight=driver(inputs,preflight_only=True)
        prepared=preflight['prepared_inputs']
        common=prepared
        tests=preflight.get('self_test_messages',preflight.get('self_tests',()))
        settings_list=[dict(target_acceptance=d) for d in prepared.target_acceptance_values]
        n_blocks=len(common.seed_blocks); total_blocks=len(settings_list)*n_blocks
        saved=store.count()
        if saved>total_blocks: raise RuntimeError('Checkpoint larger than declared grid')
        print(f'{method}: {len(settings_list)} settings; {saved}/{total_blocks} blocks saved.',flush=True)
        new=0; start=time.monotonic()
        for i,settings in enumerate(settings_list):
            for b,seeds in enumerate(common.seed_blocks):
                if store.contains(i,b): continue
                if STOP_REQUESTED or (stop_after_blocks is not None and new>=stop_after_blocks):
                    print(f'Saved for resume: {saved}/{total_blocks} blocks.',flush=True); return 75
                value=module._run_configuration_block(**settings,block_index=b,seeds=seeds,prepared=prepared)
                store.put(i,b,value); saved+=1; new+=1
                progress=f'{method}: saved {saved}/{total_blocks} blocks; setting {i+1}/{len(settings_list)}, block {b+1}/{n_blocks}; session {time.monotonic()-start:.1f}s'
                atomic_text(run_dir/'results/PROGRESS.txt',progress+'\n'+json.dumps(settings,sort_keys=True)+'\n')
                print(progress,flush=True)
        if store.count()!=total_blocks: raise RuntimeError('Incomplete checkpoint: no final report written')
        if STOP_REQUESTED:
            print('All blocks saved; resume to generate the full report.',flush=True); return 75
        try:
            best=write_report(store,module,settings_list,prepared,tests,manifest,destination)
        except InterruptedError:
            print('All blocks saved; report generation will restart on resume.',flush=True)
            return 75
        complete={'method':method,'settings':len(settings_list),'blocks':total_blocks,
                  'report':str(destination),'report_sha256':file_sha256(destination),
                  'selected_settings':{'target_acceptance':best.target_acceptance} if best else None,
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
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    return campaign(load_tuner(args.source.resolve(),'nuts'),'nuts',args.source.resolve(),args.run_dir)


if __name__=='__main__': raise SystemExit(main())

