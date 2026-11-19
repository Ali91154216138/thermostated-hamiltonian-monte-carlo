"""Full-grid, block-resumable 6D funnel tuning with disjoint parallel parts.

Original numerical kernels, diagnostics and strict nonfinite score rule remain
unchanged. Every setting receives ALL ten full blocks. No held-out runs.
Only load the checksum-protected pickle checkpoints produced by this campaign.
"""
from __future__ import annotations
import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import io
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
TERMINATED = False
KEYS = {'periodic':('alpha','phi','a','Q','epsilon','L'), 'sqhmc':('mu_m','sigma_m','epsilon','L')}

def request_stop(signum, frame):
    global STOP_REQUESTED, TERMINATED
    STOP_REQUESTED = True
    if signum == signal.SIGTERM: TERMINATED = True
    print('Stop notice: finish and save the current block, then stop.', flush=True)

def stop_code(): return 143 if TERMINATED else 75

def atomic_text(path, text):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as out:
            out.write(text); out.flush(); os.fsync(out.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)

def jsonable(v):
    if dataclasses.is_dataclass(v): return jsonable(dataclasses.asdict(v))
    if isinstance(v,dict): return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
    if hasattr(v,'tolist'): return jsonable(v.tolist())
    if isinstance(v,Path): return str(v)
    if isinstance(v,float) and not math.isfinite(v): return str(v)
    return v

def encode(v): return json.dumps(jsonable(v),sort_keys=True,separators=(',',':'),allow_nan=False)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

@contextlib.contextmanager
def file_lock(path, blocking=False):
    f=Path(path).open('a+')
    locked=False
    try:
        if os.name=='posix':
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        else:
            import msvcrt
            if f.tell()==0: f.write('0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(),msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK,1)
        locked=True
        yield
    finally:
        if locked and os.name!='posix':
            f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        f.close()

def load_tuner(source, method):
    sys.path.insert(0,str(Path(source).resolve().parent))
    name='_expanded_funnel_'+method
    spec=importlib.util.spec_from_file_location(name,source)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

class Grid:
    """Indexable Cartesian product without allocating 1.3 million dictionaries."""
    def __init__(self,keys,values):
        self.keys=tuple(keys); self.values=tuple(tuple(v) for v in values)
        self.size=math.prod(map(len,self.values))
    def __len__(self): return self.size
    def __getitem__(self,i):
        if not 0<=i<self.size: raise IndexError(i)
        result={}
        for k,v in reversed(tuple(zip(self.keys,self.values))):
            i,j=divmod(i,len(v)); result[k]=v[j]
        return {k:result[k] for k in self.keys}

def common_inputs(p,method): return p.common

def configurations(p,method):
    c=common_inputs(p,method)
    keys=KEYS[method]
    return Grid(keys,[getattr(c if k in ('epsilon','L') else p,
               ('lambda' if k=='lambda_value' else k)+'_values') for k in keys])

class Checkpoints:
    def __init__(self,path,manifest):
        self.db=sqlite3.connect(path,timeout=60)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS blocks(config INTEGER,block INTEGER,digest TEXT NOT NULL,payload BLOB NOT NULL,PRIMARY KEY(config,block))')
        old=self.db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if old and old[0]!=encode(manifest):
            self.db.close(); raise RuntimeError('Checkpoint inputs, source, runtime or part layout changed. Use a NEW campaign folder.')
        if not old:
            with self.db: self.db.execute('INSERT INTO metadata VALUES (?,?)',('manifest',encode(manifest)))
    def contains(self,i,b):
        return self.db.execute('SELECT 1 FROM blocks WHERE config=? AND block=?',(i,b)).fetchone() is not None
    def get(self,i,b):
        row=self.db.execute('SELECT digest,payload FROM blocks WHERE config=? AND block=?',(i,b)).fetchone()
        if row is None or hashlib.sha256(row[1]).hexdigest()!=row[0]:
            raise RuntimeError(f'Missing/corrupt block {i},{b}')
        return pickle.loads(zlib.decompress(row[1]))
    def put(self,i,b,v):
        blob=zlib.compress(pickle.dumps(v,protocol=5),1)
        with self.db:
            self.db.execute('INSERT INTO blocks VALUES (?,?,?,?)',(i,b,hashlib.sha256(blob).hexdigest(),blob))
    def count(self): return self.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]
    def validate_indices(self,nconfig,nblocks,part,parts):
        bad=self.db.execute('SELECT COUNT(*) FROM blocks WHERE config<0 OR config>=? OR block<0 OR block>=? OR config % ? != ?',
            (nconfig,nblocks,parts,part)).fetchone()[0]
        if bad: raise RuntimeError('Checkpoint contains out-of-grid or wrong-part records')
    def close(self): self.db.close()

def physical(b,method): return b.physical

def aggregate(mod,method,settings,blocks):
    fn=mod._aggregate_configuration if method=='sqhmc' else mod._aggregate_thc_configuration
    return fn(**settings,block_results=blocks)

def select_pair(mod,method,best,new):
    if new.status!='ok': return best
    if best is None: return new
    fn=mod._select_representative if method=='sqhmc' else mod._select_thc_representative
    return fn((best,new))

def iat_stats(result,method):
    v=[physical(b,method).worst_iat for b in result.block_results]
    if not all(math.isfinite(x) and x>0 for x in v): return math.inf,math.inf
    return max(v),math.exp(math.fsum(math.log(x) for x in v)/len(v))

def iat_key(result,method,index):
    worst,gm=iat_stats(result,method)
    return worst,gm,-result.geometric_mean_efficiency,index

def block_summary(wrapped,method):
    b=physical(wrapped,method)
    worst=next((d for d in b.observable_diagnostics if d.name==b.worst_observable),None)
    return dict(block=b.block_index+1,seeds=b.seeds,status=b.status,
        worst_observable=b.worst_observable,worst_IAT=b.worst_iat,
        efficiency=b.block_efficiency,acceptance=b.acceptance,
        retained_gradient_calls=b.retained_gradient_calls,gradients_per_draw=b.gradients_per_retained_draw,
        maximum_Rhat=b.maximum_rhat,maximum_moment_error=b.maximum_standardized_moment_error,
        nonfinite_warmup=b.warmup_nonfinite_trajectories,nonfinite_retained=b.retained_nonfinite_trajectories,
        Geyer_worst=dataclasses.asdict(worst.iat) if worst else None)

def result_summary(r,method,index):
    fields=('status','geometric_mean_efficiency','minimum_block_efficiency','maximum_block_efficiency',
        'mean_acceptance','maximum_rhat','maximum_standardized_moment_error','total_nonfinite_trajectories')
    worst,gm=iat_stats(r,method)
    return dict(global_configuration_index=index,settings={k:getattr(r,k) for k in KEYS[method]},
        **{k:getattr(r,k) for k in fields},max_block_worst_IAT=worst,geometric_mean_worst_IAT=gm,
        blocks=[block_summary(b,method) for b in r.block_results])

def pack(r,index,method):
    if r is None: return None
    return dict(summary=jsonable(result_summary(r,method,index)),
        digest=hashlib.sha256(pickle.dumps(r,protocol=5)).hexdigest(),
        payload=zlib.compress(pickle.dumps(r,protocol=5),1).hex())

def unpack(p):
    if p is None: return None
    blob=zlib.decompress(bytes.fromhex(p['payload']))
    if hashlib.sha256(blob).hexdigest()!=p['digest']: raise RuntimeError('Representative checksum mismatch')
    return pickle.loads(blob)

def header(method,c,grid,manifest):
    return '\n'.join([
        f'EXPANDED 6D NEALS FUNNEL: {method.upper()} - TUNING ONLY',
        f'{len(grid)} settings; {len(c.seed_blocks)} common blocks; {c.n_chains} chains per block.',
        f'Per chain and setting: {c.n_warmup} warmup + {c.n_retained} retained draws.',
        'Full Cartesian evaluation. No halving, short screening, pruning, held-out sampling or adaptive expansion.',
        'Same beta=1 target, M0=I6, physical starts, 27 physical observables and physical random streams.',
        'Unequal numbers of settings: equal per-setting budgets, NOT equal total tuning compute.',
        'Primary: geometric mean of all ten block efficiencies, original method-specific tie rules.',
        'Efficiency = 1 / (actual retained new target gradients per draw * worst-observable IAT).',
        'Secondary: smallest MAXIMUM blockwise worst IAT among score-defined settings; ties: GM IAT, higher efficiency, lower global index.',
        'Secondary choice is separately labelled. Neither tuning choice proves a held-out win.',
        'Original strict score rule retained for all four comparison methods: any nonfinite proposal (warmup or retained),',
        'undefined required IAT or invalid gradient cost makes the full efficiency score undefined.',
        'Nonfinite proposals are rejected and previous valid states are retained. All failure counts and descriptive IATs are kept.',
        'No block is discarded, re-seeded, replaced or omitted; no Rhat or moment-error eligibility threshold.',
        'IAT: original all-chain within/between variance Geyer IPS/IMS, autocovariances divided by N.',
        'Rhat: maximum of rank-normalized split and folded split Rhat. All exact-moment errors retained.',
        'Cache endpoint gradients once. Only actual target-gradient calls after warmup are charged. Gradient cost is not runtime.',
        'Periodic: h=1+alpha*cos(s/a-phi), rho=N(0,1), fixed s0=(-1,1,1,-1); s persists.',
        'S-QHMC: omega~N(mu_m,sigma_m^2); m=10^omega fixed within each trajectory; mass refreshed each transition.',
        'Periodic THC refreshes momenta only. Exact P/S/K/S/P subflows, full momentum reversal and full-state Metropolis.',
        'PHYSICAL STARTS: '+encode(c.physical_starts),
        'OBSERVABLES: '+encode(c.observable_names),
        'EXACT MEANS: '+encode(c.exact_observable_means),
        'SOURCE, INPUTS, SEEDS AND RUNTIME: '+encode(manifest),''])

def write_part_report(store,mod,method,grid,c,manifest,part,parts,folder):
    fd,name=tempfile.mkstemp(prefix='part-report.',dir=folder)
    best=None; best_i=None; low=None; low_i=None; defined=0
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as out:
            out.write(header(method,c,grid,manifest))
            out.write(f'PART {part+1}/{parts}; global indices are zero-based.\n')
            for i in range(part,len(grid),parts):
                if STOP_REQUESTED: raise InterruptedError()
                r=aggregate(mod,method,grid[i],[store.get(i,b) for b in range(len(c.seed_blocks))])
                chosen=select_pair(mod,method,best,r)
                if chosen is r: best,best_i=r,i
                if r.status=='ok':
                    defined+=1
                    if low is None or iat_key(r,method,i)<iat_key(low,method,low_i): low,low_i=r,i
                out.write(encode(result_summary(r,method,i))+'\n')
            for label,r,i in [('BEST EFFICIENCY',best,best_i),('LOWEST MAXIMUM WORST IAT',low,low_i)]:
                out.write(label+'\n'+encode(result_summary(r,method,i) if r else None)+'\n')
                if r:
                    out.write('ALL OBSERVABLES FOR THIS REPRESENTATIVE\n')
                    for b in r.block_results: out.write(encode(b)+'\n')
            out.write(f'DEFINED/TESTED IN PART: {defined}/{len(range(part,len(grid),parts))}\nEND OF PART REPORT\n')
            out.flush(); os.fsync(out.fileno())
        destination=folder/'configuration_results.txt'; os.replace(name,destination)
    finally:
        if os.path.exists(name): os.unlink(name)
    return dict(part=part,parts=parts,settings=len(range(part,len(grid),parts)),defined=defined,
        blocks=store.count(),report_sha256=sha(destination),manifest_sha256=hashlib.sha256(encode(manifest).encode()).hexdigest(),
        best_efficiency=pack(best,best_i,method),lowest_worst_IAT=pack(low,low_i,method))

def finalize(mod,method,grid,c,manifest,run_dir,parts):
    markers=[run_dir/f'parts/part_{p:03d}/results/COMPLETED.txt' for p in range(parts)]
    if not all(p.is_file() for p in markers): return False
    with file_lock(run_dir/'finalize.lock',blocking=True):
        result_dir=run_dir/'results'; result_dir.mkdir(exist_ok=True)
        destination=result_dir/c.output_filename
        done=result_dir/'COMPLETED.txt'
        if done.exists():
            stamp=json.loads(done.read_text())
            if sha(destination)!=stamp['report_sha256']: raise RuntimeError('Final report checksum mismatch')
            return True
        best=None; best_i=None; low=None; low_i=None; defined=0; total=0; records=[]
        expected=hashlib.sha256(encode(manifest).encode()).hexdigest()
        for p,marker in enumerate(markers):
            r=json.loads(marker.read_text(encoding='utf-8'))
            if (r['manifest_sha256']!=expected or r['part']!=p or r['parts']!=parts or
                r['settings']!=len(range(p,len(grid),parts)) or r['blocks']!=r['settings']*len(c.seed_blocks)):
                raise RuntimeError('Part identity, manifest or count mismatch')
            report=marker.parent/'configuration_results.txt'
            if sha(report)!=r['report_sha256']: raise RuntimeError('Part report checksum mismatch')
            defined+=r['defined']; total+=r['settings']
            candidate=unpack(r['best_efficiency'])
            if candidate is not None:
                chosen=select_pair(mod,method,best,candidate)
                if chosen is candidate:
                    best=candidate; best_i=r['best_efficiency']['summary']['global_configuration_index']
            candidate=unpack(r['lowest_worst_IAT'])
            if candidate is not None:
                i=r['lowest_worst_IAT']['summary']['global_configuration_index']
                if low is None or iat_key(candidate,method,i)<iat_key(low,method,low_i): low,low_i=candidate,i
            records.append(dict(part=p,settings=r['settings'],defined=r['defined'],report=str(report),sha256=r['report_sha256']))
        if total!=len(grid): raise RuntimeError('Final grid coverage mismatch')
        text=header(method,c,grid,manifest)+f'ALL PARTS COMPLETE. DEFINED/TESTED: {defined}/{total}\n'
        for label,r,i in [('BEST EFFICIENCY',best,best_i),('LOWEST MAXIMUM WORST IAT',low,low_i)]:
            text+=label+'\n'+json.dumps(jsonable(result_summary(r,method,i) if r else None),indent=2)+'\n'
            if r:
                text+='ALL OBSERVABLES, EXACT MOMENTS, RHAT AND IAT CUTOFFS\n'
                text+='\n'.join(encode(b) for b in r.block_results)+'\n'
                for k,values in zip(grid.keys,grid.values):
                    if getattr(r,k) in (min(values),max(values)):
                        text+=f'GRID BOUNDARY: {k}={getattr(r,k)}. Best tested value, not a proven optimum.\n'
        text+='COMPLETE CONFIGURATION REPORTS (ALL PARTS; no missing or discarded blocks)\n'+json.dumps(records,indent=2)+'\nEND OF REPORT\n'
        atomic_text(destination,text)
        atomic_text(done,encode(dict(method=method,settings=total,blocks=total*len(c.seed_blocks),
            report_sha256=sha(destination),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))+'\n')
        print(text,flush=True)
        print(f'ALL PARTS COMPLETE: {destination}',flush=True)
        return True

def campaign(mod,method,source,run_dir,part=0,parts=1,inputs=None,stop_after_blocks=None):
    run_dir=Path(run_dir).resolve()
    folder=run_dir/f'parts/part_{part:03d}'
    for p in (run_dir/'results',folder/'results',folder/'checkpoints',folder/'logs'): p.mkdir(parents=True,exist_ok=True)
    inputs=dataclasses.replace(inputs or mod.USER_INPUTS,output_directory=str(run_dir/'results'))
    manifest=dict(format='expanded-funnel-parts-v1',method=method,parts=parts,source_sha256=sha(source),
        protocol_sha256=sha(Path(source).parent/'expanded_protocol.py'),launcher_sha256=sha(__file__),
        code_sha256={p.name:sha(p) for p in sorted(Path(source).parent.glob('*.py'))},
        inputs=jsonable(inputs),python=platform.python_version(),numpy=mod.np.__version__,
        numba=importlib.metadata.version('numba'),llvmlite=importlib.metadata.version('llvmlite'))
    with file_lock(folder/'checkpoints/campaign.lock'):
        store=Checkpoints(folder/'checkpoints/blocks.sqlite3',dict(common=manifest,part=part))
        try:
            driver=mod.run_s_qhmc_tuning if method=='sqhmc' else mod.run_thc_tuning
            with contextlib.redirect_stdout(io.StringIO()): preflight=driver(inputs,preflight_only=True)
            p=preflight['prepared_inputs']; c=common_inputs(p,method); grid=configurations(p,method)
            if not (1<=parts<=len(grid) and 0<=part<parts): raise ValueError('Invalid part layout')
            if not mod.NUMBA_AVAILABLE: raise RuntimeError('Production requires Numba')
            tests=preflight.get('self_test_messages',preflight.get('self_tests'))
            if not tests: raise RuntimeError('Preflight did not return verified test results')
            if 'native_runtime' in preflight: tests=(*tests,preflight['native_runtime'])
            if 'preflight_message' in preflight: tests=(*tests,preflight['preflight_message'])
            atomic_text(folder/'PREFLIGHT.txt','\n'.join(tests)+'\n')
            for t in tests: print(t,flush=True)
            atomic_text(folder/'run_manifest.txt',encode(manifest)+'\n')
            store.validate_indices(len(grid),len(c.seed_blocks),part,parts)
            total=len(range(part,len(grid),parts))*len(c.seed_blocks)
            saved=store.count()
            print(f'PREFLIGHT PASSED: {method} part {part+1}/{parts}; {saved}/{total} full blocks saved.',flush=True)
            marker=folder/'results/COMPLETED.txt'
            if marker.exists():
                record=json.loads(marker.read_text())
                if sha(folder/'results/configuration_results.txt')!=record['report_sha256']:
                    raise RuntimeError('Part report checksum mismatch')
                if saved!=total: raise RuntimeError('Completed part has missing checkpoint blocks')
                finalize(mod,method,grid,c,manifest,run_dir,parts); return 0
            cache=tuple(tuple(mod._make_random_streams(seed,c.n_warmup+c.n_retained,c.dimension)
                for seed in block) for block in c.seed_blocks)
            new=0; started=time.monotonic()
            for i in range(part,len(grid),parts):
                settings=grid[i]
                for b,seeds in enumerate(c.seed_blocks):
                    if store.contains(i,b): continue
                    if STOP_REQUESTED or (stop_after_blocks is not None and new>=stop_after_blocks): return stop_code()
                    value=mod._run_configuration_block(**settings,block_index=b,seeds=seeds,stream_cache=cache[b],prepared=p)
                    store.put(i,b,value); saved+=1; new+=1
                    summary=block_summary(value,method)
                    elapsed=time.monotonic()-started
                    progress=dict(method=method,part=part,parts=parts,saved_blocks=saved,total_part_blocks=total,
                        global_configuration_index=i,settings=settings,last_block=summary,
                        session_elapsed_seconds=elapsed,session_blocks=new,
                        session_seconds_per_block=elapsed/new)
                    atomic_text(folder/'results/PROGRESS.txt',json.dumps(jsonable(progress),indent=2)+'\n')
                    print(f'{method} part {part+1}: saved {saved}/{total}; config {i}; block {b+1}; '
                        f'worst IAT={summary["worst_IAT"]:.7g}; Rhat={summary["maximum_Rhat"]:.7g}; '
                        f'efficiency={summary["efficiency"]:.7g}; status={summary["status"]}',flush=True)
            if store.count()!=total: raise RuntimeError('Incomplete checkpoint')
            if STOP_REQUESTED: return stop_code()
            try:
                record=write_part_report(store,mod,method,grid,c,manifest,part,parts,folder/'results')
            except InterruptedError: return stop_code()
            atomic_text(marker,encode(record)+'\n')
            print(f'PART COMPLETE: {folder}',flush=True)
            finalize(mod,method,grid,c,manifest,run_dir,parts)
            return 0
        finally: store.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=tuple(KEYS),required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--part',type=int,default=0)
    parser.add_argument('--parts',type=int,default=1)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    mod=load_tuner(args.source.resolve(),args.method)
    if args.preflight_only:
        driver=mod.run_s_qhmc_tuning if args.method=='sqhmc' else mod.run_thc_tuning
        driver(mod.USER_INPUTS,preflight_only=True); return 0
    return campaign(mod,args.method,args.source.resolve(),args.run_dir,args.part,args.parts)

if __name__=='__main__': raise SystemExit(main())
