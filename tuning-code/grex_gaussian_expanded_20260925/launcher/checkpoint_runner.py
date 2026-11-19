"""Run tuning jobs on the five-dimensional multiscale Gaussian with checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import importlib.metadata
import itertools
import json
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
    print("Time-limit notice: finishing and saving the current block.", flush=True)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_tuner(source, method):
    name = "_grex_frozen_" + method
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Checkpoints:
    def __init__(self, path, manifest):
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS blocks (config INTEGER, block INTEGER, digest TEXT NOT NULL, payload BLOB NOT NULL, PRIMARY KEY(config, block))")
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        old = self.db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if old and old[0] != encoded:
            raise RuntimeError("Checkpoint source, inputs, or numerical runtime differs. Use a new run folder.")
        if not old:
            with self.db:
                self.db.execute("INSERT INTO metadata VALUES ('manifest', ?)", (encoded,))

    def get(self, config, block):
        row = self.db.execute("SELECT digest, payload FROM blocks WHERE config=? AND block=?", (config, block)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[1]).hexdigest() != row[0]:
            raise RuntimeError(f"Checkpoint checksum failed: configuration {config}, block {block}.")
        return pickle.loads(zlib.decompress(row[1]))

    def put(self, config, block, result):
        payload = zlib.compress(pickle.dumps(result, protocol=5), level=1)
        with self.db:
            self.db.execute("INSERT INTO blocks VALUES (?, ?, ?, ?)", (config, block, hashlib.sha256(payload).hexdigest(), payload))

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]

    def close(self):
        self.db.close()


def configurations(prepared, method):
    if method == "hmc":
        return [dict(epsilon=e, L=L) for e, L in itertools.product(prepared.epsilon_values, prepared.L_values)]
    return [dict(lambda_value=l, a=a, Q=Q, epsilon=e, L=L)
            for l, a, Q, e, L in itertools.product(prepared.lambda_values, prepared.a_values,
                prepared.Q_values, prepared.common.epsilon_values, prepared.common.L_values)]


def aggregate(module, method, settings, blocks):
    if method == "hmc":
        return module._aggregate_configuration(settings["epsilon"], settings["L"], blocks)
    return module._aggregate_thc_configuration(settings["lambda_value"], settings["a"], settings["Q"], settings["epsilon"], settings["L"], blocks)


def campaign(module, method, source, run_dir, inputs=None, stop_after_blocks=None):
    """The optional inputs/stop count are used only for isolated resume checks."""
    run_dir = Path(run_dir).resolve()
    results_dir = run_dir / "results"
    checkpoint_dir = run_dir / "checkpoints"
    for folder in (results_dir, checkpoint_dir, run_dir / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    inputs = dataclasses.replace(inputs or module.USER_INPUTS, output_directory=str(results_dir))
    destination = results_dir / inputs.output_filename
    complete_path = results_dir / "COMPLETED.txt"
    manifest = {
        "format": "gaussian-expanded-block-checkpoints-v1",
        "method": method,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": json.loads(json.dumps(dataclasses.asdict(inputs), default=str)),
        "python": platform.python_version(),
        "numpy": module.np.__version__,
        "numba": module.numba.__version__ if module.NUMBA_AVAILABLE else None,
        "llvmlite": importlib.metadata.version("llvmlite") if module.NUMBA_AVAILABLE else None,
    }
    # Separate process lock prevents duplicate jobs writing this campaign.
    lock_handle = (checkpoint_dir / "campaign.lock").open("a+")
    if os.name == "posix":
        import fcntl
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process already owns this run folder.") from exc
    store = Checkpoints(checkpoint_dir / "blocks.sqlite3", manifest)
    try:
        if complete_path.exists():
            completed = json.loads(complete_path.read_text(encoding="utf-8"))
            if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != completed["report_sha256"]:
                raise RuntimeError("Completed report is missing or its checksum differs.")
            print(f"Already complete: {destination}", flush=True)
            return 0
        atomic_text(run_dir / "run_manifest.txt", json.dumps(manifest, indent=2) + "\n")
        driver = module.run_hmc_tuning if method == "hmc" else module.run_thc_tuning
        preflight = driver(inputs, preflight_only=True)
        prepared = preflight["prepared_inputs"]
        common = prepared if method == "hmc" else prepared.common
        settings_list = configurations(prepared, method)
        n_blocks = len(common.seed_blocks)
        total_blocks = len(settings_list) * n_blocks
        saved = store.count()
        if saved > total_blocks:
            raise RuntimeError("Checkpoint has more blocks than the declared grid.")
        print(f"{method.upper()}: {len(settings_list)} configurations; {n_blocks} blocks each; {saved}/{total_blocks} blocks already saved.", flush=True)
        streams = [[module._make_random_streams(seed, common.n_warmup + common.n_retained, common.dimension)
                    for seed in seeds] for seeds in common.seed_blocks]
        aux_moments = tuple(module._retained_standardized_momentum_second_moment(cache, common.n_warmup) for cache in streams)
        new_blocks = 0
        start = time.monotonic()
        for config_index, settings in enumerate(settings_list):
            for block_index, seeds in enumerate(common.seed_blocks):
                if store.get(config_index, block_index) is not None:
                    continue
                if STOP_REQUESTED or (stop_after_blocks is not None and new_blocks >= stop_after_blocks):
                    print(f"SAVED FOR RESUME: {saved}/{total_blocks} blocks. No completed block will be rerun.", flush=True)
                    return 75
                block = module._run_configuration_block(**settings, block_index=block_index,
                    seeds=seeds, stream_cache=streams[block_index], prepared=prepared)
                store.put(config_index, block_index, block)
                saved += 1
                new_blocks += 1
                progress = f"{method.upper()} saved {saved}/{total_blocks} blocks; configuration {config_index + 1}/{len(settings_list)}, block {block_index + 1}/{n_blocks}; session elapsed {time.monotonic() - start:.1f}s\n"
                atomic_text(results_dir / "PROGRESS.txt", progress + json.dumps(settings, sort_keys=True) + "\n")
                print(progress.rstrip(), flush=True)
        if store.count() != total_blocks:
            raise RuntimeError("Full-grid checkpoint is incomplete; final report was not written.")
        # Build the original report and select the representative with the
        # frozen, original aggregation and tie-breaking functions.
        all_results = [aggregate(module, method, settings,
            [store.get(i, b) for b in range(n_blocks)]) for i, settings in enumerate(settings_list)]
        select = module._select_representative if method == "hmc" else module._select_thc_representative
        representative = select(all_results)
        report = module._build_report(configuration_results=all_results, representative=representative,
            auxiliary_momentum_moments=aux_moments, prepared=prepared,
            self_test_messages=preflight["self_test_messages"])
        atomic_text(destination, report + "\n")
        completed = {
            "method": method, "configurations": len(settings_list), "blocks": total_blocks,
            "report": str(destination),
            "report_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        atomic_text(complete_path, json.dumps(completed, indent=2) + "\n")
        if common.print_report:
            print(report, flush=True)
        print(f"COMPLETE: {destination}", flush=True)
        return 0
    finally:
        store.close()
        lock_handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("hmc", "thc"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    module = load_tuner(args.source.resolve(), args.method)
    return campaign(module, args.method, args.source.resolve(), args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
