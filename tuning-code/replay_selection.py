"""Recompute a recorded THC tuning selection from its original inputs and seeds.

The original worker reused partial GREX checkpoints. This entry point runs
the identical selected indices from scratch through the archived Adapter.
Output is one JSON record for each completed configuration.
"""
import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parent
TARGETS = SOURCE.parent / "targets"
WRAPPER = SOURCE / "grex_thc_budgeted_20260927"


def with_original_container(template, value):
    if isinstance(template, (tuple, list)):
        return type(template)(with_original_container(t, v) for t, v in zip(template, value, strict=True))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("multiscale-gaussian", "banana", "neals-funnel"))
    parser.add_argument("method", help="Method key from that target's tuning/winners.json")
    parser.add_argument("--part", type=int, required=True, help="Zero-based original worker number")
    parser.add_argument("--configuration", type=int, help="Replay just this original global grid index")
    parser.add_argument("--check-only", action="store_true", help="Verify frozen inputs without sampling")
    args = parser.parse_args()
    folder = TARGETS / args.target / "tuning"
    catalog = json.loads((folder / "winners.json").read_text())
    method = catalog.get(args.method, {})
    if "search_key" not in method:
        parser.error("Use a THC method whose tuning guide links to replay_selection.py")
    settings_dir = folder / "records" / args.method
    frozen = json.loads((settings_dir / "FROZEN.txt").read_text())
    if not 0 <= args.part < frozen["workers"]:
        parser.error(f"Part number must be between zero and {frozen['workers'] - 1}")
    selected_file = settings_dir / "selection" / f"part_{args.part:03d}.json"
    selected = selected_file.read_bytes()
    expected_digest = frozen["parts"][args.part]["indices_sha256"]
    if hashlib.sha256(selected).hexdigest() != expected_digest:
        raise ValueError("Selected index list differs from the frozen search")
    indices = json.loads(selected)
    if args.configuration is not None:
        if args.configuration not in indices:
            parser.error("That grid index is absent from this worker's frozen selection")
        indices = [args.configuration]

    sys.path.insert(0, str(WRAPPER))
    import budgeted_runner as original
    adapter = original.Adapter(frozen["key"], local_base=SOURCE)
    if adapter.sources() != frozen["source_checksums"] or original.wrapper_sources() != frozen["wrapper_checksums"]:
        raise ValueError("Original tuner or wrapper checksum differs")
    # libm can round a grid endpoint such as log(1.05) differently across
    # platforms. Rebuild differing scientific inputs from the stored values.
    replacements = {}
    for name, value in frozen["inputs"].items():
        current = getattr(adapter.inputs, name)
        if name != "output_directory" and original.plain(current) != value:
            replacements[name] = with_original_container(current, value)
    if replacements:
        # These are the log-amplitude grids calculated with the system libm.
        # Keep the already-loaded original module and its own dataclass types.
        if set(replacements) != {"lambda_values"} or "lambda_value" not in adapter.grid.keys:
            raise ValueError(f"Unexpected differences in scientific inputs: {list(replacements)}")
        adapter.inputs = dataclasses.replace(adapter.inputs, **replacements)
        adapter.prepared = adapter.module._prepare_thc_inputs(adapter.inputs)
        adapter.common = getattr(adapter.prepared, "common", adapter.prepared)
        axes = [adapter.prepared.lambda_values if key == "lambda_value" else axis
                for key, axis in zip(adapter.grid.keys, adapter.grid.values, strict=True)]
        adapter.grid = original.Grid(adapter.grid.keys, axes)
    adapter.check_production()
    inputs = original.plain(adapter.inputs)
    # This field was an absolute GREX path. All scientific inputs must match.
    inputs["output_directory"] = frozen["inputs"]["output_directory"]
    if inputs != frozen["inputs"]:
        raise ValueError("Scientific inputs differ from the frozen search")
    if args.check_only:
        print(f"Verified {len(indices)} selected configuration(s) and ten seed blocks per configuration.")
        return
    for index in indices:
        blocks = [adapter.run_block(index, block) for block in range(10)]
        result = adapter.aggregate(index, blocks)
        print(original.encode(original.result_summary(adapter, index, result)), flush=True)


if __name__ == "__main__":
    main()
