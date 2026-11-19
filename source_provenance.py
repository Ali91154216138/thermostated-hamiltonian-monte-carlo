"""Check released source files against the original experiment records."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def source_records():
    manifest = ROOT / "data" / "source-checksums.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["files"]


def verify_source(path, recorded_sha256, current_sha256=None):
    relative = path.relative_to(ROOT).as_posix()
    actual = current_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == ".py":
        record = source_records()[relative]
        if record["original_sha256"] != recorded_sha256:
            raise ValueError(f"{relative}: original checksum differs from the run record")
        if actual in (record["original_sha256"], record["current_sha256"]):
            return
    elif actual == recorded_sha256:
        return
    raise ValueError(f"{relative}: file checksum differs from the verified source")
