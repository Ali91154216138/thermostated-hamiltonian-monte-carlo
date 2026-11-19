"""Read the tuning tables' primary representatives from the GREX result ZIPs."""
import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path


def read_winner(lines):
    """Parse original human-readable reports while keeping only the winning entry."""
    winner = None
    counts = None
    gaussian_rows = []
    in_gaussian_grid = False
    for raw in lines:
        line = raw.rstrip("\n")
        match = re.search(r"defined/tested(?: settings)?:\s*(\d+)/(\d+)", line, re.I)
        if match:
            counts = tuple(map(int, match.groups()))
        if line.startswith("epsilon") and "status" in line and "GM_eff" in line:
            in_gaussian_grid = True
            continue
        if in_gaussian_grid:
            if not line.strip():
                in_gaussian_grid = False
            else:
                gaussian_rows.append(line.split()[2] == "ok")
        if line in ("SELECTED FIXED REPRESENTATIVE", "PRIMARY EFFICIENCY REPRESENTATIVE", "BEST EFFICIENCY"):
            content = []
            for row in lines:
                content.append(row)
                if row.rstrip() == "}":
                    break
            record = json.loads("".join(content))
            winner = {"settings": record["settings"], "score": record["geometric_mean_efficiency"]}
            for key in ("global_index", "global_configuration_index"):
                if key in record:
                    winner["global_index"] = record[key]
        elif line in ("SELECTED FIXED HMC REPRESENTATIVE", "SELECTED FIXED NUTS REPRESENTATIVE"):
            next(lines)  # underline
            setting = next(lines)
            if "HMC" in line:
                match = re.match(r"epsilon=([^,]+), L=(\d+)", setting)
                parameters = {"epsilon": float(match[1]), "L": int(match[2])}
            else:
                parameters = {"target_acceptance": float(setting.split("delta=")[1])}
            score_line = next(row for row in lines if row.startswith("Geometric mean="))
            winner = {"settings": parameters, "score": float(score_line.split("=")[1])}
        if winner is not None and counts is not None:
            break
    if counts is None and gaussian_rows:
        counts = sum(gaussian_rows), len(gaussian_rows)
    if winner is None or counts is None:
        raise ValueError("Incomplete winner or defined/tested counts in the supplied report")
    return dict(winner, defined=counts[0], tested=counts[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Original tuning ZIP for one target")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(args.archive) as archive:
        members = set(archive.namelist())
        catalogs = [json.loads(c.read_text()) for c in sorted((root / "targets").glob("*/tuning/winners.json"))]
        matching = [c for c in catalogs if any(v["report"] in members for v in c.values())]
        if len(matching) != 1:
            raise ValueError("Expected one original target archive")
        for method, expected in matching[0].items():
            if expected["report"] not in members:
                raise ValueError(f"Missing report for {method}")
            with archive.open(expected["report"]) as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != expected["report_sha256"]:
                    raise ValueError(f"Report checksum mismatch for {method}")
            with archive.open(expected["report"]) as source, io.TextIOWrapper(source, encoding="utf-8") as text:
                observed = read_winner(text)
            for key in ("defined", "tested"):
                if expected[key] != observed[key]:
                    raise ValueError(f"{method}: {key} differs")
            if not math.isclose(observed["score"], expected["score"], rel_tol=1e-12, abs_tol=0):
                raise ValueError(f"{method}: score differs")
            if any(expected["settings"].get(k) != v for k, v in observed["settings"].items()):
                raise ValueError(f"{method}: frozen setting differs")
            print(json.dumps({"method": method, **observed}, allow_nan=False))


if __name__ == "__main__":
    main()
