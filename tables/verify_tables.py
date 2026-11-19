"""Recompute the manuscript table entries from the distributed numerical results."""

import argparse
import json
import math
import sys
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source_provenance import verify_source

# Each comparison has its own held-out blocks, including comparisons that
# reuse a winner from an earlier tuning search.
COMPARISONS = {
    1: ("multiscale-gaussian", "hmc"), 2: ("multiscale-gaussian", "hmc"),
    3: ("multiscale-gaussian", "s-qhmc"), 4: ("multiscale-gaussian", "s-qhmc"),
    5: ("multiscale-gaussian", "nuts"), 6: ("multiscale-gaussian", "nuts"),
    7: ("banana", "hmc"), 8: ("banana", "hmc"),
    9: ("banana", "s-qhmc"), 10: ("banana", "s-qhmc"),
    11: ("banana", "nuts"), 12: ("banana", "nuts"),
    13: ("neals-funnel", "hmc"), 14: ("neals-funnel", "hmc"),
    15: ("neals-funnel", "hmc"), 16: ("neals-funnel", "s-qhmc"),
    17: ("neals-funnel", "s-qhmc"), 18: ("neals-funnel", "nuts"),
    19: ("neals-funnel", "nuts"),
}
METHODS = {
    "HMC": "HMC", "THC": "THC_bounded", "THC (bounded scale)": "THC_bounded",
    "S-QHMC": "S_QHMC", "NUTS": "NUTS", "Original slice-NUTS": "NUTS",
    "Refreshed-mixture THC-v3": "THC_v3", "THC-v4": "THC_v4",
    "Periodic THC-v2": "THC_v2", "Scalar THC": "Scalar_THC",
    "Periodic THC": "Periodic_THC", "Conditional THC": "Conditional_THC",
}


def printed_rows(number):
    page = ROOT / "tables" / f"table-{number:02}.md"
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in page.read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    ]
    return [dict(zip(rows[0], row, strict=True)) for row in rows[2:]]


@lru_cache(maxsize=None)
def heldout_report(folder, method):
    report = (folder / f"{method}_results.txt").read_text(encoding="utf-8")
    data = json.loads(report.split("BEGIN COMPARISON_JSON\n", 1)[1]
                      .split("\nEND COMPARISON_JSON", 1)[0])
    verify_source(folder / f"{method}.py", data["metadata"]["script_sha256"])
    if data["metadata"]["testing"] or method != data["metadata"]["method"]:
        raise ValueError(f"Expected a final {method} report in {folder}")
    blocks = sorted(data["blocks"], key=lambda block: block["block_index"])
    paired_seeds = [block["seeds"] for block in blocks]
    protocol = data["metadata"]["protocol"]
    if ([block["block_index"] for block in blocks] != list(range(10))
            or paired_seeds != protocol["seed_blocks"]):
        raise ValueError(f"{method} blocks do not match the ten frozen seed blocks")
    return data["metadata"], blocks


def metrics(blocks):
    costs = [block["gradients_per_retained_draw"] for block in blocks]
    iats = [block["worst_iat"] for block in blocks]
    efficiencies = []
    for block, cost, iat in zip(blocks, costs, iats, strict=True):
        if block["status"] != "ok":
            if block["block_efficiency"] is not None:
                raise ValueError("Unexpected defined efficiency for an invalid block")
            continue
        efficiency = 1 / (cost * iat)
        if not math.isclose(efficiency, block["block_efficiency"], rel_tol=1e-12):
            raise ValueError("Block efficiency differs from 1/(gradient cost × worst IAT)")
        efficiencies.append(efficiency)
    all_defined = len(efficiencies) == len(blocks)
    values = {
        "Gradients/draw": fmean(costs),
        "Maximum worst IAT": max(iats),
        "Minimum efficiency": min(efficiencies) if all_defined else None,
        "Held-out score": math.exp(fmean(map(math.log, efficiencies))) if all_defined else None,
        "Maximum R-hat": max(b["maximum_rhat"] for b in blocks),
        "Maximum moment error": max(b["maximum_standardized_moment_error"] for b in blocks),
    }
    if "acceptance" in blocks[0]:
        values["Acceptance"] = fmean(b["acceptance"] for b in blocks)
    if "retained_nonfinite_trajectories" in blocks[0]:
        values["Nonfinite proposals"] = sum(b["retained_nonfinite_trajectories"] for b in blocks)
    return values


def paired_wins(first, second):
    wins = 0
    for left, right in zip(first, second, strict=True):
        if (left["block_index"], left["seeds"]) != (right["block_index"], right["seeds"]):
            raise ValueError("The two reports use different seed blocks")
        wins += left["block_efficiency"] > right["block_efficiency"]
    return f"{wins}/{len(first)}"


def check_value(table, method, key, printed, actual):
    # The exponent of a Decimal preserves the last digit displayed, even for
    # entries such as 1.00130 or 8.7912e-3.
    if actual is None:
        matches = printed in ("--", "undefined", "—")
    elif isinstance(actual, str):
        matches = actual == printed
    else:
        value = Decimal(printed)
        half_last_digit = Decimal(1).scaleb(value.as_tuple().exponent) / 2
        matches = abs(value - Decimal(str(actual))) <= half_last_digit
    if not matches:
        raise ValueError(f"Table {table}, {method}, {key}: printed {printed}, calculated {actual}")


def setting_value(token):
    if token.startswith("log(") and token.endswith(")"):
        return math.log(float(token[4:-1]))
    if token == "pi/2":
        return math.pi / 2
    return float(token)


def normalized_settings(settings):
    result = dict(settings)
    if "lambda_value" in result:
        result["lambda"] = result["lambda_value"]
    if "epsilon" in result and "L" in result:
        result["T"] = result["epsilon"] * result["L"]
    if "coupling_code" in result:
        code = result["coupling_code"]
        result["alpha"] = abs(code)
        result["phi"] = math.pi / 2 if code < 0 else 0
    return result


def check_table(number, show=False):
    target, comparison = COMPARISONS[number]
    folder = ROOT / "targets" / target
    rows = printed_rows(number)
    if "Tuning score" in rows[0]:
        winners = json.loads((folder / "tuning/winners.json").read_text(encoding="utf-8"))
        for row in rows:
            method = METHODS[row["Method"]]
            winner = winners[method]
            script = ROOT / winner["script"]
            manifest = json.loads((ROOT / winner["manifest"]).read_text(encoding="utf-8"))
            if "source_checksums" in manifest:
                original_path = Path(*Path(winner["script"]).parts[2:]).as_posix()
                recorded = manifest["source_checksums"][original_path]
            else:
                recorded = manifest["source_sha256"]
            if recorded != winner["script_sha256"]:
                raise ValueError(f"{method}: winner and tuning manifest disagree on source checksum")
            verify_source(script, recorded)
            check_value(number, method, "score", row["Tuning score"], winner["score"])
            if "Defined/tested" in row:
                counts = f'{winner["defined"]}/{winner["tested"]}'
                check_value(number, method, "defined/tested", row["Defined/tested"], counts)
            settings = normalized_settings(winner["settings"])
            metadata, _ = heldout_report(folder / "held-out" / comparison, method)
            used = normalized_settings(metadata["protocol"]["methods"][method])
            for item in row["Frozen settings"].split(", "):
                key, printed = item.split("=")
                expected = setting_value(printed)
                for source, record in (("tuning", settings), ("held-out", used)):
                    if not math.isclose(record[key], expected, rel_tol=2e-13, abs_tol=1e-15):
                        raise ValueError(f"Table {number}, {method}: {item} differs from {source} {record[key]}")
            if show:
                print(method, "settings", settings, "score", winner["score"])
        return

    folder = folder / "held-out" / comparison
    reports = {METHODS[r["Method"]]: heldout_report(folder, METHODS[r["Method"]]) for r in rows}
    if len({metadata["protocol_sha256"] for metadata, blocks in reports.values()}) != 1:
        raise ValueError(f"Table {number} mixes different held-out protocols")
    for row in rows:
        method = METHODS[row["Method"]]
        blocks = reports[method][1]
        values = metrics(blocks)
        if "Wins" in row:
            if method != "NUTS":
                values["Wins"] = paired_wins(blocks, reports["NUTS"][1])
            elif len(reports) == 2:
                other = next(blocks for m, (_, blocks) in reports.items() if m != "NUTS")
                values["Wins"] = paired_wins(blocks, other)
            else:
                values["Wins"] = None  # Table 19 marks NUTS as the common reference.
        for key, printed in row.items():
            if key != "Method":
                check_value(number, method, key, printed, values[key])
        if show:
            print(method, {key: values[key] for key in row if key != "Method"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", type=int, nargs="*", metavar="N", help="table numbers 1–19 (default: all)")
    parser.add_argument("--show", action="store_true", help="also print unrounded computed values")
    args = parser.parse_args()
    numbers = args.tables or list(COMPARISONS)
    if any(number not in COMPARISONS for number in numbers):
        parser.error("table numbers must be between 1 and 19")
    for number in numbers:
        check_table(number, args.show)
        print(f"Table {number}: verified")


if __name__ == "__main__":
    main()
