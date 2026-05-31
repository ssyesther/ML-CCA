import argparse
import json
import os
import re
from typing import Dict, List, Tuple


def parse_demand_lines(
    logfile_path: str,
    bidder_start: int,
    bidder_end: int,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Parse demand lines like:
    INFO:root:Bidder 0 @ station_4: slots=[6] time_value=27.09 distance=0.35 cost=0.08 → 27.02

    Returns: {bidder_id: [(item_id, value), ...]}
    """
    # Unicode arrow and possible ASCII fallback
    arrow_pattern = r"\u2192|->|→"
    # Regex to capture bidder, station, slot, value
    line_re = re.compile(
        rf"INFO:root:Bidder\s+(?P<bidder>\d+)\s+@\s+station_(?P<station>\d+):\s+"
        rf"slots=\[(?P<slot>\d+)\].*?(?:{arrow_pattern})\s*(?P<value>[-+]?\d+(?:\.\d+)?)"
    )

    # Track results and seen bidders for first pass only
    results: Dict[int, List[Tuple[str, float]]] = {}
    seen_bidders_order: List[int] = []

    with open(logfile_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            bidder = int(m.group("bidder"))
            if bidder < bidder_start or bidder > bidder_end:
                # Ignore bidders outside the requested range
                continue

            station = int(m.group("station"))
            slot = int(m.group("slot"))
            value = float(m.group("value"))
            item_id = f"station_{station}_{slot}"

            # Initialize bidder list on first encounter to preserve first-pass ordering
            if bidder not in results:
                results[bidder] = []
                seen_bidders_order.append(bidder)

            # Avoid duplicates if the same item appears again later
            if item_id not in [it for it, _ in results[bidder]]:
                results[bidder].append((item_id, value))

            # If we've seen all requested bidders at least once, we can continue collecting lines
            # but to be safe and keep to "first pass", we stop when every bidder collected
            # has a complete set of items for typical cases (e.g., 6 stations). We cannot rely
            # on exact count, so we'll break once all bidders in range have been encountered
            # AND the next bidder encountered would be outside range. This heuristic balances
            # correctness without over-reading very long logs.
            if len(seen_bidders_order) == (bidder_end - bidder_start + 1):
                # We won't break here directly to ensure we capture all station lines of the last bidder
                # The loop continues; a later non-matching line simply won't affect results.
                pass

    # Sort items per bidder by value descending (optional: preserve file order)
    # The user example preserves file order; keep insertion order as parsed.
    # Ensure bidders with no lines still appear with empty list
    for b in range(bidder_start, bidder_end + 1):
        results.setdefault(b, [])

    return results


def write_outputs(
    demands: Dict[int, List[Tuple[str, float]]],
    outdir: str,
    run_id: str,
    bidder_start: int,
    bidder_end: int,
) -> Tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    base = f"demands_{run_id}_bidders{bidder_start}-{bidder_end}"
    csv_path = os.path.join(outdir, base + ".csv")
    json_path = os.path.join(outdir, base + ".json")

    # CSV: bidder_id,item,value
    with open(csv_path, "w", encoding="utf-8") as cf:
        cf.write("bidder_id,item,value\n")
        for bidder in range(bidder_start, bidder_end + 1):
            for item, val in demands.get(bidder, []):
                cf.write(f"{bidder},{item},{val}\n")

    # JSON: { "Bidder_0": [{"item": "station_4_6", "value": 27.02}, ...], ... }
    json_obj = {
        f"Bidder_{bidder}": [
            {"item": item, "value": value} for item, value in demands.get(bidder, [])
        ]
        for bidder in range(bidder_start, bidder_end + 1)
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_obj, jf, ensure_ascii=False, indent=2)

    return csv_path, json_path


def derive_run_id_from_path(path: str) -> str:
    m = re.search(r"(run-[^/]+)", path)
    return m.group(1) if m else "unknown_run"


def main():
    parser = argparse.ArgumentParser(description="Parse bidder demands from output.log")
    parser.add_argument("--input", required=True, help="Path to output.log")
    parser.add_argument(
        "--bidders",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=[0, 49],
        help="Bidder ID range inclusive (default: 0 49)",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join("src", "wandb_custom_plots", "results"),
        help="Directory to write CSV/JSON outputs",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Optional run identifier for file naming; derived from input path if omitted",
    )

    args = parser.parse_args()

    bidder_start, bidder_end = args.bidders
    if bidder_start > bidder_end:
        raise SystemExit("Invalid bidders range: START must be <= END")

    run_id = args.run_id or derive_run_id_from_path(args.input)

    demands = parse_demand_lines(
        logfile_path=args.input,
        bidder_start=bidder_start,
        bidder_end=bidder_end,
    )

    csv_path, json_path = write_outputs(
        demands=demands,
        outdir=args.outdir,
        run_id=run_id,
        bidder_start=bidder_start,
        bidder_end=bidder_end,
    )

    print("Wrote:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()