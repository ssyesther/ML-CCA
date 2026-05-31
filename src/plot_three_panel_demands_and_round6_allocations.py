import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

import matplotlib.pyplot as plt
import parse_allocations


def sort_items(items: List[str]) -> List[str]:
    def key(item: str) -> Tuple[int, int, str]:
        m = re.match(r"station_(\d+)_(\d+)", item)
        if m:
            return (int(m.group(1)), int(m.group(2)), item)
        return (999999, 999999, item)
    return sorted(items, key=key)


def load_demands_csv(csv_path: str) -> Dict[str, int]:
    demands: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = row.get("item")
            if not item:
                continue
            try:
                demand_count = int(float(row.get("value", "0")))  # fallback if misheader
            except Exception:
                # correct header is demand_count column; if not present we build later
                pass
            # Expected header is bidder_id,item,value in original demands CSV; build counts by aggregation
            # But here we use the aggregated CSV from previous step: item_demand_counts_*.csv
            # If this function is used with aggregated CSV, column names are item,demand_count
    # Try aggregated CSV path format
    item_to_count: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Determine header type
        if header == ["item", "demand_count"]:
            for item, count in reader:
                try:
                    item_to_count[item] = int(float(count))
                except Exception:
                    item_to_count[item] = 0
        else:
            # Fallback: original detailed CSV bidder_id,item,value -> aggregate
            idx_item = header.index("item") if "item" in header else 1
            idx_bidder = header.index("bidder_id") if "bidder_id" in header else 0
            seen: Dict[str, Set[str]] = defaultdict(set)
            for row in reader:
                if not row:
                    continue
                item = row[idx_item]
                bidder = row[idx_bidder]
                seen[item].add(bidder)
            item_to_count = {it: len(bidders) for it, bidders in seen.items()}
    return item_to_count


def find_price_vector_length(data: Any) -> int:
    # Try to infer number of goods from Price Vector per Iteration
    for key in data.keys() if isinstance(data, dict) else []:
        if isinstance(key, str) and key.lower().strip() == "price vector per iteration".lower():
            pvi = data[key]
            if isinstance(pvi, dict):
                for _, arr in pvi.items():
                    if isinstance(arr, list):
                        return len(arr)
    # Fallback: search any list of floats of large length
    def scan(obj: Any) -> int:
        if isinstance(obj, list):
            if obj and all(isinstance(x, (int, float)) or (isinstance(x, str) and re.match(r"^[-+]?\d+(?:\.\d+)?$", x)) for x in obj):
                return len(obj)
            return 0
        if isinstance(obj, dict):
            for v in obj.values():
                n = scan(v)
                if n:
                    return n
        return 0
    return scan(data)


def find_items_list(data: Any, n_goods: int) -> List[str]:
    candidates: List[List[str]] = []

    def collect_lists(obj: Any):
        if isinstance(obj, list):
            if obj and all(isinstance(x, str) for x in obj):
                candidates.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                collect_lists(v)

    collect_lists(data)

    # Prefer lists with length == n_goods and matching station pattern
    for lst in candidates:
        if len(lst) == n_goods and all(re.match(r"station_\d+_\d+", s or "") for s in lst):
            return lst
    # Otherwise pick any list dominated by station-like strings
    for lst in candidates:
        station_like = sum(1 for s in lst if isinstance(s, str) and re.match(r"station_\d+_\d+", s))
        if station_like >= max(1, len(lst) // 2):
            return lst
    return []


def extract_allocated_items_round(data: Any, round_idx: int, items_list: List[str]) -> Set[str]:
    allocated_items: Set[str] = set()
    # Locate Allocation per Iteration block
    alloc_block = None
    for key in data.keys() if isinstance(data, dict) else []:
        if isinstance(key, str) and key.lower().strip() == "allocation per iteration".lower():
            alloc_block = data[key]
            break
    if alloc_block is None:
        # Try fuzzy
        for key in data.keys() if isinstance(data, dict) else []:
            if isinstance(key, str) and "allocation" in key.lower() and "iteration" in key.lower():
                alloc_block = data[key]
                break
    if not isinstance(alloc_block, dict):
        return allocated_items

    round_key = str(round_idx)
    if round_key not in alloc_block:
        return allocated_items

    round_data = alloc_block[round_key]
    if not isinstance(round_data, dict):
        return allocated_items

    # Each bidder entry may have good_ids and allocated_bundle
    for bidder_key, bidder_entry in round_data.items():
        if not isinstance(bidder_entry, dict):
            continue
        good_ids = bidder_entry.get("good_ids") or bidder_entry.get("goods_ids")
        allocated_bundle = bidder_entry.get("allocated_bundle")
        if not isinstance(good_ids, list) or not isinstance(allocated_bundle, list):
            continue
        for gid, flag in zip(good_ids, allocated_bundle):
            try:
                allocated = (float(flag) >= 0.5) and (not math.isnan(float(flag)))
            except Exception:
                allocated = False
            if allocated:
                if items_list and isinstance(gid, int) and 0 <= gid < len(items_list):
                    allocated_items.add(items_list[gid])
                else:
                    allocated_items.add(f"good_{gid}")

    return allocated_items


def extract_allocated_items_round_with_helper(results_path: str, auction_path: str, round_idx: int) -> Set[str]:
    """Use project helper parse_allocations to robustly get allocated items for a given round."""
    allocated_items: Set[str] = set()
    # First try the helper
    try:
        parsed = parse_allocations.parse_allocation_for_iteration(results_path, auction_path, round_idx)
        for _, info in parsed.items():
            items = info.get("items") or []
            for it in items:
                if it:
                    allocated_items.add(str(it))
        if allocated_items:
            return allocated_items
    except Exception:
        # proceed to fallbacks below
        pass

    # Fallback 1: pre-parsed allocations file (e.g., parsed_allocations_mlcca_round6.json)
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # If the file is already a parsed mapping with bidder -> {items: [...]}
        if isinstance(data, dict) and all(isinstance(v, dict) and ("items" in v) for v in data.values()):
            for _, info in data.items():
                items = info.get("items") or []
                for it in items:
                    if it:
                        allocated_items.add(str(it))
            if allocated_items:
                return allocated_items
    except Exception:
        pass

    # Fallback 2: raw results.json with "Allocation per Iteration" and potential missing good_ids
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        alloc_iters = results.get("Allocation per Iteration") or results.get("allocation per iteration") or {}
        round_key = str(round_idx)
        if isinstance(alloc_iters, dict) and round_key in alloc_iters:
            it_alloc = alloc_iters.get(round_key, {})
            # Build goods list from auction for mapping indices -> station_i_j
            goods, num_slots = parse_allocations.build_goods_list_from_auction(auction_path)

            def vector_to_indices(vec):
                return [i for i, v in enumerate(vec) if float(v) != 0.0]

            def indices_to_items(indices):
                items = []
                if goods:
                    for idx in indices:
                        if 0 <= idx < len(goods):
                            items.append(goods[idx])
                        else:
                            station = idx // num_slots
                            t = idx % num_slots
                            items.append(f"station_{station}_{t}")
                    return items
                for idx in indices:
                    station = idx // num_slots
                    t = idx % num_slots
                    items.append(f"station_{station}_{t}")
                return items

            for _, info in it_alloc.items():
                if not isinstance(info, dict):
                    continue
                good_ids = info.get("good_ids") or info.get("goods_ids") or []
                bundle_vec = info.get("allocated_bundle") or []
                indices = good_ids if good_ids else vector_to_indices(bundle_vec)
                for it in indices_to_items(indices):
                    allocated_items.add(str(it))
            if allocated_items:
                return allocated_items
    except Exception:
        pass

    # If all fail, return empty set
    return allocated_items


def plot_three_panel(
    item_order: List[str],
    demand_counts: Dict[str, int],
    mlcca_alloc_items: Set[str],
    cca_alloc_items: Set[str],
    out_path: str,
):
    # Single panel: demand counts with ML-CCA-only highlight
    width = max(14, 0.28 * len(item_order))
    height = 6
    fig, ax = plt.subplots(1, 1, figsize=(width, height))

    counts = [demand_counts.get(it, 0) for it in item_order]
    ml_only = set(mlcca_alloc_items) - set(cca_alloc_items)

    # Colors: revert to original blue (others) and orange (ML-CCA allocated only)
    base_color = "#4472C4"        # original blue
    highlight_color = "#ED7D31"   # original orange
    colors = [highlight_color if it in ml_only else base_color for it in item_order]

    ax.bar(range(len(item_order)), counts, color=colors, edgecolor="#333333", linewidth=0.2)
    ax.set_ylabel("Demand Count")
    ax.set_title("Item Demand Counts (highlight: ML-CCA only allocated)")
    ax.grid(axis="y", alpha=0.3)

    # Legend matching requested color mapping
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=base_color, label="Demand (others)"),
        mpatches.Patch(color=highlight_color, label="ML-CCA allocated only"),
    ]
    ax.legend(handles=handles, fontsize=8)

    # X ticks
    plt.xticks(range(len(item_order)), item_order, rotation=90, fontsize=8)
    plt.xlabel("Items")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot 3-panel: demand + MLCCA round6 + CCA round6")
    parser.add_argument("--demands_csv", required=True, help="Path to item demand counts CSV (or raw demands CSV)")
    parser.add_argument("--mlcca_json", required=True, help="Path to MLCCA results.json")
    parser.add_argument("--cca_json", required=True, help="Path to CCA results.json")
    parser.add_argument("--auction_json", default=os.path.join("src", "auction_instance.json"), help="Path to auction_instance.json for goods mapping")
    parser.add_argument("--round", type=int, default=6, help="Round index to visualize allocations (default: 6)")
    parser.add_argument(
        "--outdir",
        default=os.path.join("src", "wandb_custom_plots", "results"),
        help="Directory to write output PNG",
    )
    args = parser.parse_args()

    # Load demand counts (handles both aggregated counts CSV and raw bidder-level CSV)
    demand_counts = load_demands_csv(args.demands_csv)

    # Robust extraction of allocated items using helper (handles missing good_ids by deriving indices from vectors)
    ml_alloc_items = extract_allocated_items_round_with_helper(args.mlcca_json, args.auction_json, args.round)
    cca_alloc_items = extract_allocated_items_round_with_helper(args.cca_json, args.auction_json, args.round)

    # Union of all items to make consistent x-axis
    all_items = set(demand_counts.keys()) | ml_alloc_items | cca_alloc_items
    item_order = sort_items(list(all_items))

    os.makedirs(args.outdir, exist_ok=True)
    out_png = os.path.join(args.outdir, "three_panel_item_demand_and_round6_allocations.png")
    plot_three_panel(item_order, demand_counts, ml_alloc_items, cca_alloc_items, out_png)
    print(out_png)


if __name__ == "__main__":
    main()