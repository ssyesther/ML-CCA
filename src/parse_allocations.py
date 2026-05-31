import json
import os
import argparse


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_goods_list_from_auction(auction_path):
    """
    Build ordered goods list [station_i_j] using auction_instance.json
    Falls back to MSVM default (6 stations, 24 slots) if file missing.
    """
    if auction_path and os.path.exists(auction_path):
        data = load_json(auction_path)
        stations = data.get("stations", [])
        num_slots = int(data.get("num_slots", 24))
        # Sort station ids by numeric suffix to guarantee station_0..N-1 order
        def station_num(sid):
            try:
                return int(str(sid.get('id', 'station_0')).split('_')[-1])
            except Exception:
                return 0
        stations_sorted = sorted(stations, key=station_num)
        goods = []
        for s in stations_sorted:
            sid = str(s.get('id', 'station_0'))
            for t in range(num_slots):
                goods.append(f"{sid}_{t}")
        return goods, num_slots
    # Fallback
    num_stations = 6
    num_slots = 24
    goods = [f"station_{i}_{j}" for i in range(num_stations) for j in range(num_slots)]
    return goods, num_slots


def indices_to_items(indices, num_slots, goods):
    items = []
    # Prefer direct mapping via goods order if provided
    if goods and len(goods) > 0:
        for idx in indices:
            if 0 <= idx < len(goods):
                items.append(goods[idx])
            else:
                # Fallback arithmetic mapping
                station = idx // num_slots
                t = idx % num_slots
                items.append(f"station_{station}_{t}")
        return items
    # Arithmetic mapping
    for idx in indices:
        station = idx // num_slots
        t = idx % num_slots
        items.append(f"station_{station}_{t}")
    return items


def vector_to_indices(vec):
    return [i for i, v in enumerate(vec) if float(v) != 0.0]


def parse_allocation_for_iteration(results_path, auction_path, iteration):
    results = load_json(results_path)
    # Load goods list
    goods, num_slots = build_goods_list_from_auction(auction_path)

    alloc_iters = results.get("Allocation per Iteration", {})
    it_key = str(iteration)
    if it_key not in alloc_iters:
        raise KeyError(f"Iteration {iteration} not found in 'Allocation per Iteration'.")

    it_alloc = alloc_iters[it_key]
    parsed = {}
    for bidder_key, info in it_alloc.items():
        bundle_vec = info.get("allocated_bundle", [])
        good_ids = info.get("good_ids", [])
        # Robust: if good_ids absent or empty, derive from vector
        indices = good_ids if good_ids else vector_to_indices(bundle_vec)
        items = indices_to_items(indices, num_slots, goods)
        parsed[bidder_key] = {
            "indices": indices,
            "items": items,
            "inferred_value": info.get("inferred_value"),
            "true_value": info.get("true_value"),
        }
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Parse MLCCA/CCA results allocations to station_x_y labels.")
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--iteration", type=int, default=6, help="Iteration number to parse (default: 6)")
    parser.add_argument("--auction", default=os.path.join("src", "auction_instance.json"), help="Path to auction_instance.json")
    parser.add_argument("--output", default=None, help="Optional output file to save parsed mapping")
    args = parser.parse_args()

    parsed = parse_allocation_for_iteration(args.results, args.auction, args.iteration)

    # Pretty print to console
    print(f"Parsed allocation for iteration {args.iteration}:")
    for bidder, d in sorted(parsed.items(), key=lambda kv: int(kv[0].split('_')[-1])):
        items_str = ", ".join(d["items"]) if d["items"] else "NONE"
        print(f"{bidder}: {items_str}")

    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print(f"Saved parsed mapping to {args.output}")


if __name__ == "__main__":
    main()