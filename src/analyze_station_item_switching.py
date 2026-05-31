import os
import re
import json
from typing import Dict, List, Tuple, Any

from custom_msvm_domain import CustomMSVMAuction
import parse_allocations


OUT_DIR = "/Users/y./Documents/ML-CCA-main/src/wandb_custom_plots/results"


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json_relaxed(path: str) -> dict:
    """读取JSON，允许 NaN/Infinity/-Infinity（替换为 null）。"""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    txt = re.sub(r"(?<![\w\-])NaN(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])Infinity(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])\-Infinity(?![\w\-])", "null", txt)
    return json.loads(txt)


def extract_initial_qinit_prices(initial_path: str) -> Tuple[int, List[List[float]]]:
    """
    从 initial_cca_result.json 的 "clock_rounds" 中提取前 Qinit 轮的价格向量。
    返回 (Qinit, [prices_round1, prices_round2, ...])
    """
    data = load_json_relaxed(initial_path)
    qinit = int(data.get("Qinit", 50))
    rounds = data.get("clock_rounds", []) or []
    prices_rounds: List[List[float]] = []
    for i in range(min(qinit, len(rounds))):
        pv = rounds[i].get("price_vector", []) or []
        try:
            pv = [float(x) for x in pv]
        except Exception:
            pv = []
        prices_rounds.append(pv)
    return qinit, prices_rounds


def build_goods_list_from_auction(auction_path: str) -> List[str]:
    goods, _ = parse_allocations.build_goods_list_from_auction(auction_path)
    return goods


def compute_bidder_top1_for_prices(prices: List[float], auction_path: str) -> Dict[int, List[str]]:
    """
    给定一轮的价格向量，返回 bidder -> top-1 bundle（最多一个item）。
    """
    goods_order, num_slots = parse_allocations.build_goods_list_from_auction(auction_path)
    data = load_json_relaxed(auction_path)
    stations = data.get("stations", [])
    bidders = data.get("bidders", [])
    ns = int(data.get("num_slots", num_slots if num_slots else 24))

    auction = CustomMSVMAuction(stations=stations, bidders=bidders, num_slots=ns)
    n = min(len(goods_order), len(prices))
    goods_order = goods_order[:n]
    prices = prices[:n]

    bidder_to_items: Dict[int, List[str]] = {}
    for b in bidders:
        try:
            bid_num = int(str(b.get("id", "Bidder_0")).split("_")[-1])
        except Exception:
            continue
        top_bundles = auction.get_best_bundles_original(bidder_id=bid_num, prices=prices, max_bundle_size=1)
        bundle_items = top_bundles[0] if top_bundles else []
        bidder_to_items[bid_num] = [str(it) for it in bundle_items]
    return bidder_to_items


def extract_ml_prices_for_round(results_path: str, round_num: int) -> List[float]:
    results = load_json_relaxed(results_path)
    pv = results.get("Price Vector per Iteration", {}) or {}
    arr = pv.get(str(round_num), []) or []
    try:
        return [float(x) for x in arr]
    except Exception:
        return []


def get_item_price_from_prices(goods: List[str], prices: List[float], item_id: str) -> float:
    if not goods or not prices:
        return float("nan")
    try:
        idx = goods.index(item_id)
        return float(prices[idx]) if 0 <= idx < len(prices) else float("nan")
    except Exception:
        return float("nan")


def parse_ml_final_allocations(results_path: str, auction_path: str, round_num: int) -> Dict[int, List[str]]:
    parsed = parse_allocations.parse_allocation_for_iteration(results_path, auction_path, round_num) or {}
    out: Dict[int, List[str]] = {}
    for bidder_key, info in parsed.items():
        try:
            bid_num = int(str(bidder_key).split("_")[-1])
        except Exception:
            continue
        items = [str(x) for x in (info.get("items") or [])]
        out[bid_num] = items
    return out


def build_auction(auction_path: str) -> CustomMSVMAuction:
    data = load_json_relaxed(auction_path)
    stations = data.get("stations", [])
    bidders = data.get("bidders", [])
    num_slots = int(data.get("num_slots", 24))
    return CustomMSVMAuction(stations=stations, bidders=bidders, num_slots=num_slots)


def compute_net_utility_for_item(auction: CustomMSVMAuction, goods: List[str], prices: List[float], item_id: str, bidder_num: int) -> float:
    try:
        idx = goods.index(item_id)
    except Exception:
        return float("-inf")
    price = prices[idx] if 0 <= idx < len(prices) else float("nan")
    val = auction.calculate_value(bidder_num, [item_id])
    return float(val - price)


def generate_station_schedule_table(
    auction_path: str,
    initial_path: str,
    results_path: str,
    station_item: str,
    bidders: List[int],
    out_csv_path: str,
):
    import csv
    auction = build_auction(auction_path)
    goods = build_goods_list_from_auction(auction_path)
    qinit, prices_rounds_initial = extract_initial_qinit_prices(initial_path)
    results = load_json_relaxed(results_path)
    pv_ml_dict = results.get("Price Vector per Iteration", {}) or {}
    ml_keys = sorted(pv_ml_dict.keys(), key=lambda x: int(x))
    ml_prices = [pv_ml_dict[k] for k in ml_keys]

    # bidder top-1 per round
    top1_initial = {r: compute_bidder_top1_for_prices(pv, auction_path) for r, pv in enumerate(prices_rounds_initial, start=1)}
    top1_ml = {i: compute_bidder_top1_for_prices(pv, auction_path) for i, pv in enumerate(ml_prices, start=1)}

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "round", "phase", "bidder_id", "station_item", "station_price",
            "station_net_utility", "top1_item", "top1_net_utility", "switched_from_station"
        ])
        # initial rounds
        for r_idx, prices in enumerate(prices_rounds_initial, start=1):
            station_price = get_item_price_from_prices(goods, prices, station_item)
            for b in bidders:
                station_util = compute_net_utility_for_item(auction, goods, prices, station_item, b)
                top_item = (top1_initial.get(r_idx, {}) or {}).get(b, [""])
                top_item_id = top_item[0] if isinstance(top_item, list) and top_item else ""
                top_util = compute_net_utility_for_item(auction, goods, prices, top_item_id, b) if top_item_id else float("-inf")
                switched = (top_item_id != station_item)
                w.writerow([r_idx, "initial", b, station_item, station_price, station_util, top_item_id, top_util, switched])

        # ml rounds appended
        for i_idx, prices in enumerate(ml_prices, start=1):
            round_num = qinit + i_idx
            station_price = get_item_price_from_prices(goods, prices, station_item)
            for b in bidders:
                station_util = compute_net_utility_for_item(auction, goods, prices, station_item, b)
                top_item = (top1_ml.get(i_idx, {}) or {}).get(b, [""])
                top_item_id = top_item[0] if isinstance(top_item, list) and top_item else ""
                top_util = compute_net_utility_for_item(auction, goods, prices, top_item_id, b) if top_item_id else float("-inf")
                switched = (top_item_id != station_item)
                w.writerow([round_num, "ml", b, station_item, station_price, station_util, top_item_id, top_util, switched])

    return out_csv_path


def analyze_station_switch(initial_path: str,
                           ml_results_path: str,
                           auction_path: str,
                           item_id: str = "station_5_1",
                           ml_offset: int = 50,
                           ml_final_round: int = 6) -> Tuple[str, str]:
    """
    合并前50轮(initial)与后6轮(ML)的逐轮top-1，跟踪 item_id 的需求变化与投标者转向，
    输出两份CSV：
      1) station_5_1_bidders_trace_seed_8.csv: 每轮每投标者top-1及该item价格
      2) station_5_1_switch_summary_seed_8.csv: 曾经选择该item的投标者的转向与最终结果
    返回两个CSV的路径。
    """
    ensure_out_dir(OUT_DIR)
    goods = build_goods_list_from_auction(auction_path)

    # 1) 前Qinit轮
    qinit, prices_rounds_initial = extract_initial_qinit_prices(initial_path)
    timeline_rounds: List[int] = list(range(1, qinit + 1))
    bidder_top1_per_round: Dict[int, Dict[int, List[str]]] = {}  # round -> {bidder -> [item]}
    item_price_per_round: Dict[int, float] = {}

    for r_idx, pv in enumerate(prices_rounds_initial, start=1):
        bidder_map = compute_bidder_top1_for_prices(pv, auction_path)
        bidder_top1_per_round[r_idx] = bidder_map
        item_price_per_round[r_idx] = get_item_price_from_prices(goods, pv, item_id)

    # 2) ML 6轮，偏移到 51..56
    for ml_r in range(1, ml_final_round + 1):
        pv_ml = extract_ml_prices_for_round(ml_results_path, ml_r)
        bidder_map_ml = compute_bidder_top1_for_prices(pv_ml, auction_path)
        combined_round = ml_offset + ml_r
        timeline_rounds.append(combined_round)
        bidder_top1_per_round[combined_round] = bidder_map_ml
        item_price_per_round[combined_round] = get_item_price_from_prices(goods, pv_ml, item_id)

    # 3) 生成逐轮追踪CSV
    trace_csv = os.path.join(OUT_DIR, "station_5_1_bidders_trace_seed_8.csv")
    with open(trace_csv, "w", encoding="utf-8") as f:
        f.write("round,bidder_id,top1_item,is_station_5_1,price_station_5_1\n")
        rounds_sorted = sorted(bidder_top1_per_round.keys())
        for rnd in rounds_sorted:
            bmap = bidder_top1_per_round.get(rnd, {})
            price_ = item_price_per_round.get(rnd, float("nan"))
            for bidder_id, bundle in sorted(bmap.items()):
                top1 = bundle[0] if bundle else ""
                is_target = (top1 == item_id)
                f.write(f"{rnd},{bidder_id},{top1},{1 if is_target else 0},{price_}\n")

    # 4) 统计切换：从上一轮top1是目标item，当前轮不是 => 发生转向
    #    记录第一次选择该item与最后一次选择该item的轮次，以及转向后的首个item
    switch_info: Dict[int, Dict[str, Any]] = {}
    rounds_sorted = sorted(bidder_top1_per_round.keys())
    prev_choice: Dict[int, str] = {}
    first_on_target: Dict[int, int] = {}
    last_on_target: Dict[int, int] = {}
    switch_to_item: Dict[int, str] = {}

    for rnd in rounds_sorted:
        bmap = bidder_top1_per_round.get(rnd, {})
        for bidder_id, bundle in bmap.items():
            choice = bundle[0] if bundle else ""
            # 记录第一次/最后一次选择目标item
            if choice == item_id:
                if bidder_id not in first_on_target:
                    first_on_target[bidder_id] = rnd
                last_on_target[bidder_id] = rnd

            # 检测从目标item转向其他item
            prev = prev_choice.get(bidder_id)
            if prev == item_id and choice != item_id and choice:
                # 首次记录转向的目标
                switch_to_item.setdefault(bidder_id, choice)
            prev_choice[bidder_id] = choice

    # 5) 解析ML最终分配，并归纳结果
    final_alloc_ml = parse_ml_final_allocations(ml_results_path, auction_path, ml_final_round)
    switch_csv = os.path.join(OUT_DIR, "station_5_1_switch_summary_seed_8.csv")
    with open(switch_csv, "w", encoding="utf-8") as f:
        f.write("bidder_id,first_round_on_station_5_1,last_round_on_station_5_1,switch_to_item,final_allocation_ml,final_outcome\n")
        bidders_considered = sorted(set(list(first_on_target.keys()) + list(last_on_target.keys())))
        for bidder_id in bidders_considered:
            first_r = first_on_target.get(bidder_id, "")
            last_r = last_on_target.get(bidder_id, "")
            switched_item = switch_to_item.get(bidder_id, "")
            final_items = final_alloc_ml.get(bidder_id, [])
            final_item = final_items[0] if final_items else ""
            if final_item == item_id:
                outcome = "won_station_5_1"
            elif final_item:
                outcome = "won_other"
            else:
                outcome = "none"
            f.write(f"{bidder_id},{first_r},{last_r},{switched_item},{final_item},{outcome}\n")

    return trace_csv, switch_csv


def main():
    initial_path = \
        "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/initial_cca_result.json"
    ml_results_path = \
        "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_gd_linear_prices_on_W_v3/ML_config_hpo1/8/results.json"
    auction_path = \
        "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/auction_instance_seed_8.json"

    trace_csv, switch_csv = analyze_station_switch(
        initial_path=initial_path,
        ml_results_path=ml_results_path,
        auction_path=auction_path,
        item_id="station_5_1",
        ml_offset=50,
        ml_final_round=6,
    )
    print(f"Saved trace CSV: {trace_csv}")
    print(f"Saved switch summary CSV: {switch_csv}")

    # Generate detailed schedule table for bidders 16, 19, 36
    schedule_csv = os.path.join(OUT_DIR, "station_5_1_schedule_table_seed_8.csv")
    _ = generate_station_schedule_table(
        auction_path=auction_path,
        initial_path=initial_path,
        results_path=ml_results_path,
        station_item="station_5_1",
        bidders=[16, 19, 36],
        out_csv_path=schedule_csv,
    )
    print(f"Saved schedule table CSV: {schedule_csv}")


if __name__ == "__main__":
    main()