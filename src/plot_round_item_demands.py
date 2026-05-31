import os
import re
import json
import argparse
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt

from custom_msvm_domain import CustomMSVMAuction
import parse_allocations


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json_relaxed(path: str) -> dict:
    """
    读取JSON，允许文件中出现 NaN/Infinity/-Infinity（替换为 null）以避免解析失败。
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    txt = re.sub(r"(?<![\w\-])NaN(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])Infinity(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])-Infinity(?![\w\-])", "null", txt)
    return json.loads(txt)


def extract_initial_qinit_prices(initial_path: str) -> Tuple[int, List[List[float]]]:
    """
    从 initial_cca_result.json 中提取前 Qinit 轮的 price_vector 列表。
    返回 (Qinit, [price_vector_round1, ..., price_vector_roundQinit])。
    若存在缺失或类型不符，跳过该轮。
    """
    data = load_json_relaxed(initial_path)
    qinit = int(data.get("Qinit", 50))
    rounds = data.get("clock_rounds", []) or []
    try:
        rounds_sorted = sorted(
            rounds,
            key=lambda r: int(r.get("round", 0)) if isinstance(r.get("round", 0), (int, float)) else 0,
        )
    except Exception:
        rounds_sorted = rounds

    prices_per_round: List[List[float]] = []
    for r in rounds_sorted:
        try:
            ridx = int(r.get("round", 0))
        except Exception:
            continue
        if ridx < 1 or ridx > qinit:
            continue
        pv = r.get("price_vector", []) or []
        try:
            arr = [float(x) for x in pv]
        except Exception:
            arr = []
        prices_per_round.append(arr)

    return qinit, prices_per_round


def extract_price_vector_for_round(results: dict, round_num: int) -> Tuple[int, List[float]]:
    """
    从 results 中提取指定迭代(round_num)的价格向量；若缺失，回退为最大迭代键。
    返回 (使用的迭代号, 价格列表)
    """
    pv = results.get("Price Vector per Iteration", {})
    if not isinstance(pv, dict) or not pv:
        return round_num, []
    try:
        keys = sorted(int(k) for k in pv.keys())
    except Exception:
        keys = [round_num]
    use_round = round_num if str(round_num) in pv else (keys[-1] if keys else round_num)
    arr = pv.get(str(use_round), [])
    return use_round, [float(x) for x in arr]


def infer_auction_path_from_results(results_path: str) -> str:
    """
    在 results.json 同目录尝试查找 auction_instance_seed_*.json 或 auction_instance.json。
    找不到时返回默认路径 src/auction_instance.json。
    """
    base_dir = os.path.dirname(os.path.abspath(results_path))
    # 尝试匹配 seed 文件
    for name in os.listdir(base_dir):
        if name.startswith("auction_instance_seed_") and name.endswith(".json"):
            return os.path.join(base_dir, name)
        if name == "auction_instance.json":
            return os.path.join(base_dir, name)
    # 默认回退
    return os.path.join("src", "auction_instance.json")


def sort_items(items: List[str]) -> List[str]:
    def key(item: str):
        m = re.match(r"station_(\d+)_(\d+)", item)
        if m:
            return (int(m.group(1)), int(m.group(2)), item)
        return (999999, 999999, item)
    return sorted(items, key=key)


def compute_round_item_demands(results_path: str, auction_path: str, round_num: int) -> Tuple[List[str], Dict[str, int], Dict[int, List[str]]]:
    """
    计算第 round_num 轮价格下每个 bidder 的最优 bundle（取top-1），并聚合为每个 item 的需求计数。
    返回：
      goods_order: List[str] 全部物品顺序（来自 auction）
      item_counts: Dict[item, count] 需求计数（包含为0的补齐）
      bidder_to_items: Dict[bidder_id, List[item]] 每个投标者的top-1 bundle
    """
    results = load_json_relaxed(results_path)
    used_round, prices = extract_price_vector_for_round(results, round_num)

    # 构建 goods 顺序，以保证与项目内一致
    goods_order, num_slots = parse_allocations.build_goods_list_from_auction(auction_path)

    # 加载 auction_instance
    data = load_json_relaxed(auction_path)
    stations = data.get("stations", [])
    bidders = data.get("bidders", [])
    ns = int(data.get("num_slots", num_slots if num_slots else 24))

    auction = CustomMSVMAuction(stations=stations, bidders=bidders, num_slots=ns)

    # 对齐价格长度与 goods 长度
    n = min(len(goods_order), len(prices))
    goods_order = goods_order[:n]
    prices = prices[:n]

    # 计算每个投标者的 top-1 bundle
    bidder_to_items: Dict[int, List[str]] = {}
    item_counts: Dict[str, int] = {g: 0 for g in goods_order}

    for b in bidders:
        try:
            bid_num = int(str(b.get("id", "Bidder_0")).split("_")[-1])
        except Exception:
            continue
        # 使用 original 接口返回实际物品ID列表
        top_bundles = auction.get_best_bundles_original(bidder_id=bid_num, prices=prices, max_bundle_size=1)
        bundle_items = top_bundles[0] if top_bundles else []
        # 归一为字符串
        bundle_items = [str(it) for it in bundle_items]
        bidder_to_items[bid_num] = bundle_items
        for it in bundle_items:
            if it in item_counts:
                item_counts[it] += 1

    return goods_order, item_counts, bidder_to_items


def compute_item_demands_first_qinit_from_initial(initial_path: str, auction_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    使用 initial_cca_result.json 的前 Qinit 轮价格，计算每轮每个投标者的 top-1 bundle，
    累积得到每个 item 的被选择计数（跨前Qinit轮的总和）。
    返回 (goods_order, item_counts_qinit)
    """
    qinit, prices_rounds = extract_initial_qinit_prices(initial_path)

    # 构建 goods 顺序
    goods_order, num_slots = parse_allocations.build_goods_list_from_auction(auction_path)

    # 加载 auction_instance
    data = load_json_relaxed(auction_path)
    stations = data.get("stations", [])
    bidders = data.get("bidders", [])
    ns = int(data.get("num_slots", num_slots if num_slots else 24))

    auction = CustomMSVMAuction(stations=stations, bidders=bidders, num_slots=ns)

    item_counts_qinit: Dict[str, int] = {g: 0 for g in goods_order}

    for pv in prices_rounds:
        if not pv:
            continue
        n = min(len(goods_order), len(pv))
        goods_n = goods_order[:n]
        prices_n = pv[:n]
        for b in bidders:
            try:
                bid_num = int(str(b.get("id", "Bidder_0")).split("_")[-1])
            except Exception:
                continue
            top_bundles = auction.get_best_bundles_original(bidder_id=bid_num, prices=prices_n, max_bundle_size=1)
            bundle_items = top_bundles[0] if top_bundles else []
            bundle_items = [str(it) for it in bundle_items]
            for it in bundle_items:
                if it in item_counts_qinit:
                    item_counts_qinit[it] += 1

    return goods_order, item_counts_qinit


def compute_item_demands_first_k_from_initial(initial_path: str, auction_path: str, k: int = 10) -> Tuple[List[str], Dict[str, int]]:
    """
    使用 initial_cca_result.json 的前 k 轮价格（k 默认为 10），计算每轮每个投标者的 top-1 bundle，
    累积得到每个 item 的被选择计数（跨前 k 轮的总和）。
    返回 (goods_order, item_counts_first_k)
    """
    qinit, prices_rounds = extract_initial_qinit_prices(initial_path)
    # 只取前 k 轮（若 k 超过 Qinit 则被截断）
    k = max(0, int(k))
    prices_rounds = prices_rounds[:k]

    # 构建 goods 顺序
    goods_order, num_slots = parse_allocations.build_goods_list_from_auction(auction_path)

    # 加载 auction_instance
    data = load_json_relaxed(auction_path)
    stations = data.get("stations", [])
    bidders = data.get("bidders", [])
    ns = int(data.get("num_slots", num_slots if num_slots else 24))

    auction = CustomMSVMAuction(stations=stations, bidders=bidders, num_slots=ns)

    item_counts_first_k: Dict[str, int] = {g: 0 for g in goods_order}

    for pv in prices_rounds:
        if not pv:
            continue
        n = min(len(goods_order), len(pv))
        goods_n = goods_order[:n]
        prices_n = pv[:n]
        for b in bidders:
            try:
                bid_num = int(str(b.get("id", "Bidder_0")).split("_")[-1])
            except Exception:
                continue
            top_bundles = auction.get_best_bundles_original(bidder_id=bid_num, prices=prices_n, max_bundle_size=1)
            bundle_items = top_bundles[0] if top_bundles else []
            bundle_items = [str(it) for it in bundle_items]
            for it in bundle_items:
                if it in item_counts_first_k:
                    item_counts_first_k[it] += 1

    return goods_order, item_counts_first_k


def plot_item_demand_counts(goods_order: List[str], item_counts: Dict[str, int], out_png: str):
    ensure_out_dir(os.path.dirname(out_png))
    # 保持顺序
    xs = list(range(len(goods_order)))
    counts = [int(item_counts.get(it, 0)) for it in goods_order]
    # 排序（按station/slot）
    goods_sorted = sort_items(goods_order)
    idx_map = {it: i for i, it in enumerate(goods_order)}
    counts_sorted = [counts[idx_map[it]] for it in goods_sorted]

    fig_w = max(12.0, len(goods_sorted) / 6.0)
    fig_h = 5.0
    plt.figure(figsize=(fig_w, fig_h), facecolor="w")
    ax = plt.gca()
    x = list(range(len(goods_sorted)))
    ax.bar(x, counts_sorted, width=1.0, align="edge", color="#4C9F70", edgecolor="#444444", linewidth=0.2)
    ax.set_xlim(0, len(goods_sorted))
    ax.set_title("Round 6 Item Demand Counts (Top-1 bundles per bidder)")
    ax.set_ylabel("Demand count (bidders)")
    ax.set_xlabel("Item")
    ax.set_xticks(x)
    ax.set_xticklabels(goods_sorted, rotation=90, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def write_bidder_demands_csv(bidder_to_items: Dict[int, List[str]], out_csv: str):
    import csv
    ensure_out_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bidder", "items", "count"])
        for b in sorted(bidder_to_items.keys()):
            items = bidder_to_items[b]
            w.writerow([b, ";".join(items), len(items)])


def list_round_keys(results: dict) -> List[int]:
    keys: set = set()
    pv = results.get("Price Vector per Iteration") or {}
    if isinstance(pv, dict):
        for k in pv.keys():
            try:
                keys.add(int(str(k)))
            except Exception:
                pass
    alloc = results.get("Allocation per Iteration") or results.get("allocation per iteration") or {}
    if isinstance(alloc, dict):
        for k in alloc.keys():
            try:
                keys.add(int(str(k)))
            except Exception:
                pass
    return sorted(list(keys))


def get_prices_for_round_exact(results: dict, round_num: int) -> List[float]:
    pv = results.get("Price Vector per Iteration") or {}
    arr = pv.get(str(round_num), [])
    try:
        return [float(x) for x in arr]
    except Exception:
        return []


def compute_item_demands_sum_all_rounds(results_path: str, auction_path: str) -> Tuple[List[str], Dict[str, int]]:
    results = load_json_relaxed(results_path)
    goods_order, num_slots = parse_allocations.build_goods_list_from_auction(auction_path)

    data = load_json_relaxed(auction_path)
    stations = data.get("stations", [])
    bidders = data.get("bidders", [])
    ns = int(data.get("num_slots", num_slots if num_slots else 24))

    auction = CustomMSVMAuction(stations=stations, bidders=bidders, num_slots=ns)

    item_counts_total: Dict[str, int] = {g: 0 for g in goods_order}
    rounds = list_round_keys(results)
    for r in rounds:
        prices = get_prices_for_round_exact(results, r)
        if not prices:
            continue
        n = min(len(goods_order), len(prices))
        goods_n = goods_order[:n]
        prices_n = prices[:n]

        for b in bidders:
            try:
                bid_num = int(str(b.get("id", "Bidder_0")).split("_")[-1])
            except Exception:
                continue
            top_bundles = auction.get_best_bundles_original(bidder_id=bid_num, prices=prices_n, max_bundle_size=1)
            bundle_items = top_bundles[0] if top_bundles else []
            bundle_items = [str(it) for it in bundle_items]
            for it in bundle_items:
                if it in item_counts_total:
                    item_counts_total[it] += 1

    return goods_order, item_counts_total


def compute_item_demands_sum_rounds_after_threshold(results_path: str, auction_path: str, threshold: int = 50, offset: int = 0) -> Tuple[List[str], Dict[str, int]]:
    """
    统计 results.json 中迭代轮次严格大于 threshold 的价格下，每轮每个投标者的 top-1，
    累积为每个 item 的被选择计数。默认 threshold=50，即统计第 51 轮及之后。
    返回 (goods_order, item_counts_post_threshold)
    """
    results = load_json_relaxed(results_path)
    goods_order, num_slots = parse_allocations.build_goods_list_from_auction(auction_path)

    data = load_json_relaxed(auction_path)
    stations = data.get("stations", [])
    bidders = data.get("bidders", [])
    ns = int(data.get("num_slots", num_slots if num_slots else 24))

    auction = CustomMSVMAuction(stations=stations, bidders=bidders, num_slots=ns)

    item_counts_total: Dict[str, int] = {g: 0 for g in goods_order}
    rounds = list_round_keys(results)
    for r in rounds:
        # 允许对 ML-CCA 的轮次进行偏移映射（例如 ML 的1..6对应真实的51..56），因此选择条件使用 r+offset
        if (r + int(offset)) <= int(threshold):
            continue
        prices = get_prices_for_round_exact(results, r)
        if not prices:
            continue
        n = min(len(goods_order), len(prices))
        goods_n = goods_order[:n]
        prices_n = prices[:n]

        for b in bidders:
            try:
                bid_num = int(str(b.get("id", "Bidder_0")).split("_")[-1])
            except Exception:
                continue
            top_bundles = auction.get_best_bundles_original(bidder_id=bid_num, prices=prices_n, max_bundle_size=1)
            bundle_items = top_bundles[0] if top_bundles else []
            bundle_items = [str(it) for it in bundle_items]
            for it in bundle_items:
                if it in item_counts_total:
                    item_counts_total[it] += 1

    return goods_order, item_counts_total


def sum_item_counts(goods_order: List[str], *counts_list: Dict[str, int]) -> Dict[str, int]:
    """对齐 goods_order 后将多个计数字典逐项相加。"""
    out: Dict[str, int] = {g: 0 for g in goods_order}
    for counts in counts_list:
        if not counts:
            continue
        for k, v in counts.items():
            if k in out:
                try:
                    out[k] += int(v)
                except Exception:
                    pass
    return out


def plot_item_demand_counts_named(goods_order: List[str], item_counts: Dict[str, int], out_png: str, color: str = "#4C9F70"):
    ensure_out_dir(os.path.dirname(out_png))
    goods_sorted = sort_items(goods_order)
    counts_sorted = [int(item_counts.get(it, 0)) for it in goods_sorted]

    fig_w = max(12.0, len(goods_sorted) / 6.0)
    fig_h = 5.0
    plt.figure(figsize=(fig_w, fig_h), facecolor="w")
    ax = plt.gca()
    x = list(range(len(goods_sorted)))
    ax.bar(x, counts_sorted, width=1.0, align="edge", color=color, edgecolor="#444444", linewidth=0.2)
    ax.set_xlim(0, len(goods_sorted))
    ax.set_ylabel("Demand counts", fontsize=15)
    ax.set_xlabel("Item", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(goods_sorted, rotation=90, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_item_demand_counts_all_rounds(goods_order: List[str], item_counts: Dict[str, int], out_png: str):
    ensure_out_dir(os.path.dirname(out_png))
    goods_sorted = sort_items(goods_order)
    idx_map = {it: i for i, it in enumerate(goods_order)}
    counts_sorted = [int(item_counts.get(it, 0)) for it in goods_sorted]

    fig_w = max(12.0, len(goods_sorted) / 6.0)
    fig_h = 5.0
    plt.figure(figsize=(fig_w, fig_h), facecolor="w")
    ax = plt.gca()
    x = list(range(len(goods_sorted)))
    ax.bar(x, counts_sorted, width=1.0, align="edge", color="#3B82F6", edgecolor="#444444", linewidth=0.2)
    ax.set_xlim(0, len(goods_sorted))
    ax.set_title("All Rounds Item Demand Counts")
    ax.set_ylabel("Total demand count across rounds")
    ax.set_xlabel("Item")
    ax.set_xticks(x)
    ax.set_xticklabels(goods_sorted, rotation=90, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_item_demand_counts_overlay_all_vs_first50(goods_order: List[str],
                                                   counts_all: Dict[str, int],
                                                   counts_first50: Dict[str, int],
                                                   out_png: str):
    """
    在同一图上叠加绘制：
    - 所有轮次的总需求计数（蓝色）
    - 前50轮（或Qinit轮）的被选择计数（橙色）
    按 station_i_j 排序对齐。
    """
    ensure_out_dir(os.path.dirname(out_png))
    goods_sorted = sort_items(goods_order)
    idx_map = {it: i for i, it in enumerate(goods_order)}
    values_all = [int(counts_all.get(it, 0)) for it in goods_sorted]
    values_f50 = [int(counts_first50.get(it, 0)) for it in goods_sorted]

    fig_w = max(12.0, len(goods_sorted) / 6.0)
    fig_h = 5.0
    plt.figure(figsize=(fig_w, fig_h), facecolor="w")
    ax = plt.gca()
    x = list(range(len(goods_sorted)))
    ax.bar(x, values_all, width=1.0, align="edge", color="#3B82F6", edgecolor="#444444", linewidth=0.2, label="All rounds (top-1)")
    ax.bar(x, values_f50, width=0.55, align="edge", color="#ED7D31", alpha=0.85, edgecolor="#444444", linewidth=0.2, label="First 50 rounds (top-1)")
    ax.set_xlim(0, len(goods_sorted))
    # ax.set_title("Item Demand Counts (Top-1 per bidder per round)",fontsize=16)
    ax.set_ylabel("Total demand count",fontsize=12)
    ax.set_xlabel("Item",fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(goods_sorted, rotation=90, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="upper right",fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot item demand counts: single round, sum across all rounds, overlay first-50, or combine first-10 + post-50")
    parser.add_argument("--results", required=True, help="Path to results.json (e.g., ML-CCA run)")
    parser.add_argument("--auction", required=False, help="Path to auction_instance.json (if omitted, inferred from results dir)")
    parser.add_argument("--round", type=str, default="6", help="Round number, or 'all' to sum across all rounds")
    parser.add_argument("--initial", required=False, help="Path to initial_cca_result.json for first-50 overlay")
    parser.add_argument("--overlay", action="store_true", help="If set with --round=all and --initial, produce overlay PNG of all vs first-50")
    parser.add_argument("--combine", action="store_true", help="Combine counts: first 10 rounds (initial) + rounds >50 (results), plotted in single color")
    parser.add_argument("--first_k", type=int, default=10, help="Number of initial rounds to include when --combine is set (default: 10)")
    parser.add_argument("--post_threshold", type=int, default=50, help="Threshold for results rounds when --combine (use >threshold, default: 50)")
    parser.add_argument("--ml_offset", type=int, default=0, help="Round index offset for results when using ML-CCA (e.g., 50 to map 1..6 -> 51..56)")
    parser.add_argument("--outdir", default=os.path.join("src", "wandb_custom_plots", "results"), help="Output directory for PNG/CSV")
    args = parser.parse_args()

    auction_path = args.auction or infer_auction_path_from_results(args.results)

    # 兼容字体
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # 文件名前缀来源于 results 路径
    run_id = os.path.basename(os.path.dirname(os.path.abspath(args.results)))

    # 优先处理合并模式（前10 + 50后），生成单色图
    if args.combine and args.initial:
        goods_order_init, counts_first_k = compute_item_demands_first_k_from_initial(args.initial, auction_path, args.first_k)
        goods_order_post, counts_post = compute_item_demands_sum_rounds_after_threshold(args.results, auction_path, args.post_threshold, offset=args.ml_offset)
        # 使用 auction 的物品顺序（两者应一致）
        goods_order = goods_order_init
        counts_combined = sum_item_counts(goods_order, counts_first_k, counts_post)

        out_png_combined = os.path.join(args.outdir, f"combined_first{args.first_k}_post{args.post_threshold}_item_demands_counts_{run_id}.png")
        out_csv_combined = os.path.join(args.outdir, f"combined_first{args.first_k}_post{args.post_threshold}_item_demands_counts_{run_id}.csv")
        out_csv_first = os.path.join(args.outdir, f"first{args.first_k}_item_demands_counts_{run_id}.csv")
        out_csv_post = os.path.join(args.outdir, f"post>{args.post_threshold}_item_demands_counts_{run_id}.csv")
        # title = f"Item Demand Counts (Top-1 per bidder per round)"
        plot_item_demand_counts_named(goods_order, counts_combined, out_png_combined, color="#4C9F70")

        # 写 CSV（按排序后的物品顺序）
        import csv
        ensure_out_dir(os.path.dirname(out_csv_combined))
        with open(out_csv_combined, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "combined_count"])
            for it in sort_items(goods_order):
                w.writerow([it, int(counts_combined.get(it, 0))])

        # 分项 CSV：前k轮 与 50后轮次
        with open(out_csv_first, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", f"first_{args.first_k}_count"])
            for it in sort_items(goods_order):
                w.writerow([it, int(counts_first_k.get(it, 0))])
        with open(out_csv_post, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", f"post_gt_{args.post_threshold}_count"])
            for it in sort_items(goods_order):
                w.writerow([it, int(counts_post.get(it, 0))])

        print("Wrote:")
        print(out_png_combined)
        print(out_csv_combined)
        print(out_csv_first)
        print(out_csv_post)
        return

    if str(args.round).lower().strip() == "all":
        goods_order, item_counts_total = compute_item_demands_sum_all_rounds(args.results, auction_path)
        out_png = os.path.join(args.outdir, f"all_rounds_item_demands_counts_{run_id}.png")
        out_csv = os.path.join(args.outdir, f"all_rounds_item_demands_counts_{run_id}.csv")

        if args.overlay and args.initial:
            # 叠加前50轮（initial）
            goods_order_init, item_counts_f50 = compute_item_demands_first_qinit_from_initial(args.initial, auction_path)
            # 使用共同的 goods_order（以 auction 为准）；如存在缺少键，直接对齐
            out_png_overlay = os.path.join(args.outdir, f"overlay_all_vs_first50_item_demands_counts_{run_id}.png")
            plot_item_demand_counts_overlay_all_vs_first50(goods_order, item_counts_total, item_counts_f50, out_png_overlay)
            print("Wrote:")
            print(out_png)
            print(out_csv)
            print(out_png_overlay)
        else:
            plot_item_demand_counts_all_rounds(goods_order, item_counts_total, out_png)
            print("Wrote:")
            print(out_png)
            print(out_csv)
        # 将每个 item 的总计数写为 CSV
        import csv
        ensure_out_dir(os.path.dirname(out_csv))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "total_count"])
            for it in sort_items(goods_order):
                w.writerow([it, int(item_counts_total.get(it, 0))])

        # 已在上方打印输出路径
    else:
        round_num = int(args.round)
        goods_order, item_counts, bidder_to_items = compute_round_item_demands(args.results, auction_path, round_num)
        out_png = os.path.join(args.outdir, f"round{round_num}_item_demands_counts_{run_id}.png")
        out_csv = os.path.join(args.outdir, f"round{round_num}_bidder_demands_{run_id}.csv")

        plot_item_demand_counts(goods_order, item_counts, out_png)
        write_bidder_demands_csv(bidder_to_items, out_csv)

        print("Wrote:")
        print(out_png)
        print(out_csv)


if __name__ == "__main__":
    main()