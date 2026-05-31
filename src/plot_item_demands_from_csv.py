import argparse
import csv
import os
import re
import json
from collections import defaultdict
from typing import Dict, Set, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import parse_allocations

# 手动高亮集合：如非空则覆盖 ML-only 自动推断
MANUAL_HIGHLIGHT_ITEMS: Set[str] = {
    "station_1_11",
    "station_0_23",
    "station_2_1",
    "station_3_18",
    "station_3_17",
    "station_5_10",
    "station_5_6",
    "station_1_21",
    "station_4_8",
    "station_4_20",
}


def parse_csv(csv_path: str) -> Dict[str, Set[int]]:
    """
    Build mapping: item -> set(bidder_ids) from CSV rows.
    CSV columns: bidder_id,item,value
    """
    item_to_bidders: Dict[str, Set[int]] = defaultdict(set)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bidder_id = int(row["bidder_id"]) if "bidder_id" in row else int(row.get("bidder", "0"))
            except ValueError:
                # Skip malformed rows
                continue
            item = row.get("item")
            if not item:
                continue
            item_to_bidders[item].add(bidder_id)
    return item_to_bidders


def load_item_to_bidders_from_allowed_slots(auction_path: str) -> Dict[str, Set[int]]:
    """
    从 auction_instance.json 的每个 bidder 的 "allowed_slots" 统计：item -> 需求该 item 的投标者集合。
    bidder 标识使用其 id 的数字后缀（例如 Bidder_23 -> 23）。
    """
    item_to_bidders: Dict[str, Set[int]] = defaultdict(set)
    if not auction_path or not os.path.exists(auction_path):
        return {}
    with open(auction_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bidders = data.get("bidders", []) or []
    for b in bidders:
        bid = str(b.get("id", "Bidder_0"))
        try:
            bnum = int(bid.split("_")[-1])
        except Exception:
            bnum = len(item_to_bidders)  # fallback index
        slots = b.get("allowed_slots", []) or []
        for it in slots:
            if it:
                item_to_bidders[str(it)].add(bnum)
    return item_to_bidders

def build_weighted_demand_counts_from_allowed_slots(auction_path: str) -> Dict[str, float]:
    """
    基于 auction_instance.json 的每个 bidder 的 allowed_slots 计算加权需求：
    每个 bidder 对其 allowed 的每个 item 贡献 1/len(allowed_slots)。
    """
    item_to_count: Dict[str, float] = defaultdict(float)
    if not auction_path or not os.path.exists(auction_path):
        return {}
    with open(auction_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bidders = data.get("bidders", []) or []
    for b in bidders:
        slots = (b.get("allowed_slots", []) or [])
        m = len(slots)
        if m <= 0:
            continue
        contrib = 1.0 / float(m)
        for it in slots:
            if it:
                item_to_count[str(it)] += contrib
    return item_to_count


def build_unweighted_demand_counts_from_allowed_slots(auction_path: str) -> Dict[str, float]:
    """
    基于 auction_instance.json 的每个 bidder 的 allowed_slots 计算非加权需求：
    每个 bidder 对其 allowed 的每个 item 贡献 1。（与早期版本一致）
    """
    item_to_count: Dict[str, float] = defaultdict(float)
    if not auction_path or not os.path.exists(auction_path):
        return {}
    with open(auction_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bidders = data.get("bidders", []) or []
    for b in bidders:
        slots = (b.get("allowed_slots", []) or [])
        for it in slots:
            if it:
                item_to_count[str(it)] += 1.0
    return item_to_count


def sort_items(items: List[str]) -> List[str]:
    """
    Sort items by station then slot when pattern matches; otherwise lexicographically.
    Items look like 'station_4_6'.
    """
    def key(item: str) -> Tuple[int, int, str]:
        m = re.match(r"station_(\d+)_(\d+)", item)
        if m:
            return (int(m.group(1)), int(m.group(2)), item)
        return (999999, 999999, item)

    return sorted(items, key=key)


def build_goods_list_from_auction(auction_path: str) -> List[str]:
    """
    基于 auction_instance.json 生成完整 goods 列表：station_i_j（按station编号升序、slot 0..num_slots-1）。
    若文件缺失或解析失败，则按默认 6×24 生成。
    """
    try:
        if auction_path and os.path.exists(auction_path):
            with open(auction_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stations = data.get("stations", [])
            num_slots = int(data.get("num_slots", 24))

            def station_num(s):
                try:
                    sid = str(s.get("id", "station_0"))
                    return int(sid.split("_")[-1])
                except Exception:
                    return 0

            stations_sorted = sorted(stations, key=station_num)
            goods: List[str] = []
            for s in stations_sorted:
                sid = str(s.get("id", "station_0"))
                for t in range(num_slots):
                    goods.append(f"{sid}_{t}")
            return goods
    except Exception:
        pass

    # Fallback: 默认 6 站 × 24 槽
    return [f"station_{i}_{j}" for i in range(6) for j in range(24)]


def derive_run_id_from_filename(path: str) -> str:
    # e.g., demands_run-20250907_160052-bhaz7tsu_bidders0-49.csv
    m = re.search(r"demands_(run-[^_]+_[^/]+)_bidders", os.path.basename(path))
    if m:
        return m.group(1)
    # fallback: try generic run-* pattern
    m2 = re.search(r"(run-[^/]+)", path)
    return m2.group(1) if m2 else "unknown_run"


def get_num_slots_from_auction(auction_path: str, default_slots: int = 24) -> int:
    """
    读取 auction_instance.json 的 num_slots 字段用于时段聚合；失败则返回默认值。
    """
    try:
        if auction_path and os.path.exists(auction_path):
            with open(auction_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data.get("num_slots", default_slots))
    except Exception:
        pass
    return default_slots


def aggregate_counts_by_slot(item_to_counts: Dict[str, float], num_slots: int = 24) -> Dict[int, float]:
    """
    将各 station_i_j 的需求计数按 j（time slot）聚合为跨所有站的总需求。
    """
    slot_to_count: Dict[int, float] = defaultdict(float)
    for it, val in item_to_counts.items():
        m = re.match(r"station_(\d+)_(\d+)", str(it))
        if not m:
            continue
        slot = int(m.group(2))
        slot_to_count[slot] += float(val)
    # 补齐缺失的时段为0
    for s in range(num_slots):
        if s not in slot_to_count:
            slot_to_count[s] = 0.0
    return slot_to_count


def write_slots_csv(slot_to_counts: Dict[int, float], outdir: str, run_id: str, weighted: bool) -> str:
    os.makedirs(outdir, exist_ok=True)
    prefix = "slot_weighted_demand" if weighted else "slot_demand_counts"
    out_csv = os.path.join(outdir, f"{prefix}_{run_id}.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        header = "weighted_demand" if weighted else "demand_count"
        f.write(f"slot,{header}\n")
        for s in sorted(slot_to_counts.keys()):
            f.write(f"{s},{float(slot_to_counts.get(s, 0.0)):.6f}\n")
    return out_csv


def plot_slot_demands(slot_to_counts: Dict[int, float], outdir: str, run_id: str, *, label_mode: str = "counts") -> str:
    """
    绘制按时段聚合后的需求柱状图（跨所有站总和）。
    """
    os.makedirs(outdir, exist_ok=True)
    slots_sorted = sorted(slot_to_counts.keys())
    counts = [float(slot_to_counts[s]) for s in slots_sorted]

    width = max(12, 0.45 * len(slots_sorted))
    height = 5.5
    plt.figure(figsize=(width, height))
    base_color = "#e15759"  # 暖色方案：柔和红
    x_pos = list(range(len(slots_sorted)))
    plt.bar(x_pos, counts, color=base_color, alpha=0.9, width=0.8, edgecolor="none")

    ax = plt.gca()
    ax.margins(x=0)
    ax.set_xlim(-0.4, (len(slots_sorted) - 1) + 0.4)
    ax.set_facecolor("#ffffff")  # 纯白背景
    # 移除网格
    ax.grid(False)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)

    plt.xticks(x_pos, [str(s) for s in slots_sorted], rotation=0, fontsize=11)
    plt.yticks(fontsize=11)
    if label_mode == "weighted":
        plt.ylabel("Weighted Demand", fontsize=16)
    else:
        plt.ylabel("Demand Counts", fontsize=16)
    plt.xlabel("Time Slots", fontsize=16)
    plt.tight_layout()

    prefix = "slot_weighted_demand" if label_mode == "weighted" else "slot_demand_counts"
    out_path = os.path.join(outdir, f"{prefix}_{run_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_item_demands_weighted(item_to_counts: Dict[str, float], outdir: str, run_id: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    items = sort_items(list(item_to_counts.keys()))
    counts = [float(item_to_counts.get(it, 0.0)) for it in items]

    # Figure size adaptive to number of items
    width = max(12, 0.25 * len(items))
    height = 6

    plt.figure(figsize=(width, height))
    bars = plt.bar(range(len(items)), counts, color="#4C78A8")
    # 去除左右空隙：默认柱宽为0.8，因此边界应设为 -0.4 与 (n-1)+0.4
    ax = plt.gca()
    ax.margins(x=0)
    ax.set_xlim(-0.4, (len(items) - 1) + 0.4)
    plt.xticks(range(len(items)), items, rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    plt.ylabel("Weighted Demand", fontsize=30)
    plt.xlabel("Items", fontsize=30)
    plt.title("Item Weighted Demand (1/allowed_slots per bidder)", fontsize=40)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(outdir, f"item_weighted_demand_{run_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def write_weighted_counts_csv(item_to_counts: Dict[str, float], outdir: str, run_id: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, f"item_weighted_demand_{run_id}.csv")
    items = sort_items(list(item_to_counts.keys()))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("item,weighted_demand\n")
        for it in items:
            f.write(f"{it},{float(item_to_counts.get(it, 0.0)):.6f}\n")
    return out_csv


def plot_item_demands_with_highlight(
    item_to_counts: Dict[str, float],
    outdir: str,
    run_id: str,
    auction_path: str,
    cca_results_path: str,
    mlcca_results_path: str,
    round_no: int,
    *,
    label_mode: str = "weighted",  # "counts" | "weighted"
    use_manual_highlight: bool = True,
    output_prefix: str = "item_weighted_demand",
) -> str:
    """
    绘制需求柱状图，并以橙色在同一位置叠加高亮 ML-CCA only 分配的物品。
    """
    os.makedirs(outdir, exist_ok=True)
    items = sort_items(list(item_to_counts.keys()))
    counts = [float(item_to_counts.get(it, 0.0)) for it in items]

    width = max(12, 0.25 * len(items))
    height = 6
    plt.figure(figsize=(width, height))
    x_pos = list(range(len(items)))

    # 解析 allocations，定位 ML-CCA allocated-only；仅在启用手动高亮时使用预设集合
    ml_only: Set[str] = set(MANUAL_HIGHLIGHT_ITEMS if use_manual_highlight else set())
    if not ml_only:
        try:
            if cca_results_path and mlcca_results_path:
                parsed_ml = parse_allocations.parse_allocation_for_iteration(mlcca_results_path, auction_path, round_no) or {}
                parsed_cca = parse_allocations.parse_allocation_for_iteration(cca_results_path, auction_path, round_no) or {}

                ml_items: Set[str] = set()
                cca_items: Set[str] = set()
                for _, info_ml in parsed_ml.items():
                    for it in (info_ml.get("items", []) or []):
                        if it:
                            ml_items.add(str(it))
                for _, info_cca in parsed_cca.items():
                    for it in (info_cca.get("items", []) or []):
                        if it:
                            cca_items.add(str(it))
                ml_only = ml_items - cca_items
        except Exception:
            ml_only = set()

    # 直接改变被高亮项的柱子颜色（即便计数为0也可见）
    base_color = "#4472C4"
    highlight_color = "#ED7D31"
    colors = [highlight_color if it in ml_only else base_color for it in items]
    plt.bar(x_pos, counts, color=colors, width=0.8, edgecolor="#333333", linewidth=0.2)
    # 去除左右空隙：柱宽0.8 -> 左右各0.4
    ax = plt.gca()
    ax.margins(x=0)
    ax.set_xlim(-0.4, (len(items) - 1) + 0.4)

    plt.xticks(x_pos, items, rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    if label_mode == "counts":
        plt.ylabel("Demand Counts", fontsize=20)
        plt.xlabel("Items", fontsize=20)
        # plt.title("Item Demand Counts", fontsize=25)
    else:
        plt.ylabel("Weighted Demand", fontsize=20)
        plt.xlabel("Items", fontsize=20)
        plt.title("Item Weighted Demand", fontsize=25)
    plt.grid(axis="y", alpha=0.3)
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=base_color, label=("Demand" if label_mode == "counts" else "Demand")),
        mpatches.Patch(color=highlight_color, label="EV-MLCCA allocated only"),
    ]
    plt.legend(handles=handles, loc="upper right", fontsize=15)
    plt.tight_layout()

    out_path = os.path.join(outdir, f"{output_prefix}_{run_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot item demand counts (CSV optional; default from allowed_slots)")
    parser.add_argument("--csv", required=False, help="Path to demands CSV (optional)")
    parser.add_argument(
        "--auction",
        default="/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/auction_instance_seed_8.json",
        help="Path to auction_instance.json or specific seed instance for complete item list",
    )
    parser.add_argument("--from_allowed_slots", action="store_true", help="Compute demand from allowed_slots in auction_instance")
    parser.add_argument("--cca_results", help="Path to baseline CCA results.json for allocation parsing")
    parser.add_argument("--mlcca_results", help="Path to ML-CCA results.json for allocation parsing")
    parser.add_argument("--round", type=int, default=6, help="Round number used for allocations")
    parser.add_argument("--unweighted", action="store_true", help="Use unweighted counts (1 per allowed item per bidder)")
    parser.add_argument("--use_manual_highlight", action="store_true", help="Use built-in manual highlight item set")
    parser.add_argument("--plot_slots", action="store_true", help="Aggregate demand by time slot across all stations")
    parser.add_argument(
        "--outdir",
        default=os.path.join("src", "wandb_custom_plots", "results"),
        help="Directory to write outputs",
    )
    args = parser.parse_args()

    # 建立加权需求映射：优先从 allowed_slots；若提供 CSV 且未指定 from_allowed_slots 则将 CSV 聚合为未加权计数后转为 float
    mode = "weighted"
    output_prefix = "item_weighted_demand"

    if args.from_allowed_slots or not args.csv:
        if args.unweighted:
            item_to_counts = build_unweighted_demand_counts_from_allowed_slots(args.auction)
            run_id = f"allowed_slots_unweighted_{os.path.basename(args.auction).replace('.json','')}"
            mode = "counts"
            output_prefix = "item_demand_counts"
        else:
            item_to_counts = build_weighted_demand_counts_from_allowed_slots(args.auction)
            run_id = f"allowed_slots_weighted_{os.path.basename(args.auction).replace('.json','')}"
    else:
        item_to_bidders = parse_csv(args.csv)
        item_to_counts = {it: float(len(bset)) for it, bset in item_to_bidders.items()}
        run_id = derive_run_id_from_filename(args.csv)
        mode = "counts"
        output_prefix = "item_demand_counts"
    # 补齐所有可能的物品，包括各站的第0个槽位，使其以0计数出现
    goods_all = build_goods_list_from_auction(args.auction)
    for g in goods_all:
        if g not in item_to_counts:
            item_to_counts[g] = 0.0
    # 写 CSV 文件名与模式一致
    def write_counts_csv(item_to_counts: Dict[str, float], outdir: str, run_id: str, weighted: bool) -> str:
        os.makedirs(outdir, exist_ok=True)
        prefix = "item_weighted_demand" if weighted else "item_demand_counts"
        out_csv = os.path.join(outdir, f"{prefix}_{run_id}.csv")
        items = sort_items(list(item_to_counts.keys()))
        with open(out_csv, "w", encoding="utf-8") as f:
            header = "weighted_demand" if weighted else "demand_count"
            f.write(f"item,{header}\n")
            for it in items:
                f.write(f"{it},{float(item_to_counts.get(it, 0.0)):.6f}\n")
        return out_csv

    counts_csv = write_counts_csv(item_to_counts, args.outdir, run_id, weighted=(mode == "weighted"))
    # 始终使用高亮绘制；若未提供results路径，则仅使用手动高亮集合
    png_path = plot_item_demands_with_highlight(
        item_to_counts,
        args.outdir,
        run_id,
        args.auction,
        args.cca_results or "",
        args.mlcca_results or "",
        args.round,
        label_mode=mode,
        use_manual_highlight=args.use_manual_highlight,
        output_prefix=output_prefix,
    )

    # 若启用按时段聚合，则生成对应 CSV 和 PNG
    if args.plot_slots:
        num_slots = get_num_slots_from_auction(args.auction)
        slot_to_counts = aggregate_counts_by_slot(item_to_counts, num_slots=num_slots)
        slots_csv = write_slots_csv(slot_to_counts, args.outdir, run_id, weighted=(mode == "weighted"))
        slots_png = plot_slot_demands(slot_to_counts, args.outdir, run_id, label_mode=("weighted" if mode == "weighted" else "counts"))
        print(slots_csv)
        print(slots_png)

    print("Wrote:")
    print(counts_csv)
    print(png_path)


if __name__ == "__main__":
    main()