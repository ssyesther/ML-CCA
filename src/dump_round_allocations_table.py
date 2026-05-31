import argparse
import csv
import os
import json
from typing import Dict, List, Tuple

import parse_allocations


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_round_allocations(results_path: str, auction_path: str, round_no: int) -> Dict[str, List[str]]:
    """
    返回 bidder -> items 的映射（items 为字符串列表），使用项目内的解析函数以保证与其他脚本一致。
    """
    parsed = parse_allocations.parse_allocation_for_iteration(results_path, auction_path, round_no) or {}
    out = {}
    for bidder_key, info in parsed.items():
        items = [str(x) for x in (info.get("items") or [])]
        out[str(bidder_key)] = items
    return out


def invert_to_item_bidder_map(alloc_map: Dict[str, List[str]]) -> Dict[str, str]:
    """
    将 bidder->items 反转为 item->bidder（若同一 item 被多个 bidder 标记，保留第一个出现）。
    """
    item_to_bidder: Dict[str, str] = {}
    for bidder, items in alloc_map.items():
        for it in items or []:
            it = str(it)
            if it and it not in item_to_bidder:
                item_to_bidder[it] = str(bidder)
    return item_to_bidder


def build_compare_rows(item_to_bidder_cca: Dict[str, str], item_to_bidder_ml: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    构建对比行：item, bidder_cca, bidder_mlcca。包含两个集合的并集。
    """
    all_items = sorted(set(list(item_to_bidder_cca.keys()) + list(item_to_bidder_ml.keys())))
    rows: List[Tuple[str, str, str]] = []
    for it in all_items:
        rows.append((it, item_to_bidder_cca.get(it, ""), item_to_bidder_ml.get(it, "")))
    return rows


def write_csv(rows: List[Tuple[str, str, str]], out_path: str):
    ensure_out_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "bidder_cca", "bidder_mlcca"])  # 表头
        for it, b_cca, b_ml in rows:
            w.writerow([it, b_cca, b_ml])


def main():
    parser = argparse.ArgumentParser(description="导出第N轮中 ML-CCA 与 CCA 的 item->bidder 对比表")
    parser.add_argument("--auction", required=True, help="auction_instance.json 路径")
    parser.add_argument("--cca_results", required=True, help="CCA results.json 路径")
    parser.add_argument("--mlcca_results", required=True, help="ML-CCA results.json 路径")
    parser.add_argument("--round", type=int, default=6, help="轮次，默认 6")
    parser.add_argument("--outdir", default=os.path.join("src", "wandb_custom_plots", "results"), help="输出目录")
    args = parser.parse_args()

    # 解析两份分配
    alloc_cca = parse_round_allocations(args.cca_results, args.auction, args.round)
    alloc_ml = parse_round_allocations(args.mlcca_results, args.auction, args.round)

    # 转为 item->bidder
    item_to_bidder_cca = invert_to_item_bidder_map(alloc_cca)
    item_to_bidder_ml = invert_to_item_bidder_map(alloc_ml)

    # 构建对比并导出
    rows = build_compare_rows(item_to_bidder_cca, item_to_bidder_ml)
    out_csv = os.path.join(args.outdir, f"round{args.round}_item_allocations_compare.csv")
    write_csv(rows, out_csv)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()