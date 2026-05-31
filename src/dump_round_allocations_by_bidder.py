import argparse
import csv
import os
from typing import Dict, List

import parse_allocations


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_round_allocations(results_path: str, auction_path: str, round_no: int) -> Dict[str, List[str]]:
    """
    返回 bidder -> items 的映射（items 为字符串列表），使用项目内的解析函数以保证与其他脚本一致。
    并按 auction 中的 goods 顺序进行排序。
    """
    parsed = parse_allocations.parse_allocation_for_iteration(results_path, auction_path, round_no) or {}
    goods, _ = parse_allocations.build_goods_list_from_auction(auction_path)
    order = {str(g): i for i, g in enumerate(goods)}
    out = {}
    for bidder_key, info in parsed.items():
        items = [str(x) for x in (info.get("items") or [])]
        items_sorted = sorted(items, key=lambda it: order.get(str(it), 10**9))
        out[str(bidder_key)] = items_sorted
    return out


def write_csv(alloc_cca: Dict[str, List[str]], alloc_ml: Dict[str, List[str]], out_path: str):
    ensure_out_dir(os.path.dirname(out_path))
    # 统一 bidder 列表
    bidders = sorted(set(list(alloc_cca.keys()) + list(alloc_ml.keys())), key=lambda k: int(str(k).split('_')[-1]) if '_' in str(k) else 0)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bidder", "items_cca", "items_mlcca", "count_cca", "count_mlcca"])  # 表头
        for b in bidders:
            items_cca = alloc_cca.get(b, []) or []
            items_ml = alloc_ml.get(b, []) or []
            w.writerow([
                b,
                ";".join(items_cca),
                ";".join(items_ml),
                len(items_cca),
                len(items_ml),
            ])


def main():
    parser = argparse.ArgumentParser(description="导出第N轮中 ML-CCA 与 CCA 的 bidder->items 对比表")
    parser.add_argument("--auction", required=True, help="auction_instance.json 路径")
    parser.add_argument("--cca_results", required=True, help="CCA results.json 路径")
    parser.add_argument("--mlcca_results", required=True, help="ML-CCA results.json 路径")
    parser.add_argument("--round", type=int, default=6, help="轮次，默认 6")
    parser.add_argument("--outdir", default=os.path.join("src", "wandb_custom_plots", "results"), help="输出目录")
    args = parser.parse_args()

    # 解析两份分配（并排序）
    alloc_cca = parse_round_allocations(args.cca_results, args.auction, args.round)
    alloc_ml = parse_round_allocations(args.mlcca_results, args.auction, args.round)

    # 导出 CSV
    out_csv = os.path.join(args.outdir, f"round{args.round}_bidder_allocations_compare.csv")
    write_csv(alloc_cca, alloc_ml, out_csv)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()