import argparse
import json
import os
from typing import Dict, Set

import parse_allocations


def load_allocation_round_keys(results_path: str) -> Set[int]:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    alloc = data.get("Allocation per Iteration") or data.get("allocation per iteration") or {}
    keys = set()
    if isinstance(alloc, dict):
        for k in alloc.keys():
            try:
                keys.add(int(str(k)))
            except Exception:
                pass
    return keys


def bidder_items_set_map(results_path: str, auction_path: str, round_no: int) -> Dict[str, Set[str]]:
    parsed = parse_allocations.parse_allocation_for_iteration(results_path, auction_path, round_no) or {}
    out: Dict[str, Set[str]] = {}
    for bidder_key, info in parsed.items():
        items = [str(x) for x in (info.get("items") or [])]
        out[str(bidder_key)] = set(items)
    return out


def compare_round_vs_final(results_path: str, auction_path: str, round_no: int):
    keys = load_allocation_round_keys(results_path)
    if not keys:
        print("未找到任何轮次的分配数据。")
        return
    last_round = max(keys)

    m_round = bidder_items_set_map(results_path, auction_path, round_no)
    m_last = bidder_items_set_map(results_path, auction_path, last_round)

    equal = (m_round == m_last)
    print(f"结果文件: {results_path}")
    print(f"对比轮次: round={round_no} vs last_round={last_round}")
    print(f"是否完全一致: {equal}")

    if not equal:
        bidders_all = sorted(set(m_round.keys()) | set(m_last.keys()), key=lambda k: int(str(k).split('_')[-1]) if '_' in str(k) else 0)
        diffs = []
        for b in bidders_all:
            s6 = m_round.get(b, set())
            sl = m_last.get(b, set())
            if s6 != sl:
                only_6 = sorted(list(s6 - sl))
                only_last = sorted(list(sl - s6))
                diffs.append((b, only_6, only_last))

        if not diffs:
            # 键不同但集合相同的极端情况
            print("键集合不同，但每个投标者的items集合比较无差异。")
        else:
            print("发现以下投标者在两轮分配中存在差异：")
            for b, o6, ol in diffs:
                print(f"- {b}: 仅在round{round_no}有 {o6}; 仅在round{last_round}有 {ol}")


def main():
    parser = argparse.ArgumentParser(description="比较第N轮与最后一轮的 CCA/ML-CCA 分配是否一致")
    parser.add_argument("--results", required=True, help="results.json 路径（CCA 或 ML-CCA）")
    parser.add_argument("--auction", required=True, help="auction_instance.json 路径，用于物品映射")
    parser.add_argument("--round", type=int, default=6, help="要比较的轮次，默认 6")
    args = parser.parse_args()

    compare_round_vs_final(args.results, args.auction, args.round)


if __name__ == "__main__":
    main()