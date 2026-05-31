import os
import re
import json
import math
from typing import List, Tuple, Set

import matplotlib
import matplotlib.pyplot as plt
import parse_allocations

# 为了与现有脚本一致，输出到自定义plots目录
OUT_DIR = "/Users/y./Documents/ML-CCA-main/src/wandb_custom_plots/results"

# 与需求图保持一致：若此集合非空，则使用此手动集合作为高亮项
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


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json_relaxed(path: str) -> dict:
    """
    读取JSON，允许文件中出现 NaN/Infinity/-Infinity（替换为 null）以避免解析失败。
    """
    with open(path, "r") as f:
        txt = f.read()
    # 规整非标准常量
    txt = re.sub(r"(?<![\w\-])NaN(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])Infinity(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])-Infinity(?![\w\-])", "null", txt)
    return json.loads(txt)


def extract_price_vector_for_round(results: dict, round_num: int) -> Tuple[int, List[float]]:
    """
    从 results 中提取指定迭代(round_num)的价格向量；若缺失，回退为最大迭代键。
    返回 (使用的迭代号, 价格列表)
    """
    pv = results.get("Price Vector per Iteration", {})
    if not isinstance(pv, dict) or not pv:
        return round_num, []
    # 将键转换为整数编号
    try:
        keys = sorted(int(k) for k in pv.keys())
    except Exception:
        keys = [round_num]
    use_round = round_num if str(round_num) in pv else (keys[-1] if keys else round_num)
    arr = pv.get(str(use_round), [])
    return use_round, [float(x) for x in arr]


def build_goods_list_from_auction(auction_path: str) -> List[str]:
    """
    基于 auction_instance.json 生成标准 goods 顺序：station_i_j（按station编号升序、slot 0..num_slots-1）。
    若文件缺失，则按 MSVM 默认（6站×24槽）。
    """
    if auction_path and os.path.exists(auction_path):
        data = load_json_relaxed(auction_path)
        stations = data.get("stations", [])
        num_slots = int(data.get("num_slots", 24))

        def station_num(s):
            try:
                sid = str(s.get("id", "station_0"))
                return int(sid.split("_")[-1])
            except Exception:
                return 0

        stations_sorted = sorted(stations, key=station_num)
        goods = []
        for s in stations_sorted:
            sid = str(s.get("id", "station_0"))
            for t in range(num_slots):
                goods.append(f"{sid}_{t}")
        return goods

    # Fallback: 6 stations × 24 slots
    return [f"station_{i}_{j}" for i in range(6) for j in range(24)]


def sort_items(items: List[str]) -> List[str]:
    """按 station 编号+slot 编号排序。"""
    def key(item: str):
        m = re.match(r"station_(\d+)_(\d+)", item)
        if m:
            return (int(m.group(1)), int(m.group(2)), item)
        return (999999, 999999, item)

    return sorted(items, key=key)


def plot_round6_prices_compare(
    path_cca: str,
    path_mlcca: str,
    auction_path: str,
    out_png: str,
    round_num: int = 6,
):
    ensure_out_dir(os.path.dirname(out_png))

    data_cca = load_json_relaxed(path_cca)
    data_mlcca = load_json_relaxed(path_mlcca)

    used_round_cca, prices_cca = extract_price_vector_for_round(data_cca, round_num)
    used_round_mlcca, prices_mlcca = extract_price_vector_for_round(data_mlcca, round_num)

    goods = build_goods_list_from_auction(auction_path)

    # 对齐长度（有时价格向量长度可能与 goods 不一致）
    n = min(len(goods), len(prices_cca), len(prices_mlcca))
    goods = goods[:n]
    prices_cca = prices_cca[:n]
    prices_mlcca = prices_mlcca[:n]

    # 排序并同步价格顺序
    items_sorted = sort_items(goods)
    idx_map = {item: i for i, item in enumerate(goods)}
    cca_sorted = [prices_cca[idx_map[item]] for item in items_sorted]
    mlcca_sorted = [prices_mlcca[idx_map[item]] for item in items_sorted]

    # 计算高亮集合：全局差集（第 round_num 轮 ML-CCA 分配的所有 item）−（CCA 分配的所有 item）。
    def extract_ml_only_items_global(results_ml_path: str, results_cca_path: str, auction_path: str, round_idx: int) -> Set[str]:
        try:
            parsed_ml = parse_allocations.parse_allocation_for_iteration(results_ml_path, auction_path, round_idx) or {}
            parsed_cca = parse_allocations.parse_allocation_for_iteration(results_cca_path, auction_path, round_idx) or {}
        except Exception:
            return set()

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
        return ml_items - cca_items

    # 优先使用手动集合；为空时才根据 allocations 解析 ML-only 集合
    ml_only_items = set(MANUAL_HIGHLIGHT_ITEMS or set())
    if not ml_only_items:
        ml_only_items = extract_ml_only_items_global(path_mlcca, path_cca, auction_path, round_num)

    # 绘图：每个item两根并排柱状（CCA vs ML-CCA），并在ML-only处高亮
    width = 0.45
    xs = list(range(n))
    fig_w = max(12.0, n / 6.0)
    fig_h = 5.0
    plt.figure(figsize=(fig_w, fig_h), facecolor="w")
    ax = plt.gca()

    cca_color = "#4472C4"   
    mlcca_color = "#FFC000" 
    highlight_color = "#ED7D31"  

    ax.bar([x - width/2 for x in xs], cca_sorted, width=width, color=cca_color, edgecolor="#444444", linewidth=0.2, label="CCA")

    # 为ML-CCA每个柱子设置颜色：ML-only使用高亮色
    ml_colors = [highlight_color if item in ml_only_items else mlcca_color for item in items_sorted]
    ax.bar([x + width/2 for x in xs], mlcca_sorted, width=width, color=ml_colors, edgecolor="#444444", linewidth=0.2, label="EV-MLCCA")

    # ax.set_title(f"Item Price Comparison: CCA vs ML-CCA", fontsize=20)
    ax.set_ylabel("Price", fontsize=16)
    ax.set_xlabel("Item", fontsize=16)
    ax.set_xticks(xs)
    ax.set_xticklabels(items_sorted, rotation=90, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    # 构建图例（包含ML-only高亮说明）
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=cca_color, label="CCA"),
        mpatches.Patch(color=mlcca_color, label="EV-MLCCA"),
        mpatches.Patch(color=highlight_color, label="EV-MLCCA allocated only"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    # 去除左右空隙：让首个与最后一个柱子紧贴左右边界
    # 组合柱：最左边缘位于 -width，最右边缘位于 (n-1) + width
    ax.margins(x=0)
    ax.set_xlim(-width, (n - 1) + width)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    # 两个results.json路径（用户提供）
    path_cca = \
        "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/results.json"
    path_mlcca = \
        "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_gd_linear_prices_on_W_v3/ML_config_hpo1/8/results.json"

    # 与其他图保持一致：使用用户实验的 seed_8 auction 实例
    auction_path = "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/auction_instance_seed_8.json"
    out_png = os.path.join(OUT_DIR, "round6_item_price_compare_cca_vs_mlcca_consistent.png")

    # 嵌入字体兼容性设置（PDF/PS）；PNG不受影响，但保持一致性
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    plot_round6_prices_compare(path_cca, path_mlcca, auction_path, out_png, round_num=6)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()