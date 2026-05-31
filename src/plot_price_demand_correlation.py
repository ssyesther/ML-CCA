import os
import re
import json
from typing import List, Tuple, Dict

import matplotlib
import matplotlib.pyplot as plt
import parse_allocations

# 统一输出目录到自定义plots路径
OUT_DIR = "/Users/y./Documents/ML-CCA-main/src/wandb_custom_plots/results"


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json_relaxed(path: str) -> dict:
    """
    读取JSON，允许文件中出现 NaN/Infinity/-Infinity（替换为 null）以避免解析失败。
    """
    with open(path, "r") as f:
        txt = f.read()
    txt = re.sub(r"(?<![\w\-])NaN(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])Infinity(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])\-Infinity(?![\w\-])", "null", txt)
    return json.loads(txt)


def extract_price_vector_for_round(results: dict, round_num: int) -> Tuple[int, List[float]]:
    """从results提取指定迭代的价格向量，若缺失则回退为最大迭代键。"""
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
        goods: List[str] = []
        for s in stations_sorted:
            sid = str(s.get("id", "station_0"))
            for t in range(num_slots):
                goods.append(f"{sid}_{t}")
        return goods

    return [f"station_{i}_{j}" for i in range(6) for j in range(24)]


def sort_items(items: List[str]) -> List[str]:
    """按 station 编号+slot 编号排序。"""
    def key(item: str):
        m = re.match(r"station_(\d+)_(\d+)", item)
        if m:
            return (int(m.group(1)), int(m.group(2)), item)
        return (999999, 999999, item)

    return sorted(items, key=key)


def load_demands_csv(csv_path: str) -> Dict[str, int]:
    """
    加载需求CSV：支持聚合版
    - (item, demand_count)
    - (item, combined_count)
    - (item, total_count)
    若为明细版(bidder_id,item,value)，则按唯一投标者聚合为计数。
    """
    import csv
    item_to_count: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header == ["item", "demand_count"] or header == ["item", "combined_count"] or header == ["item", "total_count"]:
            for item, count in reader:
                try:
                    item_to_count[item] = int(float(count))
                except Exception:
                    item_to_count[item] = 0
        else:
            idx_item = header.index("item") if "item" in header else 1
            idx_bidder = header.index("bidder_id") if "bidder_id" in header else 0
            from collections import defaultdict
            seen: Dict[str, set] = defaultdict(set)
            for row in reader:
                if not row:
                    continue
                item = row[idx_item]
                bidder = row[idx_bidder]
                seen[item].add(bidder)
            item_to_count = {it: len(bidders) for it, bidders in seen.items()}
    return item_to_count


def build_demand_counts_from_allowed_slots(auction_path: str) -> Dict[str, int]:
    """
    基于 auction_instance.json 的每个 bidder 的 allowed_slots 计算 item 的总需求（唯一投标者数量）。
    """
    if not auction_path or not os.path.exists(auction_path):
        return {}
    try:
        data = load_json_relaxed(auction_path)
        bidders = data.get("bidders", []) or []
        counts: Dict[str, int] = {}
        seen: Dict[str, set] = {}
        for b in bidders:
            bid = str(b.get("id", "Bidder_0"))
            try:
                bnum = int(bid.split("_")[-1])
            except Exception:
                bnum = len(seen)
            for it in (b.get("allowed_slots", []) or []):
                k = str(it)
                if k not in seen:
                    seen[k] = set()
                seen[k].add(bnum)
        for k, s in seen.items():
            counts[k] = len(s)
        return counts
    except Exception:
        return {}


def pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0 or len(ys) != n:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx ** 0.5) * (vy ** 0.5)
    if denom == 0:
        return float('nan')
    return cov / denom

def _ranks(values: List[float]) -> List[float]:
    pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    r = 1
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = avg_rank
        r += (j - i + 1)
        i = j + 1
    return ranks

def spearman_r(xs: List[float], ys: List[float]) -> float:
    if not xs or len(xs) != len(ys):
        return float('nan')
    rx = _ranks(xs)
    ry = _ranks(ys)
    return pearson_r(rx, ry)


def plot_price_demand_correlation(
    path_cca: str,
    path_mlcca: str,
    auction_path: str,
    demands_csv: str,
    out_png: str,
    round_num: int = 6,
):
    ensure_out_dir(os.path.dirname(out_png))

    data_cca = load_json_relaxed(path_cca)
    data_mlcca = load_json_relaxed(path_mlcca)
    _, prices_cca = extract_price_vector_for_round(data_cca, round_num)
    _, prices_mlcca = extract_price_vector_for_round(data_mlcca, round_num)

    goods = build_goods_list_from_auction(auction_path)
    demand_counts = load_demands_csv(demands_csv) if demands_csv else build_demand_counts_from_allowed_slots(auction_path)

    # 对齐长度（价格向量与goods可能不一致）
    n = min(len(goods), len(prices_cca), len(prices_mlcca))
    goods = goods[:n]
    prices_cca = prices_cca[:n]
    prices_mlcca = prices_mlcca[:n]

    # 排序并同步价格顺序（使用原goods索引映射），不再过滤 station_0_0
    idx_map = {item: i for i, item in enumerate(goods)}
    items_sorted = sort_items(goods)
    cca_prices_sorted = [prices_cca[idx_map[item]] for item in items_sorted]
    mlcca_prices_sorted = [prices_mlcca[idx_map[item]] for item in items_sorted]
    demands_sorted = [int(demand_counts.get(item, 0)) for item in items_sorted]

    # y轴切换：采用标准化（Z-score），减弱极端值影响
    use_log = False
    standardize = False
    y_raw = demands_sorted
    if standardize:
        m = sum(y_raw) / len(y_raw) if y_raw else 0.0
        var = sum((y - m) ** 2 for y in y_raw) / len(y_raw) if y_raw else 0.0
        sd = (var ** 0.5) if var > 0 else 1.0
        y_plot = [(y - m) / sd for y in y_raw]
        y_label = "Demand (standardized)"
    else:
        y_plot = y_raw[:]
        y_label = "Demand (unique bidders)"
        if use_log:
            y_label = "Demand (symlog)"

    # 绘制散点图（x=price, y=demand）
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # 改为正方形画布比例
    fig_w = 8.0
    fig_h = 8.0
    plt.figure(figsize=(fig_w, fig_h), facecolor="w")
    ax = plt.gca()

    # 使用与现有配色一致的两色
    cca_color = "#4472C4"   # 蓝
    mlcca_color = "#ED7D31"  # 橙

    # 皮尔逊相关（线性相关）
    r_cca = pearson_r(cca_prices_sorted, y_raw)
    r_mlcca = pearson_r(mlcca_prices_sorted, y_raw)

    ax.scatter(cca_prices_sorted, y_plot, s=26, alpha=0.75, color=cca_color,
               edgecolor="#222222", linewidth=0.3, label=f"CCA (Pearson r={r_cca:.3f})")
    ax.scatter(mlcca_prices_sorted, y_plot, s=26, alpha=0.75, color=mlcca_color,
               edgecolor="#222222", linewidth=0.3, label=f"EV-MLCCA (Pearson r={r_mlcca:.3f})")

    # 绘制相关趋势线（线性拟合），与原图风格一致
    def fit_line(xs, ys):
        n = len(xs)
        if n == 0:
            return None
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        if vx == 0:
            return None
        b = cov / vx
        a = my - b * mx
        return a, b

    line_cca = fit_line(cca_prices_sorted, y_plot)
    line_ml = fit_line(mlcca_prices_sorted, y_plot)
    if line_cca:
        a, b = line_cca
        x0, x1 = min(cca_prices_sorted), max(cca_prices_sorted)
        ax.plot([x0, x1], [a + b * x0, a + b * x1], color=cca_color, linestyle="-", alpha=0.6, linewidth=1.2)
    if line_ml:
        a, b = line_ml
        x0, x1 = min(mlcca_prices_sorted), max(mlcca_prices_sorted)
        ax.plot([x0, x1], [a + b * x0, a + b * x1], color=mlcca_color, linestyle="-", alpha=0.6, linewidth=1.2)

    # ax.set_title("Price vs Demand Correlation")
    ax.set_xlabel("Price")
    ax.set_ylabel("Demand")
    # 使用固定坐标范围以与参考图保持一致；并以此确定画布比例
    ax.set_xlim(2.5, 20.0)
    ax.set_ylim(0.0, 25.0)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", fontsize=9)

    # 不添加点标注，保持与原图一致的简洁样式

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_faceted_by_station(
    path_cca: str,
    path_mlcca: str,
    auction_path: str,
    demands_csv: str,
    out_png: str,
    round_num: int = 6,
):
    ensure_out_dir(os.path.dirname(out_png))

    data_cca = load_json_relaxed(path_cca)
    data_mlcca = load_json_relaxed(path_mlcca)
    _, prices_cca = extract_price_vector_for_round(data_cca, round_num)
    _, prices_mlcca = extract_price_vector_for_round(data_mlcca, round_num)

    goods = build_goods_list_from_auction(auction_path)
    demand_counts = load_demands_csv(demands_csv) if demands_csv else build_demand_counts_from_allowed_slots(auction_path)

    n = min(len(goods), len(prices_cca), len(prices_mlcca))
    goods = goods[:n]
    prices_cca = prices_cca[:n]
    prices_mlcca = prices_mlcca[:n]

    invalid_items = {"station_0_0"}
    idx_map = {item: i for i, item in enumerate(goods)}

    # 提取station前缀分组
    import re as _re
    station_groups: Dict[str, List[str]] = {}
    for item in goods:
        if item in invalid_items:
            continue
        m = _re.match(r"^(station_\d+_\d+)", item)
        key = m.group(1) if m else "other"
        station_groups.setdefault(key, []).append(item)

    # 子图布局
    stations = sorted(station_groups.keys())
    cols = min(3, max(1, len(stations)))
    rows = (len(stations) + cols - 1) // cols
    fig_w = max(8.0, cols * 4.0)
    fig_h = max(6.0, rows * 3.5)
    plt.figure(figsize=(fig_w, fig_h), facecolor="w")

    cca_color = "#4472C4"
    mlcca_color = "#ED7D31"

    # y轴使用标准化（Z-score），相关为皮尔逊
    for idx, st in enumerate(stations, start=1):
        ax = plt.subplot(rows, cols, idx)
        items = sort_items(station_groups[st])
        xs_cca = [prices_cca[idx_map[i]] for i in items]
        xs_ml = [prices_mlcca[idx_map[i]] for i in items]
        ys_raw = [int(demand_counts.get(i, 0)) for i in items]
        # Z-score 标准化
        if ys_raw:
            m = sum(ys_raw) / len(ys_raw)
            var = sum((y - m) ** 2 for y in ys_raw) / len(ys_raw)
            sd = (var ** 0.5) if var > 0 else 1.0
            ys_plot = [(y - m) / sd for y in ys_raw]
        else:
            ys_plot = []

        ax.scatter(xs_cca, ys_plot, s=22, alpha=0.75, color=cca_color, edgecolor="#222222", linewidth=0.25,
                   label=f"CCA (Pearson r={pearson_r(xs_cca, ys_raw):.2f})")
        ax.scatter(xs_ml, ys_plot, s=22, alpha=0.75, color=mlcca_color, edgecolor="#222222", linewidth=0.25,
                   label=f"ML-CCA (Pearson r={pearson_r(xs_ml, ys_raw):.2f})")

        # 趋势线（基于当前y_plot）
        def _fit(xs, ys):
            n = len(xs)
            if n == 0:
                return None
            mx = sum(xs) / n
            my = sum(ys) / n
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            vx = sum((x - mx) ** 2 for x in xs)
            if vx == 0:
                return None
            b = cov / vx
            a = my - b * mx
            return a, b

        l1 = _fit(xs_cca, ys_plot)
        l2 = _fit(xs_ml, ys_plot)
        if l1:
            a, b = l1
            x0, x1 = min(xs_cca), max(xs_cca)
            ax.plot([x0, x1], [a + b * x0, a + b * x1], color=cca_color, linestyle="--", alpha=0.5, linewidth=1.0)
        if l2:
            a, b = l2
            x0, x1 = min(xs_ml), max(xs_ml)
            ax.plot([x0, x1], [a + b * x0, a + b * x1], color=mlcca_color, linestyle="--", alpha=0.5, linewidth=1.0)

        ax.set_title(st)
        ax.set_xlabel("Price")
        ax.set_ylabel("Demand (Z-score)")
        # 让坐标轴的物理长度与数值跨度一致：等比例 + 统一跨度
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
        ax.set_aspect('equal', adjustable='box')
        try:
            x_all = (xs_cca or []) + (xs_ml or [])
            y_all = ys_plot or []
            if x_all and y_all:
                x_min, x_max = min(x_all), max(x_all)
                y_min, y_max = min(y_all), max(y_all)
                x_center = 0.5 * (x_min + x_max)
                y_center = 0.5 * (y_min + y_max)
                span = max(x_max - x_min, y_max - y_min)
                pad = (0.05 * span) if span > 0 else 1.0
                span = span + 2 * pad
                ax.set_xlim(x_center - span / 2.0, x_center + span / 2.0)
                ax.set_ylim(y_center - span / 2.0, y_center + span / 2.0)
        except Exception:
            pass
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=8, loc="best")

    plt.suptitle(f"Round {round_num} Price vs Demand by Station")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_allocated_items_correlation_two_panel(
    path_cca: str,
    path_mlcca: str,
    auction_path: str,
    demands_csv: str,
    out_png: str,
    round_num: int = 6,
):
    ensure_out_dir(os.path.dirname(out_png))

    # Load data and prices
    data_cca = load_json_relaxed(path_cca)
    data_mlcca = load_json_relaxed(path_mlcca)
    _, prices_cca = extract_price_vector_for_round(data_cca, round_num)
    _, prices_mlcca = extract_price_vector_for_round(data_mlcca, round_num)
    goods = build_goods_list_from_auction(auction_path)
    demand_counts = load_demands_csv(demands_csv) if demands_csv else build_demand_counts_from_allowed_slots(auction_path)

    n = min(len(goods), len(prices_cca), len(prices_mlcca))
    goods = goods[:n]
    prices_cca = prices_cca[:n]
    prices_mlcca = prices_mlcca[:n]
    idx_map = {item: i for i, item in enumerate(goods)}

    # Extract allocated item sets per algorithm
    def allocated_set(results_path: str) -> set:
        try:
            parsed = parse_allocations.parse_allocation_for_iteration(results_path, auction_path, round_num)
            s = set()
            for _, info in parsed.items():
                for it in info.get("items", []) or []:
                    if it:
                        s.add(str(it))
            return s
        except Exception:
            return set()

    cca_alloc = allocated_set(path_cca)
    ml_alloc = allocated_set(path_mlcca)
    invalid_items = {"station_0_0"}
    cca_alloc = {it for it in cca_alloc if it in idx_map and it not in invalid_items}
    ml_alloc = {it for it in ml_alloc if it in idx_map and it not in invalid_items}

    # Prepare plotting data per panel
    def prepare(items: List[str]) -> Tuple[List[float], List[float], List[str]]:
        items_sorted = sort_items(list(items))
        xs = [prices_cca[idx_map[i]] if panel == "CCA" else prices_mlcca[idx_map[i]] for i in items_sorted]
        ys = [int(demand_counts.get(i, 0)) for i in items_sorted]
        return xs, ys, items_sorted

    # Build two panels
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 17.14), facecolor="w", sharey=True)

    colors = {"CCA": "#4472C4", "ML-CCA": "#ED7D31"}
    panels = [("CCA", cca_alloc), ("ML-CCA", ml_alloc)]

    for ax, (panel, items_set) in zip(axes, panels):
        xs, ys_raw, items_sorted = prepare(items_set)
        # Pearson r（线性相关），对Z-score不变
        r_sp = pearson_r(xs, ys_raw) if xs and ys_raw else float('nan')
        # Z-score 标准化 y
        if ys_raw:
            m = sum(ys_raw) / len(ys_raw)
            var = sum((y - m) ** 2 for y in ys_raw) / len(ys_raw)
            sd = (var ** 0.5) if var > 0 else 1.0
            ys = [(y - m) / sd for y in ys_raw]
        else:
            ys = []
        ax.scatter(xs, ys, s=28, alpha=0.8, color=colors[panel], edgecolor="#222222", linewidth=0.3,
                   label=f"{panel} (Pearson r={r_sp:.3f}, n={len(items_sorted)})")

        # Trend line
        def _fit(xv, yv):
            n_ = len(xv)
            if n_ == 0:
                return None
            mx = sum(xv) / n_
            my = sum(yv) / n_
            cov = sum((x - mx) * (y - my) for x, y in zip(xv, yv))
            vx = sum((x - mx) ** 2 for x in xv)
            if vx == 0:
                return None
            b = cov / vx
            a = my - b * mx
            return a, b
        line = _fit(xs, ys)
        if line:
            a, b = line
            x0, x1 = (min(xs), max(xs)) if xs else (0, 0)
            ax.plot([x0, x1], [a + b * x0, a + b * x1], color=colors[panel], linestyle="--", alpha=0.6, linewidth=1.2)

        # Annotate top demand items (top-4)
        try:
            top_idx = sorted(range(len(items_sorted)), key=lambda i: ys_raw[i], reverse=True)[:4]
            for i in top_idx:
                name = items_sorted[i]
                ax.annotate(name, (xs[i], ys[i]), textcoords="offset points", xytext=(4, 4), fontsize=8, color="#444444")
        except Exception:
            pass

        ax.set_title(f"{panel} 分配的items：价格 vs 需求")
        ax.set_xlabel("Price")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="best", fontsize=9)
        # 标准化后不再使用symlog
        # 等比例 + 统一跨度，确保两轴的可视长度一致
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass
        ax.set_aspect('equal', adjustable='box')
        try:
            if xs and ys:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                x_center = 0.5 * (x_min + x_max)
                y_center = 0.5 * (y_min + y_max)
                span = max(x_max - x_min, y_max - y_min)
                pad = (0.05 * span) if span > 0 else 1.0
                span = span + 2 * pad
                ax.set_xlim(x_center - span / 2.0, x_center + span / 2.0)
                ax.set_ylim(y_center - span / 2.0, y_center + span / 2.0)
        except Exception:
            pass

    axes[0].set_ylabel("Demand (Z-score)")
    plt.suptitle(f"Round {round_num} 已分配items的价格-需求相关性（分别按 CCA / ML-CCA）")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    # 两个results.json路径（与现有脚本保持一致）
    path_cca = \
        "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/results.json"
    path_mlcca = \
        "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_gd_linear_prices_on_W_v3/ML_config_hpo1/8/results.json"

    auction_path = \
        "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/auction_instance_seed_8.json"
    import glob
    # 自动选择最新生成的 combined_first*_post*_item_demands_counts_*.csv
    combined_csvs = sorted(
        glob.glob(os.path.join(OUT_DIR, "combined_first*_post*_item_demands_counts_*.csv")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    demands_csv = combined_csvs[0] if combined_csvs else None
    out_png1 = os.path.join(OUT_DIR, "price_demand_correlation_combined_round6.png")
    # 仅生成单张相关图（按用户要求）

    plot_price_demand_correlation(path_cca, path_mlcca, auction_path, demands_csv, out_png1, round_num=6)
    print(f"Saved: {out_png1}")


if __name__ == "__main__":
    main()