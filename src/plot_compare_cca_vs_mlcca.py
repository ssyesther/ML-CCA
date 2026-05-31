import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# 顶部工具函数区域：新增smooth_series
def round_list(values, ndigits=2):
    return [round(float(v), ndigits) for v in values]

def extract_efficiency_series_per_50(eff_dict, default_for_missing=None):
    """
    返回长度为50的列表，对应迭代1..50。
    - eff_dict: 形如 {"1": 0.96, "2": 0.97, ...} 的字典
    - default_for_missing: 缺失迭代用该默认值填充（float）
    """
    series = []
    for i in range(1, 51):
        key = str(i)
        if key in eff_dict and eff_dict[key] is not None:
            series.append(float(eff_dict[key]))
        elif default_for_missing is not None:
            series.append(float(default_for_missing))
        else:
            # 如果没有默认值，用已知最后一个值或0填充
            if len(series) > 0:
                series.append(series[-1])
            else:
                series.append(0.0)
    return series

def extract_sats_allocation_allocs(data):
    """
    提取"SATS Efficient Allocation"字典，返回 bidder -> bundle(list[str]) 的映射。
    """
    alloc = data.get("SATS Efficient Allocation", {})
    # 规范成 int -> list[str]，允许缺失时返回空列表
    result = {}
    # 尝试推断 bidder 范围（通常0..49）
    max_bidder = 49
    for i in range(0, max_bidder + 1):
        key = str(i)
        bundle = alloc.get(key, [])
        # bundle 是 list[str]（可能为空或一个元素）
        result[i] = bundle if isinstance(bundle, list) else [str(bundle)]
    return result

def build_allocation_matrix(alloc_cca, alloc_mlcca):
    """
    将两个分配映射转换为 2×N 的矩阵，用颜色编码 station 字符串。
    返回:
      matrix (2 x N np.array of int codes),
      station_to_code (dict),
      code_to_station (list)
    """
    bidders = sorted(set(list(alloc_cca.keys()) + list(alloc_mlcca.keys())))
    # 收集所有 station 名字（可能每个 bidder 是一个元素的list）
    stations = set()
    for i in bidders:
        for s in alloc_cca.get(i, []):
            stations.add(s)
        for s in alloc_mlcca.get(i, []):
            stations.add(s)
    # 考虑空分配（None）作为一种类目
    stations = sorted(list(stations))
    station_to_code = {s: idx for idx, s in enumerate(stations)}
    code_to_station = stations

    n = len(bidders)
    mat = np.zeros((2, n), dtype=int)

    for col, b in enumerate(bidders):
        # 将bundle（list[str]）转为一个字符串（如多元素，以逗号拼接）
        def bundle_to_str(bundle_list):
            if not bundle_list:
                return "NONE"
            # 通常是一个元素；若有多个，就合并
            return ", ".join(bundle_list)

        cca_label = bundle_to_str(alloc_cca.get(b, []))
        mlcca_label = bundle_to_str(alloc_mlcca.get(b, []))

        if cca_label not in station_to_code:
            stations.append(cca_label)
            station_to_code[cca_label] = len(stations) - 1
            code_to_station = stations
        if mlcca_label not in station_to_code:
            stations.append(mlcca_label)
            station_to_code[mlcca_label] = len(stations) - 1
            code_to_station = stations

        mat[0, col] = station_to_code[cca_label]
        mat[1, col] = station_to_code[mlcca_label]

    return mat, station_to_code, code_to_station, bidders

def extract_price_vectors(data):
    """
    提取"Price Vector per Iteration"，返回：
      iterations: list[int]（排序后的迭代编号）
      prices_per_iter: dict[int] -> list[float]
    """
    pv = data.get("Price Vector per Iteration", {})
    # 键是字符串迭代号
    iterations = sorted([int(k) for k in pv.keys()])
    prices_per_iter = {}
    for it in iterations:
        arr = pv.get(str(it), [])
        prices_per_iter[it] = [float(x) for x in arr]
    return iterations, prices_per_iter

# 新增：将 allocation 转为 item 计数（每个 item 被分配给多少个 bidder）
# 分配计数后：新增生成全量item的函数
def compute_item_counts(alloc_map):
    counts = {}
    for bidder, bundle in alloc_map.items():
        for item in bundle:
            counts[item] = counts.get(item, 0) + 1
    return counts

def main():
    # 1) 设置输入路径
    path_cca = "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/results.json"
    path_mlcca = "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_gd_linear_prices_on_W_v3/ML_config_hpo1/8/results.json"

    # 2) 读取 JSON
    data_cca = load_json(path_cca)
    data_mlcca = load_json(path_mlcca)

    out_dir = ensure_out_dir("/Users/y./Documents/ML-CCA-main/src/wandb_custom_plots/results")

    # =========================
    # 图一：Efficiency per Iteration（统一用 SATS Optimal SCW 归一化；第0轮=0；断轴放大0..10区间；平滑）
    # =========================
    baseline = get_sats_optimal_scw(data_mlcca, data_cca)
    if baseline is None or baseline == 0:
        print("[WARN] 无法获取 SATS Optimal SCW 基线或为 0，效率对比将跳过归一化。")

    def to_eff_series(scw_series: dict, base: float):
        if not isinstance(scw_series, dict) or not scw_series or not base:
            return [0], [0.0]
        try:
            iters = sorted(scw_series.keys(), key=lambda k: int(k))
        except Exception:
            iters = list(scw_series.keys())
        xs, ys = [], []
        for k in iters:
            v = scw_series.get(k)
            if isinstance(v, (int, float)):
                xs.append(int(k) if str(k).isdigit() else len(xs) + 1)
                ys.append(float(v) / base)
        if not xs or xs[0] != 0:
            xs = [0] + xs
            ys = [0.0] + ys
        return xs, ys

    scw_cca = data_cca.get("SCW per Iteration", {})
    scw_mlcca = data_mlcca.get("SCW per Iteration", {})
    cca_x, cca_y = to_eff_series(scw_cca, baseline)
    mlcca_x, mlcca_y = to_eff_series(scw_mlcca, baseline)

    # 平滑
    cca_y_sm = smooth_series(cca_y, window_size=5) if len(cca_y) > 3 else cca_y
    mlcca_y_sm = smooth_series(mlcca_y, window_size=5) if len(mlcca_y) > 3 else mlcca_y

    # 断轴布局：左轴显示[0,10]，右轴显示[10, max_x]
    x_full_left = np.array(cca_x if len(cca_x) > len(mlcca_x) else mlcca_x)
    max_x = int(x_full_left[-1]) if len(x_full_left) else 50

    fig = plt.figure(figsize=(10.5, 4.8), facecolor='w')
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.05)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)

    # 左轴绘制 0..10
    mask_left_cca = [(i >= 0 and i <= 10) for i in cca_x]
    mask_left_ml = [(i >= 0 and i <= 10) for i in mlcca_x]
    ax_left.plot(np.array(cca_x)[mask_left_cca], np.array(cca_y_sm)[mask_left_cca], label="CCA (SCW/SATS)", linewidth=2.0, alpha=0.9, color='tab:blue')
    ax_left.plot(np.array(mlcca_x)[mask_left_ml], np.array(mlcca_y_sm)[mask_left_ml], label="ML-CCA (SCW/SATS)", linewidth=2.0, alpha=0.9, color='tab:orange')
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0.0, 1.05)
    ax_left.set_xticks(np.arange(0, 11, 1))
    ax_left.grid(alpha=0.3, linestyle='--')

    # 原点坐标轴相交（左轴）
    ax_left.spines['left'].set_position('zero')
    ax_left.spines['bottom'].set_position('zero')
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.xaxis.set_ticks_position('bottom')
    ax_left.yaxis.set_ticks_position('left')

    # 右轴绘制 10..max_x
    mask_right_cca = [(i >= 10 and i <= max_x) for i in cca_x]
    mask_right_ml = [(i >= 10 and i <= max_x) for i in mlcca_x]
    ax_right.plot(np.array(cca_x)[mask_right_cca], np.array(cca_y_sm)[mask_right_cca], linewidth=2.0, alpha=0.9, color='tab:blue')
    ax_right.plot(np.array(mlcca_x)[mask_right_ml], np.array(mlcca_y_sm)[mask_right_ml], linewidth=2.0, alpha=0.9, color='tab:orange')
    ax_right.set_xlim(10, max_x)
    ax_right.set_xticks(np.arange(10, max_x + 1, 5))
    ax_right.grid(alpha=0.3, linestyle='--')
    ax_right.spines['left'].set_visible(False)

    fig.suptitle("Efficiency per Iteration (CCA vs ML-CCA, normalized by SATS Optimal SCW)", y=0.98)
    ax_left.set_ylabel("Efficiency")
    ax_right.set_xlabel("Iteration")
    ax_left.legend(loc='upper left')
    out_eff = os.path.join(out_dir, "compare_efficiency_per_iteration.png")
    plt.tight_layout()
    plt.savefig(out_eff, dpi=200)
    plt.close()
    print(f"Saved: {out_eff}")

    # =========================
    # 图二：SATS Efficient Allocation（两个子图，x=所有item）
    # =========================
    alloc_cca = extract_sats_allocation_allocs(data_cca)
    alloc_mlcca = extract_sats_allocation_allocs(data_mlcca)
    counts_cca = compute_item_counts(alloc_cca)
    counts_mlcca = compute_item_counts(alloc_mlcca)

    # 生成MSVM域的所有item，并与已有的item做并集（更稳健）
    all_items_domain = generate_all_items_msvm(num_stations=6, num_slots=24)
    all_items = sorted(set(all_items_domain) | set(counts_cca.keys()) | set(counts_mlcca.keys()))

    y_cca = [counts_cca.get(item, 0) for item in all_items]
    y_mlcca = [counts_mlcca.get(item, 0) for item in all_items]

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), facecolor='w', sharex=True)
    axes[0].bar(range(len(all_items)), y_cca, color='tab:blue', alpha=0.85)
    axes[0].set_title("SATS Efficient Allocation (CCA)", fontsize=12)
    axes[0].set_ylabel("Count (allocated to bidders)")
    axes[0].grid(alpha=0.2, axis='y')

    axes[1].bar(range(len(all_items)), y_mlcca, color='tab:orange', alpha=0.85)
    axes[1].set_title("SATS Efficient Allocation (ML-CCA)", fontsize=12)
    axes[1].set_ylabel("Count (allocated to bidders)")
    axes[1].set_xlabel("Item")
    axes[1].grid(alpha=0.2, axis='y')

    plt.xticks(range(len(all_items)), all_items, rotation=90, fontsize=6)
    out_alloc = os.path.join(out_dir, "compare_sats_allocation.png")
    plt.tight_layout()
    plt.savefig(out_alloc, dpi=200)
    plt.close()
    print(f"Saved: {out_alloc}")

    # =========================
    # 图三：Price Vector per Iteration（第6轮价格对比，横坐标与SATS相同的item）
    # =========================
    iters_cca, prices_cca = extract_price_vectors(data_cca)
    iters_mlcca, prices_mlcca = extract_price_vectors(data_mlcca)

    target_iter = 6
    p_cca_t = prices_cca.get(target_iter, [])
    p_mlcca_t = prices_mlcca.get(target_iter, [])

    # 将价格向量对齐到all_items的长度与顺序（假定价格向量的顺序与域内all_items一致）
    def align_prices(price_list, item_count):
        arr = [float(x) for x in (price_list or [])]
        if len(arr) >= item_count:
            return arr[:item_count]
        else:
            return arr + [0.0] * (item_count - len(arr))

    if len(p_cca_t) == 0 or len(p_mlcca_t) == 0:
        print(f"Warning: 第{target_iter}轮价格向量缺失，无法绘制第{target_iter}轮价格对比。CCA len={len(p_cca_t)}, ML-CCA len={len(p_mlcca_t)}")
    else:
        p_cca_aligned = align_prices(p_cca_t, len(all_items))
        p_mlcca_aligned = align_prices(p_mlcca_t, len(all_items))

        indices = np.arange(len(all_items))
        width = 0.4
        fig, ax = plt.subplots(figsize=(18, 6), facecolor='w')

        ax.bar(indices - width/2, p_cca_aligned, width=width, label="CCA", color='tab:blue', alpha=0.85)
        ax.bar(indices + width/2, p_mlcca_aligned, width=width, label="ML-CCA", color='tab:orange', alpha=0.85)

        ax.set_title(f"Iteration {target_iter} Item Prices: CCA vs ML-CCA")
        ax.set_xlabel("Item")
        ax.set_ylabel("Price")
        ax.set_xticks(indices)
        ax.set_xticklabels(all_items, rotation=90, fontsize=6)
        ax.grid(alpha=0.2, axis='y')
        ax.legend()

        out_prices = os.path.join(out_dir, "compare_price_vectors.png")
        plt.tight_layout()
        plt.savefig(out_prices, dpi=200)
        plt.close()
        print(f"Saved: {out_prices}")

def smooth_series(values, window_size=5):
    """
    对序列做简单移动平均平滑，window_size为奇数效果更好。
    """
    if not values or window_size <= 1:
        return values
    w = int(window_size)
    if w % 2 == 0:
        w += 1  # 保证奇数窗口
    pad = w // 2
    # 边界填充：用边界值进行填充
    padded = np.pad(np.array(values, dtype=float), (pad, pad), mode='edge')
    kernel = np.ones(w) / w
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed.tolist()

def generate_all_items_msvm(num_stations=6, num_slots=24):
    """
    返回MSVM域中所有item名称的列表，形如 station_i_j。
    默认6站，24个slot。
    """
    return [f"station_{i}_{j}" for i in range(num_stations) for j in range(num_slots)]

# 新增：统一获取 SATS Optimal SCW（优先 ML-CCA，缺失时退回 CCA）
def get_sats_optimal_scw(mlcca_results: dict, cca_results: dict):
    val = mlcca_results.get("SATS Optimal SCW", None)
    if not isinstance(val, (int, float)):
        val = cca_results.get("SATS Optimal SCW", None)
    return float(val) if isinstance(val, (int, float)) else None

# 顶部辅助函数区域
def extract_final_scw(results: dict) -> float:
    """
    统一从 results.json 中提取“最终 SCW”，优先级：
    1) Final SCW 为数值
    2) SCW per Iteration 的最后(或最大)迭代值
    3) Inferred SCW per Iteration 的最后(或最大)迭代值
    若都不存在则返回 None
    """
    # Final SCW 可能是数值或列表
    if isinstance(results.get("Final SCW"), (int, float)):
        return float(results["Final SCW"])

    final_scw_field = results.get("Final SCW")
    if isinstance(final_scw_field, list) and len(final_scw_field) > 0:
        last_val = final_scw_field[-1]
        if isinstance(last_val, (int, float)):
            return float(last_val)

    # SCW per Iteration 可能是迭代号->数值的字典
    scw_per_iter = results.get("SCW per Iteration")
    if isinstance(scw_per_iter, dict) and len(scw_per_iter) > 0:
        # 键可能是字符串的迭代号，这里尽量转成 int 排序
        try:
            iters = sorted(scw_per_iter.keys(), key=lambda k: int(k))
            last_key = iters[-1]
        except Exception:
            # 如果无法转 int，则按字典插入顺序/键排序兜底
            iters = list(scw_per_iter.keys())
            last_key = iters[-1]
        last_val = scw_per_iter.get(last_key)
        if isinstance(last_val, (int, float)):
            return float(last_val)

    # Inferred SCW per Iteration 兜底
    inferred_scw_per_iter = results.get("Inferred SCW per Iteration")
    if isinstance(inferred_scw_per_iter, dict) and len(inferred_scw_per_iter) > 0:
        try:
            iters = sorted(inferred_scw_per_iter.keys(), key=lambda k: int(k))
            last_key = iters[-1]
        except Exception:
            iters = list(inferred_scw_per_iter.keys())
            last_key = iters[-1]
        last_val = inferred_scw_per_iter.get(last_key)
        if isinstance(last_val, (int, float)):
            return float(last_val)

    return None

if __name__ == "__main__":
    # 保证PDF/PS兼容性（如需导出PDF可用）
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    main()