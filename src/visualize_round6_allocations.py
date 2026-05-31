import os
import json
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from parse_allocations import parse_allocation_for_iteration


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def normalize_bidder_key(k):
    """
    将投标者键规范化为整数：
    - 如果已是 int，直接返回
    - 如果是字符串，提取其中的数字（如 "bidder_12" -> 12；"12" -> 12）
    - 若无法提取数字，则原样返回（极端情况）
    """
    if isinstance(k, int):
        return k
    s = str(k)
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    try:
        return int(s)
    except Exception:
        return s


def normalize_alloc_map_keys(alloc_map):
    """
    返回一个键规范化后的分配映射副本，键尽量转换为整数。
    值保持不变。
    """
    norm = {}
    for k, v in alloc_map.items():
        nk = normalize_bidder_key(k)
        norm[nk] = v
    return norm


def build_alloc_maps_from_compare(compare_data):
    """
    从 compare_cca_vs_mlcca_round6.json 构建两个分配映射：
    返回 alloc_cca, alloc_mlcca（dict[int] -> list[str]）
    每个投标者的 bundle 用 list[str] 表示；若为空则用 []。
    """
    alloc_cca = {}
    alloc_mlcca = {}
    # bidder keys in compare file are strings; normalize to int
    for bk, entry in compare_data.items():
        try:
            b = int(bk)
        except Exception:
            # try extract digits
            digits = ''.join([c for c in bk if c.isdigit()])
            b = int(digits) if digits else bk
        cca_items = entry.get('cca', []) or []
        ml_items = entry.get('mlcca', []) or []
        # 统一格式：list[str]
        alloc_cca[b] = [str(x) for x in cca_items]
        alloc_mlcca[b] = [str(x) for x in ml_items]
    return alloc_cca, alloc_mlcca


def build_alloc_maps_from_results(cca_results_path: str, mlcca_results_path: str, auction_path: str, round_num: int):
    """
    直接从两份 results.json 解析第 round_num 轮的分配，返回 alloc_cca, alloc_mlcca。
    映射格式：dict[int] -> list[str]
    """
    parsed_cca_raw = parse_allocation_for_iteration(cca_results_path, auction_path, round_num) or {}
    parsed_ml_raw = parse_allocation_for_iteration(mlcca_results_path, auction_path, round_num) or {}

    # 规范化键，确保后续的列顺序为数值升序（如 0..49）
    parsed_cca = normalize_alloc_map_keys(parsed_cca_raw)
    parsed_ml = normalize_alloc_map_keys(parsed_ml_raw)

    bidders = sorted(set(list(parsed_cca.keys()) + list(parsed_ml.keys())), key=lambda x: (isinstance(x, int), x))
    # 上面排序规则：优先将 int 视为数值排序；若混有非 int 键，保持其相对顺序在末尾。
    alloc_cca = {}
    alloc_mlcca = {}
    for b in bidders:
        cca_items = (parsed_cca.get(b, {}) or {}).get('items', []) or []
        ml_items = (parsed_ml.get(b, {}) or {}).get('items', []) or []
        alloc_cca[b] = [str(x) for x in cca_items]
        alloc_mlcca[b] = [str(x) for x in ml_items]
    return alloc_cca, alloc_mlcca


def build_allocation_matrix(alloc_cca, alloc_mlcca):
    """
    将两个分配映射转换为 2×N 的矩阵：颜色按 station_i 统一；文本标注分配的 j。
    返回:
      matrix (2 x N np.array of int station codes),
      station_to_code (dict[str,int]),
      code_to_station (list[str]),
      bidders (sorted list[int]),
      labels (2 x N list[str])
    """
    bidders = sorted(set(list(alloc_cca.keys()) + list(alloc_mlcca.keys())))

    def parse_station_and_js(bundle):
        if not bundle:
            return 'NONE', []
        station_label = None
        js = []
        for it in bundle:
            m = re.match(r"(station_\d+)_([0-9]+)", str(it))
            if not m:
                continue
            st = m.group(1)
            j = int(m.group(2))
            if station_label is None:
                station_label = st
            if st == station_label:
                js.append(j)
        if station_label is None:
            return 'NONE', []
        js = sorted(js)
        return station_label, js

    stations = set(['NONE'])
    for b in bidders:
        st_cca, _ = parse_station_and_js(alloc_cca.get(b, []))
        st_ml, _ = parse_station_and_js(alloc_mlcca.get(b, []))
        stations.add(st_cca)
        stations.add(st_ml)
    stations = sorted(list(stations))
    station_to_code = {s: idx for idx, s in enumerate(stations)}
    code_to_station = stations

    n = len(bidders)
    mat = np.zeros((2, n), dtype=int)
    labels = [["" for _ in range(n)] for _ in range(2)]

    def js_to_label(js):
        if not js:
            return ""
        if len(js) == 1:
            return f"{js[0]}"
        is_contig = all(js[i] + 1 == js[i+1] for i in range(len(js)-1))
        if is_contig:
            return f"{js[0]}–{js[-1]}"
        if len(js) <= 4:
            return ",".join(str(x) for x in js)
        return ",".join(str(x) for x in js[:3]) + "…"

    for col, b in enumerate(bidders):
        st_cca, js_cca = parse_station_and_js(alloc_cca.get(b, []))
        st_ml, js_ml = parse_station_and_js(alloc_mlcca.get(b, []))
        mat[0, col] = station_to_code.get(st_cca, station_to_code['NONE'])
        mat[1, col] = station_to_code.get(st_ml, station_to_code['NONE'])
        labels[0][col] = js_to_label(js_cca)
        labels[1][col] = js_to_label(js_ml)

    return mat, station_to_code, code_to_station, bidders, labels


def compute_item_counts(alloc_map):
    counts = {}
    for bidder, bundle in alloc_map.items():
        if not bundle:
            counts['NONE'] = counts.get('NONE', 0) + 1
        else:
            for item in bundle:
                counts[item] = counts.get(item, 0) + 1
    return counts


def plot_per_bidder_allocation(mat, code_to_station, bidders, out_path, labels=None):
    """
    使用 imshow 将 2×N 的矩阵渲染为彩色块图：行=方法（CCA, ML-CCA），列=投标者。
    颜色表示分配的站点（或 NONE）。
    """
    # 图幅高度调小，使 y 轴更“短”且整体更扁平
    fig, ax = plt.subplots(figsize=(18, 2.2), facecolor='w')

    def user_palette_rgba(n, lighten=0.5):
        """
        返回更“浅”的粉彩颜色：将基础色与白色按比例混合。
        lighten=0.5 表示向白色偏移 50%，降低亮度饱和度，避免过于刺眼。
        """
        base_hex = [
            "#D62728",  # red
            "#FF7F0E",  # orange
            "#1F77B4",  # blue
            "#2CA02C",  # green
            "#FFC000",  # yellow
            "#7030A0",  # purple
        ]
        rgba_list = []
        for i in range(n):
            r, g, b, a = matplotlib.colors.to_rgba(base_hex[i % len(base_hex)])
            # 与白色(1,1,1)混合，得到更浅的色调
            r = r + (1.0 - r) * lighten
            g = g + (1.0 - g) * lighten
            b = b + (1.0 - b) * lighten
            rgba_list.append((r, g, b, 1.0))
        return rgba_list

    stations = [lbl for lbl in code_to_station if lbl != 'NONE']
    color_map = {'NONE': (1.0, 1.0, 1.0, 1.0)}
    palette_rgba = user_palette_rgba(len(stations))
    for lbl, rgba in zip(stations, palette_rgba):
        color_map[lbl] = rgba
    colors = [color_map.get(lbl, (0.92, 0.92, 0.92, 1.0)) for lbl in code_to_station]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), len(colors))
    im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['CCA', 'EV-MLCCA'])
    ax.set_xticks(np.arange(len(bidders)))
    ax.set_xticklabels([str(b) for b in bidders], fontsize=8)
    ax.set_xlabel('EV')
    # ax.set_title('Allocation by Bidder: CCA vs EV-MLCCA')

    # 细微网格线增强可读性
    ax.set_xticks(np.arange(-0.5, len(bidders), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)

    # 构建图例：仅显示 station_i（不展示具体 item station_i_j）
    handles = []
    legend_labels = []
    for code, station in enumerate(code_to_station):
        if station == 'NONE':
            continue
        color = colors[code]
        handles.append(matplotlib.patches.Patch(color=color))
        legend_labels.append(station)
    if handles:
        ax.legend(handles, legend_labels, title='Station', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)

    # 在色块上标注 j（CCA、ML-CCA 都标），并根据背景色动态选择文字颜色
    if labels is not None:
        n_cols = mat.shape[1]
        def text_color_for_bg(rgba):
            r, g, b, a = rgba
            # sRGB 近似亮度
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return 'black' if luminance > 0.6 else 'white'
        for r in range(2):
            for c in range(n_cols):
                txt = labels[r][c] if r < len(labels) and c < len(labels[r]) else ""
                if not txt:
                    continue
                bg = colors[mat[r, c]] if 0 <= mat[r, c] < len(colors) else (1, 1, 1, 1)
                ax.text(c, r, txt, ha='center', va='center', fontsize=8, color=text_color_for_bg(bg))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_station_distribution(counts_cca, counts_mlcca, out_path):
    stations = sorted(set(list(counts_cca.keys()) + list(counts_mlcca.keys())))
    indices = np.arange(len(stations))
    width = 0.42

    y_cca = [counts_cca.get(s, 0) for s in stations]
    y_mlcca = [counts_mlcca.get(s, 0) for s in stations]

    # 使用与上图一致的用户指定5色配色；NONE=白色
    def user_palette_rgba(n):
        base_hex = [
            "#4472C4", "#ED7D31", "#70AD47", "#FFC000", "#E45756",
        ]
        return [matplotlib.colors.to_rgba(base_hex[i % len(base_hex)]) for i in range(n)]
    non_none = [lbl for lbl in stations if lbl != 'NONE']
    bar_map = {'NONE': (1.0, 1.0, 1.0, 1.0)}
    pal = user_palette_rgba(len(non_none))
    for lbl, rgba in zip(non_none, pal):
        bar_map[lbl] = rgba
    bar_colors = [bar_map.get(lbl, (0.92, 0.92, 0.92, 1.0)) for lbl in stations]

    fig, ax = plt.subplots(figsize=(18, 6.5), facecolor='w')
    # CCA 与 ML-CCA 条形采用相同站点色，但透明度不同以区分方法
    ax.bar(indices - width/2, y_cca, width=width, label='CCA', color=bar_colors, alpha=0.70, edgecolor='#666666', linewidth=0.2)
    ax.bar(indices + width/2, y_mlcca, width=width, label='ML-CCA', color=bar_colors, alpha=0.40, edgecolor='#666666', linewidth=0.2)

    ax.set_title('Station Allocation Distribution: CCA vs ML-CCA')
    ax.set_xlabel('Station')
    ax.set_ylabel('Count (allocated to bidders)')
    ax.set_xticks(indices)
    ax.set_xticklabels(stations, rotation=90, fontsize=7)
    ax.grid(alpha=0.25, axis='y')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    auction_path = "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/auction_instance_seed_8.json"
    cca_results = "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/results.json"
    mlcca_results = "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_gd_linear_prices_on_W_v3/ML_config_hpo1/8/results.json"
    round_num = 6

    out_dir = ensure_out_dir(
        "/Users/y./Documents/ML-CCA-main/src/wandb_custom_plots/results"
    )

    alloc_cca, alloc_mlcca = build_alloc_maps_from_results(cca_results, mlcca_results, auction_path, round_num)

    # 计算全局 ML-only（第 round_num 轮 ML-CCA 分配的所有 item − CCA 分配的所有 item）
    ml_items_set = set()
    cca_items_set = set()
    for bundle in alloc_mlcca.values():
        for it in bundle or []:
            ml_items_set.add(str(it))
    for bundle in alloc_cca.values():
        for it in bundle or []:
            cca_items_set.add(str(it))
    ml_only_items = list(ml_items_set - cca_items_set)

    # 图一：按投标者对比分配（颜色代表站点；叠加 j 标签）
    mat, station_to_code, code_to_station, bidders, labels = build_allocation_matrix(alloc_cca, alloc_mlcca)
    out_fig1 = os.path.join(out_dir, "round6_allocation_by_bidder.png")
    plot_per_bidder_allocation(mat, code_to_station, bidders, out_fig1, labels)

    # 图二：站点分配分布对比（计数）
    counts_cca = compute_item_counts(alloc_cca)
    counts_mlcca = compute_item_counts(alloc_mlcca)
    out_fig2 = os.path.join(out_dir, "round6_station_distribution.png")
    plot_station_distribution(counts_cca, counts_mlcca, out_fig2)

    # 同时输出 ML-only 的站点分布（全局差集）以便核对（文件名带 ml_only）
    counts_ml_only = compute_item_counts({'ML_ONLY': ml_only_items})
    out_fig2b = os.path.join(out_dir, "round6_station_distribution_ml_only.png")
    plot_station_distribution(counts_cca, counts_ml_only, out_fig2b)


if __name__ == "__main__":
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    main()