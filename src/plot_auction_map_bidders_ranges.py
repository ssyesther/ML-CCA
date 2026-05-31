import os
import re
import json
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects


OUT_DIR = "/Users/y./Documents/ML-CCA-main/src/wandb_custom_plots/results"

# 坐标轴脊线（spines）加粗，提升轴线可见度
AXIS_SPINES_WIDTH = 8.0

# 在靠近站点时自动隐藏竞标者文字，避免遮挡
BIDDER_TEXT_HIDE_NEAR_STATION = True
# 以数据坐标为单位（当前坐标范围固定在 [0, 100]）
BIDDER_TEXT_HIDE_RADIUS_POINT = 5.0      # 与站点点的距离阈值
BIDDER_TEXT_HIDE_RADIUS_LABEL = 5.0      # 与站点标签位置的距离阈值

# 在彼此很近的竞标者之间，仅保留一个 bidder 标签
BIDDER_TEXT_HIDE_NEAR_BIDDER = True
BIDDER_TEXT_HIDE_BIDDER_RADIUS = 3.0     # bidder 文本之间的距离阈值（数据坐标）

# 当 bidder 位于主图的上/下边框线附近时隐藏文字标签
BIDDER_TEXT_HIDE_NEAR_TOP_BOTTOM = True
BIDDER_TEXT_HIDE_BORDER_MARGIN = 1.0     # 距离 y=0 或 y=100 的容忍边距（数据坐标）


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json_relaxed(path: str) -> dict:
    """读取JSON，允许 NaN/Infinity/-Infinity（替换为 null）。"""
    with open(path, "r") as f:
        txt = f.read()
    txt = re.sub(r"(?<![\w\-])NaN(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])Infinity(?![\w\-])", "null", txt)
    txt = re.sub(r"(?<![\w\-])\-Infinity(?![\w\-])", "null", txt)
    return json.loads(txt)


def bidder_label(bid: str) -> str:
    """将原始 bidder id 映射为标签格式 EV_序号。
    - 若 id 末尾带数字（如 Bidder_23 或 Any_23），则提取末尾数字并输出 EV_23。
    - 否则回退为 EV_<原始id>（避免丢失信息）。
    """
    try:
        m = re.search(r"(\d+)$", str(bid))
        if m:
            return f"EV_{m.group(1)}"
    except Exception:
        pass
    return f"EV_{str(bid)}"


def extract_stations(data: dict) -> Dict[str, Tuple[float, float]]:
    """返回 station_id -> (x, y)。"""
    stations = {}
    for s in (data.get("stations") or []):
        sid = str(s.get("id", "station_0"))
        x = float(s.get("x", 0.0))
        y = float(s.get("y", 0.0))
        stations[sid] = (x, y)
    return stations


def extract_station_clusters(data: dict) -> Dict[str, int]:
    """返回 station_id -> cluster_id（若缺失则为 0）。"""
    clusters = {}
    for s in (data.get("stations") or []):
        sid = str(s.get("id", "station_0"))
        cid = int(s.get("cluster_id", 0))
        clusters[sid] = cid
    return clusters


def extract_bidders(data: dict) -> List[dict]:
    """返回 bidder 列表，包含 id、坐标、类型、可达的 station 集合。"""
    bidders = []
    for b in (data.get("bidders") or []):
        bid = str(b.get("id", "Bidder"))
        coords = b.get("coordinates") or {}
        bx = float(coords.get("x", 0.0))
        by = float(coords.get("y", 0.0))
        btype = str(b.get("bidder_type", "unknown"))
        allowed = set()
        for it in (b.get("allowed_slots") or []):
            m = re.match(r"station_\d+_\d+", str(it))
            if m:
                # station_3_15 -> station_3
                st = "_".join(m.group(0).split("_")[:2])
                allowed.add(st)
        bidders.append({
            "id": bid,
            "x": bx,
            "y": by,
            "type": btype,
            "allowed_stations": sorted(list(allowed)),
        })
    return bidders


def color_for_type(t: str) -> str:
    mapping = {
        "premium_urgent": "#D62728",  # red
        "high_value_short": "#FF7F0E",  # orange
        "medium_value_medium": "#1F77B4",  # blue
        "low_value_long": "#2CA02C",  # green
    }
    return mapping.get(t, "#6A5ACD")  # default slate-blue


def plot_map(auction_path: str, out_png: str):
    ensure_out_dir(os.path.dirname(out_png))

    data = load_json_relaxed(auction_path)
    stations = extract_stations(data)
    bidders = extract_bidders(data)

    # 画布设置
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(100, 100), facecolor="w")
    ax = plt.gca()

    # 加粗坐标轴脊线（四边框线）
    for sp in ax.spines.values():
        sp.set_linewidth(AXIS_SPINES_WIDTH)

    # 计算全局范围并将原始坐标等比例映射到 [0, 100]
    xs_all_raw = [sx for sx, _ in stations.values()] + [b["x"] for b in bidders]
    ys_all_raw = [sy for _, sy in stations.values()] + [b["y"] for b in bidders]
    if xs_all_raw and ys_all_raw:
        min_x, max_x = min(xs_all_raw), max(xs_all_raw)
        min_y, max_y = min(ys_all_raw), max(ys_all_raw)
        span_x = max_x - min_x
        span_y = max_y - min_y
        span_max = max(span_x, span_y)
        scale = (100.0 / span_max) if span_max > 0 else 1.0
    else:
        min_x = min_y = 0.0
        scale = 1.0

    def sx_map(x: float) -> float:
        return (x - min_x) * scale

    def sy_map(y: float) -> float:
        return (y - min_y) * scale

    # 构建缩放后的站点坐标
    stations_scaled = {sid: (sx_map(sx), sy_map(sy)) for sid, (sx, sy) in stations.items()}

    # 绘制站点（改为 "Station_" 前缀，标签更大并添加描边以提升可读性）
    station_label_positions = {}
    for sid, (sx, sy) in stations_scaled.items():
        ax.scatter([sx], [sy], s=5000, marker="s", color="#444444", alpha=0.9, label=None)
        sid_cap = sid.replace("station_", "Station_")
        txt = ax.text(sx, sy+0.5, sid_cap, fontsize=100, ha="center", va="bottom", zorder=6)
        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])
        # 与实际文字位置一致，便于遮挡检测
        station_label_positions[sid] = (sx, sy + 0.5)

    # 为每个 bidder 仅绘制点与连线到可达站点（不再绘制圆圈半径）
    # 记录已放置的 bidder 文本位置，用于近邻抑制
    placed_bidder_label_positions = []
    for b in bidders:
        bx, by = sx_map(b["x"]), sy_map(b["y"])  # 缩放后的竞标者坐标
        color = color_for_type(b["type"])
        allowed = b["allowed_stations"]

        # 连线到可达站点
        for sid in allowed:
            if sid in stations_scaled:
                sx, sy = stations_scaled[sid]
                ax.plot([bx, sx], [by, sy], color=color, alpha=0.4, linewidth=3)

        # bidder 点与标签（仅显示 id，使用数据坐标右偏移并添加描边以降低重叠感）
        ax.scatter([bx], [by], s=4000, marker="o", color=color, edgecolor="#222222", linewidth=0.8, zorder=4)

        # 根据与站点点/站点标签的距离决定是否隐藏文字，避免遮挡
        show_bidder_text = True
        # 顶/底边框附近：若 y 接近 0 或 100，则不显示文字标签
        if BIDDER_TEXT_HIDE_NEAR_TOP_BOTTOM:
            if by <= BIDDER_TEXT_HIDE_BORDER_MARGIN or by >= 100.0 - BIDDER_TEXT_HIDE_BORDER_MARGIN:
                show_bidder_text = False
        if BIDDER_TEXT_HIDE_NEAR_STATION:
            for sid, (sx, sy) in stations_scaled.items():
                dxp, dyp = bx - sx, by - sy
                dist_point = (dxp * dxp + dyp * dyp) ** 0.5
                lx, ly = station_label_positions.get(sid, (sx, sy + 1.0))
                dxl, dyl = (bx + 1) - lx, by - ly
                dist_label = (dxl * dxl + dyl * dyl) ** 0.5
                if dist_point < BIDDER_TEXT_HIDE_RADIUS_POINT or dist_label < BIDDER_TEXT_HIDE_RADIUS_LABEL:
                    show_bidder_text = False
                    break

        # 近邻 bidder 文本抑制：与已放置文本过近则隐藏
        if show_bidder_text and BIDDER_TEXT_HIDE_NEAR_BIDDER:
            for (px, py) in placed_bidder_label_positions:
                dx, dy = (bx + 1) - px, by - py
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < BIDDER_TEXT_HIDE_BIDDER_RADIUS:
                    show_bidder_text = False
                    break

        if show_bidder_text:
            txtb = ax.text(bx + 1, by, bidder_label(b["id"]), fontsize=70, ha="left", va="center", zorder=6)
            txtb.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])
            placed_bidder_label_positions.append((bx + 1, by))

    # 固定坐标范围为 [0, 100]，展示被拉大后的缩放空间
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.set_title("Auction Map: Stations, Bidders & Connections", fontsize=100, pad=60)
    ax.set_xlabel("x", fontsize=100)
    ax.set_ylabel("y", fontsize=100)
    ax.set_xticks(list(range(0, 101, 10)))
    ax.set_yticks(list(range(0, 101, 10)))
    ax.tick_params(axis='both', labelsize=100)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linestyle=":")

    plt.tight_layout()
    # 输出 PNG + SVG + PDF
    base, _ = os.path.splitext(out_png)
    png_path = base + ".png"
    svg_path = base + ".svg"
    pdf_path = base + ".pdf"
    plt.savefig(png_path, dpi=180)
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    plt.close()


def plot_map_by_cluster(auction_path: str, out_png: str):
    """按 cluster_id 分区绘制子图：每个子图显示该簇的站点与可达的竞标者。"""
    ensure_out_dir(os.path.dirname(out_png))

    data = load_json_relaxed(auction_path)
    stations = extract_stations(data)
    station_clusters = extract_station_clusters(data)
    bidders = extract_bidders(data)

    # 构建 cluster -> station 列表 与 坐标列表
    cluster_to_stations: Dict[int, List[str]] = {}
    cluster_coords: Dict[int, List[Tuple[float, float]]] = {}
    for sid, cid in station_clusters.items():
        cluster_to_stations.setdefault(cid, []).append(sid)
        sx, sy = stations.get(sid, (0.0, 0.0))
        cluster_coords.setdefault(cid, []).append((sx, sy))

    cluster_ids = sorted(cluster_to_stations.keys())
    if not cluster_ids:
        raise ValueError("stations 中未找到 cluster_id")

    # 子图排版：尽量接近方阵
    n = len(cluster_ids)
    cols = min(3, max(1, int(round(n ** 0.5))))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), facecolor="w")
    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel())

    for idx, cid in enumerate(cluster_ids):
        ax = axes_list[idx]
        station_ids = set(cluster_to_stations[cid])
        coords = cluster_coords[cid]

        # 绘制该簇的站点
        for sid in station_ids:
            sx, sy = stations[sid]
            ax.scatter([sx], [sy], s=80, marker="s", color="#444444", alpha=0.85)
            ax.text(sx, sy + 0.03, sid, fontsize=8, ha="center", va="bottom")

        # 过滤可达该簇站点的竞标者（不再计算或绘制半径）
        relevant_bidders = []
        for b in bidders:
            allowed_in_cluster = [sid for sid in b["allowed_stations"] if sid in station_ids]
            if not allowed_in_cluster:
                continue
            bx, by = b["x"], b["y"]
            relevant_bidders.append({
                **b,
                "_allowed_cluster": allowed_in_cluster,
            })

        # 绘制竞标者：点与连线到该簇内的可达站点（不绘制圆圈）
        for b in relevant_bidders:
            bx, by = b["x"], b["y"]
            color = color_for_type(b["type"])
            for sid in b["_allowed_cluster"]:
                sx, sy = stations[sid]
                ax.plot([bx, sx], [by, sy], color=color, alpha=0.5, linewidth=1.0)
            ax.scatter([bx], [by], s=80, marker="o", color=color, edgecolor="#222222", linewidth=0.7)
            ax.text(bx, by - 0.05, f"{bidder_label(b['id'])}\n{b['type']}", fontsize=9, ha="center", va="top")

        # 设置坐标范围：基于该簇站点与相关竞标者的包围盒并添加边距
        xs = [x for x, _ in coords] + [b["x"] for b in relevant_bidders]
        ys = [y for _, y in coords] + [b["y"] for b in relevant_bidders]
        if xs and ys:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = max_x - min_x
            span_y = max_y - min_y
            pad_x = max(span_x * 0.25, 0.3)
            pad_y = max(span_y * 0.25, 0.3)
            ax.set_xlim(min_x - pad_x, max_x + pad_x)
            ax.set_ylim(min_y - pad_y, max_y + pad_y)

        ax.set_title(f"Cluster {cid}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linestyle=":")

    # 隐藏未使用的子图
    for j in range(len(cluster_ids), rows * cols):
        axes_list[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    auction_path = "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/auction_instance_seed_8.json"
    # 仅生成总图（不再输出分区子图）
    out_png = os.path.join(OUT_DIR, "auction_map_stations_bidders_ranges.png")
    plot_map(auction_path, out_png)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()