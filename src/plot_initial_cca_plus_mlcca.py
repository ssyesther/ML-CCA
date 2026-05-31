import json
import os
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# 输入路径（按你的需求可改为命令行参数）
INITIAL_JSON = (
    
    "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/initial_cca_result.json"
)
# 50轮之后的CCA与ML-CCA结果文件
CCA_RESULTS_JSON = (
    
    "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/results.json"
)
MLCCA_RESULTS_JSON = (
    
    "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_gd_linear_prices_on_W_v3/ML_config_hpo1/8/results.json"
)

OUTPUT_DIR = "/Users/y./Documents/ML-CCA-main/src/wandb_custom_plots/results"

# 颜色与风格（与此前图一致）
CCA_COLOR = "#4472C4"  # 蓝色
MLCCA_COLOR = "#ED7D31"  # 橙色


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_initial_series(data: Dict) -> Tuple[List[float], List[float]]:
    """从 initial_cca_result.json 中提取前 Qinit 轮的效率与SCW序列。

    返回 (eff_list, scw_list)，索引对应轮次 1..Qinit。
    若存在缺失，则用前一个值或0填充。
    """
    rounds = data.get("clock_rounds", [])
    # 先按 round 排序，保证顺序一致
    rounds_sorted = sorted(
        rounds,
        key=lambda r: int(r.get("round", 0)) if isinstance(r.get("round", 0), (int, float)) else 0,
    )

    eff: List[float] = []
    scw: List[float] = []

    for r in rounds_sorted:
        e = r.get("efficiency", None)
        s = r.get("scw", None)

        if isinstance(e, (int, float)):
            eff.append(float(e))
        else:
            eff.append(eff[-1] if eff else 0.0)

        if isinstance(s, (int, float)):
            scw.append(float(s))
        else:
            scw.append(scw[-1] if scw else 0.0)

    return eff, scw


def get_series_map(d: Dict, key: str) -> Dict[int, float]:
    """从 results.json 中读取某个 per-iteration 字段，转换为 int->float 的映射。"""
    raw = d.get(key, {})
    out: Dict[int, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                ik = int(k)
            except Exception:
                continue
            try:
                out[ik] = float(v) if v is not None else math.nan
            except Exception:
                out[ik] = math.nan
    return out


def stitch_series(initial_eff: List[float], initial_scw: List[float],
                  cca_eff_map: Dict[int, float], cca_scw_map: Dict[int, float],
                  ml_eff_map: Dict[int, float], ml_scw_map: Dict[int, float],
                  qinit: int = 50) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """拼接 0..(qinit+6) 共 qinit+7 个点：
    - 0 轮设为 0（视觉起点）；
    - 1..qinit 使用 initial_cca_result.json 的效率与SCW；
    - 51..(qinit+6) 使用 results.json 中 0..6 的 ML-CCA迭代（偏移到 51..56）。
    同时返回 CCA 的延伸序列（51..56 复制第50轮值）。
    """
    total_rounds = qinit + 7  # 0..(qinit+6)
    x = list(range(total_rounds))

    # 初始化序列
    cca_eff = [math.nan] * total_rounds
    ml_eff = [math.nan] * total_rounds
    cca_scw = [math.nan] * total_rounds
    ml_scw = [math.nan] * total_rounds

    # 第0轮起点
    cca_eff[0] = 0.0
    ml_eff[0] = 0.0
    cca_scw[0] = 0.0
    ml_scw[0] = 0.0

    # 1..qinit：取 initial
    for i in range(1, qinit + 1):
        idx = i
        val_e = initial_eff[i - 1] if i - 1 < len(initial_eff) else initial_eff[-1]
        val_s = initial_scw[i - 1] if i - 1 < len(initial_scw) else initial_scw[-1]
        cca_eff[idx] = val_e
        ml_eff[idx] = val_e
        cca_scw[idx] = val_s
        ml_scw[idx] = val_s

    # 50..(qinit+6)：CCA 后续轮次（将 0..6 映射到 50..56），缺失则延续第50轮
    for r in range(0, 7):
        target_cca = qinit + r  # 50..56
        if 0 <= target_cca < total_rounds:
            if r in cca_eff_map:
                cca_eff[target_cca] = cca_eff_map[r]
            else:
                cca_eff[target_cca] = initial_eff[-1] if initial_eff else 0.0

            if r in cca_scw_map:
                cca_scw[target_cca] = cca_scw_map[r]
            else:
                cca_scw[target_cca] = initial_scw[-1] if initial_scw else 0.0

    # 51..(qinit+6)：ML-CCA 后续轮次（将 0..6 映射到 51..56），缺失则延续第50轮
    for r in range(0, 7):
        target_ml = qinit + 1 + r  # 51..57，其中57越界需跳过
        if 0 <= target_ml < total_rounds:
            if r in ml_eff_map:
                ml_eff[target_ml] = ml_eff_map[r]
            else:
                ml_eff[target_ml] = initial_eff[-1] if initial_eff else 0.0

            if r in ml_scw_map:
                ml_scw[target_ml] = ml_scw_map[r]
            else:
                ml_scw[target_ml] = initial_scw[-1] if initial_scw else 0.0

    return x, cca_eff, ml_eff, cca_scw, ml_scw


def plot_initial_cca_plus_mlcca():
    initial = load_json(INITIAL_JSON)
    results_cca = load_json(CCA_RESULTS_JSON)
    results_ml = load_json(MLCCA_RESULTS_JSON)

    qinit = int(initial.get("Qinit", 50))
    initial_eff, initial_scw = extract_initial_series(initial)

    cca_eff_map = get_series_map(results_cca, "Efficiency per Iteration")
    cca_scw_map = get_series_map(results_cca, "SCW per Iteration")
    ml_eff_map = get_series_map(results_ml, "Efficiency per Iteration")
    ml_scw_map = get_series_map(results_ml, "SCW per Iteration")

    x, eff_cca, eff_ml, scw_cca, scw_ml = stitch_series(
        initial_eff,
        initial_scw,
        cca_eff_map,
        cca_scw_map,
        ml_eff_map,
        ml_scw_map,
        qinit=qinit,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Efficiency
    ax = axes[0]
    ax.plot(x, eff_cca, color=CCA_COLOR, linewidth=2.0, marker="x", markersize=3, label="CCA")
    ax.plot(x, eff_ml, color=MLCCA_COLOR, linewidth=2.0, marker="o", markersize=3, label="EV-MLCCA")
    ax.axvline(qinit + 0.5, color="#888888", linestyle="--", linewidth=1.2)
    ax.set_title("Efficiency per Iteration")
    ax.set_xlabel("Round")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    ax.set_xlim(0, qinit + 6)
    ax.set_xticks(list(range(0, qinit + 7, 5)))

    # SCW
    ax2 = axes[1]
    ax2.plot(x, scw_cca, color=CCA_COLOR, linewidth=2.0, marker="x", markersize=3, label="CCA")
    ax2.plot(x, scw_ml, color=MLCCA_COLOR, linewidth=2.0, marker="o", markersize=3, label="EV-MLCCA")
    ax2.axvline(qinit + 0.5, color="#888888", linestyle="--", linewidth=1.2)
    ax2.set_title("SCW per Iteration")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("SCW")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best")
    ax2.set_xlim(0, qinit + 6)
    ax2.set_xticks(list(range(0, qinit + 7, 5)))

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"initial_cca_plus_mlcca_efficiency_scw_qinit_{qinit}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = plot_initial_cca_plus_mlcca()
    print(f"Saved figure to: {path}")