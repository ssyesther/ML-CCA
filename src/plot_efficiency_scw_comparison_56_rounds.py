import json
import os
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# Absolute paths provided by the user
# 数据源：CCA 后50轮使用 results.json；前50轮使用 initial_cca_result.json
CCA_JSON = \
    "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/results.json"
INITIAL_CCA_JSON = \
    "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_cca/ML_config_hpo1/8/initial_cca_result.json"
MLCCA_JSON = \
    "/Users/y./Documents/ML-CCA-main/src/results/MSVM_qinit_50_initial_demand_query_method_cca_cca_initial_prices_multiplier_0.2_increment_0.05_new_query_option_gd_linear_prices_on_W_v3/ML_config_hpo1/8/results.json"

OUTPUT_DIR = "/Users/y./Documents/ML-CCA-main/src/wandb_custom_plots/results"

# Palette consistent with prior figures
CCA_COLOR = "#4472C4"  # deep blue
MLCCA_COLOR = "#ED7D31"  # deep orange


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def get_series(d: Dict, key: str) -> Dict[int, float]:
    """Return iteration->value mapping with int keys, ignoring non-numeric entries."""
    raw = d.get(key, {})
    out: Dict[int, float] = {}
    for k, v in raw.items():
        try:
            ik = int(k)
            out[ik] = float(v) if v is not None else math.nan
        except Exception:
            # skip entries like strings or Nones
            continue
    return out


def extract_series_from_initial_cca(d: Dict, field: str) -> Dict[int, float]:
    """
    从 initial_cca_result.json 的 clock_rounds 提取序列。
    - field: "efficiency" 或 "scw"
    返回 {round(1..50) -> value}。
    """
    out: Dict[int, float] = {}
    rounds = d.get("clock_rounds", [])
    for rec in rounds:
        try:
            r = int(rec.get("round"))
        except Exception:
            continue
        val = rec.get(field)
        try:
            out[r] = float(val) if val is not None else math.nan
        except Exception:
            out[r] = math.nan
    return out


def combine_initial_and_post50(initial_map: Dict[int, float], cca_results_map: Dict[int, float]) -> Dict[int, float]:
    """
    合并序列：
    - 1..50 使用 initial_map（来自 initial_cca_result.json 的原始钟拍数据）
    - 50轮之后（严格 >50）使用 cca_results_map 中的条目（若存在）
    注意：通常 CCA 的 results.json 只有到 50，因此 >50 条目可能不存在。
    """
    out: Dict[int, float] = {}
    # 先拷贝初始的 1..50
    for r, v in initial_map.items():
        out[int(r)] = float(v) if v is not None else math.nan
    # 合并 >50 的 results.json（如果存在）
    for r, v in cca_results_map.items():
        try:
            rr = int(r)
        except Exception:
            rr = r
        if isinstance(rr, int) and rr > 50:
            out[rr] = float(v) if v is not None else math.nan
    return out


def align_to_56_rounds(cca_map: Dict[int, float], ml_map: Dict[int, float]) -> Tuple[List[int], List[float], List[float]]:
    """
    生成全局轮次 0..56：
    - 0 轮置为 0；
    - 1..50 使用 CCA 初始数据（支持键为 1..50 或 0..49）；
    - 1..50 的 ML-CCA 复制 CCA 值；
    - 51..56 使用 ML-CCA 的 6 个轮次（按键排序取前6）。
    """
    total_rounds = 57
    x = list(range(total_rounds))

    cca_series = [math.nan] * total_rounds
    ml_series = [math.nan] * total_rounds

    # round 0
    cca_series[0] = 0.0
    ml_series[0] = 0.0

    # 填充 1..50 CCA
    for r in range(1, 51):
        if r in cca_map:
            cca_series[r] = float(cca_map[r]) if cca_map[r] is not None else math.nan
        elif (r - 1) in cca_map:
            v = cca_map[r - 1]
            cca_series[r] = float(v) if v is not None else math.nan

    # ML-CCA 前50复制 CCA
    for r in range(1, 51):
        ml_series[r] = cca_series[r]

    # ML-CCA 51..56 映射 6 个轮次
    ml_keys = sorted(list(ml_map.keys()))
    ml_vals: List[float] = []
    for k in ml_keys:
        v = ml_map.get(k)
        try:
            ml_vals.append(float(v) if v is not None else math.nan)
        except Exception:
            ml_vals.append(math.nan)
        if len(ml_vals) >= 6:
            break
    for i, v in enumerate(ml_vals):
        tgt = 51 + i
        if tgt < total_rounds:
            ml_series[tgt] = v

    return x, cca_series, ml_series


def plot_efficiency_scw_comparison():
    # 读取 CCA 后50轮（results.json）与初始前50轮（initial_cca_result.json）
    cca_data = load_json(CCA_JSON)
    initial_cca_data = load_json(INITIAL_CCA_JSON)
    ml_data = load_json(MLCCA_JSON)

    # 初始钟拍前50轮（从 initial_cca_result.json 的 clock_rounds 中提取）
    initial_eff = extract_series_from_initial_cca(initial_cca_data, "efficiency")
    initial_scw = extract_series_from_initial_cca(initial_cca_data, "scw")

    # results.json 中的标准键（若包含 >50 的条目则用于合并）
    cca_eff_post = get_series(cca_data, "Efficiency per Iteration")
    cca_scw_post = get_series(cca_data, "SCW per Iteration")

    # 合并为混合的 CCA 显示序列：前50轮来自 initial_cca_result.json，之后来自 results.json（若存在）
    cca_eff = combine_initial_and_post50(initial_eff, cca_eff_post)
    cca_scw = combine_initial_and_post50(initial_scw, cca_scw_post)

    ml_eff = get_series(ml_data, "Efficiency per Iteration")
    ml_scw = get_series(ml_data, "SCW per Iteration")

    x, eff_cca, eff_ml = align_to_56_rounds(cca_eff, ml_eff)
    _, scw_cca, scw_ml = align_to_56_rounds(cca_scw, ml_scw)

    # 根据用户要求：后50轮的 CCA 数据对应 results.json 的 1..6 轮（映射到全局 51..56）
    def extract_post6(d: Dict, key: str) -> List[float]:
        raw = d.get(key, {}) or {}
        vals: List[float] = []
        # 优先使用键 1..6
        keys_pref = [str(i) for i in range(1, 7)]
        if all(k in raw for k in keys_pref):
            for i in range(1, 7):
                try:
                    vals.append(float(raw[str(i)]) if raw[str(i)] is not None else math.nan)
                except Exception:
                    vals.append(math.nan)
            return vals
        # 回退：使用键 0..5
        keys_alt = [str(i) for i in range(0, 6)]
        if all(k in raw for k in keys_alt):
            for i in range(0, 6):
                try:
                    vals.append(float(raw[str(i)]) if raw[str(i)] is not None else math.nan)
                except Exception:
                    vals.append(math.nan)
            return vals
        # 最后回退：按键排序取前6
        for k in sorted(raw.keys(), key=lambda s: int(s))[:6]:
            try:
                vals.append(float(raw[k]) if raw[k] is not None else math.nan)
            except Exception:
                vals.append(math.nan)
        return vals

    post_eff6 = extract_post6(cca_data, "Efficiency per Iteration")
    post_scw6 = extract_post6(cca_data, "SCW per Iteration")

    for i, v in enumerate(post_eff6):
        tgt = 51 + i
        if 0 <= tgt < len(eff_cca):
            eff_cca[tgt] = v
    for i, v in enumerate(post_scw6):
        tgt = 51 + i
        if 0 <= tgt < len(scw_cca):
            scw_cca[tgt] = v

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Efficiency comparison
    ax = axes[0]
    ax.plot(x, eff_cca, color=CCA_COLOR, linewidth=2.0, marker="x", markersize=3, label="CCA Efficiency")
    ax.plot(x, eff_ml, color=MLCCA_COLOR, linewidth=2.0, marker="o", markersize=3, label="EV-MLCCA Efficiency")
    ax.axvline(50.5, color="#888888", linestyle="--", linewidth=1.2)
    ax.set_title("Efficiency per Iteration")
    ax.set_xlabel("Round")
    ax.set_ylabel("Efficiency")
    # y轴从0开始，体现“逐渐收敛到50轮”的趋势
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    ax.set_xlim(0, 56)
    ax.set_xticks(list(range(0, 57, 5)))

    # SCW comparison
    ax2 = axes[1]
    ax2.plot(x, scw_cca, color=CCA_COLOR, linewidth=2.0, marker="x", markersize=3, label="CCA Social Welfare")
    ax2.plot(x, scw_ml, color=MLCCA_COLOR, linewidth=2.0, marker="o", markersize=3, label="EV-MLCCA Social Welfare")
    ax2.axvline(50.5, color="#888888", linestyle="--", linewidth=1.2)
    ax2.set_title("Social Welfare per Iteration")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Social Welfare")
    # y轴从0开始
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best")
    ax2.set_xlim(0, 56)
    ax2.set_xticks(list(range(0, 57, 5)))

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "efficiency_social_welfare_comparison_56_rounds.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_mlcca_clearing_error():
    ml_data = load_json(MLCCA_JSON)
    ce_map = ml_data.get("Clearing Error per Iteration", {})

    rounds: List[int] = []
    values: List[float] = []

    # keys are strings like "1".."6"
    for k in sorted(ce_map.keys(), key=lambda s: int(s)):
        try:
            rounds.append(int(k))
            values.append(float(ce_map[k]))
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, values, color=MLCCA_COLOR, linewidth=2.0, marker="o", markersize=4)
    ax.set_title("EV-MLCCA Clearing Error")
    ax.set_xlabel("Round")
    ax.set_ylabel("Clearing Error")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xticks(rounds)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mlcca_clearing_error_rounds.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p1 = plot_efficiency_scw_comparison()
    p2 = plot_mlcca_clearing_error()
    print("Saved:", p1)
    print("Saved:", p2)