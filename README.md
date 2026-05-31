# 基于组合拍卖机制的充电定价与协同调度优化算法研究 (ML-CCA)

本项目实现了一种机器学习驱动的组合时钟拍卖机制 (Machine Learning-powered Combinatorial Clock Auction, ML-CCA)，专门用于解决城市级电动汽车 (EV) 的充电调度与动态定价问题。通过单调价值神经网络 (MVNN) 学习用户复杂的非加性偏好，并结合非单调价格发现算法实现社会福利最大化。

## 核心领域：MSVM (城市级充电仿真)

本项目中最核心的仿真场景为 **MSVM (Multi-Station Value Model)**，它模拟了一个具有 50 辆电动汽车和 6 个充电站的城市调度环境。

### 1. 投标者建模 (CustomMSVMBidder)
在 `src/custom_msvm_domain.py` 中，每一辆电动汽车被建模为一个具有异质偏好的投标者：
- **时空坐标**：模拟车辆在城市中的初始位置。
- **距离惩罚 (Distance Penalty)**：反映用户对前往不同充电站的地理成本敏感度。
- **基础价值 (Base Value)**：用户对电量的基本支付意愿，服从对数正态分布。
- **连续性偏好 (Required Length & Continuity Bonus)**：模拟用户希望连续充电的需求。如果分配的时段满足连续性要求，将获得额外的效用加成。

### 2. 价值函数逻辑
用户的估值 $v_n(x)$ 并非简单的线性加和，而是包含以下部分：
- **时间效用**：基于边际效用递减原则。
- **空间成本**：基于欧几里得距离的非线性惩罚。
- **连续性奖励**：鼓励系统分配连续的充电时间槽。

### 3. 拍卖环境 (CustomMSVMAuction)
环境负责管理 144 个资源项（6 个站点 × 24 个时段），并协调所有投标者的需求响应。

## 如何运行 MSVM 仿真

### 环境准备
- **Python 3.8**
- **Gurobi Optimizer**: 用于求解每一轮的价格发现子问题。
- **IBM ILOG CPLEX**: 用于处理大规模组合优化。
- **PySats**: 频谱拍卖测试套件的扩展支持。

### 运行命令
进入 `src` 目录，运行以下命令启动 MSVM 场景下的 ML-CCA 拍卖：

```bash
python3 sim_mlca_dq.py --domain MSVM --qinit 50 --seed 1 --new_query_option gd_linear_prices_on_W_v3
```

**关键参数说明：**
- `--domain MSVM`: 指定使用城市充电调度仿真场景。
- `--qinit 50`: 初始采样轮数。系统先进行 50 轮传统拍卖以收集初始数据。
- `--seed 1`: 随机种子，确保实验可复现。
- `--new_query_option gd_linear_prices_on_W_v3`: 启用基于对偶清算势能梯度的机器学习定价引擎。

## 项目结构
- `src/mvnns/`: 单调价值神经网络的底层实现，包含权重投影和单调性约束。
- `src/mlca_demand_queries/`: 拍卖机制的主控逻辑，包括 `mlca_dq_mechanism.py`。
- `src/milps/`: 封装了 Gurobi 接口，用于高效求解赢家决定问题 (WDP)。
- `src/custom_msvm_domain.py`: 专门为本研究定制的 MSVM 仿真领域定义。

## 联系方式
Maintainer: shaoshiyu (ssyesther)

