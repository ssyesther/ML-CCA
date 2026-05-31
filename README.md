# 基于组合拍卖机制的充电定价与协同调度优化算法研究 (ML-CCA)

本项目主要研究基于机器学习驱动的组合时钟拍卖机制（ML-CCA）在城市级充电资源调度中的应用。

## 技术报告
本项目包含一份详细的技术报告，涵盖了理论研究与代码实现深度解析：
- [技术报告_最终深度超长版_20页目标.md](技术报告_最终深度超长版_20页目标.md)
- [技术报告_第四章源码深度扩充版.md](技术报告_第四章源码深度扩充版.md)

## 项目背景
本项目基于 AAAI 2024 发表的论文：
**Machine Learning-powered Combinatorial Clock Auction**<br/>
Ermis Soumalias*, Jakob Weissteiner*, Jakob Heiss, and Sven Seuken.<br/>
*In [Proceedings of the AAAI Conference on Artificial Intelligence Vol 38](https://doi.org/10.1609/aaai.v38i9.28850), Vancouver, CAN, Feb 2024* <br/>
Full paper version including appendix: [[pdf](http://arxiv.org/abs/2308.10226)]

## 环境要求
* Python 3.8
* Java 8 (or later)
* Gurobi Python API
* CPLEX Python API

## 快速开始
进入 `src` 目录运行以下命令：
```bash
python3 sim_mlca_dq.py --domain GSVM --qinit 20 --seed 157 --new_query_option gd_linear_prices_on_W_v3
```

## 联系方式
Maintainer: shaoshiyu (ssyesther)


