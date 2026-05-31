import gurobipy as gp
from gurobipy import GRB
import numpy as np

# 步骤 1：创建模型
model = gp.Model("10000Var_Model")

# 步骤 2：添加 10000 个非负连续变量（名称格式 x_0 到 x_9999）
x = model.addVars(10000, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

# 步骤 3：设置目标函数（最小化所有变量的和）
model.setObjective(gp.quicksum(x[i] for i in x.keys()), GRB.MINIMIZE)

# 步骤 4：添加约束（变量总和不超过 1000000）
model.addConstr(gp.quicksum(x[i] for i in x.keys()) <= 1000000, "Total_Constraint")

# 步骤 5：求解模型（关闭日志输出）
model.setParam('OutputFlag', 0)
model.optimize()

# 步骤 6：输出结果
if model.status == GRB.OPTIMAL:
    print("求解成功！")
    print("最优目标值:", model.objVal)
    # 输出前 5 个变量的值
    for i in range(5):
        print(f"x[{i}] =", x[i].X)
else:
    print("求解失败，状态码:", model.status)