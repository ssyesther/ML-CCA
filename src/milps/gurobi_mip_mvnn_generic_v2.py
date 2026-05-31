# Libs
import time
from timeit import default_timer as timer
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch

#%% NEW SUCCINCT MVNN MIP FOR CALCULATING MAX UTILITY BUNDLE FOR SINGLE BIDDER: argmax_x {MVNN_i(x)-p*x}
class GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX_V2:

    def __init__(self,
                 model,
                 SATS_domain,
                 capacities,
                 bidder_id
                 ):
        
        # MVNN PARAMETERS
        self.model = model  # MVNN TORCH MODEL
        # 只处理中间隐藏层（跳过输入层和输出层）
        self.ts = [
            layer.ts.data.cpu().detach().numpy().reshape(-1, 1)
            for i, layer in enumerate(model.layers)
            if i != 0  # 跳过输入层
            and i != len(model.layers)-1  # 跳过输出层
            and hasattr(layer, 'ts')  # 确保是bReLU层
        ]
        # 调整断言逻辑
        for ts_idx, ts in enumerate(self.ts):
            # 获取对应的隐藏层索引（model.layers中第1层到倒数第2层）
            layer_idx = ts_idx + 1  # 因为跳过了输入层
            corresponding_layer = model.layers[layer_idx]
            
            # 验证ts维度与层输出维度匹配
            assert ts.shape[0] == corresponding_layer.out_features, (
                f"层{layer_idx}({corresponding_layer.__class__.__name__}) "
                f"ts参数维度错误，期望{corresponding_layer.out_features}，"
                f"实际{ts.shape[0]}"
            )
        assert len(self.ts) == len(model.layers) - 2, f"模型层数不匹配，期望隐藏层数：{len(model.layers)-2}，实际收集ts数：{len(self.ts)}"
        # MIP VARIABLES
        self.y_variables = []  # CONT VARS 1
        self.a_variables = []  # BINARY VARS 1
        self.b_variables = []  # BINARY VARS 2
        self.case_counter = {'Case1': 0, 'Case2': 0, 'Case3': 0, 'Case4': 0, 'Case5': 0}
        # SATS PARAMETERS
        self.SATS_domain = SATS_domain
        self.bidder_id = bidder_id
        self.capacities = capacities

    def calc_preactivated_box_bounds(self,
                                     input_upper_bound=1,
                                     input_lower_bound=0,
                                     verbose = False):

        # BOX-bounds for y variable (preactivated!!!!) as column vectors

        # Initialize
        self.upper_box_bounds = [np.array([input_upper_bound] * self.model.layers[0].in_features, dtype=np.int64).reshape(-1, 1)]
        self.lower_box_bounds = [np.array([input_lower_bound] * self.model.layers[0].in_features, dtype=np.int64).reshape(-1, 1)]

        # Propagate through Network
        for i, layer in enumerate(self.model.layers[1:-1], start=1):
            W = layer.weight.data.cpu().detach().numpy()
            b = layer.bias.data.cpu().detach().numpy().reshape(-1, 1) if layer.bias is not None else np.zeros((W.shape[0], 1))

            # 获取对应层的t值（维度需与当前层输出一致）
            t = self.ts[i-1]  # 假设self.ts已正确对齐
        
            # 调整维度对齐：将前一层输出转换为行向量，t转换为列向量
            prev_upper = self.upper_box_bounds[-1].T  # 形状变为(1, 前层神经元数)
            prev_lower = self.lower_box_bounds[-1].T  # 形状变为(1, 前层神经元数)
        
            # 计算激活值（保持维度兼容性）
            activated_upper = np.minimum(t, np.maximum(0, prev_upper))
            activated_lower = np.minimum(t, np.maximum(0, prev_lower))
        
            # 矩阵乘法维度对齐
            upper = activated_upper @ W.T + b.T  # 形状(1, 当前层神经元数)
            lower = activated_lower @ W.T + b.T
        
        # 转置回列向量存储
        self.upper_box_bounds.append(upper.reshape(-1, 1))
        self.lower_box_bounds.append(lower.reshape(-1, 1))

        if verbose:
            print('Upper Box Bounds:')
            print(self.upper_box_bounds)
        if verbose:
            print('Lower Box Bounds:')
            print(self.lower_box_bounds)

    def phi(self, x, t):
        # Bounded ReLU (bReLU) activation function for MVNNS with cutoff t
        return np.minimum(t, np.maximum(0, x)).reshape(-1, 1)

    def generate_mip(self,
                     prices = None,
                     MIPGap = None,
                     verbose = False,
                     ):

        self.mip = gp.Model("MVNN MIP2")

        # Add IntFeasTol, primal feasibility
        if MIPGap:
            self.mip.Params.MIPGap = MIPGap

        self.calc_preactivated_box_bounds(verbose=verbose)

        # --- Variable declaration -----
        # NOTE: the difference to the original MIP is the following. 
        # suppose the capacities for the 3 items are: {a:3, b:2 , c:1}
        # then the extended vector is: {x_a_0, x_a_1, x_a_2, x_b_0, x_b_1, x_c_0}
        if isinstance(self.capacities, dict):
            capacity_list = list(self.capacities.values())
        else:
            capacity_list = self.capacities.tolist()
        print(f"[MIP生成] 原始价格维度: {len(prices)} 容量列表长度: {len(capacity_list)}")
        input_dim = sum(capacity_list)  # 计算扩展后的总维度
        print(f"\n[Debug] Capacities解析:")
        print(f"输入capacities参数类型: {type(self.capacities)}")
        print(f"解析后capacity_list: {capacity_list}")
        print(f"扩展后总维度: {input_dim} vs 模型输入层: {self.model.layers[0].in_features}")
        print(f"Capacity config check:")
        print(f"SATS Domain: {self.SATS_domain}")
        print(f"Bidder ID: {self.bidder_id}")
        print(f"Capacities: {self.capacities}")
        assert self.model.layers[0].in_features == input_dim, (
            f"输入层维度不匹配！"
            f"模型期望: {self.model.layers[0].in_features}, "
            f"实际扩展维度: {input_dim}"
        )
        # 创建扁平化的输入层变量
        input_vars = []
        for good_idx, capacity in enumerate(capacity_list):
            for unit_idx in range(capacity):
                input_vars.append(
                    self.mip.addVar(
                        name=f"x_{good_idx}_{unit_idx}", 
                        vtype=GRB.BINARY
                    )
                )
        self.y_variables.append(input_vars)  # 输入层变量存储为单个列表
        print(f"[MIP验证] 输入层变量数: {len(input_vars)} 模型期望: {self.model.layers[0].in_features}")  # 新添加
        assert len(input_vars) == self.model.layers[0].in_features, "MIP输入变量数与模型输入层不匹配"
        # self.y_variables = [self.y_variables.flatten()]   # so that they have the expected shape, i.e., all variables of the 
        # input layer are in the first list element of the list y_variables.

        for (i, layer) in enumerate(self.model.layers[1:-1], start=1):

            # ----------------------------
            tmp_y_variables = []
            ts_index = i - 1
            for j in range(len(layer.weight.data)):
                tmp_y_variables.append(self.mip.addVar(name=f'y_{i+1}_{j}', vtype = GRB.CONTINUOUS, lb = 0, ub=self.ts[ts_index][j, 0]))
            self.y_variables.append(tmp_y_variables)
            # ----------------------------

            self.a_variables.append(self.mip.addVars([j for j in range(len(layer.weight.data))], name=f'a_{i+1}_', vtype = GRB.BINARY))
            self.b_variables.append(self.mip.addVars([j for j in range(len(layer.weight.data))], name=f'b_{i+1}_', vtype = GRB.BINARY))

        layer = self.model.output_layer
        self.y_variables.append(
            self.mip.addVars(
                [j for j in range(len(layer.weight.data))], 
                name='y_output_', 
                vtype=GRB.CONTINUOUS, 
                lb=0
            )
        )
        
        # ---  MVNN Contraints ---
        for layer_idx, layer in enumerate(self.model.layers[1:-1], start=1):
            current_layer_index = layer_idx
            prev_layer_index = layer_idx - 1
            ts_index = layer_idx - 1  # ts索引从0开始
            current_ts = self.ts[ts_index]
            var_index = layer_idx - 1
            print(f"Layer {layer_idx} weight shape: {layer.weight.data.shape}")
            print(f"ts for layer {layer_idx}: {current_ts.shape if ts_index < len(self.ts) else 'N/A'}")
            print(f"Processing layer {layer_idx}, ts shape: {current_ts.shape}")
            W = layer.weight.data
            input_dim = W.shape[1]  # 输入维度（前一层神经元数）
            output_dim = W.shape[0]  # 输出维度（当前层神经元数）
            # 添加维度验证断言
            assert current_ts.shape[0] == output_dim, (
                f"层{layer_idx}的ts维度不匹配，"
                f"期望{output_dim}，实际{current_ts.shape[0]}"
            )
            for j in range(output_dim):
                bias = layer.bias.data[j].item() if layer.bias is not None else 0.0
                weight = W[j]
                weight_cols = weight.shape[0]
                quicksum_terms = (
                    weight[k].item() * self.y_variables[prev_layer_index][k]
                    for k in range(input_dim)
                )
                # CASE 1 -> REMOVAL:
                if self.lower_box_bounds[current_layer_index][j, 0] >= current_ts[j, 0]:
                    self.y_variables[current_layer_index][j] = current_ts[j, 0]
                    self.case_counter['Case1'] += 1
                # CASE 2 -> REMOVAL:
                elif self.upper_box_bounds[current_layer_index][j, 0] <= 0:
                    self.y_variables[current_layer_index][j] = 0
                    self.case_counter['Case2'] += 1
                # CASE 3 -> REMOVAL:
                elif (self.lower_box_bounds[current_layer_index][j, 0] >= 0 and self.lower_box_bounds[current_layer_index][j, 0] <= current_ts[j, 0]) and (self.upper_box_bounds[current_layer_index][j, 0] >= 0 and self.upper_box_bounds[current_layer_index][j, 0] <= current_ts[j, 0]):
                    self.y_variables[current_layer_index][j] = gp.quicksum(weight[k] * self.y_variables[prev_layer_index][k] for k in range(len(weight))) + bias
                    self.case_counter['Case3'] += 1
                # CASE 4 -> REMOVAL:
                elif self.lower_box_bounds[current_layer_index][j, 0] >= 0:
                    # TYPE 1 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] <= current_ts[j, 0], name=f'HLayer_{current_layer_index}_{j}_Case4_CT1')
                    # TYPE 2 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] <= gp.quicksum(weight[k] * self.y_variables[prev_layer_index][k] for k in range(len(weight))) + bias, name=f'HLayer_{current_layer_index}_{j}_Case4_CT2')
                    # TYPE 3 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] >= self.b_variables[var_index][j] * current_ts[j, 0], name=f'HLayer_{current_layer_index}_{j}_Case4_CT3')
                    # TYPE 4 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] >= gp.quicksum(weight[k] * self.y_variables[prev_layer_index][k] for k in range(len(weight))) + bias + (current_ts[j, 0] - self.upper_box_bounds[current_layer_index][j, 0]) * self.b_variables[var_index][j], name=f'HLayer_{current_layer_index}_{j}_Case4_CT4')
                    self.case_counter['Case4'] += 1
                # CASE 5 -> REMOVAL:
                elif self.upper_box_bounds[current_layer_index][j, 0] <= current_ts[j, 0]:
                    # TYPE 1 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] <= self.a_variables[var_index][j] * current_ts[j, 0], name=f'HLayer_{current_layer_index}_{j}_Case5_CT1')
                    # TYPE 2 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] <= gp.quicksum(quicksum_terms) + bias - self.lower_box_bounds[current_layer_index][j, 0]*(1-self.a_variables[var_index][j]), name=f'HLayer_{current_layer_index}_{j}_Case5_CT2')
                    # TYPE 3 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] >= 0, name=f'HLayer_{current_layer_index}_{j}_Case5_CT3')
                    # TYPE 4 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] >= gp.quicksum(weight[k] * self.y_variables[prev_layer_index][k] for k in range(len(weight))) + bias, name=f'HLayer_{current_layer_index}_{j}_Case5_CT4')
                    self.case_counter['Case5'] += 1
                # DEFAULT CASE -> NO REMOVAL:
                else:
                    # TYPE 1 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] <= self.a_variables[var_index][j] * current_ts[j, 0], name=f'HLayer_{current_layer_index}_{j}_Default_CT1')
                    # TYPE 2 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] <= gp.quicksum(weight[k] * self.y_variables[prev_layer_index][k] for k in range(len(weight))) + bias - self.lower_box_bounds[current_layer_index][j, 0]*(1-self.a_variables[var_index][j]), name=f'HLayer_{current_layer_index}_{j}_Default_CT2')
                    # TYPE 3 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] >= self.b_variables[var_index][j] * current_ts[j, 0], name=f'HLayer_{current_layer_index}_{j}_Default_CT3')
                    # TYPE 4 Constraints for the whole network (except the output layer)
                    self.mip.addConstr(self.y_variables[current_layer_index][j] >= gp.quicksum(weight[k] * self.y_variables[prev_layer_index][k] for k in range(len(weight))) + bias + (current_ts[j, 0] - self.upper_box_bounds[current_layer_index][j, 0]) * self.b_variables[var_index][j], name=f'HLayer_{current_layer_index}_{j}_Default_CT4')

        output_weight = self.model.output_layer.weight.data[0]
        if (self.model.output_layer.bias is not None):
            output_bias = self.model.output_layer.bias.data
        else:
            output_bias = 0

        if output_bias!=0:
            raise ValueError('output_bias is not 0')

        # Final output layer of MVNN
        # Linear Constraints for the output layer WITH lin_skip_layer: W*y
        if hasattr(self.model, 'lin_skip_layer'):
            lin_skip_W = self.model.lin_skip_layer.weight.detach().cpu().numpy() 
            self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables[-2][k] for k in range(len(output_weight))) + output_bias + gp.quicksum(lin_skip_W[0, i]*self.y_variables[0][i] for i in range(lin_skip_W.shape[1])) == self.y_variables[-1][0], name='output_layer')
        # Linear Constraints for the output layer WIHTOUT lin_skip_layer: W*y + W_0*x
        else:
            self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables[-2][k] for k in range(len(output_weight))) + output_bias == self.y_variables[-1][0], name='output_layer')

        # GSVM specific allocation constraints
        if self.SATS_domain == 'GSVM':
            if self.bidder_id == 6:
                #print(f'Adding GSVM specific constraints for national bidder: {self.bidder_id}')
                GSVM_national_bidder_goods_of_interest_one_hot_encoding_complement = [i not in self.GSVM_national_bidder_goods_of_interest for i in range(len(prices))]
                self.mip.addConstr(gp.quicksum(self.y_variables[0][i]*GSVM_national_bidder_goods_of_interest_one_hot_encoding_complement[i] for i in range(len(prices)))==0, name="GSVM_CT_NationalBidder")
            else:
                #print(f'Adding GSVM specific constraints for regional bidder: {self.bidder_id}')
                self.mip.addConstr(gp.quicksum(self.y_variables[0][i] for i in range(len(prices)))<=4, name="GSVM_CT_RegionalBidder")
        
        # Constraint forcing you to pick the units corresponding to the same good on the right order
        # i.e, x_0 before x_1, x_1 before x_2, etc.
        capacity_offset = 0 
        for i in range(len(capacity_list)):
            for j in range(1, capacity_list[i]):
                self.mip.addConstr(self.y_variables[0][capacity_offset + j] <= self.y_variables[0][capacity_offset + j-1], name=f'ordering_{i}_{j}')
            capacity_offset += capacity_list[i]  # we need the offset to jump to the next good
        
        # --- Objective Declaration ---
        # need to conver the prices to the extended representation. 
        prices_extended_representation = []
        if isinstance(self.capacities, dict):
            capacity_values = list(self.capacities.values())  # 直接获取容量数值
        else:
            capacity_values = self.capacities
        
        # 添加类型验证
        print(f"[DEBUG] 转换后容量数值类型: {type(capacity_values)}")
        print(f"样例值: {capacity_values[:3]}")
        for idx, capacity in enumerate(capacity_values):  # 现在capacities是list类型
            assert isinstance(capacity, int), f"容量值必须为整数，当前类型: {type(capacity)}, 值: {capacity}"
            if idx >= len(prices):
                raise IndexError(f"商品索引{idx}超出价格向量长度{len(prices)}")
        
            # 添加数值稳定性处理
            price_val = max(prices[idx], 1e-10)
            prices_extended_representation += [price_val] * capacity   
        self.mip.setObjective(self.y_variables[-1][0] - gp.quicksum(self.y_variables[0][i] * prices_extended_representation[i] for i in range(len(prices_extended_representation))), GRB.MAXIMIZE)

        if (verbose):
            self.mip.write('MVNN_mip2_'+'_'.join(time.ctime().replace(':', '-').split(' '))+'.lp')


    def update_prices_in_objective(self, prices):
        # convert prices to extended representation
        # 转换价格为列表格式
        if isinstance(prices, dict):
            prices = [prices[k] for k in sorted(prices.keys())]

        # 生成扩展价格表示
        prices_extended_representation = []
        for idx, (good_id, capacity) in enumerate(sorted(self.capacities.items())):
            # 添加索引保护（使用enumerate生成的数字索引）
            if idx >= len(prices):
                raise IndexError(f"商品索引{idx}超出价格向量长度{len(prices)}")
    
            # 添加数值稳定性处理
            price_val = max(prices[idx], 1e-10)  # 使用数字索引访问价格
            prices_extended_representation += [price_val] * capacity
        self.mip.setObjective(self.y_variables[-1][0] - gp.quicksum(self.y_variables[0][i] * prices_extended_representation[i] for i in range(len(prices_extended_representation))), GRB.MAXIMIZE)
        return

    def add_forbidden_bundle(self, bundle):
        expr_list = [self.y_variables[0][m] if bundle[m] == 1 else (1 - self.y_variables[0][m]) for m in range(len(bundle))]
        self.mip.addConstr(gp.quicksum(expr_list) <= len(bundle) - 0.1, name='alreadyQueried')
        self.mip.update()
        return

    def solve_mip(self,
                  outputFlag = False,
                  verbose = True,
                  timeLimit = np.inf,
                  MIPGap = 1e-04,
                  IntFeasTol = 1e-5,
                  FeasibilityTol = 1e-6
                  ):
        
        if not verbose:
            self.mip.Params.LogToConsole = 0
            self.mip.Params.OutputFlag = 0

        # set solve parameter (if not sepcified, default values are used)
        self.mip.Params.timeLimit = timeLimit # Default +inf
        self.mip.Params.MIPGap = MIPGap # Default 1e-04
        self.mip.Params.IntFeasTol = IntFeasTol # Default 1e-5
        self.mip.Params.FeasibilityTol = FeasibilityTol # Default 1e-6
        #

        self.start = timer()
        self.mip.Params.OutputFlag = outputFlag
        self.mip.optimize()
        self.end = timer()

        self.optimal_schedule = []
        # TODO: test try-catch for non-feasible solution
        try:
            for i in range(len(self.y_variables[0])):
                if(self.y_variables[0][i].x >= 0.99):
                    self.optimal_schedule.append(1)
                else:
                    self.optimal_schedule.append(0)
        except:
            self._print_info()
            raise ValueError('MIP did not solve succesfully!')

        if verbose:
            self._print_info()

        return self.optimal_schedule

    def solve_mip_rv(self,
                    outputFlag = False,
                    verbose = True,
                    timeLimit = np.inf,
                    MIPGap = 1e-04,
                    IntFeasTol = 1e-5,
                    FeasibilityTol = 1e-6
                    ):
        
        if not verbose:
            self.mip.Params.LogToConsole = 0
            self.mip.Params.OutputFlag = 0
        
        # set solve parameter (if not sepcified, default values are used)
        self.mip.Params.timeLimit = timeLimit # Default +inf
        self.mip.Params.MIPGap = MIPGap # Default 1e-04
        self.mip.Params.IntFeasTol = IntFeasTol # Default 1e-5
        self.mip.Params.FeasibilityTol = FeasibilityTol # Default 1e-6
        #

        self.start = timer()
        self.mip.Params.OutputFlag = outputFlag
        self.mip.optimize()
        self.end = timer()

        self.optimal_schedule = []
        # TODO: test try-catch for non-feasible solution
        try:
            for i in range(len(self.y_variables[0])):
                if(self.y_variables[0][i].x >= 0.99):
                    self.optimal_schedule.append(1)
                else:
                    self.optimal_schedule.append(0)
        except:
            self._print_info()
            raise ValueError('MIP did not solve succesfully!')

        if verbose:
            self._print_info()

        if self.mip.Params.MIPGap > 0: 
            print('MIPGap larger than 0: ', self.mip.Params.MIPGap)

        return self.optimal_schedule, self.mip.getObjective().getValue()

    def _print_info(self):
        print(*['*']*30)
        print('MIP INFO:')
        print(*['-']*30)
        print(f'Name: {self.mip.ModelName}')
        print(f'Goal: {self._model_sense_converter(self.mip.ModelSense)}')
        print(f'Objective: {self.mip.getObjective()}')
        print(f'Number of variables: {self.mip.NumVars}')
        print(f' - Binary {self.mip.NumBinVars}')
        print(f'Number of linear constraints: {self.mip.NumConstrs}')
        print(f'Primal feasibility tolerance for constraints: {self.mip.Params.FeasibilityTol}')
        print(f'Integer feasibility tolerance: {self.mip.Params.IntFeasTol}')
        print(f'Relative optimality gap: {self.mip.Params.MIPGap}')  # we may want this 
        print(f'Time Limit: {self.mip.Params.TimeLimit}')
        print('')
        print('MIP SOLUTION:')
        print(*['-']*30)
        print(f'Status: {self._status_converter(self.mip.status)}')
        print(f'Elapsed in sec: {self.end - self.start}')
        print(f'Reached Relative optimality gap: {self.mip.MIPGap}')   
        print(f'Optimal Allocation: {self.optimal_schedule}')
        print(f'Objective Value: {self.mip.ObjVal}')
        print(f'Number of stored solutions: {self.mip.SolCount}')
        print('IA Case Statistics:')
        for k, v in self.case_counter.items():
            print(f' - {k}: {v}')
        print(*['*']*30)

    def _status_converter(self, int_status):
        status_table = ['woopsies!', 'LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED', 'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT', 'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL', 'INPROGRESS', 'USER_OBJ_LIMIT']
        return status_table[int_status]

    def _model_sense_converter(self, int_sense):
        if int_sense == 1:
            return 'Minimize'
        elif int_sense == -1:
            return 'Maximize'
        else:
            raise ValueError('int_sense needs to be -1:maximize or 1: minimize')
