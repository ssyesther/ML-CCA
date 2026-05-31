import numpy as np
from itertools import combinations
from collections import defaultdict
import logging

class CustomMSVMBidder:
    def __init__(self, bidder_id, stations, params):
        self.bidder_id = bidder_id
        self.x = params['coordinates']['x']
        self.y = params['coordinates']['y']
        self.distance_penalty = params['distance_penalty']
        self.station_info = {s['id']:s for s in stations}
        self.station_capacities = {s['id']: s['num_piles'] for s in stations}
        
        # 允许的充电站-时段组合
        try:
            self.required_length = params['required_length']
            self.base_value = float(params['base_value'])
            self.continuity_bonus = float(params['continuity_bonus'])
            self.allowed_slots = self._parse_allowed_slots(params['allowed_slots'])
            self.num_slots = len(params['allowed_slots'])

            self.distance_penalty = float(params['distance_penalty'])
        except Exception as e:
            raise ValueError(f"Error initializing bidder {bidder_id}: {str(e)}")
        
        # 添加参数校验
        self._validate_params(params)
        
    def _validate_params(self, params):
        """验证投标者参数有效性"""
        errors = []
        
        # 检查允许时段是否为空
        if not self.allowed_slots:
            errors.append("allowed_slots cannot be empty")
            
        # 检查价值参数合理性
        if self.base_value <= 0:
            errors.append(f"base_value must be positive (got {self.base_value})")
            
        if self.required_length < 1:
            errors.append(f"required_length must be >=1 (got {self.required_length})")
            
        # 检查坐标范围
        if not (-1000 <= self.x <= 1000):
            errors.append(f"x coordinate out of range (got {self.x})")
            
        if errors:
            raise ValueError(f"Invalid parameters for bidder {self.bidder_id}:\n" + "\n".join(errors))

    def _parse_allowed_slots(self, slots):
        """解析允许的时段（增强去重和排序）"""
        allowed = defaultdict(list)
        try:
            seen = set()  # 新增：防止重复slot
            for slot in slots:
                parts = slot.split('_')
                if len(parts) != 3:
                    continue  # 跳过无效格式
                
                # 标准化station_id格式
                station_id = f"{parts[0]}_{parts[1]}"
                try:
                    time_slot = int(parts[2])
                    unique_key = f"{station_id}_{time_slot}"
                    if unique_key in seen:
                        continue  # 跳过重复slot
                    seen.add(unique_key)
                    
                    allowed[station_id].append(time_slot)
                except ValueError:
                    continue
            
            # 去重、排序、检查连续性
            clean_allowed = {}
            for station_id, slots in allowed.items():
                # 去重并排序
                unique_slots = sorted(list(set(slots)))
                clean_allowed[station_id] = unique_slots
            return clean_allowed
            
        except Exception as e:
            raise ValueError(f"Error parsing slots: {str(e)}")

    def calculate_value(self, bundle):
        """
        bundle: ["station0_9", "station0_10", "station1_14"]
        """
        total_value = 0
        
        # 按充电站分组处理
        for station_id, slots in self._group_by_station(bundle).items():
            # 计算距离成本
            station = self.station_info[station_id]
            distance = np.sqrt((self.x-station['x'])**2 + (self.y-station['y'])**2)
            cost = distance * self.distance_penalty
            
            # 计算时间价值
            time_value = self._calculate_time_value(slots)
            
            # 总价值 = 时间价值 - 距离成本
            station_value = max(0.0, time_value - cost)
            logging.info(f"Bidder {self.bidder_id} @ {station_id}: "
                         f"slots={slots} time_value={time_value:.2f} "
                         f"distance={distance:.2f} cost={cost:.2f} → {station_value:.2f}")
            
            total_value += station_value
            
        return total_value
    
    def _group_by_station(self, bundle):
        """分组处理不同充电站的时段（增强类型转换）返回格式：{station_id: [t1, t2...]}"""
        grouped = defaultdict(list)
        for slot in bundle:
            # 统一转换为字符串处理
            try:
                str_slot = str(int(slot)) if isinstance(slot, (np.generic, float)) else str(slot)
            except:
                continue
            
            # 格式验证
            parts = str_slot.split('_')
            if len(parts) < 3:
                continue  # 跳过非法格式
                
            station_id = f"{parts[0]}_{parts[1]}"
            if station_id in self.allowed_slots:
                try:
                    grouped[station_id].append(int(parts[2]))
                except ValueError:
                    continue
        return grouped

    def _calculate_time_value(self, slots):
        """时间价值计算逻辑"""
        sorted_slots = sorted(slots)
        
        max_streak = current = 1
        for i in range(1, len(sorted_slots)):
            if sorted_slots[i] == sorted_slots[i-1] + 1:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1

        if len(slots) == 1:
            return self.base_value
        
        # 计算有效长度（考虑充电桩需求）
        effective_length = min(max_streak, self.required_length)
        
        # 价值计算（每个时段只需要1个充电桩）
        if effective_length >= self.required_length:
            return self.base_value * effective_length * 1.5
        elif effective_length > 1:
            return self.base_value * effective_length + \
                   self.continuity_bonus * (effective_length ** 1.2)
        return 0

class CustomMSVMAuction:
    def __init__(self, stations, bidders, num_slots):
            self.stations = stations
            self.bidders = [
                CustomMSVMBidder(int(b['id'].split('_')[-1]), stations, b)
                for b in bidders
            ]
            self.num_slots = num_slots
            self.all_goods = self._generate_all_goods()
            self.good_index = {good: idx for idx, good in enumerate(self.all_goods)}  # 新增商品索引
            self.good_capacities = self._get_good_capacities()
            self.capacities = {good: cap for good, cap in zip(self.all_goods, self.good_capacities)}  # 保留字典格式
            try:
                # 添加投标者参数日志
                logging.info("Initializing CustomMSVMAuction with parameters:")
                logging.info(f"Number of stations: {len(stations)}")
                logging.info(f"Number of bidders: {len(bidders)}")
                for i, bidder in enumerate(self.bidders):
                    logging.info(f"Bidder {i} allowed slots: {bidder.allowed_slots}")
                    logging.info(f"Bidder {i} base_value: {bidder.base_value}")         
            except Exception as e:
                raise ValueError(f"Error initializing auction: {str(e)}")

    def get_capacities(self):  # 保留SATS接口方法
        """返回商品容量字典（兼容SATS接口）"""
        for k, v in self.capacities.items():
            if not isinstance(v, int):
                raise TypeError(f"容量值必须为整数，商品ID: {k} 类型: {type(v)} 值: {v}")
        return self.capacities
        
    def _generate_all_goods(self):
        """生成所有商品ID"""
        try:
            return [
                f"{s['id']}_{t}" 
                for s in self.stations 
                for t in range(self.num_slots)
            ]
        except Exception as e:
            raise ValueError(f"Error generating goods: {str(e)}")

    def get_best_bundles(self, bidder_id, prices, max_bundle_size):
        """获取最优充电套餐组合（考虑多充电站）"""
        # 统一处理字符串和数字类型的bidder_id
        if isinstance(bidder_id, str):
            bidder_num = int(bidder_id.split('_')[-1])  # 处理"Bidder_X"格式
        else:
            bidder_num = int(bidder_id)
        
        # 修改匹配条件为数字ID比较
        bidder = next(b for b in self.bidders if b.bidder_id == bidder_num)  # 修正匹配条件
        
        candidate_bundles = []
        
        # 遍历每个允许的充电站生成候选组合
        for station_id, slots in bidder.allowed_slots.items():
            # 确保有时段可用
            if not slots:
                continue
            
            # 生成该充电站的连续时段组合
            max_possible_length = min(len(slots), bidder.required_length)
            for l in range(1, max_possible_length + 1):  # 包含单个时隙
                for i in range(len(slots) - l + 1):
                    selected = slots[i:i+l]
                    # 必须保持连续性
                    if all(selected[j]+1 == selected[j+1] for j in range(len(selected)-1)) or len(selected) == 1:
                        bundle = [f"{station_id}_{t}" for t in selected]
                        price = sum(prices[self.good_index[g]] for g in bundle)
                        value = bidder.calculate_value(bundle) - price
                        if value >= 0:
                            candidate_bundles.append((bundle, value))
        
        # 筛选最优组合
        candidate_bundles.sort(key=lambda x: -x[1])
        print(f'最优组合{candidate_bundles}')
        # 添加调试信息输出
        top_bundles = candidate_bundles[:max_bundle_size]
        logging.info(f"Bidder_{bidder_num} Top {len(top_bundles)} bundles:")
        for i, (bundle, val) in enumerate(top_bundles, 1):
            logging.info(f"Bundle {i}: {bundle} | Raw Value: {val + sum(prices[self.good_index[g]] for g in bundle):.2f} | Net Value: {val:.2f}")
        
        # 在返回前添加二进制向量转换
        all_items = [f'station_{i}_{j}' for i in range(6) for j in range(24)]
        binary_bundles = []
        if top_bundles:
            for bundle in [bundle for bundle, _ in top_bundles]:
                vector = np.zeros(len(all_items), dtype=int)
                for item in bundle:
                    if item in all_items:
                        vector[all_items.index(item)] = 1
                binary_bundles.append(vector.tolist())
        else:
            # 如果没有候选组合，返回一个空的二进制向量
            binary_bundles = [np.zeros(len(all_items), dtype=int).tolist()]
    
        return binary_bundles

    def get_best_bundles_original(self, bidder_id, prices, max_bundle_size):
        """获取最优充电套餐组合（考虑多充电站）"""
        # 统一处理字符串和数字类型的bidder_id
        if isinstance(bidder_id, str):
            bidder_num = int(bidder_id.split('_')[-1])  # 处理"Bidder_X"格式
        else:
            bidder_num = int(bidder_id)
        
        # 修改匹配条件为数字ID比较
        bidder = next(b for b in self.bidders if b.bidder_id == bidder_num)  # 修正匹配条件
        
        candidate_bundles = []
        
        # 遍历每个允许的充电站生成候选组合
        for station_id, slots in bidder.allowed_slots.items():
            # 确保有时段可用
            if not slots:
                continue
            
            # 生成该充电站的连续时段组合
            max_possible_length = min(len(slots), bidder.required_length)
            for l in range(1, max_possible_length + 1):  # 包含单个时隙
                for i in range(len(slots) - l + 1):
                    selected = slots[i:i+l]
                    # 必须保持连续性
                    if all(selected[j]+1 == selected[j+1] for j in range(len(selected)-1)) or len(selected) == 1:
                        bundle = [f"{station_id}_{t}" for t in selected]
                        price = sum(prices[self.good_index[g]] for g in bundle)
                        value = bidder.calculate_value(bundle) - price
                        if value >= 0:
                            candidate_bundles.append((bundle, value))
        
        # 筛选最优组合
        candidate_bundles.sort(key=lambda x: -x[1])
        # 添加调试信息输出
        top_bundles = candidate_bundles[:max_bundle_size]
        logging.info(f"Bidder_{bidder_num} Top {len(top_bundles)} bundles:")
        for i, (bundle, val) in enumerate(top_bundles, 1):
            logging.info(f"Bundle {i}: {bundle} | Raw Value: {val + sum(prices[self.good_index[g]] for g in bundle):.2f} | Net Value: {val:.2f}")
    
        return [bundle[0] for bundle in top_bundles]
    
    def get_bidder_ids(self):
        """返回所有竞标者ID列表"""
        return [f"{i}" for i in range(len(self.bidders))]

    def get_good_ids(self):
        return self.all_goods.copy()

    def get_good_indexs(self):
        return list(self.good_index.values())

    def calculate_value(self, bidder_id, bundle):
        """统一使用Bidder的计算逻辑，需传入bidder_id"""
        try:
            # 转换bidder_id格式
            if isinstance(bidder_id, (np.ndarray, list)):
                bidder_num = int(bidder_id[0])
            else:
                bidder_num = int(str(bidder_id).replace('Bidder_', ''))
            # 获取对应投标者
            bidder = next(b for b in self.bidders if b.bidder_id == bidder_num)

            # 新增：转换numpy数组为商品ID列表
            if isinstance(bundle, np.ndarray):
                all_goods = self.all_goods
                bundle = [all_goods[i] for i in np.where(bundle > 0.5)[0]]

            raw_value = bidder.calculate_value(bundle)
            logging.debug(f"Raw value components: {raw_value}")
            return raw_value
        except Exception as e:
            logging.error(f"Value calculation error: {str(e)}")
            return 0.0

    def _get_good_capacities(self):
        """生成商品容量列表"""
        return [
            next(s['num_piles'] for s in self.stations 
                 if s['id'] == '_'.join(good.split('_')[:2]))
            for good in self.all_goods
        ]

    def get_efficient_allocation(self):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise RuntimeError("Gurobi not installed. Required for efficient allocation calculation.")
    
        # 创建模型并设置参数
        model = gp.Model("EfficientAllocation")
        model.setParam('MIPGap', 0.01)
    
        # 为每个投标者的候选bundle创建变量
        x = {}
        candidate_bundles = {}
        for bidder in self.bidders:
            candidate_bundles[bidder.bidder_id] = []
            
            for station_id, slots in bidder.allowed_slots.items():
                if not slots:
                    continue
            
                max_possible_length = min(len(slots), bidder.required_length)
                for l in range(1, max_possible_length + 1):
                    for i in range(len(slots) - l + 1):
                        selected = slots[i:i+l]
                        # 必须保持连续性
                        if all(selected[j]+1 == selected[j+1] for j in range(len(selected)-1)) or len(selected) == 1:
                            bundle = [f"{station_id}_{t}" for t in selected]
                            candidate_bundles[bidder.bidder_id].append(bundle)
            print(f'candidate_bundles[{bidder.bidder_id}]:{candidate_bundles[bidder.bidder_id]}')

            for idx in range(len(candidate_bundles[bidder.bidder_id])):
                x_key = (bidder.bidder_id, idx)
                x[x_key] = model.addVar(vtype=GRB.BINARY, name=f"x_{x_key}")
        
        # 构建目标函数（使用bundle的完整价值）
        obj_terms = []
        for (bidder_id, idx), var in x.items():
            bundle = candidate_bundles[bidder_id][idx]
            value = self.bidders[bidder_id].calculate_value(bundle)
            obj_terms.append(value * var)

        logging.info(f"目标函数包含 {len(obj_terms)} 个有效项")
        if len(obj_terms) == 0:
            logging.error("目标函数为空，可能原因："
                        "\n1. 所有投标者的候选bundle为空"
                        "\n2. 所有bundle的价值计算为零"
                        "\n3. 价格参数导致净价值为负")
            raise RuntimeError("No valid objective terms found. Check bidder allowed_slots and good definitions")
        
        model.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)
    
        # 添加约束：每个投标者最多选择一个bundle
        for bidder in self.bidders:
            model.addConstr(
                gp.quicksum(x.get((bidder.bidder_id, idx), 0) for idx in range(len(candidate_bundles[bidder.bidder_id]))) <= 1,
                name=f"single_bundle_{bidder.bidder_id}"
            )
    
        # 添加容量约束（基于bundle包含的goods）
        capacity_constrs = defaultdict(list)
        for (bidder_id, idx), var in x.items():
            bundle = candidate_bundles[bidder_id][idx]
            for good in bundle:
                if good in self.good_index:
                    capacity_constrs[good].append(var)
        
        for good, vars_list in capacity_constrs.items():
            model.addConstr(
                gp.quicksum(vars_list) <= self.capacities[good],
                name=f"capacity_{good}"
            )
    
        # 添加连续性约束的回调函数
        def continuity_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                for bidder in self.bidders:
                    selected = []
                    for (bid_id, bundle_idx), var in x.items():
                        if bid_id == bidder.bidder_id and model.cbGetSolution(var) > 0.5:
                            # 获取实际分配的bundle
                            bundle = candidate_bundles[bid_id][bundle_idx]
                            for good in bundle:  # 遍历bundle中的每个商品
                                parts = good.split('_')
                                if len(parts) < 3:
                                    continue
                                station_id = f"{parts[0]}_{parts[1]}"
                                t = int(parts[2])
                                selected.append((station_id, t))
                    
                    # 检查每个充电站的时段连续性
                    grouped = defaultdict(list)
                    for s_id, t in selected:
                        grouped[s_id].append(t)
                    
                    for s_id, slots in grouped.items():
                        sorted_slots = sorted(slots)
                        if len(sorted_slots) > 1 and any(sorted_slots[i+1] != sorted_slots[i]+1 for i in range(len(sorted_slots)-1)):
                            # 使用实际分配的商品创建约束
                            conflict_goods = [f"{s_id}_{t}" for t in sorted_slots]
                            expr = gp.quicksum(x[(bidder.bidder_id, idx)] 
                                         for idx, b in enumerate(candidate_bundles[bidder.bidder_id])
                                         if any(g in conflict_goods for g in b))
                            model.cbLazy(expr <= 1)
    
        if len(obj_terms) == 0:
            raise RuntimeError("No valid objective terms found. Check bidder allowed_slots and good definitions")
        
        logging.info(f"模型统计：变量数={len(x)}, 约束数={model.NumConstrs}")
        # 求解模型
        model.optimize(continuity_callback)
    
        logging.info(f"MIP model contains {len(x)} variables and {model.NumConstrs} constraints")
    
        # 解析结果
        allocation = defaultdict(list)
        total_value = 0.0
        
        if model.status == GRB.OPTIMAL:
            logging.info("===== 最优解详情 =====")
            total_value = 0.0
            allocated_bundles = []
            
            # 新增分配详情日志
            for (bidder_id, idx), var in x.items():
                if var.X > 0.5:
                    bundle = candidate_bundles[bidder_id][idx]
                    value = self.bidders[bidder_id].calculate_value(bundle)
                    
                    # 记录每个分配的详细信息
                    logging.info(f"Bidder {bidder_id} 分配 bundle: {bundle}")
                    logging.info(f"  包含商品数: {len(bundle)}")
                    logging.info(f"  计算价值: {value:.2f}")
                    logging.info(f"  投标者参数: base_value={self.bidders[bidder_id].base_value}, required_length={self.bidders[bidder_id].required_length}")
                    
                    allocation[str(bidder_id)].extend(bundle)
                    total_value += value
                    allocated_bundles.append((bidder_id, bundle))

            # 新增零值诊断日志
            if total_value <= 1e-6:
                error_info = [
                    "零值错误诊断信息:",
                    f"候选bundle总数: {sum(len(b) for b in candidate_bundles.values())}",
                    "各投标者候选bundle情况:"
                ]
                
                for bidder_id, bundles in candidate_bundles.items():
                    error_info.append(f"Bidder {bidder_id}:")
                    error_info.append(f"  allowed_slots: {self.bidders[bidder_id].allowed_slots}")
                    error_info.append(f"  候选bundle数: {len(bundles)}")
                    if bundles:
                        best_value = max(self.bidders[bidder_id].calculate_value(b) for b in bundles)
                        error_info.append(f"  最高价值bundle: {best_value:.2f}")
                
                error_info.append("实际分配的bundle列表:")
                for bidder_id, bundle in allocated_bundles:
                    error_info.append(f"  Bidder {bidder_id}: {bundle}")
                
                logging.error("\n".join(error_info))
                raise RuntimeError("Optimal solution has zero value. Check bidder parameters")
        else:
            logging.error(f"模型求解失败，状态码: {model.status}")
            raise RuntimeError("No optimal solution found")
    
        return dict(allocation), total_value