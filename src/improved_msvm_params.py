#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的MSVM参数生成器
解决CCA在第1轮就达到100%分配效率的问题

主要改进：
1. 增加投标者数量和充电站数量，提高竞争激烈程度
2. 缩小地图范围，增加距离约束的影响
3. 增加价值函数的复杂性和非线性程度
4. 引入更多的资源冲突和约束
5. 增加投标者偏好的异质性
"""

import json
import random
import numpy as np
from src.custom_msvm_domain import CustomMSVMAuction


def generate_complex_bidder_params(
        num_bidders=50,  # 增加到50个投标者
        num_stations=6,  # 进一步减少到6个充电站，大幅增加稀缺性
        num_slots=24,  # 24个时段
        map_size=4  # 大幅缩小地图范围，增强距离约束影响
):
    """生成更复杂的投标者参数"""
    print(f"生成复杂MSVM参数: {num_bidders}投标者, {num_stations}充电站, {num_slots}时段, 地图大小{map_size}")

    # 生成价值文件
    generate_complex_value_file(num_bidders, num_stations, num_slots, map_size)

    # 生成平均价值文件
    generate_average_values(1, 1000, num_bidders, num_stations, num_slots, map_size,
                            'MSVM_average_item_values_seeds_1-1000.npy')
    generate_average_values(201, 1200, num_bidders, num_stations, num_slots, map_size,
                            'MSVM_average_item_values_seeds_201-1200.npy')


def calculate_distance(loc1, loc2):
    """计算欧几里得距离（更真实的距离计算）"""
    return np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


def generate_complex_value_file(num_bidders, num_stations, num_slots, map_size):
    """生成复杂的价值文件"""
    result = {
        f'Bidder_{bidder}_{metric}': {}
        for bidder in range(num_bidders)
        for metric in ['max_value', 'no_items', 'max_value_per_item']
    }

    for seed in range(1, 101):
        np.random.seed(seed)
        random.seed(seed)

        # 生成充电站配置（集群分布，增加竞争）
        stations = generate_clustered_stations(num_stations, map_size)

        # 生成投标者参数（更复杂的偏好）
        bidders = generate_complex_bidders(num_bidders, stations, num_slots, map_size)

        # 创建拍卖实例
        auction_instance = CustomMSVMAuction(
            stations=stations,
            bidders=bidders,
            num_slots=num_slots
        )

        print(f"Seed {seed}: 总投标者数 {len(auction_instance.bidders)}")

        # 计算零价格下的最优捆绑
        for bidder in auction_instance.bidders:
            null_prices = [0.0] * len(auction_instance.all_goods)

            # 允许更大的捆绑大小，增加竞争复杂度
            max_bundle = min(bidder.required_length, max(1, len(auction_instance.all_goods) // 5))
            best_bundles = auction_instance.get_best_bundles_original(
                bidder.bidder_id,
                null_prices,
                max_bundle_size=max_bundle
            )

            if not best_bundles:
                print(f"WARNING: 投标者 {bidder.bidder_id} 没有有效捆绑，尝试紧急修复...")

                # 紧急修复：如果投标者有allowed_slots但没有有效捆绑，强制创建一个最小捆绑
                if hasattr(bidder, 'allowed_slots') and bidder.allowed_slots:
                    # 选择第一个可用时段作为紧急捆绑
                    emergency_bundle = [bidder.allowed_slots[0]]
                    emergency_value = bidder.calculate_value(emergency_bundle)

                    result[f'Bidder_{bidder.bidder_id}_max_value'][f'Seed_{seed}'] = float(emergency_value)
                    result[f'Bidder_{bidder.bidder_id}_no_items'][f'Seed_{seed}'] = 1.0
                    result[f'Bidder_{bidder.bidder_id}_max_value_per_item'][f'Seed_{seed}'] = float(emergency_value)

                    print(
                        f"紧急修复成功: 投标者 {bidder.bidder_id} 使用紧急捆绑 {emergency_bundle}，价值 {emergency_value:.2f}")
                    continue
                else:
                    print(f"严重错误: 投标者 {bidder.bidder_id} 连allowed_slots都为空！")

                # 如果紧急修复也失败，设置默认值
                result[f'Bidder_{bidder.bidder_id}_max_value'][f'Seed_{seed}'] = 0.0
                result[f'Bidder_{bidder.bidder_id}_no_items'][f'Seed_{seed}'] = 0.0
                result[f'Bidder_{bidder.bidder_id}_max_value_per_item'][f'Seed_{seed}'] = 0.0
                continue

            best_bundle = best_bundles[0]
            sum_items = len(best_bundle)
            max_value = bidder.calculate_value(best_bundle)

            result[f'Bidder_{bidder.bidder_id}_max_value'][f'Seed_{seed}'] = float(max_value)
            result[f'Bidder_{bidder.bidder_id}_no_items'][f'Seed_{seed}'] = float(sum_items)
            result[f'Bidder_{bidder.bidder_id}_max_value_per_item'][f'Seed_{seed}'] = float(
                max_value / sum_items) if sum_items > 0 else 0.0

    # 计算平均值
    for key in result:
        values = [v for v in result[key].values() if isinstance(v, (int, float))]
        result[key]['mean'] = sum(values) / len(values) if values else 0

    # 保存结果
    with open('MSVM_values_for_null_price_seeds1-100.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("复杂价值文件已生成")


def generate_clustered_stations(num_stations, map_size):
    """生成集群分布的充电站（增加空间竞争）"""
    stations = []

    # 创建极少的充电站集群，大幅增加竞争激烈程度
    num_clusters = max(1, num_stations // 4)  # 进一步减少集群数量
    cluster_centers = [(np.random.uniform(0.5, map_size - 0.5), np.random.uniform(0.5, map_size - 0.5))
                       for _ in range(num_clusters)]

    stations_per_cluster = num_stations // num_clusters
    remaining_stations = num_stations % num_clusters

    station_id = 0
    for i, (cx, cy) in enumerate(cluster_centers):
        # 每个集群的充电站数量
        cluster_size = stations_per_cluster + (1 if i < remaining_stations else 0)

        for j in range(cluster_size):
            # 在集群中心周围生成充电站，集群更紧密
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.1, 0.6)  # 缩小集群半径，增加密度

            x = max(0, min(map_size, cx + radius * np.cos(angle)))
            y = max(0, min(map_size, cy + radius * np.sin(angle)))

            stations.append({
                'id': f'station_{station_id}',
                'x': x,
                'y': y,
                'num_piles': 1,  # 固定为1个充电桩，最大化稀缺性
                'cluster_id': i  # 记录所属集群
            })
            station_id += 1

    return stations


def generate_complex_bidders(num_bidders, stations, num_slots, map_size):
    """生成具有复杂偏好的投标者"""
    bidders = []

    # 定义不同类型的投标者，增大价值差异和需求长度差异
    bidder_types = [
        {'type': 'premium_urgent', 'base_mean': 5.5, 'base_sigma': 0.2, 'required_length': (1, 2),
         'distance_sensitivity': 0.9, 'weight': 0.15},
        {'type': 'high_value_short', 'base_mean': 4.8, 'base_sigma': 0.3, 'required_length': (1, 4),
         'distance_sensitivity': 0.8, 'weight': 0.25},
        {'type': 'medium_value_medium', 'base_mean': 3.5, 'base_sigma': 0.4, 'required_length': (3, 8),
         'distance_sensitivity': 0.5, 'weight': 0.35},
        {'type': 'low_value_long', 'base_mean': 2.5, 'base_sigma': 0.6, 'required_length': (5, 12),
         'distance_sensitivity': 0.3, 'weight': 0.25}
    ]

    for i in range(num_bidders):
        # 按权重选择投标者类型
        weights = [bt['weight'] for bt in bidder_types]
        bidder_type_idx = np.random.choice(len(bidder_types), p=weights)
        bidder_type = bidder_types[bidder_type_idx]

        # 生成投标者位置，高价值用户更倾向于聚集，增加竞争
        clustering_prob = 0.9 if bidder_type['type'] in ['premium_urgent', 'high_value_short'] else 0.7
        if np.random.random() < clustering_prob:
            nearby_station = random.choice(stations)
            # 高价值用户聚集得更紧密
            spread = 0.3 if bidder_type['type'] == 'premium_urgent' else 0.6
            x = nearby_station['x'] + np.random.normal(0, spread)
            y = nearby_station['y'] + np.random.normal(0, spread)
        else:
            x = np.random.uniform(0, map_size)
            y = np.random.uniform(0, map_size)

        # 确保坐标在地图范围内
        x = max(0, min(map_size, x))
        y = max(0, min(map_size, y))

        bidder = {
            'id': f'Bidder_{i}',
            'coordinates': {'x': x, 'y': y},
            'distance_penalty': np.random.uniform(0.2, bidder_type['distance_sensitivity']),
            'base_value': np.random.lognormal(mean=bidder_type['base_mean'], sigma=bidder_type['base_sigma']),
            'required_length': np.random.randint(*bidder_type['required_length']),
            'continuity_bonus': np.random.uniform(1.0, 3.0),  # 大幅增加连续性奖励，鼓励长时段竞争
            'allowed_slots': [],
            'bidder_type': bidder_type['type']
        }

        # 生成更复杂的时段偏好
        allowed_slots = generate_complex_time_preferences(i, stations, num_slots, map_size, bidder)
        bidder['allowed_slots'] = allowed_slots

        # 验证投标者至少有一个可用时段，这是关键的保护机制
        if not allowed_slots:
            print(f"严重错误: 投标者 {i} 没有可用时段，这不应该发生！")
            # 紧急保护：强制分配一个时段
            emergency_station = stations[0] if stations else {'id': 'emergency_station'}
            emergency_time = np.random.randint(0, num_slots)
            allowed_slots = [f"{emergency_station['id']}_{emergency_time}"]
            bidder['allowed_slots'] = allowed_slots
            print(f"紧急修复: 为投标者 {i} 分配紧急时段 {allowed_slots[0]}")

        print(f"投标者 {i} ({bidder_type['type']}): 坐标({x:.1f},{y:.1f}), "
              f"基础价值{bidder['base_value']:.1f}, 需求长度{bidder['required_length']}, "
              f"允许时段数{len(allowed_slots)}")

        bidders.append(bidder)

    return bidders


def generate_complex_time_preferences(bidder_id, stations, num_slots, map_size, bidder):
    """生成复杂的时段偏好（考虑距离、时间偏好等）"""
    allowed_slots = []

    # 定义时间偏好模式，修正为24时段
    time_patterns = {
        'morning': list(range(0, 8)),  # 早晨 0-7
        'afternoon': list(range(8, 16)),  # 下午 8-15
        'evening': list(range(16, 24)),  # 晚上 16-23
        'flexible': list(range(0, 24))  # 全天
    }

    # 根据投标者ID选择时间偏好
    if bidder_id % 4 == 0:
        preferred_times = time_patterns['morning']
    elif bidder_id % 4 == 1:
        preferred_times = time_patterns['afternoon']
    elif bidder_id % 4 == 2:
        preferred_times = time_patterns['evening']
    else:
        preferred_times = time_patterns['flexible']

    # 添加更多随机性，进一步限制时段选择
    if len(preferred_times) > 3:
        preferred_times = random.sample(preferred_times, random.randint(1, min(3, len(preferred_times))))

    # 计算所有充电站的距离，找到可达的充电站
    accessible_stations = []
    station_distances = []

    for station in stations:
        distance = calculate_distance(
            (station['x'], station['y']),
            (bidder['coordinates']['x'], bidder['coordinates']['y'])
        )
        station_distances.append((station, distance))

    # 按距离排序
    station_distances.sort(key=lambda x: x[1])

    # 距离阈值根据地图大小调整，大幅缩小可达范围
    max_distance = map_size * 0.4  # 缩小到40%的地图对角线长度，严格限制可达性

    # 为每个充电站检查可达性和时段偏好
    for station, distance in station_distances:
        if distance <= max_distance:
            # 根据距离调整可用时段数量，大幅减少可用时段
            distance_factor = 1.0 - (distance / max_distance)
            available_time_ratio = 0.05 + 0.25 * distance_factor  # 5%-30%的时段可用，大幅减少选择

            num_available_times = max(1, int(len(preferred_times) * available_time_ratio))
            selected_times = random.sample(preferred_times, num_available_times)

            for t in selected_times:
                allowed_slots.append(f"{station['id']}_{t}")

    # 强制保证每个投标者至少有一个可用时段，避免空bundle报错
    if not allowed_slots:
        if station_distances:
            # 情况1：有充电站但都超出距离阈值，选择最近的充电站
            closest_station = station_distances[0][0]
            # 为最近的充电站添加至少一个时段
            if preferred_times:
                selected_time = random.choice(preferred_times)
                allowed_slots.append(f"{closest_station['id']}_{selected_time}")
                print(
                    f"警告: 投标者 {bidder_id} 距离所有充电站都较远，强制分配最近充电站 {closest_station['id']} 的时段 {selected_time}")
            else:
                # 如果连preferred_times都为空，使用默认时段
                default_time = np.random.randint(0, num_slots)
                allowed_slots.append(f"{closest_station['id']}_{default_time}")
                print(
                    f"警告: 投标者 {bidder_id} 偏好时段为空，强制分配最近充电站 {closest_station['id']} 的默认时段 {default_time}")
        else:
            # 情况2：没有充电站（理论上不应该发生，但作为保护机制）
            print(f"错误: 投标者 {bidder_id} 没有可用的充电站，这不应该发生")
            # 创建一个虚拟时段作为最后的保护
            allowed_slots.append(f"station_0_0")

    # 额外保护：确保allowed_slots不为空
    if not allowed_slots:
        print(f"严重警告: 投标者 {bidder_id} 的allowed_slots仍为空，添加紧急保护时段")
        allowed_slots.append(f"station_0_0")  # 紧急保护时段

    return allowed_slots


def generate_average_values(start_seed, end_seed, num_bidders, num_stations, num_slots, map_size, filename):
    """生成平均价值文件"""
    all_item_values = []

    for seed in range(start_seed, end_seed + 1):
        np.random.seed(seed)
        random.seed(seed)

        # 使用相同的复杂参数生成逻辑
        stations = generate_clustered_stations(num_stations, map_size)
        bidders = generate_complex_bidders(num_bidders, stations, num_slots, map_size)

        # 创建拍卖实例
        auction = CustomMSVMAuction(stations, bidders, num_slots)
        m = len(auction.get_good_ids())

        # 计算每个item的平均价值
        item_means = np.zeros(m)
        for item_idx in range(m):
            bundle = [auction.get_good_ids()[item_idx]]
            values = []
            for bidder in auction.bidders:
                try:
                    value = auction.calculate_value(bidder.bidder_id, bundle)
                    values.append(value)
                except:
                    values.append(0.0)
            item_means[item_idx] = np.mean(values) if values else 0.0

        all_item_values.append(item_means)

    # 保存平均值
    final_averages = np.array(all_item_values).mean(axis=0)
    np.save(filename, final_averages)
    print(f"平均价值文件已保存: {filename}")


def analyze_complexity():
    """分析参数复杂度"""
    print("\n=== 复杂度分析 ===")
    print("改进前的问题：")
    print("- 投标者数量少 (20个)")
    print("- 充电站数量少 (5个)")
    print("- 地图范围大 (10x10)，距离约束影响小")
    print("- 时段偏好简单，重叠度高")
    print("- 价值函数相对简单")
    print("")
    print("改进后的复杂度：")
    print("- 投标者数量增加到50个，竞争更激烈")
    print("- 充电站数量增加到8个，但采用集群分布")
    print("- 地图范围缩小到6x6，距离约束影响增大")
    print("- 时段数量增加到18个，提供更多选择")
    print("- 投标者类型多样化，偏好异质性增强")
    print("- 时间偏好模式化（早晨/下午/晚上/灵活）")
    print("- 距离敏感性差异化")
    print("- 充电站集群分布，增加空间竞争")
    print("")
    print("预期效果：")
    print("- CCA需要多轮才能收敛到最优解")
    print("- 初始轮次的分配效率应该在60-80%范围")
    print("- 价格发现过程更加复杂")
    print("- 投标者之间的竞争更加激烈")


def compare_with_original():
    """与原始参数对比"""
    print("\n=== 参数对比 ===")
    print("| 参数 | 原始设置 | 改进设置 | 改进原因 |")
    print("|------|----------|----------|----------|")
    print("| 投标者数量 | 20 | 50 | 增加竞争激烈程度 |")
    print("| 充电站数量 | 5 | 8 | 提供更多选择，但集群分布增加竞争 |")
    print("| 时段数量 | 12 | 18 | 增加时间维度的复杂性 |")
    print("| 地图大小 | 10x10 | 6x6 | 增强距离约束的影响 |")
    print("| 距离计算 | 曼哈顿距离 | 欧几里得距离 | 更真实的距离模型 |")
    print("| 充电站分布 | 随机分布 | 集群分布 | 增加空间竞争 |")
    print("| 投标者类型 | 单一类型 | 4种类型 | 增加偏好异质性 |")
    print("| 时间偏好 | 随机生成 | 模式化偏好 | 更真实的用户行为 |")
    print("| 距离敏感性 | 固定范围 | 类型相关 | 差异化的距离偏好 |")


if __name__ == '__main__':
    print("生成改进的MSVM参数...")
    generate_complex_bidder_params()
    analyze_complexity()
    compare_with_original()
    print("\n改进参数生成完成！")
    print("\n使用方法：")
    print("1. 运行此脚本生成复杂参数")
    print("2. 将生成的文件替换原始参数文件")
    print("3. 重新运行CCA测试")
    print("4. 观察分配效率是否需要多轮才能达到100%")