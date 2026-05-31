import json
import random

import numpy as np


from src.custom_msvm_domain import CustomMSVMAuction


def generate_bidder_params(
    num_bidders=20,
    num_stations=5,
    num_slots=12,
    map_size=10
):
    # 生成价值文件 (seeds1-100.json)
    generate_value_file(num_bidders, num_stations, num_slots, map_size)
    
    # 生成两个时期的npy文件
    generate_average_values(1, 1000, num_bidders, num_stations, num_slots, map_size, 
                          'MSVM_average_item_values_seeds_1-1000.npy')
    generate_average_values(201, 1200, num_bidders, num_stations, num_slots, map_size,
                          'MSVM_average_item_values_seeds_201-1200.npy')

def calculate_distance(loc1, loc2):
    """计算两个GPS坐标间的距离"""
    # 简单欧式距离
    return abs(loc1[0]-loc2[0]) + abs(loc1[1]-loc2[1]) 

def generate_value_file(num_bidders, num_stations, num_slots, map_size):
    result = {
        f'Bidder_{bidder}_{metric}': {}
        for bidder in range(num_bidders)
        for metric in ['max_value', 'no_items', 'max_value_per_item']
    }
    
    for seed in range(1, 101):

        # 生成充电站配置
        stations = [{
            'id': f'station_{i}',
            'x': np.random.uniform(0, map_size),
            'y': np.random.uniform(0, map_size),
            'num_piles': np.random.randint(2,3)
        } for i in range(num_stations)]

        # 生成投标者参数
        bidders = []
        for i in range(num_bidders):
            bidder = {
                'id': f'Bidder_{i}',
                'coordinates': {
                    'x': np.random.uniform(0, map_size),
                    'y': np.random.uniform(0, map_size)
                },
                'distance_penalty': np.random.uniform(0.1, 0.5),
                'base_value': np.random.lognormal(mean=3, sigma=0.5),
                'required_length': np.random.randint(2,5),
                'continuity_bonus': np.random.uniform(0.2, 1.0),
                'allowed_slots': []
            }

            # 生成5个不重复的起始时间（0-12）
            start_points = random.sample(range(12), 5)

            # 生成连续时段
            allowed_t = []
            for start in start_points:
                length = random.choice([1, 2, 3])
                allowed_t.extend([(start + i) % 10 for i in range(length)])

            # 去重排序
            allowed_t = sorted(list(set(allowed_t)))

            print(f"\nProcessing Bidder {bidder['id']}:")
            print(f"Bidder Coordinates: {bidder['coordinates']}")

            for station in stations:
                distance = calculate_distance(
                    (station['x'], station['y']),
                    (bidder['coordinates']['x'], bidder['coordinates']['y'])
                )
                print(f"  Station {station['id']} distance: {distance:.2f}")
                if distance <= 18.0:
                    for t in allowed_t:
                        bidder['allowed_slots'].append(f"{station['id']}_{t}")
            print(f"Bidder {i} allowed_slots: {bidder['allowed_slots']}")
            bidders.append(bidder)

        # 创建自定义拍卖实例
        auction_instance = CustomMSVMAuction(
            stations=stations,
            bidders=bidders,
            num_slots=num_slots
        )

        print(f"Total bidders: {len(auction_instance.bidders)}")

        # 计算零价格下的最优捆绑
        for bidder in auction_instance.bidders:
            null_prices = [0.0] * len(auction_instance.all_goods)
            
            best_bundles = auction_instance.get_best_bundles_original(
                bidder.bidder_id, 
                null_prices,
                max_bundle_size=1
            )

            if not best_bundles:
                print(f"WARNING: No valid bundles for {bidder.bidder_id}")

            print(f"Bidder {bidder.bidder_id} best_bundles: {best_bundles}")
            
            if best_bundles:
                best_bundle = best_bundles[0]
                sum_items = len(best_bundle)
                max_value = bidder.calculate_value(best_bundle)
                result[f'Bidder_{bidder.bidder_id}_max_value'][f'Seed_{seed}'] = float(max_value)
                result[f'Bidder_{bidder.bidder_id}_no_items'][f'Seed_{seed}'] = float(sum_items)
                result[f'Bidder_{bidder.bidder_id}_max_value_per_item'][f'Seed_{seed}'] = float(max_value / sum_items) if sum_items > 0 else 0.0
    
    for key in result:
        values = list(result[key].values())
        result[key]['mean'] = sum(values) / len(values) if values else 0       
    
    # 保存结果
    with open('MSVM_values_for_null_price_seeds1-100.json', 'w') as f:
        json.dump(result, f, indent=2)

def generate_average_values(start_seed, end_seed, num_bidders, num_stations, num_slots, map_size, filename):
    from custom_msvm_domain import CustomMSVMAuction
    
    all_item_values = []

    for seed in range(start_seed, end_seed + 1):
        np.random.seed(seed)
        
        # 生成与generate_value_file相同的参数逻辑
        stations = [{
            'id': f'station_{i}',
            'x': np.random.uniform(0, map_size),
            'y': np.random.uniform(0, map_size),
            'num_piles': np.random.randint(2,3)
        } for i in range(num_stations)]

        bidders = []
        for i in range(num_bidders):
            bidder = {
                'id': f'Bidder_{i}',
                'coordinates': {
                    'x': np.random.uniform(0, map_size),
                    'y': np.random.uniform(0, map_size)
                },
                'distance_penalty': np.random.uniform(0.1, 0.5),
                'base_value': np.random.lognormal(mean=3, sigma=0.5),
                'required_length': np.random.randint(2,5),
                'continuity_bonus': np.random.uniform(0.2, 1.0),
                'allowed_slots': []
            }

            # 生成5个不重复的起始时间（0-12）
            start_points = random.sample(range(12), 5)

            # 生成连续时段
            allowed_t = []
            for start in start_points:
                length = random.choice([1, 2, 3])
                allowed_t.extend([(start + i) % 10 for i in range(length)])

            # 去重排序
            allowed_t = sorted(list(set(allowed_t)))

            print(f"\nProcessing Bidder {bidder['id']}:")
            print(f"Bidder Coordinates: {bidder['coordinates']}")

            for station in stations:
                distance = calculate_distance(
                    (station['x'], station['y']),
                    (bidder['coordinates']['x'], bidder['coordinates']['y'])
                )
                print(f"  Station {station['id']} distance: {distance:.2f}")
                if distance <= 18.0:
                    for t in allowed_t:
                        bidder['allowed_slots'].append(f"{station['id']}_{t}")
                print(f"Bidder {i} allowed_slots: {bidder['allowed_slots']}")
            bidders.append(bidder)

        # 创建拍卖实例
        auction = CustomMSVMAuction(stations, bidders, num_slots)
        m = len(auction.get_good_ids())
        
        # 计算每个item的平均价值
        item_means = np.zeros(m)
        for item_idx in range(m):
            bundle = [auction.get_good_ids()[item_idx]]
            values = []
            for bidder in auction.bidders:
                values.append(auction.calculate_value(bidder.bidder_id, bundle))
            item_means[item_idx] = np.mean(values)
        
        all_item_values.append(item_means)
    
    # 保存所有种子的平均值
    np.save(filename, np.array(all_item_values).mean(axis=0))

if __name__ == '__main__':
    generate_bidder_params()