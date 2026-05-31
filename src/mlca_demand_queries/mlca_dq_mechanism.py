# Libs
import json
import logging
from datetime import datetime
import random
import torch
import numpy as np
from numpyencoder import NumpyEncoder
from pysats import PySats

from custom_msvm_domain import CustomMSVMAuction

# make sure you set the classpath before loading any other modules
PySats.getInstance()
import os
# Own Libs
from mlca_demand_queries.mlca_dq_economies import MLCA_DQ_ECONOMIES
from pysats_ext import GenericWrapper
# from pdb import set_trace

def save_msvm_auction_instance_json(json_path, stations, bidders, num_slots):
    payload = {
        "stations": stations,
        "bidders": bidders,
        "num_slots": num_slots,
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    json.dump(payload,
              open(json_path, 'w'),
              indent=4,
              sort_keys=False,
              separators=(', ', ': '),
              ensure_ascii=False,
              cls=NumpyEncoder)

def calculate_distance(loc1, loc2):
    """计算欧几里得距离（更真实的距离计算）"""
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def generate_clustered_stations(num_stations, map_size):
    """生成集群分布的充电站（增加空间竞争）"""
    stations = []
    
    # 创建极少的充电站集群，大幅增加竞争激烈程度
    num_clusters = max(1, num_stations // 4)  # 进一步减少集群数量
    cluster_centers = [(np.random.uniform(0.5, map_size-0.5), np.random.uniform(0.5, map_size-0.5)) 
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
        {'type': 'premium_urgent', 'base_mean': 5.5, 'base_sigma': 0.2, 'required_length': (1, 2), 'distance_sensitivity': 0.9, 'weight': 0.15},
        {'type': 'high_value_short', 'base_mean': 4.8, 'base_sigma': 0.3, 'required_length': (1, 4), 'distance_sensitivity': 0.8, 'weight': 0.25},
        {'type': 'medium_value_medium', 'base_mean': 3.5, 'base_sigma': 0.4, 'required_length': (3, 8), 'distance_sensitivity': 0.5, 'weight': 0.35},
        {'type': 'low_value_long', 'base_mean': 2.5, 'base_sigma': 0.6, 'required_length': (5, 12), 'distance_sensitivity': 0.3, 'weight': 0.25}
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
        
        bidders.append(bidder)
    
    return bidders

def generate_complex_time_preferences(bidder_id, stations, num_slots, map_size, bidder):
    """生成复杂的时段偏好（考虑距离、时间偏好等）"""
    allowed_slots = []
    
    # 定义时间偏好模式，修正为24时段
    time_patterns = {
        'morning': list(range(0, 8)),      # 早晨 0-7
        'afternoon': list(range(8, 16)),   # 下午 8-15
        'evening': list(range(16, 24)),    # 晚上 16-23
        'flexible': list(range(0, 24))     # 全天
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
                print(f"警告: 投标者 {bidder_id} 距离所有充电站都较远，强制分配最近充电站 {closest_station['id']} 的时段 {selected_time}")
            else:
                # 如果连preferred_times都为空，使用默认时段
                default_time = np.random.randint(0, num_slots)
                allowed_slots.append(f"{closest_station['id']}_{default_time}")
                print(f"警告: 投标者 {bidder_id} 偏好时段为空，强制分配最近充电站 {closest_station['id']} 的默认时段 {default_time}")
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

# %% MECHANISM
def mechanism(SATS_parameters: dict,
              TRAIN_parameters: dict,
              MVNN_parameters: dict,
              mechanism_parameters: dict,
              MIP_parameters: dict,
              res_path: str, 
              wandb_tracking: bool,
              wandb_project_name: str
              ):

    SATS_seed = SATS_parameters['SATS_seed']
    SATS_domain = SATS_parameters['SATS_domain']
    Qinit =mechanism_parameters['Qinit']
    Qmax = mechanism_parameters['Qmax']
    new_query_option = mechanism_parameters['new_query_option']
    isLegacy = SATS_parameters['isLegacy']
    calc_efficiency_per_iteration = mechanism_parameters['calculate_efficiency_per_iteration']
    initial_demand_query_method = mechanism_parameters['initial_demand_query_method']
    

    # Save config dict
    config_dict = {'SATS_parameters':SATS_parameters,
                    'TRAIN_parameters':TRAIN_parameters,
                    'MVNN_parameters':MVNN_parameters,
                    'mechanism_parameters':mechanism_parameters,
                    'MIP_parameters':MIP_parameters
                    }
    
    json.dump(config_dict,
            open(os.path.join(res_path,'config.json'), 'w'),
            indent=4,
            sort_keys=False,
            separators=(', ', ': '),
            ensure_ascii=False,
            cls=NumpyEncoder)

    start_time = datetime.now()

    # SEEDING ------------------
    np.random.seed(SATS_seed)
    torch.manual_seed(SATS_seed)
    random.seed(SATS_seed)
    # ---------------------------

    logging.warning('START MLCA:')
    logging.warning('-----------------------------------------------')
    logging.warning(f'Model: {SATS_domain}')
    logging.warning(f'Seed SATS Instance: {SATS_seed}')
    logging.warning(f'Qinit:{Qinit}')
    logging.warning(f'Qmax: {Qmax}')
    logging.warning(f'new_query_option: {new_query_option}')
    logging.warning(f'initial_demand_query_method: {initial_demand_query_method}')
    logging.warning('')

    # Instantiate Economies
    logging.warning('Instantiate SATS Instance')
    if SATS_domain == 'LSVM':
        SATS_auction_instance = PySats.getInstance().create_lsvm(seed=SATS_seed,
                                                                 isLegacyLSVM=isLegacy)  # create SATS auction instance
        logging.warning('####### ATTENTION #######')
        logging.warning('isLegacyLSVM: %s', SATS_auction_instance.isLegacy)
        logging.warning('#########################\n')
        GSVM_national_bidder_goods_of_interest = None

    elif SATS_domain == 'GSVM':
        SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_seed,
                                                                 isLegacyGSVM=isLegacy)  # create SATS auction instance
        logging.warning('####### ATTENTION #######')
        logging.warning('isLegacyGSVM: %s', SATS_auction_instance.isLegacy)
        logging.warning('#########################\n')
        GSVM_national_bidder_goods_of_interest = SATS_auction_instance.get_goods_of_interest(6) # national bidder is bidder 6

    elif SATS_domain == 'MRVM':
        mrvm_non_generic = PySats.getInstance().create_mrvm(seed=SATS_seed)  # create SATS auction instance
        SATS_auction_instance = GenericWrapper(mrvm_non_generic) # wrap non-generic auction instance
        GSVM_national_bidder_goods_of_interest = None


    elif SATS_domain == 'SRVM':
        srvm_non_generic = PySats.getInstance().create_srvm(seed=SATS_seed)  # create SATS auction instance
        SATS_auction_instance = GenericWrapper(srvm_non_generic) # wrap non-generic auction instance
        GSVM_national_bidder_goods_of_interest = None
    
    elif SATS_domain == 'MSVM':
        # 生成复杂的MSVM拍卖参数
        num_stations = 6   # 进一步减少到6个充电站，大幅增加稀缺性
        num_bidders = 50   # 增加到50个投标者
        num_slots = 24     # 24个时段
        map_size = 4       # 大幅缩小地图范围，增强距离约束影响

        # 生成集群分布的充电站配置
        stations = generate_clustered_stations(num_stations, map_size)
        
        # 生成具有复杂偏好的投标者参数
        bidders = generate_complex_bidders(num_bidders, stations, num_slots, map_size)

        # 直接实例化CustomMSVMAuction
        SATS_auction_instance = CustomMSVMAuction(
            stations=stations,
            bidders=bidders,
            num_slots=num_slots
        )
        GSVM_national_bidder_goods_of_interest = None

        # 记录拍卖实例配置
        auction_config = {
            "stations": stations,
            "bidders": bidders,
        }
        logging.info("SATS Auction Configuration:\n%s",
                     json.dumps(auction_config, indent=4, default=str))

        # 保存当前拍卖实例到结果路径，文件名包含种子，便于复现
        json_save_path = os.path.join(res_path, f"auction_instance_seed_{SATS_seed}.json")
        save_msvm_auction_instance_json(json_save_path, stations, bidders, num_slots)
        logging.info("Saved MSVM auction instance JSON: %s", json_save_path)

    else:
        raise ValueError(f'SATS_domain {SATS_domain} not yet implemented')
    
    SATS_parameters['GSVM_national_bidder_goods_of_interest'] = GSVM_national_bidder_goods_of_interest

    # create economy instance
    E = MLCA_DQ_ECONOMIES(SATS_auction_instance = SATS_auction_instance,
                          SATS_parameters = SATS_parameters,
                          TRAIN_parameters = TRAIN_parameters,
                          MVNN_parameters= MVNN_parameters,
                          mechanism_parameters = mechanism_parameters,
                          start_time=start_time,
                          wandb_tracking = wandb_tracking,
                          wandb_project_name = wandb_project_name
                          )
    # set NN parameters
    E.set_ML_parameters(parameters=MVNN_parameters) 
    # set MIP parameters
    E.set_MIP_parameters(parameters=MIP_parameters)  

    # SAMPLE INITIAL DEMAND QUERIES
    E.set_initial_dqs(method = initial_demand_query_method)

    # Calculate efficient allocation given current elicited bids
    if calc_efficiency_per_iteration: 
        E.calculate_efficiency_per_iteration()
        # Also record initial CCA per-round efficiency and SCW for plotting
        E.log_Qinit_efficiency()


    # Global while loop: check if for all bidders one addtitional auction round is feasible
    for iteration in range(1, (E.Qmax-E.Qinit) + 1):
        print(f'当前迭代轮次: {iteration} / {(E.Qmax - E.Qinit)}')

        # Increment iteration
        E.mlca_iteration += 1

        # logging info
        E.get_info()

        # Reset attributes
        logging.info('RESET: ML Models')
        E.reset_ML_models()

        # Train ML Models
        E.estimation_step()

        # DQ generation (only in Main Economy), also updates the elicited DQ object
        E.generate_dq()
        
        # check if the DQ found clears the market.
        if E.found_clearing_prices: 
            logging.warning('EARLY STOPPING - CLEARED THE MARKET')
            logging.info('')
            logging.info('CALCULATE FINAL CLEARING ALLOCATION')
            logging.info('---------------------------------------------')

            # we have clearing prices -> we do not need to calculate the final allocation based on inferred bids. 
            E.calculate_clearing_allocation(E.demand_vector_per_iteration[iteration], E.price_vector_per_iteration[iteration], is_final_allocation = True)
            #E.final_allocation_efficiency = E.calculate_efficiency_of_allocation(E.final_allocation, E.final_allocation_scw, verbose=1)

            # Save results per iteration 
            E.calc_time_spent()
            E.save_results(res_path)
            E.extend_per_iteration_results()
            break

        # Calculate efficieny per iteration
        if calc_efficiency_per_iteration:
            E.calculate_efficiency_per_iteration()

        # Save results per iteration
        E.calc_time_spent() # Calculate timings
        E.save_results(res_path)

    # allocation & payments
    if not E.found_clearing_prices:
        E.calculate_final_allocation()
        #E.final_allocation_efficiency = E.calculate_efficiency_of_allocation(E.final_allocation, E.final_allocation_scw, verbose=1)
        E.calculate_vcg_payments()

    # Calculate timings
    E.calc_time_spent()

    # FINAL SAVING OF RESULTS
     # save results here but NO wandb tracking (since this was done already in the iteration forloop)
    E.save_results(res_path, no_wandb_logging=True)

    
    #Save final wandb table
    E.wandb_final_table()

    # Final Info
    E.get_info(final_summary=True)
    return
