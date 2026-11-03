"""测试用的公共构造工具。

只依赖 numpy / pandas（仓库核心依赖），不引入 YAML、matplotlib、gurobipy。

提供三样东西：
  * make_config()          —— 与 config/params_small.yaml 等价的纯 dict 配置
  * make_tiny_dataset()    —— 手算可验证的 3 客户小算例（坐标单位：米）
  * load_fixture_dataset() —— 仓库内置的 20 客户算例（tests/fixtures/small_scale）
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "small_scale"


def deep_merge(base, override):
    """递归合并两个 dict，override 优先（不修改入参）。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def make_config(**overrides):
    """返回一份完整可用的配置字典（数值取自 config/params_small.yaml）。"""
    config = {
        "geography": {
            "area_size": [10, 10],
            "depot_location": [5, 5],
            "num_stations": 4,
            "num_charging_stations": 2,
            "num_customers": 6,
        },
        "ev": {
            "speed": 40,
            "battery_capacity": 60,
            "energy_consumption": 0.20,
            "load_capacity": 500,
            "fixed_cost": 200,
            "charging_rate": 50,
            "charging_cost": 0.8,
        },
        "uav": {
            "num_drones": 3,
            "speed": 60,
            "battery_capacity": 1.5,
            "energy_consumption": 0.03,
            "load_capacity": 5,
            "fixed_cost": 1.0,
            "launch_time": 2,
            "landing_time": 1,
        },
        "customers": {
            "demand_range": [0.5, 4.0],
            "service_time": 3,
            "time_window_type": "medium",
            "time_window_width": {"wide": 120, "medium": 60, "narrow": 30},
        },
        "costs": {
            "energy_weight": 1.0,
            "time_weight": 0.5,
            "carbon_weight": 0.3,
            "delay_penalty": 10.0,
            "carbon_factor": 0.5,
        },
        "alns": {
            "max_iterations": 20,
            "segment_size": 5,
            "weights": {"start": [10, 5, 2, 1], "decay": 0.95},
            "temperature": {"start": 10000, "end": 1, "cooling_rate": 0.9975},
            "destroy": {"min_removal": 3, "max_removal": 8, "removal_percentage": 0.2},
            "repair": {"noise_parameter": 0.05, "k_regret": 3},
            "output": {"log_interval": 100},
        },
    }
    return deep_merge(config, overrides)


def make_tiny_dataset():
    """构造一个可以手算的小算例。

    坐标系单位是 **米**（solution.py 计算 UAV 距离时按米换算成公里），
    而 distances / times / energy 矩阵的单位是 **公里 / 分钟 / kWh**，
    与 DataGenerator 的输出保持一致。

    节点：0=仓库(0,0)，1=站点(0,2000)，2=充电站(2000,0)
    客户：c0(0,3000) demand=1.0，c1(0,5000) demand=2.0，c2(3000,0) demand=4.0
    """
    nodes = pd.DataFrame(
        [
            {"node_id": 0, "type": "depot", "x": 0.0, "y": 0.0, "name": "Depot"},
            {"node_id": 1, "type": "station", "x": 0.0, "y": 2000.0, "name": "Station_1"},
            {"node_id": 2, "type": "charging_station", "x": 2000.0, "y": 0.0, "name": "Charging_1"},
        ]
    )
    customers = pd.DataFrame(
        [
            {"customer_id": 0, "x": 0.0, "y": 3000.0, "demand": 1.0, "service_time": 3,
             "time_window_start": 0.0, "time_window_end": 60.0, "name": "C0"},
            {"customer_id": 1, "x": 0.0, "y": 5000.0, "demand": 2.0, "service_time": 3,
             "time_window_start": 0.0, "time_window_end": 60.0, "name": "C1"},
            {"customer_id": 2, "x": 3000.0, "y": 0.0, "demand": 4.0, "service_time": 3,
             "time_window_start": 0.0, "time_window_end": 60.0, "name": "C2"},
        ]
    )

    node_xy = nodes[["x", "y"]].values
    n_nodes = len(nodes)
    dist_ev = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            # 坐标是米，矩阵按公里存
            dist_ev[i, j] = np.linalg.norm(node_xy[i] - node_xy[j]) / 1000.0

    config = make_config()
    return {
        "nodes": nodes,
        "customers": customers,
        "distances": {"ev": dist_ev, "uav": dist_ev.copy()},
        "times": {
            "ev": dist_ev / config["ev"]["speed"] * 60,
            "uav": dist_ev / config["uav"]["speed"] * 60,
        },
        "energy": {
            "ev": dist_ev * config["ev"]["energy_consumption"],
            "uav": dist_ev * config["uav"]["energy_consumption"],
        },
        "config": config,
    }


def load_fixture_dataset(fixture_dir=None):
    """加载仓库内置的 20 客户小算例（走真实的 DataGenerator.load_dataset 路径）。"""
    from src.data_generator import DataGenerator

    directory = str(fixture_dir or FIXTURE_DIR)
    return DataGenerator.load_dataset(directory)


def served_structure(solution):
    """把解里所有"已服务的客户"按出现次数统计出来，用于查重。"""
    counter = {}
    for sorties in solution.ev_route.uav_sorties.values():
        for sortie in sorties:
            for customer in sortie.customers:
                counter[customer] = counter.get(customer, 0) + 1
    for customer in solution.ev_route.direct_customers:
        counter[customer] = counter.get(customer, 0) + 1
    return counter


def integrity_problems(solution):
    """检查解的结构不变量，返回问题描述列表（空列表表示没问题）。

    覆盖四条不变量：
      1. served / unserved 是两个互不相交的集合
      2. 两者的并集恰好是全部客户
      3. 结构里的每个客户只出现一次（不重复配送）
      4. served_customers 与结构里的客户集合完全一致
    """
    problems = []
    n_customers = len(solution.dataset["customers"])
    served = set(solution.served_customers)
    unserved = set(solution.unserved_customers)

    if served & unserved:
        problems.append("served 与 unserved 有交集: %s" % sorted(served & unserved))
    if served | unserved != set(range(n_customers)):
        missing = set(range(n_customers)) - (served | unserved)
        problems.append("client 集合并集不是全集，缺少: %s" % sorted(missing))

    counter = served_structure(solution)
    duplicated = sorted(c for c, count in counter.items() if count > 1)
    if duplicated:
        problems.append("客户被重复配送: %s" % duplicated)

    structured = set(counter)
    if structured != served:
        problems.append(
            "结构中的客户集合与 served 不一致: 结构多 %s / served 多 %s"
            % (sorted(structured - served), sorted(served - structured))
        )
    return problems


def sortie_demands(solution):
    """返回 [(站点, 客户列表, 该 sortie 的总需求)]。"""
    demands = solution.dataset["customers"]["demand"].values
    out = []
    for node, sorties in solution.ev_route.uav_sorties.items():
        for sortie in sorties:
            total = float(demands[sortie.customers].sum()) if sortie.customers else 0.0
            out.append((node, list(sortie.customers), total))
    return out
