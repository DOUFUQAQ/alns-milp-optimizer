"""
解的表示与评价模块
定义EV路径、UAV sortie和完整解的数据结构
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import copy


@dataclass
class UAVSortie:
    """UAV单次飞行任务"""
    uav_id: int
    launch_node: int  # EV停靠节点
    customers: List[int]  # 服务的客户列表
    launch_time: float = 0.0
    return_time: float = 0.0
    distance: float = 0.0
    energy: float = 0.0
    is_feasible: bool = True
    
    def copy(self):
        """深拷贝"""
        return copy.deepcopy(self)


@dataclass
class EVRoute:
    """EV路径"""
    nodes: List[int]  # 访问的节点序列（包括仓库）
    arrival_times: List[float] = field(default_factory=list)
    departure_times: List[float] = field(default_factory=list)
    battery_levels: List[float] = field(default_factory=list)
    uav_sorties: Dict[int, List[UAVSortie]] = field(default_factory=dict)  # 每个节点的UAV sorties
    direct_customers: List[int] = field(default_factory=list)  # EV直接服务的客户
    
    def copy(self):
        """深拷贝"""
        return copy.deepcopy(self)


class Solution:
    """完整解"""
    
    def __init__(self, dataset: Dict, config: Dict):
        """
        初始化解
        
        Args:
            dataset: 数据集
            config: 配置参数
        """
        self.dataset = dataset
        self.config = config
        
        self.ev_route = EVRoute(nodes=[0])  # 从仓库开始
        self.served_customers = set()  # 已服务的客户集合
        self.unserved_customers = set(range(len(dataset['customers'])))  # 未服务的客户
        
        # 缓存坐标数据（性能优化）
        self._customer_coords = dataset['customers'][['x', 'y']].values
        self._node_coords = dataset['nodes'][['x', 'y']].values
        self._customer_demands = dataset['customers']['demand'].values
        
        # 性能指标
        self.total_cost = float('inf')
        self.ev_cost = 0.0
        self.uav_cost = 0.0
        self.energy_cost = 0.0
        self.time_cost = 0.0
        self.carbon_cost = 0.0
        self.delay_penalty = 0.0
        
        self.total_distance = 0.0
        self.total_time = 0.0
        self.total_energy = 0.0
        self.is_feasible = True
        self.infeasibility_reasons = []
    
    def objective(self) -> float:
        """
        计算目标函数值（用于ALNS）
        
        Returns:
            目标函数值
        """
        if not self.is_feasible:
            return float('inf')
        return self.total_cost
    
    def evaluate(self) -> Dict:
        """
        评估解的质量，计算所有成本
        
        Returns:
            包含各项成本的字典
        """
        self._reset_metrics()
        
        # 检查客户覆盖
        if len(self.served_customers) != len(self.dataset['customers']):
            self.is_feasible = False
            self.infeasibility_reasons.append("Not all customers served")
            return self._get_metrics()
        
        # 评估EV路径
        self._evaluate_ev_route()
        
        # 评估UAV sorties
        self._evaluate_uav_sorties()
        
        # 评估EV直接服务的客户
        self._evaluate_direct_delivery()
        
        # 计算总成本
        self._calculate_total_cost()
        
        return self._get_metrics()
    
    def _reset_metrics(self):
        """重置所有性能指标"""
        self.total_cost = 0.0
        self.ev_cost = 0.0
        self.uav_cost = 0.0
        self.energy_cost = 0.0
        self.time_cost = 0.0
        self.carbon_cost = 0.0
        self.delay_penalty = 0.0
        self.total_distance = 0.0
        self.total_time = 0.0
        self.total_energy = 0.0
        self.is_feasible = True
        self.infeasibility_reasons = []
    
    def _evaluate_ev_route(self):
        """评估EV路径"""
        route = self.ev_route.nodes
        if len(route) < 2:
            return
        
        distances = self.dataset['distances']['ev']
        times = self.dataset['times']['ev']
        energies = self.dataset['energy']['ev']
        
        # 计算路径距离、时间和能耗
        ev_distance = 0.0
        ev_time = 0.0
        ev_energy = 0.0
        
        for i in range(len(route) - 1):
            ev_distance += distances[route[i], route[i+1]]
            ev_time += times[route[i], route[i+1]]
            ev_energy += energies[route[i], route[i+1]]
        
        # 检查电池容量约束
        if ev_energy > self.config['ev']['battery_capacity']:
            self.is_feasible = False
            self.infeasibility_reasons.append("EV battery capacity exceeded")
        
        self.total_distance += ev_distance
        self.total_time += ev_time
        self.total_energy += ev_energy
        
        # EV固定成本
        self.ev_cost = self.config['ev']['fixed_cost']
    
    def _evaluate_uav_sorties(self):
        """评估所有UAV sorties"""
        num_sorties = 0
        
        for node, sorties in self.ev_route.uav_sorties.items():
            for sortie in sorties:
                num_sorties += 1
                
                # 计算sortie的距离和能耗
                self._evaluate_single_sortie(sortie, node)
                
                if not sortie.is_feasible:
                    self.is_feasible = False
                
                self.total_distance += sortie.distance
                self.total_energy += sortie.energy
        
        # UAV固定成本
        self.uav_cost = num_sorties * self.config['uav']['fixed_cost']
    
    def _evaluate_single_sortie(self, sortie: UAVSortie, launch_node: int):
        """
        评估单个UAV sortie（优化版）
        
        Args:
            sortie: UAV sortie
            launch_node: 发射节点ID
        """
        if not sortie.customers:
            sortie.is_feasible = False
            return
        
        # 使用缓存的坐标数据
        launch_coord = self._node_coords[launch_node]
        
        total_distance_meters = 0.0
        
        # 从发射点到第一个客户
        first_customer = sortie.customers[0]
        customer_coord = self._customer_coords[first_customer]
        dist = np.linalg.norm(launch_coord - customer_coord)
        total_distance_meters += dist
        
        # 客户之间 - 向量化计算
        if len(sortie.customers) > 1:
            customer_indices = np.array(sortie.customers)
            coords = self._customer_coords[customer_indices]
            diffs = np.diff(coords, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_distance_meters += distances.sum()
        
        # 从最后一个客户返回发射点
        last_customer = sortie.customers[-1]
        customer_coord = self._customer_coords[last_customer]
        dist = np.linalg.norm(customer_coord - launch_coord)
        total_distance_meters += dist
        
        # 转换为公里
        total_distance_km = total_distance_meters / 1000
        
        # 计算能耗
        total_energy = total_distance_km * self.config['uav']['energy_consumption']
        
        # 加上服务时间
        total_time = total_distance_km / self.config['uav']['speed'] * 60  # 转换为分钟
        total_time += len(sortie.customers) * self.config['customers']['service_time']
        total_time += self.config['uav']['launch_time'] + self.config['uav']['landing_time']
        
        # 检查电池容量
        if total_energy > self.config['uav']['battery_capacity']:
            sortie.is_feasible = False
            self.infeasibility_reasons.append(f"UAV sortie battery capacity exceeded: {total_energy:.2f} > {self.config['uav']['battery_capacity']}")
        
        # 检查载重 - 使用缓存的需求数据
        total_demand = self._customer_demands[sortie.customers].sum()
        if total_demand > self.config['uav']['load_capacity']:
            sortie.is_feasible = False
            self.infeasibility_reasons.append(f"UAV sortie load capacity exceeded")
        
        sortie.distance = total_distance_km
        sortie.energy = total_energy
        sortie.return_time = sortie.launch_time + total_time
    
    def _evaluate_direct_delivery(self):
        """评估EV直接配送的客户"""
        # 对于直接配送，成本已包含在EV路径评估中
        # 这里主要检查时间窗约束
        pass
    
    def _calculate_total_cost(self):
        """计算总成本"""
        # 能耗成本
        self.energy_cost = self.total_energy * self.config['costs']['energy_weight']
        
        # 时间成本
        self.time_cost = self.total_time * self.config['costs']['time_weight']
        
        # 碳排放成本
        carbon_emission = self.total_energy * self.config['costs']['carbon_factor']
        self.carbon_cost = carbon_emission * self.config['costs']['carbon_weight']
        
        # 总成本
        self.total_cost = (
            self.ev_cost +
            self.uav_cost +
            self.energy_cost +
            self.time_cost +
            self.carbon_cost +
            self.delay_penalty
        )
    
    def _get_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            'total_cost': self.total_cost,
            'ev_cost': self.ev_cost,
            'uav_cost': self.uav_cost,
            'energy_cost': self.energy_cost,
            'time_cost': self.time_cost,
            'carbon_cost': self.carbon_cost,
            'delay_penalty': self.delay_penalty,
            'total_distance': self.total_distance,
            'total_time': self.total_time,
            'total_energy': self.total_energy,
            'is_feasible': self.is_feasible,
            'num_served': len(self.served_customers),
            'num_total': len(self.dataset['customers'])
        }
    
    def copy(self):
        """深拷贝解"""
        return copy.deepcopy(self)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'ev_route': {
                'nodes': self.ev_route.nodes,
                'arrival_times': self.ev_route.arrival_times,
                'departure_times': self.ev_route.departure_times,
                'direct_customers': self.ev_route.direct_customers
            },
            'uav_sorties': {
                node: [
                    {
                        'uav_id': s.uav_id,
                        'customers': s.customers,
                        'launch_time': s.launch_time,
                        'return_time': s.return_time,
                        'distance': s.distance,
                        'energy': s.energy
                    }
                    for s in sorties
                ]
                for node, sorties in self.ev_route.uav_sorties.items()
            },
            'metrics': self._get_metrics()
        }
    
    def __repr__(self):
        """字符串表示"""
        return (
            f"Solution(cost={self.total_cost:.2f}, "
            f"served={len(self.served_customers)}/{len(self.dataset['customers'])}, "
            f"feasible={self.is_feasible})"
        )
