"""
ALNS算子模块
包含破坏算子（Destroy）和修复算子（Repair）
"""

import numpy as np
import random
from typing import List, Set, Dict, Optional
from .solution import Solution, UAVSortie


class DestroyOperators:
    """破坏算子集合"""
    
    @staticmethod
    def random_removal(solution: Solution, num_remove: int) -> Set[int]:
        """
        随机移除算子
        
        Args:
            solution: 当前解
            num_remove: 移除的客户数
            
        Returns:
            移除的客户集合
        """
        if len(solution.served_customers) == 0:
            return set()
        
        num_remove = min(num_remove, len(solution.served_customers))
        removed = set(random.sample(list(solution.served_customers), num_remove))
        
        # 从解中移除这些客户
        DestroyOperators._remove_customers(solution, removed)
        
        return removed
    
    @staticmethod
    def worst_removal(solution: Solution, num_remove: int, p: float = 3.0) -> Set[int]:
        """
        最差移除算子 - 移除成本最高的客户
        
        Args:
            solution: 当前解
            num_remove: 移除的客户数
            p: 随机化参数
            
        Returns:
            移除的客户集合
        """
        if len(solution.served_customers) == 0:
            return set()
        
        # 计算每个客户的贡献成本
        customer_costs = DestroyOperators._calculate_customer_costs(solution)
        
        # 按成本排序
        sorted_customers = sorted(
            customer_costs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 使用随机化选择
        removed = set()
        num_remove = min(num_remove, len(sorted_customers))
        
        for _ in range(num_remove):
            if not sorted_customers:
                break
            
            # 随机选择（偏向成本高的）
            idx = int(random.random() ** p * len(sorted_customers))
            customer, _ = sorted_customers.pop(idx)
            removed.add(customer)
        
        # 从解中移除
        DestroyOperators._remove_customers(solution, removed)
        
        return removed
    
    @staticmethod
    def station_removal(solution: Solution, num_remove: int) -> Set[int]:
        """
        站点移除算子 - 随机选择一个站点，移除其所有UAV sorties
        
        Args:
            solution: 当前解
            num_remove: 目标移除数量（实际可能不同）
            
        Returns:
            移除的客户集合
        """
        if not solution.ev_route.uav_sorties:
            return set()
        
        # 选择一个有UAV sorties的站点
        stations_with_uav = list(solution.ev_route.uav_sorties.keys())
        if not stations_with_uav:
            return set()
        
        station = random.choice(stations_with_uav)
        
        # 移除该站点的所有sorties
        removed = set()
        for sortie in solution.ev_route.uav_sorties[station]:
            removed.update(sortie.customers)
        
        # 从解中移除
        DestroyOperators._remove_customers(solution, removed)
        
        return removed
    
    @staticmethod
    def route_removal(solution: Solution, num_remove: int) -> Set[int]:
        """
        路线移除算子 - 移除相邻的客户（优化版）
        
        Args:
            solution: 当前解
            num_remove: 移除的客户数
            
        Returns:
            移除的客户集合
        """
        if len(solution.served_customers) == 0:
            return set()
        
        # 随机选择一个起始客户
        start_customer = random.choice(list(solution.served_customers))
        removed = {start_customer}
        
        num_remove = min(num_remove, len(solution.served_customers))
        
        # 预提取所有客户坐标到numpy数组
        customers_df = solution.dataset['customers']
        coords = customers_df[['x', 'y']].values  # 一次性提取所有坐标
        
        served_list = list(solution.served_customers)
        
        while len(removed) < num_remove:
            # 使用向量化计算距离
            removed_list = list(removed)
            candidates = [c for c in served_list if c not in removed]
            
            if not candidates:
                break
            
            # 计算所有候选客户到已移除客户的最小距离
            min_distances = []
            for cand in candidates:
                cand_coord = coords[cand]
                dists = np.sqrt(np.sum((coords[removed_list] - cand_coord)**2, axis=1))
                min_distances.append((cand, dists.min()))
            
            # 选择距离最近的
            nearest, _ = min(min_distances, key=lambda x: x[1])
            removed.add(nearest)
        
        # 从解中移除
        DestroyOperators._remove_customers(solution, removed)
        
        return removed
    
    @staticmethod
    def _calculate_customer_costs(solution: Solution) -> Dict[int, float]:
        """
        计算每个客户的贡献成本
        
        Args:
            solution: 解
            
        Returns:
            客户成本字典
        """
        customer_costs = {}
        
        # 计算UAV sorties中的客户成本
        for node, sorties in solution.ev_route.uav_sorties.items():
            for sortie in sorties:
                cost_per_customer = sortie.energy / max(len(sortie.customers), 1)
                for c in sortie.customers:
                    customer_costs[c] = cost_per_customer
        
        # EV直接服务的客户
        for c in solution.ev_route.direct_customers:
            # 简化：使用固定成本
            customer_costs[c] = solution.config['ev']['fixed_cost'] / max(len(solution.ev_route.direct_customers), 1)
        
        return customer_costs
    
    @staticmethod
    def _remove_customers(solution: Solution, removed: Set[int]):
        """
        从解中移除客户
        
        Args:
            solution: 解
            removed: 要移除的客户集合
        """
        # 从UAV sorties中移除
        for node in list(solution.ev_route.uav_sorties.keys()):
            sorties = solution.ev_route.uav_sorties[node]
            new_sorties = []
            
            for sortie in sorties:
                # 移除客户
                sortie.customers = [c for c in sortie.customers if c not in removed]
                
                # 如果sortie还有客户，保留
                if sortie.customers:
                    new_sorties.append(sortie)
            
            if new_sorties:
                solution.ev_route.uav_sorties[node] = new_sorties
            else:
                del solution.ev_route.uav_sorties[node]
        
        # 从EV直接服务中移除
        solution.ev_route.direct_customers = [
            c for c in solution.ev_route.direct_customers
            if c not in removed
        ]
        
        # 更新served和unserved集合
        solution.served_customers -= removed
        solution.unserved_customers |= removed


class RepairOperators:
    """修复算子集合"""
    
    @staticmethod
    def greedy_insertion(
        solution: Solution,
        unserved: Set[int],
        noise: float = 0.0,
        debug: bool = False
    ) -> None:
        """
        贪心插入算子
        
        Args:
            solution: 当前解
            unserved: 未服务的客户集合
            noise: 噪声参数（0-1之间）
            debug: 是否打印调试信息
        """
        customers_df = solution.dataset['customers']
        nodes_df = solution.dataset['nodes']
        
        unserved_list = list(unserved)
        random.shuffle(unserved_list)
        
        debug_count = 0
        for customer in unserved_list:
            demand = customers_df.iloc[customer]['demand']
            
            # 检查是否可以由UAV服务
            if demand <= solution.config['uav']['load_capacity']:
                best_cost = float('inf')
                best_node = None
                best_sortie_idx = None
                
                # 尝试插入到现有sorties或创建新sortie
                for node in solution.ev_route.nodes[:-1]:  # 排除最后返回仓库
                    if nodes_df.iloc[node]['type'] != 'station':
                        continue
                    
                    # 尝试创建新sortie
                    cost = RepairOperators._calculate_uav_insertion_cost(
                        solution, node, customer, None
                    )
                    
                    # 添加噪声
                    if noise > 0:
                        cost *= (1 + random.uniform(-noise, noise))
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_node = node
                        best_sortie_idx = None
                    
                    # 尝试插入到现有sorties
                    if node in solution.ev_route.uav_sorties:
                        for idx, sortie in enumerate(solution.ev_route.uav_sorties[node]):
                            cost = RepairOperators._calculate_uav_insertion_cost(
                                solution, node, customer, sortie
                            )
                            
                            if noise > 0:
                                cost *= (1 + random.uniform(-noise, noise))
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_node = node
                                best_sortie_idx = idx
                
                # 执行插入
                if best_node is not None and best_cost < float('inf'):
                    if debug and debug_count < 3:
                        print(f"    [修复] 客户{customer}: UAV成本={best_cost:.2f}, 使用UAV")
                        debug_count += 1
                    RepairOperators._insert_uav_customer(
                        solution, best_node, customer, best_sortie_idx
                    )
                    continue
                else:
                    if debug and debug_count < 3:
                        # 计算实际距离看为什么是inf
                        if best_node is not None:
                            node_coord = solution._node_coords[best_node]
                            cust_coord = solution._customer_coords[customer]
                            dist = 2 * np.linalg.norm(node_coord - cust_coord)
                            energy = dist * solution.config['uav']['energy_consumption']
                            battery = solution.config['uav']['battery_capacity']
                            print(f"    [修复] 客户{customer}: 距离={dist/1000:.2f}km, 能耗={energy:.2f}kWh, 电池={battery:.2f}kWh → UAV成本=inf, 使用EV")
                        else:
                            print(f"    [修复] 客户{customer}: 无可用节点 → UAV成本=inf, 使用EV")
                        debug_count += 1
            
            # 否则由EV直接服务
            solution.ev_route.direct_customers.append(customer)
            solution.served_customers.add(customer)
            solution.unserved_customers.discard(customer)
    
    @staticmethod
    def regret_insertion(
        solution: Solution,
        unserved: Set[int],
        k: int = 3
    ) -> None:
        """
        k-regret插入算子
        
        Args:
            solution: 当前解
            unserved: 未服务的客户集合
            k: regret值的k
        """
        customers_df = solution.dataset['customers']
        
        while unserved:
            best_regret = -float('inf')
            best_customer = None
            best_position = None
            
            # 对每个未服务客户计算regret值
            for customer in unserved:
                # 找到k个最佳插入位置
                insertion_costs = []
                
                demand = customers_df.iloc[customer]['demand']
                
                # UAV插入
                if demand <= solution.config['uav']['load_capacity']:
                    for node in solution.ev_route.nodes[:-1]:
                        if solution.dataset['nodes'].iloc[node]['type'] != 'station':
                            continue
                        
                        cost = RepairOperators._calculate_uav_insertion_cost(
                            solution, node, customer, None
                        )
                        insertion_costs.append((cost, ('uav', node, None)))
                
                # EV插入
                ev_cost = solution.config['ev']['fixed_cost'] / 10  # 简化
                insertion_costs.append((ev_cost, ('ev', None, None)))
                
                # 排序
                insertion_costs.sort(key=lambda x: x[0])
                
                # 计算regret值
                if len(insertion_costs) >= k:
                    regret = sum(insertion_costs[i][0] for i in range(1, k)) - insertion_costs[0][0] * (k - 1)
                else:
                    regret = insertion_costs[-1][0] - insertion_costs[0][0]
                
                if regret > best_regret:
                    best_regret = regret
                    best_customer = customer
                    best_position = insertion_costs[0][1]
            
            if best_customer is None:
                break
            
            # 执行插入
            mode, node, sortie_idx = best_position
            if mode == 'uav':
                RepairOperators._insert_uav_customer(solution, node, best_customer, sortie_idx)
            else:
                solution.ev_route.direct_customers.append(best_customer)
                solution.served_customers.add(best_customer)
            
            unserved.remove(best_customer)
            solution.unserved_customers.discard(best_customer)
    
    @staticmethod
    def _calculate_uav_insertion_cost(
        solution: Solution,
        node: int,
        customer: int,
        sortie: UAVSortie = None
    ) -> float:
        """
        计算UAV插入成本（优化版）
        
        Args:
            solution: 解
            node: 节点
            customer: 客户
            sortie: 现有sortie（None表示创建新sortie）
            
        Returns:
            插入成本
        """
        # 使用缓存的坐标数据
        node_coord = solution._node_coords[node]
        cust_coord = solution._customer_coords[customer]
        
        if sortie is None:
            # 创建新sortie：往返距离
            dist_meters = 2 * np.linalg.norm(node_coord - cust_coord)
            dist_km = dist_meters / 1000  # 转换为公里
            energy = dist_km * solution.config['uav']['energy_consumption']
            
            # 检查电池容量
            if energy > solution.config['uav']['battery_capacity']:
                return float('inf')
            
            return energy + solution.config['uav']['fixed_cost']
        else:
            # 插入到现有sortie：计算增量成本
            if not sortie.customers:
                return float('inf')
            
            last_cust = sortie.customers[-1]
            last_coord = solution._customer_coords[last_cust]
            
            # 原来的返回距离（米）
            old_return_dist_m = np.linalg.norm(last_coord - node_coord)
            
            # 新的距离：last -> new -> node（米）
            new_dist_m = (
                np.linalg.norm(last_coord - cust_coord) +
                np.linalg.norm(cust_coord - node_coord)
            )
            
            # 转换为公里
            incremental_dist_km = (new_dist_m - old_return_dist_m) / 1000
            incremental_energy = incremental_dist_km * solution.config['uav']['energy_consumption']
            
            # 检查总能耗
            total_energy = sortie.energy + incremental_energy
            if total_energy > solution.config['uav']['battery_capacity']:
                return float('inf')
            
            # 检查总需求 - 使用缓存数据
            total_demand = solution._customer_demands[sortie.customers].sum()
            total_demand += solution._customer_demands[customer]
            if total_demand > solution.config['uav']['load_capacity']:
                return float('inf')
            
            return incremental_energy
    
    @staticmethod
    def _insert_uav_customer(
        solution: Solution,
        node: int,
        customer: int,
        sortie_idx: Optional[int]
    ):
        """
        将客户插入到UAV sortie中
        
        Args:
            solution: 解
            node: 节点
            customer: 客户
            sortie_idx: sortie索引（None表示创建新sortie）
        """
        if sortie_idx is None:
            # 创建新sortie
            new_sortie = UAVSortie(
                uav_id=0,
                launch_node=node,
                customers=[customer]
            )
            
            if node not in solution.ev_route.uav_sorties:
                solution.ev_route.uav_sorties[node] = []
            
            solution.ev_route.uav_sorties[node].append(new_sortie)
        else:
            # 插入到现有sortie
            sortie = solution.ev_route.uav_sorties[node][sortie_idx]
            sortie.customers.append(customer)
        
        solution.served_customers.add(customer)
        solution.unserved_customers.discard(customer)
