"""
ALNS启发式算法求解器
自适应大邻域搜索用于大规模EV-UAV协同配送问题
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple, Callable
import copy

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 没装 tqdm 时退化为空进度条
    tqdm = None

from .solution import Solution
from .operators import DestroyOperators, RepairOperators


class _NullProgressBar:
    """tqdm 缺失时的极简替身，只实现主循环用到的接口。"""

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get('total')
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def update(self, n=1):
        self.n += n

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _progress_bar(total: int, desc: str):
    """有 tqdm 就用 tqdm，没有就用空实现，保证核心算法不依赖可选包。"""
    if tqdm is not None:
        return tqdm(total=total, desc=desc)
    return _NullProgressBar(total=total)



class ALNSSolver:
    """自适应大邻域搜索求解器"""
    
    def __init__(self, dataset: Dict, config: Dict):
        """
        初始化ALNS求解器
        
        Args:
            dataset: 数据集
            config: 配置参数
        """
        self.dataset = dataset
        self.config = config
        self.alns_config = config.get('alns', {})
        
        # 算子及其权重
        self.destroy_operators = [
            DestroyOperators.random_removal,
            DestroyOperators.worst_removal,
            DestroyOperators.station_removal,
            DestroyOperators.route_removal
        ]
        
        self.repair_operators = [
            lambda sol, uns: RepairOperators.greedy_insertion(
                sol, uns, noise=self.alns_config.get('repair', {}).get('noise_parameter', 0.05)
            ),
            lambda sol, uns: RepairOperators.regret_insertion(
                sol, uns, k=self.alns_config.get('repair', {}).get('k_regret', 4)
            )
        ]
        
        # 算子权重
        num_destroy = len(self.destroy_operators)
        num_repair = len(self.repair_operators)
        
        self.destroy_weights = np.ones(num_destroy)
        self.repair_weights = np.ones(num_repair)
        
        # 算子使用次数和得分
        self.destroy_usage = np.zeros(num_destroy)
        self.repair_usage = np.zeros(num_repair)
        self.destroy_scores = np.zeros(num_destroy)
        self.repair_scores = np.zeros(num_repair)
        
        # 求解状态
        self.current_solution = None
        self.best_solution = None
        self.best_cost = float('inf')
        
        # 温度参数（模拟退火）
        self.temperature = None
        self.initial_temp = self.alns_config.get('temperature', {}).get('start', 10000)
        self.final_temp = self.alns_config.get('temperature', {}).get('end', 1)
        self.cooling_rate = self.alns_config.get('temperature', {}).get('cooling_rate', 0.9975)
        
        # 统计信息
        self.iteration_history = []
        self.best_history = []
    
    def solve(self, initial_solution: Solution = None, max_iterations: int = None) -> Solution:
        """
        执行ALNS算法
        
        Args:
            initial_solution: 初始解（如果为None则生成随机解）
            max_iterations: 最大迭代次数
            
        Returns:
            最优解
        """
        if max_iterations is None:
            max_iterations = self.alns_config.get('max_iterations', 10000)
        
        print("开始ALNS求解...")
        start_time = time.time()
        
        # 初始化
        if initial_solution is None:
            self.current_solution = self._generate_initial_solution()
        else:
            self.current_solution = initial_solution.copy()
        
        self.current_solution.evaluate()
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_solution.objective()
        
        self.temperature = self.initial_temp
        
        print(f"初始解成本: {self.best_cost:.2f}")
        
        # 主循环
        segment_size = self.alns_config.get('segment_size', 100)
        
        with _progress_bar(max_iterations, "ALNS迭代") as pbar:
            for iteration in range(max_iterations):
                # 选择算子
                destroy_idx = self._select_operator(self.destroy_weights)
                repair_idx = self._select_operator(self.repair_weights)
                
                # 破坏-修复
                new_solution = self._destroy_repair(
                    self.current_solution,
                    destroy_idx,
                    repair_idx,
                    iteration
                )
                
                # 评估新解
                new_solution.evaluate()
                new_cost = new_solution.objective()
                
                # 调试：检查解是否真的改变了（仅前3次迭代）
                if iteration < 3:
                    current_cost = self.current_solution.objective()
                    if abs(new_cost - current_cost) < 0.01:
                        print(f"\n[DEBUG] 迭代{iteration}: new_cost={new_cost:.2f}, current={current_cost:.2f}, diff={abs(new_cost-current_cost):.4f}")
                
                # 接受准则
                accept, reason = self._accept_solution(
                    self.current_solution.objective(),
                    new_cost
                )
                
                # 更新权重分数
                score = self._calculate_score(reason)
                self.destroy_scores[destroy_idx] += score
                self.repair_scores[repair_idx] += score
                self.destroy_usage[destroy_idx] += 1
                self.repair_usage[repair_idx] += 1
                
                # 接受新解
                if accept:
                    self.current_solution = new_solution
                    
                    # 更新最优解
                    if new_cost < self.best_cost:
                        self.best_solution = new_solution.copy()
                        self.best_cost = new_cost
                        pbar.set_postfix({'最优成本': f'{self.best_cost:.2f}'})
                
                # 降温
                self.temperature *= self.cooling_rate
                
                # 记录历史
                self.iteration_history.append(self.current_solution.objective())
                self.best_history.append(self.best_cost)
                
                # 每个segment更新权重
                if (iteration + 1) % segment_size == 0:
                    self._update_weights()
                
                # 更新进度条
                pbar.update(1)
                
                # 定期输出日志
                log_interval = self.alns_config.get('output', {}).get('log_interval', 100)
                if (iteration + 1) % log_interval == 0:
                    print(f"\n迭代 {iteration + 1}: 当前={self.current_solution.objective():.2f}, "
                          f"最优={self.best_cost:.2f}, 温度={self.temperature:.2f}")
        
        solve_time = time.time() - start_time
        
        print(f"\n求解完成！")
        print(f"最优成本: {self.best_cost:.2f}")
        print(f"求解时间: {solve_time:.2f}秒")
        
        self.solve_time = solve_time
        
        return self.best_solution
    
    def _generate_initial_solution(self) -> Solution:
        """
        生成初始解（EV-UAV协同贪心策略）
        
        Returns:
            初始解
        """
        solution = Solution(self.dataset, self.config)
        
        # 获取站点列表
        stations = [
            i for i, row in self.dataset['nodes'].iterrows()
            if row['type'] == 'station'
        ]
        
        # 构建EV路径：仓库 -> 所有站点 -> 仓库
        solution.ev_route.nodes = [0] + stations + [0]
        
        # 获取客户坐标
        customers_df = self.dataset['customers']
        nodes_df = self.dataset['nodes']
        n_customers = len(customers_df)
        
        # 为每个站点分配最近的客户使用UAV配送
        max_uav_per_station = self.config['uav'].get('num_drones', self.config['uav'].get('num_uavs', 5))
        
        for station in stations:
            station_coord = nodes_df.iloc[station][['x', 'y']].values
            
            # 计算所有未服务客户到该站点的距离
            unserved = list(solution.unserved_customers)
            if not unserved:
                break
            
            distances = []
            for cust_id in unserved:
                cust_coord = customers_df.iloc[cust_id][['x', 'y']].values
                dist_meters = np.linalg.norm(station_coord - cust_coord)
                dist_km = dist_meters / 1000  # 转换为公里
                
                # 检查UAV电池约束
                round_trip_dist_km = 2 * dist_km
                energy = round_trip_dist_km * self.config['uav']['energy_consumption']
                
                if energy <= self.config['uav']['battery_capacity']:
                    # 检查载重约束
                    demand = customers_df.iloc[cust_id]['demand']
                    if demand <= self.config['uav']['load_capacity']:
                        distances.append((cust_id, dist_km, energy))
            
            # 按距离排序，选择最近的客户
            distances.sort(key=lambda x: x[1])
            
            # 为该站点创建UAV sorties
            num_assigned = 0
            for cust_id, dist, energy in distances:
                if num_assigned >= max_uav_per_station:
                    break
                
                # 创建UAV sortie
                from .solution import UAVSortie
                sortie = UAVSortie(
                    uav_id=num_assigned,
                    launch_node=station,
                    customers=[cust_id],
                    launch_time=0.0,
                    return_time=0.0,
                    distance=2 * dist,  # dist已经是公里
                    energy=energy,
                    is_feasible=True
                )
                
                # 添加到解中
                if station not in solution.ev_route.uav_sorties:
                    solution.ev_route.uav_sorties[station] = []
                solution.ev_route.uav_sorties[station].append(sortie)
                
                # 更新服务状态
                solution.served_customers.add(cust_id)
                solution.unserved_customers.discard(cust_id)
                
                num_assigned += 1
        
        # 剩余客户由EV直接服务
        solution.ev_route.direct_customers = list(solution.unserved_customers)
        solution.served_customers.update(solution.unserved_customers)
        solution.unserved_customers.clear()
        
        # 调试：输出初始解统计
        total_uav_sorties = sum(len(sorties) for sorties in solution.ev_route.uav_sorties.values())
        total_ev_direct = len(solution.ev_route.direct_customers)
        print(f"初始解生成完成: UAV架次={total_uav_sorties}, EV直接={total_ev_direct}")
        
        return solution
    
    def _destroy_repair(
        self,
        solution: Solution,
        destroy_idx: int,
        repair_idx: int,
        iteration: int = -1
    ) -> Solution:
        """
        执行破坏-修复操作
        
        Args:
            solution: 当前解
            destroy_idx: 破坏算子索引
            repair_idx: 修复算子索引
            iteration: 当前迭代次数（用于调试）
            
        Returns:
            新解
        """
        # 复制解
        new_solution = solution.copy()
        
        # 计算移除数量
        n_customers = len(self.dataset['customers'])
        destroy_config = self.alns_config.get('destroy', {})
        min_removal = destroy_config.get('min_removal', 10)
        max_removal = destroy_config.get('max_removal', 50)
        removal_pct = destroy_config.get('removal_percentage', 0.25)
        
        num_remove = min(
            max_removal,
            max(min_removal, int(n_customers * removal_pct))
        )
        
        # 调试：记录破坏前的状态
        served_before = len(new_solution.served_customers)
        ev_direct_before = len(new_solution.ev_route.direct_customers)
        uav_sorties_before = sum(len(sorties) for sorties in new_solution.ev_route.uav_sorties.values())
        
        # 破坏
        destroy_op = self.destroy_operators[destroy_idx]
        removed = destroy_op(new_solution, num_remove)
        
        # 调试：检查是否真的移除了客户
        unserved_before_repair = len(new_solution.unserved_customers)
        served_after_destroy = len(new_solution.served_customers)
        
        # 调试输出
        if iteration < 3:
            print(f"\n[DEBUG-详细] 迭代{iteration}:")
            print(f"  破坏前: 服务{served_before}, EV直接{ev_direct_before}, UAV架次{uav_sorties_before}")
            print(f"  EV路径: {new_solution.ev_route.nodes}")
            print(f"  破坏: 应移除{num_remove}, 实际移除{len(removed) if removed else 0}")
            print(f"  破坏后: 服务{served_after_destroy}, 未服务{unserved_before_repair}")
        
        if unserved_before_repair == 0:
            if iteration < 3:
                print(f"  ⚠ 警告: 破坏后未服务客户数=0! 破坏操作失败!")
        
        # 修复
        repair_op = self.repair_operators[repair_idx]
        # 在前3次迭代直接调用原始方法启用debug
        if iteration < 3 and repair_idx == 0:  # greedy_insertion是第一个修复算子
            RepairOperators.greedy_insertion(
                new_solution,
                new_solution.unserved_customers.copy(),
                noise=self.alns_config.get('repair', {}).get('noise_parameter', 0.05),
                debug=True
            )
        else:
            repair_op(new_solution, new_solution.unserved_customers.copy())
        
        # 调试：检查修复后的状态
        unserved_after_repair = len(new_solution.unserved_customers)
        served_after_repair = len(new_solution.served_customers)
        ev_direct_after = len(new_solution.ev_route.direct_customers)
        uav_sorties_after = sum(len(sorties) for sorties in new_solution.ev_route.uav_sorties.values())
        
        if iteration < 3:
            print(f"  修复后: 服务{served_after_repair}, 未服务{unserved_after_repair}, EV直接{ev_direct_after}, UAV架次{uav_sorties_after}")
            
            # 检查解结构是否真的改变了
            if (ev_direct_before == ev_direct_after and uav_sorties_before == uav_sorties_after):
                print(f"  ⚠⚠⚠ 关键问题: 解的结构完全没变! (EV直接和UAV架次数都相同)")
        
        # 如果有未服务客户，说明修复失败
        if unserved_after_repair > 0:
            # 强制将剩余客户分配给EV
            for cust in list(new_solution.unserved_customers):
                new_solution.ev_route.direct_customers.append(cust)
                new_solution.served_customers.add(cust)
                new_solution.unserved_customers.discard(cust)
        
        return new_solution
    
    def _select_operator(self, weights: np.ndarray) -> int:
        """
        轮盘赌选择算子
        
        Args:
            weights: 权重数组
            
        Returns:
            选中的算子索引
        """
        total = np.sum(weights)
        if total == 0:
            return np.random.randint(len(weights))
        
        probabilities = weights / total
        return np.random.choice(len(weights), p=probabilities)
    
    def _accept_solution(self, current_cost: float, new_cost: float) -> Tuple[bool, str]:
        """
        接受准则（模拟退火）
        
        Args:
            current_cost: 当前解成本
            new_cost: 新解成本
            
        Returns:
            (是否接受, 原因)
        """
        if new_cost == float('inf'):
            return False, 'infeasible'
        
        if new_cost < self.best_cost:
            return True, 'new_best'
        
        if new_cost < current_cost:
            return True, 'better'
        
        # 模拟退火接受劣解
        delta = new_cost - current_cost
        probability = np.exp(-delta / max(self.temperature, 0.01))
        
        if random.random() < probability:
            return True, 'accepted'
        
        return False, 'rejected'
    
    def _calculate_score(self, reason: str) -> float:
        """
        根据接受原因计算分数
        
        Args:
            reason: 接受原因
            
        Returns:
            分数
        """
        weights = self.alns_config.get('weights', {}).get('start', [10, 5, 2, 1])
        
        score_map = {
            'new_best': weights[0],
            'better': weights[1],
            'accepted': weights[2],
            'rejected': weights[3]
        }
        
        return score_map.get(reason, 0)
    
    def _update_weights(self):
        """更新算子权重"""
        decay = self.alns_config.get('weights', {}).get('decay', 0.95)
        
        # 更新破坏算子权重
        for i in range(len(self.destroy_operators)):
            if self.destroy_usage[i] > 0:
                score = self.destroy_scores[i] / self.destroy_usage[i]
                self.destroy_weights[i] = decay * self.destroy_weights[i] + (1 - decay) * score
            
            # 重置计数
            self.destroy_scores[i] = 0
            self.destroy_usage[i] = 0
        
        # 更新修复算子权重
        for i in range(len(self.repair_operators)):
            if self.repair_usage[i] > 0:
                score = self.repair_scores[i] / self.repair_usage[i]
                self.repair_weights[i] = decay * self.repair_weights[i] + (1 - decay) * score
            
            # 重置计数
            self.repair_scores[i] = 0
            self.repair_usage[i] = 0
    
    def get_statistics(self) -> Dict:
        """
        获取求解统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'solve_time': self.solve_time,
            'best_cost': self.best_cost,
            'num_iterations': len(self.iteration_history),
            'final_temperature': self.temperature,
            'destroy_weights': self.destroy_weights.tolist(),
            'repair_weights': self.repair_weights.tolist()
        }


def main():
    """测试ALNS算法"""
    from data_generator import DataGenerator
    
    print("生成测试数据...")
    gen = DataGenerator('config/params_large.yaml')
    dataset = gen.generate_dataset()
    
    print("\n运行ALNS算法...")
    alns = ALNSSolver(dataset, dataset['config'])
    solution = alns.solve(max_iterations=1000)
    
    if solution:
        print("\n最优解信息:")
        print(solution)
        print("\n性能指标:")
        metrics = solution.evaluate()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print("\n求解统计:")
        stats = alns.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
