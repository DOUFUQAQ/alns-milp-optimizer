"""
MILP精确求解模型
使用Gurobi求解小规模EV-UAV协同配送问题
"""

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于本机是否装了商业求解器
    # gurobipy 是可选依赖：没装时模块依然可以导入（纯 Python 的辅助方法仍可用），
    # 只有真正构建模型时才会报错。
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from .solution import Solution, UAVSortie, EVRoute


class MILPModel:
    """MILP精确求解器"""
    
    def __init__(self, dataset: Dict, config: Dict):
        """
        初始化MILP模型
        
        Args:
            dataset: 数据集
            config: 配置参数
        """
        self.dataset = dataset
        self.config = config
        
        self.model = None
        self.solution = None
        
        # 决策变量
        self.x = {}  # EV路径
        self.y = {}  # UAV分配
        self.z = {}  # EV直接服务
        self.T = {}  # 时间变量
        self.E = {}  # 能耗变量
        
        self.solve_time = 0.0
        self.obj_value = None
        self.gap = None
    
    def build_model(self):
        """构建MILP模型"""
        if not GUROBI_AVAILABLE:
            raise RuntimeError(
                "MILPModel 需要 Gurobi 求解器：\n"
                "  1) pip install gurobipy\n"
                "  2) 申请 license（学术免费）: https://www.gurobi.com/downloads/\n"
                "  3) 用 grbgetkey <key> 激活，或设置环境变量 GRB_LICENSE_FILE\n"
                "只想跑启发式的话用 ALNSSolver，不需要 Gurobi。"
            )
        print("构建MILP模型...")
        
        # 创建模型
        self.model = gp.Model("EV_UAV_Routing")
        
        # 设置求解器参数
        if self.config.get('solver'):
            if self.config['solver'].get('time_limit'):
                self.model.Params.TimeLimit = self.config['solver']['time_limit']
            if self.config['solver'].get('mip_gap'):
                self.model.Params.MIPGap = self.config['solver']['mip_gap']
            if self.config['solver'].get('threads', -1) > 0:
                self.model.Params.Threads = self.config['solver']['threads']
            if not self.config['solver'].get('log_output', True):
                self.model.Params.OutputFlag = 0
        
        # 提取数据
        nodes = self.dataset['nodes']
        customers = self.dataset['customers']
        n_nodes = len(nodes)
        n_customers = len(customers)
        
        depot = 0
        stations = [i for i in range(1, n_nodes) if nodes.iloc[i]['type'] == 'station']
        charging_stations = [i for i in range(1, n_nodes) if nodes.iloc[i]['type'] == 'charging_station']
        
        dist_ev = self.dataset['distances']['ev']
        dist_uav_to_customer = self._compute_uav_customer_distances()
        
        print(f"节点数: {n_nodes}, 客户数: {n_customers}")
        
        # 决策变量
        self._create_variables(n_nodes, n_customers, stations)
        
        # 约束条件
        self._add_constraints(n_nodes, n_customers, depot, stations, charging_stations)
        
        # 目标函数
        self._set_objective(n_nodes, n_customers, stations)
        
        print("MILP模型构建完成")
    
    def _create_variables(self, n_nodes: int, n_customers: int, stations: List[int]):
        """
        创建决策变量
        
        Args:
            n_nodes: 节点数量
            n_customers: 客户数量
            stations: 站点列表
        """
        # EV路径变量 x[i,j]: EV是否从节点i行驶到节点j
        self.x = self.model.addVars(
            n_nodes, n_nodes,
            vtype=GRB.BINARY,
            name="x"
        )
        
        # UAV分配变量 y[i,c]: 客户c是否由UAV在节点i处服务
        self.y = self.model.addVars(
            stations, range(n_customers),
            vtype=GRB.BINARY,
            name="y"
        )
        
        # EV直接服务变量 z[c]: 客户c是否由EV直接服务
        self.z = self.model.addVars(
            n_customers,
            vtype=GRB.BINARY,
            name="z"
        )
        
        # 时间变量 T[i]: EV到达节点i的时间
        self.T = self.model.addVars(
            n_nodes,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="T"
        )
        
        # 客户服务时间 Tc[c]: 客户c被服务的时间
        self.Tc = self.model.addVars(
            n_customers,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="Tc"
        )
        
        # UAV能耗变量（用于每个站点的sortie）
        self.E_uav = self.model.addVars(
            stations, range(n_customers),
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="E_uav"
        )
    
    def _add_constraints(
        self,
        n_nodes: int,
        n_customers: int,
        depot: int,
        stations: List[int],
        charging_stations: List[int]
    ):
        """
        添加约束条件
        
        Args:
            n_nodes: 节点数量
            n_customers: 客户数量
            depot: 仓库节点
            stations: 站点列表
            charging_stations: 充电站列表
        """
        customers_df = self.dataset['customers']
        dist_ev = self.dataset['distances']['ev']
        energy_ev = self.dataset['energy']['ev']
        time_ev = self.dataset['times']['ev']
        
        # 1. 客户覆盖约束：每个客户必须被服务一次
        for c in range(n_customers):
            self.model.addConstr(
                self.z[c] + gp.quicksum(self.y[i, c] for i in stations) == 1,
                name=f"customer_coverage_{c}"
            )
        
        # 2. EV流平衡约束
        # 仓库：出发
        self.model.addConstr(
            gp.quicksum(self.x[depot, j] for j in range(1, n_nodes)) == 1,
            name="depot_outflow"
        )
        # 仓库：返回
        self.model.addConstr(
            gp.quicksum(self.x[i, depot] for i in range(1, n_nodes)) == 1,
            name="depot_inflow"
        )
        
        # 其他节点：流平衡
        for i in range(1, n_nodes):
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for j in range(n_nodes) if j != i) ==
                gp.quicksum(self.x[j, i] for j in range(n_nodes) if j != i),
                name=f"flow_balance_{i}"
            )
        
        # 禁止自环
        for i in range(n_nodes):
            self.model.addConstr(self.x[i, i] == 0, name=f"no_self_loop_{i}")
        
        # 3. UAV发射条件：只有EV访问的站点才能发射UAV
        for i in stations:
            for c in range(n_customers):
                self.model.addConstr(
                    self.y[i, c] <= gp.quicksum(self.x[j, i] for j in range(n_nodes) if j != i),
                    name=f"uav_launch_{i}_{c}"
                )
        
        # 4. UAV能耗约束（简化版：单客户sortie）
        uav_battery = self.config['uav']['battery_capacity']
        uav_consumption = self.config['uav']['energy_consumption']
        
        for i in stations:
            node_x = self.dataset['nodes'].iloc[i]['x']
            node_y = self.dataset['nodes'].iloc[i]['y']
            
            for c in range(n_customers):
                customer_x = customers_df.iloc[c]['x']
                customer_y = customers_df.iloc[c]['y']
                
                # 往返距离
                round_trip_dist = 2 * np.sqrt((node_x - customer_x)**2 + (node_y - customer_y)**2)
                energy = round_trip_dist * uav_consumption
                
                # 如果能耗超过电池容量，则不能分配
                if energy > uav_battery:
                    self.model.addConstr(self.y[i, c] == 0, name=f"uav_battery_{i}_{c}")
                else:
                    self.model.addConstr(
                        self.E_uav[i, c] >= energy * self.y[i, c],
                        name=f"uav_energy_{i}_{c}"
                    )
        
        # 5. EV电池容量约束（简化）
        ev_battery = self.config['ev']['battery_capacity']
        total_ev_energy = gp.quicksum(
            energy_ev[i, j] * self.x[i, j]
            for i in range(n_nodes)
            for j in range(n_nodes)
            if i != j
        )
        self.model.addConstr(
            total_ev_energy <= ev_battery,
            name="ev_battery"
        )
        
        # 6. 时间约束（MTZ formulation for subtour elimination）
        M = 10000  # 大M
        
        for i in range(n_nodes):
            for j in range(1, n_nodes):
                if i != j:
                    self.model.addConstr(
                        self.T[j] >= self.T[i] + time_ev[i, j] - M * (1 - self.x[i, j]),
                        name=f"time_consistency_{i}_{j}"
                    )
        
        # 7. 客户时间窗约束
        for c in range(n_customers):
            tw_start = customers_df.iloc[c]['time_window_start']
            tw_end = customers_df.iloc[c]['time_window_end']
            
            self.model.addConstr(
                self.Tc[c] >= tw_start,
                name=f"time_window_start_{c}"
            )
            self.model.addConstr(
                self.Tc[c] <= tw_end,
                name=f"time_window_end_{c}"
            )
            
            # 服务时间与EV/UAV的关系（简化）
            # 如果由EV服务，服务时间约等于EV到达时间
            # 如果由UAV服务，服务时间约等于UAV到达时间
            # 这里简化处理
        
        # 8. UAV载重约束（简化：单客户sortie）
        uav_capacity = self.config['uav']['load_capacity']
        for i in stations:
            for c in range(n_customers):
                demand = customers_df.iloc[c]['demand']
                if demand > uav_capacity:
                    # 超重客户不能由UAV服务
                    self.model.addConstr(self.y[i, c] == 0, name=f"uav_capacity_{i}_{c}")
    
    def _set_objective(self, n_nodes: int, n_customers: int, stations: List[int]):
        """
        设置目标函数
        
        Args:
            n_nodes: 节点数量
            n_customers: 客户数量
            stations: 站点列表
        """
        dist_ev = self.dataset['distances']['ev']
        energy_ev = self.dataset['energy']['ev']
        
        # 成本权重
        energy_weight = self.config['costs']['energy_weight']
        time_weight = self.config['costs']['time_weight']
        carbon_weight = self.config['costs']['carbon_weight']
        carbon_factor = self.config['costs']['carbon_factor']
        
        # EV固定成本
        ev_fixed_cost = self.config['ev']['fixed_cost']
        
        # UAV固定成本（每次起飞）
        uav_fixed_cost = self.config['uav']['fixed_cost']
        total_uav_launches = gp.quicksum(
            self.y[i, c] for i in stations for c in range(n_customers)
        )
        
        # EV能耗成本
        ev_energy = gp.quicksum(
            energy_ev[i, j] * self.x[i, j]
            for i in range(n_nodes)
            for j in range(n_nodes)
            if i != j
        )
        
        # UAV能耗成本
        uav_energy = gp.quicksum(
            self.E_uav[i, c]
            for i in stations
            for c in range(n_customers)
        )
        
        # 总能耗成本
        energy_cost = (ev_energy + uav_energy) * energy_weight
        
        # 碳排放成本
        carbon_cost = (ev_energy + uav_energy) * carbon_factor * carbon_weight
        
        # 时间成本（简化：使用总距离作为代理）
        total_distance = gp.quicksum(
            dist_ev[i, j] * self.x[i, j]
            for i in range(n_nodes)
            for j in range(n_nodes)
            if i != j
        )
        time_cost = total_distance * time_weight
        
        # 总目标函数
        objective = (
            ev_fixed_cost +
            uav_fixed_cost * total_uav_launches +
            energy_cost +
            carbon_cost +
            time_cost
        )
        
        self.model.setObjective(objective, GRB.MINIMIZE)
    
    def solve(self) -> Solution:
        """
        求解模型
        
        Returns:
            Solution对象
        """
        if self.model is None:
            self.build_model()
        
        print("\n开始求解MILP模型...")
        start_time = time.time()
        
        self.model.optimize()
        
        self.solve_time = time.time() - start_time
        
        if self.model.Status == GRB.OPTIMAL:
            print(f"\n找到最优解！")
            self.obj_value = self.model.ObjVal
            self.gap = 0.0
        elif self.model.Status == GRB.TIME_LIMIT:
            print(f"\n达到时间限制，返回当前最优解")
            self.obj_value = self.model.ObjVal
            self.gap = self.model.MIPGap
        else:
            print(f"\n求解失败，状态码: {self.model.Status}")
            return None
        
        print(f"目标函数值: {self.obj_value:.2f}")
        print(f"求解时间: {self.solve_time:.2f}秒")
        if self.gap > 0:
            print(f"MIP Gap: {self.gap*100:.2f}%")
        
        # 提取解
        self.solution = self._extract_solution()
        
        return self.solution
    
    def _extract_solution(self) -> Solution:
        """
        从Gurobi模型中提取解
        
        Returns:
            Solution对象
        """
        solution = Solution(self.dataset, self.config)
        
        nodes = self.dataset['nodes']
        customers = self.dataset['customers']
        n_nodes = len(nodes)
        n_customers = len(customers)
        
        depot = 0
        stations = [i for i in range(1, n_nodes) if nodes.iloc[i]['type'] == 'station']
        
        # 提取EV路径
        ev_route = [depot]
        current = depot
        visited = {depot}
        
        while len(visited) < n_nodes:
            for j in range(n_nodes):
                if j not in visited and self.x[current, j].X > 0.5:
                    ev_route.append(j)
                    visited.add(j)
                    current = j
                    break
            else:
                # 返回仓库
                if current != depot:
                    for j in range(n_nodes):
                        if j == depot and self.x[current, j].X > 0.5:
                            ev_route.append(depot)
                            break
                break
        
        solution.ev_route.nodes = ev_route
        
        # 提取UAV分配
        uav_sorties = {}
        for i in stations:
            sorties = []
            for c in range(n_customers):
                if self.y[i, c].X > 0.5:
                    sortie = UAVSortie(
                        uav_id=0,
                        launch_node=i,
                        customers=[c]
                    )
                    sorties.append(sortie)
                    solution.served_customers.add(c)
            
            if sorties:
                uav_sorties[i] = sorties
        
        solution.ev_route.uav_sorties = uav_sorties
        
        # 提取EV直接服务的客户
        direct_customers = []
        for c in range(n_customers):
            if self.z[c].X > 0.5:
                direct_customers.append(c)
                solution.served_customers.add(c)
        
        solution.ev_route.direct_customers = direct_customers
        
        # 更新未服务客户集合
        solution.unserved_customers = set(range(n_customers)) - solution.served_customers
        
        # 评估解
        solution.evaluate()
        
        return solution
    
    def _compute_uav_customer_distances(self) -> np.ndarray:
        """
        计算UAV到客户的距离矩阵
        
        Returns:
            距离矩阵 [n_nodes x n_customers]
        """
        nodes = self.dataset['nodes']
        customers = self.dataset['customers']
        
        n_nodes = len(nodes)
        n_customers = len(customers)
        
        distances = np.zeros((n_nodes, n_customers))
        
        for i in range(n_nodes):
            node_x, node_y = nodes.iloc[i][['x', 'y']]
            for c in range(n_customers):
                customer_x, customer_y = customers.iloc[c][['x', 'y']]
                distances[i, c] = np.sqrt((node_x - customer_x)**2 + (node_y - customer_y)**2)
        
        return distances
    
    def get_statistics(self) -> Dict:
        """
        获取求解统计信息
        
        Returns:
            统计信息字典
        """
        if self.model is None:
            return {}
        
        return {
            'solve_time': self.solve_time,
            'objective_value': self.obj_value,
            'mip_gap': self.gap,
            'num_variables': self.model.NumVars,
            'num_constraints': self.model.NumConstrs,
            'num_nonzeros': self.model.NumNZs
        }


def main():
    """测试MILP模型"""
    from data_generator import DataGenerator
    
    print("生成测试数据...")
    gen = DataGenerator('config/params_small.yaml')
    dataset = gen.generate_dataset()
    
    print("\n构建并求解MILP模型...")
    milp = MILPModel(dataset, dataset['config'])
    solution = milp.solve()
    
    if solution:
        print("\n解的详细信息:")
        print(solution)
        print("\n性能指标:")
        metrics = solution.evaluate()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print("\n求解统计:")
        stats = milp.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
