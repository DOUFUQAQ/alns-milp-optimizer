# 模型与数据表示

本文档说明仓库里**真实实现**的问题模型：数据怎么组织、解怎么表示、成本怎么算、
约束在哪里检查、MILP 建了哪些变量。凡是代码与旧的文字说明不一致的地方，以代码为准，
并在此显式标注「实现现状」。

---

## 1. 问题定义

一个**两层**末端配送问题（two-echelon last-mile delivery）：

* **第一层 —— EV（电动车）**：从仓库出发，沿城市道路依次访问若干**停靠站点**（station），
  最后返回仓库。EV 也可以顺路直接给客户送货（EV 直送）。
* **第二层 —— UAV（无人机）**：从某个 EV 停靠站点起飞，服务 1..n 个客户后飞回**同一个**站点。
  一次起降称为一个 **sortie**。
* 每个客户必须被服务**恰好一次**：要么由 UAV 在某个站点服务，要么由 EV 直送。

所以它既不是纯 VRP，也不是纯选址：**站点的访问顺序、UAV 从哪个站点起飞、
哪些客户交给 UAV、EV 是否直送**都耦合在同一个解里。

> 代码不做站点选址优化。`data_generator` 按配置生成站点坐标，
> `osm_data_loader` 从 OSM 的 POI/路网节点里采样站点，之后位置固定。

---

## 2. 数据表示

`DataGenerator.generate_dataset()` 和 `DataGenerator.load_dataset()` 返回同一个字典结构：

```python
dataset = {
    'nodes':     pd.DataFrame,   # node_id, type(depot/station/charging_station), x, y, name
    'customers': pd.DataFrame,   # customer_id, x, y, demand, service_time,
                                 # time_window_start, time_window_end, name
    'distances': {'ev': ndarray[n_nodes, n_nodes], 'uav': ndarray},   # km
    'times':     {'ev': ndarray, 'uav': ndarray},                     # 分钟
    'energy':    {'ev': ndarray, 'uav': ndarray},                     # kWh
    'config':    dict,           # 完整参数配置
}
```

矩阵的换算关系（`data_generator.py`）：

```
times[mode]  = distances[mode] / config[mode]['speed'] * 60
energy[mode] = distances[mode] * config[mode]['energy_consumption']
```

**单位约定（很重要）**：

| 来源 | 坐标单位 | 距离矩阵 |
| --- | --- | --- |
| `data_generator.py`（合成算例） | km | km |
| `osm_data_loader.py`（真实路网，UTM 投影） | m | km（已除 1000） |

而 `solution.py` 计算 UAV sortie 距离时对**坐标差值**除以 1000（按米处理）。
这在 OSM 数据下是对的，在合成算例下会把 UAV 距离缩小 1000 倍 —— 详见
README 的「已知局限」。

---

## 3. 解的表示

```python
@dataclass
class UAVSortie:
    uav_id: int
    launch_node: int          # 起降的 EV 站点
    customers: List[int]      # 这个架次服务的客户（按访问顺序）
    launch_time: float = 0.0
    return_time: float = 0.0  # 由 evaluate() 写入
    distance: float = 0.0     # km，由 evaluate() 写入
    energy: float = 0.0       # kWh，由 evaluate() 写入
    is_feasible: bool = True

@dataclass
class EVRoute:
    nodes: List[int]                                  # 访问序列，首尾都是 0（仓库）
    arrival_times / departure_times / battery_levels   # 预留字段，当前未填充
    uav_sorties: Dict[int, List[UAVSortie]]           # 站点 -> 架次列表
    direct_customers: List[int]                       # EV 直送的客户

class Solution:
    ev_route: EVRoute
    served_customers: Set[int]
    unserved_customers: Set[int]
    # 指标：total_cost / ev_cost / uav_cost / energy_cost / time_cost /
    #       carbon_cost / delay_penalty / total_distance / total_time / total_energy
```

`served_customers` / `unserved_customers` 与 `uav_sorties` / `direct_customers`
是**两份**信息，必须同步维护；`tests/helpers.integrity_problems()` 就是在检查这件事
（互斥、并集完整、无重复配送、两份信息一致）。

---

## 4. 目标函数（`Solution._calculate_total_cost`）

```
total_energy = Σ EV 路径能耗 + Σ UAV sortie 能耗
total_time   = EV 路径行驶时间                      # 注意：不含 UAV 飞行时间
energy_cost  = total_energy × costs.energy_weight
time_cost    = total_time   × costs.time_weight
carbon_cost  = total_energy × costs.carbon_factor × costs.carbon_weight
total_cost   = ev.fixed_cost                      # 常数，每辆 EV 一份
             + n_sorties × uav.fixed_cost          # 每个架次一份
             + energy_cost + time_cost + carbon_cost
             + delay_penalty                        # 当前恒为 0
```

`objective()` 在不可行时直接返回 `inf`，这是 ALNS 的排序依据。

**实现现状**：`_evaluate_direct_delivery()` 是空函数，EV 直送的客户不贡献任何
距离/时间/能耗（只影响"客户是否全覆盖"）。`delay_penalty` 没有任何写入点，
时间窗在评估阶段不参与计算。

---

## 5. 可行性检查（`Solution.evaluate`）

按顺序检查以下条件，任一不满足就把 `is_feasible` 置 `False`，并在
`infeasibility_reasons` 里追加一条说明：

1. **客户全覆盖**：`len(served_customers) == len(customers)`，否则立即返回（其余指标不计算）
2. **EV 电池**：整条 EV 路径的累计能耗 > `ev.battery_capacity`
3. **UAV 电池**：单个 sortie 的能耗 > `uav.battery_capacity`
4. **UAV 载重**：单个 sortie 服务客户的需求之和 > `uav.load_capacity`
5. **空 sortie**：`sortie.customers` 为空

**没有检查**：时间窗、EV 载重、充电站使用、无人机数量上限（`num_drones` 只在
生成初始解时限制每个站点的架次数，评估阶段不检查）。

---

## 6. MILP 模型（`src/milp_model.py`）

求解器：**Gurobi**（`gurobipy`）。变量与论文记号一致：

| 变量 | 含义 | 类型 |
| --- | --- | --- |
| `x[i,j]` | EV 是否从 i 行驶到 j | binary，`n_nodes × n_nodes` |
| `y[i,c]` | 客户 c 是否由站点 i 的 UAV 服务 | binary，`stations × n_customers` |
| `z[c]` | 客户 c 是否由 EV 直送 | binary |
| `T[i]` | EV 到达节点 i 的时间 | continuous ≥ 0 |
| `Tc[c]` | 客户 c 的被服务时间 | continuous ≥ 0 |
| `E_uav[i,c]` | 站点 i 服务客户 c 的 UAV 能耗 | continuous ≥ 0 |

约束：

```
(1)  客户覆盖   z[c] + Σ_i y[i,c] = 1                     ∀c
(2)  仓库出度   Σ_j x[0,j] = 1
(3)  仓库入度   Σ_i x[i,0] = 1
(4)  流平衡     Σ_j x[i,j] = Σ_j x[j,i]                    ∀i≠0
(5)  禁止自环   x[i,i] = 0
(6)  发射条件   y[i,c] ≤ Σ_j x[j,i]                        ∀i∈stations, c
(7)  UAV 能耗   E_uav[i,c] ≥ energy(i,c)·y[i,c]；超电池则 y[i,c]=0
(8)  EV 电池    Σ energy_ev[i,j]·x[i,j] ≤ ev.battery_capacity
(9)  MTZ 时间   T[j] ≥ T[i] + time_ev[i,j] − M(1 − x[i,j])
(10) 时间窗     tw_start[c] ≤ Tc[c] ≤ tw_end[c]
(11) UAV 载重   demand[c] > capacity ⇒ y[i,c] = 0
```

目标：

```
min  ev.fixed_cost + uav.fixed_cost·Σ y[i,c]
   + energy_weight·(EV 能耗 + Σ E_uav)
   + carbon_factor·carbon_weight·(同上)
   + time_weight·Σ dist_ev[i,j]·x[i,j]
```

**实现现状（建模层面的弱化，写清楚避免误读）**：

* `Tc[c]` 只被自己的时间窗上下界夹住，**没有**和 `T[i]`、`x`、`y` 建立任何联系，
  所以 (10) 在模型里是空约束。
* MILP 的时间成本用**总距离**乘 `time_weight`（量纲与 `Solution` 里的分钟不一致）。
* `E_uav` 只有下界 `≥ energy·y`，而目标是**最小化** `Σ E_uav`，求解器会把它压到下界；
  它的距离口径（往返直线距离）与 `Solution.evaluate()` 里的算法是两套实现。
* MILP 里每个客户单独构成一个 UAV 架次（`y[i,c]`），不支持"一个架次串多个客户"，
  这点与 ALNS 的解空间不同。

因此 MILP 的结果**不能**直接当作 ALNS 的严格下界使用，只适合做小规模的最优性参照。

---

## 7. ALNS 相关参数（`config/params_large.yaml` 的 `alns` 段）

| 键 | 代码里的用途 |
| --- | --- |
| `max_iterations` | 默认迭代次数（`solve(max_iterations=None)` 时使用） |
| `segment_size` | 每多少轮更新一次算子权重 |
| `weights.start` | `[new_best, better, accepted, rejected]` 四档得分 |
| `weights.decay` | 权重指数平滑系数 |
| `temperature.start / cooling_rate` | 模拟退火初温与降温系数（`end` 当前没被读取） |
| `destroy.min_removal / max_removal / removal_percentage` | 每轮移除客户数 = `min(max, max(min, n×pct))` |
| `repair.noise_parameter` | 贪心插入的成本噪声 |
| `repair.k_regret` | k-regret 的 k |
| `output.log_interval` | 每 N 轮打印一次日志 |

> `config/params_small.yaml` / `params_real_world.yaml` 里的 `num_destroy_operators`、
> `num_repair_operators` 代码没有读取：算子数量是写死的（4 个破坏 + 2 个修复）。

---

## 8. 求解流程

```
配置 YAML ──► DataGenerator ──► dataset(节点/客户/矩阵)
                                   │
                     ┌─────────────┴──────────────┐
                     ▼                            ▼
             MILPModel.solve()             ALNSSolver.solve()
             （小规模，Gurobi）            （大规模，启发式）
                     │                            │
                     └──────────► Solution ◄──────┘
                                     │
                              Solution.evaluate()
                                     │
                              Visualizer / JSON / CSV
```
