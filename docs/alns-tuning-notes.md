# ALNS 调参与排障记录

这份笔记记录开发过程中真实踩过的坑：为什么某一版 ALNS "怎么跑成本都不变"、
初始解是怎么修的、温度参数怎么选。**文中的数字来自原始工程在某台机器上的实验记录，
不同环境不会完全复现**，只作为趋势参考。

---

## 1. 第一版：无人机一架都没用上

**现象**：跑大规模算例，`uav_cost == 0`，所有客户都由 EV 直送，
最优解从初始解开始一动不动，实质退化成了单车 VRP。

**原因**：初始解把所有客户直接塞给 EV 直送，`uav_sorties` 是空的：

```python
# 错误的初始解构建
solution.ev_route.direct_customers = list(range(n_customers))
solution.ev_route.uav_sorties = {}
```

破坏算子只能移除"已经在解里的客户"，修复算子算出来的 UAV 插入成本又总比
"EV 直送"贵（EV 直送的边缘成本当时按 0 处理），于是任何一次迭代都不会主动
把客户从 EV 挪到 UAV —— 解空间里根本没有 UAV 解。

**修法**（就是现在 `ALNSSolver._generate_initial_solution` 的做法）：

1. EV 路径固定为 `仓库 → 所有站点 → 仓库`；
2. 对每个站点，按"距离最近 + 满足电池与载重"挑选客户，最多 `uav.num_drones` 个架次；
3. 剩下的客户交给 EV 直送。

修完之后初始解成本会**变高**（因为开始真的用 UAV 了，固定成本 + 能耗都上来了），
但 ALNS 从此有东西可优化。原始记录：初始解 396.91 → 682.39，随后被优化到 536 左右。

---

## 2. 第二版：解在动，但成本几乎不动

**现象**：1000 次迭代，界面上"最优成本"一直显示 231.56，看起来完全没收敛。

**原因不是算法 bug，而是成本结构**。用 `params_large.yaml` 的旧参数算一下占比
（150 客户、5 站点）：

| 成本项 | 量级 | 占比 |
| --- | --- | --- |
| EV 固定成本 | 200 元 | ≈ 86% |
| UAV 固定成本 | 10–30 元 | 4–13% |
| 能耗成本 | 1–3 元 | 0.4–1.3% |
| 时间成本 | 0.5–2 元 | 0.2–0.9% |
| 碳排放成本 | 0.3–1 元 | 0.1–0.4% |

EV 固定成本是**常数**（当前实现固定用 1 辆 EV，`Solution._evaluate_ev_route`
直接写 `ev_cost = ev.fixed_cost`）。也就是说 86% 的目标值对任何解都一样，
ALNS 能影响的只有剩下 14%，而这部分里 UAV 固定成本每个架次才 1 元、
切换 EV/UAV 的收益不到 1 元、调 EV 访问顺序带来的时间差不到 0.5 元。

**结论**：变化量小于 0.01 元的显示精度，看起来就像"没优化"。

**当时采用的缓解措施**（参数层面，已写进 `params_large.yaml` / `params_real_world.yaml`）：

```yaml
ev:  { fixed_cost: 50 }        # 200 → 50，降低常数占比
uav: { fixed_cost: 2.0 }       # 1.0 → 2.0，让"用不用 UAV"更值钱
costs: { energy_weight: 3.0, time_weight: 1.5 }   # 提高可变成本权重
```

真正的根治办法是**让 EV 数量成为决策变量**（多车调度），这样 EV 固定成本不再是
常数，但当前代码没有实现（解结构里只有一条 `EVRoute`）。

---

## 3. 模拟退火在小 Δ 下的行为

```python
delta = new_cost - current_cost
probability = np.exp(-delta / max(self.temperature, 0.01))
accept = random.random() < probability
```

初温 `temperature.start = 10000` 时，`delta < 100` 的劣解接受概率 > 99%，
前几千轮基本是随机游走。配合 `cooling_rate = 0.9975`：

| 迭代数 | 温度（从 10000 起） |
| --- | --- |
| 100 | ≈ 7785 |
| 1000 | ≈ 818 |
| 5000 | ≈ 0.04 |
| 10000 | ≈ 0 |

也就是说 **10000 次迭代的预算下，前 20% 基本在探索，后 20% 才真正在利用**。
迭代次数很少（比如测试里的 20 次）时温度几乎没降下来，别指望它收敛到好解。

`temperature.end` 这个配置项代码里没有读取，降温完全由 `cooling_rate` 控制。

---

## 4. 权重更新的实际公式

代码里是标准 Ropke & Pisinger 式的自适应权重：

```python
score  = {new_best: 10, better: 5, accepted: 2, rejected: 1}   # alns.weights.start
weight = decay * weight + (1 - decay) * (累计得分 / 使用次数)      # decay = 0.95
# 每个 segment_size 轮重置一次分数与计数
```

注意旧文档里写的 `σ = 33 / 9 / 3`、`λ = 0.8` 与代码不符，以代码为准。

---

## 5. 相对原型的性能优化

原型每轮迭代 ~2.3 it/s（0.43 s/iter），主要开销在两处：每次评估都遍历
`DataFrame.iloc` 取坐标、以及用 Python 循环算距离。改成下面三件事后提升到
4–6 it/s（原始记录，机器性能相关）：

1. `Solution.__init__` 里一次性把 `customers[['x','y']]`、`nodes[['x','y']]`、
   `demands` 缓存成 numpy 数组；
2. UAV sortie 距离用 `np.linalg.norm` / `np.diff` 向量化；
3. 插入成本计算中的需求求和直接用缓存的 numpy 数组索引。

---

## 6. 现在怎么判断"算法是否在工作"

排查顺序（`tools/diagnose_solution.py` 就是按这个思路写的）：

1. `uav_cost` 是否为 0 —— 为 0 说明 UAV 完全没用上，先查初始解；
2. `ev_cost / total_cost` 是否 > 0.8 —— 是的话目标值基本被常数主导，考虑调权重；
3. `len(best_history)` 是否等于迭代数，`best_history` 是否单调不增；
4. 最优解与初始解差多少 —— 差 < 0.01 元在显示上就是"没变化"，需要看相对值；
5. 用 `tests/test_alns_solver.py` 里的小算例（20 客户）先确认主循环行为正常，
   再上大规模算例。
