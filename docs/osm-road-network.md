# 真实路网（OpenStreetMap）数据链路

`src/osm_data_loader.py` 负责把真实城市路网变成优化器能吃的 `dataset`，
`src/map_visualizer.py` 负责把解画回真实地图上，`experiments/run_real_world.py`
把它们串成一条完整实验流程。

**这条链路需要额外依赖**（`osmnx` / `geopandas` / `networkx` / `shapely` /
`folium` / `branca`）并且需要联网。只跑合成算例或单元测试时完全用不到它。

---

## 1. 路网提取

两种入口，最终都得到一个 `networkx.MultiDiGraph`：

```python
loader = OSMDataLoader(location='Chaoyang District, Beijing, China')

# 方式一：按地名（依赖 Overpass 的地理编码，边界可能不是矩形）
loader.extract_road_network(simplify=True)

# 方式二：按经纬度矩形（推荐，快且边界确定）
loader.extract_bbox_network(north=39.925, south=39.900, east=116.475, west=116.445)
```

`extract_bbox_network` 会做两件事：

1. 保留一份**未投影**的图 `graph_unprojected`（WGS84 经纬度，给 Folium 用）；
2. 用 `ox.project_graph` 把图投影到 **UTM**（单位：米），存进 `self.graph`，
   用于最短路径与距离计算。

`experiments/run_real_world.py` 里预置了几个北京区域（单位：度）：

```python
test_areas = {
    'beijing_cbd_small': (39.915, 39.905, 116.465, 116.450),  # 约 1×1.5 km
    'beijing_cbd':       (39.925, 39.900, 116.475, 116.445),  # 约 3×3 km
    'beijing_large':     (39.945, 39.880, 116.495, 116.425),  # 约 7×7 km
    'beijing_xlarge':    (39.980, 39.850, 116.540, 116.380),  # 约 14×16 km
}
```

---

## 2. 客户与站点采样

```python
customers = loader.sample_customer_locations(num_customers=50, strategy='random')
stations  = loader.select_station_locations(num_stations=5, strategy='poi')
```

* **客户**：随机策略从路网节点里**有放回**采样；`residential` 策略会尝试抓
  `building=residential/apartments/house` 的质心，失败则回退到随机。
* **站点**：`poi` 策略用 `amenity=[parking,fuel]` + `shop=[convenience,supermarket,mall]`
  的 POI 作为候选，数量不够时回退到路网节点随机采样。
* 需求与时间窗当前是**写死**的随机分布（`demand ~ U(1,5)` kg、
  服务时刻 `U(0,240)` 分钟），**不读取** `config` 里的 `demand_range` /
  `time_window_width`。这是与合成数据生成器不一致的地方。

---

## 3. 距离矩阵

```python
loader.calculate_road_distance((x1, y1), (x2, y2))          # 单点对，返回 km
matrix = loader.calculate_distance_matrix(locations, use_road_network=True)  # n×n，km
```

实现要点：

* `ox.nearest_nodes` 吸附到最近路网节点，`networkx.shortest_path(weight='length')` 求和；
* 结果**除以 1000 换算成公里**，与 `data_generator` 的矩阵单位一致；
* 找不到路径（`NetworkXNoPath`）或任何异常时，静默回退到欧氏距离 ——
  好处是跑得下去，坏处是失败不会报错，需要自己看日志；
* `use_road_network=False` 时直接算欧氏距离（同样除 1000），适合快速试算。

因为投影后的坐标单位是米，而 `Solution._evaluate_single_sortie` 也按米处理坐标，
**真实路网这条链路的单位是自洽的**；合成算例（坐标是 km）反而不自洽，
详见 README 的「已知局限」。

---

## 4. 禁飞区

```python
zones = loader.identify_no_fly_zones()   # [{'geometry':..., 'bounds':..., 'type':...}]
```

按 `aeroway=[aerodrome,helipad,heliport]`、`military=*`、`landuse=military` 抓多边形。

**实现现状**：这个函数只做**识别与返回**，返回的列表在优化过程中没有被任何
代码读取，也就是说禁飞区目前**只是数据，不是约束**。要让它生效，需要在
`Solution.evaluate()` 或修复算子里对 UAV 直线航段做多边形相交判断。

---

## 5. 缓存

```python
loader.save_to_cache('cache/beijing_cbd.graphml')
loader.load_from_cache('cache/beijing_cbd.graphml')
```

GraphML 文件通常几 MB 到几十 MB，已被 `.gitignore` 排除，仓库里不放。

---

## 6. 地图可视化

`src/map_visualizer.py` 用 Folium 生成交互式 HTML：仓库/站点/客户打点、
EV 路径沿道路画粗线、UAV 航段画虚线、客户需求加热力图。

底图默认用 Folium 自带的 OpenStreetMap 瓦片；要换地图源时用环境变量，
仓库里不写死任何第三方瓦片服务地址：

```bash
export MAP_TILE_URL='https://{s}.tile.example.com/{z}/{x}/{y}.png'
export MAP_TILE_ATTR='Example Maps'
```

---

## 7. 一次真实实验的记录（原始工程，未在本仓库复现）

以下数字来自原始工程的一次运行，**换机器、换 OSM 数据、换参数都会不同**，
仅说明链路曾经跑通：

| 项 | 值 |
| --- | --- |
| 区域 | 北京国贸 CBD，`(39.925, 39.900, 116.475, 116.445)` |
| 路网规模 | 457 节点 / 907 边 |
| 客户数 / 站点数 | 50 / 3 |
| ALNS 迭代 | 1000 |
| 求解时间 | 约 22 s |
| 最优成本 | 206.79 元 |
| 总距离 | 6.93 km（真实道路距离） |
| UAV 架次 | 0 |

UAV 架次为 0 是可以解释的：区域小、客户密集时，EV 绕行成本很低，
而每多一个 UAV 架次就要多付一份固定成本（`uav.fixed_cost`），
在当时的参数下不划算。要看到 UAV 被启用，需要更大的区域、更分散的客户，
或者把 `uav.fixed_cost` 调低。

---

## 8. 版本约束

代码用的是 **osmnx 1.x** 的 API：

* `ox.geometries_from_place`（1.x 的写法，2.0 中已改名/移除）
* `ox.graph_from_bbox(north=..., south=..., east=..., west=...)`（2.0 改成单个 `bbox` 元组）

所以 `requirements.txt` 里把上限锁在 `<2.0`。要迁移到 2.x，需要改这几处调用。
