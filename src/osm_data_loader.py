"""
OpenStreetMap数据加载器
从真实地图提取路网、POI、客户位置等数据
"""

from __future__ import annotations

try:
    import osmnx as ox
    import networkx as nx
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    OSM_AVAILABLE = True
except ImportError:  # pragma: no cover - osmnx/geopandas 是可选的重依赖
    ox = None
    nx = None
    gpd = None
    Point = None
    Polygon = None
    OSM_AVAILABLE = False

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


OSM_INSTALL_HINT = (
    "真实路网功能需要可选依赖：pip install osmnx geopandas networkx shapely\n"
    "（只跑合成算例/ALNS 不需要它们）"
)


def _require_osm():
    """在真正调用 OSMnx 之前检查可选依赖是否就绪。"""
    if not OSM_AVAILABLE:
        raise RuntimeError(OSM_INSTALL_HINT)


class OSMDataLoader:
    """从OpenStreetMap提取真实路网数据"""
    
    def __init__(self, location: str, network_type: str = 'drive'):
        """
        初始化OSM数据加载器
        
        Args:
            location: 地点名称，如"Chaoyang District, Beijing, China"
                     或bbox坐标 (north, south, east, west)
            network_type: 'drive' (汽车), 'walk' (步行), 'bike' (自行车), 'all'
        """
        self.location = location
        self.network_type = network_type
        self.graph = None
        self.graph_unprojected = None  # 保存未投影的图（用于地图可视化）
        self.bounds = None
        
    def extract_road_network(self, simplify: bool = True) -> nx.MultiDiGraph:
        """
        提取道路网络图
        
        Args:
            simplify: 是否简化网络（合并直线段）
            
        Returns:
            道路网络图
        """
        _require_osm()
        print(f"  正在从OSM下载道路网络: {self.location}")
        
        try:
            # 下载道路网络
            self.graph = ox.graph_from_place(
                self.location, 
                network_type=self.network_type,
                simplify=simplify
            )
            
            # 投影到UTM坐标系（便于距离计算）
            self.graph = ox.project_graph(self.graph)
            
            # 获取边界
            nodes = ox.graph_to_gdfs(self.graph, edges=False)
            self.bounds = nodes.total_bounds  # (minx, miny, maxx, maxy)
            
            print(f"  ✓ 道路网络: {len(self.graph.nodes)} 节点, {len(self.graph.edges)} 边")
            print(f"  ✓ 边界范围: {self.bounds}")
            
            return self.graph
            
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            raise
    
    def extract_bbox_network(self, 
                            north: float, south: float, 
                            east: float, west: float) -> nx.MultiDiGraph:
        """
        通过经纬度边界框提取道路网络
        
        Args:
            north, south, east, west: 边界框坐标
        """
        print(f"  正在从OSM下载道路网络 (bbox)")
        
        graph_orig = ox.graph_from_bbox(
            north=north, south=south, east=east, west=west,
            network_type=self.network_type,
            simplify=True
        )
        
        # 保存未投影的图（用于地图可视化）
        self.graph_unprojected = graph_orig.copy()
        
        # 投影图（用于距离计算）
        self.graph = ox.project_graph(graph_orig)
        nodes = ox.graph_to_gdfs(self.graph, edges=False)
        self.bounds = nodes.total_bounds
        
        print(f"  ✓ 道路网络: {len(self.graph.nodes)} 节点, {len(self.graph.edges)} 边")
        
        return self.graph
    
    def extract_pois(self, 
                    poi_tags: Dict[str, List[str]]) -> gpd.GeoDataFrame:
        """
        提取兴趣点(POI)作为候选站点
        
        Args:
            poi_tags: 例如 {'amenity': ['parking', 'fuel'],
                           'shop': ['convenience', 'supermarket']}
        
        Returns:
            POI的GeoDataFrame
        """
        print(f"  正在提取POI...")
        
        try:
            pois = ox.geometries_from_place(
                self.location, 
                tags=poi_tags
            )
            
            # 只保留点状POI
            pois = pois[pois.geometry.type == 'Point']
            
            print(f"  ✓ 找到 {len(pois)} 个POI")
            return pois
            
        except Exception as e:
            print(f"  ⚠ POI提取失败: {e}")
            return gpd.GeoDataFrame()
    
    def sample_customer_locations(self, 
                                  num_customers: int,
                                  strategy: str = 'random') -> pd.DataFrame:
        """
        采样客户位置
        
        Args:
            num_customers: 客户数量
            strategy: 'random' (随机), 'residential' (居民区), 'poi' (基于POI)
        
        Returns:
            客户位置DataFrame
        """
        _require_osm()
        print(f"  正在采样 {num_customers} 个客户位置 (策略: {strategy})...")
        
        if strategy == 'residential':
            # 尝试提取居民建筑
            try:
                buildings = ox.geometries_from_place(
                    self.location,
                    tags={'building': ['residential', 'apartments', 'house']}
                )
                
                # 过滤出点或多边形
                if len(buildings) > 0:
                    # 取中心点
                    centroids = buildings.geometry.centroid
                    sampled_indices = np.random.choice(
                        len(centroids), 
                        size=min(num_customers, len(centroids)),
                        replace=False
                    )
                    sampled_points = centroids.iloc[sampled_indices]
                else:
                    raise ValueError("没有找到居民建筑")
                    
            except Exception as e:
                print(f"  ⚠ 居民区提取失败，使用随机采样: {e}")
                strategy = 'random'
        
        if strategy == 'random' or strategy == 'poi':
            # 在道路网络节点附近随机采样
            if self.graph is None:
                raise ValueError("请先调用 extract_road_network()")
            
            # 获取所有节点坐标
            nodes_gdf = ox.graph_to_gdfs(self.graph, edges=False)
            
            # 随机选择节点
            sampled_nodes = nodes_gdf.sample(n=num_customers, replace=True)
            sampled_points = sampled_nodes.geometry
        
        # 转换为DataFrame
        customers = pd.DataFrame({
            'id': range(num_customers),
            'x': [p.x for p in sampled_points],
            'y': [p.y for p in sampled_points],
            'lon': [p.x for p in sampled_points],  # 投影后的坐标
            'lat': [p.y for p in sampled_points],
            'demand': np.random.uniform(1, 5, num_customers),
            'time_window_start': np.random.uniform(0, 240, num_customers),  # 0-4小时
        })
        
        customers['time_window_end'] = customers['time_window_start'] + \
                                       np.random.uniform(60, 180, num_customers)  # +1-3小时
        
        print(f"  ✓ 已采样 {num_customers} 个客户位置")
        return customers
    
    def select_station_locations(self, 
                                 num_stations: int,
                                 strategy: str = 'poi') -> pd.DataFrame:
        """
        选择站点位置
        
        Args:
            num_stations: 站点数量
            strategy: 'poi' (基于POI), 'grid' (网格), 'random' (随机)
        
        Returns:
            站点位置DataFrame
        """
        print(f"  正在选择 {num_stations} 个站点位置 (策略: {strategy})...")
        
        if strategy == 'poi':
            # 基于POI（停车场、加油站、商超）
            poi_tags = {
                'amenity': ['parking', 'fuel'],
                'shop': ['convenience', 'supermarket', 'mall']
            }
            
            pois = self.extract_pois(poi_tags)
            
            if len(pois) >= num_stations:
                sampled_pois = pois.sample(n=num_stations)
                stations = pd.DataFrame({
                    'id': range(num_stations),
                    'x': [p.x for p in sampled_pois.geometry],
                    'y': [p.y for p in sampled_pois.geometry],
                    'type': 'station',
                    'name': [f"站点{i+1}" for i in range(num_stations)]
                })
            else:
                print(f"  ⚠ POI不足，使用随机策略")
                strategy = 'random'
        
        if strategy == 'random' or strategy == 'grid':
            # 在道路节点中随机选择
            nodes_gdf = ox.graph_to_gdfs(self.graph, edges=False)
            sampled_nodes = nodes_gdf.sample(n=num_stations)
            
            stations = pd.DataFrame({
                'id': range(num_stations),
                'x': [p.x for p in sampled_nodes.geometry],
                'y': [p.y for p in sampled_nodes.geometry],
                'type': 'station',
                'name': [f"站点{i+1}" for i in range(num_stations)]
            })
        
        print(f"  ✓ 已选择 {num_stations} 个站点")
        return stations
    
    def calculate_road_distance(self, 
                               orig: Tuple[float, float],
                               dest: Tuple[float, float]) -> float:
        """
        计算基于道路网络的实际行驶距离
        
        Args:
            orig: 起点坐标 (x, y)
            dest: 终点坐标 (x, y)
        
        Returns:
            距离（公里）
        """
        _require_osm()
        if self.graph is None:
            raise ValueError("请先调用 extract_road_network()")
        
        try:
            # 找到最近的网络节点
            orig_node = ox.nearest_nodes(self.graph, orig[0], orig[1])
            dest_node = ox.nearest_nodes(self.graph, dest[0], dest[1])
            
            # 计算最短路径
            route = nx.shortest_path(
                self.graph, 
                orig_node, 
                dest_node, 
                weight='length'
            )
            
            # 路径总长度（米转公里）
            length = sum(
                self.graph[u][v][0]['length'] 
                for u, v in zip(route[:-1], route[1:])
            ) / 1000.0
            
            return length
            
        except nx.NetworkXNoPath:
            # 如果没有路径，返回欧氏距离
            return np.linalg.norm(np.array(orig) - np.array(dest)) / 1000.0
        except Exception as e:
            # 其他错误也返回欧氏距离
            return np.linalg.norm(np.array(orig) - np.array(dest)) / 1000.0
    
    def calculate_distance_matrix(self, 
                                 locations: pd.DataFrame,
                                 use_road_network: bool = True) -> np.ndarray:
        """
        计算位置间的距离矩阵
        
        Args:
            locations: 包含x, y列的DataFrame
            use_road_network: True=道路距离, False=欧氏距离
        
        Returns:
            距离矩阵 (n × n)
        """
        n = len(locations)
        dist_matrix = np.zeros((n, n))
        
        print(f"  正在计算 {n}×{n} 距离矩阵 ({'道路距离' if use_road_network else '欧氏距离'})...")
        
        coords = locations[['x', 'y']].values
        
        if use_road_network:
            # 使用道路网络距离（较慢）
            total = n * (n - 1) // 2
            count = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = self.calculate_road_distance(
                        (coords[i, 0], coords[i, 1]),
                        (coords[j, 0], coords[j, 1])
                    )
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"    进度: {count}/{total} ({count/total*100:.1f}%)")
        else:
            # 使用欧氏距离（快速）
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(coords[i] - coords[j]) / 1000.0
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
        
        print(f"  ✓ 距离矩阵计算完成")
        return dist_matrix
    
    def identify_no_fly_zones(self) -> List[Dict]:
        """
        识别禁飞区（机场、军事区等）
        
        Returns:
            禁飞区列表，每个元素包含geometry和bounds
        """
        print(f"  正在识别禁飞区...")
        
        no_fly_tags = {
            'aeroway': ['aerodrome', 'helipad', 'heliport'],
            'military': True,
            'landuse': ['military']
        }
        
        try:
            zones = ox.geometries_from_place(
                self.location,
                tags=no_fly_tags
            )
            
            no_fly_zones = []
            for idx, zone in zones.iterrows():
                if zone.geometry.type in ['Polygon', 'MultiPolygon']:
                    no_fly_zones.append({
                        'geometry': zone.geometry,
                        'bounds': zone.geometry.bounds,  # (minx, miny, maxx, maxy)
                        'type': zone.get('aeroway', zone.get('military', zone.get('landuse', 'unknown')))
                    })
            
            print(f"  ✓ 找到 {len(no_fly_zones)} 个禁飞区")
            return no_fly_zones
            
        except Exception as e:
            print(f"  ⚠ 禁飞区识别失败: {e}")
            return []
    
    def save_to_cache(self, filepath: str):
        """保存图到缓存文件"""
        if self.graph:
            ox.save_graphml(self.graph, filepath)
            print(f"  ✓ 道路网络已缓存到: {filepath}")
    
    def load_from_cache(self, filepath: str):
        """从缓存文件加载图"""
        self.graph = ox.load_graphml(filepath)
        self.graph = ox.project_graph(self.graph)
        print(f"  ✓ 从缓存加载道路网络: {filepath}")
        return self.graph


# 便捷函数
def quick_extract_city_data(city: str, 
                            num_customers: int = 50,
                            num_stations: int = 3) -> Dict:
    """
    快速提取城市数据（简化接口）
    
    Args:
        city: 城市名称，如 "Chaoyang District, Beijing, China"
        num_customers: 客户数量
        num_stations: 站点数量
    
    Returns:
        包含nodes和customers的数据字典
    """
    loader = OSMDataLoader(city)
    
    # 提取道路网络
    loader.extract_road_network()
    
    # 采样客户
    customers = loader.sample_customer_locations(num_customers, strategy='random')
    
    # 选择站点
    stations = loader.select_station_locations(num_stations, strategy='random')
    
    # 创建depot（仓库，选第一个站点）
    depot = pd.DataFrame({
        'id': [0],
        'x': [stations.iloc[0]['x']],
        'y': [stations.iloc[0]['y']],
        'type': ['depot'],
        'name': ['仓库']
    })
    
    # 合并所有节点
    nodes = pd.concat([depot, stations], ignore_index=True)
    nodes['id'] = range(len(nodes))
    
    return {
        'loader': loader,
        'nodes': nodes,
        'customers': customers,
        'graph': loader.graph
    }
