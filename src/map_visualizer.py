"""
基于真实地图的可视化模块
使用Folium在真实地图上展示配送路径
"""

from __future__ import annotations

import os

try:
    import folium
    from folium import plugins
    import osmnx as ox
    import branca.colormap as cm
    MAP_AVAILABLE = True
except ImportError:  # pragma: no cover - folium/osmnx 属于可选依赖
    folium = None
    plugins = None
    ox = None
    cm = None
    MAP_AVAILABLE = False

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


MAP_INSTALL_HINT = (
    "地图可视化需要可选依赖：pip install folium branca osmnx\n"
    "（只跑算法/单元测试不需要它们）"
)


def _require_map_libs():
    """在真正要画地图之前检查可选依赖是否就绪。"""
    if not MAP_AVAILABLE:
        raise RuntimeError(MAP_INSTALL_HINT)


class MapVisualizer:
    """在真实地图上可视化配送方案"""
    
    def __init__(self, osm_loader):
        """
        初始化地图可视化器
        
        Args:
            osm_loader: OSMDataLoader对象
        """
        _require_map_libs()
        self.osm_loader = osm_loader
        self.graph = osm_loader.graph  # 投影图（用于路径计算）
        self.graph_unprojected = osm_loader.graph_unprojected  # 未投影图（用于可视化）
        
        # 保存图的CRS信息
        self.graph_crs = self.graph.graph.get('crs', None)
        
        # 从未投影的图获取地图中心点
        if self.graph_unprojected is not None:
            nodes_gdf = ox.graph_to_gdfs(self.graph_unprojected, edges=False)
            self.center_lat = nodes_gdf.geometry.y.mean()
            self.center_lon = nodes_gdf.geometry.x.mean()
            print(f"  地图中心: ({self.center_lat:.4f}, {self.center_lon:.4f})")
        else:
            # 默认中心点（北京）
            self.center_lat = 39.9
            self.center_lon = 116.4
            print(f"  ⚠ 使用默认中心点")
    
    def create_base_map(self, zoom_start: int = 14) -> folium.Map:
        """
        创建基础地图
        
        底图默认使用 folium 自带的 OpenStreetMap 瓦片；如果在内网或想换成
        别的地图源，设置环境变量 MAP_TILE_URL / MAP_TILE_ATTR 覆盖即可。
        仓库里不写死任何第三方瓦片服务地址。
        
        Args:
            zoom_start: 初始缩放级别
        
        Returns:
            Folium Map对象
        """
        _require_map_libs()
        tile_url = os.environ.get('MAP_TILE_URL', '').strip()
        tile_attr = os.environ.get('MAP_TILE_ATTR', '').strip()
        
        map_kwargs = {
            'location': [self.center_lat, self.center_lon],
            'zoom_start': zoom_start,
            'prefer_canvas': True  # 使用canvas渲染，提升性能
        }
        if tile_url:
            map_kwargs['tiles'] = tile_url
            if tile_attr:
                map_kwargs['attr'] = tile_attr
        
        m = folium.Map(**map_kwargs)
        
        # 添加图层控制
        folium.LayerControl().add_to(m)
        
        return m
    
    def add_road_network(self, m: folium.Map, 
                        line_weight: float = 1.0,
                        line_opacity: float = 0.3):
        """
        在地图上添加道路网络
        
        Args:
            m: Folium Map对象
            line_weight: 线条粗细
            line_opacity: 线条透明度
        """
        if self.graph_unprojected is None:
            print("  ⚠ 未投影图不可用，跳过道路网络显示")
            return
        
        # 使用未投影的图（已经是WGS84坐标）
        edges_gdf = ox.graph_to_gdfs(self.graph_unprojected, nodes=False, edges=True)
        
        # 限制显示的道路数量（提升性能）
        max_edges = 200  # 只显示前200条重要道路
        if len(edges_gdf) > max_edges:
            print(f"  道路网络: 简化显示 ({max_edges}/{len(edges_gdf)} 条边)")
            edges_gdf = edges_gdf.head(max_edges)
        
        # 使用简单的PolyLine（避免GeoJson的NumPy兼容性问题）
        road_layer = folium.FeatureGroup(name='道路网络', show=False)
        
        try:
            for idx, row in edges_gdf.iterrows():
                # 跳过过长的边（优化性能）
                if len(list(row.geometry.coords)) > 50:
                    continue
                coords = [(point[1], point[0]) for point in row.geometry.coords]
                folium.PolyLine(
                    coords,
                    color='gray',
                    weight=line_weight,
                    opacity=line_opacity,
                    popup=None  # 不添加popup以减少HTML大小
                ).add_to(road_layer)
        except Exception as e:
            print(f"  ⚠ 道路网络显示跳过: {e}")
        
        road_layer.add_to(m)
    
    def add_nodes_and_customers(self, m: folium.Map,
                               nodes: pd.DataFrame,
                               customers: pd.DataFrame):
        """
        在地图上添加节点和客户
        
        Args:
            m: Folium Map对象
            nodes: 节点DataFrame (包含仓库和站点)
            customers: 客户DataFrame
        """
        print(f"  添加标记: {len(nodes)}个节点, {len(customers)}个客户")
        
        # 添加仓库
        depot = nodes[nodes['type'] == 'depot'].iloc[0]
        
        folium.Marker(
            location=[depot['lat'], depot['lon']],
            popup=f"<b>{depot.get('name', '仓库')}</b>",
            icon=folium.Icon(color='red', icon='home', prefix='fa'),
            tooltip='仓库'
        ).add_to(m)
        
        # 添加站点
        stations = nodes[nodes['type'] == 'station']
        for idx, station in stations.iterrows():
            folium.Marker(
                location=[station['lat'], station['lon']],
                popup=f"<b>{station.get('name', f'站点{idx}')}</b>",
                icon=folium.Icon(color='green', icon='truck', prefix='fa'),
                tooltip=station.get('name', f'站点{idx}')
            ).add_to(m)
        
        # 添加客户（使用聚类显示）
        customer_cluster = plugins.MarkerCluster(name='客户', show=True)
        
        for idx, customer in customers.iterrows():
            folium.CircleMarker(
                location=[customer['lat'], customer['lon']],
                radius=3,
                popup=f"客户{idx}<br>需求: {customer.get('demand', 0):.1f}kg",
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.7
            ).add_to(customer_cluster)
        
        customer_cluster.add_to(m)
        print(f"  ✓ 标记已添加")
    
    def add_ev_route(self, m: folium.Map,
                    solution: Dict,
                    nodes: pd.DataFrame,
                    color: str = '#0066FF',  # 鲜艳蓝色
                    weight: int = 6):
        """
        在地图上添加EV路径（沿道路网络）
        
        Args:
            m: Folium Map对象
            solution: 解的字典
            nodes: 节点DataFrame
            color: 路径颜色
            weight: 线条粗细
        """
        ev_route = solution.get('ev_route', {}).get('nodes', [])
        
        if len(ev_route) < 2:
            print("  ⚠ EV路径节点少于2个，跳过")
            return
        
        print(f"  绘制EV路径: {len(ev_route)}个节点")
        
        # EV路径图层
        ev_layer = folium.FeatureGroup(name='EV路径', show=True)
        
        # 为每条边绘制路径
        for i in range(len(ev_route) - 1):
            node1_idx = ev_route[i]
            node2_idx = ev_route[i + 1]
            
            # 使用经纬度坐标
            lat1 = nodes.iloc[node1_idx]['lat']
            lon1 = nodes.iloc[node1_idx]['lon']
            lat2 = nodes.iloc[node2_idx]['lat']
            lon2 = nodes.iloc[node2_idx]['lon']
            
            # 获取投影坐标（用于路径计算）
            x1 = nodes.iloc[node1_idx]['x']
            y1 = nodes.iloc[node1_idx]['y']
            x2 = nodes.iloc[node2_idx]['x']
            y2 = nodes.iloc[node2_idx]['y']
            
            # 尝试获取真实道路路径
            try:
                route_coords = self._get_road_path((x1, y1), (x2, y2))
                if route_coords and len(route_coords) > 1:
                    # 转换为(lat, lon)格式
                    locations = [(lat, lon) for lon, lat in route_coords]
                else:
                    # 失败则用直线
                    locations = [(lat1, lon1), (lat2, lon2)]
            except:
                # 异常则用直线
                locations = [(lat1, lon1), (lat2, lon2)]
            
            folium.PolyLine(
                locations=locations,
                color=color,
                weight=weight,
                opacity=0.9,
                popup=f"<b>EV路径</b><br>{nodes.iloc[node1_idx]['name']} → {nodes.iloc[node2_idx]['name']}"
            ).add_to(ev_layer)
        
        ev_layer.add_to(m)
        print(f"  ✓ EV路径已添加")
    
    def add_uav_sorties(self, m: folium.Map,
                       solution: Dict,
                       nodes: pd.DataFrame,
                       customers: pd.DataFrame):
        """
        在地图上添加UAV飞行轨迹（直线）
        
        Args:
            m: Folium Map对象
            solution: 解的字典
            nodes: 节点DataFrame
            customers: 客户DataFrame
        """
        uav_sorties = solution.get('uav_sorties', {})
        
        if not uav_sorties:
            print("  ⚠ 无UAV路径")
            return
        
        print(f"  绘制UAV路径: {len(uav_sorties)}个站点的sorties")
        
        # UAV路径图层
        uav_layer = folium.FeatureGroup(name='UAV路径', show=True)
        
        # 颜色映射（使用鲜艳颜色）
        colors = ['#FF0000', '#FF6600', '#9900FF', '#FF00FF', '#CC0000', '#00CC00', '#00CCCC']
        color_idx = 0
        total_sorties = 0
        
        for node_id_str, sorties in uav_sorties.items():
            node_id = int(node_id_str) if isinstance(node_id_str, str) else node_id_str
            
            # 使用经纬度
            node_lat = nodes.iloc[node_id]['lat']
            node_lon = nodes.iloc[node_id]['lon']
            
            for sortie in sorties:
                sortie_customers = sortie.get('customers', [])
                
                if not sortie_customers:
                    continue
                
                color = colors[color_idx % len(colors)]
                color_idx += 1
                total_sorties += 1
                
                # 站点到第一个客户
                first_cust = sortie_customers[0]
                first_cust_lat = customers.iloc[first_cust]['lat']
                first_cust_lon = customers.iloc[first_cust]['lon']
                
                folium.PolyLine(
                    locations=[
                        [node_lat, node_lon],
                        [first_cust_lat, first_cust_lon]
                    ],
                    color=color,
                    weight=4,  # 增加粗细
                    opacity=0.8,  # 提高不透明度
                    dash_array='8, 4',  # 更明显的虚线
                    popup=f"<b>UAV架次</b><br>从{nodes.iloc[node_id]['name']}出发"
                ).add_to(uav_layer)
                
                # 客户之间
                for i in range(len(sortie_customers) - 1):
                    c1 = sortie_customers[i]
                    c2 = sortie_customers[i + 1]
                    
                    c1_lat = customers.iloc[c1]['lat']
                    c1_lon = customers.iloc[c1]['lon']
                    c2_lat = customers.iloc[c2]['lat']
                    c2_lon = customers.iloc[c2]['lon']
                    
                    folium.PolyLine(
                        locations=[
                            [c1_lat, c1_lon],
                            [c2_lat, c2_lon]
                        ],
                        color=color,
                        weight=4,
                        opacity=0.8,
                        dash_array='8, 4'
                    ).add_to(uav_layer)
                
                # 最后一个客户返回站点
                last_cust = sortie_customers[-1]
                last_cust_lat = customers.iloc[last_cust]['lat']
                last_cust_lon = customers.iloc[last_cust]['lon']
                
                folium.PolyLine(
                    locations=[
                        [last_cust_lat, last_cust_lon],
                        [node_lat, node_lon]
                    ],
                    color=color,
                    weight=4,
                    opacity=0.8,
                    dash_array='8, 4',
                    popup=f"<b>UAV架次</b><br>返回{nodes.iloc[node_id]['name']}"
                ).add_to(uav_layer)
        
        uav_layer.add_to(m)
        print(f"  ✓ UAV路径已添加: {total_sorties}个架次")
    
    def add_heatmap(self, m: folium.Map, customers: pd.DataFrame):
        """
        添加客户需求热力图
        
        Args:
            m: Folium Map对象
            customers: 客户DataFrame
        """
        # 创建热力图数据（使用经纬度）
        heat_data = [
            [row['lat'], row['lon'], row.get('demand', 1.0)]
            for _, row in customers.iterrows()
        ]
        
        # 添加热力图层
        plugins.HeatMap(
            heat_data,
            name='需求热力图',
            min_opacity=0.2,
            radius=15,
            blur=20,
            show=False
        ).add_to(m)
    
    def visualize_solution(self, 
                          solution: Dict,
                          nodes: pd.DataFrame,
                          customers: pd.DataFrame,
                          save_path: str = 'osm_solution_map.html'):
        """
        完整可视化解决方案
        
        Args:
            solution: 解的字典
            nodes: 节点DataFrame
            customers: 客户DataFrame
            save_path: 保存HTML文件路径
        
        Returns:
            Folium Map对象
        """
        print(f"\n正在生成真实地图可视化...")
        
        # 创建基础地图
        m = self.create_base_map()
        
        # 添加道路网络（可选）
        self.add_road_network(m)
        
        # 添加节点和客户
        self.add_nodes_and_customers(m, nodes, customers)
        
        # 添加EV路径
        self.add_ev_route(m, solution, nodes)
        
        # 添加UAV路径
        self.add_uav_sorties(m, solution, nodes, customers)
        
        # 添加热力图
        self.add_heatmap(m, customers)
        
        # 添加图例
        self._add_legend(m, solution)
        
        # 保存
        m.save(save_path)
        print(f"✓ 地图已保存: {save_path}")
        print(f"  请在浏览器中打开查看")
        
        return m
    
    def _get_road_path(self, orig: tuple, dest: tuple) -> List[tuple]:
        """获取两点间的道路路径坐标（返回经纬度）"""
        try:
            # 在投影图中计算最短路径
            orig_node = ox.nearest_nodes(self.graph, orig[0], orig[1])
            dest_node = ox.nearest_nodes(self.graph, dest[0], dest[1])
            route = ox.shortest_path(self.graph, orig_node, dest_node, weight='length')
            
            if route is None:
                return [orig, dest]
            
            # 从未投影图中提取经纬度坐标
            if self.graph_unprojected is not None:
                coords = []
                for node in route:
                    if node in self.graph_unprojected.nodes:
                        node_data = self.graph_unprojected.nodes[node]
                        coords.append((node_data['x'], node_data['y']))  # 经纬度
                return coords if coords else [orig, dest]
            else:
                # 如果没有未投影图，转换坐标
                coords = []
                for node in route:
                    node_data = self.graph.nodes[node]
                    coords.append((node_data['x'], node_data['y']))
                # 转换为经纬度
                coords_latlon = self._project_to_latlon(np.array(coords))
                return [(lon, lat) for lon, lat in coords_latlon]
        except:
            # 失败则返回直线
            return [orig, dest]
    
    def _project_to_latlon(self, coords: np.ndarray) -> np.ndarray:
        """将投影坐标转换为经纬度"""
        import geopandas as gpd
        from shapely.geometry import Point
        
        try:
            # 创建GeoDataFrame
            points = [Point(x, y) for x, y in coords]
            
            # 获取CRS
            crs = self.graph_crs
            if crs is None:
                # 尝试从图中获取
                crs = self.graph.graph.get('crs', None)
            
            if crs is not None:
                gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
                # 转换到WGS84
                gdf_latlon = gdf.to_crs(epsg=4326)
            else:
                # 没有CRS信息，假设已经是WGS84
                gdf_latlon = gpd.GeoDataFrame(geometry=points, crs='EPSG:4326')
            
            # 提取经纬度
            latlon = np.array([[p.x, p.y] for p in gdf_latlon.geometry])
            
            return latlon
        except Exception as e:
            print(f"  ⚠ 坐标转换失败: {e}")
            # 返回原坐标
            return coords
    
    def _add_legend(self, m: folium.Map, solution: Dict):
        """添加图例"""
        metrics = solution.get('metrics', {})
        
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 250px; height: auto; 
                    background-color: white; z-index:9999; font-size:14px;
                    border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin:0; font-weight:bold; text-align:center">配送方案信息</p>
        <hr style="margin: 5px 0">
        <p style="margin:3px 0"><i class="fa fa-home" style="color:red"></i> 红色: 仓库</p>
        <p style="margin:3px 0"><i class="fa fa-truck" style="color:green"></i> 绿色: 站点</p>
        <p style="margin:3px 0"><i class="fa fa-circle" style="color:blue"></i> 蓝色: 客户</p>
        <p style="margin:3px 0">━━━ 蓝色实线: EV路径</p>
        <p style="margin:3px 0">- - - 彩色虚线: UAV路径</p>
        <hr style="margin: 5px 0">
        <p style="margin:3px 0; font-weight:bold">性能指标:</p>
        <p style="margin:3px 0">总成本: {metrics.get('total_cost', 0):.2f} 元</p>
        <p style="margin:3px 0">总距离: {metrics.get('total_distance', 0):.2f} km</p>
        <p style="margin:3px 0">总时间: {metrics.get('total_time', 0):.2f} 分钟</p>
        <p style="margin:3px 0">服务客户: {metrics.get('num_served', 0)} 个</p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
