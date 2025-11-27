"""
基于OpenStreetMap真实路网的实验
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime

from src.osm_data_loader import OSMDataLoader
from src.map_visualizer import MapVisualizer
from src.alns_solver import ALNSSolver
from src.data_generator import DataGenerator


def generate_dataset_from_osm(osm_data: dict, config: dict) -> dict:
    """
    基于OSM数据生成ALNS求解器所需的数据集
    
    Args:
        osm_data: OSM提取的数据
        config: 配置参数
    
    Returns:
        dataset字典
    """
    loader = osm_data['loader']
    nodes = osm_data['nodes']
    customers = osm_data['customers']
    
    print("\n[步骤3] 生成数据集...")
    
    # 1. 准备节点数据
    print("  准备节点数据...")
    nodes_df = nodes.copy()
    
    # 2. 准备客户数据
    print("  准备客户数据...")
    customers_df = customers.copy()
    
    # 3. 计算距离矩阵
    print("  计算距离矩阵...")
    
    # 合并所有位置（节点+客户）
    all_locations = pd.concat([
        nodes_df[['x', 'y']],
        customers_df[['x', 'y']]
    ], ignore_index=True)
    
    n_nodes = len(nodes_df)
    n_customers = len(customers_df)
    n_total = n_nodes + n_customers
    
    # EV距离：使用真实道路距离
    print("    计算EV距离（真实道路）...")
    dist_ev = np.zeros((n_total, n_total))
    
    # 只计算节点到节点、节点到客户的距离（EV只在节点间移动）
    for i in range(n_total):
        for j in range(i+1, n_total):
            if i < n_nodes or j < n_nodes:  # 至少一个是节点
                coords_i = all_locations.iloc[i][['x', 'y']].values
                coords_j = all_locations.iloc[j][['x', 'y']].values
                
                dist = loader.calculate_road_distance(
                    (coords_i[0], coords_i[1]),
                    (coords_j[0], coords_j[1])
                )
                dist_ev[i, j] = dist
                dist_ev[j, i] = dist
        
        if (i+1) % 10 == 0:
            print(f"      进度: {i+1}/{n_total}")
    
    # UAV距离：使用欧氏距离（直线飞行）
    print("    计算UAV距离（直线）...")
    dist_uav = np.zeros((n_total, n_total))
    
    for i in range(n_total):
        for j in range(i+1, n_total):
            coords_i = all_locations.iloc[i][['x', 'y']].values
            coords_j = all_locations.iloc[j][['x', 'y']].values
            
            dist = np.linalg.norm(coords_i - coords_j) / 1000.0  # 转为公里
            dist_uav[i, j] = dist
            dist_uav[j, i] = dist
    
    # 4. 计算时间和能耗矩阵
    print("  计算时间和能耗矩阵...")
    
    ev_speed = config['ev']['speed']
    uav_speed = config['uav']['speed']
    
    time_ev = dist_ev / ev_speed * 60  # 转为分钟
    time_uav = dist_uav / uav_speed * 60
    
    ev_energy_rate = config['ev']['energy_consumption']
    uav_energy_rate = config['uav']['energy_consumption']
    
    energy_ev = dist_ev * ev_energy_rate
    energy_uav = dist_uav * uav_energy_rate
    
    # 5. 构建数据集（按ALNS求解器期待的格式）
    dataset = {
        'nodes': nodes_df,
        'customers': customers_df,
        'distances': {
            'ev': dist_ev,
            'uav': dist_uav
        },
        'times': {
            'ev': time_ev,
            'uav': time_uav
        },
        'energy': {
            'ev': energy_ev,
            'uav': energy_uav
        },
        'osm_loader': loader  # 保留loader用于可视化
    }
    
    print(f"  ✓ 数据集生成完成")
    print(f"    节点数: {len(nodes_df)}")
    print(f"    客户数: {len(customers_df)}")
    
    return dataset


def run_real_world_experiment(city: str = None,
                              bbox: tuple = None,
                              num_customers: int = 50,
                              num_stations: int = 3,
                              max_iterations: int = 1000):
    """
    运行真实场景实验
    
    Args:
        city: 城市名称（可选）
        bbox: 边界框 (north, south, east, west)（推荐使用）
        num_customers: 客户数量
        num_stations: 站点数量
        max_iterations: 最大迭代次数
    """
    
    print("=" * 80)
    if city:
        print(f"真实路网实验: {city}")
    else:
        print(f"真实路网实验: bbox {bbox}")
    print("=" * 80)
    
    # 加载配置（使用真实世界专用配置）
    with open('config/params_real_world.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改迭代次数
    config['alns']['max_iterations'] = max_iterations
    
    # [步骤1] 提取OSM数据
    print("\n[步骤1] 从OpenStreetMap提取数据...")
    
    loader = OSMDataLoader(city if city else "Custom Area")
    
    try:
        # 提取道路网络
        if bbox:
            print(f"  使用bbox方式提取: {bbox}")
            loader.extract_bbox_network(bbox[0], bbox[1], bbox[2], bbox[3])
        else:
            loader.extract_road_network(simplify=True)
    except Exception as e:
        print(f"✗ 道路网络提取失败: {e}")
        print("提示: 请检查网络连接，或尝试使用bbox方式提取")
        return None, None
    
    # 采样客户位置
    customers = loader.sample_customer_locations(
        num_customers=num_customers,
        strategy='random'
    )
    
    # 确保customers有lat/lon列
    if 'lat' not in customers.columns:
        from shapely.geometry import Point
        import geopandas as gpd
        
        cust_gdf = gpd.GeoDataFrame(
            customers,
            geometry=[Point(x, y) for x, y in zip(customers['x'], customers['y'])],
            crs=loader.graph.graph.get('crs')
        )
        cust_wgs84 = cust_gdf.to_crs('EPSG:4326')
        customers['lon'] = cust_wgs84.geometry.x
        customers['lat'] = cust_wgs84.geometry.y
    
    # 选择站点位置（多选1个用作仓库）
    all_stations = loader.select_station_locations(
        num_stations=num_stations + 1,  # 多选一个用作仓库
        strategy='random'
    )
    
    # 创建仓库（使用第一个站点位置）
    first_station = all_stations.iloc[0]
    depot = pd.DataFrame({
        'id': [0],
        'x': [first_station['x']],
        'y': [first_station['y']],
        'lon': [first_station.get('lon', first_station['x'])],
        'lat': [first_station.get('lat', first_station['y'])],
        'type': ['depot'],
        'name': ['仓库']
    })
    
    # 使用剩余站点（移除第一个，避免与仓库重叠）
    stations = all_stations.iloc[1:].copy()
    stations['id'] = range(1, len(stations) + 1)
    
    # 确保stations有lat/lon列
    if 'lat' not in stations.columns:
        # 从loader的未投影图获取经纬度
        from shapely.geometry import Point
        import geopandas as gpd
        
        # 将投影坐标转换为经纬度
        gdf = gpd.GeoDataFrame(
            stations,
            geometry=[Point(x, y) for x, y in zip(stations['x'], stations['y'])],
            crs=loader.graph.graph.get('crs')
        )
        gdf_wgs84 = gdf.to_crs('EPSG:4326')
        stations['lon'] = gdf_wgs84.geometry.x
        stations['lat'] = gdf_wgs84.geometry.y
        
        # 同样转换depot
        depot_gdf = gpd.GeoDataFrame(
            depot,
            geometry=[Point(depot.iloc[0]['x'], depot.iloc[0]['y'])],
            crs=loader.graph.graph.get('crs')
        )
        depot_wgs84 = depot_gdf.to_crs('EPSG:4326')
        depot['lon'] = [depot_wgs84.geometry.x.iloc[0]]
        depot['lat'] = [depot_wgs84.geometry.y.iloc[0]]
    
    # 合并所有节点
    nodes = pd.concat([depot, stations], ignore_index=True)
    
    osm_data = {
        'loader': loader,
        'nodes': nodes,
        'customers': customers,
        'graph': loader.graph
    }
    
    print(f"  ✓ OSM数据提取完成")
    
    # [步骤2] 生成数据集
    dataset = generate_dataset_from_osm(osm_data, config)
    
    # [步骤3] ALNS求解
    print("\n[步骤4] ALNS求解...")
    print(f"  迭代次数: {max_iterations}")
    print(f"  客户数量: {num_customers}")
    print(f"  站点数量: {num_stations}")
    
    solver = ALNSSolver(dataset, config)
    solution = solver.solve()
    
    print(f"\n求解完成！")
    print(f"最优成本: {solution.total_cost:.2f}")
    
    # [步骤4] 评估解
    print("\n[步骤5] 评估解...")
    metrics = solution.to_dict()['metrics']
    
    print("\n性能指标:")
    print(f"  - 总成本: {metrics['total_cost']:.2f} 元")
    print(f"  - EV成本: {metrics['ev_cost']:.2f} 元")
    print(f"  - UAV成本: {metrics['uav_cost']:.2f} 元")
    print(f"  - 能耗成本: {metrics['energy_cost']:.2f} 元")
    print(f"  - 时间成本: {metrics['time_cost']:.2f} 元")
    print(f"  - 碳排放成本: {metrics['carbon_cost']:.2f} 元")
    print(f"  - 总距离: {metrics['total_distance']:.2f} km")
    print(f"  - 总时间: {metrics['total_time']:.2f} 分钟")
    print(f"  - 服务客户数: {metrics['num_served']}/{len(customers)}")
    print(f"  - 可行性: {'✓' if metrics['is_feasible'] else '✗'}")
    
    # 创建结果目录
    result_dir = Path('data/results/real_world')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # [步骤5] 先可视化道路网络（无底图）
    print("\n[步骤6] 生成道路网络可视化（无底图）...")
    
    import folium
    import osmnx as ox
    
    # 获取未投影的图
    graph_wgs84 = loader.graph_unprojected
    
    # 计算中心点
    nodes_gdf = ox.graph_to_gdfs(graph_wgs84, edges=False)
    center_lat = nodes_gdf.geometry.y.mean()
    center_lon = nodes_gdf.geometry.x.mean()
    
    print(f"  地图中心: ({center_lat:.4f}, {center_lon:.4f})")
    
    # 创建空白地图（无底图）
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None  # 无底图
    )
    
    # 绘制道路网络
    edges_gdf = ox.graph_to_gdfs(graph_wgs84, nodes=False, edges=True)
    print(f"  绘制道路: {len(edges_gdf)} 条边")
    
    road_layer = folium.FeatureGroup(name='🛣️ 道路网络', show=False)  # 默认关闭，避免太杂乱
    
    for idx, row in edges_gdf.iterrows():
        coords = [(point[1], point[0]) for point in row.geometry.coords]
        
        # 道路类型
        highway_type = row.get('highway', 'unknown')
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else 'unknown'
        
        # 颜色分类
        if highway_type in ['motorway', 'trunk']:
            color = 'red'
            weight = 3
        elif highway_type in ['primary', 'secondary']:
            color = 'orange'
            weight = 2
        elif highway_type in ['tertiary', 'residential']:
            color = 'blue'
            weight = 1.5
        else:
            color = 'gray'
            weight = 1
        
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.6,
            popup=f"道路: {highway_type}"
        ).add_to(road_layer)
    
    road_layer.add_to(m)
    
    # 添加边界框
    bbox_coords = [
        (bbox[1], bbox[2]),
        (bbox[0], bbox[2]),
        (bbox[0], bbox[3]),
        (bbox[1], bbox[3]),
        (bbox[1], bbox[2])
    ]
    folium.Polygon(
        locations=bbox_coords,
        color='green',
        weight=3,
        fill=False,
        popup="区域边界"
    ).add_to(m)
    
    # 保存道路网络图
    network_path = result_dir / 'osm_network_only.html'
    m.save(str(network_path))
    
    print(f"✓ 道路网络图已保存: {network_path}")
    print(f"  - 红色: 高速/主干道")
    print(f"  - 橙色: 主要道路") 
    print(f"  - 蓝色: 次要道路")
    print(f"  - 灰色: 其他道路")
    print(f"  - 绿色框: 区域边界")
    print(f"\n请先打开此文件查看道路网络是否正确！")
    print(f"确认道路网络正确后，我们再继续添加路径...")
    
    # 生成完整地图（道路网络 + EV路径 + UAV路径）
    print("\n[步骤6.2] 添加EV和UAV路径到地图...")
    
    # 获取解的字典
    solution_dict = solution.to_dict()
    
    # 确保customers有正确的lat/lon（经纬度）
    if 'lat' not in customers.columns or customers.iloc[0]['lat'] > 180:
        print("  ⚠️ 修复customers坐标...")
        from shapely.geometry import Point
        import geopandas as gpd
        
        cust_gdf = gpd.GeoDataFrame(
            customers,
            geometry=[Point(x, y) for x, y in zip(customers['x'], customers['y'])],
            crs=loader.graph.graph.get('crs')
        )
        cust_wgs84 = cust_gdf.to_crs('EPSG:4326')
        customers['lon'] = cust_wgs84.geometry.x
        customers['lat'] = cust_wgs84.geometry.y
    
    # 在道路网络基础上添加路径
    print("  添加节点和客户标记...")
    
    # 创建标记图层组
    markers_layer = folium.FeatureGroup(name='📍 节点和客户标记', show=True)
    
    # 添加仓库
    depot_row = nodes[nodes['type'] == 'depot'].iloc[0]
    folium.Marker(
        location=[depot_row['lat'], depot_row['lon']],
        popup=f"<b>{depot_row.get('name', '仓库')}</b>",
        icon=folium.Icon(color='red', icon='home', prefix='fa'),
        tooltip='仓库'
    ).add_to(markers_layer)
    
    # 添加站点
    stations = nodes[nodes['type'] == 'station']
    for idx, station in stations.iterrows():
        folium.Marker(
            location=[station['lat'], station['lon']],
            popup=f"<b>{station.get('name', f'站点{station['id']}')}</b>",
            icon=folium.Icon(color='green', icon='truck', prefix='fa'),
            tooltip=station.get('name', f'站点{station["id"]}')
        ).add_to(markers_layer)
    
    markers_layer.add_to(m)
    
    # 添加客户（简化显示，只用圆圈）
    print("  添加客户位置...")
    print(f"    客户DataFrame索引范围: {customers.index.min()} ~ {customers.index.max()}")
    print(f"    客户数量: {len(customers)}")
    
    customers_layer = folium.FeatureGroup(name='👥 客户位置', show=True)
    for idx, customer in customers.iterrows():
        folium.CircleMarker(
            location=[customer['lat'], customer['lon']],
            radius=3,
            popup=f"客户{idx}",
            color='blue',
            fill=True,
            fillColor='lightblue',
            fillOpacity=0.7
        ).add_to(customers_layer)
    
    customers_layer.add_to(m)
    
    # 添加EV路径
    print("  添加EV路径...")
    ev_route = solution_dict.get('ev_route', {}).get('nodes', [])
    
    if len(ev_route) >= 2:
        ev_layer = folium.FeatureGroup(name='🚗 EV路径', show=True)
        
        for i in range(len(ev_route) - 1):
            node1_idx = ev_route[i]
            node2_idx = ev_route[i + 1]
            
            lat1 = nodes.iloc[node1_idx]['lat']
            lon1 = nodes.iloc[node1_idx]['lon']
            lat2 = nodes.iloc[node2_idx]['lat']
            lon2 = nodes.iloc[node2_idx]['lon']
            
            # 获取投影坐标
            x1 = nodes.iloc[node1_idx]['x']
            y1 = nodes.iloc[node1_idx]['y']
            x2 = nodes.iloc[node2_idx]['x']
            y2 = nodes.iloc[node2_idx]['y']
            
            # 尝试获取真实道路路径
            from src.map_visualizer import MapVisualizer
            try:
                temp_viz = MapVisualizer(loader)
                route_coords = temp_viz._get_road_path((x1, y1), (x2, y2))
                if route_coords and len(route_coords) > 1:
                    locations = [(lat, lon) for lon, lat in route_coords]
                else:
                    locations = [(lat1, lon1), (lat2, lon2)]
            except:
                locations = [(lat1, lon1), (lat2, lon2)]
            
            folium.PolyLine(
                locations=locations,
                color='#0066FF',
                weight=6,
                opacity=0.9,
                popup=f"<b>EV路径</b><br>{nodes.iloc[node1_idx]['name']} → {nodes.iloc[node2_idx]['name']}"
            ).add_to(ev_layer)
        
        # 添加EV直接服务的客户连线
        ev_direct_customers = solution_dict.get('ev_route', {}).get('direct_customers', [])
        if ev_direct_customers:
            print(f"    添加EV直接服务的{len(ev_direct_customers)}个客户...")
            for cust_idx in ev_direct_customers:
                if cust_idx >= len(customers):
                    continue
                
                cust_lat = customers.iloc[cust_idx]['lat']
                cust_lon = customers.iloc[cust_idx]['lon']
                
                # 找到最近的EV路径节点
                min_dist = float('inf')
                nearest_node_idx = None
                for node_idx in ev_route:
                    node_lat = nodes.iloc[node_idx]['lat']
                    node_lon = nodes.iloc[node_idx]['lon']
                    dist = ((cust_lat - node_lat)**2 + (cust_lon - node_lon)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node_idx = node_idx
                
                if nearest_node_idx is not None:
                    # 画一条从最近节点到客户的线
                    folium.PolyLine(
                        locations=[
                            [nodes.iloc[nearest_node_idx]['lat'], nodes.iloc[nearest_node_idx]['lon']],
                            [cust_lat, cust_lon]
                        ],
                        color='#0066FF',
                        weight=3,
                        opacity=0.6,
                        dash_array='4, 4',
                        popup=f"<b>EV直接服务</b><br>客户{cust_idx}"
                    ).add_to(ev_layer)
        
        ev_layer.add_to(m)
        print(f"    ✓ EV路径已添加: {len(ev_route)-1}段主路径 + {len(ev_direct_customers)}个客户服务")
    
    # 添加UAV路径
    print("  添加UAV路径...")
    uav_sorties = solution_dict.get('uav_sorties', {})
    
    # 调试：检查customers的列
    print(f"    Customers columns: {list(customers.columns)}")
    print(f"    First customer coords: lat={customers.iloc[0]['lat']:.6f}, lon={customers.iloc[0]['lon']:.6f}")
    
    if uav_sorties:
        uav_layer = folium.FeatureGroup(name='🚁 UAV路径', show=True)
        colors = ['#FF0000', '#FF6600', '#9900FF', '#FF00FF', '#CC0000', '#00CC00', '#00CCCC']
        color_idx = 0
        total_sorties = 0
        
        # 打印调试信息
        print(f"    UAV sorties站点数: {len(uav_sorties)}")
        
        for node_id_str, sorties in uav_sorties.items():
            node_id = int(node_id_str) if isinstance(node_id_str, str) else node_id_str
            
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
                
                # 检查索引是否有效
                if first_cust >= len(customers):
                    print(f"    ⚠️ 警告: 客户索引{first_cust}超出范围(最大{len(customers)-1}), 跳过此sortie")
                    continue
                
                first_cust_lat = customers.iloc[first_cust]['lat']
                first_cust_lon = customers.iloc[first_cust]['lon']
                
                folium.PolyLine(
                    locations=[[node_lat, node_lon], [first_cust_lat, first_cust_lon]],
                    color=color,
                    weight=4,
                    opacity=0.8,
                    dash_array='8, 4',
                    popup=f"<b>UAV架次</b><br>从{nodes.iloc[node_id]['name']}出发"
                ).add_to(uav_layer)
                
                # 客户之间
                for i in range(len(sortie_customers) - 1):
                    c1 = sortie_customers[i]
                    c2 = sortie_customers[i + 1]
                    
                    # 检查索引
                    if c1 >= len(customers) or c2 >= len(customers):
                        continue
                    
                    folium.PolyLine(
                        locations=[
                            [customers.iloc[c1]['lat'], customers.iloc[c1]['lon']],
                            [customers.iloc[c2]['lat'], customers.iloc[c2]['lon']]
                        ],
                        color=color,
                        weight=4,
                        opacity=0.8,
                        dash_array='8, 4'
                    ).add_to(uav_layer)
                
                # 返回站点
                last_cust = sortie_customers[-1]
                
                # 检查索引
                if last_cust >= len(customers):
                    continue
                
                folium.PolyLine(
                    locations=[
                        [customers.iloc[last_cust]['lat'], customers.iloc[last_cust]['lon']],
                        [node_lat, node_lon]
                    ],
                    color=color,
                    weight=4,
                    opacity=0.8,
                    dash_array='8, 4',
                    popup=f"<b>UAV架次</b><br>返回{nodes.iloc[node_id]['name']}"
                ).add_to(uav_layer)
        
        uav_layer.add_to(m)
        print(f"    ✓ UAV路径已添加: {total_sorties}个架次")
    
    # 添加图层控制器
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # 添加图例
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 320px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <h4 style="margin-top:0">🗺️ 图例说明</h4>
    <p style="margin:5px 0;"><i>💡 使用右上角图层控制器切换显示</i></p>
    <hr style="margin:10px 0;">
    
    <p><b>📍 标记：</b></p>
    <p style="margin:3px 0;">🏠 <span style="color:red">红色房子</span> = 仓库</p>
    <p style="margin:3px 0;">🚚 <span style="color:green">绿色卡车</span> = EV停靠站点</p>
    <p style="margin:3px 0;">🔵 <span style="color:blue">蓝色小点</span> = 客户位置</p>
    
    <p style="margin-top:10px;"><b>🚗 路径：</b></p>
    <p style="margin:3px 0;"><span style="color:#0066FF; font-weight:bold">━━━</span> 蓝色粗实线 = EV路径（沿道路）</p>
    <p style="margin:3px 0;"><span style="color:red; font-weight:bold">╌╌╌</span> 彩色虚线 = UAV路径（直线飞行）</p>
    
    <p style="margin-top:10px;"><b>🛣️ 道路网络：</b></p>
    <p style="margin:3px 0;"><span style="color:red">━</span> 红色 = 高速/主干道</p>
    <p style="margin:3px 0;"><span style="color:orange">━</span> 橙色 = 主要道路</p>
    <p style="margin:3px 0;"><span style="color:blue">━</span> 蓝色 = 次要道路</p>
    <p style="margin:3px 0;"><span style="color:gray">━</span> 灰色 = 其他道路</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 保存完整地图
    map_path = result_dir / 'osm_solution_with_routes.html'
    m.save(str(map_path))
    
    print(f"\n✓ 完整地图已保存: {map_path}")
    print(f"  包含: 道路网络 + EV路径 + UAV路径 + 节点标记 + 客户位置 + 图层控制 + 图例")
    print(f"\n💡 使用方法:")
    print(f"  1. 打开地图后，在右上角可以看到图层控制器")
    print(f"  2. 勾选/取消勾选图层名称前的复选框来显示/隐藏图层")
    print(f"  3. 建议先关闭'道路网络'图层，只查看路径和标记")
    
    # [步骤6] 保存结果
    print("\n[步骤7] 保存结果...")
    
    # 保存解
    with open(result_dir / 'solution.json', 'w', encoding='utf-8') as f:
        json.dump(solution_dict, f, indent=2, ensure_ascii=False)
    
    # 转换坐标为经纬度后保存
    import geopandas as gpd
    from shapely.geometry import Point
    
    # 转换nodes坐标
    nodes_points = [Point(x, y) for x, y in nodes[['x', 'y']].values]
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=nodes_points, crs=loader.graph.graph['crs'])
    nodes_latlon = nodes_gdf.to_crs(epsg=4326)
    nodes_save = nodes.copy()
    nodes_save['lon'] = nodes_latlon.geometry.x
    nodes_save['lat'] = nodes_latlon.geometry.y
    nodes_save.to_csv(result_dir / 'nodes.csv', index=False)
    
    # 转换customers坐标
    customers_points = [Point(x, y) for x, y in customers[['x', 'y']].values]
    customers_gdf = gpd.GeoDataFrame(customers, geometry=customers_points, crs=loader.graph.graph['crs'])
    customers_latlon = customers_gdf.to_crs(epsg=4326)
    customers_save = customers.copy()
    customers_save['lon'] = customers_latlon.geometry.x
    customers_save['lat'] = customers_latlon.geometry.y
    customers_save.to_csv(result_dir / 'customers.csv', index=False)
    
    # 保存配置
    with open(result_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    # 保存实验信息
    experiment_info = {
        'city': city,
        'num_customers': num_customers,
        'num_stations': num_stations,
        'max_iterations': max_iterations,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    with open(result_dir / 'experiment_info.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到: {result_dir}")
    print(f"  - 道路网络图（无路径）: {network_path}")
    print(f"  - 完整地图（含路径）: {map_path}")
    print(f"  - 解数据: solution.json")
    print(f"  - 节点数据: nodes.csv")
    print(f"  - 客户数据: customers.csv")
    
    print("\n" + "=" * 80)
    print("实验完成！请在浏览器中打开完整地图查看")
    print(f"完整地图路径: {map_path}")
    print("此地图包含: 道路网络 + EV路径(蓝色粗线) + UAV路径(彩色虚线) + 所有标记")
    print("=" * 80)
    
    return solution, osm_data


if __name__ == '__main__':
    # 推荐使用bbox方式（更快更稳定）
    
    # 预定义区域（使用较小范围先验证地图）
    test_areas = {
        'beijing_cbd_small': (39.915, 39.905, 116.465, 116.450),  # 北京国贸小范围 (1km×1.5km)
        'beijing_cbd': (39.925, 39.900, 116.475, 116.445),        # 北京国贸 (3km×3km)
        'beijing_large': (39.945, 39.880, 116.495, 116.425),      # 北京扩大版 (7km×7km)
        'beijing_xlarge': (39.980, 39.850, 116.540, 116.380),     # 北京超大范围 (14km×16km)
    }
    
    # 选择区域 - 使用超大范围
    area_name = 'beijing_xlarge'  # 超大范围
    bbox = test_areas[area_name]
    
    # 参数配置
    num_customers = 100  # 增加到100个客户
    num_stations = 5     # 增加到5个站点
    max_iterations = 500
    
    print(f"\n开始真实路网实验")
    print(f"区域: {area_name}")
    print(f"范围: {bbox}")
    print(f"客户数: {num_customers}")
    print(f"站点数: {num_stations}")
    print(f"迭代数: {max_iterations}")
    print("\n" + "=" * 80)
    
    solution, osm_data = run_real_world_experiment(
        bbox=bbox,
        num_customers=num_customers,
        num_stations=num_stations,
        max_iterations=max_iterations
    )
