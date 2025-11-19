"""
数据生成模块
生成仿真数据集，包括客户点、站点、充电站等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - 只在没装 PyYAML 的环境里触发
    yaml = None


class DataGenerator:
    """数据生成器"""
    
    def __init__(self, config_path):
        """
        初始化数据生成器
        
        Args:
            config_path: 配置文件路径（str / Path），或已经解析好的配置字典。
                         传字典时不需要安装 PyYAML，便于在测试里直接构造配置。
        """
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            if yaml is None:
                raise ImportError(
                    "读取 YAML 配置需要 PyYAML：pip install pyyaml\n"
                    "（也可以直接传一个已解析的 dict 给 DataGenerator）"
                )
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        self.area_size = self.config['geography']['area_size']
        self.depot_location = self.config['geography']['depot_location']
        
        np.random.seed(42)  # 设置随机种子确保可重复性
    
    def generate_dataset(self, save_dir: Optional[str] = None) -> Dict:
        """
        生成完整数据集
        
        Args:
            save_dir: 保存目录，如果为None则不保存
            
        Returns:
            包含所有数据的字典
        """
        dataset = {
            'nodes': self._generate_nodes(),
            'customers': self._generate_customers(),
            'distances': {},
            'times': {},
            'config': self.config
        }
        
        # 计算距离和时间矩阵
        dataset['distances']['ev'] = self._calculate_distance_matrix(
            dataset['nodes'], mode='ev'
        )
        dataset['distances']['uav'] = self._calculate_distance_matrix(
            dataset['nodes'], mode='uav'
        )
        
        dataset['times']['ev'] = dataset['distances']['ev'] / self.config['ev']['speed'] * 60  # 转换为分钟
        dataset['times']['uav'] = dataset['distances']['uav'] / self.config['uav']['speed'] * 60
        
        # 计算能耗矩阵
        dataset['energy'] = {
            'ev': dataset['distances']['ev'] * self.config['ev']['energy_consumption'],
            'uav': dataset['distances']['uav'] * self.config['uav']['energy_consumption']
        }
        
        if save_dir:
            self._save_dataset(dataset, save_dir)
        
        return dataset
    
    def _generate_nodes(self) -> pd.DataFrame:
        """
        生成节点（仓库、站点、充电站）
        
        Returns:
            节点数据DataFrame
        """
        nodes = []
        
        # 仓库（节点0）
        depot = self.config['geography']['depot_location']
        nodes.append({
            'node_id': 0,
            'type': 'depot',
            'x': depot[0],
            'y': depot[1],
            'name': 'Depot'
        })
        
        # EV停靠站点
        num_stations = self.config['geography']['num_stations']
        for i in range(1, num_stations + 1):
            # 使用聚类分布生成站点
            cluster_center = np.random.uniform(
                [self.area_size[0] * 0.2, self.area_size[1] * 0.2],
                [self.area_size[0] * 0.8, self.area_size[1] * 0.8]
            )
            x = np.clip(cluster_center[0] + np.random.normal(0, 1), 0, self.area_size[0])
            y = np.clip(cluster_center[1] + np.random.normal(0, 1), 0, self.area_size[1])
            
            nodes.append({
                'node_id': i,
                'type': 'station',
                'x': x,
                'y': y,
                'name': f'Station_{i}'
            })
        
        # 充电站
        num_charging = self.config['geography']['num_charging_stations']
        start_id = num_stations + 1
        for i in range(num_charging):
            x = np.random.uniform(0, self.area_size[0])
            y = np.random.uniform(0, self.area_size[1])
            
            nodes.append({
                'node_id': start_id + i,
                'type': 'charging_station',
                'x': x,
                'y': y,
                'name': f'Charging_{i+1}'
            })
        
        return pd.DataFrame(nodes)
    
    def _generate_customers(self) -> pd.DataFrame:
        """
        生成客户点
        
        Returns:
            客户数据DataFrame
        """
        num_customers = self.config['geography']['num_customers']
        demand_range = self.config['customers']['demand_range']
        service_time = self.config['customers']['service_time']
        tw_type = self.config['customers']['time_window_type']
        tw_width = self.config['customers']['time_window_width'][tw_type]
        
        customers = []
        
        for i in range(num_customers):
            # 客户位置（倾向于聚集分布）
            if np.random.random() < 0.7:  # 70%的客户聚集在某些区域
                cluster_idx = np.random.randint(0, 3)
                cluster_centers = [
                    [self.area_size[0] * 0.3, self.area_size[1] * 0.3],
                    [self.area_size[0] * 0.7, self.area_size[1] * 0.3],
                    [self.area_size[0] * 0.5, self.area_size[1] * 0.7]
                ]
                center = cluster_centers[cluster_idx]
                x = np.clip(center[0] + np.random.normal(0, 2), 0, self.area_size[0])
                y = np.clip(center[1] + np.random.normal(0, 2), 0, self.area_size[1])
            else:  # 30%的客户随机分布
                x = np.random.uniform(0, self.area_size[0])
                y = np.random.uniform(0, self.area_size[1])
            
            # 需求量
            demand = np.random.uniform(demand_range[0], demand_range[1])
            
            # 时间窗（早上8点到晚上8点，共720分钟）
            earliest = np.random.uniform(0, 720 - tw_width)
            latest = earliest + tw_width
            
            customers.append({
                'customer_id': i,
                'x': x,
                'y': y,
                'demand': demand,
                'service_time': service_time,
                'time_window_start': earliest,
                'time_window_end': latest,
                'name': f'Customer_{i+1}'
            })
        
        return pd.DataFrame(customers)
    
    def _calculate_distance_matrix(
        self, 
        nodes: pd.DataFrame, 
        mode: str = 'ev'
    ) -> np.ndarray:
        """
        计算距离矩阵（欧几里得距离）
        
        Args:
            nodes: 节点数据
            mode: 'ev' 或 'uav'
            
        Returns:
            距离矩阵
        """
        n = len(nodes)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = nodes.iloc[i]['x'] - nodes.iloc[j]['x']
                    dy = nodes.iloc[i]['y'] - nodes.iloc[j]['y']
                    distances[i, j] = np.sqrt(dx**2 + dy**2)
        
        return distances
    
    def _save_dataset(self, dataset: Dict, save_dir: str):
        """
        保存数据集
        
        Args:
            dataset: 数据集字典
            save_dir: 保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存节点和客户信息
        dataset['nodes'].to_csv(save_path / 'nodes.csv', index=False)
        dataset['customers'].to_csv(save_path / 'customers.csv', index=False)
        
        # 保存距离、时间和能耗矩阵
        np.save(save_path / 'distance_ev.npy', dataset['distances']['ev'])
        np.save(save_path / 'distance_uav.npy', dataset['distances']['uav'])
        np.save(save_path / 'time_ev.npy', dataset['times']['ev'])
        np.save(save_path / 'time_uav.npy', dataset['times']['uav'])
        np.save(save_path / 'energy_ev.npy', dataset['energy']['ev'])
        np.save(save_path / 'energy_uav.npy', dataset['energy']['uav'])
        
        # 保存配置
        with open(save_path / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"数据集已保存到: {save_path}")
    
    @staticmethod
    def load_dataset(load_dir: str) -> Dict:
        """
        加载已保存的数据集
        
        Args:
            load_dir: 数据集目录
            
        Returns:
            数据集字典
        """
        load_path = Path(load_dir)
        
        dataset = {
            'nodes': pd.read_csv(load_path / 'nodes.csv'),
            'customers': pd.read_csv(load_path / 'customers.csv'),
            'distances': {
                'ev': np.load(load_path / 'distance_ev.npy'),
                'uav': np.load(load_path / 'distance_uav.npy')
            },
            'times': {
                'ev': np.load(load_path / 'time_ev.npy'),
                'uav': np.load(load_path / 'time_uav.npy')
            },
            'energy': {
                'ev': np.load(load_path / 'energy_ev.npy'),
                'uav': np.load(load_path / 'energy_uav.npy')
            }
        }
        
        with open(load_path / 'config.json', 'r', encoding='utf-8') as f:
            dataset['config'] = json.load(f)
        
        return dataset


def main():
    """测试数据生成"""
    # 小规模测试
    print("生成小规模数据集...")
    gen_small = DataGenerator('config/params_small.yaml')
    dataset_small = gen_small.generate_dataset('data/processed/small_scale')
    
    print(f"节点数: {len(dataset_small['nodes'])}")
    print(f"客户数: {len(dataset_small['customers'])}")
    print("\n节点信息:")
    print(dataset_small['nodes'])
    print("\n客户信息（前5个）:")
    print(dataset_small['customers'].head())
    
    print("\n" + "="*50 + "\n")
    
    # 大规模测试
    print("生成大规模数据集...")
    gen_large = DataGenerator('config/params_large.yaml')
    dataset_large = gen_large.generate_dataset('data/processed/large_scale')
    
    print(f"节点数: {len(dataset_large['nodes'])}")
    print(f"客户数: {len(dataset_large['customers'])}")


if __name__ == '__main__':
    main()
