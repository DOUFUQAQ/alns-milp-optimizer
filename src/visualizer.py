"""
可视化模块
用于绘制路径、性能指标和对比分析
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.font_manager as fm
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:  # pragma: no cover - 没装 matplotlib/seaborn 时降级
    plt = None
    patches = None
    fm = None
    sns = None
    PLOTTING_AVAILABLE = False

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import json
from pathlib import Path
import warnings

# 强制设置中文字体（使用Windows系统字体文件）
def setup_chinese_font():
    """强制设置中文字体 - 返回字体属性对象"""
    if not PLOTTING_AVAILABLE:
        return None
    
    import os
    import sys
    
    # Windows系统字体目录
    if sys.platform == 'win32':
        font_dirs = [r'C:\Windows\Fonts']
        
        # 优先级：微软雅黑 > 黑体 > 宋体
        font_files = [
            ('Microsoft YaHei', ['msyh.ttc', 'msyh.ttf']),
            ('SimHei', ['simhei.ttf']),
            ('SimSun', ['simsun.ttc', 'simsun.ttf'])
        ]
        
        # 查找第一个可用字体
        for font_name, files in font_files:
            for font_dir in font_dirs:
                for font_file in files:
                    font_path = os.path.join(font_dir, font_file)
                    if os.path.exists(font_path):
                        # 创建字体属性对象
                        font_prop = fm.FontProperties(fname=font_path)
                        # 同时设置rcParams（作为备份）
                        plt.rcParams['font.sans-serif'] = [font_name]
                        plt.rcParams['font.family'] = 'sans-serif'
                        plt.rcParams['axes.unicode_minus'] = False
                        print(f"[字体] 加载: {font_name} ({font_file})")
                        return font_prop
    
    # 如果没找到，返回None
    warnings.warn("未找到中文字体文件")
    return None

# 初始化全局字体属性（缺少 matplotlib 时为 None）
CHINESE_FONT = setup_chinese_font() if PLOTTING_AVAILABLE else None


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, dataset: Dict, config: Dict):
        """
        初始化可视化器
        
        Args:
            dataset: 数据集
            config: 配置参数
        """
        self.dataset = dataset
        self.config = config
        self.font_prop = CHINESE_FONT  # 使用全局字体属性
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_solution(
        self,
        solution,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制完整解的路径图
        
        Args:
            solution: Solution对象或字典
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        nodes_df = self.dataset['nodes']
        customers_df = self.dataset['customers']
        
        # 兼容字典和对象格式
        if isinstance(solution, dict):
            ev_route_nodes = solution['ev_route']['nodes']
            uav_sorties = solution['ev_route'].get('uav_sorties', {})
            total_cost = solution.get('metrics', {}).get('total_cost', 0)
        else:
            ev_route_nodes = solution.ev_route.nodes
            uav_sorties = solution.ev_route.uav_sorties
            total_cost = solution.total_cost
        
        # 绘制客户点
        ax.scatter(
            customers_df['x'],
            customers_df['y'],
            c='lightblue',
            s=100,
            marker='o',
            label='客户',
            alpha=0.6,
            edgecolors='blue'
        )
        
        # 绘制节点
        depot = nodes_df[nodes_df['type'] == 'depot']
        stations = nodes_df[nodes_df['type'] == 'station']
        charging = nodes_df[nodes_df['type'] == 'charging_station']
        
        ax.scatter(
            depot['x'], depot['y'],
            c='red', s=300, marker='s',
            label='仓库', edgecolors='darkred', linewidths=2
        )
        
        ax.scatter(
            stations['x'], stations['y'],
            c='green', s=200, marker='^',
            label='停靠站点', edgecolors='darkgreen', linewidths=2
        )
        
        if not charging.empty:
            ax.scatter(
                charging['x'], charging['y'],
                c='orange', s=200, marker='D',
                label='充电站', edgecolors='darkorange', linewidths=2
            )
        
        # 绘制EV路径
        ev_route = ev_route_nodes
        for i in range(len(ev_route) - 1):
            node1 = nodes_df.iloc[ev_route[i]]
            node2 = nodes_df.iloc[ev_route[i + 1]]
            
            ax.plot(
                [node1['x'], node2['x']],
                [node1['y'], node2['y']],
                'b-',
                linewidth=2,
                alpha=0.7,
                label='EV路径' if i == 0 else ''
            )
            
            # 添加箭头
            dx = node2['x'] - node1['x']
            dy = node2['y'] - node1['y']
            ax.arrow(
                node1['x'], node1['y'],
                dx * 0.5, dy * 0.5,
                head_width=0.3, head_length=0.2,
                fc='blue', ec='blue', alpha=0.5
            )
        
        # 绘制UAV sorties
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        color_idx = 0
        
        for node_id_str, sorties in uav_sorties.items():
            node_id = int(node_id_str) if isinstance(node_id_str, str) else node_id_str
            node = nodes_df.iloc[node_id]
            
            for sortie in sorties:
                color = colors[color_idx % len(colors)]
                color_idx += 1
                
                # 兼容字典和对象格式
                if isinstance(sortie, dict):
                    sortie_customers = sortie.get('customers', [])
                else:
                    sortie_customers = sortie.customers
                
                # 从节点到第一个客户
                if sortie_customers:
                    first_customer = customers_df.iloc[sortie_customers[0]]
                    ax.plot(
                        [node['x'], first_customer['x']],
                        [node['y'], first_customer['y']],
                        '--',
                        color=color,
                        linewidth=1.5,
                        alpha=0.6,
                        label='UAV路径' if color_idx == 1 else ''
                    )
                    
                    # 客户之间
                    for i in range(len(sortie_customers) - 1):
                        c1 = customers_df.iloc[sortie_customers[i]]
                        c2 = customers_df.iloc[sortie_customers[i + 1]]
                        ax.plot(
                            [c1['x'], c2['x']],
                            [c1['y'], c2['y']],
                            '--',
                            color=color,
                            linewidth=1.5,
                            alpha=0.6
                        )
                    
                    # 从最后客户返回节点
                    last_customer = customers_df.iloc[sortie_customers[-1]]
                    ax.plot(
                        [last_customer['x'], node['x']],
                        [last_customer['y'], node['y']],
                        '--',
                        color=color,
                        linewidth=1.5,
                        alpha=0.6
                    )
        
        # 标注（使用中文字体）
        if self.font_prop:
            ax.set_xlabel('X 坐标 (km)', fontproperties=self.font_prop, fontsize=12)
            ax.set_ylabel('Y 坐标 (km)', fontproperties=self.font_prop, fontsize=12)
            ax.set_title(f'EV-UAV协同配送路径\n总成本: {total_cost:.2f}元', 
                        fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel('X 坐标 (km)', fontsize=12)
            ax.set_ylabel('Y 坐标 (km)', fontsize=12)
            ax.set_title(f'EV-UAV协同配送路径\n总成本: {total_cost:.2f}元', fontsize=14, fontweight='bold')
        
        # 去重图例（使用中文字体）
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if self.font_prop:
            ax.legend(by_label.values(), by_label.keys(), loc='best', prop=self.font_prop, fontsize=10)
        else:
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"路径图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_convergence(
        self,
        history: List[float],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制收敛曲线
        
        Args:
            history: 目标函数历史值
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(history) + 1)
        ax.plot(iterations, history, 'b-', linewidth=2, label='最优解')
        
        # 使用中文字体
        if self.font_prop:
            ax.set_xlabel('迭代次数', fontproperties=self.font_prop, fontsize=12)
            ax.set_ylabel('目标函数值', fontproperties=self.font_prop, fontsize=12)
            ax.set_title('ALNS算法收敛曲线', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
            ax.legend(prop=self.font_prop, fontsize=10)
        else:
            ax.set_xlabel('迭代次数', fontsize=12)
            ax.set_ylabel('目标函数值', fontsize=12)
            ax.set_title('ALNS算法收敛曲线', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"收敛曲线已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_cost_breakdown(
        self,
        metrics: Dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制成本分解饼图
        
        Args:
            metrics: 性能指标字典
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 提取各项成本
        costs = {
            'EV固定成本': metrics.get('ev_cost', 0),
            'UAV固定成本': metrics.get('uav_cost', 0),
            '能耗成本': metrics.get('energy_cost', 0),
            '时间成本': metrics.get('time_cost', 0),
            '碳排放成本': metrics.get('carbon_cost', 0),
            '延迟惩罚': metrics.get('delay_penalty', 0)
        }
        
        # 过滤掉零值
        costs = {k: v for k, v in costs.items() if v > 0}
        
        labels = list(costs.keys())
        sizes = list(costs.values())
        colors = sns.color_palette('pastel', len(labels))
        
        # 绘制饼图（使用中文字体）
        textprops = {'fontsize': 11}
        if self.font_prop:
            textprops['fontproperties'] = self.font_prop
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops=textprops
        )
        
        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 设置标题（使用中文字体）
        if self.font_prop:
            ax.set_title('成本分解', fontproperties=self.font_prop, fontsize=14, fontweight='bold')
        else:
            ax.set_title('成本分解', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"成本分解图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制多个方法的对比图
        
        Args:
            results: {方法名: metrics字典}
            save_path: 保存路径
            show: 是否显示
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        methods = list(results.keys())
        
        # 1. 总成本对比
        ax = axes[0, 0]
        costs = [results[m]['total_cost'] for m in methods]
        bars = ax.bar(methods, costs, color=sns.color_palette('muted', len(methods)))
        ax.set_ylabel('总成本 (元)', fontsize=11)
        ax.set_title('总成本对比', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9
            )
        
        # 2. 距离对比
        ax = axes[0, 1]
        distances = [results[m].get('total_distance', 0) for m in methods]
        bars = ax.bar(methods, distances, color=sns.color_palette('bright', len(methods)))
        ax.set_ylabel('总距离 (km)', fontsize=11)
        ax.set_title('总距离对比', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9
            )
        
        # 3. 能耗对比
        ax = axes[1, 0]
        energies = [results[m].get('total_energy', 0) for m in methods]
        bars = ax.bar(methods, energies, color=sns.color_palette('dark', len(methods)))
        ax.set_ylabel('总能耗 (kWh)', fontsize=11)
        ax.set_title('总能耗对比', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9
            )
        
        # 4. 时间对比
        ax = axes[1, 1]
        times = [results[m].get('total_time', 0) for m in methods]
        bars = ax.bar(methods, times, color=sns.color_palette('pastel', len(methods)))
        ax.set_ylabel('总时间 (分钟)', fontsize=11)
        ax.set_title('总时间对比', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def export_results(
        solution,
        metrics: Dict,
        save_dir: str,
        method_name: str
    ):
        """
        导出结果到文件
        
        Args:
            solution: Solution对象
            metrics: 性能指标
            save_dir: 保存目录
            method_name: 方法名称
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON格式
        result_dict = {
            'method': method_name,
            'solution': solution.to_dict(),
            'metrics': metrics
        }
        
        with open(save_path / f'{method_name}_solution.json', 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式
        metrics_df = pd.DataFrame([metrics])
        metrics_df.insert(0, 'method', method_name)
        metrics_df.to_csv(save_path / f'{method_name}_metrics.csv', index=False)
        
        print(f"结果已导出到: {save_path}")


def main():
    """测试可视化"""
    from data_generator import DataGenerator
    from alns_solver import ALNSSolver
    
    print("生成测试数据...")
    gen = DataGenerator('config/params_small.yaml')
    dataset = gen.generate_dataset()
    
    print("\n运行ALNS算法...")
    alns = ALNSSolver(dataset, dataset['config'])
    solution = alns.solve(max_iterations=500)
    
    print("\n生成可视化...")
    viz = Visualizer(dataset, dataset['config'])
    
    # 绘制路径
    viz.plot_solution(solution, save_path='data/results/test_route.png', show=False)
    
    # 绘制收敛曲线
    viz.plot_convergence(alns.best_history, save_path='data/results/test_convergence.png', show=False)
    
    # 绘制成本分解
    metrics = solution.evaluate()
    viz.plot_cost_breakdown(metrics, save_path='data/results/test_cost_breakdown.png', show=False)
    
    # 导出结果
    viz.export_results(solution, metrics, 'data/results', 'ALNS')
    
    print("\n可视化完成！")


if __name__ == '__main__':
    main()
