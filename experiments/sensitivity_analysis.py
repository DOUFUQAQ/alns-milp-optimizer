"""
敏感性分析实验
分析关键参数对系统性能的影响
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import copy
import yaml

from src.data_generator import DataGenerator
from src.alns_solver import ALNSSolver
from src.visualizer import Visualizer


def sensitivity_analysis_uav_battery(
    base_config_path: str,
    output_dir: str,
    battery_range: list = [1.0, 1.5, 2.0, 2.5, 3.0]
):
    """
    UAV电池容量敏感性分析
    
    Args:
        base_config_path: 基础配置文件
        output_dir: 输出目录
        battery_range: 电池容量范围 (kWh)
    """
    print("\n" + "="*80)
    print("敏感性分析: UAV电池容量")
    print("="*80)
    
    output_path = Path(output_dir) / 'uav_battery'
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for battery in battery_range:
        print(f"\n测试电池容量: {battery} kWh")
        
        # 加载并修改配置
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['uav']['battery_capacity'] = battery
        
        # 生成数据集
        gen = DataGenerator(base_config_path)
        gen.config = config  # 使用修改后的配置
        dataset = gen.generate_dataset()
        
        # 运行ALNS
        alns = ALNSSolver(dataset, config)
        solution = alns.solve(max_iterations=1000)
        
        if solution:
            metrics = solution.evaluate()
            results.append({
                'battery_capacity': battery,
                'total_cost': metrics['total_cost'],
                'total_distance': metrics['total_distance'],
                'total_energy': metrics['total_energy'],
                'total_time': metrics['total_time'],
                'uav_cost': metrics['uav_cost'],
                'solve_time': alns.solve_time
            })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'results.csv', index=False)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 总成本
    axes[0, 0].plot(results_df['battery_capacity'], results_df['total_cost'], 'o-', linewidth=2)
    axes[0, 0].set_xlabel('UAV电池容量 (kWh)')
    axes[0, 0].set_ylabel('总成本 (元)')
    axes[0, 0].set_title('总成本 vs UAV电池容量')
    axes[0, 0].grid(alpha=0.3)
    
    # 总距离
    axes[0, 1].plot(results_df['battery_capacity'], results_df['total_distance'], 's-', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('UAV电池容量 (kWh)')
    axes[0, 1].set_ylabel('总距离 (km)')
    axes[0, 1].set_title('总距离 vs UAV电池容量')
    axes[0, 1].grid(alpha=0.3)
    
    # 总能耗
    axes[1, 0].plot(results_df['battery_capacity'], results_df['total_energy'], '^-', linewidth=2, color='green')
    axes[1, 0].set_xlabel('UAV电池容量 (kWh)')
    axes[1, 0].set_ylabel('总能耗 (kWh)')
    axes[1, 0].set_title('总能耗 vs UAV电池容量')
    axes[1, 0].grid(alpha=0.3)
    
    # UAV成本
    axes[1, 1].plot(results_df['battery_capacity'], results_df['uav_cost'], 'd-', linewidth=2, color='red')
    axes[1, 1].set_xlabel('UAV电池容量 (kWh)')
    axes[1, 1].set_ylabel('UAV成本 (元)')
    axes[1, 1].set_title('UAV成本 vs UAV电池容量')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ UAV电池容量敏感性分析完成！结果保存在: {output_path}")
    
    return results_df


def sensitivity_analysis_uav_number(
    base_config_path: str,
    output_dir: str,
    uav_range: list = [3, 5, 8, 10, 15]
):
    """
    UAV数量敏感性分析
    
    Args:
        base_config_path: 基础配置文件
        output_dir: 输出目录
        uav_range: UAV数量范围
    """
    print("\n" + "="*80)
    print("敏感性分析: UAV数量")
    print("="*80)
    
    output_path = Path(output_dir) / 'uav_number'
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for num_uav in uav_range:
        print(f"\n测试UAV数量: {num_uav}")
        
        # 加载并修改配置
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['uav']['num_drones'] = num_uav
        
        # 生成数据集
        gen = DataGenerator(base_config_path)
        gen.config = config
        dataset = gen.generate_dataset()
        
        # 运行ALNS
        alns = ALNSSolver(dataset, config)
        solution = alns.solve(max_iterations=1000)
        
        if solution:
            metrics = solution.evaluate()
            results.append({
                'num_drones': num_uav,
                'total_cost': metrics['total_cost'],
                'total_distance': metrics['total_distance'],
                'total_energy': metrics['total_energy'],
                'total_time': metrics['total_time'],
                'uav_cost': metrics['uav_cost'],
                'solve_time': alns.solve_time
            })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'results.csv', index=False)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(results_df['num_drones'], results_df['total_cost'], 'o-', linewidth=2)
    axes[0, 0].set_xlabel('UAV数量')
    axes[0, 0].set_ylabel('总成本 (元)')
    axes[0, 0].set_title('总成本 vs UAV数量')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(results_df['num_drones'], results_df['total_distance'], 's-', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('UAV数量')
    axes[0, 1].set_ylabel('总距离 (km)')
    axes[0, 1].set_title('总距离 vs UAV数量')
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(results_df['num_drones'], results_df['uav_cost'], '^-', linewidth=2, color='green')
    axes[1, 0].set_xlabel('UAV数量')
    axes[1, 0].set_ylabel('UAV成本 (元)')
    axes[1, 0].set_title('UAV成本 vs UAV数量')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(results_df['num_drones'], results_df['solve_time'], 'd-', linewidth=2, color='red')
    axes[1, 1].set_xlabel('UAV数量')
    axes[1, 1].set_ylabel('求解时间 (秒)')
    axes[1, 1].set_title('求解时间 vs UAV数量')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ UAV数量敏感性分析完成！结果保存在: {output_path}")
    
    return results_df


def sensitivity_analysis_customer_density(
    base_config_path: str,
    output_dir: str,
    customer_range: list = [50, 100, 150, 200, 250]
):
    """
    客户密度敏感性分析
    
    Args:
        base_config_path: 基础配置文件
        output_dir: 输出目录
        customer_range: 客户数量范围
    """
    print("\n" + "="*80)
    print("敏感性分析: 客户数量")
    print("="*80)
    
    output_path = Path(output_dir) / 'customer_density'
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for num_customers in customer_range:
        print(f"\n测试客户数量: {num_customers}")
        
        # 加载并修改配置
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['geography']['num_customers'] = num_customers
        
        # 生成数据集
        gen = DataGenerator(base_config_path)
        gen.config = config
        dataset = gen.generate_dataset()
        
        # 运行ALNS
        alns = ALNSSolver(dataset, config)
        solution = alns.solve(max_iterations=1000)
        
        if solution:
            metrics = solution.evaluate()
            results.append({
                'num_customers': num_customers,
                'total_cost': metrics['total_cost'],
                'total_distance': metrics['total_distance'],
                'total_energy': metrics['total_energy'],
                'total_time': metrics['total_time'],
                'solve_time': alns.solve_time
            })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'results.csv', index=False)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(results_df['num_customers'], results_df['total_cost'], 'o-', linewidth=2)
    axes[0, 0].set_xlabel('客户数量')
    axes[0, 0].set_ylabel('总成本 (元)')
    axes[0, 0].set_title('总成本 vs 客户数量')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(results_df['num_customers'], results_df['total_distance'], 's-', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('客户数量')
    axes[0, 1].set_ylabel('总距离 (km)')
    axes[0, 1].set_title('总距离 vs 客户数量')
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(results_df['num_customers'], results_df['total_energy'], '^-', linewidth=2, color='green')
    axes[1, 0].set_xlabel('客户数量')
    axes[1, 0].set_ylabel('总能耗 (kWh)')
    axes[1, 0].set_title('总能耗 vs 客户数量')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(results_df['num_customers'], results_df['solve_time'], 'd-', linewidth=2, color='red')
    axes[1, 1].set_xlabel('客户数量')
    axes[1, 1].set_ylabel('求解时间 (秒)')
    axes[1, 1].set_title('求解时间 vs 客户数量')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 客户数量敏感性分析完成！结果保存在: {output_path}")
    
    return results_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行敏感性分析')
    parser.add_argument(
        '--config',
        type=str,
        default='config/params_large.yaml',
        help='基础配置文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/sensitivity',
        help='输出目录'
    )
    parser.add_argument(
        '--analysis',
        type=str,
        choices=['battery', 'uav_num', 'customers', 'all'],
        default='all',
        help='分析类型'
    )
    
    args = parser.parse_args()
    
    try:
        output_dir = args.output
        
        if args.analysis in ['battery', 'all']:
            print("\n" + "="*80)
            print("执行UAV电池容量敏感性分析")
            print("="*80)
            sensitivity_analysis_uav_battery(args.config, output_dir)
        
        if args.analysis in ['uav_num', 'all']:
            print("\n" + "="*80)
            print("执行UAV数量敏感性分析")
            print("="*80)
            sensitivity_analysis_uav_number(args.config, output_dir)
        
        if args.analysis in ['customers', 'all']:
            print("\n" + "="*80)
            print("执行客户数量敏感性分析")
            print("="*80)
            sensitivity_analysis_customer_density(args.config, output_dir)
        
        print("\n" + "="*80)
        print("敏感性分析全部完成！")
        print("="*80)
        return 0
    
    except Exception as e:
        print(f"\n❌ 分析出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
