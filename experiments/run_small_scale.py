"""
小规模算例实验
使用MILP精确求解小规模EV-UAV协同配送问题
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import json
import time

from src.data_generator import DataGenerator
from src.milp_model import MILPModel
from src.visualizer import Visualizer


def run_small_scale_experiment(config_path: str, output_dir: str = 'data/results/small_scale'):
    """
    运行小规模实验
    
    Args:
        config_path: 配置文件路径
        output_dir: 输出目录
    """
    print("="*80)
    print("小规模算例实验 - MILP精确求解")
    print("="*80)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成数据集
    print("\n[步骤1] 生成数据集...")
    gen = DataGenerator(config_path)
    dataset = gen.generate_dataset(save_dir=str(output_path / 'dataset'))
    
    print(f"  - 节点数: {len(dataset['nodes'])}")
    print(f"  - 客户数: {len(dataset['customers'])}")
    
    # 2. 构建并求解MILP模型
    print("\n[步骤2] 构建MILP模型...")
    milp = MILPModel(dataset, dataset['config'])
    
    print("\n[步骤3] 求解MILP模型...")
    solution = milp.solve()
    
    if solution is None:
        print("❌ 求解失败！")
        return None
    
    # 3. 评估解
    print("\n[步骤4] 评估解...")
    metrics = solution.evaluate()
    
    print("\n性能指标:")
    print(f"  - 总成本: {metrics['total_cost']:.2f} 元")
    print(f"  - EV成本: {metrics['ev_cost']:.2f} 元")
    print(f"  - UAV成本: {metrics['uav_cost']:.2f} 元")
    print(f"  - 能耗成本: {metrics['energy_cost']:.2f} 元")
    print(f"  - 时间成本: {metrics['time_cost']:.2f} 元")
    print(f"  - 碳排放成本: {metrics['carbon_cost']:.2f} 元")
    print(f"  - 总距离: {metrics['total_distance']:.2f} km")
    print(f"  - 总能耗: {metrics['total_energy']:.2f} kWh")
    print(f"  - 总时间: {metrics['total_time']:.2f} 分钟")
    print(f"  - 服务客户数: {metrics['num_served']}/{metrics['num_total']}")
    print(f"  - 可行性: {'✓' if metrics['is_feasible'] else '✗'}")
    
    # 4. 可视化
    print("\n[步骤5] 生成可视化...")
    viz = Visualizer(dataset, dataset['config'])
    
    # 路径图
    viz.plot_solution(
        solution,
        save_path=str(output_path / 'milp_route.png'),
        show=False
    )
    
    # 成本分解
    viz.plot_cost_breakdown(
        metrics,
        save_path=str(output_path / 'milp_cost_breakdown.png'),
        show=False
    )
    
    # 5. 导出结果
    print("\n[步骤6] 导出结果...")
    
    # 求解统计
    stats = milp.get_statistics()
    print("\n求解统计:")
    print(f"  - 求解时间: {stats['solve_time']:.2f} 秒")
    print(f"  - 目标函数值: {stats['objective_value']:.2f}")
    print(f"  - MIP Gap: {stats.get('mip_gap', 0)*100:.2f}%")
    print(f"  - 变量数: {stats['num_variables']}")
    print(f"  - 约束数: {stats['num_constraints']}")
    
    # 保存结果
    viz.export_results(solution, metrics, str(output_path), 'MILP')
    
    # 保存统计信息
    with open(output_path / 'milp_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 实验完成！结果已保存到: {output_path}")
    
    return {
        'solution': solution,
        'metrics': metrics,
        'stats': stats
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行小规模MILP实验')
    parser.add_argument(
        '--config',
        type=str,
        default='config/params_small.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/small_scale',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    try:
        result = run_small_scale_experiment(args.config, args.output)
        
        if result:
            print("\n" + "="*80)
            print("实验成功完成！")
            print("="*80)
            return 0
        else:
            print("\n" + "="*80)
            print("实验失败！")
            print("="*80)
            return 1
    
    except Exception as e:
        print(f"\n❌ 实验出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
