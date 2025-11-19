"""
EV-UAV协同物流配送优化系统 - 主入口
"""

import sys
import argparse
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_generator import DataGenerator
from src.milp_model import MILPModel
from src.alns_solver import ALNSSolver

try:  # 可视化是可选依赖（matplotlib / seaborn）
    from src import Visualizer
except ImportError:
    Visualizer = None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='EV-UAV协同物流配送路径优化系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成数据集
  python main.py generate --config config/params_small.yaml --output data/processed/test
  
  # 运行MILP求解（小规模）
  python main.py milp --config config/params_small.yaml
  
  # 运行ALNS求解（大规模）
  python main.py alns --config config/params_large.yaml --iterations 5000
  
  # 运行完整实验
  python main.py experiment --type small
  python main.py experiment --type large
  python main.py experiment --type all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 生成数据集命令
    gen_parser = subparsers.add_parser('generate', help='生成数据集')
    gen_parser.add_argument('--config', required=True, help='配置文件路径')
    gen_parser.add_argument('--output', required=True, help='输出目录')
    
    # MILP求解命令
    milp_parser = subparsers.add_parser('milp', help='MILP精确求解')
    milp_parser.add_argument('--config', required=True, help='配置文件路径')
    milp_parser.add_argument('--data', help='数据集目录（可选）')
    milp_parser.add_argument('--output', default='data/results/milp', help='输出目录')
    
    # ALNS求解命令
    alns_parser = subparsers.add_parser('alns', help='ALNS启发式求解')
    alns_parser.add_argument('--config', required=True, help='配置文件路径')
    alns_parser.add_argument('--data', help='数据集目录（可选）')
    alns_parser.add_argument('--iterations', type=int, help='最大迭代次数')
    alns_parser.add_argument('--output', default='data/results/alns', help='输出目录')
    
    # 实验命令
    exp_parser = subparsers.add_parser('experiment', help='运行实验')
    exp_parser.add_argument(
        '--type',
        choices=['small', 'large', 'sensitivity', 'all'],
        required=True,
        help='实验类型'
    )
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        # 生成数据集
        print("="*80)
        print("生成数据集")
        print("="*80)
        
        gen = DataGenerator(args.config)
        dataset = gen.generate_dataset(save_dir=args.output)
        
        print(f"\n✓ 数据集生成完成！")
        print(f"  节点数: {len(dataset['nodes'])}")
        print(f"  客户数: {len(dataset['customers'])}")
        print(f"  保存位置: {args.output}")
    
    elif args.command == 'milp':
        # MILP求解
        print("="*80)
        print("MILP精确求解")
        print("="*80)
        
        # 加载或生成数据集
        if args.data:
            dataset = DataGenerator.load_dataset(args.data)
        else:
            gen = DataGenerator(args.config)
            dataset = gen.generate_dataset()
        
        # 求解
        milp = MILPModel(dataset, dataset['config'])
        try:
            solution = milp.solve()
        except RuntimeError as error:
            print(f"\n✗ {error}")
            return 1
        
        if solution:
            # 评估
            metrics = solution.evaluate()
            
            # 可视化（matplotlib / seaborn 是可选依赖）
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if Visualizer is None:
                print("  跳过可视化：未安装 matplotlib / seaborn")
            else:
                viz = Visualizer(dataset, dataset['config'])
                viz.plot_solution(solution, str(output_path / 'route.png'), show=False)
                viz.plot_cost_breakdown(metrics, str(output_path / 'cost_breakdown.png'), show=False)
                viz.export_results(solution, metrics, str(output_path), 'MILP')
            
            print(f"\n✓ 求解完成！结果保存在: {output_path}")
    
    elif args.command == 'alns':
        # ALNS求解
        print("="*80)
        print("ALNS启发式求解")
        print("="*80)
        
        # 加载或生成数据集
        if args.data:
            dataset = DataGenerator.load_dataset(args.data)
        else:
            gen = DataGenerator(args.config)
            dataset = gen.generate_dataset()
        
        # 求解
        alns = ALNSSolver(dataset, dataset['config'])
        solution = alns.solve(max_iterations=args.iterations)
        
        if solution:
            # 评估
            metrics = solution.evaluate()
            
            # 可视化（matplotlib / seaborn 是可选依赖）
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if Visualizer is None:
                print("  跳过可视化：未安装 matplotlib / seaborn")
            else:
                viz = Visualizer(dataset, dataset['config'])
                viz.plot_solution(solution, str(output_path / 'route.png'), show=False)
                viz.plot_convergence(alns.best_history, str(output_path / 'convergence.png'), show=False)
                viz.plot_cost_breakdown(metrics, str(output_path / 'cost_breakdown.png'), show=False)
                viz.export_results(solution, metrics, str(output_path), 'ALNS')
            
            print(f"\n✓ 求解完成！结果保存在: {output_path}")
    
    elif args.command == 'experiment':
        # 运行实验
        sys.path.insert(0, 'experiments')
        
        if args.type == 'small':
            from run_small_scale import run_small_scale_experiment
            run_small_scale_experiment('config/params_small.yaml')
        
        elif args.type == 'large':
            from run_large_scale import run_large_scale_experiment
            run_large_scale_experiment('config/params_large.yaml')
        
        elif args.type == 'sensitivity':
            from sensitivity_analysis import main as sensitivity_main
            sensitivity_main()
        
        elif args.type == 'all':
            from run_all_experiments import run_all_experiments
            run_all_experiments()
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
