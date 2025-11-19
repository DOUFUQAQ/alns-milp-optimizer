"""
运行所有实验
包括小规模MILP、大规模ALNS和敏感性分析
"""

import sys
import argparse
from pathlib import Path
import time

# 导入实验模块
from run_small_scale import run_small_scale_experiment
from run_large_scale import run_large_scale_experiment
from sensitivity_analysis import (
    sensitivity_analysis_uav_battery,
    sensitivity_analysis_uav_number,
    sensitivity_analysis_customer_density
)


def run_all_experiments(output_base_dir: str = 'data/results'):
    """
    运行所有实验
    
    Args:
        output_base_dir: 输出基础目录
    """
    print("="*80)
    print("开始运行完整实验套件")
    print("="*80)
    
    total_start = time.time()
    results = {}
    
    # 1. 小规模MILP实验
    print("\n\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "实验 1: 小规模MILP" + " "*25 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    try:
        small_result = run_small_scale_experiment(
            'config/params_small.yaml',
            f'{output_base_dir}/small_scale'
        )
        results['small_milp'] = small_result
        print("✓ 小规模MILP实验完成")
    except Exception as e:
        print(f"✗ 小规模MILP实验失败: {e}")
        results['small_milp'] = None
    
    # 2. 大规模ALNS实验
    print("\n\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "实验 2: 大规模ALNS" + " "*25 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    try:
        large_result = run_large_scale_experiment(
            'config/params_large.yaml',
            f'{output_base_dir}/large_scale',
            max_iterations=5000
        )
        results['large_alns'] = large_result
        print("✓ 大规模ALNS实验完成")
    except Exception as e:
        print(f"✗ 大规模ALNS实验失败: {e}")
        results['large_alns'] = None
    
    # 3. 敏感性分析
    print("\n\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "实验 3: 敏感性分析" + " "*25 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    sensitivity_dir = f'{output_base_dir}/sensitivity'
    
    # 3.1 UAV电池容量
    print("\n--- 敏感性分析 3.1: UAV电池容量 ---")
    try:
        battery_result = sensitivity_analysis_uav_battery(
            'config/params_large.yaml',
            sensitivity_dir,
            battery_range=[1.0, 1.5, 2.0, 2.5, 3.0]
        )
        results['sensitivity_battery'] = battery_result
        print("✓ UAV电池容量敏感性分析完成")
    except Exception as e:
        print(f"✗ UAV电池容量敏感性分析失败: {e}")
        results['sensitivity_battery'] = None
    
    # 3.2 UAV数量
    print("\n--- 敏感性分析 3.2: UAV数量 ---")
    try:
        uav_num_result = sensitivity_analysis_uav_number(
            'config/params_large.yaml',
            sensitivity_dir,
            uav_range=[3, 5, 8, 10, 15]
        )
        results['sensitivity_uav_num'] = uav_num_result
        print("✓ UAV数量敏感性分析完成")
    except Exception as e:
        print(f"✗ UAV数量敏感性分析失败: {e}")
        results['sensitivity_uav_num'] = None
    
    # 3.3 客户数量
    print("\n--- 敏感性分析 3.3: 客户数量 ---")
    try:
        customer_result = sensitivity_analysis_customer_density(
            'config/params_large.yaml',
            sensitivity_dir,
            customer_range=[50, 100, 150, 200, 250]
        )
        results['sensitivity_customers'] = customer_result
        print("✓ 客户数量敏感性分析完成")
    except Exception as e:
        print(f"✗ 客户数量敏感性分析失败: {e}")
        results['sensitivity_customers'] = None
    
    # 总结
    total_time = time.time() - total_start
    
    print("\n\n" + "="*80)
    print("="*80)
    print(" "*30 + "实验总结")
    print("="*80)
    print("="*80)
    
    print(f"\n总运行时间: {total_time/60:.2f} 分钟")
    print("\n实验完成状态:")
    
    success_count = 0
    total_count = len(results)
    
    for name, result in results.items():
        status = "✓ 成功" if result is not None else "✗ 失败"
        print(f"  {name:30s} {status}")
        if result is not None:
            success_count += 1
    
    print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"\n所有结果已保存到: {output_base_dir}")
    
    print("\n" + "="*80)
    print("实验套件运行完成！")
    print("="*80)
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行所有实验')
    parser.add_argument(
        '--output',
        type=str,
        default='data/results',
        help='输出基础目录'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_all_experiments(args.output)
        return 0
    except Exception as e:
        print(f"\n❌ 实验套件出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
