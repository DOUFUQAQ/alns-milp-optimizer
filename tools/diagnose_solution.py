"""诊断一份 ALNS/MILP 解的健康度。

读取 `experiments/*` 写出的结果目录（里面有 `solution.json`、`nodes.csv`、
`customers.csv`），检查几件最容易出问题的事：UAV 有没有被用上、
EV 固定成本是不是吃掉了目标函数、总距离是否小到不合理。

用法::

    python tools/diagnose_solution.py --data-dir data/results/real_world
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def load_result(data_dir):
    nodes = pd.read_csv(os.path.join(data_dir, "nodes.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    with open(os.path.join(data_dir, "solution.json"), "r", encoding="utf-8") as handle:
        solution = json.load(handle)
    return nodes, customers, solution


def customer_spread(customers):
    """客户两两距离的统计（坐标原始单位，不做单位换算）。"""
    if "lon" in customers.columns and "lat" in customers.columns:
        coords = customers[["lon", "lat"]].values
    else:
        coords = customers[["x", "y"]].values
    distances = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            distances.append(float(np.linalg.norm(coords[i] - coords[j])))
    distances = np.array(distances) if distances else np.array([0.0])
    return {
        "mean": float(np.mean(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "std": float(np.std(distances)),
    }


def describe_structure(solution):
    """解的结构。

    `Solution.to_dict()` 把 `uav_sorties` 放在**顶层**（与 `ev_route` 同级），
    JSON 序列化之后站点 key 会变成字符串；这里也兼容把它塞进 `ev_route` 的旧格式。
    """
    ev_route = solution.get("ev_route", {}) or {}
    sorties = solution.get("uav_sorties")
    if sorties is None:
        sorties = ev_route.get("uav_sorties", {})
    sorties = sorties or {}
    return {
        "ev_nodes": ev_route.get("nodes", []),
        "direct_customers": len(ev_route.get("direct_customers", []) or []),
        "num_sorties": sum(len(items) for items in sorties.values()),
        "sorties": sorties,
    }


def diagnose(metrics, structure, spread):
    issues = []
    total = metrics.get("total_cost", 0.0) or 0.0

    if metrics.get("uav_cost", 0) == 0:
        issues.append("UAV 完全没被使用（uav_cost=0），先检查初始解与 uav.fixed_cost")
    if total > 0 and metrics.get("ev_cost", 0) / total > 0.8:
        issues.append(
            "EV 固定成本占 %.0f%%，目标值被常数主导，ALNS 几乎没有优化空间"
            % (metrics["ev_cost"] / total * 100)
        )
    if metrics.get("total_distance", 0) < 10:
        issues.append("总距离小于 10 km，区域可能太小或坐标单位不对")
    if len(structure["ev_nodes"]) <= 2:
        issues.append("EV 只访问了 1-2 个节点，可能不需要多站点")
    if spread["std"] == 0.0:
        issues.append("所有客户坐标相同，数据生成可能出错了")
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description="诊断一份解的结构与成本构成")
    parser.add_argument(
        "--data-dir",
        default="data/results/real_world",
        help="包含 solution.json / nodes.csv / customers.csv 的目录",
    )
    args = parser.parse_args(argv)

    try:
        nodes, customers, solution = load_result(args.data_dir)
    except FileNotFoundError as error:
        print("读取失败（目录或文件不存在）: %s" % error)
        print("先把 experiments/run_real_world.py 跑一遍生成结果目录。")
        return 1

    metrics = solution.get("metrics", {})
    structure = describe_structure(solution)
    spread = customer_spread(customers)

    print("=" * 78)
    print("目录: %s" % args.data_dir)
    print("=" * 78)
    print("节点数: %d   客户数: %d" % (len(nodes), len(customers)))
    print("客户间距: 均值 %.4f / 最小 %.4f / 最大 %.4f" % (
        spread["mean"], spread["min"], spread["max"]))

    print("\n[解结构]")
    print("  EV 路径: %s" % structure["ev_nodes"])
    print("  EV 直送客户: %d" % structure["direct_customers"])
    print("  UAV 架次: %d" % structure["num_sorties"])
    for node, sorties in structure["sorties"].items():
        print("    站点 %s: %d 个架次" % (node, len(sorties)))

    print("\n[成本]")
    for key in ("total_cost", "ev_cost", "uav_cost", "energy_cost",
                "time_cost", "carbon_cost", "delay_penalty"):
        if key in metrics:
            print("  %-14s %10.2f" % (key, metrics[key]))
    if metrics.get("total_cost"):
        for key in ("ev_cost", "uav_cost", "energy_cost", "time_cost", "carbon_cost"):
            if key in metrics:
                share = metrics[key] / metrics["total_cost"] * 100
                print("  %-14s %9.1f%%" % ("占比 " + key, share))
    print("  总距离 %.2f km，总时间 %.2f min，总能耗 %.2f kWh" % (
        metrics.get("total_distance", 0.0),
        metrics.get("total_time", 0.0),
        metrics.get("total_energy", 0.0),
    ))

    print("\n[诊断]")
    issues = diagnose(metrics, structure, spread)
    if issues:
        for issue in issues:
            print("  ! %s" % issue)
    else:
        print("  未发现明显异常")

    return 0


if __name__ == "__main__":
    sys.exit(main())
