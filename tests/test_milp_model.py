"""MILP 模型模块。

Gurobi 是商业求解器，CI 里通常没有 license，所以这里分两层：
  * 不需要求解器就能测的部分（构造、UAV 距离矩阵、统计接口、缺依赖时的报错）
    始终执行；
  * 真正调用求解器的用例用 skipUnless 跳过。
"""
import unittest

import numpy as np

from src.milp_model import GUROBI_AVAILABLE, MILPModel
from tests.helpers import make_config, make_tiny_dataset


class TestModelWithoutSolver(unittest.TestCase):
    def setUp(self):
        self.dataset = make_tiny_dataset()
        self.config = make_config()
        self.model = MILPModel(self.dataset, self.config)

    def test_construct_does_not_need_gurobi(self):
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.solution)
        self.assertIsNone(self.model.obj_value)
        self.assertEqual(self.model.solve_time, 0.0)

    def test_uav_customer_distance_matrix(self):
        distances = self.model._compute_uav_customer_distances()
        nodes = self.dataset["nodes"]
        customers = self.dataset["customers"]

        self.assertEqual(distances.shape, (len(nodes), len(customers)))
        for i in range(len(nodes)):
            for c in range(len(customers)):
                expected = np.hypot(
                    nodes.iloc[i]["x"] - customers.iloc[c]["x"],
                    nodes.iloc[i]["y"] - customers.iloc[c]["y"],
                )
                self.assertAlmostEqual(distances[i, c], expected, places=9)

    def test_uav_distance_is_not_scaled(self):
        """这个矩阵用的是原始坐标单位（米），没有除 1000。"""
        distances = self.model._compute_uav_customer_distances()
        # 站点 1 (0,2000) → 客户 0 (0,3000) 相距 1000 米
        self.assertAlmostEqual(distances[1, 0], 1000.0, places=9)

    def test_statistics_empty_before_build(self):
        self.assertEqual(self.model.get_statistics(), {})

    @unittest.skipIf(GUROBI_AVAILABLE, "本机装了 Gurobi，测不到缺依赖的分支")
    def test_build_model_without_gurobi_raises(self):
        with self.assertRaises(RuntimeError) as context:
            self.model.build_model()
        message = str(context.exception)
        self.assertIn("Gurobi", message)
        self.assertIn("gurobipy", message)

    @unittest.skipIf(GUROBI_AVAILABLE, "本机装了 Gurobi，测不到缺依赖的分支")
    def test_solve_without_gurobi_raises(self):
        with self.assertRaises(RuntimeError):
            self.model.solve()


@unittest.skipUnless(GUROBI_AVAILABLE, "需要 Gurobi license")
class TestModelWithSolver(unittest.TestCase):
    def test_build_and_solve_tiny_instance(self):
        dataset = make_tiny_dataset()
        config = make_config(solver={"time_limit": 30, "mip_gap": 0.01,
                                     "threads": 1, "log_output": False})
        model = MILPModel(dataset, config)

        solution = model.solve()

        self.assertIsNotNone(solution)
        stats = model.get_statistics()
        self.assertGreater(stats["num_variables"], 0)
        self.assertGreater(stats["num_constraints"], 0)
        self.assertIsNotNone(stats["objective_value"])


if __name__ == "__main__":
    unittest.main()
