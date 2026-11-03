"""Solution 的代价计算与可行性校验。

这里的小算例是手工设计的，所有期望值都能用纸笔算出来：

    节点 0 = 仓库 (0,0)     节点 1 = 站点 (0,2000)     节点 2 = 充电站 (2000,0)   [单位：米]
    客户 c0 (0,3000) demand=1.0   c1 (0,5000) demand=2.0   c2 (3000,0) demand=4.0

  EV 路径 0→1→0 = 4 km，行驶时间 4/40*60 = 6 min，能耗 4*0.2 = 0.8 kWh
  站点 1 往返 c0 = 2*1000 m = 2 km，UAV 能耗 2*0.03 = 0.06 kWh
"""
import unittest

from src.solution import Solution, UAVSortie
from tests.helpers import integrity_problems, make_tiny_dataset


class SolutionSetupMixin:
    def build(self, config_overrides=None):
        dataset = make_tiny_dataset()
        config = dataset["config"]
        solution = Solution(dataset, config)
        solution.ev_route.nodes = [0, 1, 0]
        return solution

    @staticmethod
    def serve_all(solution, direct=(), sorties=None):
        """把客户按 direct + sorties 的分配塞进解里。"""
        solution.ev_route.direct_customers = list(direct)
        solution.ev_route.uav_sorties = sorties or {}
        served = set(direct)
        for node_sorties in (sorties or {}).values():
            for sortie in node_sorties:
                served.update(sortie.customers)
        solution.served_customers = served
        solution.unserved_customers = set(range(len(solution.dataset["customers"]))) - served
        return solution


class TestEvaluateEVRoute(SolutionSetupMixin, unittest.TestCase):
    def test_ev_route_distance_time_energy(self):
        solution = self.build()
        self.serve_all(solution, direct=(0, 1, 2))
        metrics = solution.evaluate()

        self.assertTrue(metrics["is_feasible"])
        self.assertAlmostEqual(metrics["total_distance"], 4.0, places=9)
        self.assertAlmostEqual(metrics["total_time"], 6.0, places=9)
        self.assertAlmostEqual(metrics["total_energy"], 0.8, places=9)
        self.assertAlmostEqual(metrics["ev_cost"], 200.0, places=9)
        # 只有 EV 在动，所以各项成本都能直接算出来
        self.assertAlmostEqual(metrics["energy_cost"], 0.8, places=9)
        self.assertAlmostEqual(metrics["time_cost"], 3.0, places=9)
        self.assertAlmostEqual(metrics["carbon_cost"], 0.8 * 0.5 * 0.3, places=9)

    def test_direct_customers_add_no_distance(self):
        """已知实现：EV 直接配送的客户不贡献任何距离/能耗。

        `_evaluate_direct_delivery()` 是空实现，注释里写的"成本已包含在 EV
        路径评估中"并不成立——所以 3 个客户全部直送，总距离仍然只有 EV 路径
        本身的 4 km。
        """
        solution = self.build()
        self.serve_all(solution, direct=(0, 1, 2))
        metrics = solution.evaluate()

        self.assertTrue(metrics["is_feasible"])
        self.assertAlmostEqual(metrics["total_distance"], 4.0, places=9)
        self.assertAlmostEqual(metrics["total_energy"], 0.8, places=9)


class TestEvaluateUAVSorties(SolutionSetupMixin, unittest.TestCase):
    def test_single_sortie_cost(self):
        solution = self.build()
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[0])
        self.serve_all(solution, direct=(1, 2), sorties={1: [sortie]})
        metrics = solution.evaluate()

        self.assertTrue(metrics["is_feasible"])
        # sortie: 往返 2 km，0.06 kWh
        self.assertAlmostEqual(sortie.distance, 2.0, places=9)
        self.assertAlmostEqual(sortie.energy, 0.06, places=9)
        # 总距离 = EV 4 km + UAV 2 km
        self.assertAlmostEqual(metrics["total_distance"], 6.0, places=9)
        self.assertAlmostEqual(metrics["total_energy"], 0.86, places=9)
        # 成本 = EV固定200 + UAV固定1 + 能耗0.86 + 时间6*0.5 + 碳0.86*0.5*0.3
        self.assertAlmostEqual(metrics["total_cost"], 204.989, places=9)
        self.assertAlmostEqual(metrics["uav_cost"], 1.0, places=9)

    def test_sortie_return_time(self):
        solution = self.build()
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[0])
        self.serve_all(solution, direct=(1, 2), sorties={1: [sortie]})
        solution.evaluate()

        # 飞行 2 km / 60 km/h = 2 min，服务 1 客户 3 min，起飞 2 + 降落 1
        self.assertAlmostEqual(sortie.return_time, 2 + 3 + 2 + 1, places=9)

    def test_two_sorties_fixed_cost_per_launch(self):
        solution = self.build()
        sorties = [
            UAVSortie(uav_id=0, launch_node=1, customers=[0]),
            UAVSortie(uav_id=1, launch_node=1, customers=[1]),
        ]
        self.serve_all(solution, direct=(2,), sorties={1: sorties})
        metrics = solution.evaluate()

        self.assertAlmostEqual(metrics["total_distance"], 4.0 + 2.0 + 6.0, places=9)
        self.assertAlmostEqual(metrics["total_energy"], 0.8 + 0.06 + 0.18, places=9)
        self.assertAlmostEqual(metrics["uav_cost"], 2.0, places=9)
        self.assertAlmostEqual(metrics["total_cost"], 206.196, places=9)

    def test_uav_flight_time_not_counted_in_total_time(self):
        """已知实现：UAV 飞行时间只写入 sortie.return_time，没有累加到 total_time。"""
        solution = self.build()
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[0])
        self.serve_all(solution, direct=(1, 2), sorties={1: [sortie]})
        metrics = solution.evaluate()

        self.assertAlmostEqual(metrics["total_time"], 6.0, places=9)  # 只有 EV 的 6 min
        self.assertGreater(sortie.return_time, 0.0)

    def test_empty_sortie_is_infeasible(self):
        solution = self.build()
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[])
        self.serve_all(solution, direct=(0, 1, 2), sorties={1: [sortie]})
        solution.evaluate()

        self.assertFalse(sortie.is_feasible)
        self.assertFalse(solution.is_feasible)

    def test_evaluate_is_idempotent(self):
        """重复评估不能把指标累加（_reset_metrics 必须生效）。"""
        solution = self.build()
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[0])
        self.serve_all(solution, direct=(1, 2), sorties={1: [sortie]})

        first = solution.evaluate()
        second = solution.evaluate()

        self.assertAlmostEqual(first["total_cost"], second["total_cost"], places=9)
        self.assertAlmostEqual(first["total_distance"], second["total_distance"], places=9)
        self.assertAlmostEqual(first["total_energy"], second["total_energy"], places=9)


class TestInfeasibility(SolutionSetupMixin, unittest.TestCase):
    def test_unserved_customer(self):
        solution = self.build()
        self.serve_all(solution, direct=(0,), sorties={})

        metrics = solution.evaluate()
        self.assertFalse(metrics["is_feasible"])
        self.assertEqual(metrics["num_served"], 1)
        self.assertEqual(metrics["num_total"], 3)
        self.assertIn("Not all customers served", solution.infeasibility_reasons)
        self.assertEqual(solution.objective(), float("inf"))

    def test_uav_battery_violation(self):
        solution = self.build()
        solution.config["uav"]["battery_capacity"] = 0.01  # 0.06 kWh 的 sortie 装不下
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[0])
        self.serve_all(solution, direct=(1, 2), sorties={1: [sortie]})
        solution.evaluate()

        self.assertFalse(sortie.is_feasible)
        self.assertFalse(solution.is_feasible)
        self.assertTrue(
            any("battery" in reason for reason in solution.infeasibility_reasons),
            solution.infeasibility_reasons,
        )

    def test_uav_load_violation(self):
        solution = self.build()
        solution.config["uav"]["load_capacity"] = 1.5  # c1 需求 2.0 kg
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[1])
        self.serve_all(solution, direct=(0, 2), sorties={1: [sortie]})
        solution.evaluate()

        self.assertFalse(sortie.is_feasible)
        self.assertTrue(
            any("load capacity" in reason for reason in solution.infeasibility_reasons),
            solution.infeasibility_reasons,
        )

    def test_ev_battery_violation(self):
        solution = self.build()
        solution.config["ev"]["battery_capacity"] = 1.0
        solution.ev_route.nodes = [0, 1, 2, 0]  # 0.2*(2+2.83+2) = 1.37 kWh
        self.serve_all(solution, direct=(0, 1, 2))
        solution.evaluate()

        self.assertFalse(solution.is_feasible)
        self.assertIn("EV battery capacity exceeded", solution.infeasibility_reasons)

    def test_objective_returns_cost_when_feasible(self):
        solution = self.build()
        self.serve_all(solution, direct=(0, 1, 2))
        solution.evaluate()

        self.assertTrue(solution.is_feasible)
        self.assertAlmostEqual(solution.objective(), solution.total_cost, places=9)


class TestSolutionUtilities(SolutionSetupMixin, unittest.TestCase):
    def test_copy_is_deep(self):
        solution = self.build()
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[0])
        self.serve_all(solution, direct=(1, 2), sorties={1: [sortie]})
        solution.evaluate()

        clone = solution.copy()
        clone.ev_route.uav_sorties[1][0].customers.append(2)
        clone.served_customers.discard(0)

        self.assertEqual(solution.ev_route.uav_sorties[1][0].customers, [0])
        self.assertIn(0, solution.served_customers)
        self.assertIsNot(clone.ev_route, solution.ev_route)
        self.assertIsNot(clone.dataset, solution.dataset)
        self.assertAlmostEqual(clone.total_cost, solution.total_cost, places=9)

    def test_to_dict(self):
        solution = self.build()
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[0])
        self.serve_all(solution, direct=(1, 2), sorties={1: [sortie]})
        solution.evaluate()

        payload = solution.to_dict()
        self.assertEqual(payload["ev_route"]["nodes"], [0, 1, 0])
        self.assertEqual(payload["ev_route"]["direct_customers"], [1, 2])
        self.assertEqual(list(payload["uav_sorties"].keys()), [1])
        self.assertEqual(payload["uav_sorties"][1][0]["customers"], [0])
        self.assertIn("metrics", payload)
        self.assertAlmostEqual(payload["metrics"]["total_cost"], solution.total_cost, places=9)

    def test_metrics_keys(self):
        solution = self.build()
        self.serve_all(solution, direct=(0, 1, 2))
        metrics = solution.evaluate()

        expected = {
            "total_cost", "ev_cost", "uav_cost", "energy_cost", "time_cost",
            "carbon_cost", "delay_penalty", "total_distance", "total_time",
            "total_energy", "is_feasible", "num_served", "num_total",
        }
        self.assertEqual(set(metrics), expected)

    def test_repr(self):
        solution = self.build()
        self.serve_all(solution, direct=(0, 1, 2))
        solution.evaluate()

        text = repr(solution)
        self.assertIn("Solution(cost=", text)
        self.assertIn("served=3/3", text)

    def test_fresh_solution_is_empty(self):
        solution = Solution(make_tiny_dataset(), make_tiny_dataset()["config"])
        self.assertEqual(solution.ev_route.nodes, [0])
        self.assertEqual(solution.served_customers, set())
        self.assertEqual(solution.unserved_customers, {0, 1, 2})
        self.assertEqual(solution.total_cost, float("inf"))
        self.assertEqual(integrity_problems(solution), [])


if __name__ == "__main__":
    unittest.main()
