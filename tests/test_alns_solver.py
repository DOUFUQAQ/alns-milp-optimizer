"""ALNS 主循环：小规模实例上能在有限迭代内收敛到一个可行解。

除了"跑得通"，这里还固定住几条性质：
  * 初始解结构合法（客户全覆盖、EV 路径首尾是仓库）
  * 迭代过程中 best_history 单调不增、长度等于迭代次数
  * 接受准则 / 算子权重更新 / 轮盘赌选择的边界行为
  * 同一种子下结果可复现
"""
import contextlib
import io
import random
import unittest

import numpy as np

from src.alns_solver import ALNSSolver
from tests.helpers import (
    integrity_problems,
    load_fixture_dataset,
    make_config,
    sortie_demands,
)


def quiet(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return func(*args, **kwargs)


class ALNSTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_fixture_dataset()
        cls.config = make_config(
            alns={
                "max_iterations": 20,
                "segment_size": 5,
                "weights": {"start": [10, 5, 2, 1], "decay": 0.95},
                "temperature": {"start": 10000, "end": 1, "cooling_rate": 0.9},
                "destroy": {"min_removal": 3, "max_removal": 8, "removal_percentage": 0.2},
                "repair": {"noise_parameter": 0.05, "k_regret": 3},
                "output": {"log_interval": 100},
            }
        )

    def make_solver(self, config=None):
        return ALNSSolver(self.dataset, config or self.config)

    def make_initial(self, solver=None):
        solver = solver or self.make_solver()
        return quiet(solver._generate_initial_solution)


class TestInitialSolution(ALNSTestBase):
    def test_covers_all_customers(self):
        solution = self.make_initial()
        solution.evaluate()

        self.assertEqual(solution.unserved_customers, set())
        self.assertEqual(
            len(solution.served_customers), len(self.dataset["customers"])
        )
        self.assertEqual(integrity_problems(solution), [])

    def test_route_starts_and_ends_at_depot(self):
        solution = self.make_initial()
        self.assertEqual(solution.ev_route.nodes[0], 0)
        self.assertEqual(solution.ev_route.nodes[-1], 0)

    def test_uav_assignments_are_feasible(self):
        solution = self.make_initial()
        solution.evaluate()

        for sortie in [
            s for sorties in solution.ev_route.uav_sorties.values() for s in sorties
        ]:
            self.assertTrue(sortie.is_feasible)
            self.assertLessEqual(
                sortie.energy, self.config["uav"]["battery_capacity"]
            )
        for _, _, demand in sortie_demands(solution):
            self.assertLessEqual(demand, self.config["uav"]["load_capacity"])

    def test_drone_limit_per_station(self):
        solution = self.make_initial()
        for node, sorties in solution.ev_route.uav_sorties.items():
            self.assertLessEqual(len(sorties), self.config["uav"]["num_drones"])


class TestSolve(ALNSTestBase):
    def test_converges_to_feasible_solution(self):
        solver = self.make_solver()
        random.seed(0)
        np.random.seed(0)

        solution = quiet(solver.solve, max_iterations=20)

        self.assertIsNotNone(solution)
        solution.evaluate()
        self.assertTrue(solution.is_feasible, solution.infeasibility_reasons)
        self.assertTrue(np.isfinite(solution.objective()))
        self.assertGreater(solution.objective(), 0.0)

    def test_history_bookkeeping(self):
        solver = self.make_solver()
        random.seed(1)
        np.random.seed(1)

        quiet(solver.solve, max_iterations=15)

        self.assertEqual(len(solver.best_history), 15)
        self.assertEqual(len(solver.iteration_history), 15)
        self.assertAlmostEqual(solver.best_solution.objective(), solver.best_cost, places=9)

    def test_best_history_is_non_increasing(self):
        solver = self.make_solver()
        random.seed(2)
        np.random.seed(2)

        quiet(solver.solve, max_iterations=15)

        for previous, current in zip(solver.best_history, solver.best_history[1:]):
            self.assertLessEqual(current, previous + 1e-9)

    def test_never_worse_than_initial_solution(self):
        solver = self.make_solver()
        random.seed(3)
        np.random.seed(3)
        initial = self.make_initial(self.make_solver())
        initial.evaluate()
        initial_cost = initial.objective()

        solution = quiet(solver.solve, max_iterations=10)

        self.assertLessEqual(solution.objective(), initial_cost + 1e-9)

    def test_all_customers_served_after_solve(self):
        solver = self.make_solver()
        random.seed(4)
        np.random.seed(4)

        solution = quiet(solver.solve, max_iterations=12)

        self.assertEqual(solution.unserved_customers, set())
        self.assertEqual(integrity_problems(solution), [])

    def test_deterministic_with_same_seed(self):
        costs = []
        for _ in range(2):
            solver = self.make_solver()
            random.seed(99)
            np.random.seed(99)
            solution = quiet(solver.solve, max_iterations=8)
            costs.append(solution.objective())

        self.assertAlmostEqual(costs[0], costs[1], places=6)

    def test_temperature_cools_down(self):
        solver = self.make_solver()
        random.seed(5)
        np.random.seed(5)

        quiet(solver.solve, max_iterations=10)

        self.assertLess(solver.temperature, solver.initial_temp)
        self.assertGreater(solver.temperature, 0.0)

    def test_statistics(self):
        solver = self.make_solver()
        random.seed(6)
        np.random.seed(6)
        quiet(solver.solve, max_iterations=6)

        stats = solver.get_statistics()
        self.assertEqual(stats["num_iterations"], 6)
        self.assertEqual(len(stats["destroy_weights"]), 4)
        self.assertEqual(len(stats["repair_weights"]), 2)
        self.assertGreaterEqual(stats["solve_time"], 0.0)


class TestAcceptanceRule(ALNSTestBase):
    def setUp(self):
        self.solver = self.make_solver()
        self.solver.best_cost = 100.0
        self.solver.temperature = 1000.0

    def test_infeasible_is_rejected(self):
        accept, reason = self.solver._accept_solution(100.0, float("inf"))
        self.assertFalse(accept)
        self.assertEqual(reason, "infeasible")

    def test_new_global_best(self):
        accept, reason = self.solver._accept_solution(100.0, 90.0)
        self.assertTrue(accept)
        self.assertEqual(reason, "new_best")

    def test_better_but_not_best(self):
        self.solver.best_cost = 50.0
        accept, reason = self.solver._accept_solution(100.0, 90.0)
        self.assertTrue(accept)
        self.assertEqual(reason, "better")

    def test_worse_solution_rejected_when_cold(self):
        self.solver.temperature = 0.01
        accept, reason = self.solver._accept_solution(100.0, 10_000.0)
        self.assertFalse(accept)
        self.assertEqual(reason, "rejected")

    def test_worse_solution_accepted_when_hot(self):
        self.solver.temperature = 1e6
        accept, reason = self.solver._accept_solution(100.0, 100.5)
        self.assertTrue(accept)
        self.assertEqual(reason, "accepted")

    def test_score_mapping(self):
        weights = self.config["alns"]["weights"]["start"]
        self.assertEqual(self.solver._calculate_score("new_best"), weights[0])
        self.assertEqual(self.solver._calculate_score("better"), weights[1])
        self.assertEqual(self.solver._calculate_score("accepted"), weights[2])
        self.assertEqual(self.solver._calculate_score("rejected"), weights[3])
        self.assertEqual(self.solver._calculate_score("unknown"), 0)


class TestOperatorSelection(ALNSTestBase):
    def test_select_returns_valid_index(self):
        np.random.seed(0)
        solver = self.make_solver()
        weights = np.ones(4)
        for _ in range(50):
            index = solver._select_operator(weights)
            self.assertIn(index, range(4))

    def test_zero_weights_fall_back_to_uniform(self):
        np.random.seed(0)
        solver = self.make_solver()
        index = solver._select_operator(np.zeros(4))
        self.assertIn(index, range(4))

    def test_weight_update_resets_counters(self):
        solver = self.make_solver()
        solver.destroy_scores = np.array([10.0, 0.0, 0.0, 0.0])
        solver.destroy_usage = np.array([2.0, 0.0, 0.0, 0.0])
        solver.repair_scores = np.array([5.0, 0.0])
        solver.repair_usage = np.array([5.0, 0.0])
        old_destroy = solver.destroy_weights[0]

        solver._update_weights()

        self.assertNotAlmostEqual(solver.destroy_weights[0], old_destroy)
        np.testing.assert_allclose(solver.destroy_usage, 0.0)
        np.testing.assert_allclose(solver.destroy_scores, 0.0)
        np.testing.assert_allclose(solver.repair_usage, 0.0)
        np.testing.assert_allclose(solver.repair_scores, 0.0)


class TestDestroyRepairStep(ALNSTestBase):
    def test_step_keeps_solution_complete(self):
        solver = self.make_solver()
        solution = self.make_initial()
        solution.evaluate()

        for destroy_idx in range(4):
            for repair_idx in range(2):
                with self.subTest(destroy=destroy_idx, repair=repair_idx):
                    random.seed(destroy_idx * 10 + repair_idx)
                    np.random.seed(destroy_idx * 10 + repair_idx)
                    candidate = quiet(
                        solver._destroy_repair,
                        solution.copy(),
                        destroy_idx,
                        repair_idx,
                        -1,
                    )
                    self.assertEqual(candidate.unserved_customers, set())
                    self.assertEqual(integrity_problems(candidate), [])

    def test_step_never_leaves_unserved_customers(self):
        """修复算子即便失败，_destroy_repair 也会把剩余客户兜底给 EV。"""
        solver = self.make_solver()
        solution = self.make_initial()
        random.seed(0)
        np.random.seed(0)

        candidate = quiet(solver._destroy_repair, solution, 0, 1, -1)

        self.assertEqual(candidate.unserved_customers, set())
        self.assertEqual(
            len(candidate.served_customers), len(self.dataset["customers"])
        )


if __name__ == "__main__":
    unittest.main()
