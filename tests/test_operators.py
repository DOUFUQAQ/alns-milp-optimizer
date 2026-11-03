"""ALNS 破坏 / 修复算子的输入输出不变量。

破坏算子必须做到：只动自己该动的客户、返回集合大小可控、移完之后
served / unserved / 结构三者仍然自洽。

修复算子必须做到：把传进来的客户全部安置好、不重复配送、不违反载重约束。
"""
import contextlib
import io
import random
import unittest

from src.alns_solver import ALNSSolver
from src.operators import DestroyOperators, RepairOperators
from src.solution import Solution, UAVSortie
from tests.helpers import (
    integrity_problems,
    load_fixture_dataset,
    make_config,
    make_tiny_dataset,
    served_structure,
    sortie_demands,
)


def quiet(func, *args, **kwargs):
    """吞掉算子内部 print，保持测试输出干净。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return func(*args, **kwargs)


class FixtureSolutionMixin:
    """用仓库内置的 20 客户算例构造一个结构完整的初始解。"""

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_fixture_dataset()
        cls.config = make_config()
        # 用两个数据集各自的配置保持一致：算例自带 config，测试配置补上 alns 段
        cls.solution_template = quiet(
            ALNSSolver(cls.dataset, cls.config)._generate_initial_solution
        )

    def fresh_solution(self):
        return self.solution_template.copy()

    def assert_integrity(self, solution):
        self.assertEqual(integrity_problems(solution), [], "解结构不变量被破坏")


class TestInitialSolutionShape(FixtureSolutionMixin, unittest.TestCase):
    def test_every_customer_served_once(self):
        solution = self.fresh_solution()
        self.assert_integrity(solution)
        self.assertEqual(solution.unserved_customers, set())
        self.assertEqual(len(solution.served_customers), len(self.dataset["customers"]))

    def test_ev_route_visits_all_stations(self):
        solution = self.fresh_solution()
        stations = [
            int(row.node_id) for _, row in self.dataset["nodes"].iterrows()
            if row["type"] == "station"
        ]
        nodes = solution.ev_route.nodes
        self.assertEqual(nodes[0], 0)
        self.assertEqual(nodes[-1], 0)
        for station in stations:
            self.assertIn(station, nodes)

    def test_sortie_count_respects_drone_limit(self):
        solution = self.fresh_solution()
        limit = self.config["uav"]["num_drones"]
        for node, sorties in solution.ev_route.uav_sorties.items():
            self.assertLessEqual(len(sorties), limit, "站点 %s 的架次超过无人机数量" % node)

    def test_sorties_respect_uav_constraints(self):
        solution = self.fresh_solution()
        solution.evaluate()
        for node, customers, demand in sortie_demands(solution):
            self.assertLessEqual(demand, self.config["uav"]["load_capacity"])
        for sorties in solution.ev_route.uav_sorties.values():
            for sortie in sorties:
                self.assertLessEqual(sortie.energy, self.config["uav"]["battery_capacity"])


class TestDestroyOperators(FixtureSolutionMixin, unittest.TestCase):
    def test_random_removal_returns_exact_size(self):
        solution = self.fresh_solution()
        before = set(solution.served_customers)
        random.seed(7)

        removed = DestroyOperators.random_removal(solution, 5)

        self.assertEqual(len(removed), 5)
        self.assertTrue(removed <= before)
        self.assert_integrity(solution)
        self.assertEqual(len(solution.served_customers), len(before) - 5)
        self.assertTrue(removed <= solution.unserved_customers)
        self.assertFalse(removed & solution.served_customers)

    def test_random_removal_caps_at_served_count(self):
        solution = self.fresh_solution()
        total = len(solution.served_customers)
        random.seed(1)

        removed = DestroyOperators.random_removal(solution, total + 100)

        self.assertEqual(len(removed), total)
        self.assertEqual(solution.served_customers, set())

    def test_random_removal_on_empty_solution(self):
        solution = Solution(make_tiny_dataset(), make_config())
        self.assertEqual(DestroyOperators.random_removal(solution, 3), set())

    def test_worst_removal_picks_known_customers(self):
        solution = self.fresh_solution()
        before = set(solution.served_customers)
        random.seed(3)

        removed = DestroyOperators.worst_removal(solution, 4)

        self.assertEqual(len(removed), 4)
        self.assertTrue(removed <= before)
        self.assert_integrity(solution)

    def test_worst_removal_caps_at_known_customers(self):
        solution = self.fresh_solution()
        before = len(solution.served_customers)
        random.seed(3)

        removed = DestroyOperators.worst_removal(solution, before + 50)

        self.assertLessEqual(len(removed), before)
        self.assert_integrity(solution)

    def test_station_removal_clears_exactly_one_station(self):
        solution = self.fresh_solution()
        before_keys = set(solution.ev_route.uav_sorties.keys())
        before_customers = {
            node: [c for sortie in sorties for c in sortie.customers]
            for node, sorties in solution.ev_route.uav_sorties.items()
        }
        random.seed(11)

        removed = quiet(DestroyOperators.station_removal, solution, 5)

        after_keys = set(solution.ev_route.uav_sorties.keys())
        self.assertEqual(len(before_keys - after_keys), 1, "应当正好清空一个站点")
        cleared = (before_keys - after_keys).pop()
        self.assertEqual(removed, set(before_customers[cleared]))
        # 其它站点的客户不能被动到
        for node in before_keys - {cleared}:
            self.assertIn(node, after_keys)
        self.assert_integrity(solution)

    def test_station_removal_without_uav(self):
        solution = self.fresh_solution()
        solution.ev_route.uav_sorties = {}
        self.assertEqual(DestroyOperators.station_removal(solution, 5), set())

    def test_route_removal_returns_requested_size(self):
        solution = self.fresh_solution()
        before = set(solution.served_customers)
        random.seed(5)

        removed = DestroyOperators.route_removal(solution, 6)

        self.assertEqual(len(removed), 6)
        self.assertTrue(removed <= before)
        self.assert_integrity(solution)

    def test_route_removal_keeps_customers_geographically_close(self):
        """相邻移除：被移除的客户之间的最大距离，应当小于全体客户的平均距离。"""
        dataset = self.dataset
        coords = dataset["customers"][["x", "y"]].values
        random.seed(2)
        solution = self.fresh_solution()

        removed = DestroyOperators.route_removal(solution, 5)

        removed = list(removed)
        spread = max(
            sum((coords[i] - coords[j]) ** 2) ** 0.5
            for i in removed for j in removed
        )
        all_pairs = [
            sum((coords[i] - coords[j]) ** 2) ** 0.5
            for i in range(len(coords)) for j in range(len(coords)) if i < j
        ]
        self.assertLess(spread, max(all_pairs))
        self.assert_less_than_mean(spread, all_pairs)

    def assert_less_than_mean(self, value, values):
        mean = sum(values) / len(values)
        self.assertLess(value, mean, "相邻移除的客户分布得比平均水平还散")

    def test_all_destroy_operators_keep_partition(self):
        operators = [
            ("random", DestroyOperators.random_removal),
            ("worst", DestroyOperators.worst_removal),
            ("station", DestroyOperators.station_removal),
            ("route", DestroyOperators.route_removal),
        ]
        for name, operator in operators:
            for seed in (0, 4, 9):
                with self.subTest(operator=name, seed=seed):
                    random.seed(seed)
                    solution = self.fresh_solution()
                    removed = quiet(operator, solution, 4)
                    self.assert_integrity(solution)
                    self.assertTrue(removed <= self.solution_template.served_customers)
                    self.assertFalse(removed & solution.served_customers)

    def test_remove_customers_drops_empty_station(self):
        solution = self.fresh_solution()
        node = next(iter(solution.ev_route.uav_sorties))
        victims = {
            c for sortie in solution.ev_route.uav_sorties[node] for c in sortie.customers
        }

        DestroyOperators._remove_customers(solution, victims)

        self.assertNotIn(node, solution.ev_route.uav_sorties)
        self.assert_integrity(solution)


class TestRepairOperators(FixtureSolutionMixin, unittest.TestCase):
    def test_greedy_insertion_serves_everything(self):
        solution = self.fresh_solution()
        random.seed(13)
        removed = DestroyOperators.random_removal(solution, 8)
        self.assertGreater(len(removed), 0)

        RepairOperators.greedy_insertion(solution, solution.unserved_customers.copy())

        self.assertEqual(solution.unserved_customers, set())
        self.assertEqual(
            len(solution.served_customers), len(self.dataset["customers"])
        )
        self.assert_integrity(solution)

    def test_greedy_insertion_respects_load_capacity(self):
        solution = self.fresh_solution()
        random.seed(21)
        DestroyOperators.random_removal(solution, 8)
        RepairOperators.greedy_insertion(solution, solution.unserved_customers.copy())

        for _, _, demand in sortie_demands(solution):
            self.assertLessEqual(demand, self.config["uav"]["load_capacity"])

    def test_greedy_insertion_no_duplicate_customer(self):
        solution = self.fresh_solution()
        random.seed(31)
        DestroyOperators.random_removal(solution, 8)
        RepairOperators.greedy_insertion(solution, solution.unserved_customers.copy())

        counter = served_structure(solution)
        self.assertEqual([c for c, k in counter.items() if k > 1], [])
        self.assertEqual(len(counter), len(self.dataset["customers"]))

    def test_greedy_insertion_with_noise_still_valid(self):
        solution = self.fresh_solution()
        random.seed(41)
        DestroyOperators.random_removal(solution, 6)
        RepairOperators.greedy_insertion(
            solution, solution.unserved_customers.copy(), noise=0.5
        )

        self.assertEqual(solution.unserved_customers, set())
        self.assert_integrity(solution)

    def test_regret_insertion_serves_everything(self):
        solution = self.fresh_solution()
        random.seed(17)
        DestroyOperators.random_removal(solution, 6)

        RepairOperators.regret_insertion(solution, solution.unserved_customers.copy(), k=3)

        self.assertEqual(solution.unserved_customers, set())
        self.assert_integrity(solution)

    def test_regret_insertion_falls_back_to_ev_when_uav_infeasible(self):
        """电池容量被压到不可能飞时，regret 必须把客户交给 EV 直送、且不新建 sortie。"""
        solution = self.fresh_solution()
        solution.config["uav"]["battery_capacity"] = 1e-9
        random.seed(23)
        removed = DestroyOperators.random_removal(solution, 6)
        sorties_before = sum(len(s) for s in solution.ev_route.uav_sorties.values())
        direct_before = set(solution.ev_route.direct_customers)

        RepairOperators.regret_insertion(solution, solution.unserved_customers.copy(), k=3)

        self.assertEqual(solution.unserved_customers, set())
        self.assert_integrity(solution)
        self.assertEqual(
            sum(len(s) for s in solution.ev_route.uav_sorties.values()), sorties_before
        )
        new_direct = set(solution.ev_route.direct_customers) - direct_before
        self.assertEqual(new_direct, removed)

    def test_greedy_uses_ev_when_uav_infeasible(self):
        solution = self.fresh_solution()
        solution.config["uav"]["battery_capacity"] = 1e-9
        random.seed(29)
        DestroyOperators.random_removal(solution, 5)
        before_direct = len(solution.ev_route.direct_customers)

        RepairOperators.greedy_insertion(solution, solution.unserved_customers.copy())

        self.assertEqual(solution.unserved_customers, set())
        self.assertEqual(len(solution.ev_route.direct_customers), before_direct + 5)


class TestInsertionCostRules(unittest.TestCase):
    """UAV 插入成本是一个纯函数，可以直接用小数验证。"""

    def setUp(self):
        dataset = make_tiny_dataset()
        self.config = make_config()
        self.solution = Solution(dataset, self.config)

    def test_new_sortie_cost_includes_fixed_cost(self):
        # 站点 1 (0,2000) → 客户 0 (0,3000)：往返 2 km，能耗 0.06 kWh
        cost = RepairOperators._calculate_uav_insertion_cost(self.solution, 1, 0, None)
        self.assertAlmostEqual(cost, 0.06 + self.config["uav"]["fixed_cost"], places=9)

    def test_new_sortie_battery_violation_returns_inf(self):
        self.solution.config["uav"]["battery_capacity"] = 0.01
        cost = RepairOperators._calculate_uav_insertion_cost(self.solution, 1, 0, None)
        self.assertEqual(cost, float("inf"))

    def test_insert_into_existing_sortie_is_incremental(self):
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[0], energy=0.06)
        # c0→c1 = 2 km，c1→站点 = 3 km，原来 c0→站点 = 1 km，增量 4 km = 0.12 kWh
        cost = RepairOperators._calculate_uav_insertion_cost(self.solution, 1, 1, sortie)
        self.assertAlmostEqual(cost, 0.12, places=9)

    def test_insert_into_existing_sortie_load_violation(self):
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[2], energy=0.06)
        # c2 需求 4.0，再加 c1 需求 2.0 = 6.0 > load_capacity 5
        cost = RepairOperators._calculate_uav_insertion_cost(self.solution, 1, 1, sortie)
        self.assertEqual(cost, float("inf"))

    def test_empty_sortie_cannot_accept(self):
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[])
        cost = RepairOperators._calculate_uav_insertion_cost(self.solution, 1, 0, sortie)
        self.assertEqual(cost, float("inf"))

    def test_stale_energy_of_fresh_sortie_underestimates_total_energy(self):
        """已知问题：新建 sortie 的 energy 字段在 evaluate() 之前一直是 0，

        所以往刚建好的 sortie 里插第二个客户时，电池检查用的是 0 + 增量 而不是
        真实累计能耗。这里把两种口径都算出来，固定住当前行为。
        """
        self.solution.config["uav"]["battery_capacity"] = 0.15
        sortie = UAVSortie(uav_id=0, launch_node=1, customers=[0])  # energy 默认 0.0

        cost = RepairOperators._calculate_uav_insertion_cost(self.solution, 1, 1, sortie)

        self.assertAlmostEqual(cost, 0.12, places=9)          # 0.0 + 0.12 ≤ 0.15，被放行
        sortie.customers.append(1)
        self.solution.ev_route.nodes = [0, 1, 0]
        self.solution.ev_route.uav_sorties = {1: [sortie]}
        self.solution.ev_route.direct_customers = [2]
        self.solution.served_customers = set(range(len(self.solution.dataset["customers"])))
        self.solution.unserved_customers = set()
        self.solution.evaluate()
        self.assertGreater(sortie.energy, self.solution.config["uav"]["battery_capacity"])
        self.assertFalse(sortie.is_feasible)


class TestInsertUAVCustomer(unittest.TestCase):
    def setUp(self):
        dataset = load_fixture_dataset()
        self.solution = Solution(dataset, make_config())
        self.solution.ev_route.nodes = [0, 1, 2, 3, 4, 0]

    def test_creates_new_sortie(self):
        RepairOperators._insert_uav_customer(self.solution, 1, 0, None)

        self.assertIn(1, self.solution.ev_route.uav_sorties)
        self.assertEqual(len(self.solution.ev_route.uav_sorties[1]), 1)
        self.assertEqual(self.solution.ev_route.uav_sorties[1][0].customers, [0])
        self.assertIn(0, self.solution.served_customers)
        self.assertNotIn(0, self.solution.unserved_customers)

    def test_appends_to_existing_sortie(self):
        RepairOperators._insert_uav_customer(self.solution, 2, 5, None)
        RepairOperators._insert_uav_customer(self.solution, 2, 6, 0)

        sorties = self.solution.ev_route.uav_sorties[2]
        self.assertEqual(len(sorties), 1)
        self.assertEqual(sorties[0].customers, [5, 6])


if __name__ == "__main__":
    unittest.main()
