"""数据生成器：生成出来的算例必须满足讲得清楚的约束。

覆盖：节点/客户数量与类型、坐标范围、需求区间、时间窗宽度、
距离矩阵的对称性与零对角、时间/能耗矩阵的换算关系、随机种子可复现、
以及保存到磁盘再 load_dataset 读回来的往返一致性。
"""
import json
import os
import tempfile
import unittest
import contextlib
import io

import numpy as np

from src.data_generator import DataGenerator
from tests.helpers import make_config

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def generator_config(**overrides):
    base = {
        "geography": {
            "area_size": [10, 10],
            "depot_location": [5, 5],
            "num_stations": 4,
            "num_charging_stations": 2,
            "num_customers": 12,
        }
    }
    config = make_config()
    config.update(base)
    for key, value in overrides.items():
        config[key] = value
    return config


def save_quietly(generator, dataset, directory):
    """_save_dataset 会打印一行提示，测试里没必要看到。"""
    with contextlib.redirect_stdout(io.StringIO()):
        generator._save_dataset(dataset, directory)


class TestDatasetShapes(unittest.TestCase):
    def setUp(self):
        self.config = generator_config()
        self.dataset = DataGenerator(self.config).generate_dataset()

    def test_node_count_and_types(self):
        nodes = self.dataset["nodes"]
        expected = (
            1
            + self.config["geography"]["num_stations"]
            + self.config["geography"]["num_charging_stations"]
        )
        self.assertEqual(len(nodes), expected)

        self.assertEqual(nodes.iloc[0]["type"], "depot")
        self.assertEqual(
            list(nodes["type"]).count("station"),
            self.config["geography"]["num_stations"],
        )
        self.assertEqual(
            list(nodes["type"]).count("charging_station"),
            self.config["geography"]["num_charging_stations"],
        )

    def test_depot_is_at_configured_position(self):
        nodes = self.dataset["nodes"]
        self.assertAlmostEqual(nodes.iloc[0]["x"], 5.0)
        self.assertAlmostEqual(nodes.iloc[0]["y"], 5.0)

    def test_nodes_inside_area(self):
        nodes = self.dataset["nodes"]
        width, height = self.config["geography"]["area_size"]
        self.assertTrue((nodes["x"] >= 0).all() and (nodes["x"] <= width).all())
        self.assertTrue((nodes["y"] >= 0).all() and (nodes["y"] <= height).all())

    def test_customer_count(self):
        self.assertEqual(
            len(self.dataset["customers"]),
            self.config["geography"]["num_customers"],
        )

    def test_customer_demand_within_range(self):
        low, high = self.config["customers"]["demand_range"]
        demand = self.dataset["customers"]["demand"]
        self.assertTrue((demand >= low).all())
        self.assertTrue((demand <= high).all())

    def test_time_window_width(self):
        customers = self.dataset["customers"]
        width = self.config["customers"]["time_window_width"]["medium"]
        actual = customers["time_window_end"] - customers["time_window_start"]
        np.testing.assert_allclose(actual.values, width, rtol=0, atol=1e-9)
        self.assertTrue((customers["time_window_start"] >= 0).all())
        self.assertTrue((customers["time_window_start"] <= 720 - width).all())

    def test_customer_coordinates_inside_area(self):
        customers = self.dataset["customers"]
        width, height = self.config["geography"]["area_size"]
        self.assertTrue((customers["x"] >= 0).all() and (customers["x"] <= width).all())
        self.assertTrue((customers["y"] >= 0).all() and (customers["y"] <= height).all())


class TestMatrices(unittest.TestCase):
    def setUp(self):
        self.config = generator_config()
        self.dataset = DataGenerator(self.config).generate_dataset()

    def test_distance_matrix_properties(self):
        for mode in ("ev", "uav"):
            matrix = self.dataset["distances"][mode]
            n = len(self.dataset["nodes"])
            self.assertEqual(matrix.shape, (n, n))
            np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
            self.assertTrue((matrix >= 0).all())

    def test_time_matrix_is_distance_over_speed(self):
        for mode, key in (("ev", "ev"), ("uav", "uav")):
            expected = (
                self.dataset["distances"][key] / self.config[key]["speed"] * 60
            )
            np.testing.assert_allclose(self.dataset["times"][key], expected, atol=1e-12)

    def test_energy_matrix_is_distance_times_consumption(self):
        for key in ("ev", "uav"):
            expected = (
                self.dataset["distances"][key] * self.config[key]["energy_consumption"]
            )
            np.testing.assert_allclose(self.dataset["energy"][key], expected, atol=1e-12)

    def test_distance_matches_coordinates(self):
        nodes = self.dataset["nodes"]
        matrix = self.dataset["distances"]["ev"]
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                dx = nodes.iloc[i]["x"] - nodes.iloc[j]["x"]
                dy = nodes.iloc[i]["y"] - nodes.iloc[j]["y"]
                self.assertAlmostEqual(matrix[i, j], np.hypot(dx, dy), places=12)


class TestReproducibility(unittest.TestCase):
    def test_same_config_gives_same_dataset(self):
        config = generator_config()
        first = DataGenerator(config).generate_dataset()
        second = DataGenerator(config).generate_dataset()

        np.testing.assert_allclose(
            first["distances"]["ev"], second["distances"]["ev"], atol=0.0
        )
        np.testing.assert_allclose(
            first["customers"]["demand"].values,
            second["customers"]["demand"].values,
            atol=0.0,
        )
        np.testing.assert_allclose(
            first["nodes"][["x", "y"]].values, second["nodes"][["x", "y"]].values, atol=0.0
        )

    def test_config_is_embedded_in_dataset(self):
        config = generator_config()
        dataset = DataGenerator(config).generate_dataset()
        self.assertIs(dataset["config"], config)


class TestSaveAndLoad(unittest.TestCase):
    def test_round_trip(self):
        config = generator_config()
        dataset = DataGenerator(config).generate_dataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_quietly(DataGenerator(config), dataset, tmpdir)

            expected_files = [
                "nodes.csv", "customers.csv", "config.json",
                "distance_ev.npy", "distance_uav.npy",
                "time_ev.npy", "time_uav.npy",
                "energy_ev.npy", "energy_uav.npy",
            ]
            for name in expected_files:
                self.assertTrue(
                    os.path.exists(os.path.join(tmpdir, name)), "缺少文件 %s" % name
                )

            loaded = DataGenerator.load_dataset(tmpdir)

        self.assertEqual(len(loaded["nodes"]), len(dataset["nodes"]))
        self.assertEqual(len(loaded["customers"]), len(dataset["customers"]))
        for group, key in (("distances", "ev"), ("distances", "uav"),
                           ("times", "ev"), ("times", "uav"),
                           ("energy", "ev"), ("energy", "uav")):
            np.testing.assert_allclose(loaded[group][key], dataset[group][key], atol=1e-12)
        self.assertEqual(loaded["config"]["geography"], config["geography"])

    def test_saved_config_is_valid_json(self):
        config = generator_config()
        dataset = DataGenerator(config).generate_dataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_quietly(DataGenerator(config), dataset, tmpdir)
            path = os.path.join(tmpdir, "config.json")
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["geography"]["num_customers"], 12)


class TestFixtureDataset(unittest.TestCase):
    def test_fixture_can_be_loaded(self):
        from tests.helpers import FIXTURE_DIR

        dataset = DataGenerator.load_dataset(str(FIXTURE_DIR))

        self.assertEqual(len(dataset["nodes"]), 7)
        self.assertEqual(len(dataset["customers"]), 20)
        self.assertEqual(dataset["distances"]["ev"].shape, (7, 7))
        self.assertEqual(dataset["config"]["geography"]["num_customers"], 20)
        self.assertEqual(dataset["nodes"].iloc[0]["type"], "depot")


class TestConfigLoading(unittest.TestCase):
    @unittest.skipUnless(HAS_YAML, "没装 PyYAML，跳过 YAML 配置读取测试")
    def test_load_yaml_config(self):
        from tests.helpers import REPO_ROOT

        generator = DataGenerator(str(REPO_ROOT / "config" / "params_small.yaml"))
        self.assertEqual(generator.config["geography"]["num_customers"], 20)

    @unittest.skipIf(HAS_YAML, "装了 PyYAML，这个降级路径测不到")
    def test_yaml_missing_gives_helpful_error(self):
        with self.assertRaises(ImportError) as context:
            DataGenerator("config/params_small.yaml")
        self.assertIn("pyyaml", str(context.exception).lower())

    def test_dict_config_needs_no_yaml(self):
        generator = DataGenerator(generator_config())
        self.assertEqual(generator.area_size, [10, 10])
        self.assertEqual(generator.depot_location, [5, 5])


if __name__ == "__main__":
    unittest.main()
