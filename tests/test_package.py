"""包的导入契约：核心链路在只有 numpy/pandas 的环境里也必须可用。

这条测试是有意义的：仓库里 4 个模块依赖可选的重型库
（gurobipy / matplotlib / folium / osmnx），如果哪天有人在模块顶层写死
`import matplotlib`，这里会立刻红。
"""
import importlib
import unittest


class TestPackageImport(unittest.TestCase):
    def test_import_package(self):
        import src

        self.assertIsInstance(src.__version__, str)

    def test_core_names_are_available(self):
        import src

        for name in ("DataGenerator", "Solution", "EVRoute", "UAVSortie",
                     "ALNSSolver", "MILPModel"):
            self.assertTrue(hasattr(src, name), "缺少 %s" % name)
            self.assertIsNotNone(getattr(src, name))

    def test_gurobi_flag_is_boolean(self):
        import src

        self.assertIsInstance(src.GUROBI_AVAILABLE, bool)

    def test_optional_modules_report_availability(self):
        import src.map_visualizer as map_visualizer
        import src.osm_data_loader as osm_data_loader
        import src.visualizer as visualizer

        self.assertIsInstance(visualizer.PLOTTING_AVAILABLE, bool)
        self.assertIsInstance(map_visualizer.MAP_AVAILABLE, bool)
        self.assertIsInstance(osm_data_loader.OSM_AVAILABLE, bool)

    def test_optional_classes_degrade_to_none(self):
        import src
        import src.map_visualizer as map_visualizer
        import src.visualizer as visualizer

        if not visualizer.PLOTTING_AVAILABLE:
            self.assertIsNone(src.Visualizer)
        else:
            self.assertIsNotNone(src.Visualizer)
        if not map_visualizer.MAP_AVAILABLE:
            self.assertIsNone(src.MapVisualizer)
        else:
            self.assertIsNotNone(src.MapVisualizer)

    def test_submodules_are_importable(self):
        for name in ("src.solution", "src.operators", "src.data_generator",
                     "src.alns_solver", "src.milp_model", "src.visualizer",
                     "src.map_visualizer", "src.osm_data_loader"):
            module = importlib.import_module(name)
            self.assertIsNotNone(module)

    def test_alns_solver_works_without_tqdm(self):
        """tqdm 是可选依赖，缺了也不能影响主循环。"""
        import src.alns_solver as alns_solver

        bar = alns_solver._progress_bar(3, "test")
        bar.update(1)
        bar.set_postfix({"a": 1})
        bar.close()

    def test_all_lists_only_real_names(self):
        import src

        for name in src.__all__:
            self.assertTrue(hasattr(src, name), "__all__ 里的 %s 不存在" % name)


if __name__ == "__main__":
    unittest.main()
