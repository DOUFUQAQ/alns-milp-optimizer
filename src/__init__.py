"""
电动汽车与无人机协同的城市末端物流配送路径优化系统
EV-UAV Collaborative Urban Last-Mile Logistics Delivery Routing Optimization System

只有核心求解链路（Solution / operators / DataGenerator / ALNSSolver）是强依赖，
运行时只需要 numpy + pandas + pyyaml + tqdm。其余模块按可选依赖处理：

    MILPModel      -> 需要 gurobipy（商业求解器，需自行申请 license）
    Visualizer     -> 需要 matplotlib + seaborn
    MapVisualizer  -> 需要 folium + branca + osmnx

缺失依赖时对应名字为 None，`import src` 仍然可用。
"""

__version__ = "1.0.0"

from .data_generator import DataGenerator
from .solution import Solution, EVRoute, UAVSortie
from .alns_solver import ALNSSolver

# milp_model 内部已把 gurobipy 做成可选导入，因此这里总能导入；
# 真正缺 gurobipy 时，MILPModel.build_model() 会抛出带安装提示的 RuntimeError。
from .milp_model import MILPModel, GUROBI_AVAILABLE

# 可视化模块对外部库的依赖更硬：模块本身可以导入，但缺少依赖时导出为 None
try:  # pragma: no cover - 取决于本机安装的可选依赖
    from .visualizer import Visualizer, PLOTTING_AVAILABLE
except ImportError:  # pragma: no cover
    Visualizer = None
    PLOTTING_AVAILABLE = False

try:  # pragma: no cover - 取决于本机安装的可选依赖
    from .map_visualizer import MapVisualizer, MAP_AVAILABLE
except ImportError:  # pragma: no cover
    MapVisualizer = None
    MAP_AVAILABLE = False

if not PLOTTING_AVAILABLE:
    Visualizer = None
if not MAP_AVAILABLE:
    MapVisualizer = None

__all__ = [
    'DataGenerator',
    'Solution',
    'EVRoute',
    'UAVSortie',
    'MILPModel',
    'ALNSSolver',
    'Visualizer',
    'MapVisualizer',
    'GUROBI_AVAILABLE',
]
