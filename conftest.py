"""pytest 入口：把仓库根目录放进 sys.path，并固定无界面绘图后端。

本仓库的测试用标准库 unittest 编写，两种方式都能跑：

    python -m unittest discover -s tests -t .    # 不需要额外依赖
    python -m pytest -q                          # 需要 pytest
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 无显示器环境下 matplotlib 必须走 Agg 后端（测试本身不依赖 matplotlib）
os.environ.setdefault("MPLBACKEND", "Agg")
