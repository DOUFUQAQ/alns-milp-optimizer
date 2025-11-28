"""对比两份参数配置的差异。

用法::

    python tools/compare_configs.py --base config/params_small.yaml \
                                    --other config/params_large.yaml

只输出真实算出的人均差异，不做任何"预期效果"的推测。
"""
import argparse
import sys
import unicodedata

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("需要 PyYAML：pip install pyyaml")


def display_width(text):
    """按终端显示宽度计算字符数（中文算 2 格），用于对齐表格。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text, width):
    text = str(text)
    return text + " " * max(0, width - display_width(text))


def flatten(config, prefix=""):
    """把嵌套配置拍平成 {'ev.speed': 40, ...}。"""
    flat = {}
    for key, value in config.items():
        name = "%s.%s" % (prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        else:
            flat[name] = value
    return flat


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def format_value(value):
    if isinstance(value, list):
        return "[%s]" % ", ".join(str(item) for item in value)
    return str(value)


def relative_change(base, other):
    """数值型参数返回相对变化百分比，其它返回 None。"""
    if isinstance(base, bool) or isinstance(other, bool):
        return None
    if isinstance(base, (int, float)) and isinstance(other, (int, float)):
        if base == 0:
            return None
        return (other - base) / base * 100.0
    return None


def compare(base_config, other_config):
    base_flat = flatten(base_config)
    other_flat = flatten(other_config)
    keys = sorted(set(base_flat) | set(other_flat))

    rows = []
    for key in keys:
        base_value = base_flat.get(key, "<缺失>")
        other_value = other_flat.get(key, "<缺失>")
        if base_value == other_value:
            continue
        change = relative_change(base_value, other_value)
        rows.append((key, base_value, other_value, change))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="对比两份 YAML 参数配置")
    parser.add_argument("--base", default="config/params_small.yaml", help="基准配置")
    parser.add_argument("--other", default="config/params_large.yaml", help="对照配置")
    parser.add_argument("--all", action="store_true", help="连未变化的参数也打印")
    args = parser.parse_args(argv)

    base_config = load(args.base)
    other_config = load(args.other)

    print("=" * 78)
    print("基准: %s" % args.base)
    print("对照: %s" % args.other)
    print("=" * 78)

    rows = compare(base_config, other_config)
    if not rows:
        print("\n两份配置在全部 %d 个参数上完全相同。" % len(flatten(base_config)))
        return 0

    width = max(display_width(row[0]) for row in rows)
    width = max(width, display_width("参数"))
    header = " | ".join([pad("参数", width), pad("基准", 16), pad("对照", 16), "变化"])
    print("\n" + header)
    print("-" * display_width(header))
    for key, base_value, other_value, change in rows:
        change_text = "%+.1f%%" % change if change is not None else "-"
        print(
            " | ".join(
                [
                    pad(key, width),
                    pad(format_value(base_value), 16),
                    pad(format_value(other_value), 16),
                    change_text,
                ]
            )
        )

    only_in_base = sorted(set(flatten(base_config)) - set(flatten(other_config)))
    only_in_other = sorted(set(flatten(other_config)) - set(flatten(base_config)))
    if only_in_base:
        print("\n只在基准里出现的参数: %s" % ", ".join(only_in_base))
    if only_in_other:
        print("只在对照里出现的参数: %s" % ", ".join(only_in_other))

    print("\n共 %d 个参数不同。" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
