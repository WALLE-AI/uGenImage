"""YAML 配置系统 —— 超参的唯一真相。

原实现把超参在 argparse / scripts/*.sh / configs/*.yaml 三处各写一遍，
且 yaml 根本没有代码读取（PLAN.md P2-1）。这里统一为：

    python train_vqgan.py --config configs/vqgan.yaml --set train.lr=1e-4 data.batch_size=64

命令行 --set 用点号路径覆盖任意层级，取值按 YAML 语法解析。
"""
import argparse
import copy
import re
from pathlib import Path

import yaml


class Config(dict):
    """支持属性访问的嵌套 dict。"""

    def __init__(self, data=None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = Config(value) if isinstance(value, dict) else value

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.items()}

    def get_path(self, dotted, default=None):
        node = self
        for part in dotted.split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted, value):
        parts = dotted.split('.')
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = Config(value) if isinstance(value, dict) else value


# YAML 1.1 不把 1e-5 / 3e-4 当浮点数（要求写成 1.0e-5），会静默解析成字符串。
# 而 lr=3e-4 恰恰是最自然的写法，因此这里补上这一种情况。
_SCI_NOTATION = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$')


def _coerce_numeric(value):
    if isinstance(value, str) and _SCI_NOTATION.match(value.strip()):
        return float(value)
    if isinstance(value, dict):
        return {k: _coerce_numeric(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_numeric(v) for v in value]
    return value


def _parse_scalar(text):
    """按 YAML 语法解析命令行取值，使 lr=1e-4 / amp=false / limit=null 都正确。"""
    return _coerce_numeric(yaml.safe_load(text))


def load_config(path, overrides=None, strict=True):
    """读取 YAML 并应用 `a.b=c` 形式的覆盖。

    strict=True 时，覆盖一个 YAML 中不存在的键会报错 —— 防止拼错键名后
    参数被静默忽略（这类错误在训练跑了几小时后才发现代价极高）。
    """
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    cfg = Config(_coerce_numeric(raw))  # YAML 文件里写 lr: 3e-4 同样会被解析成字符串
    for item in overrides or []:
        if '=' not in item:
            raise SystemExit(f"--set 需要 key=value 形式，收到: {item!r}")
        key, _, raw = item.partition('=')
        key = key.strip()
        if strict and cfg.get_path(key, _MISSING) is _MISSING:
            raise SystemExit(f"配置中不存在键 {key!r}（如确需新增，用 --set-new）")
        cfg.set_path(key, _parse_scalar(raw))
    return cfg


class _Missing:
    pass


_MISSING = _Missing()


def base_parser(description=''):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', required=True, help='YAML 配置路径')
    # action='extend'：允许多次出现 --set 而不是后者覆盖前者
    p.add_argument('--set', nargs='*', action='extend', default=None, metavar='KEY=VALUE',
                   help='覆盖配置，点号路径，例如 train.lr=1e-4 data.batch_size=64')
    p.add_argument('--set-new', nargs='*', action='extend', default=None, metavar='KEY=VALUE',
                   help='同 --set，但允许新增配置中不存在的键')
    return p


def config_from_args(args):
    cfg = load_config(args.config, args.set or [], strict=True)
    for item in args.set_new or []:
        key, _, raw = item.partition('=')
        cfg.set_path(key.strip(), _parse_scalar(raw))
    return cfg


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )


def deepcopy_config(cfg):
    return Config(copy.deepcopy(cfg.to_dict()))
