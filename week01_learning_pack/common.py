"""已提供的基础工具。合成数据无须下载，也不对应任何真实业务。"""
from __future__ import annotations
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any
import torch
from torch import Tensor


def make_batch(n: int = 32, seed: int = 2026) -> tuple[Tensor, Tensor]:
    """返回 X:[n,2]、y:[n,1]，真实关系 y = 2*x1 - 3*x2 + 0.5。

    使用独立 CPU Generator，避免数据生成改变模型初始化所用的全局随机数序列。
    标签公式只在此处生成数据；模型 forward 不得写入真实系数。
    """
    if n < 1:
        raise ValueError('n 必须是正整数。')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(n, 2, generator=generator, dtype=torch.float32)
    y = x @ torch.tensor([[2.0], [-3.0]]) + 0.5
    return x, y


def check_xy(pred: Tensor, target: Tensor) -> None:
    """回归任务禁止静默广播标签。"""
    if pred.shape != target.shape:
        raise ValueError(f'预测和目标形状必须相同：{pred.shape} != {target.shape}')
    if pred.numel() == 0:
        raise ValueError('不能对空张量计算本实验的 MSE。')


def environment_info() -> dict[str, Any]:
    mps = getattr(torch.backends, 'mps', None)
    return {
        'python': sys.version.split()[0],
        'python_executable': sys.executable,
        'os': platform.platform(),
        'pytorch': str(torch.__version__),
        'default_device_for_this_week': 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'mps_available': bool(mps is not None and mps.is_available()),
        'cpu_threads': torch.get_num_threads(),
    }


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
