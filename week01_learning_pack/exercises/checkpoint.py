"""第 4 次课：模型保存与加载。完整续训状态留在第 5 周。"""
from pathlib import Path
import torch
from torch import nn
from .model import TinyRegressor


def save_model(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # TODO 14：保存 model.state_dict()，不是整个模型对象。
    torch.save(model.state_dict(), path)


def load_model(path: str | Path) -> TinyRegressor:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'模型文件不存在：{path}')
    # TODO 15：重建相同结构；torch.load 使用 map_location="cpu"、weights_only=True。
    model = TinyRegressor()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    # TODO 16：严格加载 state_dict，设 eval 模式并返回模型。
    model.eval()
    return model
