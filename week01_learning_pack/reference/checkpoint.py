"""第 4 次课：只保存模型 state_dict，不声称完整恢复训练状态。"""
from pathlib import Path
import torch
from torch import nn
from .model import TinyRegressor


def save_model(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path) -> TinyRegressor:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'模型文件不存在：{path}')
    model = TinyRegressor()
    # 只加载自己生成或可信来源的文件；weights_only=True 不是万能安全沙箱。
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
