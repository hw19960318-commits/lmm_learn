"""第 3 次课：两特征、单输出的极小回归模型，总共 3 个参数。"""
from torch import Tensor, nn


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError('模型输入必须是 [B,2]。')
        return self.linear(x)
