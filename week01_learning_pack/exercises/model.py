"""第 3 次课：本周允许使用 nn.Linear，不使用现成训练器。"""
from torch import Tensor, nn


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO 8：建立名为 self.linear 的 nn.Linear(2,1)。
        # 参数应随机初始化，不得在此写入标签生成公式的真实系数。
        self.linear = nn.Linear(2, 1)
        
        

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError('模型输入必须是 [B,2]。')
        # TODO 9：调用 self.linear。
        logits = self.linear(x)
        return logits
