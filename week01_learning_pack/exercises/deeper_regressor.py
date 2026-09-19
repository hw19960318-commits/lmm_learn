from torch import Tensor, nn

class DeeperRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError('模型输入必须是 [B,2]。')
        return self.layers(x)