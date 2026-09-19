from common import make_batch
from exercises.deeper_regressor import DeeperRegressor
from exercises.training import train_step, evaluate
import torch

torch.manual_seed(42)
model = DeeperRegressor()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
x, y = make_batch()
print('params:', sum(p.numel() for p in model.parameters()))
print('init loss:', evaluate(model, x, y))
for _ in range(200):
    train_step(model, optimizer, x, y)
print('final loss:', evaluate(model, x, y))