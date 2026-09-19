"""第 3 次课：单步训练和评价。每个 train_step 都真实执行反向与更新。"""
import math
import torch
from torch import Tensor, nn
from .tensor_ops import mse


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               x: Tensor, y: Tensor) -> dict[str, float]:
    model.train()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = mse(pred, y)
    if not torch.isfinite(loss).item():
        raise FloatingPointError('loss 非有限值，请检查数据或学习率。')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if not grads or any(g is None or not torch.isfinite(g).all().item() for g in grads):
        raise FloatingPointError('参数梯度为空或含 NaN/Inf。')
    grad_norm = math.sqrt(sum(g.detach().square().sum().item() for g in grads))
    optimizer.step()
    with torch.no_grad():
        delta = math.sqrt(sum((p - old).square().sum().item()
                              for p, old in zip(model.parameters(), before)))
    return {'loss_before': loss.item(), 'grad_norm': grad_norm, 'parameter_delta': delta}


def evaluate(model: nn.Module, x: Tensor, y: Tensor) -> float:
    model.eval()
    with torch.no_grad():
        value = mse(model(x), y)
    if not torch.isfinite(value).item():
        raise FloatingPointError('评价 loss 非有限值。')
    return value.item()
