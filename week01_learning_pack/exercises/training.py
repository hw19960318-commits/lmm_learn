"""第 3 次课：补训练主线，日志与数值检查已提供。"""
import math
import torch
from torch import Tensor, nn
from .tensor_ops import mse


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
               x: Tensor, y: Tensor) -> dict[str, float]:
    model.train()
    before = [p.detach().clone() for p in model.parameters()]
    # TODO 10：清梯度 -> 前向 -> mse；把标量损失放入 loss。
    optimizer.zero_grad()
    logits = model(x)
    loss = mse(logits, y)
    
    if not torch.isfinite(loss).item():
        raise FloatingPointError('loss 非有限值，请检查数据或学习率。')
    # TODO 11：反向传播；必须在下方检查梯度之前完成。
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if not grads or any(g is None or not torch.isfinite(g).all().item() for g in grads):
        raise FloatingPointError('参数梯度为空或含 NaN/Inf。')
    grad_norm = math.sqrt(sum(g.detach().square().sum().item() for g in grads))
    # TODO 12：让优化器更新参数。
    optimizer.step()
    with torch.no_grad():
        delta = math.sqrt(sum((p - old).square().sum().item()
                              for p, old in zip(model.parameters(), before)))
    return {'loss_before': loss.item(), 'grad_norm': grad_norm, 'parameter_delta': delta}


def evaluate(model: nn.Module, x: Tensor, y: Tensor) -> float:
    """eval 模式、不建立梯度图、不更新参数；返回 Python float。"""
    # TODO 13：实现评价逻辑。参考 train_step 中的非有限值检查。
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = mse(logits, y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError('loss 非有限值，请检查数据或学习率。')
        return loss.item()
