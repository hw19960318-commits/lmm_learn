"""第 2 次课：手算值、自动求导、梯度累积与清零。"""
import torch


def single_example_gradients(w_value: float = 1.0, b_value: float = 0.0) -> dict[str, float]:
    """x=2,y=5,pred=w*x+b,loss=(pred-y)^2；只反向，不更新。"""
    x = torch.tensor(2.0)
    target = torch.tensor(5.0)
    w = torch.tensor(w_value, dtype=torch.float32, requires_grad=True)
    b = torch.tensor(b_value, dtype=torch.float32, requires_grad=True)
    prediction = w * x + b
    loss = (prediction - target).square()
    loss.backward()
    if w.grad is None or b.grad is None:
        raise RuntimeError('参数梯度不应为 None。')
    return {'prediction': prediction.item(), 'loss': loss.item(),
            'grad_w': w.grad.item(), 'grad_b': b.grad.item(),
            'w_after_backward': w.item(), 'b_after_backward': b.item()}


def accumulation_demo() -> tuple[float, float, float]:
    """w=1，三次都重新前向计算 w^2；第二次不清梯度，第三次先清。"""
    w = torch.tensor(1.0, requires_grad=True)
    w.square().backward()
    first = w.grad.item()
    w.square().backward()  # 新图；没有 retain_graph=True。
    second = w.grad.item()
    w.grad = None
    w.square().backward()
    third = w.grad.item()
    return first, second, third
