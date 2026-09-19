"""第 2 次课：补全反向传播和梯度累积实验。"""
import torch


def single_example_gradients(w_value: float = 1.0, b_value: float = 0.0) -> dict[str, float]:
    """x=2,y=5,pred=w*x+b,loss=(pred-y)^2。只反向，不更新。

    返回键：prediction、loss、grad_w、grad_b、w_after_backward、b_after_backward。
    所有值为 Python float。必须真正调用 autograd，不可硬编码答案。
    """
    # TODO 5：创建浮点叶子参数，设置 requires_grad=True。
    w = torch.tensor(w_value,requires_grad=True)
    b = torch.tensor(b_value,requires_grad=True)
    x = torch.tensor(2.0)
    y = torch.tensor(5.0)
    
    # TODO 6：前向、loss、backward，读取梯度并返回字典。
    pred = x*w+b
    loss = (pred-y)**2
    loss.backward()
    return {'prediction': pred.item(), 'loss': loss.item(), 'grad_w': w.grad.item(), 'grad_b': b.grad.item(), 'w_after_backward': w.item(), 'b_after_backward': b.item()}
    


def accumulation_demo() -> tuple[float, float, float]:
    """w=1；对 w^2 重新前向并反向三次。

    第一次直接反向；第二次不清梯度；第三次先将 w.grad 设为 None。
    返回这三次之后的梯度值；不更新 w，不使用 retain_graph=True。
    """
    # TODO 7：输出应能说明梯度默认累加，而不是被覆盖。
    w = torch.tensor(1.0,requires_grad=True)
    w2 = w**2
    w2.backward()
    grad1 = w.grad.item()
    w2 = w**2
    w2.backward()
    grad2 = w.grad.item()
    w.grad = None
    w2 = w**2
    w2.backward()
    grad3 = w.grad.item()
    return grad1, grad2, grad3
