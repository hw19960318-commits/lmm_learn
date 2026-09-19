"""第 1 次课参考实现：形状、索引、矩阵乘法和显式 MSE。"""
import torch
from torch import Tensor
from common import check_xy


def create_demo_tensor() -> Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(3, 4)


def last_column(x: Tensor) -> Tensor:
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError('输入应为至少一列的二维张量。')
    # 切片保留列维度；x[:, -1] 会变成 [B]。
    return x[:, -1:]


def affine(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """X:[B,D]、W:[D,K]、b:[K] -> [B,K]。这里 W 的约定不同于 nn.Linear.weight。"""
    if x.ndim != 2 or w.ndim != 2 or b.ndim != 1:
        raise ValueError('要求 X、W 为二维，b 为一维。')
    if x.shape[1] != w.shape[0] or w.shape[1] != b.shape[0]:
        raise ValueError('X、W、b 的形状不匹配。')
    return x @ w + b


def mse(pred: Tensor, target: Tensor) -> Tensor:
    check_xy(pred, target)
    return (pred - target).square().mean()
