"""第 1 次课：补全 4 个 TODO。完成后运行 tests/test_01_tensors.py。"""
import torch
from torch import Tensor
from common import check_xy


def create_demo_tensor() -> Tensor:
    """返回 float32 的 [3,4] 张量，元素按行依次为 0...11。"""
    # TODO 1：使用 arange 和 reshape。
    tensor1 = torch.arange(12).reshape(3, 4).float()
    return tensor1


def last_column(x: Tensor) -> Tensor:
    """返回最后一列，必须保留 [B,1] 形状；输入非法时抛出 ValueError。"""
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError('输入应为至少一列的二维张量。')
    # TODO 2：用切片保留列维度。
    tensor2 = x[:,-1:].reshape(-1,1)
    return tensor2


def affine(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """X:[B,D]、W:[D,K]、b:[K] -> [B,K]；不要使用 nn.Linear。"""
    if x.ndim != 2 or w.ndim != 2 or b.ndim != 1:
        raise ValueError('要求 X、W 为二维，b 为一维。')
    if x.shape[1] != w.shape[0] or w.shape[1] != b.shape[0]:
        raise ValueError('X、W、b 的形状不匹配。')
    # TODO 3：矩阵乘法，再按行广播偏置。
    tensor3 = x @w +b
    return tensor3


def mse(pred: Tensor, target: Tensor) -> Tensor:
    """返回标量 MSE；不得调用现成 MSELoss。"""
    check_xy(pred, target)
    # TODO 4：逐元素误差 -> 平方 -> 均值。
    tensor4 = (pred-target)**2
    tensor4 = tensor4.mean()
    return tensor4
