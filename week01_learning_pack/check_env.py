"""直接运行：python check_env.py。只检查环境，不联网，不下载模型。"""
from pathlib import Path
import sys
try:
    import torch
except ImportError as exc:
    raise SystemExit('当前 Python 没有 torch。请核对虚拟环境并按 PyTorch 官方安装页安装。') from exc
from common import environment_info, write_json


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit('本练习使用 Python 3.10+ 语法；新建环境建议 3.11 或 3.12。')
    torch.set_num_threads(1)
    info = environment_info()
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x.square().mean().backward()
    expected = 2 * x.detach() / x.numel()
    if x.grad is None:
        raise RuntimeError('自动求导没有产生梯度。')
    torch.testing.assert_close(x.grad, expected)
    info['cpu_autograd_check'] = 'passed'
    for k, v in info.items():
        print(f'{k}: {v}')
    path = Path(__file__).resolve().parent / 'reports' / 'environment.json'
    write_json(path, info)
    print(f'环境记录：{path}')


if __name__ == '__main__':
    main()
