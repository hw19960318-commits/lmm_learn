"""统一练习入口。示例：python run_lesson.py --lesson 1 --impl exercises。

reference 是可运行参考答案；exercises 在补完对应 TODO 前会提示未实现。
本周所有计算固定 CPU。输出写入 runs/<impl>/<run-name>/，不访问网络。
"""
from __future__ import annotations
import argparse
import csv
import importlib
import re
from pathlib import Path
import torch
from common import make_batch, write_json, environment_info

ROOT = Path(__file__).resolve().parent


def module(impl: str, name: str):
    return importlib.import_module(f'{impl}.{name}')


def fit(impl: str, steps: int, lr: float):
    torch.manual_seed(42)
    model = module(impl, 'model').TinyRegressor()
    train = module(impl, 'training')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    x, y = make_batch()
    initial = train.evaluate(model, x, y)
    rows = []
    for step in range(1, steps + 1):
        row = train.train_step(model, optimizer, x, y)
        row['loss_after'] = train.evaluate(model, x, y)
        row['step'] = step
        rows.append(row)
        if step in {1, 2, 5, 20, 50, 100, steps}:
            print(f"step={step:3d}  before={row['loss_before']:.6g}  "
                  f"after={row['loss_after']:.6g}  grad={row['grad_norm']:.6g}  "
                  f"delta={row['parameter_delta']:.6g}")
    return model, x, y, initial, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument('--impl', choices=['exercises', 'reference'], default='exercises')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--run-name', default='baseline')
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.lr <= 10:
        parser.error('steps 必须为正；教学实验的 lr 限制在 [0,10]。')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('run-name 只能包含英文字母、数字、下划线和短横线。')
    torch.set_num_threads(1)
    out = ROOT / 'runs' / args.impl / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    if args.lesson == 1:
        op = module(args.impl, 'tensor_ops')
        a = op.create_demo_tensor()
        x = torch.tensor([[1., 2.], [3., 4.]])
        w = torch.tensor([[2.], [-3.]])
        y = op.affine(x, w, torch.tensor([0.5]))
        result = {'tensor': a.tolist(), 'shape': list(a.shape), 'dtype': str(a.dtype),
                  'last_column_shape': list(op.last_column(a).shape),
                  'affine_output': y.tolist(), 'zero_mse': op.mse(y, y).item()}
        write_json(out/'lesson01.json', result)
        print(result)
    elif args.lesson == 2:
        lab = module(args.impl, 'autograd_lab')
        result = {'single_example': lab.single_example_gradients(),
                  'accumulation': lab.accumulation_demo()}
        write_json(out/'lesson02.json', result)
        print(result)
    else:
        model, x, y, initial, rows = fit(args.impl, args.steps, args.lr)
        if args.lesson == 3:
            with (out/'training.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['step', 'loss_before', 'loss_after',
                                                           'grad_norm', 'parameter_delta'])
                writer.writeheader(); writer.writerows(rows)
            report = {'environment': environment_info(), 'seed': 42, 'data_seed': 2026,
                      'n_samples': x.shape[0], 'steps': args.steps, 'learning_rate': args.lr,
                      'initial_loss': initial, 'final_loss': rows[-1]['loss_after'],
                      'weight': model.linear.weight.detach().tolist(),
                      'bias': model.linear.bias.detach().tolist(),
                      'parameter_count': sum(p.numel() for p in model.parameters()),
                      'interpretation': '同一 batch 的拟合实验，不是泛化评估。'}
            write_json(out/'training_summary.json', report)
            print(f'训练记录：{out / "training.csv"}')
        else:
            ckpt = module(args.impl, 'checkpoint')
            probe, _ = make_batch(n=8, seed=88)
            model.eval()
            with torch.no_grad():
                pred_before = model(probe).clone()
            path = out/'tiny_regressor.pt'
            ckpt.save_model(model, path)
            restored = ckpt.load_model(path)
            with torch.no_grad():
                pred_after = restored(probe)
            torch.testing.assert_close(pred_before, pred_after, rtol=1e-6, atol=1e-7)
            diff = (pred_before-pred_after).abs().max().item()
            write_json(out/'checkpoint_summary.json', {
                'path': str(path), 'max_abs_diff': diff, 'loaded_training_mode': restored.training,
                'scope': '同输入、同模式下的输出一致性；未验证完整断点续训。'})
            print(f'保存加载通过；最大输出差异={diff:.3g}；文件：{path}')


if __name__ == '__main__':
    try:
        main()
    except NotImplementedError as exc:
        raise SystemExit(f'练习尚未完成：{exc}。请补对应 TODO，或用 --impl reference 查看参考运行。')
