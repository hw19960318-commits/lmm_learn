# 第 1 周复习笔记

根据仓库 `lmm_learn` 与第 1 周自学对话整理。正式手册、24 道题解、复盘数字分别在：

- `大模型全链路学习计划_教程与逐次任务版.md`
- `week01_learning_pack/docs/第一周学习手册.md`
- `week01_learning_pack/docs/问题与解答.md`
- `week01_learning_pack/reports/week01_template.md`

本文件只收**自己问过、容易再忘**的点。

## 仓库在做什么

12 周、每周 4 次，顺序是：PyTorch → 分词 → Transformer → 现代组件 → 预训练 → SFT → LoRA → DPO → 最小强化学习 → KV cache → 量化。当前完成的是第 1 周。

第 1 周学习包在 `week01_learning_pack/`。默认 CPU，不下载大模型和外部数据。数据是 32 条合成样本：

```text
y = 2*x1 - 3*x2 + 0.5
```

公式只用来造标签。模型必须从随机参数学，不能把系数写进 `forward`。

完整链路：

```text
X:[32,2], y:[32,1]
  → TinyRegressor（2 个权重 + 1 个偏置）
  → pred:[32,1] → MSE
  → backward → SGD.step
  → state_dict → 新模型加载 → 同输入预测一致
```

**同一 batch 上 loss 降到接近 0 = 拟合成功，不是泛化评估。** 要谈泛化，需要独立验证/测试集。

本机环境：Python 3.13.9，PyTorch 2.14.0+cpu，Windows。`cuda_available: False` 符合本周设定。

## 怎么跑

在 `week01_learning_pack` 里、已激活 `.venv` 后：

```powershell
python check_env.py
python -m pytest -q --impl reference          # 先确认参考答案/测试本身没问题
python run_lesson.py --lesson 1 --impl exercises
python -m pytest -q tests/test_01_tensors.py --impl exercises
python -m pytest -q --impl exercises          # 第1–4课全部测试
```

| 命令 | 作用 |
|---|---|
| `python -m venv .venv` | 在当前目录建隔离环境，依赖装这里，不装进系统 Python |
| `--impl reference` | 测 `reference/` 参考答案 |
| `--impl exercises`（默认） | 测 `exercises/` 自己写的代码 |
| `run_lesson.py --lesson N` | 试跑第 N 课演示，写 `runs/` |
| `pytest tests/test_0X_*.py` | 只测这一课 |
| `pytest` 不指定文件 | 跑全集（约 24 项） |
| `-q` | 输出更短，和 `--impl` 无关 |

主业是填 `exercises/` 的 TODO。`run_lesson` 和 `pytest` 是自检，不能代替填写。

`check_env` 通过即可继续。缺 NumPy 时 PyTorch 可能警告 `Failed to initialize NumPy`：本周 `requirements.txt` 只有 `torch` 和 `pytest`，纯用 `torch.Tensor` 可先忽略；要用 `tensor.numpy()` 再 `pip install numpy`。

## 第 1 课：张量与仿射

热身：在交互里打印标量、向量、矩阵，同时看 `shape` / `dtype` / `device`。标量 `()`，向量 `(n,)`，矩阵 `(行, 列)`。本周设备应是 `cpu`。

```python
import torch
s = torch.tensor(3.14)                          # 标量
v = torch.tensor([1.0, 2.0, 3.0])               # 向量
m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])      # 矩阵
```

### `torch.tensor` 与 `from_numpy`

| | `torch.tensor(ary)` | `torch.from_numpy(ary)` |
|---|---|---|
| 内存 | 拷贝 | 和 NumPy 共用同一块 |
| 改一边 | 另一边不变 | 另一边跟着变 |
| 输入 | 列表、NumPy、标量等 | 只能是 NumPy 数组 |

要独立、安全用 `torch.tensor`；要零拷贝且两边不会误改对方，用 `from_numpy`。`torch.as_tensor` 对 NumPy 也尽量共享，但还能接受更多类型。

### `view` 与 `reshape`、什么叫连续

- `view`：必须和原张量共享连续内存，条件不满足就报错。
- `reshape`：能共享就共享，不行就拷贝。日常更省心。

**连续**指按当前 shape 从头扫到尾，内存地址一路递增。转置、某些切片后，逻辑顺序和内存顺序不一致，就是不连续。`x.is_contiguous()` 可检查；`y.contiguous()` 必要时拷贝成连续副本。本周练习用 `arange(...).reshape(...)` 即可。

### 仿射：`pred = X @ W + b`

- `X`：`[B, D]`，`W`：`[D, K]`，`b`：`[K]`
- `X @ W` 得到 `[B, K]`，再 `+ b` 时 `b` 按输出维对齐、沿样本维广播

`b.reshape(-1, 1)` 会变成 `[K, 1]`，多输出时和 `[B, K]` 对不上。正确写法就是 `x @ w + b`，或显式 `b.reshape(1, -1)`。

选做「两个输出偏置分别设为 1 和 2」：`b = [1., 2.]`，每一行第 1 列都 `+1`、第 2 列都 `+2`。这是 `K=2` 的线性层，不是多头注意力。

本周损失是 **MSE**：`mean((pred - target)²)`。BCE 是二分类损失，公式是 \(\ell = -[y\log p + (1-y)\log(1-p)]\)，和本周作业不是一套。

## 第 2 课：自动求导

`loss.backward()` 和 `torch.autograd.grad()` 都是反向算梯度，副作用不同：

| | `loss.backward()` | `grad(loss, ...)` |
|---|---|---|
| 梯度放哪 | 写入 `param.grad` | 作为返回值，默认不写 `.grad` |
| 典型用法 | 训练：`backward` → `optimizer.step()` | 教学/调试，指定要求导的量 |

训练循环：

```text
zero_grad → 前向算 loss → backward → step
```

- `zero_grad`：清掉旧的 `.grad`（默认会累加）
- `backward`：只算梯度，**不改参数**
- `step`：按优化器规则用 `.grad` 更新参数（SGD 大致是 `param ← param - lr * grad`）

### 计算图：`retain_graph` 与重新前向

默认一次反向后图就拆掉。`retain_graph=True` 必须加在**第一次**反向，意思是「这次算完先别拆」。已经拆了再写也救不回来。

不想用 `retain_graph` 时：

```python
dw1, db = grad(loss, (w1, b))   # 一次求多个，推荐
# 或重新前向，得到新 loss 再 grad
```

Python 变量 `loss` 还在，不等于图还在。值还在、图没了，再 `grad` 会报错，只能重做前向。

日常训练每步都是新前向，一般不必 `retain_graph`。

### 梯度累积实验（预期 2、4、2）

`w = 1`，每次对 `loss = w²` 重新前向再反向。单次梯度是 2。

| 次序 | 操作 | `w.grad` |
|---|---|---|
| 1 | 前向 → `backward` | 2 |
| 2 | **不清梯度**，再前向 → `backward` | 4（2+2） |
| 3 | `w.grad = None`，再前向 → `backward` | 2 |

每次都要重新写 `w**2`。第三次清完也必须再前向，不能对已经用过的旧图再 `backward`。忘了 `zero_grad`，更新会偏大。

### 数值差分（选做）

中心差分近似导数，用来核对「导数大概对不对」，不替代 `backward`：

\[
f'(w) \approx \frac{f(w+\varepsilon)-f(w-\varepsilon)}{2\varepsilon}
\]

手册例子在 `w=1` 时解析梯度是 -12。跑：

```powershell
python -m pytest -q tests/test_02_gradients.py::test_finite_difference_matches_analytic_gradient
```

本周不要自己实现完整 autograd 引擎。

## 第 3 课：最小训练

`TinyRegressor`：`nn.Linear(2, 1)`，属性名必须是 `self.linear`。

```python
def __init__(self) -> None:
    super().__init__()
    self.linear = nn.Linear(2, 1)   # 只赋值，不要 return
```

`__init__` 只能返回 `None`。写成 `self.layer` 时训练能跑，收尾读 `model.linear.weight` 会炸。

看学到的系数：打开 `runs/exercises/baseline/training_summary.json`，或训练后 `print(model.linear.weight)` / `print(model.linear.bias)`。`weight` 形状是 `[1, 2]`，两个数仍对应 `x1`、`x2`。

本机 baseline（`lr=0.05`，200 步）：训练前 MSE ≈ 14.61，训练后 ≈ 3.5e-12；weight ≈ `[[2, -3]]`，bias ≈ `0.5`。

| 对照 | 结果 |
|---|---|
| `lr=0` | 有梯度但不更新，loss 停在约 14.61 |
| `lr=0.005` | 朝真值走，200 步后 MSE ≈ 0.33，未充分收敛 |

有梯度 ≠ 会更新，必须靠非零学习率的 `step`。

### `nn.Linear` 的形状

`Linear(in, out)` 默认带 bias。权重存成 `[out, in]`，前向是：

```text
y = x @ weight.T + bias
```

`bias` 是 `[out]`，不是 `[1, out]`；加法时自动按 `[1, out]` 广播。

这和第 1 课手写 `X @ W`（`W` 为 `[D, K]`）数值等价，只是 `nn.Linear` 把权重存成了 `W` 的转置。所以 `Linear(50, 30)` 的 `weight` 是 `[30, 50]`，不是写反了。

`model(x)` 会走 `__call__` 再进 `forward`，不要直接调 `model.forward(x)`（会绕过 hook）。

### 训练 vs 推理

| | 训练 | 只看输出 |
|---|---|---|
| 模式 | `model.train()` | `model.eval()` |
| 梯度 | 建图，可 `backward` | `torch.no_grad()` |
| 是否改参数 | `step` 会改 | 不改 |

`eval()` 把 `training` 设为 `False`：Dropout 关闭，BatchNorm 用 running 统计。它**不清梯度、不更新参数、不关 autograd**。纯 `Linear+ReLU` 时 `eval`/`train` 数值通常一样，评估仍习惯先 `eval()`。`no_grad` 只管记不记梯度。两个常一起用；训练要切回 `train()`。

`loss` 必须是有限实数。`NaN` / `Inf` 就是非有限值，常见原因是学习率过大或数据已坏。本包会直接报错拦住。

### Softmax、最后一层、argmax

分类网络最后一层通常**不加** Softmax / ReLU，只返回 logits。`CrossEntropyLoss` 内部已做 `log_softmax + NLL`。模型里再 Softmax 再接 `CrossEntropyLoss` 等于做两遍。

隐藏层要激活（否则多层线性塌成一层）；回归输出层也不加 ReLU，避免截断负数。

`softmax(..., dim=1)`：在类别维上归一化，使该维和为 1。`[batch, classes]` 用 `dim=1` 或 `dim=-1`，不要用 batch 维。

`argmax(logits)` 和 `argmax(softmax(logits))` 结果相同：Softmax 不改变「谁最大」，只把分数变成概率。只要类别，直接 `argmax(logits)`；要置信度才需要 Softmax。

### DataLoader 与杂项

- `num_workers=0`：主进程自己取数据。本周小数据保持 0，少踩 Windows 多进程坑。`>0` 是后台预取，不改变 `batch_size` 或模型。
- `drop_last=True`：丢掉最后一个不满 `batch_size` 的小批次。不删数据集里的样本；验证/测试一般不要丢。
- `torch.set_printoptions(sci_mode=False)`：打印不用科学计数法，不改数值。
- `.item()`：把 0 维张量取出成 Python 数字。已经是 `float` 就不能再 `.item()`。
- `correct = 0.0` 只是习惯，方便除出准确率；Python 3 里 `int / int` 也是 float。

### 选做更深网络

不要改 `TinyRegressor`。另建 `exercises/deeper_regressor.py`，脚本放根目录 `run_deeper.py`（不要叫 `test_*.py`）。

```text
Linear(2,8) → Tanh → Linear(8,1)
参数：8×2+8 + 8+1 = 33
```

同数据、约 200 步后 final loss ≈ 0.057，远不如单层线性的 1e-12。数据本身是线性的，更深加 Tanh 是多余约束。参数多 ≠ 一定更好。

## 第 4 课：保存加载

保存的是 `model.state_dict()`（权重和偏置），不是完整续训检查点。缺优化器状态、步数、调度器、随机数、数据读到何处。只能保证同输入预测一致，不能保证从断点按原轨迹接着训。

验收：同一组 probe 输入，保存前、加载后都前向，用 `torch.testing.assert_close` 加容差比较，并记录 `max_abs_diff`。参考运行可以是 0，跨机器不保证逐位相同；容差内一致即可。

```powershell
python run_lesson.py --lesson 4 --impl exercises
python -m pytest -q tests/test_04_checkpoint.py --impl exercises
```

## 自己踩过的坑

1. `affine` 里 `b.reshape(-1, 1)`：单输出碰巧能过，`K=2` 才暴露。
2. 第一次 `grad` 没加 `retain_graph`，图已拆；之后怎么写都晚了。
3. 累积实验第三次只 `w.grad = None`，忘了再前向。
4. `__init__` 里 `return self.linear`：`TypeError: __init__() should return None`。
5. 属性写成 `self.layer`：训练能降 loss，写日志读不到 `model.linear`。
6. `run_deeper.py` 空文件或未保存就跑，终端 `exit 0` 且没输出；`optimizer` 和 `opt` 名字不一致会在循环里炸。
7. `make_batch(200, 2)` 的第二个参数是 `seed`，不是特征维。

## 本周数字（exercises）

| 项 | 值 |
|---|---|
| 数据 | 32 条，`y = 2*x1 - 3*x2 + 0.5` |
| 主线参数量 | 3 |
| baseline | lr=0.05，200 步，MSE 14.61 → 3.51e-12，系数 ≈ `[2, -3]`、`0.5` |
| lr=0 | loss 不动，`parameter_delta=0` |
| lr=0.005 | 200 步后 MSE ≈ 0.333 |
| 保存前后 `max_abs_diff` | 0.0 |
| 选做 33 参数网络 | 同设定约 200 步后 loss ≈ 0.057 |

## 下周

环境和第 1 周测试能独立跑通后，进入文本与分词（UTF-8、字节级编解码、BPE）。本周不做 BPE、Transformer、LoRA、完整断点续训。
