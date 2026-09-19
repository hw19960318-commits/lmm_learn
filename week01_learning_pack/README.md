# 第1周学习包

四次课：张量 → 自动求导 → 最小训练 → 保存加载。默认 CPU，不下载大模型和外部数据。

## 打开顺序

1. `docs/第一周课件.html`：离线中文讲义，浏览器打开即可。也提供可编辑 `docs/第一周课件.pptx`。
2. `docs/第一周学习手册.md`：阅读范围、每次任务、安装与命令、24道题解。
3. `exercises/`：填写16个编号TODO；`reference/` 是独立参考答案。

## 安装

在本目录建立并激活虚拟环境。PyTorch 按 https://pytorch.org/get-started/locally/ 选择本机 CPU 安装方式；然后安装 requirements.txt。具体 Windows/macOS/Linux 命令见学习手册。

```bash
python -m venv .venv
# 先激活环境，再按官方安装页安装 torch
python -m pip install -r requirements.txt
python check_env.py
python -m pytest -q --impl reference
```

## 每次课怎么做

```bash
# 第1课：补 exercises/tensor_ops.py
python run_lesson.py --lesson 1 --impl exercises
python -m pytest -q tests/test_01_tensors.py --impl exercises

# 第2课：补 exercises/autograd_lab.py
python run_lesson.py --lesson 2 --impl exercises
python -m pytest -q tests/test_02_gradients.py --impl exercises

# 第3课：补 exercises/model.py、training.py
python run_lesson.py --lesson 3 --impl exercises
python -m pytest -q tests/test_03_training.py --impl exercises
python run_lesson.py --lesson 3 --impl exercises --lr 0 --run-name lr0

# 第4课：补 exercises/checkpoint.py
python run_lesson.py --lesson 4 --impl exercises
python -m pytest -q --impl exercises
```

查看参考运行，把命令中的 `exercises` 改为 `reference`。默认 `pytest` 检查 exercises；未补完时测试失败是正常的。不要修改测试来凑通过。

## 项目结构

```text
week01_learning_pack/
├── README.md
├── requirements.txt
├── check_env.py            # 现成：环境检查
├── common.py               # 现成：合成数据与基本检查
├── run_lesson.py           # 现成：四次课统一入口
├── conftest.py             # pytest --impl 选择器
├── docs/                   # 中文讲义、PPT、资料、题解
├── exercises/              # 你要补全的5个练习模块
├── reference/              # 完整参考答案，接口相同
├── tests/                  # 24项测试，按课次分组
├── runs/reference/         # 制作时的真实参考日志
└── reports/                # 环境、测试记录、你的复盘模板
```

## 说明

本包的任务、代码和中文课件为本次定制编写，官方来源链接见学习手册。无需账号或收费API。只加载自己生成或可信来源的权重。日志中的参考结果不代表你本机的结果；实测环境与限制见 reports/verification.md。参考测试通过不等于你已经完成练习。
