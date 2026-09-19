# 本机低算力大模型全链路实战学习计划（8周敏捷版）

> **设计原则**：缩小模型规模（20M~50M 参数），跑通 100% 工业级全链路；依赖纯 PyTorch 张量实现，拒绝高层黑盒库封装；完全适配本地 CPU / Apple Silicon (MPS) / 消费级显卡（4GB~8GB 显存）。

---

## 一、 课程与开源基建映射表

| 学习阶段 | 核心理论映射 | 核心代码实践参考 | 算力与硬件基准 |
| :--- | :--- | :--- | :--- |
| **W1: 分词与数据** | CS336 Assignment 1 (Tokenization) | `karpathy/minbpe`, Hugging Face `tokenizers` | 纯 CPU，内存 < 2GB |
| **W2-W3: 架构手搓** | 清华唐杰课程（Transformer/GLM架构演进） | `rasbt/LLMs-from-scratch`, CS336 Assignment 1 | 纯 CPU / MPS / 2GB 显存 |
| **W4-W5: 预训练工程** | CS336 Assignment 2 (Training Systems) | `karpathy/nanoGPT`, `jingyaogong/minimind` | 4GB~8GB 显存或 Colab T4 |
| **W6: SFT与对齐** | 清华唐杰课程（RLHF/DPO）、DPO 论文 | `huggingface/alignment-handbook`, MiniMind | 4GB~8GB 显存或 Colab T4 |
| **W7: 推理系统与优化** | UC Berkeley CS294-196 (LLM Systems) | `ggerganov/llama.cpp`, vLLM 核心算法仿真 | 纯 CPU（量化部署）/ 本机 GPU |
| **W8: 综合交付与整合** | 系统整合与工程沉淀 | 产出个人专属 GitHub 仓库 | 本机环境 |

---

## 二、 8 周逐周攻坚执行方案

### Week 1：分词器与字节对编码（Byte-level BPE from Scratch）
* **核心目标**：从零构建现代分词器，理解 BPE 算法、Regex 切分规则与 UTF-8 编解码底层机制。
* **攻坚任务**：
  * **Day 1-2**：研读 CS336 Assignment 1 分词部分，理解 GPT-2/LLaMA-3 正则表达式切分原理，避免标点与数字发生跨边界错误合并。
  * **Day 3-5**：用纯 Python 手写 Byte-level BPE 分词器：
    * 初始化 256 个基础字节词表及特殊控制字符（如 `<|im_start|>`, `<|im_end|>`, `<|pad|>`）。
    * 实现字符对频次统计与循环合并逻辑，输出 `vocab.json` 和 `merges.txt`。
    * 实现 `encode()` 与 `decode()` 函数，严格保证 UTF-8 多字节字符（中文、Emoji）编解码的可逆性。
  * **Day 6-7**：对基准文本（如 Wiki 中英文混合切片）进行分词，对比官方 `tiktoken`，验证压缩比与吞吐速度。
* **交付件**：`tokenizer.py`，包含完整的特殊字符处理、单测用例与分词一致性校验。

---

### Week 2：现代 Transformer 核心算子手写
* **核心目标**：脱离 `torch.nn.Transformer`，用 PyTorch 纯张量手写主流开源大模型（LLaMA-3 / Qwen-2）的底层组件。
* **攻坚任务**：
  * **Day 1-2**：实现 **RMSNorm**，推导并验证无均值计算下的数值稳定性与前向/后向梯度流。
  * **Day 3-4**：实现 **RoPE（旋转位置编码）**：
    * 计算预置复数频率矩阵 $\cos$ 与 $\sin$。
    * 编写二维复数旋转计算逻辑，实现序列维度的相对位置编码变换。
  * **Day 5-6**：实现 **SwiGLU 门控前馈网络**：
    * 按照 $SwiGLU(x) = (xW_{gate} \odot \text{silu}(xW_{up})) W_{down}$ 搭建两层升维与一层降维线性层。
  * **Day 7**：编写针对各个算子的 PyTest 单元测试，打印并验证各组件的张量 Shape 变换。
* **交付件**：`modules.py`，包含手写的 RMSNorm、RoPE 与 SwiGLU 模块。

---

### Week 3：多头/分组注意力与因果解码器组装
* **核心目标**：实现带因果掩码的 GQA（分组查询注意力），组装出可端到端运行的微型模型 `NanoLLM`。
* **攻坚任务**：
  * **Day 1-3**：手写 `CausalSelfAttention`：
    * 支持 Grouped-Query Attention (GQA)：设置 $N_{kv\_heads} < N_{q\_heads}$，实现 KV 头向 Q 头的广播/复制机制。
    * 构建严格下三角因果注意力掩码（Causal Mask），保证自回归生成的因果约束。
    * 结合上周的 RoPE 对 Query 和 Key 张量进行位置注入。
  * **Day 4-5**：组装完整的 `TransformerBlock` 与解码器主干网络：
    * 嵌入层 Embedding $\rightarrow$ $N \times$ Block（RMSNorm $\rightarrow$ GQA $\rightarrow$ 残差连接 $\rightarrow$ RMSNorm $\rightarrow$ SwiGLU $\rightarrow$ 残差连接） $\rightarrow$ 最终 RMSNorm $\rightarrow$ 输出投影层 LM Head。
  * **Day 6-7**：定义模型超参数（25M 规格）：`dim=512`, `n_layers=8`, `n_heads=8`, `n_kv_heads=2`, `vocab_size=6400`。
* **交付件**：`model.py`，完成整网前向推理测试，确保输入随机 Tensor 能稳定输出 Logits。

---

### Week 4：高效预训练基建与单机训练工程
* **核心目标**：建立高效数据流管道与单机训练循环，掌握混合精度与显存压缩核心工程技术。
* **攻坚任务**：
  * **Day 1-2**：数据工程与二进制序列化：
    * 下载 **TinyStories** 数据集（约 500MB，非常适合低算力学习语法与连贯叙事）。
    * 使用 Week 1 的 Tokenizer 将纯文本转为一维 `uint16` 二进制文件，利用 `np.memmap` 实现零内存常驻的高性能切片读取。
  * **Day 3-4**：编写工业级训练器 `train.py`：
    * 引入 `torch.autocast` 开启 BF16 / FP16 混合精度训练。
    * 编写梯度累加（Gradient Accumulation）逻辑，用小显存等效模拟大 Batch Size。
    * 引入梯度裁剪（Gradient Clipping）与带有预热（Warmup）的余弦退火（Cosine Decay）学习率调度器。
  * **Day 5-7**：训练状态监控与 Checkpoint 管理：
    * 手写周期性保存权重（包含 Model、Optimizer、Step、Loss 等元信息）与断点续训机制。
    * 集成 TensorBoard / WandB 实时监控训练 Loss 与 GPU/CPU 内存占用。
* **交付件**：`dataset.py` 与 `train.py`，完成训练工程闭环构建。

---

### Week 5：微型模型预训练实战与显存解构
* **核心目标**：在低成本资源下完成一次从零预训练，从理论与实践两侧解构显存消耗与收敛状态。
* **攻坚任务**：
  * **Day 1-3**：启动 25M 模型预训练（耗时约 2~4 小时）：
    * 配置：`batch_size=16`, `grad_accum_steps=4`（等效 Batch Size = 64），`seq_len=512`。
    * 观察训练损失从 8.0+ 稳步下降到 1.5 左右，确认模型完成初步语法收敛。
  * **Day 4-5**：理论推导与显存实测对齐：
    * 显式计算模型静态显存（参数量 $\times 2$ 字节）、优化器状态显存（AdamW：两份状态动量 $\times 4$ 字节）与前向激活值显存。
    * 打印 `torch.cuda.memory_allocated()` 与理论公式做对照验证。
  * **Day 6-7**：自回归生成采样初体验：
    * 编写最简贪心采样（Greedy Search）与温度采样（Temperature + Top-p 采样）脚本，给入前缀 Prompt 观察模型续写能力。
* **交付件**：预训练权重文件 `pretrained.pt`，以及包含显存推导公式与 Loss 收敛曲线的实验报告。

---

### Week 6：指令微调（SFT）与偏好对齐（DPO）手搓实战
* **核心目标**：深入后训练（Post-Training）阶段，掌握 Prompt Loss Masking 原理及手写 DPO 闭式解优化。
* **攻坚任务**：
  * **Day 1-2**：手写 SFT 数据整理与动态掩码：
    * 采用标准 Chat 模板构建对话流（`<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n{A}<|im_end|>`）。
    * 在计算交叉熵损失时，将 Prompt 部分的目标 Token 统一设为 `-100`，强制模型仅对 Assistant 的生成内容计算梯度。
  * **Day 3-4**：微调训练与行为对比：
    * 使用约 5,000 条高质量指令微调数据集（如精简版 Alpaca-ZH）运行 SFT 训练。
    * 对比观察模型由“纯文本自由续写”转变为“理解问答指令并作出回应”的行为变化。
  * **Day 5-6**：手写 DPO（直接偏好优化）对齐损失函数：
    * 构建策略模型 $\pi_\theta$ 与冻结参考模型 $\pi_{ref}$。
    * 根据数学公式纯手写 DPO Loss：
      $$\mathcal{L}_{DPO} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w\vert{}x)}{\pi_{ref}(y_w\vert{}x)} - \beta \log \frac{\pi_\theta(y_l\vert{}x)}{\pi_{ref}(y_l\vert{}x)} \right)$$
    * 在少量偏好数据对（Choose/Reject Pair）上运行 200~500 步轻量对齐更新。
  * **Day 7**：对齐能力评测：对比 SFT 与 DPO 模型在拒绝恶意输入或输出风格上的差异。
* **交付件**：`sft_train.py` 与 `dpo_train.py`，产出微调模型权重 `nano_chat.pt`。

---

### Week 7：推理加速、KV-Cache、量化与分布式仿真
* **核心目标**：解构大模型推理中的“显存受限”（Memory-bound）瓶颈，掌握单机分布式并行与低比特部署方案。
* **攻坚任务**：
  * **Day 1-3**：手写 KV-Cache 动态缓存逻辑：
    * 改造 `CausalSelfAttention`，在自回归 Decode 循环中存储历史 Key 和 Value 向量。
    * 逐 Token 推进推理，消解 $O(N^2)$ 重复计算降至 $O(N)$，对比测算加速比。
  * **Day 4**：投机采样（Speculative Decoding）算法实现：
    * 以 20M 模型为 Draft Model，加载同词表的 50M 模型为 Target Model，手写前向验证并计算 Token 接受率（Acceptance Rate）。
  * **Day 5**：模型导出与端侧推理体验：
    * 编写权重导出脚本，将自研 PyTorch 权重转化为 GGUF 格式。
    * 本地编译并运行 `llama.cpp`，体验 CPU 极低延迟推理。
  * **Day 6-7**：分布式系统仿真（单机多进程 DDP 实践）：
    * 使用 `torch.distributed.run` 模拟 2 个进程环境。
    * 手写单步 All-Reduce 梯度平均同步过程，理解分布式数据并行的底层逻辑。
* **交付件**：`inference.py`（含 KV-Cache 与投机采样），模型 GGUF 文件与部署说明文档。

---

### Week 8：全链路工程闭环与个人代码库打磨
* **核心目标**：重构代码目录，补充完善单元测试，固化全流程研发成果。
* **攻坚任务**：
  * **Day 1-3**：项目代码模块化重构：
    * 建立符合开源标准的项目结构（数据处理、网络模型、训练管道、部署脚本相互解耦）。
  * **Day 4-5**：补齐基准测试与说明文档：
    * 编写性能基准压测脚本（测定 Prefill 与 Decode 阶段的每秒吞吐 Token 数）。
    * 撰写完整的 `README.md`，附带模型架构参数配置、显存占用计算表与 Loss 变化动图。
  * **Day 6-7**：全链路端到端联合验收回归，复现从零开始的分词、预训练、微调与终端交互推理。
* **交付件**：一个结构完备、可独立运行的个人开源项目仓库（例如 `NanoLLM-Engine`）。

---

## 三、 本地低显存核心配置参考

若在显存受限设备上运行，可直接采用以下参数矩阵以确保稳定不 OOM：

```python
# configs/nano_config.py
class NanoLLMConfig:
    vocab_size: int = 6400        # 精简词表，大幅削减 Embedding 显存
    max_seq_len: int = 512        # 序列长度
    dim: int = 512                # 隐藏层维度
    n_layers: int = 8             # Transformer 层数
    n_heads: int = 8              # Query 注意力头数
    n_kv_heads: int = 2           # GQA 分组 KV 头数 (4:1 比例压缩)
    hidden_dim: int = 1408        # SwiGLU 中间层升维大小 (约 8/3 * dim)
    norm_eps: float = 1e-6

# 训练阶段显存压榨配置
class TrainConfig:
    batch_size: int = 16          # 实体 Batch Size
    gradient_accumulation: int = 4 # 累加步数，等效 Batch Size = 64
    mixed_precision: str = "bf16" # 支持 MPS/现代英伟达架构，老卡切 "fp16"
    gradient_clipping: float = 1.0
    weight_decay: float = 0.1
    learning_rate: float = 5e-4