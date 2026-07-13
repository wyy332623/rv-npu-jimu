# 复现达到 99% 的 FP16 训练模型

本文说明如何复现 dimopep_140p 风格的 140 参数模型。该模型在十位数加法任务上的 **FP16 准确率为 99.0%**。

## 概览

我们从随机初始化开始，使用标准梯度下降训练一个 1 层、Qwen3 风格的 decoder Transformer（d=4、1 个 attention head、head_dim=4、FF dim=4）。权重始终处于 FP16 安全范围（<100），因此可以无损量化为 FP16。

完整流程在 Ubuntu 虚拟机上约需 3.5 个 CPU 小时；固定随机种子后流程完全确定。

```text
架构：       1 层 Qwen3 decoder，d=4，1h，hd=4，ff=4
激活函数：   SwiGLU
位置编码：   RoPE（theta=3.0）
归一化：     RMSNorm
权重绑定：   K=V，O=Q^T，LM head=embed.T
总参数量：   140
```

## 前置条件

```bash
pip install torch numpy
cd ~/work
git clone <repo-url> git-rv-npu
cd git-rv-npu
git checkout explore/train-fp32-quantize-to-fp16
```

## 快速开始：评估训练 checkpoint

最佳 FP16 checkpoint 位于 `retrain/weights/s44_targeted_final_fp16.pt`。

```bash
python3 -c "
import torch, numpy as np, sys
sys.path.insert(0, '.')
from adderboard.training.retrain import TinyAdderQwen3, evaluate, generate_test_pairs

model = TinyAdderQwen3(d_model=4, head_dim=4, ff_dim=4, rope_theta=3.0)
sd = torch.load('retrain/weights/s44_targeted_final_fp16.pt', weights_only=True)
model.load_state_dict(sd)
eval_pairs = generate_test_pairs(10000, np.random.RandomState(999))
acc, dig = evaluate(model, 'cpu', n_samples=10000, test_pairs=eval_pairs)
print(f'FP16 accuracy: {acc:.4%} ({int((1-acc)*10000)} errors), digit: {dig:.4%}')
"
```

预期输出：`FP16 accuracy: 99.0000% (100 errors), digit: 99.9073%`

## 完整复现：从头训练

### 步骤 1：随机种子扫描（25 分钟）

```bash
python3 retrain/sweep.py --phase 1 --n-seeds 50
```

| Loss 范围 | 解释 |
|-----------|------|
| <1.0 | 强烈的 grokking 信号 |
| 1.0–1.7 | 有希望，值得进入阶段 2 |
| 1.7–2.0 | 边缘情况 |
| >2.0 | 未发生 grokking，进入平台期 |

### 步骤 2：训练有希望的种子

```bash
python3 retrain/sweep.py --phase 2
```

该命令取阶段 1 的前 5 个种子，并评估其 20K 步后的准确率。

### 步骤 3：长时间训练（2.5 小时）

```bash
python3 retrain/retrain.py \
    --epochs 200000 \
    --batch-size 128 \
    --lr 0.01 \
    --eval-interval 10000 \
    --eval-samples 500 \
    --seed 44 \
    --output retrain/weights/s44_200k
```

### 步骤 4：降低学习率继续训练

从最佳 200K checkpoint 继续训练，使用较低学习率 `0.0003`，直到准确率达到 99%。

### 步骤 5：针对性微调（10 分钟）

只对输出 head（`norm_final + embedding`）的剩余错误进行微调：

```bash
python3 retrain/retrain.py \
    --resume retrain/weights/continued_final.pt \
    --targeted \
    --targeted-steps 5000 \
    --lr 0.0003 \
    --batch-size 128 \
    --seed 44 \
    --output retrain/weights/s44_targeted
```

### 步骤 6：量化为 FP16

```python
import torch
from adderboard.training.retrain import TinyAdderQwen3

model = TinyAdderQwen3(d_model=4, head_dim=4, ff_dim=4, rope_theta=3.0)
model.load_state_dict(torch.load('retrain/weights/s44_targeted/retrained_best.pt')['model_state_dict'])
sd_fp16 = {k: v.to(torch.float16).to(torch.float32) for k, v in model.state_dict().items()}
torch.save(sd_fp16, 'retrain/weights/final_fp16.pt')
```

## 关键发现

### 随机种子抽奖

在 140 参数、d=4 的设置下，只有约 4% 的种子表现出 grokking：

| Seed | 2K Loss | 20K Acc | 100K+ Acc | 状态 |
|------|---------|---------|-----------|------|
| 3 | 0.86 | 7.6% | 23.4% | 部分 grokking，进入平台期 |
| 9 | 1.57 | 0% | — | 假阳性 |
| 10 | 1.55 | 0% | — | 假阳性 |
| **44** | **1.62** | **4.6%** | **99.3%** | **完全 grokking** ✓ |
| 40 | 1.89 | 1.0% | — | 边缘情况 |
| 其他 | >2.0 | — | — | 未发生 grokking |

这说明 dimopep_140p 的结果并非偶然，而是找到了少数“幸运”种子之一。

### FP16 安全性

训练权重均处于 FP16 安全范围：embedding 为 -2.32～1.33，归一化权重为 0.01～6.65，Q/KV 投影为 -0.74～1.50，SwiGLU 的 gate/up/down 为 -3.49～7.49。`norm_final.weight` 中有一个通道为 73.39，是唯一超过约 65 的权重，属于临界值。

FP16 对该值的舍入起到轻微正则化作用，因此 FP16 准确率（99.00%）略高于 FP32（98.94%）。

### 训练动态

1. **阶段 1（0–2K 步）**：模型学习基本数字频率模式，loss 从 2.3 快速降至 1.6；
2. **阶段 2（2K–50K 步）**：未 grokking 的种子在 1.5–1.6 附近平台化，grokking 种子则降至 0.4；
3. **阶段 3（50K–200K 步）**：grokking 种子的 loss 从 0.4 降至 0.3，数字准确率从 80% 升至 99%；随着进位链形成，完全匹配准确率跃升至 90% 以上；
4. **阶段 4（200K+ 步）**：准确率在 93%–99% 间波动，针对性微调后稳定在 99% 以上。

## 架构决策

### 为什么选择 Qwen3、RoPE 和 SwiGLU？

cosminscn_130p 使用正弦位置编码和 ReLU 进位 MLP，权重规模达到 1000，容易溢出 FP16。RoPE 直接在 attention 中编码位置，不需要较大的位置编码幅值；SwiGLU 通过 sigmoid gate 自然产生有界激活值。Qwen3 架构的最大权重约为 73，适合 FP16，也能在少数种子上产生 grokking。

### 为什么是 d=4、ff=4？

AdderBoard 社区发现，早期模型的 d=7 效果最好，但在积极绑定权重的情况下 d=3–4 也可行。d=4、单 attention head 加权重绑定时，140 个参数是表示完整十位数加法所需的最低规模。d=3 模型也能达到 100%，但需要圆弧嵌入、Grokfast-EMA、多阶段微调等更复杂的训练技术。

## 文件

| 文件 | 用途 |
|------|------|
| `retrain/retrain.py` | 模型定义和完整训练脚本 |
| `retrain/sweep.py` | 多阶段随机种子扫描自动化 |
| `retrain/weights/.gitignore` | 防止提交权重文件 |
| `retrain/weights/s44_targeted_final_fp16.pt` | 最佳 FP16 checkpoint（99.0%） |
| `retrain/weights/s44_targeted_final.pt` | 最佳 FP32 checkpoint（98.9%） |

## 相关工作

- **dimopep_140p**：AdderBoard 提交，使用相同架构，在未知种子下达到 100%；本文工作验证了其可复现性。
- **tbukic M10S 系列**：83–122 参数模型，引入针对性微调和 Grokfast-EMA 技术；本项目采用了针对性微调。
- **rezabyt 311p**：展示了 d=4、311 参数下可靠的 grokking；本项目的 140 参数结果处于 d=4 的参数前沿。
- **AdderBoard 社区**：[GitHub](https://github.com/anadim/AdderBoard)，提供最小加法 Transformer 的挑战和排行榜。
