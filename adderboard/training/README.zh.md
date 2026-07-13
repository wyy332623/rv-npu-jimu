> 本文件由自动翻译生成，仅供参考；以英文原文为准。

# 复制FP16 99%的训练模式

此文档描述如何复制训练有素的140 Parm dimopp  140p 风格
在10位数添加时达到**99.0%FP16精度**的模型。

## 概览

我们训练一台1层的Quen3型解码器变压器(d=4,1注意头,头 dim=4,).
FF dim=4)从使用标准梯度下降的随机初始化. 重量留着
在FP16-安全范围内(<100),使无损失的量化达到FP16。

整条管道在Ubuntu VM上耗时~3.5 CPU小时. 所有步骤都是决定性的
给一个固定的种子。

```
Architecture:  1L Qwen3 decoder, d=4, 1h, hd=4, ff=4
Activation:    SwiGLU
Positional:    RoPE (theta=3.0)
Normalization: RMSNorm
Tied weights:  K=V, O=Q^T, LM head = embed.T
Total params:  140
```

## 先决条件

```bash
# Python 3.10+ with PyTorch and numpy
pip install torch numpy

# Clone the repo and checkout the branch
cd ~/work
git clone <repo-url> git-rv-npu
cd git-rv-npu
git checkout explore/train-fp32-quantize-to-fp16
```

## 快速启动: 评价训练有素的检查站

最好的FP16检查站位于ZPROT000XQZ.

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

预期产出:

## 完全复制:从涂鸦出发的列车

### 步骤1:种子扫荡(25分钟)

寻找有希望的种子,在50种不同种子上跑2000步:

```bash
python3 retrain/sweep.py --phase 1 --n-seeds 50
```

这会产生 QQZPROT000XQZ ,每个种子都有损失值.
损失大大低于~2.15基线的种子有希望:

|损失范围|口译|
|---|---|
| <1.0 |强烈的杂音信号|
| 1.0-1.7 |前景——值得第二阶段|
| 1.7-2.0 |边际|
| >2.0 |没有杂耍( 台词)|

### 第2步:火车有希望种子(每个种子5个)

运行每个有希望的种子 20 000 步:

```bash
python3 retrain/sweep.py --phase 2
```

这取自第一阶段前5名种子,并在20K步骤后评价精度.

### 第3步:长期培训(2.5小时)

运行最好的种子20万步:

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

### 第4步:在低纬度地区继续培训(30分钟)

从最好的200K检查站继续学习,降低学习率:

```python
# script: continue_training.py
import sys, torch, numpy as np
sys.path.insert(0, '.')
from adderboard.training.retrain import TinyAdderQwen3, generate_batch, evaluate, train_step

model = TinyAdderQwen3(d_model=4, head_dim=4, ff_dim=4, rope_theta=3.0)
model.load_state_dict(torch.load('retrain/weights/s44_200k/retrained_best.pt')['model_state_dict'])

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
rng = np.random.RandomState(86)

for step in range(1, 100001):
    full_seq, labels = generate_batch(128, 'cpu', 10, rng=rng)
    loss = train_step(model, full_seq, labels)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 5000 == 0:
        acc, _ = evaluate(model, 'cpu', n_samples=1000)
        print(f"Step {step}: accuracy={acc:.4f}")
        if acc >= 0.99:
            torch.save(model.state_dict(), 'retrain/weights/checkpoint_99pct.pt')

torch.save(model.state_dict(), 'retrain/weights/continued_final.pt')
```

### 第5步:定向罚款(10分钟)

只调整剩余错误的输出头( Norm  final + 嵌入) :

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

### 步骤6:量化至FP16

```python
import torch
from adderboard.training.retrain import TinyAdderQwen3

model = TinyAdderQwen3(d_model=4, head_dim=4, ff_dim=4, rope_theta=3.0)
model.load_state_dict(torch.load('retrain/weights/s44_targeted/retrained_best.pt')['model_state_dict'])

sd_fp16 = {k: v.to(torch.float16).to(torch.float32) for k, v in model.state_dict().items()}
torch.save(sd_fp16, 'retrain/weights/final_fp16.pt')
```

## 主要调查结果

### 种子彩票

只有~4%的种子(每50个种子中有2个)在140个parms中表现出杂乱行为,其中d=4:

|种子|2K 损失|20K Acc (英语).|100K+ Acc( 百K+ Acc) (百K+ Acc) (百K+ Acc)|状态|
|---|---|---|---|---|
| 3 | 0.86 | 7.6% | 23.4% |部分沟槽, 固定|
| 9 | 1.57 | 0% | — |假阳性|
| 10 | 1.55 | 0% | — |假阳性|
| **44** | **1.62** | **4.6%** | **99.3%** |** 全部杂务 * * * * * * *|
| 40 | 1.89 | 1.0% | — |边际|
|所有其他国家| >2.0 | — | — |没有杂音|

这证实了 Dimopep  140p 呈文不是一个故障—— 它需要找到
罕见的"幸运"种子之一。

### FP16 安全性

所有受过训练的重量均在FP16安全范围内:

|重量|最小数|马克斯|FP16安全吗?|
|---|---|---|---|
|嵌入| -2.32 | 1.33 | ✅ |
|规范权重| 0.01 | 6.65 | ✅ |
|规范( L)| -13.31 | 73.39 |QQ 边线|
|ZPROT000Z 预测| -0.74 | 1.50 | ✅ |
|SwiGLU(ZPROT000XZ) 互联网档案馆的存檔,存档日期2014-09-02.| -3.49 | 7.49 | ✅ |

QQZPROT000XQZ有一个频道73.39——唯一重量超过~65.
FP16这一数值的四舍五入作用为轻度规范化,这解释了为什么FP16
精确度(99.00%)略高于FP32(98.94%)。

### 培训动态

拼接过程遵循一个特征模式:

1. ** 第1阶段(0-2K步)**: 随着模型的学习,损失从2.3~1.6迅速下降
   基本数字频率模式
2. ** 第2阶段(2K-50K步)**: 非葡萄种子损失高原1.5-1.6左右,
   或减到0.4 对于杂食种子。 数字精度从~10%上升到~80%
3. ** 第3阶段(50K-200K步)**: 对于杂交种子,损失从0.4 ~ 0.3减少
   而数字精度则从80%~99%攀升. 精确匹配的精度跳跃
   随着承载链结晶,近零至90QQ
4. ** 第4阶段(200K+步骤)**: 随着模型的完善,精确振荡率为93-99%
   高位数负责决策。 目标FT稳定在99QQ

## 结构决定

### 为什么Quen3 / RoPE / SwiGLU?

Cosminscn 130p架构(sinusoidal PE,ReLU 承载 MLP)使用1000级
超过FP16的重量。 我们需要一个所有重量 自然而然的建筑
保持在100以下:

|建筑|最大重量|FP16安全吗?|格罗克斯?|
|---|---|---|---|
|cosminscn 130p(sin PE + ReLU) (中文(简体) ).| 44,248 | ❌ |ZPROT000Z(手工制作)|
|Quen3 (ROPE + SwiGLU) (英语).| 73 | ✅ |是(稀有种子)|

RoPE 直接编码位置,从而不再需要大型PE
振动. SwiGLU通过Sigmoid门自然产生有界激活.

### 为什么d=4,ff=4?

AdderBoard社区发现d=7 是早期模型的甜点,但
d = 3-4 具有攻击性重量捆绑。 在D=4时,有1个注意头和捆绑
权重,140个参数是能够代表完全10位加法的最小值.

d=3种模式(36-101段)达到100%,但需要更细致的培训
技术(圆弧嵌入、Grokfast-EMA、多级微调)。

## 文件

|文件|目的|
|---|---|
|津巴布韦|模式定义+完整的培训脚本|
|津巴布韦|多阶段种子扫描自动化|
|津巴布韦|防止执行重量文件|
|津巴布韦|最佳FP16检查站(99.0%)|
|津巴布韦|最佳FP32检查站(98.9%)|

## 有关工作

- ** dimopep 140p** (AddBoard提交): 相同的结构,100%精确度
  种子不详。 我们的工作证实了可复制性。
- ** tbukic M10S系列**(83-122段): 引入有针对性的微调和
  Grokfast-EMA技术. 我们通过了目标FT。
- **雷扎比特311p**: d=4有311个参数的显示可靠凹槽。
  我们的140段结果位于当前d=4的参数边框.
- ** AdderBoard社区**:ZPROT000Z——挑战
  和最小增压变压器的导板。
