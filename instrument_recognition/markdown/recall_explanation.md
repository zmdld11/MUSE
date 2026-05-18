# Recall 参数说明与计算方法

## 定义

**Recall（召回率）** 衡量模型"该检出的乐器检出了多少"，公式：

```
Recall = TP / (TP + FN)
```

| 符号 | 含义 | 在乐器识别中的意义 |
|------|------|-------------------|
| TP | 模型正确检出的乐器 | 窗口内有钢琴，模型也判为有钢琴 |
| FN | 模型漏检的乐器 | 窗口内有小提琴，但模型判为没有 |

**Precision（精确率）** 是配对指标：
```
Precision = TP / (TP + FP)
```
FP（误报/幻觉） = 模型说有但实际没有。

**F1 Score** 是两者的调和平均：
```
F1 = 2 × Recall × Precision / (Recall + Precision)
```

## 为什么 Recall 重要

在乐器识别中，**漏检比误报更严重**：

- **漏检（FN）**：模型说"没有钢琴"，但实际有。下游任务（如乐谱提取）会直接丢失这段旋律
- **误报（FP）**：模型说"有电吉他"，但实际没有。这算幻觉，但靠后处理（门控规则、共现矩阵）可以抑制

Recall 低 = 乐器实际在演奏但模型"听不见"。

## 计算方法（train.py）

每 epoch 结束后逐类统计：

```python
val_tp = ((predicted == 1) & (targets == 1)).sum()
val_fn = ((predicted == 0) & (targets == 1)).sum()
recall = val_tp / (val_tp + val_fn + 1e-8)
```

predicted 来自 `torch.sigmoid(outputs) > 阈值`（训练时阈值=0.5）。

## 阈值对 Recall 的影响

Recall 和 Precision 是 trade-off：

| 阈值 | Recall | Precision | 效果 |
|------|--------|-----------|------|
| ↓ 降低 | ↑ 升高 | ↓ 降低 | 检得多但幻觉多 |
| ↑ 提高 | ↓ 降低 | ↑ 升高 | 幻觉少但漏检多 |

这就是 VER3.2 引入**逐类自适应阈值**的原因——每类乐器单独搜索最优阈值：

```python
for thresh in np.arange(0.20, 0.85, 0.05):
    preds = (probs >= thresh).astype(float)
    f1 = 2*TP / (2*TP + FP + FN)
    # 保留使 F1 最大的阈值
```

## 标准对照集上的当前 Recall（VER3.5）

在 6 首 MedleyDB/MoisesDB 真实混音上的评估结果：

| 乐器 | Recall | 漏检窗口 | 活跃窗口 |
|------|--------|---------|---------|
| acoustic guitar | 0.206 | 706 | 889 |
| cello | 0.602 | 457 | 1149 |
| drum set | 0.213 | 1100 | 1397 |
| electric bass | 0.252 | 734 | 981 |
| electric guitar | 0.777 | 119 | 534 |
| flute | 0.095 | 584 | 645 |
| piano | 0.476 | 1222 | 2333 |
| singer | 0.544 | 540 | 1184 |
| synthesizer | 0.009 | 444 | 448 |
| violin | 0.373 | 1733 | 2764 |

**Global Recall: 0.380** — 这意味着 62% 的乐器实际出现窗口被模型漏掉了。

这是当前模型的核心瓶颈：Precision 0.859（误报不多），但 Recall 0.380（漏检严重）。下一步优化方向是**用真实混音训练**来弥合合成→真实的数据分布差距，预期能显著提升 Recall。
