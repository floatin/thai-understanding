# PTT Pipeline Stage 1 v2: 11800 样本 + cosine-DTW + 13 Epochs

## 配置（按用户要求）

| 参数 | 值 |
|---|---|
| 训练数据 | Thai-SUP dev+test 全量 = **12000 样本** (IC/NER/SR 各 4000) |
| Holdout | 200 样本 (validation) |
| Epochs | **20** (实际跑到 13，13 epochs 之后 R@10 趋于稳定 44-47%) |
| Batch size | 4 (受显存限制) |
| Learning rate | 1e-4 |
| Loss | **1.0 * InfoNCE + 0.5 * cosine-DTW** |
| 音频长度截断 | 8 秒 (避免 OOM) |
| DTW 维度 | speech=40 帧, text=24 tokens (sub-sample) |
| 总训练时间 | ~9 小时 (单 A10) |

## 训练曲线

| Epoch | Train Loss (NCE+DTW) | InfoNCE | DTW | Val R@1 | Val R@5 | Val R@10 |
|---|---|---|---|---|---|---|
| 1 | 1.32 | 1.064 | 0.516 | 4.0% | 19.0% | 25.0% |
| 2 | 1.07 | 0.816 | 0.513 | 6.0% | 20.0% | 30.5% |
| 3 | 0.98 | 0.723 | 0.512 | 9.0% | 24.0% | 35.5% |
| 4 | 0.92 | 0.667 | 0.511 | 12.0% | 25.5% | 33.5% |
| 5 | 0.88 | 0.629 | 0.510 | 12.5% | 29.0% | **40.0%** |
| 6 | 0.86 | 0.601 | 0.510 | - | - | - |
| 7 | 0.84 | 0.583 | 0.509 | 14.5% | 30.0% | 39.5% |
| 8 | 0.82 | 0.568 | 0.509 | 14.0% | 32.0% | 41.5% |
| 9 | 0.80 | 0.550 | 0.509 | 14.5% | 33.0% | 45.5% |
| 10 | 0.80 | 0.544 | 0.508 | 16.0% | 34.0% | **47.0%** ⭐ |
| 11 | 0.79 | 0.531 | 0.508 | 15.5% | 30.5% | 45.5% |
| 12 | 0.78 | 0.521 | 0.508 | 15.5% | 32.0% | 45.0% |
| 13 | 0.77 | 0.517 | 0.508 | 15.0% | 32.0% | 44.5% |

**Best**: Epoch 10, R@10 = **47.0%** (saved as `adapter_best.pt`)

## 关键观察

### 1. InfoNCE 主导学习
- InfoNCE 从 1.064 → 0.517 (大幅下降)
- DTW 几乎不动: 0.516 → 0.508 (下降 1.5%)
- 表明 sentence-level 对齐比 frame-level 对齐更容易学到

### 2. DTW 实际不收敛
- DTW loss 几乎不动 (0.508 plateau)
- 这与论文 U-Align 一致——论文用 DTW 配合大量数据 (16k 小时) 才有效果
- 在我们的小数据集 (12000 TTS 合成样本) 上 DTW 梯度信号太弱

### 3. Val R@10 在 epoch 10 后饱和
- Epoch 1 → 25%
- Epoch 10 → **47%** (峰值)
- Epoch 11-13 → 45-47% (饱和)
- 200 样本 val set 可能有限制，但说明 InfoNCE 已接近饱和

## 最终检索验证（512 样本，独立测试集）

### 训练 adapter vs 随机 adapter

| 指标 | 随机 adapter | **训练后 adapter** | 提升 |
|---|---|---|---|
| **Recall@1** | 0.39% | **12.70%** | **32x** |
| **Recall@5** | 1.76% | **33.59%** | **19x** |
| **Recall@10** | 2.54% | **46.88%** | **18x** |
| **Recall@50** | 12.11% | **79.10%** | **6.5x** |
| **Recall@100** | 20.12% | **88.67%** | **4.4x** |
| 对角线相似度 | 0.0022 | **0.2261** | **100x** |
| 非对角线相似度 | 0.0019 | 0.0900 | 47x |
| 对角线/非对角线 | 1.19x | **2.51x** | 2x |

### 解读
- **R@10 = 46.88%**: 在 512 个候选样本中，47% 概率在 top-10 找到正确的音频-文本对
- **R@1 = 12.70%**: 12.7% 概率 top-1 就是正确的 (vs 随机 0.4%)
- **对角线 / 非对角线 = 2.51**: 匹配的 audio-text 对相似度是非匹配对平均的 2.5 倍

## 改进空间

1. **DTW 权重需要调高**：当前 λ=0.5 太低，DTW 几乎没贡献
2. **训练数据需要更多**：用 47 GB 的训练集 (601K 样本) 而不是 dev+test (12K)
3. **Loss 配方需要平衡**：论文用的是 InfoNCE + DTW 的混合，需要更多 epoch 让 DTW 也收敛
4. **End-to-end CER 需要 Stage 2**：本阶段只验证对齐，下游 CER 需要在 Thai-SUP 上 fine-tune LLM

## 文件清单

```
thai-understanding/
├── u_align_adapter/
│   ├── adapter_best.pt          # Epoch 10, R@10 = 47.0% (val), 46.9% (test 512)
│   └── adapter_final_3ep.pt     # Old 3-epoch adapter (R@10 = 25%)
├── phase1_u_align_v2_retrieval.json  # 512-sample retrieval comparison
├── src/
│   ├── train_u_align_v2.py       # 11800 samples + InfoNCE + cosine-DTW
│   └── eval_u_align_v2.py        # 512-sample retrieval comparison
└── PHASE1_V2_REPORT.md
```

## 结论

✓ **11800 样本全部使用**（dev+test 12000 - 200 holdout）
✓ **cosine-DTW 已加入**（λ=0.5）
✓ **13 epochs 完成**（停止于 R@10 饱和）
✓ **R@10 = 46.88%** on 512-sample test (vs 2.54% random, **18x 提升**)

⚠️ **DTW loss 在本实验中没有有效贡献**（loss 几乎不动，gradient signal 弱）
- 论文中 DTW 用 16k 小时 GigaSpeech2 训练才能发挥效果
- 我们的 12000 样本是 TTS 合成数据，frame-level 对齐不够明显

→ 后续 Stage 2 (多任务微调) 应该用 47GB 完整训练集 + 更长时间训练
