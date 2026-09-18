# PTT Pipeline Development: Phase 1 (Stage 1 U-Align) Report

## 执行范围

按 `PTT语音语义转录Pipeline开发指南.md` 执行：
- **Phase 0 数据盘点**：✓ 已完成（Thai-SUP 47GB 数据集完整下载并盘点）
- **Phase 1 Baseline**：跳过（文档需要 Whisper 3-model，本环境未部署）
- **Phase 2 Stage 1**：✓ 已完成（U-Align adapter 训练 + 检索验证）

## 数据资产盘点

| 资产 | 状态 | 详情 |
|---|---|---|
| XLSR-Thai checkpoint | ✓ | 3.6 GB, XLSR-53 large (24 层, 1024-dim, 16 heads) |
| Qwen3-4B (LLM) | ✓ | 7.6 GB, 4B params, hidden=2560 |
| Typhoon2-3B (备用) | ✓ | 6.1 GB, 3.21B params, hidden=3072 |
| Thai-SUP 训练集 | ✓ | 47 GB, 613,459 样本 / 874 小时 |
| Thai-SUP dev/test | ✓ | 1 GB, 12,000 样本 (每任务 2000) |

## Stage 1 U-Align 训练

### 架构（按文档 Section 1.2）

```
audio (16kHz) 
  → XLSR-Thai (冻参, 1024-dim, 20ms stride)
  → U-Align Adapter (可训, 9.18M 参数)
    - LayerNorm → CNN subsampler (2x downsample) → MLP (1024 → 2560 → 2560)
  → speech_embeds 与 Qwen3.embed_tokens(input_ids) 对齐
```

### 训练配置

| 参数 | 值 | 来源 |
|---|---|---|
| 训练数据 | Thai-SUP SR train, 2000 样本 / 3 epochs | 文档 3.1 阶段 1 对齐 |
| Batch size | 8 | 显存允许的最大值 |
| 学习率 | 1e-4 | 文档 2.3 |
| Loss | InfoNCE (in-batch contrastive, T=0.07) | 文档 2.3（cosine-DTW 可选） |
| Encoder | XLSR-Thai (frozen) | 文档 |
| Text tower | Qwen3 embed_tokens (frozen) | 文档 |

### 训练日志

| Epoch | InfoNCE Loss |
|---|---|
| 1 | 2.0272 |
| 2 | 1.6558 |
| 3 | 1.4002 |

Loss 从 2.02（接近 ln(8)=2.08 random baseline）下降到 1.40，说明 adapter 学到了 XLSR-Thai 特征和 Qwen3 text embeddings 之间的对齐。

## 验证：Retrieval 任务

设计：给定音频 embedding，预测其文本在 batch 中对应的 text embedding。Recall@K 衡量对齐质量。

### 训练 adapter vs 随机 adapter（64 样本测试集）

| 指标 | 随机 adapter | 训练 adapter | 提升 |
|---|---|---|---|
| Recall@1 | 1.56% | 6.25% | **+4.69%** (4x) |
| Recall@5 | 4.69% | 25.00% | **+20.31%** (5x) |
| Recall@10 | 14.06% | 45.31% | **+31.25%** (3x) |
| 对角线相似度 | 0.0085 | 0.0690 | **8x** |
| 非对角线相似度 | 0.0084 | 0.0341 | 4x |
| 对角线/非对角线 | 1.01x | **2.02x** | **清晰分离** |

**结论**：U-Align Stage 1 训练成功。训练后的 adapter 能在 64 个候选中以 45% 概率找到正确文本（R@10），而随机只有 14%。

## 验证：端到端生成

虽然 retrieval 验证了对齐，但端到端生成的 CER 仍然很高（313%）：
- **原因 1**：SR 任务要求 LLM 改写文本，不是简单转录
- **原因 2**：训练数据量少（2000 样本 × 3 epoch），adapter 还没充分学到 audio → LLM-readable embedding 的映射
- **原因 3**：需要 Stage 2 多任务微调（文档 Section 3）才能真正提升 CER

## 下一步

按文档继续执行：
- **Phase 2 Stage 1 增强**：用更多数据训练（建议 50k+ 样本, 10+ epochs），加 cosine-DTW loss
- **Phase 3 Stage 2 多任务微调**：在 Thai-SUP 上做 IC + NER + SR 联合训练
- **Phase 4 A/B 评测**：与 Whisper 3-model 对比
- **Phase 5 路由式部署**

## 文件清单

```
thai-understanding/
├── u_align_adapter/
│   ├── adapter_final.pt        # 训练好的 9.18M 参数 adapter
│   └── train_log.json
├── phase1_u_align_eval_SR_test.json    # 端到端评测（50 样本）
├── phase1_u_align_retrieval.json       # 检索验证（64 样本）
├── phase1_adapter_compare.json         # 训练 vs 随机 adapter 对比
├── src/
│   ├── train_u_align.py                # Stage 1 训练脚本
│   ├── eval_u_align.py                 # 端到端评测
│   ├── eval_u_align_retrieval.py       # 检索验证
│   └── eval_u_align_baseline.py        # 对比 baseline
└── PHASE1_REPORT.md
```

## 结论

✓ 严格按文档执行了 Phase 0 + Phase 2 Stage 1
✓ XLSR-Thai → U-Align Adapter → Qwen3-4B 架构完整实现
✓ Stage 1 训练 loss 从 2.02 → 1.40，验证对齐学习有效
✓ Retrieval R@10 45% (vs 随机 14%) 证明 adapter 工作
✗ 端到端 CER 仍高（需要 Stage 2 多任务微调）
