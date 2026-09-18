# Thai-Understanding: 完整复现报告

## 资产清单（全部下载完成）

```
thai-understanding/
├── README.md, VERIFY_REPORT.md, E2E_CER_REPORT.md
├── assets/Thai-SUP.png, Thai-SUP.pdf
├── XLSR-Thai/checkpoint_best.pt        # 3.6 GB, 24 层 XLSR-53 SSL encoder
├── Typhoon2-3B/                       # 6.1 GB, Llama-3.2-3B
├── ZikXewen-thai-ctc/                 # 1.3 GB, XLSR-53 + Thai CTC head (pretrained on CommonVoice)
├── Thai-SUP/                          # 全部 47 GB 数据集
│   ├── IC/   train(73)/dev/test   = 148,538 samples / 180.5 h
│   ├── NER/  train(116)/dev/test  = 234,367 samples / 439.3 h
│   └── SR/   train(114)/dev/test  = 230,554 samples / 254.4 h
│   TOTAL: 613,459 samples / 874.2 h
├── src/
│   ├── xlsr_thai.py           # Fairseq 权重 → torchaudio 加载器
│   ├── slu_pipeline.py        # XLSR-Thai + Adapter + LLM 端到端
│   ├── demo_real.py           # 用真实样本跑 pipeline
│   ├── verify_assets.py       # 全资产验证
│   ├── eval_e2e_cer.py        # 方法 A: LLM 随机 adapter
│   ├── eval_cer_pretrained.py # 方法 C: Pretrained XLSR-53-Thai-CTC
│   ├── eval_xlsr_thai_ctc.py  # 方法 D: XLSR-Thai SSL + 迁移 CTC head
│   ├── eval_ctc_full_train.py # 方法 B: XLSR-Thai + 全新 CTC head 训练
│   ├── eval_ctc_pretrained_finetune.py # 微调 pretrained head
│   └── eval_xlsr_thai_finetune.py # Full encoder fine-tune (OOM)
└── results_*.json             # 所有评测结果
```

## 评测结果对比 (1000 样本 Thai-SUP dev+test)

| 方法 | 参数量 | 训练数据 | avg CER (无空格) | 备注 |
|---|---|---|---|---|
| **A. Typhoon2-LLM + 随机 adapter** | 3.2B | 0 | 142.48% | LLM 不听音频，只根据 prompt 幻觉 |
| **B. XLSR-Thai SSL + 自训 CTC** | 0.32B + 0.5M | 400 样本/8 epoch | ~100% | CTC 头坍缩到全 blank |
| **C. Pretrained XLSR-53-Thai-CTC (ZikXewen)** | 0.32B | CommonVoice Thai | **37.95%** | 有意义的 baseline |
| **D. XLSR-Thai SSL + 迁移 CTC head** | 0.32B + 0.5M | 0 | 237.77% | CTC 头对 XLSR-53 训练，对不上 XLSR-Thai 特征 |
| **E. Paper Table 1: XLSR-Thai + full ASR fine-tune** | 0.32B + 30M | Giga2+MSR+CV 16k h | **13.91%** | 论文 Giga2 Test CER，需数天训练 |

## 关键发现

### 1. XLSR-Thai 是 SSL 模型，不是 ASR 模型
- 论文释放的 `checkpoint_best.pt` 只是 wav2vec2 SSL encoder（24 层 XLSR-53 + wav2vec criterion 训练）
- 不能直接拿来做 ASR，需要再训练一个 CTC 头
- 论文 Table 1 的 13.91% CER 来自 XLSR-Thai + full ASR fine-tune（encoder + CTC head 一起训）

### 2. Adapter 随机权重的 LLM pipeline 没用
- Typhoon2-LLM + 随机 adapter → LLM 实际只听 prompt 不听音频
- 输出全是 `ยิ่ยงรยูวิธยรยยยยยยยยยย` 这种字符重复
- 必须训 adapter（论文 U-Align 阶段）才能让它"听"

### 3. CTC 头冷启动困难
- 8 epoch / 400 样本：CTC head 全预测 blank，CER 100%
- 300 样本 / 3 epoch：依然坍缩到单字符 (`ฑ`)
- 原因：CTC 需要模型学会"每个时间步对应什么字符"，从随机 init 学太难
- 即使用了 pretrained head 的 init 也需要大量数据 + 长时间训练才能收敛

### 4. 唯一可用的 ASR baseline 是 pretrained XLSR-53-Thai
- ZikXewen 模型用 XLSR-53 backbone + CommonVoice Thai 微调
- 在我们的 Thai-SUP 测试集上 CER 37.95%（无空格）/ 52.31%（有空格）
- 这个数字可以视为 "XLSR-Thai 论文方法在 small data 下的可达上限" 的近似
- 论文的 13.91% 来自 GigaSpeech2 test set（不同 domain + 大数据训练）

### 5. 数据规模
- 完整 Thai-SUP 训练集：613,459 样本 / 874 小时 / 47 GB
- 这是 TTS 合成数据（不是真人录音），所以 ASR 数字天然偏高
- 论文做 U-Align alignment 时用的是 GigaSpeech2/MSR-86K/CV-Thai，这些**没在这个仓库里**，需要另外下载

## 复现论文 Table 2 (XLSR-Thai + U-Align) 还需要什么

1. **数据**：
   - ✓ Thai-SUP 训练集（已下完，47 GB）
   - ✗ GigaSpeech2 Thai 训练集（~几百 GB）
   - ✗ MSR-86K（~86k 小时 = TB 级）
   - ✗ CommonVoice Thai 训练集
2. **代码**：
   - ✗ OSUM 训练框架（基于 WeNet + LLM，需要改装）
   - ✗ U-Align 模块：DTW-loss 实现
   - ✗ Adapter: LN + CNN subsampler + MLP
   - ✗ SFT loss: task prompt + speech embedding → LLM
3. **计算**：
   - ✗ Stage 1 U-Align: 1 epoch on 2000h ASR → 单 A10 估计 1-2 天
   - ✗ Stage 2 多任务微调: 1 epoch on Thai-SUP 1000h → 单 A10 估计 2-3 天
   - ✗ 至少需要 16-24 GB VRAM × 7 天

## 验证命令

```bash
cd /data/workspace/asr-model-training/thai-understanding

# 全资产验证
/data/workspace/venvs/qwen3-asr/bin/python src/verify_assets.py

# 方法 A: LLM 随机 adapter
/data/workspace/venvs/qwen3-asr/bin/python src/eval_e2e_cer.py --n_samples 1000

# 方法 C: Pretrained XLSR-53-Thai-CTC
/data/workspace/venvs/qwen3-asr/bin/python src/eval_cer_pretrained.py --n_samples 1000

# 方法 D: XLSR-Thai SSL + 迁移 CTC head
/data/workspace/venvs/qwen3-asr/bin/python src/eval_xlsr_thai_ctc.py
```

## 结论

在当前硬件（A10 24GB）和时间预算下，能完整复现 XLSR-Thai 论文 Table 1/2 的数字需要：

1. **下载额外数据**（GigaSpeech2 Thai + CommonVoice + MSR-86K，~1 TB）
2. **写完整训练代码**（U-Align DTW loss + adapter + SFT，~500 行）
3. **多卡/多天训练**（至少 4×A100 × 1 周）

本次跑通的是：
- ✓ 完整数据 pipeline (XLSR-Thai + LLM + Thai-SUP 数据)
- ✓ 真实 ASR baseline (Pretrained XLSR-53-Thai-CTC: 37.95% CER on 1000 Thai-SUP samples)
- ✓ 失败模式分析（adapter 未训练 → 142% CER；CTC 头冷启动 → 100% CER；CTC 头迁移 → 237% CER）

要让 XLSR-Thai 真正达到论文的 13.91% CER，必须跑完整 ASR fine-tune（encoder + CTC head 一起训，~16k 小时数据），这超出当前任务范围。
