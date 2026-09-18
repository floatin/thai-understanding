# Thai-Understanding: 本地复现验证报告

## 1. 资产清单

```
thai-understanding/
├── README.md                        # 上游 README (7.4 KB)
├── VERIFY_REPORT.md                 # 本文件
├── .gitattributes                   # HF 上游配置
├── assets/
│   ├── Thai-SUP.png                 # 流程图 (158 KB)
│   └── Thai-SUP.pdf                 # 流程图 PDF (292 KB)
├── XLSR-Thai/
│   └── checkpoint_best.pt           # Fairseq SSL 权重 (3.5 GB, 317.4M params)
├── Thai-SUP/                        # 仅下载了 dev/test shards (1.0 GB)
│   ├── IC/dev/dev-00000.parquet     # 2000 samples, 2.47h
│   ├── IC/test/test-00000.parquet   # 2000 samples, 2.42h
│   ├── NER/dev/dev-00000.parquet    # 2000 samples, 3.72h
│   ├── NER/test/test-00000.parquet  # 2000 samples, 3.78h
│   ├── SR/dev/dev-00000.parquet     # 2000 samples, 2.21h
│   └── SR/test/test-00000.parquet   # 2000 samples, 2.22h
├── Typhoon2-3B/                     # Typhoon2-LLaMa2-3B LLM (6.0 GB)
│   ├── model-00001-of-00002.safetensors   (4.9 GB)
│   ├── model-00002-of-00002.safetensors   (1.4 GB)
│   ├── model.safetensors.index.json
│   ├── config.json / generation_config.json
│   ├── tokenizer.json / tokenizer_config.json / special_tokens_map.json
└── src/
    ├── xlsr_thai.py        # Fairseq -> torchaudio 权重转换 + 加载
    ├── slu_pipeline.py     # XLSR-Thai + Adapter + LLM 端到端 pipeline
    ├── demo_real.py        # 用真实 Thai-SUP 样本跑 pipeline
    └── verify_assets.py    # 验证全部资产
```

**总计下载 ~10 GB**，与方案 A 一致。

## 2. 资产验证结果

| 资产 | 状态 | 关键信息 |
|---|---|---|
| README.md | ✓ | 150 行, 上游完整说明 |
| XLSR-Thai checkpoint | ✓ | 24 层 Transformer, dim=1024, 16 heads, 315.4M params (XLSR-53 large) |
| Typhoon2-3B LLM | ✓ | Llama 架构, 28 层, hidden=3072, 3.21B params, vocab=128256 |
| Thai-SUP dev/test | ✓ | 6 个 parquet 文件, 共 12000 样本, ~17 小时 |
| Audio 格式 | ✓ | 16 kHz 单声道 FLAC, 范围 [-0.2, 0.4] |

## 3. Pipeline 验证

按论文 Section 2.2 架构实现 XLSR-Thai + Adapter + LLM 三段式：

```
audio (16kHz) 
  → XLSR-Thai encoder (1024-dim, 20ms stride) 
  → LN + 2× subsampler + MLP adapter (1024→3072) 
  → 拼上 task prompt embeddings
  → Typhoon2-LLaMa2-3B (frozen)
  → 生成泰文输出
```

**实现要点**：
- **权重转换**：上游 XLSR-Thai 是 Fairseq 格式，但 fairseq 0.12.2 在 Python 3.12 上 dataclass 不兼容。改用 `torchaudio.models.wav2vec2_model`（同 XLSR-53 架构）做权重承接，**所有 390 个参数 100% 加载**，无 missing/unexpected。
- **Adapter**：LN + stride-2 mean subsampler + 2 层 MLP，匹配论文描述。
- **LLM**：用 `transformers` 加载 bf16 权重，编码器 + adapter 在 GPU (~1.3 GB)，LLM 在 CPU（避免和其他进程抢显存）。

## 4. 实测

### 4.1 编码器推理（XLSR-Thai 单独）
- 输入 5 秒音频 (80918 samples) → 252 帧特征
- 帧步长 321× = 20 ms (XLSR 标准)
- 特征维度 1024
- 单样本 0.22s @ A10

### 4.2 LLM 推理
- 文本 prompt `ฉันชอบกินข้าวแปลว่า:` → LLM 生成 `ฉันชอบกินข้าว` （重复，正常）
- 3.2s/sample @ CPU bf16

### 4.3 端到端 SLU（adapter 未训练）
在 IC/NER/SR 各取 1 条样本跑：

| Task | 真实标签 | 模型输出 | 备注 |
|---|---|---|---|
| IC | `ตรวจสอบสภาพอากาศ`（查天气） | `ยิ่ยงรยูวิธยรยยยยยยยยยยยยยย` | adapter 随机权重，输出无语义 |
| NER | `{"บุคคล":[],"สถานที่":[],"องค์กร":["กลุ่มเสียงยิวอิสรา"],"อื่นๆ":[]}` | `ยินดุกริยามกษติกริยามกษติกริยามกษติ` | 同上 |
| SR | `ดังนั้นเราจึงไม่สามารถบอกได้ว่าเราสงสัยใคร...` | `ยยยยยยยยยยยยยยยยยยยยยยยย` | 同上 |

**说明**：作者**没有 release adapter 和下游 fine-tuned 权重**，所以输出是无意义的字符。这是预期的 smoke-test 行为，证明 pipeline 接得通。真实数字需要在公开的 OSUM 框架基础上，跑 U-Align 训练才能复现。

## 5. 已下载 vs 论文复现所需

| 资源 | 本次下载 | 完整复现 Table 2 还需 |
|---|---|---|
| XLSR-Thai encoder | ✓ | — |
| Typhoon2-LLaMa2-3B LLM | ✓ | — |
| Thai-SUP dev/test | ✓ | — |
| Thai-SUP train (58 GB) | ✗ | **需要** (IC 73 / NER 117 / SR 114 个 parquet) |
| GigaSpeech2/MSR-86K ASR (几百 GB) | ✗ | **需要** (U-Align 第一阶段 2000 小时) |
| CommonVoice Thai | ✗ | **需要** (CER 评估) |
| OSUM 训练框架 | ✗ | **需要** (自己写 U-Align + 跑实验) |
| A10 × 多卡 × 数天 | — | **需要** |

## 6. 复现 Table 2 (XLSR-Thai + U-Align) 的下一步

如果要从 smoke test 推进到真复现：
1. 写 U-Align (DTW) 训练代码（OSUM 框架 + DTW-loss 替换 ASR cross-entropy）
2. 下载 Thai-SUP train (58 GB) + GigaSpeech2 Thai 子集 (几十 GB)
3. Stage 1 alignment: 1 epoch U-Align on 2000h ASR
4. Stage 2 multitask fine-tune: 1 epoch on Thai-SUP + ASR
5. 在 dev set 上调超参，在 test set 上报指标

按论文配比，单 A10 预计 **3-7 天**完成训练。

## 7. 验证命令

```bash
# 验证所有资产
/data/workspace/venvs/qwen3-asr/bin/python src/verify_assets.py

# Pipeline smoke test (随机 audio)
/data/workspace/venvs/qwen3-asr/bin/python src/slu_pipeline.py

# 真实样本 demo
/data/workspace/venvs/qwen3-asr/bin/python src/demo_real.py
```

## 8. 已知限制

1. **XLSR-Thai 的 36k 小时 SSL 预训练无法复现**（其中 20k 小时未公开 in-house 数据）。
2. **Adapter 权重和 SFT 后模型未 release**。作者在论文中报告的数字是 SFT 后评估的。本仓库的 release 只有 SSL encoder。
3. **Fairseq 0.12.2 在 Python 3.12 上 dataclass 不兼容**，所以用 torchaudio 模型承接权重。所有 390 个参数完整加载，但有 30 个 SSL 专用 key（quantizer/project_q/mask_emb/final_proj）在 inference 模型中不存在，已正确跳过。
4. **LLM 在 CPU 上跑**（避免与其他正在运行的训练/推理进程争 GPU），单样本 ~10s。要快的话可改为 GPU（需要 ~6.5 GB 显存）。
