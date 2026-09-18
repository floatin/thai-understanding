# PTT 评测报告：XLSR-Thai + U-Align Adapter + Qwen3-4B

**日期**: 2026-09-16
**评测设计**: `EVAL_DESIGN.md`
**模板**: `PTT_PROMPT_TEMPLATE.md`

---

## 1. 评测配置

| 项 | 值 |
|---|---|
| 评测集 | 273 GT + 273 Silver (strong_silver, 剔除 59 个 GT 重叠 wav) = **546 样本** |
| Audio 时长截断 | 10 秒 |
| Adapter A | 3 epoch, InfoNCE only, 2000 SR 样本 |
| Adapter B | 10 epoch, InfoNCE + cosine-DTW, 11800 样本 |
| LLM 生成 | Qwen3-4B (bf16) on GPU, greedy, max_new_tokens=64 |
| 主指标 | deepseek-v4.1-flash Yes/No LLM-judge (1092 calls) |
| 辅助指标 1 | bge-m3-Thai cosine similarity |
| 辅助指标 2/3 | chrF / CER norm（未跑）|

**生成 prompt**（业务场景 + 热词 + 纠偏规则，~150 tokens）

---

## 2. 主结果

| 子集 | n | Embed sim A | Embed sim B | **Judge SER A [主]** | **Judge SER B [主]** |
|---|---|---|---|---|---|
| GT-273 | 273 | 0.37 | 0.35 | **100.00%** | **100.00%** |
| Silver-273 | 273 | 0.36 | 0.34 | **99.63%** | **99.63%** |
| **PTT-546** | 546 | 0.37 | 0.35 | **99.82%** | **99.82%** |

**主要发现**：

- 两个 adapter 在 PTT 真实语音上 **SER ≈ 100%** ——基本都生成了与 ref 不等价的输出
- Adapter A 与 Adapter B **差异极小**（Judge SER 几乎相同）
- Adapter A 在 embedding sim 上略高于 B（0.37 vs 0.35），但都在低区间

---

## 3. 输出样本分析

| ref | hyp_A (3-ep) | hyp_B (10-ep InfoNCE+DTW) |
|---|---|---|
| ขออนุญาตเช็ดสัญญาณ | "[เสียง] ถูกต้องสะท้อนเป็น: ศักราชจ์ หัวหน้า..." | "หัวหน้า หัวหน้า หัวหน้า..." (重复) |
| คลื่นหนึ่ง ว สอง ว สิบหก | "สันสันสันสันสัน..." (重复) | "าาาาาาาาา..." (重复) |
| ประตูสองออกสิบหกห้าค่ะ | "[เสียง] ถอดเสียงเป็นข้อความภาษาไทย..." (拒答) | "คคคคคคคคค..." (重复) |

**模式**：
- Adapter A: 经常**幻觉**生成看似泰文但语义不对的句子
- Adapter B: 几乎总是**重复**同一 token（更明显的失败模式）
- 两个 adapter 都没真正转录 PTT 音频

---

## 4. 根因分析

### 4.1 Stage 1 训练的局限

两个 adapter 都用 **InfoNCE + 可选 DTW** 在 **Thai-SUP SR/IC/NER** 任务上训练：

| 训练数据 | PTT 评测数据 |
|---|---|
| Thai-SUP (TTS 合成语音) | PTT (真实用户录音) |
| 改写任务 (SR) | 转录任务 (ASR) |
| ~17小时 | 12000 句 |
| 无背景噪声 | 真实环境噪声 |
| 标准发音 | 各种口音 / 速度 |

Stage 1 只让 adapter 学会"speech embedding 靠近 text embedding 空间"。但**没学会**：
1. 处理 PTT 真实声学环境
2. 听懂保安对讲机专业术语
3. 让 Qwen3-4B 用 PTT 上下文生成有意义的回复

### 4.2 Adapter B (10-ep) 没显著优于 A (3-ep)

虽然 Adapter B 在 Thai-SUP SR retrieval 上 R@10 提升 22pp（25% → 47%），但**这个提升没迁移到 PTT 端到端任务**：
- B 的输出更差（重复模式）vs A 的幻觉模式
- Embed sim 反而 A 略高（0.37 vs 0.35）

可能原因：
- InfoNCE + DTW 让 B 过拟合到 Thai-SUP 的 speech↔text 对应，但没学到**生成**
- 端到端生成需要 LLM 学会"读" speech embedding，这需要 Stage 2 fine-tune

---

## 5. 评测方法学验证

### 5.1 评测 pipeline 工作正常

- ✓ LLM-judge API 4 个 sanity case 全对（เช็ด/เช็ค, หนึ่ง/สอง, ครับ/ค่ะ, ครับ/无）
- ✓ bge-m3-Thai sanity check 通过（identical=1.0, unrelated=0.26）
- ✓ 全 GPU 推理 ~2.3s/样本，**稳定**
- ✓ 数据保存完整（546×2 = 1092 评估结果）

### 5.2 已知边界 case

**1 条 Silver 样本被判 Yes**（边界 case）：

- ref: `'ครับ'`（单 token）
- hyp: `'ค่ะ ค่ะ ค่ะ ...'`（重复 22 次）
- 按 §5.2 "礼貌词尾互换 = 等价" → Yes
- 但严格看：ref 是 1 token，hyp 是 22 tokens，明显不同

**建议规则补充**（如果认为这是 bug）：
- 选项 1：ref 与 hyp 的有效 token 数应大致相当
- 选项 2：单 token 情况下才生效，多 token ref 不适用
- 选项 3：保持当前规则，接受这是规则边界

当前未改，按你确认的 §5.2 规则执行。

---

## 6. 结论

### 主要结论

1. **评测 pipeline 可用**：4 类 sanity check 全过，端到端能跑 1092 个样本
2. **Stage 1 训练不足以让 PTT 工作**：两个 adapter 在 PTT 端到端 SER ≈ 100%
3. **Adapter B 优势没显现**：在 PTT 上 A 略优于 B，与 Thai-SUP 上的 R@10 趋势相反

### 业务价值

❌ **当前 adapter 不能直接用于 PTT 业务**（SER 100%）
✅ **但评测 pipeline 已就绪**，可在 Stage 2 fine-tune 后直接复用

### 下一步建议

| 优先级 | 行动 |
|---|---|
| 高 | **Stage 2 fine-tune**：在 PTT 音频 + GT/Silver 文本上 end-to-end 训练，让 Qwen3-4B 学会消费 adapter 输出 |
| 中 | 跑 chrF / CER norm 补充辅助指标（已留接口）|
| 中 | 边界 case 规则调整（单 token 礼貌词） |
| 低 | 评测更细分的子集（按时长 / 群组 / 子任务） |

---

## 7. 文件清单

```
thai-understanding/
├── ptt_eval_gen.json        # Phase 1: 546 样本 × 2 adapter 的 hyp
├── ptt_eval_metrics.json    # Phase 2+3: 所有指标（embed sim, judge SER）
├── src/
│   ├── eval_ptt_gen.py      # Phase 1 generation 脚本
│   └── eval_ptt_metrics.py  # Phase 2+3 metrics 脚本
└── PTT_EVAL_REPORT.md       # 本报告

设计文档:
├── EVAL_DESIGN.md           # 评测设计
├── PTT_PROMPT_TEMPLATE.md   # prompt 模板
└── AGENTS.md §5             # 评测铁律
```

---

## 8. 后续

如果进入 Stage 2 fine-tune：
1. 训练数据：PTT 273 GT + 273 Silver = 546 高质量样本（外加更多 silver 池）
2. 训练目标：让 Qwen3-4B 学"看到 adapter 输出后生成对应 Thai 文本"
3. 训练方式：end-to-end，**冻结 XLSR-Thai + Qwen3-4B**，只训练 adapter + LoRA on Qwen3
4. 评测：复用 `eval_ptt_metrics.py`，相同指标
