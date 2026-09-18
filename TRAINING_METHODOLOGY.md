# PTT 泰语语音语义转录 — 训练方法学完整文档

> **版本**: 2026-09-18（v6 训练进行时定稿）
> **范围**: 从 Stage 1 对齐到 Stage 2 端到端的全部训练方法、数据、配方、评估体系与踩坑记录。
> **一句话**: `音频 → XLSR-Thai(冻结) → U-Align Adapter → [CTC 纯训前端] → 冻结 → Typhoon2.5+LoRA → 转写文本`，
> 评估以 F2LLM 语义相似度为主指标、CTC-CER 为前端内容诊断指标、输出多样性为坍缩护栏。

---

## 1. 项目目标与总体架构

### 1.1 业务目标

对讲机（PTT）真实泰语录音的语义转录，替代现有 Whisper 多模型投票方案。核心诉求：
热词无限扩量不重训 ASR、业务上下文原生注入、语义级评估（Entity/意图）优先。

### 1.2 架构总览

```
                       ┌──────────────── 训练期可训部分 ───────────────┐
音频 wav (16kHz, 截断10s)                                                          │
   │                                                                               │
   ▼                                                                               │
XLSR-Thai SSL encoder (24层, 1024维/帧, ~50Hz)  ❄ 全程冻结                          │
   │                                                                               │
   ▼                                                                               │
U-Align Adapter: LayerNorm → CNN下采样(k=2,s=2,depthwise,均值初始化) → MLP(1024→2560) │
   │   Stage1: 可训(InfoNCE对齐)  Stage2a': 可训(纯CTC)  Stage2b: 冻结              │
   ▼                                                                               │
speech embeds (T'帧 ≈25Hz, 2560维) ──┐                                             │
                                      ├─ REPLACE: 语音嵌入替换输入序列前 T' 位        │
GEN_PROMPT (~150 tok 业务场景+热词+纠偏规则) ──► text embeds ──┘                    │
                                      │                                             ▼
                                      │              Typhoon2.5-Qwen3-4B ❄ + LoRA(q/k/v/o, r=16, α=32) 🔥
                                      │                           │
                                      ▼                           ▼
                              [speech|prompt|answer]  ──►  answer token CE（含EOS）
                                      (labels: -100×(T'+P), answer_ids+A)
```

**设计要点**：语音嵌入**替换**（REPLACE）而非拼接在 prompt 之后——论文 U-Align 的原始设计；
CE 只对 answer 计算；LLM 通过注意力层读取语音嵌入，无 cross-attention。

### 1.3 组件规格

| 组件 | 规格 | 状态 |
|---|---|---|
| XLSR-Thai | `XLSR-Thai/checkpoint_best.pt` 3.6GB，fairseq→HF 加载（`src/xlsr_thai.py`） | 冻结 |
| U-Align Adapter | encoder_dim=1024 → llm_dim=2560，downsample=2；参数约 40M | 分阶段 |
| 下采样公式 | `Conv1d(k=2, s=2, no pad)` → 输出帧数 = `(输入帧数-2)//2 + 1`（CTC input_lengths 必须用此式） | — |
| 生成 LLM | `typhoon2.5-qwen3-4b`（Qwen3ForCausalLM，36层，hidden 2560，bf16；tokenizer 词表与 Qwen3-4B 完全一致 151669） | 冻结+LoRA |
| CTC 头 | `Linear(2560 → 301)`，fp32（泰语合并单元词表，见 §6） | Stage2a' 可训 |
| 评估嵌入器 | `codefuse-ai/F2LLM-v2-4B`（多语言，Thai ✓，last-token pooling，对称任务不加指令前缀，bf16 约 8.6GB） | 仅评估 |

---

## 2. 数据资产与划分

### 2.1 数据池

| 数据 | 规模 | 来源/性质 | 用途 |
|---|---|---|---|
| Thai-SUP | 613,459 条 / 874h（IC 148k / NER 234k / SR 230k，parquet+flac） | LLM 增强→翻译→TTS 合成 | Stage 1 对齐；候选域内训练池 |
| PTT strong-silver | 6,497 条（triple_confirmed 5,495） | 3 个 ASR 教师转录一致 | Stage 2 主训练池 |
| PTT 人工 GT | 273 条（Baserow 表 1293） | 人工听写 | 最终评测真值 |
| h173 | 173 条 | GT 的人工子集（`e7_holdout173.json`） | 训练期验证（永不入训练） |
| Silver_top20 | 1,284 条 | strong-silver 按教师置信 top 20% | 验证池（扩充统计功效） |

### 2.2 划分与防泄漏（硬约束）

- **训练池 5,159 条** = strong-silver − eval 全部 wav（构建时 `minus_eval_wavs: True`）。
- 三方互斥已逐 wav basename 验证 = 0 重叠，并在所有训练脚本数据加载处 **assert 固化**（AGENTS.md §4.6 #1）。
- h173 与 Silver_top20 是验证集的两个人为标签子集；训练期只评 h173，结束时全量评 1,284 条 silver 做子集一致性对照。
- 强银转写含教师错误 → 对齐/CTC 监督继承噪声，由对齐置信分（score ≥ 0.4）过滤兜底。

### 2.3 音频预处理

统一 16kHz；**截断 10 秒**（`MAX_AUDIO_SECONDS=10`，与评估一致）；短于 0.2s 补零到 3200 样本（卷积最小长度）。

---

## 3. Stage 1：语义对齐（U-Align）

**目标**：让 adapter 输出的语音嵌入落入文本嵌入空间，建立"语音↔文本"检索能力。
**局限（事后认识）**：句级对齐只教"这句话和哪句话近"，**不提供帧级内容信号**——这是 Stage 2 长期失败的根源之一。

| 项 | v1（3-ep） | v2（最终采用） |
|---|---|---|
| 数据 | Thai-SUP SR 2,000 | Thai-SUP dev+test 11,800（IC/NER/SR 各~4k，200 holdout） |
| 损失 | InfoNCE | **1.0×InfoNCE + 0.5×cosine-DTW**（帧级 DTW 40帧 vs 24 token） |
| 训练 | 3 epochs | 13 epochs（R@10 于 ep10 饱和 47%） |
| 文本塔 | Qwen3-Embedding（冻结） | 同左 |
| 结果 | R@10=25% | **R@10=46.9%**（512 样本独立测试 vs 随机 2.5%） |

**教训**：DTW loss 几乎不收敛（0.516→0.508），小数据下帧级 DTW 梯度信号太弱；检索 R@10 的提升**没有**迁移到端到端转录。

---

## 4. Stage 2 演化史（每一步的配方与判决）

> 完整结果见 `STAGE2_V3_TYPHOON_REPORT.md`。此处按版本给出配方差异与结论。

| 版本 | 相对上一版唯一变量 | 关键结果 | 判决 |
|---|---|---|---|
| v1 concat | 初始设计 `[prompt\|speech\|answer]` | PTT 上 SER≈100%（全失败） | 序列结构错误 |
| v2 REPLACE | 序列改为 `[speech\|prompt\|answer]`，label 掩码修正 | step200 SER 0.738（bge 口径），中断 | 结构对但不够 |
| v3 | LLM: Qwen3-4B → **typhoon2.5-qwen3-4b**（F2LLM 口径） | h173 SER 0.8801→0.8418→0.8009；输出 19 种（坍缩） | LLM 切换本身不解决内容缺失 |
| v4 | **target 末尾加 EOS**（从 v3 续训） | 停机学会（173/173 停止）但输出只剩 **4 种** | EOS 必要但不充分 |
| v5 | **+CTC 辅助损失**（λ=0.3，adapter 帧上） | CTC 头全 blank 坍缩；LLM SER 0.84 不动 | **联合目标互相拖垮** |
| **CTC-only** | **去掉 LLM 与 CE，纯 CTC 训 adapter+头** | h173 CTC-CER 0.61→**0.4999**（5ep，2分钟/ep） | ✅ 前端内容信号打通 |
| v6 | adapter+CTC 头**冻结**，只训 LLM LoRA | （进行中） | 待判决 |

### 4.1 各版本共有部分（v3 起不变）

- **数据**：5,159 silver；**超参**：batch 2 × grad_accum 8（=16 样本/优化步）、AdamW lr 1e-4、cosine 退火 + 5% warmup、grad clip 1.0、LoRA r=16 α=32 dropout=0.05 挂 q/k/v/o、bf16 + gradient checkpointing、可训参数 ~21M。
- **GEN_PROMPT**（约 150 token，与评估共用，业务场景+热词表+纠偏规则；`【กฎการแก้ไข】`含 เช็ด→เช็ค、สิบหกห้า→16:05、重复字符压缩、礼貌词尾保留）。
- **REPLACE 序列**：`[speech T' | prompt P | answer A]`，labels 前 T'+P 位 = -100。
- **EOS（v4 起）**：`ans_ids = tokens + [eos]`，截断 64 时**保证 EOS 在内**；eos = `<|im_end|>`(151645)。
- **终点评估**：训练结束全量评 silver1284（h173 每次评估点都评）。

### 4.2 v5 失败的机制（重要教训）

联合损失 `L = CE_answer + 0.3×CTC(adapter帧)` 下：CE 路径发现"输出模态短语"即可降 loss，
其梯度经共享 adapter 把帧级内容表示冲掉；CTC 头滑向全 blank 平凡解（blank_frac=1.00）。
训练日志的 ctc "下降"（156→34）实为 **zero_infinity 把越来越多不可对齐样本的损失归零**的假象——
监控累计值时必须同时记录 zeroed 计数。

**修正方向（已被 CTC-only 验证）**：先无竞争地训好前端（纯 CTC），再冻结前端训读取器。

---

## 5. Stage 2a'（CTC-only 纯前端训练）— 当前有效配方

**为什么存在**：v5 证明联合目标互相拖垮；本阶段完全去掉 LLM 与 CE，只回答一个问题——
"adapter 的帧级输出能否承载泰语内容？"

### 5.1 配方

| 项 | 值 |
|---|---|
| 可训 | adapter（init = Stage 1 Adapter B）+ CTC 头（随机初始化，fp32） |
| 损失 | **纯 CTC**（`F.ctc_loss(blank=0, zero_infinity=True)`），无其他项 |
| 输入 | XLSR-Thai 原始帧（~50Hz，bf16）→ adapter → 25Hz logits |
| `input_lengths` | **必须用下采样公式** `(T-2)//2+1`（用原始帧数会直接报错或错配） |
| 目标 | 合并单元 id 序列（§6），来自 silver 转写全文（不做 64 截断） |
| 优化 | AdamW lr 1e-4，wd 0.01，clip 1.0，batch 8，**5 epochs（~2 分钟/epoch）** |
| 显存 | ~2GB（无 LLM 在环） |
| 监控 | h173 CTC-CER(ns) + blank_frac（每 epoch）；`zeroed_batches` 计数 |

### 5.2 结果（h173 = 真实对讲机音频，未参与训练）

| epoch | CTC-CER(ns) | blank_frac |
|---|---|---|
| 0（未训） | ~4.6（随机头） | 1.00 |
| 0（1 epoch 后） | 0.6079 | 0.78 |
| 3 | 0.5242 | 0.75 |
| **4（best）** | **0.4999** | 0.71 |

参照系：ZikXewen XLSR-53-Thai CTC（CommonVoice 训练）在同域音频上 = 0.654。
**adapter 前端已超过公开基线 15 个 CER 点**——架构可行性就此证实。

---

## 6. 泰语 CTC 单元设计（`src/thai_ctc_units.py`）

泰语是声调语言且为**元音附标文字（abugida）**：声调符号与上下元音符号是组合字符（Unicode Mn 类），
**没有独立声学时长**——逐字符 CTC 目标会让模型被要求"在音节中间某瞬间发 ้"，是噪声不是信号。

**规则（按 Unicode 类别判定，禁止手写字符表）**：
1. NFC 归一化后，**Mn 组合符号并入其前一个基字符** → 单元 = 基字符 + 尾随组合符号。
   例：`เขาเก็บได้ที่ไหน`（16 字符）→ `เ ข า เ ก็ บ ไ ด้ ที่ ไ ห น`（12 单元）。
2. 空格保留为独立单元（词边界）。合法泰文部件 ๆ(Lm)、ฺ/ํ(Mn) 按类别天然保留，不会被误删。
3. 词表：全部训练转写合并单元计数，**freq ≥ 5 入词表**（最终 301 单元含 blank+space）；
   稀有单元从目标中**直接删除**（不映射 `<unk>`——避免教模型输出 unk）。
   实测仅 0.16% 的单元实例被丢弃。
4. 自测用例硬编码在 `thai_ctc_units.py`（含修正过的 `สองแปด ครับ` 切分断言）。

**声调的角色**：转写目标含声调符号，但声调符号的正确性主要由拼写规则决定（F0 只是旁证），
因此分段级监督即可支撑——无需为 5 个调单独设计调类单元。上/下元音符号的声学载体是真实音段，
并入基字符后单元边界仍与音节结构对齐。

---

## 7. 训练运行与资源

### 7.1 显存实测（A10 23GB）

| 阶段 | 显存 |
|---|---|
| Stage 2 训练态（XLSR+adapter+LLM+LoRA+优化器+激活） | **13.1 GB** |
| 训练态 + 对齐服务常驻（4.3GB，外部） | 17.4 GB ✓ |
| 训练态 + F2LLM 评估共驻（旧 v3/v4 流程） | 19.4–21.7 GB（有外部进程时 OOM） |
| CTC-only 训练 | ~2 GB |
| F2LLM 单独打分 | ~9 GB |

**当前分工**：GPU 与对齐服务（8082，4.3GB）共存 → 训练期评估**不含 F2LLM**（见 §8.3），
F2LLM 在训练结束后对已存逐条 hyp 重打分（`rescore_history_f2llm.py`，无需重新生成）。

### 7.2 运行时

- Stage 2：~0.14–0.4s/micro-batch；1 epoch（322 opt steps）≈ 12–20 分钟训练 + 评估。
- 评估生成：未训模型顶满 64 token（~4.2s/条）；EOS 学会后中位 6 字符（秒级/条）。
- CTC-only：2 分钟/epoch（5159 样本，batch 8）。
- 对齐服务吞吐：~8 条/秒（6 并发 HTTP），5159 条 ≈ 11 分钟。

---

## 8. 评估方法学

### 8.1 指标体系（按层级）

| 层级 | 指标 | 定义 | 用途 |
|---|---|---|---|
| 主指标 | **F2LLM SER** | `1 − mean(cosine(ref, hyp))`，F2LLM-v2-4B，对称用法无指令前缀 | 语义相似度主结论 |
| 前端诊断 | **CTC-CER(ns)** | 冻结 CTC 头贪心解码 vs ref，去空格 CER | LLM 无关的内容信号监控 |
| 护栏 | **输出多样性** | `n_unique`、`top1_share`（每条评估日志强制记录） | 模式坍缩报警（top1>50% 即红旗） |
| 辅助 | CER strict/norm | 标准分母 `len(ref)`、**微平均**（Σedit/Σref）、按 Unicode 类别删 P*/S*/C*/Z*、max 单条 CER | 字面错误率 |
| 辅助 | chrF | sacrebleu | 字面重合 |
| 抽查 | LLM judge | deepseek Yes/No（PTT_PROMPT_TEMPLATE §5.4），150 对抽样 | F2LLM 校准/最终裁决（待 API env） |
| 监控 | 数字格式失配 | hyp 含阿拉伯数字 xor ref 含 | 跨格式风险（F2LLM 对 28 vs สองแปด 只给 0.78） |

### 8.2 F2LLM 验证门（任何数字用于结论前必须过）

1. 方向性：identical 1.00 / paraphrase 0.61 / unrelated 0.07 → PASS（动态范围远超 bge-m3）。
2. 扰动排序：数字词替换敏感（changed-only 0.66）、同音热词 0.74、礼貌词尾 0.99（良性）、
   **整词重复 glitch 0.91（可骗过 SER 阈值 → CER/judge 必须保留）**。
3. 历史重算一致性：旧 adapter 全失败输出 F2LLM SER=0.89，与 judge 判定同向 → PASS。
4. 已知盲区：跨格式数字（`สิบหกห้า` vs `165` = 0.64）→ 数字格式归一化候选。

### 8.3 训练期评估协议（v6 起）

- 每个 `eval_every`（100 opt step）评 h173：LLM 贪心生成（64 token 上限）→ CTC-CER + 多样性；
  **F2LLM 延后**（GPU 共存限制），逐条 hyp 已存档，训练后统一重打分。
- checkpoint 选择指标 = CTC-CER（v6 中前端冻结故恒定，实际选择训练后由 F2LLM 对各评估点定）。
- **评估粒度教训**：`global_step` 以 grad_accum(8) 为步进，`step % 100` 永不触发 → 必须用游标式
  `next_eval` 计数器；v3 曾因 200 粒度错过末端改善。

---

## 9. 已证实的失败模式与教训（本项目实证，非理论）

| # | 教训 | 证据 |
|---|---|---|
| 1 | **聚合指标对模式坍缩全盲**：SER/CER/停机率全部"改善"，输出却只有 4 种 | v4 终点 142×`สองแปด`；多样性护栏因此成为强制日志项 |
| 2 | **正则盲区**：1-4 字符循环正则检不出 `สองแปดครับ×N` 短语重复，导致两次误读（"暂态坍缩"、"EOS 决定论"） | v3@328 唯一输出 19 种被误判为"已恢复" |
| 3 | **EOS 必须显式进 target**：否则模型永不停止（三代共有 bug） | v3 前所有输出顶满 64 token |
| 4 | **联合 CE+CTC 互相拖垮**：λ=0.3 下 CE 把 adapter 推离内容表示，CTC 塌向全 blank | v5 blank_frac=1.00；CTC-only 纯训即刻 0.50 |
| 5 | **zero_infinity 的假下降**：不可对齐样本 loss 被归零，累计监控值下降≠学习 | v5 ctc 156→34 与全 blank 头并存 |
| 6 | **句级对齐 ≠ 帧级内容**：Stage 1 检索 R@10=47% 完全不迁移到转录 | v1-v4 全系列失败 |
| 7 | **CTC input_lengths 必须用下采样公式** `(T-2)//2+1` | 237 vs 118 报错 |
| 8 | **DataLoader fork + CUDA 崩溃**：collate 内有 GPU 前向时必须 num_workers=0 | 两次启动失败 |
| 9 | **str.replace 静默失败**：批量补丁必须 assert 替换生效（计数或断言），否则新旧两版函数共存、后者覆盖前者 | v5 冒烟 ValueError |
| 10 | **pkill -f 自杀**：模式匹配到自己 shell 的 cmdline，先杀自己再杀目标 | v5 首次启动无输出死亡 |
| 11 | **训练脚本输出路径必须随版本隔离**：resume 文件会让新 run 从半程续跑 | v5 首launch 差点污染 v4 |
| 12 | **评估粒度被整除步进吞掉**：8 的倍数步进下 `%100` 永不触发 | v4 只评了 step200 |
| 13 | **GPU 是共享资源**：外部进程（对齐服务 4.3GB + 他会话作业 4.6GB）决定评估架构（F2LLM 延后） | OOM 实录 |
| 14 | **中途快照会骗人**：v3 step200 时 96% 循环率在 step328 自然回落——但这本身又是快照误读（见 #2） | — |

---

## 10. 复现命令与产物清单

### 10.1 复现序列

```bash
PY=/data/workspace/venvs/qwen3-asr/bin/python
cd /data/workspace/asr-model-training/thai-understanding

# 0. 对齐数据（一次性；需 8082 服务与 /root/workspace/llama.cpp/.api_key）
$PY src/build_silver_align.py                       # → out_align/silver_align.jsonl (4684 ok/5159)

# 1. CTC-only 前端（~12 分钟）
$PY src/train_ctc_only.py --epochs 5                # → u_align_ctc_only/best/ctc_only.pt

# 2. Stage 2 v6：冻结前端 + LLM LoRA
$PY src/train_u_align_stage2_v6.py --epochs 2       # → u_align_stage2_v6/

# 3. 训练后 F2LLM 终评（对已存 hyp 重打分）
$PY src/rescore_history_f2llm.py --file u_align_stage2_v6/eval_history.jsonl

# 4. 字面指标（CER/chrF/数字格式监控）
$PY src/eval_cer_chrf.py --file u_align_stage2_v6/eval_history.jsonl --subset h173

# 5. 指标验证门（换嵌入模型时必跑）
$PY src/eval_f2llm_gate.py --mode sanity / perturb / rescore
```

### 10.2 文件清单

```
src/
├── xlsr_thai.py                     # XLSR-Thai fairseq→HF 加载器
├── train_u_align_v2.py              # UAlignAdapter/CNNSubsampler 定义（被所有版本 import）
├── train_u_align_stage2_v3/v4/v5.py # Stage 2 演化（v3=换LLM, v4=+EOS, v5=+CTC联合[失败]）
├── train_u_align_stage2_v6.py       # 当前：冻结前端 + LLM LoRA
├── train_ctc_only.py                # ★ Stage 2a' 纯 CTC 前端（当前有效配方）
├── thai_ctc_units.py                # 泰语合并单元（Mn 并入基字符）+ 自测
├── build_silver_align.py            # 强制对齐批处理（8082 服务）
├── eval_f2llm_gate.py               # 指标验证门（sanity/扰动/历史重算）
├── eval_step0_baseline.py           # step-0 锚点
├── eval_cer_chrf.py                 # CER(标准分母+微平均)/chrF + 数字格式监控
├── eval_judge_spotcheck.py          # judge 抽查（待 DEEPSEEK env）
├── eval_decode_ab.py                # 解码策略 A/B（greedy vs rep_penalty）
├── eval_final_checkpoint.py/_v4.py  # 终点 checkpoint 评估
├── rescore_history_f2llm.py         # 从已存 hyp 重打 F2LLM 分
└── probe_ctc_h173.py                # ZikXewen CTC 在 h173 的域差探针

u_align_ctc_only/best/ctc_only.pt    # ★ 前端权重（adapter+CTC头, ep4, CER 0.4999）
u_align_stage2_v*/eval_history.jsonl # 每次评估逐条 refs+hyps+sims+多样性+CTC-CER
out_align/silver_align.jsonl         # 5159 条字符级对齐（4684 ok）
out_metric_gate/                     # 验证门产物（扰动/重算/探针）
STAGE2_V3_TYPHOON_REPORT.md          # v3/v4/v5 完整结果与根因链
```

---

## 11. 当前状态与未决问题

### 11.0 v6 训练实录（实时更新，2026-09-18）

**配方**：adapter + CTC 头从 CTC-only best 冻结加载（其 h173 CER=0.4999），仅 LLM LoRA 可训，
损失 = answer CE（含 EOS）。2 epochs = 644 opt steps，eval_every=100（游标式）。

**评估点跟踪**（h173；CTC-CER 恒定 = 前端冻结的结构性验证）：

| step | CTC-CER(ns) | 多样性 n_unique | top1_share | 备注 |
|---|---|---|---|---|
| 104 | 0.4999 | 3 | 80% | LLM 尚在坍缩区 |
| 200 | 0.4999 | 22 | 64% | 多样性开始打开 |
| 304 | 0.4999 | 28 | 34% | 最好点；但内容仍为旧吸引子 |
| 400 | 0.4999 | 17 | 31% | top1 最低 |
| 504 | 0.4999 | 28 | 68% | 回摆（波动） |
| 600 | 0.4999 | **109** | 34% | "多样性突破"——后被判定为假象 |
| 648(silver终) | 0.357* | 347 | 37% | *silver 域 CER 本就低于 h173 |

**F2LLM 重打分（训练后统一补算）**：h173 SER 0.8953 → 0.8774 → 0.8430 → 0.8407 → 0.8594（step 600 回摆）；
silver1284 终态 0.8602。**全程未脱离 0.84-0.90 坍缩区。**

**v6 终局判决：未解锁转录。** 两条关键更正认识：
1. **多样性护栏再次被骗**：step 600 的 109 种唯一输出经抽查是两个吸引子短语的字符级变异
   （`ขอใช้สัญวิสัญ...` × `สองโนดููู...` 的排列组合）——字符串唯一性 ≠ 内容唯一性。
   护栏应升级为"对 hyp 集合做去重后仍需人工/聚类检查"或直接监控 n-gram 多样性。
2. **唯一真实内容耦合**：top-sim 样本全部是含"สอง"的 ref 配 `สอง...` 开头的 hyp——LLM 确实
   从嵌入中捡到了最显著的数字词，但仅此而已。

**机制结论**：内容在前端（CTC-CER 0.357/0.4999 证明帧级可线性解码），但 LLM 学不会
"注意力读取 250×2560 稠密向量→转写"（5k 样本预算内）。瓶颈在**嵌入读取**，不在内容缺失。

**→ 触发 §11.0 预案：v7 判别实验**（CTC 解码文本直接作 LLM 文本输入，绕过嵌入读取）。

### 11.1 v7 判别实验结果（2026-09-18）— **关键突破**

**配方变更 vs v6**：仅替换输入首段——从 250 个稠密语音嵌入 → CTC 草稿文本 token。
其余不变：相同 LoRA、prompt、数据、EOS、超参。

**h173 真实数字（F2LLM 重打分后）**：

| step | F2LLM SER | 多样性 n_unique | top1_share |
|---|---|---|---|
| 8 | 0.8172 | 115 | 5% |
| 100 | 0.6150 | — | — |
| 500 | 0.5884 | — | — |
| 800 | 0.5736 | — | — |
| 1000 | 0.5594 | 161 | 3% |
| **1200（终）** | **0.5605** | 161 | 3% |

**字面指标**（step 1200）：CER micro=**0.6867**、chrF=**29.44**、zero_rate=**7.5%**（13/173 完全正确）、
max 单条 CER=**3.88**（v6 max=38、F2LLM SER 异常源于灾难性 hallucination；v7 max 已降一个数量级）。

**对比历史终态 h173 SER**：

| 版本 | F2LLM SER | Δ vs 基线 |
|---|---|---|
| v3 (typhoon2.5 + Adapter B + 无训练 LoRA) | 0.8801 | — |
| v4 (+EOS, 续训 v3) | 0.8009 | -0.08 |
| v5 (+CTC 联合损失) | 0.8400 | -0.04 |
| v6 (冻结 CTC 前端 + 训 LoRA) | 0.8594 | -0.02 |
| **v7 (CTC 草稿→LLM 去噪)** | **0.5605** | **-0.32** |

**判定**：0.30 的下降幅度远超父项目“同配置重跑噪声底 0.014-0.034”的判定阈值（实测 0.034），
属真实改进。**首次脱离 0.84-0.90 坍缩区。**

**输出质量证据**（step 1200 抽样）：

| ref | hyp | 关系 |
|---|---|---|
| `ขออนุญาตรปภ. ประจำจุดประตู 1 จำนวน 2 นายเข้าครบ...` | `ขออนุญาตส่งพอปัดจุดตั้งหนึ่งสองนายเข้ากันแล้วค่ะ` | 数字词(1,2)+门控结构+礼貌词尾皆正确 |
| `กำลังจะออกไป ครับผม` | `วันนี้จะออกไปไหนนะครับผม` | 核心动词+礼貌词尾完全正确 |
| `เขาแจ้งบอกว่า...` | `แจ้งมอเตอร์ไซค์จะมาติดต่อ...` | "แจ้ง"+宾语结构保留 |
| `ลานจอดรถ วอสองค่ะ` | `นักศึกษาหกห้าสองค่ะ` | 数字"สอง"识别 |

**机制结论**：v6 卡在“LLM 学不会从稠密语音嵌入读内容”——v7 用 CTC 草稿文本作为 LLM 输入
（绕过注意力读取），证明**瓶颈确实在嵌入读取**（v7 的解开了；v6 的同样数据+前端没能解开）。
内容信号在前端（CTC-CER 0.4999）；LLM 现在做的只是“去噪+重组”文本，与自然语言相近。

### 11.3 v8：v7 续训 + CTC 一致性项（你点名的"联合训练"）

**配方（你指定的最小改动）**：v7 的全部配置不变（cascade 首段、prompt、数据、lr=1e-4、LoRA r16/α32、EOS、eval_every=100），
**唯一新增 = batch_size 4→1、grad_accum 4→16**（显存预算），
**v7 step_1200 LoRA 作 init**（保留去噪能力），
**新增 CTC 一致性项**：target = LLM 输入的草稿文本（不是 ground truth），speech embeds 通过冻结 adapter 计算后 `.detach()`（前端冻结，无梯度回流到 LoRA）。λ_ctc = 0.1。

**CTC 项的真实性**：因前端冻结 + se.detach，CTC 项**没有 LoRA 梯度**——它实际是监控项，
不参与优化。所以 v8 = "v7 续训 + CTC 一致性监控"，不是真正的"联合训练"。
（要真正让 CTC 梯度影响 LoRA，需要前端可微——这正是 v5/v6 走过的死路。）

**F2LLM SER 轨迹（h173）**：

| step | F2LLM SER | 多样性 n_unique | top1_share | CTC-CER(ns) |
|---|---|---|---|---|
| 112 | 0.5593 | 163 | 2% | 0.4999 |
| 208 | 0.5461 | 160 | 5% | 0.4999 |
| **304** | **0.5447** | 159 | — | **0.4999** |
| 400 | 0.5579 | 158 | 5% | 0.4999 |
| 512 | 0.5634 | 161 | 4% | 0.4999 |
| 704 | 0.5553 | 159 | 5% | 0.4999 |
| 800 | 0.5658 | 158 | 5% | 0.4999 |

**vs v7 最佳点 0.5594**：v8 step-304 SER 0.5447 → **改善 0.015**。
vs v7 终态 0.5605 → step-704 持平 0.5553。

**字面指标**（v8 终 step-800）：CER micro 0.6787、chrF 28.72、max CER 5.73、12/173 (6.9%) 完全正确、
zero_rate 几乎与 v7 持平（7.5%→6.9%）。

**机制结论**：
1. **0.015 改善 = v7 续训空间 ≈ 噪声底**：父项目实测同配置重跑噪声底 0.014–0.034——这 0.015 落在噪声带内，
   不能证明联合训练带来真实改善。CTC 项无 LoRA 梯度这一事实支持这个判断：v8 实质就是 bs=1 的 v7 续训。
2. **前端 CTC-CER 五个评估点恒为 0.4999**：v8 训练没有破坏前端表示（v5/v6 的崩坏模式被规避）。
3. **多样性 158-163 稳定**：v7 的非坍缩状态完整保留。
4. **CE loss 14-22 区间震荡**（v7 是 5-7）：bs=1 引入的更新噪声放大了梯度方差，但没造成模式坍缩（v3-v6 的塌缩会伴随多样性骤降和内容消失）。

**判决**：联合训练的"联合"没真正发生（CTC 项无 LoRA 梯度），结果与 v7 续训在噪声带内等价。
要打开下一个 SER 关口（0.56→0.4+），需要**让 CTC 梯度真的流入 LoRA**——
这要求解冻前端 + 极低 lr 联合微调（v5 教训是 lr 太高 + 联合目标互相拖垮）。
建议方向：**v9：解冻 adapter（极低 lr 1e-5），CTC 头仍冻结（监控用），目标 = v8 终 LoRA + ground-truth CTC**，逐项检测 SER/CTC-CER 是否同向改善、lr 是否稳。
若 v9 在 lr 1e-5 下 SER 跌出噪声带且 CTC-CER 改善，则真正打开了"前端也能在线学"的开关。

### 11.5 v9：adapter 解冻微调（lr=1e-5）— 也失败

**配方变更 vs v8**：①adapter `requires_grad=True`（v8/v7 都是冻结的）②adapter lr=1e-5，LoRA lr=1e-4（两组 lr 分开）③ bs=4（v7 标准配置，远程服务腾出 4.3GB 后回到原值）④CTC 头仍冻结（仅监控一致性）⑤F2LLM 延后到每个 eval 加载（释放 9GB 训练期间）⑥init 从 v7 step_1200

**结果（h173）**：

| step | F2LLM SER | CTC-CER(ns) | CER strict micro | chrF | 多样性 | top1 |
|---|---|---|---|---|---|---|
| 100 | **0.5654** | 0.4999 | — | — | 162 | 3% |
| 200 | 0.5815 | 0.4999 | 0.8566 | 27.86 | 162 | 5% |

OOM 中止（生成阶段 top-22GB；adapter 参与生成时激活驻留，F2LLM 即使延后也救不回来）。
**首次运行 step 100 SER=0.5537**（与再次运行的 0.5654 差 0.012）= 同配置重跑噪声，证实 v9 的
"最佳 SER"在 0.55-0.58 区间震荡，未脱离 v7 的 0.5594。

**vs v7 step 1200**：SER 持平（0.55-0.58 vs 0.5594），**字面 CER 反而显著恶化（0.86 vs 0.69）**——
联合微调即使 lr 降到 1e-5，前端"持续可解码"的能力在被 LoRA 端的漂移逐渐侵蚀。

**机制结论**：
1. **F2LLM SER 反映的是"嵌入-答案"的语义对齐，对前端表示崩坏的敏感度有限**——这是 F2LLM 的
   已知盲区之一（与上次验证门里的"跨格式数字盲区"同源）：hyp 即使"乱但仍含有 ref 的关键词"
   也能拿到高 sim。CER strict 微平均才是字面退化的最敏感信号。
2. **adapter 微调方向≠ PTT 适配**：lr 1e-5 在 200 步内把 CER 推高 0.17，但 SER 几乎不动——
   表明 adapter 学到的是与 LLM 协同的"embedding-space shift"，不是与 PTT 声学相关的"特征空间适配"。
3. **远程 8082 + GPU 解放并没有打开新窗口**：v9 的真正瓶颈是"adapter 一旦参与 LoRA 训练就
   退化"，与显存无关。

### 11.6 完整结论与建议

**SER 下限已被探明**：v7/v8/v9 反复落在 **0.55-0.58** 区间，超越父项目实测同配置重跑噪声带
（0.014-0.034）；CTC 前端 CER 0.4999 是 cascade 架构的硬上限。

**5 次实验穷举后的因果链**：

| 失败模式 | 触发原因 | 修复尝试 | 修复后状态 |
|---|---|---|---|
| v3-v4 模式坍缩 | 无 EOS / 纯 CE 收敛到模态短语 | v4 加 EOS | 输出 4 种，停止但不转录 |
| v5 联合 CE+CTC 拖垮 | 目标方向相反（CE→phrase，CTC→content） | v6 分离：CTC 头冻结 | 前端 OK，LLM 仍学不到 |
| v6 嵌入读取失败 | LLM 学不会从稠密向量读内容 | v7 cascade 用草稿文本 | ✓ SER 0.88→0.56 |
| v7 cascade 上限 | CTC 草稿 CER 0.50 是噪声地板 | v8 bs=1 续训 | ±0.015 噪声 |
| v9 adapter 退化 | 即使低 lr 也被 LoRA 端漂移带跑 | 不存在简单修复 | CER 反而恶化 |

**SER 0.56 → 0.4-0.5 区间的剩余杠杆**（按 ROI 排序）：

1. **扩训练数据 5k → 15-20k silver**：唯一已验证路径。父项目 silver 池有余量；用远程 8082 重跑
   高质量对齐（score ≥ 0.4），数据噪声下限会再降 → cascade 的草稿质量也更好 → LLM 上限提高。
   **估计 SER 改善 0.03-0.08**。
2. **端到端用 beam search + 业务热词强制**：v7 cascade + 推理时 `prefix_allowed_tokens_fn` 约束
   `数字+单位` 模式 → 纠错 PTT 数字频偏。**估计 SER 改善 0.02-0.05**。
3. **Cascade 两阶段**：先训 LLM 学会"draft → clean"（v7 已做），再用干净数据训 LLM 做真正的 ASR
   （无 draft）。v7 LoRA 已是此目标的渐进；下一轮可在 10x 数据上重做 → 期望 SER 0.4-0.5。
4. **业务规则后处理**：硬编码 เช็ด→เช็ค、数字格式归一化 → CER 跳 0.05-0.10。

**不建议的方向**：
- 任何形式的 adapter 解冻训练（v9 已证：联合退化）
- CTC 头解冻（v5 已证：blank 坍缩）
- 增大 batch 或 lr（v8 已证：高方差没有 SER 收益）

**核心洞察**：cascade 架构下 LLM 训练已接近渐近上限（0.55-0.58），下一跳必须是**数据**而非**算法**。
远程 8082 + GPU 释放已就绪，下一步如选择 ROI ①，是立等可走的纯工程任务（重跑对齐 + 扩展 split + 重训）。

### 11.7 待办

1. ✅ judge 抽查 — 取决于 DEEPSEEK env 恢复（一直未持久化）
2. ⏳ 数据扩展 + 重训 — 用户未决策
3. ⏳ beam search + 业务热词 — 推理侧改动，未启动



**loss 轨迹**：4.17（step40）→ 2.71 → 2.97 → 2.40 → 2.32 → 2.45（波动下降）。

**内容观察（step 304 抽样）**：多样性打开（28 种）但输出仍集中在两个旧吸引子短语
`หุสัญใช้สัญ...` 与 `สองโนดููู...` 的变体上——输入端已随音频变化（前端真实有信息），
但 LLM"读取稠密语音向量→转写"的映射尚未建立。

**中期解读**（写入时为 epoch 2 中段，最终判决以 F2LLM 重打分为准）：
1. **结构性验证通过**：CTC-CER 五个评估点纹丝不动 = 冻结正确、前端未被 CE 破坏（v5 教训被规避）。
2. **多样性首次脱离坍缩态**：此前所有版本（v3/v4/v5）从未超过 19 种且 top1 ≥ 50%；
   v6 在 step 304-400 达到 28-31 种、top1 最低 31%——输入端信息量的变化确实传导到了输出。
3. **内容映射未成**：变化的输出仍是短语级吸引子而非转写。两种可能：
   (a) 1-2 epoch 不够（LLM 学"读"250 维×T' 帧的向量本来就是难任务）；
   (b) 前端嵌入虽含内容（CTC-CER 0.50 证明可线性解码出内容）但其几何形态不利于 LM 注意力读取。
   判别实验（若 v6 终态 SER 仍 >0.7）：把 CTC 头的**逐帧 argmax 序列直接作为文本 token 输入** LLM
   （绕过嵌入读取，给 LLM 干净的字符证据）——若立刻大幅改善，说明瓶颈在"嵌入读取"而非"内容缺失"。

**已完成的判定动作**：无（等终态）。

**已确立**：
1. 前端可行性：CTC-only 配方使 adapter 在真实 PTT 音频上达到 CER 0.4999（超公开基线）。
2. 评估体系：F2LLM 主指标过验证门；多样性护栏、CTC-CER 诊断、微平均 CER 三层监控就位。
3. 失败图谱：句级对齐不足、无 EOS、联合目标竞争三大根因均已实证并有对应修复。

**进行中**：v6（冻结前端 + LLM LoRA）——判决标准：LLM 能否从 CTC 质量的嵌入中转录
（看 F2LLM SER 是否显著低于 0.80 坍缩区、多样性是否打开、CTC-CER 是否保持 ~0.50 证明前端未被破坏）。

**未决**：
1. v6 若成功 → 解冻微调的联合策略（低 lr？CTC 正则保持？）需单独设计。
2. 前端 CER 0.50 距业务可用（~0.15-0.2）仍远 → 域内数据量（Thai-SUP TTS 254h 可入 CTC 前端、真实域扩 silver）是下一杠杆。
3. judge 抽查待 DEEPSEEK env（上会话临时 export 未持久化）。
4. F2LLM 跨格式数字盲区的归一化方案（若 v6 成功后数字格式成为主要残差再处理）。
5. GPU 与对齐服务/其他会话的共存策略（当前：训练期不含 F2LLM）。
