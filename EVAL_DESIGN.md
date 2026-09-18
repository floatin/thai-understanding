# PTT 评测设计方案

> 目标：评估 XLSR-Thai + U-Align adapter + Qwen3-4B pipeline 在 PTT 真实业务语音上的语义转录质量。

**状态**：设计中（核心原则已对齐，剩余 8 项细节待确认）。带 `🟡 待定` 标记的条目尚未确定，需用户确认。

**最新更新**：
- §1 设计原则（5 条）已与 AGENTS.md §5 对齐
- §4 指标层级：LLM-judge 为主指标，embedding/chrF/CER 为辅助
- §10 报告格式：Judge SER 放主表首行

**相关文档**：
- `AGENTS.md` §5：评测铁律（伪指标使用边界、3 类独立指标要求、sanity check、threshold 校准）
- `PTT_PROMPT_TEMPLATE.md`：prompt 设计模板（待用户 check）

---

## 1. 设计原则（铁律见 AGENTS.md §5）

1. **主打语义一致性**——评测核心问题是"hyp 是否与 ref 语义等价"，不是字面重合；以 LLM-judge 为主要判断依据，n-gram / embedding 等指标仅作辅助证据
2. **至少 3 类独立指标**（n-gram / embedding / LLM-judge）——防止单一指标的偏置
3. **每个指标必须做 sanity check** 验证方向性
4. **threshold 必须基于带标签数据校准**，不能拍脑袋
5. **judge 与生成模型不同源**，避免自我评判偏差

---

## 2. 数据来源

### 2.1 Gold Standard（GT，金指标）

| 项 | 值 |
|---|---|
| 路径 | `/data/workspace/asr-model-training/out/gt_baserow_108.json` |
| 来源 | Baserow 表 1293，人工标注（`是否人工标注=已标注`） |
| 数量 | **107 条** |
| 字段 | `wav_name`, `turn_key`, `gt`, `baserow_id`, `source` |
| 音频位置 | `/data/workspace/asr-model-training/wav_sodexo/{wav_name}` |
| 用途 | **主评测集，唯一可信参考** |

### 2.2 Strong Silver（强银）

| 项 | 值 |
|---|---|
| 路径 | `/data/workspace/asr-model-training/out/pseudo_labels.json` |
| 来源 | 3 个 ASR 教师模型（teacher_a/b/c）转录结果一致 |
| 数量 | strong_silver 共 6497 条（`triple_confirmed=True` 5495 条） |
| 用途 | **混入主评测集**（**已确认**：用户决定混合） |

### 2.3 参考标准说明

GT（人工）和 silver（3 教师共识）**作为同一评测集的不同子集混入**。两个子集**分别统计指标并附注**，同时给出**合并数字**供业务侧决策参考。silver 的局限性已知（教师共识可能掩盖系统性偏置），在报告 §12 中列出。

---

## 3. 评测集

### 3.1 评测集：GT-107 + Silver-100 = PTT-207

| 项 | 值 |
|---|---|
| 总样本数 | **207**（GT 107 + Silver 100） |
| 数据来源 | `gt_baserow_108.json` 全部 + `pseudo_labels.json` top-100 strong silver |
| 抽样方式 | GT 全部使用；Silver 100 按 **🟡 待定** 策略抽取（见 §8） |
| Silver 筛选条件 | **🟡 待定**：`triple_confirmed=True AND cer=0`（5138 候选）或仅 `triple_confirmed=True`（5495 候选） |
| 报告位置 | 主报告含合并数字 + 两个子集分别数字 |

### 3.2 子集分别报告

| 子集 | 样本 | 报告位置 |
|---|---|---|
| GT-107 | 107 | 主表一行 |
| Silver-100 | 100 | 主表一行 |
| PTT-207 (合并) | 207 | 主表一行 |

---

## 4. 评测指标（主指标 + 辅助指标层级）

### 4.0 指标层级

| 层级 | 指标 | 用途 | 业务决策依据 |
|---|---|---|---|
| **主指标** | LLM-judge (deepseek-v4.1-flash) | 语义等价判断 | ✅ 是 |
| **辅助 1** | Semantic SER (bge-m3-Thai cosine) | 快速 sanity / 阈值筛选 | ❌ 否 |
| **辅助 2** | chrF (sacrebleu) | 字面重合度参考 | ❌ 否 |
| **辅助 3** | CER (Thai 规范化) | 字面错误率参考 | ❌ 否 |

**判断逻辑**：以 LLM-judge 数字为主要结论；embedding/chrF/CER 出现与 LLM-judge 冲突时，**以 LLM-judge 为准**并在报告中解释。

### 4.1 主指标：LLM-as-Judge

| 项 | 值 |
|---|---|
| Judge 模型 | **deepseek-v4.1-flash**（Alibaba Cloud MaaS，OpenAI 兼容 API；与生成模型 Qwen3 系列不同源） |
| 调用方式 | OpenAI Python SDK，base_url + api_key 从环境变量读取 |
| 评分维度 | **Yes / No 二分类**（语义等价 = Yes） |
| 样本量 | **全 207 样本**（作为主指标必须全覆盖） |
| 输出 | 1) judge-SER (主结论)<br>2) judge 分数分布<br>3) 与 embedding sim 的相关性<br>4) 两个子集（GT vs silver）的 judge 偏差 |
| Prompt 模板 | `PTT_PROMPT_TEMPLATE.md §5` |
| max_tokens | 2000（给 reasoning model 足够思考空间） |
| Sanity check | 3 个固定样本已验证（详 EVAL_DESIGN.md §9 附） |

### 4.2 辅助指标 1：Semantic SER (bge-m3-Thai cosine)

| 项 | 值 |
|---|---|
| 嵌入模型 | `jaeyong2/bge-m3-Thai`（基于 BAAI/bge-m3，Thai fine-tuned，2.2GB，~5GB GPU） |
| Pooling | sentence-transformers 默认（last token pooling + Normalize） |
| 相似度 | cosine |
| 用途 | 辅助 LLM-judge；快速 sanity check；不需要调用 API judge |
| Threshold | **🟡 待定**：需在 GT 子集上校准（见 §6） |
| 输出 | `semantic_sim` 分布 + 不同 threshold 下的 SER（仅作参考） |
| Sanity | 已验证：identical=1.0 / typo 0.84 / polite 0.91 / paraphrase 0.96 / unrelated 0.26-0.44 |

### 4.3 辅助指标 2：chrF

| 项 | 值 |
|---|---|
| 库 | `sacrebleu` |
| 计算粒度 | 字符级 n-gram（chrf / chrf++） |
| 优点 | 语言无关，对 Thai 友好 |
| 用途 | 字面重合度参考 |
| 输出 | `chrf` 和 `chrf++` 两个分数 |

### 4.4 辅助指标 3：CER（带 Thai 规范化）

| 项 | 值 |
|---|---|
| 实现 | Levenshtein 编辑距离 / ref 长度 |
| 规范化步骤（**🟡 待定**，待实现时定稿） | 1) 去标点 `. , ! ? ; :` 等<br>2) 去空白<br>3) 连续重复字符截断（>3 保留前 3 个）<br>4) tone marks 标准化（如 `\u0e4d` → `\u0e3a`） |
| 输出 | avg / median / min / max CER |

---

## 5. Pipeline

### 5.1 端到端流程

```
输入: wav 文件 + (gt | silver_text)
   ↓
Step 1: XLSR-Thai 提取 1024 维特征
   ↓
Step 2: U-Align adapter → 2560 维（与 Qwen3 输入空间对齐）
   ↓
Step 3: 拼接 prompt_embeds + speech_embeds → Qwen3-4B.generate()
   ↓
Step 4: hyp 文本
   ↓
Step 5: 计算 chrF / CER / Embed-sim / (judge-SER)
```

### 5.2 推理配置

| 项 | 值 |
|---|---|
| GPU 状态 | 等 PID 917327 释放（当前 19.7/23 GB used） |
| Adapter A | `/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_final_3ep.pt` |
| Adapter B | `/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_best.pt` |
| 推理硬件 | XLSR-Thai + adapter: GPU；Qwen3-4B: CPU（避免与 GPU embedding 抢显存） |
| LLM 推理参数 | greedy decoding, `max_new_tokens=64`, `do_sample=False` |
| Prompt | **🟡 待定**：见 §7 |
| 音频截断 | **10 秒**（**已确认**） |

---

## 6. Threshold 校准（🟡 待定流程）

**目的**：确定 bge-m3-Thai cosine 多少算"语义等价"。

**候选流程**：
1. 取 GT-107 中 20 条，先用 Adapter B 跑出 hyp
2. 人工（或用规则）将 20 对 (ref, hyp) 标 3 档：
   - 完全等价（应得高 cosine）
   - 部分等价（中等 cosine）
   - 不等价（低 cosine）
3. 观察 cosine 分布，找两个 cluster 之间的自然断点
4. 候选阈值报告：`0.7 / 0.75 / 0.8 / 0.85`，给出 sensitivity

---

## 7. Prompt 设计（仅 LLM-judge prompt）

生成 prompt 不在本设计文档讨论范围，见 `PTT_PROMPT_TEMPLATE.md §4`。

**LLM-judge prompt** 设计原则：

- 使用**长 prompt**，将业务场景 + 热词 + 纠偏规则装入上下文
- judge 模型为 SOTA LLM（deepseek-v4.1-flash），指令跟随能力强，能处理长 prompt
- judge 与生成 prompt 共享业务上下文和纠偏规则，避免"judge 严、生成松"的不一致
- 强制输出格式（只 Yes / No），便于后续程序解析

**LLM-judge prompt 模板**：见 `PTT_PROMPT_TEMPLATE.md §5`

---

## 8. 抽样策略

**GT-107**：全部使用，不过滤。

**Silver-100**：候选策略

| 策略 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| 随机 + seed | `random.sample(..., seed=42)` | 简单、可复现 | 可能与 GT 分布不齐 |
| 分层 | 按音频时长分桶（短/中/长），按群组分桶 | 与 GT 分布对齐 | 实现复杂 |
| 简单前 100 | 按 storage 顺序 | 最简单 | 可能偏置 |

**🟡 待定**：选用哪种策略。

---

## 9. Sanity Check（强制）

在报告任何指标前，先做以下 sanity check：

### 9.1 嵌入模型方向性
- 相同文本 vs 自身：cosine ≈ 1.0
- 相同文本 vs 完全无关文本（不同话题）：cosine 应较低
- 改写 vs 无关：cosine 应介于两者之间

### 9.2 指标方向性
- 字符完全相同的 (ref, hyp) 应得 CER = 0、chrF = 100、embedding sim ≈ 1.0
- 字符完全不同的 (ref, hyp) 应得 CER > 0.5、chrF 较低、embedding sim 较低

### 9.3 Adapter 行为
- Adapter A vs B 在 GT-107 上的输出应有 **可观察到的差异**（不是完全相同的输出）
- 之前 Thai-SUP 上观察到 B 更"像 Thai"输出，PTT 上应观察同样趋势或反向趋势

---

## 10. 报告格式

**业务决策依据是 LLM-judge，其他指标是辅助证据。** 报告以主指标为主。

### 10.1 主表（以 LLM-judge 为首行）

| 指标 (层级) | Adapter A (3-ep) | Adapter B (10-ep) | Δ |
|---|---|---|---|
| **Judge SER [主]** | x% | y% | ±Δ% |
| Embedding SER @ threshold=X [辅] | x% | y% | ±Δ% |
| chrF [辅] | x | y | ±Δ |
| CER (norm) [辅] | x% | y% | ±Δ% |

每个指标拆为三行子集：合并 / GT-107 / Silver-100，12 行总计。

### 10.2 指标冲突分析

**关键**：如果 LLM-judge 与辅助指标结论冲突（例如 LLM-judge 说 B > A，但 chrF 说 A > B），**以 LLM-judge 为准**，并在报告中列出冲突原因的可能解释。

### 10.3 附信息

- 样本数、threshold 值、prompt 文本（生成 + judge 两个 prompt）
- cosine similarity 分布（histogram，按子集分别画）
- sanity check 结果（4 项）
- 失败 case 例子（前 5 条 hyp vs ref）
- 两个子集（GT vs silver）的指标差异分析

---

## 11. 时间预算

| 阶段 | 内容 | 估计时间 |
|---|---|---|
| Sanity check (embedding) | 20 对样本 | 5 min（CPU） |
| Phase 1: Generation | 207 × 2 adapter × CPU Qwen3-4B | ~95 min |
| Phase 2: Embedding | 414 句 × GPU bge-m3-Thai | ~5 min（等 GPU） |
| **Phase 3: LLM-judge [主指标]** | 414 对 × deepseek-v4.1-flash API | ~12 min sequential（可并行加速至 2-3 min） |
| Phase 4: 计算 + 报告 | chrF / CER / embedding sim / judge SER | 5 min |
| **总计** | | **~130 min** |

---

## 12. 限制与已知风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| GT 仅 107 条 | 95% CI 半宽 ±9% on SER | 报告附 CI；不夸大数字意义；混入 100 silver 扩到 207 |
| Silver 标签可能与 GT 系统性偏置 | silver 子集 SER 可能偏乐观 | 子集分别报告，便于发现系统性偏置 |
| LLM-judge 作为主指标的依赖 | judge 不准 → 主指标不准 | judge 模型与生成模型不同源；sanity check 通过 |
| bge-m3-Thai 在 Thai 上精度已 sanity check（方向性 ✓），但具体 threshold 需校准 | sanity check ✓ + sensitivity report |
| 端到端 pipeline 在 PTT 上从未跑过 | 输出分布未知 | pilot 5 句先看 |
| GPU 资源争抢 | generation 时间不可控 | 等显存后再开 |
| deepseek-v4.1-flash reasoning tokens 开销 | 414 对 × ~40 reasoning tokens 额外消耗 | max_tokens=2000 预留空间 |
| Judge 与 embedding/chrF 结论冲突 | 需人工解读 | §10.2 列出冲突分析 |

---

## 13. 待确认事项汇总

| # | 待确认 | 状态 |
|---|---|---|
| 1 | GT 与 silver 是否分开评测 | ✅ **已确认**：混合（合并 + 子集分别报告） |
| 2 | 是否启用 Silver-100 评测 | ✅ **已确认**：启用 |
| 3 | 音频截断长度 | ✅ **已确认**：10 秒 |
| 4 | 评测核心是语义一致性，LLM-judge 为主指标 | ✅ **已确认** |
| 5 | Silver 筛选条件 | ✅ **默认采用 A**：`triple_confirmed=True AND cer=0`（5138 候选） |
| 6 | Silver 抽样方式 | ✅ **默认采用**：random seed=42（简单可复现） |
| 7 | Prompt 设计 | ✅ **已确认**（`PTT_PROMPT_TEMPLATE.md` 你 check 过） |
| 8 | CER 规范化具体规则 | ✅ **默认采用** §4.4 规则（去标点/空白/重复>3/tone marks） |
| 9 | Semantic SER threshold 校准流程 | **🟡 待定**（见 §6，可在生成后定） |
| 10 | LLM-judge 评分维度 | ✅ **已确认**：Yes / No 二分类 |
| 11 | LLM-judge prompt 设计 | ✅ **已确认**（`PTT_PROMPT_TEMPLATE.md §5`） |
| 12 | Judge 模型 | ✅ **已确认**：deepseek-v4.1-flash（OpenAI 兼容 API） |

**剩余 1 项待定**：Semantic SER threshold 校准（不阻断实施，在 generation 完成后做）。

---

**确认所有 🟡 待定项后，正式进入实施阶段。**
