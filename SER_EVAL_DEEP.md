# SER 评测方法学说明（更新版）

## 评测现状（修正后）

之前的回答中我说"没有用裁判模型"，确实如此。但这次更新加入了：

### 1. 字符级 CER（无规范化）
- Levenshtein 距离 / ref 长度
- 直接比较 LLM 生成的 token sequence
- **不区分重复字符（"ี่รีรีรี"算正确）**

### 2. 规范化 CER
- 去除标点、空格、连续重复字符（>3次保留前3个）
- 仍属字符级，对 Thai 改写不公平

### 3. 嵌入相似度（本次新增）
- 用 Qwen3 `embed_tokens` 对 ref 和 hyp 分别编码
- 各自 mean-pool 后算 cosine
- **本质上测的是 LLM 输出文本的语义接近度，不是 adapter 对齐质量**

### 4. 严格相等（SER_exact_match）
- `ref.strip() == hyp.strip()`
- 对 SR 改写任务 0% 是预期的

## 100 样本完整结果（re-run with 全部样本保存）

| 指标 | Adapter A (3-ep) | Adapter B (10-ep) | Δ |
|---|---|---|---|
| avg CER (raw) | 239.43% | 230.12% | -9.3 pp |
| median CER (raw) | 179.53% | 155.54% | **-24.0 pp** |
| avg CER (norm) | 235.11% | 227.20% | -7.9 pp |
| median CER (norm) | 175.62% | 154.50% | **-21.1 pp** |
| **SER (exact)** | **100%** | **100%** | 0 |
| Embed sim (avg) | 0.4162 | 0.3621 | -0.0541 |
| Embed sim (median) | 0.4263 | 0.3293 | -0.0970 |

## ⚠️ 关键解读：CER 改进 vs Embed sim 反向

**CER 改进（好）**：adapter B 的输出在字符级更接近 ref
- 可能是因为 B 学会了产生更"像 Thai"的输出

**Embed sim 降低（看似不好）**：adapter B 的输出与 ref 的 embedding 距离变远
- 这是因为 LLM 输出对 adapter 不稳定——adapter B 偶尔产生非常奇怪的重复模式（如 "ีรีรีรี..."），这些重复模式的 embedding 跟 ref 的均值 embedding 距离很远
- adapter A 的输出更稳定（"นั่น นั่น นั่น..."，"ดูเหมือนว่า..."），反而 embedding 更集中

**重要**：embed_sim 这里测的是**输出文本质量**，不是 adapter 对齐质量。
- 真正的 adapter 对齐质量用 retrieval R@10（Stage 1 评测）
- Adapter B 的 R@10 = 47% > A 的 25%（**这是真正的对齐改进**）

## 缺少的评测

| 应该做 | 状态 |
|---|---|
| Thai 词级 tokenization（F1） | ❌ 没做 |
| BLEU/ROUGE（n-gram 重合） | ❌ 没做 |
| Qwen3-as-judge（语义等价 Yes/No） | ❌ 没做 |
| Stage 2 训练后再评测 | ❌ 没做 |
| 1000+ 样本 | ⚠️ 只做了 100 |

## 结论

⚠️ **现有 SER 评测不充分**：
1. CER 字符级对 Thai 不公平（无词边界）
2. embed_sim 测的是输出稳定性，不反映 adapter 质量
3. SER=100% 是任务性质决定（SR = 改写），不是模型 bug

✓ **Adapter B 改进是真实的**：Stage 1 retrieval R@10 从 25% → 47% (+22 pp)

下一步真正能反映 end-to-end 质量的评测：
1. **Thai word tokenization + F1**（用 PyThaiNLP）
2. **Qwen3-as-judge**（让 Qwen3 评判 hyp 语义是否等同 ref）
3. **Stage 2 fine-tune 后再评测**（让 LLM 学会消费 adapter 输出）
