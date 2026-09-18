# Stage 2 v3/v4 报告：typhoon2.5-qwen3-4b + F2LLM 评估体系

**日期**: 2026-09-17/18
**结论速览**: LLM 切换为 typhoon2.5 后，同配方 v3 一个 epoch 内 h173 SER 0.8801→0.8009（单调、未收敛）。
事后诊断发现**训练 target 从未包含 EOS**（v1/v2/v3 三代共有），模型学不会停止——短数字句内容已
转写正确但无限重复。v4（=v3+EOS target）验证中。

---

## 1. 本轮变更（单变量纪律）

| 变更 | 内容 | 不变量 |
|---|---|---|
| LLM | Qwen3-4B → **typhoon2.5-qwen3-4b**（Qwen3ForCausalLM, hidden 2560, tokenizer 词表与 Qwen3-4B 完全一致 151669） | prompt、数据、超参、REPLACE 结构全部不变 |
| 评估 | LLM-judge 主指标 → **F2LLM-v2-4B embedding 相似度为主**，CER/chrF 保留，judge 降级为抽查/校准 | SER 定义不变：1 − mean(cosine) |
| 数据划分 | **不动**（用户确认；h173+silver1284 与训练池 5159 逐 wav 零重叠已验证 + assert 固化） | |

架构：`音频(10s) → XLSR-Thai❄ → U-Align Adapter🔥(init=Adapter B) → 2560维 → REPLACE 输入前 T' 位 → typhoon2.5❄+LoRA(q/k/v/o, r16)🔥 → CE(answer-only)`。训练态显存峰值 13.1GB；eval 时叠加 F2LLM 19.4GB（A10 23GB 内）。

## 2. F2LLM 验证门（下结论前完成，AGENTS.md §5.5）

| 检验 | 结果 | 判定 |
|---|---|---|
| 方向性 sanity | identical 1.00 / paraphrase 0.61 / unrelated 0.07（动态范围远大于 bge-m3 的 0.26） | PASS |
| 数字词替换（changed-only, n=24） | mean sim **0.66** —— 对数字错误高度敏感（"数字盲区"假设被证伪） | 敏感 ✓ |
| **跨格式数字**（泰文词 vs 阿拉伯数字） | `สองแปด` vs `28` = **0.78**；`สิบหกห้า` vs `165` = 0.64 —— 正确转写但格式不同会被记为错误 | ⚠ 已列监控项 |
| 同音热词 เช็ด→เช็ค | 0.74（业务混淆可见） | ✓ |
| 整词重复故障（dup-glitch） | **0.91** —— 重复输出可骗过 SER<0.10 阈值 | ⚠ CER+judge 必须保留 |
| 礼貌词尾互换 | 0.99（按设计视为良性） | ✓ |
| 历史产物重算（旧 2 adapter × 1457） | F2LLM SER=0.89，与 judge 判定的 ~100% 失败一致 | ✓ 与主指标同向 |

产物：`out_metric_gate/perturb_summary.json`、`rescore_adapter_{A,B}.json`；脚本 `src/eval_f2llm_gate.py`。

## 3. v3 训练结果（typhoon2.5，1 epoch = 328 opt steps）

**训练曲线（h173, F2LLM SER）**：

| step | SER | 备注 |
|---|---|---|
| 0（未训 LoRA 锚点） | 0.8801 | 输出 = 无视音频的流利业务幻觉 |
| 200 | 0.8418 | **输出 96% 为音节循环**（暂态，见 §4） |
| 328（终点） | **0.8009** | 循环率降至 3%；loss 3.97→2.50 |

silver1284（终态）：F2LLM SER=0.8864 ｜ CER strict micro=3.02 ｜ chrF=2.09 ｜ max 单条 CER=31.7
h173（step-0 锚点）：CER micro=3.28 ｜ chrF=10.8
子集一致性：h173 与 silver 同向，无发散红旗。

**评估粒度教训**：eval_every=200 差点漏掉末端改善（step200→328 还有 −0.041）；best-checkpoint 逻辑据此误存了 step_200。v4 起改 eval_every=100。

## 4. 事后诊断（两条关键发现）

### 4.1 循环坍缩是暂态，不是病理
step-200 时 96% 输出为音节循环（`สองฟังดูููู...`），一度疑似训练崩坏；终点循环率自然回落到 3%。
**如果按中途快照停止训练或调参，会错杀一个仍在正常学习的过程。**

### 4.2 根因：训练 target 无 EOS（v1/v2/v3 共有）
step-328 的全部 top 输出呈现同一模式：**内容正确 + 短语无限重复**：
```
ref: สิบหกห้า          hyp: สิบหกห้าครับสิบหกห้าครับสิบหกห้าครับ...   (sim 0.786)
ref: สองแปดครับ       hyp: สองแปดครับสองแปดครับสองแปดครับ...       (sim 0.758)
```
回查 `collate_batch`：answer = ref token 序列，**末尾无 `<|im_end|>`**，labels 亦无 → 模型从未获得
"何时停"的梯度。greedy 推理必然顶满 max_new_tokens=64。这解释了历代 Stage 2 输出中
"重复/不停"失败形态的主导部分。

**排除项**（已检验）：greedy 解码伪影 —— rep_penalty=1.15 仅 −0.018（噪声底量级），短语循环依旧；
管道权重 — adapter 权重确认已更新（max|Δ|≈3e-3）；评估管道 — greedy 重跑与训练内评估逐位一致。

## 5. v4：EOS 修复验证（已完结——结论被 case 抽查推翻）

v4 终点（step 328 续训 1 epoch）：h173 SER=0.7508，173/173 输出正常停止、循环率 0、CER micro 0.88。
表面改善，但 **case 抽查发现输出总数只有 4 种**（142× `สองแปด`、22× `ขออนุญาตครับ`、8× `สอง`、1× 其他）——
模型退化为常数输出机，"精确转写"只是 ref 恰好等于模态短语。silver1284 上则是 945× `สองฟังดููู...`。

**完整多样性账目（本报告最关键的一张表）**：

| 检查点 | 输出去重数 / 173 | 主模态（占条数） |
|---|---|---|
| v3 @200 | 9 | `สองโนดููู...` (86)、`สองฟังดููู...` (80) |
| v3 @328 | 19 | `สองแปดครับ×N` (59)、`สิบหกห้าครับ×N` (57) |
| v4 @328 | 4 | `สองแปด` (142) |

**所有 Stage 2 运行（v2/v3/v4）都坍缩到少数模态短语，从未发生真实内容转写。**
v3 的"循环坍缩是暂态"与 v4 的"EOS 决定性证实"都是聚合指标（SER/CER/停机率）造成的误读——
我自己的循环正则只认 1-4 字符单元，`สองแปดครับ×N` 短语级重复完全漏检（已修：多样性护栏写入 eval 日志）。

### 5.1 根因链（证据齐了）

1. **声学信号存在但难**：XLSR-53+CTC（Thai-SUP 上 37.95%）在 h173 真实音频上 CER 65.4%，
   解码结果语音学邻近但错乱 → 域差距 severe，非零信号（`probe_ctc_h173.py`）。
2. **Adapter 在真实音频上输入敏感**：50 条真实音频两两 cosine 0.27（白噪声对照 0.98，确定性 1.0）
   → adapter 不是常量瓶颈（`train_u_align_stage2_v3.py` 环境内的探针）。
3. **致命缺口：Stage 1 是句级 InfoNCE 对齐**——它只教了"这句话和哪句话语义近"，
   没有提供任何帧级/音素级内容信号。LLM 面对约 250 个"不知道怎么读"的语音嵌入 +
   5k 样本的 CE 训练 → 收敛到平凡解：**输出答案分布的模态短语 + EOS**，
   loss 平台 2.2-2.5 ≈ 答案分布熵。论文 Stage 2 用 16k 小时数据补这个信号，我们只有 ~17h。
4. EOS 修复只改变了坍缩的表面形态（无限循环 → 短语+停机），没有触及内容缺失。

### 5.2 为什么 SER 降不到 0.3 以下

常数输出 vs 任意 ref 的 sim ≈ 0.13-0.25（按 ref 长度分层实测），SER 的下限就被钉在 ~0.75 附近。
0.3 以下需要真实内容转写发生——在当前"句级对齐 + 纯 CE Stage 2"的配方下不会发生。

## 6. 下一步选项（按证据强度排序）

1. **在域内先证明配方（推荐）**：用 Thai-SUP SR 的 TTS 音频（adapter 的 Stage 1 同域，230k/254h 可采样
   10-20k）跑 Stage 2。若 TTS 上也转写不了 → 配方/架构是墙（需 CTC 辅助损失或更大规模）；
   若能转写 → 问题归结为域迁移，再决定真实域数据策略。一次实验消掉最大的混杂变量。
2. CTC 辅助损失：给 speech 特征加帧级监督，直接补内容信号（改动较大）。
3. 数据量：真实域 silver 只有 5k；父项目有大量未用 silver 池可扩。
4. judge 抽查 150 对（需 DEEPSEEK env）——注意：在输出全部坍缩的当前状态下 judge 抽查意义有限，
   建议等有混合质量输出时再做。
5. ~~跨格式数字监控~~：在全模型能转写之前是二阶问题（当前 8/173）。

## 7. 产物清单

```
src/train_u_align_stage2_v3.py / v4.py      # 训练脚本（v4=EOS fix + init_from）
src/eval_f2llm_gate.py                       # 指标验证门（sanity/扰动/重算）
src/eval_step0_baseline.py                   # step-0 锚点
src/eval_cer_chrf.py                         # CER(标准分母+微平均)/chrF + 跨格式监控
src/eval_judge_spotcheck.py                  # judge 抽查（待 DEEPSEEK env）
src/eval_decode_ab.py / eval_final_checkpoint.py  # 解码 A/B 与终点评估
u_align_stage2_v3/eval_history.jsonl         # 全部逐条 sims+hyps+refs（原始预测留档）
u_align_stage2_v4/eval_history.jsonl         # v4 同上
out_metric_gate/                             # 验证门产物
```
