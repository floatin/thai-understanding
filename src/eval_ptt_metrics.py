"""
Phase 2 + 3: Compute metrics on PTT eval results.

- Phase 2: bge-m3-Thai cosine similarity (辅助指标 1)
- Phase 3: deepseek-v4.1-flash Yes/No judge (主指标) with thinking disabled

Features:
- Incremental save every 10 judge entries
- Resume from partial state (checks ptt_eval_metrics.partial.json on startup)
- Per-pair judge timing logged
- Thinking explicitly disabled via extra_body

Reads: ptt_eval_gen.json
Writes: ptt_eval_metrics.json (final) + ptt_eval_metrics.partial.json (intermediate)
"""
import warnings
warnings.filterwarnings("ignore")
import os
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

GEN_FILE = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_gen.json'
OUT_FILE = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_metrics.json'
PARTIAL_FILE = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_metrics.partial.json'
EMBED_MODEL = '/data/workspace/models/bge-m3-Thai'
SAVE_EVERY = 10  # save every N judge calls


# ============ 1. Compute embedding similarity ============
def compute_embedding_sim(samples, samples_b, existing=None):
    """Compute embedding similarity. Skip if already in partial."""
    if existing and existing.get('sim_a') and existing.get('sim_b'):
        print("\n=== Phase 2: bge-m3-Thai (from cache) ===")
        print(f"  Loaded {len(existing['sim_a'])} sim_a, {len(existing['sim_b'])} sim_b from partial")
        return existing['sim_a'], existing['sim_b']

    print("\n=== Phase 2: bge-m3-Thai embedding ===")
    print(f"Loading {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL, device='cuda')
    print(f"  Loaded. mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    refs = [s['ref'] for s in samples]
    hyps_a = [s['hyp_a'] for s in samples]
    hyps_b = [s['hyp_b'] for s in samples_b]
    all_texts = refs + hyps_a + hyps_b
    print(f"Encoding {len(all_texts)} texts...")
    embs = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=16)
    embs = np.array(embs)
    n = len(refs)
    ref_embs = embs[:n]
    hyp_a_embs = embs[n:2*n]
    hyp_b_embs = embs[2*n:]

    def cos(a, b):
        return float((a * b).sum(-1))

    sim_a = [cos(ref_embs[i], hyp_a_embs[i]) for i in range(n)]
    sim_b = [cos(ref_embs[i], hyp_b_embs[i]) for i in range(n)]

    print(f"  Adapter A: avg sim = {np.mean(sim_a):.4f}, median = {np.median(sim_a):.4f}")
    print(f"  Adapter B: avg sim = {np.mean(sim_b):.4f}, median = {np.median(sim_b):.4f}")

    del model
    torch.cuda.empty_cache()
    return sim_a, sim_b


# ============ 2. Compute judge scores ============
def compute_judge(samples, samples_b, existing=None):
    """Compute LLM judge with thinking disabled, incremental save + resume."""
    print("\n=== Phase 3: deepseek-v4.1-flash judge (thinking disabled) ===")
    client = OpenAI(
        api_key=os.environ['DEEPSEEK_API_KEY'],
        base_url=os.environ['DEEPSEEK_BASE_URL'],
    )
    model_id = os.environ['DEEPSEEK_MODEL']

    # System prompt (from PTT_PROMPT_TEMPLATE.md §5)
    system_prompt = """你是 PTT 业务语音转录质量的评审员。判断 hyp 与 ref 是否语义等价。

【业务场景】
音频来自物业保安使用对讲机沟通。

【评判标准】
算等价（Yes）：
- 同义词替换、同义改写
- 礼貌词尾互换或省略（ค่ะ/ครับ/นะ/คะ 任意组合、保留、省略——都只算礼貌标记，不携带语义）
- 应用纠偏规则后字符一致（เช็ด + 设备上下文 → เช็ค）
- 数字规范化后等价
- 重复字符压缩后等价

不算等价（No）：
- 关键信息缺失或增加
- 主体意思反转
- 严重语义偏离

【输出格式】
只回答 Yes 或 No。"""

    # Load existing partial results
    judge_a = list(existing.get('judge_a', [])) if existing else []
    judge_b = list(existing.get('judge_b', [])) if existing else []
    start_idx = len(judge_a)
    n = len(samples)
    print(f"  Resuming from index {start_idx}/{n}")

    if start_idx >= n:
        print(f"  Already complete ({start_idx} >= {n})")
        return judge_a, judge_b

    refs = [s['ref'] for s in samples]
    hyps_a = [s['hyp_a'] for s in samples]
    hyps_b = [s['hyp_b'] for s in samples_b]

    def judge_one(ref, hyp):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"ref: {ref}\nhyp: {hyp}"},
                ],
                max_tokens=2000,
                temperature=0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content.strip()
            first = content.split()[0] if content else 'No'
            return first == 'Yes', content
        except Exception as e:
            print(f"    judge error: {e}")
            return None, str(e)

    t0 = time.time()
    for i in range(start_idx, n):
        yes_a, _ = judge_one(refs[i], hyps_a[i])
        yes_b, _ = judge_one(refs[i], hyps_b[i])
        judge_a.append(yes_a)
        judge_b.append(yes_b)

        # Save every SAVE_EVERY entries
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            # ETA based on processed-from-start
            processed = i + 1 - start_idx
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {elapsed:.0f}s, ETA {eta:.0f}s, saved to partial", flush=True)
            save_partial(judge_a, judge_b, sim_a=None, sim_b=None,
                         last_idx=i+1, complete=False)
        elif (i + 1) % 50 == 0:
            # Quick progress log without save
            elapsed = time.time() - t0
            processed = i + 1 - start_idx
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {elapsed:.0f}s, ETA {eta:.0f}s", flush=True)

    return judge_a, judge_b


# ============ Save/load partial ============
def save_partial(judge_a, judge_b, sim_a, sim_b, last_idx, complete):
    """Save partial state. Only judge results are checkpointed; sim saved once."""
    partial = {
        'last_idx': last_idx,
        'complete': complete,
        'judge_a': [bool(x) if x is not None else None for x in judge_a],
        'judge_b': [bool(x) if x is not None else None for x in judge_b],
    }
    if sim_a is not None:
        partial['sim_a'] = sim_a
    if sim_b is not None:
        partial['sim_b'] = sim_b
    Path(PARTIAL_FILE).write_text(json.dumps(partial, ensure_ascii=False))


def load_partial():
    """Load partial state if exists."""
    if not Path(PARTIAL_FILE).exists():
        return None
    try:
        with open(PARTIAL_FILE) as f:
            d = json.load(f)
        print(f"\n[Resume] Found partial: {d.get('last_idx', 0)} entries, complete={d.get('complete', False)}")
        return d
    except Exception as e:
        print(f"\n[Resume] Partial read failed: {e}")
        return None


# ============ 3. Aggregate ============
def aggregate(samples, samples_b, sim_a, sim_b, judge_a, judge_b):
    n = len(samples)
    out = []
    for i in range(n):
        out.append({
            **samples[i],
            'hyp_a': samples[i]['hyp_a'],
            'hyp_b': samples_b[i]['hyp_b'],
            'embed_sim_a': sim_a[i] if i < len(sim_a) else None,
            'embed_sim_b': sim_b[i] if i < len(sim_b) else None,
            'judge_yes_a': judge_a[i] if i < len(judge_a) else None,
            'judge_yes_b': judge_b[i] if i < len(judge_b) else None,
        })

    def stats(items, key):
        vals = [r[key] for r in items if r.get(key) is not None]
        if not vals:
            return None
        return {
            'n': len(vals),
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
        }

    def judge_ser(items, key):
        vals = [r[key] for r in items if r.get(key) is not None]
        if not vals:
            return None
        return float(1.0 - np.mean([1 if v else 0 for v in vals]))

    subsets = {}
    for r in out:
        subsets.setdefault(r['subset'], []).append(r)

    summary = {}
    for subset, items in subsets.items():
        summary[subset] = {
            'n': len(items),
            'embed_sim_a': stats(items, 'embed_sim_a'),
            'embed_sim_b': stats(items, 'embed_sim_b'),
            'judge_ser_a': judge_ser(items, 'judge_yes_a'),
            'judge_ser_b': judge_ser(items, 'judge_yes_b'),
        }
    summary['PTT-1457'] = {
        'n': n,
        'embed_sim_a': stats(out, 'embed_sim_a'),
        'embed_sim_b': stats(out, 'embed_sim_b'),
        'judge_ser_a': judge_ser(out, 'judge_yes_a'),
        'judge_ser_b': judge_ser(out, 'judge_yes_b'),
    }
    return out, summary


def main():
    print("=== Phase 2 + 3: Metrics on PTT-1457 (with resume) ===\n")
    print(f"Loading {GEN_FILE}...")
    with open(GEN_FILE) as f:
        d = json.load(f)
    samples = d['samples']
    samples_b = d['samples_b']
    print(f"  {len(samples)} samples, {len(samples_b)} adapter B results")

    # Load partial state if exists
    partial = load_partial()
    if partial and partial.get('complete'):
        print("Previous run completed! Just re-aggregating.")
        sim_a, sim_b = partial.get('sim_a'), partial.get('sim_b')
        judge_a, judge_b = partial.get('judge_a'), partial.get('judge_b')
    else:
        sim_a, sim_b = compute_embedding_sim(samples, samples_b, existing=partial)
        # Save sims
        save_partial(partial.get('judge_a', []) if partial else [],
                     partial.get('judge_b', []) if partial else [],
                     sim_a, sim_b, last_idx=partial.get('last_idx', 0) if partial else 0,
                     complete=False)
        judge_a, judge_b = compute_judge(samples, samples_b, existing=partial)

    # Mark complete and save final
    save_partial(judge_a, judge_b, sim_a, sim_b,
                 last_idx=len(judge_a), complete=True)
    out, summary = aggregate(samples, samples_b, sim_a, sim_b, judge_a, judge_b)

    print(f"\n=== Summary ===")
    for name, s in summary.items():
        print(f"\n[{name}] n={s['n']}")
        if s.get('embed_sim_a'):
            print(f"  Embed sim A: mean={s['embed_sim_a']['mean']:.4f}, median={s['embed_sim_a']['median']:.4f}")
            print(f"  Embed sim B: mean={s['embed_sim_b']['mean']:.4f}, median={s['embed_sim_b']['median']:.4f}")
        if s.get('judge_ser_a') is not None:
            print(f"  Judge SER A: {s['judge_ser_a']*100:.2f}%")
            print(f"  Judge SER B: {s['judge_ser_b']*100:.2f}%")

    Path(OUT_FILE).write_text(json.dumps({
        'config': {
            'n_samples': len(out),
            'embed_model': 'jaeyong2/bge-m3-Thai',
            'judge_model': 'deepseek-v4.1-flash',
            'judge_thinking': 'disabled',
        },
        'summary': summary,
        'samples': out,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved: {OUT_FILE}")


if __name__ == '__main__':
    main()