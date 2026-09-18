"""Judge spot-check (Phase 4): deepseek Yes/No on a sample of (ref, hyp) pairs.

Role changed per user decision: judge is no longer the primary metric; it is
used (a) once to validate F2LLM ranking consistency, (b) at final go/no-go.

Reads an eval_history.jsonl line (subset + step), samples n pairs (seed 42),
runs the PTT_PROMPT_TEMPLATE §5.4 judge prompt, parses first-token Yes/No,
reports judge-SER (= No rate) + agreement with F2LLM sims (Spearman, and
judge-No rate in F2LLM sim bands).

Env required: DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL (not persisted
from previous session — set before running).

Usage: python eval_judge_spotcheck.py --subset h173 --step -1 --n 150
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

EVAL_HISTORY = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v3/eval_history.jsonl'
OUT_DIR = '/data/workspace/asr-model-training/thai-understanding/out_metric_gate'

JUDGE_PROMPT = """你是一名 PTT 业务语音转录质量的评审员。你将评估一个转录系统（hyp）与人工标注的标准答案（ref）是否语义等价。

【业务场景】
你是一名泰国的语音转录助手。当前音频来自[物业]的[保安]使用对讲机进行日常工作沟通。
常见内容包括门禁报备、巡逻签到、异常上报、设备状态确认、换班交接等。
请将听到的泰语语音逐字转录为标准书面泰文。

【热词提示】
岗位角色: 班长, 队长, 中控, 前台
设备设施: 对讲机, 门禁, 监控, 信号, 报警器, 摄像头
地点方位: 大门, 后门, 侧门, 地下停车场, 一楼大厅, 楼顶, 围墙
数字编号: 门牌号, 时间编号, 车位号

【纠偏规则】(对 ref 和 hyp 应用相同规则后再判等价)
- เช็ด + 设备上下文 → เช็ค
- "สิบหกห้า" + 时间上下文 → "16:05"
- 连续重复字符超过3次 → 压缩为1次
- 礼貌词尾 ค่ะ/ครับ/นะ/คะ 的互换、保留、省略不携带语义

【评判标准】
算等价(Yes): 同义词替换 / 同义改写 / 礼貌词尾互换或省略 / 应用纠偏规则后一致 / 数字规范化后等价("สิบหกห้า"="16:05"="165"格式差异不算错) / 重复字符压缩后等价
不算等价(No): 关键信息缺失 / 关键信息增加 / 主体意思反转 / 严重语义偏离

【输出格式】只回答 Yes 或 No，一行，不要解释。

ref: {ref}
hyp: {hyp}

语义等价吗？(Yes/No):"""

SANITY_CASES = [
    ('ขออนุญาตเช็ดสัญญาณครับ', 'ขออนุญาตเช็คสัญญาณครับ', 'Yes'),   # hotword rule
    ('ประตูหนึ่งค่ะ', 'ประตูสองค่ะ', 'No'),                        # number differs
    ('รับทราบครับ', 'รับทราบค่ะ', 'Yes'),                          # polite swap
    ('ส่งเวรแล้วครับ', 'ส่งเวรแล้วครับ', 'Yes'),                    # identical
]


def parse_verdict(text):
    m = re.search(r'\b(yes|no)\b', (text or '').strip(), re.I)
    return m.group(1).capitalize() if m else 'UNPARSED'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='h173')
    ap.add_argument('--step', type=int, default=-1)
    ap.add_argument('--n', type=int, default=150)
    args = ap.parse_args()

    lines = [json.loads(l) for l in open(EVAL_HISTORY)]
    lines = [r for r in lines if r.get('subset') == args.subset and r['step'] != 8]  # step8 = smoke
    rec = lines[args.step] if args.step >= 0 else lines[-1]
    print(f"judging: step={rec['step']} subset={rec['subset']} n={len(rec['refs'])} (F2LLM SER={rec['ser']:.4f})")

    client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url=os.environ['DEEPSEEK_BASE_URL'])
    model_id = os.environ['DEEPSEEK_MODEL']

    print("\n=== judge sanity cases (must match EVAL_DESIGN expectations) ===")
    for ref, hyp, expect in SANITY_CASES:
        r = client.chat.completions.create(
            model=model_id, max_tokens=2000,
            messages=[{'role': 'user', 'content': JUDGE_PROMPT.format(ref=ref, hyp=hyp)}],
            extra_body={'enable_thinking': False},
        )
        got = parse_verdict(r.choices[0].message.content)
        print(f"  expect={expect} got={got}  ref={ref!r} hyp={hyp!r} {'OK' if got == expect else 'FAIL'}")

    rng = np.random.RandomState(42)
    idx = rng.choice(len(rec['refs']), size=min(args.n, len(rec['refs'])), replace=False)
    pairs = [(rec['refs'][i], rec['hyps'][i], rec['sims'][i], int(i)) for i in idx]

    results, t0 = [], time.time()
    for k, (ref, hyp, sim, i) in enumerate(pairs):
        try:
            r = client.chat.completions.create(
                model=model_id, max_tokens=2000,
                messages=[{'role': 'user', 'content': JUDGE_PROMPT.format(ref=ref, hyp=hyp)}],
                extra_body={'enable_thinking': False},
            )
            verdict = parse_verdict(r.choices[0].message.content)
        except Exception as e:
            verdict = 'API_ERR'
        results.append({'i': i, 'ref': ref, 'hyp': hyp, 'f2llm_sim': sim, 'verdict': verdict})
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(pairs)} judged ({time.time()-t0:.0f}s)", flush=True)

    Path(OUT_DIR).mkdir(exist_ok=True)
    with open(f"{OUT_DIR}/judge_spotcheck_{args.subset}_step{rec['step']}.json", 'w') as f:
        json.dump({'step': rec['step'], 'subset': args.subset, 'n': len(results),
                   'results': results, 'prompt': JUDGE_PROMPT}, f, ensure_ascii=False, indent=1)

    ok = [r for r in results if r['verdict'] in ('Yes', 'No')]
    no_rate = np.mean([r['verdict'] == 'No' for r in ok]) if ok else float('nan')
    print(f"\n  judge SER (=No rate): {no_rate:.4f}  (n={len(ok)}, unparsed/api_err={len(results)-len(ok)})")

    from scipy.stats import spearmanr
    sims = np.array([r['f2llm_sim'] for r in ok])
    judge_yes = np.array([r['verdict'] == 'Yes' for r in ok], dtype=float)
    rho, p = spearmanr(sims, judge_yes)
    print(f"  F2LLM sim vs judge-Yes Spearman: rho={rho:.4f} (p={p:.4g})")
    for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 2.0)):
        band = [(r['verdict'] == 'Yes') for r in ok if lo <= r['f2llm_sim'] < hi]
        if band:
            print(f"    sim [{lo:.1f},{hi:.1f}): n={len(band)} judge-Yes rate={np.mean(band):.3f}")


if __name__ == '__main__':
    main()
