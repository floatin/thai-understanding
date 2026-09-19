"""CER / chrF computation from eval_history.jsonl lines (Phase 4 auxiliary metrics).

CER conventions (parent-project lessons applied):
  - denominator = len(ref) (standard), NOT max(len)
  - aggregation = micro (Σedits / Σref_chars), macro reported alongside
  - normalization: NFC + strip P*/S*/C*/Z* by Unicode category (category-based,
    so Thai-legal ๆ (Lm) / ฺ ํ (Mn) are preserved by construction)
  - two variants: cer_strict (no repeat handling) and cer_norm (runs of >3 chars
    truncated to 3, EVAL_DESIGN §4.4) — collapse can mask dup-glitch failures,
    so strict is the primary.
  - always report max single-item CER + ref-length stats (AGENTS.md §4.6 macro-mean lesson)

Usage:
  python eval_cer_chrf.py --subset h173 [--step N]   # pick line from eval_history.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

# 归一化接入全项目唯一入口 thai_norm（2026-09-16 审计确立；此前本地副本是当年三份分叉的残留，
# 与 thai_norm 的差异：缺小写/泰文数字/sara am/Mn 排序，在 h173 上实测数值影响 <0.001）
_ROOT = '/data/workspace/asr-model-training'
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from thai_norm import norm_text as _thai_norm  # noqa: E402

EVAL_HISTORY = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v3/eval_history.jsonl'


def norm_text(s, collapse_repeats=False):
    s = _thai_norm(s)
    if collapse_repeats:
        prev, run, res = None, 0, []
        for ch in s:
            if ch == prev:
                run += 1
                if run < 3:
                    res.append(ch)
            else:
                prev, run = ch, 0
                res.append(ch)
        s = ''.join(res)
    return s


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default=EVAL_HISTORY)
    ap.add_argument('--subset', required=True)
    ap.add_argument('--step', type=int, default=-1, help='-1 = last matching line')
    args = ap.parse_args()

    lines = []
    for line in open(args.file):
        rec = json.loads(line)
        if rec.get('subset') == args.subset:
            lines.append(rec)
    assert lines, f"no lines for subset {args.subset}"
    rec = lines[args.step] if args.step >= 0 else lines[-1]
    print(f"line: step={rec['step']} subset={rec['subset']} n={len(rec['refs'])} (F2LLM SER={rec['ser']:.4f})")

    strict, norm, refs_n, hyps_n = [], [], [], []
    n_digit_fmt_mismatch = 0
    for ref, hyp in zip(rec['refs'], rec['hyps']):
        r, h = norm_text(ref), norm_text(hyp)
        r2, h2 = norm_text(ref, collapse_repeats=True), norm_text(hyp, collapse_repeats=True)
        refs_n.append(r)
        hyps_n.append(h)
        strict.append(levenshtein(h, r) / max(1, len(r)))
        norm.append(levenshtein(h2, r2) / max(1, len(r2)))
        if any('0' <= c <= '9' for c in h) != any('0' <= c <= '9' for c in r):
            n_digit_fmt_mismatch += 1

    strict, norm = np.array(strict), np.array(norm)
    ref_lens = [len(r) for r in refs_n]
    total_ref = sum(ref_lens)
    edits = [levenshtein(h, r) for h, r in zip(hyps_n, refs_n)]
    micro = sum(edits) / max(1, total_ref)
    chrf = CHRF().corpus_score(hyps_n, [refs_n]).score

    print(f"\n  CER strict : micro={micro:.4f}  macro={strict.mean():.4f}  max_item={strict.max():.4f}  zero_rate={(strict == 0).mean():.4f}")
    print(f"  CER norm(>3 collapse): micro={sum(levenshtein(norm_text(h, True), norm_text(r, True)) for h, r in zip(hyps_n, refs_n)) / total_ref:.4f}  macro={norm.mean():.4f}  max_item={norm.max():.4f}")
    print(f"  chrF       : {chrf:.2f}")
    print(f"  ref chars total={total_ref}  mean={np.mean(ref_lens):.1f}  p10={np.percentile(ref_lens, 10):.0f}  p90={np.percentile(ref_lens, 90):.0f}")
    print(f"  digit-format mismatches (hyp has Arabic digits xor ref has): {n_digit_fmt_mismatch}/{len(rec['refs'])}")

    out = Path(args.file).parent / f"cer_chrf_{args.subset}_step{rec['step']}.json"
    with open(out, 'w') as f:
        json.dump({'step': rec['step'], 'subset': rec['subset'], 'n': len(rec['refs']),
                   'cer_strict_micro': micro, 'cer_strict_macro': float(strict.mean()),
                   'cer_strict_max': float(strict.max()),
                   'cer_norm_micro': float(norm.mean()), 'chrf': chrf,
                   'digit_fmt_mismatch': n_digit_fmt_mismatch,
                   'per_item_cer_strict': [round(float(x), 4) for x in strict]}, f, indent=1)
    print(f"\n  saved {out}")


if __name__ == '__main__':
    main()
