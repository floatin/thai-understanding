"""Build expanded PTT training pool (v10) — silver usage sanctioned by user
(2026-09-18), hard constraint: eval wavs must never enter training.

Pool rule (superset of the current ptt_train_split.json):
  - keep:  subtier=='strong_silver' minus eval wavs   (= the current 5,159)
  - add:   status=='accepted' & subtier!='strong_silver' & drop_reason=='' minus eval wavs
           (teacher-accepted segments that passed upstream quality gates)
  - never: conflict rows, eval wavs (h173 + Silver_top20), missing audio.

Writes ptt_train_split_v10.json; the original split file is left untouched so
old runs stay reproducible.
"""
import warnings
warnings.filterwarnings("ignore")
import json
import os
from collections import Counter
from pathlib import Path

PTT = Path('/data/workspace/asr-model-training')
PSEUDO = PTT / 'out' / 'pseudo_labels.json'
EVAL_SPLIT = PTT / 'thai-understanding' / 'ptt_eval_split.json'
TRAIN_OLD = PTT / 'thai-understanding' / 'ptt_train_split.json'
TRAIN_OUT = PTT / 'thai-understanding' / 'ptt_train_split_v10.json'


def main():
    pseudo = json.load(open(PSEUDO))
    eval_samples = json.load(open(EVAL_SPLIT))['samples']
    eval_wavs = {os.path.basename(s['wav_path']) for s in eval_samples}
    old = json.load(open(TRAIN_OLD))['samples']
    old_ids = {s['id'] for s in old}
    h173_wavs = {os.path.basename(s['wav_path']) for s in eval_samples if s['subset'] == 'GT_human'}

    keep, add, drop_stats = [], [], Counter()
    for r in pseudo:
        bn = os.path.basename(r['wav'])
        if r.get('status') != 'accepted':
            drop_stats['not_accepted'] += 1
            continue
        if bn in eval_wavs:
            drop_stats['eval_wav'] += 1
            continue
        if not os.path.exists(r['wav']):
            drop_stats['missing_wav'] += 1
            continue
        row = {
            'wav_path': r['wav'],
            'ref': r['text'],
            'source': f"pseudo_labels_{r.get('subtier') or 'c_resolved'}",
            'score': 1.0 - float(r.get('cer', 1.0)),
            'cer': float(r.get('cer', 1.0)),
            'id': f"sil_{bn}_{int(r.get('start', 0) * 1000)}",
        }
        if r.get('subtier') == 'strong_silver':
            keep.append(row)
        elif not r.get('drop_reason'):
            add.append(row)
        else:
            drop_stats['added_tier_flagged'] += 1

    # 去重（同 id 保留第一条）
    seen, pool = set(), []
    for row in keep + add:
        if row['id'] in seen:
            drop_stats['dup_id'] += 1
            continue
        seen.add(row['id'])
        pool.append(row)

    # 防泄漏断言（硬约束）
    pool_wavs = {os.path.basename(s['wav_path']) for s in pool}
    assert not (pool_wavs & eval_wavs), "LEAK: pool wav in eval"
    assert not (pool_wavs & h173_wavs), "LEAK: pool wav in h173"
    assert old_ids <= {s['id'] for s in pool}, "old train pool is not a subset"
    assert all(s['ref'].strip() for s in pool), "empty ref"

    comp = Counter(s['source'] for s in pool)
    hours = sum(os.path.getsize(s['wav_path']) for s in pool) / 2 / 32000 / 3600  # 16kHz*2B
    print(f"pool: {len(pool)}  (kept strong_silver {len(keep)} + added {len(add)})")
    print(f"drop stats: {dict(drop_stats)}")
    print(f"composition: {dict(comp)}")
    print(f"audio hours (approx): {hours:.1f}")
    cers = sorted(s['cer'] for s in pool)
    print(f"cer mean={sum(cers)/len(cers):.4f} p50={cers[len(cers)//2]:.4f} p90={cers[int(len(cers)*0.9)]:.4f}")

    TRAIN_OUT.write_text(json.dumps({
        'config': {
            'source': 'pseudo_labels accepted; strong_silver kept as-is, other tiers drop_reason-clean',
            'minus_eval_wavs': True,
            'eval_split': str(EVAL_SPLIT),
            'sanction': 'user 2026-09-18: silver usage allowed, no eval leakage',
        },
        'n_samples': len(pool),
        'samples': pool,
    }, indent=2, ensure_ascii=False))
    print(f"saved {TRAIN_OUT}")


if __name__ == '__main__':
    main()
