"""
Build PTT eval/train split per the new design:
- Eval: 173 h173 (真人工标注) + top 20% strong_silver by 1-cer score
- Train: all other strong_silver
- Wav dedup across both sets

Reads:
  - e7_holdout173.json (173 h173)
  - pseudo_labels.json (6497 strong_silver)

Writes:
  - ptt_eval_split.json (eval set)
  - ptt_train_split.json (train set)
"""
import warnings
warnings.filterwarnings("ignore")
import json
import os
import sys
from collections import Counter
from pathlib import Path

PTT_ROOT = Path('/data/workspace/asr-model-training')
OUT_DIR = PTT_ROOT / 'out'

H173_FILE = OUT_DIR / 'e7_holdout173.json'
PSEUDO_FILE = OUT_DIR / 'pseudo_labels.json'
EVAL_OUT = PTT_ROOT / 'thai-understanding' / 'ptt_eval_split.json'
TRAIN_OUT = PTT_ROOT / 'thai-understanding' / 'ptt_train_split.json'
TOP_PCT = 0.20  # top 20% of strong_silver


def main():
    print("=== Building PTT eval/train split ===\n")

    # Load h173
    print(f"[1] Loading h173 from {H173_FILE}...")
    with open(H173_FILE) as f:
        h173_raw = json.load(f)['e7']
    print(f"  h173 entries: {len(h173_raw)}")

    h173 = []
    for r in h173_raw:
        h173.append({
            'wav_path': os.path.join(str(PTT_ROOT / 'wav_sodexo'), r['wav']),
            'wav_name': r['wav'],
            'ref': r['gt'],
            'subset': 'GT_human',
            'source': 'e7_holdout173',
            'score': float(r['conf']),
            'id': f"h173_{r['wav']}",
        })

    # Load strong_silver
    print(f"\n[2] Loading strong_silver from {PSEUDO_FILE}...")
    with open(PSEUDO_FILE) as f:
        pseudo = json.load(f)
    silver = [r for r in pseudo if r.get('subtier') == 'strong_silver']
    print(f"  strong_silver total: {len(silver)}")

    # Compute score = 1 - cer
    for r in silver:
        r['score'] = 1.0 - float(r.get('cer', 1.0))

    # Sort by score desc
    silver.sort(key=lambda r: r['score'], reverse=True)
    print(f"  score range: {silver[-1]['score']:.4f} (lowest) to {silver[0]['score']:.4f} (highest)")
    print(f"  cer=0 entries: {sum(1 for r in silver if r.get('cer', 1.0) == 0)}")

    # Top 20%
    n_top = int(len(silver) * TOP_PCT)
    top_silver = silver[:n_top]
    print(f"  top {TOP_PCT*100:.0f}% (top {n_top} by score 1-cer)")

    # Build eval silver
    eval_silver = []
    for r in top_silver:
        eval_silver.append({
            'wav_path': r['wav'],
            'ref': r['text'],
            'subset': 'Silver_top20',
            'source': 'pseudo_labels_strong_silver',
            'score': r['score'],
            'id': f"sil_{os.path.basename(r['wav'])}_{int(r.get('start', 0)*1000)}",
        })

    # Wav dedup: h173 wavs (high quality wins)
    h173_wavs = set(r['wav_name'] for r in h173)
    eval_silver_dedup = []
    for s in eval_silver:
        bn = os.path.basename(s['wav_path'])
        if bn in h173_wavs:
            # Skip this silver entry, h173 has it
            continue
        eval_silver_dedup.append(s)
    n_dedup = len(eval_silver) - len(eval_silver_dedup)
    print(f"\n[3] Wav dedup: removed {n_dedup} silver entries that overlap with h173")

    # Final eval set
    eval_set = h173 + eval_silver_dedup
    print(f"\n[4] Final eval set: {len(eval_set)} samples")
    print(f"   - h173: {len(h173)}")
    print(f"   - Silver top 20%: {len(eval_silver_dedup)} (after dedup)")

    # Train set = all strong_silver MINUS eval set
    eval_wavs = set(os.path.basename(r['wav_path']) for r in eval_set)
    train = []
    for r in silver:
        bn = os.path.basename(r['wav'])
        if bn in eval_wavs:
            continue
        train.append({
            'wav_path': r['wav'],
            'ref': r['text'],
            'source': 'pseudo_labels_strong_silver',
            'score': r['score'],
            'cer': r.get('cer', 1.0),
            'id': f"sil_{os.path.basename(r['wav'])}_{int(r.get('start', 0)*1000)}",
        })
    print(f"\n[5] Train set: {len(train)} samples (strong_silver minus eval)")

    # Quality stats
    train_cer0 = sum(1 for r in train if r['cer'] == 0)
    train_score_avg = sum(r['score'] for r in train) / len(train) if train else 0
    print(f"   - cer=0 in train: {train_cer0} ({train_cer0/len(train)*100:.1f}%)")
    print(f"   - avg score: {train_score_avg:.4f}")

    # Save
    EVAL_OUT.write_text(json.dumps({
        'config': {
            'h173': str(H173_FILE),
            'pseudo': str(PSEUDO_FILE),
            'top_pct': TOP_PCT,
            'score_def': '1 - cer',
            'wav_dedup': True,
        },
        'n_samples': len(eval_set),
        'h173_count': len(h173),
        'silver_count': len(eval_silver_dedup),
        'samples': eval_set,
    }, indent=2, ensure_ascii=False))
    print(f"\n[6] Saved eval split: {EVAL_OUT}")

    TRAIN_OUT.write_text(json.dumps({
        'config': {
            'source': 'pseudo_labels_strong_silver',
            'minus_eval_wavs': True,
        },
        'n_samples': len(train),
        'samples': train,
    }, indent=2, ensure_ascii=False))
    print(f"    Saved train split: {TRAIN_OUT}")

    # Final summary
    print(f"\n=== Summary ===")
    print(f"Eval: {len(eval_set)} (h173 {len(h173)} + Silver {len(eval_silver_dedup)})")
    print(f"Train: {len(train)}")
    print(f"Total strong_silver used: {len(eval_silver_dedup) + len(train)} (out of {len(silver)})")


if __name__ == '__main__':
    main()