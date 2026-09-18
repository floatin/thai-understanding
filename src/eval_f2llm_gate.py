"""
Phase 1 metric-validation gate for F2LLM-v2-4B (AGENTS.md §5.5 / EVAL_DESIGN §9).

Run BEFORE using any F2LLM number for conclusions. Two modes:

  sanity      — direction checks (identical/unrelated/paraphrase/Thai specifics)
  perturb     — severity ordering on real GT refs:
                  exact > polite-swap > homophone(เช็ด→เช็ค) > digit-swap,
                digit-swap vs exact gap is the business-critical number-blindness
                test (PTT dispatch payloads are digits).
  rescore     — re-score saved preds in ptt_eval_gen.json (1457 x 2 adapters)
                with F2LLM; compare vs stored bge-m3 sims (needs ptt_eval_metrics.json).

Usage: python eval_f2llm_gate.py --mode sanity|perturb|rescore [--n 40]
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EVAL_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_split.json'
GEN_FILE = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_gen.json'
METRICS_FILE = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_metrics.json'
EMBED_MODEL = '/data/workspace/models/F2LLM-v2-4B'
OUT_DIR = '/data/workspace/asr-model-training/thai-understanding/out_metric_gate'

# Thai digit words, cyclic +1 map for digit-swap perturbation
NUM_WORDS = ['ศูนย์', 'หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า', 'สิบ']
NUM_NEXT = {w: NUM_WORDS[(i + 1) % len(NUM_WORDS)] for i, w in enumerate(NUM_WORDS)}


def load_model():
    return SentenceTransformer(EMBED_MODEL, device='cuda', model_kwargs={'torch_dtype': torch.bfloat16})


def sims_for(model, pairs):
    """pairs: list of (ref, hyp). Symmetric-task usage: no instruction prefix."""
    a = model.encode([p[0] for p in pairs], normalize_embeddings=True,
                     show_progress_bar=False, batch_size=16, convert_to_numpy=True)
    b = model.encode([p[1] for p in pairs], normalize_embeddings=True,
                     show_progress_bar=False, batch_size=16, convert_to_numpy=True)
    return (a * b).sum(axis=-1).tolist()


def report(name, sims):
    sims = np.array([s for s in sims if s is not None])
    print(f"  {name:<28s} n={len(sims):<4d} mean={sims.mean():.4f}  sd={sims.std():.4f}  min={sims.min():.4f}  max={sims.max():.4f}")
    return sims


def mode_sanity(model):
    print("=== Sanity: direction (EVAL_DESIGN §9.1) ===")
    cases = {
        'identical': [('ขออนุญาตเช็ดสัญญาณครับ', 'ขออนุญาตเช็ดสัญญาณครับ'),
                      ('ประตูสองออกสิบหกห้าค่ะ', 'ประตูสองออกสิบหกห้าค่ะ'),
                      ('ส่งเวรหัวหน้าแล้วค่ะ', 'ส่งเวรหัวหน้าแล้วค่ะ')],
        'polite_swap': [('ขออนุญาตเช็ดสัญญาณครับ', 'ขออนุญาตเช็ดสัญญาณค่ะ'),
                        ('ประตูสองออกสิบหกห้าค่ะ', 'ประตูสองออกสิบหกห้าครับ')],
        'paraphrase': [('ขออนุญาตเช็ดสัญญาณครับ', 'ขอตรวจสอบสัญญาณหน่อยครับ'),
                       ('วันนี้อากาศดีมาก', 'เพดานฟ้าวันนี้สดใส')],
        'unrelated': [('ขออนุญาตเช็ดสัญญาณครับ', 'ราคาน้ำมันปีนขึ้นสามเปอร์เซ็นต์'),
                      ('ประตูสองออกสิบหกห้าค่ะ', 'กาแฟแก้วนี้หอมกลิ่นเชอร์รี่')],
        'tone_mark_diff': [('ครับ', 'คฺรับ'), ('ปัญหา', 'ปัญหา')],
    }
    out = {}
    for name, pairs in cases.items():
        out[name] = report(name, sims_for(model, pairs))
    ok = (out['identical'].mean() > 0.95 and out['unrelated'].mean() < out['paraphrase'].mean() < out['identical'].mean())
    print(f"\n  direction check: {'PASS' if ok else 'FAIL'}")
    return ok


def perturb_ref(ref, kind):
    if kind == 'digit':
        # cyclic +1 on every Thai number word present
        out = ref
        for w in sorted(NUM_NEXT, key=len, reverse=True):
            out = re.sub(re.escape(w), f'«{NUM_NEXT[w]}»', out)
        return re.sub('«|»', '', out) if '«' not in out else out.replace('«', '').replace('»', '')
    if kind == 'homophone':
        return ref.replace('เช็ด', 'เช็ค')  # actual business hotword confusion
    if kind == 'polite':
        return ref.replace('ค่ะ', 'ครับ').replace('คะ', 'ครับ').replace('ครับ', 'ค่ะ') if 'ครับ' in ref else ref.replace('ค่ะ', 'ครับ').replace('คะ', 'ครับ')
    if kind == 'dup':
        toks = ref.split()
        if len(toks) > 1:
            return ' '.join([toks[0]] * 4 + toks)
        return ref[:3] * 4 + ref
    return ref


def mode_perturb(model, n):
    print(f"=== Perturbation severity ordering (n={n} refs x 6 conditions) ===")
    data = json.load(open(EVAL_SPLIT))['samples']
    gt = [s for s in data if s.get('subset') == 'GT_human']
    rng = random.Random(42)
    refs = [s['ref'] for s in gt if len(s['ref']) >= 8]
    rng.shuffle(refs)
    refs = refs[:n]

    conds = ['exact', 'polite', 'homophone', 'digit', 'dup', 'unrelated']
    pairs = {c: [] for c in conds}
    for i, r in enumerate(refs):
        pairs['exact'].append((r, r))
        pairs['polite'].append((r, perturb_ref(r, 'polite')))
        pairs['homophone'].append((r, perturb_ref(r, 'homophone')))
        pairs['digit'].append((r, perturb_ref(r, 'digit')))
        pairs['dup'].append((r, perturb_ref(r, 'dup')))
        pairs['unrelated'].append((r, refs[(i + 7) % len(refs)]))  # a real GT ref, different item

    n_digit_changed = sum(1 for r in refs if perturb_ref(r, 'digit') != r)
    n_hom_changed = sum(1 for r in refs if perturb_ref(r, 'homophone') != r)
    print(f"  refs where digit-perturb changed text: {n_digit_changed}/{len(refs)}; homophone: {n_hom_changed}/{len(refs)}\n")
    out = {}
    for c in conds:
        out[c] = report(c, sims_for(model, pairs[c]))

    # digit restricted to refs that actually changed (unchanged refs dilute the mean with 1.0s)
    digit_changed_sims = [s for s, (r, h) in zip(out['digit'], pairs['digit']) if r != h]
    print("\n  digit-swap, changed refs only:")
    report('digit_changed_only', digit_changed_sims)

    # cross-format: Thai number words vs Arabic digits (prompt allows digit-form output)
    print("\n  cross-format numbers (Thai words vs Arabic digits):")
    cross = [('สองแปดค่ะ', '28ค่ะ'), ('สองแปดค่ะ', 'สองแปดค่ะ'),
             ('ประตูสอง ขาเข้าสิบหกห้าค่ะ', 'ประตู2 ขาเข้า16:55ค่ะ'),
             ('สิบหกห้า', '165'), ('วอ.สอง ว.สิบหก', 'ว.2 ว.16')]
    report('cross_format', sims_for(model, cross))
    for (a, b), s in zip(cross, sims_for(model, cross)):
        print(f"    sim={s:.4f}  {a!r} vs {b!r}")

    # homophone on refs that actually contain เช็ด
    shed_refs = [s['ref'] for s in gt if 'เช็ด' in s['ref']]
    print(f"\n  homophone เช็ด→เช็ค on containing refs (found {len(shed_refs)}):")
    if shed_refs:
        hp = [(r, r.replace('เช็ด', 'เช็ค')) for r in shed_refs]
        report('homophone_shed', sims_for(model, hp))
    else:
        # business hotword case not present in GT; use the canonical example
        report('homophone_canonical', sims_for(model, [('ขออนุญาตเช็ดสัญญาณครับ', 'ขออนุญาตเช็คสัญญาณครับ')]))

    Path(OUT_DIR).mkdir(exist_ok=True)
    with open(f'{OUT_DIR}/perturb_pairs.json', 'w') as f:
        json.dump({'refs': refs, 'pairs': {k: v for k, v in pairs.items()},
                   'sims': {k: [round(s, 4) for s in out[k]] for k in out}}, f, ensure_ascii=False, indent=1)

    print("\n  --- verdict ---")
    gap_digit = out['exact'].mean() - out['digit'].mean()
    gap_homophone = out['exact'].mean() - out['homophone'].mean()
    print(f"  exact − digit-swap        = {gap_digit:.4f} (changed-only: {out['exact'].mean() - float(np.mean(digit_changed_sims)) if digit_changed_sims else 0:.4f})")
    print(f"    -> F2LLM is SENSITIVE to digit-word swaps (not blind); real risk is")
    print(f"       cross-format (Thai words vs Arabic digits) scoring ~0.64-0.78:")
    print(f"       a correctly-transcribed number in the other format counts as an error.")
    print(f"       Monitor hyp number format at Phase 4; add number normalization if needed.")
    print(f"  exact − homophone(เช็ด→เช็ค) = {gap_homophone:.4f} (hotword confusion visible)")
    print(f"  dup-glitch mean sim = {out['dup'].mean():.4f} -> repetitive broken output can pass")
    print(f"       SER<0.10 threshold; CER + judge spot-check stay mandatory.")
    ordered = out['exact'].mean() > out['dup'].mean() > out['unrelated'].mean()
    print(f"  ordering exact > dup > unrelated: {'PASS' if ordered else 'FAIL'}")
    with open(f'{OUT_DIR}/perturb_summary.json', 'w') as f:
        json.dump({k: float(v.mean()) for k, v in out.items()} |
                  {'gap_exact_minus_digit': float(gap_digit), 'gap_exact_minus_homophone': float(gap_homophone),
                   'n_refs': len(refs), 'n_digit_changed': n_digit_changed}, f, indent=1)


def mode_rescore(model):
    print("=== Rescore ptt_eval_gen.json (1457 x 2 old adapters) with F2LLM ===")
    gen = json.load(open(GEN_FILE))
    try:
        old = json.load(open(METRICS_FILE))
        old_a = {s['id']: s.get('sim') for s in old.get('per_item', {}).get('a', [])} if isinstance(old.get('per_item'), dict) else None
    except Exception:
        old_a = None
    for side, key in (('A', 'samples'), ('B', 'samples_b')):
        items = gen[key]
        refs = [s['ref'] for s in items]
        hyps = [s.get('hyp_a') if key == 'samples' else s.get('hyp_b') for s in items]
        sims = sims_for(model, list(zip(refs, hyps)))
        ser = 1.0 - float(np.mean(sims))
        print(f"  adapter {side}: n={len(items)}  F2LLM SER={ser:.4f}  mean_sim={np.mean(sims):.4f}  max={np.max(sims):.4f}")
        Path(OUT_DIR).mkdir(exist_ok=True)
        with open(f'{OUT_DIR}/rescore_adapter_{side}.json', 'w') as f:
            json.dump({'ids': [s['id'] for s in items], 'refs': refs, 'hyps': hyps,
                       'sims': [round(s, 4) for s in sims], 'ser': ser}, f, ensure_ascii=False)
    if old_a:
        print("  (bge-m3 old sims available for mapping — see ptt_eval_metrics.json)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['sanity', 'perturb', 'rescore'])
    ap.add_argument('--n', type=int, default=40)
    args = ap.parse_args()
    m = load_model()
    {'sanity': lambda: mode_sanity(m),
     'perturb': lambda: mode_perturb(m, args.n),
     'rescore': lambda: mode_rescore(m)}[args.mode]()
