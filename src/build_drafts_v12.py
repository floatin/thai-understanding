"""Build v12 multi-hypothesis drafts: A|B|C|S (3 teachers + CTC beam).

Rationale (§12.5): silver consensus text = majority vote of teacher a/b/c.
Feeding ALL THREE teacher transcripts + the CTC beam draft gives the LLM the
exact information the consensus was computed from — a far easier denoising
function than single-draft cleanup. Zero-LLM baselines (2026-09-19):
  teacher_c raw: h173 SER 0.3312 / silver1284 SER 0.0631
  silver consensus vs h173 GT: SER 0.4362 (= target-noise ceiling for any
  silver-target-trained denoiser)

Format: "A: <ta> | B: <tb> | C: <tc> | S: <ctc>"  (S = our CTC beam draft)
v7 collate truncates at MAX_DRAFT_TOKENS=192 (right-truncation, B tail goes
first as the weakest source is last... order: C best first is safer).
Format chosen: "C: .. | A: .. | S: .. | B: .." — best sources first so
truncation drops the weakest.
"""
import json
from pathlib import Path

ROOT = Path('/data/workspace/asr-model-training/thai-understanding')
OUT = ROOT / 'ctc_drafts_v12.json'

pseudo = json.load(open('/data/workspace/asr-model-training/out/pseudo_labels.json'))
by_wav = {}
for r in pseudo:
    by_wav.setdefault(r['wav'].split('/')[-1], []).append(r)

ctc = json.load(open(ROOT / 'ctc_drafts_v10.json'))
train = json.load(open(ROOT / 'ptt_train_split_v10.json'))['samples']
evals = json.load(open(ROOT / 'ptt_eval_split.json'))['samples']

def teacher_row(wav_path):
    rows = by_wav.get(Path(wav_path).name, [])
    if not rows:
        return {}
    return rows[0] if len(rows) == 1 else max(rows, key=lambda x: x.get('dur_ms', 0))

drafts, n_miss = {}, 0
for s in train + evals:
    r = teacher_row(s['wav_path'])
    ta, tb, tc = r.get('teacher_a') or '', r.get('teacher_b') or '', r.get('teacher_c') or ''
    cs = ctc.get(s['id'], '')
    if not (ta or tb or tc or cs):
        n_miss += 1
    parts = []
    if tc:
        parts.append(f"C: {tc}")
    if ta:
        parts.append(f"A: {ta}")
    if cs:
        parts.append(f"S: {cs}")
    if tb:
        parts.append(f"B: {tb}")
    drafts[s['id']] = ' | '.join(parts)

OUT.write_text(json.dumps(drafts, ensure_ascii=False))
lens = [len(v) for v in drafts.values()]
import statistics
print(f"saved {len(drafts)} drafts (empty-ish {sum(1 for v in drafts.values() if len(v)<4)}, "
      f"no-source {n_miss}) -> {OUT}")
print(f"char len: mean={statistics.mean(lens):.0f} p90={sorted(lens)[int(len(lens)*0.9)]}")
for sid in list(drafts)[:2]:
    print(f"  {drafts[sid][:110]!r}")
