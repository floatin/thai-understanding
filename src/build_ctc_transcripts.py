"""Precompute CTC greedy-decoded draft transcripts for train + eval splits,
using the FROZEN CTC-only front-end (adapter + head, h173 CER 0.4999).

Output: ctc_drafts.json — {id: draft_text} for every sample in
ptt_train_split.json and ptt_eval_split.json. v7 consumes this; the LLM
denoising stage needs no audio at all (fast text-only SFT iteration).
"""
import warnings
warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v5 as v5
from thai_ctc_units import build_vocab, text_to_ids, ids_to_text
from xlsr_thai import load_xlsr_thai, extract_features

OUT = '/data/workspace/asr-model-training/thai-understanding/ctc_drafts.json'
CTC_ONLY_BEST = '/data/workspace/asr-model-training/thai-understanding/u_align_ctc_only/best/ctc_only.pt'


def main():
    train = json.load(open(v5.TRAIN_SPLIT))['samples']
    evals = json.load(open(v5.EVAL_SPLIT))['samples']
    vocab = json.load(open('u_align_stage2_v5/ctc_vocab.json'))
    inv = {v: k for k, v in vocab.items()}

    encoder, _ = load_xlsr_thai(v5.XLSR_CKPT, device='cuda')
    encoder.eval()
    ckpt = torch.load(CTC_ONLY_BEST, map_location='cpu', weights_only=False)
    adapter = v5.UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    adapter.eval()
    head = torch.nn.Linear(2560, len(vocab)).cuda().float()
    head.load_state_dict(ckpt['ctc_head_state_dict'])
    head.eval()

    drafts = {}
    all_items = train + evals
    for k, s in enumerate(all_items):
        audio = v5.load_audio(s['wav_path'])
        t = torch.from_numpy(audio).unsqueeze(0).cuda()
        with torch.no_grad():
            feats, _ = extract_features(encoder, t, torch.tensor([t.shape[1]]).cuda())
            logp = F.log_softmax(head(adapter(feats.to(torch.bfloat16)).float().squeeze(0)), dim=-1)
        am = logp.argmax(-1).tolist()
        prev, out = -1, []
        for i in am:
            if i != prev and i != 0:
                out.append(i)
            prev = i
        drafts[s['id']] = ids_to_text(out, inv)
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(all_items)}", flush=True)

    Path(OUT).write_text(json.dumps(drafts, ensure_ascii=False, indent=0))
    n_empty = sum(1 for v in drafts.values() if not v.strip())
    print(f"DONE {len(drafts)} drafts -> {OUT}  (empty: {n_empty})")
    for sid in list(drafts)[:3]:
        print(f"  {sid}: {drafts[sid][:60]!r}")


if __name__ == '__main__':
    main()
