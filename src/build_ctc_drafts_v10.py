"""Rebuild CTC draft transcripts with the v10 front-end (retrained on the
expanded pool) + prefix beam search with a unit n-gram LM.

Outputs:
  ctc_drafts_v10_greedy.json  — greedy decode (fallback / ablation)
  ctc_drafts_v10.json         — beam decode at the chosen alpha
Alpha is selected by h173 CTC-CER (validation set — never trained on).
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v5 as v5
from ctc_beam import beam_search_decode, UnitLM, LM_PATH
from thai_ctc_units import ids_to_text
from xlsr_thai import load_xlsr_thai, extract_features
from eval_cer_pretrained import cer_no_space

ROOT = Path('/data/workspace/asr-model-training/thai-understanding')
CKPT = ROOT / 'u_align_ctc_only_v10' / 'best' / 'ctc_only_v10.pt'
OUT_GREEDY = ROOT / 'ctc_drafts_v10_greedy.json'
OUT_BEAM = ROOT / 'ctc_drafts_v10.json'


def load_front_end(ckpt_path=CKPT):
    encoder, _ = load_xlsr_thai(v5.XLSR_CKPT, device='cuda')
    encoder.eval()
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'encoder_state_dict' in ck:
        encoder.load_state_dict(ck['encoder_state_dict'])
        print(f"encoder weights from ckpt (v10b-style)")
    vocab = ck['vocab']
    inv = {v: k for k, v in vocab.items()}
    adapter = v5.UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ck['adapter_state_dict'])
    adapter.eval()
    head = torch.nn.Linear(2560, len(vocab)).cuda().float()
    head.load_state_dict(ck['ctc_head_state_dict'])
    head.eval()
    return encoder, adapter, head, vocab, inv


@torch.no_grad()
def get_logp(encoder, adapter, head, wav_path):
    audio = v5.load_audio(wav_path)
    t = torch.from_numpy(audio).unsqueeze(0).cuda()
    feats, _ = extract_features(encoder, t, torch.tensor([t.shape[1]]).cuda())
    return F.log_softmax(head(adapter(feats.to(torch.bfloat16)).float().squeeze(0)), dim=-1)


def greedy_text(logp, inv):
    am = logp.argmax(-1).tolist()
    prev, out = -1, []
    for i in am:
        if i != prev and i != 0:
            out.append(i)
        prev = i
    return ids_to_text(out, inv)


_W = {}


def _init_worker(inv, lm):
    _W['inv'] = inv
    _W['lm'] = lm


def _beam(job):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ctc_beam import beam_search_decode
    from eval_cer_pretrained import cer_no_space
    i, alpha, width = job
    hyp = beam_search_decode(_LOGPS[i], _W['inv'], lm=_W['lm'], alpha=alpha, beam_width=width)
    return i, hyp, cer_no_space(_REFS[i], hyp)


_LOGPS, _REFS = None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep_alpha', type=float, nargs='*', default=None,
                    help='h173-only alpha sweep; skips draft building')
    ap.add_argument('--alpha', type=float, default=0.0,
                    help='LM weight; 0 = pure beam (LM shown to hurt, §12.2a)')
    ap.add_argument('--beam_width', type=int, default=30)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    encoder, adapter, head, vocab, inv = load_front_end()
    lm = UnitLM(LM_PATH) if args.alpha != 0 else None
    evals = json.load(open(ROOT / 'ptt_eval_split.json'))['samples']
    h173 = [s for s in evals if s['subset'] == 'GT_human']

    if args.sweep_alpha:
        for alpha in args.sweep_alpha:
            t0, cers = time.time(), []
            for s in h173:
                lp = get_logp(encoder, adapter, head, s['wav_path']).cpu().numpy().astype(np.float32)
                hyp = beam_search_decode(lp, inv, lm=lm, alpha=alpha, beam_width=args.beam_width)
                cers.append(cer_no_space(s['ref'], hyp))
            print(f"alpha={alpha}: h173 CTC-CER(ns)={float(np.mean(cers)):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return

    train = json.load(open(ROOT / 'ptt_train_split_v10.json'))['samples']
    items = train + evals
    if args.limit:
        items = items[:args.limit]
    print(f"items: {len(items)} (train {len(train)} + eval {len(evals)})", flush=True)

    # 1) batched logp extraction (one frozen-encoder forward per chunk)
    logps, drafts_g = [], {}
    t0 = time.time()
    B = 12
    for i0 in range(0, len(items), B):
        chunk = items[i0:i0 + B]
        audios = [v5.load_audio(s['wav_path']) for s in chunk]
        maxlen = max(len(a) for a in audios)
        wav = torch.zeros(len(chunk), maxlen, device='cuda')
        for i, a in enumerate(audios):
            wav[i, :len(a)] = torch.from_numpy(np.ascontiguousarray(a))
        with torch.no_grad():
            feats, lens = extract_features(encoder, wav,
                                           torch.tensor([len(a) for a in audios]).cuda())
            lp_all = F.log_softmax(head(adapter(feats.to(torch.bfloat16)).float()), dim=-1)
        for i, s in enumerate(chunk):
            # adapter 真实帧数 = (enc_len-2)//2+1（与训练侧 input_lengths 同式）。
            # pad 区帧是垃圾 logits——按编码器帧数切片会让短样本混入几十~几百帧垃圾（§12.2b bug）
            n_frames = max(1, (int(lens[i]) - 2) // 2 + 1)
            lp = lp_all[i, :n_frames].cpu().numpy().astype(np.float32)
            logps.append(lp)
            drafts_g[s['id']] = greedy_text(lp, inv)
        if (i0 // B) % 50 == 0:
            el = time.time() - t0
            print(f"  extract {i0+len(chunk)}/{len(items)}  {el:.0f}s", flush=True)

    # 2) multiprocessing beam decode
    from multiprocessing import Pool
    global _LOGPS, _REFS
    jobs = [(i, args.alpha, args.beam_width) for i in range(len(items))]
    ids = [s['id'] for s in items]
    refs = [s['ref'] for s in items]
    _LOGPS, _REFS = logps, refs

    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker,
                  initargs=({v: k for k, v in vocab.items()}, lm)) as pool:
            results = pool.map(_beam, jobs, chunksize=8)
    else:
        _init_worker({v: k for k, v in vocab.items()}, lm)
        results = [_beam(j) for j in jobs]

    drafts_b, h173_cers = {}, []
    h173_ids = {s['id'] for s in h173}
    for i, hyp, cer in results:
        drafts_b[ids[i]] = hyp
        if ids[i] in h173_ids:
            h173_cers.append(cer)
    OUT_GREEDY.write_text(json.dumps(drafts_g, ensure_ascii=False))
    OUT_BEAM.write_text(json.dumps(drafts_b, ensure_ascii=False))
    n_empty = sum(1 for v in drafts_b.values() if not v.strip())
    if h173_cers:
        print(f"self-check h173 beam CER(ns)={float(np.mean(h173_cers)):.4f} (n={len(h173_cers)})", flush=True)
    print(f"DONE {len(drafts_b)} drafts (empty {n_empty}) -> {OUT_BEAM} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
