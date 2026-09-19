"""CTC prefix beam search with a unit-level interpolated n-gram LM.

Why not pyctcdecode: it scores the LM at word boundaries; Thai orthography
has few spaces, so a character/unit-level LM scored at every step fits this
domain better. Units = thai_ctc_units merged units (base char + trailing Mn),
NOT raw unicode chars (§6 of TRAINING_METHODOLOGY.md).

LM: order-4 interpolated (Jelinek-Mercer) over unit sequences:
    P(u|c1c2c3) = w4*ML4 + w3*ML3 + w2*ML2 + w1*ML1
Serialized as JSON { "c1<c2<c3<u" contexts } -> keep it dependency-free.

Decode: log-domain prefix beam search (Hannun-style), per-frame top-k unit
pruning, ranking by acoustic logprob + alpha * LM cumulative logprob.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/data/workspace/asr-model-training/thai-understanding')
LM_PATH = ROOT / 'out_align' / 'unit_ngram_lm.json'
LM_ORDER = 4
WEIGHTS = (0.6, 0.25, 0.1, 0.05)  # 4/3/2/1-gram interpolation


def build_unit_lm(texts, vocab, order=LM_ORDER):
    """texts: list of Thai strings; vocab: {unit: id} from thai_ctc_units."""
    from thai_ctc_units import merge_units
    seqs = []
    vocab_set = set(vocab)
    for t in texts:
        units = [u for u in merge_units(t) if u in vocab_set]
        seqs.append(units)
    c1, c2, c3, c4 = Counter(), Counter(), Counter(), Counter()
    for units in seqs:
        for i, u in enumerate(units):
            c1[u] += 1
            if i >= 1:
                c2[units[i - 1] + '<' + u] += 1
            if i >= 2:
                c3[units[i - 2] + '<' + units[i - 1] + '<' + u] += 1
            if i >= 3:
                c4[units[i - 3] + '<' + units[i - 2] + '<' + units[i - 1] + '<' + u] += 1
    tot1 = sum(c1.values())
    m1 = {k: v / tot1 for k, v in c1.items()}
    m2 = _cond(c2, lambda k: k.rsplit('<', 1)[0])
    m3 = _cond(c3, lambda k: k.rsplit('<', 2)[0] if k.count('<') >= 2 else '')
    m4 = _cond(c4, lambda k: k.rsplit('<', 3)[0] if k.count('<') >= 3 else '')
    n4, n3, n2 = sum(len(v) for v in m4.values()), sum(len(v) for v in m3.values()), sum(len(v) for v in m2.values())
    print(f"unit LM: 4-gram entries={n4} 3-gram={n3} 2-gram={n2} 1-gram={len(m1)} corpus_units={tot1}")
    return {'order': order, 'w': WEIGHTS, 'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4}


def _cond(counter, ctx_fn):
    """Counter('a<b<c' -> count) -> {context: {unit: logP_ml}}."""
    ctx_tot = defaultdict(int)
    for k, v in counter.items():
        ctx_tot[ctx_fn(k)] += v
    out = defaultdict(dict)
    for k, v in counter.items():
        ctx = ctx_fn(k)
        out[ctx][k.rsplit('<', 1)[-1]] = math.log(v / ctx_tot[ctx])
    return dict(out)


def _split_units(text):
    from thai_ctc_units import merge_units
    return merge_units(text)


class UnitLM:
    def __init__(self, path, unseen=-23.0):
        d = json.load(open(path))
        self.w = d['w']
        self.m1, self.m2, self.m3, self.m4 = d['m1'], d['m2'], d['m3'], d['m4']
        self.unseen = unseen
        self._cache = {}

    def logp(self, unit, ctx3):
        """log P(unit | ctx3) with JM interpolation. ctx3: tuple of up to 3 units."""
        key = (unit, ctx3)
        if key in self._cache:
            return self._cache[key]
        w4, w3, w2, w1 = self.w
        if len(ctx3) >= 3:
            k4 = '<'.join(ctx3[-3:]) + '<' + unit
            p4 = self.m4.get('<'.join(ctx3[-3:]), {}).get(unit, self.unseen)
        else:
            p4 = None
        if len(ctx3) >= 2:
            p3 = self.m3.get('<'.join(ctx3[-2:]), {}).get(unit, self.unseen)
        else:
            p3 = None
        if len(ctx3) >= 1:
            p2 = self.m2.get(ctx3[-1], {}).get(unit, self.unseen)
        else:
            p2 = None
        p1 = self.m1.get(unit, self.unseen)
        terms = []
        if p4 is not None:
            terms.append(w4 * math.exp(p4))
        if p3 is not None:
            terms.append(w3 * math.exp(p3))
        if p2 is not None:
            terms.append(w2 * math.exp(p2))
        terms.append(w1 * math.exp(p1))
        lp = math.log(max(sum(terms), 1e-300))
        if len(self._cache) < 2_000_000:
            self._cache[key] = lp
        return lp


def beam_search_decode(logp, inv, lm=None, alpha=0.3, beam_width=30, topk=20):
    """logp: (T, V) np.float32 log-softmax; blank = id 0; returns decoded text.

    Prefix beam search (log domain). beams: prefix -> [logpb, logpnb,
    lm_state(last 3 units), lm_cum]. LM is charged alpha*logP(unit|ctx) when
    the decoded text extends; blank/collapse extensions don't extend text.
    """
    T, V = logp.shape
    beams = {(): [0.0, -math.inf, (), 0.0]}
    for t in range(T):
        row = logp[t]
        top = [int(i) for i in np.argsort(row)[-(topk + 1):] if i != 0]
        cand = {}
        for prefix, (lpb, lpnb, lmstate, lmcum) in beams.items():
            # blank emission: prefix unchanged, now ends with blank
            _merge(cand, prefix, _logaddexp(lpb, lpnb) + row[0], -math.inf,
                   lmstate, lmcum)
            for c in top:
                unit, ac = inv[c], row[c]
                if prefix and prefix[-1] == c:
                    # repeat of last unit, no blank between -> SAME token, text unchanged
                    _merge(cand, prefix, -math.inf, lpnb + ac, lmstate, lmcum)
                    # blank-then-c -> NEW token: text extends by c
                    lmv = alpha * lm.logp(unit, lmstate) if lm is not None else 0.0
                    _merge(cand, prefix + (c,), -math.inf, lpb + ac,
                           (lmstate + (unit,))[-3:], lmcum + lmv)
                else:
                    # text extends by unit c (from blank- or non-blank-ending beam)
                    lmv = alpha * lm.logp(unit, lmstate) if lm is not None else 0.0
                    _merge(cand, prefix + (c,), -math.inf,
                           _logaddexp(lpb, lpnb) + ac,
                           (lmstate + (unit,))[-3:], lmcum + lmv)
        scored = sorted(cand.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]) + kv[1][3],
                        reverse=True)
        beams = {p: v for p, v in scored[:beam_width]}
    best = max(beams.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]) + kv[1][3])
    return ''.join(inv[i] for i in best[0])


def _merge(cand, prefix, lpb, lpnb, lmstate, lmcum):
    if prefix in cand:
        cur = cand[prefix]
        cur[0] = _logaddexp(cur[0], lpb)
        cur[1] = _logaddexp(cur[1], lpnb)
    else:
        cand[prefix] = [lpb, lpnb, lmstate, lmcum]


def _logaddexp(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-lm', action='store_true')
    ap.add_argument('--eval-h173', action='store_true')
    ap.add_argument('--ckpt', default=str(ROOT / 'u_align_ctc_only_v10' / 'best' / 'ctc_only_v10.pt'))
    ap.add_argument('--vocab_pool', choices=['ckpt', 'v10', 'old'], default='ckpt',
                    help="where to get the unit vocab when the checkpoint doesn't embed one")
    ap.add_argument('--alphas', type=float, nargs='*', default=[0.0, 0.2, 0.4])
    ap.add_argument('--beam_width', type=int, default=30)
    ap.add_argument('--limit', type=int, default=173)
    args = ap.parse_args()

    if args.build_lm:
        import torch
        train = json.load(open(ROOT / 'ptt_train_split_v10.json'))['samples']
        ckpt_p = ROOT / 'u_align_ctc_only_v10' / 'best' / 'ctc_only_v10.pt'
        ckpt = torch.load(ckpt_p, map_location='cpu', weights_only=False) if ckpt_p.exists() else None
        vocab = ckpt['vocab'] if ckpt else build_vocab([s['ref'] for s in train], min_freq=5)[0]
        lm = build_unit_lm([s['ref'] for s in train], vocab)
        LM_PATH.write_text(json.dumps(lm, ensure_ascii=False))
        print(f"saved {LM_PATH}")
        return

    if args.eval_h173:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        import torch
        import torch.nn.functional as F
        import train_u_align_stage2_v5 as v5
        from train_u_align_v2 import UAlignAdapter
        from thai_ctc_units import ids_to_text, build_vocab
        from xlsr_thai import load_xlsr_thai, extract_features
        from eval_cer_pretrained import cer_no_space

        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        if 'vocab' in ck:
            vocab = ck['vocab']
        elif args.vocab_pool == 'v10':
            pool = json.load(open(ROOT / 'ptt_train_split_v10.json'))['samples']
            vocab = build_vocab([s['ref'] for s in pool], min_freq=5)[0]
        else:
            pool = json.load(open(ROOT / 'ptt_train_split.json'))['samples']
            vocab = build_vocab([s['ref'] for s in pool], min_freq=5)[0]
        inv = {v: k for k, v in vocab.items()}
        print(f"vocab: {len(vocab)} units (source: {'ckpt' if 'vocab' in ck else args.vocab_pool})")
        lm = UnitLM(LM_PATH) if LM_PATH.exists() else None
        print(f"LM loaded: {LM_PATH.exists()}, alpha grid: {args.alphas}")

        encoder, _ = load_xlsr_thai(v5.XLSR_CKPT, device='cuda')
        encoder.eval()
        adapter = UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
        adapter.load_state_dict(ck['adapter_state_dict'])
        ctc_head = torch.nn.Linear(2560, len(vocab)).cuda().float()
        ctc_head.load_state_dict(ck['ctc_head_state_dict'])
        adapter.eval()
        ctc_head.eval()

        eval_full = json.load(open(ROOT / 'ptt_eval_split.json'))['samples']
        h173 = [s for s in eval_full if s['subset'] == 'GT_human'][:args.limit]

        @torch.no_grad()
        def get_logp(s):
            audio = v5.load_audio(s['wav_path'])
            t = torch.from_numpy(audio).unsqueeze(0).cuda()
            feats, _ = extract_features(encoder, t, torch.tensor([t.shape[1]]).cuda())
            lp = F.log_softmax(ctc_head(adapter(feats.to(torch.bfloat16)).float().squeeze(0)), dim=-1)
            return lp.cpu().numpy().astype(np.float32)

        for alpha in args.alphas:
            cers = []
            for s in h173:
                lp = get_logp(s)
                hyp = beam_search_decode(lp, inv, lm=lm, alpha=alpha, beam_width=args.beam_width)
                cers.append(cer_no_space(s['ref'], hyp))
            print(f"alpha={alpha} beam={args.beam_width}: h173 CTC-CER(ns)={float(np.mean(cers)):.4f}", flush=True)


if __name__ == '__main__':
    main()
