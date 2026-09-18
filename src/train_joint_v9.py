"""Stage 2 v9: v7 cascade + adapter unfrozen (lr=1e-5).

v7 works (SER 0.56) — cascade front-end frozen, LLM LoRA learns to denoise
draft text. CTC-CER stays at 0.4999 throughout.

v9 hypothesis: PTT acoustic domain differs enough from TTS that the TTS-trained
adapter is the bottleneck. Unfreeze adapter at very low lr (1e-5, 10x lower
than v5's 1e-4) so it can adapt to PTT without the LLM destroying it.

Lessons applied from v5 / v6:
  - lr=1e-4 with adapter unfrozen killed both losses (v5). lr=1e-5 lets
    adapter move slowly enough to stay consistent with CTC head.
  - CTC head FROZEN — it's the stability anchor. If its CER drifts > 0.55
    on h173, we know the adapter broke.
  - F2LLM REAL-TIME during training (no longer deferred: remote align service
    freed 4.3GB, F2LLM fits in 23GB).

Recipe (single variable vs v7):
  - cascade unchanged: [draft_tokens | prompt | answer+EOS]
  - LoRA init from v7 step_1200 (cascade denoising preserved)
  - **adapter trainable, lr=1e-5** (NEW)
  - CTC head frozen (stability probe)
  - F2LLM scoring added to eval loop
  - bs=4 / grad_accum=4 (back to v7 standard — GPU free now)
  - eval_every=100 steps, full h173 + F2LLM
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v5 as base
import train_llm_denoise_v7 as v7

LLM_PATH = '/data/workspace/models/Typhoon2.5-qwen3-4b'
F2LLM_PATH = '/data/workspace/models/F2LLM-v2-4B'
CTC_ONLY_BEST = '/data/workspace/asr-model-training/thai-understanding/u_align_ctc_only/best/ctc_only.pt'
V7_INIT_LORA = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v7/checkpoints/step_1200/lora'
TRAIN_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_train_split.json'
EVAL_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_split.json'
DRAFTS = '/data/workspace/asr-model-training/thai-understanding/ctc_drafts.json'
V5_VOCAB = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v5/ctc_vocab.json'
XLSR_CKPT = '/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt'
OUTPUT_DIR = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v9'
EVAL_HISTORY = f'{OUTPUT_DIR}/eval_history.jsonl'


def main(args):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    drafts_map = json.load(open(DRAFTS))
    train_data = json.load(open(TRAIN_SPLIT))['samples']
    evals = json.load(open(EVAL_SPLIT))['samples']
    h173 = [s for s in evals if s.get('subset') == 'GT_human']
    vocab = json.load(open(V5_VOCAB))
    inv_vocab = {v: k for k, v in vocab.items()}

    before = len(train_data)
    train_data = [s for s in train_data if drafts_map.get(s['id'], '').strip()]
    print(f"train: {before} -> {len(train_data)} (empty-draft filter)")

    # Frozen backbone (XLSR-Thai)
    encoder, _ = base.load_xlsr_thai(XLSR_CKPT, device='cuda'); encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    # Adapter — UNFROZEN at lr 1e-5 (the v9 variable)
    ckpt = torch.load(CTC_ONLY_BEST, map_location='cpu', weights_only=False)
    from train_u_align_v2 import UAlignAdapter
    adapter = UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    for p in adapter.parameters(): p.requires_grad = True  # v9 change
    adapter.train()

    # CTC head — frozen (stability anchor)
    ctc_head = torch.nn.Linear(2560, len(vocab)).cuda().float()
    ctc_head.load_state_dict(ckpt['ctc_head_state_dict'])
    for p in ctc_head.parameters(): p.requires_grad = False
    ctc_head.eval()

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map='cuda')
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(base_model, V7_INIT_LORA, is_trainable=True)
    model.print_trainable_parameters()

    # Two LR groups: adapter at 1e-5 (gentle), LoRA at 1e-4 (v7 standard)
    adapter_params = list(adapter.parameters())
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n]
    opt = torch.optim.AdamW([
        {'params': adapter_params, 'lr': args.adapter_lr, 'weight_decay': 0.01},
        {'params': lora_params, 'lr': args.lr, 'weight_decay': 0.01},
    ])
    print(f"adapter lr={args.adapter_lr}, LoRA lr={args.lr}")

    # collate: v7 cascade only (no CTC path in training — CTC head used only as monitor)
    from xlsr_thai import extract_features
    MAX_DRAFT = v7.MAX_DRAFT_TOKENS
    MAX_ANS = v7.MAX_ANS_TOKENS

    def collate(batch):
        prompt_ids = tokenizer(base.GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0]
        P = prompt_ids.shape[0]
        embeds_list, labels_list, speech_frames_list, draft_text_list = [], [], [], []
        for b in batch:
            draft_ids = tokenizer(b['draft'], return_tensors='pt', add_special_tokens=False).input_ids[0][:MAX_DRAFT]
            ans_ids = tokenizer(b['ref'], return_tensors='pt', add_special_tokens=False).input_ids[0]
            ans_ids = torch.cat([ans_ids, torch.tensor([tokenizer.eos_token_id])])[:MAX_ANS]
            ids = torch.cat([draft_ids, prompt_ids, ans_ids]).cuda()
            emb = base_model.get_input_embeddings()(ids).to(torch.bfloat16)
            D, A = len(draft_ids), len(ans_ids)
            labels = torch.full((D + P + A,), -100, dtype=torch.long, device='cuda')
            labels[D + P:] = ans_ids.cuda()
            embeds_list.append(emb); labels_list.append(labels)
            # Speech embeds via adapter (which IS being trained; this is how adapter gets grad)
            audio = base.load_audio(b['wav_path'])
            t = torch.from_numpy(audio).unsqueeze(0).cuda()
            with torch.no_grad():
                feats, _ = extract_features(encoder, t, torch.tensor([t.shape[1]]).cuda())
            se = adapter(feats.to(torch.bfloat16)).squeeze(0)  # grad flows to adapter
            speech_frames_list.append(se)
            draft_text_list.append(b['draft'])
        max_len = max(e.shape[0] for e in embeds_list)
        B = len(batch)
        fe = torch.zeros(B, max_len, 2560, dtype=torch.bfloat16, device='cuda')
        fl = torch.full((B, max_len), -100, dtype=torch.long, device='cuda')
        fm = torch.zeros(B, max_len, dtype=torch.long, device='cuda')
        for i, (e, l) in enumerate(zip(embeds_list, labels_list)):
            fe[i, :e.shape[0]] = e; fl[i, :l.shape[0]] = l; fm[i, :l.shape[0]] = 1
        return {'inputs_embeds': fe, 'labels': fl, 'attention_mask': fm,
                'speech_frames': speech_frames_list, 'draft_texts': draft_text_list}

    # dataset must carry draft + wav_path
    class _Wrap(Dataset):
        def __init__(self, samples, drafts):
            self.samples = samples; self.drafts = drafts
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            s = self.samples[i]
            return {'draft': self.drafts.get(s['id'], ''), 'ref': s['ref'], 'id': s['id'],
                    'wav_path': s['wav_path']}
    loader = DataLoader(_Wrap(train_data, drafts_map), batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    warmup = int(0.05 * total_steps)
    from torch.optim.lr_scheduler import LambdaLR
    sched = LambdaLR(opt, lambda s: s / max(1, warmup) if s < warmup else
                     0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup))))

    # F2LLM loaded LAZILY in quick_eval (only needed at eval time — saves 9GB during training).
    # The remote align service freed 4.3GB; adapter unfreezing adds memory; loading F2LLM
    # up-front would force bs=4 OOM. Lazy load is the v6 pattern proven safe.
    f2llm_holder = {'m': None}
    def _load_f2llm():
        if f2llm_holder['m'] is not None:
            return f2llm_holder['m']
        from sentence_transformers import SentenceTransformer
        torch.cuda.empty_cache()
        f2llm_holder['m'] = SentenceTransformer(F2LLM_PATH, device='cuda', model_kwargs={'torch_dtype': torch.bfloat16})
        return f2llm_holder['m']
    def _free_f2llm():
        if f2llm_holder['m'] is not None:
            f2llm_holder['m'] = None
            torch.cuda.empty_cache()

    def quick_eval():
        adapter.eval(); ctc_head.eval(); model.eval()
        prompt_ids = tokenizer(base.GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0].cuda()
        prompt_emb = base_model.get_input_embeddings()(prompt_ids).to(torch.bfloat16)
        from eval_cer_pretrained import cer_no_space
        from thai_ctc_units import ids_to_text
        refs, hyps, ctc_hyps = [], [], []
        with torch.no_grad():
            for s in h173:
                audio = base.load_audio(s['wav_path'])
                t = torch.from_numpy(audio).unsqueeze(0).cuda()
                feats, _ = extract_features(encoder, t, torch.tensor([t.shape[1]]).cuda())
                se = adapter(feats.to(torch.bfloat16))
                logp = F.log_softmax(ctc_head(se.float().squeeze(0)), dim=-1)
                am = logp.argmax(-1).tolist()
                prev, out = -1, []
                for i in am:
                    if i != prev and i != 0: out.append(i)
                    prev = i
                ctc_hyps.append(ids_to_text(out, inv_vocab))
                draft = drafts_map.get(s['id'], '')
                d_ids = tokenizer(draft, return_tensors='pt', add_special_tokens=False).input_ids[0][:MAX_DRAFT].cuda()
                d_emb = base_model.get_input_embeddings()(d_ids).to(torch.bfloat16)
                inputs_embeds = torch.cat([d_emb, prompt_emb], dim=0).unsqueeze(0)
                o = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=MAX_ANS,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
                hyps.append(base.strip_think(tokenizer.decode(o[0], skip_special_tokens=True).strip()))
                refs.append(s['ref'])
        ctc_cer = float(np.mean([cer_no_space(r, h) for r, h in zip(refs, ctc_hyps)]))
        # F2LLM SER (lazy-load on first eval, free after)
        used = _load_f2llm()
        torch.cuda.empty_cache()
        embs = used.encode(refs + hyps, normalize_embeddings=True, batch_size=16,
                            show_progress_bar=False, convert_to_numpy=True)
        n = len(refs)
        sims = (embs[:n] * embs[n:]).sum(axis=-1)
        ser = 1.0 - float(np.mean(sims))
        _free_f2llm()
        return refs, hyps, ctc_cer, ser, sims.tolist()

    print("=== v9: v7 cascade + adapter UNFROZEN (lr=1e-5), bs=4 ===\n")
    Path(OUTPUT_DIR, 'checkpoints').mkdir(exist_ok=True)
    gstep, acc, t0 = 0, 0.0, time.time()
    next_eval = args.eval_every
    opt.zero_grad()
    model.train()
    adapter.train()
    best_ser = 1.0
    for ep in range(args.epochs):
        for bi, batch in enumerate(loader):
            out_lm = model(inputs_embeds=batch['inputs_embeds'], attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            ce_loss = out_lm.loss
            total = ce_loss / args.grad_accum
            total.backward()
            acc += ce_loss.item()
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(adapter_params + lora_params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                gstep += args.grad_accum
                if gstep % 20 == 0:
                    print(f"  step {gstep}/{total_steps} ce={acc:.3f} {time.time()-t0:.0f}s", flush=True)
                acc = 0.0
                if args.max_steps and gstep >= args.max_steps:
                    refs, hyps, ctc_cer, ser, sims = quick_eval()
                    uniq = len(set(hyps)); top1 = max([hyps.count(h) for h in set(hyps)], default=0)
                    print(f"  >> step {gstep} F2LLM SER={ser:.4f}  CTC-CER={ctc_cer:.4f}  [div: {uniq} unique, top1 {top1}]", flush=True)
                    with open(EVAL_HISTORY, 'a') as f:
                        f.write(json.dumps({'step': gstep, 'subset': 'h173', 'ser': round(ser, 4),
                                            'ctc_cer_ns': round(ctc_cer, 4),
                                            'n_unique': uniq, 'top1_share': round(top1/len(hyps), 3),
                                            'sims': sims, 'hyps': hyps, 'refs': refs}) + '\n')
                    return
                if gstep >= next_eval:
                    next_eval += args.eval_every
                    refs, hyps, ctc_cer, ser, sims = quick_eval()
                    uniq = len(set(hyps)); top1 = max([hyps.count(h) for h in set(hyps)], default=0)
                    print(f"  >> step {gstep} F2LLM SER={ser:.4f}  CTC-CER={ctc_cer:.4f}  [div: {uniq} unique, top1 {top1}]", flush=True)
                    with open(EVAL_HISTORY, 'a') as f:
                        f.write(json.dumps({'step': gstep, 'subset': 'h173', 'ser': round(ser, 4),
                                            'ctc_cer_ns': round(ctc_cer, 4),
                                            'n_unique': uniq, 'top1_share': round(top1/len(hyps), 3),
                                            'sims': sims, 'hyps': hyps, 'refs': refs}) + '\n')
                    if ser < best_ser:
                        best_ser = ser
                        # Free every transient before disk write (saves ~1-2GB peak)
                        _free_f2llm()
                        import gc; gc.collect(); torch.cuda.empty_cache()
                        model.save_pretrained(f'{OUTPUT_DIR}/checkpoints/step_{gstep}/lora')
                        torch.save({'adapter_state_dict': adapter.state_dict()},
                                   f'{OUTPUT_DIR}/checkpoints/step_{gstep}/adapter.pt')
                        gc.collect(); torch.cuda.empty_cache()
                        print(f"  >> saved best (step {gstep} SER={ser:.4f})", flush=True)
                    if ctc_cer > 0.65:  # safety check: front-end broken
                        print(f"  !! front-end broken (CTC-CER {ctc_cer:.3f} > 0.65), stopping")
                        return
                    adapter.train(); model.train()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4, help='LoRA lr')
    ap.add_argument('--adapter_lr', type=float, default=1e-5, help='adapter lr (v5 lesson: keep low)')
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=0)
    main(ap.parse_args())
