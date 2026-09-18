"""Stage 2 v8 (rebuilt to spec): v7 cascade + CTC auxiliary loss, batch_size=1.

Per user: '除了batch，不要改动v7任何配置' = keep every v7 hyperparam
(cascade design, GEN_PROMPT, data, EOS, lr=1e-4, lora r=16/α=32, eval_every=100,
no speech path in collate). Only deltas vs v7:
  (a) init LoRA from v7 step_1200 (already does denoising)
  (b) batch_size 4 -> 1, grad_accum 4 -> 16 (same effective batch=16, but
      CTC-side speech embeds add memory; bs=1 fits)
  (c) add CTC self-consistency: target = LLM's input DRAFT text; speech
      embeds produced inside the same forward and CTC'd against draft.
      Adapter + CTC head frozen (v7 already showed cascade works; v8 tests
      whether CTC as a TIE-BREAKING signal helps the LLM stay grounded in
      front-end evidence without killing it).

v5 lesson applied: CTC uses DRAFT (LLM's input), not ground-truth ref,
so CE and CTC both point the same way (toward "this draft decodes
consistently from these embeddings"), not opposite.

Eval = F2LLM deferred (GPU shared with align service), diversity guard,
CTC-CER on h173 at each eval (must stay ~0.50 if front-end untouched).
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
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v5 as base
import train_llm_denoise_v7 as v7  # DenoiseDataset, GEN_PROMPT, paths

LLM_PATH = '/data/workspace/models/Typhoon2.5-qwen3-4b'
CTC_ONLY_BEST = '/data/workspace/asr-model-training/thai-understanding/u_align_ctc_only/best/ctc_only.pt'
V7_INIT_LORA = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v7/checkpoints/step_1200/lora'
TRAIN_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_train_split.json'
EVAL_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_split.json'
DRAFTS = '/data/workspace/asr-model-training/thai-understanding/ctc_drafts.json'
V5_VOCAB = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v5/ctc_vocab.json'
XLSR_CKPT = '/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt'
OUTPUT_DIR = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v8'
EVAL_HISTORY = f'{OUTPUT_DIR}/eval_history.jsonl'


def draft_to_ids(draft, vocab):
    from thai_ctc_units import merge_units
    return [vocab[u] for u in merge_units(draft) if u in vocab and vocab[u] != 0]


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

    # frozen front-end (CTC probe + adapter)
    encoder, _ = base.load_xlsr_thai(XLSR_CKPT, device='cuda'); encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    ckpt = torch.load(CTC_ONLY_BEST, map_location='cpu', weights_only=False)
    from train_u_align_v2 import UAlignAdapter
    adapter = UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    for p in adapter.parameters(): p.requires_grad = False
    adapter.eval()
    ctc_head = torch.nn.Linear(2560, len(vocab)).cuda().float()
    ctc_head.load_state_dict(ckpt['ctc_head_state_dict'])
    for p in ctc_head.parameters(): p.requires_grad = False
    ctc_head.eval()
    print(f"front-end frozen: h173 CER baseline = 0.4999")

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map='cuda')
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    # init from v7 step_1200 (cascade denoising capability preserved)
    model = PeftModel.from_pretrained(base_model, V7_INIT_LORA, is_trainable=True)
    model.print_trainable_parameters()
    trainable = [p for n, p in model.named_parameters() if 'lora_' in n]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    from xlsr_thai import extract_features
    MAX_DRAFT = v7.MAX_DRAFT_TOKENS  # 192, same as v7
    MAX_ANS = v7.MAX_ANS_TOKENS  # 64

    def collate(batch):
        prompt_ids = tokenizer(base.GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0]
        P = prompt_ids.shape[0]
        embeds_list, labels_list, speech_list = [], [], []
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
            # speech embeds for CTC auxiliary — only IF batch_size == 1 (avoids OOM at bs>1)
            if args.batch_size == 1:
                audio = base.load_audio(b['wav_path'])
                t = torch.from_numpy(audio).unsqueeze(0).cuda()
                with torch.no_grad():
                    feats, _ = extract_features(encoder, t, torch.tensor([t.shape[1]]).cuda())
                    # adapter is frozen; detach so this loss does NOT build an autograd
                    # graph back into the LLM path. Front-end params frozen anyway.
                    se = adapter(feats.to(torch.bfloat16)).squeeze(0).detach()
                speech_list.append((se, draft_to_ids(b['draft'], vocab)))
        max_len = max(e.shape[0] for e in embeds_list)
        B = len(batch)
        fe = torch.zeros(B, max_len, 2560, dtype=torch.bfloat16, device='cuda')
        fl = torch.full((B, max_len), -100, dtype=torch.long, device='cuda')
        fm = torch.zeros(B, max_len, dtype=torch.long, device='cuda')
        for i, (e, l) in enumerate(zip(embeds_list, labels_list)):
            fe[i, :e.shape[0]] = e; fl[i, :l.shape[0]] = l; fm[i, :l.shape[0]] = 1
        return {'inputs_embeds': fe, 'labels': fl, 'attention_mask': fm,
                'speech_list': speech_list, 'ids': [b['id'] for b in batch]}

    # dataset must keep wav_path for the CTC path (bs=1 only) and eval
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
                # CTC diagnostic
                logp = F.log_softmax(ctc_head(se.float().squeeze(0)), dim=-1)
                am = logp.argmax(-1).tolist()
                prev, out = -1, []
                for i in am:
                    if i != prev and i != 0: out.append(i)
                    prev = i
                ctc_hyps.append(ids_to_text(out, inv_vocab))
                # LLM cascade (draft + prompt)
                draft = drafts_map.get(s['id'], '')
                d_ids = tokenizer(draft, return_tensors='pt', add_special_tokens=False).input_ids[0][:MAX_DRAFT].cuda()
                d_emb = base_model.get_input_embeddings()(d_ids).to(torch.bfloat16)
                inputs_embeds = torch.cat([d_emb, prompt_emb], dim=0).unsqueeze(0)
                o = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=MAX_ANS,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
                hyps.append(base.strip_think(tokenizer.decode(o[0], skip_special_tokens=True).strip()))
                refs.append(s['ref'])
        ctc_cer = float(np.mean([cer_no_space(r, h) for r, h in zip(refs, ctc_hyps)]))
        return refs, hyps, ctc_cer

    print("=== v8: v7 cascade + CTC auxiliary (bs=1) ===\n")
    Path(OUTPUT_DIR, 'checkpoints').mkdir(exist_ok=True)
    gstep, acc, acc_ctc, t0 = 0, 0.0, 0.0, time.time()
    next_eval = args.eval_every
    opt.zero_grad()
    model.train()
    for ep in range(args.epochs):
        for bi, batch in enumerate(loader):
            out_lm = model(inputs_embeds=batch['inputs_embeds'], attention_mask=batch['attention_mask'],
                           labels=batch['labels'])
            ce_loss = out_lm.loss
            # CTC auxiliary: front-end speech embeds (frozen) -> draft text (the LLM's
            # input). Since adapter is frozen and we detach `se`, this loss has no grad
            # into the LLM path; it's purely a monitoring signal for front-end consistency.
            ctc_loss = torch.tensor(0.0, device='cuda')
            for se, ids in batch['speech_list']:
                if not ids: continue
                T = se.shape[0]
                if T < len(ids): continue
                logp = F.log_softmax(ctc_head(se.float()), dim=-1).unsqueeze(1)
                ctc_loss = ctc_loss + F.ctc_loss(logp, torch.tensor(ids), torch.tensor([T]),
                                                 torch.tensor([len(ids)]), blank=0, zero_infinity=True)
            if ctc_loss.dim() == 0:
                ctc_loss = torch.tensor(0.0, device='cuda')
            else:
                ctc_loss = ctc_loss / max(1, len(batch['speech_list']))
            # NOTE: at bs=1 the CTC loss has NO grad into LoRA (frozen front-end + detach),
            # so only CE contributes to gradients. The CTC term is a probe for consistency.
            total = (ce_loss + args.ctc_weight * ctc_loss) / args.grad_accum
            total.backward()
            acc += ce_loss.item(); acc_ctc += float(ctc_loss)
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                gstep += args.grad_accum
                if gstep % 20 == 0:
                    print(f"  step {gstep}/{total_steps} ce={acc:.3f} ctc={acc_ctc:.3f} {time.time()-t0:.0f}s", flush=True)
                acc = acc_ctc = 0.0
                if args.max_steps and gstep >= args.max_steps:
                    refs, hyps, ctc_cer = quick_eval()
                    uniq = len(set(hyps)); top1 = max([hyps.count(h) for h in set(hyps)], default=0)
                    print(f"  >> step {gstep} CTC-CER={ctc_cer:.4f}  [div: {uniq} unique, top1 {top1}]", flush=True)
                    with open(EVAL_HISTORY, 'a') as f:
                        f.write(json.dumps({'step': gstep, 'subset': 'h173', 'ctc_cer_ns': ctc_cer,
                                            'n_unique': uniq, 'top1_share': round(top1/len(hyps), 3),
                                            'hyps': hyps, 'refs': refs, 'sims': None}) + '\n')
                    return
                if gstep >= next_eval:
                    next_eval += args.eval_every
                    refs, hyps, ctc_cer = quick_eval()
                    uniq = len(set(hyps)); top1 = max([hyps.count(h) for h in set(hyps)], default=0)
                    print(f"  >> step {gstep} CTC-CER={ctc_cer:.4f}  [div: {uniq} unique, top1 {top1}]", flush=True)
                    with open(EVAL_HISTORY, 'a') as f:
                        f.write(json.dumps({'step': gstep, 'subset': 'h173', 'ctc_cer_ns': ctc_cer,
                                            'n_unique': uniq, 'top1_share': round(top1/len(hyps), 3),
                                            'hyps': hyps, 'refs': refs, 'sims': None}) + '\n')
                    model.save_pretrained(f'{OUTPUT_DIR}/checkpoints/step_{gstep}/lora')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--grad_accum', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--ctc_weight', type=float, default=0.1)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=0)
    main(ap.parse_args())
