"""Stage 2 v11: v7 recipe unchanged, re-run on v10 beam+LM drafts + expanded pool.

(v7 discriminating experiment): CTC-draft text cascade.

v6 showed the LLM cannot learn to READ dense speech embeddings (SER stuck
0.84-0.90 while the frozen front-end holds CER 0.4999). v7 bypasses embedding
reading entirely: the input's first segment is the CTC greedy-decoded DRAFT
as ordinary text tokens (from ctc_drafts.json), then GEN_PROMPT, then answer.

Sequence: [draft_tokens (D)] + [prompt (P)] + [answer+EOS (A)]
Labels:   [-100 × D+P]        + [answer ids]
vs v6, the ONLY change: first segment = text-token embeddings of the draft
instead of adapter speech embeddings. Same prompt, same data, same LoRA
hyperparams, same eval protocol (F2LLM deferred, diversity guard).

This directly tests: is CER~0.5 draft + business prompt + LoRA enough for the
LLM to transcribe? (Cascade architecture: XLSR→adapter→CTC→LLM-denoise.)
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v5 as base  # GEN_PROMPT, strip_think, paths convention

LLM_PATH = '/data/workspace/models/Typhoon2.5-qwen3-4b'
DRAFTS = '/data/workspace/asr-model-training/thai-understanding/ctc_drafts_v10.json'
TRAIN_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_train_split_v10.json'
EVAL_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_split.json'
OUTPUT_DIR = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v11'
EVAL_HISTORY = f'{OUTPUT_DIR}/eval_history.jsonl'
RESUME_FILE = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v11_resume.json'

MAX_DRAFT_TOKENS = 192
MAX_ANS_TOKENS = 64


class DenoiseDataset(Dataset):
    def __init__(self, samples, drafts):
        self.samples = samples
        self.drafts = drafts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {'draft': self.drafts.get(s['id'], ''), 'ref': s['ref'], 'id': s['id']}


def collate_batch(batch, tokenizer, embed_layer):
    prompt_ids = tokenizer(base.GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0]
    P = prompt_ids.shape[0]
    embeds_list, labels_list = [], []
    for b in batch:
        draft_ids = tokenizer(b['draft'], return_tensors='pt', add_special_tokens=False).input_ids[0][:MAX_DRAFT_TOKENS]
        ans_ids = tokenizer(b['ref'], return_tensors='pt', add_special_tokens=False).input_ids[0]
        ans_ids = torch.cat([ans_ids, torch.tensor([tokenizer.eos_token_id])])[:MAX_ANS_TOKENS]
        ids = torch.cat([draft_ids, prompt_ids, ans_ids]).cuda()
        emb = embed_layer(ids).to(torch.bfloat16)
        D, A = len(draft_ids), len(ans_ids)
        labels = torch.full((D + P + A,), -100, dtype=torch.long, device='cuda')
        labels[D + P:] = ans_ids.cuda()
        embeds_list.append(emb)
        labels_list.append(labels)
    max_len = max(e.shape[0] for e in embeds_list)
    B = len(batch)
    final_embeds = torch.zeros(B, max_len, 2560, dtype=torch.bfloat16, device='cuda')
    final_labels = torch.full((B, max_len), -100, dtype=torch.long, device='cuda')
    final_mask = torch.zeros(B, max_len, dtype=torch.long, device='cuda')
    for i, (e, l) in enumerate(zip(embeds_list, labels_list)):
        final_embeds[i, :e.shape[0]] = e
        final_labels[i, :l.shape[0]] = l
        final_mask[i, :l.shape[0]] = 1
    return {'inputs_embeds': final_embeds, 'labels': final_labels, 'attention_mask': final_mask,
            'refs': [b['ref'] for b in batch], 'drafts': [b['draft'] for b in batch]}


def quick_eval(model, tokenizer, embed_layer, data, drafts):
    model.eval()
    prompt_ids = tokenizer(base.GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0].cuda()
    prompt_embeds = embed_layer(prompt_ids).to(torch.bfloat16)
    refs, hyps = [], []
    for s in data:
        draft = drafts.get(s['id'], '')
        draft_ids = tokenizer(draft, return_tensors='pt', add_special_tokens=False).input_ids[0][:MAX_DRAFT_TOKENS].cuda()
        draft_embeds = embed_layer(draft_ids).to(torch.bfloat16)
        inputs_embeds = torch.cat([draft_embeds, prompt_embeds], dim=0).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=MAX_ANS_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        hyps.append(base.strip_think(tokenizer.decode(out[0], skip_special_tokens=True).strip()))
        refs.append(s['ref'])
    model.train()
    return refs, hyps


def run_eval_and_log(step, data, drafts, model, tokenizer, embed_layer):
    refs, hyps = quick_eval(model, tokenizer, embed_layer, data, drafts)
    uniq = len(set(hyps))
    top1 = max([hyps.count(h) for h in set(hyps)], default=0)
    share = top1 / max(1, len(hyps))
    print(f"  >> step {step} h173 SER = n/a (F2LLM deferred)  [diversity: {uniq} unique, top1 {top1} ({share:.0%})]", flush=True)
    with open(EVAL_HISTORY, 'a') as f:
        f.write(json.dumps({'step': step, 'subset': 'h173', 'ser': None,
                            'n_unique': uniq, 'top1_share': round(share, 3), 'ctc_cer_ns': None,
                            'sims': None, 'hyps': hyps, 'refs': refs}) + '\n')
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    model.save_pretrained(f'{OUTPUT_DIR}/checkpoints/step_{step}/lora')
    return hyps


def main(args):
    print("=== Stage 2 v7: CTC-draft text cascade (LLM denoise) ===\n")
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    drafts = json.load(open(DRAFTS))
    train_data = json.load(open(TRAIN_SPLIT))['samples']
    eval_full = json.load(open(EVAL_SPLIT))['samples']
    h173 = [s for s in eval_full if s.get('subset') == 'GT_human']
    assert not ({Path(s['wav_path']).name for s in train_data} &
                {Path(s['wav_path']).name for s in h173}), "LEAK"
    print(f"train: {len(train_data)}, eval h173: {len(h173)}, drafts: {len(drafts)} [disjoint OK]")

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map='cuda')
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    embed_layer = base_model.get_input_embeddings()
    model = get_peft_model(base_model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    trainable = [p for n, p in model.named_parameters() if 'lora_' in n]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    ds = DenoiseDataset(train_data, drafts)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_batch(b, tokenizer, embed_layer))
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    warmup = int(0.05 * total_steps)
    from torch.optim.lr_scheduler import LambdaLR
    sched = LambdaLR(opt, lambda s: s / max(1, warmup) if s < warmup else
                     0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup))))
    print(f"batches/epoch: {len(loader)}, total opt steps: {total_steps}")

    model.train()
    gstep, acc, t0 = 0, 0.0, time.time()
    next_eval = args.eval_every
    opt.zero_grad()
    for ep in range(args.epochs):
        for bi, batch in enumerate(loader):
            out = model(inputs_embeds=batch['inputs_embeds'], attention_mask=batch['attention_mask'],
                        labels=batch['labels'])
            (out.loss / args.grad_accum).backward()
            acc += out.loss.item()
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                gstep += args.grad_accum
                if gstep % 20 == 0:
                    print(f"  step {gstep}/{total_steps} loss={acc:.4f} {time.time()-t0:.0f}s", flush=True)
                acc = 0.0
                if args.max_steps and gstep >= args.max_steps:
                    run_eval_and_log(gstep, h173, drafts, model, tokenizer, embed_layer)
                    print("  [smoke] done"); return
                if gstep >= next_eval:
                    next_eval += args.eval_every
                    run_eval_and_log(gstep, h173, drafts, model, tokenizer, embed_layer)
    run_eval_and_log(gstep, h173, drafts, model, tokenizer, embed_layer)
    print(f"=== v7 done. steps={gstep} ===")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--grad_accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora_rank', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--max_steps', type=int, default=0)
    main(ap.parse_args())
