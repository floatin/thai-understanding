"""
Stage 2 v4: v3 + EOS in training targets (the stop-control fix).

Root cause found in v3 postmortem: training answers never included an EOS
token, so the model never learned when to stop; greedy generation ran to
max_new_tokens, repeating the (often correct) phrase. Evidence: step-328 best
outputs had CORRECT content with phrase repetition (sim 0.76-0.79).

Changes from v3 (everything else identical, single variable = EOS):
  - collate_batch: answer ids = ref tokens + [eos], truncated to 64 WITH eos kept
  - --init_from: continue from a v3 checkpoint dir (adapter.pt + lora/)
  - output paths v4, eval_every default 100 (200-granularity missed v3's
    end-of-run improvement)

Changes from v2 lineage (v3, unchanged here):
  1. LLM: Qwen3-4B -> /data/workspace/models/Typhoon2.5-qwen3-4b
     (same Qwen3ForCausalLM arch, hidden 2560, identical tokenizer vocab 151669)
  2. Quick-eval embedding: bge-m3-Thai (CPU) -> F2LLM-v2-4B (GPU during eval)
     F2LLM-v2-4B: multilingual embedder (Thai supported), last-token pooling,
     symmetric-task usage = no instruction prefix on either side (per model card).
     SER definition unchanged: SER = 1 - mean(cosine). Absolute scale differs
     from bge-m3; threshold meaning re-calibrated in Phase 1 gate.
  3. Guardrail asserts: train split must be disjoint from BOTH eval subsets
     (GT_human=h173, Silver_top20=1284) — AGENTS.md §4.6 leakage check.
  4. Eval history: per-item sims+hyps appended to eval_history.jsonl
     (rule: always save raw predictions, not just aggregate metrics).
  5. --max_steps smoke flag: stop after N optimizer steps (runs one quick_eval).

Sequence structure (unchanged, REPLACE mode per paper U-Align):
  [speech_embeds (T')] + [text_prompt_embeds (P)] + [text_answer_embeds (A)]
  Labels: [-100 (T'+P)] + [answer_ids (A)]

Pipeline:
  audio → XLSR-Thai (frozen) → Adapter (trainable, init from Adapter B)
       → 2560-dim embeddings
       → REPLACE first T' positions of [prompt; answer] embeddings
       → typhoon2.5-qwen3-4b + LoRA (trainable, attention layers)
       → cross-entropy on answer tokens only

Train on:  5159 PTT silver (ptt_train_split.json, verified disjoint from eval)
Quick eval: h173 (GT_human) every eval_every steps
At best checkpoint: full val pool (h173 + Silver_top20 1284) SER for
                    subset-consistency monitoring (AGENTS.md §4.6 #8).
Target:    SER < 0.10 on h173 (definition: 1 - mean F2LLM cosine)
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import json
import io
import sys
import time
import random
import re
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai, extract_features
from train_u_align_v2 import UAlignAdapter


# ============ Paths ============
ADAPTER_B = '/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_best.pt'
XLSR_CKPT = '/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt'
LLM_PATH = '/data/workspace/models/Typhoon2.5-qwen3-4b'
EMBED_MODEL = '/data/workspace/models/F2LLM-v2-4B'
TRAIN_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_train_split.json'
EVAL_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_split.json'
OUTPUT_DIR = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v4'
RESUME_FILE = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v4_resume.json'
EVAL_HISTORY = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_v4/eval_history.jsonl'

# Prompt - business context + hotwords + correction rules (identical to v2 — do not change)
GEN_PROMPT = """คุณคือผู้ช่วยถอดเสียงภาษาไทย เสียงนี้มาจากเจ้าหน้าที่รักษาความปลอดภัยในอาคารใช้วิทยุสื่อสารในการทำงานประจำวัน เช่น รายงานประตู ตรวจตรา แจ้งเหตุ ตรวจสอบอุปกรณ์ ส่งเวร

โปรดถอดเสียงเป็นข้อความภาษาไทยที่ถูกต้อง โดย:
- คงคำลงท้ายสุภาพไว้ (ค่ะ/ครับ/นะ/คะ)
- ตัวเลขคงไว้ในรูปแบบที่อ่าน
- ไม่สรุป ไม่ตัดทอน ไม่เพิ่มข้อความที่ไม่มีในเสียง

【คำสำคัญ】
ตำแหน่ง: หัวหน้า, รองหัวหน้า, ผู้ควบคุม, เจ้าหน้าที่
อุปกรณ์: วิทยุสื่อสาร, ประตู, กล้อง, สัญญาณ, สัญญาณเตือน, กล้องวงจรปิด
สถานที่: ประตูใหญ่, ประตูหลัง, ประตูข้าง, ลานจอดรถใต้ดิน, โถงชั้น 1, ดาดฟ้า, รั้ว
ตัวเลข: หมายเลขห้อง, รหัสเวลา, หมายเลขช่องจอด
การกระทำ: ตรวจตรา, ส่งเวร, ส่งมอบงาน, รายงาน, ขออนุญาต, ยืนยัน

【กฎการแก้ไข】
- เช็ด + บริบทอุปกรณ์ → เช็ค
- "สิบหกห้า" + บริบทเวลา → "16:05"
- ตัวอักษรซ้ำต่อเนื่องเกิน 3 ครั้ง → บีบให้เหลือ 1
- คงคำลงท้ายสุภาพ ค่ะ/ครับ/นะคะ/นะครับ/คะ ไว้

ถอดเสียงเป็นข้อความ:"""


# ============ SER (continuous: 1 - mean cosine) ============
def compute_ser_continuous(cosine_sims):
    sims = [s for s in cosine_sims if s is not None]
    if not sims:
        return None
    return 1.0 - float(np.mean(sims))


def compute_embed_sim_batch(embed_model, refs, hyps):
    embs_r = embed_model.encode(refs, normalize_embeddings=True, show_progress_bar=False, batch_size=16, convert_to_numpy=True)
    embs_h = embed_model.encode(hyps, normalize_embeddings=True, show_progress_bar=False, batch_size=16, convert_to_numpy=True)
    sims = (embs_r * embs_h).sum(axis=-1)
    return sims.tolist()


# ============ Dataset ============
MAX_AUDIO_SECONDS = 10
SAMPLE_RATE = 16000


def load_audio(wav_path):
    audio, sr = sf.read(wav_path, dtype='float32')
    if sr != SAMPLE_RATE:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * SAMPLE_RATE / sr))
    max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    if len(audio) < 3200:
        audio = np.pad(audio, (0, 3200 - len(audio)))
    return audio


class PTTDataset(Dataset):
    def __init__(self, samples, max_text_tokens=64):
        self.samples = samples
        self.max_text_tokens = max_text_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'wav_path': s['wav_path'],
            'ref': s['ref'],
            'id': s.get('id', str(idx)),
        }


def collate_batch(batch, encoder, adapter, tokenizer, base_model):
    """REPLACE mode: speech_embeds replace first T' positions of text_embeddings.

    Final sequence: [speech (T)] + [prompt (P)] + [answer (A)]
    Labels:        [-100 (T)] + [-100 (P)] + [answer_ids (A)]
    """
    B = len(batch)
    inputs_embeds_list = []
    labels_list = []
    speech_lens = []

    # Pre-tokenize prompt (shared)
    prompt_ids = tokenizer(GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0]
    P = prompt_ids.shape[0]

    for b in batch:
        # 1. Tokenize answer + EOS (v4 fix: teach stop control; eos kept inside the 64 cap)
        ans_ids = tokenizer(b['ref'], return_tensors='pt', add_special_tokens=False).input_ids[0]
        ans_ids = torch.cat([ans_ids, torch.tensor([tokenizer.eos_token_id])])[:64]
        A = len(ans_ids)

        # 2. Get text embeddings for prompt + answer (no speech in input_ids)
        text_ids = torch.cat([prompt_ids, ans_ids]).cuda()
        text_embeds = base_model.get_input_embeddings()(text_ids).to(torch.bfloat16)  # (P+A, 2560)

        # 3. Get speech features
        audio = load_audio(b['wav_path'])
        audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
        length = torch.tensor([audio_t.shape[1]]).cuda()
        with torch.no_grad():
            feats, _ = extract_features(encoder, audio_t, length)
        feats_b = feats.to(torch.bfloat16)
        se = adapter(feats_b)  # (1, T, 2560)
        T = se.shape[1]
        speech_lens.append(T)

        # 4. REPLACE mode: speech REPLACES first T positions of text embeddings
        full = torch.cat([se.squeeze(0), text_embeds], dim=0)  # (T+P+A, 2560)
        inputs_embeds_list.append(full)

        # 5. Labels: -100 for speech + prompt positions, real ids for answer
        labels = torch.full((T + P + A,), -100, dtype=torch.long, device=full.device)
        labels[T + P:] = ans_ids.cuda()  # answer positions get real ids
        labels_list.append(labels)

    # Right-pad to max length in batch
    max_len = max(e.shape[0] for e in inputs_embeds_list)
    final_embeds = torch.zeros(B, max_len, 2560, dtype=torch.bfloat16, device='cuda')
    final_labels = torch.full((B, max_len), -100, dtype=torch.long, device='cuda')
    final_mask = torch.zeros(B, max_len, dtype=torch.long, device='cuda')
    for i, (e, l) in enumerate(zip(inputs_embeds_list, labels_list)):
        T = e.shape[0]
        final_embeds[i, :T] = e
        final_labels[i, :T] = l
        final_mask[i, :T] = 1

    return {
        'inputs_embeds': final_embeds,
        'labels': final_labels,
        'attention_mask': final_mask,
        'refs': [b['ref'] for b in batch],
        'ids': [b['id'] for b in batch],
    }


def strip_think(text):
    """Guard: typhoon2.5 is a Qwen3 hybrid-thinking model; without a chat
    template it may still emit a <think> block that would silently corrupt
    hyps (and every downstream metric). Remove the block if present."""
    t = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    return t.replace('<think>', '').replace('</think>', '').strip()


# ============ Training & Eval ============
def train(args):
    print("=== Stage 2 v4: REPLACE mode + typhoon2.5 + EOS targets ===\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR + '/checkpoints', exist_ok=True)

    print(f"[1/6] Loading data splits + leakage asserts...")
    with open(TRAIN_SPLIT) as f:
        train_data = json.load(f)['samples']
    with open(EVAL_SPLIT) as f:
        eval_data_full = json.load(f)['samples']
    eval_data_h173 = [s for s in eval_data_full if s.get('subset') == 'GT_human']
    eval_data_silver = [s for s in eval_data_full if s.get('subset') == 'Silver_top20']
    assert eval_data_h173 and eval_data_silver, "eval subsets missing"

    # Guardrail (AGENTS.md §4.6 #1): training must not see any eval wav
    train_wavs = {Path(s['wav_path']).name for s in train_data}
    for name, subset in (('GT_human(h173)', eval_data_h173), ('Silver_top20', eval_data_silver)):
        sub_wavs = {Path(s['wav_path']).name for s in subset}
        leak = train_wavs & sub_wavs
        assert not leak, f"LEAK: {len(leak)} train wavs also in eval {name}, e.g. {sorted(leak)[:3]}"
    print(f"  train: {len(train_data)}, eval h173: {len(eval_data_h173)}, eval silver: {len(eval_data_silver)}  [disjoint OK]")

    # Resume
    resume = None
    if Path(RESUME_FILE).exists():
        try:
            with open(RESUME_FILE) as f:
                resume = json.load(f)
            print(f"  [Resume] from step {resume.get('step', 0)}, last SER={resume.get('ser', 'N/A')}")
        except Exception as e:
            print(f"  [Resume] load failed: {e}")

    print(f"\n[2/6] Loading XLSR-Thai encoder...")
    encoder, _ = load_xlsr_thai(XLSR_CKPT, device='cuda')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"\n[3/6] Loading Adapter B (Stage 1, 10-ep InfoNCE+DTW)...")
    ckpt = torch.load(ADAPTER_B, map_location='cpu', weights_only=False)
    adapter = UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    for p in adapter.parameters():
        p.requires_grad = True

    print(f"\n[4/6] Loading typhoon2.5-qwen3-4b + LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH, dtype=torch.bfloat16, device_map='cuda',
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    if args.init_from:
        print(f"  [init] continuing from {args.init_from}")
        model = PeftModel.from_pretrained(base_model, f'{args.init_from}/lora', is_trainable=True)
        trained_ckpt = torch.load(f'{args.init_from}/adapter.pt', map_location='cpu', weights_only=False)
        adapter.load_state_dict(trained_ckpt['adapter_state_dict'])
    else:
        model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    trainable_params = [p for p in adapter.parameters() if p.requires_grad] + \
                       [p for n, p in model.named_parameters() if 'lora_' in n]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    print(f"  total trainable: {sum(p.numel() for p in trainable_params):,}")

    print(f"\n[5/6] Loading F2LLM-v2-4B judge (CPU resident, GPU during eval)...")
    embed_model = SentenceTransformer(EMBED_MODEL, device='cpu', model_kwargs={'torch_dtype': torch.bfloat16})

    print(f"\n[6/6] Building dataset...")
    train_ds = PTTDataset(train_data)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_batch(b, encoder, adapter, tokenizer, base_model),
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    print(f"  batches/epoch: {len(train_loader)}, total opt steps: {total_steps}")

    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = int(0.05 * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)
    start_step = resume.get('step', 0) if resume else 0

    def run_eval_and_log(step, data, subset_name):
        ser, sims, refs, hyps = quick_eval(adapter, tokenizer, base_model, encoder, embed_model, data)
        # diversity guard: aggregate SER/CER cannot see mode collapse to modal
        # phrases (v3/v4 lesson) — always log unique-output stats
        uniq = len(set(hyps))
        top1 = max([hyps.count(h) for h in set(hyps)], default=0)
        share = top1 / max(1, len(hyps))
        print(f"  >> step {step} {subset_name} SER = {ser:.4f} (avg sim = {1-ser:.4f})  "
              f"[diversity: {uniq} unique, top1 {top1} ({share:.0%})]", flush=True)
        with open(EVAL_HISTORY, 'a') as f:
            f.write(json.dumps({'step': step, 'subset': subset_name, 'ser': ser,
                                'n_unique': uniq, 'top1_share': round(share, 3),
                                'sims': sims, 'hyps': hyps, 'refs': refs}) + '\n')
        return ser, hyps

    print(f"\n=== Training (replace mode, resume from step {start_step}) ===")
    model.train()
    adapter.train()
    global_step = start_step
    accumulated_loss = 0.0
    optimizer.zero_grad()
    t_start = time.time()
    best_ser = resume.get('ser', 1.0) if resume else 1.0

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= total_steps:
                break
            opt_step = global_step // args.grad_accum
            if opt_step >= total_steps:
                break

            outputs = model(
                inputs_embeds=batch['inputs_embeds'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += args.grad_accum

                if global_step % 20 == 0:
                    elapsed = time.time() - t_start
                    rate = (global_step - start_step) / elapsed if elapsed > 0 else 0
                    eta = (total_steps - global_step) / rate if rate > 0 else 0
                    print(f"  step {global_step}/{total_steps} loss={accumulated_loss:.4f} {elapsed:.0f}s ETA {eta:.0f}s", flush=True)
                accumulated_loss = 0.0

                smoke_stop = args.max_steps and global_step >= args.max_steps
                # next_eval cursor: global_step advances in grad_accum units, so
                # `global_step % eval_every` silently skips non-multiple targets
                if 'next_eval' not in dir():
                    next_eval = args.eval_every
                if smoke_stop or global_step >= next_eval:
                    next_eval += args.eval_every
                    ser, hyps = run_eval_and_log(global_step, eval_data_h173, 'h173')
                    Path(RESUME_FILE).write_text(json.dumps({'step': global_step, 'ser': ser}))
                    if ser < best_ser:
                        best_ser = ser
                        save_checkpoint(model, adapter, tokenizer, global_step, ser)
                        print(f"  >> step {global_step} saved (best h173 SER={ser:.4f})")
                    if ser < args.target_ser:
                        print(f"  >> step {global_step} reached target SER < {args.target_ser}, stopping")
                        return
                    if smoke_stop:
                        print(f"  [smoke] max_steps={args.max_steps} reached, exiting after eval")
                        return

        if global_step >= total_steps:
            break

    save_checkpoint(model, adapter, tokenizer, global_step, best_ser)
    # Final: full val pool silver subset (subset-consistency check vs h173, §4.6 #8)
    ser_final_silver, _ = run_eval_and_log(global_step, eval_data_silver, 'silver1284')
    print(f"\n=== Training done. Final step {global_step}. Best h173 SER={best_ser:.4f}, "
          f"final silver1284 SER={ser_final_silver:.4f} ===")


def quick_eval(adapter, tokenizer, base_model, encoder, embed_model, eval_data):
    """REPLACE mode inference: [speech | prompt] → generate. F2LLM on GPU."""
    adapter.eval()
    base_model.config.use_cache = True
    print(f"  [Eval] running inference on {len(eval_data)} samples...")

    # Pre-tokenize prompt
    prompt_ids = tokenizer(GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0].cuda()
    prompt_embeds = base_model.get_input_embeddings()(prompt_ids).to(torch.bfloat16)  # (P, 2560)

    refs, hyps = [], []
    t0 = time.time()
    for s in eval_data:
        try:
            audio = load_audio(s['wav_path'])
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            with torch.no_grad():
                feats, _ = extract_features(encoder, audio_t, length)
                se = adapter(feats.to(torch.bfloat16))  # (1, T, 2560)
                # REPLACE: [speech | prompt]
                inputs_embeds = torch.cat([se.squeeze(0), prompt_embeds], dim=0).unsqueeze(0)
                out = base_model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            hyp = strip_think(tokenizer.decode(out[0], skip_special_tokens=True).strip())
        except Exception as e:
            hyp = ""
        refs.append(s['ref'])
        hyps.append(hyp)
    print(f"  [Eval] inference {time.time()-t0:.0f}s. Computing F2LLM embeddings...")

    torch.cuda.empty_cache()
    embed_model = embed_model.to('cuda')
    sims = compute_embed_sim_batch(embed_model, refs, hyps)
    ser = compute_ser_continuous(sims)
    embed_model = embed_model.to('cpu')
    torch.cuda.empty_cache()
    base_model.config.use_cache = False
    adapter.train()
    return ser, sims, refs, hyps


def save_checkpoint(model, adapter, tokenizer, step, ser):
    Path(f'{OUTPUT_DIR}/checkpoints/step_{step}').mkdir(parents=True, exist_ok=True)
    torch.save({
        'adapter_state_dict': adapter.state_dict(),
        'encoder_dim': 1024, 'llm_dim': 2560, 'downsample': 2,
        'step': step, 'ser': ser,
    }, f'{OUTPUT_DIR}/checkpoints/step_{step}/adapter.pt')
    model.save_pretrained(f'{OUTPUT_DIR}/checkpoints/step_{step}/lora')
    print(f"  saved to {OUTPUT_DIR}/checkpoints/step_{step}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora_rank', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--target_ser', type=float, default=0.10)
    ap.add_argument('--max_steps', type=int, default=0, help='smoke test: stop after N optimizer steps (0=full)')
    ap.add_argument('--init_from', type=str, default='', help='checkpoint dir to continue from')
    args = ap.parse_args()
    train(args)
