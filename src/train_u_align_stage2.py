"""
Stage 2: End-to-end fine-tuning of Adapter + LoRA on Qwen3-4B.

Pipeline:
  audio → XLSR-Thai (frozen) → Adapter (trainable, init from Adapter B)
       → 2560-dim embeddings
       → concat with prompt_embeds + answer_embeds
       → Qwen3-4B + LoRA (trainable, attention layers)
       → cross-entropy on answer tokens

Start from: Adapter B (10-ep InfoNCE+DTW on Thai-SUP)
Train on:  5159 PTT silver (excluding h173 to prevent eval leakage)
Quick eval: h173 (173 gold samples) with bge-m3-Thai cosine similarity
Target:    SER_continuous < 0.10 on h173

SER formula (continuous, aligned with bge-m3-Thai judge):
  SER = 1 - mean(cosine(ref_embed, hyp_embed))
  where cosine is over normalized 1024-dim vectors

Usage:
  python train_u_align_stage2.py --epochs 1 --batch_size 4 --grad_accum 4
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
LLM_PATH = '/data/workspace/asr-model-training/thai-understanding/Qwen3-4B'
EMBED_MODEL = '/data/workspace/models/bge-m3-Thai'
TRAIN_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_train_split.json'
EVAL_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_split.json'
OUTPUT_DIR = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2'
RESUME_FILE = '/data/workspace/asr-model-training/thai-understanding/u_align_stage2_resume.json'

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

[เสียง]"""


# ============ SER (continuous, aligned with bge-m3-Thai) ============
def compute_ser_continuous(cosine_sims):
    """SER = 1 - mean(cosine_similarity). Range [0, 1]. Aligned with bge-m3-Thai judge."""
    sims = [s for s in cosine_sims if s is not None]
    if not sims:
        return None
    return 1.0 - float(np.mean(sims))


def compute_embed_sim_batch(embed_model, refs, hyps):
    """Compute pairwise cosine similarity (refs[i], hyps[i]) using bge-m3-Thai."""
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
    """For each item: load audio, compute feats, project via adapter, build inputs_embeds with prompt + answer."""
    B = len(batch)
    prompt_ids = tokenizer(GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0]
    P = prompt_ids.shape[0]
    prompt_embeds = base_model.get_input_embeddings()(prompt_ids.cuda()).to(torch.bfloat16)  # (P, 2560)

    inputs_embeds_list = []
    labels_list = []
    attention_mask_list = []
    speech_lens = []
    for b in batch:
        audio = load_audio(b['wav_path'])
        audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
        length = torch.tensor([audio_t.shape[1]]).cuda()
        with torch.no_grad():
            feats, _ = extract_features(encoder, audio_t, length)
        feats_b = feats.to(torch.bfloat16)
        se = adapter(feats_b)  # (1, T, 2560)
        T = se.shape[1]
        speech_lens.append(T)

        ans_ids = tokenizer(b['ref'], return_tensors='pt', add_special_tokens=False).input_ids[0][:64]
        ans_embeds = base_model.get_input_embeddings()(ans_ids.cuda()).to(torch.bfloat16)  # (A, 2560)

        full = torch.cat([prompt_embeds, se.squeeze(0), ans_embeds], dim=0)  # (P+T+A, 2560)
        inputs_embeds_list.append(full)

        # Labels: -100 for prompt + speech, real ids for answer
        labels = torch.full((full.shape[0],), -100, dtype=torch.long, device=full.device)
        labels[P + T: P + T + len(ans_ids)] = ans_ids.cuda()
        labels_list.append(labels)
        attention_mask_list.append(torch.ones(full.shape[0], dtype=torch.long, device=full.device))

    # Stack with right-padding to max len in batch
    max_len = max(e.shape[0] for e in inputs_embeds_list)
    final_embeds = torch.zeros(B, max_len, 2560, dtype=torch.bfloat16, device='cuda')
    final_labels = torch.full((B, max_len), -100, dtype=torch.long, device='cuda')
    final_mask = torch.zeros(B, max_len, dtype=torch.long, device='cuda')
    for i, (e, l, m) in enumerate(zip(inputs_embeds_list, labels_list, attention_mask_list)):
        T = e.shape[0]
        final_embeds[i, :T] = e
        final_labels[i, :T] = l
        final_mask[i, :T] = m

    return {
        'inputs_embeds': final_embeds,
        'labels': final_labels,
        'attention_mask': final_mask,
        'refs': [b['ref'] for b in batch],
        'ids': [b['id'] for b in batch],
    }


# ============ Training & Eval ============
def train(args):
    print("=== Stage 2: End-to-end fine-tuning (Adapter B + LoRA, bge-m3-Thai judge) ===\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR + '/checkpoints', exist_ok=True)

    print(f"[1/6] Loading data splits...")
    with open(TRAIN_SPLIT) as f:
        train_data = json.load(f)['samples']
    with open(EVAL_SPLIT) as f:
        eval_data_full = json.load(f)['samples']
    eval_data = [s for s in eval_data_full if s.get('subset') == 'GT_human']
    print(f"  train: {len(train_data)}, eval (h173): {len(eval_data)}")

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

    print(f"\n[3/6] Loading Adapter B (10-ep InfoNCE+DTW)...")
    ckpt = torch.load(ADAPTER_B, map_location='cpu', weights_only=False)
    adapter = UAlignAdapter(
        encoder_dim=ckpt.get('encoder_dim', 1024),
        llm_dim=ckpt.get('llm_dim', 2560),
        downsample=ckpt.get('downsample', 2),
    ).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    for p in adapter.parameters():
        p.requires_grad = True

    print(f"\n[4/6] Loading Qwen3-4B + LoRA...")
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
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Optimizer
    trainable_params = [p for p in adapter.parameters() if p.requires_grad] + \
                       [p for n, p in model.named_parameters() if 'lora_' in n]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    print(f"  total trainable: {sum(p.numel() for p in trainable_params):,}")

    # Embed model for quick eval
    print(f"\n[5/6] Loading bge-m3-Thai judge...")
    embed_model = SentenceTransformer(EMBED_MODEL, device='cpu')  # CPU to save GPU mem; eval moves to GPU
    print(f"  embed model mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Dataset
    train_ds = PTTDataset(train_data)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_batch(b, encoder, adapter, tokenizer, base_model),
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    print(f"\n[6/6] Training setup: {len(train_loader)} batches/epoch, {total_steps} opt steps total")

    # LR schedule
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = int(0.05 * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)
    start_step = resume.get('step', 0) if resume else 0

    # Training loop
    print(f"\n=== Training (resume from step {start_step}) ===")
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

                if global_step % args.eval_every == 0:
                    ser, sims = quick_eval(adapter, tokenizer, base_model, encoder, embed_model, eval_data)
                    print(f"  >> step {global_step} h173 SER = {ser:.4f} (avg sim = {1-ser:.4f})", flush=True)
                    Path(RESUME_FILE).write_text(json.dumps({'step': global_step, 'ser': ser, 'sims': sims[:10]}))
                    if ser < best_ser:
                        best_ser = ser
                        save_checkpoint(model, adapter, tokenizer, global_step, ser)
                        print(f"  >> step {global_step} saved (best SER={ser:.4f})")
                    if ser < 0.10:
                        print(f"  >> step {global_step} reached target SER < 0.10, stopping")
                        return

        if global_step >= total_steps:
            break

    save_checkpoint(model, adapter, tokenizer, global_step, ser if 'ser' in dir() else None)
    print(f"\n=== Training done. Final step {global_step}. Best SER={best_ser:.4f} ===")


def quick_eval(adapter, tokenizer, base_model, encoder, embed_model, eval_data):
    """Run inference on h173, then compute bge-m3-Thai cosine sim. Return (SER, sims)."""
    adapter.eval()
    embed_model = embed_model.to('cuda')  # Move to GPU for fast encoding
    # Re-enable cache for inference (was disabled for grad checkpointing)
    base_model.config.use_cache = True
    print(f"  [Eval] running inference on {len(eval_data)} samples...")

    prompt_ids = tokenizer(GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0].cuda()
    prompt_embeds = base_model.get_input_embeddings()(prompt_ids).to(torch.bfloat16)

    refs, hyps = [], []
    t0 = time.time()
    for s in eval_data:
        try:
            audio = load_audio(s['wav_path'])
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            with torch.no_grad():
                feats, _ = extract_features(encoder, audio_t, length)
                speech_embeds = adapter(feats.to(torch.bfloat16))
                inputs_embeds = torch.cat([prompt_embeds.unsqueeze(0), speech_embeds], dim=1)
                out = base_model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            hyp = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            hyp = ""
        refs.append(s['ref'])
        hyps.append(hyp)
    print(f"  [Eval] inference {time.time()-t0:.0f}s. Computing embeddings...")

    sims = compute_embed_sim_batch(embed_model, refs, hyps)
    ser = compute_ser_continuous(sims)
    embed_model = embed_model.to('cpu')  # Free GPU
    torch.cuda.empty_cache()
    # Re-disable cache for next training step
    base_model.config.use_cache = False
    adapter.train()
    return ser, sims


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
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora_rank', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--eval_every', type=int, default=200)
    args = ap.parse_args()
    train(args)