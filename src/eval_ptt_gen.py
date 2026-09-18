"""
Phase 1: Generate PTT-207 hyp with both adapters on GPU.

Adapter A: 3-epoch InfoNCE only
Adapter B: 10-epoch InfoNCE + cosine-DTW (best on Thai-SUP retrieval)

Output: JSON with (wav, ref, hyp_A, hyp_B) for Phase 2/3.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai, extract_features
from train_u_align_v2 import UAlignAdapter

# ============ Config ============
EVAL_SPLIT = '/data/workspace/asr-model-training/thai-understanding/ptt_eval_split.json'

WAV_DIR = '/data/workspace/asr-model-training/wav_sodexo'

ADAPTER_A = '/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_final_3ep.pt'
ADAPTER_B = '/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_best.pt'

XLSR_CKPT = '/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt'
LLM_PATH = '/data/workspace/asr-model-training/thai-understanding/Qwen3-4B'

# Generation prompt (from PTT_PROMPT_TEMPLATE.md §4 — business context + hotwords + rules)
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


def load_adapter(path, device='cuda'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    a = UAlignAdapter(
        encoder_dim=ckpt.get('encoder_dim', 1024),
        llm_dim=ckpt.get('llm_dim', 2560),
        downsample=ckpt.get('downsample', 2),
    ).to(device).to(torch.bfloat16)
    a.load_state_dict(ckpt['adapter_state_dict'])
    a.eval()
    return a, ckpt


def load_audio(wav_path, max_seconds=10, sr_target=16000):
    audio, sr = sf.read(wav_path, dtype='float32')
    if sr != sr_target:
        # naive resample if needed
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * sr_target / sr))
        sr = sr_target
    max_samples = max_seconds * sr
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    if len(audio) < 3200:  # torchaudio minimum
        audio = np.pad(audio, (0, 3200 - len(audio)))
    return audio, sr


def build_eval_set():
    """Load prebuilt eval split (h173 + top 20% strong_silver, wav-deduped)."""
    with open(EVAL_SPLIT) as f:
        d = json.load(f)
    samples = d['samples']
    return [{
        'wav_path': s['wav_path'],
        'ref': s['ref'],
        'subset': s['subset'],
        'id': s['id'],
        'source': s.get('source', ''),
    } for s in samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/data/workspace/asr-model-training/thai-understanding/ptt_eval_gen.json')
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0, help='limit samples for debugging')
    args = ap.parse_args()

    print("=== Phase 1: Generation on PTT-207 (all on GPU) ===\n")

    # Build eval set
    print("[1/4] Building eval set...")
    samples = build_eval_set()
    if args.limit:
        samples = samples[:args.limit]
    print(f"  {len(samples)} samples: GT={sum(1 for s in samples if s['subset']=='GT')}, Silver={sum(1 for s in samples if s['subset']=='Silver')}")

    # Verify wav files exist
    missing = [s['wav_path'] for s in samples if not os.path.exists(s['wav_path'])]
    if missing:
        print(f"  ⚠️ Missing {len(missing)} wav files, e.g.: {missing[0]}")
        samples = [s for s in samples if os.path.exists(s['wav_path'])]
        print(f"  Reduced to {len(samples)} samples")

    # Load encoder on GPU
    print(f"\n[2/4] Loading XLSR-Thai on GPU...")
    encoder, _ = load_xlsr_thai(XLSR_CKPT, device='cuda')
    encoder.eval()

    # Load adapters on GPU
    print(f"  Loading adapters on GPU...")
    adapter_a, ckpt_a = load_adapter(ADAPTER_A, device='cuda')
    adapter_b, ckpt_b = load_adapter(ADAPTER_B, device='cuda')

    # Load Qwen3-4B on CPU (GPU occupied by other tenants)
    print(f"  Loading Qwen3-4B on CPU (GPU has other tenants)...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map='cuda')
    llm.eval()
    embed_tokens = llm.get_input_embeddings()
    print(f"  GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Pre-compute prompt embeddings (same for all samples)
    prompt_ids = tokenizer(GEN_PROMPT, return_tensors="pt").input_ids.cuda()
    prompt_embeds = embed_tokens(prompt_ids).to(torch.bfloat16)

    def gen_all(adapter, name):
        print(f"\n[3/4] Generating with {name}...")
        results = []
        t0 = time.time()
        for i, s in enumerate(samples):
            audio, _ = load_audio(s['wav_path'])
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            with torch.no_grad():
                feats, _ = extract_features(encoder, audio_t, length)
                speech_embeds = adapter(feats.to(torch.bfloat16))
                inputs_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
                out = llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            hyp = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            results.append({
                **s,
                'hyp': hyp,
            })
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (len(samples) - i - 1) / (i + 1)
                print(f"  [{i+1}/{len(samples)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
        print(f"  Done in {time.time()-t0:.0f}s")
        return results

    res_a = gen_all(adapter_a, 'Adapter A (3-ep)')
    res_b = gen_all(adapter_b, 'Adapter B (10-ep InfoNCE+DTW)')

    # Save
    print(f"\n[4/4] Saving results to {args.out}...")
    Path(args.out).write_text(json.dumps({
        'prompt': GEN_PROMPT,
        'adapter_a': {'path': ADAPTER_A, 'final_epoch': ckpt_a.get('final_epoch', '?')},
        'adapter_b': {'path': ADAPTER_B, 'final_epoch': ckpt_b.get('epoch', '?')},
        'samples': [{**r, 'hyp_a': r.pop('hyp')} for r in res_a],
        'samples_b': [{**s, 'hyp_b': r['hyp']} for s, r in zip(samples, res_b)],
    }, indent=2, ensure_ascii=False))
    print(f"  Saved {len(res_a)} samples x 2 adapters")

    # Sample output
    print("\n=== First 3 samples (Adapter A vs B) ===")
    for s, a, b in list(zip(samples, res_a, res_b))[:3]:
        print(f"  ref: {s['ref']}")
        print(f"  hyp_A: {a['hyp']}")
        print(f"  hyp_B: {b['hyp']}")
        print()


if __name__ == '__main__':
    main()