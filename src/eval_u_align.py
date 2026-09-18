"""
Stage 1 evaluation: Run U-Align-trained pipeline on Thai-SUP dev/test.

Approach:
  1. audio → XLSR-Thai (frozen) → adapter (trained) → speech_embeds
  2. Concatenate prompt_embeds + speech_embeds → LLM (Qwen3-4B, frozen)
  3. LLM generates output

For evaluation, we focus on:
  - ASR via SR samples (audio → transcript, compare to text)
  - SLU: SR task (rewriting) - compare to text and label
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import io
import json
import time
import gc
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai, extract_features
from train_u_align import UAlignAdapter


def cer(ref, hyp):
    n = max(len(ref), 1)
    m = len(hyp)
    if m == 0:
        return 1.0
    dp = list(range(m+1))
    for i in range(1, len(ref)+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cur = dp[j]
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m] / n


def cer_ns(ref, hyp):
    return cer(ref.replace(' ', ''), hyp.replace(' ', ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder_ckpt', default='/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt')
    ap.add_argument('--llm_path', default='/data/workspace/asr-model-training/thai-understanding/Qwen3-4B')
    ap.add_argument('--adapter_ckpt', default='/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_final.pt')
    ap.add_argument('--task', default='SR', choices=['IC', 'NER', 'SR'])
    ap.add_argument('--split', default='test', choices=['dev', 'test'])
    ap.add_argument('--n_samples', type=int, default=100)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--gpu_memory_limit', type=float, default=0.6,
                    help='Fraction of GPU memory to use for LLM (0=no GPU, 1=all)')
    args = ap.parse_args()

    base = Path("/data/workspace/asr-model-training/thai-understanding")
    print(f"=== U-Align Evaluation ===")
    print(f"Task: {args.task}/{args.split}, samples: {args.n_samples}")

    # Encoder (frozen)
    print(f"\n[1/4] Loading XLSR-Thai encoder...")
    encoder, cfg = load_xlsr_thai(args.encoder_ckpt, device='cuda')
    encoder.eval()

    # Adapter
    print(f"\n[2/4] Loading U-Align adapter...")
    ckpt = torch.load(args.adapter_ckpt, map_location='cpu', weights_only=False)
    adapter = UAlignAdapter(
        encoder_dim=ckpt['encoder_dim'],
        llm_dim=ckpt['llm_dim'],
        downsample=ckpt['downsample'],
    ).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    adapter.eval()
    print(f"  Trained {ckpt['final_epoch']} epochs, final loss={ckpt['log'][-1].get('nce', '?')}")

    # LLM (full, for inference)
    print(f"\n[3/4] Loading LLM from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if args.gpu_memory_limit > 0:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path, dtype=torch.bfloat16, device_map='cuda'
        )
        llm_device = 'cuda'
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path, dtype=torch.bfloat16, device_map='cpu'
        )
        llm_device = 'cpu'
    llm.eval()
    embed_tokens = llm.get_input_embeddings()
    print(f"  LLM device: {llm_device}, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load test data
    print(f"\n[4/4] Loading {args.task}/{args.split} samples...")
    parquet_path = base / f"Thai-SUP/{args.task}/{args.split}/{args.split}-00000.parquet"
    df = pq.read_table(str(parquet_path)).to_pandas()
    df = df.head(args.n_samples)
    print(f"  Loaded {len(df)} samples")

    # Task-specific prompts
    prompts = {
        'SR': "มันเป็นข้อความเสียงโปรดเขียนมันใหม่:",  # rewrite
        'IC': "นี่คือเสียง โปรดจำแนกเจตนา:",  # classify intent
        'NER': "นี่คือเสียง โปรดระบุชื่อเฉพาะ:",  # identify entities
    }
    prompt = prompts[args.task]

    # Inference
    print(f"\n=== Running inference ({len(df)} samples) ===")
    results = []
    t0 = time.time()
    for i, row in df.iterrows():
        audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
        if len(audio) < 3200:
            audio = np.pad(audio, (0, 3200 - len(audio)))
        audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
        length = torch.tensor([audio_t.shape[1]]).cuda()
        with torch.no_grad():
            feats, _ = extract_features(encoder, audio_t, length)
            speech_embeds = adapter(feats.to(torch.bfloat16))
            prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(llm_device)
            prompt_embeds = embed_tokens(prompt_ids).to(torch.bfloat16)
            inputs_embeds = torch.cat([prompt_embeds, speech_embeds.to(llm_device)], dim=1)
            out = llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        hyp = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        ref = str(row['text'])
        c = cer_ns(ref, hyp)
        results.append({
            'ref': ref, 'hyp': hyp, 'cer': c,
            'data_id': row['data_id'],
            'duration': float(row['duration_s']),
        })
        if (i+1) % 20 == 0:
            elapsed = time.time() - t0
            avg_cer = np.mean([r['cer'] for r in results])
            print(f"  [{i+1}/{len(df)}] avg CER={avg_cer:.2%} ({elapsed:.0f}s)", flush=True)

    # Stats
    cers = [r['cer'] for r in results]
    print(f"\n=== Results ===")
    print(f"Samples: {len(results)}")
    print(f"Avg CER (no_space): {np.mean(cers):.2%}")
    print(f"Median CER: {np.median(cers):.2%}")
    print(f"Min: {min(cers):.2%}, Max: {max(cers):.2%}")

    print(f"\nFirst 10 examples:")
    for r in results[:10]:
        print(f"  ref: {r['ref'][:80]!r}")
        print(f"  hyp: {r['hyp'][:80]!r}")
        print(f"  CER: {r['cer']:.2%}")

    # Save
    out = base / f"phase1_u_align_eval_{args.task}_{args.split}.json"
    out.write_text(json.dumps({
        'task': args.task,
        'split': args.split,
        'n_samples': len(results),
        'avg_cer_ns': float(np.mean(cers)),
        'median_cer': float(np.median(cers)),
        'min_cer': float(min(cers)),
        'max_cer': float(max(cers)),
        'samples': results[:20],
        'training_log': ckpt['log'],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
