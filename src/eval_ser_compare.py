"""
End-to-end SER evaluation: compare 3-epoch vs 10-epoch (InfoNCE+DTW) adapter.

SER = Sentence Error Rate = fraction of samples where output != ground truth.
CER = Character Error Rate (also reported).
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import io
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai, extract_features
from train_u_align import UAlignAdapter as UAlignAdapterV1
from train_u_align_v2 import UAlignAdapter as UAlignAdapterV2


def cer(ref, hyp):
    n = max(len(ref), 1)
    m = len(hyp)
    if m == 0:
        return 1.0
    dp = list(range(m + 1))
    for i in range(1, len(ref) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m] / n


def cer_ns(ref, hyp):
    return cer(ref.replace(' ', ''), hyp.replace(' ', ''))


def load_adapter(adapter_path, device='cuda'):
    ckpt = torch.load(adapter_path, map_location='cpu', weights_only=False)
    # v2 ckpts include downsample; v1 ckpts default downsample=2
    enc_dim = ckpt.get('encoder_dim', 1024)
    llm_dim = ckpt.get('llm_dim', 2560)
    downsample = ckpt.get('downsample', 2)
    # Both v1 and v2 have identical UAlignAdapter class shape
    adapter = UAlignAdapterV1(encoder_dim=enc_dim, llm_dim=llm_dim, downsample=downsample).to(device).to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    adapter.eval()
    return adapter, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=100)
    ap.add_argument('--task', default='SR')
    ap.add_argument('--split', default='test')
    ap.add_argument('--max_new_tokens', type=int, default=48)
    ap.add_argument('--gpu_memory_limit', type=float, default=0.6)
    ap.add_argument('--adapter_a', default='/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_final_3ep.pt')
    ap.add_argument('--adapter_b', default='/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_best.pt')
    ap.add_argument('--encoder_ckpt', default='/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt')
    ap.add_argument('--llm_path', default='/data/workspace/asr-model-training/thai-understanding/Qwen3-4B')
    ap.add_argument('--out', default='/data/workspace/asr-model-training/thai-understanding/ser_compare.json')
    args = ap.parse_args()

    base = Path('/data/workspace/asr-model-training/thai-understanding')
    print(f'=== End-to-end SER comparison ===')
    print(f'Task: {args.task}/{args.split}, samples: {args.n_samples}')

    # Load encoder (frozen, on GPU)
    print(f'\n[1/4] Loading XLSR-Thai encoder...')
    encoder, cfg = load_xlsr_thai(args.encoder_ckpt, device='cuda')
    encoder.eval()

    # Load LLM
    print(f'\n[2/4] Loading LLM...')
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if args.gpu_memory_limit > 0:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path, dtype=torch.bfloat16, device_map='cuda')
        llm_device = 'cuda'
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path, dtype=torch.bfloat16, device_map='cpu')
        llm_device = 'cpu'
    llm.eval()
    embed_tokens = llm.get_input_embeddings()
    print(f'  LLM device: {llm_device}')

    # Load test samples
    print(f'\n[3/4] Loading test samples...')
    pq_path = base / f'Thai-SUP/{args.task}/{args.split}/{args.split}-00000.parquet'
    df = pq.read_table(str(pq_path)).to_pandas().head(args.n_samples)
    print(f'  Loaded {len(df)} samples')

    prompts = {
        'SR': 'มันเป็นข้อความเสียงโปรดเขียนมันใหม่:',
        'IC': 'นี่คือเสียง โปรดจำแนกเจตนา:',
        'NER': 'นี่คือเสียง โปรดระบุชื่อเฉพาะ:',
    }
    prompt = prompts[args.task]
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(llm_device)
    prompt_embeds = embed_tokens(prompt_ids).to(torch.bfloat16)

    def evaluate(adapter, adapter_name):
        print(f'\n[4/4] Evaluating with {adapter_name}...')
        results = []
        t0 = time.time()
        for i, row in df.iterrows():
            audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
            max_samples = 8 * 16000
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            if len(audio) < 3200:
                audio = np.pad(audio, (0, 3200 - len(audio)))
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            with torch.no_grad():
                feats, _ = extract_features(encoder, audio_t, length)
                speech_embeds = adapter(feats.to(torch.bfloat16))
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
                'ref': ref,
                'hyp': hyp,
                'cer': c,
                'sentence_correct': (ref.strip() == hyp.strip()),
            })
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                avg_cer = np.mean([r['cer'] for r in results])
                ser = 1.0 - np.mean([r['sentence_correct'] for r in results])
                print(f'  [{i+1}/{len(df)}] avg CER={avg_cer:.2%}, SER={ser:.2%} ({elapsed:.0f}s)', flush=True)
        return results, time.time() - t0

    # Adapter A (3-epoch)
    adapter_a, ckpt_a = load_adapter(args.adapter_a, device='cuda')
    print(f'  Adapter A: {ckpt_a.get("final_epoch", "?")} epoch, loss={ckpt_a["log"][-1].get("nce", "?")}')
    results_a, time_a = evaluate(adapter_a, 'adapter A (3-epoch)')

    # Free adapter A
    del adapter_a
    torch.cuda.empty_cache()

    # Adapter B (10-epoch InfoNCE+DTW)
    adapter_b, ckpt_b = load_adapter(args.adapter_b, device='cuda')
    print(f'  Adapter B: epoch={ckpt_b.get("epoch", "?")}, best_r10={ckpt_b.get("best_r10", "?")}')
    results_b, time_b = evaluate(adapter_b, 'adapter B (10-epoch InfoNCE+DTW)')

    # Aggregate metrics
    def summarize(results):
        cers = [r['cer'] for r in results]
        corrects = [r['sentence_correct'] for r in results]
        return {
            'avg_cer_ns': float(np.mean(cers)),
            'median_cer': float(np.median(cers)),
            'min_cer': float(min(cers)),
            'max_cer': float(max(cers)),
            'ser': float(1.0 - np.mean(corrects)),  # sentence error rate
            'sentence_accuracy': float(np.mean(corrects)),
            'n_samples': len(results),
        }

    sum_a = summarize(results_a)
    sum_b = summarize(results_b)

    print(f'\n{"="*60}')
    print(f'Results: {args.task}/{args.split}, {len(df)} samples')
    print(f'{"="*60}')
    print(f'{"Metric":<22} {"Adapter A (3-ep)":<20} {"Adapter B (10-ep)":<20} {"Δ":<10}')
    print(f'{"-"*72}')
    for key in ['avg_cer_ns', 'median_cer', 'ser', 'sentence_accuracy']:
        delta = sum_b[key] - sum_a[key]
        print(f'{key:<22} {sum_a[key]:<20.4f} {sum_b[key]:<20.4f} {delta:+.4f}')

    # Save detailed results
    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        'config': {
            'task': args.task,
            'split': args.split,
            'n_samples': len(df),
            'prompt': prompt,
            'gpu_memory_limit': args.gpu_memory_limit,
        },
        'adapter_a': {
            'path': args.adapter_a,
            'final_epoch': ckpt_a.get('final_epoch', '?'),
            'summary': sum_a,
            'time_seconds': time_a,
        },
        'adapter_b': {
            'path': args.adapter_b,
            'final_epoch': ckpt_b.get('epoch', '?'),
            'best_r10': ckpt_b.get('best_r10', '?'),
            'summary': sum_b,
            'time_seconds': time_b,
        },
        'samples_a': results_a[:20],
        'samples_b': results_b[:20],
    }, indent=2, ensure_ascii=False))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
