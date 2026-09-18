"""
Save all 100 hypotheses + add embedding-based semantic SER.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse, io, json, time, sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai, extract_features
from train_u_align import UAlignAdapter


def cer(ref, hyp):
    n = max(len(ref), 1)
    m = len(hyp)
    if m == 0: return 1.0
    dp = list(range(m + 1))
    for i in range(1, len(ref) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m] / n


def cer_ns(ref, hyp):
    return cer(ref.replace(' ', ''), hyp.replace(' ', ''))


def normalize_thai(text):
    """Light Thai normalization for fairer comparison."""
    import re
    # Remove common punctuation
    text = re.sub(r'[\s\u200b-\u200f\ufeff]+', '', text)
    text = re.sub(r'[.,!?;:()"\'\-—–_/\\]', '', text)
    # Decompose/normalize tone marks and sara am
    text = text.replace('\u0e4d', '\u0e3a')  # mai chattawa -> phinthu
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=100)
    ap.add_argument('--max_new_tokens', type=int, default=48)
    ap.add_argument('--adapter_a', default='/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_final_3ep.pt')
    ap.add_argument('--adapter_b', default='/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_best.pt')
    ap.add_argument('--encoder_ckpt', default='/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt')
    ap.add_argument('--llm_path', default='/data/workspace/asr-model-training/thai-understanding/Qwen3-4B')
    ap.add_argument('--out', default='/data/workspace/asr-model-training/thai-understanding/ser_full.json')
    args = ap.parse_args()

    base = Path('/data/workspace/asr-model-training/thai-understanding')
    print(f'=== Full SER eval (100 samples, save all) ===')

    print(f'\n[1] Loading encoder...')
    encoder, _ = load_xlsr_thai(args.encoder_ckpt, device='cuda')
    encoder.eval()

    print(f'\n[2] Loading LLM (CPU)...')
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, dtype=torch.bfloat16, device_map='cpu')
    llm.eval()
    embed_tokens = llm.get_input_embeddings()  # stays on CPU
    embed_cuda = embed_tokens.weight.detach().to("cuda").float()  # separate copy for similarity

    pq_path = base / f'Thai-SUP/SR/test/test-00000.parquet'
    df = pq.read_table(str(pq_path)).to_pandas().head(args.n_samples)
    print(f'\n[3] Loaded {len(df)} samples')

    prompt = "มันเป็นข้อความเสียงโปรดเขียนมันใหม่:"
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids
    prompt_embeds = embed_tokens(prompt_ids).to(torch.bfloat16)

    def load_adapter(p):
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        a = UAlignAdapter(encoder_dim=ckpt.get('encoder_dim', 1024),
                          llm_dim=ckpt.get('llm_dim', 2560),
                          downsample=ckpt.get('downsample', 2)).cuda().to(torch.bfloat16)
        a.load_state_dict(ckpt['adapter_state_dict'])
        a.eval()
        return a, ckpt

    def gen_all(adapter, name):
        print(f'\n[4] Generating with {name}...')
        results = []
        t0 = time.time()
        for i, row in df.iterrows():
            audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
            audio = audio[:8*16000] if len(audio) > 8*16000 else audio
            if len(audio) < 3200:
                audio = np.pad(audio, (0, 3200 - len(audio)))
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            with torch.no_grad():
                feats, _ = extract_features(encoder, audio_t, length)
                speech_embeds = adapter(feats.to(torch.bfloat16))
                inputs_embeds = torch.cat([prompt_embeds, speech_embeds.cpu()], dim=1)
                out = llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
            hyp = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            results.append({
                'data_id': row['data_id'],
                'ref': str(row['text']),
                'hyp': hyp,
                'duration_s': float(row['duration_s']),
            })
            if (i + 1) % 25 == 0:
                print(f'  [{i+1}/{len(df)}] ({time.time()-t0:.0f}s)', flush=True)
        return results, time.time() - t0

    adapter_a, ckpt_a = load_adapter(args.adapter_a)
    res_a, t_a = gen_all(adapter_a, 'adapter A')
    del adapter_a; torch.cuda.empty_cache()

    adapter_b, ckpt_b = load_adapter(args.adapter_b)
    res_b, t_b = gen_all(adapter_b, 'adapter B')
    del adapter_b; torch.cuda.empty_cache()

    # Compute all metrics including embedding similarity
    print(f'\n[5] Computing metrics including embedding similarity...')
    import torch.nn.functional as F

    def add_metrics(results):
        for r in results:
            ref, hyp = r['ref'], r['hyp']
            r['cer_ns'] = cer_ns(ref, hyp)
            r['cer_ns_norm'] = cer_ns(normalize_thai(ref), normalize_thai(hyp))
            r['exact_match'] = (ref.strip() == hyp.strip())
            # Embedding similarity via Qwen3 embed_tokens
            with torch.no_grad():
                rid = tokenizer(ref, return_tensors='pt').input_ids.cuda()
                hid = tokenizer(hyp, return_tensors='pt').input_ids.cuda()
                re = F.normalize(F.embedding(rid, embed_cuda).float().mean(dim=1), dim=-1)
                he = F.normalize(F.embedding(hid, embed_cuda).float().mean(dim=1), dim=-1)
                r['embed_sim'] = float((re * he).sum().item())
        return results

    res_a = add_metrics(res_a)
    res_b = add_metrics(res_b)

    def summary(results, thresholds=(0.7, 0.8, 0.9)):
        cers = [r['cer_ns'] for r in results]
        cers_n = [r['cer_ns_norm'] for r in results]
        em = [r['exact_match'] for r in results]
        sims = [r['embed_sim'] for r in results]
        s = {
            'avg_cer_ns': float(np.mean(cers)),
            'median_cer_ns': float(np.median(cers)),
            'avg_cer_ns_norm': float(np.mean(cers_n)),
            'median_cer_ns_norm': float(np.median(cers_n)),
            'avg_embed_sim': float(np.mean(sims)),
            'median_embed_sim': float(np.median(sims)),
            'ser_exact_match': float(1.0 - np.mean(em)),
            'n_samples': len(results),
        }
        for t in thresholds:
            s[f'ser_embed@{t}'] = float(1.0 - np.mean([s > t for s in sims]))
        return s

    sum_a = summary(res_a)
    sum_b = summary(res_b)

    print(f'\n{"="*70}')
    print(f'{"Metric":<25} {"Adapter A (3-ep)":<22} {"Adapter B (10-ep)":<22}')
    print(f'{"-"*70}')
    keys = ['avg_cer_ns', 'median_cer_ns', 'avg_cer_ns_norm', 'median_cer_ns_norm',
            'avg_embed_sim', 'median_embed_sim', 'ser_exact_match',
            'ser_embed@0.7', 'ser_embed@0.8', 'ser_embed@0.9']
    for k in keys:
        print(f'{k:<25} {sum_a[k]:<22.4f} {sum_b[k]:<22.4f}')

    Path(args.out).write_text(json.dumps({
        'config': {'n_samples': args.n_samples, 'task': 'SR/test'},
        'adapter_a_summary': sum_a, 'adapter_b_summary': sum_b,
        'adapter_a_time_s': t_a, 'adapter_b_time_s': t_b,
        'samples_a': res_a, 'samples_b': res_b,
    }, indent=2, ensure_ascii=False))
    print(f'\nSaved: {args.out} (with all {len(res_a)} samples per adapter)')


if __name__ == '__main__':
    main()
