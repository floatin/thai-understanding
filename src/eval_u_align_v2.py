"""
Evaluate v2 adapter (InfoNCE + cosine-DTW, 11800 samples, 20+ epochs).
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, '/data/workspace/asr-model-training/thai-understanding/src')
from xlsr_thai import load_xlsr_thai, extract_features
from train_u_align_v2 import UAlignAdapter

# Load
print("Loading encoder...")
encoder, _ = load_xlsr_thai('/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt')
encoder.eval()

print("Loading adapter (best checkpoint)...")
ckpt = torch.load('/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_best.pt', map_location='cpu', weights_only=False)
adapter = UAlignAdapter(encoder_dim=ckpt['encoder_dim'], llm_dim=ckpt['llm_dim'], downsample=ckpt['downsample']).cuda().to(torch.bfloat16)
adapter.load_state_dict(ckpt['adapter_state_dict'])
adapter.eval()
print(f"  Best epoch: {ckpt.get('epoch', '?')}, R@10: {ckpt.get('best_r10', '?')}")

print("Loading LLM embed_tokens...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('/data/workspace/asr-model-training/thai-understanding/Qwen3-4B')
llm = AutoModelForCausalLM.from_pretrained('/data/workspace/asr-model-training/thai-understanding/Qwen3-4B', dtype=torch.bfloat16, device_map='cpu')
embed_tokens = llm.get_input_embeddings().to('cuda')

# Compare with random
print("\nCreating random adapter for comparison...")
torch.manual_seed(0)
random_adapter = UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
random_adapter.eval()

# Load all 12000 + 200 val
import io, soundfile as sf, numpy as np, pyarrow.parquet as pq
print("\nLoading test data...")
all_dfs = []
for task in ['IC', 'NER', 'SR']:
    for subset in ['dev', 'test']:
        f = f'/data/workspace/asr-model-training/thai-understanding/Thai-SUP/{task}/{subset}/{subset}-00000.parquet'
        all_dfs.append(pq.read_table(f, columns=['text', 'audio_flac']).to_pandas())
import pandas as pd
df = pd.concat(all_dfs, ignore_index=True)
df = df.sample(n=512, random_state=42).reset_index(drop=True)  # random sample for fast eval
print(f"  Using {len(df)} samples for retrieval test")

def compute_embeddings(adapter):
    audio_embeds, text_embeds = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
            max_samples = 8 * 16000
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            if len(audio) < 3200:
                audio = np.pad(audio, (0, 3200 - len(audio)))
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            feats, _ = extract_features(encoder, audio_t, length)
            speech_emb = adapter(feats.to(torch.bfloat16)).mean(dim=1)
            audio_embeds.append(speech_emb)
            ids = tokenizer(str(row['text']), return_tensors='pt').input_ids.cuda()
            text_emb = embed_tokens(ids).mean(dim=1)
            text_embeds.append(text_emb)
    return torch.cat(audio_embeds, dim=0).float(), torch.cat(text_embeds, dim=0).float()

print("Computing trained adapter embeddings...")
a_trained, t = compute_embeddings(adapter)
print("Computing random adapter embeddings...")
a_random, _ = compute_embeddings(random_adapter)

def retrieval_metrics(audio, text):
    a = F.normalize(audio, dim=-1)
    te = F.normalize(text, dim=-1)
    sim = torch.mm(a, te.t())
    metrics = {}
    for K in [1, 5, 10, 50, 100]:
        topk = sim.topk(K, dim=-1).indices
        targets = torch.arange(len(df)).cuda()
        metrics[f'R@{K}'] = float((topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item())
    metrics['diag'] = float(torch.diagonal(sim).mean().item())
    off_diag = float((sim.sum() - sim.diagonal().sum()) / (len(df) * (len(df) - 1)))
    metrics['off_diag'] = off_diag
    metrics['ratio'] = metrics['diag'] / off_diag
    return metrics

print("\n=== Trained adapter ===")
m_trained = retrieval_metrics(a_trained, t)
for k, v in m_trained.items():
    print(f"  {k}: {v:.4f}")

print("\n=== Random adapter (baseline) ===")
m_random = retrieval_metrics(a_random, t)
for k, v in m_random.items():
    print(f"  {k}: {v:.4f}")

print("\n=== Improvement ===")
for K in [1, 5, 10, 50, 100]:
    print(f"  R@{K}: {m_trained[f'R@{K}'] - m_random[f'R@{K}']:+.2%}")
print(f"  Diagonal ratio: {m_trained['ratio']:.2f}x vs {m_random['ratio']:.2f}x")

import json
out = Path('/data/workspace/asr-model-training/thai-understanding/phase1_u_align_v2_retrieval.json')
out.write_text(json.dumps({
    'n_samples': len(df),
    'epochs_trained': ckpt.get('epoch', '?'),
    'best_r10_val': ckpt.get('best_r10', '?'),
    'trained': m_trained,
    'random': m_random,
}, indent=2))
print(f"\nSaved: {out}")
