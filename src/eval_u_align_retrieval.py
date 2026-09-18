"""
Test Stage 1 alignment via retrieval:
  - Given audio embedding from adapter, can we retrieve its matching text embedding
    among many other text embeddings?
  - Higher retrieval accuracy = better alignment
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai, extract_features
from train_u_align import UAlignAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder_ckpt', default='/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt')
    ap.add_argument('--llm_path', default='/data/workspace/asr-model-training/thai-understanding/Qwen3-4B')
    ap.add_argument('--adapter_ckpt', default='/data/workspace/asr-model-training/thai-understanding/u_align_adapter/adapter_final.pt')
    ap.add_argument('--n_samples', type=int, default=64)
    args = ap.parse_args()

    base = Path("/data/workspace/asr-model-training/thai-understanding")
    print(f"=== U-Align Retrieval Evaluation ===")

    # Load encoder
    print(f"\n[1/4] Loading XLSR-Thai encoder...")
    encoder, cfg = load_xlsr_thai(args.encoder_ckpt, device='cuda')
    encoder.eval()

    # Load adapter
    print(f"\n[2/4] Loading U-Align adapter...")
    ckpt = torch.load(args.adapter_ckpt, map_location='cpu', weights_only=False)
    adapter = UAlignAdapter(
        encoder_dim=ckpt['encoder_dim'], llm_dim=ckpt['llm_dim'], downsample=ckpt['downsample'],
    ).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    adapter.eval()

    # Load LLM (embed_tokens only)
    print(f"\n[3/4] Loading LLM embed_tokens...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, dtype=torch.bfloat16, device_map='cpu')
    embed_tokens = llm.get_input_embeddings().to('cuda')
    print(f"  embed_tokens: {embed_tokens.weight.shape}")

    # Load SR test data
    print(f"\n[4/4] Loading test samples for retrieval...")
    df = pq.read_table(str(base / "Thai-SUP/SR/test/test-00000.parquet")).to_pandas().head(args.n_samples)
    print(f"  {len(df)} samples")

    # Compute all audio and text embeddings
    audio_embeds = []
    text_embeds = []
    texts = []
    print(f"\nComputing embeddings...")
    t0 = time.time()
    with torch.no_grad():
        for i, row in df.iterrows():
            audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
            if len(audio) < 3200:
                audio = np.pad(audio, (0, 3200 - len(audio)))
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            feats, _ = extract_features(encoder, audio_t, length)
            speech_emb = adapter(feats.to(torch.bfloat16))  # (1, T_s, D)
            speech_emb = speech_emb.mean(dim=1)  # mean pool to (1, D)
            audio_embeds.append(speech_emb)
            # Text embedding
            ids = tokenizer(str(row['text']), return_tensors='pt').input_ids.cuda()
            text_emb = embed_tokens(ids).mean(dim=1)  # mean pool
            text_embeds.append(text_emb)
            texts.append(str(row['text']))
    audio_embeds = torch.cat(audio_embeds, dim=0).float()  # (N, D)
    text_embeds = torch.cat(text_embeds, dim=0).float()  # (N, D)
    print(f"  Computed {len(audio_embeds)} pairs in {time.time()-t0:.0f}s")

    # Normalize
    audio_norm = F.normalize(audio_embeds, dim=-1)
    text_norm = F.normalize(text_embeds, dim=-1)

    # Compute similarity matrix
    sim = torch.mm(audio_norm, text_norm.t())  # (N, N)
    # Recall@K: for each audio i, does the matching text i appear in top-K?
    print(f"\n=== Retrieval Results ===")
    for K in [1, 5, 10, 50]:
        if K > len(df): continue
        # Top-K indices for each audio
        topk = sim.topk(K, dim=-1).indices  # (N, K)
        # Check if true index is in top-K
        targets = torch.arange(len(df)).cuda()
        recall = (topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
        print(f"  Recall@{K}: {recall:.2%}")

    # Check diagonal similarity vs off-diagonal
    diag = torch.diagonal(sim).mean().item()
    off_diag = (sim.sum() - sim.diagonal().sum()) / (len(df) * (len(df) - 1))
    print(f"\n  Diagonal similarity (matched pairs): {diag:.4f}")
    print(f"  Off-diagonal similarity (random pairs): {off_diag:.4f}")
    print(f"  Ratio: {diag/off_diag:.2f}x")

    # Top-1 match examples
    print(f"\n  Top-1 retrieval examples (correct = ✓, wrong = ✗):")
    top1_idx = sim.argmax(dim=-1).cpu().numpy()
    for i in [0, 1, 5, 10, 20, 30]:
        if i >= len(df): continue
        match = "✓" if top1_idx[i] == i else "✗"
        print(f"    [{i}] {match} audio(text={texts[i][:30]!r}) → top1({texts[top1_idx[i]][:30]!r})")

    # Save
    report = {
        'n_samples': len(df),
        'recall_at_1': float((sim.argmax(dim=-1) == torch.arange(len(df)).cuda()).float().mean().item()),
        'diag_similarity': diag,
        'off_diag_similarity': float(off_diag),
        'similarity_ratio': diag / float(off_diag),
        'training_nce_loss': [e['nce'] for e in ckpt['log']],
    }
    for K in [1, 5, 10]:
        if K <= len(df):
            topk = sim.topk(K, dim=-1).indices
            targets = torch.arange(len(df)).cuda()
            report[f'recall_at_{K}'] = float((topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item())
    out = base / "phase1_u_align_retrieval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
