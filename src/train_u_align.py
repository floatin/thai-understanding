"""
Stage 1: U-Align training per the PTT Pipeline Development Guide.

Per document Section 2.3:
  "损失函数 | InfoNCE（句级对比）+ 可选 cosine-DTW（帧级对齐）"

Note: cosine-DTW is OPTIONAL. This implementation uses InfoNCE only for speed,
which is functionally equivalent to the U-Align Stage 1 goal (aligning speech
representations with text representations in the LLM embedding space).
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import time
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).parent))
from xlsr_thai import load_xlsr_thai, extract_features


class CNNSubsampler(nn.Module):
    def __init__(self, dim: int, factor: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=factor, stride=factor, groups=dim, bias=False)
        with torch.no_grad():
            w = torch.zeros(dim, 1, factor)
            for i in range(factor):
                w[:, 0, i] = 1.0 / factor
            self.conv.weight.copy_(w)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        return self.conv(x_t).transpose(1, 2)


class UAlignAdapter(nn.Module):
    """U-Align adapter: LN + CNN subsampler + MLP projection."""
    def __init__(self, encoder_dim: int, llm_dim: int, downsample: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(encoder_dim)
        self.subsampler = CNNSubsampler(dim=encoder_dim, factor=downsample)
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):
        return self.proj(self.subsampler(self.norm(x)))


def info_nce_loss(speech_embeds, text_embeds, temperature=0.07):
    """Sentence-level InfoNCE (in-batch contrastive).
    Pool speech and text to single vectors, compute cosine similarity,
    cross-entropy with diagonal as positives."""
    s_pool = F.normalize(speech_embeds.mean(dim=1), dim=-1)
    t_pool = F.normalize(text_embeds.mean(dim=1), dim=-1)
    sim = torch.mm(s_pool, t_pool.t()) / temperature
    labels = torch.arange(speech_embeds.shape[0], device=speech_embeds.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2


def load_train_samples(shard_paths, n_per_shard=200, seed=42):
    rng = random.Random(seed)
    for shard in shard_paths:
        df = pq.read_table(str(shard), columns=["text", "audio_flac"]).to_pandas()
        n = min(n_per_shard, len(df))
        indices = rng.sample(range(len(df)), n)
        for i in indices:
            row = df.iloc[i]
            audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
            yield {'audio': torch.from_numpy(audio), 'text': str(row['text'])}


def make_batch(samples, tokenizer, max_text_tokens=32):
    valid = [s for s in samples if 5 < len(s['text']) < 200]
    if not valid:
        return None
    audios, texts = [], []
    for s in valid:
        a = s['audio']
        if len(a) < 3200:
            a = torch.cat([a, torch.zeros(3200 - len(a))])
        audios.append(a)
        texts.append(s['text'])
    max_len = max(a.shape[0] for a in audios)
    audio_pad = torch.zeros(len(audios), max_len)
    for j, a in enumerate(audios):
        audio_pad[j, :a.shape[0]] = a
    enc = tokenizer(texts, return_tensors='pt', padding='max_length',
                    truncation=True, max_length=max_text_tokens)
    return {
        'audio': audio_pad,
        'input_ids': enc.input_ids,
        'attention_mask': enc.attention_mask,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder_ckpt', default='/data/workspace/asr-model-training/thai-understanding/XLSR-Thai/checkpoint_best.pt')
    ap.add_argument('--llm_path', default='/data/workspace/asr-model-training/thai-understanding/Qwen3-4B')
    ap.add_argument('--train_shards_glob', default='/data/workspace/asr-model-training/thai-understanding/Thai-SUP/SR/train')
    ap.add_argument('--n_shards', type=int, default=10)
    ap.add_argument('--n_per_shard', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--downsample', type=int, default=2)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--max_text_tokens', type=int, default=32)
    ap.add_argument('--embed_only', action='store_true')
    ap.add_argument('--output_dir', default='/data/workspace/asr-model-training/thai-understanding/u_align_adapter')
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Stage 1 U-Align Training (InfoNCE only) ===")
    print(f"Args: bs={args.batch_size}, epochs={args.epochs}, lr={args.lr}, "
          f"temp={args.temperature}, max_text_tokens={args.max_text_tokens}")

    # Encoder
    print(f"\n[1/5] Loading XLSR-Thai encoder...")
    encoder, cfg = load_xlsr_thai(args.encoder_ckpt, device='cuda')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder_dim = cfg['encoder_embed_dim']

    # LLM embed_tokens
    print(f"\n[2/5] Loading LLM embed_tokens from {args.llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, dtype=torch.bfloat16, device_map='cpu')
    embed_tokens = llm.get_input_embeddings().to('cuda')
    llm_dim = embed_tokens.embedding_dim
    print(f"  llm_dim={llm_dim}")

    # Adapter
    print(f"\n[3/5] Building adapter ({encoder_dim} -> {llm_dim})...")
    adapter = UAlignAdapter(encoder_dim=encoder_dim, llm_dim=llm_dim, downsample=args.downsample).cuda().to(torch.bfloat16)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  adapter params: {n_params/1e6:.2f}M")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

    # Data
    print(f"\n[4/5] Loading training data...")
    base_dir = Path(args.train_shards_glob)
    shard_paths = sorted(base_dir.glob("*.parquet"))[:args.n_shards]
    all_samples = list(load_train_samples(shard_paths, n_per_shard=args.n_per_shard))
    print(f"  Using {len(shard_paths)} shards, {len(all_samples)} samples")

    # Train
    print(f"\n[5/5] Training {args.epochs} epochs...")
    adapter.train()
    log = []
    for epoch in range(args.epochs):
        random.shuffle(all_samples)
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for i in range(0, len(all_samples), args.batch_size):
            batch = make_batch(all_samples[i:i+args.batch_size], tokenizer,
                               max_text_tokens=args.max_text_tokens)
            if batch is None:
                continue
            audio = batch['audio'].cuda()
            input_ids = batch['input_ids'].cuda()
            attn_mask = batch['attention_mask'].cuda()
            lengths = torch.tensor([a.shape[0] for a in audio], dtype=torch.long).cuda()
            with torch.no_grad():
                feats, _ = extract_features(encoder, audio, lengths)
                # Mask out padding in text embeddings
                text_embeds = embed_tokens(input_ids).to(torch.bfloat16)
                # Set padding positions to zero (so they don't contribute to mean pooling)
                text_embeds = text_embeds * attn_mask.unsqueeze(-1).to(torch.bfloat16)
            speech_embeds = adapter(feats.to(torch.bfloat16))
            # Mean-pool over valid positions
            # Mask out padding audio frames... actually all frames are valid since we pad to max_len
            # But we should ignore very late frames after audio ends
            # Use feat_lens to mask
            feat_lens = torch.tensor([f.shape[1] for f in feats], dtype=torch.long).cuda()
            mask = torch.arange(speech_embeds.shape[1], device='cuda').unsqueeze(0) < feat_lens.unsqueeze(1)
            speech_embeds = speech_embeds * mask.unsqueeze(-1).to(torch.bfloat16)
            nce = info_nce_loss(speech_embeds.float(), text_embeds.float(), temperature=args.temperature)
            optimizer.zero_grad()
            nce.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            total_loss += nce.item()
            n_batches += 1
            if n_batches % 5 == 0:
                elapsed = time.time() - t0
                rate = n_batches / elapsed
                eta = (len(all_samples)//args.batch_size - n_batches) / max(rate, 0.001)
                print(f"    [Epoch {epoch+1}] batch {n_batches}, nce={total_loss/n_batches:.4f}, "
                      f"{elapsed:.0f}s, ETA {eta:.0f}s", flush=True)
        epoch_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{args.epochs}: nce={epoch_loss:.4f}  ({elapsed:.0f}s)", flush=True)
        log.append({'epoch': epoch+1, 'nce': epoch_loss})

    # Save
    torch.save({
        'adapter_state_dict': adapter.state_dict(),
        'encoder_dim': encoder_dim,
        'llm_dim': llm_dim,
        'downsample': args.downsample,
        'final_epoch': args.epochs,
        'log': log,
        'args': vars(args),
    }, output_dir / "adapter_final.pt")
    (output_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nAdapter saved to {output_dir}")


if __name__ == "__main__":
    main()
