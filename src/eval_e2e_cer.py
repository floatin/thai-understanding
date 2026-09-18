"""
End-to-end evaluation: extract 1000 samples, run pipeline, compute CER.

Two experiments:
  A. Random adapter + LLM (current pipeline): baseline showing random output
  B. Frozen XLSR-Thai encoder + trained CTC head: meaningful ASR baseline

The "LLM transcription" is computed using the trained CTC head (Exp B), since
the LLM with random adapter just hallucinates and isn't a real transcription.

For Exp A, we also report what the LLM generates with random adapter for comparison.

GT = `text` field from the dataset (the spoken content).
"""
import warnings
warnings.filterwarnings("ignore")
import io
import time
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, '/data/workspace/asr-model-training/thai-understanding/src')
from xlsr_thai import load_xlsr_thai, extract_features

# Thai characters vocabulary (basic set for CER)
THAI_CHARS = (
    " กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
    "ะัาำิีุูเแโใไๅๆ่้๊๋์ํ"
    "๐๑๒๓๔๕๖๗๘๙"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ".,!?\"'()[]{}:;-"
)


def cer(ref: str, hyp: str) -> float:
    """Character error rate (edit distance / len(ref))."""
    if not ref:
        return 0.0 if not hyp else float(len(hyp))
    # Levenshtein distance
    n, m = len(ref), len(hyp)
    if m == 0:
        return n / n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m] / n


def load_samples(parquet_path, indices=None, n=1000):
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()
    if indices is None:
        random.seed(42)
        indices = random.sample(range(len(df)), min(n, len(df)))
    samples = []
    for i in indices:
        row = df.iloc[i]
        audio_bytes = row['audio_flac']
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        samples.append({
            'task_id': row['task_id'],
            'data_id': row['data_id'],
            'text': str(row['text']),  # GT transcription
            'label': str(row.get('label', '')),
            'audio': torch.from_numpy(audio),
            'sr': sr,
            'duration_s': float(row['duration_s']),
        })
    return samples


class CTCHead(nn.Module):
    """Linear classifier on top of XLSR features for character-level ASR."""

    def __init__(self, in_dim=1024, n_chars=None):
        super().__init__()
        self.proj = nn.Linear(in_dim, n_chars)
        # blank = 0

    def forward(self, x):  # (B, T, 1024)
        return self.proj(x)  # (B, T, n_chars)


def build_char_vocab(samples, extra_chars=""):
    """Build char vocabulary from training samples."""
    chars = set(THAI_CHARS)
    for s in samples:
        for c in s['text']:
            chars.add(c)
        for c in s.get('label', ''):
            chars.add(c)
    chars = sorted(chars)
    char_to_id = {c: i+1 for i, c in enumerate(chars)}  # 0 = blank
    id_to_char = {i+1: c for i, c in enumerate(chars)}
    id_to_char[0] = '<blk>'
    return char_to_id, id_to_char


def greedy_decode(logits, id_to_char):
    """Greedy CTC decode: argmax then collapse repeats and remove blanks."""
    # logits: (B, T, C)
    ids = logits.argmax(dim=-1).cpu().numpy()  # (B, T)
    results = []
    for seq in ids:
        out = []
        prev = -1
        for i in seq:
            if i != prev and i != 0:
                out.append(id_to_char.get(i, '?'))
            prev = i
        results.append(''.join(out))
    return results


def train_ctc_head(encoder, samples, char_to_id, n_epochs=10, batch_size=4, lr=1e-3, device='cuda'):
    """Train a CTC head on top of frozen XLSR encoder."""
    n_chars = max(char_to_id.values()) + 1
    head = CTCHead(in_dim=1024, n_chars=n_chars).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"  Training CTC head: {n_chars} chars, {len(samples)} samples, {n_epochs} epochs")
    encoder.eval()
    head.train()
    t0 = time.time()
    for epoch in range(n_epochs):
        random.shuffle(samples)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            audios = [s['audio'].to(device) for s in batch]
            texts = [s['text'] for s in batch]
            lengths_audio = torch.tensor([a.shape[0] for a in audios])
            # Pad audio
            max_len = lengths_audio.max().item()
            audio_pad = torch.zeros(len(batch), max_len)
            for j, a in enumerate(audios):
                audio_pad[j, :a.shape[0]] = a
            # Encode
            with torch.no_grad():
                feats, feat_lens = extract_features(encoder, audio_pad, lengths_audio)
            feats = feats.detach()  # (B, T', 1024)
            # Forward head
            logits = head(feats)  # (B, T', C)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T', B, C)
            # Targets
            targets = []
            target_lengths = []
            for t in texts:
                ids = [char_to_id[c] for c in t if c in char_to_id]
                if not ids:
                    ids = [1]  # at least one non-blank
                targets.extend(ids)
                target_lengths.append(len(ids))
            targets = torch.tensor(targets, dtype=torch.long).to(device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)
            # CTC loss expects (T, B, C), (B, T), targets, target_lengths
            loss = ctc_loss(log_probs, targets, feat_lens, target_lengths)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{n_epochs}: ctc_loss={avg:.4f}  ({time.time()-t0:.1f}s)")
    head.eval()
    return head


def run_experiment(encoder, head, samples, id_to_char, device='cuda'):
    """Run CTC inference and compute CER."""
    encoder.eval()
    head.eval()
    cers = []
    details = []
    n_skip = 0
    with torch.no_grad():
        for s in samples:
            # Pad/clip audio to at least 3200 samples (0.2s) so torchaudio conv works
            min_samples = 3200
            audio = s['audio']
            if audio.shape[0] < min_samples:
                pad = torch.zeros(min_samples - audio.shape[0])
                audio = torch.cat([audio, pad])
            audio = audio.unsqueeze(0).to(device)
            length = torch.tensor([audio.shape[1]]).to(device)
            try:
                feats, _ = extract_features(encoder, audio, length)
                logits = head(feats)
                hyp = greedy_decode(logits, id_to_char)[0]
            except Exception as e:
                n_skip += 1
                continue
            ref = s['text']
            c = cer(ref, hyp)
            cers.append(c)
            if len(details) < 5:
                details.append({'data_id': s['data_id'], 'task': s['task_id'], 'ref': ref, 'hyp': hyp, 'cer': round(c, 3)})
    return cers, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=1000)
    ap.add_argument('--ctc_epochs', type=int, default=8)
    ap.add_argument('--ctc_samples', type=int, default=400)
    args = ap.parse_args()

    base = Path("/data/workspace/asr-model-training/thai-understanding")

    # Load encoder on GPU
    print("[1/4] Loading XLSR-Thai on GPU...")
    encoder, cfg = load_xlsr_thai(str(base / "XLSR-Thai/checkpoint_best.pt"), device='cuda')

    # Load 1000 samples from dev+test (mixed tasks to be representative)
    print(f"\n[2/4] Loading {args.n_samples} samples from Thai-SUP dev+test...")
    all_samples = []
    per_split = args.n_samples // 6  # 3 tasks x 2 splits
    for task in ["IC", "NER", "SR"]:
        for subset in ["dev", "test"]:
            parquet = base / f"Thai-SUP/{task}/{subset}/{subset}-00000.parquet"
            samples = load_samples(parquet, n=per_split)
            all_samples.extend(samples)
            print(f"  {task}/{subset}: +{len(samples)}")
    print(f"  Total: {len(all_samples)} samples")
    # Take only samples where text is reasonable (length 5-200 chars)
    all_samples = [s for s in all_samples if 5 < len(s['text']) < 200][:args.n_samples]
    print(f"  After filter: {len(all_samples)} samples")

    # Build vocab
    char_to_id, id_to_char = build_char_vocab(all_samples)
    print(f"  Vocab size: {len(char_to_id)} chars")

    # === Experiment A: LLM with random adapter (smoke check) ===
    print(f"\n[3/4] Experiment A: LLM with random adapter (smoke check on first 10)...")
    from slu_pipeline import ThaiSLUPipeline
    pipeline = ThaiSLUPipeline(
        encoder_ckpt=str(base / "XLSR-Thai/checkpoint_best.pt"),
        llm_path=str(base / "Typhoon2-3B"),
        encoder_device="cuda", llm_device="cpu",
    )
    llm_cers = []
    for i, s in enumerate(all_samples[:10]):
        # For SR: text is the spoken content; prompt asks to rewrite
        # For IC/NER: text is the spoken content; prompt asks for intent/entities
        # We just measure CER between LLM output and GT text
        out = pipeline.generate(
            s['audio'].unsqueeze(0),
            "มันเป็นข้อความเสียงโปรดเขียนมันใหม่",
            max_new_tokens=32,
        )
        c = cer(s['text'], out)
        llm_cers.append(c)
        if i < 3:
            print(f"  [{i}] task={s['task_id']} CER={c:.2f}  ref={s['text'][:60]!r}  hyp={out[:60]!r}")
    print(f"  LLM (random adapter) avg CER on 10 samples: {sum(llm_cers)/len(llm_cers):.2%}")

    # === Experiment B: Train CTC head on a subset, then evaluate ===
    print(f"\n[4/4] Experiment B: Frozen XLSR-Thai + trained CTC head...")
    # Use a portion of samples for CTC training (could use train shards if downloaded)
    ctc_train_samples = all_samples[:args.ctc_samples]
    head = train_ctc_head(
        encoder, ctc_train_samples, char_to_id,
        n_epochs=args.ctc_epochs, batch_size=4, lr=1e-3,
    )
    # Evaluate on remaining samples
    eval_samples = all_samples[args.ctc_samples:]
    print(f"  Evaluating CTC on {len(eval_samples)} samples...")
    t0 = time.time()
    cers, details = run_experiment(encoder, head, eval_samples, id_to_char)
    avg_cer = sum(cers) / len(cers)
    print(f"  CTC avg CER on {len(cers)} samples: {avg_cer:.2%}  ({time.time()-t0:.1f}s)")

    # Save report
    report = {
        'n_samples_total': len(all_samples),
        'n_samples_ctc_train': len(ctc_train_samples),
        'n_samples_ctc_eval': len(eval_samples),
        'n_vocab': len(char_to_id),
        'exp_A_llm_random_adapter_cer_avg': sum(llm_cers)/len(llm_cers),
        'exp_A_llm_random_adapter_cers': llm_cers,
        'exp_B_ctc_cer_avg': avg_cer,
        'exp_B_ctc_cer_std': (sum((c-avg_cer)**2 for c in cers)/len(cers))**0.5,
        'exp_B_samples': details,
    }
    out = base / "results_e2e_cer.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Report saved to: {out}")
    print(f"\n  === Summary ===")
    print(f"  LLM (random adapter) CER: {sum(llm_cers)/len(llm_cers):.2%}  (10 samples, baseline)")
    print(f"  CTC head (small training) CER: {avg_cer:.2%}  ({len(eval_samples)} samples)")


if __name__ == "__main__":
    main()
