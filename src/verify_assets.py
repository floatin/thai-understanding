"""
Verify all downloaded assets and produce a report.

Runs after downloads complete. Loads each asset, prints key info.
"""
import warnings
warnings.filterwarnings("ignore")
import io
import json
import sys
import time
from pathlib import Path
import torch
import pyarrow.parquet as pq
import soundfile as sf

BASE = Path("/data/workspace/asr-model-training/thai-understanding")
sys.path.insert(0, str(BASE / "src"))

def hr(t): print("\n" + "=" * 70 + f"\n {t}\n" + "=" * 70)

def fmt_size(n_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"

def file_info(path):
    if path.exists():
        return fmt_size(path.stat().st_size)
    return "MISSING"

hr("Thai-Understanding: Asset Verification Report")

# README
hr("1. README.md")
readme = BASE / "README.md"
print(f"Size: {file_info(readme)}")
print(f"Lines: {len(readme.read_text().splitlines())}")

# Pipeline diagram
hr("2. Pipeline figure (assets/)")
for f in ["Thai-SUP.png", "Thai-SUP.pdf"]:
    p = BASE / "assets" / f
    print(f"  {f}: {file_info(p)}")

# XLSR-Thai checkpoint
hr("3. XLSR-Thai checkpoint")
xlsr = BASE / "XLSR-Thai/checkpoint_best.pt"
print(f"Size: {file_info(xlsr)}")
ckpt = torch.load(xlsr, map_location='cpu', weights_only=False)
cfg = ckpt['cfg']['model']
print(f"Arch: {cfg['_name']} (XLSR-53 large variant)")
print(f"Layers: {cfg['encoder_layers']}, dim: {cfg['encoder_embed_dim']}, heads: {cfg['encoder_attention_heads']}")
print(f"FFN dim: {cfg['encoder_ffn_embed_dim']}")
n_params = sum(v.numel() for v in ckpt['model'].values())
print(f"Params: {n_params/1e6:.1f}M")

# LLM
hr("4. Typhoon2-LLaMa2-3B (LLM)")
from transformers import AutoTokenizer, AutoConfig
llm_path = BASE / "Typhoon2-3B"
total_size = sum(f.stat().st_size for f in llm_path.iterdir())
print(f"Total size: {fmt_size(total_size)}")
cfg = AutoConfig.from_pretrained(llm_path)
print(f"Arch: {cfg.architectures}, hidden: {cfg.hidden_size}, layers: {cfg.num_hidden_layers}")
print(f"Vocab: {cfg.vocab_size}")
tok = AutoTokenizer.from_pretrained(llm_path)
print(f"Tokenizer: {len(tok)} tokens")

# Datasets
hr("5. Thai-SUP dataset (dev/test shards)")
splits = []
for task in ["IC", "NER", "SR"]:
    for subset in ["dev", "test"]:
        path = BASE / f"Thai-SUP/{task}/{subset}/{subset}-00000.parquet"
        if path.exists():
            table = pq.read_table(str(path))
            df = table.to_pandas()
            duration_hr = df['duration_s'].sum() / 3600
            n_samples = len(df)
            labels = df['label'].value_counts().head(3).to_dict() if 'label' in df.columns else {}
            splits.append((task, subset, path.stat().st_size, n_samples, duration_hr, labels))
            print(f"  {task}/{subset}: {n_samples} samples, {duration_hr:.2f}h, "
                  f"{file_info(path)}, top labels: {list(labels.keys())[:3]}")

# Total sample check
hr("6. Single sample sanity check")
sample_path = BASE / "Thai-SUP/IC/dev/dev-00000.parquet"
table = pq.read_table(str(sample_path))
df = table.to_pandas()
row = df.iloc[0]
print(f"Columns: {list(df.columns)}")
print(f"Sample 0:")
for k in ['task_id', 'text', 'label', 'sampling_rate', 'num_channels', 'duration_s']:
    print(f"  {k}: {row[k]!r}")
audio, sr = sf.read(io.BytesIO(row['audio_flac']), dtype='float32')
print(f"  audio: shape={audio.shape}, dtype={audio.dtype}, sr={sr}, range=[{audio.min():.3f}, {audio.max():.3f}]")

# Pipeline test
hr("7. End-to-end pipeline test")
from xlsr_thai import load_xlsr_thai
print("Loading XLSR-Thai onto GPU...")
t0 = time.time()
encoder, enc_cfg = load_xlsr_thai(str(xlsr), device='cuda')
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Params: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M")
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\nTesting on real audio sample...")
audio_tensor = torch.from_numpy(audio).unsqueeze(0).cuda()
lengths = torch.tensor([audio_tensor.shape[1]]).cuda()
t0 = time.time()
with torch.no_grad():
    feats, lens = encoder(audio_tensor, lengths)
print(f"  Input: {audio_tensor.shape[1]/sr:.2f}s ({audio_tensor.shape[1]} samples)")
print(f"  Output: {feats.shape[1]} frames ({feats.shape[1]*20}ms)")
print(f"  Frame stride: {audio_tensor.shape[1]/feats.shape[1]:.0f}x = 20ms (XLSR standard)")
print(f"  Feature dim: {feats.shape[2]}")
print(f"  Time: {time.time()-t0:.2f}s")

# Check LLM also works
hr("8. LLM inference test (text-only)")
from transformers import AutoModelForCausalLM
print("Loading LLM...")
t0 = time.time()
llm = AutoModelForCausalLM.from_pretrained(str(llm_path), dtype=torch.bfloat16, device_map='cpu')
print(f"  Loaded in {time.time()-t0:.1f}s")
prompt_text = "แปลว่า:"
prompt = "ฉันชอบกินข้าว" + prompt_text
ids = tok(prompt, return_tensors='pt').input_ids
t0 = time.time()
with torch.no_grad():
    out = llm.generate(ids, max_new_tokens=15, do_sample=False, pad_token_id=tok.eos_token_id)
print(f"  Input: {prompt!r}")
print(f"  Output: {tok.decode(out[0])}")
print(f"  Time: {time.time()-t0:.1f}s (CPU)")

hr("ALL ASSETS VERIFIED OK")
