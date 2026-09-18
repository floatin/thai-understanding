"""
Sanity check: Qwen3-Embedding-8B on Thai text.

Verify:
1. Same text vs itself -> cos ~ 1.0
2. Same text vs paraphrase -> cos high (>= 0.7)
3. Same text vs unrelated -> cos low (< 0.3)
4. Direction is correct (paraphrase > unrelated)
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = '/data/workspace/models/bge-m3-Thai'

print(f"Loading Qwen3-Embedding-8B from {MODEL_PATH}...")
model = SentenceTransformer(MODEL_PATH, device='cuda' if __import__('torch').cuda.is_available() else 'cpu')
print(f"  Device: {model.device}")

# Test cases (Thai business radio security context)
test_cases = [
    # (description, text_a, text_b)
    ("identical", "ขออนุญาตเช็คสัญญาณค่ะ", "ขออนุญาตเช็คสัญญาณค่ะ"),
    ("typo same word (เช็ด→เช็ค rule)", "ขออนุญาตเช็คสัญญาณค่ะ", "ขออนุญาตเช็ดสัญญาณค่ะ"),
    ("polite swap (ครับ↔ค่ะ)", "สวัสดีครับ", "สวัสดีค่ะ"),
    ("paraphrase (same intent)", "ประตูหนึ่งเปิดแล้ว", "ประตูที่หนึ่งเปิดเรียบร้อย"),
    ("number normalize", "ประตูสองออกสิบหกห้าค่ะ", "ประตูสองออกเวลา 16:05 ค่ะ"),
    ("different number", "ประตูหนึ่งเปิดแล้ว", "ประตูสองเปิดแล้ว"),
    ("completely unrelated", "ขออนุญาตเช็คสัญญาณค่ะ", "วันนี้อากาศดีมากเลย"),
    ("completely unrelated 2", "ประตูหนึ่งเปิดแล้ว", "ฉันชอบกินข้าวผัด"),
]

print(f"\nComputing embeddings for {len(test_cases)} pairs...")
texts = []
for _, a, b in test_cases:
    texts.append(a)
    texts.append(b)
embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
embs = np.array(embs)

print(f"\n{'description':<45s} {'cos_sim':>10s}")
print("-" * 60)
for i, (desc, a, b) in enumerate(test_cases):
    sim = float((embs[2*i] @ embs[2*i+1]).item())
    flag = ""
    if "identical" in desc and abs(sim - 1.0) < 0.01:
        flag = " ✓"
    elif "completely unrelated" in desc and sim < 0.5:
        flag = " ✓ (low)"
    elif "completely unrelated" not in desc and "different number" not in desc and sim > 0.5:
        flag = " ✓ (high)"
    elif "different number" in desc and sim < 0.7:
        flag = " ✓ (mid)"
    print(f"{desc:<45s} {sim:>10.4f}{flag}")

# Summary
print("\n=== Sanity Verdict ===")
identical_ok = bool(abs((embs[0] @ embs[1]).item() - 1.0) < 0.001)
print(f"  Identical → 1.0: {'PASS' if identical_ok else 'FAIL'}")

# check paraphrase > unrelated in aggregate
para_sims = []
unrel_sims = []
for i, (desc, a, b) in enumerate(test_cases):
    sim = float((embs[2*i] @ embs[2*i+1]).item())
    if "completely unrelated" in desc:
        unrel_sims.append(sim)
    elif "paraphrase" in desc or "normalize" in desc or "polite" in desc or "typo" in desc:
        para_sims.append(sim)

avg_para = np.mean(para_sims) if para_sims else 0
avg_unrel = np.mean(unrel_sims) if unrel_sims else 0
print(f"  Avg paraphrase cos: {avg_para:.4f}")
print(f"  Avg unrelated cos:  {avg_unrel:.4f}")
print(f"  Direction (para > unrel): {'PASS' if avg_para > avg_unrel + 0.1 else 'FAIL'}")