"""Diagnostic probe: does the real PTT audio (h173) contain learnable content
for our encoders at all?

Feed h173 audio through the VERIFIED XLSR-53+Thai-CTC head (CER 37.95% on
Thai-SUP TTS). If CER here is moderate (<=~60%), real-audio features carry
content and the Stage-2 failure is in the adapter/training recipe. If CER is
garbage-level, the acoustic-domain gap itself is the wall.
"""
import warnings
warnings.filterwarnings("ignore")
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

sys.path.insert(0, str(Path(__file__).parent))
from eval_cer_pretrained import cer, cer_no_space
import train_u_align_stage2_v3 as v3  # load_audio (10s cap, 16kHz)

OUT = '/data/workspace/asr-model-training/thai-understanding/out_metric_gate/ctc_probe_h173.json'


def main():
    eval_full = json.load(open(v3.EVAL_SPLIT))['samples']
    h173 = [s for s in eval_full if s.get('subset') == 'GT_human']

    base = Path('/data/workspace/asr-model-training/thai-understanding')
    model_path = base / 'ZikXewen-thai-ctc'
    tokenizer = Wav2Vec2CTCTokenizer(vocab_file=str(model_path / 'vocab.json'),
                                     unk_token='<unk>', pad_token='<pad>', word_delimiter_token='|')
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer)
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path)).cuda().eval()

    cers_ws, cers_ns, details = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, s in enumerate(h173):
            audio = v3.load_audio(s['wav_path'])  # same 10s cap as the main pipeline
            inputs = processor(audio, sampling_rate=16000, return_tensors='pt').input_values.cuda()
            pred_ids = torch.argmax(model(inputs).logits, dim=-1)
            pred = processor.batch_decode(pred_ids)[0]
            ref = s['ref']
            cers_ws.append(cer(ref, pred))
            cers_ns.append(cer_no_space(ref, pred))
            details.append({'ref': ref, 'hyp': pred})
            if i < 8:
                print(f"  REF: {ref[:56]!r}\n  HYP: {pred[:56]!r}\n  CER_ns={cers_ns[-1]:.3f}\n")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(h173)} ({time.time()-t0:.0f}s)", flush=True)

    ws, ns = np.array(cers_ws), np.array(cers_ns)
    micro_ns = sum(lev if False else 0 for lev in [])  # placeholder, macro below
    print(f"\n=== CTC probe on h173 (n={len(h173)}) ===")
    print(f"  CER (len(ref) denom): macro={ws.mean():.4f}  median={np.median(ws):.4f}")
    print(f"  CER (no-space)      : macro={ns.mean():.4f}  median={np.median(ns):.4f}")
    print(f"  zero-CER items      : {(ns == 0).sum()}/{len(ns)}")
    print(f"  (reference: same model on Thai-SUP TTS = 37.95% avg CER)")

    Path(OUT).parent.mkdir(exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'macro_cer': float(ws.mean()), 'macro_cer_nospace': float(ns.mean()),
                   'n': len(h173), 'details': details}, f, ensure_ascii=False, indent=1)
    print(f"  saved {OUT}")


if __name__ == '__main__':
    main()
