"""Decode-strategy A/B on the final checkpoint: is the 96% loop-collapse a
decoding artifact? Greedy (baseline) vs repetition_penalty=1.15 on h173.

Zero-training diagnostic (AGENTS.md §4.6: diagnose before concluding/retuning).
Saves per-item to out_metric_gate/decode_ab.json (not eval_history — different
generation config, must not mix with the training-loop series).
"""
import warnings
warnings.filterwarnings("ignore")
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v3 as v3
from xlsr_thai import load_xlsr_thai, extract_features

OUT = '/data/workspace/asr-model-training/thai-understanding/out_metric_gate/decode_ab.json'


def generate_all(adapter, tokenizer, base_model, encoder, data, gen_kwargs):
    prompt_ids = tokenizer(v3.GEN_PROMPT, return_tensors='pt', add_special_tokens=False).input_ids[0].cuda()
    prompt_embeds = base_model.get_input_embeddings()(prompt_ids).to(torch.bfloat16)
    base_model.config.use_cache = True
    hyps = []
    for s in data:
        try:
            audio = v3.load_audio(s['wav_path'])
            audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
            length = torch.tensor([audio_t.shape[1]]).cuda()
            with torch.no_grad():
                feats, _ = extract_features(encoder, audio_t, length)
                se = adapter(feats.to(torch.bfloat16))
                inputs_embeds = torch.cat([se.squeeze(0), prompt_embeds], dim=0).unsqueeze(0)
                out = base_model.generate(inputs_embeds=inputs_embeds,
                                          pad_token_id=tokenizer.eos_token_id, **gen_kwargs)
            hyps.append(v3.strip_think(tokenizer.decode(out[0], skip_special_tokens=True).strip()))
        except Exception:
            hyps.append("")
    base_model.config.use_cache = False
    return hyps


def loop_rate(hyps):
    n = sum(1 for h in hyps if h and re.search(r'(.{1,4})\1{4,}', h))
    return n / max(1, len([h for h in hyps if h]))


def main():
    ckpt_dir = sorted(Path(v3.OUTPUT_DIR, 'checkpoints').glob('step_*'),
                      key=lambda p: int(p.name.split('_')[1]))[-1]
    print(f"checkpoint: {ckpt_dir}")
    eval_full = json.load(open(v3.EVAL_SPLIT))['samples']
    h173 = [s for s in eval_full if s.get('subset') == 'GT_human']

    encoder, _ = load_xlsr_thai(v3.XLSR_CKPT, device='cuda')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    adapter = v3.UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    trained = torch.load(ckpt_dir / 'adapter.pt', map_location='cpu', weights_only=False)
    adapter.load_state_dict(trained['adapter_state_dict'])

    tokenizer = AutoTokenizer.from_pretrained(v3.LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(v3.LLM_PATH, dtype=torch.bfloat16, device_map='cuda')
    model = PeftModel.from_pretrained(base_model, ckpt_dir / 'lora', is_trainable=False)
    embed = SentenceTransformer(v3.EMBED_MODEL, device='cpu', model_kwargs={'torch_dtype': torch.bfloat16}).to('cuda')

    refs = [s['ref'] for s in h173]
    out = {'ckpt': str(ckpt_dir), 'refs': refs}
    for name, kw in (('greedy', dict(max_new_tokens=64, do_sample=False)),
                     ('rep_pen_1.15', dict(max_new_tokens=64, do_sample=False, repetition_penalty=1.15))):
        t0 = time.time()
        hyps = generate_all(adapter, tokenizer, base_model, encoder, h173, kw)
        torch.cuda.empty_cache()
        sims = v3.compute_embed_sim_batch(embed, refs, hyps)
        ser = v3.compute_ser_continuous(sims)
        print(f"{name:>14s}: SER={ser:.4f} mean_sim={1-ser:.4f} loop_rate={loop_rate(hyps):.0%}  ({time.time()-t0:.0f}s)")
        out[name] = {'ser': ser, 'sims': [round(float(s), 4) for s in sims], 'hyps': hyps}
    Path(OUT).parent.mkdir(exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved {OUT}")
    print("\nsample comparison:")
    for i in range(3):
        print(f"  REF: {refs[i][:50]!r}")
        print(f"  greedy   : {out['greedy']['hyps'][i][:50]!r}")
        print(f"  rep_pen  : {out['rep_pen_1.15']['hyps'][i][:50]!r}\n")


if __name__ == '__main__':
    main()
