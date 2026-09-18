"""Step-0 baseline: typhoon2.5 + Adapter B + untrained LoRA, eval on h173 + silver1284.

Attribution anchor (AGENTS.md: base-model performance on the eval set must be
known before attributing training effects). Uses the exact quick_eval from the
v3 trainer so numbers are directly comparable to training-time evals.
Appends to u_align_stage2_v3/eval_history.jsonl with step=0.
"""
import warnings
warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v3 as v3
from xlsr_thai import load_xlsr_thai


def main():
    Path(v3.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    skip_silver = '--skip_silver' in sys.argv
    train_data = json.load(open(v3.TRAIN_SPLIT))['samples']  # loaded only for the leakage assert
    eval_full = json.load(open(v3.EVAL_SPLIT))['samples']
    h173 = [s for s in eval_full if s.get('subset') == 'GT_human']
    silver = [s for s in eval_full if s.get('subset') == 'Silver_top20']
    train_wavs = {Path(s['wav_path']).name for s in train_data}
    for name, sub in (('GT_human', h173), ('Silver_top20', silver)):
        assert not (train_wavs & {Path(s['wav_path']).name for s in sub}), f"LEAK {name}"

    encoder, _ = load_xlsr_thai(v3.XLSR_CKPT, device='cuda')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    ckpt = torch.load(v3.ADAPTER_B, map_location='cpu', weights_only=False)
    adapter = v3.UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])

    tokenizer = AutoTokenizer.from_pretrained(v3.LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(v3.LLM_PATH, dtype=torch.bfloat16, device_map='cuda')
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                      lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(base_model, lora)  # untrained LoRA (B=0) = base model behavior

    embed = SentenceTransformer(v3.EMBED_MODEL, device='cpu', model_kwargs={'torch_dtype': torch.bfloat16})

    for name, data in (('h173', h173), ('silver1284', None if skip_silver else silver)):
        if data is None:
            continue
        ser, sims, refs, hyps = v3.quick_eval(adapter, tokenizer, base_model, encoder, embed, data)
        print(f"step0 {name}: SER={ser:.4f} mean_sim={1-ser:.4f}")
        with open(v3.EVAL_HISTORY, 'a') as f:
            f.write(json.dumps({'step': 0, 'subset': name, 'ser': ser, 'sims': sims,
                                'hyps': hyps, 'refs': refs}) + '\n')
        for r, h, s in list(zip(refs, hyps, sims))[:5]:
            print(f"  sim={s:.3f} ref={r[:40]!r} hyp={h[:40]!r}")


if __name__ == '__main__':
    main()
