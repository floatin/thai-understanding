"""Evaluate the FINAL checkpoint (step_322) on h173 — completes the training
trend curve (step0 / step200 / step322). Run after training exits.
"""
import warnings
warnings.filterwarnings("ignore")
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
import train_u_align_stage2_v4 as v3
from xlsr_thai import load_xlsr_thai


def main():
    ckpt_dir = sorted(Path(v3.OUTPUT_DIR, 'checkpoints').glob('step_*'),
                      key=lambda p: int(p.name.split('_')[1]))[-1]
    step = int(ckpt_dir.name.split('_')[1])
    print(f"evaluating final checkpoint: {ckpt_dir} (step {step})")

    eval_full = json.load(open(v3.EVAL_SPLIT))['samples']
    h173 = [s for s in eval_full if s.get('subset') == 'GT_human']

    encoder, _ = load_xlsr_thai(v3.XLSR_CKPT, device='cuda')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    ckpt = torch.load(v3.ADAPTER_B, map_location='cpu', weights_only=False)
    adapter = v3.UAlignAdapter(encoder_dim=1024, llm_dim=2560, downsample=2).cuda().to(torch.bfloat16)
    adapter.load_state_dict(ckpt['adapter_state_dict'])
    # trained adapter weights from the checkpoint
    trained = torch.load(ckpt_dir / 'adapter.pt', map_location='cpu', weights_only=False)
    adapter.load_state_dict(trained['adapter_state_dict'])

    tokenizer = AutoTokenizer.from_pretrained(v3.LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(v3.LLM_PATH, dtype=torch.bfloat16, device_map='cuda')
    model = PeftModel.from_pretrained(base_model, ckpt_dir / 'lora', is_trainable=False)

    embed = SentenceTransformer(v3.EMBED_MODEL, device='cpu', model_kwargs={'torch_dtype': torch.bfloat16})
    ser, sims, refs, hyps = v3.quick_eval(adapter, tokenizer, base_model, encoder, embed, h173)
    print(f"FINAL step {step} h173: SER={ser:.4f} mean_sim={1-ser:.4f}")
    with open(v3.EVAL_HISTORY, 'a') as f:
        f.write(json.dumps({'step': step, 'subset': 'h173', 'ser': ser, 'sims': sims,
                            'hyps': hyps, 'refs': refs}) + '\n')


if __name__ == '__main__':
    main()
