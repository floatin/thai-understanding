"""Final-checkpoint eval for the cascade versions (v7+): text-only, no audio.

Loads the latest LoRA checkpoint of a given cascade version, generates on the
requested eval subset, appends an eval_history line (sims left empty for the
remote F2LLM rescore pass). Usage:
  $PY src/eval_final_cascade.py --module v11 --subset silver1284
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import importlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', default='train_llm_denoise_v11')
    ap.add_argument('--subset', choices=['silver1284', 'h173'], default='silver1284')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    v = importlib.import_module(args.module)
    ckpts = sorted(Path(v.OUTPUT_DIR, 'checkpoints').glob('step_*'),
                   key=lambda p: int(p.name.split('_')[1]))
    ckpt_dir = ckpts[-1]
    step = int(ckpt_dir.name.split('_')[1])
    print(f"evaluating {args.module} final checkpoint: {ckpt_dir} (step {step})")

    eval_full = json.load(open(v.EVAL_SPLIT))['samples']
    data = [s for s in eval_full if s.get('subset') ==
            ('GT_human' if args.subset == 'h173' else 'Silver_top20')]
    if args.limit:
        data = data[:args.limit]

    drafts = json.load(open(v.DRAFTS))
    tokenizer = AutoTokenizer.from_pretrained(v.LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(v.LLM_PATH, dtype=torch.bfloat16, device_map='cuda')
    base_model.config.use_cache = True
    model = PeftModel.from_pretrained(base_model, ckpt_dir / 'lora', is_trainable=False)
    embed_layer = base_model.get_input_embeddings()

    refs, hyps = v.quick_eval(model, tokenizer, embed_layer, data, drafts)
    uniq = len(set(hyps))
    top1 = max([hyps.count(h) for h in set(hyps)], default=0)
    print(f"FINAL {args.module} step {step} {args.subset}: n={len(hyps)} "
          f"[diversity: {uniq} unique, top1 {top1} ({top1/len(hyps):.0%})]")
    with open(v.EVAL_HISTORY, 'a') as f:
        f.write(json.dumps({'step': step, 'subset': args.subset, 'ser': None,
                            'n_unique': uniq, 'top1_share': round(top1 / max(1, len(hyps)), 3),
                            'sims': None, 'hyps': hyps, 'refs': refs}) + '\n')
    print("line appended — run f2llm_remote.py --rescore to score it")


if __name__ == '__main__':
    main()
