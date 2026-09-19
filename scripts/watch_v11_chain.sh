#!/bin/bash
# Watch v11 training; on exit run the post-sequence automatically:
# rescore h173 eval points -> silver1284 final eval -> rescore again.
PY=/data/workspace/venvs/qwen3-asr/bin/python
R=/data/workspace/asr-model-training/thai-understanding

V11PID=$(pgrep -f "qwen3-asr/bin/python /data/workspace/asr-model-training/thai-understanding/src/train_llm_denoise_v11[.]py" | head -1)
echo "watching v11 pid=$V11PID at $(date)"
while [ -n "$V11PID" ] && kill -0 "$V11PID" 2>/dev/null; do sleep 60; done
echo "v11 exited at $(date)"
tail -3 "$R/u_align_stage2_v11/train.log"
grep -q "v11 done" "$R/u_align_stage2_v11/train.log" && echo "TRAIN_COMPLETED_MARKER=yes" || echo "TRAIN_COMPLETED_MARKER=no"

echo "== [1/3] rescore h173 eval_history =="
$PY "$R/src/f2llm_remote.py" --rescore --file "$R/u_align_stage2_v11/eval_history.jsonl"

echo "== [2/3] silver1284 final eval =="
$PY "$R/src/eval_final_cascade.py" --module train_llm_denoise_v11 --subset silver1284

echo "== [3/3] rescore silver line =="
$PY "$R/src/f2llm_remote.py" --rescore --file "$R/u_align_stage2_v11/eval_history.jsonl"

echo "CHAIN_DONE $(date)"
