#!/bin/bash
# Watch v12 training; on exit: rescore h173 -> silver1284 final eval -> rescore.
PY=/data/workspace/venvs/qwen3-asr/bin/python
R=/data/workspace/asr-model-training/thai-understanding

V12PID=$(pgrep -f "qwen3-asr/bin/python /data/workspace/asr-model-training/thai-understanding/src/train_llm_denoise_v12[.]py" | head -1)
echo "watching v12 pid=$V12PID at $(date)"
while [ -n "$V12PID" ] && kill -0 "$V12PID" 2>/dev/null; do sleep 60; done
echo "v12 exited at $(date)"
tail -3 "$R/u_align_stage2_v12/train.log"
grep -q "steps=" "$R/u_align_stage2_v12/train.log" && echo "TRAIN_COMPLETED_MARKER=yes" || echo "TRAIN_COMPLETED_MARKER=no"

echo "== [1/3] rescore h173 eval_history =="
$PY "$R/src/f2llm_remote.py" --rescore --file "$R/u_align_stage2_v12/eval_history.jsonl"

echo "== [2/3] silver1284 final eval =="
$PY "$R/src/eval_final_cascade.py" --module train_llm_denoise_v12 --subset silver1284

echo "== [3/3] rescore silver line =="
$PY "$R/src/f2llm_remote.py" --rescore --file "$R/u_align_stage2_v12/eval_history.jsonl"

echo "CHAIN_DONE $(date)"
