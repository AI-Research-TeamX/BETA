#!/bin/bash
# Merge each Nash-GRPO run's final checkpoint to HF, then vLLM-eval on the common
# answer metric (ID 200 + OOD 750). Same reward across conditions; external eval
# gives Nash/BR/difficulty breakdowns + clean held-out numbers.
set -u
W=/root/storage/cuisijia.csj/vlmarch/BETA; cd $W
CKPT=$W/results/nashgrpo/ckpt; MERGED=$W/results/nashgrpo/merged; EVAL=$W/eval_results/nashgrpo
mkdir -p $MERGED $EVAL $W/results/nashgrpo/logs
export NO_PROXY=127.0.0.1,localhost

RUNS=(grpo_s0 grpo_s1 grpo_s2 nash4_s0 nash4_s1 nash4_s2 rank_s0 rank_s1 rank_s2 nash1_s0 nash1_s1 nash1_s2 nash16_s0 nash16_s1 nash16_s2)

for r in "${RUNS[@]}"; do
  src=$CKPT/$r/global_step_240/actor; dst=$MERGED/$r
  [ -d "$dst" ] && [ -f "$dst/config.json" ] && { echo "[merge] $r exists"; continue; }
  [ -d "$src" ] || { echo "[merge] MISSING $src"; continue; }
  echo "[merge] $r"
  python3 verl/scripts/legacy_model_merger.py merge --backend fsdp --local_dir "$src" --target_dir "$dst" \
    > $W/results/nashgrpo/logs/merge_$r.log 2>&1
done

declare -a JOBS
for r in "${RUNS[@]}"; do JOBS+=("$r:id"); JOBS+=("$r:ood"); done
run_eval(){ local job=$1 gpu=$2; IFS=':' read -r r kind <<< "$job"
  local m=$MERGED/$r; [ -d "$m" ] || { echo "skip $job"; return; }
  if [ "$kind" = id ]; then bench=gamesolve_bench.jsonl; out=$EVAL/$r; extra="--max_samples 200";
  else bench=gamesolve_ood_bench.jsonl; out=$EVAL/${r}_ood; extra=""; fi
  CUDA_VISIBLE_DEVICES=$gpu python3 vllm_eval/eval.py --model_path "$m" --bench_path $bench \
    --output_dir "$out" --tp 1 --gpu_mem 0.85 $extra > $W/results/nashgrpo/logs/eval_${r}_${kind}.log 2>&1
  echo "[eval done] $job"; }
i=0; ng=8
while [ $i -lt ${#JOBS[@]} ]; do
  pids=()
  for g in $(seq 0 $((ng-1))); do [ $i -lt ${#JOBS[@]} ] || break; run_eval "${JOBS[$i]}" $g & pids+=($!); i=$((i+1)); done
  wait "${pids[@]}"
done
echo "ALL NASHGRPO EVAL COMPLETE"
