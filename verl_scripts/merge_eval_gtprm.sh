#!/bin/bash
# After GT-PRM training: merge each run's final checkpoint to HF, then vLLM-eval
# every checkpoint on the SAME held-out answer-quality metric (ID 200 + OOD 750).
# Cross-condition comparison MUST use this common metric (training rewards differ).
set -u
W=/root/storage/cuisijia.csj/vlmarch/BETA; cd $W
CKPT=$W/results/gtprm/ckpt
MERGED=$W/results/gtprm/merged
EVAL=$W/eval_results/gtprm
mkdir -p $MERGED $EVAL
export NO_PROXY=127.0.0.1,localhost

RUNS=(orm_s0 orm_s1 orm_s2 proc_s0 proc_s1 proc_s2 hyb_s0 hyb_s1 hyb_s2 typed_s0 typed_s1 typed_s2)

# 1) merge (sequential; CPU-bound)
for r in "${RUNS[@]}"; do
  src=$CKPT/$r/global_step_240/actor
  dst=$MERGED/$r
  if [ -d "$dst" ] && [ -f "$dst/config.json" ]; then echo "[merge] $r exists"; continue; fi
  if [ ! -d "$src" ]; then echo "[merge] MISSING $src"; continue; fi
  echo "[merge] $r"
  python3 verl/scripts/legacy_model_merger.py merge --backend fsdp \
    --local_dir "$src" --target_dir "$dst" > $W/results/gtprm/logs/merge_$r.log 2>&1
done

# 2) parallel eval: 8 GPUs, queue of (run, bench) jobs
declare -a JOBS
for r in "${RUNS[@]}"; do JOBS+=("$r:id"); JOBS+=("$r:ood"); done

run_eval(){ local job=$1 gpu=$2; IFS=':' read -r r kind <<< "$job"
  local m=$MERGED/$r; [ -d "$m" ] || { echo "skip $job (no merge)"; return; }
  if [ "$kind" = id ]; then bench=gamesolve_bench.jsonl; out=$EVAL/$r; extra="--max_samples 200";
  else bench=gamesolve_ood_bench.jsonl; out=$EVAL/${r}_ood; extra=""; fi
  CUDA_VISIBLE_DEVICES=$gpu python3 vllm_eval/eval.py --model_path "$m" --bench_path $bench \
    --output_dir "$out" --tp 1 --gpu_mem 0.85 $extra > $W/results/gtprm/logs/eval_${r}_${kind}.log 2>&1
  echo "[eval done] $job"; }

i=0; ng=8
while [ $i -lt ${#JOBS[@]} ]; do
  pids=()
  for g in $(seq 0 $((ng-1))); do
    [ $i -lt ${#JOBS[@]} ] || break
    run_eval "${JOBS[$i]}" $g & pids+=($!); i=$((i+1))
  done
  wait "${pids[@]}"
done
echo "ALL GTPRM EVAL COMPLETE"
