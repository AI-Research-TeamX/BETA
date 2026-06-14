#!/bin/bash
# 2-slot parallel orchestrator for Nash-GRPO matrix (15 runs, 4 GPUs each).
# Queue entry: EXP_NAME:ADV:NASH_BETA:SEED
set -u
W=/root/storage/cuisijia.csj/vlmarch/BETA
LOG=$W/results/nashgrpo/logs; mkdir -p $LOG
CKPT=$W/results/nashgrpo/ckpt

# wave1 = grpo_s0 (A) + nash4_s0 (B) to validate the new estimator immediately
QUEUE=(
  "grpo_s0:grpo:4:0"      "nash4_s0:nash_grpo:4:0"
  "grpo_s1:grpo:4:1"      "nash4_s1:nash_grpo:4:1"
  "grpo_s2:grpo:4:2"      "nash4_s2:nash_grpo:4:2"
  "rank_s0:rank_grpo:4:0" "rank_s1:rank_grpo:4:1" "rank_s2:rank_grpo:4:2"
  "nash1_s0:nash_grpo:1:0" "nash1_s1:nash_grpo:1:1" "nash1_s2:nash_grpo:1:2"
  "nash16_s0:nash_grpo:16:0" "nash16_s1:nash_grpo:16:1" "nash16_s2:nash_grpo:16:2"
)

run_one(){ local e=$1 gpus=$2 tmp=$3; IFS=':' read -r name adv beta seed <<< "$e"
  EXP_NAME=$name ADV=$adv NASH_BETA=$beta SEED=$seed \
    GPUS=$gpus N_GPUS=4 RAY_TMP=$tmp STEPS_EPOCHS=2 OUT_DIR=$CKPT/$name \
    bash $W/verl_scripts/run_nashgrpo.sh > $LOG/$name.log 2>&1; echo "FINISHED $name exit=$?"; }

idx=0; n=${#QUEUE[@]}; pA=""; pB=""; nA=""; nB=""
while [ $idx -lt $n ] || [ -n "$pA" ] || [ -n "$pB" ]; do
  if [ -z "$pA" ] && [ $idx -lt $n ]; then nA=${QUEUE[$idx]}; idx=$((idx+1)); run_one "$nA" 0,1,2,3 /tmp/rayA & pA=$!; echo "[A] ${nA%%:*} ($idx/$n)"; sleep 20; fi
  if [ -z "$pB" ] && [ $idx -lt $n ]; then nB=${QUEUE[$idx]}; idx=$((idx+1)); run_one "$nB" 4,5,6,7 /tmp/rayB & pB=$!; echo "[B] ${nB%%:*} ($idx/$n)"; sleep 20; fi
  if [ -n "$pA" ] && ! kill -0 "$pA" 2>/dev/null; then wait "$pA"; echo "[A] done ${nA%%:*}"; pA=""; fi
  if [ -n "$pB" ] && ! kill -0 "$pB" 2>/dev/null; then wait "$pB"; echo "[B] done ${nB%%:*}"; pB=""; fi
  sleep 15
done
echo "ALL NASHGRPO RUNS COMPLETE"
