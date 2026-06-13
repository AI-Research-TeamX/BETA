#!/bin/bash
# 2-slot parallel orchestrator for GT-PRM matrix (12 runs, 4 GPUs each).
# Queue entry: EXP_NAME:MODE:VARIANT:SEED
set -u
W=/root/storage/cuisijia.csj/vlmarch/BETA
LOG=$W/results/gtprm/logs; mkdir -p $LOG
CKPT=$W/results/gtprm/ckpt

QUEUE=(
  "orm_s0:outcome:binary:0"   "orm_s1:outcome:binary:1"   "orm_s2:outcome:binary:2"
  "proc_s0:process:binary:0"  "proc_s1:process:binary:1"  "proc_s2:process:binary:2"
  "hyb_s0:hybrid:binary:0"    "hyb_s1:hybrid:binary:1"    "hyb_s2:hybrid:binary:2"
  "typed_s0:process:typed:0"  "typed_s1:process:typed:1"  "typed_s2:process:typed:2"
)

run_one(){ local e=$1 gpus=$2 tmp=$3; IFS=':' read -r name mode var seed <<< "$e"
  EXP_NAME=$name GT_PRM_MODE=$mode GT_PRM_VARIANT=$var SEED=$seed \
    GPUS=$gpus N_GPUS=4 RAY_TMP=$tmp STEPS_EPOCHS=2 OUT_DIR=$CKPT/$name \
    bash $W/verl_scripts/run_gtprm.sh > $LOG/$name.log 2>&1; echo "FINISHED $name exit=$?"; }

idx=0; n=${#QUEUE[@]}; pA=""; pB=""; nA=""; nB=""
while [ $idx -lt $n ] || [ -n "$pA" ] || [ -n "$pB" ]; do
  if [ -z "$pA" ] && [ $idx -lt $n ]; then nA=${QUEUE[$idx]}; idx=$((idx+1)); run_one "$nA" 0,1,2,3 /tmp/rayA & pA=$!; echo "[A] ${nA%%:*} ($idx/$n)"; sleep 20; fi
  if [ -z "$pB" ] && [ $idx -lt $n ]; then nB=${QUEUE[$idx]}; idx=$((idx+1)); run_one "$nB" 4,5,6,7 /tmp/rayB & pB=$!; echo "[B] ${nB%%:*} ($idx/$n)"; sleep 20; fi
  if [ -n "$pA" ] && ! kill -0 "$pA" 2>/dev/null; then wait "$pA"; echo "[A] done ${nA%%:*}"; pA=""; fi
  if [ -n "$pB" ] && ! kill -0 "$pB" 2>/dev/null; then wait "$pB"; echo "[B] done ${nB%%:*}"; pB=""; fi
  sleep 15
done
echo "ALL GTPRM RUNS COMPLETE"
