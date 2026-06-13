#!/bin/bash
# 2-slot parallel orchestrator for the probe ablation matrix.
# Slot A = GPUs 0-3 (ray /tmp/rayA), Slot B = GPUs 4-7 (ray /tmp/rayB).
# Each queue entry: EXP_NAME:PROBE_ENABLED:PROBE_LAYER:SEED
set -u
W=/root/storage/cuisijia.csj/vlmarch/BETA
LOGDIR=$W/results/phase2/ablation_logs
mkdir -p $LOGDIR

# queue (15 runs). no-probe layer field ignored.
QUEUE=(
  "np_s0:0:17:0"  "np_s1:0:17:1"  "np_s2:0:17:2"
  "p17_s0:1:17:0" "p17_s1:1:17:1" "p17_s2:1:17:2"
  "p6_s0:1:6:0"   "p6_s1:1:6:1"   "p6_s2:1:6:2"
  "p30_s0:1:30:0" "p30_s1:1:30:1" "p30_s2:1:30:2"
  "p35_s0:1:35:0" "p35_s1:1:35:1" "p35_s2:1:35:2"
)

run_one() {
  local entry=$1 gpus=$2 ntmp=$3
  IFS=':' read -r name pe pl seed <<< "$entry"
  EXP_NAME=$name PROBE_ENABLED=$pe PROBE_LAYER=$pl SEED=$seed \
    GPUS=$gpus N_GPUS=4 RAY_TMP=$ntmp STEPS_EPOCHS=2 \
    OUT_DIR=/tmp/ablation_ckpt/$name \
    bash $W/verl_scripts/run_probe_ablation.sh > $LOGDIR/$name.log 2>&1
  echo "FINISHED $name (exit $?)"
}

idx=0
n=${#QUEUE[@]}
pidA=""; pidB=""; nameA=""; nameB=""
while [ $idx -lt $n ] || [ -n "$pidA" ] || [ -n "$pidB" ]; do
  # launch into free slot A
  if [ -z "$pidA" ] && [ $idx -lt $n ]; then
    nameA=${QUEUE[$idx]}; idx=$((idx+1))
    run_one "$nameA" "0,1,2,3" /tmp/rayA &
    pidA=$!
    echo "[A] launched ${nameA%%:*} pid=$pidA ($idx/$n)"
    sleep 20   # stagger ray startup to avoid races
  fi
  if [ -z "$pidB" ] && [ $idx -lt $n ]; then
    nameB=${QUEUE[$idx]}; idx=$((idx+1))
    run_one "$nameB" "4,5,6,7" /tmp/rayB &
    pidB=$!
    echo "[B] launched ${nameB%%:*} pid=$pidB ($idx/$n)"
    sleep 20
  fi
  # poll
  if [ -n "$pidA" ] && ! kill -0 "$pidA" 2>/dev/null; then
    wait "$pidA"; echo "[A] done ${nameA%%:*}"; pidA=""
  fi
  if [ -n "$pidB" ] && ! kill -0 "$pidB" 2>/dev/null; then
    wait "$pidB"; echo "[B] done ${nameB%%:*}"; pidB=""
  fi
  sleep 15
done
echo "ALL ABLATION RUNS COMPLETE"
