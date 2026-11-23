#!/usr/bin/env bash
# Runs the KS precision ladder in order. A failed run does not stop the queue.
mkdir -p logs
for r in r0_fp32 r1_fwd_ste r2_fwd_full r3_acc r4_grads_res \
         r5n_params_nearest r5s_params_stoch x_e8m10 m8_r2 m6_r2 m4_r2; do
  n=ks_$r
  [ -f logs/$n.done ] && { echo "skip $n (already done)"; continue; }
  echo "$(date '+%F %T') start $n" | tee -a logs/queue.log
  if python3 main.py --config=ladder/$n.py --workdir=. > logs/$n.log 2>&1
  then touch logs/$n.done; s=finished; else s="CRASHED (see logs/$n.log)"; fi
  echo "$(date '+%F %T') $s $n" | tee -a logs/queue.log
done
echo "$(date '+%F %T') queue complete" | tee -a logs/queue.log
