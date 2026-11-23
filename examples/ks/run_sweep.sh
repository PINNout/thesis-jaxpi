#!/usr/bin/env bash
mkdir -p logs
for r in s_e8m7 s_e8m8 s_e7m10 s_e6m10 s_e8m6 s_e8m4; do
  n=ks_$r
  [ -f logs/$n.done ] && continue
  echo "$(date '+%F %T') start $n" | tee -a logs/queue.log
  if python3 main.py --config=ladder/$n.py --workdir=. > logs/$n.log 2>&1
  then touch logs/$n.done; s=finished; else s=CRASHED; fi
  echo "$(date '+%F %T') $s $n" | tee -a logs/queue.log
done
echo "$(date '+%F %T') sweep complete" | tee -a logs/queue.log
