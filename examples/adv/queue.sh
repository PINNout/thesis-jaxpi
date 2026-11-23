#!/usr/bin/env bash
# usage: ./queue.sh adv_4x128_50k_fp32_s0 adv_4x128_50k_fp32_s1 ...
cd ~/thesis/jaxpi/examples/adv
source ~/thesis/jaxpi/.venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=${WANDB_MODE:-online}
mkdir -p runs

for name in "$@"; do
  if [ -d "runs/$name" ]; then echo "SKIP $name (runs/$name exists)"; continue; fi
  if [ ! -f "benchmark_configs/$name.py" ]; then echo "SKIP $name (no config)"; continue; fi
  echo "=== $(date '+%F %T') START $name ==="
  start=$(date +%s)
  python3 main.py --config=benchmark_configs/$name.py --workdir=./runs/$name \
    > runs/$name.log 2>&1
  status=$?
  echo "=== $(date '+%F %T') END $name exit=$status time=$(( $(date +%s) - start ))s ===" | tee -a runs/queue.log
done
