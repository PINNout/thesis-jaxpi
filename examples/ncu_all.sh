#!/usr/bin/env bash
# Nsight Compute hardware-counter profile of every PDE x precision run (about 3 training steps each).
# Run from the examples folder:   cd ~/thesis/jaxpi/examples && ./ncu_all.sh
# Output per run:  <example>/profiles/<run>_ncu_all.csv   (+ .log).  Runs that already have a CSV are skipped,
# so you can stop it (Ctrl+C) and start it again.
#
# Optional settings (environment variables):
#   EXAMPLES="adv ks"            only these examples        (default: all six)
#   TAGS="fp32 fp16_native"      only these precision tags  (default: fp32 fp16 int8 fp16_native)
#   COVER=3.3                    how many training steps of kernels to profile per run
#   PYTHON=python3
set -u
PY=${PYTHON:-python3}
ROOT=$(pwd)
EXAMPLES=${EXAMPLES:-"adv burgers allen_cahn ks ldc ns_steady_cylinder"}
TAGS=${TAGS:-"fp32 fp16 int8 fp16_native"}
COVER=${COVER:-3.3}
declare -A EQ=( [adv]=adv [burgers]=burgers [allen_cahn]=allen_cahn [ks]=ks [ldc]=ldc [ns_steady_cylinder]=ns_cyl )
METRICS=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size
export PI_PROFILE=1 WANDB_MODE=disabled
ok=0; skipped=0; failed=0
for ex in $EXAMPLES; do
  [ -d "$ROOT/$ex" ] || { echo "no folder $ex, skipping"; continue; }
  cd "$ROOT/$ex" || continue
  for tag in $TAGS; do
    name=${EQ[$ex]}_$tag
    out=profiles/${name}_ncu_all.csv
    if [ -s "$out" ]; then echo "[skip] $name (already done)"; skipped=$((skipped+1)); continue; fi
    if [ ! -f "benchmark_configs/$name.py" ]; then echo "[skip] $name (no config benchmark_configs/$name.py)"; continue; fi
    # kernels per training step = total kernel launches in the nsys profile / 40 steps
    launches=$($PY - "$name" "$COVER" <<'PYEOF'
import csv, math, sys
name, cover = sys.argv[1], float(sys.argv[2])
try:
    rows = list(csv.DictReader(open(f"profiles/{name}/nsys_kernels_cuda_gpu_kern_sum.csv")))
    per_step = sum(int(float(r["Instances"])) for r in rows) / 40
    print(max(800, min(3000, math.ceil(per_step * cover))))
except Exception:
    print(1500)
PYEOF
)
    echo "[run ] $name  ($launches kernel launches)  $(date +%H:%M:%S)"
    rm -rf "./profiles/${name}_ncu_wd"
    ncu --csv --log-file "$out" --profile-from-start off --target-processes all \
        --launch-count "$launches" --metrics "$METRICS" \
        $PY profile_main.py --config="benchmark_configs/$name.py" --workdir="./profiles/${name}_ncu_wd" \
        > "profiles/${name}_ncu_all.log" 2>&1
    rc=$?
    rm -rf "./profiles/${name}_ncu_wd"
    if [ $rc -ne 0 ] || [ ! -s "$out" ] || ! grep -q dram__bytes_read "$out"; then
      echo "[FAIL] $name (exit $rc) - see $ex/profiles/${name}_ncu_all.log"; rm -f "$out"; failed=$((failed+1))
    else
      echo "[ ok ] $name  $(grep -c dram__bytes_read.sum "$out") kernels profiled"; ok=$((ok+1))
    fi
  done
  cd "$ROOT"
done
echo; echo "finished: $ok ok, $skipped skipped, $failed failed  ($(date +%H:%M:%S))"
mkdir -p "$ROOT/ncu_all"
for f in "$ROOT"/*/profiles/*_ncu_all.csv; do [ -s "$f" ] && cp "$f" "$ROOT/ncu_all/"; done
echo "all CSVs copied to $ROOT/ncu_all/   ->  attach that folder's files, or:  cd $ROOT && zip -j ncu_all.zip ncu_all/*.csv"
