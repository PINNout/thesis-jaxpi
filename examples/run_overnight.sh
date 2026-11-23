#!/usr/bin/env bash
# Overnight queue for the precision experiments. Safe to start again: finished
# runs are skipped, and a second copy refuses to start.
#
#   nohup ./run_overnight.sh > /dev/null 2>&1 &
#   tail -f overnight.log
#
ROOT=$(cd "$(dirname "$0")" && pwd)
LOG="$ROOT/overnight.log"
exec 9> "$ROOT/.overnight.lock"
flock -n 9 || { echo "$(date '+%F %T') another queue is already running" >> "$LOG"; exit 1; }
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

GEN=""
for g in "$ROOT/make_ladder.py" "$ROOT/ks/make_ladder.py"; do [ -f "$g" ] && GEN="$g" && break; done
[ -n "$GEN" ] || { say "ERROR: make_ladder.py not found in $ROOT or $ROOT/ks"; exit 1; }
grep -q -- "--sweep" "$GEN" && grep -q allen_cahn "$GEN" || { say "ERROR: $GEN is an old version"; exit 1; }
python3 -c "import jax, jaxpi" 2>/dev/null || { say "ERROR: jax/jaxpi not importable (wrong terminal? use the .venv one)"; exit 1; }

# name used by make_ladder.py -> folder under examples/ (first one that exists)
folder_of() {
  case "$1" in
    ks)         c="ks" ;;
    burgers)    c="burgers" ;;
    adv)        c="adv advection" ;;
    allen_cahn) c="allen_cahn allen-cahn ac" ;;
    ldc)        c="ldc" ;;
    ns_cyl)     c="ns_cyl ns_steady_cylinder ns_cylinder cylinder" ;;
  esac
  for d in $c; do [ -f "$ROOT/$d/main.py" ] && { echo "$d"; return; }; done
}

# run one config:  run <pde> <run-name> [seed]
run() {
  local pde=$1 r=$2 seed=${3:-} dir n tag
  dir=$(folder_of "$pde"); [ -n "$dir" ] || return
  n=${pde}_$r; tag=$n${seed:+_s$seed}
  cd "$ROOT/$dir" && mkdir -p logs
  [ -f "ladder/$n.py" ] || { say "MISSING config $dir/ladder/$n.py"; return; }
  [ -f "logs/$tag.done" ] && return
  say "start    $tag"
  if env ${seed:+PI_SEED=$seed} python3 main.py --config="ladder/$n.py" --workdir=. > "logs/$tag.log" 2>&1
  then touch "logs/$tag.done"; say "finished $tag"; else say "CRASHED  $tag (see $dir/logs/$tag.log)"; fi
}

# generate configs and smoke-test a PDE; returns 1 if it cannot run
prepare() {
  local pde=$1 dir; dir=$(folder_of "$pde")
  [ -n "$dir" ] || { say "SKIP $pde: no folder with main.py found"; return 1; }
  cd "$ROOT/$dir" && mkdir -p logs
  python3 "$GEN" "$pde" > /dev/null && python3 "$GEN" "$pde" --sweep e6m9 e8m7 > /dev/null \
    || { say "SKIP $pde: config generation failed"; return 1; }
  [ -f logs/smoke_ok ] && return 0
  if PI_SMOKE=1 python3 main.py --config="ladder/${pde}_r2_fwd_full.py" --workdir=. > logs/smoke.log 2>&1
  then touch logs/smoke_ok; else say "SKIP $pde: smoke test failed (see $dir/logs/smoke.log)"; return 1; fi
}

say "=== queue started, generator $GEN"

# 1. KS: every part of training in the 6/9 format
if prepare ks; then
  (cd "$ROOT/$(folder_of ks)" && python3 "$GEN" ks --fmt e6m9 --acc-chunk 16 > /dev/null)
  for r in e6m9_r3_acc e6m9_r4_grads_res e6m9_r5n_params_nearest e6m9_r5s_params_stoch; do run ks $r; done
  # 2. KS: two more seeds for the reference and the 6/9 run
  for s in 1 2; do run ks r0_fp32 $s; run ks s_e6m9 $s; done
fi

# 3. The other PDEs: FP32, FP16 forward-only, FP16 all values, 6/9, BF16
for pde in burgers adv allen_cahn ldc ns_cyl; do
  prepare $pde || continue
  for r in r0_fp32 r1_fwd_ste r2_fwd_full s_e6m9 s_e8m7; do run $pde $r; done
done

# 4. Burgers: exact accumulator (rounds after every product; slow), 6/9 then FP16
if prepare burgers; then
  (cd "$ROOT/$(folder_of burgers)" && python3 "$GEN" burgers --fmt e6m9 > /dev/null)
  run burgers e6m9_r3_acc
  run burgers r3_acc
fi

say "=== queue complete"
