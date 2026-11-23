#!/usr/bin/env bash
# Queue for the microscaling (MX) runs. Safe to start again: finished
# runs are skipped, and a second copy refuses to start.
#
#   nohup ./run_mx.sh > /dev/null 2>&1 &
#   tail -f overnight_mx.log
#
ROOT=$(cd "$(dirname "$0")" && pwd)
LOG="$ROOT/overnight_mx.log"
exec 9> "$ROOT/.overnight.lock"   # shared lock: never runs alongside run_overnight.sh
flock -n 9 || { echo "$(date '+%F %T') another queue is already running" >> "$LOG"; exit 1; }
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

GEN=""
for g in "$ROOT/make_ladder.py" "$ROOT/ks/make_ladder.py"; do [ -f "$g" ] && GEN="$g" && break; done
[ -n "$GEN" ] || { say "ERROR: make_ladder.py not found in $ROOT or $ROOT/ks"; exit 1; }
grep -q -- "--mx" "$GEN" && grep -q allen_cahn "$GEN" || { say "ERROR: $GEN is an old version"; exit 1; }
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

say "=== MX queue started, generator $GEN"

# MX on the five PDEs that use reverse-mode derivatives (not KS: Taylor mode).
# Order: one reference per PDE, then the formats from most to least precise.
ELEMS="e4m3 e5m2 int8 e2m3 e3m2 e2m1"
for pde in burgers allen_cahn ns_cyl adv ldc; do
  prepare $pde || continue
  d="$ROOT/$(folder_of $pde)"
  (cd "$d" && python3 "$GEN" "$pde" --mx $ELEMS > /dev/null)
  if [ ! -f "$d/logs/smoke_mx_ok" ]; then
    if (cd "$d" && PI_SMOKE=1 python3 main.py --config="ladder/${pde}_s_mx_e4m3.py" --workdir=. > logs/smoke_mx.log 2>&1)
    then touch "$d/logs/smoke_mx_ok"; else say "SKIP $pde: MX smoke test failed (see $(folder_of $pde)/logs/smoke_mx.log)"; continue; fi
  fi
  run $pde s_e6m9
  for e in $ELEMS; do run $pde s_mx_$e; done
done

say "=== MX queue complete"
