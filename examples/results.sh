#!/usr/bin/env bash
# Summary of every finished run: NaN count and the last logged error values.
ROOT=$(cd "$(dirname "$0")" && pwd)
tail -3 "$ROOT/overnight.log" 2>/dev/null; echo
for d in "$ROOT"/*/logs; do
  ls "$d"/*.done > /dev/null 2>&1 || continue
  echo "## $(basename "$(dirname "$d")")"
  for f in $(ls -tr "$d"/*.done); do
    l=${f%.done}.log
    # the last logging block: every metric whose name contains "error"
    errs=$(tac "$l" | awk '/Iter:/{exit} /error/{print $(NF-1)"="$NF}' | tac | tr '\n' ' ')
    last4=$(grep -iE "l2_error|l2_err" "$l" | tail -4 | awk '{print $NF}' | tr '\n' ' ')
    printf "%-34s nan-lines=%-5s %s| last4: %s\n" "$(basename "${f%.done}")" "$(grep -ci nan "$l")" "$errs" "$last4"
  done; echo
done
