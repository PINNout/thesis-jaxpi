"""
Summarise every finished sweep run from its log into one table.

    python3 collect.py                 # prints the table, writes examples/summary.csv
    python3 collect.py --logdir smoke_logs   # (debug) read other log folders

For each <example>/runs/<name>.log it reports the accuracy metric (l2_error, or
u_error for the NS cylinder), the final value, the best value and where it
happened, the first logged step with NaN/inf, and the wall time from sweep_results.csv.
LDC logs restart the iteration counter for every Reynolds number, so those steps are
shown as  stage:iter  (e.g. 2:1500 = stage 2 = Re 400, iteration 1500).
"""
import csv, glob, math, os, re, sys

ROOT = os.environ.get("JAXPI_EXAMPLES", os.path.expanduser("~/thesis/jaxpi/examples"))
LOGDIR = "runs"
if "--logdir" in sys.argv:
    LOGDIR = sys.argv[sys.argv.index("--logdir") + 1]

HDR = re.compile(r"Iter:\s*(\d+)\s+Time:")
ROW = re.compile(r"\]\s+([A-Za-z_0-9]+)\s+(\S+)\s*$")
PRIMARY = ["l2_error", "u_error"]


def parse(path):
    """-> list of (stage, iter, {metric: value})"""
    recs, cur, stage, last_iter = [], None, 1, -1
    for line in open(path, errors="ignore"):
        m = HDR.search(line)
        if m:
            it = int(m.group(1))
            if it < last_iter:
                stage += 1
            last_iter = it
            cur = (stage, it, {})
            recs.append(cur)
            continue
        if cur is not None:
            r = ROW.search(line)
            if r and not r.group(1).startswith("-"):
                try:
                    cur[2][r.group(1)] = float(r.group(2))
                except ValueError:
                    pass
    return recs


def wall_times():
    out = {}
    p = os.path.join(ROOT, "sweep_results.csv")
    if os.path.isfile(p):
        for row in csv.DictReader(open(p)):
            out[row["run"]] = (row["exit_code"], row["wall_seconds"])
    return out


rows, wt = [], wall_times()
for log in sorted(glob.glob(os.path.join(ROOT, "*", LOGDIR, "*.log"))):
    name = os.path.basename(log)[:-4]
    recs = parse(log)
    if not recs:
        continue
    key = next((k for k in PRIMARY if k in recs[-1][2]), None)
    series = [((s, i), d[key]) for s, i, d in recs if key in d]
    bad = [(s, i) for s, i, d in recs if any(not math.isfinite(v) for v in d.values())]
    finite = [(si, v) for si, v in series if math.isfinite(v)]
    best = min(finite, key=lambda t: t[1]) if finite else (None, float("nan"))
    final = series[-1][1] if series else float("nan")
    fmt = lambda si: "-" if si is None else (f"{si[0]}:{si[1]}" if recs[-1][0] > 1 else str(si[1]))
    rows.append(dict(
        run=name, metric=key or "?", last_step=fmt(series[-1][0]) if series else "-",
        final=final, best=best[1], best_step=fmt(best[0]),
        final_over_best=(final / best[1]) if finite and best[1] > 0 and math.isfinite(final) else float("nan"),
        first_nan=fmt(bad[0]) if bad else "-",
        exit_code=wt.get(name, ("?", "?"))[0], wall_s=wt.get(name, ("?", "?"))[1]))

cols = ["run", "metric", "last_step", "final", "best", "best_step", "final_over_best", "first_nan", "exit_code", "wall_s"]
with open(os.path.join(ROOT, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def cell(v):
    return f"{v:.3e}" if isinstance(v, float) else str(v)


widths = [max(len(c), *(len(cell(r[c])) for r in rows)) if rows else len(c) for c in cols]
print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
for r in rows:
    print("  ".join(cell(r[c]).ljust(w) for c, w in zip(cols, widths)))
print(f"\nwritten: {os.path.join(ROOT, 'summary.csv')}")
print("final_over_best >> 1 means the error got worse again after its best point.")
