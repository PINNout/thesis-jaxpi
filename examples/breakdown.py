"""
Kernel-time breakdown by operation type for every profiled run.

    python3 breakdown.py                 # reads <example>/profiles/<run>/nsys_kernels_cuda_gpu_kern_sum.csv
    python3 breakdown.py --top 5         # also print the 5 most expensive kernels of every run

Prints one row per run and writes examples/breakdown.csv.
Categories are assigned from the kernel NAME (first match wins), so they are approximate:
a fused kernel gets one label even if it contains several kinds of operation.
  matmul          gemm / cutlass / cublas ...        (the dense layers)
  reduction       *reduce*                           (sums, e.g. loss and weight gradients)
  data_movement   concatenate / transpose / copy / slice / pad / sort / convert / scal
  transcendental  sin / cos / tanh / exp / log
  elementwise     everything else (merged adds, multiplies, ...)
total_ms is the summed kernel time of the profiled window (about 40 training steps);
ms_per_step = total_ms / 40. Profiling slows things down, so use these numbers for
proportions between runs, not as absolute training speed.
"""
import csv, glob, os, re, sys

ROOT = os.environ.get("JAXPI_EXAMPLES", os.path.expanduser("~/thesis/jaxpi/examples"))
STEPS = 40
CATS = ["matmul", "reduction", "data_movement", "transcendental", "elementwise"]
RULES = [
    ("matmul", re.compile(r"gemm|cutlass|cublas|nvjet|wgmma|hmma|imma")),
    ("reduction", re.compile(r"reduce")),
    ("data_movement", re.compile(r"concat|transpose|copy|slice|pad|sort|convert|scal_kernel")),
    ("transcendental", re.compile(r"cosine|sine|tanh|exp|log")),
]


def classify(name):
    n = name.lower()
    for cat, rx in RULES:
        if rx.search(n):
            return cat
    return "elementwise"


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((r["Name"], float(r["Total Time (ns)"]), int(r["Instances"])))
            except (KeyError, ValueError):
                continue
    return rows


top_n = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 0
out_rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "*", "profiles", "*", "nsys_kernels_cuda_gpu_kern_sum.csv"))):
    run = os.path.basename(os.path.dirname(path))
    kernels = load(path)
    total = sum(t for _, t, _ in kernels)
    if total == 0:
        continue
    by = {c: 0.0 for c in CATS}
    for name, t, _ in kernels:
        by[classify(name)] += t
    row = {"run": run, "total_ms": total / 1e6, "ms_per_step": total / 1e6 / STEPS,
           "n_kernels": sum(i for _, _, i in kernels)}
    row.update({c: 100 * by[c] / total for c in CATS})
    out_rows.append(row)
    if top_n:
        out_rows[-1]["_top"] = sorted(kernels, key=lambda k: -k[1])[:top_n]

cols = ["run", "total_ms", "ms_per_step", "n_kernels"] + CATS
with open(os.path.join(ROOT, "breakdown.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in out_rows:
        w.writerow([r["run"]] + [f"{r[c]:.1f}" if c in CATS or c.endswith("ms") or c == "ms_per_step" else r[c] for c in cols[1:]])

hdr = f"{'run':24s}{'total_ms':>10s}{'ms/step':>9s}{'kernels':>9s}" + "".join(f"{c:>16s}" for c in CATS)
print(hdr)
for r in out_rows:
    print(f"{r['run']:24s}{r['total_ms']:10.1f}{r['ms_per_step']:9.2f}{r['n_kernels']:9d}" + "".join(f"{r[c]:15.1f}%" for c in CATS))
    for name, t, inst in r.get("_top", []):
        print(f"      {100*t/(r['total_ms']*1e6):5.1f}%  {inst:5d}x  {name[:90]}")
print(f"\nwritten: {os.path.join(ROOT, 'breakdown.csv')}")
