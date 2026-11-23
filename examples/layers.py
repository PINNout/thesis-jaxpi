"""
Attribute profiled GPU time to layers, phases, operation classes, and estimate how close every
kernel is to the memory-bandwidth / compute limit of the GPU.

    python3 layers.py                       # every run that has profiles/<run>/hlo and the kernel CSV
    python3 layers.py adv_fp32 ks_fp32      # only these runs
    python3 layers.py --top 20              # show 20 kernels in the per-kernel table (default 12)
    python3 layers.py --brief               # only the one-line-per-run summary table at the end
    python3 layers.py --explain <kernel> <run>   # operands/ops of one kernel, e.g. --explain loop_multiply_fusion_9 ks_fp32
    PEAK_GBPS=448 PEAK_TFLOPS=7.5 python3 layers.py     # GPU limits (defaults = RTX 2070 FP32)

What it does
  1. In <run>/hlo it picks the TRAINING-STEP module (the optimised HLO that contains the backward
     pass, "transpose(jvp", and is not a weight-update / evaluation module).
  2. Every compiler-generated (fused) kernel is matched by name to its HLO instruction
     (HLO "fusion.29" == Nsight "fusion_29"), so its time is MEASURED.  From the HLO we also read
       - what is INSIDE the fusion (tanh, sine, reduce, multiply, concatenate, ...) -> operation class
         (reduction | transcendental | elementwise | data_movement | other), instead of guessing from
         the kernel name
       - the bytes it reads+writes (operand + result sizes, an UPPER bound: broadcasts/slices inside a
         fusion may read less) -> achieved GB/s = bytes / kernel time, and % of PEAK_GBPS
       - a rough arithmetic intensity (arith ops x elements / bytes)
       - layer / phase from the op_name metadata (if the fusion itself carries none, the first
         op_name inside it that does is used)
  3. cuBLAS matmuls have no layer in their kernel name: their total measured time is split over
     the HLO matmuls in proportion to FLOPs (an ESTIMATE).  Per matmul the script also computes
     bytes and arithmetic intensity, the estimated achieved TFLOP/s and how it compares to PEAK_TFLOPS
     (is the matmul memory-bound or compute-bound for these skinny shapes?).

Labels
  layer  embedding | gate_u_v (Dense_0/1 of ModifiedMlp) | hidden | output | in_net_other | loss_other
  phase  net_fwd   network evaluated under jvp (primal AND tangent/derivative evaluation, they cannot be
                   told apart in op_name)
         net_bwd   transpose (backprop) through the network, including through the derivative path
         loss_fwd / loss_bwd   the same outside the network (loss terms, residual arithmetic, weights)
  branch residual | other_loss, only when the op_names of this model contain branch information
         (adv/burgers/allen_cahn: res_and_w).  For ldc / ns_cyl they do not, so it is shown as n/a.
BYTES: for fused kernels the bytes are operand + result sizes, where operands that are only sliced inside
the fusion count just the slice and in-place dynamic-update-slice buffers count only the update.  Still an
UPPER bound, so a LOW % of peak is trustworthy (real bandwidth is even lower) but a high one is not.
Kernels above 120% of peak are physically impossible: they are FLAGGED, left out of the GB/s averages and their
time share is printed.  Likely cause: nsys merges kernels with the same name from different XLA modules (step
module, logging/eval module), so calls/time of the step-module instruction can be inflated.  Use --explain.
Percentages are of the kernel time of the ~40 profiled steps (profiling slows kernels, use ratios).
Caveats: bytes are upper bounds; the 4 MB L2 of the RTX 2070 can serve small tensors, so "% of peak"
can exceed 100 or understate DRAM traffic; matmul layer split and TFLOP/s assume the step module ran
once per profiled step (a launch-count sanity check is printed).

Output: tables on screen, <run>/layers.csv, <run>/fusions.csv (all kernels), and in examples/:
layers_summary.csv, class_summary.csv, run_summary.csv.
"""
import csv, glob, hashlib, os, re, sys
from collections import defaultdict, Counter

ROOT = os.environ.get("JAXPI_EXAMPLES", os.path.expanduser("~/thesis/jaxpi/examples"))
PEAK_GBPS = float(os.environ.get("PEAK_GBPS", "448"))       # RTX 2070 memory bandwidth
PEAK_TFLOPS = float(os.environ.get("PEAK_TFLOPS", "7.5"))   # RTX 2070 FP32
PEAK_TFLOPS_HALF = float(os.environ.get("PEAK_TFLOPS_HALF", "59.7"))  # nominal FP16 tensor-core peak (used for *_native runs;
#                                                                       with FP32 accumulate roughly half of that)
STEPS = 40
EXCLUDE = ("compute_weights", "compute_l2_error", "compute_diag_ntk", "update_weights")
MATMUL = re.compile(r"gemm|cutlass|cublas|nvjet|wgmma|hmma|imma")
SHAPE = re.compile(r"\b(f8e\w+|bf16|[fsu]\d+|pred)\[([\d,]*)\]")
DTB = {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4,
       "u32": 4, "s64": 8, "u64": 8, "pred": 1}
LAYER = re.compile(r"(Dense_\d+|FourierEmbs_\d+|PeriodEmbs_\d+)")
ARCH = re.compile(r"(?:Modified)?Mlp")
BRANCH = re.compile(r"res_and_w|r_net|r_pred|ru_net|rv_net|rc_net")
HEADER = re.compile(r"^\s*(ENTRY\s+)?%?([\w.\-]+)\s*\(.*\)\s*->\s*.*\{\s*$")
DM = {"parameter", "constant", "bitcast", "reshape", "broadcast", "convert", "copy", "transpose", "slice",
      "dynamic-slice", "dynamic-update-slice", "concatenate", "pad", "tuple", "get-tuple-element",
      "reverse", "iota", "copy-start", "copy-done", "bitcast-convert"}
TRANS = {"tanh", "exponential", "exponential-minus-one", "log", "log-plus-one", "sine", "cosine", "power",
         "sqrt", "rsqrt", "cbrt", "logistic", "atan2", "tan", "erf"}
REDUCE = {"reduce", "reduce-window"}
OTHER = {"sort", "gather", "scatter", "rng-bit-generator", "rng", "all-reduce"}
CLASSES = ["matmul", "reduction", "transcendental", "elementwise", "data_movement", "other", "unmapped"]


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def split_type(rest):
    if rest.startswith("("):
        depth = 0
        for i, ch in enumerate(rest):
            depth += (ch == "(") - (ch == ")")
            if depth == 0:
                return rest[: i + 1], rest[i + 1:].lstrip()
    i = rest.find(" ")
    return rest[:i], rest[i + 1:]


def arrays(typ):
    return [(m.group(1), [int(x) for x in m.group(2).split(",") if x]) for m in SHAPE.finditer(typ)]


def nbytes(arrs):
    return sum(DTB.get(d, 4) * prod(s) for d, s in arrs)


SLICE_OPS = {"slice", "dynamic-slice", "gather"}
PASS_OPS = {"bitcast", "reshape"}


NEEDS_ALL = REDUCE | {"dot", "convolution", "sort", "scatter", "select-and-scatter", "cholesky", "triangular-solve"}


def fusion_bytes(ins, comp_instrs, arrs):
    """(bytes, per-operand detail) for one top-level fusion instruction.
    Loop fusions compute each output element from exactly the input elements it needs, so a parameter is read
    only as far as the demand of its consumers reaches: slices/gathers read just the slice, bitcast/convert/
    elementwise pass the demand through, reductions/dots need the whole operand, dynamic-update-slice buffers
    are updated in place.  Result = upper bound on the DRAM bytes."""
    body = comp_instrs.get(ins["calls"])
    own = nbytes(arrs.get(ins["name"], []))
    ops = ins["operands"]
    clean = lambda o: re.sub(r"#\d+$", "", o)
    if not body:
        return own + sum(nbytes(arrs.get(clean(o), [])) for o in ops), []
    elems = {b["name"]: (prod(arrs[b["name"]][0][1]) if arrs.get(b["name"]) else 1) for b in body}
    cons = defaultdict(list)
    for b in body:
        for o in b["operands"]:
            cons[o].append(b)
    root = next((b for b in body if b.get("root")), None)
    memo = {}

    def demand(x):
        if x in memo:
            return memo[x]
        memo[x] = elems[x]                                   # cycle guard
        cs = cons.get(x, [])
        if not cs:
            return elems[x]
        tot = 0
        for c in cs:
            ops_c = c["operands"]
            if c["op"] in NEEDS_ALL:
                tot += elems[x]
            elif c["op"] in SLICE_OPS:
                tot += min(elems[c["name"]], demand(c["name"])) if ops_c and ops_c[0] == x else 1
            elif c["op"] == "dynamic-update-slice":
                tot += 0 if ops_c and ops_c[0] == x else elems[x]
            else:
                tot += min(elems[x], demand(c["name"]))
        memo[x] = min(tot, elems[x])
        return memo[x]

    pname = {b["pidx"]: b["name"] for b in body if b["op"] == "parameter" and b.get("pidx") is not None}
    total, detail = 0, []
    for k, o in enumerate(ops):
        arr = arrs.get(clean(o), [])
        full = nbytes(arr)
        p = pname.get(k)
        used = min(full, demand(p) * (DTB.get(arr[0][0], 4) if arr else 4)) if p else full
        total += used
        detail.append((o, full, used))
    if root is not None and root["op"] == "dynamic-update-slice" and len(root["operands"]) > 1:
        own = nbytes(arrs.get(root["operands"][1], [])) or own   # in-place update: only the update is written
    return total + own, detail


def parse_module(path):
    arrs, instrs = {}, []
    comp_ops, comp_names = defaultdict(Counter), defaultdict(list)
    comp_instrs = defaultdict(list)
    cur = None
    for line in open(path, errors="ignore"):
        if line.strip() == "}":
            cur = None
            continue
        h = HEADER.match(line)
        if h:
            cur = h.group(2)
            continue
        m = re.match(r"^\s*(?:ROOT\s+)?%([^\s=]+)\s*=\s*(.*)$", line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        typ, rem = split_type(rest)
        arrs[name] = arrays(typ)
        om = re.match(r"([\w\-]+)\(([^)]*)\)", rem)
        if not om:
            continue
        op, args = om.group(1), om.group(2)
        md = re.search(r'op_name="([^"]*)"', line)
        calls = re.search(r"calls=%?([\w.\-]+)", line)
        ins = dict(name=name, op=op, op_name=md.group(1) if md else "", comp=cur,
                   operands=re.findall(r"%([\w.\-]+)", args), calls=calls.group(1) if calls else None,
                   flops=0.0, is_matmul=False, root=bool(re.match(r"^\s*ROOT\s", line)),
                   pidx=int(args) if op == "parameter" and args.strip().isdigit() else None)
        if op == "custom-call" and "cublas" in line:
            ins["is_matmul"] = True
            lc = re.search(r'"lhs_contracting_dimensions":\[([^\]]*)\]', line)
            ins["lc"] = [int(x) for x in re.findall(r"\d+", lc.group(1))] if lc else []
        instrs.append(ins)
        if cur:
            comp_instrs[cur].append(ins)
            comp_ops[cur][op] += 1
            if ins["op_name"] and ins["op_name"] not in comp_names[cur] and len(comp_names[cur]) < 40:
                comp_names[cur].append(ins["op_name"])
    fused = {i["calls"] for i in instrs if i["op"] == "fusion" and i["calls"]}
    top = [i for i in instrs if i["comp"] not in fused]
    for ins in top:
        own = arrs.get(ins["name"], [])
        ops_arr = [a for o in ins["operands"] for a in arrs.get(re.sub(r"#\d+$", "", o), [])]
        if ins["is_matmul"]:
            out = own[:1]                                # 2nd tuple element is cuBLAS workspace
            ins["bytes"] = nbytes(out) + nbytes(ops_arr[:2] if len(ops_arr) >= 2 else ops_arr)
            lhs = arrs.get(re.sub(r"#\d+$", "", ins["operands"][0]), []) if ins["operands"] else []
            try:
                k = prod(lhs[0][1][d] for d in ins["lc"])
            except (IndexError, TypeError):
                k = 0
            n_out = prod(out[0][1]) if out else 0
            ins["flops"] = 2.0 * n_out * k if (n_out and k) else 0.0
        elif ins["op"] == "fusion":
            ins["bytes"], ins["detail"] = fusion_bytes(ins, comp_instrs, arrs)
        elif ins["op"] in SLICE_OPS:
            ins["bytes"] = 2 * nbytes(own)
        elif ins["op"] == "dynamic-update-slice" and len(ins["operands"]) > 1:
            ins["bytes"] = 2 * nbytes(arrs.get(ins["operands"][1], []))
        else:
            ins["bytes"] = nbytes(own) + nbytes(ops_arr)
        ins["elems"] = max([prod(s) for _, s in own + ops_arr] or [0])
    return top, comp_ops, comp_names


def pick_step_modules(hlo_dir):
    files = sorted(glob.glob(os.path.join(hlo_dir, "*_gpu_after_optimizations.txt")))
    seen, picks = set(), []
    for f in files:
        txt = open(f, errors="ignore").read()
        if "transpose(jvp" not in txt or any(x in txt for x in EXCLUDE):
            continue
        h = hashlib.md5(txt.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        picks.append((len(txt), f))
    picks.sort(reverse=True)
    return [f for _, f in picks]


def classify(op_name, max_dense, modified, has_branch):
    lm = LAYER.findall(op_name)
    if lm:
        lay = lm[-1]
        if lay.startswith(("FourierEmbs", "PeriodEmbs")):
            layer = "embedding"
        else:
            i = int(lay.split("_")[1])
            layer = "output" if i == max_dense else ("gate_u_v" if modified and i in (0, 1) else "hidden")
    elif ARCH.search(op_name):
        layer = "in_net_other"
    else:
        layer = "loss_other"
    bwd = "transpose(" in op_name
    if layer != "loss_other":
        phase = "net_bwd" if bwd else "net_fwd"
    else:
        phase = "loss_bwd" if bwd else "loss_fwd"
    branch = ("residual" if BRANCH.search(op_name) else "other_loss") if has_branch else "n/a"
    return layer, phase, branch


def pick_opname(ins, comp_names):
    cands = ([ins["op_name"]] if ins["op_name"] else []) + comp_names.get(ins["calls"], [])
    for pat in (LAYER, ARCH):
        for c in cands:
            if pat.search(c):
                return c
    return cands[0] if cands else ""


def content_class(ins, comp_ops):
    ops = comp_ops.get(ins["calls"]) if ins["op"] == "fusion" and ins["calls"] in comp_ops else Counter({ins["op"]: 1})
    names = {o for o in ops if o not in ("parameter", "tuple", "get-tuple-element")}
    arith = sum(c for o, c in ops.items() if o not in DM)
    if names & REDUCE:
        cls = "reduction"
    elif names & TRANS:
        cls = "transcendental"
    elif names & OTHER:
        cls = "other"
    elif arith == 0:
        cls = "data_movement"
    else:
        cls = "elementwise"
    top = [o for o, _ in sorted(((o, c) for o, c in ops.items() if o not in DM or o in ("concatenate", "transpose", "copy")),
                                key=lambda kv: -kv[1])][:3]
    return cls, arith, top


def name_class(name):
    n = name.lower()
    if "reduce" in n:
        return "reduction"
    if re.search(r"cosine|sine|tanh|exp|log", n):
        return "transcendental"
    if re.search(r"concat|transpose|copy|slice|pad|convert", n):
        return "data_movement"
    return "elementwise"


def load_kernels(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out.append((r["Name"], float(r["Total Time (ns)"]), int(float(r.get("Instances", 1) or 1))))
            except (KeyError, ValueError):
                pass
    return out


def analyse(run_dir):
    run = os.path.basename(run_dir)
    kcsv = os.path.join(run_dir, "nsys_kernels_cuda_gpu_kern_sum.csv")
    mods = pick_step_modules(os.path.join(run_dir, "hlo"))
    if not (os.path.isfile(kcsv) and mods):
        return None
    top, comp_ops, comp_names = parse_module(mods[0])
    onames = [i["op_name"] for i in top] + [n for v in comp_names.values() for n in v]
    dense = [int(x) for n in onames for x in re.findall(r"Dense_(\d+)", n)]
    max_dense = max(dense) if dense else -1
    modified = any("ModifiedMlp" in n for n in onames)
    has_branch = any(BRANCH.search(n) for n in onames)
    by_kname = {re.sub(r"[.\-]", "_", i["name"]): i for i in top if not i["is_matmul"]}
    matmuls = [i for i in top if i["is_matmul"]]
    total_flops = sum(i["flops"] for i in matmuls)

    agg = defaultdict(lambda: [0.0, 0.0])      # (layer, phase, branch, cls) -> [measured_ns, estimated_ns]
    cls_stat = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # cls -> [ns, credible bytes, credible ns, flagged ns]
    kernels, total, mm_time, mm_launch, unmapped = [], 0.0, 0.0, 0, 0.0
    for name, t, n in load_kernels(kcsv):
        total += t
        if MATMUL.search(name.lower()):
            mm_time += t
            mm_launch += n
            continue
        ins = by_kname.get(name)
        if ins is None:
            unmapped += t
            kernels.append(dict(kernel=name, ns=t, inst=n, cls=name_class(name), layer="?", phase="?", branch="?",
                                ops="", bytes_call=0, gbps=0.0, ai=0.0, mapped=False))
            continue
        lay, ph, br = classify(pick_opname(ins, comp_names), max_dense, modified, has_branch)
        cls, arith, ops = content_class(ins, comp_ops)
        agg[(lay, ph, br, cls)][0] += t
        cls_stat[cls][0] += t
        b = ins["bytes"]
        if b and (b * n / t) <= 1.2 * PEAK_GBPS:         # credible estimate: not above the physical limit
            cls_stat[cls][1] += b * n
            cls_stat[cls][2] += t
        elif b:
            cls_stat[cls][3] += t                          # implausible (>120% of peak): bytes overcounted
        kernels.append(dict(kernel=name, ns=t, inst=n, cls=cls, layer=lay, phase=ph, branch=br,
                            ops="+".join(ops), bytes_call=b, gbps=(b * n / t if t else 0.0),
                            ai=(arith * ins["elems"] / b if b else 0.0), mapped=True))
    if total_flops > 0:
        for i in matmuls:
            lay, ph, br = classify(pick_opname(i, comp_names), max_dense, modified, has_branch)
            agg[(lay, ph, br, "matmul")][1] += mm_time * i["flops"] / total_flops
    else:
        unmapped += mm_time
    mm = dict(n=len(matmuls), flops=total_flops, bytes=sum(i["bytes"] for i in matmuls), time=mm_time,
              launches=mm_launch)
    return dict(run=run, total=total, mm_time=mm_time, unmapped=unmapped, agg=agg, cls_stat=cls_stat,
                kernels=kernels, mm=mm, module=mods[0], has_branch=has_branch)


def table(title, agg, total, idx, note=""):
    d = defaultdict(lambda: [0.0, 0.0])
    for k, (m, e) in agg.items():
        d[k[idx]][0] += m
        d[k[idx]][1] += e
    print(f"  by {title}:  (measured fusions + estimated matmul){note}")
    for k, (m, e) in sorted(d.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        print(f"    {k:14s} {100*(m+e)/total:5.1f}%   measured {100*m/total:5.1f}%  est.matmul {100*e/total:5.1f}%")


def class_times(r):
    """class -> ns (matmul = all matmul kernels, unmapped = kernels not found in HLO)"""
    c = defaultdict(float)
    c["matmul"] = r["mm_time"]
    for k, (ns, _, _, _) in r["cls_stat"].items():
        c[k] += ns
    for k in r["kernels"]:
        if not k["mapped"]:
            c["unmapped"] += k["ns"]
    return c


def peak_tflops(r):
    return PEAK_TFLOPS_HALF if r["run"].endswith("_native") else PEAK_TFLOPS


def matmul_numbers(r):
    mm = r["mm"]
    if not (mm["flops"] and mm["bytes"] and mm["time"]):
        return None
    ai = mm["flops"] / mm["bytes"]
    tf = mm["flops"] * STEPS / mm["time"] / 1e3          # flop / ns = GFLOP/s ; /1e3 -> TFLOP/s
    return ai, tf, mm["launches"] / mm["n"] if mm["n"] else 0.0


def show(r, topn):
    total = r["total"]
    print(f"\n== {r['run']}   (HLO: {os.path.basename(r['module'])}, {r['mm']['n']} matmuls, kernel time {total/1e6:.1f} ms over ~{STEPS} steps = {total/1e6/STEPS:.2f} ms/step)")
    print(f"  matmul share {100*r['mm_time']/total:.1f}%   unmapped {100*r['unmapped']/total:.1f}%")
    agg = r["agg"]
    table("layer", agg, total, 0)
    table("phase", agg, total, 1, "   [net_* = inside the network, loss_* = outside]")
    if r["has_branch"]:
        table("branch", agg, total, 2)
    else:
        print("  by branch: n/a (op_names of this model carry no residual/boundary information)")
    ct = class_times(r)
    print("  by operation class (content of the HLO fusion):  time%   GB/s (credible kernels only)  %peak   time with implausible bytes")
    for c in CLASSES:
        if ct.get(c, 0) <= 0:
            continue
        ns, by, cns, fl = r["cls_stat"].get(c, (0, 0, 0, 0))
        bw = f"{by/cns:8.0f}   {100*by/cns/PEAK_GBPS:6.0f}%   flagged {100*fl/ns:3.0f}% of class time" if c not in ("matmul", "unmapped") and cns else ""
        print(f"    {c:15s} {100*ct[c]/total:5.1f}%   {bw}")
    mn = matmul_numbers(r)
    if mn:
        ai, tf, per = mn
        pk = peak_tflops(r)
        print(f"  matmul roofline: mean arithmetic intensity {ai:.1f} flop/B (ridge at {pk*1e3/PEAK_GBPS:.1f} flop/B for a {pk:g} TFLOP/s peak), "
              f"est. achieved {tf:.2f} TFLOP/s = {100*tf/pk:.0f}% of peak   "
              f"[launches per HLO matmul in window: {per:.1f}, expected ~{STEPS}]")
    # class x phase (as % of total)
    phases = ["net_fwd", "net_bwd", "loss_fwd", "loss_bwd"]
    mat = defaultdict(float)
    for (lay, ph, br, cls), (m, e) in agg.items():
        mat[(cls, ph)] += m + e
    print("  class x phase (% of total kernel time):   " + "  ".join(f"{p:>8s}" for p in phases))
    for c in CLASSES[:-1]:
        row = [100 * mat[(c, p)] / total for p in phases]
        if sum(row) > 0:
            print(f"    {c:15s}                       " + "  ".join(f"{v:8.1f}" for v in row))
    print(f"  top {topn} non-matmul kernels:   (* = above 120% of peak bandwidth: bytes overcounted, ignore GB/s)")
    print("    {:<30s} {:>6s} {:>5s} {:>5s} {:<14s} {:<12s} {:<9s} {:<20s} {:>7s} {:>6s} {:>5s} {:>6s}".format(
        "kernel", "ms", "%tot", "calls", "class", "layer", "phase", "ops", "MB/call", "GB/s", "%pk", "flop/B"))
    for k in sorted(r["kernels"], key=lambda k: -k["ns"])[:topn]:
        print("    {:<30s} {:6.1f} {:5.1f} {:5d} {:<14s} {:<12s} {:<9s} {:<20s} {:7.2f} {:6.0f} {:>5s} {:6.2f}".format(
            k["kernel"][:30], k["ns"] / 1e6, 100 * k["ns"] / total, k["inst"], k["cls"], k["layer"], k["phase"],
            k["ops"][:20], k["bytes_call"] / 1e6, k["gbps"],
            ("%.0f" % (100 * k["gbps"] / PEAK_GBPS)) + ("*" if k["gbps"] > 1.2 * PEAK_GBPS else ""), k["ai"]))


def explain(run_dir, kname):
    mods = pick_step_modules(os.path.join(run_dir, "hlo"))
    top, comp_ops, comp_names = parse_module(mods[0])
    key = lambda i: re.sub(r"[.\-]", "_", i["name"])
    hits = [i for i in top if key(i) == kname or key(i).startswith(kname)]
    if not hits:
        print(f"no instruction matching {kname} in {os.path.basename(mods[0])}")
        return
    for i in hits[:3]:
        print(f"\n{i['name']}  op={i['op']}  calls={i['calls']}  bytes counted={i['bytes']/1e6:.1f} MB")
        print(f"  op_name: {pick_opname(i, comp_names)}")
        if i["calls"]:
            print("  ops inside: " + ", ".join(f"{o}x{c}" for o, c in comp_ops[i['calls']].most_common(12)))
        print("  operands (outer buffer MB -> MB actually read after slice analysis):")
        for o, full, used in sorted(i.get("detail", []), key=lambda d: -d[1])[:12]:
            print(f"    {o:40s} {full/1e6:9.2f} -> {used/1e6:9.2f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--explain" in args:
        j = args.index("--explain")
        explain(os.path.join(ROOT, "*", "profiles", args[j + 2]) if False else
                (glob.glob(os.path.join(ROOT, "*", "profiles", args[j + 2])) or [None])[0], args[j + 1])
        sys.exit(0)
    topn, brief = 12, "--brief" in args
    if "--top" in args:
        topn = int(args[args.index("--top") + 1])
    wanted = [a for a in args if not a.startswith("-") and not a.isdigit()]
    summary, csum, rsum = [], [], []
    for run_dir in sorted(glob.glob(os.path.join(ROOT, "*", "profiles", "*"))):
        run = os.path.basename(run_dir)
        if wanted and run not in wanted:
            continue
        if not os.path.isdir(run_dir) or run.endswith("_wd"):
            continue
        r = analyse(run_dir)
        if r is None:
            print(f"{run}: no kernel CSV or no training-step HLO module found, skipped")
            continue
        if not brief:
            show(r, topn)
        total, agg = r["total"], r["agg"]
        with open(os.path.join(run_dir, "layers.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "phase", "branch", "class", "measured_ms", "estimated_matmul_ms", "pct_of_total"])
            for (l, p, b, c), (m, e) in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
                w.writerow([l, p, b, c, f"{m/1e6:.3f}", f"{e/1e6:.3f}", f"{100*(m+e)/total:.2f}"])
                summary.append([r["run"], l, p, b, c, f"{m/1e6:.3f}", f"{e/1e6:.3f}", f"{100*(m+e)/total:.2f}"])
            w.writerow(["unmapped", "", "", "", f"{r['unmapped']/1e6:.3f}", "", f"{100*r['unmapped']/total:.2f}"])
        with open(os.path.join(run_dir, "fusions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "total_ms", "pct_of_total", "calls", "class", "layer", "phase", "branch", "ops",
                        "mb_per_call", "gb_per_s", "pct_peak_bw", "flop_per_byte", "mapped_to_hlo"])
            for k in sorted(r["kernels"], key=lambda k: -k["ns"]):
                w.writerow([k["kernel"], f"{k['ns']/1e6:.3f}", f"{100*k['ns']/total:.2f}", k["inst"], k["cls"],
                            k["layer"], k["phase"], k["branch"], k["ops"], f"{k['bytes_call']/1e6:.3f}",
                            f"{k['gbps']:.1f}", f"{100*k['gbps']/PEAK_GBPS:.1f}", f"{k['ai']:.3f}", k["mapped"]])
        ct = class_times(r)
        nm_ns = sum(v[2] for v in r["cls_stat"].values())
        nm_by = sum(v[1] for v in r["cls_stat"].values())
        nm_fl = sum(v[3] for v in r["cls_stat"].values())
        nm_all = sum(v[0] for v in r["cls_stat"].values())
        for c in CLASSES:
            ns, by, cns, fl = r["cls_stat"].get(c, (0, 0, 0, 0))
            csum.append([r["run"], c, f"{100*ct.get(c,0)/total:.2f}", f"{by/cns:.1f}" if cns else "",
                         f"{100*by/cns/PEAK_GBPS:.1f}" if cns else "", f"{100*fl/ns:.0f}" if ns else ""])
        mn = matmul_numbers(r)
        rsum.append([r["run"], f"{total/1e6/STEPS:.2f}"] + [f"{100*ct.get(c,0)/total:.1f}" for c in CLASSES]
                    + [f"{nm_by/nm_ns:.0f}" if nm_ns else "", f"{100*nm_by/nm_ns/PEAK_GBPS:.0f}" if nm_ns else "", f"{100*nm_fl/nm_all:.0f}" if nm_all else "",
                       f"{mn[0]:.1f}" if mn else "", f"{mn[1]:.2f}" if mn else ""])
    hdr = ["run", "ms/step"] + [c + "%" for c in CLASSES] + ["nonmm_GB/s", "nonmm_%peakBW", "nonmm_%flagged", "matmul_flop/B", "matmul_TFLOPs"]
    if rsum:
        print("\n== summary (classes from HLO fusion content; % of kernel time)")
        print("  " + " ".join(f"{h:>14s}" if i else f"{h:<22s}" for i, h in enumerate(hdr)))
        for row in rsum:
            print("  " + " ".join(f"{v:>14s}" if i else f"{v:<22s}" for i, v in enumerate(row)))
        with open(os.path.join(ROOT, "run_summary.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rsum)
        with open(os.path.join(ROOT, "class_summary.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["run", "class", "pct_of_total", "credible_gb_per_s", "pct_peak_bw", "pct_class_time_flagged"]); w.writerows(csum)
        with open(os.path.join(ROOT, "layers_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "layer", "phase", "branch", "class", "measured_ms", "estimated_matmul_ms", "pct_of_total"])
            w.writerows(summary)
        print(f"\nwritten: {ROOT}/run_summary.csv, class_summary.csv, layers_summary.csv ; per run: layers.csv, fusions.csv")
