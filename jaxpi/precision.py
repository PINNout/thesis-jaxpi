"""
Reduced-precision emulation for the PINN experiments (v2).

Everything is computed in FP32 and *rounded* to the target format at named
hooks, so any single part of training can be lowered on its own.

FORMATS
  fp32                      no rounding
  fp16, bf16                IEEE half / bfloat16
  e<E>m<M>                  any float with E exponent and M mantissa bits,
                            e.g. e5m10 (= fp16), e5m7, e5m4, e8m10
  fp8_e4m3, fp8_e5m2, int8  block-scaled 8-bit formats (scale = max|x| of the
                            tensor; under vmap that is one scale per sample)

HOOKS (config.precision.<hook> = "<format>")
  network (read in archs.py)
    embed      output of the Fourier / periodic embedding
    weights    effective weight matrix W (= g*v) of hidden layers
    act_in     input X of each hidden dense layer
    bias       bias of every dense layer (the Y operand of Z = X.W + Y)
    acc        accumulator of the matmul, see qdot()
    act_out    tanh output
    out_layer  X and W of the last layer
  training loop (need one call each in the model / train step, see bottom)
    residual   PDE residual before it is squared
    grads      parameter gradient before the optimiser
    params     stored parameters after the optimiser update

MIXED PRECISION
  Every hook has its own format, so parts can differ, e.g. weights in 8 bits
  and everything else in a 16-bit format. A format may carry its own
  derivative mode as a suffix, "<format>:ste" or "<format>:full", which
  overrides the global `ad` for that hook only:
      precision.ad      = "full"
      precision.act_in  = "e8m7"            # values, derivatives, gradients
      precision.weights = "fp8_e4m3:ste"    # only the weight values are 8-bit;
                                            # their gradient is not rounded
  ":ste" on the weights models "narrow stored weights, wide master copy".

MICROSCALING (MX), see jaxpi/mx.py
  mx             element format of every matmul operand: e4m3, e5m2 (MXFP8),
                 e3m2, e2m3 (MXFP6), e2m1 (MXFP4), int8 (MXINT8); "none" = off.
                 Applies to all dense layers at every derivative order.
                 The weights, act_in, out_layer and acc hooks must then be fp32.
                 Not usable with Taylor mode (KS).
  mx_block       values per shared scale (default 32)
  mx_out         format the matmul result is rounded to (default fp32), e.g. e6m9

OPTIONS
  ad             "ste"  round forward values only; derivative values and
                        gradients stay FP32 (the old behaviour)
                 "full" also round every derivative value and every gradient
                        that flows through a hook. Float formats only.
  acc_chunk      products added per rounding in qdot (1 = round after every
                 multiply-add, like a chain of FP16 FMAs; larger = cheaper
                 to run but rounds less often, so it is milder than hardware)
  acc_impl       "scan" (default) or "unroll". Use "unroll" with a large
                 acc_chunk for KS: Taylor mode (jet) cannot trace a scan, and
                 unrolled compile time doubles each time acc_chunk is halved.
  param_rounding "nearest" or "stochastic" (for the params hook)
  scale_pow2     True: block scales of the 8-bit formats are powers of two
  native         "fp16"/"bf16": really run the network body in that dtype
                 (all hooks must be fp32)

Notes
  * Rounding uses lax.reduce_precision (XLA removes astype round trips).
  * reduce_precision overflows to inf like the real format, but flushes
    subnormals to zero (fp16: |x| < 6.1e-5 becomes 0).
  * ad="full" cannot separate "derivative chain" from "backward pass": JAX-PI
    computes the input derivatives with reverse mode too, so both are the same
    kind of value. Use the grads hook to lower the final gradient on its own.
"""
import re

import jax
import jax.numpy as jnp
from jax import lax

try:  # lets jax.experimental.jet (KS example) see reduce_precision as linear
    from jax.experimental import jet as _jet
    if lax.reduce_precision_p not in _jet.jet_rules:
        _jet.deflinear(lax.reduce_precision_p)
except Exception:  # pragma: no cover
    pass

_NAMED = {"fp16": (5, 10), "bf16": (8, 7)}
_FP8 = {"fp8_e4m3": (4, 3), "fp8_e5m2": (5, 2)}
_FP8_MAX = {"fp8_e4m3": 240.0, "fp8_e5m2": 57344.0}
FORMATS = ("fp32", "fp16", "bf16", "e<E>m<M>", "fp8_e4m3", "fp8_e5m2", "int8")

NET_HOOKS = ("weights", "act_in", "act_out", "embed", "out_layer", "bias", "acc")
TRAIN_HOOKS = ("residual", "grads", "params")
HOOKS = NET_HOOKS + TRAIN_HOOKS

_PRECISION = {k: "fp32" for k in HOOKS}
_OPT = {"ad": "ste", "acc_chunk": 1, "acc_impl": "scan", "param_rounding": "nearest", "scale_pow2": True}
_NATIVE_DTYPES = {"fp16": jnp.float16, "bf16": jnp.bfloat16}
_NATIVE = {"name": None, "dtype": None}


# --------------------------------------------------------------------------
# formats
# --------------------------------------------------------------------------
def split_format(fmt):
    """'fp16:ste' -> ('fp16', 'ste'); 'fp16' -> ('fp16', None)."""
    if ":" in fmt:
        base, ad = fmt.split(":", 1)
        if ad not in ("ste", "full"):
            raise ValueError(f"format '{fmt}': suffix must be ':ste' or ':full'")
        return base, ad
    return fmt, None


def float_bits(fmt):
    """(exponent_bits, mantissa_bits) for a plain float format, else None."""
    fmt = split_format(fmt)[0]
    if fmt in _NAMED:
        return _NAMED[fmt]
    m = re.fullmatch(r"e(\d+)m(\d+)", fmt)
    if m:
        e, mb = int(m.group(1)), int(m.group(2))
        if not (2 <= e <= 8 and 1 <= mb <= 23):
            raise ValueError(f"format '{fmt}': need 2<=E<=8 and 1<=M<=23")
        return e, mb
    return None


def check_format(fmt):
    full = fmt
    fmt = split_format(fmt)[0]
    if fmt == "fp32":
        return "fp32"
    if fmt in _FP8 or fmt == "int8" or float_bits(fmt):
        return full
    raise ValueError(f"Unknown precision format '{full}'. Choose from {FORMATS}")


def _scale(x, max_val):
    """Block scale mapping max|x| onto the format's largest value."""
    s = jnp.maximum(jnp.max(jnp.abs(x)), 1e-30) / max_val
    if _OPT["scale_pow2"]:
        mant, ex = jnp.frexp(s)                    # s = mant * 2^ex, mant in [0.5, 1)
        s = jnp.ldexp(jnp.ones_like(s), jnp.where(mant == 0.5, ex - 1, ex))
    return lax.stop_gradient(s)


# --------------------------------------------------------------------------
# rounding
# --------------------------------------------------------------------------
def quantize(x, fmt, ad=None):
    """Round x to fmt, returned as float32.

    ad="ste":  derivatives are those of x (only forward values are rounded).
    ad="full": derivative values and gradients passing through are rounded too.
    Default: the format's own ":ste"/":full" suffix, else config.precision.ad.
    """
    fmt, ad_hook = split_format(fmt)
    if fmt == "fp32":
        return x
    ad = ad or ad_hook or _OPT["ad"]
    bits = float_bits(fmt)
    with jax.named_scope("quantize"):
        if ad == "full":
            if bits is None:
                raise ValueError(
                    f"ad='full' needs a plain float format, got '{fmt}'. "
                    "Use ad='ste' for fp8_*/int8, or an e<E>m<M> format."
                )
            return lax.reduce_precision(x, *bits)
        xs = lax.stop_gradient(x)
        if bits is not None:
            q = lax.reduce_precision(xs, *bits)
        elif fmt in _FP8:
            s = _scale(xs, _FP8_MAX[fmt])
            q = lax.reduce_precision(xs / s, *_FP8[fmt]) * s
        elif fmt == "int8":
            s = _scale(xs, 127.0)
            q = jnp.clip(jnp.round(xs / s), -127, 127) * s
        else:
            check_format(fmt)
        return x + (q - xs)  # straight-through: value q, derivative of x


def qdot(x, kernel, bias=None, fmt=None):
    """x @ kernel + bias with a reduced-precision accumulator.

    fmt="fp32" (default: the "acc" hook): ordinary dot, wide accumulator.
    Otherwise the accumulator starts at the bias and is rounded to fmt after
    every acc_chunk products, in order along the inner dimension:
        acc = round(acc + x[k] * W[k, :])
    which models a chain of narrow fused multiply-adds. The result is in fmt.
    (Each step is computed in FP32 and then rounded, i.e. rounded twice, so
    after 128 terms about 0.3% of results differ from a true single-rounding
    FMA chain by one unit in the last place.)
    """
    fmt = split_format(_PRECISION["acc"] if fmt is None else fmt)[0]
    if fmt == "fp32":
        y = jnp.dot(x, kernel)
        return y if bias is None else y + bias
    n = kernel.shape[0]
    c = min(n, max(1, int(_OPT["acc_chunk"])))
    acc = jnp.zeros(x.shape[:-1] + (kernel.shape[1],), x.dtype)
    if bias is not None:
        acc = quantize(acc + bias, fmt)

    def add(acc, k, width):
        xs = lax.dynamic_slice_in_dim(x, k, width, axis=-1)
        ws = lax.dynamic_slice_in_dim(kernel, k, width, axis=0)
        return quantize(acc + jnp.dot(xs, ws), fmt)

    with jax.named_scope("qdot"):
        full = n // c
        if _OPT["acc_impl"] == "scan":
            acc, _ = lax.scan(lambda a, k: (add(a, k, c), None), acc, jnp.arange(full) * c)
        else:  # "unroll": plain ops only, needed for jax.experimental.jet (KS)
            for i in range(full):
                acc = add(acc, i * c, c)
        if n % c:
            acc = add(acc, full * c, n % c)
    return acc


def stochastic_round(x, key, fmt):
    """Round x up or down to a neighbour in fmt, with probability by distance.

    E[result] = x, so updates smaller than the format spacing survive on
    average instead of being dropped. Plain float formats only.
    """
    bits = float_bits(fmt)
    if bits is None:
        raise ValueError(f"stochastic rounding needs a plain float format, got '{fmt}'")
    e, m = bits
    emin = 2 - 2 ** (e - 1)                       # exponent of smallest normal
    fmax = (2.0 - 2.0 ** -m) * 2.0 ** (2 ** (e - 1) - 1)
    _, ex = jnp.frexp(x)                          # |x| in [2^(ex-1), 2^ex)
    ulp = jnp.ldexp(jnp.ones_like(x), jnp.maximum(ex - 1, emin) - m)  # exact 2^n
    lo = jnp.floor(x / ulp) * ulp
    up = jax.random.uniform(key, x.shape, x.dtype) < (x - lo) / ulp
    return jnp.clip(lo + ulp * up, -fmax, fmax)


# --------------------------------------------------------------------------
# training-loop hooks
# --------------------------------------------------------------------------
_USED = set()


def quantize_residual(r):
    """Call on each PDE residual before squaring it."""
    _USED.add("residual")
    return quantize(r, _PRECISION["residual"])


def assert_hooks_used():
    """Raise if the residual hook is configured but was never called."""
    if _PRECISION["residual"] != "fp32" and "residual" not in _USED:
        raise RuntimeError(
            "config.precision.residual is set, but this PDE's losses() never "
            "calls jaxpi.precision.quantize_residual. Add that call or set "
            "residual back to 'fp32'."
        )


def quantize_grads(grads):
    """Call on the gradient pytree before the optimiser."""
    fmt = _PRECISION["grads"]
    if fmt == "fp32":
        return grads
    return jax.tree_util.tree_map(lambda g: quantize(g, fmt, ad="ste"), grads)


def round_params(params, key=None):
    """Call on the parameter pytree after the optimiser update.

    Makes the stored (master) parameters live in the format. With
    param_rounding="stochastic" a PRNG key is required (e.g. fold the step
    number into a fixed key).
    """
    fmt = _PRECISION["params"]
    if fmt == "fp32":
        return params
    leaves, treedef = jax.tree_util.tree_flatten(params)
    if _OPT["param_rounding"] == "stochastic":
        if key is None:
            raise ValueError("stochastic param rounding needs a PRNG key")
        keys = jax.random.split(key, len(leaves))
        leaves = [stochastic_round(l, k, fmt) for l, k in zip(leaves, keys)]
    else:
        leaves = [quantize(l, fmt, ad="ste") for l in leaves]
    return jax.tree_util.tree_unflatten(treedef, leaves)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def set_precision(cfg):
    """Call once at the start of a run, before anything is jitted.

    cfg is config.precision (or None for all FP32).
    """
    get = (lambda k, d: d) if cfg is None else (lambda k, d: cfg.get(k, d))
    _USED.clear()
    _OPT.pop("mx", None)
    for k in HOOKS:
        _PRECISION[k] = check_format(get(k, "fp32"))
    _OPT["ad"] = get("ad", "ste")
    _OPT["acc_chunk"] = int(get("acc_chunk", 1))
    _OPT["acc_impl"] = get("acc_impl", "scan")
    if _OPT["acc_impl"] not in ("scan", "unroll"):
        raise ValueError("precision.acc_impl must be 'scan' or 'unroll'")
    _OPT["param_rounding"] = get("param_rounding", "nearest")
    _OPT["scale_pow2"] = bool(get("scale_pow2", True))
    if _OPT["ad"] not in ("ste", "full"):
        raise ValueError("precision.ad must be 'ste' or 'full'")
    if _OPT["param_rounding"] not in ("nearest", "stochastic"):
        raise ValueError("precision.param_rounding must be 'nearest' or 'stochastic'")
    if _PRECISION["acc"] != "fp32" and float_bits(_PRECISION["acc"]) is None:
        raise ValueError("the acc hook needs a plain float format")
    def _eff_ad(k):
        return split_format(_PRECISION[k])[1] or _OPT["ad"]

    if True:
        bad = [k for k in NET_HOOKS + ("residual",)
               if _PRECISION[k] != "fp32" and _eff_ad(k) == "full"
               and float_bits(_PRECISION[k]) is None]
        if bad:
            raise ValueError(
                f"ad='full' needs plain float formats; offending hooks: {bad}. "
                "Add ':ste' to those formats (e.g. 'fp8_e4m3:ste') to round only their values."
            )
    if _OPT["param_rounding"] == "stochastic" and _PRECISION["params"] != "fp32" \
            and float_bits(_PRECISION["params"]) is None:
        raise ValueError("stochastic param rounding needs a plain float format")

    from jaxpi import mx as _mx
    mx_elem = get("mx", "none")
    if mx_elem in (None, "", "none"):
        _mx.configure(None)
    else:
        clash = [k for k in ("weights", "act_in", "out_layer", "acc") if _PRECISION[k] != "fp32"]
        if clash:
            raise ValueError(f"with precision.mx set, these hooks must be fp32 (MX quantizes the "
                             f"matmul operands itself): {clash}")
        out = get("mx_out", "fp32")
        _mx.configure(mx_elem, int(get("mx_block", 32)),
                      None if out == "fp32" else float_bits(check_format(out)))
        _OPT["mx"] = f"{mx_elem}/block {int(get('mx_block', 32))}/out {out}"

    nat = get("native", None)
    if nat in (None, "", "none"):
        _NATIVE["name"], _NATIVE["dtype"] = None, None
    else:
        if nat not in _NATIVE_DTYPES:
            raise ValueError(f"Unknown native dtype '{nat}'. Choose from {list(_NATIVE_DTYPES)}")
        if any(v != "fp32" for v in _PRECISION.values()) or _mx.enabled():
            raise ValueError("native mode needs all emulated formats set to fp32 and mx off")
        _NATIVE["name"], _NATIVE["dtype"] = nat, _NATIVE_DTYPES[nat]
    print("Precision settings:", dict(_PRECISION), dict(_OPT), "native:", _NATIVE["name"])


def to_native(x):
    """Cast to the native compute dtype (no-op unless native mode is on)."""
    return x if _NATIVE["dtype"] is None else x.astype(_NATIVE["dtype"])


def from_native(x):
    """Cast the network output back to float32 (no-op unless native mode is on)."""
    return x if _NATIVE["dtype"] is None else x.astype(jnp.float32)


def get_precision():
    return _PRECISION


# --------------------------------------------------------------------------
# self-test:  python3 -m jaxpi.precision
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    ok = True

    def check(name, cond):
        global ok
        ok &= bool(cond)
        print(("PASS  " if cond else "FAIL  ") + name)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4096,)) * 3
    h16 = lambda a: np.asarray(a).astype(np.float16).astype(np.float32)

    for mode in ("ste", "full"):
        q = jax.jit(lambda a: quantize(a, "fp16", ad=mode))(x)
        check(f"fp16 {mode}: value equals a real float16 round trip", np.array_equal(q, h16(x)))
    check("e5m10 is fp16", np.array_equal(quantize(x, "e5m10"), quantize(x, "fp16")))
    check("fp16 overflow gives inf", np.isinf(quantize(jnp.float32(7e4), "fp16")))

    f = lambda a, mode: quantize(jnp.sin(a), "e5m4", ad=mode)
    t = jnp.cos(x) * 1.2345
    n_ste = len(np.unique(jax.jvp(lambda a: f(a, "ste"), (x,), (t,))[1]))
    n_full = len(np.unique(jax.jvp(lambda a: f(a, "full"), (x,), (t,))[1]))
    check(f"ste leaves derivative values alone ({n_ste} distinct)", n_ste > 3000)
    check(f"full rounds derivative values ({n_full} distinct)", n_full < 300)
    g = jax.grad(lambda a: (f(a, "full") * 1.2345).sum())(x)
    check("full rounds gradients (reverse mode)",
          len(np.unique(np.asarray(g) / np.cos(np.asarray(x)))) < 300)
    net = lambda a: quantize(jnp.tanh(quantize(2 * a, "fp16", ad="full")), "fp16", ad="full")
    d2 = jax.grad(jax.grad(net))(0.3)
    check("full: second derivative finite and close to exact",
          abs(d2 - jax.grad(jax.grad(lambda a: jnp.tanh(2 * a)))(0.3)) < 1e-2)
    try:
        from jax.experimental import jet
        out = jet.jet(net, (0.3,), ((1.0, 0.0, 0.0, 0.0),))
        check("full works under Taylor mode (jet), 4th order", np.all(np.isfinite(out[1])))
    except Exception as ex:
        check(f"jet: {type(ex).__name__}: {ex}", False)

    _OPT["ad"] = "full"
    wq = lambda a: quantize(a, "fp8_e4m3:ste")
    gq = jax.grad(lambda a: (wq(a) * 1.2345).sum())(x)
    check("per-hook ':ste' overrides global ad=full (8-bit value, exact gradient)",
          len(np.unique(wq(x))) <= 256 and np.allclose(gq, 1.2345))
    _OPT["ad"] = "ste"
    check("per-hook ':full' overrides global ad=ste",
          len(np.unique(jax.jvp(lambda a: quantize(jnp.sin(a), "e5m4:full"), (x,), (t,))[1])) < 300)
    for sp in (True, False):
        _OPT["scale_pow2"] = sp
        sc = float(_scale(x, 240.0))
        check(f"block scale pow2={sp}: {sc:.6g}", (np.log2(sc) % 1 == 0) == sp and sc * 240 >= float(jnp.max(jnp.abs(x))))
    _OPT["scale_pow2"] = True
    for fmt in ("fp8_e4m3", "fp8_e5m2", "int8"):
        q = jax.jit(lambda a, fmt=fmt: quantize(a, fmt, ad="ste"))(x)
        check(f"{fmt}: at most 256 distinct values ({len(np.unique(q))})", len(np.unique(q)) <= 256)

    # sequential narrow accumulator against a float64 reference
    X = np.asarray(h16(jax.random.normal(jax.random.PRNGKey(1), (64, 128))))
    W = np.asarray(h16(jax.random.normal(jax.random.PRNGKey(2), (128, 32)) * 0.1))
    b = np.asarray(h16(jax.random.normal(jax.random.PRNGKey(3), (32,)) * 0.1))
    for impl in ("scan", "unroll"):
        _OPT["acc_chunk"], _OPT["acc_impl"] = 48, impl   # 128 = 2*48 + 32
        y48 = np.asarray(qdot(X, W, b, "fp16"))
        check(f"qdot {impl}, chunk 48 with remainder: on fp16 grid, close to wide",
              np.array_equal(y48, h16(y48)) and np.allclose(y48, X @ W + b, rtol=0.05, atol=1e-2))
    _OPT["acc_chunk"], _OPT["acc_impl"] = 1, "scan"
    got = np.asarray(jax.jit(lambda a, w, c: qdot(a, w, c, "fp16"))(X, W, b))
    ref = np.tile(b.astype(np.float16), (64, 1))
    for k in range(128):
        ref = (ref.astype(np.float64) + np.outer(X[:, k].astype(np.float64), W[k].astype(np.float64))).astype(np.float16)
    mism = np.mean(got != ref.astype(np.float32))
    check(f"qdot fp16 matches sequential FMA reference (mismatch {mism:.1e})", mism < 1e-2)
    wide = X @ W + b
    print(f"      narrow vs wide accumulator, median rel. diff: "
          f"{np.median(np.abs(got - wide) / np.abs(wide)):.1e}")

    # stochastic rounding
    w = jax.random.normal(jax.random.PRNGKey(4), (2000,))
    keys = jax.random.split(jax.random.PRNGKey(5), 400)
    sr = jax.vmap(lambda k: stochastic_round(w, k, "fp16"))(keys)
    check("stochastic rounding lands on fp16 values", np.array_equal(sr, h16(sr)))
    bias_sr = float(jnp.max(jnp.abs(sr.mean(0) - w) / jnp.abs(w)))
    check(f"stochastic rounding is unbiased (max rel. bias {bias_sr:.1e})", bias_sr < 2e-4)
    upd = w * 1e-5  # an update far below fp16 spacing
    near = quantize(quantize(w, "fp16") + upd, "fp16")
    check("round-to-nearest drops a 1e-5 relative update",
          np.array_equal(near, quantize(w, "fp16")))
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
