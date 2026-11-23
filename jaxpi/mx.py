"""
Microscaling (MX) matmuls for the reduced-precision PINN emulation.

MX format (OCP Microscaling spec v1.0): along the summed dimension of a matmul,
every block of `block` (default 32) consecutive values shares one power-of-two
scale (E8M0), and each value is stored in a narrow element format:

  element   bits  exp/man  largest value   notes
  e4m3       8    4 / 3    448             MXFP8 (OCP E4M3, no infinity)
  e5m2       8    5 / 2    57344           MXFP8
  e3m2       6    3 / 2    28              MXFP6
  e2m3       6    2 / 3    7.5             MXFP6
  e2m1       4    2 / 1    6               MXFP4
  int8       8    -        127/64          MXINT8 (two's complement, implicit 2^-6)

shared exponent = floor(log2(max |x| in block)) - emax(element), as in the spec;
values above the element maximum saturate.

How derivatives are treated (the part that matters for PINNs)
------------------------------------------------------------
mx_matmul(a, b) is a custom-VJP function whose backward pass is built from
mx_matmul itself:

    forward      z  = out( mx(a, summed axis) @ mx(b, summed axis) )
    backward     da = mx_matmul(g, b^T)       g and b^T quantized along their own summed axis
                 db = sum over samples of mx(a) (x) mx(g)

So every matmul at every derivative order (u_x, u_xx and the parameter
gradient on top) quantizes its own operands, with fresh block scales, along
the dimension that product sums over, which is what MX hardware does.

Limitations
  * db: inside JAX-PI the network is vmapped over the batch, so the sum over
    the batch for the weight gradient is done by JAX outside this function, in
    FP32. Its operands are quantized in feature blocks rather than in blocks
    along the batch. db is also not rounded to the output format.
  * Forward-mode / Taylor-mode differentiation (jax.jvp, jax.experimental.jet)
    is not supported by custom_vjp, so this does not run on the KS example.
  * The matmul accumulates exactly and rounds once to `mx_out`.
"""
import jax
import jax.numpy as jnp
from jax import lax

ELEMENTS = {
    "e4m3": dict(E=4, M=3, maxval=448.0),
    "e5m2": dict(E=5, M=2, maxval=57344.0),
    "e3m2": dict(E=3, M=2, maxval=28.0),
    "e2m3": dict(E=2, M=3, maxval=7.5),
    "e2m1": dict(E=2, M=1, maxval=6.0),
    "int8": dict(int=True, maxval=127.0 / 64.0),
}
_EMAX = {"e4m3": 8, "e5m2": 15, "e3m2": 4, "e2m3": 2, "e2m1": 2, "int8": 0}

_CFG = {"elem": None, "block": 32, "out": None}   # out: (exp_bits, man_bits) or None for FP32


def configure(elem=None, block=32, out_bits=None):
    """Set by precision.set_precision; elem=None switches MX off."""
    if elem is not None and elem not in ELEMENTS:
        raise ValueError(f"unknown MX element format '{elem}', choose from {sorted(ELEMENTS)}")
    _CFG.update(elem=elem, block=int(block), out=out_bits)


def enabled():
    return _CFG["elem"] is not None


def _round_elem(v, elem):
    """Round v (already divided by the block scale) to the element format, saturating."""
    spec = ELEMENTS[elem]
    if spec.get("int"):
        return jnp.clip(jnp.round(v * 64.0), -127.0, 127.0) / 64.0
    E, M, maxval = spec["E"], spec["M"], spec["maxval"]
    emin = 2 - 2 ** (E - 1)                       # exponent of the smallest normal
    _, ex = jnp.frexp(jnp.abs(v))                 # |v| = m * 2**ex, m in [0.5, 1)
    e = jnp.maximum(ex - 1, emin)
    q = jnp.ldexp(jnp.ones_like(v), e - M)        # spacing in this binade (subnormals share emin)
    r = jnp.round(v / q) * q                      # jnp.round: half to even
    return jnp.clip(r, -maxval, maxval)


def mx_quantize(x, axis, elem=None, block=None):
    """Quantize x to MX with blocks of `block` values along `axis`. Returns float32 values."""
    elem = _CFG["elem"] if elem is None else elem
    block = _CFG["block"] if block is None else block
    x = lax.stop_gradient(x)
    xm = jnp.moveaxis(x, axis, -1)
    n = xm.shape[-1]
    b = block if (n >= block and n % block == 0) else n   # short or uneven axis: one block
    xb = xm.reshape(xm.shape[:-1] + (n // b, b))
    amax = jnp.max(jnp.abs(xb), axis=-1, keepdims=True)
    _, ex = jnp.frexp(jnp.where(amax > 0, amax, 1.0))
    shared = jnp.clip(ex - 1 - _EMAX[elem], -127, 127)  # floor(log2(amax)) - emax
    scale = jnp.ldexp(jnp.ones_like(amax), shared)
    q = _round_elem(xb / scale, elem) * scale
    return jnp.moveaxis(q.reshape(xm.shape), -1, axis)


def _out(z):
    bits = _CFG["out"]
    return z if bits is None else lax.reduce_precision(z, *bits)


def _ste(x, q):
    return x + lax.stop_gradient(q - x)


@jax.custom_vjp
def mx_matmul(a, b):
    """a [..., k] @ b [k, n] with both operands in MX along k; result rounded to mx_out."""
    return _out(jnp.matmul(mx_quantize(a, -1), mx_quantize(b, 0)))


def _fwd(a, b):
    return mx_matmul(a, b), (a, b)


def _bwd(res, g):
    a, b = res
    da = mx_matmul(g, b.T)                                     # sums over n
    aq = _ste(a, mx_quantize(a, -1))
    gq = _ste(g, mx_quantize(g, -1))
    db = jnp.einsum("...k,...n->kn", aq, gq)                   # sums over samples (if batched)
    return da, db


mx_matmul.defvjp(_fwd, _bwd)


# ---- self-test:  python3 -m jaxpi.mx ------------------------------------------
if __name__ == "__main__":
    import numpy as np

    ok = True

    def check(name, cond):
        global ok
        ok &= bool(cond)
        print(("PASS  " if cond else "FAIL  ") + name)

    rng = np.random.default_rng(0)

    def ref_quant(x, elem, block=32):
        """Independent numpy implementation along the last axis."""
        spec, out = ELEMENTS[elem], np.empty_like(x)
        for i in range(0, x.shape[-1], block):
            blk = x[..., i:i + block]
            amax = np.max(np.abs(blk), axis=-1, keepdims=True)
            sh = np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - _EMAX[elem]
            v = blk / 2.0 ** sh
            if spec.get("int"):
                r = np.clip(np.round(v * 64), -127, 127) / 64
            else:
                E, M = spec["E"], spec["M"]
                emin = 2 - 2 ** (E - 1)
                e = np.maximum(np.floor(np.log2(np.maximum(np.abs(v), 1e-300))), emin)
                q = 2.0 ** (e - M)
                r = np.clip(np.round(v / q) * q, -spec["maxval"], spec["maxval"])
            out[..., i:i + block] = r * 2.0 ** sh
        return out

    x = (rng.normal(0, 1, (4, 128)) * 10.0 ** rng.uniform(-3, 3, (4, 128))).astype(np.float32)
    for elem in ELEMENTS:
        got = np.asarray(mx_quantize(jnp.asarray(x), -1, elem))
        exp = ref_quant(x.astype(np.float64), elem)
        check(f"{elem}: matches independent reference", np.allclose(got, exp, rtol=0, atol=0))
        blk = got[0, :32] / 2.0 ** (np.floor(np.log2(np.max(np.abs(x[0, :32])))) - _EMAX[elem])
        check(f"{elem}: elements within +-{ELEMENTS[elem]['maxval']}",
              np.max(np.abs(blk)) <= ELEMENTS[elem]["maxval"])
    big = jnp.full((32,), 3e30)
    check("huge block: scale clipped, no overflow to inf",
          bool(jnp.all(jnp.isfinite(mx_quantize(big, -1, "e4m3")))))

    configure("e4m3", 32, (6, 9))
    a = jnp.asarray(rng.normal(0, 1, (8, 128)), jnp.float32)
    b = jnp.asarray(rng.normal(0, 0.1, (128, 64)), jnp.float32)
    z = mx_matmul(a, b)
    zr = lax.reduce_precision(mx_quantize(a, -1) @ mx_quantize(b, 0), 6, 9)
    check("forward: quantized operands, rounded output", np.array_equal(z, zr))
    g = jnp.asarray(rng.normal(0, 1, (8, 64)), jnp.float32)
    _, vjp = jax.vjp(mx_matmul, a, b)
    da, db = vjp(g)
    check("backward da = mx_matmul(g, b^T)", np.array_equal(da, mx_matmul(g, b.T)))
    check("backward db = sum over samples of mx(a) x mx(g)",
          np.allclose(db, mx_quantize(a, -1).T @ mx_quantize(g, -1), rtol=1e-6))

    # nested reverse mode, as PINN derivatives use
    W1 = jnp.asarray(rng.normal(0, 0.3, (32, 128)), jnp.float32)
    W2 = jnp.asarray(rng.normal(0, 0.1, (128, 1)), jnp.float32)
    f = lambda s, W1, W2: mx_matmul(jnp.tanh(mx_matmul(s * jnp.ones(32), W1)), W2)[0]
    d1 = jax.grad(f)(0.3, W1, W2)
    d2 = jax.grad(jax.grad(f))(0.3, W1, W2)
    gW = jax.grad(lambda W1: jax.grad(jax.grad(f))(0.3, W1, W2))(W1)
    check("1st/2nd derivative and parameter gradient of u_xx are finite",
          np.isfinite(d1) and np.isfinite(d2) and bool(jnp.all(jnp.isfinite(gW))))
    configure(None)
    ex = jax.grad(jax.grad(lambda s: (jnp.tanh((s * jnp.ones(32)) @ W1) @ W2)[0]))(0.3)
    configure("e4m3", 32, (6, 9))
    print(f"      u_xx with MXFP8: {float(d2):.4f}, exact FP32: {float(ex):.4f}")
    configure(None)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
