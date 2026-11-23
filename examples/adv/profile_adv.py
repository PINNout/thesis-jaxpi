import os, sys, time, functools

NAME = "adv_int8"      # adv_fp32 | adv_fp16 | adv_fp16_native | adv_bf16 | adv_int8

ADV = os.path.expanduser("~/thesis/jaxpi/examples/adv")
OUT = os.path.expanduser(f"~/thesis/profiling/results/{NAME}")
os.makedirs(OUT, exist_ok=True)
os.environ["XLA_FLAGS"] = (f"--xla_dump_to={OUT}/hlo --xla_dump_hlo_as_text --xla_gpu_enable_command_buffer=")

sys.path.insert(0, ADV); os.chdir(ADV)

import importlib.util
import jax, jax.numpy as jnp
import ml_collections
from jaxpi.utils import restore_checkpoint
from jaxpi.precision import set_precision
from jaxpi.samplers import UniformSampler
import models
from utils import get_dataset

# ---- config: sota.py + the overrides from the sweep file ----
spec = importlib.util.spec_from_file_location("sota", os.path.join(ADV, "configs", "sota.py"))
sota = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sota)

ALL = ["weights", "act_in", "act_out", "embed", "out_layer"]
def emu(fmt): return {k: fmt for k in ALL}
PREC = {
    "adv_fp32":        {**emu("fp32"), "native": None},
    "adv_fp16":        {**emu("fp16"), "native": None},
    "adv_fp16_native": {**emu("fp32"), "native": "fp16"},
    "adv_bf16":        {**emu("bf16"), "native": None},
    "adv_int8":        {**emu("int8"), "native": None},
}[NAME]

config = sota.get_config()
config.seed = 0
config.wandb.name = NAME
config.arch.num_layers = 4
config.arch.hidden_dim = 128
config.weighting.scheme = "grad_norm"
config.training.batch_size_per_device = 8192
config.training.max_steps = 50000
config.precision = ml_collections.ConfigDict(PREC)

print(config)
set_precision(config.get("precision"))   # must come before the model is built

# ---- setup copied from train.py ----
T, L, c, n_t, n_x = 1.0, 2 * jnp.pi, 80, 200, 128
u_ref, t_star, x_star = get_dataset(T, L, c, n_t, n_x)
u0 = u_ref[0, :]
dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]]])

model = models.Advection(config, u0, t_star, x_star, c)
batch = next(iter(UniformSampler(dom, config.training.batch_size_per_device)))

try:   # optional; a failure means the config doesn't match the trained run
    model.state = restore_checkpoint(model.state, os.path.join(ADV, NAME, "ckpt"))
except Exception as e:
    print("WARNING: checkpoint restore failed:", e)

# ---- named scopes (before the first jitted call) ----
def scoped(fn, name):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with jax.named_scope(name):
            return fn(*a, **k)
    return wrapper

nets = [n for n in dir(model) if n.endswith("_net") and callable(getattr(model, n))]
for n in nets + ["res_and_w", "losses"]:
    if hasattr(model, n):
        setattr(model, n, scoped(getattr(model, n), n))
print("scoped:", nets)

# ---- warm-up, wall-clock check, trace ----
state = model.state
for _ in range(5):
    state = model.step(state, batch)
jax.block_until_ready(state)

t0 = time.perf_counter()
for _ in range(20):
    state = model.step(state, batch)
jax.block_until_ready(state)
print(f"{(time.perf_counter() - t0) / 20 * 1e3:.2f} ms/step")

jax.profiler.start_trace(f"{OUT}/trace")
for _ in range(10):
    state = model.step(state, batch)
jax.block_until_ready(state)
jax.profiler.stop_trace()

if config.weighting.scheme in ("grad_norm", "ntk"):    # separate jitted function
    for _ in range(3):
        state = model.update_weights(state, batch)
    jax.block_until_ready(state)
    jax.profiler.start_trace(f"{OUT}/trace_weights")
    for _ in range(5):
        state = model.update_weights(state, batch)
    jax.block_until_ready(state)
    jax.profiler.stop_trace()

print("done ->", OUT)
