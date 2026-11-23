"""
RUN TEMPLATE: advection PINN benchmark (JAX-PI)

How to use
----------
1. Copy this file for every run and name it after the run, e.g.
       cp configs/run_template.py configs/adv_4x128_fp16_s0.py
2. Edit ONLY the "EDIT PER RUN" block below (and the notes).
3. Run it from examples/adv with:
       time WANDB_MODE=offline python main.py \
         --config=configs/adv_4x128_fp16_s0.py \
         --workdir=./runs/adv_4x128_fp16_s0 2>&1 | tee runs/adv_4x128_fp16_s0.log
   Do not override values on the command line. The file itself is then the
   complete record of the run, and re-running it reproduces the experiment.

Baseline recipe: sota.py (ModifiedMlp, 3*pi period, batch 8192) with
grad_norm weighting instead of NTK (NTK ran out of memory on the 8 GB RTX 2070).
"""
import ml_collections
import jax.numpy as jnp


def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "train"

    # =====================================================================
    # EDIT PER RUN
    # =====================================================================
    SEED = 2
    NUM_LAYERS = 4
    HIDDEN_DIM = 128
    MAX_STEPS = 50000
    BATCH_SIZE = 8192

    # Precision per component. Options: fp32, fp16, bf16, fp8_e4m3, fp8_e5m2, int8
    # (only has an effect once the hooks in jaxpi/archs.py use config.precision)
    PREC_WEIGHTS = "int8"
    PREC_ACT_IN = "int8"
    PREC_ACT_OUT = "fp32"
    PREC_EMBED = "fp32"
    PREC_OUT_LAYER = "fp32"

    GPU = "RTX2070"          # record where it ran
    NOTES = "baseline fp32 reference, 4x128, 10k steps"
    # =====================================================================

    # Weights & Biases (name is built from the settings above)
    prec_tag = "-".join([PREC_WEIGHTS, PREC_ACT_IN, PREC_ACT_OUT, PREC_EMBED, PREC_OUT_LAYER])
    if len(set([PREC_WEIGHTS, PREC_ACT_IN, PREC_ACT_OUT, PREC_EMBED, PREC_OUT_LAYER])) == 1:
        prec_tag = PREC_WEIGHTS
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Advection"
    wandb.name = f"adv_{NUM_LAYERS}x{HIDDEN_DIM}_{prec_tag}_s{SEED}"
    wandb.tag = None

    # Run metadata (stored with the config for traceability)
    config.meta = meta = ml_collections.ConfigDict()
    meta.gpu = GPU
    meta.notes = NOTES
    meta.baseline = "sota.py with grad_norm instead of ntk"

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = NUM_LAYERS
    arch.hidden_dim = HIDDEN_DIM
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (3 * jnp.pi, 1.0), "axis": (0, 1), "trainable": (True, False)}
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

    # Precision (read by the hooks you add to jaxpi/archs.py)
    config.precision = prec = ml_collections.ConfigDict()
    prec.weights = PREC_WEIGHTS
    prec.act_in = PREC_ACT_IN
    prec.act_out = PREC_ACT_OUT
    prec.embed = PREC_EMBED
    prec.out_layer = PREC_OUT_LAYER

    # Optim (keep identical across all runs that you compare)
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = MAX_STEPS
    training.batch_size_per_device = BATCH_SIZE

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 1.0, "res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000
    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False   # set True to diagnose gradient underflow
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = 10

    config.input_dim = 2
    config.seed = SEED

    return config
