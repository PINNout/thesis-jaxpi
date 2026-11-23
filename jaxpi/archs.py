from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict

from flax import linen as nn
from flax.core.frozen_dict import freeze

from jax import random, jit, vmap
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant

from jaxpi.precision import quantize, qdot, get_precision, to_native, from_native
from jaxpi import mx

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v

    return init


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied
    trainable: Tuple[
        bool
    ]  # Specifies whether the period for each axis is trainable or not

    def setup(self):
        # Initialize period parameters as trainable or constant and store them in a flax frozen dict
        period_params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                period_params[f"period_{idx}"] = self.param(
                    f"period_{idx}", constant(self.period[idx]), ()
                )
            else:
                period_params[f"period_{idx}"] = self.period[idx]

        self.period_params = freeze(period_params)

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []

        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None
    # PRECISION HOOK: which entries of config.precision apply to this layer.
    # Hidden layers use "weights" / "act_in"; the final layer overrides both
    # with "out_layer" (see Mlp / ModifiedMlp below).
    prec_w: str = "weights"
    prec_x: str = "act_in"

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )

        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        # Native mode: run the layer in the dtype of its input (fp16/bf16);
        # parameters themselves stay FP32 (master copy). No-op in FP32 mode.
        kernel = kernel.astype(x.dtype)
        bias = bias.astype(x.dtype)

        # PRECISION HOOK: round the layer input X, the effective weight W and
        # the bias Y; qdot computes Z = X.W + Y with the "acc" accumulator
        # format (fp32 = wide accumulator, same as jnp.dot(x, w) + bias).
        p = get_precision()
        if mx.enabled():
            # Microscaling: both operands quantized in blocks along the summed
            # dimension, at every derivative order (see jaxpi/mx.py).
            y = mx.mx_matmul(x, kernel) + quantize(bias, p["bias"])
        else:
            y = qdot(
                quantize(x, p[self.prec_x]),
                quantize(kernel, p[self.prec_w]),
                quantize(bias, p["bias"]),
            )

        return y


# TODO: Make it more general, e.g. imposing periodicity for the given axis


class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        p = get_precision()

        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
            x = quantize(x, p["embed"])  # PRECISION HOOK: embedding output

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)
            x = quantize(x, p["embed"])  # PRECISION HOOK: embedding output

        x = to_native(x)  # native mode only

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = quantize(self.activation_fn(x), p["act_out"])  # PRECISION HOOK

        x = Dense(
            features=self.out_dim,
            reparam=self.reparam,
            prec_w="out_layer",
            prec_x="out_layer",
        )(x)
        return from_native(x)


class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        p = get_precision()

        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
            x = quantize(x, p["embed"])  # PRECISION HOOK: embedding output

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)
            x = quantize(x, p["embed"])  # PRECISION HOOK: embedding output

        x = to_native(x)  # native mode only

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)

        u = quantize(self.activation_fn(u), p["act_out"])  # PRECISION HOOK
        v = quantize(self.activation_fn(v), p["act_out"])  # PRECISION HOOK

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = quantize(self.activation_fn(x), p["act_out"])  # PRECISION HOOK
            x = x * u + (1 - x) * v  # gating arithmetic stays FP32

        x = Dense(
            features=self.out_dim,
            reparam=self.reparam,
            prec_w="out_layer",
            prec_x="out_layer",
        )(x)
        return from_native(x)


class MlpBlock(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    activation: str
    reparam: Union[None, Dict]
    final_activation: bool

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        if self.final_activation:
            x = self.activation_fn(x)

        return x


class DeepONet(nn.Module):
    arch_name: Optional[str] = "DeepONet"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u, x):
        u = MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            final_activation=False,
            reparam=self.reparam,
        )(u)

        x = Mlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)

        y = u * x
        y = self.activation_fn(y)
        y = Dense(features=self.out_dim, reparam=self.reparam)(y)
        return y
