"""
nn_core.py
----------
A minimal, dependency-free (NumPy-only) neural-network toolkit: a Dense
(fully-connected) layer with manual forward/backward passes, standard
activations with their derivatives, and an Adam optimiser.

This project deliberately avoids PyTorch/TensorFlow so that the whole
simulation runs anywhere NumPy runs, and so that every gradient used to train
the Transformer-lite encoder, the GNN, the Dirichlet actor and the
distributional critic is fully visible and auditable (useful for a thesis
appendix). Each layer/activation below has been verified with a numerical
gradient check (see `gradcheck.py`).
"""
import numpy as np


# ----------------------------------------------------------------------
# Activations (forward + derivative given the *input* to the activation)
# ----------------------------------------------------------------------
def tanh_fwd(x):
    return np.tanh(x)


def tanh_grad(x, dy):
    t = np.tanh(x)
    return dy * (1.0 - t ** 2)


def relu_fwd(x):
    return np.maximum(0.0, x)


def relu_grad(x, dy):
    return dy * (x > 0.0)


def softplus_fwd(x):
    # numerically stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softplus_grad(x, dy):
    return dy * (1.0 / (1.0 + np.exp(-x)))


def identity_fwd(x):
    return x


def identity_grad(x, dy):
    return dy


ACTIVATIONS = {
    "tanh": (tanh_fwd, tanh_grad),
    "relu": (relu_fwd, relu_grad),
    "softplus": (softplus_fwd, softplus_grad),
    "linear": (identity_fwd, identity_grad),
}


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ----------------------------------------------------------------------
# Adam optimiser (works on an arbitrary dict of {name: array} params/grads)
# ----------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ----------------------------------------------------------------------
# Dense layer: y = act(x @ W + b), supports batched input x: (batch, in)
# ----------------------------------------------------------------------
class Dense:
    def __init__(self, n_in, n_out, activation="linear", name="dense", rng=None):
        rng = rng or np.random.default_rng()
        limit = np.sqrt(6.0 / (n_in + n_out))
        self.name = name
        self.act_fwd, self.act_grad = ACTIVATIONS[activation]
        self.params = {
            f"{name}.W": rng.uniform(-limit, limit, size=(n_in, n_out)),
            f"{name}.b": np.zeros(n_out),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._cache = None

    def forward(self, x):
        """x: (..., n_in) -> y: (..., n_out). Caches for backward."""
        W = self.params[f"{self.name}.W"]
        b = self.params[f"{self.name}.b"]
        z = x @ W + b
        y = self.act_fwd(z)
        self._cache = (x, z)
        return y

    def backward(self, dy):
        """dy: (..., n_out) -> dx: (..., n_in). Accumulates self.grads."""
        x, z = self._cache
        dz = self.act_grad(z, dy)
        W = self.params[f"{self.name}.W"]
        # flatten leading (batch) dims for the matmul grads
        x2 = x.reshape(-1, x.shape[-1])
        dz2 = dz.reshape(-1, dz.shape[-1])
        self.grads[f"{self.name}.W"] += x2.T @ dz2
        self.grads[f"{self.name}.b"] += dz2.sum(axis=0)
        dx = dz @ W.T
        return dx

    def zero_grad(self):
        for k in self.grads:
            self.grads[k][...] = 0.0
