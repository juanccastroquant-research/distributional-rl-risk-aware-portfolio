"""
encoders.py
-----------
Two learned modules corresponding to sections 4.1-4.3 of the research
proposal:

  * AttentionEncoder  -- a lightweight, single-head additive-attention
    ("Transformer-lite") pooling over a lookback window of per-asset time
    series features. Plays the role of the Transformer encoder: it learns
    *which past days matter* for the current state instead of using a fixed
    lag structure.

  * GraphConvLayer -- a single Graph Convolutional layer (Kipf & Welling,
    2017 style: H' = act(A_hat @ H @ W)) operating over the physically-
    informed asset graph (fuel exposure / geography / correlation edges
    built in market_sim.py). Plays the role of the GNN: it propagates a
    shock at one node (e.g. a gas-heavy utility) to its physically-connected
    neighbours.

Both are hand-differentiated (no autograd) and unit-tested numerically in
gradcheck_encoders.py.
"""
import numpy as np
from nn_core import tanh_fwd, relu_fwd, relu_grad, softmax


class AttentionEncoder:
    """
    Additive-attention pooling over a lookback window.

    Input:  X of shape (n_assets, L, F)      -- L days, F features per asset
    Output: H of shape (n_assets, d_model)   -- one temporal embedding per asset

    K = tanh(X @ Wk + bk)            (n, L, dk)
    scores = K @ v                   (n, L)
    alpha = softmax(scores, axis=L)  (n, L)
    V = X @ Wv + bv                  (n, L, d_model)
    H = sum_L alpha * V              (n, d_model)
    """

    def __init__(self, n_features, d_attn, d_model, rng=None):
        rng = rng or np.random.default_rng()
        lim_k = np.sqrt(6.0 / (n_features + d_attn))
        lim_v = np.sqrt(6.0 / (n_features + d_model))
        self.params = {
            "attn.Wk": rng.uniform(-lim_k, lim_k, size=(n_features, d_attn)),
            "attn.bk": np.zeros(d_attn),
            "attn.v": rng.uniform(-0.1, 0.1, size=(d_attn,)),
            "attn.Wv": rng.uniform(-lim_v, lim_v, size=(n_features, d_model)),
            "attn.bv": np.zeros(d_model),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._cache = None

    def forward(self, X):
        Wk, bk, v = self.params["attn.Wk"], self.params["attn.bk"], self.params["attn.v"]
        Wv, bv = self.params["attn.Wv"], self.params["attn.bv"]

        Kpre = X @ Wk + bk                 # (n, L, dk)
        K = tanh_fwd(Kpre)                 # (n, L, dk)
        scores = K @ v                     # (n, L)
        alpha = softmax(scores, axis=-1)   # (n, L)
        V = X @ Wv + bv                     # (n, L, d_model)
        H = np.einsum("nl,nld->nd", alpha, V)  # (n, d_model)

        self._cache = (X, Kpre, K, scores, alpha, V)
        return H

    def backward(self, dH):
        """dH: (n, d_model) -> dX: (n, L, F). Accumulates self.grads."""
        X, Kpre, K, scores, alpha, V = self._cache
        n, L, F = X.shape
        d_model = V.shape[-1]
        Wk, v, Wv = self.params["attn.Wk"], self.params["attn.v"], self.params["attn.Wv"]

        # H = sum_l alpha_l * V_l
        dalpha = np.einsum("nd,nld->nl", dH, V)              # (n, L)
        dV = np.einsum("nd,nl->nld", dH, alpha)               # (n, L, d_model)

        # softmax backward: dscores = alpha * (dalpha - sum_l alpha_l dalpha_l)
        s = np.sum(alpha * dalpha, axis=-1, keepdims=True)
        dscores = alpha * (dalpha - s)                          # (n, L)

        # scores = K @ v
        dK = dscores[..., None] * v[None, None, :]              # (n, L, dk)
        dv = np.einsum("nl,nld->d", dscores, K)                 # (dk,)

        # K = tanh(Kpre)
        dKpre = dK * (1.0 - K ** 2)                              # (n, L, dk)

        # Kpre = X @ Wk + bk   ,  V = X @ Wv + bv
        X2 = X.reshape(-1, F)
        dKpre2 = dKpre.reshape(-1, dKpre.shape[-1])
        dV2 = dV.reshape(-1, d_model)

        dWk = X2.T @ dKpre2
        dbk = dKpre2.sum(axis=0)
        dWv = X2.T @ dV2
        dbv = dV2.sum(axis=0)

        dX = dKpre @ Wk.T + dV @ Wv.T                            # (n, L, F)

        self.grads["attn.Wk"] += dWk
        self.grads["attn.bk"] += dbk
        self.grads["attn.v"] += dv
        self.grads["attn.Wv"] += dWv
        self.grads["attn.bv"] += dbv
        return dX

    def zero_grad(self):
        for k in self.grads:
            self.grads[k][...] = 0.0


class GraphConvLayer:
    """
    Single Graph Convolutional layer: H' = relu(A_hat @ H @ W + b).
    A_hat is the fixed (precomputed, row-normalised, self-loop-added)
    physically-informed adjacency matrix, shape (n_assets, n_assets).
    """

    def __init__(self, d_in, d_out, rng=None):
        rng = rng or np.random.default_rng()
        lim = np.sqrt(6.0 / (d_in + d_out))
        self.params = {
            "gnn.W": rng.uniform(-lim, lim, size=(d_in, d_out)),
            "gnn.b": np.zeros(d_out),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._cache = None

    def forward(self, H, A_hat):
        W, b = self.params["gnn.W"], self.params["gnn.b"]
        AH = A_hat @ H                 # (n, d_in)
        Z = AH @ W + b                 # (n, d_out)
        out = relu_fwd(Z)
        self._cache = (H, A_hat, AH, Z)
        return out

    def backward(self, dout):
        H, A_hat, AH, Z = self._cache
        W = self.params["gnn.W"]
        dZ = relu_grad(Z, dout)                 # (n, d_out)
        dW = AH.T @ dZ
        db = dZ.sum(axis=0)
        dAH = dZ @ W.T                           # (n, d_in)
        dH = A_hat.T @ dAH                        # (n, d_in)
        self.grads["gnn.W"] += dW
        self.grads["gnn.b"] += db
        return dH

    def zero_grad(self):
        for k in self.grads:
            self.grads[k][...] = 0.0
