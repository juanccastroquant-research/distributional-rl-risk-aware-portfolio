"""
models.py
---------
The two decision-making components of the pipeline (LaTeX proposal S3):

  * DirichletActor   -- state -> Dirichlet concentration parameters alpha,
                        with the alpha-floor fix from S3.5 to avoid the
                        zero-weight boundary pathology, closed-form entropy
                        and log-density for a REINFORCE/score-function
                        policy-gradient update (the standard way Dirichlet
                        portfolio policies are trained in the literature,
                        e.g. Andre & Coqueret 2021).

  * QuantileCritic   -- state+action -> N fixed quantiles of the return
                        distribution (QR-DQN style distributional critic),
                        used both to bootstrap TD targets and to compute
                        Expected Shortfall directly from the learned tail
                        (S5, "Option A").

Both wrap an AttentionEncoder + GraphConvLayer state encoder (S4) feeding a
small Dense MLP head. NOTE: the actor and critic each own a SEPARATE
_EncoderStack instance (independent attention/GNN weights) -- they do not
share encoder parameters. Each network exposes its own merged params/grads
dict (all_params()/all_grads()) so it can be handed to its own Adam
optimiser; train.py creates two independent optimisers (actor_opt,
critic_opt), not one shared optimiser across both networks.
"""
import numpy as np
from scipy.special import gammaln, digamma, polygamma

from nn_core import Dense, softplus_fwd, softplus_grad
from encoders import AttentionEncoder, GraphConvLayer


def _flatten_state(H_temporal, H_graph, prev_w):
    """Concatenate per-asset temporal + graph embeddings and previous
    weights into a single flat global state vector (S4.3)."""
    fused = np.concatenate([H_temporal, H_graph], axis=-1)  # (n, d_model+d_g)
    return np.concatenate([fused.reshape(-1), prev_w.reshape(-1)])


class _EncoderStack:
    """Shared plumbing used by both the actor and the critic: attention
    encoder + GNN + state flattening, with a combined backward pass."""

    def __init__(self, n_features, n_assets, d_attn, d_model, d_graph, rng):
        self.attn = AttentionEncoder(n_features, d_attn, d_model, rng=rng)
        self.gnn = GraphConvLayer(d_model, d_graph, rng=rng)
        self.n_assets = n_assets
        self.d_model = d_model
        self.d_graph = d_graph
        self.state_dim = n_assets * (d_model + d_graph) + n_assets
        self._cache = None

    def forward(self, X, A_hat, prev_w):
        H_temporal = self.attn.forward(X)          # (n, d_model)
        H_graph = self.gnn.forward(H_temporal, A_hat)  # (n, d_graph)
        S = _flatten_state(H_temporal, H_graph, prev_w)
        self._cache = (H_temporal, H_graph, prev_w)
        return S

    def backward(self, dS):
        """dS: (state_dim,) -> pushes gradient back through GNN + attention."""
        n, dmod, dg = self.n_assets, self.d_model, self.d_graph
        fused_dim = n * (dmod + dg)
        dfused_flat = dS[:fused_dim]
        # d(prev_w) is not needed further (prev_w is environment state, not a param)
        dfused = dfused_flat.reshape(n, dmod + dg)
        dH_temporal_from_fuse = dfused[:, :dmod]
        dH_graph = dfused[:, dmod:]

        dH_temporal_from_gnn = self.gnn.backward(dH_graph)
        dH_temporal = dH_temporal_from_fuse + dH_temporal_from_gnn
        self.attn.backward(dH_temporal)

    def all_params(self):
        return {**self.attn.params, **self.gnn.params}

    def all_grads(self):
        return {**self.attn.grads, **self.gnn.grads}

    def zero_grad(self):
        self.attn.zero_grad()
        self.gnn.zero_grad()


class DirichletActor:
    def __init__(self, n_features, n_assets, cfg, rng=None):
        rng = rng or np.random.default_rng()
        self.n_assets = n_assets
        self.alpha_floor = cfg["alpha_floor"]
        self.encoder = _EncoderStack(
            n_features, n_assets,
            cfg["d_attn"], cfg["d_model"], cfg["d_graph"], rng,
        )
        self.h1 = Dense(self.encoder.state_dim, cfg["actor_hidden"], activation="tanh",
                         name="actor_h1", rng=rng)
        self.out = Dense(cfg["actor_hidden"], n_assets, activation="linear",
                          name="actor_out", rng=rng)
        self._cache = None

    def forward(self, X, A_hat, prev_w):
        S = self.encoder.forward(X, A_hat, prev_w)
        z1 = self.h1.forward(S)
        pre_alpha = self.out.forward(z1)
        alpha = softplus_fwd(pre_alpha) + self.alpha_floor
        self._cache = (S, pre_alpha)
        return alpha

    def sample(self, alpha, rng):
        return rng.dirichlet(alpha)

    @staticmethod
    def log_prob(w, alpha):
        w = np.clip(w, 1e-8, None)
        return (gammaln(alpha.sum()) - np.sum(gammaln(alpha))
                + np.sum((alpha - 1.0) * np.log(w)))

    @staticmethod
    def dlogprob_dalpha(w, alpha):
        """Analytic d/dalpha of log Dirichlet density (score function)."""
        w = np.clip(w, 1e-8, None)
        return digamma(alpha.sum()) - digamma(alpha) + np.log(w)

    @staticmethod
    def entropy(alpha):
        a0 = alpha.sum()
        n = alpha.shape[0]
        logB = np.sum(gammaln(alpha)) - gammaln(a0)
        return logB + (a0 - n) * digamma(a0) - np.sum((alpha - 1.0) * digamma(alpha))

    @staticmethod
    def dentropy_dalpha(alpha):
        """Exact gradient of the closed-form Dirichlet entropy w.r.t. alpha:
        dH/dalpha_i = (a0 - K) * psi1(a0) - (alpha_i - 1) * psi1(alpha_i),
        where psi1 is the trigamma function. Derived by differentiating
        H(alpha) = logB(alpha) + (a0-K)psi(a0) - sum_j (alpha_j-1)psi(alpha_j)."""
        a0 = alpha.sum()
        K = alpha.shape[0]
        psi1_a0 = polygamma(1, a0)
        psi1_alpha = polygamma(1, alpha)
        return (a0 - K) * psi1_a0 - (alpha - 1.0) * psi1_alpha

    def backward(self, dalpha):
        """dalpha: (n_assets,) gradient of the loss w.r.t. alpha (already
        includes the softplus derivative? No -- we pass gradient wrt alpha
        and apply the softplus chain rule here)."""
        S, pre_alpha = self._cache
        dpre_alpha = softplus_grad(pre_alpha, dalpha)
        dz1 = self.out.backward(dpre_alpha)
        dS = self.h1.backward(dz1)
        self.encoder.backward(dS)

    def all_params(self):
        return {**self.encoder.all_params(), **self.h1.params, **self.out.params}

    def all_grads(self):
        return {**self.encoder.all_grads(), **self.h1.grads, **self.out.grads}

    def zero_grad(self):
        self.encoder.zero_grad()
        self.h1.zero_grad()
        self.out.zero_grad()


class QuantileCritic:
    """QR-DQN-style distributional critic: predicts N_quantiles fixed
    quantile levels tau_i = (i+0.5)/N of the return distribution given
    (state, action)."""

    def __init__(self, n_features, n_assets, cfg, rng=None):
        rng = rng or np.random.default_rng()
        self.n_assets = n_assets
        self.n_quantiles = cfg["n_quantiles"]
        self.taus = (np.arange(self.n_quantiles) + 0.5) / self.n_quantiles
        self.encoder = _EncoderStack(
            n_features, n_assets,
            cfg["d_attn"], cfg["d_model"], cfg["d_graph"], rng,
        )
        in_dim = self.encoder.state_dim + n_assets
        self.h1 = Dense(in_dim, cfg["critic_hidden"], activation="tanh",
                         name="critic_h1", rng=rng)
        self.out = Dense(cfg["critic_hidden"], self.n_quantiles, activation="linear",
                          name="critic_out", rng=rng)
        self._cache = None

    def forward(self, X, A_hat, prev_w, action_w):
        S = self.encoder.forward(X, A_hat, prev_w)
        sa = np.concatenate([S, action_w])
        z1 = self.h1.forward(sa)
        quantiles = self.out.forward(z1)
        self._cache = (S, sa)
        return quantiles

    def backward(self, dquantiles):
        S, sa = self._cache
        dz1 = self.out.backward(dquantiles)
        dsa = self.h1.backward(dz1)
        dS = dsa[: len(S)]
        self.encoder.backward(dS)

    def expected_shortfall(self, quantiles, alpha_level):
        """Average of the worst (1 - alpha_level) fraction of the predicted
        quantiles (S5, 'critic-derived ES')."""
        k = max(1, int(np.ceil((1 - alpha_level) * self.n_quantiles)))
        sorted_q = np.sort(quantiles)
        return sorted_q[:k].mean(), k

    def all_params(self):
        return {**self.encoder.all_params(), **self.h1.params, **self.out.params}

    def all_grads(self):
        return {**self.encoder.all_grads(), **self.h1.grads, **self.out.grads}

    def zero_grad(self):
        self.encoder.zero_grad()
        self.h1.zero_grad()
        self.out.zero_grad()


def quantile_huber_loss_grad(pred, target, taus, kappa=1.0):
    """
    Pairwise quantile Huber loss (QR-DQN, Dabney et al. 2018) and its
    gradient w.r.t. `pred` (shape (N,)), against `target` quantile samples
    (shape (M,)); taus: shape (N,).
    Returns (loss_value, grad_wrt_pred).
    """
    N, M = len(pred), len(target)
    u = target[None, :] - pred[:, None]        # (N, M)
    abs_u = np.abs(u)
    huber = np.where(abs_u <= kappa, 0.5 * u ** 2, kappa * (abs_u - 0.5 * kappa))
    huber_grad_u = np.where(abs_u <= kappa, u, kappa * np.sign(u))
    indicator = (u < 0).astype(np.float64)
    weight = np.abs(taus[:, None] - indicator)
    loss = np.mean(weight * huber)
    # loss = (1/(N*M)) sum_ij weight_ij*huber_ij, so
    # d loss / d pred_i = -(1/N) * mean_j (weight_ij * dHuber/du_ij)
    grad = -np.mean(weight * huber_grad_u, axis=1) / N
    return loss, grad


def polyak_update(target_params, online_params, tau):
    for k in target_params:
        target_params[k][...] = (1 - tau) * target_params[k] + tau * online_params[k]


def clone_params(params):
    return {k: v.copy() for k, v in params.items()}
