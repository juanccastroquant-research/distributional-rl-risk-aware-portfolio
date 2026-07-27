"""
train.py
--------
Orchestrates the full simulation described in the LaTeX research proposal:

  1. Simulate a physically-driven synthetic European energy market (S2, S6)
     with a calm training segment, a stressed test segment, and a calm test
     segment (S7).
  2. Build the Transformer-lite + GNN state encoder (S4), the Dirichlet
     actor (S3), and the QR distributional critic (S3), and train them with
     an entropy-regularised (S3.2), critic-derived-ES-aware (S5, Option A)
     actor-critic loop with a Polyak-averaged target critic and automatic
     temperature tuning (S3.2).
  3. Evaluate the trained policy on both out-of-sample test segments and
     compare it against the recommended benchmark suite (S7): 1/N,
     buy-and-hold, static cap-weighted, rolling CVaR-optimal LP, and risk
     parity, with a bootstrap significance test (S8.1).
  4. Save a metrics table and diagnostic plots.

Run with:  python3 train.py [path/to/config.yaml]
"""
import sys
import copy
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import market_sim
import data_loader
from market_sim import (simulate_market, build_asset_graph, causal_rolling_normalise,
                         PortfolioEnv, project_capped_simplex)

# Fallback names kept for anything that imports these directly from train.py;
# make_report/train() below always resolve the ACTUAL asset names/count from
# the chosen data source at runtime rather than relying on these.
N_ASSETS = market_sim.N_ASSETS
ASSET_NAMES = market_sim.ASSET_NAMES
from models import DirichletActor, QuantileCritic, quantile_huber_loss_grad, polyak_update, clone_params
from nn_core import Adam
from benchmarks import (equal_weight, static_cap_weight, cvar_optimal_weights,
                         risk_parity_weights, RebalancingBenchmark)
from metrics import performance_summary, bootstrap_sharpe_diff, relative_performance


# ----------------------------------------------------------------------
def scale_grads(grads, factor):
    for k in grads:
        grads[k] *= factor


def grad_norm(grads):
    return float(np.sqrt(sum(np.sum(g ** 2) for g in grads.values())))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def add(self, item):
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self.buf), size=min(batch_size, len(self.buf)))
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


class RollingES:
    """Causal rolling-window historical ES tracker (S5, Option B / H4 ablation)."""

    def __init__(self, window, alpha_level):
        self.window = window
        self.alpha = alpha_level
        self.hist = []

    def update_and_query(self, port_return):
        self.hist.append(port_return)
        if len(self.hist) > self.window:
            self.hist.pop(0)
        if len(self.hist) < 10:
            return 0.0
        arr = np.array(self.hist)
        k = max(1, int(np.ceil((1 - self.alpha) * len(arr))))
        return np.sort(arr)[:k].mean()


# ----------------------------------------------------------------------
def build_agent(cfg, n_features, n_assets, rng):
    actor_cfg = dict(d_attn=cfg["state"]["d_attn"], d_model=cfg["state"]["d_model"],
                      d_graph=cfg["state"]["d_graph"], actor_hidden=cfg["actor"]["actor_hidden"],
                      alpha_floor=cfg["actor"]["alpha_floor"])
    critic_cfg = dict(d_attn=cfg["state"]["d_attn"], d_model=cfg["state"]["d_model"],
                       d_graph=cfg["state"]["d_graph"], critic_hidden=cfg["critic"]["critic_hidden"],
                       n_quantiles=cfg["critic"]["n_quantiles"])

    actor = DirichletActor(n_features, n_assets, actor_cfg, rng=rng)
    critic = QuantileCritic(n_features, n_assets, critic_cfg, rng=rng)
    critic_target = QuantileCritic(n_features, n_assets, critic_cfg, rng=rng)
    # sync target critic to online critic at init
    tgt_params, online_params = critic_target.all_params(), critic.all_params()
    for k in tgt_params:
        tgt_params[k][...] = online_params[k]
    return actor, critic, critic_target


def critic_update(critic, critic_target, actor, batch, A_hat_eff, gamma, taus, kappa,
                   critic_opt, episode=None, grad_attribution_log=None):
    critic.zero_grad()
    total_loss = 0.0
    spreads = []
    for (X_t, prev_w, w_t, reward, X_next, prev_w_next, done) in batch:
        quantiles_online = critic.forward(X_t, A_hat_eff, prev_w, w_t)
        spreads.append(float(quantiles_online.max() - quantiles_online.min()))
        alpha_next = actor.forward(X_next, A_hat_eff, prev_w_next)
        w_next_mean = alpha_next / alpha_next.sum()
        quantiles_next = critic_target.forward(X_next, A_hat_eff, prev_w_next, w_next_mean)
        target = reward + gamma * (1.0 - float(done)) * quantiles_next
        loss, grad = quantile_huber_loss_grad(quantiles_online, target, taus, kappa)
        critic.backward(grad)
        total_loss += loss
    n = len(batch)

    # ADDED: gradient-attribution check. critic_h1.W has shape
    # (state_dim + n_assets, hidden) -- its rows are indexed by the INPUT
    # dimension of the concatenated [state, action_w] vector fed into
    # critic_h1. The gradient accumulated on the last n_assets rows is
    # therefore exactly the gradient contribution attributable to the
    # action input; the gradient on the first state_dim rows is attributable
    # to the state encoding. This tells us whether the critic's OWN training
    # signal (from the TD/quantile-Huber loss) is meaningfully using the
    # action input at all, as opposed to the forward-pass counterfactual
    # sensitivity test (which showed the critic's PREDICTIONS don't vary
    # much with action) -- this checks whether the critic is even trying
    # to learn a relationship with the action, from the gradient's own
    # perspective, independent of what it has learned so far.
    if grad_attribution_log is not None:
        state_dim = critic.encoder.state_dim
        dW = critic.h1.grads["critic_h1.W"]  # (state_dim + n_assets, hidden), accumulated over batch
        state_grad_norm = float(np.linalg.norm(dW[:state_dim, :]))
        action_grad_norm = float(np.linalg.norm(dW[state_dim:, :]))
        grad_attribution_log.append(dict(
            episode=episode,
            state_grad_norm=state_grad_norm,
            action_grad_norm=action_grad_norm,
            action_grad_fraction=action_grad_norm / (state_grad_norm + action_grad_norm + 1e-12),
        ))

    scale_grads(critic.all_grads(), 1.0 / n)
    gn = grad_norm(critic.all_grads())
    critic_opt.step(critic.all_params(), critic.all_grads())
    critic.zero_grad()
    return total_loss / n, gn, float(np.mean(spreads))


def actor_update(actor, critic, rollout, A_hat_eff, cfg, beta, es_alpha, lambda_es,
                  es_mode, historical_es_tracker_values, baseline_state, actor_opt,
                  apply_update=True, episode=None, per_asset_log=None):
    actor.zero_grad()
    entropies, alpha_sums, max_weights, objectives = [], [], [], []
    adv_term_sq_sum, ent_term_sq_sum = 0.0, 0.0
    for i, (X_t, prev_w, w_t) in enumerate(rollout):
        alpha = actor.forward(X_t, A_hat_eff, prev_w)
        quantiles = critic.forward(X_t, A_hat_eff, prev_w, w_t)
        q_mean = float(quantiles.mean())
        if es_mode == "critic":
            es_hat, _ = critic.expected_shortfall(quantiles, es_alpha)
        else:
            es_hat = historical_es_tracker_values[i]
        objective = q_mean - lambda_es * es_hat

        # Advantage is computed against the baseline as it stood BEFORE this
        # step's objective is folded in. (Previously the EMA baseline was
        # updated first and the advantage taken against the *updated* value,
        # which silently shrinks every advantage by a factor of
        # (1 - baseline_ema) since part of the current objective had already
        # been mixed into the thing being subtracted from it.)
        advantage = objective - baseline_state["value"]
        baseline_state["value"] = ((1 - cfg["actor"]["baseline_ema"]) * baseline_state["value"]
                                    + cfg["actor"]["baseline_ema"] * objective)

        dlogp = DirichletActor.dlogprob_dalpha(w_t, alpha)
        dent = DirichletActor.dentropy_dalpha(alpha)
        entropies.append(DirichletActor.entropy(alpha))
        alpha_sums.append(float(alpha.sum()))
        max_weights.append(float(w_t.max()))
        objectives.append(objective)

        # Diagnostic (proposal #4): how much of the actor's gradient comes from
        # the entropy bonus vs. the actual return/ES advantage signal, per step.
        adv_term = advantage * dlogp
        ent_term = beta * dent
        adv_term_sq_sum += float(np.sum(adv_term ** 2))
        ent_term_sq_sum += float(np.sum(ent_term ** 2))

        # ADDED: per-asset diagnostic. adv_term is already per-asset (dlogp
        # varies by asset i via w_i, alpha_i) -- this is the actual per-asset
        # gradient contribution driving the policy update. Its cross-sectional
        # std (across the n_assets, this step) vs. how much the *scalar*
        # advantage itself moves step-to-step tells us whether there is a
        # real, exploitable cross-asset signal or whether adv_term's spread
        # is basically noise. NOTE: `advantage` itself is a single scalar per
        # step (see objective = q_mean - lambda_es*es_hat, both portfolio-
        # level) -- there is no per-asset value signal in this critic
        # architecture at all; any cross-asset differentiation can only come
        # through dlogp (i.e. through which asset got more/less weight when
        # advantage happened to be positive/negative), which is a REINFORCE-
        # style, high-variance credit-assignment mechanism, not a direct
        # per-asset value estimate.
        if per_asset_log is not None:
            per_asset_log.append(dict(
                episode=episode, step=i,
                advantage=float(advantage),
                adv_term_std_across_assets=float(adv_term.std()),
                adv_term_max_minus_min=float(adv_term.max() - adv_term.min()),
                dlogp_std_across_assets=float(dlogp.std()),
                w_t_std_across_assets=float(np.std(w_t)),
            ))

        dloss_dalpha = -(adv_term + ent_term)
        actor.backward(dloss_dalpha)

    n = len(rollout)
    scale_grads(actor.all_grads(), 1.0 / n)
    gn = grad_norm(actor.all_grads())
    if apply_update:
        actor_opt.step(actor.all_params(), actor.all_grads())
    actor.zero_grad()

    adv_norm = np.sqrt(adv_term_sq_sum)
    ent_norm = np.sqrt(ent_term_sq_sum)
    entropy_frac = ent_norm / (adv_norm + ent_norm + 1e-12)
    diag = dict(avg_entropy=float(np.mean(entropies)), avg_alpha_sum=float(np.mean(alpha_sums)),
                avg_max_weight=float(np.mean(max_weights)), grad_norm=gn,
                avg_objective=float(np.mean(objectives)), entropy_frac=float(entropy_frac))
    return diag


# ----------------------------------------------------------------------
# ADDED: critic counterfactual probe.
#
# QuantileCritic.forward(X, A_hat, prev_w, action_w) accepts ANY action_w --
# it evaluates "what does the critic think the return distribution looks
# like if THIS allocation is chosen", for a state that already happened.
# There is no per-asset value head in this architecture, but we can still
# directly test whether the critic differentiates between assets by asking
# it to evaluate a set of counterfactual allocations, each concentrated in
# a different single asset, all using the SAME state (X_t, prev_w). If the
# critic's predicted quantile mean varies meaningfully across which asset
# is concentrated in, there IS a real cross-asset signal available for the
# actor to exploit (and a flat policy would point to an actor-side issue --
# entropy control, transaction-cost penalty, or optimisation). If the
# critic predicts essentially the SAME value regardless of which asset is
# overweighted, there is no cross-asset signal at that state for the actor
# to act on at all -- pointing at the reward/critic side, not the actor.
# ----------------------------------------------------------------------
def critic_counterfactual_probe(critic, X_t, A_hat_eff, prev_w, w_max, n_assets):
    q_means = []
    for i in range(n_assets):
        w_i = np.full(n_assets, (1.0 - w_max) / (n_assets - 1))
        w_i[i] = w_max
        quantiles = critic.forward(X_t, A_hat_eff, prev_w, w_i)
        q_means.append(float(quantiles.mean()))
    q_means = np.array(q_means)
    return {
        "q_mean_per_asset": q_means,
        "cross_asset_std": float(q_means.std()),
        "cross_asset_spread": float(q_means.max() - q_means.min()),
        "best_asset_idx": int(np.argmax(q_means)),
        "worst_asset_idx": int(np.argmin(q_means)),
    }


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def train(cfg):
    rng = np.random.default_rng(cfg["seed"])

    data_source = cfg["market"].get("data_source", "real")
    if data_source == "real":
        data = data_loader.load_real_market(cfg["market"])
        asset_names = list(data_loader.TICKERS if "tickers" not in cfg["market"]
                            else cfg["market"]["tickers"])
        n_assets = len(asset_names)
        # No unit-interval channel in the real OHLCV-derived feature set (see
        # data_loader.build_features docstring) -- every channel gets fully
        # z-scored so all 6 channels end up on a comparable scale.
        feat_norm = causal_rolling_normalise(data["features"], window=cfg["state"]["normalise_window"],
                                              unit_interval_channels=())
        A_hat_physical = data_loader.build_empirical_asset_graph(
            data["returns"], data["regime"], cfg["state"]["graph_similarity_threshold"])
    elif data_source == "synthetic":
        data = simulate_market(cfg["market"], rng=np.random.default_rng(cfg["seed"]))
        asset_names = list(market_sim.ASSET_NAMES)
        n_assets = market_sim.N_ASSETS
        feat_norm = causal_rolling_normalise(data["features"], window=cfg["state"]["normalise_window"])
        A_hat_physical = build_asset_graph(cfg["state"]["graph_similarity_threshold"])
    else:
        raise ValueError(f"Unknown cfg['market']['data_source']: {data_source!r} "
                          f"(expected 'real' or 'synthetic')")

    # Keep the reported day-counts in sync with what actually loaded, for
    # both data sources (a no-op for synthetic, since it's built to match
    # these counts exactly by construction; load-bearing for real data).
    for label, key in (("train", "n_train_days"), ("stressed_test", "n_stressed_test_days"),
                        ("calm_test", "n_calm_test_days")):
        cfg["market"][key] = int(np.sum(np.asarray(data["regime"]) == label))

    A_hat_identity = np.eye(n_assets)
    A_hat_eff = A_hat_physical if cfg["state"]["use_gnn"] else A_hat_identity

    env = PortfolioEnv(feat_norm, data["returns"], data["regime"],
                        lookback=cfg["state"]["lookback"], lambda_tc=cfg["costs"]["lambda_tc"],
                        w_max=cfg["actor"]["w_max"])

    n_features = feat_norm.shape[-1]
    actor, critic, critic_target = build_agent(cfg, n_features, n_assets, rng)
    actor_opt = Adam(actor.all_params(), lr=cfg["actor"]["lr"])
    critic_opt = Adam(critic.all_params(), lr=cfg["critic"]["lr"])

    replay = ReplayBuffer(cfg["critic"]["replay_capacity"])

    log_beta = np.log(cfg["actor"]["beta_init"])
    h_uniform = DirichletActor.entropy(np.ones(n_assets))  # H_max: entropy ceiling
                                                            # given alpha_floor >= 1
    target_entropy = h_uniform - cfg["actor"]["entropy_target_margin"]  # always < H_max
    log_beta_min = np.log(cfg["actor"]["beta_min"])
    log_beta_max = np.log(cfg["actor"]["beta_max"])
    baseline_state = {"value": 0.0}

    train_start, train_end = env.segment_bounds("train")
    start_t = train_start + cfg["state"]["lookback"]

    taus = critic.taus
    kappa = cfg["critic"]["huber_kappa"]
    gamma = cfg["critic"]["gamma"]

    history = {"episode": [], "avg_reward": [], "avg_entropy": [], "beta": [], "critic_loss": [],
                "actor_grad_norm": [], "critic_grad_norm": [], "avg_alpha_sum": [],
                "avg_max_weight": [], "avg_quantile_spread": [], "avg_objective": [],
                "episode_cum_return": [], "entropy_frac": [], "actor_warmed_up": [],
                "actor_updates_this_episode": []}
    total_critic_updates = 0
    total_actor_updates = 0
    warmup_target = cfg["training"]["critic_warmup_updates"]

    # ADDED: diagnostic collectors (see actor_update's per_asset_log and the
    # critic_counterfactual_probe call below). These are returned in the
    # trained dict and saved separately -- they do not affect training at all.
    per_asset_log = []
    counterfactual_log = []
    grad_attribution_log = []

    # ------------------------------------------------------------------
    # FIX: previously the actor took exactly ONE gradient step per episode
    # (after accumulating the whole ~1500+ step rollout), so n_episodes=120
    # meant only 120 actor parameter updates in the entire run -- two to
    # three orders of magnitude fewer than the critic gets (which updates
    # every 5 steps). That mismatch, not the entropy bonus, is why the
    # actor's gradient norm was still steadily declining (not plateaued) at
    # the final logged episode: it simply hadn't had enough updates to
    # converge. `actor_update_every_steps` lets the actor update every K
    # environment steps instead of once per full episode, using the SAME
    # total amount of rollout data but far more gradient steps. Set it to
    # None (or omit it from config.yaml) to recover the exact old
    # once-per-episode behaviour.
    # ------------------------------------------------------------------
    actor_update_every = cfg["training"].get("actor_update_every_steps") or (train_end - start_t)

    print(f"Training on {train_end - start_t} days/episode, {cfg['training']['n_episodes']} episodes, "
          f"actor update every {actor_update_every} steps...")
    t0 = time.time()
    for episode in range(cfg["training"]["n_episodes"]):
        obs = env.reset(start_t)
        done = False
        es_tracker = RollingES(cfg["risk"]["historical_es_window"], cfg["risk"]["es_alpha"])
        rewards = []
        episode_port_returns = []
        critic_losses, critic_grad_norms, quantile_spreads = [], [], []
        chunk_rollout, chunk_es_values = [], []
        episode_diags = []  # list of (diag, n_steps_in_chunk, warmed_up), for within-episode aggregation
        step_i = 0

        def _flush_actor_update():
            """Run one actor gradient step on whatever is in chunk_rollout,
            update beta, log the diagnostic, and clear the chunk buffers."""
            nonlocal log_beta, total_actor_updates
            beta = np.exp(log_beta)
            warmed_up = total_critic_updates >= warmup_target
            diag = actor_update(actor, critic, chunk_rollout, A_hat_eff, cfg, beta,
                                 cfg["risk"]["es_alpha"], cfg["risk"]["lambda_es"],
                                 cfg["risk"]["es_mode"], chunk_es_values,
                                 baseline_state, actor_opt, apply_update=warmed_up,
                                 episode=episode, per_asset_log=per_asset_log)
            if warmed_up:
                log_beta = float(np.clip(
                    log_beta + cfg["actor"]["beta_lr"] * (target_entropy - diag["avg_entropy"]),
                    log_beta_min, log_beta_max))
                total_actor_updates += 1
            episode_diags.append((diag, len(chunk_rollout), warmed_up))
            chunk_rollout.clear()
            chunk_es_values.clear()

        while env.t < train_end:
            X_t, prev_w = obs

            # ADDED: once per episode (first step only, to keep cost low),
            # probe the critic's cross-asset differentiation at this state.
            if step_i == 0:
                cf = critic_counterfactual_probe(critic, X_t, A_hat_eff, prev_w,
                                                  cfg["actor"]["w_max"], n_assets)
                cf["episode"] = episode
                counterfactual_log.append(cf)

            alpha = actor.forward(X_t, A_hat_eff, prev_w)
            w_raw = actor.sample(alpha, rng)
            w_t = project_capped_simplex(w_raw, cfg["actor"]["w_max"])  # the action actually executed
            (X_next, prev_w_next), reward, done, info = env.step(w_t)

            chunk_es_values.append(es_tracker.update_and_query(info["port_return"]))
            chunk_rollout.append((X_t, prev_w, w_t))
            rewards.append(reward)
            episode_port_returns.append(info["port_return"])
            replay.add((X_t, prev_w, w_t, reward, X_next, prev_w_next, done))

            step_i += 1
            if len(replay) >= cfg["critic"]["batch_size"] and step_i % 5 == 0:
                batch = replay.sample(cfg["critic"]["batch_size"], rng)
                cl, cgn, spread = critic_update(critic, critic_target, actor, batch, A_hat_eff,
                                                 gamma, taus, kappa, critic_opt,
                                                 episode=episode, grad_attribution_log=grad_attribution_log)
                total_critic_updates += 1
                critic_losses.append(cl)
                critic_grad_norms.append(cgn)
                quantile_spreads.append(spread)
                polyak_update(critic_target.all_params(), critic.all_params(),
                              cfg["critic"]["target_tau"])

            obs = (X_next, prev_w_next)

            if len(chunk_rollout) >= actor_update_every:
                _flush_actor_update()

        if chunk_rollout:  # leftover partial chunk at the end of the episode
            _flush_actor_update()

        # Aggregate the (possibly several) actor updates this episode into a
        # single set of episode-level diagnostics, weighted by chunk length,
        # so history/plots/reports keep exactly one row per episode.
        n_total = sum(n for _, n, _ in episode_diags)
        def _wavg(key):
            return sum(d[key] * n for d, n, _ in episode_diags) / n_total
        avg_entropy = _wavg("avg_entropy")
        diag = dict(avg_alpha_sum=_wavg("avg_alpha_sum"), avg_max_weight=_wavg("avg_max_weight"),
                    grad_norm=_wavg("grad_norm"), avg_objective=_wavg("avg_objective"),
                    entropy_frac=_wavg("entropy_frac"))
        warmed_up = episode_diags[-1][2]  # whether the actor was updating by episode's end
        n_actor_updates_this_ep = sum(1 for _, _, w in episode_diags if w)

        history["episode"].append(episode)
        history["avg_reward"].append(float(np.mean(rewards)))
        history["episode_cum_return"].append(float(np.prod(1.0 + np.array(episode_port_returns)) - 1.0))
        history["avg_entropy"].append(avg_entropy)
        history["beta"].append(float(np.exp(log_beta)))
        history["critic_loss"].append(float(np.mean(critic_losses)) if critic_losses else np.nan)
        history["actor_grad_norm"].append(diag["grad_norm"])
        history["critic_grad_norm"].append(float(np.mean(critic_grad_norms)) if critic_grad_norms else np.nan)
        history["avg_alpha_sum"].append(diag["avg_alpha_sum"])
        history["avg_max_weight"].append(diag["avg_max_weight"])
        history["avg_quantile_spread"].append(float(np.mean(quantile_spreads)) if quantile_spreads else np.nan)
        history["avg_objective"].append(diag["avg_objective"])
        history["entropy_frac"].append(diag["entropy_frac"])
        history["actor_warmed_up"].append(bool(warmed_up))
        history["actor_updates_this_episode"].append(n_actor_updates_this_ep)

        if episode % cfg["training"]["log_every"] == 0 or episode == cfg["training"]["n_episodes"] - 1:
            warmup_note = "" if warmed_up else f"  [actor WARMUP: {total_critic_updates}/{warmup_target} critic updates]"
            print(f"  episode {episode:4d}  avg_reward={history['avg_reward'][-1]:+.5f}  "
                  f"entropy={avg_entropy:.3f} (target {target_entropy:.3f})  "
                  f"beta={history['beta'][-1]:.4f}  critic_loss={history['critic_loss'][-1]:.5f}  "
                  f"actor_gn={diag['grad_norm']:.4f}  critic_gn={history['critic_grad_norm'][-1]:.4f}  "
                  f"entropy_frac={diag['entropy_frac']:.3f}  actor_updates/ep={n_actor_updates_this_ep}"
                  f"  total_actor_updates={total_actor_updates}{warmup_note}")

    print(f"Training finished in {time.time()-t0:.1f}s "
          f"({total_actor_updates} total actor updates, {total_critic_updates} total critic updates)")
    return dict(actor=actor, critic=critic, env=env, data=data, A_hat_eff=A_hat_eff,
                history=history, target_entropy=target_entropy,
                asset_names=asset_names, n_assets=n_assets, data_source=data_source,
                total_actor_updates=total_actor_updates, total_critic_updates=total_critic_updates,
                per_asset_log=per_asset_log, counterfactual_log=counterfactual_log,
                grad_attribution_log=grad_attribution_log)


def evaluate_policy(actor, env, A_hat_eff, segment_name, rng):
    lo, hi = env.segment_bounds(segment_name)
    obs = env.reset(lo)
    port_returns, turnovers, weights_hist = [], [], []
    while env.t < hi:
        X_t, prev_w = obs
        alpha = actor.forward(X_t, A_hat_eff, prev_w)
        w_t = alpha / alpha.sum()  # deterministic (mean-action) evaluation policy
        obs, reward, done, info = env.step(w_t)
        port_returns.append(info["port_return"])
        turnovers.append(info["turnover"])
        weights_hist.append(info["weights"])
    return np.array(port_returns), np.array(turnovers), np.array(weights_hist)


def evaluate_benchmarks(cfg, data, segment_name):
    raw = data["returns"].values
    regime = data["regime"]
    mask = regime == segment_name
    n = raw.shape[1]
    w_max = cfg["actor"]["w_max"]
    bcfg = cfg["benchmarks"]

    results = {}

    # 1/N (rebalanced at the same frequency as the other benchmarks)
    rb = RebalancingBenchmark(lambda w_hist, wm: equal_weight(n), bcfg["rebalance_every"],
                               bcfg["scenario_window"])
    results["1/N"] = rb.run(raw, mask, w_max)

    # static cap-weighted (no active rule; still "rebalance" back to the fixed target)
    # VALIDATION: `benchmarks.static_cap_weights` is edited independently of
    # `market.tickers` in config.yaml -- previously there was no check here at
    # all, so changing the asset universe (adding/removing a ticker, or
    # switching data sources) without updating static_cap_weights to match
    # crashed downstream with an opaque numpy broadcast ValueError inside
    # RebalancingBenchmark.run(). Fall back to equal weights (with a warning)
    # instead of crashing.
    static_w_cfg = bcfg["static_cap_weights"]
    if len(static_w_cfg) != n:
        print(f"[benchmarks] WARNING: benchmarks.static_cap_weights has "
              f"{len(static_w_cfg)} entries but the current asset universe has "
              f"{n} assets -- falling back to equal weights for the "
              f"'CapWeighted' benchmark. Update static_cap_weights in your "
              f"config to match market.tickers to use real cap weights.")
        static_w = np.ones(n) / n
    else:
        static_w = np.array(static_w_cfg)
    rb = RebalancingBenchmark(lambda w_hist, wm: static_cap_weight(static_w),
                               bcfg["rebalance_every"], bcfg["scenario_window"])
    results["CapWeighted"] = rb.run(raw, mask, w_max)

    # buy-and-hold: rebalance_every set larger than the segment length
    rb = RebalancingBenchmark(lambda w_hist, wm: equal_weight(n), 10 ** 9, bcfg["scenario_window"])
    results["BuyAndHold"] = rb.run(raw, mask, w_max)

    # rolling CVaR-optimal LP
    rb = RebalancingBenchmark(
        lambda w_hist, wm: cvar_optimal_weights(w_hist, bcfg["cvar_alpha"], wm,
                                                 bcfg["cvar_lambda_return"]),
        bcfg["rebalance_every"], bcfg["scenario_window"])
    results["CVaR-LP"] = rb.run(raw, mask, w_max)

    # risk parity / ERC
    rb = RebalancingBenchmark(lambda w_hist, wm: risk_parity_weights(w_hist, wm),
                               bcfg["rebalance_every"], bcfg["scenario_window"])
    results["RiskParity"] = rb.run(raw, mask, w_max)

    return results


def make_report(cfg, trained, out_dir="energy_rl_sim_results"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"] + 1)

    actor, env, data, A_hat_eff = (trained["actor"], trained["env"], trained["data"],
                                    trained["A_hat_eff"])

    all_rows = []
    curves = {}
    for segment in ["stressed_test", "calm_test"]:
        port_ret, turn, wh = evaluate_policy(actor, env, A_hat_eff, segment, rng)
        summ = performance_summary(port_ret, turn, wh, es_alpha=cfg["risk"]["es_alpha"])
        summ["strategy"] = "RL-Dirichlet-DistCritic"
        summ["segment"] = segment
        all_rows.append(summ)
        curves[(segment, "RL-Dirichlet-DistCritic")] = port_ret

        bres = evaluate_benchmarks(cfg, data, segment)
        for name, (bp, bt, bw) in bres.items():
            s = performance_summary(bp, bt, bw, es_alpha=cfg["risk"]["es_alpha"])
            s["strategy"] = name
            s["segment"] = segment
            all_rows.append(s)
            curves[(segment, name)] = bp

    df = pd.DataFrame(all_rows).set_index(["segment", "strategy"])
    es_a = int(cfg["risk"]["es_alpha"] * 100)
    cols = ["ann_return", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown",
            "ulcer_index", f"VaR{es_a}_daily", f"ES{es_a}_daily", "ES99_daily",
            "omega_ratio", "skew", "excess_kurtosis", "hit_rate", "best_day", "worst_day",
            "avg_turnover", "annual_turnover", "avg_HHI", "avg_n_effective_assets",
            "max_single_weight", "cum_return", "n_days"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(f"{out_dir}/metrics_table.csv")
    print("\n=== Performance summary ===")
    print(df.round(4).to_string())

    # relative performance vs. 1/N (tracking error / information ratio)
    rel_rows = []
    for segment in ["stressed_test", "calm_test"]:
        ref = curves[(segment, "1/N")]
        for name in ["RL-Dirichlet-DistCritic", "CapWeighted", "CVaR-LP", "RiskParity"]:
            rel = relative_performance(curves[(segment, name)], ref)
            rel["strategy"] = name
            rel["segment"] = segment
            rel_rows.append(rel)
    rel_df = pd.DataFrame(rel_rows).set_index(["segment", "strategy"])
    rel_df.to_csv(f"{out_dir}/relative_performance_vs_1N.csv")
    print("\n=== Relative performance vs. 1/N (tracking error, ann. excess return, Information Ratio) ===")
    print(rel_df.round(4).to_string())

    # significance test: RL vs each benchmark, on the stressed segment (H1)
    print("\n=== Bootstrap significance (stressed_test, RL vs each benchmark) ===")
    sig_rows = []
    ra = curves[("stressed_test", "RL-Dirichlet-DistCritic")]
    for name in ["1/N", "CapWeighted", "BuyAndHold", "CVaR-LP", "RiskParity"]:
        rb_ = curves[("stressed_test", name)]
        diff, p = bootstrap_sharpe_diff(ra, rb_, block_size=cfg["evaluation"]["bootstrap_block_size"],
                                         n_boot=cfg["evaluation"]["bootstrap_n_boot"], seed=cfg["seed"])
        sig_rows.append({"benchmark": name, "sharpe_diff": diff, "p_value_RL_better": p})
        print(f"  RL vs {name:12s}: Sharpe diff = {diff:+.3f}   p(RL better) = {p:.3f}")
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(f"{out_dir}/significance_stressed_test.csv", index=False)

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, segment in zip(axes, ["stressed_test", "calm_test"]):
        for name in ["RL-Dirichlet-DistCritic", "1/N", "CapWeighted", "CVaR-LP", "RiskParity"]:
            r = curves[(segment, name)]
            cum = np.cumprod(1 + r)
            ax.plot(cum, label=name, linewidth=1.6 if "RL" in name else 1.0)
        ax.set_title(f"Cumulative return -- {segment}")
        ax.set_xlabel("Trading day")
        ax.set_ylabel("Growth of 1")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/equity_curves.png", dpi=150)
    plt.close(fig)

    # ---- training diagnostics: quick overview (kept for continuity) ----
    hist = trained["history"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(hist["episode"], hist["avg_reward"]); axes[0].set_title("Avg reward / episode"); axes[0].grid(alpha=0.3)
    axes[1].plot(hist["episode"], hist["avg_entropy"]); axes[1].set_title("Avg Dirichlet entropy / episode"); axes[1].grid(alpha=0.3)
    axes[2].plot(hist["episode"], hist["beta"]); axes[2].set_title("SAC temperature (beta) / episode"); axes[2].grid(alpha=0.3)
    axes[3].plot(hist["episode"], hist["critic_loss"]); axes[3].set_title("Critic quantile-Huber loss / episode")
    axes[3].set_yscale("log"); axes[3].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/training_diagnostics.png", dpi=150)
    plt.close(fig)

    # ---- dedicated ACTOR training diagnostics ----
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    warmup_eps = [e for e, w in zip(hist["episode"], hist["actor_warmed_up"]) if not w]
    warmup_end = max(warmup_eps) if warmup_eps else None

    axes[0, 0].plot(hist["episode"], hist["avg_entropy"], label="actual")
    axes[0, 0].axhline(trained.get("target_entropy", np.nan), color="red", linestyle="--", label="target")
    axes[0, 0].set_title("Policy (Dirichlet) entropy"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(hist["episode"], hist["beta"])
    axes[0, 1].set_title("SAC temperature beta (auto-tuned)"); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(hist["episode"], hist["avg_alpha_sum"])
    axes[0, 2].set_title("Avg sum(alpha) -- higher = more concentrated policy"); axes[0, 2].grid(alpha=0.3)

    axes[0, 3].plot(hist["episode"], hist["entropy_frac"])
    axes[0, 3].set_title("Entropy term's share of the actor gradient\n"
                          "||beta*dH|| / (||beta*dH|| + ||advantage*dlogp||)")
    axes[0, 3].set_ylim(-0.02, 1.02); axes[0, 3].grid(alpha=0.3)

    axes[1, 0].plot(hist["episode"], hist["avg_max_weight"])
    axes[1, 0].axhline(cfg["actor"]["w_max"], color="red", linestyle="--", label="position cap")
    axes[1, 0].set_title("Avg max single-asset weight (sampled)"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(hist["episode"], hist["actor_grad_norm"])
    axes[1, 1].set_title("Actor gradient norm / episode"); axes[1, 1].set_yscale("log"); axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(hist["episode"], hist["avg_objective"])
    axes[1, 2].set_title("Avg actor objective: E[Z] - lambda_ES*ES"); axes[1, 2].grid(alpha=0.3)

    axes[1, 3].axis("off")
    if warmup_end is not None:
        axes[1, 3].text(0.05, 0.7, f"Critic warmup:\nepisodes 0-{warmup_end}\n"
                                    f"(actor gradients computed\nfor diagnostics but NOT\napplied during this window)",
                         fontsize=10, va="top")
    else:
        axes[1, 3].text(0.05, 0.7, "No critic warmup period\n(warmup_target already met\nby end of episode 0)", fontsize=10, va="top")

    for ax in axes.flat[:7]:
        ax.set_xlabel("episode")
        if warmup_end is not None:
            ax.axvspan(hist["episode"][0], warmup_end, color="grey", alpha=0.15)
    fig.suptitle("Actor training diagnostics (grey band = critic warmup, actor not yet updating)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/actor_training_diagnostics.png", dpi=150)
    plt.close(fig)

    # ---- dedicated CRITIC training diagnostics ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(hist["episode"], hist["critic_loss"])
    axes[0].set_title("Critic quantile-Huber loss"); axes[0].set_yscale("log"); axes[0].grid(alpha=0.3)

    axes[1].plot(hist["episode"], hist["critic_grad_norm"])
    axes[1].set_title("Critic gradient norm"); axes[1].set_yscale("log"); axes[1].grid(alpha=0.3)

    axes[2].plot(hist["episode"], hist["avg_quantile_spread"])
    axes[2].set_title("Avg predicted quantile spread (max-min)\n(distributional calibration -- should stabilise, not vanish)")
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel("episode")
    fig.suptitle("Critic training diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/critic_training_diagnostics.png", dpi=150)
    plt.close(fig)

    # full training history as CSV, for the user's own further analysis/plots
    pd.DataFrame(hist).to_csv(f"{out_dir}/training_history.csv", index=False)

    # weights-over-time for the RL policy on both segments
    for segment in ["stressed_test", "calm_test"]:
        _, _, wh = evaluate_policy(actor, env, A_hat_eff, segment, rng)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.stackplot(range(len(wh)), wh.T, labels=trained.get("asset_names", ASSET_NAMES))
        ax.set_title(f"RL policy portfolio weights over time -- {segment} segment")
        ax.set_xlabel("Trading day"); ax.set_ylabel("Weight")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/rl_weights_{segment}.png", dpi=150)
        plt.close(fig)

    write_text_report(cfg, trained, df, rel_df, sig_df, out_dir)
    write_results_summary(cfg, trained, curves, out_dir)

    print(f"\nSaved metrics table, significance test, training diagnostics, and text report to: "
          f"{os.path.abspath(out_dir)}")
    return df


def write_results_summary(cfg, trained, curves, out_dir):
    """Human-readable results summary formatted to match the individual-result-block
    + comparison-table + training-history-summary template style (per-strategy blocks,
    then a compact comparison table, then a training history section at the end)."""
    import datetime
    hist = trained["history"]
    lines = []
    add = lines.append

    def individual_block(name, s):
        add("=" * 60)
        add(f"  Strategy     : {name}")
        add(f"  Annual Return: {s['ann_return']:.4f}  ({s['ann_return']*100:.2f}%)")
        add(f"  Total Return : {s['cum_return']:.4f}  ({s['cum_return']*100:.2f}%)")
        add(f"  Sharpe Ratio : {s['sharpe']:.4f}")
        add(f"  Max Drawdown : {abs(s['max_drawdown']):.4f}  ({abs(s['max_drawdown'])*100:.2f}%)")
        add(f"  Daily Ret Std: {s['daily_ret_std']*100:.4f}%")
        add(f"  Daily Ret Min: {s['worst_day']*100:.4f}%")
        add(f"  Daily Ret Max: {s['best_day']*100:.4f}%")
        add("=" * 60)

    def comparison_table(summaries):
        name_w = max(16, max(len(n) for n, _ in summaries) + 2)
        add(f"{'Strategy':<{name_w}}{'Annual Ret':>12}{'Total Ret':>12}{'Sharpe':>10}"
            f"{'MDD':>12}{'DailyStd':>12}")
        add("-" * (name_w + 58))
        for name, s in summaries:
            add(f"{name:<{name_w}}{s['ann_return']*100:>11.2f}%{s['cum_return']*100:>11.2f}%"
                f"{s['sharpe']:>10.4f}{abs(s['max_drawdown']):>12.4f}"
                f"{s['daily_ret_std']*100:>11.4f}%")
        add("-" * (name_w + 58))

    add("=" * 60)
    add("  DISTRIBUTIONAL-SAC / DIRICHLET PORTFOLIO MANAGEMENT -- FULL RESULTS SUMMARY")
    add(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("=" * 60)

    _src_label = "real market data" if trained.get("data_source") == "real" else "synthetic"
    segment_labels = {
        "stressed_test": f"TEST PERIOD 1 ({_src_label}, {cfg['market']['n_stressed_test_days']} trading days)",
        "calm_test": f"TEST PERIOD 2 ({_src_label}, {cfg['market']['n_calm_test_days']} trading days)",
    }
    strategy_order = ["RL-Dirichlet-DistCritic", "1/N", "CapWeighted", "CVaR-LP", "RiskParity"]

    for segment in ["stressed_test", "calm_test"]:
        add("")
        add(f"[ {segment_labels[segment]} -- INDIVIDUAL RESULTS ]")
        add("")
        summaries = []
        for name in strategy_order:
            r = curves[(segment, name)]
            s = performance_summary(r, es_alpha=cfg["risk"]["es_alpha"])
            summaries.append((name, s))
            individual_block(name, s)
        add("")
        add(f"[ {segment_labels[segment]} -- COMPARISON TABLE ]")
        add("")
        comparison_table(summaries)

    add("")
    add("[ TRAINING HISTORY SUMMARY ]")
    add("")
    n_ep = len(hist["episode"])
    last10 = slice(-10, None)

    add("  Actor (Dirichlet policy)")
    n_warmup = sum(1 for w in hist["actor_warmed_up"] if not w)
    add(f"    Episodes trained    : {n_ep}   ({n_warmup} withheld for critic warmup)")
    add(f"    Reward  - mean      : {np.mean(hist['avg_reward']):.6f}")
    add(f"    Reward  - std       : {np.std(hist['avg_reward']):.6f}")
    add(f"    Reward  - min/max   : {np.min(hist['avg_reward']):.6f} / {np.max(hist['avg_reward']):.6f}")
    ep_ret = np.array(hist["episode_cum_return"]) * 100
    add(f"    Ep Ret (%)          : mean={ep_ret.mean():.2f}%  std={ep_ret.std():.2f}%")
    add(f"    Ep Ret min/max      : {ep_ret.min():.2f}% / {ep_ret.max():.2f}%")
    add(f"    Final beta          : {hist['beta'][-1]:.4f}   "
        f"(SAC temperature -- this project's analogue of an epsilon-greedy rate: it is")
    add(f"                          auto-tuned toward a target entropy rather than manually decayed)")
    add(f"    GradNorm (pre-clip) : mean={np.mean(hist['actor_grad_norm']):.4f}  "
        f"final10={np.mean(hist['actor_grad_norm'][last10]):.4f}")
    add(f"    Objective - mean    : {np.mean(hist['avg_objective']):.6f}   "
        f"(E[Z]-lambda_ES*ES + entropy bonus; the actor has no classical 'loss',")
    add(f"    Objective - final10 : {np.mean(hist['avg_objective'][last10]):.6f}   "
        f"since it is trained to maximise this objective, not minimise a loss)")
    add(f"    Entropy frac - mean : {np.mean(hist['entropy_frac']):.3f}  "
        f"final10={np.mean(hist['entropy_frac'][last10]):.3f}   "
        f"(share of the actor gradient coming from the entropy bonus")
    add(f"                          rather than the return/ES advantage; 1.0 = entropy-dominated, "
        f"0.0 = return/ES-dominated)")
    add("")
    add("  Critic (distributional, quantile regression)")
    add(f"    Episodes trained    : {n_ep}")
    add(f"    GradNorm (pre-clip) : mean={np.nanmean(hist['critic_grad_norm']):.4f}  "
        f"final10={np.nanmean(hist['critic_grad_norm'][last10]):.4f}")
    add(f"    Loss - mean         : {np.nanmean(hist['critic_loss']):.6f}")
    add(f"    Loss - final10      : {np.nanmean(hist['critic_loss'][last10]):.6f}")
    add(f"    Quantile spread - mean    : {np.nanmean(hist['avg_quantile_spread']):.6f}   "
        f"(calibration diagnostic -- should stabilise, not vanish)")
    add(f"    Quantile spread - final10: {np.nanmean(hist['avg_quantile_spread'][last10]):.6f}")

    add("")
    add("=" * 60)
    add("  END OF REPORT")
    add("=" * 60)

    with open(f"{out_dir}/results_summary.txt", "w") as f:
        f.write("\n".join(lines))


def write_text_report(cfg, trained, df, rel_df, sig_df, out_dir):
    """Human-readable plain-text summary of the whole run: configuration,
    full performance table, relative performance, significance test, and a
    training-dynamics summary for the actor and the critic."""
    hist = trained["history"]
    lines = []
    add = lines.append

    add("=" * 78)
    add("EUROPEAN ENERGY PORTFOLIO RL -- SIMULATION RESULTS REPORT")
    add("=" * 78)
    add("")
    add("1. RUN CONFIGURATION")
    add("-" * 78)
    add(f"  seed                        : {cfg['seed']}")
    add(f"  data source                 : {trained.get('data_source', cfg['market'].get('data_source', 'unknown'))}")
    add(f"  asset universe ({trained.get('n_assets', '?')})           : "
        f"{', '.join(trained.get('asset_names', []))}")
    add(f"  train / stressed / calm days: {cfg['market']['n_train_days']} / "
        f"{cfg['market']['n_stressed_test_days']} / {cfg['market']['n_calm_test_days']}")
    add(f"  episodes                    : {cfg['training']['n_episodes']}")
    add(f"  lookback window (L)         : {cfg['state']['lookback']}")
    add(f"  use_gnn                     : {cfg['state']['use_gnn']}")
    add(f"  es_mode                     : {cfg['risk']['es_mode']}")
    add(f"  lambda_ES / ES confidence   : {cfg['risk']['lambda_es']} / {cfg['risk']['es_alpha']}")
    add(f"  lambda_TC (transaction cost): {cfg['costs']['lambda_tc']}")
    add(f"  position cap (w_max)        : {cfg['actor']['w_max']}")
    add(f"  alpha_floor                 : {cfg['actor']['alpha_floor']}")
    add(f"  n_quantiles (critic)        : {cfg['critic']['n_quantiles']}")
    add(f"  critic_warmup_updates       : {cfg['training']['critic_warmup_updates']} "
        f"(critic TD updates required before the actor starts learning)")
    add(f"  entropy_target_margin       : {cfg['actor']['entropy_target_margin']} "
        f"(target = H_max - margin, H_max = H(Dir(1,...,1)))")
    add(f"  beta bounds (min/max)       : {cfg['actor']['beta_min']} / {cfg['actor']['beta_max']}")
    add("")

    add("2. PERFORMANCE SUMMARY (both test segments, all strategies)")
    add("-" * 78)
    add(df.round(4).to_string())
    add("")

    add("3. RELATIVE PERFORMANCE VS. 1/N (tracking error, ann. excess return, Information Ratio)")
    add("-" * 78)
    add(rel_df.round(4).to_string())
    add("")

    add("4. BOOTSTRAP SIGNIFICANCE TEST (stressed_test segment, RL vs. each benchmark)")
    add("-" * 78)
    add("   H0: RL's Sharpe ratio does not exceed the benchmark's. p_value_RL_better is the")
    add("   fraction of stationary block-bootstrap resamples in which the RL-vs-benchmark")
    add("   Sharpe gap is at least as large as observed once centred on zero -- a low value")
    add("   is evidence the observed gap is unlikely to be pure sampling noise.")
    add(sig_df.round(4).to_string(index=False))
    add("")

    add("5. ACTOR TRAINING SUMMARY")
    add("-" * 78)
    n_warmup = sum(1 for w in hist["actor_warmed_up"] if not w)
    if n_warmup > 0:
        add(f"   critic warmup                 : actor updates were withheld for the first "
            f"{n_warmup} episode(s)")
        add(f"                                   ({cfg['training']['critic_warmup_updates']} "
            f"critic TD updates required before the actor starts learning)")
    else:
        add(f"   critic warmup                 : none needed (warmup target already met by "
            f"episode 0)")
    add(f"   entropy target                : {trained.get('target_entropy', float('nan')):.3f}")
    add(f"   entropy, first episode -> last   : {hist['avg_entropy'][0]:.3f} -> {hist['avg_entropy'][-1]:.3f}")
    add(f"   SAC temperature beta, first -> last : {hist['beta'][0]:.4f} -> {hist['beta'][-1]:.4f}")
    add(f"   avg sum(alpha), first -> last  : {hist['avg_alpha_sum'][0]:.3f} -> {hist['avg_alpha_sum'][-1]:.3f}")
    add(f"   avg max single weight, first -> last : {hist['avg_max_weight'][0]:.3f} -> {hist['avg_max_weight'][-1]:.3f}")
    add(f"   actor gradient norm, first -> last   : {hist['actor_grad_norm'][0]:.5f} -> {hist['actor_grad_norm'][-1]:.5f}")
    add(f"   avg actor objective (E[Z]-lambda_ES*ES), first -> last : "
        f"{hist['avg_objective'][0]:.5f} -> {hist['avg_objective'][-1]:.5f}")
    add(f"   entropy term's share of actor gradient, first -> last : "
        f"{hist['entropy_frac'][0]:.3f} -> {hist['entropy_frac'][-1]:.3f}")
    add("   (entropy_frac = ||beta*dH|| / (||beta*dH|| + ||advantage*dlogp||) per step, averaged")
    add("   over the episode -- close to 1.0 means the entropy bonus is dominating the update and")
    add("   the return/ES signal is being drowned out; close to 0.0 means entropy has negligible")
    add("   influence and the policy is being driven almost entirely by the return/ES objective.)")
    final_entropy = hist['avg_entropy'][-1]
    target_ent = trained.get('target_entropy', float('nan'))
    final_beta = hist['beta'][-1]
    beta_max_cfg = cfg['actor']['beta_max']
    beta_min_cfg = cfg['actor']['beta_min']
    if not np.isnan(target_ent) and abs(final_entropy - target_ent) > 0.05:
        add("   NOTE: entropy had not yet reached its target by the final episode.")
        if final_entropy < target_ent and final_beta >= beta_max_cfg - 1e-9:
            add("   Entropy is BELOW target and beta is pinned at beta_max: the entropy bonus")
            add("   is already at its strongest allowed setting and still isn't enough to push")
            add("   entropy up to target -- consider raising beta_max, or more episodes.")
        elif final_entropy > target_ent and final_beta <= beta_min_cfg + 1e-9:
            add("   Entropy is ABOVE target and beta is pinned at beta_min: the entropy-control")
            add("   loop has done everything it can (weakened the entropy bonus as far as")
            add("   allowed) and entropy is *still* rising -- this means the return/ES advantage")
            add("   term itself is what's pushing the policy toward higher entropy (more diffuse")
            add("   weights), not the entropy bonus. Check entropy_frac (section above): if it's")
            add("   small (as here) while entropy keeps drifting away from target, lowering")
            add("   beta_min further will not help -- the policy is being driven there by the")
            add("   return/ES signal itself.")
        else:
            add("   Beta has not saturated at either bound, so more training episodes (or a")
            add("   larger beta_lr) may still close the gap.")
    add("")

    add("6. CRITIC TRAINING SUMMARY")
    add("-" * 78)
    add(f"   quantile-Huber loss, first -> last   : {hist['critic_loss'][0]:.5f} -> {hist['critic_loss'][-1]:.5f}")
    add(f"   critic gradient norm, first -> last  : {hist['critic_grad_norm'][0]:.5f} -> {hist['critic_grad_norm'][-1]:.5f}")
    add(f"   avg predicted quantile spread, first -> last : "
        f"{hist['avg_quantile_spread'][0]:.5f} -> {hist['avg_quantile_spread'][-1]:.5f}")
    add("   (the quantile spread is a calibration diagnostic: it should stabilise at a")
    add("   value reflecting genuine day-to-day return uncertainty, not collapse to ~0,")
    add("   which would indicate the critic has stopped representing tail risk at all.)")
    add("")

    add("7. FILES IN THIS REPORT")
    add("-" * 78)
    add("   metrics_table.csv                 -- full performance metrics, both segments")
    add("   relative_performance_vs_1N.csv     -- tracking error / Information Ratio vs 1/N")
    add("   significance_stressed_test.csv     -- bootstrap Sharpe-difference test")
    add("   training_history.csv               -- every per-episode diagnostic tracked during training")
    add("   equity_curves.png                  -- cumulative return, RL vs benchmarks, both segments")
    add("   training_diagnostics.png           -- quick overview: reward/entropy/beta/critic loss")
    add("   actor_training_diagnostics.png     -- detailed actor diagnostics (6 panels)")
    add("   critic_training_diagnostics.png    -- detailed critic diagnostics (3 panels)")
    add("   rl_weights_stressed_test.png       -- RL policy weights over time, stressed segment")
    add("   rl_weights_calm_test.png           -- RL policy weights over time, calm segment")
    add("")
    add("=" * 78)
    add("Reminder: hyperparameters here are illustrative, not tuned (see README.md,")
    add("'Honest simplifications'). Treat magnitudes as a demonstration that every")
    add("mechanism runs correctly end-to-end, not as a calibrated empirical result.")
    add("=" * 78)

    with open(f"{out_dir}/results_report.txt", "w") as f:
        f.write("\n".join(lines))
    return df


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "energy_rl_sim_results"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    trained = train(cfg)
    make_report(cfg, trained, out_dir=out_dir)
