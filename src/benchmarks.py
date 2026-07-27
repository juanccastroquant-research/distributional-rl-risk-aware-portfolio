"""
benchmarks.py
-------------
The benchmark suite recommended in S7 of the proposal:
  Tier 1 (naive/passive): 1/N equal-weight, buy-and-hold, static cap-weighted
  Tier 2 (classical convex optimisation): rolling CVaR-optimal LP
    (Rockafellar & Uryasev, 2000), equal-risk-contribution / risk parity

All benchmarks respect the same hard constraints as the RL agent: long-only,
fully invested, and the position cap w_max, so comparisons are on equal
footing.
"""
import numpy as np
from scipy.optimize import linprog, minimize

from market_sim import project_capped_simplex


def equal_weight(n_assets):
    return np.ones(n_assets) / n_assets


def static_cap_weight(weights):
    """A fixed, size-proxy 'index' weight vector (does not change with the
    market), representing the low-cost passive alternative to actively
    managing the same universe."""
    w = np.asarray(weights, dtype=np.float64)
    return w / w.sum()


def cvar_optimal_weights(scenario_returns, alpha_level, w_max, lambda_ret=1.0):
    """
    Rockafellar-Uryasev (2000) linear program:

        min_{w, zeta, u}  -lambda_ret * mu^T w + zeta + 1/(S(1-alpha)) * sum_s u_s
        s.t.   u_s >= -(r_s . w) - zeta ,  u_s >= 0
               sum(w) = 1 ,  0 <= w_i <= w_max

    scenario_returns: (S, n) trailing historical scenario matrix.
    Returns the optimal weight vector w (n,).
    """
    S, n = scenario_returns.shape
    mu = scenario_returns.mean(axis=0)

    n_vars = n + 1 + S  # w (n), zeta (1), u (S)
    c = np.zeros(n_vars)
    c[:n] = -lambda_ret * mu
    c[n] = 1.0
    c[n + 1:] = 1.0 / (S * (1 - alpha_level))

    # inequality constraints: -r_s.w - zeta - u_s <= 0  for each scenario s
    A_ub = np.zeros((S, n_vars))
    A_ub[:, :n] = -scenario_returns
    A_ub[:, n] = -1.0
    A_ub[np.arange(S), n + 1 + np.arange(S)] = -1.0
    b_ub = np.zeros(S)

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, w_max)] * n + [(None, None)] + [(0.0, None)] * S

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                   method="highs")
    if not res.success:
        # fall back to equal weight if the solver fails on a degenerate window
        return equal_weight(n)
    w = res.x[:n]
    return project_capped_simplex(w, w_max)  # numerical safety


def risk_parity_weights(scenario_returns, w_max):
    """
    Equal-risk-contribution (ERC) portfolio via the convex reformulation of
    Maillard, Roncalli & Teiletche (2010):
        min_w  0.5 w^T Sigma w - (1/n) sum_i log(w_i) ,  w > 0
    then renormalised to sum to 1 and clipped to the position cap.
    """
    Sigma = np.cov(scenario_returns.T)
    n = Sigma.shape[0]
    Sigma = Sigma + 1e-6 * np.eye(n)  # numerical stability

    def objective(x):
        # x is unconstrained; map to positive w via softplus-like transform
        w = np.log1p(np.exp(x))  # softplus, always > 0
        val = 0.5 * w @ Sigma @ w - np.mean(np.log(w))
        return val

    x0 = np.zeros(n)
    res = minimize(objective, x0, method="L-BFGS-B")
    w = np.log1p(np.exp(res.x))
    w = w / w.sum()
    return project_capped_simplex(w, w_max)


class RebalancingBenchmark:
    """
    Wraps a weight-decision rule (a function scenario_window -> weights) so
    it can be simulated on the same PortfolioEnv-shaped return data, with a
    fixed rebalancing frequency and trailing scenario window, mirroring how
    the RL agent's own weights are evaluated.
    """

    def __init__(self, rule, rebalance_every, scenario_window):
        self.rule = rule
        self.rebalance_every = rebalance_every
        self.scenario_window = scenario_window

    def run(self, raw_returns, regime_mask, w_max):
        """raw_returns: (T, n) array restricted conceptually to the whole
        series; regime_mask: boolean array selecting the evaluation segment
        (e.g. regime == 'stressed_test'). Returns (port_returns, turnovers,
        weights_history) for the evaluation segment only, using only
        information available up to the current day for each rebalance."""
        T, n = raw_returns.shape
        idx = np.where(regime_mask)[0]
        start, end = idx[0], idx[-1]

        w_prev = equal_weight(n)
        port_returns, turnovers, weight_hist = [], [], []
        w_current = w_prev.copy()

        for t in range(start, end):
            if (t - start) % self.rebalance_every == 0:
                lo = max(0, t - self.scenario_window)
                window = raw_returns[lo:t] if t > lo else raw_returns[max(0, t - 5):t + 1]
                if window.shape[0] < 10:
                    w_target = equal_weight(n)
                else:
                    w_target = self.rule(window, w_max)
                turnover = np.sum(np.abs(w_target - w_current))
                w_current = w_target
            else:
                turnover = 0.0
            r = float(w_current @ raw_returns[t + 1])
            port_returns.append(r)
            turnovers.append(turnover)
            weight_hist.append(w_current.copy())

        return np.array(port_returns), np.array(turnovers), np.array(weight_hist)
