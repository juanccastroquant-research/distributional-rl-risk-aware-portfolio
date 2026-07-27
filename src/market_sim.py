"""
market_sim.py
-------------
Synthetic, physically-motivated European energy market (S2, S6 of the
proposal). Real vendor data (LSEG/Bloomberg, ICE Endex TTF/EUA, ENTSO-E
day-ahead prices) is not reachable from this environment, so this module
generates a stylised but structurally faithful substitute: a small universe
of assets with explicit fuel/technology exposures, driven by shared
macro/physical factors (gas, carbon, wind capacity factor, a spark-spread-like
electricity proxy), with an explicit "stressed" regime that reproduces the
qualitative shape of the 2021-2022 European energy crisis: a rising gas price
initially rewards gas-heavy names (via the base return term during the
transition-to-stress ramp, see `simulate_market`), while supply-shock jump
events -- more frequent once the stressed regime is reached -- carry a
separate, asymmetric penalty concentrated on gas-heavy names and cushioned
for hydro/grid names (the `crash` term). This gives the ES-aware,
physically-informed pipeline something genuine to learn and be evaluated
against.

Everything here is SYNTHETIC. Swapping in real TTF/EUA/ENTSO-E/equity data
would only require replacing `simulate_market()`'s output with the real
panel in the same shape; no other module needs to change.
"""
import numpy as np
import pandas as pd

ASSET_NAMES = [
    "GasUtility",        # RWE-like
    "PureRenewable",     # Orsted-like
    "IntegratedUtility",  # Iberdrola-like
    "GridTSO",           # Terna-like
    "OilGasMajor",       # TotalEnergies-like
    "GasInfrastructure",  # Snam-like
    "HydroUtility",      # Verbund-like
    "RenewableFuels",    # Neste-like
]

# Exposure profile per asset: [wind, gas, carbon, hydro, grid/regulated]
# Used both to generate returns (S6, "exploit the physical drivers") and to
# build the physically-informed asset graph (S4.2).
EXPOSURES = np.array([
    [0.15, 0.70, 0.55, 0.00, 0.10],  # GasUtility
    [0.90, 0.00, 0.10, 0.00, 0.05],  # PureRenewable
    [0.50, 0.20, 0.30, 0.10, 0.20],  # IntegratedUtility
    [0.05, 0.05, 0.05, 0.05, 0.90],  # GridTSO
    [0.05, 0.60, 0.55, 0.00, 0.05],  # OilGasMajor
    [0.05, 0.80, 0.20, 0.00, 0.30],  # GasInfrastructure
    [0.10, 0.00, 0.00, 0.90, 0.10],  # HydroUtility
    [0.15, 0.30, 0.40, 0.00, 0.05],  # RenewableFuels
])

N_ASSETS = len(ASSET_NAMES)


def build_asset_graph(threshold=0.35):
    """Physically-informed adjacency: connect assets whose exposure profiles
    are similar (shared fuel/technology exposure), add self-loops, row-
    normalise (S4.2 / S6.4). Returns A_hat, shape (n, n)."""
    E = EXPOSURES / (np.linalg.norm(EXPOSURES, axis=1, keepdims=True) + 1e-8)
    sim = E @ E.T  # cosine similarity
    A = (sim >= threshold).astype(np.float64)
    np.fill_diagonal(A, 1.0)  # self-loops
    A_hat = A / A.sum(axis=1, keepdims=True)
    return A_hat


def _ou_process(rng, n_steps, mu, theta, sigma, x0):
    x = np.zeros(n_steps)
    x[0] = x0
    for t in range(1, n_steps):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.normal()
    return x


def simulate_market(cfg, rng=None):
    """
    Generates one contiguous synthetic panel covering, in order:
    train (calm) -> [transition_to_stress buffer] -> stressed_test ->
    [transition_to_calm buffer] -> calm_test.

    The two buffer blocks exist to fix a face-validity problem: the gas-price
    factor mean-reverts slowly (rate `gas_theta`), so switching regime
    parameters exactly at a segment boundary meant the "stressed_test" window
    was mostly spent *ramping up* to the crisis level (a sustained, one-directional
    rise that -- since returns depend on the day-to-day CHANGE in gas price --
    produced systematically positive returns for every strategy), while
    "calm_test" was mostly spent *ramping back down* (a sustained fall,
    producing systematic losses). Neither segment was actually testing what its
    name claims. The buffers let gas equilibrate to each regime's level BEFORE
    the named/evaluated segment begins, so "stressed_test" now captures the
    market actually *being* in a high-risk regime (still with real jump-driven
    drawdowns) rather than transitioning into one, and "calm_test" starts from
    an already-settled level rather than mid-unwind.

    Returns a dict with:
      returns   : DataFrame (T, n_assets) daily simple returns
      features  : ndarray (T, n_assets, F) per-asset raw feature panel
                  (own return, own rolling vol, gas, carbon, elec, wind)
      regime    : array (T,) of {"train","transition_to_stress","stressed_test",
                  "transition_to_calm","calm_test"}
      macro     : DataFrame of the raw macro factor levels (for inspection)
    """
    rng = rng or np.random.default_rng(cfg["seed"])
    n_train = cfg["n_train_days"]
    n_stress = cfg["n_stressed_test_days"]
    n_calm_test = cfg["n_calm_test_days"]
    n_buffer = cfg.get("transition_buffer_days", 80)
    T = n_train + n_buffer + n_stress + n_buffer + n_calm_test
    regime = np.array(
        ["train"] * n_train
        + ["transition_to_stress"] * n_buffer
        + ["stressed_test"] * n_stress
        + ["transition_to_calm"] * n_buffer
        + ["calm_test"] * n_calm_test
    )
    # High-risk regime PARAMETERS (elevated gas level/vol, higher jump and
    # drought probability) apply from the start of the ramp-up buffer through
    # the end of stressed_test; calm parameters apply everywhere else. This is
    # deliberately a *superset* of the "stressed_test" label itself.
    is_stressed = np.isin(regime, ["transition_to_stress", "stressed_test"])

    # --- macro factors -------------------------------------------------
    gas_vol = np.where(is_stressed, cfg["gas_vol_stressed"], cfg["gas_vol_calm"])
    gas_mu = np.where(is_stressed, cfg["gas_level_stressed"], cfg["gas_level_calm"])
    gas = np.zeros(T)
    gas[0] = cfg["gas_level_calm"]
    for t in range(1, T):
        gas[t] = gas[t - 1] + cfg["gas_theta"] * (gas_mu[t] - gas[t - 1]) + gas_vol[t] * rng.normal()

    # Poisson supply-shock jumps, far more likely in the stressed regime
    jump_prob = np.where(is_stressed, cfg["jump_prob_stressed"], cfg["jump_prob_calm"])
    jumps = (rng.uniform(size=T) < jump_prob).astype(np.float64)
    jump_size = jumps * rng.uniform(cfg["jump_size_min"], cfg["jump_size_max"], size=T)
    gas = gas + np.cumsum(jump_size) * 0.0  # jumps affect returns directly, not the level path
    # (jumps are applied as one-off shocks to *returns* below, not the level,
    #  to avoid an ever-growing gas level series)

    carbon = _ou_process(rng, T, mu=cfg["carbon_level"], theta=cfg["carbon_theta"],
                          sigma=cfg["carbon_vol"], x0=cfg["carbon_level"])
    carbon = carbon + 0.3 * (gas - cfg["gas_level_calm"])  # gas/carbon co-movement

    day_of_year = np.arange(T) % 365
    seasonal = 0.5 + 0.35 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)  # winter-high wind proxy
    wind_cf = np.clip(seasonal + cfg["wind_noise"] * rng.normal(size=T), 0.05, 0.95)
    # occasional multi-week "wind drought" episodes, more likely when stressed
    drought_prob = np.where(is_stressed, cfg["drought_prob_stressed"], cfg["drought_prob_calm"])
    t = 0
    while t < T:
        if rng.uniform() < drought_prob[t]:
            dur = rng.integers(10, 25)
            wind_cf[t:t + dur] *= 0.5
            t += dur
        else:
            t += 1
    wind_cf = np.clip(wind_cf, 0.05, 0.95)

    elec = 1.0 * gas + 0.3 * carbon - 0.6 * wind_cf + cfg["elec_noise"] * rng.normal(size=T)

    d_gas = np.diff(gas, prepend=gas[0])
    d_carbon = np.diff(carbon, prepend=carbon[0])
    d_wind = np.diff(wind_cf, prepend=wind_cf[0])

    # --- asset returns from exposures + idiosyncratic noise + crash term ---
    beta_wind, beta_gas, beta_carbon, beta_hydro, beta_grid = EXPOSURES.T
    idio_vol = cfg["idio_vol"] * (1.0 + 0.4 * beta_gas)  # gas-heavy names slightly noisier

    returns = np.zeros((T, N_ASSETS))
    for i in range(N_ASSETS):
        base = (cfg["beta_gas_scale"] * beta_gas[i] * d_gas
                + cfg["beta_carbon_scale"] * beta_carbon[i] * (-d_carbon)
                + cfg["beta_wind_scale"] * beta_wind[i] * d_wind)
        # (grid/TSO names are low-beta by construction via their small
        #  gas/carbon/wind exposure entries in EXPOSURES, so no separate
        #  grid term is needed here.)
        idio = idio_vol[i] * rng.normal(size=T)
        # crash term: a supply-shock jump carries an asymmetric penalty
        # concentrated on gas/carbon-heavy names (regulatory caps, demand
        # destruction, margin calls) and is *cushioned* (small positive
        # flight-to-quality effect) for hydro/grid/renewables -- this is the
        # "overweight gas -> gets punished" tail risk the ES-aware policy
        # is specifically designed to guard against.
        # BUG FIX: this term previously had the opposite sign (a jump was a
        # BOOST for gas-heavy names and a hit for hydro/grid), which combined
        # with the higher jump probability in the stressed regime to produce
        # systematically positive "stressed_test" returns for most of the
        # portfolio -- exactly backwards from what a crash should do.
        crash = jumps * jump_size * (-cfg["jump_beta_gas"] * beta_gas[i]
                                      + cfg["jump_beta_safe"] * (beta_hydro[i] + beta_grid[i]))
        returns[:, i] = base + idio + crash

    features = np.zeros((T, N_ASSETS, 6))
    features[:, :, 0] = returns
    ret_df = pd.DataFrame(returns, columns=ASSET_NAMES)
    features[:, :, 1] = ret_df.rolling(20, min_periods=5).std().bfill().values
    features[:, :, 2] = np.repeat(gas[:, None], N_ASSETS, axis=1)
    features[:, :, 3] = np.repeat(carbon[:, None], N_ASSETS, axis=1)
    features[:, :, 4] = np.repeat(elec[:, None], N_ASSETS, axis=1)
    features[:, :, 5] = np.repeat(wind_cf[:, None], N_ASSETS, axis=1)

    macro = pd.DataFrame({"gas": gas, "carbon": carbon, "elec": elec, "wind_cf": wind_cf,
                           "jump": jumps})
    return {
        "returns": ret_df,
        "features": features,
        "regime": regime,
        "macro": macro,
    }


def causal_rolling_normalise(features, window=252, min_periods=20, unit_interval_channels=(5,)):
    """Rolling (past-only) z-score normalisation per feature channel, as
    recommended in S4.6 to avoid look-ahead bias.

    `unit_interval_channels` lists channel indices that are already bounded
    in [0,1] (e.g. the synthetic wind-capacity-factor channel, index 5, by
    default) and so are only mean-centred, not rescaled by their rolling std.
    Every other channel is fully z-scored (mean-centred AND divided by its
    rolling std) so all channels end up on a comparable scale -- pass
    unit_interval_channels=() if none of your feature channels are already
    naturally bounded (e.g. the real-data OHLCV-derived feature set in
    data_loader.py, none of whose 6 channels are unit-interval)."""
    T, n, F = features.shape
    out = np.zeros_like(features)
    for f in range(F):
        df = pd.DataFrame(features[:, :, f])
        # STRICTLY CAUSAL fill for the leading rows before the rolling window has
        # `min_periods` observations: previously `.bfill()` pulled in a value
        # computed from a LATER row (using data past the current one) into these
        # early rows -- a genuine look-ahead leak, even though it's confined to
        # the first `min_periods` rows of the series. An expanding (min_periods=1
        # for the mean, 2 for the std) statistic only ever uses data up to and
        # including the current row, so it can never leak future information --
        # swap it in for exactly the rows where the rolling stat is still NaN.
        roll_mean = df.rolling(window, min_periods=min_periods).mean()
        roll_mean = roll_mean.fillna(df.expanding(min_periods=1).mean())
        if f in unit_interval_channels:
            out[:, :, f] = (df - roll_mean).values
        else:
            roll_std = df.rolling(window, min_periods=min_periods).std()
            roll_std = roll_std.fillna(df.expanding(min_periods=2).std()).fillna(1.0)
            roll_std = roll_std.clip(lower=1e-4)
            out[:, :, f] = ((df - roll_mean) / roll_std).values
    return out


def project_capped_simplex(w, w_max):
    """Euclidean projection of w onto {x : sum(x)=1, 0<=x<=w_max}.
    Standard water-filling / bisection algorithm (S3.6 correction #3)."""
    n = len(w)
    if w_max * n < 1.0 - 1e-9:
        raise ValueError("Infeasible: w_max * n_assets must be >= 1.")
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        x = np.clip(w - mid, 0.0, w_max)
        if x.sum() > 1.0:
            lo = mid
        else:
            hi = mid
    x = np.clip(w - 0.5 * (lo + hi), 0.0, w_max)
    x = x / x.sum()  # numerical safety renormalisation
    return x


class PortfolioEnv:
    """
    Minimal sequential portfolio-allocation environment.

    On each step the agent (or a benchmark rule) supplies portfolio weights
    decided using information up to and including day t; those weights earn
    the day t+1 return (no look-ahead), net of an L1 turnover cost and
    subject to a hard position cap enforced by Euclidean projection.
    """

    def __init__(self, features_norm, raw_returns, regime, lookback, lambda_tc, w_max):
        self.X = features_norm       # (T, n, F), causally normalised
        self.R = raw_returns.values  # (T, n) raw simple returns
        self.regime = regime
        self.L = lookback
        self.lambda_tc = lambda_tc
        self.w_max = w_max
        self.n_assets = raw_returns.shape[1]
        self.T = raw_returns.shape[0]

    def segment_bounds(self, name):
        idx = np.where(self.regime == name)[0]
        return idx[0], idx[-1]

    def reset(self, start_t):
        self.t = start_t
        self.prev_w = np.ones(self.n_assets) / self.n_assets
        return self._obs()

    def _obs(self):
        lo = max(0, self.t - self.L + 1)
        window = self.X[lo:self.t + 1]  # (<=L, n, F)
        if window.shape[0] < self.L:
            pad = np.repeat(window[:1], self.L - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
        X_t = np.transpose(window, (1, 0, 2))  # (n, L, F)
        return X_t, self.prev_w.copy()

    def step(self, action_w):
        action_w = project_capped_simplex(action_w, self.w_max)
        turnover = np.sum(np.abs(action_w - self.prev_w))
        next_t = self.t + 1
        port_return = float(action_w @ self.R[next_t]) if next_t < self.T else 0.0
        reward = port_return - self.lambda_tc * turnover
        self.prev_w = action_w
        self.t = next_t
        done = (self.t >= self.T - 1) or (self.regime[self.t] != self.regime[self.t - 1]
                                          if self.t > 0 else False)
        obs = self._obs()
        info = {"port_return": port_return, "turnover": turnover, "weights": action_w.copy()}
        return obs, reward, done, info
