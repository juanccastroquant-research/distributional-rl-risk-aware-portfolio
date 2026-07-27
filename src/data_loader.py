"""
data_loader.py
--------------
Real-market-data counterpart to market_sim.py's simulate_market(). Downloads
daily OHLCV bars from Yahoo Finance (via the `yfinance` package) for a fixed
European energy universe, derives a 6-channel per-asset feature panel from
Open/High/Low/Close/Volume, and labels each trading day into the same three
regime buckets the rest of the pipeline already reads: "train", "stressed_test",
"calm_test" (train.py, run_seed_sweep.py and compare_ablation.py all hard-code
exactly these three strings -- see the grep-confirmed usage list in the code
review that produced this file).

WHY NO "transition_to_stress"/"transition_to_calm" BUFFERS HERE:
market_sim.py needed those buffers because its synthetic gas-price factor is
a slow mean-reverting (OU) process, so switching regime PARAMETERS exactly at
a labelled segment boundary meant the labelled window was mostly spent
ramping into/out of the new regime rather than actually being in it. That is
purely an artifact of how the synthetic generator works. Real historical
prices have no such artificial ramp: the actual 2023-01-01 news, prices and
volatility are exactly what happened at that date, so no buffer is needed --
each day is assigned to exactly one of the three windows below, back to back.

WHY THE ASSET GRAPH IS BUILT DIFFERENTLY HERE:
market_sim.build_asset_graph() thresholds cosine similarity between a
hand-specified fundamental exposure table (EXPOSURES: wind/gas/carbon/hydro/
grid loadings) that only exists for the 8 synthetic archetypes. There is no
equivalent fundamental exposure table shipped in this codebase for real
tickers, so build_empirical_asset_graph() below builds the physically-
informed adjacency empirically instead: pairwise return correlation over the
TRAIN segment only (no lookahead into either test window), thresholded,
self-loops added, row-normalised. Same output contract (n x n, row-stochastic)
as market_sim.build_asset_graph(), so GraphConvLayer / the use_gnn toggle in
train.py need no changes at all.

NETWORK NOTE: this module requires outbound internet access to Yahoo Finance
(pip install yfinance). It will NOT run inside a network-sandboxed
environment that only allows package-registry domains -- run it on a machine
with normal internet access.
"""
import os
import time

import numpy as np
import pandas as pd

# yfinance is imported lazily inside download_ohlcv(), NOT here at module load
# time -- this lets `import data_loader` succeed (and train.py's synthetic
# fallback path keep working) even in environments where yfinance/internet
# access isn't set up; the clear ImportError below only fires when the
# real-data path is actually invoked.


# ----------------------------------------------------------------------
# European energy universe (real tickers, as specified for this study)
# ----------------------------------------------------------------------
TICKERS = [
    # -- Integrated Oil & Gas --
    "SHEL",       # Shell plc          - UK,      Europe's largest energy co.; LNG & renewables transition
    "BP",         # BP plc             - UK,      Oil, gas, hydrogen & offshore wind exposure
    "TTE",        # TotalEnergies SE   - France,  Major LNG producer and renewable investor
    # -- Electric & Renewable Utilities --
    "ENEL.MI",    # Enel S.p.A.        - Italy,   One of the world's largest renewable electricity producers
    "IBE.MC",     # Iberdrola S.A.     - Spain,   Global leader in wind power and smart grids
    "RWE.DE",     # RWE AG             - Germany, Coal-to-offshore-wind & storage transition
    "EOAN.DE",    # E.ON SE            - Germany, Large European electricity network operator
    # -- Oil, Gas & Offshore Wind --
    "EQNR",       # Equinor ASA        - Norway,  Europe's strategic gas supplier post-2022
                  #                      (NYSE-listed ADR; no exchange suffix given, so yfinance
                  #                       resolves this to the US ADR line, not the Oslo listing EQNR.OL)
    # -- Pure-play Offshore Wind --
    "ORSTED.CO",  # Orsted A/S         - Denmark, Pure-play offshore wind benchmark
    # -- Integrated Energy (Refining, Renewables & Hydrogen) --
    "REP.MC",     # Repsol S.A.        - Spain,   Refining, renewables and hydrogen exposure
]

N_ASSETS = len(TICKERS)
ASSET_NAMES = list(TICKERS)  # ticker symbols double as the display names

# ----------------------------------------------------------------------
# Study periods (real calendar dates -- edit here to change the windows)
# ----------------------------------------------------------------------
TRAIN_START = "2011-01-01"
TRAIN_END   = "2022-12-31"   # NOTE: this window includes the real 2021-2022
                             # European energy crisis, so the agent trains on
                             # genuine crisis dynamics rather than a held-out
                             # synthetic replica of one.
TEST1_START = "2023-01-01"  # -> regime label "stressed_test" (see note below)
TEST1_END   = "2024-06-30"
TEST2_START = "2024-07-01"  # -> regime label "calm_test"
TEST2_END   = "2025-12-31"

# The regime labels "stressed_test"/"calm_test" are carried over unchanged
# from market_sim.py purely because train.py/run_seed_sweep.py/
# compare_ablation.py hard-code those exact strings throughout. With real
# data they just mean "test period 1" (2023-01 to 2024-06) and "test period
# 2" (2024-07 to 2025-12) respectively -- they are NOT a claim that the first
# window was necessarily more volatile than the second. Check the actual
# realised ann_vol per segment in the results report rather than relying on
# the label.


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
def download_ohlcv(tickers, start, end, cache_dir=".yf_cache", max_retries=3, pause=2.0):
    """Downloads daily OHLCV bars for each ticker individually (simpler and
    more robust than a single multi-ticker call, which returns awkward
    MultiIndex columns), with a small on-disk CSV cache so repeated runs
    (e.g. across seeds in run_seed_sweep.py) don't re-hit the network."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "The real-data path (cfg['market']['data_source'] == 'real') requires "
            "the `yfinance` package. Install it with:\n"
            "    pip install yfinance --break-system-packages\n"
            "(or just `pip install yfinance` in a normal environment), or switch "
            "cfg['market']['data_source'] to 'synthetic' to use market_sim.py instead."
        ) from e
    os.makedirs(cache_dir, exist_ok=True)
    data = {}
    for t in tickers:
        cache_path = os.path.join(cache_dir, f"{t.replace('.', '_')}_{start}_{end}.csv")
        df = None
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            for attempt in range(max_retries):
                try:
                    print(f"[data_loader] downloading {t} ({start} -> {end}) "
                          f"[attempt {attempt + 1}/{max_retries}]...")
                    df = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
                    if df is not None and len(df) > 0:
                        break
                except Exception as e:
                    print(f"  [{t}] download failed: {e}")
                time.sleep(pause)
            if df is None or len(df) == 0:
                raise RuntimeError(
                    f"Failed to download any data for ticker '{t}' after {max_retries} "
                    f"attempts. Check the ticker symbol and your internet connection."
                )
            # yfinance sometimes returns MultiIndex columns even for a single
            # ticker (depends on version) -- flatten defensively.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(cache_path)
        data[t] = df
    return data


# ----------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------
def _adjust_ohlc(df):
    """Back-adjusts Open/High/Low by the same factor implied by
    Adj Close / Close, so that stock splits and large dividend adjustments
    don't create spurious jumps in the intraday-range / overnight-gap
    features (mirrors what Adj Close already does for the close-to-close
    return series)."""
    factor = df["Adj Close"] / df["Close"]
    out = pd.DataFrame(index=df.index)
    out["Open"] = df["Open"] * factor
    out["High"] = df["High"] * factor
    out["Low"] = df["Low"] * factor
    out["Close"] = df["Adj Close"]
    out["Volume"] = df["Volume"]
    return out


def build_features(raw_data, tickers, vol_window=20):
    """
    Builds a 6-channel-per-asset feature panel from raw OHLCV data, using
    ALL FIVE raw fields (open, close, volume, high/"max", low/"min"):

      channel 0 : daily close-to-close simple return          (own return)
      channel 1 : rolling `vol_window`-day std of channel 0    (own realised vol)
      channel 2 : same-day open-to-close return  (Close/Open - 1)
      channel 3 : intraday high-low range as a fraction of close ((High-Low)/Close)
      channel 4 : overnight gap return  (Open_t / Close_{t-1} - 1)
      channel 5 : log-volume change     (log1p(Volume_t) - log1p(Volume_{t-1}))

    None of these 6 channels is naturally bounded in [0,1] (unlike the
    synthetic wind-capacity-factor channel in market_sim.py), so the caller
    should apply causal_rolling_normalise(..., unit_interval_channels=())
    to properly z-score every channel -- see the patched signature in
    market_sim.py.

    Returns (returns_df, features, dates):
      returns_df : DataFrame (T, n_assets), columns = tickers, close-to-close
                   simple returns, first day dropped (no prior close)
      features   : ndarray (T, n_assets, 6), same T/row-alignment as returns_df
      dates      : DatetimeIndex of length T, the common trading-day index
    """
    adj = {t: _adjust_ohlc(raw_data[t]) for t in tickers}

    # Align on the INTERSECTION of trading days across all exchanges/tickers.
    # European exchanges (LSE, Borsa Italiana, BME, Xetra, Oslo Bors,
    # Nasdaq Copenhagen, Euronext Paris) don't share identical holiday
    # calendars, and a ticker that IPO'd after TRAIN_START will simply have
    # no rows before its IPO date -- taking the intersection handles both
    # cases automatically without manual calendar bookkeeping.
    #
    # IMPORTANT: this intersection can silently truncate the requested
    # training window. E.g. ORSTED.CO (Orsted, formerly DONG Energy) only
    # IPO'd on Nasdaq Copenhagen on 2016-06-09 -- if it's in your universe,
    # NO ticker's common_idx can start before that date, regardless of
    # `train_start`. Print each ticker's own history range plus the final
    # intersected range so this is visible rather than silently absorbed
    # into a slightly-smaller-than-expected day count.
    print("[data_loader] per-ticker downloaded history range:")
    for t in tickers:
        idx_t = adj[t].index
        print(f"[data_loader]   {t:12s}: {idx_t[0].date()} -> {idx_t[-1].date()}  ({len(idx_t)} rows)")

    common_idx = adj[tickers[0]].index
    for t in tickers[1:]:
        common_idx = common_idx.intersection(adj[t].index)
    common_idx = common_idx.sort_values()
    print(f"[data_loader] common (intersected) trading-day range: "
          f"{common_idx[0].date()} -> {common_idx[-1].date()}  ({len(common_idx)} rows)")

    n = len(tickers)
    T = len(common_idx)
    returns = np.zeros((T, n))
    features = np.zeros((T, n, 6))

    for i, t in enumerate(tickers):
        d = adj[t].reindex(common_idx)
        close = d["Close"].values.astype(np.float64)
        open_ = d["Open"].values.astype(np.float64)
        high = d["High"].values.astype(np.float64)
        low = d["Low"].values.astype(np.float64)
        vol = d["Volume"].values.astype(np.float64)

        ret = np.zeros(T)
        ret[1:] = close[1:] / close[:-1] - 1.0
        returns[:, i] = ret

        oc_ret = np.zeros(T)
        valid_open = open_ != 0
        oc_ret[valid_open] = close[valid_open] / open_[valid_open] - 1.0

        hl_range = np.zeros(T)
        valid_close = close != 0
        hl_range[valid_close] = (high[valid_close] - low[valid_close]) / close[valid_close]

        gap_ret = np.zeros(T)
        prev_close_valid = close[:-1] != 0
        gap_ret[1:][prev_close_valid] = (open_[1:][prev_close_valid]
                                          / close[:-1][prev_close_valid] - 1.0)

        log_vol = np.log1p(vol)
        vol_chg = np.zeros(T)
        vol_chg[1:] = log_vol[1:] - log_vol[:-1]

        # STRICTLY CAUSAL fill for the leading rows before the rolling window has
        # `min_periods` observations: previously `.bfill()` pulled in a value
        # computed from a LATER row (using data past the current one) -- a real
        # look-ahead leak, confined to the first few rows but still a leak. An
        # expanding (min_periods=2) std only ever uses data up to and including
        # the current row, so swap it in for exactly the rows that are still NaN;
        # the single very first row (no std defined at all) falls back to 0.0.
        ret_s = pd.Series(ret)
        roll_vol = ret_s.rolling(vol_window, min_periods=5).std()
        roll_vol = roll_vol.fillna(ret_s.expanding(min_periods=2).std()).fillna(0.0).values

        features[:, i, 0] = ret
        features[:, i, 1] = roll_vol
        features[:, i, 2] = oc_ret
        features[:, i, 3] = hl_range
        features[:, i, 4] = gap_ret
        features[:, i, 5] = vol_chg

    returns_df = pd.DataFrame(returns, index=common_idx, columns=tickers)
    # Day 0 has no prior close (return/gap are undefined, set to 0 above) --
    # drop it so every remaining row is a genuine, fully-defined observation.
    return returns_df.iloc[1:], features[1:], common_idx[1:]


# ----------------------------------------------------------------------
# Regime assignment (real calendar dates -> the 3 labels the pipeline reads)
# ----------------------------------------------------------------------
def assign_regime(dates, train_end, test1_start, test1_end, test2_start, test2_end):
    dates = pd.DatetimeIndex(dates)
    regime = np.full(len(dates), "unassigned", dtype=object)
    regime[dates <= pd.Timestamp(train_end)] = "train"
    regime[(dates >= pd.Timestamp(test1_start)) & (dates <= pd.Timestamp(test1_end))] = "stressed_test"
    regime[(dates >= pd.Timestamp(test2_start)) & (dates <= pd.Timestamp(test2_end))] = "calm_test"
    return regime


# ----------------------------------------------------------------------
# Asset graph (empirical replacement for market_sim.build_asset_graph)
# ----------------------------------------------------------------------
def build_empirical_asset_graph(returns_df, regime, threshold=0.5):
    """Row-stochastic adjacency from pairwise return correlation, computed
    on the TRAIN segment only (no lookahead), thresholded and self-looped.
    Same output contract as market_sim.build_asset_graph() (n x n,
    row-normalised) so GraphConvLayer needs no changes.

    `threshold` is on a correlation scale (typically 0.2-0.7 for large-cap
    European energy names), NOT the cosine-similarity-of-exposures scale
    market_sim.py's default (0.35) was tuned for -- 0.5 is a reasonable
    starting point but worth sanity-checking against the actual correlation
    matrix for your ticker set (very high thresholds can produce a
    disconnected graph for some rows if no pair clears the bar; the
    self-loop guarantees each row sums to at least 1 either way)."""
    train_mask = (np.asarray(regime) == "train")
    R = returns_df.values[train_mask]
    corr = np.corrcoef(R.T)
    A = (corr >= threshold).astype(np.float64)
    np.fill_diagonal(A, 1.0)
    A_hat = A / A.sum(axis=1, keepdims=True)
    return A_hat


# ----------------------------------------------------------------------
# Top-level entry point -- same output contract as market_sim.simulate_market
# ----------------------------------------------------------------------
def load_real_market(market_cfg):
    """
    market_cfg: the cfg["market"] sub-dict. Recognised keys (all optional,
    fall back to the module-level defaults above):
      tickers, train_start, train_end, test1_start, test1_end,
      test2_start, test2_end, feature_vol_window, cache_dir

    Returns a dict with the SAME shape as market_sim.simulate_market():
      returns  : DataFrame (T, n_assets) daily simple returns
      features : ndarray (T, n_assets, 6) raw (not yet normalised) feature panel
      regime   : ndarray (T,) of {"train","stressed_test","calm_test"}
      dates    : DatetimeIndex (T,) -- extra key, not present in the synthetic
                 output, kept for convenience/debugging
      macro    : placeholder DataFrame, kept only for interface parity
    """
    tickers = market_cfg.get("tickers", TICKERS)
    train_start = market_cfg.get("train_start", TRAIN_START)
    train_end = market_cfg.get("train_end", TRAIN_END)
    test1_start = market_cfg.get("test1_start", TEST1_START)
    test1_end = market_cfg.get("test1_end", TEST1_END)
    test2_start = market_cfg.get("test2_start", TEST2_START)
    test2_end = market_cfg.get("test2_end", TEST2_END)
    vol_window = market_cfg.get("feature_vol_window", 20)
    cache_dir = market_cfg.get("cache_dir", ".yf_cache")

    print(f"[data_loader] loading {len(tickers)} real tickers "
          f"({train_start} -> {test2_end})...")
    raw = download_ohlcv(tickers, train_start, test2_end, cache_dir=cache_dir)
    returns_df, features, dates = build_features(raw, tickers, vol_window=vol_window)

    # WARNING: flag (rather than silently absorb) a meaningfully shortened
    # training window caused by a late-IPO ticker in the universe -- see the
    # per-ticker range printout in build_features() above for which ticker
    # is the limiting one.
    requested_train_start = pd.Timestamp(train_start)
    actual_start = dates[0]
    if actual_start > requested_train_start + pd.Timedelta(days=30):
        gap_days = (actual_start - requested_train_start).days
        print(f"[data_loader] WARNING: requested train_start={train_start}, but the earliest "
              f"date common to ALL {len(tickers)} tickers is {actual_start.date()} -- the "
              f"effective TRAINING window is about {gap_days} days (~{gap_days/365.25:.1f} "
              f"years) SHORTER than requested. This is caused by whichever ticker above has "
              f"the latest history-start date (a late IPO, e.g. Orsted/ORSTED.CO IPO'd "
              f"2016-06-09). Either accept the shorter effective training window, or drop "
              f"the limiting ticker from `market.tickers` for training purposes.")

    regime = assign_regime(dates, train_end, test1_start, test1_end, test2_start, test2_end)
    keep = regime != "unassigned"
    if not keep.all():
        print(f"[data_loader] dropping {int((~keep).sum())} day(s) that fall outside "
              f"the configured train/test1/test2 windows")
    returns_df = returns_df.loc[keep].reset_index(drop=True)
    features = features[keep]
    regime = regime[keep]
    dates = dates[keep]

    for name in ("train", "stressed_test", "calm_test"):
        count = int((regime == name).sum())
        print(f"[data_loader]   {name:14s}: {count} trading days")
        if count == 0:
            raise ValueError(
                f"Regime '{name}' has zero trading days after alignment -- check that "
                f"{name}'s configured date window actually overlaps the downloaded data "
                f"(e.g. a ticker with a very late IPO can shrink the common trading-day "
                f"intersection enough to empty out an early window)."
            )

    macro = pd.DataFrame({"date": dates})  # kept only for interface parity with market_sim

    return {
        "returns": returns_df,
        "features": features,
        "regime": regime,
        "dates": dates,
        "macro": macro,
    }
