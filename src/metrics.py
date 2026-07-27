"""
metrics.py
----------
Evaluation metrics recommended in S8.1 of the proposal, plus a simplified
stationary block-bootstrap significance test in the spirit of White's (2000)
Reality Check / Hansen's (2005) SPA test -- a lightweight approximation
(single pairwise comparison, not the full multiple-testing correction) that
is enough to flag whether an apparent Sharpe-ratio improvement survives a
basic resampling check.
"""
import numpy as np


def max_drawdown(cum_returns):
    running_max = np.maximum.accumulate(cum_returns)
    dd = cum_returns / running_max - 1.0
    return dd.min()


def ulcer_index(cum_returns):
    """Root-mean-square of the drawdown series (Martin & McCann, 1987) -- a
    drawdown-depth-and-duration-sensitive complement to max drawdown."""
    running_max = np.maximum.accumulate(cum_returns)
    dd_pct = (cum_returns / running_max - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def historical_var(r, level=0.95):
    """Historical (non-parametric) Value-at-Risk: the loss such that
    (1-level) of daily returns are worse than it. Reported as a negative
    daily return, consistent with the ES convention used elsewhere."""
    k = max(1, int(np.ceil((1 - level) * len(r))))
    return float(np.sort(r)[k - 1])


def historical_es(r, level=0.95):
    k = max(1, int(np.ceil((1 - level) * len(r))))
    return float(np.sort(r)[:k].mean())


def omega_ratio(r, threshold=0.0):
    """Omega ratio (Keating & Shadwick, 2002): sum of gains above a
    threshold divided by sum of losses below it. >1 is favourable."""
    gains = np.sum(np.maximum(r - threshold, 0.0))
    losses = np.sum(np.maximum(threshold - r, 0.0))
    return float(gains / (losses + 1e-12))


def _skew(r):
    m = r.mean()
    s = r.std(ddof=1) + 1e-12
    return float(np.mean(((r - m) / s) ** 3))


def _kurtosis(r):
    """Excess kurtosis (0 = normal)."""
    m = r.mean()
    s = r.std(ddof=1) + 1e-12
    return float(np.mean(((r - m) / s) ** 4) - 3.0)


def performance_summary(daily_returns, turnovers=None, weights_history=None,
                         es_alpha=0.95, ann_factor=252):
    r = np.asarray(daily_returns)
    cum = np.cumprod(1.0 + r)
    ann_return = r.mean() * ann_factor
    ann_vol = r.std(ddof=1) * np.sqrt(ann_factor)
    sharpe = ann_return / (ann_vol + 1e-12)

    downside = r[r < 0]
    downside_dev = downside.std(ddof=1) * np.sqrt(ann_factor) if len(downside) > 1 else np.nan
    sortino = ann_return / (downside_dev + 1e-12) if downside_dev == downside_dev else np.nan

    mdd = max_drawdown(cum)
    calmar = ann_return / (abs(mdd) + 1e-12)
    ui = ulcer_index(cum)

    out = {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "daily_ret_std": float(r.std(ddof=1)),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "ulcer_index": ui,
        f"VaR{int(es_alpha*100)}_daily": historical_var(r, es_alpha),
        f"ES{int(es_alpha*100)}_daily": historical_es(r, es_alpha),
        f"ES99_daily": historical_es(r, 0.99),
        "omega_ratio": omega_ratio(r),
        "skew": _skew(r),
        "excess_kurtosis": _kurtosis(r),
        "hit_rate": float(np.mean(r > 0)),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
        "cum_return": cum[-1] - 1.0,
        "n_days": len(r),
    }
    if turnovers is not None:
        out["avg_turnover"] = float(np.mean(turnovers))
        out["annual_turnover"] = float(np.mean(turnovers) * ann_factor)
    if weights_history is not None:
        wh = np.asarray(weights_history)
        out["avg_HHI"] = float(np.mean(np.sum(wh ** 2, axis=1)))
        out["max_single_weight"] = float(np.max(wh))
        out["avg_n_effective_assets"] = float(np.mean(1.0 / np.sum(wh ** 2, axis=1)))
    return out


def relative_performance(returns_a, returns_b, ann_factor=252):
    """Tracking error and Information Ratio of strategy `a` relative to a
    reference strategy `b` (e.g. the RL policy vs. 1/N), aligned by length."""
    ra, rb = np.asarray(returns_a), np.asarray(returns_b)
    T = min(len(ra), len(rb))
    diff = ra[:T] - rb[:T]
    tracking_error = float(diff.std(ddof=1) * np.sqrt(ann_factor))
    ann_excess = float(diff.mean() * ann_factor)
    information_ratio = ann_excess / (tracking_error + 1e-12)
    return {"tracking_error": tracking_error, "ann_excess_return": ann_excess,
            "information_ratio": information_ratio}


def _block_bootstrap_indices(T, block_size, rng):
    n_blocks = int(np.ceil(T / block_size))
    starts = rng.integers(0, T - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:T]
    return idx


def bootstrap_sharpe_diff(returns_a, returns_b, block_size=20, n_boot=2000,
                           ann_factor=252, seed=0):
    """
    Stationary block-bootstrap test of H0: Sharpe(a) <= Sharpe(b), following
    the spirit of White (2000) / Hansen (2005). Returns:
      observed_diff : Sharpe(a) - Sharpe(b) on the actual data
      p_value       : fraction of bootstrap resamples, after centring the
                       resampled diffs on zero (imposing H0), whose centred
                       diff is >= the observed diff -- i.e. how often a gap
                       this large or larger shows up under the null. A low
                       value is evidence the observed gap is unlikely to be
                       pure sampling noise. (NOT "fraction of raw resamples
                       with diff <= 0" -- the resampled diffs are recentred
                       first; see the two lines above `p_value`.)
    """
    rng = np.random.default_rng(seed)
    ra, rb = np.asarray(returns_a), np.asarray(returns_b)
    T = min(len(ra), len(rb))
    ra, rb = ra[:T], rb[:T]

    def sharpe(x):
        return x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(ann_factor)

    observed_diff = sharpe(ra) - sharpe(rb)
    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = _block_bootstrap_indices(T, block_size, rng)
        diffs[b] = sharpe(ra[idx]) - sharpe(rb[idx])
    # centre the bootstrap distribution on 0 (test statistic under H0) then
    # compare the observed statistic to that null distribution
    centred = diffs - diffs.mean()
    p_value = np.mean(centred >= observed_diff)
    return observed_diff, p_value
