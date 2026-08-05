# European Energy Portfolio RL — Code Review & Run Guide

This is a from-scratch (NumPy-only, no PyTorch/TensorFlow) reinforcement-learning
pipeline that trains a Dirichlet-policy portfolio-allocation agent with a
distributional, Expected-Shortfall-aware critic on a European energy-sector
asset universe, and benchmarks it against classical passive and convex-optimisation
strategies. It can run on **real Yahoo Finance data** or on a **synthetic,
physically-motivated market simulator** that needs no internet access.
 

## What's in this folder

| File | Role |
|---|---|
| `train.py` | Main orchestrator: builds the market/agent, runs the actor-critic training loop, evaluates vs. benchmarks, writes `results_report.txt` + plots. Also exposes `train()`, `evaluate_policy()`, `evaluate_benchmarks()` for the other scripts to import. |
| `market_sim.py` | Synthetic European energy market (S2/S6): 8 archetypal assets driven by shared gas/carbon/wind/electricity factors, a stressed regime with supply-shock jumps, the physically-informed asset graph, causal rolling normalisation, capped-simplex projection, and `PortfolioEnv`. |
| `data_loader.py` | Real-data counterpart: downloads OHLCV for a 10-ticker European energy universe via `yfinance`, builds a 6-channel feature panel, assigns train/stressed_test/calm_test date windows, builds an empirical (correlation-based) asset graph. **Needs internet + `pip install yfinance`.** |
| `encoders.py` | `AttentionEncoder` (Transformer-lite temporal pooling) and `GraphConvLayer` (single GCN layer over the asset graph) — hand-differentiated, no autograd. |
| `models.py` | `DirichletActor` (policy) and `QuantileCritic` (QR-DQN-style distributional critic, used both for TD targets and critic-derived ES). |
| `nn_core.py` | Dense layer, activations, Adam optimiser — the whole NN toolkit, dependency-free. |
| `benchmarks.py` | 1/N, buy-and-hold, static cap-weighted, rolling CVaR-optimal LP (Rockafellar–Uryasev), risk parity/ERC (Maillard et al.). |
| `metrics.py` | Sharpe/Sortino/Calmar/Ulcer/VaR/ES/Omega/skew/kurtosis, and a stationary block-bootstrap Sharpe-difference significance test (White/Hansen-style). |
| `run_seed_sweep.py` | Trains across multiple seeds (optionally split into independent short calls) and reports the *distribution* of outcomes, not a single point estimate. Also drives the `use_gnn`/`es_mode` ablations. |
| `run_ablation_lambda_tc0_real.py` | Runs baseline vs. `lambda_tc=0` vs. `lambda_tc=0 + lower beta_min`, wired to the real `train()`/`evaluate_policy()`. |
| `run_advantage_diagnosis.py` | Trains once with extra instrumentation and prints/saves a diagnosis of whether a flat policy is an actor problem or a critic/reward problem (per-asset advantage spread, critic counterfactual probe, gradient attribution). |
| `compare_ablation.py` | Diffs two `run_seed_sweep.py` output directories (baseline vs. an ablation) using the *same seeds*. |
| `gradcheck.py`, `gradcheck_encoders.py`, `gradcheck_models.py` | Numerical gradient checks for every hand-written backward pass. |
| `config.yaml` | Full run config, real-data path (`market.data_source: real`). |
| `config_synthetic.yaml` | **Added by this review** — same hyperparameters, synthetic-market path (`market.data_source: synthetic`), no internet required. |

---

## How to run it

### Fastest path (no internet, no API keys) — synthetic market

```bash
pip install numpy pandas scipy pyyaml matplotlib --break-system-packages

python3 gradcheck.py && python3 gradcheck_encoders.py && python3 gradcheck_models.py

python3 train.py config_synthetic.yaml energy_rl_sim_results_synthetic
```

This trains the full pipeline (120 episodes by default) on the synthetic
8-asset universe and writes a `results_report.txt`, `metrics_table.csv`,
`training_history.csv`, and several diagnostic plots to
`energy_rl_sim_results_synthetic/`.

### Real-data path (10-ticker European energy universe, 2011–2025)

```bash
pip install yfinance --break-system-packages
python3 train.py config.yaml energy_rl_sim_results
```

Requires outbound internet access to Yahoo Finance (not available in a
network-sandboxed environment — the egress allowlist most sandboxes ship with
covers package registries only, not `finance.yahoo.com`). Downloaded OHLCV is
cached under `.yf_cache/` so repeated runs/seeds don't re-hit the network.

### Multi-seed robustness sweep

A single-seed Sharpe comparison can't distinguish a genuine edge from seed
noise. Run several seeds and look at the *distribution*:

```bash
python3 run_seed_sweep.py config_synthetic.yaml --seeds 5 \
    --out energy_rl_sim_results/sweep_results
```

or split it into independent short calls (useful if your process has a
runtime limit):

```bash
python3 run_seed_sweep.py config_synthetic.yaml --single_seed 42 --out energy_rl_sim_results/sweep_results
python3 run_seed_sweep.py config_synthetic.yaml --single_seed 43 --out energy_rl_sim_results/sweep_results
python3 run_seed_sweep.py config_synthetic.yaml --aggregate_only --out energy_rl_sim_results/sweep_results
```

Read `frac_seeds_RL_wins` and `frac_seeds_p_below_010` in the output, not any
single seed's p-value.

### Ablations

```bash
# H3: does the GNN help?
python3 run_seed_sweep.py config_synthetic.yaml --use_gnn false --seeds 5 \
    --out energy_rl_sim_results/sweep_results_no_gnn
python3 compare_ablation.py energy_rl_sim_results/sweep_results \
    energy_rl_sim_results/sweep_results_no_gnn

# H4: critic-derived vs. historical Expected Shortfall
python3 run_seed_sweep.py config_synthetic.yaml --es_mode historical --seeds 5 \
    --out energy_rl_sim_results/sweep_results_historical_es
python3 compare_ablation.py energy_rl_sim_results/sweep_results \
    energy_rl_sim_results/sweep_results_historical_es

# Does removing the transaction-cost penalty let the policy concentrate?
python3 run_ablation_lambda_tc0_real.py config_synthetic.yaml ablation_real_out
```

### "Why is my policy staying near uniform?" diagnostics

```bash
python3 run_advantage_diagnosis.py config_synthetic.yaml advantage_diag_out
```

Tells you, from three independent angles (per-asset REINFORCE gradient
spread, a critic counterfactual probe, and gradient-attribution at the
critic's first layer), whether a flat policy is an **actor**-side issue
(entropy control, transaction cost, REINFORCE variance) or a
**critic/reward**-side issue (no learnable cross-asset signal at all).

---

## Config keys (both `config.yaml` and `config_synthetic.yaml`)

| Section | Key | Meaning |
|---|---|---|
| `market` | `data_source` | `real` (Yahoo Finance via `data_loader.py`) or `synthetic` (`market_sim.py`) |
| `state` | `lookback` | Attention pooling window, in trading days |
| `state` | `use_gnn` | `false` → identity adjacency (ablation H3) |
| `state` | `graph_similarity_threshold` | **Scale-dependent** — see warning below |
| `actor` | `alpha_floor` | Minimum Dirichlet concentration per asset (avoids the zero-weight boundary pathology) |
| `actor` | `w_max` | Hard position cap, enforced by Euclidean projection onto the capped simplex |
| `actor` | `beta_init/min/max/lr` | SAC-style auto-tuned entropy temperature |
| `actor` | `entropy_target_margin` | Target entropy = H(Dir(1,...,1)) − margin |
| `critic` | `n_quantiles` | QR-DQN quantile count, used for TD targets and critic-derived ES |
| `risk` | `es_mode` | `critic` (learned tail) or `historical` (rolling window, ablation H4) |
| `costs` | `lambda_tc` | L1 turnover cost coefficient |
| `training` | `critic_warmup_updates` | Critic TD updates required before the actor starts learning |
| `training` | `actor_update_every_steps` | Actor gradient step every K env steps (set `null` for once-per-episode) |

**⚠️ `graph_similarity_threshold` is not portable between the two configs.**
In `market_sim.py` it's a cosine-similarity-of-hand-specified-exposures
threshold (default `0.35`); in `data_loader.py` it's a raw return-correlation
threshold (default `0.5`). `config.yaml` and `config_synthetic.yaml` each use
the value appropriate to their own data source — don't copy this number
across the two files.

---

## Honest simplifications / things to keep in mind

- **All hyperparameters (learning rates, `gamma`, `n_quantiles`, network
  widths, etc.) are illustrative starting points, not tuned.** Treat any
  Sharpe/Sharpe-diff numbers as evidence the pipeline runs correctly
  end-to-end, not as a calibrated empirical claim. This is stated directly in
  `train.py`'s own generated report.
- **The synthetic market is synthetic.** It's structurally faithful (shared
  macro factors, an asymmetric crash term concentrated on gas-heavy names,
  regime buffers so labelled segments aren't mostly spent transitioning) but
  it is not real TTF/EUA/ENTSO-E/equity data. Swapping in real data only
  requires `market.data_source: real`.
- **A single seed is not evidence.** `run_seed_sweep.py` exists specifically
  because Sharpe-ratio gaps of the size this pipeline tends to produce
  (p ≈ 0.07–0.09 vs. 1/N in earlier single-seed runs) are exactly the regime
  where seed variance can flip the sign of the conclusion.
- **The actor and critic each own independent encoder weights** (separate
  `AttentionEncoder` + `GraphConvLayer` instances) — they do not share
  parameters, and are optimised by two separate Adam instances.
- **Advantage is a single scalar per step** (`objective = E[Z] − λ·ES`, both
  portfolio-level). There is no per-asset value head in this architecture;
  any cross-asset credit assignment happens only through `dlogp` (REINFORCE),
  which is high-variance by construction. `run_advantage_diagnosis.py` exists
  to tell you when this is actually the bottleneck vs. when the critic itself
  hasn't learned a cross-asset signal to exploit.

## Verification performed for this review

- ✅ `gradcheck.py`, `gradcheck_encoders.py`, `gradcheck_models.py` — all pass
  (max errors ~1e-10 to 1e-11, well under the 1e-4 tolerance).
- ✅ `train.py` — full run on synthetic data, both `actor_update_every_steps`
  set and `null` (once-per-episode fallback).
- ✅ `run_seed_sweep.py` — all-in-one mode, split single-seed + `--aggregate_only`
  mode, `--use_gnn false`, `--es_mode historical`.
- ✅ `compare_ablation.py` — diffing a baseline sweep against a `use_gnn=false`
  sweep.
- ✅ `run_ablation_lambda_tc0_real.py` — all three ablation configs
  (`baseline`, `lambda_tc_0`, `lambda_tc_0_lower_beta_min`).
- ✅ `run_advantage_diagnosis.py` — full diagnostic log generation and
  interpretation logic.
- ✅ New `config_synthetic.yaml` re-validated directly against `train.py`,
  `run_advantage_diagnosis.py`, and `run_ablation_lambda_tc0_real.py` (their
  own hard-coded default config path).

No exceptions, shape mismatches, or NaN blow-ups were observed in any of the
above.
