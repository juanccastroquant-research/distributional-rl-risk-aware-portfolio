"""
run_seed_sweep.py
------------------
A single 40-episode, single-seed run is not evidence -- Sharpe-ratio differences
of the size seen here (p ~ 0.07-0.09 vs 1/N) are exactly the regime where seed
variance can flip the conclusion. This script trains the agent across several
independent seeds and reports the *distribution* of outcomes: mean/std Sharpe
per strategy per segment, and how many seeds the RL policy actually beats each
benchmark on, rather than a single point estimate.

All outputs from this script (and from compare_ablation.py) are written under
the SAME root results folder as train.py's own output
("energy_rl_sim_results/" by default), just in their own subdirectories, so a
single run of the whole pipeline produces ONE results folder rather than
scattering CSVs/reports across several sibling directories.

Usage:
    # Run several seeds in one call (fine for small seed counts / episode budgets)
    python3 run_seed_sweep.py config.yaml --seeds 5 --episodes 120 --out energy_rl_sim_results/sweep_results

    # Split the sweep into short, independent calls (avoids long-process timeouts)
    python3 run_seed_sweep.py config.yaml --episodes 70 --single_seed 42 --out energy_rl_sim_results/sweep_results
    python3 run_seed_sweep.py config.yaml --episodes 70 --single_seed 43 --out energy_rl_sim_results/sweep_results
    python3 run_seed_sweep.py config.yaml --aggregate_only --out energy_rl_sim_results/sweep_results

    # Ablations (H3: GNN: value; H4: critic-derived vs historical ES) -- use a
    # SEPARATE --out directory per ablation so it doesn't overwrite the baseline,
    # and reuse the same seeds as the baseline sweep for a like-for-like comparison:
    python3 run_seed_sweep.py config.yaml --episodes 70 --use_gnn false \\
        --single_seed 42 --out energy_rl_sim_results/sweep_results_no_gnn
    python3 run_seed_sweep.py config.yaml --episodes 70 --es_mode historical \\
        --single_seed 42 --out energy_rl_sim_results/sweep_results_historical_es

    # Then compare an ablation directory against the baseline directory:
    python3 compare_ablation.py energy_rl_sim_results/sweep_results energy_rl_sim_results/sweep_results_no_gnn
"""
import argparse
import copy
import time
import yaml
import numpy as np
import pandas as pd

from train import train, evaluate_policy, evaluate_benchmarks
from metrics import performance_summary, bootstrap_sharpe_diff

BENCHMARK_NAMES = ["1/N", "CapWeighted", "BuyAndHold", "CVaR-LP", "RiskParity"]


def run_one_seed(cfg, seed):
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    trained = train(cfg)
    actor, env, data, A_hat_eff = (trained["actor"], trained["env"], trained["data"],
                                    trained["A_hat_eff"])
    rng = np.random.default_rng(seed + 1)

    rows = []
    sig_rows = []
    for segment in ["stressed_test", "calm_test"]:
        port_ret, turn, wh = evaluate_policy(actor, env, A_hat_eff, segment, rng)
        s = performance_summary(port_ret, turn, wh, es_alpha=cfg["risk"]["es_alpha"])
        s.update(seed=seed, segment=segment, strategy="RL-Dirichlet-DistCritic")
        rows.append(s)
        ra = port_ret

        bres = evaluate_benchmarks(cfg, data, segment)
        for name, (bp, bt, bw) in bres.items():
            s = performance_summary(bp, bt, bw, es_alpha=cfg["risk"]["es_alpha"])
            s.update(seed=seed, segment=segment, strategy=name)
            rows.append(s)

            if segment == "stressed_test" and name in BENCHMARK_NAMES:
                diff, p = bootstrap_sharpe_diff(
                    ra, bp, block_size=cfg["evaluation"]["bootstrap_block_size"],
                    n_boot=cfg["evaluation"]["bootstrap_n_boot"], seed=seed)
                sig_rows.append(dict(seed=seed, benchmark=name, sharpe_diff=diff,
                                      p_value_RL_better=p, rl_wins=diff > 0))
    return rows, sig_rows


def save_seed_result(rows, sig_rows, seed, out_dir):
    """Write one seed's results to its own file, so the sweep can be run as
    several independent short calls instead of one long-running process."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{out_dir}/raw_seed_{seed}.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(f"{out_dir}/sig_seed_{seed}.csv", index=False)


def load_all_seed_results(out_dir):
    """Collect every previously-saved per-seed file in out_dir (however many
    calls it took to produce them) and concatenate into the full sweep."""
    import glob
    raw_files = sorted(glob.glob(f"{out_dir}/raw_seed_*.csv"))
    sig_files = sorted(glob.glob(f"{out_dir}/sig_seed_*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw_seed_*.csv files found in {out_dir} -- "
                                 f"run at least one seed first (--single_seed).")
    df = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
    sig_df = pd.concat([pd.read_csv(f) for f in sig_files], ignore_index=True)
    seeds_found = sorted(df["seed"].unique().tolist())
    return df, sig_df, seeds_found


def aggregate_and_report(cfg, df, sig_df, seeds, out_dir):
    df.to_csv(f"{out_dir}/all_seeds_raw.csv", index=False)
    sig_df.to_csv(f"{out_dir}/all_seeds_significance.csv", index=False)

    agg = (df.groupby(["segment", "strategy"])
             .agg(sharpe_mean=("sharpe", "mean"), sharpe_std=("sharpe", "std"),
                  ann_return_mean=("ann_return", "mean"), ann_return_std=("ann_return", "std"),
                  max_drawdown_mean=("max_drawdown", "mean"),
                  ES95_daily_mean=("ES95_daily", "mean"),
                  n_seeds=("sharpe", "count"))
             .round(4))
    agg.to_csv(f"{out_dir}/aggregated_performance.csv")

    sig_agg = (sig_df.groupby("benchmark")
                 .agg(mean_sharpe_diff=("sharpe_diff", "mean"),
                      std_sharpe_diff=("sharpe_diff", "std"),
                      mean_p_value=("p_value_RL_better", "mean"),
                      frac_seeds_RL_wins=("rl_wins", "mean"),
                      frac_seeds_p_below_010=("p_value_RL_better", lambda x: (x < 0.10).mean()))
                 .round(4))
    sig_agg.to_csv(f"{out_dir}/aggregated_significance.csv")

    print(f"\n{'='*70}\nAGGREGATED PERFORMANCE (mean +/- std across {len(seeds)} seeds: {seeds})\n{'='*70}")
    print(agg.to_string())
    print(f"\n{'='*70}\nAGGREGATED SIGNIFICANCE (stressed_test, RL vs each benchmark, "
          f"{len(seeds)} seeds)\n{'='*70}")
    print(sig_agg.to_string())
    print(f"\nSaved: {out_dir}/all_seeds_raw.csv, all_seeds_significance.csv, "
          f"aggregated_performance.csv, aggregated_significance.csv")

    write_sweep_report(cfg, seeds, agg, sig_agg, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (uses 42,43,44,...)")
    ap.add_argument("--seed_list", type=str, default=None, help="comma-separated explicit seeds")
    ap.add_argument("--episodes", type=int, default=None, help="override training.n_episodes")
    ap.add_argument("--out", type=str, default="energy_rl_sim_results/sweep_results",
                    help="output directory for sweep files -- nested under the same "
                         "energy_rl_sim_results/ root that train.py writes to, so all "
                         "results from the whole pipeline live in one place")
    ap.add_argument("--use_gnn", type=str, default=None, choices=["true", "false"],
                    help="override state.use_gnn (ablation H3: false = identity adjacency, "
                         "no cross-asset propagation)")
    ap.add_argument("--es_mode", type=str, default=None, choices=["critic", "historical"],
                    help="override risk.es_mode (ablation H4: historical = rolling-window ES "
                         "instead of critic-derived ES in the actor objective)")
    ap.add_argument("--single_seed", type=int, default=None,
                    help="run ONLY this one seed and save its result to disk, then exit "
                         "(no aggregation) -- use this to split the sweep into several "
                         "short, independent calls")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="skip training entirely; just aggregate whatever raw_seed_*.csv "
                         "files already exist in --out")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.episodes is not None:
        cfg["training"]["n_episodes"] = args.episodes
    if args.use_gnn is not None:
        cfg["state"]["use_gnn"] = (args.use_gnn == "true")
    if args.es_mode is not None:
        cfg["risk"]["es_mode"] = args.es_mode

    # Record the effective ablation settings alongside the results, so a sweep
    # directory is self-describing regardless of how it was launched.
    import os as _os
    _os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/ablation_settings.txt", "w") as f:
        f.write(f"use_gnn = {cfg['state']['use_gnn']}\n")
        f.write(f"es_mode = {cfg['risk']['es_mode']}\n")
        f.write(f"n_episodes = {cfg['training']['n_episodes']}\n")

    if args.aggregate_only:
        df, sig_df, seeds_found = load_all_seed_results(args.out)
        aggregate_and_report(cfg, df, sig_df, seeds_found, args.out)
        return

    if args.single_seed is not None:
        seed = args.single_seed
        t0 = time.time()
        print(f"Running single seed {seed} ({cfg['training']['n_episodes']} episodes)...")
        rows, sig_rows = run_one_seed(cfg, seed)
        save_seed_result(rows, sig_rows, seed, args.out)
        print(f"Seed {seed} finished in {time.time()-t0:.1f}s. Saved to "
              f"{args.out}/raw_seed_{seed}.csv and sig_seed_{seed}.csv")
        print("Run more seeds the same way, then call with --aggregate_only to combine them.")
        return

    # ---- original all-in-one behaviour (fine for small seed counts / episodes) ----
    seeds = ([int(s) for s in args.seed_list.split(",")] if args.seed_list
             else [42 + i for i in range(args.seeds)])
    import os
    os.makedirs(args.out, exist_ok=True)
    all_rows, all_sig = [], []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}\nSEED {seed}  ({i+1}/{len(seeds)})\n{'='*70}")
        rows, sig_rows = run_one_seed(cfg, seed)
        all_rows.extend(rows)
        all_sig.extend(sig_rows)
        save_seed_result(rows, sig_rows, seed, args.out)  # incremental save even in this mode
    df, sig_df = pd.DataFrame(all_rows), pd.DataFrame(all_sig)
    print(f"\nTotal sweep time: {time.time()-t0:.1f}s across {len(seeds)} seeds "
          f"({cfg['training']['n_episodes']} episodes each)")
    aggregate_and_report(cfg, df, sig_df, seeds, args.out)


def write_sweep_report(cfg, seeds, agg, sig_agg, out_dir):
    lines = []
    add = lines.append
    add("=" * 78)
    add("MULTI-SEED SWEEP REPORT")
    add("=" * 78)
    add("")
    add(f"Seeds run: {seeds}")
    add(f"Episodes per seed: {cfg['training']['n_episodes']}")
    add(f"use_gnn: {cfg['state']['use_gnn']}   es_mode: {cfg['risk']['es_mode']}")
    add("")
    add("This report exists because a single-seed Sharpe-ratio comparison cannot")
    add("distinguish a genuine edge from seed variance. Read 'frac_seeds_RL_wins'")
    add("and 'frac_seeds_p_below_0.10' below as the real answer to 'is RL better',")
    add("not any single seed's p-value.")
    add("")
    add("AGGREGATED PERFORMANCE (mean +/- std across seeds)")
    add("-" * 78)
    add(agg.to_string())
    add("")
    add("AGGREGATED SIGNIFICANCE (stressed_test, RL vs each benchmark)")
    add("-" * 78)
    add("  frac_seeds_RL_wins       : fraction of seeds where RL's Sharpe > benchmark's")
    add("  frac_seeds_p_below_010   : fraction of seeds where the bootstrap test called")
    add("                             the RL-vs-benchmark gap significant at the 10% level")
    add(sig_agg.to_string())
    add("")
    add("HOW TO READ THIS")
    add("-" * 78)
    add("  frac_seeds_RL_wins close to 1.0  -> consistent, likely-genuine edge")
    add("  frac_seeds_RL_wins near 0.5      -> coin flip; the earlier single-seed result")
    add("                                      was noise, not a real effect")
    add("  sharpe_std comparable to or bigger than sharpe_mean's spread across strategies")
    add("                                   -> single-seed comparisons in this project are")
    add("                                      not reliable; always look at this sweep")
    add("=" * 78)

    with open(f"{out_dir}/sweep_report.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
