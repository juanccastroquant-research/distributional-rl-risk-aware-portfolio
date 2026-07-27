"""
compare_ablation.py
--------------------
Compares an ablation sweep (e.g. use_gnn=false, or es_mode=historical) against the
baseline sweep, using the SAME seeds in both, and reports whether the ablated
component actually changed anything -- rather than eyeballing two separate sweep
reports side by side.

This directly operationalises H3 (does the GNN help?) and H4 (critic-derived vs
historical ES) from the research proposal, using the same seeds-not-a-single-run
discipline as the main sweep.

Usage:
    python3 compare_ablation.py energy_rl_sim_results/sweep_results energy_rl_sim_results/sweep_results_no_gnn
    python3 compare_ablation.py energy_rl_sim_results/sweep_results energy_rl_sim_results/sweep_results_historical_es --out energy_rl_sim_results/my_report.txt

    Note: comparison_report.txt is written INTO ablation_dir by default (see
    --out below), so as long as ablation_dir itself lives under
    energy_rl_sim_results/ (the run_seed_sweep.py default), this report ends
    up there too rather than in a separate sibling folder.
"""
import argparse
import pandas as pd


def load_dir(d):
    perf = pd.read_csv(f"{d}/aggregated_performance.csv")
    sig = pd.read_csv(f"{d}/aggregated_significance.csv")
    try:
        with open(f"{d}/ablation_settings.txt") as f:
            settings = f.read().strip()
    except FileNotFoundError:
        settings = "(no ablation_settings.txt found -- directory predates this feature)"
    return perf, sig, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir")
    ap.add_argument("ablation_dir")
    ap.add_argument("--out", type=str, default=None,
                    help="where to write the comparison report (default: "
                         "<ablation_dir>/comparison_report.txt)")
    args = ap.parse_args()

    base_perf, base_sig, base_settings = load_dir(args.baseline_dir)
    abl_perf, abl_sig, abl_settings = load_dir(args.ablation_dir)

    out_path = args.out or f"{args.ablation_dir}/comparison_report.txt"
    lines = []
    add = lines.append

    add("=" * 78)
    add("ABLATION COMPARISON REPORT")
    add("=" * 78)
    add("")
    add(f"Baseline directory : {args.baseline_dir}")
    add(f"  settings: {base_settings}")
    add(f"Ablation directory : {args.ablation_dir}")
    add(f"  settings: {abl_settings}")
    add("")
    add("NOTE: this comparison is only meaningful if both directories used the SAME")
    add("seeds -- different seeds would confound the ablation effect with ordinary")
    add("seed variance. Check the seed lists in each directory's sweep_report.txt.")
    add("")

    # ---- performance comparison: RL-Dirichlet-DistCritic Sharpe, both segments ----
    add("1. RL POLICY PERFORMANCE: baseline vs. ablation")
    add("-" * 78)
    rl_name = "RL-Dirichlet-DistCritic"
    for segment in ["stressed_test", "calm_test"]:
        b = base_perf[(base_perf.segment == segment) & (base_perf.strategy == rl_name)]
        a = abl_perf[(abl_perf.segment == segment) & (abl_perf.strategy == rl_name)]
        if len(b) == 0 or len(a) == 0:
            add(f"  {segment}: missing data in one of the two directories, skipping")
            continue
        b, a = b.iloc[0], a.iloc[0]
        delta = a["sharpe_mean"] - b["sharpe_mean"]
        add(f"  {segment}:")
        add(f"    baseline sharpe_mean  = {b['sharpe_mean']:.4f}  (std {b['sharpe_std']:.4f}, n={int(b['n_seeds'])})")
        add(f"    ablation sharpe_mean  = {a['sharpe_mean']:.4f}  (std {a['sharpe_std']:.4f}, n={int(a['n_seeds'])})")
        add(f"    delta (ablation - baseline) = {delta:+.4f}")
    add("")

    # ---- significance comparison: RL vs each benchmark, baseline vs ablation ----
    add("2. RL vs. EACH BENCHMARK (stressed_test): baseline vs. ablation")
    add("-" * 78)
    add(f"{'benchmark':<14}{'base_diff':>11}{'abl_diff':>11}{'base_win%':>11}{'abl_win%':>11}{'delta_win%':>12}")
    merged = base_sig.merge(abl_sig, on="benchmark", suffixes=("_base", "_abl"))
    for _, row in merged.iterrows():
        delta_win = row["frac_seeds_RL_wins_abl"] - row["frac_seeds_RL_wins_base"]
        add(f"{row['benchmark']:<14}{row['mean_sharpe_diff_base']:>11.4f}"
            f"{row['mean_sharpe_diff_abl']:>11.4f}"
            f"{row['frac_seeds_RL_wins_base']*100:>10.0f}%"
            f"{row['frac_seeds_RL_wins_abl']*100:>10.0f}%"
            f"{delta_win*100:>+11.0f}%")
    add("")

    # ---- interpretation ----
    add("3. HOW TO READ THIS")
    add("-" * 78)
    add("  Large |delta| in sharpe_mean or in win-rate (say, >20-30 percentage points)")
    add("  is evidence the ablated component matters. A small delta means the removed")
    add("  component (GNN, or critic-derived ES) is not doing much work relative to the")
    add("  rest of the pipeline -- for THIS synthetic market and THIS training budget;")
    add("  it does not prove the component is useless in general.")
    add("")
    vs_1n = merged[merged.benchmark.isin(["1/N"])]
    if len(vs_1n):
        d = float(vs_1n.iloc[0]["frac_seeds_RL_wins_abl"] - vs_1n.iloc[0]["frac_seeds_RL_wins_base"])
        if abs(d) < 0.15:
            add(f"  RL-vs-1/N win-rate barely moved ({d*100:+.0f} points) -- this ablation does")
            add("  not appear to explain whatever edge (or lack of one) the baseline showed.")
        else:
            add(f"  RL-vs-1/N win-rate moved by {d*100:+.0f} points -- this ablation looks like a")
            add("  real contributor to the baseline result; worth investigating further.")
    add("=" * 78)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
