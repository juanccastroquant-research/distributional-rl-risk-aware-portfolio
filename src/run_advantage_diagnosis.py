"""
run_advantage_diagnosis.py

Runs train.train(cfg) (with the instrumentation added directly to train.py:
per-asset adv_term logging in actor_update, and a critic counterfactual
probe once per episode), then saves and interprets the two diagnostic logs.

Usage:
    python3 run_advantage_diagnosis.py config_synthetic.yaml out_dir
"""
import sys
import json
import numpy as np
import pandas as pd
import yaml

from train import train


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_synthetic.yaml"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "advantage_diag_out"
    import os
    os.makedirs(out_dir, exist_ok=True)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    trained = train(cfg)

    # ---- 1. per-asset adv_term log (from actor_update) --------------------
    padf = pd.DataFrame(trained["per_asset_log"])
    padf.to_csv(f"{out_dir}/per_asset_advantage_log.csv", index=False)

    # aggregate per episode: cross-asset spread of adv_term vs. how much the
    # scalar advantage itself varies within the episode (temporal noise)
    ep_rows = []
    for ep, g in padf.groupby("episode"):
        ep_rows.append({
            "episode": ep,
            "avg_adv_term_std_across_assets": g["adv_term_std_across_assets"].mean(),
            "avg_dlogp_std_across_assets": g["dlogp_std_across_assets"].mean(),
            "temporal_std_of_advantage": g["advantage"].std(),
            "avg_w_t_std_across_assets": g["w_t_std_across_assets"].mean(),
        })
    ep_df = pd.DataFrame(ep_rows).sort_values("episode")
    ep_df["cross_asset_to_temporal_ratio"] = (
        ep_df["avg_adv_term_std_across_assets"] / (ep_df["temporal_std_of_advantage"] + 1e-12)
    )
    ep_df.to_csv(f"{out_dir}/per_asset_advantage_by_episode.csv", index=False)

    # ---- 2. critic counterfactual probe log --------------------------------
    cf_rows = []
    for rec in trained["counterfactual_log"]:
        cf_rows.append({
            "episode": rec["episode"],
            "cross_asset_std": rec["cross_asset_std"],
            "cross_asset_spread": rec["cross_asset_spread"],
            "best_asset_idx": rec["best_asset_idx"],
            "worst_asset_idx": rec["worst_asset_idx"],
        })
    cf_df = pd.DataFrame(cf_rows).sort_values("episode")
    cf_df.to_csv(f"{out_dir}/critic_counterfactual_by_episode.csv", index=False)

    # ---- 3. gradient-attribution log (state vs action input to critic_h1) --
    ga_df = pd.DataFrame(trained["grad_attribution_log"])
    ga_df.to_csv(f"{out_dir}/gradient_attribution_log.csv", index=False)
    ga_ep = (ga_df.groupby("episode")
             .agg(state_grad_norm=("state_grad_norm", "mean"),
                  action_grad_norm=("action_grad_norm", "mean"),
                  action_grad_fraction=("action_grad_fraction", "mean"))
             .reset_index())
    ga_ep.to_csv(f"{out_dir}/gradient_attribution_by_episode.csv", index=False)

    # ---- interpretation -----------------------------------------------------
    hist = trained["history"]
    n_ep = len(hist["episode"])
    last_frac = max(1, n_ep // 4)  # last quarter of training
    late_critic_loss = np.nanmean(hist["critic_loss"][-last_frac:])
    late_cf_std = cf_df["cross_asset_std"].tail(last_frac).mean()
    late_ratio = ep_df["cross_asset_to_temporal_ratio"].tail(last_frac).mean()
    early_cf_std = cf_df["cross_asset_std"].head(last_frac).mean()

    print("\n" + "=" * 78)
    print("ADVANTAGE / CRITIC-DIFFERENTIATION DIAGNOSIS")
    print("=" * 78)
    print(f"n_episodes trained: {n_ep}")
    print(f"late critic_loss (mean, last {last_frac} eps): {late_critic_loss:.6f}")
    print()
    print("Critic counterfactual probe (does the critic predict a different")
    print("value depending on which single asset is concentrated in?):")
    print(f"  cross_asset_std, first {last_frac} eps : {early_cf_std:.6f}")
    print(f"  cross_asset_std, last {last_frac} eps  : {late_cf_std:.6f}")
    print()
    print("Per-asset actor-gradient-term diagnostic:")
    print(f"  cross_asset_to_temporal_ratio, last {last_frac} eps (mean): {late_ratio:.4f}")
    print("  (>~1: real, exploitable cross-asset signal exists in the actor's own")
    print("   gradient term; <<1: the per-step cross-asset spread is small relative")
    print("   to how much the scalar advantage swings over time -- i.e. mostly noise)")
    print()
    if late_cf_std < 1e-3:
        print("READING: the critic predicts nearly IDENTICAL portfolio value regardless")
        print("of which single asset is concentrated in, even late in training. This is")
        print("direct evidence the critic has not learned (or the reward signal does not")
        print("contain) a meaningful cross-asset differentiation -- a near-uniform actor")
        print("policy is the rational response, and the fix belongs on the reward/critic")
        print("side (features, reward shaping, more training data), not the entropy/")
        print("actor-optimisation side.")
    else:
        print("READING: the critic DOES predict meaningfully different values depending")
        print("on which asset is concentrated in. If the actor's policy is still ~uniform")
        print("despite this, the bottleneck is on the actor side (entropy control,")
        print("transaction-cost penalty, or REINFORCE-variance in dlogp credit assignment)")
        print("rather than a lack of critic signal.")
    print("=" * 78)

    # ---- gradient-attribution reading -----------------------------------
    late_action_frac = ga_ep["action_grad_fraction"].tail(last_frac).mean()
    early_action_frac = ga_ep["action_grad_fraction"].head(last_frac).mean()

    # the naive "no preference" baseline: if gradient were spread uniformly
    # across input DIMENSIONS regardless of their role, the action_w slice
    # (n_assets dims) would get this fraction of the total gradient norm.
    # Compare against this, not a fixed cutoff -- a fixed threshold like 0.05
    # is meaningless without knowing how many dimensions the action even is.
    n_assets_ = trained["n_assets"]
    state_dim_ = trained["critic"].encoder.state_dim
    naive_baseline = n_assets_ / (state_dim_ + n_assets_)

    print()
    print("=" * 78)
    print("GRADIENT-ATTRIBUTION DIAGNOSIS (critic_h1's input: [state | action_w])")
    print("=" * 78)
    print("What fraction of the critic's OWN training gradient (from the TD/")
    print("quantile-Huber loss) is attributable to the action_w input, vs. the state")
    print("encoding, at critic_h1?")
    print(f"  state_dim={state_dim_}, n_assets={n_assets_}, "
          f"naive dimension-count baseline = {naive_baseline:.4f}")
    print(f"  action_grad_fraction, first {last_frac} eps (mean): {early_action_frac:.4f}")
    print(f"  action_grad_fraction, last {last_frac} eps  (mean): {late_action_frac:.4f}")
    print("  (the naive baseline is what you'd see if gradient were spread evenly per")
    print("  input DIMENSION with no preference for state vs action; it is NOT 50/50,")
    print("  since action_w is a small slice of the total input. Compare the observed")
    print("  fraction to THIS baseline, not to 0.5 or any fixed cutoff.)")
    print()
    ratio_to_baseline = late_action_frac / (naive_baseline + 1e-12)
    if ratio_to_baseline < 0.5:
        print(f"READING: the action input gets {ratio_to_baseline:.2f}x its naive dimension-")
        print("count share of the critic's own training gradient late in training -- i.e.")
        print("gradient is flowing INTO the action pathway at a lower rate than sheer input")
        print("size would predict. This is consistent with the critic's learning dynamics")
        print("structurally underweighting the action relative to the state encoding.")
    elif ratio_to_baseline > 2.0:
        print(f"READING: the action input gets {ratio_to_baseline:.2f}x its naive dimension-")
        print("count share of gradient late in training -- MORE than proportional. The")
        print("critic's training signal is not starving the action pathway; if anything")
        print("it's giving it disproportionate attention relative to its size. This points")
        print("AWAY from 'the critic structurally ignores the action' as the explanation --")
        print("the forward-pass counterfactual-probe result (near-zero cross_asset_std)")
        print("is more likely explained by what the critic has LEARNED (e.g. converged to")
        print("a near-flat function of the action) than by the gradient never reaching it.")
    else:
        print(f"READING: the action input gets roughly its proportional dimension-count")
        print(f"share of gradient ({ratio_to_baseline:.2f}x baseline) -- neither starved nor")
        print("favoured. The near-zero counterfactual differentiation is more likely a")
        print("property of what the critic has converged to than a sign gradient never")
        print("reaches the action pathway at all.")
    print("=" * 78)

    with open(f"{out_dir}/diagnosis_summary.json", "w") as f:
        json.dump({
            "n_episodes": n_ep,
            "late_critic_loss": float(late_critic_loss),
            "early_cross_asset_std": float(early_cf_std),
            "late_cross_asset_std": float(late_cf_std),
            "late_cross_asset_to_temporal_ratio": float(late_ratio),
            "early_action_grad_fraction": float(early_action_frac),
            "late_action_grad_fraction": float(late_action_frac),
            "naive_dimension_count_baseline": float(naive_baseline),
            "late_action_grad_ratio_to_baseline": float(ratio_to_baseline),
        }, f, indent=2)

    print(f"\nSaved: {out_dir}/per_asset_advantage_log.csv, "
          f"per_asset_advantage_by_episode.csv, critic_counterfactual_by_episode.csv, "
          f"gradient_attribution_log.csv, gradient_attribution_by_episode.csv, "
          f"diagnosis_summary.json")


if __name__ == "__main__":
    main()
