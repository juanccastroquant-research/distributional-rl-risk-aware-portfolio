"""
run_ablation_lambda_tc0_real.py

Same ablation as run_ablation_lambda_tc0.py, but wired directly to your
actual train.train() / evaluate_policy() / performance_summary() functions
instead of the earlier placeholder. Runs three configs (baseline, lambda_TC=0,
lambda_TC=0 + lower beta_min), evaluates each on both test segments, and
prints/saves a side-by-side comparison.

Uses market_sim's synthetic generator (via config_synthetic.yaml) so it runs
without internet access -- rerun with your real config.yaml on a machine with
internet access for the real-data answer.
"""
import copy
import json
import os
import sys
import yaml
import numpy as np
import pandas as pd

from train import train, evaluate_policy
from metrics import performance_summary

ABLATIONS = {
    "baseline": {},
    "lambda_tc_0": {"costs": {"lambda_tc": 0.0}},
    "lambda_tc_0_lower_beta_min": {"costs": {"lambda_tc": 0.0}, "actor": {"beta_min": 0.001}},
}


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def main():
    base_cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_synthetic.yaml"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "ablation_real_out"
    only = sys.argv[3] if len(sys.argv) > 3 else None  # run just one ablation by name
    os.makedirs(out_dir, exist_ok=True)

    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    ablations = {only: ABLATIONS[only]} if only else ABLATIONS

    rows = []
    for name, overrides in ablations.items():
        cfg = copy.deepcopy(base_cfg)
        deep_update(cfg, overrides)
        run_dir = f"{out_dir}/{name}"
        os.makedirs(run_dir, exist_ok=True)
        with open(f"{run_dir}/config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"\n{'='*70}\nRUNNING: {name}  (lambda_tc={cfg['costs']['lambda_tc']}, "
              f"beta_min={cfg['actor']['beta_min']})\n{'='*70}")
        trained = train(cfg)
        hist = trained["history"]
        pd.DataFrame(hist).to_csv(f"{run_dir}/training_history.csv", index=False)

        rng = np.random.default_rng(cfg["seed"] + 1)
        for segment in ["stressed_test", "calm_test"]:
            port_ret, turn, wh = evaluate_policy(trained["actor"], trained["env"],
                                                  trained["A_hat_eff"], segment, rng)
            summ = performance_summary(port_ret, turn, wh, es_alpha=cfg["risk"]["es_alpha"])
            rows.append({
                "run": name, "segment": segment,
                "final_avg_max_weight": hist["avg_max_weight"][-1],
                "final_avg_alpha_sum": hist["avg_alpha_sum"][-1],
                "avg_HHI": summ["avg_HHI"],
                "max_single_weight": summ["max_single_weight"],
                "annual_turnover": summ.get("annual_turnover", 0.0),
                "sharpe": summ["sharpe"],
                "ann_return": summ["ann_return"],
            })

    df = pd.DataFrame(rows)
    comp_path = f"{out_dir}/comparison.csv"
    if only and os.path.exists(comp_path):
        # append to (rather than overwrite) results from earlier single-ablation calls
        prev = pd.read_csv(comp_path)
        prev = prev[prev["run"] != only]  # replace this run's rows if rerun
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(comp_path, index=False)
    print("\n" + "=" * 100)
    print("ABLATION COMPARISON: does removing the transaction-cost penalty let the")
    print("policy concentrate away from 1/N?")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nSaved to {out_dir}/comparison.csv")


if __name__ == "__main__":
    main()
