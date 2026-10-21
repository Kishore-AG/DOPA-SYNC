"""Run training, paired evaluation, policy export, and figures."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .train import load_config, train
from .evaluate import evaluate
from .export_policy import export_policy


def main():
    cfg = load_config()
    agent, training = train(cfg)
    _, summary = evaluate(cfg, agent)
    export_policy(agent, cfg)
    out = Path("results"); out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax[0,0].plot(training.episode, training["return"], alpha=.35, label="episode")
    ax[0,0].plot(training.episode, training["return"].rolling(20, min_periods=1).mean(), label="20-episode mean")
    ax[0,0].set(title="Training return", xlabel="Episode", ylabel="Reward (queue burden score)"); ax[0,0].legend()
    ax[0,1].plot(training.episode, training.mean_wait_seconds_per_completed_vehicle.rolling(20,min_periods=1).mean())
    ax[0,1].set(title="Mean completed-vehicle wait", xlabel="Episode", ylabel="Seconds / vehicle")
    ax[1,0].plot(training.episode, training.mean_queue_vehicles.rolling(20,min_periods=1).mean())
    ax[1,0].set(title="Mean queue length", xlabel="Episode", ylabel="Vehicles")
    ax[1,1].hist(agent.td_errors, bins=40, color="#527a9b"); ax[1,1].set(title="Training TD-error distribution", xlabel="TD error", ylabel="Transitions")
    fig.savefig(out / "training_diagnostics.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    metrics = ["mean_wait_seconds_per_completed_vehicle", "mean_queue_vehicles", "completed_vehicles", "episode_return", "switches"]
    labels = ["Wait (s/vehicle)", "Mean queue (vehicles)", "Completed (vehicles)", "Return", "Switches"]
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes.flat[i]
        for j, c in enumerate(("rl", "fixed")):
            item = summary["results"][c][metric]
            ax.bar(j, item["mean"], yerr=item["std"], capsize=4,
                   color="#4b9e91" if c == "rl" else "#8c9bae", label="Trained Q-learning" if c == "rl" else "Fixed time")
        ax.set_xticks([0,1], ["Trained Q-learning", "Fixed time"], rotation=12, ha="right")
        ax.set_title(label); ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Held-out paired scenarios: mean ± sample SD")
    fig.savefig(out / "evaluation_comparison.png", dpi=150); plt.close(fig)
    print(json.dumps({"training_episodes": len(training), "q_table_changed": bool(np.any(agent.q != 0)),
                      "evaluation_scenarios": cfg["evaluation"]["scenarios"],
                      "artifacts": ["artifacts/q_table.npy", "artifacts/training_metadata.json", "artifacts/policy.json"],
                      "results": ["results/training_episodes.csv", "results/evaluation_scenarios.csv", "results/experiment_summary.json", "results/training_diagnostics.png", "results/evaluation_comparison.png"]}, indent=2))


if __name__ == "__main__": main()
