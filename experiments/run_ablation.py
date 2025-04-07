import os
import yaml
import torch
from datetime import datetime
from train.train_pipeline import train, evaluate
from utils.config import load_config


def run_experiment(cfg_path):
    cfg = load_config(cfg_path)
    tag = os.path.splitext(os.path.basename(cfg_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/ablation_{tag}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"[Ablation] Running {tag} experiment...")
    train(cfg)
    evaluate(cfg)
    print(f"[✓] Experiment complete. See logs at: {log_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to ablation YAML config')
    args = parser.parse_args()

    run_experiment(args.config)
