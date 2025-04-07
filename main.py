import yaml
import argparse
from train.train_pipeline import run_experiment

def main():
    parser = argparse.ArgumentParser(description='LogTraceGuard Training and Evaluation')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help='Mode to run the program in')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if args.mode == 'train':
        print("Starting training...")
        metrics = run_experiment(args.config)
        print("Training completed with metrics:", metrics)
    else:
        print("Starting evaluation...")
        metrics = run_experiment(args.config)
        print("Evaluation completed with metrics:", metrics)

if __name__ == '__main__':
    main()
