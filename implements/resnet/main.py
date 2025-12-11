import argparse
from scripts.train import train
from scripts.eval import evaluate
from configs import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'inference'], required=True)
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.mode == 'train':
        train(config)
    elif args.mode == 'eval':
        evaluate(config, args.checkpoint)
    elif args.mode == 'inference':
        # 推理代码
        pass

if __name__ == "__main__":
    main()