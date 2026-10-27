#!/usr/bin/env python3
"""Train the depth-2 cross-branch FiLM / shared+routed experts baseline."""

import argparse
import os
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/model.yaml')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--mode', choices=('train', 'preflight'), default='train')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from engine.trainer import train
    from utils.config import load_config

    config = load_config(args.config, args.set)
    train(config, mode=args.mode)


if __name__ == '__main__':
    main()
