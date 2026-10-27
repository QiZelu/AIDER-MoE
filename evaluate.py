#!/usr/bin/env python3
"""Evaluate one checkpoint on all four dimensions, without touching saved results."""

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/model.yaml')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from utils.config import load_config
    from models import build_aider
    from dataset.loader import build_loader
    from engine.checkpoint import load_model_checkpoint
    from engine.evaluation import evaluate, export
    from engine.memory import initialize_process_memory, MemoryWatch, check_memory
    from engine.io import write_json

    config = load_config(args.config, args.set)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    initialize_process_memory()
    torch.cuda.set_per_process_memory_fraction(
        14 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    with MemoryWatch() as watcher:
        from utils.provenance import selected_source_hashes as source_hashes

        model = build_aider(config)
        model.source_hashes = source_hashes(config)
        payload = load_model_checkpoint(
            model, args.checkpoint or config['evaluation']['checkpoint'], config
        )
        epoch = payload['epoch']
        del payload
        model.cuda()
        rows, scores, routes = evaluate(model, build_loader(config, 'val'), torch.device('cuda:0'))
        export(output, 'evaluation', rows, scores, routes, epoch)
        check_memory(watcher.report())
        write_json(
            output / 'verification.json', dict(checkpoint_loaded=True, memory=watcher.report())
        )
        print(scores)


if __name__ == '__main__':
    main()
