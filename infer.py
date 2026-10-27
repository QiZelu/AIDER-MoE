#!/usr/bin/env python3
"""Predict one known task with the FiLM Transformer baseline."""

import argparse
import os
import sys
import json

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/model.yaml')
    parser.add_argument('--checkpoint')
    parser.add_argument('--video', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument(
        '--task', choices=('static', 'temporal', 'dynamic', 'text_video'), required=True
    )
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from utils.config import load_config
    from models import build_aider, TASKS
    from dataset.video_dataset import VideoInputProcessor
    from engine.checkpoint import load_model_checkpoint
    from engine.runtime import move_to_device, autocast_context
    from engine.memory import initialize_process_memory, MemoryWatch, check_memory
    from utils.provenance import selected_source_hashes as source_hashes

    config = load_config(args.config)
    torch.set_num_threads(2)
    initialize_process_memory()
    torch.cuda.set_per_process_memory_fraction(
        14 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    with MemoryWatch() as watcher:
        model = build_aider(config)
        model.source_hashes = source_hashes(config)
        payload = load_model_checkpoint(
            model, args.checkpoint or config['evaluation']['checkpoint'], config
        )
        del payload
        model.cuda().eval()
        sample = VideoInputProcessor(config['data'])(args.video, args.prompt)
        batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k, v in sample.items()}
        batch.update(
            task_id=torch.tensor([TASKS.index(args.task)]),
            label_key=[args.task],
            dimension_text=[config['task']['descriptions'][args.task]],
        )
        device = torch.device('cuda:0')
        with torch.no_grad(), autocast_context(device):
            result = model(move_to_device(batch, device))
        check_memory(watcher.report())
        print(
            json.dumps(
                dict(
                    task=args.task,
                    score=float(result['predictions'][0]),
                    frame_scores=result['frame_scores'][0].float().cpu().tolist(),
                    pools={
                        name: {
                            key: value[0].detach().cpu().tolist() for key, value in route.items()
                        }
                        for name, route in result['routing']['pools'].items()
                    },
                    memory=watcher.report(),
                )
            )
        )


if __name__ == '__main__':
    main()
