"""Unchanged baseline AdamW grouping and cosine scheduler."""

import math
import torch
from typing import Any


def _optimizer(model: torch.nn.Module, config: dict[str, Any]):
    training = config['training']
    learning_rate = float(training.get('lr', 1e-4))
    weight_decay = float(training.get('weight_decay', 0.05))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('training.lr must be finite and positive')
    module_lrs = training.get('module_lrs', {}) or {}
    if not isinstance(module_lrs, dict):
        raise TypeError('training.module_lrs must be a mapping')
    supported_groups = {
        'technical_expert',
        'motion',
        'text_video',
        'fusion',
        'vital_backbone',
        'motion_backbone',
        'text_video_backbone',
    }
    unknown_groups = sorted(set(module_lrs) - supported_groups)
    if unknown_groups:
        raise ValueError(f'Unsupported training.module_lrs groups: {unknown_groups}')
    resolved_lrs = {name: float(value) for name, value in module_lrs.items()}
    if any(not math.isfinite(value) or value <= 0 for value in resolved_lrs.values()):
        raise ValueError('Every training.module_lrs value must be finite and positive')

    def module_group(parameter_name: str) -> str:
        if parameter_name.startswith(('technical_expert.', 'technical_projector.')):
            return 'technical_expert'
        if parameter_name.startswith(('motion_position_embedding.', 'motion_projector.')):
            return 'motion'
        if parameter_name.startswith(('tv_projector.', 'blip_temporal.')):
            return 'text_video'
        if parameter_name.startswith('fusion.'):
            return 'fusion'
        if parameter_name.startswith('vital_backbone.'):
            return 'vital_backbone'
        if parameter_name.startswith('motion_backbone.'):
            return 'motion_backbone'
        if parameter_name.startswith('tv_backbone.'):
            return 'text_video_backbone'
        if parameter_name.startswith(('video_summaries.', 'video_summary.', 'alpha_gate.')):
            return 'video_condition'
        raise ValueError(f'No optimizer module group for trainable parameter {parameter_name!r}')

    groups: dict[tuple[str, float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = (
            parameter.ndim < 2
            or name.endswith('.bias')
            or 'norm' in name.lower()
            or 'embedding' in name
            or name.endswith('.queries')
        )
        decay = 0.0 if no_decay else weight_decay
        group_name = module_group(name) if resolved_lrs else 'default'
        group_lr = resolved_lrs.get(group_name, learning_rate)
        groups.setdefault((group_name, group_lr, decay), []).append(parameter)
    return torch.optim.AdamW(
        [
            {
                'params': parameters,
                'lr': group_lr,
                'weight_decay': decay,
                'group_name': group_name,
            }
            for (group_name, group_lr, decay), parameters in groups.items()
        ],
        betas=tuple(training.get('betas', [0.9, 0.999])),
    )


def _scheduler(
    optimizer,
    total_steps: int,
    warmup_ratio: float,
    warmup_steps: int | None = None,
):
    if warmup_steps is None:
        if not math.isfinite(warmup_ratio) or warmup_ratio < 0:
            raise ValueError('training.warmup_ratio must be finite and non-negative')
        warmup_steps = max(1, int(total_steps * warmup_ratio))
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError('warmup steps must be between zero and total training steps')

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def build_optimizer(model, config):
    return _optimizer(model, config)
