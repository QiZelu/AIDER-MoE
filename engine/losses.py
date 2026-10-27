"""FP32 quality objectives and optional task--expert mutual information."""

import math
import torch
from torch.nn import functional as F


def regression_loss(predictions, targets, beta=0.1):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    if predictions.ndim != 1 or targets.shape != predictions.shape or not predictions.numel():
        raise ValueError('Expected aligned nonempty [B] scores')
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise FloatingPointError('Nonfinite scores or MOS')
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        values = F.smooth_l1_loss(predictions.float(), targets.float(), beta=beta, reduction='none')
    return values.mean(), values


def moe_mi_weight(config):
    """Missing/zero weight keeps the original regression-only update path."""
    value = config.get('training', {}).get('moe_mi_weight', 0.0)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError('training.moe_mi_weight must be finite and non-negative')
    if value and ('head_rank' in config
                  or config.get('model', {}).get('head_policy') == 'frozen_private'):
        raise ValueError('MoE MI requires trainable Gates; frozen-Head training is unsupported')
    return float(value)


def moe_mi_protocol(config):
    """Record the auxiliary objective separately from the architecture contract."""
    weight = moe_mi_weight(config)
    return dict(weight=weight, task_prior='uniform', task_scope='all_registered_tasks',
                probabilities='pre_topk_softmax', layer_reduction='mean') if weight else None


def task_expert_mutual_information(probabilities, task_ids, num_tasks):
    """I(T;E), in nats; task means receive equal weight regardless of row counts.

    Each probability row describes one categorical expert assignment. All tasks
    must be represented; no detach or hard Top-K counts enter this objective.
    """
    if isinstance(num_tasks, bool) or not isinstance(num_tasks, int) or num_tasks < 1:
        raise ValueError('num_tasks must be a positive integer')
    if (probabilities.ndim != 2 or not probabilities.shape[0]
            or not probabilities.shape[1] or not probabilities.is_floating_point()):
        raise ValueError('Expected nonempty floating expert probabilities [B,E]')
    if (task_ids.dtype != torch.int64 or task_ids.shape != (len(probabilities),)
            or task_ids.device != probabilities.device
            or not bool(((task_ids >= 0) & (task_ids < num_tasks)).all())):
        raise ValueError('Expected aligned int64 task IDs in [0,num_tasks)')
    with torch.autocast(device_type=probabilities.device.type, enabled=False):
        p = probabilities.float()
        if not torch.isfinite(p).all():
            raise FloatingPointError('Nonfinite expert probabilities')
        if (p < 0).any() or not torch.allclose(
            p.sum(-1), torch.ones(len(p), device=p.device), rtol=1e-5, atol=1e-6
        ):
            raise ValueError('Expert probabilities must be non-negative and sum to one')
        if bool((torch.bincount(task_ids, minlength=num_tasks) == 0).any()):
            raise ValueError('Task--expert MI requires probabilities for every task')
        conditional = torch.stack([p[task_ids == t].mean(0) for t in range(num_tasks)])
        marginal = conditional.mean(0, keepdim=True)
        eps = 1e-8
        return (conditional * (conditional.clamp_min(eps).log()
                               - marginal.clamp_min(eps).log())).sum(-1).mean()


def task_router_mi_loss(model):
    """Negative MI averaged over routed pools, using one Gate query per task.

    Current Gates depend only on fixed task embeddings. Querying all tasks is
    exact and independent of video-batch composition, including tail batches.
    It evaluates no FFN expert and gives auxiliary gradients only to the Gates.
    Always-on shared experts are outside these categorical distributions.
    """
    if getattr(model, 'frozen_private_heads', False):
        raise ValueError('MoE MI requires trainable Gates')
    count = len(model.task_order)
    ids = torch.arange(count, device=next(model.parameters()).device)
    condition = model.dimension_encoder(ids)
    losses, per_layer = [], {}
    for name, pool in model.pools():
        route = pool.route(condition, task_ids=ids)
        mi = task_expert_mutual_information(route['probabilities'], ids, count)
        losses.append(-mi)
        per_layer[name] = float(mi.detach())
    if not losses:
        raise ValueError('MoE MI requires at least one routed expert pool')
    return torch.stack(losses).mean(), per_layer


def quality_training_loss(model, result, batch, config):
    """Keep Smooth L1 sample statistics; optionally add -lambda * mean_layer MI."""
    regression, samples = regression_loss(result['predictions'], batch['target'], 0.1)
    weight = moe_mi_weight(config)
    if not weight:
        return regression, samples, {}
    auxiliary, per_layer = task_router_mi_loss(model)
    total = regression + weight * auxiliary
    return total, samples, dict(
        regression_loss=float(regression.detach()),
        mi_loss=float(auxiliary.detach()), mutual_information=float(-auxiliary.detach()),
        weighted_mi_loss=float((weight * auxiliary).detach()), moe_mi_weight=weight,
        total_loss=float(total.detach()), mi_by_layer=per_layer,
    )


def private_head_l2(heads, task_ids, coefficient=1e-5):
    """按样本比例惩罚活跃 Head 的 Linear 权重，不触碰缺席 Head 或 bias。"""
    from models.registry import TASKS
    from models.moe import validate_task_ids

    validate_task_ids(task_ids, len(task_ids))
    if not len(task_ids) or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Invalid L2 coefficient or empty batch')
    with torch.autocast(device_type=task_ids.device.type, enabled=False):
        weighted_square = torch.zeros((), device=task_ids.device, dtype=torch.float32)
        per_head = {}
        for index, task in enumerate(TASKS):
            count = int((task_ids == index).sum())
            if count:
                square = sum(m.weight.float().square().sum() for m in heads[task].modules()
                             if isinstance(m, torch.nn.Linear))
                weighted_square = weighted_square + count / len(task_ids) * square
                per_head[task] = square.detach()
        return coefficient / 2 * weighted_square, weighted_square, per_head


def homogeneous_rank_loss(predictions, targets, task_ids):
    """同任务内全部非同分 i<j 对，sign-only logistic，空对集返回可微零。"""
    if (predictions.ndim != 1 or targets.shape != predictions.shape
            or task_ids.shape != predictions.shape or not predictions.numel()
            or torch.unique(task_ids).numel() != 1):
        raise ValueError('Ranking requires aligned nonempty single-task [B] scores')
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise FloatingPointError('Nonfinite ranking scores')
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        p, y = predictions.float(), targets.float()
        i, j = torch.triu_indices(len(p), len(p), 1, device=p.device)
        difference = y[i] - y[j]
        valid = difference != 0
        signed = difference[valid].sign() * (p[i[valid]] - p[j[valid]])
        count = signed.numel()
        loss = F.softplus(-signed).mean() if count else p.sum() * 0.
        return loss, dict(pair_count=count,
                         reversed_pairs=int((signed < 0).sum()),
                         tied_prediction_pairs=int((signed == 0).sum()))


def t2vqa_rank_loss(predictions, targets, task_ids):
    """同任务零 margin ReLU 排序，严格保留 T2VQA 的双向平均及可导缩放。"""
    if (predictions.ndim != 1 or targets.shape != predictions.shape
            or task_ids.shape != predictions.shape or not predictions.numel()
            or torch.unique(task_ids).numel() != 1):
        raise ValueError('Ranking requires aligned nonempty single-task [B] scores')
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise FloatingPointError('Nonfinite ranking scores')
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        p, y = predictions.float(), targets.float()
        signed_matrix = (p[:, None] - p[None, :]) * (y[:, None] - y[None, :]).sign()
        violations = F.relu(-signed_matrix)
        maximum = violations.max()
        denominator = len(p) * (len(p) - 1)
        raw = violations.sum() / denominator if denominator else p.sum() * 0.
        loss = raw / (1 + maximum)  # 不 detach；与参考实现的梯度相同。
        i, j = torch.triu_indices(len(p), len(p), 1, device=p.device)
        valid = y[i] != y[j]
        signed = signed_matrix[i[valid], j[valid]]
        count = signed.numel()
        reversed_pairs = int((signed < 0).sum())
        tied_pairs = int((signed == 0).sum())
        return loss, dict(
            pair_count=count, reversed_pairs=reversed_pairs,
            tied_prediction_pairs=tied_pairs, ordered_pair_denominator=denominator,
            rank_raw_mean=float(raw.detach()), rank_max_violation=float(maximum.detach()),
            rank_scale=float((1 + maximum).detach()),
            reversed_pair_ratio=reversed_pairs / count if count else None,
            tied_prediction_pair_ratio=tied_pairs / count if count else None)
