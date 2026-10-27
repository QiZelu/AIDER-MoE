"""Independent original 461/115 split fit; preflight updates are discarded."""

import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from engine.optim import build_optimizer
from engine.gradients import prepare_gradients

from dataset.loader import ActiveDataset
from engine.audit import digest, optimizer_snapshot
from engine.io import write_json
from engine.losses import quality_training_loss, moe_mi_weight
from engine.memory import check_memory
from engine.runtime import autocast_context, move_to_device
from engine.sampling import MixedPairSampler

from engine.checkpoint import save_checkpoint, load_model_checkpoint
from utils.experiment import ROOT, validate_data


def check_precision(model, optimizer=None):
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if optimizer is not None and trainable != {
        id(p) for group in optimizer.param_groups for p in group['params']
    }:
        raise AssertionError('Optimizer coverage of trainable parameters is incomplete')
    if getattr(model, 'frozen_private_heads', False):
        if any(p.requires_grad != name.startswith('fusion.heads.') for name, p in model.named_parameters()):
            raise AssertionError('Only private heads may be trainable')
    elif not all(p.requires_grad for p in model.tv_backbone.model.text_encoder.parameters()):
        raise AssertionError('BLIP text must remain trainable')
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            raise AssertionError('Non-FP32 trainable parameter: ' + name)
        if optimizer is not None:
            state = optimizer.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state and state[key].dtype != torch.float32:
                    raise AssertionError('Non-FP32 Adam state: ' + name)


def frozen_digest(model):
    state = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    state.update(dict(model.named_buffers()))
    return digest(state)


def _selected(route):
    return set(route['topk_indices'].detach().flatten().tolist())


def _verify_split_dataset(config):
    validate_data(config)
    train, val = ActiveDataset(config, 'train'), ActiveDataset(config, 'val')
    counts = (32618, 676, 163090, 3380) if 'videofeedback' in config else (461, 115, 1844, 460)
    if 'eduaigv' in config:
        counts = (900, 230, 1800, 460)
    if (train.video_count, val.video_count, len(train), len(val)) != counts:
        raise AssertionError('Split record counts do not match selected protocol')
    training_ids = {row['id'] for row in train.tasks[0].records}
    testing_ids = {row['id'] for row in val.tasks[0].records}
    if training_ids & testing_ids or len(training_ids | testing_ids) != sum(counts[:2]):
        raise AssertionError('Video leakage or missing split videos')
    return train, val


def preflight(model, config, output, watcher):
    mi_enabled = moe_mi_weight(config) > 0
    task_count = len(config['task']['order'])
    assert len(model.fusion.blocks) == 2 and len(list(model.pools())) == 2
    assert len(list(model.heads())) == (task_count if model.private_heads else 1)
    assert all(pool.num_experts == 7 and pool.top_k == 2 for _, pool in model.pools())
    assert len(model.video_summaries) == (0 if model.film_video_condition == 'cross_branch_linear' else 3)
    assert len(
        {p.data_ptr() for m in model.video_summaries.values() for p in m.parameters()}
    ) == sum(len(list(m.parameters())) for m in model.video_summaries.values())
    assert all(hasattr(block, 'shared_expert') for block in model.fusion.blocks)
    assert all(
        torch.count_nonzero(block.shared_expert[-1].weight) == 0 for block in model.fusion.blocks
    )
    check_precision(model)
    train_set, val_set = _verify_split_dataset(config)
    sampler = MixedPairSampler(len(train_set), config['loader']['sampler_seed'], 8)
    orders = []
    for epoch in range(10):
        before = torch.get_rng_state().clone()
        sampler.set_epoch(epoch)
        batches = sampler.batches()
        if (
            len(batches) != config['ablation']['steps_per_epoch']
            or any(len(batch) != 8 for batch in batches[:-1])
            or len(batches[-1]) != (len(train_set) % 8 or 8)
        ):
            raise AssertionError('Unexpected batch count or tail size')
        if sorted(index for batch in batches for index in batch) != list(range(len(train_set))):
            raise AssertionError('Sampler did not cover all records')
        if not torch.equal(before, torch.get_rng_state()):
            raise AssertionError('CPU sampler changed model RNG')
        orders.append(sampler.digest())
    reference = json.loads((ROOT / 'artifacts/reference/verification.json').read_text())
    if not any(k in config for k in ('videofeedback','eduaigv')) and orders != reference['sampling_sha256']:
        raise AssertionError('Sampler differs from the verified baseline')
    if 'private_task_modules' in config:
        from utils.private_task_modules import CONTROL
        if orders != json.loads((CONTROL/'preflight.json').read_text())['sampling_sha256']:
            raise AssertionError('Private-task sampling differs from direct control')
    model.sampling_sha256 = orders
    if 'repeat' in config.get('eduaigv', {}):
        from utils.eduaigv_splits import REFERENCE
        if orders != json.loads((REFERENCE / 'preflight.json').read_text())['sampling_sha256']:
            raise AssertionError('Repeated EduAIGV sampling differs from historical seed42')
    batch_ids = [
        [task * 461 + step * 8 + task * 2 + j for task in range(4) for j in range(2)]
        for step in range(2)
    ]
    if 'videofeedback' in config:
        batch_ids = [[task * train_set.video_count + step * 8 + task * 2 + j
                      for task, count in enumerate(quota) for j in range(count)]
                     for step, quota in enumerate(((2, 2, 2, 1, 1), (1, 1, 2, 2, 2)))]
    if 'eduaigv' in config:
        batch_ids = [[task * train_set.video_count + step * 8 + j
                      for task in range(2) for j in range(4)] for step in range(2)]
    loader = DataLoader(
        train_set,
        batch_sampler=batch_ids,
        num_workers=0,
        generator=torch.Generator().manual_seed(9042),
    )
    optimizer = build_optimizer(model, config)
    before_frozen = frozen_digest(model)
    initial = digest(model.compact_state_dict())
    device = next(model.parameters()).device
    reports = []
    for step, cpu in enumerate(loader):
        model.train(True)
        batch = move_to_device(cpu, device)
        optimizer.zero_grad(set_to_none=True)
        previous_experts = {
            name: [optimizer_snapshot(expert, optimizer) for expert in pool.experts]
            for name, pool in model.pools()
        }
        previous_always = [
            digest(block.shared_expert.state_dict()) for block in model.fusion.blocks
        ]
        previous_heads = {name: digest(head.state_dict()) for name, head in model.heads()}
        previous_private = {}
        if getattr(model, 'private_task_modules', False):
            from models.task_private import private_module_items
            previous_private = {name: optimizer_snapshot(module, optimizer)
                                for name, _, module in private_module_items(model)}
        module_before = {
            name: digest(module.state_dict())
            for name, module in model.named_children()
            if any(p.requires_grad for p in module.parameters())
        }
        with autocast_context(device):
            result = model(batch)
        if set(batch['task_id'].tolist()) != set(range(task_count)):
            raise AssertionError('Preflight batch lacks a task')
        torch.testing.assert_close(
            result['predictions'], result['frame_scores'].mean(1), atol=0, rtol=0
        )
        loss, _, loss_record = quality_training_loss(model, result, batch, config)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite preflight loss')
        loss.backward()
        for name, head in model.heads():
            if not any(p.grad is not None and torch.count_nonzero(p.grad) > 0
                       for p in head.parameters()):
                raise AssertionError('Active head has no effective gradient: ' + name)
        if model.film_video_condition == 'cross_branch_linear':
            from models.registry import ATTRIBUTES
            for attr in ATTRIBUTES:
                linear = getattr(model, attr).adaptive
                if not isinstance(linear, torch.nn.Linear) or not any(
                    p.grad is not None and torch.count_nonzero(p.grad) > 0
                    for p in linear.parameters()
                ):
                    raise AssertionError('Linear video correction has no effective gradient: ' + attr)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite gradient: ' + name)
        for name, pool in model.pools():
            active = _selected(result['routing']['pools'][name])
            torch.testing.assert_close(
                result['routing']['pools'][name]['topk_weights'].sum(-1),
                torch.ones(len(batch['task_id']), device=device),
                atol=1e-6,
                rtol=0,
            )
            if not any(
                p.grad is not None and torch.count_nonzero(p.grad) > 0
                for p in pool.gate.parameters()
            ):
                raise AssertionError('Task Gate has no effective gradient: ' + name)
            for index, expert in enumerate(pool.experts):
                if index not in active and any(p.grad is not None for p in expert.parameters()):
                    raise AssertionError('Unselected routed expert received a gradient')
        for block in model.fusion.blocks:
            if not any(
                p.grad is not None and torch.count_nonzero(p.grad) > 0
                for p in block.shared_expert[-1].parameters()
            ):
                raise AssertionError('Always-on FFN did not receive a gradient')
        # Adaptive FiLM output Linear is initialized to zero. Its first update
        # opens the gradient path to the independent video-summary MLPs.
        if step >= 1:
            for module in model.video_summaries.values():
                if not any(
                    p.grad is not None and torch.count_nonzero(p.grad) > 0
                    for p in module.parameters()
                ):
                    raise AssertionError(
                        'Cross-branch video condition has no gradient after opening'
                    )
        gradient_record = prepare_gradients(model, config, f'preflight step={step+1}')
        if getattr(model, 'private_task_modules', False):
            from engine.private_task_diagnostics import check_private_activity, module_gradient_norms
            private_activity = check_private_activity(
                model, batch['task_id'], all_task_gates=mi_enabled
            )
            gradient_record['module_norms'] = module_gradient_norms(model)
        optimizer.step()
        check_precision(model, optimizer)
        private_updates = {}
        if previous_private:
            for name, task, module in private_module_items(model):
                active = private_activity[name]
                updated = digest(module.state_dict()) != previous_private[name]['parameters']
                private_updates[name] = updated
                if not active and optimizer_snapshot(module,optimizer) != previous_private[name]:
                    raise AssertionError('Absent task private parameter/Adam state changed: ' + name)
                if active and not updated and not (step == 0 and name.startswith('alpha_gate.')):
                    raise AssertionError('Active private module failed to update: ' + name)
        head_updates = {name: digest(head.state_dict()) != previous_heads[name]
                        for name, head in model.heads()}
        if not all(head_updates.values()):
            raise AssertionError('Active head failed to update: ' + str(head_updates))
        module_updates = {
            name: digest(module.state_dict()) != module_before[name]
            for name, module in model.named_children()
            if name in module_before
        }
        if step >= 1 and not all(module_updates.values()):
            raise AssertionError('Trainable module failed to update: ' + str(module_updates))
        for i, block in enumerate(model.fusion.blocks):
            if digest(block.shared_expert.state_dict()) == previous_always[i]:
                raise AssertionError('Always-on FFN did not update')
        for name, pool in model.pools():
            active = _selected(result['routing']['pools'][name])
            for index, expert in enumerate(pool.experts):
                if (
                    index not in active
                    and optimizer_snapshot(expert, optimizer) != previous_experts[name][index]
                ):
                    raise AssertionError('Unselected expert parameter/Adam state changed')
                if (
                    index in active
                    and digest(expert.state_dict()) == previous_experts[name][index]['parameters']
                ):
                    raise AssertionError('Active expert did not update')
        if frozen_digest(model) != before_frozen:
            raise AssertionError('Frozen visual backbone or fixed buffer changed')
        check_memory(watcher.report())
        reports.append(
            dict(
                step=step + 1,
                loss=float(loss.detach()),
                **loss_record,
                always_on_updated=True,
                inactive_expert_unchanged=True,
                trainable_module_updates=module_updates,
                head_updates=head_updates,
                gradient_diagnostics=gradient_record,
                private_module_updates=private_updates,
            )
        )
        del result, loss
    model.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        checkpoint = Path(temporary) / 'preflight.pt'
        save_checkpoint(checkpoint, model, config, 0)
        reload_ids = list(range(8))
        if 'private_task_modules' in config:
            reload_ids = [t * val_set.video_count + j for t in range(4) for j in range(2)]
        if 'videofeedback' in config:
            reload_ids = [t * val_set.video_count + j
                          for t, count in enumerate((2, 2, 2, 1, 1)) for j in range(count)]
        if 'eduaigv' in config:
            reload_ids = [t * val_set.video_count + j for t in range(2) for j in range(4)]
        sample = move_to_device(
            next(iter(DataLoader(val_set, batch_sampler=[reload_ids], num_workers=0))), device
        )
        model.eval()
        with torch.no_grad(), autocast_context(device):
            expected = model(sample)['predictions'].detach().float().clone()
        with torch.no_grad():
            next(model.fusion.heads.parameters()).add_(1.0)
        load_model_checkpoint(model, checkpoint, config)
        with torch.no_grad(), autocast_context(device):
            actual = model(sample)['predictions'].detach().float()
        delta = float((actual - expected).abs().max())
        if delta > 1e-6:
            raise AssertionError('Checkpoint reload prediction drift')
    write_json(
        output / 'preflight.json',
        dict(
            passed=True,
            initial_state_sha256=initial,
            sampling_sha256=orders,
            steps=reports,
            checkpoint_reload_max_abs_delta=delta,
            peak_memory=watcher.report(),
        ),
    )
    return initial, orders
