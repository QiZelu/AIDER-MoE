"""固定已训练表征，只拟合任务私有 Head；来源仅在训练初始化时读取。"""

import gc
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.loader import build_loader
from engine.audit import digest, file_digest, parameter_counts
from engine.checkpoint import FROZEN_PRETRAINED, save_checkpoint, load_model_checkpoint
from engine.evaluation import evaluate, export
from engine.io import write_json, write_csv
from engine.losses import regression_loss, private_head_l2
from engine.memory import initialize_process_memory, MemoryWatch, check_memory
from engine.optim import build_optimizer
from engine.preflight import check_precision, frozen_digest, _verify_split_dataset
from engine.reporting import gap_diagnostics
from engine.routing import RouteCounts, task_routes, export_encoding
from engine.runtime import autocast_context, move_to_device
from engine.sampling import MixedPairSampler
from models.aider import AIDER
from models.registry import TASKS
from utils.config import seed_everything
from utils.experiment import ROOT, VIDEO_ONLY_CONTRACT, validate_data
from utils.provenance import selected_source_hashes


def validate_training(config):
    t = config['training']
    if (config['model'].get('head_policy') != 'frozen_private'
            or not math.isfinite(t['lr']) or t['lr'] <= 0
            or t['weight_decay'] != 0 or t['module_lrs']
            or t['head_l2_lambda'] != 1e-5 or t['scheduler'] != 'constant'
            or t['warmup_steps'] != 0 or t['epochs'] != 10 or t['resume'] is not None):
        raise ValueError('Expected frozen private heads, finite positive constant LR, explicit L2=1e-5 and no decay')


def optimizer_lr(optimizer, config):
    """核验恒定学习率，日志记录优化器中的实际值。"""
    expected = config['training']['lr']
    if not optimizer.param_groups or any(
            g['lr'] != expected or g['weight_decay'] != 0 for g in optimizer.param_groups):
        raise AssertionError('Head optimizer LR or weight decay differs from config')
    return optimizer.param_groups[0]['lr']


def convert_head_state(state):
    """只复制已训练共享 Head，其他 tensor 不改变。"""
    prefix = 'fusion.heads.shared.'
    if not any(k.startswith(prefix) for k in state):
        raise ValueError('Source has no shared head')
    result = {}
    for key, value in state.items():
        if key.startswith(prefix):
            for task in TASKS:
                result['fusion.heads.' + task + '.' + key[len(prefix):]] = value.clone()
        else:
            result[key] = value
    return result


def load_trained_source(model, config, sources):
    """局部授权的精确来源加载；不影响通用 checkpoint 的严格校验。"""
    source = config['frozen_head_source']
    path = Path(source['checkpoint'])
    if file_digest(path) != source['sha256']:
        raise ValueError('Pinned source checkpoint hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    expected_model = dict(config['model'])
    expected_model.pop('head_policy')
    if (payload['model_contract'] != VIDEO_ONLY_CONTRACT
            or payload['epoch'] + 1 != source['epoch']
            or payload.get('film_video_condition') != 'video_only'
            or payload['split'] != config['ablation']['split']
            or payload['config']['model'] != expected_model
            or payload['config']['task'] != config['task']
            or payload['config']['data'] != config['data']
            or payload['task_registry'] != model.task
            or payload['task_embedding_hashes'] != model.task_embedding_hashes):
        raise ValueError('Source architecture, task table or dataset mismatch')
    assets = lambda table: {k: v['sha256'] for k, v in table.items()
                            if not k.startswith('implementation/')}
    if assets(payload['source_hashes']) != assets(sources):
        raise ValueError('Source pretrained/data/fixed assets changed')
    state = convert_head_state(payload['model'])
    expected = model.compact_state_dict()
    if state.keys() != expected.keys():
        raise ValueError('Converted source tensor coverage mismatch')
    for name, value in expected.items():
        if state[name].shape != value.shape or state[name].dtype != value.dtype:
            raise ValueError('Source shape/dtype mismatch: ' + name)
        if (name.startswith('dimension_encoder.') or name.endswith('position_encoding')):
            if not torch.equal(state[name], value.cpu()):
                raise ValueError('Source fixed buffer mismatch: ' + name)
    missing, extra = model.load_state_dict(state, strict=False)
    if extra or any(not k.startswith(FROZEN_PRETRAINED) for k in missing):
        raise ValueError('Unexpected source load coverage')
    model.initialization_audit.update(
        trained_checkpoint_loaded=True, source_best_sha256=source['sha256'],
        source_best_epoch=source['epoch'], source_initial_state_sha256=digest(model.compact_state_dict()),
        head_initialization='four_independent_copies_of_trained_shared_head',
        frozen_representation=True, optimizer_inherited=False,
    )
    return model


def initialize_frozen_model(config, sources):
    validate_training(config)
    seed_everything(config['training']['seed'])
    model = AIDER(config)
    load_trained_source(model, config, sources)
    model.source_hashes = sources
    model.construction_rng_sha256 = digest({'cpu_rng': torch.get_rng_state()})
    model.train(True)
    check_precision(model)
    return model


def assert_frozen_mode(model):
    for name, module in model.named_modules():
        if not name.startswith('fusion.heads') and module.training:
            raise AssertionError('Frozen module entered training mode: ' + name)
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise AssertionError('Frozen parameter received a gradient')


def head_objective(model, result, batch, config):
    regression, samples = regression_loss(result['predictions'], batch['target'], 0.1)
    penalty, square, per_head = private_head_l2(
        model.fusion.heads, batch['task_id'], config['training']['head_l2_lambda'])
    return regression + penalty, samples, dict(
        regression_loss=float(regression.detach()), l2_loss=float(penalty.detach()),
        weighted_head_weight_square=float(square.detach()),
        head_weight_square={k: float(v) for k, v in per_head.items()},
    )


def frozen_preflight(model, config, output, watcher):
    train_set, val_set = _verify_split_dataset(config)
    assert list(model.fusion.heads) == list(TASKS)
    assert parameter_counts(model)['trainable'] == 1643908
    assert len(list(model.pools())) == 2
    sampler = MixedPairSampler(1844, config['loader']['sampler_seed'], 8)
    orders = []
    for epoch in range(10):
        before_rng = torch.get_rng_state().clone()
        sampler.set_epoch(epoch)
        batches = sampler.batches()
        assert len(batches) == 231 and len(batches[-1]) == 4
        assert all(len(b) == 8 for b in batches[:-1])
        assert sorted(i for b in batches for i in b) == list(range(1844))
        assert torch.equal(before_rng, torch.get_rng_state())
        orders.append(sampler.digest())
    reference = json.loads((ROOT / 'artifacts/reference/verification.json').read_text())
    assert orders == reference['sampling_sha256']
    model.sampling_sha256 = orders
    ids = [[t * 461 + step * 8 + t * 2 + j for t in range(4) for j in range(2)]
           for step in range(2)]
    loader = DataLoader(train_set, batch_sampler=ids, num_workers=0,
                        generator=torch.Generator().manual_seed(9042))
    optimizer = build_optimizer(model, config)
    optimizer_lr(optimizer, config)
    check_precision(model, optimizer)
    initial = digest(model.compact_state_dict())
    frozen = frozen_digest(model)
    reports = []
    device = next(model.parameters()).device
    for step, cpu in enumerate(loader):
        model.train(True)
        assert_frozen_mode(model)
        batch = move_to_device(cpu, device)
        optimizer.zero_grad(set_to_none=True)
        before_heads = {t: digest(h.state_dict()) for t, h in model.heads()}
        with autocast_context(device):
            result = model(batch)
        fused = result['routing']['fused_frames'].detach().clone()
        assert not result['routing']['fused_frames'].requires_grad
        if step == 0:
            with torch.no_grad():
                reference_fp32 = model.fusion.heads[TASKS[0]](fused.float()).squeeze(-1).mean(1)
                actual_fp32 = model.score_frames(fused.float(), batch['task_id']).mean(1)
                with autocast_context(device):
                    shared_bf16 = model.fusion.heads[TASKS[0]](fused).squeeze(-1).mean(1)
            fp32_delta = float((actual_fp32 - reference_fp32).abs().max())
            bf16_delta = float((result['predictions'].detach() - shared_bf16).abs().max())
            if max(fp32_delta, bf16_delta) > 1e-6:
                raise AssertionError(f'Private head conversion drift: {fp32_delta}, {bf16_delta}')
        loss, _, info = head_objective(model, result, batch, config)
        loss.backward()
        assert_frozen_mode(model)
        for task, head in model.heads():
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
                raise AssertionError('Invalid active head gradients: ' + task)
        torch.nn.utils.clip_grad_norm_(model.fusion.heads.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        for task, head in model.heads():
            if digest(head.state_dict()) == before_heads[task]:
                raise AssertionError('Head did not update: ' + task)
        check_precision(model, optimizer)
        if frozen_digest(model) != frozen:
            raise AssertionError('Frozen parameters or buffers changed')
        with torch.no_grad(), autocast_context(device):
            after = model(batch)['routing']['fused_frames']
        torch.testing.assert_close(after, fused, atol=0, rtol=0)
        reports.append(dict(step=step + 1, **info, frozen_unchanged=True, features_unchanged=True))
        check_memory(watcher.report())
        del result, loss, fused, after
    model.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        path = Path(temporary) / 'roundtrip.pt'
        save_checkpoint(path, model, config, -1)
        with torch.no_grad(), autocast_context(device):
            expected = model(batch)['predictions'].clone()
        with torch.no_grad():
            next(model.fusion.heads.parameters()).add_(1)
        payload = load_model_checkpoint(model, path, config)
        del payload
        with torch.no_grad(), autocast_context(device):
            actual = model(batch)['predictions']
        reload_delta = float((expected - actual).abs().max())
        assert reload_delta <= 1e-6
    write_json(output / 'preflight.json', dict(
        passed=True, steps=reports, initial_state_sha256=initial, sampling_sha256=orders,
        conversion_fp32_max_abs_delta=fp32_delta, conversion_bf16_max_abs_delta=bf16_delta,
        checkpoint_reload_max_abs_delta=reload_delta, peak_memory=watcher.report()))
    return initial, orders


def score_spread(rows):
    report = {}
    for task in TASKS:
        subset = [r for r in rows if r['task'] == task]
        prediction = np.array([r['prediction'] for r in subset])
        target = np.array([r['target'] for r in subset])
        report[task] = dict(prediction_mean=float(prediction.mean()), target_mean=float(target.mean()),
                            prediction_std=float(prediction.std()), target_std=float(target.std()),
                            std_ratio=float(prediction.std() / target.std()) if target.std() else None)
    return dict(tasks=report, dynamic_tv=gap_diagnostics(rows))


def summarize_heads(output):
    source = json.loads((output / 'source_evaluation.json').read_text())
    differences = []
    for kind in ('best', 'last'):
        current = json.loads((output / f'{kind}_evaluation.json').read_text())
        for task in (*TASKS, 'macro'):
            for metric in ('plcc', 'srocc', 'krocc', 'rmse'):
                x, y = current['metrics'][task][metric], source['metrics'][task][metric]
                differences.append(dict(checkpoint=kind, task=task, metric=metric, current=x,
                                        source_best=y, delta=x-y if x is not None and y is not None else None))
        write_json(output / f'{kind}_spread_diagnostics.json', score_spread(current['samples']))
    write_csv(output / 'source_best_comparison.csv', differences)
    config = yaml.safe_load((output / 'config.yaml').read_text())
    if config.get('head_lr_control'):
        summarize_lr_control(output, config)
    write_json(output / 'summary.json', dict(
        complete=True, source_best_epoch=source['checkpoint_epoch'],
        interpretation=['Trained source best is followed by 2310 head-only updates.',
                        'Private heads increase capacity and isolate updates; L2 also changes.',
                        'Single seed; source and new best use the same held-out selection set.',
                        'No automatic baseline replacement.']))


def summarize_lr_control(output, config):
    """只读已完成的低 LR 对照；同来源、同采样，best/last 分开比较。"""
    control = Path(config['head_lr_control'])
    other = yaml.safe_load((control / 'config.yaml').read_text())
    if json.loads((control / 'status.json').read_text())['stage'] != 'complete':
        write_json(output / 'lr_comparison_status.json', dict(complete=False, control=str(control)))
        return
    for field in ('model', 'task', 'data', 'loader', 'frozen_head_source'):
        if config[field] != other[field]:
            raise ValueError('LR control mismatch: ' + field)
    training = lambda c: {k: v for k, v in c['training'].items() if k not in ('lr', 'output_dir')}
    if training(config) != training(other):
        raise ValueError('LR control has other training changes')
    audits = [json.loads((p / 'preflight.json').read_text()) for p in (output, control)]
    for key in ('initial_state_sha256', 'sampling_sha256'):
        if audits[0][key] != audits[1][key]:
            raise ValueError('LR control audit mismatch: ' + key)
    metrics, spreads = [], {}
    for kind in ('best', 'last'):
        current = json.loads((output / f'{kind}_evaluation.json').read_text())
        reference = json.loads((control / f'{kind}_evaluation.json').read_text())
        labels = lambda e: [(r['video_id'], r['task'], r['target']) for r in e['samples']]
        if sorted(labels(current)) != sorted(labels(reference)):
            raise ValueError('LR comparison evaluation labels mismatch')
        for task in (*TASKS, 'macro'):
            for metric in ('plcc', 'srocc', 'krocc', 'rmse'):
                x, y = current['metrics'][task][metric], reference['metrics'][task][metric]
                metrics.append(dict(checkpoint=kind, task=task, metric=metric,
                                    current_lr=config['training']['lr'], control_lr=other['training']['lr'],
                                    current_epoch=current['checkpoint_epoch'],
                                    control_epoch=reference['checkpoint_epoch'], current=x, control=y,
                                    delta=x-y if x is not None and y is not None else None))
        spreads[kind] = dict(current=score_spread(current['samples']),
                             control=score_spread(reference['samples']))
    write_csv(output / 'lr_control_comparison.csv', metrics)
    write_json(output / 'lr_control_spread_comparison.json', spreads)
    write_json(output / 'lr_comparison_status.json', dict(
        complete=True, control=str(control), paired_initialization_and_sampling=True,
        source_best_sha256=config['frozen_head_source']['sha256']))


def train_frozen_heads(config, mode='train'):
    validate_training(config)
    output = Path(config['training']['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    torch.set_num_threads(config['training']['cpu_threads'])
    watcher = None
    try:
        write_json(output / 'status.json', dict(stage='initializing'))
        initialize_process_memory()
        torch.cuda.set_per_process_memory_fraction(14 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        with MemoryWatch() as watcher:
            write_json(output / 'data_audit.json', validate_data(config))
            sources = selected_source_hashes(config)
            device = torch.device(config['training']['device'])
            model = initialize_frozen_model(config, sources).to(device)
            write_json(output / 'sources.json', sources)
            write_json(output / 'parameters.json', parameter_counts(model))
            write_json(output / 'initialization_audit.json', model.initialization_audit)
            write_json(output / 'status.json', dict(stage='preflight'))
            initial, orders = frozen_preflight(model, config, output, watcher)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if mode == 'preflight':
                write_json(output / 'status.json', dict(stage='preflight_complete'))
                return
            write_json(output / 'status.json', dict(stage='reloading_source_after_preflight'))
            model = initialize_frozen_model(config, sources).to(device)
            assert digest(model.compact_state_dict()) == initial
            model.sampling_sha256 = orders
            frozen = frozen_digest(model)
            export_encoding(model, output)
            write_json(output / 'moe_routing/initial.json', task_routes(model))
            train_loader, val_loader = build_loader(config, 'train'), build_loader(config, 'val')
            write_json(output / 'status.json', dict(stage='evaluating_source_before_training'))
            rows, metrics, routes = evaluate(model, val_loader, device)
            export(output, 'source', rows, metrics, routes, config['frozen_head_source']['epoch'] - 1)
            write_json(output / 'source_spread_diagnostics.json', score_spread(rows))
            del rows, metrics, routes
            optimizer = build_optimizer(model, config)
            history, best = [], -float('inf')
            from engine.trainer import save_epoch_diagnostics

            for epoch in range(10):
                write_json(output / 'status.json', dict(stage='training', epoch=epoch+1, completed_epochs=epoch))
                train_loader.batch_sampler.set_epoch(epoch)
                counters = RouteCounts(model)
                indices, compositions, records = [], [], []
                regression_sum, l2_sum = 0., 0.
                for step, cpu in enumerate(train_loader, 1):
                    model.train(True)
                    assert_frozen_mode(model)
                    batch = move_to_device(cpu, device)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device):
                        result = model(batch)
                    loss, samples, info = head_objective(model, result, batch, config)
                    loss.backward()
                    assert_frozen_mode(model)
                    torch.nn.utils.clip_grad_norm_(model.fusion.heads.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    counters.add(batch['task_id'], result['routing'])
                    indices.append(cpu['pair_index'].tolist())
                    compositions.append([int((batch['task_id'] == i).sum()) for i in range(4)])
                    regression_sum += float(samples.detach().sum())
                    l2_sum += info['l2_loss'] * len(batch['task_id'])
                    record = dict(epoch=epoch+1, step=step, loss=float(loss.detach()),
                                  lr=optimizer_lr(optimizer, config), **info)
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    check_memory(watcher.report())
                    del loss, samples, result
                assert len(indices) == 231 and indices == train_loader.batch_sampler.batches()
                assert train_loader.batch_sampler.digest() == orders[epoch]
                check_precision(model, optimizer)
                if frozen_digest(model) != frozen:
                    raise AssertionError('Frozen representation changed during training')
                optimizer_lr(optimizer, config)
                rows, metrics, routes = evaluate(model, val_loader, device)
                save_epoch_diagnostics(output, epoch, model, dict(
                    route_counts=counters.report(), shared_gradients=[[0., 0.]], pair_indices=indices,
                    batch_task_counts=compositions, order_sha256=orders[epoch]), routes)
                write_json(output / 'loss_diagnostics' / f'epoch_{epoch+1:02d}.json', records)
                history.append(dict(epoch=epoch+1, train_loss=regression_sum/1844,
                                    train_l2_loss=l2_sum/1844, test=dict(metrics=metrics),
                                    optimizer_updates=(epoch+1)*231, frozen_state_sha256=frozen))
                write_json(output / 'history.json', history)
                save_checkpoint(output / 'last.pt', model, config, epoch, optimizer,
                                selection_metric='test_macro_srocc')
                if metrics['macro']['srocc'] is not None and metrics['macro']['srocc'] > best:
                    best = metrics['macro']['srocc']
                    save_checkpoint(output / 'best.pt', model, config, epoch, score=best,
                                    selection_metric='test_macro_srocc')
                write_json(output / 'process_memory.json', watcher.report())
            del optimizer
            for kind in ('best', 'last'):
                payload = load_model_checkpoint(model, output / f'{kind}.pt', config)
                saved_epoch = payload['epoch']
                del payload
                rows, metrics, routes = evaluate(model, val_loader, device)
                export(output, kind, rows, metrics, routes, saved_epoch)
            if file_digest(config['frozen_head_source']['checkpoint']) != config['frozen_head_source']['sha256']:
                raise AssertionError('Source checkpoint changed')
            check_memory(watcher.report())
            write_json(output / 'process_memory.json', watcher.report())
            summarize_heads(output)
            write_json(output / 'status.json', dict(stage='complete', completed_epochs=10, optimizer_updates=2310))
    except Exception as error:
        write_json(output / 'status.json', dict(stage='failed', error=repr(error)))
        if watcher is not None:
            write_json(output / 'process_memory.json', watcher.report())
        raise
