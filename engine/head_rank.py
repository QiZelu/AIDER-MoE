"""继承第9轮四个私有 Head，只在同任务 batch 上微调回归与排序。"""

import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.loader import build_loader
from engine.audit import digest, file_digest, optimizer_snapshot, parameter_counts
from engine.checkpoint import FROZEN_PRETRAINED, load_model_checkpoint, save_checkpoint
from engine.evaluation import evaluate, export
from engine.frozen_heads import assert_frozen_mode, score_spread
from engine.gradients import prepare_gradients
from engine.io import write_csv, write_json
from engine.losses import homogeneous_rank_loss, regression_loss, t2vqa_rank_loss
from engine.memory import MemoryWatch, check_memory, initialize_process_memory
from engine.optim import _scheduler, build_optimizer
from engine.preflight import _verify_split_dataset, check_precision, frozen_digest
from engine.routing import RouteCounts, export_encoding, initial_task_routes, task_routes
from engine.runtime import autocast_context, move_to_device
from engine.sampling import HomogeneousTaskSampler
from models.aider import AIDER
from models.registry import TASKS
from utils.config import seed_everything
from utils.experiment import validate_data
from utils.head_rank import LOGISTIC_KIND, T2VQA_KIND, validate_config
from utils.private_task_modules import CONTRACT as SOURCE_CONTRACT
from utils.provenance import selected_source_hashes


def load_epoch9(model, config, sources):
    """只授权精确哈希的来源；评估本组检查点不调用此函数。"""
    source = config['head_rank']
    path = Path(source['source_checkpoint'])
    if file_digest(path) != source['source_sha256']:
        raise ValueError('Epoch9 source hash mismatch')
    if file_digest(path.parent / 'config.yaml') != source['source_config_sha256']:
        raise ValueError('Epoch9 saved configuration changed')
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if (payload['model_contract'] != SOURCE_CONTRACT or payload['epoch'] != 8
            or payload['split'] != config['ablation']['split']
            or payload.get('private_task_modules') != config['private_task_modules']
            or payload.get('gradient_clip') is not None):
        raise ValueError('Wrong epoch9 source protocol')
    for key in ('model', 'task', 'data'):
        if payload['config'][key] != config[key]:
            raise ValueError('Source configuration mismatch: ' + key)
    if (payload['task_registry'] != model.task
            or payload['task_embedding_hashes'] != model.task_embedding_hashes):
        raise ValueError('Source task table mismatch')
    assets = lambda table: {k: v['sha256'] for k, v in table.items()
                            if not k.startswith('implementation/')}
    if assets(payload['source_hashes']) != assets(sources):
        raise ValueError('Source pretrained/data/fixed assets changed')
    state, expected = payload['model'], model.compact_state_dict()
    if state.keys() != expected.keys():
        raise ValueError('Source state coverage mismatch')
    for name, value in expected.items():
        if state[name].shape != value.shape or state[name].dtype != value.dtype:
            raise ValueError('Source shape/dtype mismatch: ' + name)
        if (name.startswith('dimension_encoder.') or name.endswith('position_encoding')):
            if not torch.equal(state[name], value.cpu()):
                raise ValueError('Source fixed constant mismatch: ' + name)
    missing, extra = model.load_state_dict(state, strict=False)
    if extra or any(not name.startswith(FROZEN_PRETRAINED) for name in missing):
        raise ValueError('Unexpected source load coverage')
    # 冻结文本后 compact_state_dict 把文本键追加到末尾；按同一名称顺序比较，
    # 不将保存顺序的变化误判为权重变化。
    source_digest = digest({name: state[name] for name in expected})
    if digest(model.compact_state_dict()) != source_digest:
        raise AssertionError('Source weights were not inherited exactly')
    model.initialization_audit.update(
        trained_checkpoint_loaded=True, source_epoch=9, source_sha256=source['source_sha256'],
        source_state_sha256=source_digest, head_initialization='inherit_four_trained_private_heads',
        frozen_representation=True, optimizer_inherited=False)


def initialize(config, sources):
    validate_config(config)
    seed_everything(config['training']['seed'])
    model = AIDER(config)
    load_epoch9(model, config, sources)
    model.initial_task_routes = initial_task_routes(model)
    model.source_hashes = sources
    model.train(True)
    assert_frozen_mode(model)
    check_precision(model)
    return model


def objective(result, batch, config=None):
    regression, samples = regression_loss(result['predictions'], batch['target'], .1)
    kind = config['head_rank']['rank_kind'] if config is not None else LOGISTIC_KIND
    functions = {LOGISTIC_KIND: homogeneous_rank_loss, T2VQA_KIND: t2vqa_rank_loss}
    rank, pairs = functions[kind](result['predictions'], batch['target'], batch['task_id'])
    total = regression + rank
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite Head ranking loss')
    return total, samples, dict(regression_loss=float(regression.detach()),
                                rank_loss=float(rank.detach()), **pairs)


def sampling_audit():
    sampler = HomogeneousTaskSampler()
    hashes = []
    for epoch in range(10):
        rng = torch.get_rng_state().clone()
        sampler.set_epoch(epoch)
        batches = sampler.batches()
        assert len(batches) == 232 and sorted(i for b in batches for i in b) == list(range(1844))
        assert all(len(b) == 8 for b in batches[:-4]) and all(len(b) == 5 for b in batches[-4:])
        assert all(len({i // 461 for i in b}) == 1 for b in batches)
        assert all({b[0] // 461 for b in batches[r:r+4]} == set(range(4)) for r in range(0, 232, 4))
        assert torch.equal(rng, torch.get_rng_state())
        hashes.append(sampler.digest())
    return hashes


def preflight(model, config, output, watcher):
    train_set, _ = _verify_split_dataset(config)
    assert parameter_counts(model)['trainable'] == 1643908
    assert tuple(model.fusion.heads) == TASKS and len(list(model.pools())) == 2
    orders = sampling_audit()
    model.sampling_sha256 = orders
    initial, frozen = digest(model.compact_state_dict()), frozen_digest(model)
    initial_routes = task_routes(model)
    optimizer = build_optimizer(model, config)
    scheduler = _scheduler(optimizer, 2320, .05, 116)
    check_precision(model, optimizer)
    # 两批不同任务；第一批的 Head 在第二步已有 Adam 状态但必须保持不变。
    batches = [list(range(8)), list(range(3 * 461, 3 * 461 + 8))]
    loader = DataLoader(train_set, batch_sampler=batches, num_workers=0,
                        generator=torch.Generator().manual_seed(9042))
    device = next(model.parameters()).device
    reports, equivalence = [], 0.
    for step, cpu in enumerate(loader):
        model.train(True)
        batch = move_to_device(cpu, device)
        optimizer.zero_grad(set_to_none=True)
        before = {task: optimizer_snapshot(head, optimizer) for task, head in model.heads()}
        with autocast_context(device):
            result = model(batch)
        fused = result['routing']['fused_frames'].detach().clone()
        assert not result['routing']['fused_frames'].requires_grad
        if step == 0:
            # 精确继承已审计；比较相同权重的来源 eval 前向与冻结训练前向。
            model.frozen_private_heads = False
            model.eval()
            with torch.no_grad(), autocast_context(device):
                source_prediction = model(batch)['predictions']
            model.frozen_private_heads = True
            model.train(True)
            equivalence = float((source_prediction - result['predictions'].detach()).abs().max())
            assert equivalence <= 1e-6
            # 四任务分流以对应的、已经不同的四个 Head 显式计算作参照。
            with torch.no_grad():
                ids = torch.tensor([3, 0, 2, 1, 0, 1, 2, 3], device=device)
                direct = torch.stack([model.fusion.heads[TASKS[int(t)]](x).squeeze(-1)
                                      for x, t in zip(fused.float(), ids)])
                torch.testing.assert_close(model.score_frames(fused.float(), ids), direct,
                                           atol=1e-6, rtol=0)
        loss, _, info = objective(result, batch, config)
        loss.backward()
        assert_frozen_mode(model)
        active = int(batch['task_id'][0])
        for index, (task, head) in enumerate(model.heads()):
            if index == active:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            else:
                assert all(p.grad is None for p in head.parameters())
        grad = prepare_gradients(model, config)
        optimizer.step()
        scheduler.step()
        for index, (task, head) in enumerate(model.heads()):
            after = optimizer_snapshot(head, optimizer)
            if index == active:
                assert before[task]['parameters'] != after['parameters']
            else:
                assert before[task] == after
        check_precision(model, optimizer)
        assert frozen_digest(model) == frozen and task_routes(model) == initial_routes
        with torch.no_grad(), autocast_context(device):
            after_features = model(batch)['routing']['fused_frames']
        torch.testing.assert_close(after_features, fused, atol=0, rtol=0)
        check_memory(watcher.report())
        reports.append(dict(step=step+1, task=TASKS[active], **info, gradient=grad,
                            frozen_unchanged=True, features_unchanged=True,
                            absent_heads_and_adam_unchanged=True))
        del result, loss, fused, after_features
    model.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        path = Path(temporary) / 'roundtrip.pt'
        save_checkpoint(path, model, config, -1)
        with torch.no_grad(), autocast_context(device):
            expected = model(batch)['predictions'].clone()
        with torch.no_grad():
            for head in model.fusion.heads.values():
                next(head.parameters()).add_(1)
        payload = load_model_checkpoint(model, path, config)
        del payload
        with torch.no_grad(), autocast_context(device):
            actual = model(batch)['predictions']
        delta = float((expected - actual).abs().max())
        assert delta <= 1e-6
    write_json(output / 'preflight.json', dict(
        passed=True, steps=reports, initial_state_sha256=initial, sampling_sha256=orders,
        source_eval_prediction_max_abs_delta=equivalence,
        checkpoint_reload_max_abs_delta=delta, peak_memory=watcher.report()))
    return initial, orders


def score_diagnostics(rows):
    result = score_spread(rows)
    for task in TASKS:
        subset = [r for r in rows if r['task'] == task]
        p = np.array([r['prediction'] for r in subset])
        y = np.array([r['target'] for r in subset])
        i, j = np.triu_indices(len(p), 1)
        valid = y[i] != y[j]
        signed = np.sign(y[i[valid]] - y[j[valid]]) * (p[i[valid]] - p[j[valid]])
        result['tasks'][task].update(
            pair_count=len(signed), reversed_pair_ratio=float((signed < 0).mean()) if len(signed) else None,
            prediction_tie_ratio=float((signed == 0).mean()) if len(signed) else None)
    return result


def logistic_comparison(output, config, history):
    """只比较已保存的相同轮次；取消的对照不冒充完整十轮结果。"""
    control = config['head_rank'].get('comparison_logistic')
    if control is None:
        return
    path = Path(control) / 'history.json'
    old = json.loads(path.read_text()) if path.exists() else []
    paired = []
    for current in history:
        reference = next((r for r in old if r['epoch'] == current['epoch']), None)
        if reference is None:
            continue
        for task in (*TASKS, 'macro'):
            for metric in ('plcc', 'srocc', 'krocc', 'rmse'):
                x = current['test']['metrics'][task][metric]
                y = reference['test']['metrics'][task][metric]
                paired.append(dict(epoch=current['epoch'], task=task, metric=metric,
                                   t2vqa=x, logistic=y,
                                   delta=x-y if x is not None and y is not None else None))
    if paired:
        write_csv(output / 'logistic_same_epoch_comparison.csv', paired)
    write_json(output / 'logistic_control_status.json', dict(
        path=control, status=config['head_rank']['comparison_logistic_status'],
        completed_epochs=len(old), complete_10_epoch_control=False,
        paired_epochs=sorted({r['epoch'] for r in paired}),
        note='Loss magnitudes are not directly comparable across ranking formulas.'))


def summarize(output):
    source = json.loads((output / 'source_evaluation.json').read_text())
    differences = []
    for kind in ('best', 'last'):
        result = json.loads((output / f'{kind}_evaluation.json').read_text())
        for task in (*TASKS, 'macro'):
            for metric in ('plcc', 'srocc', 'krocc', 'rmse'):
                x, y = result['metrics'][task][metric], source['metrics'][task][metric]
                differences.append(dict(checkpoint=kind, epoch=result['checkpoint_epoch'], task=task,
                                        metric=metric, current=x, source_epoch9=y,
                                        delta=x-y if x is not None and y is not None else None))
        write_json(output / f'{kind}_score_diagnostics.json', score_diagnostics(result['samples']))
    write_csv(output / 'source_epoch9_comparison.csv', differences)
    config = yaml.safe_load((output / 'config.yaml').read_text())
    logistic_comparison(output, config, json.loads((output / 'history.json').read_text()))
    write_json(output / 'summary.json', dict(
        complete=True, source_epoch=9, optimizer_updates=2320, updates_per_head=580,
        interpretation=['Inherited trained epoch9 Heads; no Head reset or copying.',
                        'Only Heads trained; representation, FiLM, Gates and experts frozen.',
                        'Training scope, sampling, ranking objective and update budget all change.',
                        'Single seed; same held-out set selects best and reports scores.',
                        'No automatic baseline replacement.']))
    if config['head_rank']['rank_kind'] == T2VQA_KIND:
        (output / 'report.md').write_text(
            '# Epoch9冻结表征：T2VQA Rank私有Head续训\n\n'
            '损失为 Smooth L1(beta=0.1) + 1.0 × T2VQA零margin ReLU Rank；'
            '双向非对角平均，除以1+最大逆序量，分母参与反向传播。\n\n'
            '正确排序及相等预测的ReLU项均可为零，Rank为零不代表分数准确或不存在预测趋同。'
            '请结合score_diagnostics中的标准差、RMSE、任务间分差及逆序/同分比例解读。\n\n'
            'source_epoch9_comparison.csv比较来源模型与best/last；'
            'logistic_same_epoch_comparison.csv只比较旧Logistic实验已完成的相同轮次。'
            '旧实验已取消，不构成完整十轮对照。\n\n'
            '本组从已训练epoch9继承四个私有Head，仅续训Head；单seed，同一held-out集'
            '选择best并报告成绩，存在选择偏差，不自动替换Baseline。\n')


def train_head_rank(config, mode='train'):
    validate_config(config)
    if mode not in ('train', 'preflight'):
        raise ValueError('Use evaluate.py for checkpoint evaluation')
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
            model = initialize(config, sources).to(device)
            write_json(output / 'sources.json', sources)
            write_json(output / 'parameters.json', parameter_counts(model))
            write_json(output / 'initialization_audit.json', model.initialization_audit)
            write_json(output / 'status.json', dict(stage='preflight'))
            initial, orders = preflight(model, config, output, watcher)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if mode == 'preflight':
                write_json(output / 'status.json', dict(stage='preflight_complete'))
                return
            write_json(output / 'status.json', dict(stage='reloading_epoch9_after_preflight'))
            model = initialize(config, sources).to(device)
            assert digest(model.compact_state_dict()) == initial
            model.sampling_sha256 = orders
            frozen, initial_routes = frozen_digest(model), task_routes(model)
            export_encoding(model, output)
            write_json(output / 'moe_routing/initial.json', initial_routes)
            train_loader, val_loader = build_loader(config, 'train'), build_loader(config, 'val')
            write_json(output / 'status.json', dict(stage='evaluating_source_before_training'))
            rows, metrics, routes = evaluate(model, val_loader, device)
            export(output, 'source', rows, metrics, routes, 8)
            write_json(output / 'source_score_diagnostics.json', score_diagnostics(rows))
            del rows, metrics, routes
            optimizer = build_optimizer(model, config)
            scheduler = _scheduler(optimizer, 2320, .05, 116)
            history, best = [], -float('inf')
            from engine.trainer import save_epoch_diagnostics

            for epoch in range(10):
                write_json(output / 'status.json', dict(stage='training', epoch=epoch+1, completed_epochs=epoch))
                train_loader.batch_sampler.set_epoch(epoch)
                counters = RouteCounts(model)
                indices, compositions, records, gradients = [], [], [], []
                loss_sum, regression_sum, rank_sum = 0., 0., 0.
                head_steps = {t: 0 for t in TASKS}
                for step, cpu in enumerate(train_loader, 1):
                    model.train(True)
                    assert_frozen_mode(model)
                    batch = move_to_device(cpu, device)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device):
                        result = model(batch)
                    loss, samples, info = objective(result, batch, config)
                    loss.backward()
                    assert_frozen_mode(model)
                    active = int(batch['task_id'][0])
                    for index, (_, head) in enumerate(model.heads()):
                        if index != active and any(p.grad is not None for p in head.parameters()):
                            raise AssertionError('Absent Head received gradients')
                    gradients.append(prepare_gradients(model, config))
                    lr = optimizer.param_groups[0]['lr']
                    optimizer.step()
                    scheduler.step()
                    head_steps[TASKS[active]] += 1
                    counters.add(batch['task_id'], result['routing'])
                    indices.append(cpu['pair_index'].tolist())
                    compositions.append([int((batch['task_id'] == i).sum()) for i in range(4)])
                    regression_sum += float(samples.detach().sum())
                    rank_sum += info['rank_loss']
                    loss_sum += float(loss.detach())
                    record = dict(epoch=epoch+1, step=step, global_step=epoch*232+step,
                                  task=TASKS[active], batch_size=len(batch['task_id']),
                                  loss=float(loss.detach()), lr=lr, **info)
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    check_memory(watcher.report())
                    del loss, samples, result
                assert len(indices) == 232 and indices == train_loader.batch_sampler.batches()
                assert train_loader.batch_sampler.digest() == orders[epoch]
                assert set(head_steps.values()) == {58}
                check_precision(model, optimizer)
                assert frozen_digest(model) == frozen and task_routes(model) == initial_routes
                rows, metrics, routes = evaluate(model, val_loader, device)
                write_json(output / 'score_diagnostics' / f'epoch_{epoch+1:02d}.json',
                           score_diagnostics(rows))
                diagnostics = dict(route_counts=counters.report(), shared_gradients=[[0., 0.]],
                                   private_module_steps={}, gradient_records=gradients,
                                   pair_indices=indices, batch_task_counts=compositions,
                                   order_sha256=orders[epoch])
                diagnostics['route_counts'].update(head_optimizer_steps=head_steps,
                    representation_frozen=True, private_task_modules_frozen=True)
                save_epoch_diagnostics(output, epoch, model, diagnostics, routes)
                write_json(output / 'loss_diagnostics' / f'epoch_{epoch+1:02d}.json', records)
                history.append(dict(epoch=epoch+1, train_loss_batch_mean=loss_sum/232,
                                    regression_loss_sample_mean=regression_sum/1844,
                                    rank_loss_batch_mean=rank_sum/232, test=dict(metrics=metrics),
                                    optimizer_updates=(epoch+1)*232, head_optimizer_steps=head_steps,
                                    frozen_state_sha256=frozen))
                write_json(output / 'history.json', history)
                logistic_comparison(output, config, history)
                save_checkpoint(output / 'last.pt', model, config, epoch, optimizer, scheduler,
                                selection_metric='test_macro_srocc')
                if metrics['macro']['srocc'] is not None and metrics['macro']['srocc'] > best:
                    best = metrics['macro']['srocc']
                    save_checkpoint(output / 'best.pt', model, config, epoch, score=best,
                                    selection_metric='test_macro_srocc')
                print(json.dumps(dict(epoch=epoch+1, metrics=metrics, best_macro_srocc=best)), flush=True)
                write_json(output / 'process_memory.json', watcher.report())
            del optimizer, scheduler
            for kind in ('best', 'last'):
                payload = load_model_checkpoint(model, output / f'{kind}.pt', config)
                saved_epoch = payload['epoch']
                del payload
                rows, metrics, routes = evaluate(model, val_loader, device)
                export(output, kind, rows, metrics, routes, saved_epoch)
            assert frozen_digest(model) == frozen
            assert file_digest(config['head_rank']['source_checkpoint']) == config['head_rank']['source_sha256']
            check_memory(watcher.report())
            write_json(output / 'process_memory.json', watcher.report())
            summarize(output)
            write_json(output / 'status.json', dict(stage='complete', completed_epochs=10, optimizer_updates=2320))
    except Exception as error:
        write_json(output / 'status.json', dict(stage='failed', error=repr(error)))
        if watcher is not None:
            write_json(output / 'process_memory.json', watcher.report())
        raise
