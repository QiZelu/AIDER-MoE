"""私有任务模块的梯度、调制与严格同划分结果比较。"""

import json
from pathlib import Path

import torch
import yaml

from engine.audit import file_digest
from engine.io import write_json, write_csv
from engine.reporting import score_diagnostics
from models.registry import TASKS, ATTRIBUTES, BRANCHES
from models.task_private import private_module_items, task_linear
from utils.private_task_modules import CONTROL, validate_config


def check_private_activity(model, ids, *, all_task_gates=False):
    """MI may update every private Gate; other modules still follow batch IDs."""
    active = set(ids.detach().cpu().tolist())
    gates = ({id(module) for _, pool in model.pools()
              for module in getattr(pool.gate, 'by_task', {}).values()}
             if all_task_gates else set())
    report = {}
    for name, task, module in private_module_items(model):
        expected = model.task_order.index(task) in active or id(module) in gates
        gradients = [p.grad for p in module.parameters()]
        if any((g is not None) != expected for g in gradients):
            raise AssertionError('Private module gradient violates task presence: ' + name)
        report[name] = expected
    return report


def module_gradient_norms(model):
    """互斥主模块范数，另列私有模块细分；不会改写梯度。"""
    squared = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            key = name.split('.')[0]
            value = p.grad.detach().double().square().sum()
            squared[key] = squared.get(key, 0) + value
    detail = {}
    for name, _, module in private_module_items(model):
        values = [p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
        detail[name] = float(torch.stack(values).sum().sqrt()) if values else None
    return dict(main={k:float(v.sqrt()) for k,v in squared.items()}, private_detail=detail)


@torch.no_grad()
def task_coefficients(model):
    ids = torch.arange(len(model.task_order), device=next(model.parameters()).device)
    condition = model.dimension_encoder(ids)
    with torch.autocast(condition.device.type, enabled=False):
        alpha = task_linear(model.alpha_gate, condition.float(), ids).sigmoid()
        result = {task: dict(alpha=float(alpha[i,0]), branches={}) for i,task in enumerate(model.task_order)}
        for branch, attr in zip(BRANCHES, ATTRIBUTES):
            gamma, beta = getattr(model,attr).coefficients(condition,ids)
            for i,task in enumerate(model.task_order):
                result[task]['branches'][branch] = dict(gamma=gamma[i].cpu().tolist(),beta=beta[i].cpu().tolist())
    return result


def summarize(output):
    output = Path(output)
    from utils.portable import relocate
    config = relocate(yaml.safe_load((output/'config.yaml').read_text()))
    validate_config(config)
    history = json.loads((output/'history.json').read_text())
    if len(history) != 10 or history[-1]['optimizer_updates'] != 2310:
        raise ValueError('Incomplete private-task training')
    best = max(range(10),key=lambda i:history[i]['test']['metrics']['macro']['srocc'])+1
    metrics, differences, diagnostics = [], [], {}
    for kind in ('best','last'):
        reports = {}
        for label,folder in (('private_task_no_clip',output),('private_heads_control',CONTROL)):
            saved_config = relocate(yaml.safe_load((folder/'config.yaml').read_text()))
            for key in ('train_manifest','val_manifest'):
                if file_digest(saved_config['data'][key]) != file_digest(config['data'][key]):
                    raise ValueError('Comparison split mismatch')
            report=json.loads((folder/f'{kind}_evaluation.json').read_text())
            if not report.get('completed') or len(report['samples']) != 460:
                raise ValueError('Incomplete comparison evaluation')
            if folder == output and report['checkpoint_epoch'] != (best if kind=='best' else 10):
                raise ValueError('Incorrect best/last selection')
            reports[label]=report
            diagnostics[label+'_'+kind]=score_diagnostics(report['samples'])
            for task in (*TASKS,'macro'):
                metrics.append(dict(experiment=label,checkpoint=kind,epoch=report['checkpoint_epoch'],task=task,
                    **{m:report['metrics'][task][m] for m in ('plcc','srocc','krocc','rmse')}))
        actual, reference = reports['private_task_no_clip'], reports['private_heads_control']
        identities = [{(r['video_id'],r['task']):r['target'] for r in x['samples']} for x in (actual,reference)]
        if identities[0] != identities[1] or len(identities[0]) != 460:
            raise ValueError('Comparison video/task/label mismatch')
        for task in (*TASKS,'macro'):
            for metric in ('plcc','srocc','krocc','rmse'):
                a,b=actual['metrics'][task][metric],reference['metrics'][task][metric]
                differences.append(dict(checkpoint=kind,task=task,metric=metric,experiment=a,control=b,
                                        delta=None if a is None or b is None else a-b))
    write_csv(output/'comparison_metrics.csv',metrics)
    write_csv(output/'comparison_deltas.csv',differences)
    write_json(output/'task_score_diagnostics.json',diagnostics)
    report=dict(complete=True,best_epoch=best,direct_control=str(CONTROL),
        interpretation=['Task FiLM, alpha and Gate privatization plus removal of gradient clipping are simultaneous changes.',
            'Heads remain private; video adaptation, attention, routed FFN pools and always-on FFNs remain shared.',
            'Private Gates do not impose expert disjointness or guarantee elimination of gradient conflicts.',
            'Single seed; same held-out set selects best and reports scores; no automatic baseline replacement.'])
    write_json(output/'summary.json',report)
    return report
