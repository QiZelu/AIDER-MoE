"""任务调制／α／Gate 私有化消融的配置与未训练复制审计。"""

import json
from pathlib import Path

import torch
import yaml

from engine.audit import digest, file_digest
from models.registry import TASKS
from models.task_private import TaskPrivateLinear

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'outputs/multitask/runs/film_video_only_private_heads_seed42'
CONTRACT = 'aider_private_task_film_alpha_gates_heads_no_clip_461_115_v1'
POLICIES = dict(task_film_policy='private', alpha_policy='private', moe_gate_policy='private')


def validate_config(config):
    from utils.portable import relocate
    config = relocate(config)
    protocol = config['private_task_modules']
    if protocol['contract'] != CONTRACT or config['ablation']['contract'] != CONTRACT:
        raise ValueError('Wrong private task module contract')
    if file_digest(CONTROL / 'config.yaml') != protocol['control_config_sha256']:
        raise ValueError('Private-head control configuration changed')
    base = relocate(yaml.safe_load((CONTROL / 'config.yaml').read_text()))
    if config['model'] != dict(base['model'], **POLICIES):
        raise ValueError('Unexpected private-task model configuration')
    for section in ('data','task','loader','mixed_task_experiment'):
        if config[section] != base[section]:
            raise ValueError('Private-task control differs: ' + section)
    for key, value in base['training'].items():
        expected = None if key == 'gradient_clip' else value
        if key not in ('output_dir','device') and config['training'][key] != expected:
            raise ValueError('Unexpected private-task training setting: ' + key)
    if config['ablation']['tasks'] != list(TASKS) or config['ablation']['task_ids'] != list(range(4)):
        raise ValueError('Expected four original tasks')
    for key in ('split','depth','epochs','warmup','routed_experts','routed_top_k',
                'layerwise_always_on_experts','steps_per_epoch','total_steps'):
        if config['ablation'][key] != base['ablation'][key]:
            raise ValueError('Unexpected private-task protocol: ' + key)
    if config['ablation'].get('gradient_clipping') != 'disabled':
        raise ValueError('Private task modules require no gradient clipping')


def collapsed_initial_state(model):
    """仅审计未训练模型：将四份相同 Linear 还原到原参数名称和顺序。"""
    wrappers = {name: module for name,module in model.named_modules() if isinstance(module,TaskPrivateLinear)}
    if len(wrappers) != 6:
        raise AssertionError('Expected three task FiLMs, one alpha, and two Gate banks')
    pointers = []
    for name, module in wrappers.items():
        if module.task_order != TASKS or tuple(module.by_task) != TASKS:
            raise AssertionError('Wrong task map: ' + name)
        hashes = [digest(linear.state_dict()) for linear in module.by_task.values()]
        if len(set(hashes)) != 1:
            raise AssertionError('Copied Linears are not identical at initialization: ' + name)
        pointers.extend(p.data_ptr() for p in module.parameters())
    if len(pointers) != len(set(pointers)):
        raise AssertionError('Private task parameters share storage')
    output, inserted = {}, set()
    for key,value in model.compact_state_dict().items():
        owner = next((name for name in wrappers if key.startswith(name+'.by_task.')), None)
        if owner is None:
            output[key] = value
        elif owner not in inserted:
            output.update({owner+'.'+k:v for k,v in wrappers[owner].by_task[TASKS[0]].state_dict().items()})
            inserted.add(owner)
    return output


def audit_initialization(model, config):
    validate_config(config)
    expected = json.loads((CONTROL / 'preflight.json').read_text())
    saved = json.loads((CONTROL / 'initialization_audit.json').read_text())
    mapped = digest(collapsed_initial_state(model))
    rng = digest({'cpu_rng':torch.get_rng_state()})
    if mapped != expected['initial_state_sha256']:
        raise AssertionError('Private-task parameter mapping differs from untrained control')
    if rng != saved['original_split']['construction_rng_sha256']:
        raise AssertionError('Private-task conversion changed construction RNG')
    if model.task_embedding_hashes != saved['task_embedding_hashes']:
        raise AssertionError('Task table differs from control')
    return dict(mapped_control_state_sha256=mapped, construction_rng_sha256=rng,
        retained_parameters_exact=True, copied_linears_equal_and_independent=True,
        task_modulators=12, alpha_generators=4, task_gates=8, private_heads=4,
        shared_expert_pools=2, trained_checkpoint_loaded=False)
