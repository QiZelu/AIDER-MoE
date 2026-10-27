"""Independent, RNG-local EduAIGV repeated generator holdouts."""

import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

from engine.audit import file_digest
from utils.eduaigv import ROOT

SEEDS = tuple(42 * i for i in range(1, 11))
REFERENCE = ROOT / 'outputs/multitask/runs/eduaigv_film_video_only_private_heads_seed42'
OUTPUT = ROOT / 'outputs/multitask/robustness/eduaigv_film_video_only_private_heads_generator_stratified_10splits_seed42'
PROTOCOL = 'eduaigv_private_heads_generator_stratified_10splits_model_seed42_v1'


def split_records(rows, seed):
    if seed not in SEEDS:
        raise ValueError('Unsupported split seed')
    groups = defaultdict(list)
    if len(rows) != 1130 or len({r['id'] for r in rows}) != 1130 or len({r['video'] for r in rows}) != 1130:
        raise ValueError('Expected 1130 unique source videos')
    for index, row in enumerate(rows):
        groups[Path(row['video']).parts[0]].append(index)
    if len(groups) != 10 or any(len(v) != 113 for v in groups.values()):
        raise ValueError('Expected ten generators of 113 videos')
    train_indices, quotas = set(), []
    for generator, indices in sorted(groups.items()):
        key = f'edu-aigv-generator-split-v1|{seed}|{generator}'.encode()
        local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')
        shuffled = list(indices)
        random.Random(local_seed).shuffle(shuffled)
        train_indices.update(shuffled[:90])
        quotas.append(dict(generator=generator, total=113, train=90, test=23, rng_seed=str(local_seed)))
    train = [r for i, r in enumerate(rows) if i in train_indices]
    test = [r for i, r in enumerate(rows) if i not in train_indices]
    a, b = ({r['prompt'] for r in subset} for subset in (train, test))
    prompts = dict(train_unique=len(a), test_unique=len(b), overlap=len(a & b),
                   prompt_disjoint=not bool(a & b), test_videos_with_train_prompt=sum(r['prompt'] in a for r in test))
    return train, test, quotas, prompts


def manifest_bytes(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows).encode()


def make_config(base, root, seed):
    root = Path(root).resolve()
    folder = root / f'seed{seed}'
    metadata = folder / 'split.json'
    split = json.loads(metadata.read_text())
    config = copy.deepcopy(base)
    config['data'].update(train_manifest=str(folder / 'train.jsonl'), val_manifest=str(folder / 'test.jsonl'))
    config['training']['output_dir'] = str(folder)
    config['evaluation']['checkpoint'] = str(folder / 'best.pt')
    config['ablation'].update(split=f'eduaigv_generator_seed{seed}_900_230',
        train_manifest_sha256=split['train_sha256'], test_manifest_sha256=split['test_sha256'])
    config['eduaigv'].update(output_root=str(folder), split_manifest=str(metadata), split_sha256=file_digest(metadata))
    config['eduaigv']['repeat'] = dict(protocol=PROTOCOL, split_seed=seed, model_seed=42,
        output_root=str(root), reference=str(REFERENCE), reference_config_sha256=file_digest(REFERENCE / 'config.yaml'))
    config['_config_path'] = str(folder / 'config.yaml')
    return config


def validate_repeat(config):
    repeat = config['eduaigv']['repeat']
    if repeat['protocol'] != PROTOCOL or repeat['split_seed'] not in SEEDS[1:]:
        raise ValueError('Wrong repeated EduAIGV identity; seed42 is read-only reuse')
    if repeat['reference'] != str(REFERENCE) or file_digest(REFERENCE / 'config.yaml') != repeat['reference_config_sha256']:
        raise ValueError('Repeated EduAIGV reference changed')
    base = yaml.safe_load((REFERENCE / 'config.yaml').read_text())
    expected = make_config(base, repeat['output_root'], repeat['split_seed'])
    actual = copy.deepcopy(config)
    # A standalone preflight writes only to its own disposable directory.
    if actual['training']['output_dir'] == str(Path(expected['training']['output_dir']) / 'preflight_only'):
        actual['training']['output_dir'] = expected['training']['output_dir']
    if actual != expected:
        raise ValueError('Repeated EduAIGV configuration differs from pinned protocol')


def validate_checkpoint_identity(payload, config):
    if payload.get('eduaigv') != config.get('eduaigv'):
        raise ValueError('Wrong EduAIGV dataset/checkpoint identity (including split and table hashes)')
