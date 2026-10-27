"""Explicit repeated-holdout opt-in; original-split guards remain unchanged."""

import json
from pathlib import Path

from engine.audit import file_digest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'outputs/multitask/runs/film_video_only_private_heads_seed42'
SPLITS = ROOT / 'data/AIGVQA-DB/splits_generator_stratified_hardcap2_10'
PROTOCOL = 'aider_video_only_private_heads_generator_hardcap2_10splits_v1'


def validate_protocol(config):
    value = config['repeated_holdout']
    if value['protocol'] != PROTOCOL or value['split'] not in [f'split_{i:02d}' for i in range(1, 11)]:
        raise ValueError('Invalid repeated-holdout protocol/split')
    if config['model'].get('head_policy') != 'private' or config['model'].get('film_video_condition') != 'video_only':
        raise ValueError('Repeated holdout requires video-only joint private heads')
    if config['ablation']['split'] != value['split']:
        raise ValueError('Split identity mismatch')
    for key, expected in (('gradient_clip', 1.0), ('seed', 42), ('epochs', 10),
                          ('warmup_steps', 116), ('lr', 1e-4), ('weight_decay', .05)):
        if config['training'][key] != expected:
            raise ValueError('Repeated-holdout training setting changed: ' + key)
    for name, expected in value['reference_hashes'].items():
        if file_digest(CONTROL / name) != expected:
            raise ValueError('Private-head reference changed: ' + name)
    if file_digest(value['cache_manifest']) != value['cache_manifest_sha256']:
        raise ValueError('Cache audit changed')
    if config['data']['artifact'].get('readonly_manifest') != value['cache_manifest']:
        raise ValueError('Repeated holdout must use audited read-only cache')
    return value


def validate_split_data(config):
    from scripts.prepare_aigvqa_splits import SOURCE, make_splits, HARD_VIDEOS
    p = validate_protocol(config)
    if file_digest(SOURCE) != p['source_sha256']:
        raise ValueError('Full dataset changed')
    rows = [json.loads(x) for x in SOURCE.read_text().splitlines()]
    _, generated = make_splits(rows)
    planned = next(s for s in generated if s['name'] == p['split'])
    report = {}
    for part, config_key in (('train', 'train_manifest'), ('test', 'val_manifest')):
        path = Path(config['data'][config_key])
        if path.resolve() != (SPLITS / p['split'] / f'AIGVQA_DB_{part}.jsonl').resolve():
            raise ValueError('Unexpected split manifest path')
        sha = file_digest(path)
        if sha != p['manifest_sha256'][part] or sha != config['ablation'][part + '_manifest_sha256']:
            raise ValueError('Split manifest hash mismatch')
        actual = [json.loads(x) for x in path.read_text().splitlines()]
        if actual != [rows[i] for i in planned[part]]:
            raise ValueError('Saved split differs from deterministic constrained split')
        report[part] = dict(path=str(path), videos=len(actual), records=4*len(actual), sha256=sha,
                            hard_videos=[r['video'] for r in actual if r['video'] in HARD_VIDEOS])
    return dict(**report, id_overlap=0, path_overlap=0, total_videos=576,
                protocol=PROTOCOL, split=p['split'], test_quotas=planned['test_quotas'])


def validate_checkpoint_protocol(payload, config):
    if payload.get('repeated_holdout') != config.get('repeated_holdout'):
        raise ValueError('Checkpoint repeated-holdout identity mismatch')


def audit_initialization(model, config, initial, orders=None):
    validate_protocol(config)
    saved = json.loads((CONTROL / 'initialization_audit.json').read_text())
    preflight = json.loads((CONTROL / 'preflight.json').read_text())
    if (initial != preflight['initial_state_sha256']
        or model.construction_rng_sha256 != saved['original_split']['construction_rng_sha256']
        or model.task_embedding_hashes != saved['task_embedding_hashes']):
        raise AssertionError('Repeated-holdout initialization differs from private-head control')
    if orders is not None and orders != preflight['sampling_sha256']:
        raise AssertionError('Repeated-holdout sampler indices differ from control')
    return dict(initial_state_equal=True, construction_rng_equal=True, task_table_equal=True,
                sampling_indices_equal=orders is not None, trained_weights_loaded=False,
                reference=str(CONTROL), split=config['repeated_holdout']['split'])
