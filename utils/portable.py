"""Path relocation and narrowly pinned historical-checkpoint support for this export."""

import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT / 'package_manifest.json').read_text())


def relocate(value):
    """Translate recorded source paths without editing historical YAML/checkpoints."""
    metadata = manifest()
    old_root = metadata['source_root']
    old_assets = metadata['pretrained_root']
    video_root = os.environ.get('AIDER_VIDEO_ROOT')
    def walk(item):
        if isinstance(item, dict):
            return {k: walk(v) for k, v in item.items()}
        if isinstance(item, list):
            return [walk(v) for v in item]
        if isinstance(item, str):
            if video_root and item == metadata['video_root']:
                return str(Path(video_root).expanduser().resolve())
            for old, new in ((old_root, ROOT), (old_assets, ROOT / 'pretrained_models')):
                if item == old or item.startswith(old + '/'):
                    if old == old_assets and not metadata['bundled_pretrained']:
                        return item
                    return str(new) + item[len(old):]
        return item
    return walk(value)


def normalize_checkpoint(payload, path):
    """Only the three byte-identical exported checkpoints receive path migration."""
    from engine.audit import file_digest
    metadata = manifest()
    if str(payload.get('config', {}).get('training', {}).get('output_dir', '')).startswith(metadata['source_root'] + '/'):
        checkpoint_hash = file_digest(path)
        if checkpoint_hash not in {v['sha256'] for v in metadata['checkpoints'].values()}:
            raise ValueError('Unrecognized historical checkpoint; path migration denied')
        payload = dict(payload)
        for key in ('config', 'ablation', 'private_task_modules', 'task_registry'):
            if key in payload:
                payload[key] = relocate(payload[key])
    return payload


def validate_export_sources(saved, current, checkpoint):
    """Keep exact checkpoint/data/weight hashes; permit only audited package code."""
    from engine.audit import file_digest
    metadata = manifest()
    if file_digest(checkpoint) not in {v['sha256'] for v in metadata['checkpoints'].values()}:
        return False
    original = json.loads((ROOT / metadata['run_dir'] / 'sources.json').read_text())
    if saved != original:
        raise ValueError('Exported checkpoint historical source record changed')
    old_inputs = {k: v['sha256'] for k, v in saved.items() if not k.startswith('implementation/')}
    new_inputs = {k: v['sha256'] for k, v in current.items() if not k.startswith('implementation/')}
    if old_inputs != new_inputs:
        raise ValueError('Exported model inputs/pretrained weights differ from the original run')
    for key, value in current.items():
        if key.startswith('implementation/'):
            relative = str(Path(value['path']).resolve().relative_to(ROOT))
            if metadata['files'].get(relative, {}).get('sha256') != value['sha256']:
                raise ValueError('Export implementation changed: ' + relative)
    return True
