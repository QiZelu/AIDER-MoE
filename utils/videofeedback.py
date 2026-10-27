"""VideoFeedback's explicit five-task protocol and fixed initialization audit."""

import copy
import json
import math
from pathlib import Path

import torch
import yaml

from engine.audit import digest, file_digest

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('static', 'temporal', 'dynamic', 'text_video', 'factual')
CONTRACT = 'aider_videofeedback_video_only_private_heads5_v1'
TRANSFORM = 'fp32_five_task_mean_subtract_then_row_l2_normalize'
CONTROL = ROOT / 'outputs/multitask/runs/film_video_only_private_heads_seed42'
COUNTS = {'train': 32618, 'test': 676}


def absolute(path):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def resolve_config(path, output=None):
    path = absolute(path)
    spec = yaml.safe_load(path.read_text())
    reference = absolute(spec['reference_config'])
    if reference != CONTROL / 'config.yaml' or file_digest(reference) != spec['reference_sha256']:
        raise ValueError('VideoFeedback source configuration changed')
    base = yaml.safe_load(reference.read_text())
    result = copy.deepcopy(base)
    root = absolute(output or spec['output_dir'])
    task = result['task']
    if tuple(spec['task_order']) != TASKS:
        raise ValueError('Expected fixed five-task order')
    task['order'] = list(TASKS)
    task['descriptions']['factual'] = spec['factual_description']
    task['table_path'] = str(root / 'preparation/task_tables.json')
    result['model']['task_embedding_transform'] = TRANSFORM
    result['experiment']['task_embedding_transform'] = TRANSFORM
    result['experiment']['task_centering_scope'] = 'all_five_tasks_once_before_training'
    result['experiment'].update(contract=CONTRACT, variant='videofeedback_video_only_private_heads5',
        historical_config=str(reference), direct_control=str(CONTROL),
        control_config_sha256=spec['reference_sha256'])
    result['experiment'].pop('secondary_control', None)
    result['baseline']['runtime_architecture']['regression_heads'] = 5
    result['baseline']['name'] = 'VideoFeedback depth2 video-only FiLM with five private heads'
    result['baseline']['selection'] = 'Independent VideoFeedback fixed train/test; does not replace AIGVQA baseline'
    for key in ('train_manifest', 'val_manifest', 'video_root'):
        result['data'][key] = str(absolute(spec['data'][key]))
    result['data']['artifact']['cache_dir'] = str(absolute(spec['data']['cache_dir']))
    result['data']['artifact']['readonly_manifest'] = str(root / 'preparation/prior_manifest.json')
    settings = spec['training']
    for key in ('seed', 'epochs', 'warmup_steps', 'lr', 'weight_decay', 'gradient_clip', 'smooth_l1_beta'):
        result['training'][key] = settings[key]
    result['training'].update(output_dir=str(root), resume=None, device='cuda:0')
    result['training']['module_lrs'] = {'text_video_backbone': settings['text_lr']}
    result['loader']['video_batch_size'] = settings['batch_size']
    result['evaluation'].update(checkpoint=str(root / 'best.pt'), device='cuda:0')
    result['inference']['device'] = 'cuda:0'
    result['ablation'].update(contract=CONTRACT, variant='videofeedback_video_only_private_heads5',
        split='videofeedback_clean_fixed', tasks=list(TASKS), task_ids=list(range(5)),
        epochs=settings['epochs'], warmup=settings['warmup_steps'],
        steps_per_epoch=settings['steps_per_epoch'], total_steps=settings['total_steps'],
        train_manifest_sha256=spec['data']['train_sha256'], test_manifest_sha256=spec['data']['test_sha256'])
    result['ablation'].pop('gpu', None)
    result['ablation'].update(control=str(CONTROL), control_config_sha256=spec['reference_sha256'])
    result['videofeedback'] = dict(contract=CONTRACT, output_root=str(root),
        reference=str(CONTROL), reference_sha256=spec['reference_sha256'],
        train_count=spec['data']['train_count'], test_count=spec['data']['test_count'],
        prepare_workers=spec['data']['prepare_workers'], source_spec=str(path),
        source_spec_sha256=file_digest(path), task_table_sha256=None, cache_manifest_sha256=None)
    result['_config_path'] = str(root / 'config.yaml')
    return result


def validate_config(config):
    vf, model, ablation = config['videofeedback'], config['model'], config['ablation']
    if vf['contract'] != CONTRACT or ablation['contract'] != CONTRACT:
        raise ValueError('Wrong VideoFeedback contract')
    if tuple(config['task']['order']) != TASKS or ablation['tasks'] != list(TASKS) or ablation['task_ids'] != list(range(5)):
        raise ValueError('Invalid five-task mapping')
    if set(config['task']['descriptions']) != set(TASKS):
        raise ValueError('Missing task descriptions')
    if file_digest(CONTROL / 'config.yaml') != vf['reference_sha256']:
        raise ValueError('Reference configuration changed')
    base = yaml.safe_load((CONTROL / 'config.yaml').read_text())
    expected_model = copy.deepcopy(base['model'])
    expected_model['task_embedding_transform'] = TRANSFORM
    if model != expected_model:
        raise ValueError('VideoFeedback model differs from selected architecture')
    for task in TASKS[:-1]:
        if config['task']['descriptions'][task] != base['task']['descriptions'][task]:
            raise ValueError('Original task description changed')
    required = dict(seed=42, epochs=10, warmup_steps=10194, gradient_clip=1., lr=1e-4,
                    weight_decay=.05, smooth_l1_beta=.1, gradient_accumulation=1, loss_kind='smooth_l1')
    if any(config['training'][k] != v for k, v in required.items()):
        raise ValueError('Unexpected VideoFeedback training schedule')
    if config['training']['module_lrs'] != {'text_video_backbone': 1e-5}:
        raise ValueError('Unexpected text LR')
    if config['loader'] != base['loader'] or config['data']['num_frames'] != 8:
        raise ValueError('Unexpected sampling/input settings')
    if (ablation['depth'], ablation['steps_per_epoch'], ablation['total_steps']) != (2, 20387, 203870):
        raise ValueError('Unexpected model depth or update count')
    if vf['train_count'] != 32618 or vf['test_count'] != 676:
        raise ValueError('Wrong data counts')


def validate_data(config):
    validate_config(config)
    sets, videos, report = {}, {}, {}
    for split, key in (('train', 'train_manifest'), ('test', 'val_manifest')):
        path = Path(config['data'][key])
        if file_digest(path) != config['ablation'][split + '_manifest_sha256']:
            raise ValueError('VideoFeedback manifest changed: ' + split)
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        sets[split], videos[split] = {r['id'] for r in rows}, {r['video'] for r in rows}
        if len(rows) != COUNTS[split] or len(sets[split]) != len(rows) or len(videos[split]) != len(rows):
            raise ValueError('Manifest count or duplicate error')
        for row in rows:
            if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                raise ValueError('Empty prompt')
            if set(row['labels']) != set(TASKS) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                or v not in (.2, .4, .6, .8) for v in row['labels'].values()):
                raise ValueError('Invalid five-dimensional labels')
        report[split] = dict(videos=len(rows), records=len(rows)*5,
                            path=str(path), sha256=file_digest(path))
    if sets['train'] & sets['test'] or videos['train'] & videos['test']:
        raise ValueError('Train/test overlap')
    return dict(**report, id_overlap=0, path_overlap=0, total_videos=33294)


def center_table(original):
    if original.shape != (5, 768) or original.dtype != torch.float32 or not torch.isfinite(original).all():
        raise ValueError('Expected finite FP32 five-task embeddings')
    if not torch.allclose(original.norm(dim=1), torch.ones(5), atol=1e-6, rtol=0):
        raise ValueError('Original task vectors must have unit norm')
    mean = original.mean(0, keepdim=True)
    centered = original - mean
    norms = centered.norm(dim=1, keepdim=True)
    if (norms <= 1e-12).any():
        raise ValueError('Degenerate centered task vector')
    return dict(encoded_feature=centered / norms, original_feature=original,
                task_mean=mean, centered_norms=norms)


def prepare_task_table(config):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    from engine.io import write_json
    saved = json.loads((ROOT / 'artifacts/reference/task_tables.json').read_text())
    original = torch.tensor(saved['original'], dtype=torch.float32)
    model_dir = config['model']['dimension_encoder']['model_dir']
    # Encoder construction is isolated from the later model's global RNG.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        tokenizer = CLIPTokenizer.from_pretrained(model_dir, local_files_only=True)
        encoder = CLIPTextModelWithProjection.from_pretrained(model_dir, local_files_only=True).float().eval()
        inputs = tokenizer([config['task']['descriptions']['factual']], padding=True,
                           truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
        factual = torch.nn.functional.normalize(encoder(**inputs).text_embeds[0].float(), dim=0)
    values = center_table(torch.cat([original, factual[None]], dim=0))
    path = Path(config['task']['table_path'])
    if path.exists():
        raise FileExistsError(path)
    write_json(path, dict(task_order=list(TASKS), descriptions=config['task']['descriptions'],
        values={k: v.tolist() for k, v in values.items()},
        hashes={k: digest({k: v}) for k, v in values.items()},
        factual_token_ids=inputs['input_ids'][0].tolist(),
        original_four_source_sha256=file_digest(ROOT / 'artifacts/reference/task_tables.json')))
    config['task']['table_sha256'] = file_digest(path)
    config['videofeedback']['task_table_sha256'] = file_digest(path)
    return values


def load_five_task_table(task):
    path = Path(task['table_path'])
    if tuple(task['order']) != TASKS or file_digest(path) != task['table_sha256']:
        raise ValueError('Five-task table hash/order mismatch')
    saved = json.loads(path.read_text())
    if saved['task_order'] != list(TASKS) or saved['descriptions'] != task['descriptions']:
        raise ValueError('Five-task table descriptions mismatch')
    values = {k: torch.tensor(v, dtype=torch.float32) for k, v in saved['values'].items()}
    recalculated = center_table(values['original_feature'])
    if values.keys() != recalculated.keys():
        raise ValueError('Wrong task buffer coverage')
    for k, value in values.items():
        if digest({k: value}) != saved['hashes'][k] or not torch.equal(value, recalculated[k]):
            raise ValueError('Invalid five-task transform: ' + k)
    old = json.loads((ROOT / 'artifacts/reference/task_tables.json').read_text())
    if not torch.equal(values['original_feature'][:4], torch.tensor(old['original'], dtype=torch.float32)):
        raise ValueError('Original four task embeddings changed')
    return values


def four_task_initial_view(model):
    from utils.initialization import load_task_table
    base = yaml.safe_load((CONTROL / 'config.yaml').read_text())
    buffers = load_task_table(base['task'])
    state = model.compact_state_dict()
    result = {}
    for key, value in state.items():
        if key.startswith('fusion.heads.factual.'):
            continue
        if key.startswith('dimension_encoder.'):
            value = buffers[key.split('.', 1)[1]]
        result[key] = value
    return result


def audit_initial_model(model, config):
    saved = json.loads((CONTROL / 'initialization_audit.json').read_text())
    expected = json.loads((CONTROL / 'preflight.json').read_text())['initial_state_sha256']
    retained = digest(four_task_initial_view(model))
    if retained != expected:
        raise AssertionError('Retained parameters differ from private-head untrained reference')
    rng = digest({'cpu_rng': torch.get_rng_state()})
    if rng != saved['original_split']['construction_rng_sha256']:
        raise AssertionError('Five-task construction changed RNG')
    heads = list(model.heads())
    if tuple(t for t, _ in heads) != TASKS:
        raise AssertionError('Expected five heads')
    pointers = [p.data_ptr() for _, h in heads for p in h.parameters()]
    if len(set(pointers)) != len(pointers) or len({digest(h.state_dict()) for _, h in heads}) != 1:
        raise AssertionError('Heads must be equal initially but storage-independent')
    return dict(retained_four_task_view_sha256=retained, retained_parameters_match=True,
                construction_rng_sha256=rng, independent_identical_heads=5,
                original_four_raw_task_vectors_preserved=True, trained_parameters_loaded=False)
