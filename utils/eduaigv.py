"""EduAIGV双任务协议：原生成器划分、独立任务表、未训练初始化审计。"""

import copy
import json
import math
from collections import Counter
from pathlib import Path

import torch
import yaml

from engine.audit import digest, file_digest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'outputs/multitask/runs/film_video_only_private_heads_seed42'
TASKS = ('overall', 'text_video')
CONTRACT = 'aider_eduaigv_video_only_private_heads2_generator_split_v1'
TRANSFORM = 'fp32_two_task_mean_subtract_then_row_l2_normalize'
COUNTS = {'train': 900, 'test': 230}
DESCRIPTIONS = {
    'overall': 'Overall quality evaluates the perceived quality of the video as a whole.',
    'text_video': 'Overall text-video alignment evaluates how well the video as a whole corresponds to the content described in the input prompt.',
}


def absolute(path):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def resolve_config(path, output=None):
    path = absolute(path)
    spec = yaml.safe_load(path.read_text())
    reference = absolute(spec['reference_config'])
    if reference != CONTROL / 'config.yaml' or file_digest(reference) != spec['reference_sha256']:
        raise ValueError('EduAIGV reference configuration changed')
    if tuple(spec['task_order']) != TASKS or spec['descriptions'] != DESCRIPTIONS:
        raise ValueError('Expected Overall quality and overall text-video alignment')
    config = copy.deepcopy(yaml.safe_load(reference.read_text()))
    root = absolute(output or spec['output_dir'])
    split_path = absolute(spec['data']['split_manifest'])
    if file_digest(split_path) != spec['data']['split_sha256']:
        raise ValueError('EduAIGV split specification changed')
    split = json.loads(split_path.read_text())
    config['task'] = dict(order=list(TASKS), descriptions=dict(DESCRIPTIONS),
        table_protocol='eduaigv', table_path=str(root / 'preparation/task_tables.json'))
    config['model']['task_embedding_transform'] = TRANSFORM
    config['experiment'].update(contract=CONTRACT, variant='eduaigv_video_only_private_heads2',
        task_embedding_transform=TRANSFORM, task_centering_scope='all_two_tasks_once_before_training',
        direct_control=str(CONTROL), historical_config=str(reference),
        control_config_sha256=spec['reference_sha256'])
    config['experiment'].pop('secondary_control', None)
    config['baseline']['runtime_architecture']['regression_heads'] = 2
    config['baseline']['name'] = 'EduAIGV depth2 video-only FiLM + shared/routed experts + two private Heads'
    config['baseline']['selection'] = 'Independent dataset experiment; no baseline replacement'
    for key in ('train_manifest', 'val_manifest', 'video_root'):
        config['data'][key] = str(absolute(spec['data'][key]))
    config['data']['artifact'].update(cache_dir=str(absolute(spec['data']['cache_dir'])),
        readonly_manifest=str(root / 'preparation/prior_manifest.json'))
    config['training'].update(output_dir=str(root), resume=None, device='cuda:0')
    config['evaluation'].update(checkpoint=str(root / 'best.pt'), device='cuda:0')
    config['inference']['device'] = 'cuda:0'
    config['ablation'].update(contract=CONTRACT, variant='eduaigv_video_only_private_heads2',
        split='eduaigv_generator_seed42_900_230', tasks=list(TASKS), task_ids=[0, 1],
        steps_per_epoch=225, total_steps=2250, control=str(CONTROL),
        control_config_sha256=spec['reference_sha256'], train_manifest_sha256=split['train_sha256'],
        test_manifest_sha256=split['test_sha256'])
    config['ablation'].pop('gpu', None)
    config['eduaigv'] = dict(contract=CONTRACT, output_root=str(root),
        reference_sha256=spec['reference_sha256'], split_manifest=str(split_path),
        split_sha256=spec['data']['split_sha256'], prepare_workers=spec['data']['prepare_workers'],
        train_count=900, test_count=230, source_spec=str(path), source_spec_sha256=file_digest(path),
        task_table_sha256=None, cache_manifest_sha256=None)
    config['_config_path'] = str(root / 'config.yaml')
    validate_config(config)
    return config


def validate_config(config):
    p, a = config['eduaigv'], config['ablation']
    if file_digest(p['source_spec']) != p['source_spec_sha256']:
        raise ValueError('EduAIGV source specification changed')
    if any(x != CONTRACT for x in (p['contract'], a['contract'], config['experiment']['contract'])):
        raise ValueError('Wrong EduAIGV contract')
    if (tuple(config['task']['order']) != TASKS or config['task']['descriptions'] != DESCRIPTIONS
            or config['task'].get('table_protocol') != 'eduaigv'
            or a['tasks'] != list(TASKS) or a['task_ids'] != [0, 1]):
        raise ValueError('Wrong EduAIGV task map/descriptions')
    if file_digest(CONTROL / 'config.yaml') != p['reference_sha256']:
        raise ValueError('EduAIGV reference configuration changed')
    base = yaml.safe_load((CONTROL / 'config.yaml').read_text())
    expected = copy.deepcopy(base['model'])
    expected['task_embedding_transform'] = TRANSFORM
    if config['model'] != expected or config['loader'] != base['loader']:
        raise ValueError('EduAIGV architecture or sampler differs from reference')
    actual_train = {k: v for k, v in config['training'].items() if k != 'output_dir'}
    expected_train = {k: v for k, v in base['training'].items() if k != 'output_dir'}
    if actual_train != expected_train:
        raise ValueError('EduAIGV training differs from reference (including clipping=1 and warmup=116)')
    if ((a['depth'], a['routed_experts'], a['routed_top_k'], a['layerwise_always_on_experts'],
         a['steps_per_epoch'], a['total_steps']) != (2, 7, 2, 1, 225, 2250)
            or (p['train_count'], p['test_count']) != (900, 230)
            or p['prepare_workers'] != 4):
        raise ValueError('Wrong EduAIGV counts or schedule')
    for key in ('num_frames', 'min_total_frames', 'vital_size', 'expert_size'):
        if config['data'][key] != base['data'][key]:
            raise ValueError('Input preprocessing changed: ' + key)
    artifact = lambda d: {k: v for k,v in d.items() if k not in ('cache_dir','readonly_manifest')}
    if artifact(config['data']['artifact']) != artifact(base['data']['artifact']):
        raise ValueError('Prior preprocessing changed')
    if any(key in config for key in ('head_rank', 'private_task_modules', 'videofeedback', 'repeated_holdout')):
        raise ValueError('Conflicting experiment protocols')
    if 'repeat' in p:
        from utils.eduaigv_splits import validate_repeat
        validate_repeat(config)


def validate_data(config):
    validate_config(config)
    p = config['eduaigv']
    if file_digest(p['split_manifest']) != p['split_sha256']:
        raise ValueError('EduAIGV split specification changed')
    split = json.loads(Path(p['split_manifest']).read_text())
    source_path = Path(p['split_manifest']).parent / split['source']
    if file_digest(source_path) != split['source_sha256']:
        raise ValueError('EduAIGV complete source changed')
    source = {r['id']: r for r in map(json.loads, source_path.read_text().splitlines())}
    ids, paths, report = {}, {}, {}
    for name, key in (('train', 'train_manifest'), ('test', 'val_manifest')):
        path = Path(config['data'][key])
        if file_digest(path) != split[name+'_sha256'] or file_digest(path) != config['ablation'][name+'_manifest_sha256']:
            raise ValueError('EduAIGV manifest changed: ' + name)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ids[name], paths[name] = {r['id'] for r in rows}, {r['video'] for r in rows}
        if (len(rows), len(ids[name]), len(paths[name])) != (COUNTS[name],)*3:
            raise ValueError('Wrong count or duplicate video')
        for r in rows:
            if r != source.get(r['id']) or not isinstance(r.get('prompt'), str) or not r['prompt'].strip():
                raise ValueError('Changed record or empty prompt')
            if set(r['labels']) != set(TASKS) or any(isinstance(v,bool) or not isinstance(v,(int,float))
                    or not math.isfinite(v) or not 0 <= v <= 1 for v in r['labels'].values()):
                raise ValueError('Invalid two-task score')
            if not (Path(config['data']['video_root']) / r['video']).is_file():
                raise FileNotFoundError(r['video'])
        generators = Counter(Path(r['video']).parts[0] for r in rows)
        if len(generators) != 10 or set(generators.values()) != {90 if name == 'train' else 23}:
            raise ValueError('Wrong generator quotas')
        report[name] = dict(videos=len(rows), records=2*len(rows), sha256=file_digest(path),
                            generator_counts=dict(generators))
    if ids['train'] & ids['test'] or paths['train'] & paths['test'] or ids['train'] | ids['test'] != set(source):
        raise ValueError('Video overlap or incomplete coverage')
    return dict(**report, total_videos=1130, id_overlap=0, path_overlap=0,
                prompt_statistics=split['prompt_statistics'])


def center_table(original):
    if original.shape != (2,768) or original.dtype != torch.float32 or not torch.isfinite(original).all():
        raise ValueError('Expected finite FP32 two-task table')
    if not torch.allclose(original.norm(dim=1), torch.ones(2), atol=1e-6, rtol=0):
        raise ValueError('Original CLIP vectors must be unit norm')
    mean = original.mean(0, keepdim=True)
    centered = original - mean
    norms = centered.norm(dim=1, keepdim=True)
    if (norms <= 1e-12).any():
        raise ValueError('Degenerate two-task centering')
    unit = centered / norms
    torch.testing.assert_close(unit[0], -unit[1], atol=1e-6, rtol=0)
    return dict(encoded_feature=unit, original_feature=original, task_mean=mean, centered_norms=norms)


def prepare_task_table(config):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    from engine.io import write_json
    folder = config['model']['dimension_encoder']['model_dir']
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        tokenizer = CLIPTokenizer.from_pretrained(folder, local_files_only=True)
        encoder = CLIPTextModelWithProjection.from_pretrained(folder, local_files_only=True).float().eval()
        inputs = tokenizer([config['task']['descriptions'][t] for t in TASKS], padding=True,
                           truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
        original = torch.nn.functional.normalize(encoder(**inputs).text_embeds.float(), dim=1)
    values = center_table(original)
    path = Path(config['task']['table_path'])
    if path.exists():
        raise FileExistsError(path)
    write_json(path, dict(protocol=CONTRACT, task_order=list(TASKS), descriptions=dict(DESCRIPTIONS),
        values={k:v.tolist() for k,v in values.items()}, hashes={k:digest({k:v}) for k,v in values.items()},
        token_ids=inputs['input_ids'].tolist(), original_cosine=(original@original.T).tolist(),
        transformed_cosine=(values['encoded_feature']@values['encoded_feature'].T).tolist(),
        note='Two-task centering makes opposite vectors; not a claim of opposite semantic meaning.'))
    config['task']['table_sha256'] = file_digest(path)
    config['eduaigv']['task_table_sha256'] = file_digest(path)


def load_two_task_table(task):
    path = Path(task['table_path'])
    if tuple(task['order']) != TASKS or file_digest(path) != task['table_sha256']:
        raise ValueError('EduAIGV task table hash/order mismatch')
    saved = json.loads(path.read_text())
    if saved['protocol'] != CONTRACT or saved['task_order'] != list(TASKS) or saved['descriptions'] != task['descriptions']:
        raise ValueError('EduAIGV task table identity mismatch')
    values = {k:torch.tensor(v,dtype=torch.float32) for k,v in saved['values'].items()}
    expected = center_table(values['original_feature'])
    if values.keys() != expected.keys():
        raise ValueError('Wrong EduAIGV task buffer coverage')
    for k,v in values.items():
        if not torch.equal(v,expected[k]) or digest({k:v}) != saved['hashes'][k]:
            raise ValueError('EduAIGV task transform mismatch: '+k)
    return values


def four_task_initial_view(model):
    """只读审计视图：恢复原四任务buffer和四份相同未训练Head的键名。"""
    from utils.initialization import load_task_table
    base = yaml.safe_load((CONTROL/'config.yaml').read_text())
    old_buffers = load_task_table(base['task'])
    heads = list(model.heads())
    if tuple(t for t,_ in heads) != TASKS or len({digest(h.state_dict()) for _,h in heads}) != 1:
        raise AssertionError('Two heads must be identical initially')
    pointers = [p.data_ptr() for _,h in heads for p in h.parameters()]
    if len(set(pointers)) != len(pointers):
        raise AssertionError('Private heads share parameter storage')
    result, inserted = {}, False
    for name, value in model.compact_state_dict().items():
        if name.startswith('dimension_encoder.'):
            result[name] = old_buffers[name.split('.',1)[1]]
        elif name.startswith('fusion.heads.'):
            if not inserted:
                for task in base['task']['order']:
                    result.update({f'fusion.heads.{task}.{k}':v for k,v in heads[0][1].state_dict().items()})
                inserted = True
        else:
            result[name] = value
    return result


def audit_initial_model(model, config):
    expected = json.loads((CONTROL/'preflight.json').read_text())['initial_state_sha256']
    retained = digest(four_task_initial_view(model))
    rng = digest({'cpu_rng':torch.get_rng_state()})
    saved_rng = json.loads((CONTROL/'initialization_audit.json').read_text())['original_split']['construction_rng_sha256']
    if retained != expected or rng != saved_rng:
        raise AssertionError('EduAIGV retained initialization or RNG differs from control')
    if 'repeat' in config['eduaigv']:
        from utils.eduaigv_splits import REFERENCE
        previous = json.loads((REFERENCE / 'initialization_audit.json').read_text())
        if (digest(model.compact_state_dict()) != previous['initial_state_sha256']
                or rng != previous['eduaigv']['construction_rng_sha256']
                or model.task_embedding_hashes != previous['task_embedding_hashes']):
            raise AssertionError('Repeated EduAIGV initial parameters/table/RNG differ from seed42')
    return dict(retained_four_task_view_sha256=retained, retained_parameters_match=True,
                construction_rng_sha256=rng, independent_identical_heads=2, trained_parameters_loaded=False)
