"""Private-head repeated holdout: preparation, scheduling and split-aware reports."""

import copy
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from engine.audit import file_digest
from engine.io import write_csv, write_json
from models.registry import TASKS
from utils.split_protocol import CONTROL, SPLITS, ROOT, PROTOCOL, validate_split_data

OUTPUT = ROOT / 'outputs/multitask/robustness/film_video_only_private_heads_generator_stratified_hardcap2_10splits_seed42'
NAMES = tuple(f'split_{i:02d}' for i in range(1, 11))
METRICS = ('plcc', 'srocc', 'krocc', 'rmse')
CAVEATS = [
    'Generator-stratified, difficulty-constrained repeated hold-out; not ten-fold CV.',
    'Six historically identified difficult videos are capped at two per test set; actual draws required no rejection.',
    'Test sets overlap; fixed model seed measures split sensitivity, not cross-seed significance.',
    'The test set selects best and reports performance; best and epoch-10 last are reported separately.',
    'No paired metric differences against the original split; no automatic baseline replacement.',
]


def audit_cache(config):
    import numpy as np
    from dataset.artifact_prior import ArtifactCache, ArtifactPriorExtractor
    from scripts.prepare_aigvqa_splits import SOURCE
    artifact = dict(config['data']['artifact'])
    cache = ArtifactCache(artifact.pop('cache_dir'))
    extractor = ArtifactPriorExtractor(**artifact)
    files = {}
    for row in map(json.loads, SOURCE.read_text().splitlines()):
        video = Path(config['data']['video_root']) / row['video']
        path = cache.path_for(video, config['data']['num_frames'], config['data']['vital_size'], extractor)
        value = cache.load(path)
        shape = (config['data']['num_frames'], config['data']['vital_size'], config['data']['vital_size'])
        if value is None or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Missing/corrupt Prior cache: ' + str(path))
        files[str(path.resolve())] = dict(video_id=row['id'], sha256=file_digest(path), shape=list(value.shape))
    if len(files) != 576:
        raise ValueError('Expected 576 distinct Prior cache entries')
    return dict(readonly=True, files=files, generation_version_verified=False,
                note='Existing cache contents are pinned; no regeneration or historical-version claim.')


def make_config(base, name, root, cache_sha, reference_hashes):
    if name not in NAMES:
        raise ValueError('Invalid split')
    config = copy.deepcopy(base)
    meta = json.loads((SPLITS / name / 'metadata.json').read_text())
    config['data']['train_manifest'] = str(SPLITS / name / 'AIGVQA_DB_train.jsonl')
    config['data']['val_manifest'] = str(SPLITS / name / 'AIGVQA_DB_test.jsonl')
    config['data']['artifact']['readonly_manifest'] = str(root / 'prior_manifest.json')
    config['training'].update(device='cuda:0', output_dir=str(root / name), resume=None)
    config['evaluation'].update(device='cuda:0', checkpoint=str(root / name / 'best.pt'))
    config['inference']['device'] = 'cuda:0'
    hashes = {p: meta['manifests'][p]['sha256'] for p in ('train', 'test')}
    config['ablation'].update(split=name, train_manifest_sha256=hashes['train'],
                              test_manifest_sha256=hashes['test'])
    # GPU assignment is runtime scheduler metadata, not a historical model field.
    config['ablation'].pop('gpu', None)
    config['repeated_holdout'] = dict(protocol=PROTOCOL, split=name, split_seed=meta['seed'],
        source_sha256=meta['source_sha256'], manifest_sha256=hashes,
        reference_hashes=reference_hashes, cache_manifest=str(root / 'prior_manifest.json'),
        cache_manifest_sha256=cache_sha, output_root=str(root), test_quotas=meta['test_quotas'])
    config['_config_path'] = str(root / 'configs' / (name + '.yaml'))
    return config


def prepare(root):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError('Refusing to overwrite run root: ' + str(root))
    base = yaml.safe_load((CONTROL / 'config.yaml').read_text())
    refs = {name: file_digest(CONTROL / name) for name in (
        'config.yaml', 'initialization_audit.json', 'preflight.json')}
    cache_report = audit_cache(base)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'status.json', dict(stage='preparing', protocol=PROTOCOL))
    write_json(root / 'prior_manifest.json', cache_report)
    cache_sha = file_digest(root / 'prior_manifest.json')
    (root / 'configs').mkdir()
    (root / 'reference').mkdir()
    for name in refs:
        shutil.copyfile(CONTROL / name, root / 'reference' / name)
    for name in ('summary.json', 'split_summary.csv', 'generator_split_counts.csv', 'test_coverage.csv', 'test_overlap.csv'):
        shutil.copyfile(SPLITS / name, root / ('data_' + name))
    audits, prompt_rows = {}, []
    for name in NAMES:
        config = make_config(base, name, root, cache_sha, refs)
        audits[name] = validate_split_data(config)
        (root / 'configs' / (name + '.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
        train = [json.loads(x) for x in Path(config['data']['train_manifest']).read_text().splitlines()]
        test = [json.loads(x) for x in Path(config['data']['val_manifest']).read_text().splitlines()]
        prompts = {r['prompt'] for r in train}
        prompt_rows.append(dict(split=name, test_videos=115,
            test_videos_with_exact_train_prompt=sum(r['prompt'] in prompts for r in test),
            shared_unique_prompts=len(prompts & {r['prompt'] for r in test})))
    write_json(root / 'data_audit.json', audits)
    write_csv(root / 'prompt_overlap.csv', prompt_rows)
    write_json(root / 'status.json', dict(stage='prepared', protocol=PROTOCOL, reference_hashes=refs))
    return root


def load_split_config(root, name):
    if name not in NAMES:
        raise ValueError('Invalid split')
    config = yaml.safe_load((Path(root) / 'configs' / (name + '.yaml')).read_text())
    validate_split_data(config)
    return config


def verified_evaluation(folder, kind, config):
    path = folder / f'{kind}_evaluation.json'
    report = json.loads(path.read_text())
    expected_rows = [json.loads(x) for x in Path(config['data']['val_manifest']).read_text().splitlines()]
    expected = {(r['id'], t): r['labels'][t] for r in expected_rows for t in TASKS}
    samples = report['samples']
    actual = {(r['video_id'], r['task']): r['target'] for r in samples}
    if not report.get('completed') or len(samples) != 460 or actual.keys() != expected.keys():
        raise ValueError('Incomplete/incorrect evaluation identities: ' + str(path))
    if any(abs(actual[k] - expected[k]) > 1e-6 for k in expected):
        raise ValueError('Evaluation labels differ from bound split')
    if not 1 <= report['checkpoint_epoch'] <= 10 or (kind == 'last' and report['checkpoint_epoch'] != 10):
        raise ValueError('Invalid checkpoint epoch')
    for task in (*TASKS, 'macro'):
        if any(report['metrics'][task][m] is None or not math.isfinite(report['metrics'][task][m]) for m in METRICS):
            raise ValueError('Nonfinite split metric')
    return report


def summarize_split(folder):
    from engine.reporting import score_diagnostics
    folder = Path(folder)
    config = yaml.safe_load((folder / 'config.yaml').read_text())
    validate_split_data(config)
    history = json.loads((folder / 'history.json').read_text())
    if len(history) != 10 or history[-1]['optimizer_updates'] != 2310:
        raise ValueError('Incomplete split training')
    best_epoch = max(range(10), key=lambda i: history[i]['test']['metrics']['macro']['srocc']) + 1
    diagnostics, rows = {}, []
    for kind in ('best', 'last'):
        report = verified_evaluation(folder, kind, config)
        if kind == 'best' and report['checkpoint_epoch'] != best_epoch:
            raise ValueError('Best checkpoint selection mismatch')
        diagnostics[kind] = score_diagnostics(report['samples'])
        for task in (*TASKS, 'macro'):
            rows.append(dict(split=config['ablation']['split'], checkpoint=kind,
                epoch=report['checkpoint_epoch'], task=task,
                **{m: report['metrics'][task][m] for m in METRICS}))
    write_csv(folder / 'metrics.csv', rows)
    write_json(folder / 'private_head_score_diagnostics.json', diagnostics)
    summary = dict(complete=True, split=config['ablation']['split'], best_epoch=best_epoch,
                   protocol=config['repeated_holdout'], interpretation=CAVEATS)
    write_json(folder / 'summary.json', summary)
    return summary


def aggregate_metrics(rows):
    result = []
    for kind in ('best', 'last'):
        for task in (*TASKS, 'macro'):
            subset = [r for r in rows if r['checkpoint'] == kind and r['task'] == task]
            for metric in METRICS:
                values = [r[metric] for r in subset]
                result.append(dict(checkpoint=kind, task=task, metric=metric, n=len(values),
                    mean=statistics.mean(values) if values else None,
                    std_ddof1=statistics.stdev(values) if len(values) > 1 else None))
    return result


def summarize(root):
    root = Path(root)
    statuses, rows, completed = {}, [], []
    for name in NAMES:
        folder = root / name
        status = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else dict(stage='pending')
        statuses[name] = status
        if status['stage'] != 'complete':
            continue
        try:
            config = load_split_config(root, name)
            history = json.loads((folder / 'history.json').read_text())
            if len(history) != 10 or history[-1]['optimizer_updates'] != 2310:
                raise ValueError('Incomplete training history')
            local = []
            for kind in ('best', 'last'):
                report = verified_evaluation(folder, kind, config)
                for task in (*TASKS, 'macro'):
                    local.append(dict(split=name, split_seed=config['repeated_holdout']['split_seed'],
                        checkpoint=kind, epoch=report['checkpoint_epoch'], task=task,
                        **{m: report['metrics'][task][m] for m in METRICS}))
            if not json.loads((folder / 'summary.json').read_text())['complete']:
                raise ValueError('Missing split acceptance')
            rows.extend(local)
            completed.append(name)
        except Exception as error:
            statuses[name] = dict(stage='invalid_result', error=repr(error))
    if rows:
        write_csv(root / 'per_split_metrics.csv', rows)
    write_csv(root / 'summary.csv', aggregate_metrics(rows))
    report = dict(complete=len(completed) == 10, completed_splits=completed,
        completed_count=len(completed), expected_splits=10, statuses=statuses,
        aggregation='Only fully completed splits; partial aggregates are not ten-split results.',
        interpretation=CAVEATS)
    write_json(root / 'summary.json', report)
    return report


def free_slots(gpus, active):
    return [gpu for gpu in sorted(gpus) if gpu not in {job['gpu'] for job in active.values()}]


def launch(root, gpus):
    root = prepare(root)
    jobs, active, queue = {}, {}, list(NAMES)
    write_json(root / 'status.json', dict(stage='running', manager_pid=os.getpid(), gpus=gpus))
    while queue or active:
        for name, job in list(active.items()):
            code = job['process'].poll()
            if code is None:
                continue
            status_file = root / name / 'status.json'
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            complete = code == 0 and status.get('stage') == 'complete'
            jobs[name].update(state='complete' if complete else 'failed', exit_code=code, finished_at=time.time())
            if not complete and status.get('stage') != 'failed':
                write_json(status_file, dict(stage='failed', error='Worker exit without completed results', exit_code=code))
            del active[name]
        for gpu in free_slots(gpus, active):
            if not queue:
                break
            name = queue.pop(0)
            command = [sys.executable, '-u', '-B', str(ROOT / 'scripts/train_private_heads_splits.py'),
                       '--mode', 'train', '--split', name, '--gpu', str(gpu), '--output-root', str(root)]
            log = root / f'worker_{name}.log'
            jobs[name] = dict(gpu=gpu, command=command, log=str(log), state='starting', started_at=time.time())
            try:
                with log.open('x') as handle:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}, start_new_session=True)
                jobs[name].update(pid=process.pid, state='running')
                active[name] = dict(process=process, gpu=gpu)
            except Exception as error:
                jobs[name].update(state='failed', error=repr(error))
                write_json(root / name / 'status.json', dict(stage='failed', error=repr(error)))
        write_json(root / 'scheduler.json', dict(manager_pid=os.getpid(), jobs=jobs, pending=queue))
        summarize(root)
        if active:
            time.sleep(10)
    report = summarize(root)
    write_json(root / 'status.json', dict(stage='complete' if report['complete'] else 'finished_with_failures',
        completed_count=report['completed_count'], manager_pid=os.getpid()))


def evaluate_split(root, name, checkpoint):
    import torch
    from utils.initialization import initialize_model
    from utils.provenance import selected_source_hashes
    from engine.checkpoint import load_model_checkpoint
    from engine.evaluation import evaluate, export
    from engine.memory import initialize_process_memory, MemoryWatch, check_memory
    from dataset.loader import build_loader
    config = load_split_config(root, name)
    folder = Path(root) / 'evaluations' / name / Path(checkpoint).stem
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    initialize_process_memory()
    torch.cuda.set_per_process_memory_fraction(14*1024**3 / torch.cuda.get_device_properties(0).total_memory)
    with MemoryWatch() as watcher:
        model = initialize_model(config)
        model.source_hashes = selected_source_hashes(config)
        payload = load_model_checkpoint(model, checkpoint, config)
        epoch = payload['epoch']
        del payload
        model.cuda()
        rows, metrics, routes = evaluate(model, build_loader(config, 'val'), torch.device('cuda:0'))
        export(folder, 'evaluation', rows, metrics, routes, epoch)
        check_memory(watcher.report())
        write_json(folder / 'process_memory.json', watcher.report())
