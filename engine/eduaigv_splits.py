"""Read-only historical reuse, ten holdouts, and isolated four-GPU scheduling."""

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from engine.audit import file_digest
from engine.io import write_csv, write_json
from utils.eduaigv import ROOT, TASKS, load_two_task_table, validate_data
from utils.eduaigv_splits import SEEDS, REFERENCE, PROTOCOL, make_config, manifest_bytes, split_records

METRICS = ('plcc', 'srocc', 'krocc', 'rmse')
CAVEATS = [
    'Repeated random generator-stratified hold-out, not ten-fold cross-validation.',
    'The test set selects best by two-task macro SROCC; ties retain the earlier epoch.',
    'Test sets overlap across splits; identical prompts may occur in train and test.',
    'Fixed model seed42 primarily measures split sensitivity, not model-seed variation.',
    'No difficult-sample filtering, prompt isolation, or automatic Baseline replacement.',
]


def read(path):
    return json.loads(Path(path).read_text())


def verify_results(folder, config, checkpoints=False):
    """Validate without writing to the source run, including early-tie selection."""
    folder = Path(folder)
    status = read(folder / 'status.json')
    history = read(folder / 'history.json')
    if (status.get('stage') != 'complete' or len(history) != 10
            or any(r['epoch'] != i + 1 or r['optimizer_updates'] != (i + 1) * 225
                   for i, r in enumerate(history))):
        raise ValueError('Incomplete ten-epoch / 2250-update history')
    scores = [r['test']['metrics']['macro']['srocc'] for r in history]
    if any(v is None or not math.isfinite(v) for v in scores):
        raise ValueError('Nonfinite selection score')
    best = max(range(10), key=scores.__getitem__) + 1
    expected = {(r['id'], t): r['labels'][t]
                for r in map(json.loads, Path(config['data']['val_manifest']).read_text().splitlines()) for t in TASKS}
    rows = []
    for kind in ('best', 'last'):
        epoch = best if kind == 'best' else 10
        report = read(folder / f'{kind}_evaluation.json')
        samples = report['samples']
        actual = {(r['video_id'], r['task']): r['target'] for r in samples}
        if (not report.get('completed') or report['checkpoint_epoch'] != epoch or len(samples) != 460
                or actual.keys() != expected.keys() or any(abs(actual[k] - expected[k]) > 1e-6 for k in expected)
                or any(not math.isfinite(r['prediction']) for r in samples)):
            raise ValueError('Evaluation identity, labels, or checkpoint epoch mismatch')
        for task in (*TASKS, 'macro'):
            values = {m: report['metrics'][task][m] for m in METRICS}
            if any(v is None or not math.isfinite(v) for v in values.values()):
                raise ValueError('Nonfinite evaluation metric')
            rows.append(dict(checkpoint=kind, epoch=epoch, best_epoch=best, task=task, **values))
        from engine.metrics import regression_metrics
        for task in TASKS:
            subset = [r for r in samples if r['task'] == task]
            computed = regression_metrics(np.array([r['prediction'] for r in subset], np.float32)[:, None],
                np.array([r['target'] for r in subset], np.float32)[:, None], ['score'])['score']
            if any(abs(computed[m] - report['metrics'][task][m]) > 1e-7 for m in METRICS):
                raise ValueError('Saved metrics disagree with predictions')
        if any(abs(report['metrics']['macro'][m] - np.mean([report['metrics'][t][m] for t in TASKS])) > 1e-7 for m in METRICS):
            raise ValueError('Incorrect two-task macro average')
        for form, count in (('long', 460), ('wide', 230)):
            with (folder / f'{kind}_predictions_{form}.csv').open() as handle:
                if len(list(csv.DictReader(handle))) != count:
                    raise ValueError('Incomplete prediction export')
        if not (folder / f'{kind}_routing.json').is_file() or not (folder / f'{kind}.pt').is_file():
            raise ValueError('Missing checkpoint or routing export')
        if checkpoints:
            payload = torch.load(folder / f'{kind}.pt', map_location='cpu', weights_only=False, mmap=True)
            from utils.eduaigv_splits import validate_checkpoint_identity
            validate_checkpoint_identity(payload, config)
            if (payload['config'] != config or payload['epoch'] != epoch - 1
                    or payload['ablation'] != config['ablation']
                    or payload['task_registry'] != config['task']
                    or payload['selection_metric'] != 'test_macro_srocc'):
                raise ValueError('Checkpoint configuration/epoch mismatch')
            from utils.experiment import checkpoint_contract
            from engine.gradients import validate_checkpoint_gradient_policy
            validate_checkpoint_gradient_policy(payload, config)
            if payload['model_contract'] != checkpoint_contract(config) or payload['film_video_condition'] != 'video_only':
                raise ValueError('Checkpoint model contract mismatch')
            if kind == 'best' and payload['score'] != scores[best - 1]:
                raise ValueError('Best checkpoint selection score mismatch')
            table = read(config['task']['table_path'])
            if payload['task_embedding_hashes'] != table['hashes']:
                raise ValueError('Checkpoint task table mismatch')
            # The complete text encoder schema is taken from the original pretrained BLIP.
            pretrained = torch.load(config['model']['backbones']['blip']['checkpoint_path'],
                                    map_location='cpu', weights_only=False)
            original = pretrained.get('model', pretrained)
            text_keys = {k for k in original if k.startswith('text_encoder.')}
            if not text_keys or any('tv_backbone.model.' + k not in payload['model'] for k in text_keys):
                raise ValueError('Incomplete fine-tuned BLIP text encoder')
            del original, pretrained
            if kind == 'last' and (payload['scheduler']['last_epoch'] != 2250 or not payload['optimizer']['state']):
                raise ValueError('Incomplete last optimizer/scheduler')
            if payload['source_hashes'] != read(folder / 'sources.json'):
                raise ValueError('Checkpoint source record mismatch')
            del payload
    for epoch in range(1, 11):
        for subdir in ('film', 'moe_routing'):
            if not (folder / subdir / f'epoch_{epoch:02d}.json').is_file():
                raise ValueError('Missing epoch FiLM/routing diagnostics')
    return dict(complete=True, best_epoch=best, metrics=rows)


def verify_reference():
    from engine.eduaigv import load_prepared
    config = load_prepared(REFERENCE)
    if yaml.safe_load((REFERENCE / 'config.yaml').read_text()) != config or 'repeat' in config['eduaigv']:
        raise ValueError('Historical seed42 configuration changed')
    result = verify_results(REFERENCE, config, checkpoints=True)
    initial, preflight = read(REFERENCE / 'initialization_audit.json'), read(REFERENCE / 'preflight.json')
    if (not preflight['passed'] or len(preflight['steps']) != 2
            or preflight['checkpoint_reload_max_abs_delta'] > 1e-6
            or initial['trained_checkpoint_loaded'] or initial['seed'] != 42
            or initial['initial_state_sha256'] != preflight['initial_state_sha256']):
        raise ValueError('Historical initialization/preflight audit failed')
    # Code has evolved; preserve its historical hash records, but require unchanged inputs.
    for key, value in read(REFERENCE / 'sources.json').items():
        if not key.startswith('implementation/') and file_digest(value['path']) != value['sha256']:
            raise ValueError('Historical data/pretrained input changed: ' + key)
    hashes = {str(p.relative_to(REFERENCE)): file_digest(p) for p in sorted(REFERENCE.rglob('*'))
              if p.is_file() and not p.name.startswith('.')}
    return config, dict(verified=True, source=str(REFERENCE), readonly=True,
        hashes=hashes, summary=result, initialization=initial, preflight=preflight)


def audit_cache(config):
    """Never rebuild: verify all 1130 existing arrays and their original video identities."""
    manifest = Path(config['data']['artifact']['readonly_manifest'])
    if file_digest(manifest) != config['eduaigv']['cache_manifest_sha256']:
        raise ValueError('Prior manifest changed')
    saved = read(manifest)
    split = read(config['eduaigv']['split_manifest'])
    source = Path(config['eduaigv']['split_manifest']).parent / split['source']
    ids = {r['id'] for r in map(json.loads, source.read_text().splitlines())}
    if not saved['readonly'] or len(saved['files']) != 1130 or {v['video_id'] for v in saved['files'].values()} != ids:
        raise ValueError('Prior cache does not cover the exact 1130 source videos')
    for i, (path, entry) in enumerate(saved['files'].items(), 1):
        if file_digest(path) != entry['sha256']:
            raise ValueError('Prior content changed: ' + path)
        value = np.load(path, allow_pickle=False, mmap_mode='r')
        if value.shape != (8, 448, 448) or value.dtype != np.float32 or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError('Invalid Prior array: ' + path)
        stat = Path(entry['video']).stat()
        if stat.st_size != entry['video_size'] or stat.st_mtime_ns != entry['video_mtime_ns']:
            raise ValueError('Prior source video changed: ' + entry['video'])
        if i % 200 == 0 or i == 1130:
            print(f'Verified read-only Prior cache: {i}/1130', flush=True)
    load_two_task_table(config['task'])
    return dict(verified=True, videos=1130, rebuilt=0, manifest=str(manifest), sha256=file_digest(manifest),
                task_table=config['task']['table_path'], task_table_sha256=config['task']['table_sha256'])


def audit_initializations(root):
    """Construct every new model on CPU before dispatch; never load a trained checkpoint."""
    import gc
    from engine.audit import digest
    from engine.eduaigv import load_prepared
    from utils.initialization import initialize_model
    torch.set_num_threads(2)
    expected = read(REFERENCE / 'initialization_audit.json')
    reports = {}
    for seed in SEEDS[1:]:
        config = load_prepared(Path(root) / f'seed{seed}')
        model = initialize_model(config)
        state = digest(model.compact_state_dict())
        if (state != expected['initial_state_sha256']
                or model.construction_rng_sha256 != expected['eduaigv']['construction_rng_sha256']
                or model.task_embedding_hashes != expected['task_embedding_hashes']):
            raise AssertionError('Initial model/table/RNG mismatch for seed ' + str(seed))
        reports[str(seed)] = dict(initial_state_sha256=state, construction_rng_sha256=model.construction_rng_sha256,
            task_embedding_hashes=model.task_embedding_hashes, trained_checkpoint_loaded=False)
        write_json(Path(root) / f'seed{seed}/preparation/initialization_audit.json', reports[str(seed)])
        del model
        gc.collect()
        print(f'Initial model, task table and construction RNG match seed42: split seed{seed}', flush=True)
    write_json(Path(root) / 'initialization_audit.json', reports)


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'status.json', dict(stage='preparing', pid=os.getpid()))
    try:
        base, reuse = verify_reference()
        write_json(root / 'seed42/reuse.json', reuse)
        write_json(root / 'seed42/summary.json', reuse['summary'])
        write_json(root / 'seed42/status.json', dict(stage='reused_verified'))
        write_json(root / 'cache_audit.json', audit_cache(base))
        original_split = read(base['eduaigv']['split_manifest'])
        source = (Path(base['eduaigv']['split_manifest']).parent / original_split['source']).resolve()
        rows = list(map(json.loads, source.read_text().splitlines()))
        before = torch.get_rng_state().clone()
        test_sets, quotas, prompts, configs, audits = {}, [], [], {}, {}
        for seed in SEEDS:
            folder = root / f'seed{seed}'
            folder.mkdir(exist_ok=seed == 42)
            train, test, counts, prompt = split_records(rows, seed)
            for name, subset in (('train', train), ('test', test)):
                (folder / f'{name}.jsonl').write_bytes(manifest_bytes(subset))
            meta = dict(protocol=PROTOCOL, seed=seed, source=str(source), source_sha256=file_digest(source),
                train_sha256=file_digest(folder / 'train.jsonl'), test_sha256=file_digest(folder / 'test.jsonl'),
                generator_counts=counts, prompt_statistics=prompt)
            write_json(folder / 'split.json', meta)
            if seed == 42:
                if any(meta[k] != original_split[k] for k in ('train_sha256', 'test_sha256')):
                    raise ValueError('seed42 fails byte-exact historical manifest reproduction')
                audits[str(seed)] = validate_data(base)
            else:
                config = make_config(base, root, seed)
                audits[str(seed)] = validate_data(config)
                text = yaml.safe_dump(config, sort_keys=False)
                (folder / 'config.yaml').write_text(text)
                (folder / 'preparation').mkdir()
                (folder / 'preparation/config.yaml').write_text(text)
                write_json(folder / 'preparation/complete.json', dict(videos=1130,
                    config_sha256=file_digest(folder / 'preparation/config.yaml')))
                # Summary needs these statistics; no tokenizer or cache reconstruction is necessary.
                write_json(folder / 'preparation/dataset_statistics.json', dict(prompt_overlap=prompt,
                    labels={name: {t: dict(mean=float(np.mean([r['labels'][t] for r in subset])),
                        std=float(np.std([r['labels'][t] for r in subset]))) for t in TASKS}
                        for name, subset in (('train', train), ('test', test))}))
                write_json(folder / 'status.json', dict(stage='prepared'))
                configs[str(seed)] = file_digest(folder / 'preparation/config.yaml')
            test_sets[seed] = {r['id'] for r in test}
            quotas.extend(dict(split_seed=seed, **r) for r in counts)
            prompts.append(dict(split_seed=seed, **prompt))
        if not torch.equal(before, torch.get_rng_state()):
            raise AssertionError('Split generation consumed model RNG')
        write_csv(root / 'generator_quotas.csv', quotas)
        write_csv(root / 'prompt_overlap.csv', prompts)
        write_csv(root / 'test_coverage.csv', [dict(video_id=r['id'], test_count=sum(r['id'] in s for s in test_sets.values())) for r in rows])
        write_csv(root / 'test_overlap.csv', [dict(seed_a=a, seed_b=b, intersection=len(test_sets[a] & test_sets[b]),
            jaccard=len(test_sets[a] & test_sets[b]) / len(test_sets[a] | test_sets[b])) for a in SEEDS for b in SEEDS])
        write_json(root / 'data_audit.json', dict(splits=audits, split_preserves_model_rng=True,
            model_seed=42, epoch_sampling_seeds=list(range(9042, 9052)), config_hashes=configs))
        audit_initializations(root)
        write_json(root / 'preparation_complete.json', dict(protocol=PROTOCOL,
            pinned_files={str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()
                          and p.name != 'status.json'}))
        write_json(root / 'status.json', dict(stage='prepared', pid=os.getpid()))
        summarize(root)
        return root
    except Exception as error:
        write_json(root / 'status.json', dict(stage='preparation_failed', error=repr(error)))
        raise


def load_config(root, seed):
    from engine.eduaigv import load_prepared
    if seed not in SEEDS[1:]:
        raise ValueError('seed42 is historical read-only reuse, never train it')
    root = Path(root).resolve()
    pinned = read(root / 'preparation_complete.json')['pinned_files']
    relative = f'seed{seed}/preparation/config.yaml'
    if file_digest(root / relative) != pinned[relative]:
        raise ValueError('Prepared split configuration changed')
    return load_prepared(root / f'seed{seed}')


def aggregate_metrics(rows):
    return [dict(checkpoint=kind, task=task, metric=metric, n=len(values),
        mean=statistics.mean(values) if values else None,
        std_ddof1=statistics.stdev(values) if len(values) > 1 else None)
        for kind in ('best', 'last') for task in (*TASKS, 'macro') for metric in METRICS
        for values in [[r[metric] for r in rows if r['checkpoint'] == kind and r['task'] == task]]]


def summarize_split(folder):
    folder = Path(folder)
    config = load_config(folder.parent, int(folder.name.removeprefix('seed')))
    report = verify_results(folder, config, checkpoints=True)
    report.update(protocol=config['eduaigv']['repeat'], interpretation=CAVEATS,
        checkpoint_hashes={kind: file_digest(folder / f'{kind}.pt') for kind in ('best', 'last')})
    write_csv(folder / 'metrics.csv', report['metrics'])
    write_json(folder / 'summary.json', report)
    return report


def summarize(root):
    root = Path(root)
    statuses, completed, rows = {}, [], []
    for seed in SEEDS:
        folder = root / f'seed{seed}'
        try:
            status = read(folder / 'status.json') if (folder / 'status.json').exists() else dict(stage='pending')
        except (ValueError, OSError) as error:
            statuses[str(seed)] = dict(stage='invalid_status', error=repr(error))
            continue
        statuses[str(seed)] = status
        if status['stage'] not in ('complete', 'reused_verified'):
            continue
        try:
            if seed == 42:
                pinned = read(root / 'preparation_complete.json')['pinned_files']
                if file_digest(folder / 'reuse.json') != pinned['seed42/reuse.json']:
                    raise ValueError('Historical reuse snapshot changed')
                reuse = read(folder / 'reuse.json')
                if not reuse['verified'] or reuse['source'] != str(REFERENCE):
                    raise ValueError('Unverified historical reuse')
                for name, sha in reuse['hashes'].items():
                    if file_digest(REFERENCE / name) != sha:
                        raise ValueError('Historical reuse source changed: ' + name)
                result = reuse['summary']
            else:
                config = load_config(root, seed)
                result = verify_results(folder, config)
                accepted = read(folder / 'summary.json')
                if not accepted['complete'] or set(accepted['checkpoint_hashes']) != {'best', 'last'} or any(file_digest(folder / f'{k}.pt') != v
                                                  for k, v in accepted['checkpoint_hashes'].items()):
                    raise ValueError('Split checkpoint acceptance missing or changed')
            rows.extend(dict(split_seed=seed, reused=seed == 42, **r) for r in result['metrics'])
            completed.append(seed)
        except Exception as error:
            statuses[str(seed)] = dict(stage='invalid_result', error=repr(error))
    if rows:
        write_csv(root / 'per_split_metrics.csv', rows)
    write_csv(root / 'summary.csv', aggregate_metrics(rows))
    report = dict(complete=completed == list(SEEDS), completed_count=len(completed), completed_seeds=completed,
        statuses=statuses, expected_splits=10, interpretation=CAVEATS,
        aggregation='Partial aggregates use completed splits only; ddof=1; not a ten-split result until all ten pass.')
    write_json(root / 'summary.json', report)
    (root / 'report.md').write_text('# EduAIGV repeated hold-out\n\n'
        f'Completed: {len(completed)}/10. Ten-split results complete: {report["complete"]}.\n\n'
        + '\n\n'.join(CAVEATS) + '\n\nMetrics: per_split_metrics.csv; mean/sample SD: summary.csv.\n'
        'Data and configuration hashes: data_audit.json, preparation_complete.json, seed*/split.json.\n'
        'Overlap: test_coverage.csv, test_overlap.csv, prompt_overlap.csv.\n')
    return report


def free_slots(gpus, active):
    return sorted(set(gpus) - {job['gpu'] for job in active.values()})


def launch(root, gpus):
    root = Path(root).resolve()
    if not root.exists():
        prepare(root)
    pinned = read(root / 'preparation_complete.json')
    for name, sha in pinned['pinned_files'].items():
        if file_digest(root / name) != sha:
            raise ValueError('Prepared artifact changed: ' + name)
    if not summarize(root)['completed_seeds'] == [42]:
        raise ValueError('Expected verified seed42 and nine unstarted splits')
    with (root / '.launch_claim').open('x') as handle:
        handle.write(str(os.getpid()))
    jobs, active, queue = {}, {}, list(SEEDS[1:])
    write_json(root / 'status.json', dict(stage='running', manager_pid=os.getpid(), gpus=gpus))
    while queue or active:
        for seed, job in list(active.items()):
            code = job['process'].poll()
            if code is None:
                continue
            path = root / f'seed{seed}/status.json'
            try:
                status = read(path) if path.exists() else {}
            except (ValueError, OSError) as error:
                status = dict(stage='invalid_status', error=repr(error))
            complete = code == 0 and status.get('stage') == 'complete'
            jobs[seed].update(state='complete' if complete else 'failed', exit_code=code, finished_at=time.time())
            if not complete:
                write_json(path, dict(stage='failed', exit_code=code, worker_status=status))
            del active[seed]
        for gpu in free_slots(gpus, active):
            if not queue:
                break
            seed = queue.pop(0)
            command = [sys.executable, '-u', '-B', str(ROOT / 'scripts/train_eduaigv_splits.py'),
                '--mode', 'train', '--split-seed', str(seed), '--gpu', str(gpu), '--output-root', str(root)]
            log = root / f'worker_seed{seed}.log'
            jobs[seed] = dict(gpu=gpu, command=command, log=str(log), state='starting', started_at=time.time())
            try:
                with log.open('x') as handle:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}, start_new_session=True)
                jobs[seed].update(pid=process.pid, state='running')
                active[seed] = dict(process=process, gpu=gpu)
            except Exception as error:
                jobs[seed].update(state='failed', error=repr(error))
                write_json(root / f'seed{seed}/status.json', dict(stage='failed', error=repr(error)))
        write_json(root / 'scheduler.json', dict(manager_pid=os.getpid(), jobs=jobs, pending=queue, automatic_retry=False))
        if active:
            time.sleep(10)
    report = summarize(root)
    write_json(root / 'status.json', dict(stage='complete' if report['complete'] else 'finished_with_failures',
        completed_count=report['completed_count'], manager_pid=os.getpid()))
