"""VideoFeedback preparation and five-task reporting; no changes to the baseline split."""

import json
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
import yaml

from engine.audit import file_digest
from engine.io import write_json, write_csv
from utils.videofeedback import TASKS, validate_data, prepare_task_table, load_five_task_table


def records(config, split):
    key = 'train_manifest' if split == 'train' else 'val_manifest'
    return [json.loads(line) for line in Path(config['data'][key]).read_text().splitlines()]


def prepare_video(job):
    """Decode every selected frame, including when a valid cache already exists."""
    import cv2
    import torch
    from dataset.artifact_prior import ArtifactCache, ArtifactPriorExtractor
    from dataset.frame_sampling import read_anchor_pairs
    from dataset.video_dataset import _center_crop_resize
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    row, data = job
    video = (Path(data['video_root']) / row['video']).resolve()
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError('Missing/empty video: ' + str(video))
    options = dict(data['artifact'])
    cache = ArtifactCache(options.pop('cache_dir'))
    options.pop('readonly_manifest', None)
    extractor = ArtifactPriorExtractor(**options)
    frames, following, anchors, next_indices = read_anchor_pairs(
        video, num_frames=8, min_total_frames=8)
    path = cache.path_for(video, 8, 448, extractor)
    existed = path.exists()
    if existed and np.load(path, allow_pickle=False, mmap_mode='r').dtype != np.float32:
        raise ValueError('Existing Prior must be stored in FP32: ' + str(path))
    value = cache.load(path)
    if value is None:
        pair = _center_crop_resize(np.concatenate([frames, following]), 448)
        value = extractor(pair[:8], pair[8:])
    if (value.shape != (8, 448, 448) or not np.isfinite(value).all()
        or value.min() < 0 or value.max() > 1):
        raise ValueError('Invalid Prior: ' + str(video))
    if not existed:
        cache.save(path, value)
    stat = video.stat()
    return str(path.resolve()), dict(video_id=row['id'], video=str(video),
        video_size=stat.st_size, video_mtime_ns=stat.st_mtime_ns,
        sha256=file_digest(path), shape=list(value.shape), reused=existed,
        anchor_indices=list(map(int, anchors)), next_indices=list(map(int, next_indices)),
        extractor_fingerprint=extractor.fingerprint())


def dataset_statistics(config):
    from transformers import BertTokenizer
    # Same vocabulary and special tokens as BLIP's init_tokenizer.
    tokenizer = BertTokenizer.from_pretrained(
        config['model']['backbones']['blip']['bert_path'], local_files_only=True)
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    sets = {split: records(config, split) for split in ('train', 'test')}
    report = {}
    for split, rows in sets.items():
        lengths = [len(tokenizer(r['prompt'], truncation=False)['input_ids']) for r in rows]
        report[split] = dict(videos=len(rows), max50_truncated=sum(n > 50 for n in lengths),
            token_lengths=dict(min=min(lengths), max=max(lengths), mean=float(np.mean(lengths))),
            labels={t: dict(mean=float(np.mean([r['labels'][t] for r in rows])),
                std=float(np.std([r['labels'][t] for r in rows])),
                counts={str(v): sum(r['labels'][t] == v for r in rows) for v in (.2,.4,.6,.8)})
                for t in TASKS})
    train_prompts = {r['prompt'] for r in sets['train']}
    report['prompt_overlap'] = dict(exact_test_videos=sum(r['prompt'] in train_prompts for r in sets['test']),
        exact_unique_prompts=len(train_prompts & {r['prompt'] for r in sets['test']}))
    report['train_mean_predictor_test_rmse'] = {
        t: float(np.sqrt(np.mean([(r['labels'][t]-report['train']['labels'][t]['mean'])**2
                                  for r in sets['test']]))) for t in TASKS}
    report['label_policy'] = 'Original discrete 0.2/0.4/0.6/0.8 regression targets, no rescaling.'
    return report


def prepare(config):
    root = Path(config['videofeedback']['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'status.json', dict(stage='preparing', pid=os.getpid()))
    try:
        (root / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(root / 'preparation/data_audit.json', validate_data(config))
        prepare_task_table(config)
        (root / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(root / 'preparation/dataset_statistics.json', dataset_statistics(config))
        rows = records(config, 'train') + records(config, 'test')
        cache_root = Path(config['data']['artifact']['cache_dir'])
        cache_root.mkdir(parents=True, exist_ok=True)
        required = len(rows)*8*448*448*4
        # Conservative check, including existing entries; never remove caches to make space.
        free = shutil.disk_usage(cache_root).free
        if free < required + 10*1024**3:
            raise RuntimeError(f'Need approximately {required/1024**3:.1f} GiB Prior space plus 10 GiB reserve; free={free/1024**3:.1f} GiB')
        workers = config['videofeedback']['prepare_workers']
        files, iterator = {}, iter(rows)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {pool.submit(prepare_video, (r, config['data'])) for r in [next(iterator) for _ in range(workers)]}
            try:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        path, entry = future.result()
                        if path in files:
                            raise ValueError('Duplicate Prior identity: ' + path)
                        files[path] = entry
                        if len(files) <= workers or len(files) % 100 == 0 or len(files) == len(rows):
                            progress = dict(stage='preparing_prior', completed=len(files), total=len(rows),
                                            workers=workers, last_video=entry['video_id'], pid=os.getpid())
                            write_json(root / 'status.json', progress)
                            print(json.dumps(progress), flush=True)
                        row = next(iterator, None)
                        if row is not None:
                            pending.add(pool.submit(prepare_video, (row, config['data'])))
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        manifest = root / 'preparation/prior_manifest.json'
        write_json(manifest, dict(readonly=True, files=files, video_count=len(rows),
            freshly_generated=sum(not f['reused'] for f in files.values()),
            validation='All videos decoded; aligned 8-frame FP32 Priors; hashes checked on every training read.'))
        config['videofeedback']['cache_manifest_sha256'] = file_digest(manifest)
        text = yaml.safe_dump(config, sort_keys=False)
        (root / 'config.yaml').write_text(text)
        (root / 'preparation/config.yaml').write_text(text)
        write_json(root / 'preparation/complete.json', dict(config_sha256=file_digest(root / 'preparation/config.yaml'),
            videos=len(files), cache_manifest_sha256=file_digest(manifest)))
        write_json(root / 'status.json', dict(stage='prepared', videos=len(files), pid=os.getpid()))
        return config
    except Exception as error:
        write_json(root / 'status.json', dict(stage='preparation_failed', error=repr(error), pid=os.getpid()))
        raise


def load_prepared(root):
    root = Path(root)
    complete = json.loads((root / 'preparation/complete.json').read_text())
    path = root / 'preparation/config.yaml'
    if file_digest(path) != complete['config_sha256']:
        raise ValueError('Prepared configuration changed')
    config = yaml.safe_load(path.read_text())
    validate_data(config)
    load_five_task_table(config['task'])
    vf = config['videofeedback']
    if (complete['videos'] != 33294 or vf['output_root'] != str(root.resolve())
        or file_digest(config['data']['artifact']['readonly_manifest']) != vf['cache_manifest_sha256']
        or vf['task_table_sha256'] != config['task']['table_sha256']):
        raise ValueError('Prepared data/table/cache protocol mismatch')
    return config


def claim_training_output(config):
    root = Path(config['videofeedback']['output_root'])
    expected = load_prepared(root)
    if config != expected or any((root / name).exists() for name in ('history.json','best.pt','last.pt','preflight.json')):
        raise ValueError('Training state already exists or configuration changed')
    with (root / '.training_claim').open('x') as handle:
        handle.write(str(os.getpid()) + '\n')


def summarize(root):
    root = Path(root)
    config = load_prepared(root)
    history = json.loads((root / 'history.json').read_text())
    if len(history) != 10 or history[-1]['optimizer_updates'] != 203870:
        raise ValueError('Incomplete VideoFeedback training')
    best_epoch = max(range(10), key=lambda i: history[i]['test']['metrics']['macro']['srocc']) + 1
    expected = {(r['id'], t): r['labels'][t] for r in records(config, 'test') for t in TASKS}
    rows, distributions = [], {}
    for kind in ('best', 'last'):
        report = json.loads((root / f'{kind}_evaluation.json').read_text())
        samples = report['samples']
        actual = {(r['video_id'], r['task']): r['target'] for r in samples}
        if (not report.get('completed') or len(samples) != 3380 or actual.keys() != expected.keys()
            or any(abs(actual[k]-expected[k]) > 1e-6 for k in expected)
            or report['checkpoint_epoch'] != (best_epoch if kind == 'best' else 10)):
            raise ValueError('Evaluation does not match VideoFeedback test set/selection')
        distributions[kind] = {}
        for task in (*TASKS, 'macro'):
            rows.append(dict(checkpoint=kind, epoch=report['checkpoint_epoch'], task=task,
                **{m: report['metrics'][task][m] for m in ('plcc','srocc','krocc','rmse')}))
            if task == 'macro':
                continue
            values = [r for r in samples if r['task'] == task]
            y, p = (np.array([r[k] for r in values]) for k in ('target','prediction'))
            distributions[kind][task] = dict(target_mean=float(y.mean()), prediction_mean=float(p.mean()),
                target_std=float(y.std()), prediction_std=float(p.std()),
                std_ratio=float(p.std()/y.std()) if y.std() > 0 else None,
                bias=float((p-y).mean()))
    write_csv(root / 'metrics.csv', rows)
    write_json(root / 'prediction_distributions.json', distributions)
    summary = dict(complete=True, tasks=list(TASKS), train_videos=32618, test_videos=676,
        best_epoch=best_epoch, updates=203870, metrics=rows,
        dataset_statistics=json.loads((root / 'preparation/dataset_statistics.json').read_text()),
        caveats=['Single seed; the test set selects best and reports performance (selection bias).',
                 'Best and epoch-10 last are reported separately; no automatic baseline replacement.',
                 'Five-task recentering changes the original four task conditions; different dataset/labels prevent paired AIGVQA comparisons.'])
    write_json(root / 'summary.json', summary)
    return summary


def evaluate_checkpoint(root, checkpoint):
    import torch
    from utils.initialization import initialize_model
    from utils.provenance import selected_source_hashes
    from engine.checkpoint import load_model_checkpoint
    from engine.evaluation import evaluate, export
    from engine.memory import initialize_process_memory, MemoryWatch, check_memory
    from dataset.loader import build_loader
    config = load_prepared(root)
    folder = Path(root) / 'evaluations' / Path(checkpoint).stem
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
