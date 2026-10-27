"""EduAIGV双任务准备和报告，不更改原AIGVQA/VideoFeedback实验。"""

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
from engine.videofeedback import prepare_video as prepare_aligned_video
from utils.eduaigv import TASKS, validate_data, prepare_task_table, load_two_task_table


def records(config, split):
    key = 'train_manifest' if split == 'train' else 'val_manifest'
    return [json.loads(line) for line in Path(config['data'][key]).read_text().splitlines()]


def prepare_video(job):
    path, entry = prepare_aligned_video(job)
    value = np.load(path, allow_pickle=False, mmap_mode='r')
    if (value.shape != (8,448,448) or value.dtype != np.float32
            or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1):
        raise ValueError('Invalid stored EduAIGV Prior: ' + path)
    return path, entry


def dataset_statistics(config):
    from transformers import BertTokenizer
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
                minimum=min(r['labels'][t] for r in rows), maximum=max(r['labels'][t] for r in rows))
                for t in TASKS})
    train_prompts = {r['prompt'] for r in sets['train']}
    report['prompt_overlap'] = dict(
        exact_test_videos=sum(r['prompt'] in train_prompts for r in sets['test']),
        exact_unique_prompts=len(train_prompts & {r['prompt'] for r in sets['test']}))
    report['train_mean_predictor_test_rmse'] = {
        t: float(np.sqrt(np.mean([(r['labels'][t]-report['train']['labels'][t]['mean'])**2
                                  for r in sets['test']]))) for t in TASKS}
    report['label_policy'] = 'Original Overall and Text-video scores unchanged; no aggregation or rescaling.'
    return report


def prepare(config):
    root = Path(config['eduaigv']['output_root'])
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
        workers = config['eduaigv']['prepare_workers']
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
        config['eduaigv']['cache_manifest_sha256'] = file_digest(manifest)
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
    load_two_task_table(config['task'])
    edu = config['eduaigv']
    if (complete['videos'] != 1130 or edu['output_root'] != str(root.resolve())
        or file_digest(config['data']['artifact']['readonly_manifest']) != edu['cache_manifest_sha256']
        or edu['task_table_sha256'] != config['task']['table_sha256']):
        raise ValueError('Prepared data/table/cache protocol mismatch')
    return config


def claim_training_output(config):
    root = Path(config['eduaigv']['output_root'])
    expected = load_prepared(root)
    if config != expected or any((root / name).exists() for name in ('history.json','best.pt','last.pt','preflight.json')):
        raise ValueError('Training state already exists or configuration changed')
    with (root / '.training_claim').open('x') as handle:
        handle.write(str(os.getpid()) + '\n')


def score_diagnostics(rows):
    grouped = {}
    distributions = {}
    for r in rows:
        pair = grouped.setdefault(r['video_id'], {})
        if r['task'] in pair:
            raise ValueError('Duplicate video/task prediction')
        pair[r['task']] = r
    if len(rows) != 460 or len(grouped) != 230 or any(set(v) != set(TASKS) for v in grouped.values()):
        raise ValueError('Expected 230 videos and exactly two task predictions each')
    for task in TASKS:
        subset = [r for r in rows if r['task'] == task]
        y, p = (np.asarray([r[k] for r in subset]) for k in ('target', 'prediction'))
        distributions[task] = dict(target_mean=float(y.mean()), prediction_mean=float(p.mean()),
            target_std=float(y.std()), prediction_std=float(p.std()),
            std_ratio=float(p.std()/y.std()) if y.std() else None,
            rmse=float(np.sqrt(np.mean((p-y)**2))), bias=float((p-y).mean()))
    gaps = [dict(video_id=vid,
        true_gap=v['overall']['target']-v['text_video']['target'],
        predicted_gap=v['overall']['prediction']-v['text_video']['prediction'])
        for vid,v in grouped.items()]
    y = np.asarray([r['true_gap'] for r in gaps])
    p = np.asarray([r['predicted_gap'] for r in gaps])
    return dict(tasks=distributions, overall_minus_text_video=dict(
        true_abs_gap_mean=float(abs(y).mean()), predicted_abs_gap_mean=float(abs(p).mean()),
        gap_mae=float(abs(p-y).mean()),
        signed_gap_pearson=float(np.corrcoef(p,y)[0,1]) if p.std() and y.std() else None,
        records=gaps))


def summarize(root):
    root = Path(root)
    config = load_prepared(root)
    history = json.loads((root/'history.json').read_text())
    if len(history) != 10 or history[-1]['optimizer_updates'] != 2250:
        raise ValueError('Incomplete EduAIGV training')
    best_epoch = max(range(10), key=lambda i: history[i]['test']['metrics']['macro']['srocc'])+1
    expected = {(r['id'], t): r['labels'][t] for r in records(config,'test') for t in TASKS}
    metrics = []
    for kind in ('best','last'):
        report = json.loads((root/f'{kind}_evaluation.json').read_text())
        samples = report['samples']
        actual = {(r['video_id'],r['task']):r['target'] for r in samples}
        if (not report.get('completed') or len(samples) != 460 or actual.keys() != expected.keys()
                or any(abs(actual[k]-expected[k]) > 1e-6 for k in expected)
                or report['checkpoint_epoch'] != (best_epoch if kind == 'best' else 10)):
            raise ValueError('EduAIGV evaluation/selection mismatch')
        for task in (*TASKS,'macro'):
            metrics.append(dict(checkpoint=kind, epoch=report['checkpoint_epoch'], task=task,
                **{m:report['metrics'][task][m] for m in ('plcc','srocc','krocc','rmse')}))
        write_json(root/f'{kind}_score_diagnostics.json',score_diagnostics(samples))
    write_csv(root/'metrics.csv', metrics)
    summary = dict(complete=True, tasks=list(TASKS), train_videos=900, test_videos=230,
        best_epoch=best_epoch, updates=2250, metrics=metrics,
        dataset_statistics=json.loads((root/'preparation/dataset_statistics.json').read_text()),
        caveats=['Single seed; same test set selects best and reports scores (selection bias).',
                 'All 99 test prompts and all generators occur in training; no unseen-prompt/generator claim.',
                 'Two centered task vectors are opposite by construction, not semantic opposition.',
                 'Different labels/dataset: no paired numerical comparison to AIGVQA.',
                 'No automatic baseline replacement.'])
    write_json(root/'summary.json', summary)
    (root/'report.md').write_text(
        '# EduAIGV-1K 双任务训练结果\n\n'
        '模型从未训练seed42及原始预训练Backbone开始；不迁移AIGVQA已训练参数。'
        '双任务为整体质量Overall和整体文本—视频对齐Text-video；900/230视频，1800/460条任务记录。\n\n'
        '两层MoE Transformer、每层7个Top2路由专家+1个常开FFN，两个私有评分Head。'
        '仅Smooth L1；混合batch8，10轮，2250步，warmup116，梯度裁剪1。\n\n'
        '详见metrics.csv（best/last）、*_score_diagnostics.json（幅度及任务分差）、'
        'preparation/dataset_statistics.json（均值预测器与Prompt截断）、film/和moe_routing/。\n\n'
        '单seed，同一测试集选择best与报告成绩；测试集99种Prompt均在训练集中出现。'
        '两个中心化任务向量互为相反数，不表示语义相反。不宣称未见Prompt/生成器泛化，'
        '不与AIGVQA不同维度成绩作直接差值比较，不自动替换Baseline。\n')
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
