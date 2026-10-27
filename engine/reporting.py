"""Same-manifest best/last comparisons, never treating ten-split means as paired."""

import json
from pathlib import Path

import numpy as np
import yaml

from engine.audit import file_digest
from engine.io import write_csv, write_json
from models.registry import TASKS
from utils.experiment import ROOT, EXPECTED_HASHES

CONTROLS = {
    "selected_baseline": ROOT
    / "outputs/multitask/ablations/depth2_cross_branch_shared_expert_original_split_seed42",
}


def gap_diagnostics(rows, tasks=TASKS, expected_videos=115):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["video_id"], {})[row["task"]] = row
    if len(rows) != expected_videos * len(tasks) or len(grouped) != expected_videos or any(set(v) != set(tasks) for v in grouped.values()):
        raise ValueError("Expected complete task/video predictions")
    pairs = []
    for video, values in grouped.items():
        d, t = values["dynamic"], values["text_video"]
        pairs.append(
            dict(
                video_id=video,
                true_gap=d["target"] - t["target"],
                predicted_gap=d["prediction"] - t["prediction"],
                dynamic_target=d["target"],
                dynamic_prediction=d["prediction"],
                text_video_target=t["target"],
                text_video_prediction=t["prediction"],
            )
        )
    gt = np.array([r["true_gap"] for r in pairs])
    pred = np.array([r["predicted_gap"] for r in pairs])
    correlation = (
        float(np.corrcoef(gt, pred)[0, 1]) if gt.std() > 0 and pred.std() > 0 else None
    )
    return dict(
        videos=expected_videos,
        true_abs_gap_mean=float(abs(gt).mean()),
        predicted_abs_gap_mean=float(abs(pred).mean()),
        gap_mae=float(abs(gt - pred).mean()),
        signed_gap_pearson=correlation,
        large_gap_ge_0p1=[r for r in pairs if abs(r["true_gap"]) >= 0.1],
        records=pairs,
    )


def score_diagnostics(rows):
    """原尺度的幅度与 TV 误差，不用相关系数替代误差分析。"""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['video_id'], {})[row['task']] = row
    spread, errors = {}, []
    for task in TASKS:
        subset = [r for r in rows if r['task'] == task]
        target = np.asarray([r['target'] for r in subset], dtype=float)
        prediction = np.asarray([r['prediction'] for r in subset], dtype=float)
        order = np.argsort(target, kind='stable')
        count = max(1, int(np.ceil(len(order) * 0.2)))
        spread[task] = dict(
            target_mean=float(target.mean()), prediction_mean=float(prediction.mean()),
            target_std=float(target.std()), prediction_std=float(prediction.std()),
            std_ratio=float(prediction.std() / target.std()) if target.std() > 0 else None,
            low20_bias=float((prediction - target)[order[:count]].mean()),
            high20_bias=float((prediction - target)[order[-count:]].mean()),
        )
    for video, values in grouped.items():
        gt = {t: values[t]['target'] for t in TASKS}
        pred = {t: values[t]['prediction'] for t in TASKS}
        others = float(np.mean([gt[t] for t in TASKS if t != 'text_video']))
        error = pred['text_video'] - gt['text_video']
        errors.append(dict(video_id=video, targets=gt, predictions=pred,
                           text_video_error=error, text_video_abs_error=abs(error),
                           others_mean_minus_tv=others - gt['text_video']))
    errors.sort(key=lambda row: -row['text_video_abs_error'])
    subgroups = {}
    for name, subset in (
        ('tv_at_least_0p1_below_others', [r for r in errors if r['others_mean_minus_tv'] >= 0.1]),
        ('tv_at_least_0p1_above_others', [r for r in errors if r['others_mean_minus_tv'] <= -0.1]),
    ):
        subgroups[name] = dict(
            count=len(subset),
            tv_bias=float(np.mean([r['text_video_error'] for r in subset])) if subset else None,
            tv_mae=float(np.mean([r['text_video_abs_error'] for r in subset])) if subset else None,
        )
    return dict(task_score_spread=spread, text_video_error_ranked=errors,
                text_video_gap_subgroups=subgroups, dynamic_tv=gap_diagnostics(rows))


def summarize(output):
    output = Path(output)
    statuses, metrics_rows, differences, evaluations = {}, [], [], {}
    controls = dict(CONTROLS)
    own_config = yaml.safe_load((output / "config.yaml").read_text())
    if 'eduaigv' in own_config:
        if 'repeat' in own_config['eduaigv']:
            from engine.eduaigv_splits import summarize_split
            return summarize_split(output)
        from engine.eduaigv import summarize
        return summarize(output)
    if 'private_task_modules' in own_config:
        from engine.private_task_diagnostics import summarize
        return summarize(output)
    if 'videofeedback' in own_config:
        from engine.videofeedback import summarize
        return summarize(output)
    if 'repeated_holdout' in own_config:
        from engine.repeated_holdout import summarize_split
        return summarize_split(output)
    private = own_config['model'].get('head_policy') == 'private'
    no_clip = own_config['ablation'].get('gradient_clipping') == 'disabled'
    if own_config["model"].get("film_video_condition") == "cross_branch_linear" or private or no_clip:
        controls["video_only"] = ROOT / "outputs/multitask/runs/film_video_only_seed42"
    folders = {"combined": output, **controls}
    for name, folder in folders.items():
        status_path = folder / "status.json"
        statuses[name] = (
            json.loads(status_path.read_text())
            if status_path.exists()
            else dict(stage="not_started")
        )
        config_path = folder / "config.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        matching = (
            file_digest(config["data"]["train_manifest"]) == EXPECTED_HASHES["train"]
            and file_digest(config["data"]["val_manifest"]) == EXPECTED_HASHES["test"]
        )
        for kind in ("best", "last"):
            path = folder / f"{kind}_evaluation.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            if not matching:
                raise ValueError(
                    "Control no longer uses the original manifests: " + name
                )
            if not report.get("completed") or len(report["samples"]) != 460:
                raise ValueError("Incomplete evaluation: " + str(path))
            evaluations[name, kind] = report
            for task in (*TASKS, "macro"):
                metrics_rows.append(
                    dict(
                        experiment=name,
                        checkpoint=kind,
                        epoch=report["checkpoint_epoch"],
                        task=task,
                        **{
                            m: report["metrics"][task][m]
                            for m in ("plcc", "srocc", "krocc", "rmse")
                        },
                    )
                )
    for kind in ("best", "last"):
        if ("combined", kind) not in evaluations:
            continue
        actual = evaluations["combined", kind]
        write_json(
            output / f"{kind}_gap_diagnostics.json", gap_diagnostics(actual["samples"])
        )
        for control in controls:
            if (control, kind) not in evaluations:
                continue
            other = evaluations[control, kind]
            a = {(r["video_id"], r["task"]): r["target"] for r in actual["samples"]}
            b = {(r["video_id"], r["task"]): r["target"] for r in other["samples"]}
            if a != b:
                raise ValueError("Control target/video identity mismatch: " + control)
            for task in (*TASKS, "macro"):
                for metric in ("plcc", "srocc", "krocc", "rmse"):
                    x, y = (
                        actual["metrics"][task][metric],
                        other["metrics"][task][metric],
                    )
                    differences.append(
                        dict(
                            checkpoint=kind,
                            control=control,
                            task=task,
                            metric=metric,
                            combined=x,
                            reference=y,
                            delta=None if x is None or y is None else x - y,
                        )
                    )
    if metrics_rows:
        write_csv(output / "comparison_metrics.csv", metrics_rows)
    if differences:
        write_csv(output / "comparison_deltas.csv", differences)
    if private:
        write_json(output / 'private_head_score_diagnostics.json', {
            name + '_' + kind: score_diagnostics(report['samples'])
            for (name, kind), report in evaluations.items()
            if name in ('combined', 'video_only')
        })
    summary = dict(
        complete=statuses["combined"].get("stage") == "complete"
        and all(("combined", kind) in evaluations for kind in ("best", "last")),
        statuses=statuses,
        comparison="same original 461/115 split; best and last separate",
        interpretation=[
            "All models are single-seed; same held-out set selects best and reports scores.",
            "Seven routed experts are shared across tasks, not task-private expert pools.",
            "Reference is the preserved selected baseline; no archive lookup is performed.",
            "No automatic baseline replacement.",
        ],
    )
    if private:
        summary['interpretation'].append(
            'Joint training from seed42 untrained weights; only Head privatization changes. '
            'Additional Head capacity and isolated update paths are confounded; '
            'no explicit L2 loss or trained-checkpoint warm start.'
        )
    if no_clip:
        summary['direct_control'] = 'video_only'
        summary['interpretation'].append(
            'Only global gradient clipping is disabled relative to film_video_only; '
            'this does not establish resolution of task gradient conflicts or explain '
            'the entire historical reproduction gap. No automatic baseline replacement.'
        )
    write_json(output / "summary.json", summary)
    return summary
