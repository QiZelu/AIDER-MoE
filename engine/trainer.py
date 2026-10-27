"""训练主流程：构建 → 预检 → 重新初始化 → 逐轮训练／评估 → best/last 导出。"""

import gc
import json
from pathlib import Path

import torch
import yaml

from utils.initialization import initialize_model
from utils.experiment import validate_data
from utils.provenance import selected_source_hashes
from engine.routing import RouteCounts
from engine.routing import task_routes, export_encoding
from dataset.loader import build_loader
from engine.audit import digest, parameter_counts
from engine.checkpoint import save_checkpoint, load_model_checkpoint
from engine.evaluation import evaluate, export
from engine.io import write_json
from engine.losses import quality_training_loss, moe_mi_weight, moe_mi_protocol
from engine.gradients import gradient_clip_limit, prepare_gradients, gradient_statistics
from engine.memory import initialize_process_memory, MemoryWatch, check_memory
from engine.optim import build_optimizer, _scheduler
from engine.preflight import preflight, check_precision
from engine.reporting import gap_diagnostics, summarize
from engine.runtime import autocast_context, move_to_device
from models.registry import BRANCHES, TASKS


def finetune_epoch(model, loader, optimizer, scheduler, device, epoch, config, watcher):
    """一轮训练。独立记录实际采样顺序和两层专家的使用情况。"""
    loader.batch_sampler.set_epoch(epoch)
    tasks = tuple(config.get('task', {}).get('order', TASKS))
    mi_enabled = moe_mi_weight(config) > 0
    counters = RouteCounts(model)
    indices, compositions, shared_gradients = [], [], []
    gradient_records = []
    private_module_steps = {}
    loss_sum = 0.0
    loss_records = []
    private_heads = list(model.heads()) if getattr(model, 'private_heads', False) else []
    head_update_steps = {name: 0 for name, _ in private_heads}

    for step, cpu_batch in enumerate(loader, 1):
        model.train(True)
        batch = move_to_device(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            result = model(batch)
        loss, sample_losses, loss_record = quality_training_loss(model, result, batch, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss: epoch={epoch+1}, step={step}')
        loss.backward()
        for name, head in private_heads:
            active = name == 'shared' or bool((batch['task_id'] == tasks.index(name)).any())
            has_gradient = any(p.grad is not None for p in head.parameters())
            if has_gradient != active:
                raise AssertionError('Head gradient does not match task presence: ' + name)
            head_update_steps[name] += int(has_gradient)

        shared_gradients.append(
            [
                sum(
                    float(parameter.grad.detach().float().square().sum())
                    for parameter in block.shared_expert.parameters()
                    if parameter.grad is not None
                )
                ** 0.5
                for block in model.fusion.blocks
            ]
        )
        gradient_record = prepare_gradients(model, config, f'epoch={epoch+1}, step={step}')
        if getattr(model, 'private_task_modules', False):
            from engine.private_task_diagnostics import check_private_activity, module_gradient_norms
            for name, active in check_private_activity(
                model, batch['task_id'], all_task_gates=mi_enabled
            ).items():
                private_module_steps[name] = private_module_steps.get(name, 0) + int(active)
            gradient_record['module_norms'] = module_gradient_norms(model)
        if gradient_record is not None:
            gradient_record.update(epoch=epoch+1, step=step,
                                   update=epoch*len(loader)+step)
            gradient_records.append(gradient_record)
        optimizer.step()
        scheduler.step()
        check_memory(watcher.report())

        counters.add(batch["task_id"], result["routing"])
        indices.append(cpu_batch["pair_index"].tolist())
        compositions.append(
            [int((batch["task_id"] == task).sum()) for task in range(len(tasks))]
        )
        loss_sum += float(sample_losses.detach().sum())
        if loss_record:
            loss_records.append(dict(step=step, samples=len(sample_losses), **loss_record))
        print(
            json.dumps(
                {
                    "split": config["ablation"]["split"],
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": float(loss.detach()),
                    **loss_record,
                    **({'gradients': gradient_record} if gradient_record is not None else {}),
                }
            ),
            flush=True,
        )
        del result, loss, sample_losses

    if indices != loader.batch_sampler.batches():
        raise AssertionError("Actual sample order differs from sampler")
    check_precision(model, optimizer)
    training = {
        "train_loss": loss_sum / len(loader.dataset),
        "pair_indices": indices,
        "batch_task_counts": compositions,
        "order_sha256": loader.batch_sampler.digest(),
        "route_counts": counters.report(),
        "shared_gradients": shared_gradients,
        "head_update_steps": head_update_steps,
        "gradient_records": gradient_records,
        "private_module_steps": private_module_steps,
    }
    if loss_records:
        # Sample-weighted epoch means match the legacy regression statistic.
        fields = ('regression_loss', 'mi_loss', 'mutual_information',
                  'weighted_mi_loss', 'total_loss')
        count = sum(record['samples'] for record in loss_records)
        summary = {key: sum(r[key] * r['samples'] for r in loss_records) / count
                   for key in fields}
        summary['regression_loss'] = training['train_loss']
        summary['mi_by_layer'] = {
            name: sum(r['mi_by_layer'][name] * r['samples'] for r in loss_records) / count
            for name in loss_records[0]['mi_by_layer']
        }
        training['loss_diagnostics'] = dict(
            protocol=moe_mi_protocol(config), epoch_reduction='sample_mean',
            **summary, steps=loss_records,
        )
    return training


def save_epoch_diagnostics(output, epoch, model, training, routes):
    """诊断与训练计算分开，保持历史输出字段和统计方式。"""
    if training.get('loss_diagnostics'):
        write_json(output / 'losses' / f'epoch_{epoch+1:02d}.json', training['loss_diagnostics'])
    if training.get('gradient_records'):
        write_json(output / 'gradients' / f'epoch_{epoch+1:02d}.json', dict(
            summary=gradient_statistics(training['gradient_records']),
            steps=training['gradient_records']))
    report = training["route_counts"]
    if getattr(model, 'private_task_modules', False):
        from engine.private_task_diagnostics import task_coefficients
        report['private_module_gradient_steps'] = training['private_module_steps']
        write_json(output / 'film' / f'epoch_{epoch+1:02d}_task_coefficients.json', task_coefficients(model))
    report["end_of_epoch_task_routes"] = task_routes(model)
    if getattr(model, 'private_heads', False) and not model.frozen_private_heads:
        report["head_optimizer_steps"] = training["head_update_steps"]
    report["always_on_ffn_grad_norm_mean_by_layer"] = (
        torch.tensor(training["shared_gradients"]).mean(0).tolist()
    )
    report["always_on_ffn_output_norm_mean_by_layer"] = [
        sum(float(record["pools"][name]["shared_output_norm"]) for record in routes)
        / len(routes)
        for name, _ in model.pools()
    ]
    report["routed_ffn_output_norm_mean_by_layer"] = [
        sum(float(record["pools"][name]["routed_output_norm"]) for record in routes)
        / len(routes)
        for name, _ in model.pools()
    ]
    write_json(output / "moe_routing" / f"epoch_{epoch + 1:02d}.json", report)
    write_json(output / "moe_routing" / f"epoch_{epoch + 1:02d}_records.json", routes)
    context_key = "cross_branch_input" if model.film_video_condition == "cross_branch_linear" else "video_summary"
    write_json(
        output / "film" / f"epoch_{epoch + 1:02d}.json",
        {
            "film_video_condition": model.film_video_condition,
            "alpha_mean": sum(record["alpha"] for record in routes) / len(routes),
            context_key + "_norm_mean_by_branch": {
                branch: sum(
                    sum(x * x for x in record[context_key][branch]) ** 0.5
                    for record in routes
                )
                / len(routes)
                for branch in BRANCHES
            },
            "effective_modulation_norm_mean_by_branch": {
                branch: sum(
                    record["film"][branch]["effective_gamma_norm"] for record in routes
                )
                / len(routes)
                for branch in BRANCHES
            },
        },
    )
    if model.film_video_condition == "cross_branch_linear" or len(model.task_order) != 4 or getattr(model,'private_task_modules',False):
        grouped = {}
        for task in model.task_order:
            subset = [r for r in routes if r["task"] == task]
            grouped[task] = {
                "alpha_mean": sum(r["alpha"] for r in subset) / len(subset),
                "branches": {b: {k: sum(r["film"][b][k] for r in subset) / len(subset)
                                  for k in subset[0]["film"][b]} for b in BRANCHES},
            }
        write_json(output / "film" / f"epoch_{epoch + 1:02d}_by_task.json", grouped)
    write_json(
        output / "sampling" / f"epoch_{epoch + 1:02d}.json",
        {
            key: training[key]
            for key in ("pair_indices", "batch_task_counts", "order_sha256")
        },
    )


def train(config, mode="train"):
    moe_mi_weight(config)  # Fail before creating output for invalid/frozen-Gate runs.
    if 'head_rank' in config:
        from engine.head_rank import train_head_rank
        return train_head_rank(config, mode)
    gradient_clip_limit(config)
    if config['model'].get('head_policy') == 'frozen_private':
        from engine.frozen_heads import train_frozen_heads
        return train_frozen_heads(config, mode)
    options = config["training"]
    output = Path(options["output_dir"])
    if 'eduaigv' in config and output.resolve() == Path(config['eduaigv']['output_root']).resolve():
        from engine.eduaigv import claim_training_output
        claim_training_output(config)
    elif 'videofeedback' in config and output.resolve() == Path(config['videofeedback']['output_root']).resolve():
        from engine.videofeedback import claim_training_output
        claim_training_output(config)
    else:
        output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    torch.set_num_threads(options["cpu_threads"])
    watcher = None

    try:
        write_json(
            output / "status.json",
            {"stage": "initializing", "split": config["ablation"]["split"]},
        )
        initialize_process_memory()
        torch.cuda.set_per_process_memory_fraction(
            14 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
        )
        with MemoryWatch() as watcher:
            # 预检使用临时更新，不能进入正式训练。
            write_json(output / "data_audit.json", validate_data(config))
            sources = selected_source_hashes(config)
            model = initialize_model(config).to(options["device"])
            model.source_hashes = sources
            export_encoding(model, output)
            write_json(output / "sources.json", sources)
            write_json(output / "initialization_audit.json", model.initialization_audit)
            write_json(output / "parameters.json", parameter_counts(model))
            write_json(output / "moe_routing" / "initial.json", task_routes(model))
            if getattr(model, 'private_task_modules', False):
                from engine.private_task_diagnostics import task_coefficients
                write_json(output / 'film' / 'initial_task_coefficients.json', task_coefficients(model))
            write_json(output / "status.json", {"stage": "preflight"})
            expected_rng = model.construction_rng_sha256
            if 'repeated_holdout' in config:
                from utils.split_protocol import audit_initialization
                audit_initialization(model, config, digest(model.compact_state_dict()))
            expected_state, orders = preflight(model, config, output, watcher)
            if 'repeated_holdout' in config:
                write_json(output / 'paired_control_audit.json',
                           audit_initialization(model, config, expected_state, orders))
            if config['ablation'].get('gradient_clipping') == 'disabled' and 'private_task_modules' not in config:
                from engine.audit import file_digest
                control = Path(config['ablation']['control'])
                if file_digest(control / 'config.yaml') != config['ablation']['control_config_sha256']:
                    raise ValueError('No-clip control configuration changed')
                saved = json.loads((control / 'preflight.json').read_text())
                audit = json.loads((control / 'initialization_audit.json').read_text())
                if (expected_state != saved['initial_state_sha256']
                    or orders != saved['sampling_sha256']
                    or expected_rng != audit['original_split']['construction_rng_sha256']
                    or model.task_embedding_hashes != audit['task_embedding_hashes']):
                    raise AssertionError('No-clip initialization/RNG/sampling differs from control')
                write_json(output / 'paired_control_audit.json', dict(
                    initial_state_equal=True, construction_rng_equal=True,
                    sampling_equal=True, task_table_equal=True,
                    control=str(control), gradient_clip=None))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if mode == "preflight":
                write_json(output / "status.json", {"stage": "preflight_complete"})
                return

            # 重新构造未训练模型，确认初始化和 RNG 与预检之前一致。
            write_json(
                output / "status.json", {"stage": "reinitializing_after_preflight"}
            )
            model = initialize_model(config).to(options["device"])
            model.source_hashes = sources
            model.sampling_sha256 = orders
            if model.construction_rng_sha256 != expected_rng:
                raise AssertionError("Reconstruction RNG changed")
            if digest(model.compact_state_dict()) != expected_state:
                raise AssertionError("Preflight updates entered formal training")
            check_precision(model)

            train_loader = build_loader(config, "train")
            val_loader = build_loader(config, "val")
            expected_eval = 3380 if 'videofeedback' in config else 460
            if len(train_loader) != config['ablation']['steps_per_epoch'] or len(val_loader.dataset) != expected_eval:
                raise AssertionError("Unexpected train/eval batches")
            device = torch.device(options["device"])
            optimizer = build_optimizer(model, config)
            total_steps = len(train_loader) * options["epochs"]
            scheduler = _scheduler(
                optimizer,
                total_steps,
                options["warmup_ratio"],
                warmup_steps=options["warmup_steps"],
            )

            history = []
            best_score = -float("inf")
            for epoch in range(options["epochs"]):
                write_json(
                    output / "status.json",
                    {
                        "stage": "training",
                        "split": config["ablation"]["split"],
                        "epoch": epoch + 1,
                        "completed_epochs": epoch,
                    },
                )
                training = finetune_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                    config,
                    watcher,
                )
                if training["order_sha256"] != orders[epoch]:
                    raise AssertionError("Formal sampler order changed")
                rows, metrics, routes = evaluate(model, val_loader, device)
                save_epoch_diagnostics(output, epoch, model, training, routes)
                if 'eduaigv' in config:
                    from engine.eduaigv import score_diagnostics
                    write_json(output / 'score_diagnostics' / f'epoch_{epoch+1:02d}.json',
                               score_diagnostics(rows))
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": training["train_loss"],
                        "test": {"metrics": metrics},
                        "optimizer_updates": (epoch + 1) * len(train_loader),
                        **({'losses': {k: v for k, v in training['loss_diagnostics'].items()
                                       if k != 'steps'}} if 'loss_diagnostics' in training else {}),
                    }
                )
                write_json(output / "history.json", history)
                save_checkpoint(
                    output / "last.pt",
                    model,
                    config,
                    epoch,
                    optimizer,
                    scheduler,
                    selection_metric="test_macro_srocc",
                )
                score = metrics["macro"]["srocc"]
                if score is not None and score > best_score:
                    best_score = score
                    save_checkpoint(
                        output / "best.pt",
                        model,
                        config,
                        epoch,
                        score=score,
                        selection_metric="test_macro_srocc",
                    )
                write_json(output / "process_memory.json", watcher.report())

            if scheduler.last_epoch != total_steps:
                raise AssertionError("Incomplete scheduler updates")
            del optimizer, scheduler
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

            # best 与 last 分别重载并评估，不覆盖历史实验。
            for kind in ("best", "last"):
                payload = load_model_checkpoint(model, output / f"{kind}.pt", config)
                rows, metrics, routes = evaluate(model, val_loader, device)
                export(output, kind, rows, metrics, routes, payload["epoch"])
                if 'eduaigv' in config:
                    from engine.eduaigv import score_diagnostics
                    write_json(output / f'{kind}_score_diagnostics.json', score_diagnostics(rows))
                else:
                    write_json(output / f"{kind}_gap_diagnostics.json", gap_diagnostics(
                        rows, tasks=tuple(config['task']['order']), expected_videos=len(val_loader.dataset)//len(config['task']['order'])))
                del payload
            write_json(output / "process_memory.json", watcher.report())
            check_memory(watcher.report())
            write_json(
                output / "status.json",
                {
                    "stage": "complete",
                    "completed_epochs": options["epochs"],
                    "optimizer_updates": total_steps,
                    "split": config["ablation"]["split"],
                },
            )
            summarize(output)
    except Exception as error:
        write_json(
            output / "status.json",
            {
                "stage": "failed",
                "error": repr(error),
                "split": config["ablation"]["split"],
            },
        )
        if watcher is not None:
            write_json(output / "process_memory.json", watcher.report())
        raise
