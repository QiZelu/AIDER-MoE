"""Original-split contract: reject multiseed and other experimental checkpoints."""

import os
from pathlib import Path

import torch

from utils.provenance import validate_sources
from utils.experiment import checkpoint_contract, film_video_condition
from engine.gradients import gradient_clip_limit, validate_checkpoint_gradient_policy
from engine.losses import moe_mi_protocol

FROZEN_PRETRAINED = (
    "vital_backbone.vision_model.",
    "motion_backbone.feature_extraction.",
    "tv_backbone.model.",
)


def validate_checkpoint_mi_policy(payload, config):
    saved = moe_mi_protocol(payload['config'])
    if payload.get('moe_mi', saved) != saved:
        raise ValueError('Checkpoint MoE MI metadata mismatch')
    if saved != moe_mi_protocol(config):
        raise ValueError('Checkpoint MoE MI training policy mismatch')


def save_checkpoint(
    path, model, config, epoch, optimizer=None, scheduler=None, **metadata
):
    state = model.compact_state_dict()
    prefix = "tv_backbone.model.text_encoder."
    if not all(
        prefix + name in state
        for name in model.tv_backbone.model.text_encoder.state_dict()
    ):
        raise ValueError("Complete fine-tuned BLIP text encoder is required")
    contract = checkpoint_contract(config)
    if model.film_video_condition != film_video_condition(config):
        raise ValueError("Model/config FiLM mode mismatch")
    payload = dict(
        eduaigv=config.get('eduaigv'),
        head_rank=config.get('head_rank'),
        private_task_modules=config.get('private_task_modules'),
        videofeedback=config.get('videofeedback'),
        repeated_holdout=config.get('repeated_holdout'),
        gradient_clip=gradient_clip_limit(config),
        moe_mi=moe_mi_protocol(config),
        model_contract=contract,
        film_video_condition=model.film_video_condition,
        split=config["ablation"]["split"],
        ablation=config["ablation"],
        model=state,
        config=config,
        task_registry=model.task,
        task_embedding_hashes=model.task_embedding_hashes,
        source_hashes=model.source_hashes,
        initialization_audit=model.initialization_audit,
        sampling_sha256=getattr(model, "sampling_sha256", None),
        epoch=epoch,
        **metadata,
    )
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if getattr(model, 'frozen_private_heads', False) and 'head_rank' not in config:
        payload['frozen_head_source'] = config['frozen_head_source']
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model_checkpoint(model, path, config):
    contract = checkpoint_contract(config)
    mode = film_video_condition(config)
    if model.film_video_condition != mode:
        raise ValueError("Model/config FiLM mode mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    from utils.portable import normalize_checkpoint
    payload = normalize_checkpoint(payload, path)
    from utils.split_protocol import validate_checkpoint_protocol
    validate_checkpoint_protocol(payload, config)
    from utils.eduaigv_splits import validate_checkpoint_identity
    validate_checkpoint_identity(payload, config)
    if payload.get('head_rank') != config.get('head_rank'):
        raise ValueError('Wrong frozen Head ranking checkpoint identity')
    if payload.get('private_task_modules') != config.get('private_task_modules'):
        raise ValueError('Wrong private-task-module checkpoint identity')
    if payload.get('videofeedback') != config.get('videofeedback'):
        raise ValueError('Wrong VideoFeedback dataset/checkpoint identity')
    if (
        payload.get("model_contract") != contract
        or payload.get("film_video_condition", "task_video") != mode
        or payload.get("split") != config["ablation"]["split"]
        or payload.get("ablation") != config["ablation"]
    ):
        raise ValueError("Wrong variant or split checkpoint")
    validate_checkpoint_gradient_policy(payload, config)
    validate_checkpoint_mi_policy(payload, config)
    if getattr(model, 'frozen_private_heads', False) and 'head_rank' not in config:
        if payload.get('frozen_head_source') != config['frozen_head_source']:
            raise ValueError('Frozen private head source mismatch')
        fields = ('lr', 'weight_decay', 'module_lrs', 'head_l2_lambda', 'scheduler', 'warmup_steps')
        if any(payload['config']['training'][k] != config['training'][k] for k in fields):
            raise ValueError('Frozen private head training protocol mismatch')
    if (
        payload.get("task_registry") != model.task
        or payload.get("task_embedding_hashes") != model.task_embedding_hashes
        or payload["config"]["model"] != config["model"]
        or payload["config"]["task"] != config["task"]
    ):
        raise ValueError("Checkpoint model/task table mismatch")
    validate_sources(payload.get("source_hashes"), model.source_hashes, path)
    state, expected = payload["model"], model.compact_state_dict()
    if state.keys() != expected.keys():
        raise ValueError("Checkpoint tensor coverage mismatch")
    for name, value in expected.items():
        saved = state[name]
        if saved.shape != value.shape or saved.dtype != value.dtype:
            raise ValueError("Checkpoint tensor shape/dtype mismatch: " + name)
        if name.startswith("dimension_encoder.") or name in (
            "fusion.position_encoding",
            "blip_temporal.position_encoding",
        ):
            if not torch.equal(value.cpu(), saved.cpu()):
                raise ValueError("Frozen task/position table mismatch: " + name)
    missing, extra = model.load_state_dict(state, strict=False)
    if extra or any(not name.startswith(FROZEN_PRETRAINED) for name in missing):
        raise ValueError("Unexpected missing/extra checkpoint state")
    return payload
