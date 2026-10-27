"""当前运行的来源校验；只为明确固定的 Baseline best/last 保留重载许可。"""

import json
from pathlib import Path

from engine.audit import file_digest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/reference"


def selected_source_hashes(config):
    """校验本地常量、数据和预训练权重，不查找任何历史归档目录。"""
    reference = json.loads((REFERENCE / "verification.json").read_text())
    for name, expected in reference["assets_sha256"].items():
        if file_digest(REFERENCE / name) != expected:
            raise ValueError("Baseline reference asset changed: " + name)
    paths = {
        "initialization": config["training"]["initialization"],
        "historical_config": REFERENCE / "historical_config.yaml",
        "train_manifest": config["data"]["train_manifest"],
        "val_manifest": config["data"]["val_manifest"],
        "med_config": config["model"]["backbones"]["blip"]["med_config"],
    }
    for branch in ("blip", "slowfast"):
        paths[branch + "_pretrained"] = config["model"]["backbones"][branch][
            "checkpoint_path"
        ]
    folders = (
        ("vital", config["model"]["backbones"]["vital"]["model_dir"]),
        ("clip", config["model"]["dimension_encoder"]["model_dir"]),
        ("bert", config["model"]["backbones"]["blip"]["bert_path"]),
    )
    for branch, folder in folders:
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file() and path.suffix in (
                ".bin",
                ".safetensors",
                ".json",
                ".txt",
            ):
                paths[branch + "/" + str(path.relative_to(folder))] = path
    fixed = [
        ROOT / "artifacts/depth2_video_adaptive_fixed_position.pt",
        ROOT / "artifacts/depth2_video_adaptive_fixed_position.json",
        REFERENCE / "config.yaml",
        REFERENCE / "preflight.json",
        REFERENCE / "task_tables.json",
    ]
    for path in fixed:
        paths["fixed/" + path.name] = path
    for folder in ("models", "engine", "dataset", "utils"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            paths["implementation/" + str(path.relative_to(ROOT))] = path
    for name in ("train.py", "evaluate.py", "infer.py"):
        paths["implementation/" + name] = ROOT / name
    if 'repeated_holdout' in config:
        protocol = config['repeated_holdout']
        paths['split/cache_manifest'] = protocol['cache_manifest']
        for name in ('scripts/train_private_heads_splits.py', 'scripts/prepare_aigvqa_splits.py'):
            paths['implementation/' + name] = ROOT / name
    if 'eduaigv' in config:
        edu = config['eduaigv']
        paths['eduaigv/task_table'] = config['task']['table_path']
        paths['eduaigv/cache_manifest'] = config['data']['artifact']['readonly_manifest']
        paths['eduaigv/spec'] = edu['source_spec']
        paths['eduaigv/split'] = edu['split_manifest']
        paths['eduaigv/reference_config'] = ROOT / 'outputs/multitask/runs/film_video_only_private_heads_seed42/config.yaml'
        paths['implementation/scripts/train_eduaigv.py'] = ROOT / 'scripts/train_eduaigv.py'
        if 'repeat' in edu:
            paths['implementation/scripts/train_eduaigv_splits.py'] = ROOT / 'scripts/train_eduaigv_splits.py'
    if 'videofeedback' in config:
        vf = config['videofeedback']
        paths['videofeedback/task_table'] = config['task']['table_path']
        paths['videofeedback/cache_manifest'] = config['data']['artifact']['readonly_manifest']
        paths['videofeedback/spec'] = vf['source_spec']
        paths['implementation/scripts/train_videofeedback.py'] = ROOT / 'scripts/train_videofeedback.py'
    if 'private_task_modules' in config:
        control = Path(config['private_task_modules']['control'])
        for name in ('config.yaml','preflight.json','initialization_audit.json'):
            paths['private_task_modules/control/' + name] = control / name
    # 将新校验常量也纳入新检查点的来源记录。
    paths["implementation/artifacts/reference/verification.json"] = (
        REFERENCE / "verification.json"
    )
    return {
        key: {"path": str(path), "sha256": file_digest(path)}
        for key, path in paths.items()
    }


def validate_sources(saved, current, checkpoint):
    if saved == current:
        return
    from utils.portable import validate_export_sources
    if validate_export_sources(saved, current, checkpoint):
        return
    pinned = json.loads((REFERENCE / "checkpoint_sources.json").read_text())
    if saved != pinned:
        raise ValueError(
            "Checkpoint sources differ from current code and the selected baseline"
        )
    reference = json.loads((REFERENCE / "verification.json").read_text())
    if file_digest(checkpoint) not in reference["checkpoint_sha256"].values():
        raise ValueError(
            "Only the exact selected baseline best/last may use the saved source contract"
        )
    old_data = {
        k: v["sha256"] for k, v in saved.items() if not k.startswith("implementation/")
    }
    new_data = {
        k: v["sha256"]
        for k, v in current.items()
        if not k.startswith("implementation/")
    }
    if old_data != new_data:
        raise ValueError(
            "Baseline data, pretrained weights or initialization constants changed"
        )
