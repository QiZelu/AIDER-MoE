"""只读恢复 Baseline 的未训练参数与固定常量，不构造历史消融模型。"""

import json
from pathlib import Path

import torch

from engine.audit import digest, file_digest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/reference"
TASK_TABLE = REFERENCE / "task_tables.json"
POSITION = ROOT / "artifacts/depth2_video_adaptive_fixed_position.pt"


def load_task_table(task):
    if task.get('table_protocol') == 'eduaigv':
        from utils.eduaigv import load_two_task_table
        return load_two_task_table(task)
    if 'table_path' in task:
        from utils.videofeedback import load_five_task_table
        return load_five_task_table(task)
    saved = json.loads(TASK_TABLE.read_text())
    if (
        list(task["order"]) != saved["task_order"]
        or task["descriptions"] != saved["descriptions"]
    ):
        raise ValueError("Task descriptions/order differ from the baseline snapshot")
    values = {
        "encoded_feature": torch.tensor(saved["transformed"], dtype=torch.float32),
        "original_feature": torch.tensor(saved["original"], dtype=torch.float32),
        "task_mean": torch.tensor(saved["mean"], dtype=torch.float32),
        "centered_norms": torch.tensor(saved["centered_norms"], dtype=torch.float32),
    }
    for name, value in values.items():
        if (
            not torch.isfinite(value).all()
            or digest({name: value}) != saved["hashes"][name]
        ):
            raise ValueError("Invalid fixed task table: " + name)
    return values


def load_position_encoding():
    audit = json.loads(POSITION.with_suffix(".json").read_text())
    key = "fusion.position_encoding"
    if audit["extracted_keys"] != [key] or audit["trained_parameters_imported"] != 0:
        raise ValueError("Position snapshot must contain only an approved fixed buffer")
    if file_digest(POSITION) != audit["snapshot_sha256"]:
        raise ValueError("Fixed position snapshot changed")
    state = torch.load(POSITION, map_location="cpu", weights_only=True)
    if set(state) != {key} or digest(state) != audit["tensor_sha256"]:
        raise ValueError("Fixed position tensor hash mismatch")
    value = state[key]
    if (
        value.shape != (1, 8, 1536)
        or value.dtype != torch.float32
        or value.requires_grad
    ):
        raise ValueError("Invalid position encoding")
    return value


def restore_encoder_initialization(model, config):
    """恢复 Technical、Motion 位置编码和 BLIP SA，以及历史构造后的 CPU RNG。"""
    snapshot = torch.load(
        config["training"]["initialization"], map_location="cpu", weights_only=True
    )
    if snapshot["seed"] != config["training"]["seed"]:
        raise ValueError("Initialization seed mismatch")
    prefixes = ("technical_expert.", "motion_position_embedding.", "blip_temporal.")
    saved = {
        name: value
        for name, value in snapshot["state"].items()
        if name.startswith(prefixes)
    }
    current = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }
    if saved.keys() != current.keys():
        raise ValueError("Encoder initialization coverage mismatch")
    for name, value in current.items():
        if value.shape != saved[name].shape or value.dtype != saved[name].dtype:
            raise ValueError("Encoder initialization shape/dtype mismatch: " + name)
    model.load_state_dict(saved, strict=False)
    torch.set_rng_state(snapshot["cpu_rng_state"])
    return len(saved)


def initialize_model(config):
    """直接构造当前网络，再核验历史未训练状态；不导入历史模型类。"""
    from models.aider import AIDER
    from utils.config import seed_everything

    seed_everything(config["training"]["seed"])
    model = AIDER(config)
    model.train(True)
    initial = digest(model.compact_state_dict())
    if 'eduaigv' in config:
        from utils.eduaigv import audit_initial_model
        model.construction_rng_sha256 = digest({'cpu_rng': torch.get_rng_state()})
        model.initialization_audit['eduaigv'] = audit_initial_model(model, config)
        model.initialization_audit['initial_state_sha256'] = initial
        return model
    if 'private_task_modules' in config:
        from utils.private_task_modules import audit_initialization
        model.construction_rng_sha256 = digest({'cpu_rng': torch.get_rng_state()})
        model.initialization_audit['private_task_modules'] = audit_initialization(model, config)
        model.initialization_audit['initial_state_sha256'] = initial
        return model
    if 'videofeedback' in config:
        from utils.videofeedback import audit_initial_model
        model.construction_rng_sha256 = digest({'cpu_rng': torch.get_rng_state()})
        model.initialization_audit['videofeedback'] = audit_initial_model(model, config)
        model.initialization_audit['initial_state_sha256'] = initial
        return model
    baseline_hash = baseline_equivalent_initial_hash(model, config)
    reference = json.loads((REFERENCE / "preflight.json").read_text())
    if baseline_hash != reference["initial_state_sha256"]:
        raise AssertionError(
            "Retained initial parameters differ from the verified baseline"
        )
    model.construction_rng_sha256 = digest({"cpu_rng": torch.get_rng_state()})
    model.initialization_audit["original_split"] = {
        "initial_state_sha256": initial,
        "historical_combined_initial_state_matches": initial == baseline_hash,
        "baseline_equivalent_initial_state_sha256": baseline_hash,
        "retained_initial_parameters_match": True,
        "construction_rng_sha256": model.construction_rng_sha256,
    }
    if config['model'].get('head_policy') == 'private':
        control = Path(config['ablation']['control'])
        if file_digest(control / 'config.yaml') != config['ablation']['control_config_sha256']:
            raise ValueError('Shared-head control configuration changed')
        paired_hash = digest(shared_head_initial_state(model))
        saved = json.loads((control / 'preflight.json').read_text())
        if paired_hash != saved['initial_state_sha256']:
            raise AssertionError('Private-head initialization differs from video-only control')
        model.initialization_audit['private_heads'] = {
            'source': 'seed42_untrained_shared_head',
            'independent_copies': 4,
            'shared_equivalent_initial_state_sha256': paired_hash,
            'control_config_sha256': config['ablation']['control_config_sha256'],
            'all_retained_parameters_match': True,
        }
    return model


def shared_head_initial_state(model):
    """仅用于未训练审计：四份相同 Head 在状态视图中还原成共享 Head。"""
    from models.registry import TASKS

    state = model.compact_state_dict()
    reference = model.fusion.heads[TASKS[0]].state_dict()
    pointers = []
    for task in TASKS:
        head = model.fusion.heads[task]
        if digest(head.state_dict()) != digest(reference):
            raise AssertionError('Private heads must start with identical values')
        pointers.extend(p.data_ptr() for p in head.parameters())
    if len(set(pointers)) != len(pointers):
        raise AssertionError('Private head parameters share storage')
    restored, inserted = {}, False
    for name, value in state.items():
        if not name.startswith('fusion.heads.'):
            restored[name] = value
        elif not inserted:
            restored.update({'fusion.heads.shared.' + k: v for k, v in reference.items()})
            inserted = True
    return restored


def baseline_equivalent_initial_hash(model, config):
    """用原修正器初值补回审计视图，严格校验其余参数；不修改模型。

    video_only 的修正器宽度改变，不能直接与原完整状态哈希比较。
    先校验新修正器本身，再将审计字典中的三个修正器换成原确定性初值。
    重建仅用于未训练初始化审计，不能用于已训练模型加载。
    """
    from models.film import video_mlp, video_linear
    from models.registry import ATTRIBUTES, WIDTHS, BRANCHES

    state = (shared_head_initial_state(model)
             if config['model'].get('head_policy') == 'private'
             else model.compact_state_dict())
    if model.film_video_condition == "cross_branch_linear":
        replacements = {}
        for index, (attribute, width, inputs) in enumerate(zip(ATTRIBUTES, WIDTHS, (1024, 1280, 768))):
            seed = config["training"]["seed"] + 820010 + index
            actual = getattr(model, attribute).adaptive.state_dict()
            if digest(actual) != digest(video_linear(inputs, 2 * width, seed).state_dict()):
                raise AssertionError("Linear correction initialization mismatch: " + attribute)
            replacements[attribute + ".adaptive."] = video_mlp(
                1024, 256, 2 * width, seed, zero_output=True
            ).state_dict()
        # 保持历史 state_dict 顺序；digest 将名称及顺序共同计入哈希。
        restored, added = {}, set()
        for name, value in state.items():
            prefix = next((p for p in replacements if name.startswith(p)), None)
            if prefix is None:
                restored[name] = value
            elif prefix not in added:
                restored.update({prefix + k: v for k, v in replacements[prefix].items()})
                added.add(prefix)
        if len(model.video_summaries):
            raise AssertionError("Linear correction must not retain summary MLPs")
        summary_state = {}
        for index, (branch, inputs) in enumerate(zip(BRANCHES, (1024, 1280, 768))):
            original = video_mlp(inputs, 512, 256, config["training"]["seed"] + 840000 + index)
            summary_state.update({f"video_summaries.{branch}.{k}": v
                                  for k, v in original.state_dict().items()})
        state = {}
        for name, value in restored.items():
            state[name] = value
            if name == "alpha_gate.bias":
                # 完整模型的文本 buffer 在 compact_state_dict 最后追加。
                state.update(summary_state)
    if model.film_video_condition == "video_only":
        for index, (attribute, width) in enumerate(zip(ATTRIBUTES, WIDTHS)):
            seed = config["training"]["seed"] + 820010 + index
            expected = video_mlp(256, 256, 2 * width, seed, zero_output=True)
            actual = getattr(model, attribute).adaptive.state_dict()
            if digest(actual) != digest(expected.state_dict()):
                raise AssertionError(
                    "Video-only correction initialization mismatch: " + attribute
                )
            original = video_mlp(1024, 256, 2 * width, seed, zero_output=True)
            for name, value in original.state_dict().items():
                state[f"{attribute}.adaptive.{name}"] = value
    return digest(state)
