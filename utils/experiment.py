"""Original 461/115 manifests, with the saved combined-model configuration."""

import json
import math
from pathlib import Path


from engine.audit import file_digest
from models.registry import TASKS

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/reference"
CONTRACT = "aider_depth2_crossbranch_alwayson_original_461_115_v1"
VIDEO_ONLY_CONTRACT = "aider_depth2_crossbranch_film_video_only_original_461_115_v1"
VIDEO_LINEAR_CONTRACT = "aider_depth2_crossbranch_film_video_linear_original_461_115_v1"
FROZEN_HEAD_CONTRACT = "aider_video_only_frozen_private_heads_l2_original_461_115_v1"
PRIVATE_HEAD_CONTRACT = "aider_video_only_joint_private_heads_original_461_115_v1"
TRAIN = ROOT / "data/AIGVQA-DB/AIGVQA_DB_train.jsonl"
TEST = ROOT / "data/AIGVQA-DB/AIGVQA_DB_test.jsonl"
EXPECTED_HASHES = {
    "train": "bb0cad63c209234d5bef5076c1a7a9877574acbd58791116d93da29931ef0f50",
    "test": "53eb08cd5e0624f5498e602e38024b0f2fc836515a5147141f292458ad0f3068",
}


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def film_video_condition(config):
    mode = config["model"].get("film_video_condition", "task_video")
    if mode not in ("task_video", "video_only", "cross_branch_linear"):
        raise ValueError("Unknown FiLM video condition: " + str(mode))
    return mode


def checkpoint_contract(config):
    if 'eduaigv' in config:
        from utils.eduaigv import CONTRACT as EDU_CONTRACT, validate_config
        validate_config(config)
        return EDU_CONTRACT
    if 'head_rank' in config:
        from utils.head_rank import validate_config
        validate_config(config)
        return config['head_rank']['contract']
    if 'private_task_modules' in config:
        from utils.private_task_modules import CONTRACT as PRIVATE_TASK_CONTRACT, validate_config
        validate_config(config)
        return PRIVATE_TASK_CONTRACT
    if 'videofeedback' in config:
        from utils.videofeedback import CONTRACT as VF_CONTRACT, validate_config
        validate_config(config)
        return VF_CONTRACT
    expected = {"task_video": CONTRACT, "video_only": VIDEO_ONLY_CONTRACT,
                "cross_branch_linear": VIDEO_LINEAR_CONTRACT}[film_video_condition(config)]
    policy = config["model"].get("head_policy", "shared")
    if policy not in ("shared", "private", "frozen_private"):
        raise ValueError("Unknown head policy")
    if policy == "private":
        if film_video_condition(config) != "video_only":
            raise ValueError("Joint private heads require the video-only control")
        expected = PRIVATE_HEAD_CONTRACT
    if policy == "frozen_private":
        if film_video_condition(config) != "video_only":
            raise ValueError("Frozen private heads require the video-only source")
        expected = FROZEN_HEAD_CONTRACT
    if config["ablation"]["contract"] != expected:
        raise ValueError("FiLM condition mode and checkpoint contract do not match")
    return expected


def validate_model_config(config):
    """显式检查当前结构，避免修改旧元数据后被误认为已经改变网络。"""
    if 'eduaigv' in config:
        from utils.eduaigv import validate_config
        validate_config(config)
        return
    if 'head_rank' in config:
        from utils.head_rank import validate_config
        validate_config(config)
        return
    if 'private_task_modules' in config:
        from utils.private_task_modules import validate_config
        validate_config(config)
        return
    if 'videofeedback' in config:
        from utils.videofeedback import validate_config
        validate_config(config)
        return
    reference = json.loads((REFERENCE / "verification.json").read_text())[
        "model_metadata"
    ]

    model = config["model"]
    ablation = config["ablation"]
    blip = model["backbones"]["blip"]
    checkpoint_contract(config)
    if (
        model["readout_variant"] != "film_transformer"
        or model["readout_spec"] != reference["readout_spec"]
        or model["task_embedding_transform"] != reference["task_embedding_transform"]
    ):
        raise ValueError("Historical model metadata differs from the selected baseline")
    if (
        ablation["depth"] != 2
        or ablation["routed_experts"] != 7
        or ablation["routed_top_k"] != 2
        or ablation["layerwise_always_on_experts"] != 1
        or not ablation["film"]
        or ablation["router"]
        or ablation["cross_branch_video_modulation"]
        != ("three_independent_raw_concat_linears"
            if film_video_condition(config) == "cross_branch_linear"
            else "three_independent_two_input_mlps")
        or ablation["tasks"] != list(TASKS)
        or ablation["task_ids"] != list(range(4))
    ):
        raise ValueError(
            "Expected depth 2, cross-branch FiLM, 7 Top2 + 1 shared FFN, four tasks"
        )
    if (
        not model["blip_temporal_sa"]
        or not blip["train_text_encoder"]
        or not blip["text_encoder_gradient_checkpointing"]
        or config["data"]["num_frames"] != 8
    ):
        raise ValueError(
            "Selected baseline requires eight frames, trainable BLIP text and temporal SA"
        )


def validate_data(config=None):
    if config is not None and 'eduaigv' in config:
        from utils.eduaigv import validate_data as validate_edu
        return validate_edu(config)
    if config is not None and 'videofeedback' in config:
        from utils.videofeedback import validate_data as validate_vf
        return validate_vf(config)
    if config is not None and 'repeated_holdout' in config:
        from utils.split_protocol import validate_split_data
        return validate_split_data(config)
    paths = (
        {"train": TRAIN, "test": TEST}
        if config is None
        else {
            "train": Path(config["data"]["train_manifest"]),
            "test": Path(config["data"]["val_manifest"]),
        }
    )
    sets, videos, report = {}, {}, {}
    for part, expected_count in (("train", 461), ("test", 115)):
        rows = read_rows(paths[part])
        sets[part], videos[part] = {r["id"] for r in rows}, {r["video"] for r in rows}
        if (len(rows), len(sets[part]), len(videos[part])) != (expected_count,) * 3:
            raise ValueError("Unexpected original manifest count/duplicates: " + part)
        if any(
            set(r["labels"]) != set(TASKS)
            or any(
                not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in r["labels"].values()
            )
            for r in rows
        ):
            raise ValueError("Missing/nonfinite four-task labels: " + part)
        sha = file_digest(paths[part])
        if sha != EXPECTED_HASHES[part]:
            raise ValueError("Original manifest content changed: " + part)
        report[part] = dict(
            path=str(paths[part]), videos=len(rows), records=4 * len(rows), sha256=sha
        )
    if sets["train"] & sets["test"] or videos["train"] & videos["test"]:
        raise ValueError("Train/test video leakage")
    if len(sets["train"] | sets["test"]) != 576:
        raise ValueError("Incomplete dataset coverage")
    return dict(**report, id_overlap=0, path_overlap=0, total_videos=576)
