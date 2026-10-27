"""小模型验证训练流程重组未改变更新顺序、采样或损失统计。"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.audit import digest
from engine.losses import regression_loss
from engine.optim import build_optimizer, _scheduler
from engine.sampling import MixedPairSampler
from engine.trainer import finetune_epoch


class TinyTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.variant = "test"
        self.tv_backbone = nn.Module()
        self.tv_backbone.model = nn.Module()
        self.tv_backbone.model.text_encoder = nn.Linear(1, 1)
        self.fusion = nn.Module()
        block = nn.Module()
        block.shared_expert = nn.Linear(1, 1)
        self.fusion.blocks = nn.ModuleList([block])

    def pools(self):
        yield "transformer.layer_00.shared", SimpleNamespace(num_experts=2)

    def forward(self, batch):
        features = self.tv_backbone.model.text_encoder(batch["features"])
        prediction = self.fusion.blocks[0].shared_expert(features).squeeze(-1)
        return {
            "predictions": prediction,
            "routing": {
                "pools": {
                    "transformer.layer_00.shared": {
                        "topk_indices": torch.tensor([[0, 1]]).repeat(
                            len(prediction), 1
                        ),
                    }
                }
            },
        }


class TrainingTests(unittest.TestCase):
    def test_two_updates_match_original_loop(self):
        torch.manual_seed(42)
        reference = TinyTrainingModel()
        model = copy.deepcopy(reference)
        config = {
            "training": {"lr": 1e-4, "weight_decay": 0.05},
            "ablation": {"split": "unit_test"},
        }
        records = [
            {
                "features": torch.tensor([index / 8.0]),
                "target": index / 10.0,
                "task_id": index % 4,
                "pair_index": index,
            }
            for index in range(8)
        ]
        loader = DataLoader(records, batch_sampler=MixedPairSampler(8, 9042, 4))
        optimizer = build_optimizer(reference, config)
        scheduler = _scheduler(optimizer, 4, 0.05, warmup_steps=2)
        loader.batch_sampler.set_epoch(1)
        total_loss = 0.0

        # 原训练入口的一步更新次序。
        for batch in loader:
            reference.train(True)
            optimizer.zero_grad(set_to_none=True)
            result = reference(batch)
            loss, losses = regression_loss(result["predictions"], batch["target"], 0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                reference.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            total_loss += float(losses.detach().sum())

        new_optimizer = build_optimizer(model, config)
        new_scheduler = _scheduler(new_optimizer, 4, 0.05, warmup_steps=2)
        # 仅在 CPU 单测中跳过 GPU 监控；正式训练的显存检查不变。
        watcher = SimpleNamespace(report=lambda: {})
        with patch("engine.trainer.check_memory"):
            training = finetune_epoch(
                model,
                loader,
                new_optimizer,
                new_scheduler,
                torch.device("cpu"),
                1,
                config,
                watcher,
            )
        self.assertEqual(digest(model.state_dict()), digest(reference.state_dict()))
        self.assertEqual(new_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(training["train_loss"], total_loss / len(records))
        self.assertEqual(training["pair_indices"], loader.batch_sampler.batches())
        for old_parameter, new_parameter in zip(
            reference.parameters(), model.parameters()
        ):
            self.assertEqual(
                digest(optimizer.state[old_parameter]),
                digest(new_optimizer.state[new_parameter]),
            )


if __name__ == "__main__":
    unittest.main()
