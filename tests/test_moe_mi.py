"""MI values, task weighting, Gate gradients and the real CPU training loop."""

import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.audit import digest
from engine.checkpoint import validate_checkpoint_mi_policy
from engine.losses import (
    moe_mi_protocol, moe_mi_weight, quality_training_loss,
    task_expert_mutual_information, task_router_mi_loss,
)
from engine.optim import build_optimizer, _scheduler
from engine.private_task_diagnostics import check_private_activity
from engine.sampling import MixedPairSampler
from engine.trainer import finetune_epoch
from models.fusion import TransformerFusion
from models.moe import TaskMoE
from models.registry import TASKS
from models.task_private import TaskPrivateLinear


class FixedTaskTable(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('table', torch.eye(768)[:len(TASKS)].clone())

    def forward(self, ids):
        return self.table[ids]


class TinyMiModel(nn.Module):
    """Use production sparse experts/private Linears without pretrained assets."""
    def __init__(self):
        super().__init__()
        self.variant = 'mi_unit_test'
        self.task_order = TASKS
        self.task = dict(order=list(TASKS))
        self.private_task_modules = self.private_heads = True
        self.frozen_private_heads = False
        self.dimension_encoder = FixedTaskTable()
        self.tv_backbone = nn.Module()
        self.tv_backbone.model = nn.Module()
        self.tv_backbone.model.text_encoder = nn.Linear(3, 3)
        self.private_adapter = TaskPrivateLinear(nn.Linear(3, 3), TASKS)
        self.fusion = nn.Module()
        self.fusion.blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.moe = TaskMoE(3, 3, num_experts=7, top_k=2)
            block.moe.gate = TaskPrivateLinear(block.moe.gate, TASKS)
            block.shared_expert = nn.Linear(3, 3)
            self.fusion.blocks.append(block)
        self.fusion.heads = nn.ModuleDict({t: nn.Linear(3, 1) for t in TASKS})

    def pools(self):
        for index, block in enumerate(self.fusion.blocks):
            yield f'transformer.layer_{index:02d}.shared', block.moe

    def heads(self):
        yield from self.fusion.heads.items()

    def forward(self, batch):
        ids = batch['task_id']
        features = self.tv_backbone.model.text_encoder(batch['features'])
        features = self.private_adapter(features, ids)[:, None]
        condition = self.dimension_encoder(ids)
        routes = {}
        for (name, _), block in zip(self.pools(), self.fusion.blocks):
            routed, routes[name] = block.moe(features, condition, ids)
            features = features + routed + block.shared_expert(features)
        predictions = torch.stack([
            self.fusion.heads[TASKS[int(task)]](features[i, 0]).squeeze(-1)
            for i, task in enumerate(ids)
        ])
        return dict(predictions=predictions, routing=dict(pools=routes))


class MutualInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_known_values_and_identical_task_distributions(self):
        ids = torch.arange(2)
        for p in (torch.full((2, 2), .5), torch.tensor([[.9, .1], [.9, .1]])):
            self.assertAlmostEqual(float(task_expert_mutual_information(p, ids, 2)), 0., places=6)
        separated = torch.tensor([[.9, .1], [.1, .9]])
        actual = task_expert_mutual_information(separated, ids, 2)
        expected = .9 * math.log(1.8) + .1 * math.log(.2)
        self.assertAlmostEqual(float(actual), expected, places=6)

    def test_equal_task_prior_with_unequal_frequency_and_unused_experts(self):
        p = torch.eye(4, 7)
        expected = math.log(4)
        self.assertAlmostEqual(float(task_expert_mutual_information(p, torch.arange(4), 4)),
                               expected, places=6)
        ids = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3])
        actual = task_expert_mutual_information(p[ids], ids, 4)
        self.assertAlmostEqual(float(actual), expected, places=6)
        order = torch.tensor([7, 2, 6, 0, 5, 1, 4, 3])
        torch.testing.assert_close(actual, task_expert_mutual_information(p[ids][order], ids[order], 4))

    def test_average_within_task_before_entropy(self):
        # Every sample picks one expert, but both tasks use the same distribution.
        p = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        mi = task_expert_mutual_information(p, torch.tensor([0, 0, 1, 1]), 2)
        self.assertEqual(float(mi), 0.)

    def test_negative_mi_gradient_increases_specialization(self):
        logits = nn.Parameter(torch.tensor([[1., 0.], [0., 1.]]))
        optimizer = torch.optim.SGD([logits], lr=.5)
        before = task_expert_mutual_information(logits.softmax(-1), torch.arange(2), 2)
        (-before).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        optimizer.step()
        after = task_expert_mutual_information(logits.softmax(-1), torch.arange(2), 2)
        self.assertGreater(float(after), float(before))

    def test_invalid_probabilities_and_missing_tasks_fail_explicitly(self):
        cases = [
            (torch.tensor([[.5, .5]]), torch.tensor([0]), 2),
            (torch.tensor([[.2, .2]]), torch.tensor([0]), 1),
            (torch.tensor([[-.1, 1.1]]), torch.tensor([0]), 1),
            (torch.tensor([[.5, .5]]), torch.tensor([1]), 1),
            (torch.tensor([[.5, .5]]), torch.tensor([0], dtype=torch.int32), 1),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                task_expert_mutual_information(*args)
        with self.assertRaises(FloatingPointError):
            task_expert_mutual_information(torch.tensor([[float('nan'), 0.]]), torch.tensor([0]), 1)

    def test_fp32_under_autocast(self):
        logits = nn.Parameter(torch.tensor([[1., 0.], [0., 1.]]))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            mi = task_expert_mutual_information(logits.softmax(-1), torch.arange(2), 2)
        self.assertEqual(mi.dtype, torch.float32)
        (-mi).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_config_and_checkpoint_policy(self):
        self.assertEqual(moe_mi_weight({}), 0.)
        self.assertIsNone(moe_mi_protocol({}))
        for value in (-1., float('nan'), float('inf'), True, None, '0.001'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                moe_mi_weight(dict(training=dict(moe_mi_weight=value)))
        config = dict(training=dict(moe_mi_weight=.001))
        protocol = moe_mi_protocol(config)
        validate_checkpoint_mi_policy(dict(config=copy.deepcopy(config), moe_mi=protocol), config)
        validate_checkpoint_mi_policy(dict(config=dict(training={})), {})
        with self.assertRaisesRegex(ValueError, 'training policy'):
            validate_checkpoint_mi_policy(dict(config=config, moe_mi=protocol), {})
        with self.assertRaisesRegex(ValueError, 'metadata'):
            validate_checkpoint_mi_policy(dict(config=config, moe_mi=dict(protocol, weight=.1)), config)
        for extra in (dict(head_rank={}), dict(model=dict(head_policy='frozen_private'))):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'trainable Gates'):
                moe_mi_weight(dict(config, **extra))

    def test_auxiliary_only_updates_gates_and_averages_layers(self):
        torch.manual_seed(13)
        model = TinyMiModel()
        with patch.object(model, 'forward', side_effect=AssertionError('Must not encode videos')):
            auxiliary, layers = task_router_mi_loss(model)
        self.assertEqual(len(layers), 2)
        self.assertAlmostEqual(float(auxiliary), -sum(layers.values()) / 2, places=7)
        auxiliary.backward()
        for name, parameter in model.named_parameters():
            if '.moe.gate.' in name:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        for _, pool in model.pools():
            for gate in pool.gate.by_task.values():
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in gate.parameters()), 0.)

    def test_mi_gradients_match_with_production_activation_checkpointing(self):
        torch.manual_seed(14)
        model = TinyMiModel().train()
        model.fusion = TransformerFusion(14, torch.zeros(1, 8, 12), width=12, heads=3)
        for _, pool in model.pools():
            pool.gate = TaskPrivateLinear(pool.gate, TASKS)
        ids = torch.tensor([0, 2, 0, 2])
        raw = torch.randn(4, 8, 12)
        reference = None
        for checkpointing in (False, True):
            model.zero_grad(set_to_none=True)
            model.fusion.use_checkpoint = checkpointing
            features = raw.clone().requires_grad_(True)
            fused, _ = model.fusion(features, model.dimension_encoder(ids), ids)
            auxiliary, _ = task_router_mi_loss(model)
            (fused[..., 0].mean() + .1 * auxiliary).backward()
            gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                         if p.grad is not None}
            gradients['input'] = features.grad.detach().clone()
            if reference is None:
                reference = gradients
            else:
                self.assertEqual(gradients.keys(), reference.keys())
                for name in gradients:
                    torch.testing.assert_close(gradients[name], reference[name], rtol=1e-5, atol=1e-7)

    def test_real_training_with_absent_tasks_preserves_other_private_modules(self):
        torch.manual_seed(21)
        model = TinyMiModel()
        config = dict(training=dict(lr=1e-4, weight_decay=.05, gradient_clip=None, moe_mi_weight=.1),
                      task=dict(order=list(TASKS)), ablation=dict(split='mi_unit_test'))
        records = [dict(features=torch.tensor([i / 4, .2, .5]), target=.1 + i / 5,
                        task_id=0 if i < 2 else 2, pair_index=i) for i in range(4)]
        loader = DataLoader(records, batch_sampler=MixedPairSampler(4, 9042, 4))
        batch = next(iter(loader))
        with torch.no_grad():
            expected_total, _, expected = quality_training_loss(model, model(batch), batch, config)
        absent = (TASKS[1], TASKS[3])
        before_heads = {t: digest(model.fusion.heads[t].state_dict()) for t in absent}
        before_adapter = {t: digest(model.private_adapter.by_task[t].state_dict()) for t in absent}
        before_gates = {(name, t): digest(pool.gate.by_task[t].state_dict())
                        for name, pool in model.pools() for t in absent}
        optimizer = build_optimizer(model, config)
        scheduler = _scheduler(optimizer, 1, 0., warmup_steps=0)
        with patch('engine.trainer.check_memory'):
            training = finetune_epoch(model, loader, optimizer, scheduler, torch.device('cpu'),
                                      0, config, SimpleNamespace(report=lambda: {}))
        with self.assertRaisesRegex(AssertionError, 'task presence'):
            check_private_activity(model, batch['task_id'])
        check_private_activity(model, batch['task_id'], all_task_gates=True)
        for task in absent:
            self.assertEqual(before_heads[task], digest(model.fusion.heads[task].state_dict()))
            self.assertEqual(before_adapter[task], digest(model.private_adapter.by_task[task].state_dict()))
        for name, pool in model.pools():
            for task in absent:
                self.assertNotEqual(before_gates[name, task], digest(pool.gate.by_task[task].state_dict()))
            unused = training['route_counts']['pools'][name]['unused_experts']
            self.assertTrue(unused)
            for index in unused:
                self.assertTrue(all(p.grad is None for p in pool.experts[index].parameters()))
        diagnostic = training['loss_diagnostics']
        self.assertAlmostEqual(diagnostic['total_loss'], float(expected_total), places=6)
        self.assertAlmostEqual(diagnostic['mutual_information'], expected['mutual_information'], places=6)
        self.assertAlmostEqual(training['train_loss'], expected['regression_loss'], places=6)
        self.assertAlmostEqual(diagnostic['total_loss'], diagnostic['regression_loss']
                               + diagnostic['weighted_mi_loss'], places=6)
        self.assertEqual(diagnostic['protocol'], moe_mi_protocol(config))
        self.assertEqual(len(diagnostic['steps']), 1)


if __name__ == '__main__':
    unittest.main()
