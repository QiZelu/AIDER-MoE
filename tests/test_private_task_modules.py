"""任务私有调制、alpha、Gate、Head 的复制、分流和无裁剪测试。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from engine.audit import digest, optimizer_snapshot, parameter_counts
from engine.checkpoint import load_model_checkpoint
from engine.gradients import prepare_gradients
from engine.private_task_diagnostics import check_private_activity, task_coefficients
from engine.routing import task_routes
from engine.sampling import MixedPairSampler
from models.aider import AIDER
from models.registry import BRANCHES, WIDTHS, TASKS
from models.task_private import TaskPrivateLinear
from utils.config import load_config, seed_everything
from utils.private_task_modules import CONTROL, collapsed_initial_state, validate_config


class PrivateTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg=load_config('configs/film_video_only_private_task_modules_no_clip.yaml')
        cls.reference=load_config(CONTROL/'config.yaml')
        seed_everything(42)
        cls.base=AIDER(cls.reference,build_backbones=False).eval()
        cls.rng=torch.get_rng_state().clone()
        seed_everything(42)
        cls.model=AIDER(cls.cfg,build_backbones=False).eval()
        cls.private_rng=torch.get_rng_state().clone()
        gen=torch.Generator().manual_seed(97)
        cls.raw={b:torch.randn(4,8,d,generator=gen) for b,d in zip(BRANCHES,WIDTHS)}

    @classmethod
    def tearDownClass(cls):
        del cls.base,cls.model

    def test_config_copies_rng_and_counts(self):
        validate_config(self.cfg)
        self.assertTrue(torch.equal(self.rng,self.private_rng))
        self.assertEqual(digest(collapsed_initial_state(self.model)),digest(self.base.compact_state_dict()))
        banks=[m for m in self.model.modules() if isinstance(m,TaskPrivateLinear)]
        self.assertEqual(len(banks),6)
        self.assertEqual(sum(len(m.by_task) for m in banks),24)
        self.assertEqual(len(list(self.model.heads())),4)
        self.assertEqual(len(list(self.model.pools())),2)
        self.assertEqual(parameter_counts(self.model)['trainable']-parameter_counts(self.base)['trainable'],7121709)
        coefficients=task_coefficients(self.model)
        for t in TASKS:
            self.assertAlmostEqual(coefficients[t]['alpha'],.1,places=7)
        wrong=copy.deepcopy(self.cfg)
        wrong['training']['gradient_clip']=1.
        with self.assertRaises(ValueError):
            validate_config(wrong)

    def test_initial_predictions_routes_and_formula(self):
        with torch.no_grad():
            for ids in (torch.tensor([3,0,2,1]),torch.tensor([2,2,2,2]),torch.tensor([0,3,0,3])):
                a=self.model.forward_from_raw(self.raw,ids)
                b=self.base.forward_from_raw(self.raw,ids)
                torch.testing.assert_close(a['predictions'],b['predictions'],rtol=0,atol=1e-6)
                torch.testing.assert_close(a['predictions'],a['frame_scores'].mean(1),rtol=0,atol=0)
                for name,route in a['routing']['pools'].items():
                    other=b['routing']['pools'][name]
                    self.assertTrue(torch.equal(route['topk_indices'],other['topk_indices']))
                    torch.testing.assert_close(route['topk_weights'],other['topk_weights'],rtol=0,atol=1e-7)
        for pool in task_routes(self.model)['pools'].values():
            self.assertEqual(pool['gate_policy'],'task_private')
            self.assertEqual(pool['expert_policy'],'cross_task_shared')

    def test_explicit_ids_not_inferred_from_features(self):
        source=torch.nn.Linear(3,2)
        bank=TaskPrivateLinear(source,TASKS)
        with torch.no_grad():
            for index,linear in enumerate(bank.by_task.values()):
                linear.weight.zero_();linear.bias.fill_(index)
        ids=torch.tensor([3,0,2,1])
        torch.testing.assert_close(bank(torch.ones(4,3),ids),ids.float()[:,None].expand(4,2))
        with self.assertRaises(ValueError):
            bank(torch.ones(4,3),None)
        with self.assertRaises(ValueError):
            bank(torch.ones(4,3),torch.tensor([0,1,2,4]))

    def test_absent_private_modules_existing_adam_states(self):
        for bank in (m for m in self.model.modules() if isinstance(m,TaskPrivateLinear)):
            state=copy.deepcopy(bank.state_dict())
            optimizer=torch.optim.AdamW(bank.parameters(),lr=1e-4,weight_decay=.05)
            try:
                for ids in (torch.arange(4),torch.tensor([3,1,3,1])):
                    optimizer.zero_grad(set_to_none=True)
                    before={t:optimizer_snapshot(m,optimizer) for t,m in bank.by_task.items()}
                    bank(torch.ones(4,bank.in_features),ids).sum().backward()
                    optimizer.step()
                    for index,(task,module) in enumerate(bank.by_task.items()):
                        if index not in ids.tolist():
                            self.assertTrue(all(p.grad is None for p in module.parameters()))
                            self.assertEqual(before[task],optimizer_snapshot(module,optimizer))
                        else:
                            self.assertNotEqual(before[task]['parameters'],digest(module.state_dict()))
            finally:
                bank.load_state_dict(state);bank.zero_grad(set_to_none=True)

    def test_full_feature_gradients_no_clip_and_checkpointing(self):
        self.model.train()
        ids=torch.tensor([0,3,0,3])
        try:
            for checkpointing in (False,True):
                self.model.fusion.use_checkpoint=checkpointing
                self.model.zero_grad(set_to_none=True)
                result=self.model.forward_from_raw(self.raw,ids)
                result['predictions'].sum().backward()
                check_private_activity(self.model,ids)
                before=digest({n:p.grad for n,p in self.model.named_parameters() if p.grad is not None})
                with patch('torch.nn.utils.clip_grad_norm_',side_effect=AssertionError('must not clip')):
                    report=prepare_gradients(self.model,self.cfg)
                self.assertEqual(report['applied_scale'],1.)
                self.assertEqual(before,digest({n:p.grad for n,p in self.model.named_parameters() if p.grad is not None}))
                if checkpointing:
                    self.assertEqual(before,plain_gradient)
                else:
                    plain_gradient=before
                for name,pool in self.model.pools():
                    active=set(result['routing']['pools'][name]['topk_indices'].flatten().tolist())
                    for index,expert in enumerate(pool.experts):
                        if index not in active:
                            self.assertTrue(all(p.grad is None for p in expert.parameters()))
        finally:
            self.model.eval();self.model.fusion.use_checkpoint=True
            self.model.zero_grad(set_to_none=True)

    def test_sampling_and_checkpoint_rejection(self):
        saved=json.loads((CONTROL/'preflight.json').read_text())
        sampler=MixedPairSampler(1844,9042,8)
        for epoch,expected in enumerate(saved['sampling_sha256']):
            sampler.set_epoch(epoch)
            self.assertEqual(sampler.digest(),expected)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'old.pt';torch.save({},path)
            with self.assertRaisesRegex(ValueError,'private-task-module'):
                load_model_checkpoint(self.model,path,self.cfg)

    def test_evaluation_diagnostics_and_summary(self):
        from engine.evaluation import evaluate
        from engine.trainer import save_epoch_diagnostics
        from engine.private_task_diagnostics import summarize
        from engine.io import write_json
        model,raw=self.model,self.raw
        class Probe:
            def __getattr__(self,name):
                return getattr(model,name)
            def __call__(self,batch):
                return model.forward_from_raw(raw,batch['task_id'])
        class Loader:
            dataset=range(4)
            def __iter__(self):
                yield dict(task_id=torch.arange(4),video_id=['probe']*4,prompt=['prompt']*4,
                           target=torch.tensor([.2,.4,.6,.8]))
        _,_,routes=evaluate(Probe(),Loader(),torch.device('cpu'))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            training=dict(route_counts={},shared_gradients=[[1.,1.]],pair_indices=[],batch_task_counts=[],
                          order_sha256='test',head_update_steps={},private_module_steps={})
            save_epoch_diagnostics(out,0,model,training,routes)
            self.assertEqual(set(json.loads((out/'film/epoch_01_by_task.json').read_text())),set(TASKS))
            self.assertTrue((out/'film/epoch_01_task_coefficients.json').exists())
            (out/'config.yaml').write_text(yaml.safe_dump(self.cfg))
            history=[dict(epoch=i+1,optimizer_updates=(i+1)*231,
                          test=dict(metrics=dict(macro=dict(srocc=1. if i==0 else 0.)))) for i in range(10)]
            write_json(out/'history.json',history)
            for kind,epoch in (('best',1),('last',10)):
                report=json.loads((CONTROL/f'{kind}_evaluation.json').read_text())
                report['checkpoint_epoch']=epoch
                write_json(out/f'{kind}_evaluation.json',report)
            result=summarize(out)
            self.assertTrue(result['complete'])
            self.assertEqual(result['best_epoch'],1)


if __name__=='__main__':
    unittest.main()
