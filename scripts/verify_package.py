"""Verify every exported file and load all three complete checkpoints on CPU."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'verification/model_load.json')
    parser.add_argument('--skip-file-hashes', action='store_true', help='Only after a separate complete file hash audit')
    args = parser.parse_args()
    import torch
    from engine.audit import file_digest, digest
    from engine.checkpoint import load_model_checkpoint
    from engine.io import write_json
    from utils.config import load_config
    from utils.initialization import initialize_model
    from utils.provenance import selected_source_hashes
    from models.registry import BRANCHES, WIDTHS
    torch.set_num_threads(2)
    metadata = json.loads((ROOT / 'package_manifest.json').read_text())
    if not args.skip_file_hashes:
        for name, record in metadata['files'].items():
            if file_digest(ROOT / name) != record['sha256']:
                raise ValueError('Package file changed: ' + name)
        print('All package file hashes verified:', len(metadata['files']), flush=True)
    config = load_config(ROOT / 'configs/model.yaml')
    model = initialize_model(config).eval()
    expected = json.loads((ROOT / metadata['run_dir'] / 'initialization_audit.json').read_text())
    if model.initialization_audit != expected:
        raise AssertionError('Initial parameters/task copies/RNG do not reproduce the source version')
    model.source_hashes = selected_source_hashes(config)
    required_text = {'tv_backbone.model.text_encoder.' + name for name in model.tv_backbone.model.text_encoder.state_dict()}
    generator = torch.Generator().manual_seed(20261023)
    raw = {b:torch.randn(4,8,d,generator=generator) for b,d in zip(BRANCHES,WIDTHS)}
    task_ids = torch.tensor([0,1,2,3])
    checkpoints = {}
    for kind, expected_epoch in (('best',7),('last',10),('epoch9',9)):
        entry = metadata['checkpoints'][kind]
        path = ROOT / entry['path']
        payload = load_model_checkpoint(model,path,config)
        if payload['epoch'] + 1 != expected_epoch:
            raise AssertionError('Wrong checkpoint epoch: ' + kind)
        if not required_text <= payload['model'].keys():
            raise AssertionError('Fine-tuned BLIP text tensor coverage is incomplete')
        original_hash = digest(payload['model'])
        loaded_hash = digest(model.compact_state_dict())
        if original_hash != loaded_hash:
            raise AssertionError('Loaded parameter bytes differ from checkpoint')
        model.eval()
        with torch.no_grad():
            output = model.forward_from_raw(raw,task_ids)
        if not torch.isfinite(output['predictions']).all():
            raise AssertionError('Nonfinite prediction')
        checkpoints[kind] = dict(epoch=expected_epoch,sha256=entry['sha256'],
            exact_tensor_reload=True,compact_state_sha256=loaded_hash,complete_blip_text_tensors=len(required_text),
            raw_feature_predictions=output['predictions'].tolist(),
            four_private_heads=len(list(model.heads())),shared_expert_pools=len(list(model.pools())))
        del payload
        gc.collect()
        print('Loaded and verified:',kind,'epoch',expected_epoch,flush=True)
    report = dict(complete=True,package_root=str(ROOT),file_hashes_checked=not args.skip_file_hashes,
        initial_parameters_and_rng_match=True,checkpoints=checkpoints,
        verification_scope='CPU full-backbone construction; exact complete checkpoint tensor reload; four-task raw-feature forward. No training or GPU jobs launched.')
    write_json(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
