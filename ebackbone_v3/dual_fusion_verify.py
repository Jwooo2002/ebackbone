"""Bounded native two-GPU verification: one real batch/step per selected mode.

This command never runs epochs, opens validation/test, resumes a study, or
updates a queue. Probe weights are disposable and never initialize training.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .dual_fusion_data import DualFusionDataset, MODES, collate
from .dual_fusion_models import DualFusionBackbone, export_backbone, load_backbone_export, profile_macs
from .hierarchy_ddp import seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        parser.error('bounded verification requires batch-size in [1,32]')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError('native verification requires exactly two GPU ranks')
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    repo = Path(__file__).resolve().parents[1]
    seed_all({'seed': 20260908, 'cpu_threads': 4}, device)
    reports = []
    for mode in args.modes:
        dataset = DualFusionDataset(repo / 'manifests/n_imagenet_mini/supervised-v1',
            '/mnt/hdd1/datasets/event/n_imagenet', 'train', mode=mode,
            raw_cache=repo.parent / 'ebackbone_v3_v1_artifacts/raw_cache',
            limit=world * args.batch_size, seed=20260908)
        started = time.perf_counter()
        samples = [dataset[i] for i in range(rank, len(dataset), world)]
        sample_ids = [s['sample_id'] for s in samples]
        counts = [s['source']['event_count'] for s in samples]
        reference = {k: v.to(device) for k, v in collate([samples[0]])['inputs'].items()}
        batch = collate(samples)
        del samples
        inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
        labels = batch['labels'].to(device)
        del batch
        preparation_seconds = time.perf_counter() - started
        model = DualFusionBackbone(mode=mode).to(device)
        before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=.05, momentum=.9, weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = ddp(inputs)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        if logits.shape != (args.batch_size, 100) or logits.dtype != torch.bfloat16:
            raise AssertionError('native logits shape/dtype mismatch')
        if not torch.isfinite(loss):
            raise AssertionError('nonfinite CE')
        loss.backward()
        gradient_l1 = {}
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f'missing/nonfinite gradient: {name}')
            key = '.'.join(name.split('.')[:2]) if name.startswith(('latent.', 'hierarchy.')) else name.split('.')[0]
            gradient_l1[key] = gradient_l1.get(key, 0.) + float(parameter.grad.abs().sum())
            digest.update(parameter.grad.detach().cpu().numpy().tobytes())
        if not all(value > 0 for value in gradient_l1.values()):
            raise AssertionError(f'disconnected component: {gradient_l1}')
        optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - started
        changed = {}
        for name, parameter in model.named_parameters():
            key = '.'.join(name.split('.')[:2]) if name.startswith(('latent.', 'hierarchy.')) else name.split('.')[0]
            changed[key] = changed.get(key, False) or not torch.equal(before[name], parameter.detach().cpu())
        if not all(changed.values()):
            raise AssertionError(f'component failed to update: {changed}')
        peak = {'allocated_bytes': torch.cuda.max_memory_allocated(device),
                'reserved_bytes': torch.cuda.max_memory_reserved(device)}
        row = {'rank': rank, 'gpu': torch.cuda.get_device_name(device),
               'sample_ids': sample_ids, 'event_counts': counts,
               'preparation_seconds': preparation_seconds, 'step_seconds': step_seconds,
               'loss': float(loss), 'gradient_l1': gradient_l1, 'changed_components': changed,
               'gradient_sha256': digest.hexdigest(), **peak}
        del logits, loss, inputs, labels, before, optimizer, ddp
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            expected_logits = model(reference)
            expected_embedding = model.forward_embedding(reference)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checkpoint_path = args.output_dir / f'{mode}_rank{rank}_checkpoint.pt'
        with checkpoint_path.open('xb') as handle:
            torch.save({'model_config': model.construction_config(), 'model': state}, handle)
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        restored = DualFusionBackbone(**saved['model_config']).to(device).eval()
        restored.load_state_dict(saved['model'], strict=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if not torch.equal(expected_logits, restored(reference)):
                raise AssertionError('live-to-checkpoint logits changed')
        del restored, saved, state
        export_path = args.output_dir / f'{mode}_rank{rank}_backbone.pt'
        export_backbone(model, export_path)
        restored = load_backbone_export(export_path, map_location=device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if not torch.equal(expected_embedding, restored(reference)):
                raise AssertionError('live-to-backbone embedding changed')
        row.update(checkpoint_logits_bit_exact=True, backbone_embedding_bit_exact=True)
        gathered = [None] * world
        dist.all_gather_object(gathered, row)
        if len({item['gradient_sha256'] for item in gathered}) != 1:
            raise AssertionError('rank gradients differ')
        ids = [sid for item in gathered for sid in item['sample_ids']]
        if len(ids) != len(set(ids)) or len(ids) != args.batch_size * world:
            raise AssertionError('probe sample duplication or omission')
        report = {'mode': mode, 'status': 'PASS', 'local_batch': args.batch_size,
                  'global_batch': args.batch_size * world, 'accumulation': 1,
                  'optimizer_steps': 1, 'ranks': gathered, 'profile': profile_macs(model),
                  'lambda_after': float(model.a.sigmoid()) if mode == 'dual' else None}
        reports.append(report)
        if rank == 0:
            print(json.dumps({'mode': mode, 'status': 'PASS', 'ranks_memory': [
                {k: r[k] for k in ('rank', 'allocated_bytes', 'reserved_bytes')} for r in gathered]}), flush=True)
        del restored, model, reference, expected_embedding, expected_logits
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        (args.output_dir / 'report.json').write_text(json.dumps({
            'status': 'PASS', 'reports': reports, 'full_training_launched': False,
            'final_test_accessed': False,
            'scope': 'one native real training batch per mode; not an epoch or worst-case memory guarantee'}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
