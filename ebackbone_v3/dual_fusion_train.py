"""Isolated full-event dual fusion study. No queue or legacy run is modified."""
from __future__ import annotations

import argparse
from datetime import timedelta
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .dual_fusion_data import DualFusionDataset, collate
from .dual_fusion_models import DualFusionBackbone, export_backbone, load_backbone_export, profile_macs
from .hierarchy_ddp import (ExactBatchShard, amp, check_gradient_average, cpu_state, flat_grad,
                            gather, resume_epoch, save_epoch, seed_all, synchronize)
from .v1_training import atomic_json, seed_worker, sha256_file

VARIANTS = ('hierarchy_only', 'latent_only', 'dual')
CONFIG = Path(__file__).resolve().parents[1] / 'configs/dual_fusion.n_imagenet_mini_b64_e50.json'
SOURCE_FILES = ('dual_fusion_train.py', 'dual_fusion_models.py', 'dual_fusion_data.py',
                'hierarchy_ddp.py', 'hierarchy_models.py', 'hierarchy_data.py',
                'v1_models.py', 'v1_data.py', 'v1_training.py', 'n_imagenet_mini_dataset.py',
                'representations.py', 'splits.py', 'n_imagenet_mini_index.py')


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    settings = config['training']
    required = dict(world_size=2, global_batch_size=64, batch_size=32,
                    accumulation_steps=1, epochs=50, learning_rate=.05, momentum=.9,
                    weight_decay=1e-4, seed=20260908, precision='bfloat16')
    for key, value in required.items():
        if settings.get(key) != value:
            raise ValueError(f'{key} must be {value!r} for this controlled study')
    for key in ('cpu_threads', 'num_workers'):
        value = settings[key]
        if type(value) is not int or value < (1 if key == 'cpu_threads' else 0):
            raise ValueError(f'invalid {key}')
    if config['model'] != dict(height=480, width=640, num_classes=100, dimension=256, latent_width=8):
        raise ValueError('the production study requires native 480x640 inputs and dimension 256')
    for key in ('manifest_dir', 'dataset_root', 'raw_cache'):
        config[key] = str((path.parent / config[key]).resolve())
    config['config_sha256'] = sha256_file(path)
    for split in ('train', 'validation'):
        if sha256_file(Path(config['manifest_dir']) / f'{split}.jsonl') != config[f'{split}_manifest_sha256']:
            raise ValueError(f'{split} manifest provenance mismatch')
    return config


def make_model(variant, config):
    return DualFusionBackbone(mode=variant, seed=config['training']['seed'], **config['model'])


def identity_for(config, variant):
    return dict(protocol=config['protocol'], variant=variant, settings=config['training'],
                config=config, world_size=2, global_batch_size=64,
                source_sha256={name: sha256_file(Path(__file__).parent / name) for name in SOURCE_FILES},
                torch_version=torch.__version__, numpy_version=np.__version__,
                input='full stored event array; no window splitting; no augmentation',
                final_test_accessed=False)


def prepare_output(path, identity, resume=False):
    """Require a new directory or the exact provenance of our own checkpoint."""
    path = Path(path)
    if resume:
        checkpoint = torch.load(path / 'checkpoint_last.pt', map_location='cpu', weights_only=False)
        if checkpoint.get('identity') != identity:
            raise ValueError('resume source/configuration/manifest identity mismatch')
        if json.loads((path / 'config.json').read_text()) != identity:
            raise ValueError('output config identity mismatch')
    else:
        path.mkdir(parents=True, exist_ok=False)
        atomic_json(path / 'config.json', identity)


def run_epoch(ddp, data, settings, device, epoch, optimizer=None, *, probe=False, height=480, width=640):
    """Exact distributed coverage; tail gradients use actual global sample count."""
    train = optimizer is not None
    model = ddp.module
    model.train(train)
    rank, world = dist.get_rank(), dist.get_world_size()
    sampler = ExactBatchShard(len(data), rank, world, settings['batch_size'], settings['seed'], epoch, train)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=partial(collate, height=height, width=width),
                        num_workers=settings['num_workers'], worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(settings['seed'] + epoch),
                        pin_memory=device.type == 'cuda')
    values = torch.zeros(4, dtype=torch.float64, device=device)
    ids, checks = [], []
    events = []
    dist.barrier()
    synchronize(device)
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        if set(batch['splits']) != {data.split} or set(batch['source_splits']) != {'train'}:
            raise ValueError('only the immutable training-source internal split is allowed')
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch['inputs'].items()}
        labels = batch['labels'].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if probe and index == 0:
                error = check_gradient_average(ddp, inputs, labels, sampler.global_sizes[index], device)
        with torch.set_grad_enabled(train), amp(device):
            logits = (ddp if train else model)(inputs)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite cross entropy')
        if train:
            (loss * world / sampler.global_sizes[index]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            if probe:
                gradient = flat_grad(model)
                digest = hashlib.sha256(gradient.cpu().numpy().tobytes()).hexdigest()
                if len(set(gather(digest))) != 1:
                    raise AssertionError('DDP gradient mismatch')
                checks.append(dict(step=index + 1, gradient_sha256=digest,
                                   average_error=error if index == 0 else None))
            optimizer.step()
        predictions = logits.detach().topk(min(5, logits.shape[1]), dim=1).indices
        values += torch.stack((loss.detach().double(), (predictions[:, 0] == labels).sum(),
                               (predictions == labels[:, None]).any(1).sum(), labels.new_tensor(len(labels))))
        ids.extend(batch['sample_ids'])
        events.extend(batch['inputs']['event_counts'].tolist())
        if train and rank == 0 and (index == 0 or (index + 1) % 100 == 0):
            print(json.dumps(dict(event='optimizer_step', epoch=epoch, step=index + 1,
                                  global_batch=sampler.global_sizes[index], loss_rank0=float(loss.detach()) / len(labels))), flush=True)
    synchronize(device)
    seconds = torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64)
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    dist.all_reduce(values)
    shards = gather(ids)
    flat = [sample for shard in shards for sample in shard]
    expected = [row.sample_id for row in data.rows]
    if len(flat) != len(data) or len(set(flat)) != len(flat) or set(flat) != set(expected):
        raise AssertionError('sample coverage must have no duplicates or missing rows')
    all_events = [n for shard in gather(events) for n in shard]
    result = dict(loss=float(values[0] / values[3]), top1=float(values[1] / values[3]),
                  top5=float(values[2] / values[3]), samples=len(flat), seconds=float(seconds),
                  samples_per_second=len(flat) / float(seconds), duplicates=0, missing=0,
                  rank_sample_counts=[len(shard) for shard in shards],
                  sample_order_sha256=hashlib.sha256(''.join(s + '\n' for s in expected).encode()).hexdigest(),
                  optimizer_steps=len(sampler) if train else 0,
                  event_count=dict(min=min(all_events), max=max(all_events), mean=sum(all_events) / len(all_events)))
    if probe:
        result['gradient_checks'] = checks
    return result


def verify_exports(output, model, inputs, config, variant, device):
    """Compare live trained logits, strict checkpoint reload and classifier-free export."""
    model.eval()
    result = {}
    for kind in ('last', 'best'):
        checkpoint_path = output / f'checkpoint_{kind}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        restored = make_model(variant, config).to(device).eval()
        restored.load_state_dict(checkpoint['model'], strict=True)
        if kind == 'last':
            for key, value in model.state_dict().items():
                if not torch.equal(value.detach().cpu(), checkpoint['model'][key]):
                    raise AssertionError(f'live/checkpoint tensor mismatch: {key}')
        with torch.no_grad(), amp(device):
            if kind == 'last':
                torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
            features = restored.forward_embedding(inputs)
        export_path = output / f'backbone_{kind}.pt'
        if not export_path.exists():
            export_backbone(restored, export_path)
        exported = load_backbone_export(export_path, map_location='cpu').to(device).eval()
        expected_state = {key: value for key, value in checkpoint['model'].items()
                          if not key.startswith('classifier.')}
        actual_state = exported.state_dict()
        if set(expected_state) != set(actual_state) or any(
                not torch.equal(value, actual_state[key].detach().cpu()) for key, value in expected_state.items()):
            raise AssertionError('checkpoint/export tensor mismatch')
        with torch.no_grad(), amp(device):
            torch.testing.assert_close(features, exported.forward_embedding(inputs), rtol=0, atol=0)
        result[kind] = dict(strict_checkpoint_load=True, backbone_embedding_bit_exact=True,
                            checkpoint_sha256=sha256_file(checkpoint_path), export_sha256=sha256_file(export_path))
    return result


def execute(args):
    config = load_config(args.config)
    settings = config['training']
    identity = identity_for(config, args.variant)
    if int(os.environ.get('WORLD_SIZE', '0')) != 2 or not torch.cuda.is_available():
        raise RuntimeError('train requires torchrun with exactly two CUDA ranks')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch')
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    try:
        rank, local = dist.get_rank(), int(os.environ['LOCAL_RANK'])
        device = torch.device('cuda', local)
        seed_all(settings, device)
        datasets = {split: DualFusionDataset(config['manifest_dir'], config['dataset_root'], split,
                                             mode=args.variant, raw_cache=config['raw_cache'])
                    for split in ('train', 'validation')}
        if (len(datasets['train']), len(datasets['validation'])) != (124395, 5000):
            raise ValueError('unexpected Mini internal split sizes')
        if {row.sample_id for row in datasets['train'].rows} & {row.sample_id for row in datasets['validation'].rows}:
            raise ValueError('train/validation overlap')
        # Propagate rank-zero output failures to every rank instead of hanging at a barrier.
        error = [None]
        if rank == 0:
            try:
                prepare_output(args.output_dir, identity, args.resume)
            except Exception as exc:
                error[0] = f'{type(exc).__name__}: {exc}'
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
        model = make_model(args.variant, config).to(device)
        ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=settings['learning_rate'],
                                    momentum=settings['momentum'], weight_decay=settings['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['epochs'])
        first, history, best = 1, [], None
        if args.resume:
            first, history, best = resume_epoch(args.output_dir, ddp, optimizer, scheduler, device, identity)
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(first, settings['epochs'] + 1):
            lr = optimizer.param_groups[0]['lr']
            training = run_epoch(ddp, datasets['train'], settings, device, epoch, optimizer)
            evaluation = run_epoch(ddp, datasets['validation'], settings, device, epoch)
            scheduler.step()
            improved = best is None or (evaluation['top1'], -evaluation['loss']) > (best['top1'], -best['loss'])
            if improved:
                best = dict(epoch=epoch, **evaluation)
            gate = float(model.a.detach().sigmoid()) if args.variant == 'dual' else None
            history.append(dict(epoch=epoch, learning_rate=lr, train=training, validation=evaluation, latent_weight=gate))
            save_epoch(args.output_dir, ddp, optimizer, scheduler, device, epoch, identity, history, best, improved)
            if rank == 0:
                print(json.dumps(dict(event='epoch_complete', variant=args.variant, **history[-1])), flush=True)
        memory = gather(dict(allocated_bytes=torch.cuda.max_memory_allocated(device),
                             reserved_bytes=torch.cuda.max_memory_reserved(device)))
        if rank == 0:
            inputs = {key: value.to(device) for key, value in collate([datasets['validation'][0]])['inputs'].items()}
            checks = verify_exports(args.output_dir, model, inputs, config, args.variant, device)
            atomic_json(args.output_dir / 'report.json', dict(status='complete', identity=identity,
                        completed_epochs=history[-1]['epoch'], best=best, final=history[-1]['validation'],
                        efficiency=profile_macs(model), gpu_memory_by_rank=memory,
                        checkpoint_verification=checks, final_test_accessed=False))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='read-only configuration/provenance/compute inspection')
    inspect.add_argument('--config', type=Path, default=CONFIG)
    train = sub.add_parser('train', help='explicit launch; requires two CUDA ranks')
    train.add_argument('--config', type=Path, default=CONFIG)
    train.add_argument('--variant', choices=VARIANTS, required=True)
    train.add_argument('--output-dir', type=Path, required=True)
    train.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'inspect':
        config = load_config(args.config)
        print(json.dumps(dict(config=config, profiles={name: profile_macs(make_model(name, config)) for name in VARIANTS},
                              training_launched=False, final_test_accessed=False), indent=2))
    else:
        execute(args)


if __name__ == '__main__':
    main()
