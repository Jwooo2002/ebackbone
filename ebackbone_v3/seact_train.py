"""Separate full-recording SeACT fine-tuning/scratch study with exact DDP accumulation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
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

from .hierarchy_ddp import ExactBatchShard, amp, cpu_state, flat_grad, gather, resume_epoch, save_epoch, seed_all, synchronize
from .v1_training import atomic_json, seed_worker, sha256_file
from .seact_data import SeActDataset, collate
from .seact_models import make_model as construct_model, export_backbone, load_backbone_export

VARIANTS = ('hierarchy_only', 'latent_only', 'dual')
REGIMES = ('finetune', 'scratch')
CONFIG = Path(__file__).resolve().parents[1] / 'configs/seact.finetune_scratch.json'
SOURCE_FILES = ('seact_train.py', 'seact_models.py', 'seact_data.py', 'dual_fusion_models.py',
                'dual_fusion_data.py', 'hierarchy_ddp.py', 'hierarchy_models.py', 'hierarchy_data.py',
                'v1_models.py', 'v1_data.py', 'v1_training.py', 'representations.py',
                'n_imagenet_mini_dataset.py', 'n_imagenet_mini_index.py', 'splits.py')


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    settings = config['training']
    required = dict(world_size=2, batch_size=1, accumulation_steps=4,
                    global_microbatch_size=2, global_batch_size=8, epochs=50,
                    learning_rate=.01, momentum=.9, weight_decay=1e-4,
                    seed=20260908, precision='bfloat16')
    for key, value in required.items():
        if settings.get(key) != value:
            raise ValueError(f'{key} must equal {value!r} for the matched six-run protocol')
    for key in ('num_workers', 'cpu_threads'):
        if type(settings[key]) is not int or settings[key] < (1 if key == 'cpu_threads' else 0):
            raise ValueError(f'invalid {key}')
    if config['dataset'] != 'SeACT' or config['model']['num_classes'] != 58:
        raise ValueError('SeACT with 58 classes is required')
    expected_model = dict(height=288, width=352, num_classes=58, dimension=256,
                          latent_width=8, point_chunk_size=65536)
    if config['model'] != expected_model:
        raise ValueError(f'model must equal the pinned SeACT geometry and architecture: {expected_model}')
    if config['split_sizes'] != dict(train=406, validation=58, test=116):
        raise ValueError('split_sizes must preserve released train464 and held-out116')
    if config.get('protocol') != 'seact-released-split-internal-val58-dual-fusion-ft-scratch-e50-v1':
        raise ValueError('unsupported SeACT protocol')
    for key in ('manifest_dir', 'dataset_root'):
        config[key] = str((path.parent / config[key]).resolve())
    config['config_sha256'] = sha256_file(path)
    return config


def make_model(variant, regime, config, pretrained_backbone=None):
    if regime not in REGIMES or variant not in VARIANTS:
        raise ValueError('unknown SeACT regime or branch variant')
    if (regime == 'finetune') != (pretrained_backbone is not None):
        raise ValueError('finetune requires exactly one pretrained backbone; scratch forbids it')
    return construct_model(variant, regime, pretrained_backbone=pretrained_backbone,
                           seed=config['training']['seed'], **config['model'])


def identity_for(config, variant, regime, command, pretrained_backbone=None):
    if (regime == 'finetune') != (pretrained_backbone is not None):
        raise ValueError('pretrained backbone is required only for finetune')
    manifests = Path(config['manifest_dir'])
    hashes = {name: sha256_file(manifests / name) for name in ('train.jsonl', 'validation.jsonl', 'test.jsonl', 'provenance.json')}
    return dict(protocol=config['protocol'], dataset='SeACT', config=config, settings=config['training'],
                variant=variant, regime=regime, command=command, world_size=2,
                source_sha256={name: sha256_file(Path(__file__).parent / name) for name in SOURCE_FILES},
                manifest_sha256=hashes,
                pretrained_backbone=str(Path(pretrained_backbone).resolve()) if pretrained_backbone else None,
                pretrained_backbone_sha256=sha256_file(pretrained_backbone) if pretrained_backbone else None,
                torch_version=str(torch.__version__), numpy_version=np.__version__,
                input='full recording; shared observed temporal support; no temporal window splitting',
                selection='internal validation only; official held-out test once on selected best after training')


def prepare_output(path, identity, resume=False):
    path = Path(path)
    if resume:
        checkpoint = torch.load(path / 'checkpoint_last.pt', map_location='cpu', weights_only=False)
        if checkpoint.get('identity') != identity or json.loads((path / 'config.json').read_text()) != identity:
            raise ValueError('resume source/config/manifests/pretrained identity mismatch')
    else:
        path.mkdir(parents=True, exist_ok=False)
        atomic_json(path / 'config.json', identity)


def accumulation_groups(global_sizes, accumulation_steps):
    if type(accumulation_steps) is not int or accumulation_steps < 1:
        raise ValueError('accumulation_steps must be positive')
    return [(start, min(start + accumulation_steps, len(global_sizes)),
             sum(global_sizes[start:start + accumulation_steps]))
            for start in range(0, len(global_sizes), accumulation_steps)]


def run_epoch(ddp, data, settings, device, epoch, optimizer=None, *, probe=False, height=288, width=352):
    train = optimizer is not None
    model = ddp.module
    model.train(train)
    rank, world = dist.get_rank(), dist.get_world_size()
    sampler = ExactBatchShard(len(data), rank, world, settings['batch_size'], settings['seed'], epoch, train)
    if train and any(n < world for n in sampler.global_sizes):
        raise ValueError('this exact sampler requires a nonempty local microbatch on every rank')
    groups = accumulation_groups(sampler.global_sizes, settings['accumulation_steps']) if train else []
    by_step = {i: (start, end, total) for start, end, total in groups for i in range(start, end)}
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=partial(collate, height=height, width=width),
                        num_workers=settings['num_workers'], worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(settings['seed'] + epoch),
                        pin_memory=device.type == 'cuda')
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    ids, events, gradients = [], [], []
    dist.barrier()
    synchronize(device)
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        if set(batch['splits']) != {data.split}:
            raise ValueError('unexpected SeACT split in minibatch')
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch['inputs'].items()}
        labels = batch['labels'].to(device, non_blocking=True)
        if train:
            start, end, total = by_step[index]
            if index == start:
                optimizer.zero_grad(set_to_none=True)
            sync_context = nullcontext() if index + 1 == end else ddp.no_sync()
        else:
            sync_context = nullcontext()
        with sync_context:
            with torch.set_grad_enabled(train), amp(device):
                logits = (ddp if train else model)(inputs)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite cross entropy')
            if train:
                # DDP averages rank gradients: multiply by world and normalize
                # by the actual whole accumulation-group count, including tails.
                (loss * world / total).backward()
        if train and index + 1 == end:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            if probe:
                gradient = flat_grad(model)
                digest = hashlib.sha256(gradient.cpu().numpy().tobytes()).hexdigest()
                if len(set(gather(digest))) != 1:
                    raise AssertionError('DDP accumulated gradients differ between ranks')
                stage_norms = {name: sum(float(p.grad.detach().abs().sum()) for p in module.parameters()
                                         if p.grad is not None) for name, module in model.named_children()
                               if any(True for _ in module.parameters())}
                if any(value <= 0 for value in stage_norms.values()):
                    raise AssertionError(f'disconnected training stage: {stage_norms}')
                if hasattr(model, 'a') and (model.a.grad is None or not torch.isfinite(model.a.grad) or model.a.grad.abs() == 0):
                    raise AssertionError('fusion scalar has no finite nonzero gradient')
                gradients.append(dict(microbatch_end=index + 1, samples=total, gradient_sha256=digest, stage_l1=stage_norms))
            optimizer.step()
            if rank == 0:
                print(json.dumps(dict(event='optimizer_step', epoch=epoch, microbatch_end=index + 1,
                                      effective_global_batch=total, loss_rank0=float(loss.detach()) / len(labels))), flush=True)
        prediction = logits.detach().topk(min(5, logits.shape[1]), dim=1).indices
        totals += torch.stack((loss.detach().double(), (prediction[:, 0] == labels).sum(),
                               (prediction == labels[:, None]).any(1).sum(), labels.new_tensor(len(labels))))
        ids.extend(batch['sample_ids'])
        events.extend(batch['inputs']['event_counts'].tolist())
    synchronize(device)
    seconds = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    dist.all_reduce(totals)
    shards = gather(ids)
    flat = [sample for shard in shards for sample in shard]
    expected = [row.sample_id for row in data.rows]
    if len(flat) != len(data) or len(set(flat)) != len(flat) or set(flat) != set(expected):
        raise AssertionError('SeACT sample coverage mismatch')
    counts = [n for shard in gather(events) for n in shard]
    result = dict(loss=float(totals[0] / totals[3]), top1=float(totals[1] / totals[3]),
                  top5=float(totals[2] / totals[3]), samples=len(flat), seconds=float(seconds),
                  samples_per_second=len(flat) / float(seconds), duplicates=0, missing=0,
                  rank_sample_counts=[len(shard) for shard in shards], optimizer_steps=len(groups),
                  optimizer_group_samples=[total for _, _, total in groups],
                  event_count=dict(min=min(counts), max=max(counts), mean=sum(counts) / len(counts)),
                  sample_order_sha256=hashlib.sha256(''.join(s + '\n' for s in expected).encode()).hexdigest())
    if probe:
        result['gradient_checks'] = gradients
    return result


class SelectedDataset:
    """Bounded preflight subset preserves original row identity and raw access."""
    def __init__(self, source, indices):
        self.source, self.indices, self.split = source, tuple(indices), source.split
        self.rows = [source.rows[index] for index in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


def preflight_subset(data, count, largest=False):
    if type(count) is not int or count < 1 or count > len(data):
        raise ValueError('preflight group exceeds dataset size')
    # All selection is label independent; worst-sized train recordings exercise
    # native full-event activation memory without truncating or splitting them.
    order = sorted(range(len(data)), key=lambda i: (-data.rows[i].event_count, data.rows[i].sample_id)
                   if largest else (data.rows[i].sample_id,))
    return SelectedDataset(data, order[:count])


def final_test_identity(identity, history, best, checkpoint_path, epochs):
    """Test access is legal only after a complete, contiguous training history."""
    if identity['command'] != 'train' or [row['epoch'] for row in history] != list(range(1, epochs + 1)):
        raise ValueError('final test requires all scheduled training epochs')
    if best is None or best['epoch'] not in range(1, epochs + 1):
        raise ValueError('final test requires an internal-validation selected checkpoint')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['identity'] != identity or checkpoint['epoch'] != best['epoch']:
        raise ValueError('final test selected checkpoint identity mismatch')
    return dict(identity=identity, checkpoint_sha256=sha256_file(checkpoint_path), selected_epoch=best['epoch'])


def verify_exports(output, model, inputs, config, variant, regime, pretrained, device):
    model.eval()
    result = {}
    for kind in ('last', 'best'):
        checkpoint_path = output / f'checkpoint_{kind}.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        restored = make_model(variant, regime, config, pretrained).to(device).eval()
        restored.load_state_dict(checkpoint['model'], strict=True)
        if kind == 'last':
            if any(not torch.equal(value.detach().cpu(), checkpoint['model'][key]) for key, value in model.state_dict().items()):
                raise AssertionError('live/last checkpoint tensor mismatch')
        with torch.no_grad(), amp(device):
            features = restored.forward_embedding(inputs)
            if kind == 'last':
                torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
        export_path = output / f'backbone_{kind}.pt'
        if not export_path.exists():
            export_backbone(restored, export_path)
        exported = load_backbone_export(export_path, map_location='cpu').to(device).eval()
        expected = {k: v for k, v in checkpoint['model'].items() if not k.startswith('classifier.')}
        actual = exported.state_dict()
        if set(expected) != set(actual) or any(not torch.equal(value, actual[key].detach().cpu()) for key, value in expected.items()):
            raise AssertionError('checkpoint/export tensor mismatch')
        with torch.no_grad(), amp(device):
            torch.testing.assert_close(features, exported.forward_embedding(inputs), rtol=0, atol=0)
        result[kind] = dict(strict_checkpoint_load=True, backbone_embedding_bit_exact=True,
                            checkpoint_sha256=sha256_file(checkpoint_path), export_sha256=sha256_file(export_path))
    return result


def execute(args):
    config = load_config(args.config)
    settings = config['training']
    if args.command == 'preflight' and args.resume:
        raise ValueError('preflight cannot resume')
    identity = identity_for(config, args.variant, args.regime, args.command, args.pretrained_backbone)
    if int(os.environ.get('WORLD_SIZE', '0')) != 2 or not torch.cuda.is_available():
        raise RuntimeError('requires exactly two CUDA ranks via torchrun')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise RuntimeError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch')
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    try:
        rank, local = dist.get_rank(), int(os.environ['LOCAL_RANK'])
        device = torch.device('cuda', local)
        seed_all(settings, device)
        datasets = {split: SeActDataset(config['manifest_dir'], config['dataset_root'], split, mode=args.variant)
                    for split in ('train', 'validation')}
        for split, data in datasets.items():
            if len(data) != config['split_sizes'][split]:
                raise ValueError(f'{split} manifest sample count mismatch')
        if {row.sample_id for row in datasets['train'].rows} & {row.sample_id for row in datasets['validation'].rows}:
            raise ValueError('SeACT train/validation overlap')
        if args.command == 'preflight':
            datasets['train'] = preflight_subset(datasets['train'], settings['global_batch_size'], largest=True)
            datasets['validation'] = preflight_subset(datasets['validation'], 2)
        error = [None]
        if rank == 0:
            try:
                prepare_output(args.output_dir, identity, args.resume)
            except Exception as exc:
                error[0] = f'{type(exc).__name__}: {exc}'
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
        model = make_model(args.variant, args.regime, config, args.pretrained_backbone).to(device)
        ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=settings['learning_rate'], momentum=settings['momentum'], weight_decay=settings['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['epochs'])
        first, history, best = 1, [], None
        if args.resume:
            first, history, best = resume_epoch(args.output_dir, ddp, optimizer, scheduler, device, identity)
        torch.cuda.reset_peak_memory_stats(device)
        last = 1 if args.command == 'preflight' else settings['epochs']
        before = cpu_state(model) if args.command == 'preflight' else None
        hw = dict(height=config['model']['height'], width=config['model']['width'])
        for epoch in range(first, last + 1):
            lr = optimizer.param_groups[0]['lr']
            training = run_epoch(ddp, datasets['train'], settings, device, epoch, optimizer,
                                 probe=args.command == 'preflight', **hw)
            validation = run_epoch(ddp, datasets['validation'], settings, device, epoch, **hw)
            scheduler.step()
            improved = best is None or (validation['top1'], -validation['loss']) > (best['top1'], -best['loss'])
            if improved:
                best = dict(epoch=epoch, **validation)
            history.append(dict(epoch=epoch, learning_rate=lr, train=training, validation=validation,
                                latent_weight=float(model.a.detach().sigmoid()) if hasattr(model, 'a') else None))
            save_epoch(args.output_dir, ddp, optimizer, scheduler, device, epoch, identity, history, best, improved)
            if rank == 0:
                print(json.dumps(dict(event='epoch_complete', variant=args.variant, regime=args.regime, **history[-1])), flush=True)
        if before is not None:
            for stage, module in model.named_children():
                if any(True for _ in module.parameters()) and not any(not torch.equal(before[name], p.detach().cpu())
                        for name, p in model.named_parameters() if name.startswith(stage + '.')):
                    raise AssertionError(f'preflight stage did not update: {stage}')
            if hasattr(model, 'a') and torch.equal(before['a'], model.a.detach().cpu()):
                raise AssertionError('preflight global fusion scalar did not update')
        memory = gather(dict(allocated_bytes=torch.cuda.max_memory_allocated(device),
                             reserved_bytes=torch.cuda.max_memory_reserved(device),
                             total_bytes=torch.cuda.get_device_properties(device).total_memory))
        if args.command == 'preflight' and any(row['reserved_bytes'] >= .9 * row['total_bytes'] for row in memory):
            raise RuntimeError('preflight lacks 10 percent GPU memory headroom')
        checks_payload = [None]
        if rank == 0:
            try:
                inputs = {k: v.to(device) for k, v in collate([datasets['validation'][0]], **hw)['inputs'].items()}
                checks_payload[0] = dict(checks=verify_exports(args.output_dir, model, inputs, config,
                                         args.variant, args.regime, args.pretrained_backbone, device))
            except Exception as exc:
                checks_payload[0] = dict(error=f'{type(exc).__name__}: {exc}')
        dist.broadcast_object_list(checks_payload, src=0)
        if 'error' in checks_payload[0]:
            raise RuntimeError(checks_payload[0]['error'])
        checks = checks_payload[0]['checks']
        # Evaluate the selected best checkpoint after export verification. Persist
        # the result immediately, so resuming after a report-write failure reuses
        # the completed evaluation instead of reopening held-out event files.
        final_test = None
        if args.command == 'train':
            final_test_path = args.output_dir / 'final_test.json'
            best_path = args.output_dir / 'checkpoint_best.pt'
            test_identity = final_test_identity(identity, history, best, best_path, settings['epochs'])
            if final_test_path.exists():
                saved = json.loads(final_test_path.read_text())
                if saved.get('test_identity') != test_identity:
                    raise ValueError('persisted final test identity mismatch')
                final_test = saved['metrics']
            else:
                test_data = SeActDataset(config['manifest_dir'], config['dataset_root'], 'test', mode=args.variant,
                                        allow_final_test=True)
                if len(test_data) != config['split_sizes']['test']:
                    raise ValueError('official SeACT test sample count mismatch')
                test_model = make_model(args.variant, args.regime, config, args.pretrained_backbone).to(device)
                checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
                test_model.load_state_dict(checkpoint['model'], strict=True)
                final_test = run_epoch(type('EvaluationModel', (), {'module': test_model})(), test_data,
                                       settings, device, checkpoint['epoch'], **hw)
                if rank == 0:
                    atomic_json(final_test_path, dict(test_identity=test_identity, metrics=final_test))
                dist.barrier()
                del test_model, checkpoint
        if rank == 0:
            atomic_json(args.output_dir / 'report.json', dict(status='PASS' if args.command == 'preflight' else 'complete',
                        identity=identity, completed_epochs=history[-1]['epoch'], best=best,
                        final=history[-1]['validation'], final_test=final_test,
                        final_test_accessed=args.command == 'train', initialization=model.init_metadata,
                        parameters=sum(p.numel() for p in model.parameters()),
                        gpu_memory_by_rank=memory, checkpoint_verification=checks))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--config', type=Path, default=CONFIG)
    for command in ('train', 'preflight'):
        child = sub.add_parser(command)
        child.add_argument('--config', type=Path, default=CONFIG)
        child.add_argument('--variant', choices=VARIANTS, required=True)
        child.add_argument('--regime', choices=REGIMES, required=True)
        child.add_argument('--pretrained-backbone', type=Path)
        child.add_argument('--output-dir', type=Path, required=True)
        child.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'inspect':
        print(json.dumps(dict(config=load_config(args.config), training_launched=False), indent=2))
    else:
        execute(args)


if __name__ == '__main__':
    main()
