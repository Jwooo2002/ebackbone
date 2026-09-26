"""Bounded CPU tests; no real SeACT recording or CUDA training is opened."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ebackbone_v3 import seact_train as d
from ebackbone_v3.dual_fusion_models import DualFusionBackbone, export_backbone as export_mini
from tests.test_dual_fusion_train import TinyData as MiniTinyData
from tests.test_hierarchy_polarity_training import assert_nested_equal


class TinyData(MiniTinyData):
    def __init__(self, mode, split='train', size=7):
        super().__init__(mode, split, size)
        self.rows = [SimpleNamespace(sample_id=f'{split}/{i}', event_count=5 + i) for i in range(size)]


def config():
    return dict(training=dict(batch_size=2, epochs=50, accumulation_steps=2, seed=20260908,
                              cpu_threads=1, num_workers=0, learning_rate=.01, momentum=.9, weight_decay=.0001),
                model=dict(height=32, width=32, num_classes=58, dimension=256, latent_width=8, point_chunk_size=4))


def test_accumulation_group_normalization_and_empty_tail_guard():
    assert d.accumulation_groups([2] * 203, 4) == [(i, min(i + 4, 203), min(4, 203 - i) * 2) for i in range(0, 203, 4)]
    assert d.accumulation_groups([4, 4, 4, 3], 2) == [(0, 2, 8), (2, 4, 7)]
    with pytest.raises(ValueError):
        d.accumulation_groups([2], 0)


def test_preflight_is_one_group_largest_full_recordings():
    data = TinyData('dual', size=10)
    selected = d.preflight_subset(data, 8, largest=True)
    assert len(selected) == 8
    assert [row.event_count for row in selected.rows] == list(range(14, 6, -1))
    assert selected[0]['sample_id'] == 'train/9'
    assert selected[0]['inputs']['event_counts'].item() == 14


def test_regime_and_output_identity_guards(tmp_path):
    with pytest.raises(ValueError, match='pretrained'):
        d.make_model('dual', 'scratch', config(), tmp_path / 'not_allowed.pt')
    with pytest.raises(ValueError, match='pretrained'):
        d.make_model('dual', 'finetune', config())
    output = tmp_path / 'run'
    identity = dict(manifests='same', pretrained='initial', sources='same')
    d.prepare_output(output, identity)
    with pytest.raises(FileExistsError):
        d.prepare_output(output, identity)
    torch.save(dict(identity=identity), output / 'checkpoint_last.pt')
    d.prepare_output(output, identity, resume=True)
    with pytest.raises(ValueError, match='identity mismatch'):
        d.prepare_output(output, {**identity, 'pretrained': 'changed'}, resume=True)


@pytest.mark.parametrize('field,value', [('height', 260), ('width', 346), ('dimension', 128),
                                        ('point_chunk_size', 0)])
def test_configuration_pins_geometry_and_architecture(tmp_path, field, value):
    cfg = json.loads(d.CONFIG.read_text())
    cfg['model'][field] = value
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='pinned SeACT'):
        d.load_config(path)


def test_final_test_requires_complete_training_and_selected_checkpoint(tmp_path):
    identity = dict(command='train', variant='dual')
    best = dict(epoch=35)
    path = tmp_path / 'checkpoint_best.pt'
    torch.save(dict(identity=identity, epoch=35), path)
    history = [dict(epoch=epoch) for epoch in range(1, 51)]
    result = d.final_test_identity(identity, history, best, path, 50)
    assert result['selected_epoch'] == 35 and result['checkpoint_sha256'] == d.sha256_file(path)
    for invalid_history in (history[:-1], history[1:], history[:20] + history[21:]):
        with pytest.raises(ValueError, match='all scheduled'):
            d.final_test_identity(identity, invalid_history, best, path, 50)
    with pytest.raises(ValueError, match='all scheduled'):
        d.final_test_identity({**identity, 'command': 'preflight'}, history, best, path, 50)
    with pytest.raises(ValueError, match='checkpoint identity'):
        d.final_test_identity(identity, history, dict(epoch=34), path, 50)


def worker(rank, init, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init, rank=rank, world_size=2)
    device = torch.device('cpu')
    cfg = config()
    settings = cfg['training']
    results = {}
    for regime in d.REGIMES:
        for variant in d.VARIANTS:
            pretrained = Path(folder) / f'mini_{variant}.pt' if regime == 'finetune' else None
            if rank == 0 and pretrained:
                mini = DualFusionBackbone(mode=variant, height=32, width=32)
                # Distinguish fine-tuned initial values from a seeded scratch model.
                with torch.no_grad():
                    for p in mini.parameters():
                        p.mul_(.9)
                export_mini(mini, pretrained)
            dist.barrier()
            d.seed_all(settings, device)
            model = d.make_model(variant, regime, cfg, pretrained)
            reference = d.make_model(variant, regime, cfg, pretrained)
            ddp = DDP(model, broadcast_buffers=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.0001)
            refopt = torch.optim.SGD(reference.parameters(), lr=.01, momentum=.9, weight_decay=.0001)
            data = TinyData(variant, size=7)
            # One accumulation group of 7 samples, comprising unequal 4+3
            # microbatches, must match direct global-mean CE and one SGD step.
            tr = d.run_epoch(ddp, data, settings, device, 1, optimizer, probe=True, height=32, width=32)
            batch = d.collate([data[i] for i in range(7)], height=32, width=32)
            torch.nn.functional.cross_entropy(reference(batch['inputs']), batch['labels']).backward()
            torch.testing.assert_close(d.flat_grad(model), d.flat_grad(reference), atol=3e-5, rtol=3e-3)
            refopt.step()
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, reference.state_dict()[key], atol=3e-6, rtol=3e-4)
            assert tr['samples'] == 7 and tr['optimizer_steps'] == 1 and tr['optimizer_group_samples'] == [7]
            val = d.run_epoch(ddp, TinyData(variant, 'validation', 5), settings, device, 1, height=32, width=32)
            assert val['rank_sample_counts'] == [3, 2] and val['optimizer_steps'] == 0
            out = Path(folder) / (regime + '_' + variant)
            if rank == 0:
                out.mkdir()
            dist.barrier()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            scheduler.step()
            identity = dict(settings=settings, regime=regime, variant=variant)
            d.save_epoch(out, ddp, optimizer, scheduler, device, 1, identity, [dict(epoch=1)], dict(epoch=1), True)
            if rank == 0:
                checks = d.verify_exports(out, model, batch['inputs'], cfg, variant, regime, pretrained, device)
                assert checks['last']['strict_checkpoint_load'] and checks['best']['backbone_embedding_bit_exact']
            dist.barrier()
            results[regime + '_' + variant] = dict(train=tr, validation=val)
            del model, reference, ddp, optimizer, refopt, scheduler
    # Actual production microbatch 1/GPU, accumulation4: 14 samples => groups8,6.
    settings = {**settings, 'batch_size': 1, 'accumulation_steps': 4}
    cfg = {**cfg, 'training': settings}
    states = []
    for branch in ('continuous', 'resumed'):
        d.seed_all(settings, device)
        ddp = DDP(d.make_model('dual', 'scratch', cfg), broadcast_buffers=False)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=.01, momentum=.9, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        output = Path(folder) / branch
        if rank == 0:
            output.mkdir()
        dist.barrier()
        identity = dict(settings=settings, protocol='SeACT CPU exact resume')
        history = []
        for epoch in (1, 2):
            if branch == 'resumed' and epoch == 2:
                del ddp, optimizer, scheduler
                ddp = DDP(d.make_model('dual', 'scratch', cfg), broadcast_buffers=False)
                optimizer = torch.optim.SGD(ddp.parameters(), lr=.01, momentum=.9, weight_decay=.0001)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
                first, history, best = d.resume_epoch(output, ddp, optimizer, scheduler, device, identity)
                assert first == 2
            tr = d.run_epoch(ddp, TinyData('dual', size=14), settings, device, epoch, optimizer, height=32, width=32)
            assert tr['optimizer_group_samples'] == [8, 6]
            scheduler.step()
            history.append(dict(epoch=epoch, train=tr))
            d.save_epoch(output, ddp, optimizer, scheduler, device, epoch, identity, history, dict(epoch=epoch), True)
        states.append(torch.load(output / 'checkpoint_last.pt', map_location='cpu', weights_only=False))
        del ddp, optimizer, scheduler
    for key in ('model', 'optimizer', 'scheduler', 'rng_by_rank', 'sampler'):
        assert_nested_equal(states[0][key], states[1][key])
    if rank == 0:
        (Path(folder) / 'checks.json').write_text(json.dumps(dict(six_regimes=results, exact_resume=True)))
    dist.destroy_process_group()


def test_two_rank_six_regimes_global_mean_accumulation_and_resume(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / 'init'), str(tmp_path)), nprocs=2, join=True)
    result = json.loads((tmp_path / 'checks.json').read_text())
    assert len(result['six_regimes']) == 6 and result['exact_resume']
