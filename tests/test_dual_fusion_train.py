import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ebackbone_v3 import dual_fusion_train as d
from ebackbone_v3.dual_fusion_data import input_keys
from tests.test_hierarchy import sample
from tests.test_hierarchy_polarity_training import assert_nested_equal


class TinyData:
    def __init__(self, mode, split='train', size=7):
        self.mode, self.split = mode, split
        self.rows = [SimpleNamespace(sample_id=f'{split}/{i}') for i in range(size)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        result = sample(index, self.split)
        g = torch.Generator().manual_seed(index + 31)
        result['inputs'].update(event_frame=torch.rand(2, 32, 32, generator=g),
                                voxel_grid=torch.rand(2, 8, 32, 32, generator=g),
                                time_surface=torch.rand(2, 32, 32, generator=g))
        result['inputs'] = {k: v for k, v in result['inputs'].items() if k in input_keys(self.mode)}
        return result


def tiny_config():
    return dict(training=dict(batch_size=2, epochs=50, accumulation_steps=1, seed=20260908,
                              cpu_threads=1, num_workers=0, learning_rate=.05, momentum=.9, weight_decay=.0001),
                model=dict(height=32, width=32, num_classes=100, dimension=256, latent_width=8))


def test_locked_config_and_readonly_inspection():
    config = d.load_config(d.CONFIG)
    assert config['training']['global_batch_size'] == 64
    assert config['training']['epochs'] == 50
    bundled_manifests = Path(__file__).resolve().parents[1] / 'manifests/n_imagenet_mini/supervised-v1'
    assert Path(config['manifest_dir']) == bundled_manifests
    identity = d.identity_for(config, 'dual')
    assert set(identity['source_sha256']) == set(d.SOURCE_FILES)
    assert identity['final_test_accessed'] is False


@pytest.mark.parametrize('key,value', [('epochs', 100), ('batch_size', 16), ('accumulation_steps', 2),
                                      ('learning_rate', .1), ('seed', 20260923)])
def test_reject_protocol_drift(tmp_path, key, value):
    config = json.loads(d.CONFIG.read_text())
    config['training'][key] = value
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        d.load_config(path)


def test_output_collision_and_resume_provenance(tmp_path):
    output = tmp_path / 'new'
    identity = dict(protocol='test', source='first', manifest='frozen')
    d.prepare_output(output, identity)
    with pytest.raises(FileExistsError):
        d.prepare_output(output, identity)
    torch.save(dict(identity=identity), output / 'checkpoint_last.pt')
    d.prepare_output(output, identity, resume=True)
    with pytest.raises(ValueError, match='identity mismatch'):
        d.prepare_output(output, {**identity, 'source': 'changed'}, resume=True)
    (output / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='config identity'):
        d.prepare_output(output, identity, resume=True)


def distributed_worker(rank, init, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init, rank=rank, world_size=2)
    device = torch.device('cpu')
    config = tiny_config()
    settings = config['training']
    reports = {}
    for mode in d.VARIANTS:
        d.seed_all(settings, device)
        model = d.make_model(mode, config)
        ddp = DDP(model, broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        data = TinyData(mode)
        before = d.cpu_state(model)
        tr = d.run_epoch(ddp, data, settings, device, 1, optimizer, probe=True, height=32, width=32)
        val = d.run_epoch(ddp, TinyData(mode, 'validation', 5), settings, device, 1, height=32, width=32)
        assert tr['samples'] == 7 and tr['optimizer_steps'] == 2 and tr['rank_sample_counts'] == [4, 3]
        assert val['samples'] == 5 and val['rank_sample_counts'] == [3, 2]
        for stage, module in model.named_children():
            assert any(not torch.equal(before[name], p) for name, p in model.named_parameters()
                       if name.startswith(stage + '.')), stage
        if mode == 'dual':
            assert not torch.equal(before['a'], model.a)
        # Independent reference: unequal local batches 2/1 equal one global mean CE.
        reference = d.make_model(mode, config)
        reference.load_state_dict(model.state_dict())
        all_batch = d.collate([data[i] for i in range(3)], height=32, width=32)
        rank_batch = d.collate([data[i] for i in range(rank, 3, 2)], height=32, width=32)
        model.train()
        reference.train()
        ddp.zero_grad(set_to_none=True)
        (torch.nn.functional.cross_entropy(ddp(rank_batch['inputs']), rank_batch['labels'], reduction='sum') * 2 / 3).backward()
        torch.nn.functional.cross_entropy(reference(all_batch['inputs']), all_batch['labels']).backward()
        torch.testing.assert_close(d.flat_grad(model), d.flat_grad(reference), atol=3e-5, rtol=3e-3)
        out = Path(folder) / mode
        if rank == 0:
            out.mkdir()
        dist.barrier()
        identity = dict(settings=settings, protocol='synthetic CPU test', variant=mode)
        scheduler.step()
        d.save_epoch(out, ddp, optimizer, scheduler, device, 1, identity, [dict(epoch=1)], dict(epoch=1), True)
        checks = None
        if rank == 0:
            checks = d.verify_exports(out, model, all_batch['inputs'], config, mode, device)
            assert checks['last']['backbone_embedding_bit_exact']
            # Retrying completed export verification is safe and does not overwrite.
            assert d.verify_exports(out, model, all_batch['inputs'], config, mode, device) == checks
        dist.barrier()
        reports[mode] = dict(train=tr, validation=val, export_checks=checks)
        del reference, ddp, model, optimizer, scheduler
    # Epoch-boundary resume preserves model, momentum, schedule and per-rank RNG.
    states = []
    for branch in ('continuous', 'resumed'):
        d.seed_all(settings, device)
        ddp = DDP(d.make_model('dual', config), broadcast_buffers=False)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=.05, momentum=.9, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        out = Path(folder) / branch
        if rank == 0:
            out.mkdir()
        dist.barrier()
        identity = dict(settings=settings, protocol='resume test')
        history = []
        for epoch in (1, 2):
            if branch == 'resumed' and epoch == 2:
                del ddp, optimizer, scheduler
                ddp = DDP(d.make_model('dual', config), broadcast_buffers=False)
                optimizer = torch.optim.SGD(ddp.parameters(), lr=.05, momentum=.9, weight_decay=.0001)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
                first, history, best = d.resume_epoch(out, ddp, optimizer, scheduler, device, identity)
                assert first == 2
            tr = d.run_epoch(ddp, TinyData('dual', size=3), settings, device, epoch, optimizer, height=32, width=32)
            scheduler.step()
            history.append(dict(epoch=epoch, train=tr))
            d.save_epoch(out, ddp, optimizer, scheduler, device, epoch, identity, history, dict(epoch=epoch), True)
        states.append(torch.load(out / 'checkpoint_last.pt', weights_only=False))
        del ddp, optimizer, scheduler
    for key in ('model', 'optimizer', 'scheduler', 'rng_by_rank', 'sampler'):
        assert_nested_equal(states[0][key], states[1][key])
    if rank == 0:
        (Path(folder) / 'checks.json').write_text(json.dumps(dict(variants=reports, exact_resume=True)))
    dist.destroy_process_group()


def test_two_rank_gloo_gradients_tail_resume_and_export(tmp_path):
    mp.spawn(distributed_worker, args=(str(tmp_path / 'init'), str(tmp_path)), nprocs=2, join=True)
    report = json.loads((tmp_path / 'checks.json').read_text())
    assert report['exact_resume']
    assert set(report['variants']) == set(d.VARIANTS)
