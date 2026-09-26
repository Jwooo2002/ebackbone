from functools import partial

import pytest
import torch

from ebackbone_v3 import hierarchy_ablation_training as training
from ebackbone_v3.hierarchy_ablation_models import EventInteraction, SpatialMeanDownsample, MODEL_NAMES, make_model, profile_macs
from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.hierarchy_data import collate
from tests.test_hierarchy import sample, FakeDataset
from tests.test_v1_training import assert_nested_equal


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def dense_sample(i):
    s = sample(i)
    s["inputs"]["voxel_lower"].zero_()
    s["inputs"]["voxel_upper"].fill_(64)
    return s


def test_skip_mean_matches_pooling_forward_and_backward():
    x=torch.randn(2,3,8,8,10,requires_grad=True)
    actual=SpatialMeanDownsample()(x)
    expected=torch.nn.functional.avg_pool3d(x,(1,2,2))
    torch.testing.assert_close(actual,expected)
    a,=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)
    b,=torch.autograd.grad(expected.square().sum(),x)
    torch.testing.assert_close(a,b)


def test_neighbor_dependence_self_control_and_group_isolation():
    torch.manual_seed(1)
    features = torch.randn(8, 32)
    points = torch.randn(8, 4)
    points[:, 2] = torch.arange(8) / 10
    bucket = torch.tensor([0, 0, 0, 0, 512, 512, 512, 512])
    local = EventInteraction(height=32, width=32)
    control = EventInteraction(local=False, height=32, width=32)
    with torch.no_grad():
        local.project.weight.normal_(std=.1)
    control.load_state_dict(local.state_dict(), strict=True)
    baseline = local(features, points, bucket)
    changed = features.clone(); changed[1] += 2
    assert not torch.allclose(baseline[0], local(changed, points, bucket)[0])
    assert torch.equal(control(features, points, bucket)[0], control(changed, points, bucket)[0])
    changed = features.clone(); changed[4:] += 20
    assert torch.equal(baseline[:4], local(changed, points, bucket)[:4])
    assert torch.equal(local(features[:1], points[:1], bucket[:1]), features[:1])
    torch.testing.assert_close(baseline, torch.cat([local(features[:4], points[:4], bucket[:4]),
                                                   local(features[4:], points[4:], bucket[4:])]))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_identity_all_event_contract_two_updates_and_bf16(name, tmp_path):
    torch.manual_seed(42)
    baseline = HierarchyV1(height=32, width=32).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = make_model(name, height=32, width=32).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[key])
    samples = [dense_sample(0), dense_sample(1)]
    inputs = collate(samples, height=32, width=32)["inputs"]
    assert torch.equal(model(inputs), baseline(inputs))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, weight_decay=0)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0, 1]))
        assert logits.dtype == torch.bfloat16 and torch.isfinite(loss)
        loss.backward()
        for key, module in model.named_children():
            params = list(module.parameters())
            if not params:
                continue
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), key
            assert sum(float(p.grad.abs().sum()) for p in params) > 0, key
        if hasattr(model, "event_interaction"):
            for module in [model.event_interaction.reduce, model.event_interaction.message]:
                grad = sum(float(p.grad.abs().sum()) for p in module.parameters())
                assert (grad == 0) if step == 0 else (grad > 0)
        optimizer.step()
    assert all(any(not torch.equal(before[k], p) for k, p in model.named_parameters() if k.startswith(name_ + '.'))
               for name_, module in model.named_children() if list(module.parameters()))
    individual = torch.cat([model(collate([s], height=32, width=32)["inputs"]) for s in samples])
    torch.testing.assert_close(model(inputs), individual, atol=3e-6, rtol=1e-5)
    path = tmp_path / 'model.pt'; torch.save(model.state_dict(), path)
    restored = make_model(name, height=32, width=32).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(model(inputs), restored(inputs))
    if hasattr(model, 'early_skip'):
        with torch.no_grad(): model.early_skip[-1].weight.zero_(); model.early_skip[-1].bias.zero_()
        baseline.load_state_dict({k: v for k,v in model.state_dict().items() if k in baseline.state_dict()})
        assert torch.equal(model(inputs), baseline(inputs))


def test_profiles_match_local_control_and_account_for_event_count():
    local, skip, control = [profile_macs(make_model(name)) for name in MODEL_NAMES]
    assert local['parameters'] == control['parameters']
    assert local['macs'] == control['macs']
    assert skip['parameters'] == 2670724 + 512*128 + 128
    assert skip['macs'] == 7374657280 + 512*128*30*40
    for name in MODEL_NAMES:
        a,b = [profile_macs(make_model(name), n) for n in (4,17)]
        assert b['macs'] - a['macs'] == 13*a['point_macs_per_event']


@pytest.mark.parametrize('name', MODEL_NAMES)
def test_exact_training_resume(name, monkeypatch, tmp_path):
    monkeypatch.setattr(training, 'HierarchyDataset', FakeDataset)
    monkeypatch.setattr(training, 'make_model', lambda name, num_classes=100, height=32, width=32:
                        make_model(name, num_classes, height, width))
    monkeypatch.setattr(training, 'collate', partial(collate, height=32, width=32))
    config=training.TrainConfig(epochs=2,batch_size=2,accumulation_steps=1,device='cpu',precision='float32',num_workers=0,cpu_threads=1)
    def run(folder, **kwargs):
        return training.run_model(name,config,manifest_dir='unused',dataset_root='unused',output_dir=tmp_path/folder,**kwargs)
    run('whole'); run('resume',stop_after_epoch=1); resumed=run('resume',resume=True)
    states=[torch.load(tmp_path/f/'checkpoint_last.pt',weights_only=False) for f in ['whole','resume']]
    for key in ['model','optimizer','scheduler']: assert_nested_equal(states[0][key],states[1][key])
    assert resumed['checkpoint_verification']['logits_bit_exact']
    with pytest.raises(ValueError,match='overwrite'): run('whole')
