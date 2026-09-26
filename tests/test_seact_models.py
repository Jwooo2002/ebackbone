import pytest
import torch

from ebackbone_v3.dual_fusion_models import DualFusionBackbone, export_backbone as export_mini
from ebackbone_v3.seact_models import SeActBackbone, make_model, export_backbone, load_backbone_export
from tests.test_dual_fusion_train import TinyData
from ebackbone_v3.dual_fusion_data import collate


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('mode', ['hierarchy_only', 'dual'])
def test_chunked_full_event_pooling_matches_original_forward_and_gradients(mode):
    reference = SeActBackbone(mode=mode, height=32, width=32, point_chunk_size=0)
    chunked = SeActBackbone(mode=mode, height=32, width=32, point_chunk_size=3)
    data = TinyData(mode)
    batch = collate([data[0], data[2]], height=32, width=32)
    # Multiple chunks cross sample boundaries without dropping or duplicating events.
    expected = reference(batch['inputs'])
    actual = chunked(batch['inputs'])
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    torch.nn.functional.cross_entropy(expected, batch['labels']).backward()
    torch.nn.functional.cross_entropy(actual, batch['labels']).backward()
    for (name, original), (other, parameter) in zip(reference.named_parameters(), chunked.named_parameters()):
        assert name == other and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, original.grad, atol=2e-5, rtol=3e-3)
    chunked.eval()
    with torch.no_grad():
        torch.testing.assert_close(chunked(batch['inputs']), expected.detach(), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('mode', ['hierarchy_only', 'latent_only', 'dual'])
def test_finetune_uses_exact_backbone_fresh_matched_head_and_portable_export(tmp_path, mode):
    source = DualFusionBackbone(mode, height=32, width=32)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if not name.startswith('classifier.'):
                parameter.add_(.01)
    mini_path = tmp_path / 'mini.pt'
    export_mini(source, mini_path)
    target = make_model(mode, 'finetune', pretrained_backbone=mini_path, height=32, width=32, point_chunk_size=3)
    scratch = make_model(mode, 'scratch', height=32, width=32, point_chunk_size=3)
    assert target.classifier.out_features == 58
    assert target.init_metadata['classifier_initial_sha256'] == scratch.init_metadata['classifier_initial_sha256']
    assert target.init_metadata['pretrained_backbone_sha256']
    assert scratch.init_metadata['pretrained_backbone_sha256'] is None
    assert all(p.requires_grad for p in target.parameters())
    for key, value in source.state_dict().items():
        if not key.startswith('classifier.'):
            assert torch.equal(target.state_dict()[key], value)
    data = TinyData(mode)
    batch = collate([data[0], data[1]], height=32, width=32)
    loss = torch.nn.functional.cross_entropy(target(batch['inputs']), batch['labels'])
    loss.backward()
    for parameter in target.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    optimizer = torch.optim.SGD(target.parameters(), lr=.01)
    optimizer.step()
    target.eval()
    with torch.no_grad():
        expected = target.forward_embedding(batch['inputs'])
    path = tmp_path / 'seact.pt'
    export_backbone(target, path)
    restored = load_backbone_export(path)
    assert not hasattr(restored, 'classifier')
    with torch.no_grad():
        torch.testing.assert_close(restored(batch['inputs']), expected, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        export_backbone(target, path)


def test_initialization_and_export_reject_wrong_regime_variant_contract(tmp_path):
    mini = tmp_path / 'mini.pt'
    export_mini(DualFusionBackbone('hierarchy_only', height=32, width=32), mini)
    with pytest.raises(ValueError, match='only finetune'):
        make_model('dual', 'scratch', pretrained_backbone=mini)
    with pytest.raises(ValueError, match='only finetune'):
        make_model('dual', 'finetune')
    with pytest.raises(ValueError, match='mismatch'):
        make_model('dual', 'finetune', pretrained_backbone=mini)
    path = tmp_path / 'seact.pt'
    export_backbone(make_model('hierarchy_only', 'scratch', height=32, width=32), path)
    payload = torch.load(path, weights_only=True)
    payload['input_contract_sha256'] = 'wrong'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='input contract'):
        load_backbone_export(path)
    clean = tmp_path / 'missing_geometry.pt'
    export_backbone(make_model('hierarchy_only', 'scratch', height=32, width=32), clean)
    payload = torch.load(clean, weights_only=True)
    del payload['model_config']['height']
    torch.save(payload, clean)
    with pytest.raises(ValueError, match='input contract'):
        load_backbone_export(clean)
