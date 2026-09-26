import numpy as np
import pytest
import torch

from ebackbone_v3.representations import SourceIdentity, compute_event_fingerprint
from ebackbone_v3.errors import DatasetError
from ebackbone_v3.v1_data import V1Dataset, prepare_raw_cache, render_v1
from ebackbone_v3.v1_models import MODEL_NAMES, V1Backbone, example_inputs, match_frame_width, profile_macs
from tests.test_manifest_dataset import fixture_release as _fixture_release

fixture_release = _fixture_release


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def fields_source(times=(0, 25, 50, 100)):
    fields = {"x": np.array([0, 0, 1, 639], dtype=np.uint16),
              "y": np.array([0, 0, 1, 479], dtype=np.uint16),
              "t": np.array(times, dtype=np.uint16),
              "p": np.array([False, False, True, True])}
    source = SourceIdentity("train/n00000001/a.npz", "train", compute_event_fingerprint(fields),
                            times[0], times[-1], "[]", 4)
    return fields, source


def test_renderer_mass_polarity_latest_and_frame_identity():
    fields, source = fields_source()
    views = render_v1(fields, source, multiview=True)
    np.testing.assert_array_equal(views["event_frame"], render_v1(fields, source, multiview=False)["event_frame"])
    np.testing.assert_allclose(np.expm1(views["voxel_grid"]).sum(1), np.expm1(views["event_frame"]), atol=1e-6)
    assert np.expm1(views["voxel_grid"])[0, 1, 0, 0] == pytest.approx(0.25)
    assert np.expm1(views["voxel_grid"])[0, 2, 0, 0] == pytest.approx(0.75)
    assert views["time_surface"][0, 0, 0] == pytest.approx(np.exp(-0.75 / 0.2))
    assert views["time_surface"][1, 479, 639] == 1
    assert views["time_surface"][1, 0, 0] == 0
    assert views["voxel_grid"].shape == (2, 8, 480, 640)
    assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in views.values())


def test_zero_duration_and_identity_rejection():
    fields, source = fields_source((10, 10, 10, 10))
    views = render_v1(fields, source, multiview=True)
    assert np.count_nonzero(views["voxel_grid"][:, :7]) == 0
    assert views["time_surface"][0, 0, 0] == 1
    fields["x"][0] = 4
    with pytest.raises(ValueError, match="identity|subset_id"):
        render_v1(fields, source, multiview=True)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_optimizer_and_strict_reload(name, tmp_path):
    torch.manual_seed(123)
    model = V1Backbone(name, frame_width=96 if name == "frame_matched" else 32)
    inputs = {k: torch.rand_like(v) for k, v in example_inputs(name, 32, 40).items()}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([3]))
    assert torch.isfinite(loss)
    loss.backward()
    for component in ("frame", "fusion", "trunk", "classifier") + (("voxel", "surface") if model.multiview else ()):
        params = [p for n, p in model.named_parameters() if n.startswith(component + ".")]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert sum(p.grad.abs().sum().item() for p in params) > 0
    optimizer.step()
    for component in ("frame", "fusion", "trunk", "classifier") + (("voxel", "surface") if model.multiview else ()):
        assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(component + "."))
    model.eval()
    expected = model(inputs).detach()
    path = tmp_path / "state.pt"
    torch.save(model.state_dict(), path)
    restored = V1Backbone(name, frame_width=model.frame_width).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
    assert model.forward_features(inputs).shape == (1, 256, 4, 5)


def test_same_arch_independent_and_temporal_structure():
    model = V1Backbone("same_arch")
    assert model.frame.body[0].in_channels == 2
    assert model.voxel.body[0].in_channels == 16
    assert model.frame.body[3].body[0].weight.data_ptr() != model.surface.body[3].body[0].weight.data_ptr()
    heterogeneous = V1Backbone("heterogeneous")
    assert any(isinstance(m, torch.nn.Conv3d) for m in heterogeneous.voxel.modules())
    assert any(isinstance(m, torch.nn.Conv2d) and m.groups == m.in_channels and m.kernel_size == (5, 5)
               for m in heterogeneous.surface.modules())


def test_compute_match_and_rng_preservation():
    state = torch.get_rng_state()
    matched = match_frame_width()
    assert torch.equal(state, torch.get_rng_state())
    assert matched["relative_mac_error"] <= 0.05
    model = V1Backbone("frame_matched", frame_width=matched["frame_width"])
    assert profile_macs(model)["macs"] == matched["macs"]


def test_fail_closed_final_test(tmp_path):
    with pytest.raises(DatasetError, match="test|final"):
        V1Dataset(tmp_path, tmp_path, "test", multiview=True)


def test_reject_wrong_inputs():
    with pytest.raises(ValueError, match="exactly"):
        V1Backbone("frame")(example_inputs("heterogeneous", 32, 40))
    inputs = example_inputs("heterogeneous", 32, 40)
    inputs["voxel_grid"] = inputs["voxel_grid"][:, :, :5]
    with pytest.raises(ValueError, match="8-bin"):
        V1Backbone("heterogeneous")(inputs)


def test_real_adapter_cache_identity_and_no_final_test(fixture_release, tmp_path, monkeypatch):
    from ebackbone_v3 import v1_data
    locations = dict(manifest_dir=fixture_release.manifest_dir, dataset_root=fixture_release.dataset_root)
    original_open = v1_data.raw.open_dataset

    def guarded_open(*args, **kwargs):
        assert kwargs["split"] != "test"
        return original_open(*args, **kwargs)

    monkeypatch.setattr(v1_data.raw, "open_dataset", guarded_open)
    cache = tmp_path / "raw"
    prepared = prepare_raw_cache(**locations, cache_root=cache)
    assert prepared == {"written": 200, "verified_existing": 0, "test_accessed": False}
    repeated = prepare_raw_cache(**locations, cache_root=cache)
    assert repeated["written"] == 0 and repeated["verified_existing"] == 200
    multi = V1Dataset(**locations, split="train", multiview=True, raw_cache=cache)
    frame = V1Dataset(**locations, split="train", multiview=False)
    a, b = multi[0], frame[0]
    assert a["source"] == b["source"]
    torch.testing.assert_close(a["inputs"]["event_frame"], b["inputs"]["event_frame"], atol=0, rtol=0)
    assert set(b["inputs"]) == {"event_frame"}
    target = cache / f"{multi.rows[0].raw_content_sha256}.npz"
    payload = target.read_bytes()
    target.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(DatasetError, match="SHA-256"):
        multi[0]
    target.unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        multi[0]


def test_diagnostic_subset_independent_of_representations(fixture_release):
    locations = dict(manifest_dir=fixture_release.manifest_dir, dataset_root=fixture_release.dataset_root)
    frame = V1Dataset(**locations, split="train", multiview=False, limit=8, seed=123)
    multi = V1Dataset(**locations, split="train", multiview=True, limit=8, seed=123)
    validation = V1Dataset(**locations, split="validation", multiview=False)
    assert [r.sample_id for r in frame.rows] == [r.sample_id for r in multi.rows]
    assert not {r.sample_id for r in frame.rows}.intersection(r.sample_id for r in validation.rows)
