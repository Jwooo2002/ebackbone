from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ebackbone_v3.probe import compute_event_subset_id
from ebackbone_v3.representations import (
    POLARITY_ORDER,
    ProductionRepresentations,
    RendererConfig,
    RepresentationError,
    SourceIdentity,
    read_representation_cache,
    render_production_representations,
    representation_cache_key,
    write_representation_cache,
)


def _fields(
    *,
    x: list[int] | None = None,
    y: list[int] | None = None,
    t: list[int] | None = None,
    p: list[bool] | None = None,
) -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([0, 639, 320, 10] if x is None else x, dtype="<u2"),
        "y": np.asarray([0, 479, 240, 10] if y is None else y, dtype="<u2"),
        "t": np.asarray([10, 20, 25, 30] if t is None else t, dtype="<u2"),
        "p": np.asarray([False, True, False, True] if p is None else p, dtype="?"),
    }


def _source(fields: dict[str, np.ndarray], sample_id: str = "train/n/sample.npz") -> SourceIdentity:
    return SourceIdentity(
        sample_id=sample_id,
        split=sample_id.split("/", 1)[0],
        event_subset_id=compute_event_subset_id(
            fields,
            {"x": "x", "y": "y", "timestamp": "t", "polarity": "p"},
        ),
        temporal_start=int(fields["t"][0]),
        temporal_end=int(fields["t"][-1]),
        interval_closure="[]",
        event_count=len(fields["t"]),
    )


def test_canonical_polarity_order_is_shared_by_every_representation() -> None:
    fields = _fields(x=[0, 639], y=[0, 479], t=[0, 10], p=[False, True])
    bundle = render_production_representations(fields, source=_source(fields))

    assert POLARITY_ORDER == ("negative", "positive")
    assert bundle.config.polarity_order == POLARITY_ORDER
    assert bundle.event_frame.shape == (2, 480, 640)
    assert bundle.voxel_grid.shape == (2, 5, 480, 640)
    assert bundle.time_surface.shape == (2, 480, 640)
    assert bundle.event_frame[0, 0, 0] == pytest.approx(np.log1p(1.0))
    assert bundle.event_frame[1, 479, 639] == pytest.approx(np.log1p(1.0))
    assert bundle.time_surface[0, 0, 0] > 0
    assert bundle.time_surface[1, 479, 639] == 1.0


def test_temporal_boundaries_and_linear_interpolation() -> None:
    fields = _fields(x=[0, 0, 0], y=[0, 0, 0], t=[0, 5, 10], p=[False] * 3)
    bundle = render_production_representations(fields, source=_source(fields))
    voxel_mass = np.expm1(bundle.voxel_grid)

    assert voxel_mass[0, 0, 0, 0] == 1.0
    assert voxel_mass[0, 2, 0, 0] == 1.0
    assert voxel_mass[0, 4, 0, 0] == 1.0
    assert voxel_mass[1].sum() == 0.0

    # The source interval must equal observed support; use a three-event fixture
    # so the middle event lands at z=4/3 for B=5.
    interpolated = _fields(x=[1, 0, 2], y=[1, 0, 2], t=[0, 1, 3], p=[False, True, False])
    result = render_production_representations(interpolated, source=_source(interpolated))
    mass = np.expm1(result.voxel_grid)
    assert mass[1, 1, 0, 0] == pytest.approx(2.0 / 3.0)
    assert mass[1, 2, 0, 0] == pytest.approx(1.0 / 3.0)


def test_zero_duration_is_finite_and_maps_to_last_bin_and_zero_background() -> None:
    fields = _fields(x=[1, 1], y=[2, 2], t=[7, 7], p=[False, True])
    bundle = render_production_representations(fields, source=_source(fields))
    mass = np.expm1(bundle.voxel_grid)

    assert mass[0, -1].sum() == pytest.approx(1.0)
    assert mass[1, -1].sum() == pytest.approx(1.0)
    assert mass[:, :-1].sum() == 0.0
    assert np.isfinite(bundle.event_frame).all()
    assert np.isfinite(bundle.voxel_grid).all()
    assert np.isfinite(bundle.time_surface).all()
    assert bundle.time_surface[:, 0, 0].tolist() == [0.0, 0.0]
    # Both polarities occupy native pixel (x=1,y=2) and are maximally recent.
    assert bundle.time_surface[0, 2, 1] == 1.0
    assert bundle.time_surface[1, 2, 1] == 1.0


def test_empty_pixels_and_empty_polarity_branch_are_exact_zero() -> None:
    fields = _fields(x=[10, 10], y=[10, 10], t=[0, 10], p=[False, False])
    bundle = render_production_representations(fields, source=_source(fields))

    assert np.count_nonzero(bundle.event_frame[1]) == 0
    assert np.count_nonzero(bundle.voxel_grid[1]) == 0
    assert np.count_nonzero(bundle.time_surface[1]) == 0
    assert bundle.time_surface[:, 479, 639].tolist() == [0.0, 0.0]


def test_determinism_and_event_count_conservation_before_log1p() -> None:
    fields = _fields()
    source = _source(fields)
    first = render_production_representations(fields, source=source)
    second = render_production_representations(fields, source=source)

    for name in first.tensors():
        np.testing.assert_array_equal(first.tensors()[name], second.tensors()[name])
    assert np.expm1(first.event_frame).sum(dtype=np.float64) == pytest.approx(len(fields["t"]))
    assert np.expm1(first.voxel_grid).sum(dtype=np.float64) == pytest.approx(len(fields["t"]))


def test_one_source_fingerprint_and_interval_bind_all_three_representations() -> None:
    fields = _fields()
    source = _source(fields)
    bundle = render_production_representations(fields, source=source)

    assert bundle.source is source
    assert set(bundle.tensors()) == {"event_frame", "voxel_grid", "time_surface"}
    assert bundle.source.event_subset_id == _source(fields).event_subset_id
    assert (bundle.source.temporal_start, bundle.source.temporal_end) == (10, 30)


def test_cache_round_trip_is_deterministic_and_parameter_changes_invalidate(tmp_path: Path) -> None:
    fields = _fields()
    source = _source(fields)
    bundle = render_production_representations(fields, source=source)
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    write_representation_cache(first_path, bundle)
    write_representation_cache(second_path, bundle)
    assert first_path.read_bytes() == second_path.read_bytes()

    loaded = read_representation_cache(
        first_path,
        expected_source=source,
        expected_config=bundle.config,
    )
    for name in bundle.tensors():
        np.testing.assert_array_equal(bundle.tensors()[name], loaded.tensors()[name])

    changed = RendererConfig(voxel_bins=6)
    assert representation_cache_key(source=source, config=changed) != bundle.cache_key
    with pytest.raises(RepresentationError, match="D012"):
        read_representation_cache(
            first_path,
            expected_source=source,
            expected_config=changed,
        )


def test_cache_rejects_forged_shapes_and_manifest_contract(tmp_path: Path) -> None:
    fields = _fields()
    source = _source(fields)
    bundle = render_production_representations(fields, source=source)
    forged = ProductionRepresentations(
        event_frame=np.zeros((1,), dtype=np.float32),
        voxel_grid=np.zeros((1,), dtype=np.float32),
        time_surface=np.zeros((1,), dtype=np.float32),
        source=source,
        config=bundle.config,
        cache_key=bundle.cache_key,
    )
    with pytest.raises(RepresentationError, match="production shape"):
        write_representation_cache(tmp_path / "forged.npz", forged)

    valid_path = tmp_path / "valid.npz"
    write_representation_cache(valid_path, bundle)
    with np.load(valid_path, allow_pickle=False) as archive:
        tensors = {name: archive[name].copy() for name in bundle.tensors()}
        manifest = json.loads(bytes(archive["manifest_utf8"]).decode("utf-8"))
    manifest["contract_name"] = "wrong-contract"
    with valid_path.open("wb") as handle:
        np.savez(
            handle,
            **tensors,
            manifest_utf8=np.frombuffer(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                dtype=np.uint8,
            ),
        )
    with pytest.raises(RepresentationError, match="contract name"):
        read_representation_cache(
            valid_path,
            expected_source=source,
            expected_config=bundle.config,
        )


def test_invalid_input_is_rejected_without_filtering_or_fabrication() -> None:
    fields = _fields()
    bad = dict(fields)
    bad["x"] = fields["x"].copy()
    bad["x"][0] = 640
    with pytest.raises(RepresentationError, match="coordinates"):
        render_production_representations(bad, source=_source(bad))

    with pytest.raises(RepresentationError, match="at least one event"):
        SourceIdentity(
            sample_id="train/empty",
            split="train",
            event_subset_id="sha256:" + "0" * 64,
            temporal_start=0,
            temporal_end=0,
            interval_closure="[]",
            event_count=0,
        )
