"""Matched SeACT fine-tune/scratch models with memory-bounded full-event pooling."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .dual_fusion_models import DualFusionBackbone, load_backbone_export as load_mini_export
from .hierarchy_models import INPUT_KEYS

EXPORT_VERSION = 'seact-dual-backbone-1'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class SeActBackbone(DualFusionBackbone):
    def __init__(self, mode='dual', num_classes=58, height=288, width=352,
                 dimension=256, seed=20260908, latent_width=8, point_chunk_size=65536,
                 *, include_classifier=True):
        super().__init__(mode, num_classes, height, width, dimension, seed, latent_width,
                         include_classifier=include_classifier)
        if type(point_chunk_size) is not int or point_chunk_size < 0:
            raise ValueError('point_chunk_size must be a nonnegative integer')
        self.point_chunk_size = point_chunk_size
        self.init_metadata = {}

    def construction_config(self):
        return {**super().construction_config(), 'point_chunk_size': self.point_chunk_size}

    def _chunk_sums(self, points, lower, upper, alpha, cells):
        features = self.hierarchy.point(points)
        values = torch.cat((features.float(), torch.ones_like(features[:, :1], dtype=torch.float32)), 1)
        weight = alpha.float().unsqueeze(1)
        sums = values.new_zeros(cells, 33)
        sums.index_add_(0, lower, values * (1 - weight))
        sums.index_add_(0, upper, values * weight)
        return sums

    def _hierarchy_features(self, inputs):
        if self.point_chunk_size == 0:
            return self.hierarchy.forward_features({key: inputs[key] for key in INPUT_KEYS})
        points = inputs['points']
        if points.ndim != 2 or points.shape[1] != 4 or not points.shape[0]:
            raise ValueError('points must be nonempty packed [N,4]')
        if any(inputs[key].shape != points.shape[:1] for key in ('voxel_lower', 'voxel_upper', 'alpha')):
            raise ValueError('routing must align with every point')
        batch = inputs['event_counts'].shape[0]
        h, w = self.height // 4, self.width // 4
        cells = batch * 8 * h * w
        total = points.new_zeros(cells, 33, dtype=torch.float32)
        for start in range(0, points.shape[0], self.point_chunk_size):
            stop = start + self.point_chunk_size
            args = (points[start:stop], inputs['voxel_lower'][start:stop],
                    inputs['voxel_upper'][start:stop], inputs['alpha'][start:stop], cells)
            if self.training and torch.is_grad_enabled():
                sums = checkpoint(self._chunk_sums, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                sums = self._chunk_sums(*args)
            total = total + sums
        # Aggregate every event before division. Chunks neither define temporal
        # windows nor change event availability. Addition order can change FP32
        # roundoff relative to the original all-at-once scatter.
        mass = total[:, 32:]
        values = torch.cat((total[:, :32] / mass.clamp_min(1e-6), mass.log1p()), 1)
        x = values.view(batch, 8, h, w, 33).permute(0, 4, 1, 2, 3).contiguous()
        hierarchy = self.hierarchy
        x = hierarchy.voxel_stage2(hierarchy.voxel_stage1(hierarchy.voxel_projection(x)))
        return hierarchy.frame_stage(hierarchy.temporal_collapse(x.flatten(1, 2)))

    def branch_embeddings(self, inputs):
        self._validate(inputs)
        result = {}
        if self.mode != 'latent_only':
            result['hierarchy'] = self.hierarchy_projection(self._hierarchy_features(inputs).mean(dim=(-2, -1)))
        if self.mode != 'hierarchy_only':
            result['latent'] = self.latent_projection(self.latent(inputs))
        return result


def make_model(variant, regime, *, pretrained_backbone=None, seed=20260908,
               height=288, width=352, num_classes=58, dimension=256, latent_width=8,
               point_chunk_size=65536):
    if regime not in ('finetune', 'scratch'):
        raise ValueError('regime must be finetune or scratch')
    if (regime == 'finetune') != (pretrained_backbone is not None):
        raise ValueError('only finetune requires a pretrained backbone')
    model = SeActBackbone(variant, num_classes, height, width, dimension, seed, latent_width,
                         point_chunk_size=point_chunk_size)
    head_before = {k: v.detach().clone() for k, v in model.classifier.state_dict().items()}
    source_hash = None
    if regime == 'finetune':
        source_hash = file_sha256(pretrained_backbone)
        pretrained = load_mini_export(pretrained_backbone)
        if (pretrained.mode, pretrained.dimension, pretrained.latent_width) != (variant, dimension, latent_width):
            raise ValueError('pretrained variant/dimension/latent width mismatch')
        backbone = pretrained.state_dict()
        target_keys = {key for key in model.state_dict() if not key.startswith('classifier.')}
        if set(backbone) != target_keys:
            raise ValueError('pretrained backbone key mismatch')
        state = {**backbone, **{'classifier.' + key: value for key, value in head_before.items()}}
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(model.state_dict()[key], value) for key, value in backbone.items()):
            raise AssertionError('pretrained initialization mismatch')
    if any(not torch.equal(value, model.classifier.state_dict()[key]) for key, value in head_before.items()):
        raise AssertionError('new classifier must remain independently initialized')
    head_hash = hashlib.sha256(b''.join(value.cpu().numpy().tobytes() for value in head_before.values())).hexdigest()
    model.init_metadata = dict(regime=regime, pretrained_backbone_sha256=source_hash,
                               classifier_initial_sha256=head_hash, all_parameters_trainable=True,
                               spatial_policy='native coordinates; padded dense canvas; no event resampling',
                               point_chunk_size=point_chunk_size)
    return model


def export_backbone(model, path):
    from .seact_data import INPUT_CONTRACT_SHA256
    config = {**model.construction_config(), 'include_classifier': False}
    payload = dict(export_version=EXPORT_VERSION, input_contract_sha256=INPUT_CONTRACT_SHA256,
                   model_config=config, init_metadata=model.init_metadata,
                   state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                               if not key.startswith('classifier.')})
    with Path(path).open('xb') as handle:
        torch.save(payload, handle)
    return Path(path)


def load_backbone_export(path, *, map_location='cpu'):
    from .seact_data import INPUT_CONTRACT_SHA256
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (set(payload) != {'export_version', 'input_contract_sha256', 'model_config', 'init_metadata', 'state_dict'}
            or payload['export_version'] != EXPORT_VERSION
            or payload['input_contract_sha256'] != INPUT_CONTRACT_SHA256
            or set(payload['model_config']) != {'mode', 'num_classes', 'height', 'width', 'dimension',
                                               'seed', 'latent_width', 'point_chunk_size', 'include_classifier'}
            or payload['model_config'].get('include_classifier') is not False):
        raise ValueError('invalid SeACT backbone export or input contract')
    model = SeActBackbone(**payload['model_config'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.init_metadata = payload['init_metadata']
    return model.to(map_location).eval()
