"""Exact optimizer-boundary resume and explicit downstream transfer artifacts.

Only caller-selected output directories are written. All ranks call save_epoch;
only rank zero writes. Full training checkpoints retain every model tensor and
rank RNG state. Transfer bundles separate reusable event features from the
source task's classifier/fixed text bank. No dataset or GPU work is launched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import distributed as dist


CHECKPOINT_SCHEMA = "hierarchy-clip-training-v1"
TRANSFER_SCHEMA = "hierarchy-clip-transfer-v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap(model):
    return model.module if isinstance(model, (torch.nn.parallel.DistributedDataParallel,
                                             torch.nn.DataParallel)) else model


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


def _atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng_state():
    """Capture this rank only; never initialize CUDA merely to inspect RNG."""
    cuda = None
    if torch.cuda.is_initialized():
        device = torch.cuda.current_device()
        cuda = {"device": device, "state": torch.cuda.get_rng_state(device).cpu()}
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": cuda}


def restore_rng_state(state):
    required = {"python", "numpy", "torch", "cuda"}
    if set(state) != required:
        raise ValueError("incomplete Python/numpy/torch/CUDA RNG state")
    cuda = state["cuda"]
    if cuda is not None:
        if not torch.cuda.is_available() or torch.cuda.current_device() != cuda["device"]:
            raise ValueError("exact resume requires the saved rank-local CUDA device")
        torch.cuda.set_rng_state(cuda["state"], cuda["device"])
    elif torch.cuda.is_initialized():
        raise ValueError("CPU checkpoint RNG cannot provide exact CUDA resume")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())


def gather_rng_states():
    state = capture_rng_state()
    if dist.is_available() and dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, state)
        return states
    return [state]


def _optimizer_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    grouped = []
    seen = set()
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            if id(parameter) not in names or id(parameter) in seen or not parameter.requires_grad:
                raise ValueError("optimizer must contain unique trainable model parameters only")
            seen.add(id(parameter))
            group_names.append(names[id(parameter)])
        grouped.append(group_names)
    if seen != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise ValueError("optimizer does not cover the complete configured training stage")
    return grouped


def _validate_sampler(sampler, world_size):
    if not {"seed", "epoch", "offset", "world_size"} <= set(sampler):
        raise ValueError("sampler state requires seed, next epoch, consumed-sample offset, world_size")
    for key in ("seed", "epoch", "offset", "world_size"):
        if isinstance(sampler[key], bool) or not isinstance(sampler[key], int):
            raise ValueError(f"sampler {key} must be an integer")
    if sampler["epoch"] < 1 or sampler["offset"] < 0 or sampler["world_size"] != world_size:
        raise ValueError("invalid sampler epoch/offset/world_size")


def _validate_identity(identity):
    _canonical(identity)
    world = identity.get("world_size")
    if isinstance(world, bool) or not isinstance(world, int) or world < 1:
        raise ValueError("identity requires a positive world_size")
    return world


def _claim_output(output, identity):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "checkpoint_identity.json"
    if marker.is_file():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("checkpoint output directory belongs to another identity")
    else:
        if any((output / name).exists() for name in
               ("checkpoint_best.pt", "checkpoint_last.pt", "transfer_best.pt", "transfer_last.pt")):
            raise FileExistsError("refusing to overwrite checkpoint artifacts without this run's identity marker")
        # The writer is rank zero only. Exclusive creation prevents silently
        # claiming another writer's directory even before its first checkpoint.
        with marker.open("x") as stream:
            stream.write(_canonical(identity) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return output


def save_training_checkpoint(output_dir, model, optimizer, scheduler, *, identity,
                             epoch, global_step, stage, sampler_state, rng_by_rank,
                             history, best, is_best=False, extra=None):
    """Rank-zero save at an optimizer-step boundary, never mid accumulation.

    Epoch means completed epoch for normal training. ``sampler_state`` describes
    the next data item (next epoch and offset zero at an epoch boundary). A caller
    saving mid epoch must provide the consumed local-sample offset and restore
    the sampler exactly; prefetched batches are not represented by this offset.
    The caller's identity must contain its complete config, source/checkpoint and
    manifest hashes, ordered class names/prompts, and world size.
    """
    model = _unwrap(model)
    world = _validate_identity(identity)
    if len(rng_by_rank) != world:
        raise ValueError("RNG rank count differs from identity world_size")
    _validate_sampler(sampler_state, world)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step must be nonnegative integers")
    if stage != getattr(model, "_fine_tuning_stage", None):
        raise ValueError("model must be configured for the saved fine-tuning stage")
    scheduler_state = scheduler.state_dict() if hasattr(scheduler, "state_dict") else scheduler
    if not isinstance(scheduler_state, dict):
        raise ValueError("scheduler must expose state_dict or be an absolute-schedule dictionary")
    if "completed_epoch" in scheduler_state and scheduler_state["completed_epoch"] != epoch:
        raise ValueError("scheduler completed_epoch must equal checkpoint epoch")
    state = {"schema": CHECKPOINT_SCHEMA, "identity": copy.deepcopy(identity),
             "identity_sha256": _digest(identity), "epoch": epoch, "global_step": global_step,
             "stage": stage, "sampler": copy.deepcopy(sampler_state),
             "boundary": "optimizer_step", "pending_accumulation_steps": 0,
             "model": _cpu(model.state_dict()), "optimizer": _cpu(optimizer.state_dict()),
             "optimizer_parameter_names": _optimizer_names(model, optimizer),
             "optimizer_group_lrs": [group["lr"] for group in optimizer.param_groups],
             "scheduler": copy.deepcopy(scheduler_state),
             "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad],
             "rng_by_rank": _cpu(rng_by_rank), "history": copy.deepcopy(history),
             "best": copy.deepcopy(best), "extra": copy.deepcopy(extra or {})}
    if any(key.startswith("module.") for key in state["model"]):
        raise ValueError("full checkpoints must contain unwrapped model keys")
    output = _claim_output(output_dir, identity)
    results = {}
    # Write improved best first: a newly advanced last always has its best file.
    for kind in (["best", "last"] if is_best else ["last"]):
        path = output / f"checkpoint_{kind}.pt"
        _atomic_save(path, state)
        results[kind] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return results


def inspect_training_checkpoint(path, identity=None):
    """Read before rebuilding the saved stage/optimizer; accepts loaded state too."""
    state = path if isinstance(path, dict) else torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported training checkpoint schema")
    saved_identity = state["identity"]
    world = _validate_identity(saved_identity)
    if state.get("identity_sha256") != _digest(saved_identity):
        raise ValueError("training checkpoint identity digest mismatch")
    if identity is not None and _canonical(identity) != _canonical(saved_identity):
        raise ValueError("training checkpoint identity mismatch (config/provenance/class order/world size)")
    if len(state["rng_by_rank"]) != world:
        raise ValueError("checkpoint RNG world size mismatch")
    _validate_sampler(state["sampler"], world)
    if state.get("boundary") != "optimizer_step" or state.get("pending_accumulation_steps") != 0:
        raise ValueError("resume of pending gradient accumulation is unsupported")
    if any(key.startswith("module.") for key in state["model"]):
        raise ValueError("checkpoint model keys must be unwrapped")
    return state


def _validate_tensors(saved, expected, label):
    if set(saved) != set(expected):
        raise ValueError(f"{label} keys differ: missing={sorted(set(expected)-set(saved))}, "
                         f"unexpected={sorted(set(saved)-set(expected))}")
    for key, tensor in saved.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
            raise ValueError(f"{label} tensor shape/dtype mismatch: {key}")


def load_training_checkpoint(path, model, optimizer, scheduler=None, *, identity,
                             rank=0, restore_rng=True):
    """Restore only after the trainer configures the checkpoint's stage and DDP."""
    state = inspect_training_checkpoint(path, identity)
    model = _unwrap(model)
    if state["stage"] != getattr(model, "_fine_tuning_stage", None):
        raise ValueError("configure checkpoint stage before restoring; transition afterward")
    if state["trainable_names"] != [name for name, p in model.named_parameters() if p.requires_grad]:
        raise ValueError("checkpoint trainable parameter policy mismatch")
    if state["optimizer_parameter_names"] != _optimizer_names(model, optimizer):
        raise ValueError("optimizer parameter order differs from checkpoint")
    if not 0 <= rank < len(state["rng_by_rank"]):
        raise ValueError("rank outside checkpoint world size")
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != identity["world_size"]:
        raise ValueError("live distributed world size differs from checkpoint identity")
    _validate_tensors(state["model"], model.state_dict(), "checkpoint model")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if [group["lr"] for group in optimizer.param_groups] != state["optimizer_group_lrs"]:
        raise ValueError("optimizer learning rates differ after resume")
    if scheduler is not None:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(state["scheduler"])
        elif isinstance(scheduler, dict):
            scheduler.clear()
            scheduler.update(copy.deepcopy(state["scheduler"]))
        else:
            raise ValueError("scheduler must accept saved state")
    if restore_rng:
        # Restore last, after object construction/loading could consume RNG.
        restore_rng_state(state["rng_by_rank"][rank])
    return state


def load_checkpoint(path, model, optimizer, *, identity, device=None, rank=None):
    del device  # Tensor optimizer states migrate to parameter devices on load.
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    return load_training_checkpoint(path, model, optimizer, identity=identity, rank=rank)


def _transfer_parts(model):
    model = _unwrap(model)
    state = model.state_dict()
    prefix = "encoder.hierarchy."
    backbone = {key[len(prefix):]: value for key, value in state.items()
                if key.startswith(prefix) and not key.startswith(prefix + "classifier.")}
    if not backbone:
        raise ValueError("transfer requires encoder.hierarchy backbone state")
    components = {}
    task = {}
    covered = {key for key in state if key.startswith(prefix)}
    for name in ("temporal", "adapter", "projection", "visual", "classifier", "text_bank"):
        module = getattr(model, name, None)
        if module is None:
            continue
        values = {key[len(name)+1:]: value for key, value in state.items() if key.startswith(name + ".")}
        (task if name in {"classifier", "text_bank"} else components)[name] = values
        covered.update(name + "." + key for key in values)
    if set(state) != covered:
        raise ValueError(f"unclassified transfer tensors: {sorted(set(state)-covered)}")
    text = getattr(model, "text_bank", None)
    metadata = text.contract() if text is not None and hasattr(text, "contract") else {}
    return backbone, components, task, metadata


def export_transfer_bundle(path, model, *, identity, epoch, stage, checkpoint_sha256=None):
    """Export learned event features and source task state as separate components.

    This bundle has no optimizer/RNG and is not a training-resume checkpoint.
    Object prompts/text bank remain attached as source-task metadata, never
    assumed appropriate for a future action-recognition dataset.
    """
    model = _unwrap(model)
    _validate_identity(identity)
    backbone, components, task, task_metadata = _transfer_parts(model)
    bundle = {"schema": TRANSFER_SCHEMA, "identity": copy.deepcopy(identity),
              "identity_sha256": _digest(identity), "epoch": epoch, "stage": stage,
              "checkpoint_sha256": checkpoint_sha256,
              "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
              "variant": getattr(model, "name", identity.get("variant", "hierarchy_temporal")),
              "backbone": _cpu(backbone), "components": _cpu(components),
              "source_task": _cpu(task), "source_task_metadata": copy.deepcopy(task_metadata),
              "excluded_unused_state": [key for key in model.state_dict()
                                        if key.startswith("encoder.hierarchy.classifier.")],
              "contract": {"hierarchy_classifier_transferred": False,
                           "event_input_only": True, "rgb_input_required": False,
                           "text_bank_scope": "source dataset fixed object prompts",
                           "new_class_transfer_requires_explicit_head_exclusion": True},
              "component_classes": {name: f"{type(getattr(model, name)).__module__}.{type(getattr(model, name)).__qualname__}"
                                    for name in components},
              "tensor_shapes": {"backbone": {key: list(value.shape) for key, value in backbone.items()},
                                **{name: {key: list(value.shape) for key, value in values.items()}
                                   for name, values in {**components, **task}.items()}}}
    path = Path(path)
    if path.is_file():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if existing.get("schema") != TRANSFER_SCHEMA or existing.get("identity_sha256") != _digest(identity):
            raise FileExistsError("refusing to overwrite transfer artifact from a different run")
        del existing
    _atomic_save(path, bundle)
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "backbone_tensors": len(backbone), "components": list(components),
            "source_task_components": list(task)}


def load_transfer_bundle(path, model, *, expected_identity=None, include_task_head=True,
                         allow_new_classes=False):
    """Strict feature transfer; explicit new-class mode retains target task state.

    ``expected_identity`` identifies the source artifact, not the target dataset.
    The target model must construct the same feature architecture. Its new class
    names/text bank or linear head are retained only with both exclusion flags.
    """
    if not include_task_head and not allow_new_classes:
        raise ValueError("excluding source task state requires explicit allow_new_classes=True")
    if include_task_head and allow_new_classes:
        raise ValueError("new-class transfer must exclude source classifier/text bank")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema") != TRANSFER_SCHEMA or bundle.get("identity_sha256") != _digest(bundle["identity"]):
        raise ValueError("invalid transfer schema/identity digest")
    if expected_identity is not None and _canonical(expected_identity) != _canonical(bundle["identity"]):
        raise ValueError("transfer source identity mismatch")
    model = _unwrap(model)
    backbone, components, task, task_metadata = _transfer_parts(model)
    _validate_tensors(bundle["backbone"], backbone, "transfer backbone")
    if set(bundle["components"]) != set(components):
        raise ValueError("transfer component architecture mismatch")
    merged = dict(model.state_dict())
    merged.update({"encoder.hierarchy." + key: value for key, value in bundle["backbone"].items()})
    for name, values in bundle["components"].items():
        _validate_tensors(values, components[name], f"transfer {name}")
        merged.update({name + "." + key: value for key, value in values.items()})
    if include_task_head:
        if set(bundle["source_task"]) != set(task) or bundle["source_task_metadata"] != task_metadata:
            raise ValueError("source task class order/prompts or task architecture mismatch")
        for name, values in bundle["source_task"].items():
            _validate_tensors(values, task[name], f"transfer source task {name}")
            merged.update({name + "." + key: value for key, value in values.items()})
    model.load_state_dict(merged, strict=True)
    return {"strict_features": True, "source_task_loaded": include_task_head,
            "new_classes": allow_new_classes, "identity": bundle["identity"],
            "epoch": bundle["epoch"], "checkpoint_sha256": bundle["checkpoint_sha256"]}


def save_epoch(output, model, optimizer, *, epoch, stage, identity, history, best,
               improved, device=None, scheduler_state, global_step=0,
               sampler_state=None, extra=None):
    """Collective wrapper used by the full trainer; writes only on rank zero."""
    del device
    rng = gather_rng_states()
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    if sampler_state is None:
        raise ValueError("save_epoch requires the explicit deterministic sampler state")
    result = None
    if rank == 0:
        try:
            checkpoints = save_training_checkpoint(output, model, optimizer, scheduler_state,
                identity=identity, epoch=epoch, global_step=global_step, stage=stage,
                sampler_state=sampler_state, rng_by_rank=rng, history=history, best=best,
                is_best=improved, extra=extra)
            exports = {kind: export_transfer_bundle(Path(output) / f"transfer_{kind}.pt", model,
                         identity=identity, epoch=epoch, stage=stage,
                         checkpoint_sha256=metadata["sha256"])
                       for kind, metadata in checkpoints.items()}
            result = {"checkpoints": checkpoints, "exports": exports}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        results = [result]
        dist.broadcast_object_list(results, src=0)
        result = results[0]
    if "error" in result:
        raise RuntimeError("checkpoint/export save failed: " + result["error"])
    return result
