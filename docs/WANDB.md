# W&B monitoring for the existing V1 experiments

The sidecar is **online and verified** as of 2026-09-10 after explicit user approval.
Project: https://wandb.ai/jwoh3923-/ebackbone-v3-n-imagenet-mini
Uploaded epoch metrics were read back through the W&B API and matched against local histories.
Existing training processes, source snapshots, optimizer state, data access and checkpoints are unchanged.

## Concrete destination and upload scope

- Verified saved W&B login: username `jwoh3923`, default workspace `jwoh3923-`.
- Requested project: `ebackbone-v3-n-imagenet-mini`.
- Run group: `mini-v1-seed20260908`.
- Runs: hierarchy (without TS), hierarchy_ts (with TS gating), frame, same_arch, heterogeneous, frame_matched. Queued models
  are attached automatically when their production configuration appears.
- Backfill every completed epoch from the existing `history.json` files.
- Upload training and internal-validation CE, top-1/top-5 percentages, learning
  rate, sample counts, epoch duration, throughput and forward latency.
- Follow the latest logged optimizer progress every 30 seconds: batch loss,
  current epoch, sample progress and estimated remaining time.
- Upload architecture/training settings, renderer or point contract, dataset
  name, train/validation manifest hashes, source-code hashes, optimizer,
  scheduler, objective, augmentation policy and package versions.
- No raw events, class labels, sample identifiers, source code, credentials or
  checkpoint files are uploaded. Saved credentials are read by W&B for login;
  no key is printed or placed in the registry.

The dashboard displays epoch curves with epoch as the x-axis. Accuracy values
use percentages. Best and latest validation results are separate summary fields.
Only a complete configured epoch budget with a successful checkpoint-reload
report marks a training run finished. W&B automatic system metrics are disabled
because the logging process is not the training process.

## Implementation and installation

- `ebackbone_v3/wandb_monitor.py`: standalone read-only artifact follower; no
  PyTorch import or modification of existing trainers.
- `tests/test_wandb_monitor.py`: metric units, data scope, incomplete log writes,
  model filtering, completion guards and configuration allowlist.
- `pyproject.toml`: optional `tracking` dependency (`wandb==0.30.0`).

For this running experiment, W&B is installed into an isolated environment:
`/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/wandb_env`.
The active training environments have not been changed. Deployment files are in
`/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/wandb_sync`:
`registry.json`, the frozen `wandb_monitor.py`, and `deployment.json`.

The running uploader uses the following command. Do not start a duplicate; inspect `wandb_sync/launch.json` and `state/status.json` first:

```bash
WANDB_CONFIG_DIR=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/wandb_sync/state/config \
WANDB_CACHE_DIR=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/wandb_sync/state/cache \
WANDB_DATA_DIR=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/wandb_sync/state/data \
  /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/wandb_env/bin/python -u \
  /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/wandb_sync/wandb_monitor.py \
  --registry /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/wandb_sync/registry.json
```

Run IDs are stable hashes of the local production-run directory. The SDK resumes
existing runs with `resume="allow"`; the remote completed-epoch summary is the
restart cursor. A process lock prevents two sidecars from publishing the same
registry simultaneously. Network/SDK errors appear in `state/status.json` and
are retried without affecting training.

SDK behavior references: [run resume](https://docs.wandb.ai/models/runs/resuming)
and [settings](https://docs.wandb.ai/models/ref/python/experiments/settings).

Verification: `python -m pytest -q tests/test_wandb_monitor.py` passed all four
tests. Online run creation and epoch-history upload were verified by remote API readback: hierarchy 100 epochs, TS gating 14 epochs, frame 100 epochs, same_arch 81 epochs at initial verification. Live runs continue syncing every 30 seconds; remaining baselines attach when their runs begin. Evidence: `wandb_sync/online_verification.json`.
