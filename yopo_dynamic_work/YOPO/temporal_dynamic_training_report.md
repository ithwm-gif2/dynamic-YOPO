# Temporal Dynamic YOPO: first formal training report

Date: 2026-07-30

## Scope

- Inference input: four consecutive depth frames, three relative camera-pose increments, and the UAV ego state/goal.
- No obstacle position, velocity, point cloud, or simulator obstacle truth is subscribed by the policy node.
- Training mixture: static 30%, mixed 30%, fully dynamic 40%.
- Dynamic obstacle speed: 0.5-3.0 m/s; density: one obstacle per 30 m^2.
- Network initialization: random; no `epoch50.pth` initialization.
- Scene geometry: axis-aligned box pillars.

## Dataset used by the formal run

| Scenario | Train samples | Validation samples | Dataset |
|---|---:|---:|---|
| Static | 39,760 | 9,940 | `dataset_static` |
| Mixed | 39,545 | 9,869 | `dataset_mixed_independent` |
| Dynamic | 77,987 | 19,550 | `dataset_dynamic_independent` |

The mixed and dynamic datasets use independent obstacle motion. Obstacles do not reflect away from the drone.

## Training

Command:

```bash
cd /workspace/YOPO/YOPO
python3 train_temporal_yopo.py \
  --epochs 10 \
  --batch-size 16 \
  --workers 4 \
  --learning-rate 1.5e-4 \
  --early-stopping-patience 3 \
  --run-name temporal_dynamic_independent_low_alt_20260730
```

The safety-selection metric chose epoch 0. Training stopped after epoch 3 because the safety score did not improve for three consecutive epochs.

Best checkpoint:

```text
/workspace/YOPO/YOPO/saved_temporal/temporal_dynamic_independent_low_alt_20260730/best.pth
```

Validation collision rates for the best checkpoint:

| Scenario | Selected collision rate | Oracle collision rate |
|---|---:|---:|
| Static | 0.0% | 0.0% |
| Mixed | 6.3% | 3.4% |
| Dynamic | 9.5% | 4.6% |

Later epochs reduced the training loss but did not improve the dynamic safety-selection metric.

## ROS closed-loop test

The fair test order is important: finish takeoff first, then start/reset the dynamic sensor simulator, then start Temporal YOPO. Starting the moving obstacles before takeoff can cause a collision before the policy is active.

During a valid 30-second fully dynamic run:

| Metric | Result |
|---|---:|
| Goal distance | 50.32 m -> 48.31 m |
| Altitude range | 2.42-2.69 m |
| Path length | 13.90 m |
| Mean / maximum speed | 0.46 / 2.91 m/s |
| Collision-positive samples | 46 / 991 (4.6%) |
| First collision | 9.03 s |

The altitude constraint fixed the earlier upward-escape behavior, but the current checkpoint is not collision-free and must not be treated as a successful dynamic-avoidance model.

ROS rates measured during the run:

- Depth: 33 Hz
- Position command: 50 Hz
- Collision monitor: 33 Hz

PyTorch inference benchmark on RTX 5070 Ti:

- Parameters: 11,315,434
- Mean forward time: 0.98 ms

## Temporal ablation

The dynamic validation subset used 1,024 samples.

| Input variant | Collision rate | Margin violation | Backward endpoint rate | Mean progress along goal |
|---|---:|---:|---:|---:|
| Normal four frames + ego pose | 7.42% | 45.80% | 25.10% | 1.14 m |
| Four copies of the last frame | 7.52% | 45.90% | 25.68% | 0.99 m |
| Four frames, zero relative pose | 6.84% | 49.61% | 26.17% | 0.51 m |
| Last frame only + zero relative pose | 7.32% | 47.36% | 26.27% | 0.42 m |

Replacing the temporal observations with a single-frame equivalent barely changes collision rate. The temporal inputs affect goal progress, but the safety decision mostly uses a single-frame shortcut.

## Required next iteration

The next training iteration should not add obstacle-state inputs at inference. It should instead:

1. Oversample temporal-critical windows, especially pre-collision and crossing events.
2. Add safety-aware ranking supervision so the score head selects a safe candidate when one exists; the selected/oracle collision gap is currently large.
3. Add a training-only motion/future-occupancy auxiliary target so the temporal encoder cannot ignore inter-frame motion.
4. Collect closed-loop/DAGGER-style rollouts from model-visited states to reduce the open-loop-to-closed-loop distribution gap.
5. Keep the static/mixed/dynamic 30/30/40 mixture and retain static-scene regression tests.

Flow Matching should remain deferred until the temporal representation and closed-loop safety are demonstrably effective.
