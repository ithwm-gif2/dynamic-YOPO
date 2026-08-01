# Geometry-aware static/dynamic YOPO experiment report

Date: 2026-07-31

## Final inference design

The policy does not receive obstacle positions, velocities, identities, or simulator truth at inference time.

```text
depth frames [t-6, t-3, t] + ego poses
  -> geometric ego-motion depth alignment
  -> signed/absolute residual, validity, occlusion, disocclusion
  -> shared ResNet-18 + residual ConvGRU
  -> current dynamic occupancy and coarse obstacle velocity

current depth -> static ResNet-18

static feature + motion feature + predicted velocity + ego state
  -> future dynamic occupancy
  -> YOPO endpoint and safety score heads
```

The simulator obstacle state is used only to build training labels and losses.

## Data

- Static: 5 episodes x 4,000 frames.
- Mixed: 5 episodes x 4,000 frames.
- Fully dynamic: 10 episodes x 4,000 frames.
- Training sampling: 30% static, 30% mixed, 40% fully dynamic.
- Obstacle density: one obstacle per 30 square metres.
- Dynamic speed: 0.5-3.0 m/s.
- Input window: frames `[t-6, t-3, t]`, approximately 0.18 s at 33 Hz.

## Geometry residual validation

After ego-motion compensation, the fraction of pixels with residual greater than 0.1 m was approximately:

| Scenario | Fraction |
|---|---:|
| Static | 0.59% |
| Mixed | 36.0% |
| Fully dynamic | 59.9% |

This shows that explicit geometry removes most apparent motion caused by the flying camera.

## Motion pretraining

The best velocity-auxiliary motion checkpoint is:

```text
saved_static_dynamic/geometry_motion_velocity_aux_full_20260731/best.pth
```

| Scenario | Dynamic-mask IoU | Static FPR | Velocity MAE |
|---|---:|---:|---:|
| Static | n/a | 0.1% | n/a |
| Mixed | 0.862 | - | 0.492 m/s |
| Fully dynamic | 0.965 | - | 0.494 m/s |

Velocity labels are simulator-only auxiliary supervision. The velocity prediction is generated from the depth residual sequence at inference.

## Planner experiments

| Variant | Static collision | Mixed collision | Dynamic collision | Result |
|---|---:|---:|---:|---|
| Basic geometry planner | 0.1% | 5.2% | 12.2% | Ranking bottleneck |
| Safety-distribution and pairwise ranking | 0.0% | 3.9% | 10.8% | Clear improvement |
| Extra candidate-risk head | 0.0% | 4.3% | 11.1% | Rejected |
| Velocity-supervised hidden feature | 0.0% | 4.9% | 9.9% | Improved dynamic result |
| Explicit predicted-velocity fusion, selected checkpoint | 0.0% | 4.5% | 10.2% | Used for closed loop |

The explicit-velocity run reached 9.9% dynamic collision in one epoch with 4.6% mixed collision, while its automatic weighted-selection checkpoint obtained 10.2% dynamic and 4.5% mixed collision. The previous non-ConvGRU temporal baseline was 0.0%, 6.3%, and 9.5%, so the new model did not establish a clear offline dynamic-safety win.

Future-occupancy diagnosis on the geometry planner showed that current motion segmentation was strong, but future occupancy remained the main perception bottleneck: mixed IoU about 0.38 and fully dynamic IoU about 0.50 for the earlier best safety-ranking checkpoint.

## Runtime

Measured with PyTorch on the RTX 5070 Ti:

| Item | Result |
|---|---:|
| Parameters | 22,778,673 |
| Mean inference | 2.77 ms |
| P95 inference | 2.82 ms |
| Maximum in 500 trials | 3.68 ms |

This is fast enough for the 30-33 Hz depth loop.

## ROS and RViz integration

The new node is:

```text
test_static_dynamic_yopo_ros.py
```

It maintains a seven-frame buffer, selects `[t-6, t-3, t]`, constructs the two relative camera transforms, publishes the original YOPO trajectory visualizations, and sends the original `PositionCommand` control messages.

## 30-second fully dynamic closed loop

The fair run completed takeoff first, preloaded the policy without depth, then started a freshly reset fully dynamic sensor simulator. Monitoring began on the first `/dynamic_collision` message. The policy subscribed only to depth and odometry.

| Metric | Result |
|---|---:|
| Goal distance | 50.00 m -> 2.81 m |
| Path length | 52.82 m |
| Mean / maximum speed | 1.76 / 4.76 m/s |
| Altitude range | 2.01-2.85 m |
| Collision-positive samples | 0 / 991 |
| First collision | none |

This is substantially better than the earlier raw ConvGRU closed loop, which had 41 / 991 collision-positive samples. It is still one simulator seed and must be repeated across multiple obstacle seeds before claiming robust dynamic avoidance.

## Conclusion

The central finding is that adding more raw frames is not enough. Explicit ego-motion compensation, dynamic segmentation, and velocity auxiliary supervision make the temporal feature represent obstacle motion rather than camera motion. The final model is computationally feasible and achieved a collision-free single-seed closed loop, but its offline dynamic collision rate did not clearly beat the earlier 9.5% baseline.

The next experiment should therefore repeat closed-loop evaluation across multiple online obstacle seeds and then collect policy rollouts for DAgger-style retraining. Flow Matching remains deferred until this deterministic temporal model passes a multi-seed safety gate.
