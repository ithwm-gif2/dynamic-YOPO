# ConvGRU Dynamic YOPO experiment report

Date: 2026-07-30

## Implemented architecture

The ConvGRU version keeps the policy input free of obstacle state truth.

```text
four depth frames
  -> shared single-channel ResNet-18 encoder
  -> per-frame 3 x 5 feature maps
  -> ConvGRU temporal fusion
  -> future-risk / motion-risk auxiliary head
  -> YOPO trajectory and score head
```

The three relative camera-pose increments are encoded separately. Training uses future obstacle boxes only to construct loss targets; inference uses depth history, relative ego pose, current ego state, and the goal.

Additional training changes:

- 30% static, 30% mixed, and 40% fully dynamic sampling.
- Motion-critical window oversampling.
- Log-cost score regression plus safety-aware trajectory ranking.
- Training-only future-risk, motion-delta, and motion-presence targets.
- A frozen-depth counterfactual branch to discourage temporal shortcuts.
- Direct future-risk features in the YOPO trajectory score.

## Runtime

| Item | Result |
|---|---:|
| Parameters | 11,548,453 |
| PyTorch forward time on RTX 5070 Ti | 1.72 ms |
| Training throughput, batch 16 | about 601 samples/s |

ConvGRU is therefore computationally feasible for the 30-33 Hz sensing loop.

## Formal training

Run:

```text
/workspace/YOPO/YOPO/saved_temporal/temporal_dynamic_convgru_final_20260730
```

The run started from random initialization and stopped after epoch 3 because the safety-selection metric did not improve for three epochs. Epoch 0 was selected.

Uniform validation subset results for the selected checkpoint:

| Scenario | Collision rate | Oracle collision rate |
|---|---:|---:|
| Static | 0.0% | 0.0% |
| Mixed | 6.7% | 2.3% |
| Dynamic | 10.8% | 3.5% |

The previous non-ConvGRU temporal model achieved 0.0%, 6.3%, and 9.5% respectively on its full validation sets. ConvGRU did not improve offline safety.

## Full dynamic validation and temporal ablation

The complete dynamic validation set contains 19,550 samples.

| Input | Collision | Margin violation | Goal-backward endpoint | Mean goal progress |
|---|---:|---:|---:|---:|
| Normal depth history + relative pose | 10.70% | 44.50% | 24.60% | 0.83 m |
| Repeated last depth + relative pose | 10.80% | 44.60% | 24.60% | 0.83 m |
| Depth history + zero relative pose | 11.79% | 48.17% | 25.18% | 0.22 m |
| Repeated last depth + zero relative pose | 11.87% | 48.28% | 25.23% | 0.22 m |

The nearly identical normal and repeated-last-frame results show that the policy mainly uses relative ego pose rather than obstacle motion visible across depth frames. A second pilot trained the frozen-depth branch with real relative pose, but normal versus frozen collision remained 9.57% versus 9.77% on its validation subset. This did not solve the shortcut.

## Closed-loop comparison

A fair 30-second fully dynamic test was run by completing takeoff first, then resetting the dynamic simulator and starting the policy.

| Metric | ConvGRU result |
|---|---:|
| Goal distance during monitored interval | 43.82 m -> 3.12 m |
| Path length | 58.15 m |
| Mean / maximum speed | 1.94 / 3.78 m/s |
| Altitude range | 2.26-2.83 m |
| Collision-positive samples | 41 / 991 (4.1%) |
| First collision | 0.79 s |

ConvGRU produced substantially stronger goal progress than the earlier temporal checkpoint, but it remained unsafe. The earlier checkpoint had about 4.6% collision-positive samples in its corresponding 30-second test, so the closed-loop safety change is small.

## Conclusion

ConvGRU is fast enough and improves temporal feature capacity, but adding it alone does not produce reliable obstacle-motion estimation. Under the current four-frame, approximately 0.09-second observation window, the network finds a shortcut through ego pose and the last depth frame.

The next defensible iteration is:

1. Increase the temporal baseline to roughly 0.3-0.6 seconds, for example 6 frames sampled at 10-15 Hz, while keeping control output at 30-50 Hz.
2. Add an explicit image-space target such as ego-motion-compensated future depth, optical flow, scene flow, or future occupancy, instead of supervising motion only through trajectory risk.
3. Ensure the motion encoder cannot directly access ego pose; combine its image-derived motion feature with ego pose only after motion prediction.
4. Collect policy rollouts and perform DAgger-style retraining on states actually visited by the closed-loop controller.
5. Keep the non-ConvGRU checkpoint as the default until a new model improves both full offline collision rate and closed-loop collision rate.

Flow Matching remains deferred because the deterministic temporal representation has not yet passed the dynamic-safety gate.
