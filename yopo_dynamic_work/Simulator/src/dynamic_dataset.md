# Continuous dynamic YOPO dataset

The `dynamic_dataset_generator` executable creates synchronized depth sequences for
training a temporal YOPO planner with a time-indexed dynamic collision loss.

## Scene definition

- The horizontal map size comes from `x_length` and `y_length`.
- Every box obstacle moves in the XY plane; there are no static box obstacles.
- The ground plane remains static so that the depth images retain realistic lower-image geometry.
- Obstacle count is computed as
  `round(x_length * y_length / area_per_obstacle)`.
- With the default 60 m by 60 m map and 30 square metres per obstacle, every frame
  contains 120 moving obstacles.
- Obstacle speeds remain within 0.5 to 3.0 m/s. Obstacles reflect at map boundaries
  and at a configurable 5 cm drone-contact margin without losing speed. A sampled
  position can repeat at the exact instant of a symmetric reflection; velocity stays nonzero.

## Build

```bash
cd /workspace/YOPO/Simulator
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release
```

## Generate data

The default configuration is in `config/config.yaml` under `dynamic_dataset`.

```bash
cd /workspace/YOPO
./Simulator/devel/lib/sensor_simulator/dynamic_dataset_generator
```

Small inspection dataset:

```bash
./Simulator/devel/lib/sensor_simulator/dynamic_dataset_generator \
  --episodes 1 --frames 180 --seed 123 --output dataset_dynamic_sample
```

Command-line overrides:

- `--episodes N`
- `--frames N`
- `--seed N`
- `--output PATH`

The selected output directory is removed before generation.

## Episode layout

```text
dataset_dynamic/
  manifest.csv
  episode_0000/
    metadata.yaml
    drone_state.csv
    relative_pose.csv
    obstacles.csv
    depth/
      img_000000.png
      img_000001.png
      ...
```

### Depth images

Depth is rendered at the camera resolution in `config.yaml`, clipped by
`max_depth_dist`, normalized to [0, 1], and saved as a 16-bit single-channel PNG.

### drone_state.csv

Contains synchronized body/camera poses, velocity, acceleration, goal, collision
flag, minimum obstacle clearance, and a `valid` training-frame flag.

A frame is valid only when:

- the requested four-frame history is available; and
- obstacle ground truth is available for the complete future planning horizon.

### relative_pose.csv

Stores the transform from the previous camera frame to the current camera frame:

`T_camera_previous_camera_current`.

Translation is expressed in the previous camera frame. The quaternion uses
`qw,qx,qy,qz` order.

### obstacles.csv

Contains one row per obstacle per frame:

- obstacle ID and timestamp;
- world-frame centre and velocity;
- box width, depth, and height.

These synchronized future rows are the ground truth for the later time-indexed
dynamic collision loss.
