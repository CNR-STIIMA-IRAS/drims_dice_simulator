# drims_dice_simulator

A ROS 2 package for spawning a dice inside a robotic cell. The face up can be chosen or selected at random. The node also publishes a `tf` frame centered on the currently face-up side.

The dice pips are rendered in RViz as a `MarkerArray` of cylinders relative to the dice face frames. Coordinate transforms are broadcast on `/tf` at 200Hz.

## Launch

```bash
ros2 launch drims_dice_simulator spawn_dice.launch.py
```

## Parameters

All parameters default to values defined in [dice_spawner_parameters.yaml](config/dice_spawner_parameters.yaml) and can be overridden at launch time:

- **`face_up`** *(int, 0–6, default: `0`)*  
  Face that should point upward. Use `1`–`6` to set the face up; `0` (or unset) picks a random face.

- **`dice_size`** *(double, default: `0.027`)*  
  Edge length of the dice (in meters).

- **`position`** *(double_array[3], default: `[0.6, 0.2, 0.0]`)*  
  XYZ spawn position in the `base_link` frame.

- **`random_position`** *(bool, default: `false`)*  
  Whether to spawn at a random position within bounds.

- **`x_min`** / **`x_max`** *(double, default: `0.40` / `0.80`)*  
  X-coordinate boundaries for random spawning.

- **`y_min`** / **`y_max`** *(double, default: `-0.20` / `0.40`)*  
  Y-coordinate boundaries for random spawning.

- **`surface_height`** *(double, default: `0.0`)*  
  Surface height (in meters).

- **`pips_distance`** *(double, default: `0.26`)*  
  Distance of pips from the center of the face (as a fraction of dice size).

- **`pip_diameter`** *(double, default: `0.21`)*  
  Diameter of the pips (as a fraction of dice size).

## Examples

Random face, default size and position:
```bash
ros2 launch drims_dice_simulator spawn_dice.launch.py
```

Fixed face (e.g., “6”), custom pip spacing and dice size:
```bash
ros2 launch drims_dice_simulator spawn_dice.launch.py face_up:=6 dice_size:=0.05 pips_distance:=0.22
```

## Coordinate Frames (TF)

The node publishes the following frames continuously on `/tf` at 200Hz:

- **`dice_com_tf`**  
  Located at the center of mass of the dice.
- **`face{1..6}_tf`**  
  Located at the center of each of the six faces on the surface of the dice.
- **`dice_tf`**  
  Aligned with the face currently pointing upward. This frame can be used for grasp planning (e.g., positioning the robot gripper over the dice as a pre-grasp pose).

## Services

- **`/dice_identification`** (`easy_motion_msgs/srv/DiceIdentification`)  
  Triggers face identification to determine which side is facing up, updates `dice_tf`, and returns the pose.
  
  *Command to call:*
  ```bash
  ros2 service call /dice_identification easy_motion_msgs/srv/DiceIdentification "{}"
  ```

- **`/reset_dice`** (`std_srvs/srv/Trigger`)  
  Resets the simulation. It detaches the dice from the gripper, removes it from the planning scene, resolves a new spawn position, spawns the monolithic body, and updates `dice_tf`.
  
  *Command to call:*
  ```bash
  ros2 service call /reset_dice std_srvs/srv/Trigger "{}"
  ```
