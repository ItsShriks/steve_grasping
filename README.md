# steve_grasping

What is mounted on Steve's UR5e flange, and how the whole-body controller (`steve_wbc`) grasps with it.

| Tool (`--tool`, launch `tool:=`) | Model | Gripper | Grasp | Status |
|---|---|---|---|---|
| `camera_only` | `models/camera_only/d415_on_flange.xml` — D415 on a short post (calibrated mount, `steve_wbc/config/gripper_camera.yaml`) | none: go to the grasp pose only | top-down, hover `grasp_dz` 5 cm above the object centre | **real robot now** (default with `use_sim_time:=false`) |
| `robotiq_2f85` | `models/robotiq_2f85/robotiq_2f85.xml` (MuJoCo Menagerie) | 2-finger, `GripperCommand` | top-down, pads at the equator (`grasp_dz` −1.5 cm) | simulation default; identical to the model `steve_wbc` used before |
| `dex5` | `models/dex5_1/dex5_{right,left}.xml` generated from `description/dex5_1` | Unitree Dex5-1, 16 actuated + 4 coupled joints | side grasp (horizontal approach along the palm normal) | **provisional**: model and mount in place, grasp synergy not validated |

All three put the WBC's end effector — the site `pinch` (z = approach, y = closing axis) — at the same
place for `camera_only` and `robotiq_2f85`: 14.9 cm beyond the flange on its axis. Figure-8s and grasp
poses are therefore unchanged between the two.

## How a tool is attached

`steve_wbc/models/mmo_700_wbc.xml` is the robot without a tool; it ends in the empty body `hand_mount` at
the flange (UR `tool0`: z out of the flange — checked equal to the real URDF's `ur5etool0` with the arm
mounted at +90°). `steve_grasping.robot_model.compose(robot_xml, tool)` puts the tool's MJCF there (XML
level, any MuJoCo version), merges its assets / actuators / equalities / contact exclusions, adds `pinch`
from the tool config, and — for a tool with its own camera (`camera_only`) — removes the robot model's
D405 wrist camera. `steve_wbc.steve_figure8_wbc.load_model(..., tool=...)` does this for every script.

```python
from steve_grasping.tools import load_tool
from steve_grasping.robot_model import load_robot_model
model = load_robot_model("steve_wbc/models/mmo_700_wbc.xml", "camera_only")   # mujoco.MjModel
```

## Tool configs (`config/tools/<tool>.yaml`)

| Key | Meaning |
|---|---|
| `model` | MJCF of the tool (root body = tool base) |
| `gripper` | `false`: nothing to close — active perception runs `--goto_grasp_only` |
| `own_camera` | the tool carries `wrist_cam` (the robot model's D405 is removed) |
| `grasp_style` | `topdown` \| `side` |
| `grasp_dz` | default `--grasp_dz`: grasp point relative to the object centre [m] |
| `mount` | `tool0` → tool base: `xyz` + `rpy_deg`, or for a hand `flange_in_hand` + `adapter_length` + `hand_axes_in_tool` + `yaw_deg` |
| `pinch` | grasp point in the tool base frame: `xyz` + `rpy_deg` |
| `sim` / `real` | gripper commands (MuJoCo actuator / `GripperCommand` positions), Dex5 synergies |

After a new eye-in-hand calibration of the D415 (`ros2 run steve_wbc gripper_camera_calibrator`), copy
its `xyz` / `quat` into the `gripper_camera_color_optical_frame` body of `d415_on_flange.xml` (only the
simulation uses it; on the robot the camera pose comes from TF).

## Dex5-1

`description/dex5_1/Dex5-URDF-{R,L}` is the URDF + meshes from
[unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/dexterous_hand_description/dex5_1)
(BSD-3). `python3 tools/build_dex5_mjcf.py` regenerates the MuJoCo models: collision as primitives (palm
boxes, one capsule per phalanx), 16 position actuators (0.93 N m), the distal joints coupled to the middle
ones. Hand frame: fingers +y, thumb +x, palm normal +z (left hand: −z). Open points: the side-grasp
synergy (thumb `Pitch_13` closes towards the fingers, `Pitch_14` away from them), the real adapter
geometry, and the hand's command interface (USB, 1 kHz; Unitree SDK topic names to be confirmed).
