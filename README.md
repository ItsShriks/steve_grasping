# steve_grasping

What is mounted on Steve's UR5e flange, and how the whole-body controller (`steve_wbc`) grasps with it.

| Tool (`--tool`, launch `tool:=`) | Model | Gripper | Grasp | Status |
|---|---|---|---|---|
| `camera_only` | `models/camera_only/d415_on_flange.xml` — D415 on a short post (calibrated mount, `steve_wbc/config/gripper_camera.yaml`) | none: go to the grasp pose only | top-down, hover `grasp_dz` 5 cm above the object centre | **real robot now** (default with `use_sim_time:=false`) |
| `robotiq_2f85` | `models/robotiq_2f85/robotiq_2f85.xml` (MuJoCo Menagerie) | 2-finger, `GripperCommand` | top-down, pads at the equator (`grasp_dz` −1.5 cm) | simulation default; identical to the model `steve_wbc` used before |
| `ability_hand` | `models/ability_hand/abh_right_large.xml` generated from [psyonicinc/ability-hand-api](https://github.com/psyonicinc/ability-hand-api) (MIT) | PSYONIC Ability Hand, 6 motors, synergies in API degrees | palm-down wrap (35° tilt), hand rolled 180°, see below | **sim: static ball, clutter, rolling ball (`--dynamic`) grasped + held**; real: bridge tested with ah_node's virtual hand |
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

## Ability Hand (PSYONIC)

**Model.** `python3 tools/build_ability_hand_mjcf.py --src <clone of ability-hand-api>` regenerates
`models/ability_hand/abh_{right,left}_{large,small}.xml`: upstream meshes, hulls, masses and the 30 touch sensors;
no free joint, every name prefixed `abh_`, the finger PIP joints coupled to the MCP joints (the 4-bar linkage,
upstream does it in Python), six position actuators = the six motors in API order
`[index, middle, ring, pinky, thumb flexor, thumb rotator]`, ctrl [rad] = API position [deg] in radians.

**Grasp** (`config/tools/ability_hand.yaml`). Palm-down wrap: the hand comes down with the palm over the object,
tilted 35° (fingers pointing down) so the forearm / wrist stay clear of the table; fingers and thumb close around
it. `tools/sim_grasp_test.py` (the tool on a mocap carriage, top-down like the WBC: descend, close, lift, hold)
lifts the 5 / 6.5 cm ball and the 5 cm cube also with the grasp point 1.2 cm too high or 1 cm off sideways.
The fingers-down grasp (palm sideways, flange vertical) only worked within a few mm and was dropped.
Two tool-config keys exist for hands: `support_contact: true` (the WBC lets the hand's own collision spheres touch
the table in DESCEND / GRASP — not the arm's) and `hold_ref: lifted` (the hold check measures slip from where the
object sat in the hand after the lift; a wrap pulls it into the palm, away from the grasp point).

**Camera.** The hand is rolled 180° on the flange (mount `rpy_deg` yaw): the wrist camera then sits beside the
palm, 1.6 cm above the grasp point, and sees the object down to contact (ball visible 100 / 100 / 82 / 25 % at
15 / 10 / 5 / 0 cm before the grasp; roll 0 — camera on top of the hand — 0 %, 135° fails). WBC sim, roll 180:
static ball, clutter cube and a ball rolling 2–3 cm/s (`--dynamic`) are all grasped, lifted and held; a rolling
ball without `--dynamic` is not (as with the other tools). The real camera mount should copy this: beside the
palm, looking along it.

**Real hand.** Clone [ItsShriks/ability-hand-ros2](https://github.com/ItsShriks/ability-hand-ros2) into `src/`,
`pip install ability-hand` (ah_wrapper, pyserial), enable UART + byte stuffing in the PSYONIC app (see its README),
`colcon build --packages-select ah_messages ah_ros_py steve_grasping`. Then

```bash
ros2 launch steve_grasping ability_hand.launch.py                     # ah_node + bridge (port auto)
ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 1.0}}"
ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false tool:=ability_hand ...   # starts both
```

`ability_hand_gripper` is a `control_msgs/GripperCommand` server (`/gripper_controller/gripper_cmd`, the WBC's
interface): position 0..1 = open..close synergy, streamed to `/ability_hand/<side>/target/position` at 100 Hz
(ah_node without its write thread; the hand leaves API mode without a command for 0.3 s); result `reached_goal`,
or `stalled` when the fingers stop on an object. Without hardware: `hand_simulated:=true` (ah_node's virtual hand).

Open points: the UR adapter plate (mount `xyz`), the real camera mount (beside the palm, as in the sim) and the
measured roll of the hand on the flange.
