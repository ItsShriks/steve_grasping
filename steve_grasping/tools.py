"""
tools.py — what is mounted on the UR5e flange, and how the WBC grasps with it.

    from steve_grasping.tools import load_tool, TOOLS
    tool = load_tool("camera_only")       # camera_only | robotiq_2f85 | dex5 | ability_hand
    tool.gripper, tool.grasp_dz, tool.model_path, tool.base_R, tool.pinch_pos ...

Every tool is a MuJoCo model (models/<tool>/...) attached at the flange by robot_model.compose(), plus
a config (config/tools/<tool>.yaml):

  model        MJCF of the tool, its root body is placed at the flange (UR tool0: z out of the flange)
  gripper      false: nothing to close (active perception only goes to the grasp pose)
  own_camera   true: the tool carries the wrist camera (`wrist_cam`), the robot model's D405 is removed
  grasp_style  topdown (grasp frame z pointing down) | side (horizontal approach)
  grasp_dz     default --grasp_dz: grasp point relative to the object centre [m]
  mount        tool0 -> tool base frame: xyz + rpy_deg, or (hands) flange_in_hand + adapter_length +
               hand_axes_in_tool (+ yaw_deg about the tool axis)
  pinch        grasp point (WBC end effector) in the tool base frame: xyz + rpy_deg
  support_contact  true: the tool may touch the support surface (table) in the final descent and while
               closing — the WBC drops the table margin for the tool's collision spheres (not the arm's)
               in DESCEND / GRASP (a hand whose fingers wrap around the object down to the table)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TOOLS = ("camera_only", "robotiq_2f85", "dex5", "ability_hand")
DEFAULT_TOOL = os.environ.get("STEVE_TOOL", "robotiq_2f85")


def package_dir():
    """steve_grasping's share folder when installed, else the source folder."""
    here = Path(__file__).resolve().parent.parent
    if (here / "config" / "tools").is_dir():
        return here
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("steve_grasping"))
    except Exception:
        return here


def rpy_matrix(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll) (URDF / ROS convention)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


@dataclass
class ToolSpec:
    name: str
    model_path: Path
    gripper: bool
    own_camera: bool
    grasp_style: str
    grasp_dz: float
    base_R: np.ndarray          # tool base frame in tool0
    base_pos: np.ndarray
    pinch_R: np.ndarray         # grasp frame in the tool base frame
    pinch_pos: np.ndarray
    support_contact: bool = False
    sim: dict = field(default_factory=dict)
    real: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def __str__(self):
        return (f"{self.name} ({'gripper' if self.gripper else 'no gripper'}, {self.grasp_style} grasp, "
                f"grasp_dz {self.grasp_dz:+.3f} m)")


def load_tool(name=None):
    import yaml
    name = name or DEFAULT_TOOL
    pkg = package_dir()
    path = Path(name)
    if not path.suffix:
        path = pkg / "config" / "tools" / f"{name}.yaml"
    if not path.is_file():
        raise SystemExit(f"[ERROR] tool '{name}' unknown ({path} not found); tools: {', '.join(TOOLS)}")
    c = yaml.safe_load(path.read_text())
    m = c.get("mount", {})
    if "flange_in_hand" in m:  # hand: its wrist flange face on the tool flange, adapter in between
        A = np.asarray(m["hand_axes_in_tool"], float)   # rows: tool axes in hand coordinates
        R = A.T                                          # columns: hand axes in tool coordinates
        yaw = np.radians(m.get("yaw_deg", 0.0))
        R = rpy_matrix(0.0, 0.0, yaw) @ R
        p = np.array([0.0, 0.0, m.get("adapter_length", 0.0)]) - R @ np.asarray(m["flange_in_hand"], float)
    else:
        R = rpy_matrix(*np.radians(m.get("rpy_deg", [0, 0, 0])))
        p = np.asarray(m.get("xyz", [0, 0, 0]), float)
    pin = c.get("pinch", {})
    return ToolSpec(name=c.get("name", path.stem), model_path=(pkg / c["model"]).resolve(),
                    gripper=bool(c.get("gripper", True)), own_camera=bool(c.get("own_camera", False)),
                    grasp_style=c.get("grasp_style", "topdown"), grasp_dz=float(c.get("grasp_dz", 0.0)),
                    base_R=R, base_pos=p, pinch_R=rpy_matrix(*np.radians(pin.get("rpy_deg", [0, 0, 0]))),
                    pinch_pos=np.asarray(pin.get("xyz", [0, 0, 0]), float),
                    support_contact=bool(c.get("support_contact", False)), sim=c.get("sim", {}) or {},
                    real=c.get("real", {}) or {}, config=c)
