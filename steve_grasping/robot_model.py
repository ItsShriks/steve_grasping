"""
robot_model.py — attach the tool (camera only, Robotiq 2F-85, Dex5-1) to the robot's MuJoCo model.

The robot model (steve_wbc/models/mmo_700_wbc.xml: base, UR5e, D405 wrist camera) ends in an empty
body `hand_mount` at the tool flange (UR tool0: z out of the flange; identical to the real URDF's
ur5etool0). `compose()` puts the tool's MJCF (models/<tool>/...) there — XML level, so it works with
every MuJoCo version the WBC runs on — together with its assets, defaults, actuators, equalities and
contact exclusions, and adds the grasp point site `pinch` (the WBC's end effector: z = approach,
y = closing axis) from the tool config. A tool with its own camera (camera_only) replaces the D405; a tool
whose config has a `camera:` section (the D415 on a bracket, e.g. on top of the Ability Hand) gets that
camera — housing, bracket boxes and the `wrist_cam` — at its calibrated pose (see _add_tool_camera).

All asset paths are made absolute, so the result loads from a string:
    xml = compose(robot_xml_path, load_tool("camera_only"))
    model = mujoco.MjModel.from_xml_string(xml)       # or mujoco.MjSpec.from_string(xml)
"""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _abs_assets(root, base_dir):
    """Asset file paths -> absolute (relative to the compiler's meshdir/texturedir/assetdir)."""
    comp = root.find("compiler")
    dirs = {"mesh": "", "texture": "", "hfield": "", "skin": ""}
    if comp is not None:
        a = comp.get("assetdir", "")
        dirs = {"mesh": comp.get("meshdir", a), "texture": comp.get("texturedir", a), "hfield": a, "skin": a}
        for k in ("meshdir", "texturedir", "assetdir"):
            comp.attrib.pop(k, None)
    asset = root.find("asset")
    if asset is None:
        return
    for el in asset:
        f = el.get("file")
        if f and not Path(f).is_absolute():
            el.set("file", str((Path(base_dir) / dirs.get(el.tag, "") / f).resolve()))


def _quat_from_R(R):
    """Rotation matrix -> (w, x, y, z), Shepperd's method (also exact for 180 deg rotations, where the
    sign-from-off-diagonals shortcut picks wrong signs)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    i = int(np.argmax([t, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = 2.0 * np.sqrt(1.0 + t)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        s = 2.0 * np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q = q if q[0] >= 0 else -q
    return q / np.linalg.norm(q)


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def _box(body, name, pos, full_size, rgba):
    """Collision box (group 3) + its visual twin (group 2)."""
    half = _fmt(np.asarray(full_size, float) / 2.0)
    ET.SubElement(body, "geom", name=name, type="box", pos=_fmt(pos), size=half, contype="1", conaffinity="1",
                  group="3", density="0", rgba="0.3 0.3 0.35 1")
    ET.SubElement(body, "geom", name=name + "_visual", type="box", pos=_fmt(pos), size=half, contype="0",
                  conaffinity="0", group="2", density="0", rgba=rgba)


def _add_tool_camera(mount, cam):
    """The tool config's `camera:` at the flange (tool0): its bracket (boxes in tool0) and the camera body = the
    colour optical frame (xyz + quat_xyzw in tool0, as gripper_camera.yaml writes them) with the housing box
    (in the optical frame) and the MuJoCo camera `wrist_cam` (looks along the optical z). Returns the body names."""
    bracket = ET.SubElement(mount, "body", name="tool_camera_bracket")
    ET.SubElement(bracket, "inertial", pos="0 0 0.01", mass="0.05", diaginertia="1e-5 1e-5 1e-5")
    for i, b in enumerate(cam.get("bracket", [])):
        _box(bracket, f"tool_camera_bracket_{i}", b["xyz"], b["size"], "0.08 0.08 0.08 1")
    qx, qy, qz, qw = cam["quat_xyzw"]
    optical = ET.SubElement(mount, "body", name="gripper_camera_color_optical_frame", pos=_fmt(cam["xyz"]),
                            quat=_fmt([qw, qx, qy, qz]))
    ET.SubElement(optical, "inertial", pos="0 0 0", mass="0.07", diaginertia="1e-5 1e-5 1e-5")
    _box(optical, "tool_camera_housing", cam.get("housing_offset", [0, 0, 0]), cam["housing_size"],
         "0.82 0.82 0.84 1")
    # MuJoCo cameras look along -z with image up +y: the optical frame (z forward, y down) turned 180 deg about x
    ET.SubElement(optical, "camera", name="wrist_cam", quat="0 1 0 0", fovy=str(cam.get("fovy_deg", 42)))
    return "tool_camera_bracket", "gripper_camera_color_optical_frame"


def compose(robot_xml, hand, mount_body="hand_mount"):
    """MJCF string of the robot with the tool `hand` (steve_grasping.tools.ToolSpec) at `mount_body`."""
    robot_xml = Path(robot_xml)
    rt = ET.parse(robot_xml).getroot()
    _abs_assets(rt, robot_xml.parent)
    mount = next((b for b in rt.iter("body") if b.get("name") == mount_body), None)
    if mount is None:
        raise ValueError(f"{robot_xml}: no body '{mount_body}' to attach the hand to")
    for b in list(mount):  # a hand attached before (e.g. a model saved with one): replace it
        if b.tag == "body":
            mount.remove(b)
    # qpos layout changes with the hand: the robot keyframe would no longer fit
    for kf in rt.findall("keyframe"):
        rt.remove(kf)
    if hand.own_camera:  # the tool brings the wrist camera: drop the model's D405 (and its exclusions)
        for parent in rt.iter():
            for b in [b for b in parent if b.tag == "body" and b.get("name") == "wrist_d405_mount"]:
                parent.remove(b)

    ht = ET.parse(hand.model_path).getroot()
    _abs_assets(ht, hand.model_path.parent)
    # defaults / assets: hand classes go into the robot's top-level default, assets merged by name
    rdef = rt.find("default")
    if rdef is None:
        rdef = ET.SubElement(rt, "default")
    hdef = ht.find("default")
    if hdef is not None:
        for d in hdef:
            rdef.append(copy.deepcopy(d))
    rasset = rt.find("asset")
    if rasset is None:
        rasset = ET.SubElement(rt, "asset")
    names = {(el.tag, el.get("name") or Path(el.get("file", "")).stem) for el in rasset}
    hasset = ht.find("asset")
    for el in (hasset if hasset is not None else []):
        key = (el.tag, el.get("name") or Path(el.get("file", "")).stem)
        if key not in names:
            rasset.append(copy.deepcopy(el))

    # mount: tool0 -> tool base frame (from the tool config)
    hb = ht.find("worldbody").find("body")
    hand_body = copy.deepcopy(hb)
    hand_body.set("pos", _fmt(hand.base_pos))
    hand_body.set("quat", _fmt(_quat_from_R(hand.base_R)))
    # grasp point (WBC end effector) on the hand base
    ET.SubElement(hand_body, "site", name="pinch", pos=_fmt(hand.pinch_pos), quat=_fmt(_quat_from_R(hand.pinch_R)),
                  type="sphere", size="0.005", group="5", rgba="0.9 0.9 0.9 1")
    mount.append(hand_body)
    cam_cfg = (hand.config or {}).get("camera") if hand.own_camera else None
    cam_bodies = _add_tool_camera(mount, cam_cfg) if cam_cfg else ()

    for tag in ("contact", "tendon", "equality", "actuator", "sensor"):
        hs = ht.find(tag)
        if hs is None or not len(hs):
            continue
        rs = rt.find(tag)
        if rs is None:
            rs = ET.SubElement(rt, tag)
        for el in hs:
            rs.append(copy.deepcopy(el))
    # the hand sits on the flange (and the wrist camera mount): no contacts there
    rc = rt.find("contact")
    if rc is None:
        rc = ET.SubElement(rt, "contact")
    for other in ("wrist_3_link", "wrist_d405_mount", "wrist_camera_link", *cam_bodies):
        if any(b.get("name") == other for b in rt.iter("body")):
            ET.SubElement(rc, "exclude", body1=other, body2=hand_body.get("name"))
    # the tool's camera and its bracket are one rigid part on the flange (the fingers still collide with them)
    for i, cb in enumerate(cam_bodies):
        ET.SubElement(rc, "exclude", body1=cb, body2="wrist_3_link")
        for other in cam_bodies[i + 1:]:
            ET.SubElement(rc, "exclude", body1=cb, body2=other)
    # exclusions that name bodies no longer in the model (the removed D405) would not compile
    bodies = {b.get("name") for b in rt.iter("body")} | {"world"}
    for ex in [e for e in rc if e.tag == "exclude"]:
        if ex.get("body1") not in bodies or ex.get("body2") not in bodies:
            rc.remove(ex)
    return ET.tostring(rt, encoding="unicode")


def load_robot_model(robot_xml, tool, spec_edit=None):
    """MjModel of the robot with `tool` (name or ToolSpec); spec_edit(spec) may edit it first (MjSpec)."""
    import mujoco
    from steve_grasping.tools import load_tool
    tool = load_tool(tool) if tool is None or isinstance(tool, str) else tool
    xml = compose(robot_xml, tool)
    if spec_edit is None:
        return mujoco.MjModel.from_xml_string(xml)
    spec = mujoco.MjSpec.from_string(xml)
    spec_edit(spec)
    return spec.compile()
