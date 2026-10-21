"""
robot_model.py — attach the tool (camera only, Robotiq 2F-85, Dex5-1) to the robot's MuJoCo model.

The robot model (steve_wbc/models/mmo_700_wbc.xml: base, UR5e, D405 wrist camera) ends in an empty
body `hand_mount` at the tool flange (UR tool0: z out of the flange; identical to the real URDF's
ur5etool0). `compose()` puts the tool's MJCF (models/<tool>/...) there — XML level, so it works with
every MuJoCo version the WBC runs on — together with its assets, defaults, actuators, equalities and
contact exclusions, and adds the grasp point site `pinch` (the WBC's end effector: z = approach,
y = closing axis) from the tool config. A tool with its own camera (camera_only) replaces the D405.

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
    """Rotation matrix -> (w, x, y, z)."""
    R = np.asarray(R, float)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


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
    for other in ("wrist_3_link", "wrist_d405_mount", "wrist_camera_link"):
        if any(b.get("name") == other for b in rt.iter("body")):
            ET.SubElement(rc, "exclude", body1=other, body2=hand_body.get("name"))
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
