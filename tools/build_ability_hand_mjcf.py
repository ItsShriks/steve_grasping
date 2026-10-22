#!/usr/bin/env python3
"""
build_ability_hand_mjcf.py — PSYONIC Ability Hand MuJoCo models (psyonicinc/ability-hand-api, MIT) ->
models/ability_hand/abh_<side>_<size>.xml, ready to be attached to the robot by steve_grasping.

    git clone --depth 1 https://github.com/psyonicinc/ability-hand-api /tmp/ability-hand-api
    python3 tools/build_ability_hand_mjcf.py --src /tmp/ability-hand-api      # all four hands

Taken from python/ah_simulators/mujoco_xml/hands/abh_<side>_<size>.xml (+ assets): meshes, collision
hulls, masses, joint ranges, the 30 touch sensors (FSR sites). Changed for the WBC / grasping:

  * no free joint (the hand sits on the flange); every name (bodies, joints, geoms, sites, classes,
    meshes) gets the prefix `abh_` so it cannot clash with the robot model it is merged into
  * the finger PIP joints follow the MCP joints through an equality (the real finger is a 4-bar
    linkage, upstream mimics it in a Python loop: pip = 0.72349796 + 1.05851325 * mcp); the PIP lower
    limit is opened to that offset (upstream 0.766 > 0.7235 would fight the coupling at mcp = 0)
  * six position actuators = the six motors of the real hand, in the order of the API
    [index, middle, ring, pinky, thumb flexor, thumb rotator]; ctrl [rad] = API position [deg] in
    radians (as upstream's ah_mujoco). Upstream drives index_mcp with a bare `general` actuator (ctrl
    = torque) and kp = 1 — too weak to hold an object; here kp / forcerange are --kp / --max_torque
  * hand geoms contype 2 / conaffinity 1 (like dex5): they touch the world but not each other

Hand frame (upstream): origin at the centre of the wrist flange face, fingers along +z, palm side +y
(the fingers curl towards +y, the rotated thumb opposes them from +y), thumb base on -x (right hand). The mount on the arm and
the grasp point (`pinch`) come from config/tools/ability_hand.yaml when the hand is attached.
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "models" / "ability_hand"
PREFIX = "abh_"
FINGERS = ("index", "middle", "ring", "pinky")
PIP_OFFSET, PIP_RATIO = 0.72349796, 1.05851325   # upstream ah_mujoco.Controller.mimic_joints
# actuator name -> joint, in API order [index, middle, ring, pinky, thumb flexor, thumb rotator]
ACTUATORS = [(f"{f}_mcp", f"{f}_mcp") for f in FINGERS] + [("thumb_flexor", "thumb_mcp"),
                                                           ("thumb_rotator", "thumb_cmc")]
NAME_ATTRS = ("name", "class", "childclass", "mesh", "joint", "joint1", "joint2", "site", "body1", "body2")


def build(src_xml, out_xml, kp, max_torque, damping):
    root = ET.parse(src_xml).getroot()
    root.set("model", out_xml.stem)
    comp = root.find("compiler")
    comp.set("meshdir", "meshes")
    # drop upstream's actuators / visual (re-made below), the free joint
    for tag in ("actuator", "visual"):
        for el in root.findall(tag):
            root.remove(el)
    hand = root.find("worldbody").find("body")
    for j in hand.findall("joint"):
        if j.get("type") == "free":
            hand.remove(j)
    hand.attrib.pop("pos", None)
    # prefix every name / reference
    for el in root.iter():
        for a in NAME_ATTRS:
            v = el.get(a)
            if v is not None and not (el.tag == "compiler"):
                el.set(a, PREFIX + v.replace(" ", "_"))
    # joint defaults: damping, PIP lower limit, no actuator defaults upstream (kp = 1, general)
    for d in root.iter("default"):
        for j in d.findall("joint"):
            if d.get("class") == PREFIX + "joints":
                j.set("damping", f"{damping:g}")
                j.set("armature", "0.001")
            if d.get("class") == PREFIX + "finger_pip_joint":
                lo, hi = (float(x) for x in j.get("range").split())
                j.set("range", f"{min(lo, PIP_OFFSET - 0.01):.4f} {hi:g}")
        for tag in ("position", "general"):
            for a in d.findall(tag):
                d.remove(a)
        if d.get("class") == PREFIX + "joints":
            ET.SubElement(d, "position", kp=f"{kp:g}", forcerange=f"{-max_torque:g} {max_torque:g}",
                          ctrllimited="true")
    # collision geoms: world contacts only (not hand-hand)
    for d in root.iter("default"):
        if d.get("class") == PREFIX + "collisions_ah":
            g = d.find("geom")
            g.set("contype", "2")
            g.set("conaffinity", "1")
            g.set("condim", "4")            # torsional friction: a ball held at the fingertips does not spin
            g.set("friction", "1.0 0.02 0.0001")
    # joint ranges per class (for the actuators' ctrlrange)
    ranges = {}
    for d in root.iter("default"):
        j = d.find("joint")
        if j is not None and j.get("range"):
            ranges[d.get("class")] = j.get("range")
    jclass = {j.get("name"): j.get("class") for j in root.iter("joint")}

    eq = ET.SubElement(root, "equality")
    for f in FINGERS:
        ET.SubElement(eq, "joint", joint1=f"{PREFIX}{f}_pip", joint2=f"{PREFIX}{f}_mcp",
                      polycoef=f"{PIP_OFFSET} {PIP_RATIO} 0 0 0", solref="0.005 1", solimp="0.95 0.99 0.001")
    act = ET.SubElement(root, "actuator")
    for name, joint in ACTUATORS:
        jn = PREFIX + joint
        ET.SubElement(act, "position", name=PREFIX + name, joint=jn, **{"class": PREFIX + "joints"},
                      ctrlrange=ranges[jclass[jn]])
    # palm site: centre of the palm, z = palm normal (out of the palm, +y of the hand frame)
    ET.SubElement(hand, "site", name=PREFIX + "palm", pos="0 0.02 0.07", quat="0.7071068 -0.7071068 0 0",
                  type="sphere", size="0.004", group="5", rgba="0 1 0 1")
    ET.indent(root, "  ")
    out_xml.write_text(f"<!-- Generated by tools/build_ability_hand_mjcf.py from {src_xml.name} "
                       "(psyonicinc/ability-hand-api, MIT). Do not edit. -->\n"
                       + ET.tostring(root, encoding="unicode") + "\n")
    meshes = {m.get("file") for m in root.iter("mesh") if m.get("file")}
    return meshes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="clone of psyonicinc/ability-hand-api")
    p.add_argument("--kp", type=float, default=5.0, help="position gain of the six motors [N m / rad]")
    p.add_argument("--max_torque", type=float, default=0.6, help="motor torque limit at the joint [N m]")
    p.add_argument("--damping", type=float, default=0.05, help="joint damping [N m s / rad]")
    a = p.parse_args()
    hands = Path(a.src) / "python" / "ah_simulators" / "mujoco_xml" / "hands"
    assets = hands.parent / "assets"
    (OUT / "meshes").mkdir(parents=True, exist_ok=True)
    meshes = set()
    for side in ("right", "left"):
        for size in ("large", "small"):
            src = hands / f"abh_{side}_{size}.xml"
            out = OUT / f"abh_{side}_{size}.xml"
            meshes |= build(src, out, a.kp, a.max_torque, a.damping)
            print(f"{out.relative_to(PKG)}")
    for m in sorted(meshes):
        shutil.copy(assets / m, OUT / "meshes" / m)
    shutil.copy(Path(a.src) / "LICENSE", OUT / "LICENSE")
    print(f"{len(meshes)} meshes -> {(OUT / 'meshes').relative_to(PKG)}")


if __name__ == "__main__":
    main()
