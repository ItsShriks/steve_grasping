#!/usr/bin/env python3
"""
build_dex5_mjcf.py — Unitree Dex5-1 URDF -> MuJoCo hand model (models/dex5_1/dex5_<side>.xml).

    python3 tools/build_dex5_mjcf.py            # both hands
    python3 tools/build_dex5_mjcf.py --side right

The URDF (description/dex5_1, from unitreerobotics/unitree_ros, BSD-3) gives the kinematics,
inertias and meshes. The MuJoCo model adds what a simulation and the WBC's collision model need:

  * collision geometry as primitives (the WBC covers every collidable geom with spheres, and
    mesh-mesh contacts are slow and noisy): the palm as two boxes (heel + knuckles), every
    phalanx as a capsule fitted to its mesh (principal axis, 90th percentile radius);
    the STL meshes stay as visuals (group 2)
  * 16 position actuators on the active joints (thumb 4, fingers 3 each: lateral roll, MCP, PIP)
    and the 4 passive distal joints (Pitch_x4) coupled to the PIP (Pitch_x3) by an equality, as
    the real hand's linkage (20 DOF = 16 active + 4 coupled, Unitree spec)
  * actuator torque limited to --max_torque (URDF effort 0.93 N m; ~10 N fingertip force)
  * hand geoms contype 2 / conaffinity 1: they touch the world (object, table) but not each other
  * sites: `dex5_palm` (palm centre, z = palm normal) and one per fingertip

The hand's base frame (URDF base_link00): fingers along +y, thumb on the +x side, palm normal +z
(right hand; left hand: -z, the mesh is mirrored in z); the round wrist flange is the face
y = -0.0398 around (x, z) = (-0.023, -0.0137) (left: z = +0.0136).
The mount on the arm and the grasp point (`pinch`) are added when the hand is attached to the
robot (steve_grasping.robot_model), from config/dex5.yaml.
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
DESC = PKG / "description" / "dex5_1"
OUT = PKG / "models" / "dex5_1"

# passive joint -> (driving joint, ratio): distal phalanx follows the middle one (ranges 1.396 / 1.684)
COUPLED = {f"Pitch_{f}4": (f"Pitch_{f}3", 1.396 / 1.684) for f in (2, 3, 4, 5)}
# links without collision geometry: tiny lateral-roll knuckles / thumb rotator inside the palm
NO_COLLISION = {"Link_11", "Link_21", "Link_31", "Link_41", "Link_51"}
# palm boxes (base frame, (lo, hi)): heel with the thumb base, knuckles where the fingers start
PALM_BOXES = {
    "heel": ((-0.070, -0.040, -0.050), (0.024, 0.035, 0.029)),
    "knuckles": ((-0.070, 0.035, -0.045), (0.024, 0.078, 0.004)),
}


def rpy_to_quat(r, p, y):
    """URDF rpy (R = Rz(y) Ry(p) Rx(r)) -> MuJoCo quaternion (w, x, y, z)."""
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def fmt(v, n=6):
    return " ".join(f"{float(x):.{n}g}" for x in v)


def capsule_fit(vertices):
    """Capsule covering a phalanx mesh: (from, to, radius) in the link frame."""
    V = np.asarray(vertices, float)
    c = V.mean(0)
    _, _, Vt = np.linalg.svd(V - c, full_matrices=False)
    a = Vt[0]
    s = (V - c) @ a
    radial = np.linalg.norm((V - c) - np.outer(s, a), axis=1)
    r = float(np.percentile(radial, 90))
    lo, hi = s.min() + r, s.max() - r
    if hi < lo:  # short and thick: a sphere-like capsule
        lo = hi = 0.5 * (s.min() + s.max())
    return c + lo * a, c + hi * a, r


def build(side, max_torque, kp, out_dir=OUT):
    import trimesh
    S = "R" if side == "right" else "L"
    udir = DESC / f"Dex5-URDF-{S}"
    urdf = ET.parse(udir / f"Dex5-URDF-{S}.urdf").getroot()
    links = {l.get("name"): l for l in urdf.findall("link")}
    joints = urdf.findall("joint")
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)
    root_link = next(n for n in links if n not in {j.find("child").get("link") for j in joints})
    strip = lambda n: n[:-1] if n.endswith(S) else n          # Link_22R -> Link_22
    meshrel = lambda f: f"../../description/dex5_1/Dex5-URDF-{S}/meshes/{Path(f).name}"
    pre = f"dex5{S.lower()}_"
    # the left hand is the right one mirrored in z (palm normal -z, wrist flange at z = +0.0136)
    mirror = np.array([1.0, 1.0, 1.0 if side == "right" else -1.0])

    mj = ET.Element("mujoco", model=f"dex5_1_{side}")
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true")
    default = ET.SubElement(mj, "default")
    dcls = ET.SubElement(default, "default", {"class": "dex5"})
    ET.SubElement(dcls, "joint", damping="0.05", armature="0.002", frictionloss="0.002")
    ET.SubElement(dcls, "position", kp=f"{kp}", forcerange=f"{-max_torque} {max_torque}", ctrllimited="true")
    vis = ET.SubElement(dcls, "default", {"class": "dex5_visual"})
    ET.SubElement(vis, "geom", type="mesh", contype="0", conaffinity="0", group="2", density="0",
                  rgba="0.80 0.80 0.82 1")
    col = ET.SubElement(dcls, "default", {"class": "dex5_collision"})
    # contype 2 / conaffinity 1: contacts with the world (object, table: 1/1), none among the hand's own
    # geoms; torsional friction (condim 4) keeps a ball from spinning out of a fingertip grasp
    ET.SubElement(col, "geom", contype="2", conaffinity="1", group="3", density="0", condim="4",
                  friction="1.5 0.02 0.002", solref="0.01 1", solimp="0.95 0.99 0.001", rgba="0.3 0.6 0.9 0.4")
    asset = ET.SubElement(mj, "asset")
    world = ET.SubElement(mj, "worldbody")
    actuators, equalities = [], []

    def mesh_name(link):
        return pre + strip(link) + "_mesh"

    def add_link(parent_el, link_name, joint=None):
        L = links[link_name]
        attrs = {"name": pre + strip(link_name)}
        if joint is not None:
            o = joint.find("origin")
            xyz = [float(x) for x in (o.get("xyz") or "0 0 0").split()]
            rpy = [float(x) for x in (o.get("rpy") or "0 0 0").split()]
            attrs["pos"] = fmt(xyz)
            q = rpy_to_quat(*rpy)
            if abs(q[0] - 1.0) > 1e-9:
                attrs["quat"] = fmt(q)
        body = ET.SubElement(parent_el, "body", attrs)
        if joint is not None:
            jn = strip(joint.get("name"))
            lim = joint.find("limit")
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            ax = joint.find("axis").get("xyz")
            ET.SubElement(body, "joint", {"name": pre + jn, "class": "dex5", "axis": ax, "range": f"{lo:.5g} {hi:.5g}"})
            if jn in COUPLED:
                drv, ratio = COUPLED[jn]
                equalities.append(dict(joint1=pre + jn, joint2=pre + drv, polycoef=f"0 {ratio:.4f} 0 0 0",
                                       solref="0.005 1", solimp="0.95 0.99 0.001"))
            else:
                actuators.append(dict(name=pre + jn, joint=pre + jn, ctrlrange=f"{lo:.5g} {hi:.5g}"))
        inert = L.find("inertial")
        if inert is not None:
            o = inert.find("origin")
            I = inert.find("inertia")
            full = [float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            ET.SubElement(body, "inertial", pos=o.get("xyz"), mass=inert.find("mass").get("value"),
                          fullinertia=fmt(full))
        mfile = L.find("visual/geometry/mesh").get("filename")
        ET.SubElement(asset, "mesh", name=mesh_name(link_name), file=meshrel(mfile))
        ET.SubElement(body, "geom", {"class": "dex5_visual", "mesh": mesh_name(link_name)})
        mesh = trimesh.load(udir / "meshes" / Path(mfile).name)
        if link_name == root_link:
            for part, (lo, hi) in PALM_BOXES.items():
                lo, hi = np.array(lo) * mirror, np.array(hi) * mirror
                lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                ET.SubElement(body, "geom", {"class": "dex5_collision", "name": f"{pre}palm_{part}", "type": "box",
                                             "pos": fmt(0.5 * (lo + hi)), "size": fmt(0.5 * (hi - lo))})
            # palm centre on the palm surface over the knuckle region, z = palm normal
            ET.SubElement(body, "site", name=f"{pre}palm", pos=fmt([-0.023, 0.055, 0.004 * mirror[2]]),
                          size="0.004", group="5", rgba="0.1 0.9 0.1 1")
        elif strip(link_name) not in NO_COLLISION:
            a, b, r = capsule_fit(mesh.vertices)
            ET.SubElement(body, "geom", {"class": "dex5_collision", "name": pre + strip(link_name) + "_col",
                                         "type": "capsule", "fromto": fmt([*a, *b]), "size": f"{r:.5g}"})
            if strip(link_name).endswith("4"):  # distal phalanx: fingertip site at the far end
                tip = b + (b - a) / max(np.linalg.norm(b - a), 1e-9) * r
                ET.SubElement(body, "site", name=pre + strip(link_name).replace("Link_", "tip_"),
                              pos=fmt(tip), size="0.004", group="5", rgba="0.9 0.2 0.2 1")
        for j in children.get(link_name, []):
            add_link(body, j.find("child").get("link"), j)
        return body

    add_link(world, root_link)
    eq = ET.SubElement(mj, "equality")
    for e in equalities:
        ET.SubElement(eq, "joint", e)
    act = ET.SubElement(mj, "actuator")
    for a in actuators:
        ET.SubElement(act, "position", {"class": "dex5", **a})
    ET.indent(mj, space="  ")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"dex5_{side}.xml"
    header = (f"<!-- Unitree Dex5-1 ({side} hand) for MuJoCo — GENERATED by tools/build_dex5_mjcf.py from\n"
              f"     description/dex5_1/Dex5-URDF-{S} (unitreerobotics/unitree_ros, BSD-3). Do not edit by hand.\n"
              f"     {len(actuators)} position actuators (max {max_torque} N m), {len(equalities)} coupled distal joints. -->\n")
    out.write_text(header + ET.tostring(mj, encoding="unicode") + "\n")
    return out, len(actuators), len(equalities)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=["right", "left", "both"], default="both")
    p.add_argument("--max_torque", type=float, default=0.93, help="actuator torque limit [N m] (URDF effort)")
    p.add_argument("--kp", type=float, default=3.0, help="position actuator gain [N m / rad]")
    a = p.parse_args()
    for side in (["right", "left"] if a.side == "both" else [a.side]):
        out, na, ne = build(side, a.max_torque, a.kp)
        print(f"{out}: {na} actuators, {ne} coupled joints")


if __name__ == "__main__":
    main()
